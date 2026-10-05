from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DATA_ROOT = (
    WORKSPACE
    / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
)
OUT = PROJECT / "results/weekly_dynamic_research"
CASH_CACHE = PROJECT / "results/requested_premium_study/spx_cash_close.csv"
LOCAL_CASH_ROOT = (
    WORKSPACE
    / "05 - Underlying Prices and Events/Cleaned Data/Yahoo Finance/FX External/processed"
    / "provider=yfinance/group=equities/series=sp500"
)
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.data_loader import SPXSurfaceArchive  # noqa: E402
from spxresearch.execution import ExecutionModel  # noqa: E402


MULTIPLIER = 100.0
MID_MODEL = ExecutionModel(spread_fraction=0.0, commission_per_contract=0.0)
REALISTIC_MODEL = ExecutionModel(spread_fraction=0.25, commission_per_contract=1.50)
NATURAL_MODEL = ExecutionModel(spread_fraction=0.50, commission_per_contract=1.50)
FILL_MODELS = {
    "mid": MID_MODEL,
    "realistic": REALISTIC_MODEL,
    "natural": NATURAL_MODEL,
}

SHORT_RATIOS = (0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03)
HEDGE_WIDTHS = (0.01, 0.02, 0.03, 0.05)
CANDIDATE_GRID = tuple(
    (short_ratio, width)
    for short_ratio in SHORT_RATIOS
    for width in HEDGE_WIDTHS
)
BASELINE_SHORT_RATIO = 0.99
BASELINE_WIDTH = 0.03

LEG_FIELDS = (
    "symbol",
    "bid",
    "ask",
    "mid",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "volume",
    "open_interest",
)


class CandidateError(ValueError):
    """A requested grid point cannot be represented by executable quotes."""


def weekly_roll_schedule(dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Map each calendar Friday to Friday or its latest session in the prior four days."""
    available = pd.DatetimeIndex(dates).normalize().unique().sort_values()
    if available.empty:
        return pd.DataFrame(columns=["scheduled_friday", "roll_date"])
    first = available.min()
    first_friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
    scheduled_fridays = pd.date_range(first_friday, available.max(), freq="W-FRI")
    available_set = set(available)
    rows: list[dict[str, pd.Timestamp]] = []
    for scheduled in scheduled_fridays:
        choices = [
            scheduled - pd.Timedelta(days=offset)
            for offset in range(5)
            if scheduled - pd.Timedelta(days=offset) in available_set
        ]
        if choices:
            rows.append(
                {
                    "scheduled_friday": scheduled.normalize(),
                    "roll_date": max(choices).normalize(),
                }
            )
    schedule = pd.DataFrame(rows)
    if not schedule.empty:
        schedule = schedule.drop_duplicates("roll_date", keep="last").reset_index(drop=True)
    return schedule


def load_cash_closes_read_only() -> pd.Series:
    """Read cached and local cash SPX closes without refreshing or rewriting either source."""
    frames: list[pd.DataFrame] = []
    if LOCAL_CASH_ROOT.exists():
        for path in sorted(LOCAL_CASH_ROOT.glob("year=*/sp500_*.parquet")):
            local = pd.read_parquet(path, columns=["observation_date", "close"])
            frames.append(
                local.rename(
                    columns={"observation_date": "date", "close": "spx_close"}
                )[["date", "spx_close"]]
            )
    if CASH_CACHE.exists():
        frames.append(pd.read_csv(CASH_CACHE, usecols=["date", "spx_close"]))
    if not frames:
        raise FileNotFoundError(
            f"No read-only cash SPX source found at {CASH_CACHE} or {LOCAL_CASH_ROOT}"
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined["spx_close"] = pd.to_numeric(combined["spx_close"], errors="coerce")
    combined = (
        combined.dropna(subset=["date", "spx_close"])
        .loc[lambda frame: frame["spx_close"].gt(0)]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )
    return combined.set_index("date")["spx_close"].astype(float)


def valid_pm_puts(chain: pd.DataFrame, expiration_date: pd.Timestamp) -> pd.DataFrame:
    bid = pd.to_numeric(chain["bid"], errors="coerce")
    ask = pd.to_numeric(chain["ask"], errors="coerce")
    strike = pd.to_numeric(chain["strike"], errors="coerce")
    valid = (
        chain["option_type"].eq("put")
        & chain["expiration_date"].eq(pd.Timestamp(expiration_date).normalize())
        & chain["settlement"].eq("PM")
        & strike.gt(0)
        & bid.ge(0)
        & ask.ge(bid)
    )
    return chain.loc[valid].copy()


def nearest_put(puts: pd.DataFrame, target_strike: float) -> pd.Series:
    if puts.empty:
        raise CandidateError("no valid PM-settled put quotes for expiration")
    ranked = puts.assign(
        _distance=(pd.to_numeric(puts["strike"]) - float(target_strike)).abs()
    ).sort_values(["_distance", "strike", "option_symbol"], kind="stable")
    return ranked.iloc[0]


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _leg_values(prefix: str, quote: pd.Series) -> dict[str, object]:
    values: dict[str, object] = {
        f"{prefix}_symbol": str(quote["option_symbol"]).strip(),
        f"{prefix}_strike": _safe_float(quote["strike"]),
    }
    for field in LEG_FIELDS[1:]:
        values[f"{prefix}_{field}"] = _safe_float(quote.get(field, np.nan))
    return values


def _sum_nonnegative(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame.get(column), errors="coerce")
    if values is None:
        return np.nan
    values = values[values.ge(0)]
    return float(values.sum()) if not values.empty else np.nan


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if np.isfinite(denominator) and denominator > 0 else np.nan


def _nearest_iv_put(puts: pd.DataFrame, target_strike: float) -> pd.Series | None:
    iv = pd.to_numeric(puts.get("implied_volatility"), errors="coerce")
    eligible = puts.loc[iv.gt(0) & np.isfinite(iv)].copy()
    if eligible.empty:
        return None
    return nearest_put(eligible, target_strike)


def chain_features(
    chain: pd.DataFrame,
    weekly_puts: pd.DataFrame,
    *,
    entry_date: pd.Timestamp,
    expiration_date: pd.Timestamp,
    spot: float,
) -> dict[str, object]:
    """Summarize only information observable in the entry snapshot."""
    same_expiry = chain[chain["expiration_date"].eq(expiration_date)]
    same_expiry_puts = same_expiry[same_expiry["option_type"].eq("put")]
    same_expiry_calls = same_expiry[same_expiry["option_type"].eq("call")]
    all_puts = chain[chain["option_type"].eq("put")]
    all_calls = chain[chain["option_type"].eq("call")]

    atm = _nearest_iv_put(weekly_puts, spot)
    put_95 = _nearest_iv_put(weekly_puts, 0.95 * spot)
    iv = pd.to_numeric(weekly_puts.get("implied_volatility"), errors="coerce")
    delta = pd.to_numeric(weekly_puts.get("delta"), errors="coerce")
    delta_eligible = weekly_puts.loc[iv.gt(0) & np.isfinite(iv) & delta.notna()].copy()
    put_25d = None
    if not delta_eligible.empty:
        put_25d = delta_eligible.assign(
            _delta_distance=(pd.to_numeric(delta_eligible["delta"]) + 0.25).abs()
        ).sort_values(["_delta_distance", "strike", "option_symbol"], kind="stable").iloc[0]

    atm_iv = _safe_float(atm["implied_volatility"]) if atm is not None else np.nan
    put_95_iv = (
        _safe_float(put_95["implied_volatility"]) if put_95 is not None else np.nan
    )
    put_25d_iv = (
        _safe_float(put_25d["implied_volatility"]) if put_25d is not None else np.nan
    )

    one_month_expiration = pd.NaT
    one_month_dte = np.nan
    one_month_atm_iv = np.nan
    pm_puts = chain[
        chain["option_type"].eq("put")
        & chain["settlement"].eq("PM")
        & pd.to_numeric(chain["implied_volatility"], errors="coerce").gt(0)
        & pd.to_numeric(chain["strike"], errors="coerce").gt(0)
        & chain["expiration_date"].gt(entry_date)
    ].copy()
    if not pm_puts.empty:
        expirations = pd.DatetimeIndex(pm_puts["expiration_date"].unique())
        one_month_expiration = min(
            expirations,
            key=lambda value: (abs((pd.Timestamp(value) - entry_date).days - 30), value),
        )
        one_month_dte = int((pd.Timestamp(one_month_expiration) - entry_date).days)
        one_month_puts = pm_puts[
            pm_puts["expiration_date"].eq(one_month_expiration)
        ]
        one_month_atm = _nearest_iv_put(one_month_puts, spot)
        if one_month_atm is not None:
            one_month_atm_iv = _safe_float(one_month_atm["implied_volatility"])

    chain_put_volume = _sum_nonnegative(all_puts, "volume")
    chain_call_volume = _sum_nonnegative(all_calls, "volume")
    chain_put_oi = _sum_nonnegative(all_puts, "open_interest")
    chain_call_oi = _sum_nonnegative(all_calls, "open_interest")
    weekly_put_volume = _sum_nonnegative(same_expiry_puts, "volume")
    weekly_call_volume = _sum_nonnegative(same_expiry_calls, "volume")
    weekly_put_oi = _sum_nonnegative(same_expiry_puts, "open_interest")
    weekly_call_oi = _sum_nonnegative(same_expiry_calls, "open_interest")

    return {
        "same_expiry_atm_put_iv": atm_iv,
        "same_expiry_95_put_iv": put_95_iv,
        "same_expiry_25d_put_iv": put_25d_iv,
        "same_expiry_put_skew_95_minus_atm": put_95_iv - atm_iv,
        "same_expiry_put_skew_25d_minus_atm": put_25d_iv - atm_iv,
        "one_month_atm_put_iv": one_month_atm_iv,
        "one_month_expiration": one_month_expiration,
        "one_month_dte": one_month_dte,
        "chain_put_volume": chain_put_volume,
        "chain_call_volume": chain_call_volume,
        "chain_put_call_volume_ratio": _ratio(chain_put_volume, chain_call_volume),
        "chain_put_open_interest": chain_put_oi,
        "chain_call_open_interest": chain_call_oi,
        "chain_put_call_open_interest_ratio": _ratio(chain_put_oi, chain_call_oi),
        "same_expiry_put_volume": weekly_put_volume,
        "same_expiry_call_volume": weekly_call_volume,
        "same_expiry_put_call_volume_ratio": _ratio(
            weekly_put_volume, weekly_call_volume
        ),
        "same_expiry_put_open_interest": weekly_put_oi,
        "same_expiry_call_open_interest": weekly_call_oi,
        "same_expiry_put_call_open_interest_ratio": _ratio(
            weekly_put_oi, weekly_call_oi
        ),
    }


def build_candidate_record(
    *,
    entry_date: pd.Timestamp,
    scheduled_entry_friday: pd.Timestamp,
    expiration_date: pd.Timestamp,
    scheduled_expiration_friday: pd.Timestamp,
    spot_entry: float,
    spot_expiration: float,
    short_ratio: float,
    width: float,
    puts: pd.DataFrame,
    shared_features: dict[str, object] | None = None,
    require_positive_credit: bool = True,
) -> dict[str, object]:
    """Account for exact quoted legs; fixed policies may retain fee-driven debits."""
    long_ratio = short_ratio - width
    short = nearest_put(puts, spot_entry * short_ratio)
    long = nearest_put(puts, spot_entry * long_ratio)
    short_strike = float(short["strike"])
    long_strike = float(long["strike"])
    width_points = short_strike - long_strike
    if width_points <= 0:
        raise CandidateError(
            f"nearest strikes are not descending ({short_strike:g}/{long_strike:g})"
        )
    width_cash = width_points * MULTIPLIER
    spot_notional = spot_entry * MULTIPLIER

    record: dict[str, object] = {
        "candidate_id": (
            f"{entry_date:%Y%m%d}_s{int(round(short_ratio * 100)):03d}"
            f"_w{int(round(width * 100)):02d}"
        ),
        "entry_date": entry_date,
        "scheduled_entry_friday": scheduled_entry_friday,
        "expiration_date": expiration_date,
        "scheduled_expiration_friday": scheduled_expiration_friday,
        "dte": int((expiration_date - entry_date).days),
        "spot_entry": spot_entry,
        "spot_expiration": spot_expiration,
        "spx_forward_return": spot_expiration / spot_entry - 1.0,
        "contracts_per_1m": 1_000_000.0 / spot_notional,
        "target_short_ratio": short_ratio,
        "target_long_ratio": long_ratio,
        "target_upper_ratio": short_ratio,
        "target_lower_ratio": long_ratio,
        "target_width_pct": width,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "upper_strike": short_strike,
        "lower_strike": long_strike,
        "actual_short_ratio": short_strike / spot_entry,
        "actual_long_ratio": long_strike / spot_entry,
        "actual_upper_ratio": short_strike / spot_entry,
        "actual_lower_ratio": long_strike / spot_entry,
        "width_points": width_points,
        "width_pct": width_points / spot_entry,
        "width_cash": width_cash,
        "is_baseline_99_96": bool(
            np.isclose(short_ratio, BASELINE_SHORT_RATIO)
            and np.isclose(width, BASELINE_WIDTH)
        ),
    }
    record.update(_leg_values("short", short))
    record.update(_leg_values("long", long))
    record["upper_symbol"] = record["short_symbol"]
    record["lower_symbol"] = record["long_symbol"]
    record["short_iv"] = record["short_implied_volatility"]
    record["long_iv"] = record["long_implied_volatility"]
    for field in LEG_FIELDS[1:]:
        record[f"upper_{field}"] = record[f"short_{field}"]
        record[f"lower_{field}"] = record[f"long_{field}"]

    short_intrinsic = max(short_strike - spot_expiration, 0.0) * MULTIPLIER
    long_intrinsic = max(long_strike - spot_expiration, 0.0) * MULTIPLIER
    expiration_value = long_intrinsic - short_intrinsic
    record["short_terminal_intrinsic_cash"] = short_intrinsic
    record["long_terminal_intrinsic_cash"] = long_intrinsic
    record["short_terminal_payoff_cash"] = -short_intrinsic
    record["long_terminal_payoff_cash"] = long_intrinsic
    record["expiration_value_cash"] = expiration_value

    for fill, model in FILL_MODELS.items():
        short_sell_fill = model.fill(short, -1)
        long_buy_fill = model.fill(long, 1)
        short_entry_cash = model.cash_flow(short, -1)
        long_entry_cash = model.cash_flow(long, 1)
        gross_credit_cash = (short_sell_fill - long_buy_fill) * MULTIPLIER
        premium_cash = short_entry_cash + long_entry_cash
        if premium_cash >= width_cash or (require_positive_credit and premium_cash <= 0):
            raise CandidateError(
                f"{fill} credit {premium_cash / MULTIPLIER:.4f} is outside "
                f"(0, {width_points:.4f}) points"
            )
        maximum_loss = width_cash - premium_cash
        short_pnl = short_entry_cash - short_intrinsic
        long_pnl = long_entry_cash + long_intrinsic
        pnl = premium_cash + expiration_value
        record.update(
            {
                f"short_sell_fill_{fill}": short_sell_fill,
                f"long_buy_fill_{fill}": long_buy_fill,
                f"short_entry_cash_{fill}": short_entry_cash,
                f"long_entry_cash_{fill}": long_entry_cash,
                f"gross_credit_{fill}_cash": gross_credit_cash,
                f"commission_{fill}_cash": gross_credit_cash - premium_cash,
                f"commission_{fill}_pct_spot_notional": (
                    gross_credit_cash - premium_cash
                )
                / spot_notional,
                f"premium_{fill}_points": premium_cash / MULTIPLIER,
                f"premium_{fill}_cash": premium_cash,
                f"premium_{fill}_pct_spot_notional": premium_cash / spot_notional,
                f"credit_{fill}_pct_of_width": premium_cash / width_cash,
                f"max_loss_{fill}_cash": maximum_loss,
                f"max_loss_{fill}_pct": maximum_loss / spot_notional,
                f"max_loss_{fill}_pct_spot_notional": maximum_loss / spot_notional,
                f"short_pnl_{fill}_cash": short_pnl,
                f"short_pnl_{fill}_pct_spot_notional": short_pnl / spot_notional,
                f"long_pnl_{fill}_cash": long_pnl,
                f"long_pnl_{fill}_pct_spot_notional": long_pnl / spot_notional,
                f"pnl_{fill}_cash": pnl,
                f"pnl_{fill}_pct_spot_notional": pnl / spot_notional,
                f"pnl_{fill}_dollars_per_1m": pnl / spot_notional * 1_000_000.0,
            }
        )
    if shared_features:
        record.update(shared_features)
    return record


def _missing_grid_rows(
    *,
    entry_date: pd.Timestamp,
    scheduled_entry_friday: pd.Timestamp,
    expiration_date: pd.Timestamp,
    scheduled_expiration_friday: pd.Timestamp,
    candidate_grid: Sequence[tuple[float, float]],
    reason: str,
) -> list[dict[str, object]]:
    return [
        {
            "entry_date": entry_date,
            "scheduled_entry_friday": scheduled_entry_friday,
            "expiration_date": expiration_date,
            "scheduled_expiration_friday": scheduled_expiration_friday,
            "target_short_ratio": short_ratio,
            "target_width_pct": width,
            "target_long_ratio": short_ratio - width,
            "reason": reason,
        }
        for short_ratio, width in candidate_grid
    ]


def build_weekly_candidate_panel(
    archive: SPXSurfaceArchive,
    cash_closes: pd.Series,
    *,
    market_features: pd.DataFrame | None = None,
    candidate_grid: Sequence[tuple[float, float]] = CANDIDATE_GRID,
    progress_every: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return candidates, missing grid points, weekly availability, and skipped weeks."""
    schedule = weekly_roll_schedule(archive.populated_dates)
    if len(schedule) < 2:
        raise RuntimeError("archive does not contain two weekly roll dates")
    if market_features is None:
        market_features = pd.DataFrame(index=cash_closes.index)
    market_features = market_features.copy()
    market_features.index = pd.to_datetime(market_features.index).normalize()

    candidates: list[dict[str, object]] = []
    missing_candidates: list[dict[str, object]] = []
    weekly_records: list[dict[str, object]] = []
    skipped_weeks: list[dict[str, object]] = []

    for number in range(len(schedule) - 1):
        entry = schedule.iloc[number]
        expiry = schedule.iloc[number + 1]
        entry_date = pd.Timestamp(entry["roll_date"])
        expiration_date = pd.Timestamp(expiry["roll_date"])
        scheduled_entry = pd.Timestamp(entry["scheduled_friday"])
        scheduled_expiration = pd.Timestamp(expiry["scheduled_friday"])
        weekly_base = {
            "entry_date": entry_date,
            "scheduled_entry_friday": scheduled_entry,
            "expiration_date": expiration_date,
            "scheduled_expiration_friday": scheduled_expiration,
            "dte": int((expiration_date - entry_date).days),
            "expected_candidates": len(candidate_grid),
        }

        missing_spot_dates = [
            date for date in (entry_date, expiration_date) if date not in cash_closes.index
        ]
        if missing_spot_dates:
            reason = "missing cash SPX close for " + ", ".join(
                date.strftime("%Y-%m-%d") for date in missing_spot_dates
            )
            missing_candidates.extend(
                _missing_grid_rows(
                    entry_date=entry_date,
                    scheduled_entry_friday=scheduled_entry,
                    expiration_date=expiration_date,
                    scheduled_expiration_friday=scheduled_expiration,
                    candidate_grid=candidate_grid,
                    reason=reason,
                )
            )
            weekly_records.append(
                {
                    **weekly_base,
                    "available_candidates": 0,
                    "missing_candidates": len(candidate_grid),
                    "baseline_99_96_available": False,
                }
            )
            skipped_weeks.append({**weekly_base, "reason": reason})
            continue

        spot_entry = float(cash_closes.loc[entry_date])
        spot_expiration = float(cash_closes.loc[expiration_date])
        chain = archive.read(entry_date)
        puts = valid_pm_puts(chain, expiration_date)
        if puts.empty:
            reason = "no valid PM-settled SPXW puts for next weekly expiration"
            missing_candidates.extend(
                _missing_grid_rows(
                    entry_date=entry_date,
                    scheduled_entry_friday=scheduled_entry,
                    expiration_date=expiration_date,
                    scheduled_expiration_friday=scheduled_expiration,
                    candidate_grid=candidate_grid,
                    reason=reason,
                )
            )
            weekly_records.append(
                {
                    **weekly_base,
                    "available_candidates": 0,
                    "missing_candidates": len(candidate_grid),
                    "baseline_99_96_available": False,
                }
            )
            skipped_weeks.append({**weekly_base, "reason": reason})
            continue

        shared = chain_features(
            chain,
            puts,
            entry_date=entry_date,
            expiration_date=expiration_date,
            spot=spot_entry,
        )
        if entry_date in market_features.index:
            shared.update(market_features.loc[entry_date].to_dict())

        week_records: list[dict[str, object]] = []
        for short_ratio, width in candidate_grid:
            try:
                record = build_candidate_record(
                    entry_date=entry_date,
                    scheduled_entry_friday=scheduled_entry,
                    expiration_date=expiration_date,
                    scheduled_expiration_friday=scheduled_expiration,
                    spot_entry=spot_entry,
                    spot_expiration=spot_expiration,
                    short_ratio=short_ratio,
                    width=width,
                    puts=puts,
                    shared_features=shared,
                )
            except (CandidateError, KeyError, TypeError, ValueError) as error:
                missing_candidates.append(
                    {
                        **weekly_base,
                        "target_short_ratio": short_ratio,
                        "target_width_pct": width,
                        "target_long_ratio": short_ratio - width,
                        "reason": str(error),
                    }
                )
            else:
                week_records.append(record)

        baseline_available = any(
            bool(record["is_baseline_99_96"]) for record in week_records
        )
        for record in week_records:
            record["baseline_99_96_available"] = baseline_available
        candidates.extend(week_records)
        weekly_records.append(
            {
                **weekly_base,
                "available_candidates": len(week_records),
                "missing_candidates": len(candidate_grid) - len(week_records),
                "baseline_99_96_available": baseline_available,
            }
        )
        if not week_records:
            skipped_weeks.append(
                {**weekly_base, "reason": "all candidate grid points failed validation"}
            )
        if progress_every and (number + 1) % progress_every == 0:
            print(
                f"Built weekly candidates {number + 1}/{len(schedule) - 1}",
                flush=True,
            )

    candidate_frame = pd.DataFrame(candidates)
    if candidate_frame.empty:
        raise RuntimeError("weekly schedule produced no executable candidates")
    candidate_frame.sort_values(
        ["entry_date", "target_short_ratio", "target_width_pct"], inplace=True
    )
    candidate_frame.reset_index(drop=True, inplace=True)
    return (
        candidate_frame,
        pd.DataFrame(missing_candidates),
        pd.DataFrame(weekly_records),
        pd.DataFrame(skipped_weeks),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = SPXSurfaceArchive(DATA_ROOT)
    cash_closes = load_cash_closes_read_only()
    candidates, missing, weekly, skipped = build_weekly_candidate_panel(
        archive,
        cash_closes,
    )

    candidates.to_parquet(OUT / "candidate_trades.parquet", index=False)
    candidates.to_csv(OUT / "candidate_trades.csv", index=False)
    missing.to_csv(OUT / "missing_candidates.csv", index=False)
    weekly.to_csv(OUT / "weekly_availability.csv", index=False)
    skipped.to_csv(OUT / "skipped_weeks.csv", index=False)

    manifest = {
        "archive_start": str(archive.populated_dates.min().date()),
        "archive_end": str(archive.populated_dates.max().date()),
        "entry_schedule": "calendar Friday, or latest archive session in the prior four calendar days",
        "expiration": "next weekly roll date; PM-settled SPXW puts held to cash settlement",
        "candidate_grid": {
            "short_strike_ratios": list(SHORT_RATIOS),
            "hedge_widths_pct_of_entry_cash_spot": list(HEDGE_WIDTHS),
            "fixed_before_outcome_observation": True,
        },
        "candidate_rows": len(candidates),
        "scheduled_weeks": len(weekly),
        "weeks_with_candidates": int(weekly["available_candidates"].gt(0).sum()),
        "fully_skipped_weeks": len(skipped),
        "missing_grid_points": len(missing),
        "baseline_99_96_available_weeks": int(
            weekly["baseline_99_96_available"].sum()
        ),
        "spot_and_settlement_source": "read-only cached/local Yahoo Finance ^GSPC unadjusted cash close",
        "normalization": "100% of entry cash-SPX spot notional; fractional contracts",
        "realistic_execution": {
            "bid_ask_spread_fraction_away_from_mid": 0.25,
            "commission_per_contract_per_leg": 1.50,
        },
        "natural_execution": {
            "bid_ask_spread_fraction_away_from_mid": 0.50,
            "commission_per_contract_per_leg": 1.50,
        },
        "feature_timing": "option features use the entry snapshot only; lagged VIX, VVIX, and cash technical features are joined by the separate market-feature stage",
        "outcome_fields": [
            "spot_expiration",
            "spx_forward_return",
            "short_terminal_intrinsic_cash",
            "long_terminal_intrinsic_cash",
            "expiration_value_cash",
            "pnl_*",
        ],
        "files": {
            "candidates": ["candidate_trades.parquet", "candidate_trades.csv"],
            "missing_grid_points": "missing_candidates.csv",
            "weekly_availability": "weekly_availability.csv",
            "fully_skipped_weeks": "skipped_weeks.csv",
        },
    }
    (OUT / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Saved {len(candidates):,} candidates across "
        f"{candidates['entry_date'].nunique():,} weeks to {OUT}"
    )


if __name__ == "__main__":
    main()
