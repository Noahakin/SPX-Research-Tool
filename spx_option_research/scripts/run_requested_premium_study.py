from __future__ import annotations

import calendar
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DATA_ROOT = (
    WORKSPACE
    / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
)
OUT = PROJECT / "results/requested_premium_study"
LOCAL_CASH_ROOT = (
    WORKSPACE
    / "05 - Underlying Prices and Events/Cleaned Data/Yahoo Finance/FX External/processed"
    / "provider=yfinance/group=equities/series=sp500"
)
CASH_CACHE = OUT / "spx_cash_close.csv"
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.execution import ExecutionModel


MULTIPLIER = 100.0
MID_MODEL = ExecutionModel(spread_fraction=0.0, commission_per_contract=0.0)
REALISTIC_MODEL = ExecutionModel(spread_fraction=0.25, commission_per_contract=1.50)
NATURAL_MODEL = ExecutionModel(spread_fraction=0.50, commission_per_contract=1.50)


@dataclass(frozen=True)
class Structure:
    label: str
    tenor_months: int
    kind: str
    upper_ratio: float
    lower_ratio: float | None = None


ONE_MONTH_STRUCTURES = (
    Structure("1M 101/98 short put spread", 1, "short_put_spread", 1.01, 0.98),
    Structure("1M 102/99 short put spread", 1, "short_put_spread", 1.02, 0.99),
    Structure("1M 103/100 short put spread", 1, "short_put_spread", 1.03, 1.00),
    Structure("1M 95/85 buffer", 1, "buffer", 0.95, 0.85),
    Structure("1M 97.5/92.5 buffer", 1, "buffer", 0.975, 0.925),
)
THREE_MONTH_STRUCTURE = Structure("3M 90% long put", 3, "long_put", 0.90)


def third_friday(year: int, month: int) -> pd.Timestamp:
    weeks = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in weeks if week[calendar.FRIDAY]]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def monthly_roll_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Third Friday, or the latest available session in the prior three days."""
    available = set(pd.DatetimeIndex(dates).normalize())
    result: list[pd.Timestamp] = []
    for period in pd.period_range(min(available), max(available), freq="M"):
        scheduled = third_friday(period.year, period.month)
        choices = [
            scheduled - pd.Timedelta(days=offset)
            for offset in range(4)
            if scheduled - pd.Timedelta(days=offset) in available
        ]
        if choices:
            result.append(max(choices))
    return sorted(set(result))


def valid_pm_puts(chain: pd.DataFrame, expiration: pd.Timestamp) -> pd.DataFrame:
    valid = (
        chain["option_type"].eq("put")
        & chain["expiration_date"].eq(expiration)
        & chain["settlement"].eq("PM")
        & chain["strike"].gt(0)
        & chain["bid"].ge(0)
        & chain["ask"].ge(chain["bid"])
    )
    return chain.loc[valid].copy()


def nearest_put(puts: pd.DataFrame, target_strike: float) -> pd.Series:
    if puts.empty:
        raise LookupError("no valid PM-settled puts for the requested expiration")
    ranked = puts.assign(
        _distance=(puts["strike"].astype(float) - float(target_strike)).abs()
    ).sort_values(["_distance", "strike", "option_symbol"], kind="stable")
    return ranked.iloc[0]


def signed_legs(
    puts: pd.DataFrame, spot: float, structure: Structure
) -> list[tuple[pd.Series, int, str]]:
    upper = nearest_put(puts, spot * structure.upper_ratio)
    if structure.kind == "long_put":
        return [(upper, 1, "long")]
    if structure.lower_ratio is None:
        raise ValueError(f"{structure.label} requires a lower strike")
    lower = nearest_put(puts, spot * structure.lower_ratio)
    if float(lower["strike"]) >= float(upper["strike"]):
        raise LookupError(f"{structure.label} selected non-descending strikes")
    if structure.kind == "short_put_spread":
        return [(upper, -1, "short"), (lower, 1, "long")]
    if structure.kind == "buffer":
        return [(upper, 1, "long"), (lower, -1, "short")]
    raise ValueError(f"unknown structure kind: {structure.kind}")


def entry_cash(legs: list[tuple[pd.Series, int, str]], model: ExecutionModel) -> float:
    return float(sum(model.cash_flow(quote, quantity) for quote, quantity, _ in legs))


def expiration_value(
    legs: list[tuple[pd.Series, int, str]], expiration_spot: float
) -> float:
    return float(
        sum(
            quantity
            * max(float(quote["strike"]) - float(expiration_spot), 0.0)
            * MULTIPLIER
            for quote, quantity, _ in legs
        )
    )


def structure_record(
    *,
    entry_date: pd.Timestamp,
    expiration_date: pd.Timestamp,
    spot: float,
    structure: Structure,
    puts: pd.DataFrame,
) -> dict[str, object]:
    legs = signed_legs(puts, spot, structure)
    cash_mid = entry_cash(legs, MID_MODEL)
    cash_realistic = entry_cash(legs, REALISTIC_MODEL)
    cash_natural = entry_cash(legs, NATURAL_MODEL)
    premium_mid = cash_mid if structure.kind == "short_put_spread" else -cash_mid
    premium_realistic = (
        cash_realistic if structure.kind == "short_put_spread" else -cash_realistic
    )
    premium_natural = cash_natural if structure.kind == "short_put_spread" else -cash_natural
    if premium_mid <= 0 or premium_realistic <= 0 or premium_natural <= 0:
        raise LookupError(f"{structure.label} did not have the expected entry premium sign")
    upper = legs[0][0]
    lower = legs[1][0] if len(legs) == 2 else None
    record: dict[str, object] = {
        "strategy": structure.label,
        "kind": structure.kind,
        "entry_date": entry_date,
        "expiration_date": expiration_date,
        "dte": int((expiration_date - entry_date).days),
        "spot_entry": spot,
        "target_upper_ratio": structure.upper_ratio,
        "target_lower_ratio": structure.lower_ratio,
        "upper_strike": float(upper["strike"]),
        "lower_strike": float(lower["strike"]) if lower is not None else np.nan,
        "actual_upper_ratio": float(upper["strike"]) / spot,
        "actual_lower_ratio": float(lower["strike"]) / spot if lower is not None else np.nan,
        "upper_symbol": str(upper["option_symbol"]).strip(),
        "lower_symbol": str(lower["option_symbol"]).strip() if lower is not None else None,
        "upper_bid": float(upper["bid"]),
        "upper_ask": float(upper["ask"]),
        "lower_bid": float(lower["bid"]) if lower is not None else np.nan,
        "lower_ask": float(lower["ask"]) if lower is not None else np.nan,
        "premium_mid_cash": premium_mid,
        "premium_realistic_cash": premium_realistic,
        "premium_natural_cash": premium_natural,
        "premium_mid_points": premium_mid / MULTIPLIER,
        "premium_realistic_points": premium_realistic / MULTIPLIER,
        "premium_natural_points": premium_natural / MULTIPLIER,
        "premium_mid_pct_spot_notional": premium_mid / (spot * MULTIPLIER),
        "premium_realistic_pct_spot_notional": premium_realistic
        / (spot * MULTIPLIER),
        "premium_natural_pct_spot_notional": premium_natural / (spot * MULTIPLIER),
    }
    record["_legs"] = legs
    record["_entry_cash_mid"] = cash_mid
    record["_entry_cash_realistic"] = cash_realistic
    record["_entry_cash_natural"] = cash_natural
    return record


def load_spx_cash_closes(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Load cash SPX closes locally and refresh the uncovered tail from Yahoo."""
    frames: list[pd.DataFrame] = []
    if CASH_CACHE.exists():
        cached = pd.read_csv(CASH_CACHE, parse_dates=["date"])
        frames.append(cached[["date", "spx_close"]])
    if LOCAL_CASH_ROOT.exists():
        for path in sorted(LOCAL_CASH_ROOT.glob("year=*/sp500_*.parquet")):
            local = pd.read_parquet(path, columns=["observation_date", "close"])
            local = local.rename(columns={"observation_date": "date", "close": "spx_close"})
            frames.append(local)
    if not frames:
        combined = pd.DataFrame(columns=["date", "spx_close"])
    else:
        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
        combined["spx_close"] = pd.to_numeric(combined["spx_close"], errors="coerce")
        combined = combined.dropna().drop_duplicates("date", keep="last")

    covered_through = combined["date"].max() if not combined.empty else start - pd.Timedelta(days=1)
    if pd.isna(covered_through) or covered_through < end:
        try:
            import yfinance as yf
        except ImportError as error:
            raise RuntimeError("yfinance is required to complete the SPX cash-close history") from error
        download_start = max(start, pd.Timestamp(covered_through) + pd.Timedelta(days=1))
        downloaded = yf.download(
            "^GSPC",
            start=download_start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
        if downloaded.empty:
            raise RuntimeError(
                f"Yahoo returned no SPX closes for {download_start:%Y-%m-%d} through {end:%Y-%m-%d}"
            )
        close = downloaded["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        refreshed = close.rename("spx_close").rename_axis("date").reset_index()
        refreshed["date"] = pd.to_datetime(refreshed["date"]).dt.tz_localize(None).dt.normalize()
        combined = pd.concat([combined, refreshed], ignore_index=True)
        combined = combined.dropna().drop_duplicates("date", keep="last")

    combined = combined.sort_values("date")
    selected = combined[combined["date"].between(start, end)].copy()
    if selected.empty or selected["date"].max() < end:
        raise RuntimeError(f"cash SPX close history does not reach {end:%Y-%m-%d}")
    selected.to_csv(CASH_CACHE, index=False)
    return selected.set_index("date")["spx_close"].astype(float)


def annualized_rate(frame: pd.DataFrame, column: str) -> float:
    contract_years = float((frame["dte"] / 365.2425).sum())
    if contract_years <= 0:
        raise ValueError("contract years must be positive")
    return float(frame[column].sum() / contract_years)


def compounded_rate(frame: pd.DataFrame, column: str) -> float:
    ordered = frame.sort_values("entry_date")
    elapsed_years = (
        pd.Timestamp(ordered["expiration_date"].max())
        - pd.Timestamp(ordered["entry_date"].min())
    ).days / 365.2425
    wealth = float((1.0 + ordered[column]).prod())
    return wealth ** (1.0 / elapsed_years) - 1.0


def build_samples(
    archive: SPXSurfaceArchive,
    cash_closes: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rolls = monthly_roll_dates(archive.populated_dates)
    one_month: list[dict[str, object]] = []
    three_month: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []

    for index, entry_date in enumerate(rolls):
        chain = archive.read(entry_date)
        if entry_date not in cash_closes.index:
            skips.append(
                {"tenor": "all", "entry_date": entry_date, "expiration_date": None, "reason": "missing cash SPX entry close"}
            )
            continue
        spot = float(cash_closes.loc[entry_date])

        if index + 1 < len(rolls):
            expiration = rolls[index + 1]
            puts = valid_pm_puts(chain, expiration)
            records: list[dict[str, object]] = []
            try:
                if puts.empty:
                    raise LookupError("no valid PM-settled puts for the requested expiration")
                for structure in ONE_MONTH_STRUCTURES:
                    records.append(
                        structure_record(
                            entry_date=entry_date,
                            expiration_date=expiration,
                            spot=spot,
                            structure=structure,
                            puts=puts,
                        )
                    )
            except (LookupError, ValueError) as error:
                skips.append(
                    {
                        "tenor": "1M",
                        "entry_date": entry_date,
                        "expiration_date": expiration,
                        "reason": str(error),
                    }
                )
            else:
                one_month.extend(records)

        if index + 3 < len(rolls):
            expiration = rolls[index + 3]
            puts = valid_pm_puts(chain, expiration)
            try:
                if puts.empty:
                    raise LookupError("no valid PM-settled puts for the requested expiration")
                three_month.append(
                    structure_record(
                        entry_date=entry_date,
                        expiration_date=expiration,
                        spot=spot,
                        structure=THREE_MONTH_STRUCTURE,
                        puts=puts,
                    )
                )
            except (LookupError, ValueError) as error:
                skips.append(
                    {
                        "tenor": "3M",
                        "entry_date": entry_date,
                        "expiration_date": expiration,
                        "reason": str(error),
                    }
                )

    one_month_frame = pd.DataFrame(one_month)
    three_month_frame = pd.DataFrame(three_month)
    if one_month_frame.empty or three_month_frame.empty:
        raise RuntimeError("the archive produced no usable strategy observations")

    for frame in (one_month_frame, three_month_frame):
        frame["spot_expiration"] = frame["expiration_date"].map(cash_closes)
        missing = frame["spot_expiration"].isna()
        if missing.any():
            missing_dates = sorted(frame.loc[missing, "expiration_date"].dt.strftime("%Y-%m-%d").unique())
            raise RuntimeError(f"missing expiration spot for {missing_dates}")
        expiry_values = []
        pnl_mid = []
        pnl_realistic = []
        pnl_natural = []
        for row in frame.to_dict("records"):
            value = expiration_value(row["_legs"], float(row["spot_expiration"]))
            expiry_values.append(value)
            pnl_mid.append(float(row["_entry_cash_mid"]) + value)
            pnl_realistic.append(float(row["_entry_cash_realistic"]) + value)
            pnl_natural.append(float(row["_entry_cash_natural"]) + value)
        frame["expiration_value_cash"] = expiry_values
        frame["pnl_mid_cash"] = pnl_mid
        frame["pnl_realistic_cash"] = pnl_realistic
        frame["pnl_natural_cash"] = pnl_natural
        frame["pnl_mid_pct_spot_notional"] = frame["pnl_mid_cash"] / (
            frame["spot_entry"] * MULTIPLIER
        )
        frame["pnl_realistic_pct_spot_notional"] = frame["pnl_realistic_cash"] / (
            frame["spot_entry"] * MULTIPLIER
        )
        frame["pnl_natural_pct_spot_notional"] = frame["pnl_natural_cash"] / (
            frame["spot_entry"] * MULTIPLIER
        )
        frame.drop(
            columns=["_legs", "_entry_cash_mid", "_entry_cash_realistic", "_entry_cash_natural"],
            inplace=True,
        )

    return one_month_frame, three_month_frame, pd.DataFrame(skips)


def summarize(one_month: pd.DataFrame, three_month: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, group in one_month.groupby("strategy", sort=False):
        kind = str(group["kind"].iloc[0])
        row: dict[str, object] = {
            "strategy": strategy,
            "kind": kind,
            "observations": len(group),
            "first_entry": group["entry_date"].min(),
            "last_expiration": group["expiration_date"].max(),
            "average_dte": float(group["dte"].mean()),
            "average_actual_upper_pct": float(group["actual_upper_ratio"].mean() * 100.0),
            "average_actual_lower_pct": float(group["actual_lower_ratio"].mean() * 100.0),
            "average_premium_mid_points": float(group["premium_mid_points"].mean()),
            "average_premium_realistic_points": float(group["premium_realistic_points"].mean()),
            "average_premium_natural_points": float(group["premium_natural_points"].mean()),
            "annualized_premium_mid_pct_spot_notional": annualized_rate(
                group, "premium_mid_pct_spot_notional"
            ),
            "annualized_premium_realistic_pct_spot_notional": annualized_rate(
                group, "premium_realistic_pct_spot_notional"
            ),
            "annualized_premium_natural_pct_spot_notional": annualized_rate(
                group, "premium_natural_pct_spot_notional"
            ),
            "annualized_pnl_mid_pct_spot_notional": annualized_rate(
                group, "pnl_mid_pct_spot_notional"
            ),
            "annualized_pnl_realistic_pct_spot_notional": annualized_rate(
                group, "pnl_realistic_pct_spot_notional"
            ),
            "annualized_pnl_natural_pct_spot_notional": annualized_rate(
                group, "pnl_natural_pct_spot_notional"
            ),
            "compounded_pnl_mid_cagr": compounded_rate(
                group, "pnl_mid_pct_spot_notional"
            ),
            "compounded_pnl_realistic_cagr": compounded_rate(
                group, "pnl_realistic_pct_spot_notional"
            ),
            "compounded_pnl_natural_cagr": compounded_rate(
                group, "pnl_natural_pct_spot_notional"
            ),
            "win_rate_realistic": float(group["pnl_realistic_cash"].gt(0).mean()),
        }
        rows.append(row)

    group = three_month
    rows.append(
        {
            "strategy": THREE_MONTH_STRUCTURE.label,
            "kind": THREE_MONTH_STRUCTURE.kind,
            "observations": len(group),
            "first_entry": group["entry_date"].min(),
            "last_expiration": group["expiration_date"].max(),
            "average_dte": float(group["dte"].mean()),
            "average_actual_upper_pct": float(group["actual_upper_ratio"].mean() * 100.0),
            "average_actual_lower_pct": np.nan,
            "average_premium_mid_points": float(group["premium_mid_points"].mean()),
            "average_premium_realistic_points": float(group["premium_realistic_points"].mean()),
            "average_premium_natural_points": float(group["premium_natural_points"].mean()),
            "annualized_premium_mid_pct_spot_notional": annualized_rate(
                group, "premium_mid_pct_spot_notional"
            ),
            "annualized_premium_realistic_pct_spot_notional": annualized_rate(
                group, "premium_realistic_pct_spot_notional"
            ),
            "annualized_premium_natural_pct_spot_notional": annualized_rate(
                group, "premium_natural_pct_spot_notional"
            ),
            "annualized_pnl_mid_pct_spot_notional": annualized_rate(
                group, "pnl_mid_pct_spot_notional"
            ),
            "annualized_pnl_realistic_pct_spot_notional": annualized_rate(
                group, "pnl_realistic_pct_spot_notional"
            ),
            "annualized_pnl_natural_pct_spot_notional": annualized_rate(
                group, "pnl_natural_pct_spot_notional"
            ),
            "compounded_pnl_mid_cagr": np.nan,
            "compounded_pnl_realistic_cagr": np.nan,
            "compounded_pnl_natural_cagr": np.nan,
            "win_rate_realistic": float(group["pnl_realistic_cash"].gt(0).mean()),
        }
    )
    summary = pd.DataFrame(rows)
    dollar_columns = {
        "annualized_premium_mid_pct_spot_notional": "annualized_premium_mid_dollars_per_1m",
        "annualized_premium_realistic_pct_spot_notional": "annualized_premium_realistic_dollars_per_1m",
        "annualized_premium_natural_pct_spot_notional": "annualized_premium_natural_dollars_per_1m",
        "annualized_pnl_mid_pct_spot_notional": "annualized_pnl_mid_dollars_per_1m",
        "annualized_pnl_realistic_pct_spot_notional": "annualized_pnl_realistic_dollars_per_1m",
        "annualized_pnl_natural_pct_spot_notional": "annualized_pnl_natural_dollars_per_1m",
    }
    for rate_column, dollar_column in dollar_columns.items():
        summary[dollar_column] = summary[rate_column] * 1_000_000.0
    return summary


def yearly_results(one_month: pd.DataFrame) -> pd.DataFrame:
    income = one_month[one_month["kind"].eq("short_put_spread")].copy()
    income["year"] = income["expiration_date"].dt.year
    rows: list[dict[str, object]] = []
    for (strategy, year), group in income.groupby(["strategy", "year"], sort=True):
        rows.append(
            {
                "strategy": strategy,
                "year": int(year),
                "trades": len(group),
                "premium_realistic_pct_spot_notional": float(
                    group["premium_realistic_pct_spot_notional"].sum()
                ),
                "pnl_realistic_pct_spot_notional": float(
                    group["pnl_realistic_pct_spot_notional"].sum()
                ),
                "wins": int(group["pnl_realistic_cash"].gt(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def yearly_hedge_results(one_month: pd.DataFrame) -> pd.DataFrame:
    hedges = one_month[one_month["kind"].eq("buffer")].copy()
    hedges["year"] = hedges["expiration_date"].dt.year
    rows: list[dict[str, object]] = []
    for (strategy, year), group in hedges.groupby(["strategy", "year"], sort=True):
        rows.append(
            {
                "strategy": strategy,
                "year": int(year),
                "trades": len(group),
                "premium_paid_realistic_pct_spot_notional": float(
                    group["premium_realistic_pct_spot_notional"].sum()
                ),
                "pnl_realistic_pct_spot_notional": float(
                    group["pnl_realistic_pct_spot_notional"].sum()
                ),
                "profitable_trades": int(group["pnl_realistic_cash"].gt(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def three_month_roll_cohorts(three_month: pd.DataFrame) -> pd.DataFrame:
    """Three staggered, non-overlapping quarterly roll schedules."""
    ordered = three_month.sort_values("entry_date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for phase in range(3):
        group = ordered.iloc[phase::3].copy()
        row: dict[str, object] = {
            "roll_phase": phase + 1,
            "observations": len(group),
            "first_entry": group["entry_date"].min(),
            "last_expiration": group["expiration_date"].max(),
            "average_dte": float(group["dte"].mean()),
            "win_rate_realistic": float(group["pnl_realistic_cash"].gt(0).mean()),
        }
        for fill in ("mid", "realistic", "natural"):
            row[f"annualized_premium_{fill}_pct_spot_notional"] = annualized_rate(
                group, f"premium_{fill}_pct_spot_notional"
            )
            row[f"annualized_pnl_{fill}_pct_spot_notional"] = annualized_rate(
                group, f"pnl_{fill}_pct_spot_notional"
            )
            row[f"compounded_pnl_{fill}_cagr"] = compounded_rate(
                group, f"pnl_{fill}_pct_spot_notional"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    one_month: pd.DataFrame,
    three_month: pd.DataFrame,
    three_month_cohorts: pd.DataFrame,
) -> None:
    def format_dollars(value: float) -> str:
        return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"

    indexed = summary.set_index("strategy")
    income = indexed.loc[[structure.label for structure in ONE_MONTH_STRUCTURES[:3]]]
    protection = indexed.loc[
        [ONE_MONTH_STRUCTURES[3].label, ONE_MONTH_STRUCTURES[4].label, THREE_MONTH_STRUCTURE.label]
    ]

    income_lines = []
    for label, row in income.iterrows():
        income_lines.append(
            f"| {label.replace(' short put spread', '')} | "
            f"{row['annualized_premium_realistic_pct_spot_notional']:.2%} | "
            f"{format_dollars(row['annualized_premium_realistic_dollars_per_1m'])} | "
            f"{row['annualized_pnl_realistic_pct_spot_notional']:.2%} | "
            f"{format_dollars(row['annualized_pnl_realistic_dollars_per_1m'])} | "
            f"{row['annualized_pnl_natural_pct_spot_notional']:.2%} | "
            f"{row['compounded_pnl_realistic_cagr']:.2%} | "
            f"{row['win_rate_realistic']:.1%} |"
        )
    protection_lines = []
    for label, row in protection.iterrows():
        if label == THREE_MONTH_STRUCTURE.label:
            cagr = float(three_month_cohorts["compounded_pnl_realistic_cagr"].mean())
            cagr_text = f"{cagr:.2%} average"
        else:
            cagr_text = f"{row['compounded_pnl_realistic_cagr']:.2%}"
        protection_lines.append(
            f"| {label} | {row['annualized_premium_realistic_pct_spot_notional']:.2%} | "
            f"{format_dollars(row['annualized_premium_realistic_dollars_per_1m'])} | "
            f"{row['annualized_pnl_realistic_pct_spot_notional']:.2%} | "
            f"{format_dollars(row['annualized_pnl_realistic_dollars_per_1m'])} | "
            f"{cagr_text} | {row['win_rate_realistic']:.1%} |"
        )

    report = f"""# Requested SPX premium study

The source archive runs from September 22, 2016 through September 22, 2026. The 1M tests use matched third-Friday-to-third-Friday PM-settled SPXW positions held to expiration. There are {one_month.groupby('strategy').size().min()} complete matched 1M cycles, from {one_month['entry_date'].min():%B %d, %Y} through {one_month['expiration_date'].max():%B %d, %Y}, with an average tenor of {one_month['dte'].mean():.1f} calendar days. Strike targets and expiration payoffs use the cash SPX close from Yahoo Finance; the option archive's expiration-specific `underlying_price` field is not used as cash spot.

All percentages and dollar equivalents use SPX spot notional at entry. For example, $1 million of notional means `contracts = $1,000,000 / (SPX × 100)`, with fractional contracts used solely to make results comparable. Annualized rates equal the sum of each normalized trade amount divided by total contract-years. The realistic execution case fills each leg one-quarter of the full quoted bid/ask spread away from mid and charges $1.50 per contract per leg. Expiration uses intrinsic value at that day's cached cash SPX close. Results are option-only and exclude interest on collateral, taxes, and settlement fees.

## Short put spreads

| Structure | Annual premium collected | Per $1M/year | Annual realized P&L | Per $1M/year | Natural-fill P&L | Compounded P&L CAGR | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(income_lines)}

## Protection premium and P&L

| Structure | Annual premium paid | Per $1M/year | Annual realized P&L | Per $1M/year | P&L CAGR | Profitable trades |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(protection_lines)}

The 3M premium and arithmetic P&L estimates use {len(three_month)} monthly entry observations with an average tenor of {three_month['dte'].mean():.1f} days. Dividing the normalized results by total contract-years makes them phase-neutral annual estimates for continuously maintaining one 3M 90% put; it does not assume twelve overlapping puts. The three actual non-overlapping quarterly roll cohorts produced realistic P&L CAGRs from {three_month_cohorts['compounded_pnl_realistic_cagr'].min():.2%} to {three_month_cohorts['compounded_pnl_realistic_cagr'].max():.2%}, averaging {three_month_cohorts['compounded_pnl_realistic_cagr'].mean():.2%}.

Premium-cost sensitivity from mid-market to natural bid/ask was {protection.iloc[0]['annualized_premium_mid_pct_spot_notional']:.2%}–{protection.iloc[0]['annualized_premium_natural_pct_spot_notional']:.2%} for the 95/85 buffer, {protection.iloc[1]['annualized_premium_mid_pct_spot_notional']:.2%}–{protection.iloc[1]['annualized_premium_natural_pct_spot_notional']:.2%} for the 97.5/92.5 buffer, and {protection.iloc[2]['annualized_premium_mid_pct_spot_notional']:.2%}–{protection.iloc[2]['annualized_premium_natural_pct_spot_notional']:.2%} for the 3M put.

“Premium collected” is entry cash flow and includes intrinsic value in the 101%–103% short strikes. Realized P&L subtracts the cash-settled expiration payoff, which is why it is much smaller than gross premium.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = SPXSurfaceArchive(DATA_ROOT)
    cash_closes = load_spx_cash_closes(archive.populated_dates.min(), archive.populated_dates.max())
    one_month, three_month, skips = build_samples(archive, cash_closes)
    summary = summarize(one_month, three_month)
    yearly = yearly_results(one_month)
    yearly_hedges = yearly_hedge_results(one_month)
    three_month_cohorts = three_month_roll_cohorts(three_month)

    one_month.to_csv(OUT / "one_month_trades.csv", index=False)
    one_month.to_parquet(OUT / "one_month_trades.parquet", index=False)
    three_month.to_csv(OUT / "three_month_put_quotes.csv", index=False)
    three_month.to_parquet(OUT / "three_month_put_quotes.parquet", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    yearly.to_csv(OUT / "yearly_short_spread_results.csv", index=False)
    yearly_hedges.to_csv(OUT / "yearly_hedge_results.csv", index=False)
    three_month_cohorts.to_csv(OUT / "three_month_quarterly_roll_cohorts.csv", index=False)
    skips.to_csv(OUT / "skips.csv", index=False)
    write_report(summary, one_month, three_month, three_month_cohorts)

    manifest = {
        "archive_start": str(archive.populated_dates.min().date()),
        "archive_end": str(archive.populated_dates.max().date()),
        "one_month_matched_cycles": int(one_month.groupby("strategy").size().min()),
        "three_month_quote_observations": len(three_month),
        "entry_schedule": "third Friday or latest available session in the prior three days",
        "spot_and_settlement_source": "Yahoo Finance ^GSPC unadjusted cash close",
        "cash_close_start": str(cash_closes.index.min().date()),
        "cash_close_end": str(cash_closes.index.max().date()),
        "cash_close_observations": len(cash_closes),
        "cash_close_cache": str(CASH_CACHE.relative_to(PROJECT)),
        "option_settlement": "PM-settled SPXW; intrinsic value at expiration-day cash SPX close",
        "normalization": "SPX spot notional at entry",
        "annualization": "sum of normalized amounts divided by total contract-years",
        "realistic_execution": {
            "bid_ask_spread_fraction_away_from_mid": 0.25,
            "commission_per_contract_per_leg": 1.50,
        },
        "natural_execution": {
            "bid_ask_spread_fraction_away_from_mid": 0.50,
            "commission_per_contract_per_leg": 1.50,
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
