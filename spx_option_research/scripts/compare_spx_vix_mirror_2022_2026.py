from __future__ import annotations

import calendar
import math
import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
SPX_CACHE = PROJECT / "data/cache/itm_structures"
VIX_ROOT = WORKSPACE / "_inventory/vix_options"
OUT = PROJECT / "results/spx_vix_mirror_2022_2026"

INITIAL_CAPITAL = 1_000_000.0
START = pd.Timestamp("2022-01-03")
END = pd.Timestamp("2026-07-27")
TARGET_DTE = 60
DTE_TOLERANCE = 4
PROFIT_LEVELS = [15, 25, 40, 60, 80]
SPX_PARAMETER_ID = "vertical_103/100_monthly_dte60_m103"
MULTIPLIER = 100.0
SPREAD_FRACTION = 0.25
COMMISSION = 1.50

VIX_COLUMNS = [
    "snapshot_date",
    "expiration_date",
    "dte",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "delta",
    "underlying_price",
    "option_symbol",
    "volume",
    "open_interest",
]


def build_vix_paths() -> dict[pd.Timestamp, Path]:
    paths: dict[pd.Timestamp, Path] = {}
    for path in VIX_ROOT.rglob("eod.parquet"):
        match = re.search(r"date=(\d{4}-\d{2}-\d{2})", str(path))
        if match and path.stat().st_size > 10_000:
            date = pd.Timestamp(match.group(1))
            if START <= date <= END:
                paths[date] = path
    return paths


def read_vix_chain(path: Path) -> pd.DataFrame:
    chain = pd.read_parquet(path, columns=VIX_COLUMNS)
    chain["expiration_date"] = pd.to_datetime(chain["expiration_date"])
    return chain


def valid_calls(chain: pd.DataFrame) -> pd.DataFrame:
    symbols = chain["option_symbol"].fillna("").astype(str)
    return chain.loc[
        chain["option_type"].astype(str).str.lower().eq("call")
        & symbols.str.match(r"^VIX")
        & chain["bid"].ge(0)
        & chain["ask"].gt(0)
        & chain["ask"].ge(chain["bid"])
    ].copy()


def executable_price(quote: pd.Series, side: str) -> float:
    mid = float(quote["mid"])
    spread = float(quote["ask"] - quote["bid"])
    if side == "buy":
        return max(0.0, mid + SPREAD_FRACTION * spread)
    if side == "sell":
        return max(0.0, mid - SPREAD_FRACTION * spread)
    raise ValueError(side)


def build_vix_entries(
    entry_dates: list[pd.Timestamp], paths: dict[pd.Timestamp, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for entry_date in entry_dates:
        if entry_date not in paths:
            skips.append({"entry_date": entry_date, "reason": "missing VIX surface"})
            continue
        calls = valid_calls(read_vix_chain(paths[entry_date]))
        expirations = (
            calls[["expiration_date", "dte"]]
            .drop_duplicates()
            .loc[lambda x: x["dte"].between(TARGET_DTE - DTE_TOLERANCE, TARGET_DTE + DTE_TOLERANCE)]
        )
        expirations = expirations[expirations["expiration_date"].le(END)]
        if expirations.empty:
            skips.append({"entry_date": entry_date, "reason": "no completed expiry near 60 DTE"})
            continue
        expirations = expirations.assign(
            distance=(expirations["dte"] - TARGET_DTE).abs()
        ).sort_values(["distance", "dte", "expiration_date"])
        expiration = pd.Timestamp(expirations.iloc[0]["expiration_date"])
        actual_dte = int(expirations.iloc[0]["dte"])
        expiry_calls = calls[calls["expiration_date"].eq(expiration)].copy()
        spot = float(expiry_calls["underlying_price"].dropna().median())

        # Mirror of the SPX short 103% / long 100% put spread:
        # short a roughly 97% call and buy a call roughly 3% of spot higher.
        short = expiry_calls.loc[
            (expiry_calls["strike"].astype(float) - 0.97 * spot).abs().idxmin()
        ]
        short_strike = float(short["strike"])
        higher = expiry_calls[expiry_calls["strike"].gt(short_strike)]
        if higher.empty:
            skips.append({"entry_date": entry_date, "reason": "no higher long-call strike"})
            continue
        long_target = short_strike + 0.03 * spot
        long = higher.loc[(higher["strike"].astype(float) - long_target).abs().idxmin()]
        long_strike = float(long["strike"])

        short_sale = executable_price(short, "sell") * MULTIPLIER - COMMISSION
        long_purchase = executable_price(long, "buy") * MULTIPLIER + COMMISSION
        entry_credit = short_sale - long_purchase
        width_cash = (long_strike - short_strike) * MULTIPLIER
        max_loss = width_cash - entry_credit
        if entry_credit <= 0 or max_loss <= 0:
            skips.append(
                {
                    "entry_date": entry_date,
                    "reason": "nonpositive credit or maximum loss",
                }
            )
            continue
        trade_id = f"vix_97_100_call_{entry_date:%Y%m%d}"
        rows.append(
            {
                "trade_id": trade_id,
                "asset": "VIX",
                "structure": "97/100 call credit spread",
                "entry_date": entry_date,
                "expiration_date": expiration,
                "actual_dte": actual_dte,
                "spot_entry": spot,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "short_strike_ratio": short_strike / spot,
                "long_strike_ratio": long_strike / spot,
                "short_delta": float(short["delta"]),
                "long_delta": float(long["delta"]),
                "short_symbol": str(short["option_symbol"]),
                "long_symbol": str(long["option_symbol"]),
                "entry_credit_per_spread": entry_credit,
                "max_profit_per_spread": entry_credit,
                "max_loss_per_spread": max_loss,
                "width_cash": width_cash,
                "entry_short_bid": float(short["bid"]),
                "entry_short_ask": float(short["ask"]),
                "entry_long_bid": float(long["bid"]),
                "entry_long_ask": float(long["ask"]),
                "entry_volume": float(short["volume"]) + float(long["volume"]),
                "entry_open_interest": float(short["open_interest"])
                + float(long["open_interest"]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(skips)


def build_vix_marks(entries: pd.DataFrame) -> pd.DataFrame:
    legs = pd.concat(
        [
            entries[
                ["trade_id", "entry_date", "expiration_date", "short_symbol"]
            ].rename(columns={"short_symbol": "option_symbol"}).assign(role="short"),
            entries[
                ["trade_id", "entry_date", "expiration_date", "long_symbol"]
            ].rename(columns={"long_symbol": "option_symbol"}).assign(role="long"),
        ],
        ignore_index=True,
    )
    glob = (VIX_ROOT / "year=*/month=*/date=*/eod.parquet").resolve().as_posix()
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.register("selected_legs", legs)
    quotes = con.execute(
        f"""
        SELECT
            l.trade_id, l.role, CAST(q.snapshot_date AS DATE) AS mark_date,
            q.bid, q.ask, q.mid, q.underlying_price, q.dte
        FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true) q
        JOIN selected_legs l
          ON CAST(q.option_symbol AS VARCHAR) = l.option_symbol
         AND CAST(q.snapshot_date AS DATE)
             BETWEEN CAST(l.entry_date AS DATE) AND CAST(l.expiration_date AS DATE)
        WHERE lower(CAST(q.option_type AS VARCHAR)) = 'call'
        ORDER BY l.trade_id, mark_date, l.role
        """
    ).fetchdf()
    con.close()
    quotes["mark_date"] = pd.to_datetime(quotes["mark_date"])
    quotes = quotes[
        quotes["bid"].ge(0)
        & quotes["ask"].gt(0)
        & quotes["ask"].ge(quotes["bid"])
    ].copy()
    short = quotes[quotes["role"].eq("short")].drop(columns="role").add_prefix("short_")
    short = short.rename(
        columns={"short_trade_id": "trade_id", "short_mark_date": "mark_date"}
    )
    long = quotes[quotes["role"].eq("long")].drop(columns="role").add_prefix("long_")
    long = long.rename(
        columns={"long_trade_id": "trade_id", "long_mark_date": "mark_date"}
    )
    marks = short.merge(long, on=["trade_id", "mark_date"], how="inner")
    metadata = entries[
        [
            "trade_id",
            "entry_date",
            "expiration_date",
            "entry_credit_per_spread",
            "max_profit_per_spread",
            "max_loss_per_spread",
        ]
    ]
    marks = marks.merge(metadata, on="trade_id", how="left")
    marks["pnl_mid"] = (
        marks["entry_credit_per_spread"]
        + (marks["long_mid"] - marks["short_mid"]) * MULTIPLIER
    )
    close_short = -(
        marks["short_mid"]
        + SPREAD_FRACTION * (marks["short_ask"] - marks["short_bid"])
    ) * MULTIPLIER - COMMISSION
    close_long_price = np.maximum(
        0.0,
        marks["long_mid"]
        - SPREAD_FRACTION * (marks["long_ask"] - marks["long_bid"]),
    )
    close_long = close_long_price * MULTIPLIER - COMMISSION
    marks["pnl_realistic"] = (
        marks["entry_credit_per_spread"] + close_short + close_long
    )
    # Individual VIX leg quotes can briefly imply a package value outside the
    # vertical's arbitrage bounds. A trader can cap the close debit at the
    # strike width (or allow the European spread to settle), so enforce the
    # defined payoff bounds on both midpoint and executable marks.
    for column in ("pnl_mid", "pnl_realistic"):
        marks[column] = np.minimum(
            marks["max_profit_per_spread"],
            np.maximum(-marks["max_loss_per_spread"], marks[column]),
        )
    marks["current_dte"] = np.minimum(
        marks["short_dte"], marks["long_dte"]
    ).astype(int)
    return marks.sort_values(["trade_id", "mark_date"]).reset_index(drop=True)


def load_spx_data(entry_dates: set[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_parquet(SPX_CACHE / "trades.parquet")
    trades = trades[trades["parameter_id"].eq(SPX_PARAMETER_ID)].copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["expiration_date"] = pd.to_datetime(trades["expiration_date"])
    trades = trades[
        trades["entry_date"].isin(entry_dates)
        & trades["expiration_date"].le(END)
    ].copy()
    trades["asset"] = "SPX"
    trades["structure"] = "103/100 put credit spread"
    trades["short_strike"] = trades["spot_entry"] * trades["short_strike_ratio"]
    trades["long_strike"] = trades["short_strike"] - (
        trades["max_profit_per_unit"] + trades["max_loss_per_unit"]
    ) / MULTIPLIER
    trades = trades.rename(
        columns={
            "entry_cash_per_unit": "entry_credit_per_spread",
            "max_profit_per_unit": "max_profit_per_spread",
            "max_loss_per_unit": "max_loss_per_spread",
        }
    )
    marks = pd.read_parquet(
        SPX_CACHE / "trade_marks.parquet",
        filters=[("parameter_id", "==", SPX_PARAMETER_ID)],
    )
    marks["mark_date"] = pd.to_datetime(marks["mark_date"])
    marks["entry_date"] = pd.to_datetime(marks["entry_date"])
    marks = marks[
        marks["trade_id"].isin(trades["trade_id"])
        & marks["mark_date"].between(START, END)
    ].copy()
    marks = marks.rename(
        columns={
            "entry_cash_per_unit": "entry_credit_per_spread",
            "max_profit_per_unit": "max_profit_per_spread",
            "max_loss_per_unit": "max_loss_per_spread",
        }
    )
    keep = [
        "trade_id",
        "mark_date",
        "entry_date",
        "expiration_date",
        "entry_credit_per_spread",
        "max_profit_per_spread",
        "max_loss_per_spread",
        "pnl_mid",
        "pnl_realistic",
        "current_dte",
    ]
    return trades.sort_values("entry_date").reset_index(drop=True), marks[keep]


def build_exit_paths(
    trades: pd.DataFrame,
    marks: pd.DataFrame,
    profit_level: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fraction = profit_level / 100.0
    paths: list[pd.DataFrame] = []
    exits: list[dict[str, object]] = []
    marks_by_trade = {
        trade_id: group.sort_values("mark_date")
        for trade_id, group in marks.groupby("trade_id", sort=False)
    }
    for trade in trades.itertuples():
        if trade.trade_id not in marks_by_trade:
            continue
        trade_marks = marks_by_trade[trade.trade_id]
        eligible = trade_marks[
            trade_marks["mark_date"].gt(pd.Timestamp(trade.entry_date))
            & trade_marks["pnl_realistic"].ge(
                fraction * float(trade.max_profit_per_spread)
            )
        ]
        exit_row = eligible.iloc[0] if not eligible.empty else trade_marks.iloc[-1]
        path = trade_marks[
            trade_marks["mark_date"].le(exit_row["mark_date"])
        ][["trade_id", "mark_date", "pnl_mid"]].copy()
        path = path.sort_values("mark_date")
        path["cumulative_pnl_per_spread"] = path["pnl_mid"].astype(float)
        path.loc[path.index[-1], "cumulative_pnl_per_spread"] = float(
            exit_row["pnl_realistic"]
        )
        path["daily_pnl_per_spread"] = path["cumulative_pnl_per_spread"].diff().fillna(
            path["cumulative_pnl_per_spread"]
        )
        paths.append(path[["trade_id", "mark_date", "daily_pnl_per_spread"]])
        exits.append(
            {
                "trade_id": trade.trade_id,
                "exit_date": exit_row["mark_date"],
                "exit_dte": int(exit_row["current_dte"]),
                "realized_pnl_per_spread": float(exit_row["pnl_realistic"]),
                "hit_profit_target": bool(not eligible.empty),
            }
        )
    return pd.concat(paths, ignore_index=True), pd.DataFrame(exits)


def simulate(
    asset: str,
    trades: pd.DataFrame,
    paths: pd.DataFrame,
    exits: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    profit_level: int,
) -> tuple[pd.Series, pd.DataFrame, dict[str, object]]:
    trades = trades.merge(exits, on="trade_id", how="inner")
    paths_by_date = {
        date: group[["trade_id", "daily_pnl_per_spread"]]
        for date, group in paths.groupby("mark_date")
    }
    entries_by_date = {
        date: group for date, group in trades.groupby("entry_date", sort=False)
    }
    contracts: dict[str, float] = {}
    records: list[dict[str, object]] = []
    equity = INITIAL_CAPITAL
    equity_values: list[float] = []
    daily_pnl_values: list[float] = []
    for date in market_index:
        pre_day_equity = equity
        if date in entries_by_date:
            for trade in entries_by_date[date].itertuples():
                trade_contracts = pre_day_equity / (
                    float(trade.short_strike) * MULTIPLIER
                )
                contracts[trade.trade_id] = trade_contracts
                records.append(
                    {
                        "asset": asset,
                        "profit_target": profit_level / 100.0,
                        "trade_id": trade.trade_id,
                        "structure": trade.structure,
                        "entry_date": trade.entry_date,
                        "exit_date": trade.exit_date,
                        "expiration_date": trade.expiration_date,
                        "actual_dte": trade.actual_dte,
                        "exit_dte": trade.exit_dte,
                        "spot_entry": trade.spot_entry,
                        "short_strike": trade.short_strike,
                        "long_strike": trade.long_strike,
                        "short_strike_ratio": trade.short_strike / trade.spot_entry,
                        "long_strike_ratio": trade.long_strike / trade.spot_entry,
                        "entry_credit_per_spread": trade.entry_credit_per_spread,
                        "max_loss_per_spread": trade.max_loss_per_spread,
                        "contracts": trade_contracts,
                        "entry_equity": pre_day_equity,
                        "entry_notional": trade_contracts
                        * float(trade.short_strike)
                        * MULTIPLIER,
                        "defined_maximum_loss": trade_contracts
                        * float(trade.max_loss_per_spread),
                        "realized_pnl_per_spread": trade.realized_pnl_per_spread,
                        "dollar_pnl": trade_contracts
                        * float(trade.realized_pnl_per_spread),
                        "hit_profit_target": trade.hit_profit_target,
                    }
                )
        day_pnl = 0.0
        if date in paths_by_date:
            for change in paths_by_date[date].itertuples(index=False):
                day_pnl += float(change.daily_pnl_per_spread) * contracts[change.trade_id]
        equity += day_pnl
        equity_values.append(equity)
        daily_pnl_values.append(day_pnl)

    equity_curve = pd.Series(equity_values, index=market_index)
    daily_pnl = pd.Series(daily_pnl_values, index=market_index)
    detail = pd.DataFrame(records)
    returns = daily_pnl / equity_curve.shift(1).fillna(INITIAL_CAPITAL)
    years = len(returns) / 252.0
    peak = pd.Series(
        np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity_curve.to_numpy()])[1:],
        index=market_index,
    )
    standard_deviation = float(returns.std(ddof=1))
    metrics = {
        "asset": asset,
        "structure": detail["structure"].iloc[0],
        "profit_target": profit_level / 100.0,
        "trades": len(detail),
        "cagr": (equity_curve.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0,
        "annualized_volatility": standard_deviation * math.sqrt(252.0),
        "sharpe_zero_cash": (
            float(returns.mean() / standard_deviation * math.sqrt(252.0))
            if standard_deviation > 0
            else np.nan
        ),
        "max_drawdown": float((equity_curve / peak - 1.0).min()),
        "ending_equity": float(equity_curve.iloc[-1]),
        "win_rate": float(detail["dollar_pnl"].gt(0).mean()),
        "profit_target_hit_rate": float(detail["hit_profit_target"].mean()),
        "average_exit_dte": float(detail["exit_dte"].mean()),
        "average_entry_credit_pct_notional": float(
            (
                detail["entry_credit_per_spread"]
                / (detail["short_strike"] * MULTIPLIER)
            ).mean()
        ),
        "average_defined_loss_pct_notional": float(
            (
                detail["max_loss_per_spread"]
                / (detail["short_strike"] * MULTIPLIER)
            ).mean()
        ),
    }
    return equity_curve, detail, metrics


def calendar_returns(equity: pd.Series, asset: str, profit_level: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior = INITIAL_CAPITAL
    for year, group in equity.groupby(equity.index.year):
        ending = float(group.iloc[-1])
        rows.append(
            {
                "asset": asset,
                "profit_target": profit_level / 100.0,
                "year": int(year),
                "option_return": ending / prior - 1.0,
            }
        )
        prior = ending
    return pd.DataFrame(rows)


def standalone_curve_metrics(equity: pd.Series, label: str) -> dict[str, object]:
    returns = equity.pct_change().fillna(0.0)
    years = len(returns) / 252.0
    peak = equity.cummax()
    standard_deviation = float(returns.std(ddof=1))
    return {
        "version": label,
        "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0,
        "annualized_volatility": standard_deviation * math.sqrt(252.0),
        "sharpe_zero_cash": float(
            returns.mean() / standard_deviation * math.sqrt(252.0)
        ),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "ending_equity_rebased": float(
            INITIAL_CAPITAL * equity.iloc[-1] / equity.iloc[0]
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vix_paths = build_vix_paths()
    market_index = pd.DatetimeIndex(sorted(vix_paths))

    spx_all = pd.read_parquet(SPX_CACHE / "trades.parquet")
    spx_all = spx_all[spx_all["parameter_id"].eq(SPX_PARAMETER_ID)].copy()
    spx_all["entry_date"] = pd.to_datetime(spx_all["entry_date"])
    spx_all["expiration_date"] = pd.to_datetime(spx_all["expiration_date"])
    scheduled = spx_all[
        spx_all["entry_date"].between(START, END)
        & spx_all["expiration_date"].le(END)
    ].sort_values("entry_date")
    entry_dates = scheduled["entry_date"].tolist()

    vix_entries, vix_skips = build_vix_entries(entry_dates, vix_paths)
    if vix_entries.empty:
        raise RuntimeError("No valid VIX mirror-spread entries")
    vix_marks = build_vix_marks(vix_entries)
    matched_dates = set(pd.to_datetime(vix_entries["entry_date"]))
    spx_trades, spx_marks = load_spx_data(matched_dates)
    vix_entries = vix_entries[vix_entries["entry_date"].isin(spx_trades["entry_date"])].copy()
    matched_dates = set(pd.to_datetime(vix_entries["entry_date"]))
    spx_trades = spx_trades[spx_trades["entry_date"].isin(matched_dates)].copy()
    spx_marks = spx_marks[spx_marks["trade_id"].isin(spx_trades["trade_id"])].copy()
    vix_marks = vix_marks[vix_marks["trade_id"].isin(vix_entries["trade_id"])].copy()

    all_metrics: list[dict[str, object]] = []
    all_details: list[pd.DataFrame] = []
    all_years: list[pd.DataFrame] = []
    curves: dict[str, pd.Series] = {}
    for profit_level in PROFIT_LEVELS:
        for asset, trades, marks in [
            ("SPX", spx_trades, spx_marks),
            ("VIX", vix_entries, vix_marks),
        ]:
            paths, exits = build_exit_paths(trades, marks, profit_level)
            equity, detail, metrics = simulate(
                asset, trades, paths, exits, market_index, profit_level
            )
            key = f"{asset}_profit_{profit_level}"
            curves[key] = equity
            all_metrics.append(metrics)
            all_details.append(detail)
            all_years.append(calendar_returns(equity, asset, profit_level))

    metrics = pd.DataFrame(all_metrics).sort_values(["profit_target", "asset"])
    details = pd.concat(all_details, ignore_index=True)
    yearly = pd.concat(all_years, ignore_index=True)
    equity_curves = pd.DataFrame(curves, index=market_index)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    details.to_csv(OUT / "trades.csv", index=False)
    yearly.to_csv(OUT / "calendar_returns.csv", index=False)
    equity_curves.to_csv(OUT / "equity_curves.csv", index_label="date")
    vix_entries.to_csv(OUT / "vix_entries.csv", index=False)
    vix_skips.to_csv(OUT / "vix_skips.csv", index=False)

    p25 = metrics[metrics["profit_target"].eq(0.25)].set_index("asset")
    spx_curve = equity_curves["SPX_profit_25"]
    vix_curve = equity_curves["VIX_profit_25"]
    legacy_raw = pd.read_parquet(
        SPX_CACHE / "equity_curves.parquet",
        columns=[f"{SPX_PARAMETER_ID}_profit_25"],
    ).iloc[:, 0]
    legacy_raw = legacy_raw.loc[START:END].reindex(market_index).ffill().bfill()
    legacy_curve = legacy_raw / legacy_raw.iloc[0] * INITIAL_CAPITAL
    legacy_metrics = standalone_curve_metrics(
        legacy_curve, "Original standout SPX sizing"
    )
    pd.DataFrame([legacy_metrics]).to_csv(OUT / "period_benchmark.csv", index=False)
    spx_returns = spx_curve.pct_change().fillna(spx_curve.iloc[0] / INITIAL_CAPITAL - 1.0)
    vix_returns = vix_curve.pct_change().fillna(vix_curve.iloc[0] / INITIAL_CAPITAL - 1.0)
    daily_correlation = float(spx_returns.corr(vix_returns))

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("white")
    years_elapsed = (market_index - market_index[0]).days / 365.2425
    ax.fill_between(
        market_index,
        1.03**years_elapsed - 1.0,
        1.05**years_elapsed - 1.0,
        color="#bbf7d0",
        alpha=0.30,
        label="3%–5% annual return path",
        zorder=1,
    )
    colors = {"SPX": "#087f8c", "VIX": "#d97706"}
    for asset, curve in [("SPX", spx_curve), ("VIX", vix_curve)]:
        row = p25.loc[asset]
        ax.plot(
            market_index,
            curve / INITIAL_CAPITAL - 1.0,
            color=colors[asset],
            linewidth=2.6,
            label=(
                f"{asset} {row['structure']}  |  CAGR {row['cagr']:.2%}  |  "
                f"Sharpe {row['sharpe_zero_cash']:.2f}  |  DD {row['max_drawdown']:.2%}"
            ),
            zorder=3,
        )
    ax.plot(
        market_index,
        legacy_curve / INITIAL_CAPITAL - 1.0,
        color="#64748b",
        linewidth=1.7,
        linestyle="--",
        label=(
            "Original standout SPX sizing  |  "
            f"CAGR {legacy_metrics['cagr']:.2%}  |  "
            f"Sharpe {legacy_metrics['sharpe_zero_cash']:.2f}  |  "
            f"DD {legacy_metrics['max_drawdown']:.2%}"
        ),
        zorder=2,
    )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_title(
        "SPX put spread vs. mirrored VIX call spread, 2022–2026",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.01,
        "Matched monthly entries · ~60 DTE · 25% profit target · 100% current-equity notional per entry",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    ax.set_ylabel("Cumulative option return", fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.8)
    ax.legend(loc="best", frameon=False, fontsize=9.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.07,
        0.015,
        "Option P&L only; no cash interest; realistic execution; fractional contracts. "
        "VIX positions not closed early exit at the last tradable quote before settlement.",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.98])
    fig.savefig(OUT / "equity_curve_profit25.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor("#f8fafc")
    for asset in ("SPX", "VIX"):
        subset = metrics[metrics["asset"].eq(asset)].sort_values("profit_target")
        axes[0].plot(
            subset["profit_target"] * 100,
            subset["cagr"] * 100,
            marker="o",
            linewidth=2.2,
            color=colors[asset],
            label=asset,
        )
        axes[1].plot(
            subset["profit_target"] * 100,
            subset["max_drawdown"] * 100,
            marker="o",
            linewidth=2.2,
            color=colors[asset],
            label=asset,
        )
    axes[0].axhspan(3, 5, color="#bbf7d0", alpha=0.35)
    axes[0].set_title("CAGR by profit target", fontweight="bold")
    axes[0].set_ylabel("Annualized return (%)")
    axes[1].set_title("Maximum drawdown by profit target", fontweight="bold")
    axes[1].set_ylabel("Maximum drawdown (%)")
    for ax in axes:
        ax.set_xlabel("Profit target (% of entry credit)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Matched 60-DTE SPX and VIX credit spreads, 2022–2026",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.03, 0.02, 0.99, 0.94])
    fig.savefig(OUT / "profit_target_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    requested = metrics.pivot(
        index="profit_target",
        columns="asset",
        values=["cagr", "sharpe_zero_cash", "max_drawdown", "win_rate"],
    )
    requested.to_csv(OUT / "profit_target_pivot.csv")
    vix_ratios = vix_entries[
        ["short_strike_ratio", "long_strike_ratio"]
    ].mean()
    report_lines = [
        "# SPX 103/100 puts vs mirrored VIX 97/100 calls, 2022–2026",
        "",
        "Both strategies use matched monthly entry dates, approximately 60 DTE, 100% of then-current equity in short-strike notional per entry, realistic execution costs, fractional contracts, and no cash interest. The VIX structure sells a call near 97% of VIX spot and buys the next available call near 100%, the moneyness mirror of selling a 103% SPX put and buying a 100% put.",
        "",
        "| Asset | Profit target | CAGR | Volatility | Sharpe | Max drawdown | Win rate | Ending equity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples():
        report_lines.append(
            f"| {row.asset} | {row.profit_target:.0%} | {row.cagr:.2%} | "
            f"{row.annualized_volatility:.2%} | {row.sharpe_zero_cash:.2f} | "
            f"{row.max_drawdown:.2%} | {row.win_rate:.1%} | ${row.ending_equity:,.0f} |"
        )
    report_lines.extend(
        [
            "",
            f"For reference, the original standout SPX maximum-loss sizing returned {legacy_metrics['cagr']:.2%} CAGR with a {legacy_metrics['max_drawdown']:.2%} maximum drawdown in this shorter window, versus 4.16% CAGR over the full 2016–2026 sample.",
            f"The 25% strategies' daily return correlation was {daily_correlation:.2f}.",
            f"Because VIX strike increments are coarse relative to the index level, the realized VIX spread averaged {vix_ratios['short_strike_ratio']:.1%}/{vix_ratios['long_strike_ratio']:.1%} rather than exactly 97%/100%.",
            "VIX expiration-day quotes in this archive are zeroed and do not provide VRO settlement values. Positions that did not reach their profit target were therefore closed using the final valid bid/ask quote before expiration.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"\nMatched entries: {len(vix_entries)}")
    print(
        f"Mean VIX realized ratios: {vix_ratios['short_strike_ratio']:.3f}/"
        f"{vix_ratios['long_strike_ratio']:.3f}"
    )
    print(f"25% daily return correlation: {daily_correlation:.3f}")
    print(f"Saved results to {OUT}")


if __name__ == "__main__":
    main()
