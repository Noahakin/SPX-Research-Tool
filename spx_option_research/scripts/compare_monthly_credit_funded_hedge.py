from __future__ import annotations

import calendar
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUT = PROJECT / "results/monthly_credit_funded_hedge"
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.execution import ExecutionModel
from spxresearch.option_selector import clean_puts
from spxresearch.warehouse import ResearchWarehouse


INITIAL_CAPITAL = 1_000_000.0
MULTIPLIER = 100.0
MODEL = ExecutionModel(spread_fraction=0.25, commission_per_contract=1.50)
STRUCTURES = {"99/96": (0.99, 0.96), "103/100": (1.03, 1.00)}
BASELINE = "99/96 monthly, no hedge"
HEDGED = "103/100 monthly, surplus-funded put"


def third_friday(year: int, month: int) -> pd.Timestamp:
    weeks = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in weeks if week[calendar.FRIDAY]]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def monthly_roll_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    available = set(dates)
    result: list[pd.Timestamp] = []
    for period in pd.period_range(dates.min(), dates.max(), freq="M"):
        scheduled = third_friday(period.year, period.month)
        choices = [
            scheduled - pd.Timedelta(days=offset)
            for offset in range(4)
            if scheduled - pd.Timedelta(days=offset) in available
        ]
        if choices:
            result.append(max(choices))
    return sorted(set(result))


def nearest_put(puts: pd.DataFrame, target_strike: float) -> pd.Series:
    return puts.loc[(puts["strike"].astype(float) - target_strike).abs().idxmin()]


def choose_vertical(puts: pd.DataFrame, spot: float, short_ratio: float, long_ratio: float) -> dict[str, object]:
    short = nearest_put(puts, spot * short_ratio)
    lower = puts[puts["strike"].astype(float) < float(short["strike"])]
    if lower.empty:
        raise LookupError("no lower long put")
    long = nearest_put(lower, spot * long_ratio)
    entry_credit_cash = MODEL.cash_flow(short, -1) + MODEL.cash_flow(long, 1)
    if entry_credit_cash <= 0:
        raise LookupError("nonpositive spread credit")
    return {
        "short_strike": float(short["strike"]),
        "long_strike": float(long["strike"]),
        "short_symbol": str(short["option_symbol"]).strip(),
        "long_symbol": str(long["option_symbol"]).strip(),
        "entry_credit_cash": float(entry_credit_cash),
        "entry_credit_points": float(entry_credit_cash / MULTIPLIER),
    }


def build_entries(archive: SPXSurfaceArchive) -> tuple[pd.DataFrame, pd.DataFrame]:
    roll_dates = monthly_roll_dates(archive.populated_dates)
    rows: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for entry_date, expiration_date in zip(roll_dates[:-1], roll_dates[1:]):
        chain = archive.read(entry_date)
        puts = clean_puts(chain, pm_only=True)
        puts = puts[puts["expiration_date"].eq(expiration_date)].copy()
        if puts.empty:
            skips.append({"entry_date": entry_date, "reason": "no matching PM expiration"})
            continue
        spot = float(puts["underlying_price"].dropna().median())
        try:
            base = choose_vertical(puts, spot, *STRUCTURES["99/96"])
            itm = choose_vertical(puts, spot, *STRUCTURES["103/100"])
        except LookupError as error:
            skips.append({"entry_date": entry_date, "reason": str(error)})
            continue

        surplus_cash = float(itm["entry_credit_cash"] - base["entry_credit_cash"])
        if surplus_cash <= 0:
            skips.append({"entry_date": entry_date, "reason": "103/100 has no credit surplus"})
            continue
        eligible = puts[puts["strike"].astype(float) < float(itm["long_strike"])].copy()
        if eligible.empty:
            skips.append({"entry_date": entry_date, "reason": "no put below 103/100 long strike"})
            continue
        eligible["hedge_debit_cash"] = eligible.apply(lambda quote: -MODEL.cash_flow(quote, 1), axis=1)
        affordable = eligible[
            eligible["hedge_debit_cash"].gt(0)
            & eligible["hedge_debit_cash"].le(surplus_cash)
        ]
        if affordable.empty:
            skips.append({"entry_date": entry_date, "reason": "no full protective put fits surplus"})
            continue
        hedge = affordable.sort_values(["strike", "hedge_debit_cash"], ascending=[False, False]).iloc[0]

        common = {
            "entry_date": entry_date,
            "expiration_date": expiration_date,
            "entry_dte": int((expiration_date - entry_date).days),
            "spot_entry": spot,
            "baseline_credit_cash": float(base["entry_credit_cash"]),
            "itm_credit_cash": float(itm["entry_credit_cash"]),
            "credit_surplus_cash": surplus_cash,
            "hedge_strike": float(hedge["strike"]),
            "hedge_symbol": str(hedge["option_symbol"]).strip(),
            "hedge_abs_delta": abs(float(hedge["delta"])),
            "hedge_otm_pct": 100.0 * (1.0 - float(hedge["strike"]) / spot),
            "hedge_debit_cash": float(hedge["hedge_debit_cash"]),
            "surplus_spent_pct": float(hedge["hedge_debit_cash"] / surplus_cash),
        }
        rows.append(
            {
                **common,
                "trade_id": f"baseline_{entry_date:%Y%m%d}",
                "strategy": BASELINE,
                "structure": "99/96",
                "short_strike": base["short_strike"],
                "long_strike": base["long_strike"],
                "short_symbol": base["short_symbol"],
                "long_symbol": base["long_symbol"],
                "entry_credit_cash": base["entry_credit_cash"],
                "entry_credit_points": base["entry_credit_points"],
                "package_entry_cash": base["entry_credit_cash"],
                "has_hedge": False,
            }
        )
        rows.append(
            {
                **common,
                "trade_id": f"hedged_{entry_date:%Y%m%d}",
                "strategy": HEDGED,
                "structure": "103/100 + put",
                "short_strike": itm["short_strike"],
                "long_strike": itm["long_strike"],
                "short_symbol": itm["short_symbol"],
                "long_symbol": itm["long_symbol"],
                "entry_credit_cash": itm["entry_credit_cash"],
                "entry_credit_points": itm["entry_credit_points"],
                "package_entry_cash": itm["entry_credit_cash"] - float(hedge["hedge_debit_cash"]),
                "has_hedge": True,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(skips)


def add_expiration_and_size(entries: pd.DataFrame, warehouse: ResearchWarehouse) -> pd.DataFrame:
    warehouse.connection.register("monthly_entries", entries)
    expiration = warehouse.execute(
        """
        SELECT
            e.trade_id,
            median(s.spot) AS spot_expiration
        FROM monthly_entries e
        JOIN surface s
          ON s.trade_date = CAST(e.expiration_date AS DATE)
         AND s.expiration_date = CAST(e.expiration_date AS DATE)
         AND s.settlement = 'PM'
        GROUP BY e.trade_id
        """
    ).fetchdf()
    result = entries.merge(expiration, on="trade_id", how="inner")
    result["core_expiration_debit_cash"] = (
        np.maximum(result["short_strike"] - result["spot_expiration"], 0.0)
        - np.maximum(result["long_strike"] - result["spot_expiration"], 0.0)
    ) * MULTIPLIER
    result["hedge_expiration_value_cash"] = np.where(
        result["has_hedge"],
        np.maximum(result["hedge_strike"] - result["spot_expiration"], 0.0) * MULTIPLIER,
        0.0,
    )
    result["pnl_per_spread"] = (
        result["package_entry_cash"]
        - result["core_expiration_debit_cash"]
        + result["hedge_expiration_value_cash"]
    )

    sized: list[pd.DataFrame] = []
    for strategy, group in result.groupby("strategy", sort=False):
        equity = INITIAL_CAPITAL
        records: list[pd.Series] = []
        for _, row in group.sort_values("entry_date").iterrows():
            row = row.copy()
            row["pre_trade_equity"] = equity
            row["contracts"] = equity / (float(row["short_strike"]) * MULTIPLIER)
            row["dollar_pnl"] = float(row["pnl_per_spread"]) * float(row["contracts"])
            equity += float(row["dollar_pnl"])
            row["post_trade_equity"] = equity
            records.append(row)
        sized.append(pd.DataFrame(records))
    return pd.concat(sized, ignore_index=True)


def daily_paths(entries: pd.DataFrame, warehouse: ResearchWarehouse) -> pd.DataFrame:
    warehouse.connection.unregister("monthly_entries")
    warehouse.connection.register("monthly_entries", entries)
    paths = warehouse.execute(
        """
        SELECT
            e.trade_id,
            e.strategy,
            e.entry_date,
            e.expiration_date,
            e.contracts,
            e.package_entry_cash,
            e.has_hedge,
            s.trade_date AS mark_date,
            CASE
                WHEN s.trade_date = CAST(e.expiration_date AS DATE) OR s.dte = 0 THEN
                    (greatest(e.short_strike - s.spot, 0.0)
                     - greatest(e.long_strike - s.spot, 0.0)) * 100.0
                ELSE (s.mid - l.mid) * 100.0
            END AS core_mark_debit_cash,
            CASE
                WHEN NOT e.has_hedge THEN 0.0
                WHEN s.trade_date = CAST(e.expiration_date AS DATE) OR s.dte = 0 THEN
                    greatest(e.hedge_strike - s.spot, 0.0) * 100.0
                ELSE h.mid * 100.0
            END AS hedge_mark_value_cash
        FROM monthly_entries e
        JOIN surface s
          ON s.option_symbol = e.short_symbol
         AND s.trade_date BETWEEN CAST(e.entry_date AS DATE) AND CAST(e.expiration_date AS DATE)
        JOIN surface l
          ON l.option_symbol = e.long_symbol
         AND l.trade_date = s.trade_date
        LEFT JOIN surface h
          ON e.has_hedge
         AND h.option_symbol = e.hedge_symbol
         AND h.trade_date = s.trade_date
        ORDER BY strategy, trade_id, mark_date
        """
    ).fetchdf()
    paths["hedge_mark_value_cash"] = (
        paths.groupby("trade_id")["hedge_mark_value_cash"].ffill().fillna(0.0)
    )
    paths["cumulative_pnl"] = (
        paths["package_entry_cash"]
        - paths["core_mark_debit_cash"]
        + paths["hedge_mark_value_cash"]
    ) * paths["contracts"]
    paths["daily_pnl"] = paths.groupby("trade_id")["cumulative_pnl"].diff()
    paths["daily_pnl"] = paths["daily_pnl"].fillna(paths["cumulative_pnl"])
    return paths


def metrics(equity: pd.Series, daily_pnl: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    returns = daily_pnl / equity.shift(1, fill_value=INITIAL_CAPITAL)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    peak = pd.Series(
        np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity.to_numpy()])[1:],
        index=equity.index,
    )
    volatility = returns.std(ddof=1) * math.sqrt(252.0)
    return {
        "trades": int(len(trades)),
        "cagr": float((equity.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0),
        "annualized_volatility": float(volatility),
        "sharpe_zero_cash": float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0)),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "ending_equity": float(equity.iloc[-1]),
        "win_rate": float((trades["dollar_pnl"] > 0).mean()),
        "average_trade_return": float((trades["dollar_pnl"] / trades["pre_trade_equity"]).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    archive = SPXSurfaceArchive(data_root)
    entries, skips = build_entries(archive)
    if entries.empty:
        raise RuntimeError("No monthly trades selected")

    database = PROJECT / "data/cache/monthly_credit_funded_hedge.duckdb"
    with ResearchWarehouse(database, data_root, threads=8, memory_limit="12GB") as warehouse:
        entries = add_expiration_and_size(entries, warehouse)
        paths = daily_paths(entries, warehouse)

    entries.to_csv(OUT / "trades.csv", index=False)
    entries.to_parquet(OUT / "trades.parquet", index=False)
    skips.to_csv(OUT / "skips.csv", index=False)
    paths["mark_date"] = pd.to_datetime(paths["mark_date"])
    paths.to_parquet(OUT / "daily_paths.parquet", index=False)

    all_dates = pd.DatetimeIndex(archive.populated_dates)
    curves: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, object]] = []
    for strategy in [BASELINE, HEDGED]:
        strategy_paths = paths[paths["strategy"].eq(strategy)]
        daily = strategy_paths.groupby("mark_date")["daily_pnl"].sum().reindex(all_dates, fill_value=0.0)
        equity = INITIAL_CAPITAL + daily.cumsum()
        trades = entries[entries["strategy"].eq(strategy)].copy()
        row: dict[str, object] = {"strategy": strategy}
        row.update(metrics(equity, daily, trades))
        row.update(
            {
                "average_entry_dte": float(trades["entry_dte"].mean()),
                "average_entry_credit_points": float(trades["entry_credit_points"].mean()),
                "average_package_credit_points": float(trades["package_entry_cash"].mean() / 100.0),
            }
        )
        if strategy == HEDGED:
            row.update(
                {
                    "average_credit_surplus_points": float(trades["credit_surplus_cash"].mean() / 100.0),
                    "average_hedge_debit_points": float(trades["hedge_debit_cash"].mean() / 100.0),
                    "average_surplus_spent_pct": float(trades["surplus_spent_pct"].mean()),
                    "average_hedge_otm_pct": float(trades["hedge_otm_pct"].mean()),
                    "average_hedge_abs_delta": float(trades["hedge_abs_delta"].mean()),
                }
            )
        summary_rows.append(row)
        curves[strategy] = equity

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    curve_frame = pd.DataFrame(curves, index=all_dates)
    curve_frame.index.name = "date"
    curve_frame.to_csv(OUT / "equity_curves.csv")
    curve_frame.to_parquet(OUT / "equity_curves.parquet")

    fig, ax = plt.subplots(figsize=(13, 7.5))
    colors = {BASELINE: "#006b76", HEDGED: "#c56a1a"}
    indexed = summary.set_index("strategy")
    for strategy in [BASELINE, HEDGED]:
        label = f"{strategy} ({indexed.loc[strategy, 'cagr']:.2%} CAGR)"
        ax.plot(curve_frame.index, curve_frame[strategy] / INITIAL_CAPITAL - 1.0, lw=2.0, color=colors[strategy], label=label)
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title("Monthly SPX put spreads: OTM baseline vs credit-funded tail put", loc="left", weight="bold", pad=20)
    ax.set_ylabel("Cumulative option return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.075,
        0.025,
        "Third-Friday monthly rolls; held to expiration; one position at a time; short-strike notional equals current equity; realistic entry execution; no cash interest.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=180)
    plt.close(fig)

    hedged_trades = entries[entries["strategy"].eq(HEDGED)]
    manifest = {
        "archive_start": str(all_dates.min().date()),
        "archive_end": str(all_dates.max().date()),
        "trades_per_strategy": int(len(hedged_trades)),
        "average_entry_dte": float(hedged_trades["entry_dte"].mean()),
        "hedge_selection": "highest strike below the 103/100 long put whose one-for-one debit does not exceed the credit surplus over 99/96",
        "average_hedge_otm_pct": float(hedged_trades["hedge_otm_pct"].mean()),
        "average_hedge_abs_delta": float(hedged_trades["hedge_abs_delta"].mean()),
        "average_surplus_spent_pct": float(hedged_trades["surplus_spent_pct"].mean()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
