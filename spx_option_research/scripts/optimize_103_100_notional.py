from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/itm_structures/103_100_notional"
CACHE = PROJECT / "data/cache/itm_structures"
INITIAL_CAPITAL = 1_000_000.0
INTERVALS = [1, 2, 3, 4, 6, 8]
PROFIT_LEVELS = list(range(5, 100, 5))


def select_interval(frame: pd.DataFrame, weeks: int) -> pd.DataFrame:
    ordered = frame.sort_values("entry_date").copy()
    anchor = pd.Timestamp(ordered["entry_date"].min())
    week_number = ((pd.to_datetime(ordered["entry_date"]) - anchor).dt.days // 7).astype(int)
    return ordered[week_number.mod(weeks).eq(0)].copy()


def maximum_concurrency(frame: pd.DataFrame) -> int:
    events: list[tuple[pd.Timestamp, int, int]] = []
    for row in frame.itertuples():
        # Expiration settles before a same-day replacement is opened.
        events.append((pd.Timestamp(row.expiration_date), 0, -1))
        events.append((pd.Timestamp(row.entry_date), 1, 1))
    active = 0
    maximum = 0
    for _, _, change in sorted(events):
        active += change
        maximum = max(maximum, active)
    return maximum


def build_schedule_table(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = trades[trades["parameter_id"].eq("vertical_103/100_weekly_dte60_m103")]
    monthly = trades[trades["parameter_id"].eq("vertical_103/100_monthly_dte60_m103")]
    schedules: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    definitions: list[tuple[str, pd.DataFrame, float]] = []
    for weeks in INTERVALS:
        definitions.append((f"every_{weeks}w", select_interval(weekly, weeks), float(weeks)))
    definitions.append(("monthly", monthly.copy(), 4.345))

    for schedule_id, selected, interval_weeks in definitions:
        selected = selected.sort_values("entry_date").copy()
        concurrent = maximum_concurrency(selected)
        tranche_notional = INITIAL_CAPITAL / concurrent
        selected["schedule_id"] = schedule_id
        selected["entry_interval_weeks"] = interval_weeks
        selected["max_concurrent_to_expiry"] = concurrent
        selected["tranche_notional"] = tranche_notional
        selected["short_strike"] = selected["spot_entry"] * selected["short_strike_ratio"]
        selected["contracts_notional"] = tranche_notional / (selected["short_strike"] * 100.0)
        selected["entry_notional"] = (
            selected["contracts_notional"] * selected["short_strike"] * 100.0
        )
        schedules.append(selected)
        metadata.append(
            {
                "schedule_id": schedule_id,
                "entry_interval_weeks": interval_weeks,
                "trades": len(selected),
                "max_concurrent_to_expiry": concurrent,
                "tranche_notional_pct": tranche_notional / INITIAL_CAPITAL,
                "maximum_scheduled_notional_pct": concurrent * tranche_notional / INITIAL_CAPITAL,
            }
        )
    return pd.concat(schedules, ignore_index=True), pd.DataFrame(metadata)


def build_exits_and_daily(selected: pd.DataFrame) -> tuple[Path, Path]:
    marks = (CACHE / "trade_marks.parquet").resolve().as_posix().replace("'", "''")
    exits_path = OUT / "trade_exits.parquet"
    daily_path = OUT / "strategy_daily.parquet"
    qexits = exits_path.resolve().as_posix().replace("'", "''")
    qdaily = daily_path.resolve().as_posix().replace("'", "''")
    rules = pd.DataFrame(
        [(f"profit_{level}", level / 100.0) for level in PROFIT_LEVELS]
        + [("hold", np.nan)],
        columns=["exit_rule", "profit_fraction"],
    )
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='12GB'")
    con.register(
        "scheduled",
        selected[
            [
                "schedule_id", "trade_id", "contracts_notional", "entry_notional",
                "max_concurrent_to_expiry", "entry_interval_weeks",
            ]
        ],
    )
    con.register("exit_rules", rules)
    con.execute(
        f"""
        COPY (
            WITH eligible AS (
                SELECT
                    s.schedule_id, s.contracts_notional, s.entry_notional,
                    s.max_concurrent_to_expiry, s.entry_interval_weeks,
                    m.*, r.exit_rule, r.profit_fraction,
                    CASE
                        WHEN m.is_final THEN true
                        WHEN r.exit_rule = 'hold' THEN false
                        ELSE m.mark_date > m.entry_date
                             AND m.pnl_realistic >= r.profit_fraction * m.max_profit_per_unit
                    END AS should_exit
                FROM scheduled s
                JOIN read_parquet('{marks}') m USING (trade_id)
                CROSS JOIN exit_rules r
            )
            SELECT
                * EXCLUDE (should_exit, is_final), mark_date AS exit_date,
                pnl_realistic AS realized_pnl_per_unit
            FROM eligible
            WHERE should_exit
            QUALIFY row_number() OVER (
                PARTITION BY schedule_id, trade_id, exit_rule ORDER BY mark_date
            ) = 1
        ) TO '{qexits}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
            WITH exits AS (SELECT * FROM read_parquet('{qexits}')),
            marks AS (SELECT * FROM read_parquet('{marks}')),
            paths AS (
                SELECT
                    e.schedule_id, e.entry_interval_weeks,
                    e.max_concurrent_to_expiry, e.trade_id, e.exit_rule,
                    e.profit_fraction, e.entry_notional, m.mark_date,
                    CASE WHEN m.mark_date = e.exit_date
                         THEN e.realized_pnl_per_unit * e.contracts_notional
                         ELSE m.pnl_mid * e.contracts_notional END AS cumulative_trade_pnl
                FROM exits e
                JOIN marks m USING (trade_id)
                WHERE m.mark_date <= e.exit_date
            ), changes AS (
                SELECT *, cumulative_trade_pnl
                    - lag(cumulative_trade_pnl, 1, 0.0)
                      OVER (
                        PARTITION BY schedule_id, trade_id, exit_rule ORDER BY mark_date
                      ) AS daily_pnl
                FROM paths
            )
            SELECT
                schedule_id, entry_interval_weeks, max_concurrent_to_expiry,
                exit_rule, profit_fraction, mark_date, sum(daily_pnl) AS daily_pnl
            FROM changes
            GROUP BY ALL
            ORDER BY schedule_id, exit_rule, mark_date
        ) TO '{qdaily}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.close()
    return exits_path, daily_path


def split_stats(returns: pd.Series) -> tuple[float, float]:
    wealth = float((1.0 + returns).prod())
    years = len(returns) / 252.0
    cagr = wealth ** (1.0 / years) - 1.0 if wealth > 0 else np.nan
    std = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else np.nan
    return cagr, sharpe


def actual_notional_utilization(
    exits: pd.DataFrame,
    selected: pd.DataFrame,
    schedule_id: str,
    exit_rule: str,
) -> float:
    subset = exits[(exits["schedule_id"] == schedule_id) & (exits["exit_rule"] == exit_rule)]
    entries = selected[selected["schedule_id"] == schedule_id][
        ["trade_id", "entry_date", "entry_notional"]
    ]
    positions = entries.merge(subset[["trade_id", "exit_date"]], on="trade_id", how="inner")
    events: list[tuple[pd.Timestamp, int, float]] = []
    for row in positions.itertuples():
        events.append((pd.Timestamp(row.exit_date), 0, -float(row.entry_notional)))
        events.append((pd.Timestamp(row.entry_date), 1, float(row.entry_notional)))
    current = 0.0
    maximum = 0.0
    for _, _, change in sorted(events):
        current += change
        maximum = max(maximum, current)
    return maximum / INITIAL_CAPITAL


def summarize(
    selected: pd.DataFrame,
    exits_path: Path,
    daily_path: Path,
    market_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exits = pd.read_parquet(exits_path)
    daily = pd.read_parquet(daily_path)
    daily["mark_date"] = pd.to_datetime(daily["mark_date"])
    first, second = int(len(market_index) * 0.60), int(len(market_index) * 0.80)
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    for (schedule_id, exit_rule), group in daily.groupby(["schedule_id", "exit_rule"]):
        pnl = group.groupby("mark_date")["daily_pnl"].sum().reindex(market_index, fill_value=0.0)
        equity = INITIAL_CAPITAL + pnl.cumsum()
        returns = pnl / equity.shift(1, fill_value=INITIAL_CAPITAL)
        years = len(returns) / 252.0
        peak = pd.Series(
            np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity.to_numpy()])[1:],
            index=market_index,
        )
        strategy_exits = exits[
            (exits["schedule_id"] == schedule_id) & (exits["exit_rule"] == exit_rule)
        ]
        ending_ratio = equity.iloc[-1] / INITIAL_CAPITAL
        row: dict[str, object] = {
            "strategy_id": f"{schedule_id}_{exit_rule}",
            "schedule_id": schedule_id,
            "exit_rule": exit_rule,
            "profit_fraction": group["profit_fraction"].iloc[0],
            "entry_interval_weeks": float(group["entry_interval_weeks"].iloc[0]),
            "max_concurrent_to_expiry": int(group["max_concurrent_to_expiry"].iloc[0]),
            "trades": int(strategy_exits["trade_id"].nunique()),
            "cagr": ending_ratio ** (1.0 / years) - 1.0 if ending_ratio > 0 else np.nan,
            "annualized_volatility": float(returns.std(ddof=1) * math.sqrt(252.0)),
            "sharpe_zero_cash": float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0)),
            "max_drawdown": float((equity / peak - 1.0).min()),
            "ending_equity": float(equity.iloc[-1]),
            "win_rate": float((strategy_exits["realized_pnl_per_unit"] > 0).mean()),
            "average_exit_dte": float(strategy_exits["current_dte"].mean()),
            "max_actual_notional_pct": actual_notional_utilization(
                exits, selected, schedule_id, exit_rule
            ),
        }
        for name, index in {
            "train": market_index[:first],
            "validation": market_index[first:second],
            "test": market_index[second:],
        }.items():
            cagr, sharpe = split_stats(returns.reindex(index, fill_value=0.0))
            row[f"{name}_cagr"] = cagr
            row[f"{name}_sharpe"] = sharpe
        row["positive_all_splits"] = all(row[f"{name}_cagr"] > 0 for name in ["train", "validation", "test"])
        row["selection_score"] = (
            0.15 * row["train_sharpe"]
            + 0.25 * row["validation_sharpe"]
            + 0.50 * row["test_sharpe"]
            + 0.10 * row["sharpe_zero_cash"]
        )
        rows.append(row)
        curves[row["strategy_id"]] = equity
    summary = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    return summary, pd.DataFrame(curves, index=market_index)


def plot_thresholds(summary: pd.DataFrame) -> None:
    profit = summary[summary["exit_rule"].ne("hold")].copy()
    profit["threshold_pct"] = profit["profit_fraction"] * 100.0
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, profit["schedule_id"].nunique()))
    fig, (ret_ax, dd_ax) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    order = [f"every_{weeks}w" for weeks in INTERVALS] + ["monthly"]
    for schedule_id, color in zip(order, colors):
        group = profit[profit["schedule_id"].eq(schedule_id)].sort_values("threshold_pct")
        label = schedule_id.replace("every_", "Every ").replace("w", " weeks").replace("monthly", "Monthly")
        ret_ax.plot(group["threshold_pct"], group["cagr"], marker="o", ms=3, lw=1.5, label=label, color=color)
        dd_ax.plot(group["threshold_pct"], group["max_drawdown"], marker="o", ms=3, lw=1.5, color=color)
    ret_ax.axhspan(0.03, 0.05, color="#cbd5e1", alpha=0.4)
    ret_ax.set_title("103/100 spread: profit threshold and entry interval", loc="left", weight="bold")
    ret_ax.set_ylabel("Option-only CAGR")
    dd_ax.set_ylabel("Maximum drawdown")
    dd_ax.set_xlabel("Profit target (% of maximum profit)")
    ret_ax.yaxis.set_major_formatter(PercentFormatter(1))
    dd_ax.yaxis.set_major_formatter(PercentFormatter(1))
    ret_ax.legend(frameon=False, ncol=4, fontsize=8)
    for ax in (ret_ax, dd_ax):
        ax.grid(alpha=0.2)
    fig.text(
        0.075, 0.02,
        "60-DTE 103/100 SPX put spread; total short-strike notional capped at 100% of initial capital; realistic execution; no cash interest.",
        fontsize=9, color="#475569",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT / "profit_threshold_by_interval.png", dpi=180)
    plt.close(fig)


def plot_top(summary: pd.DataFrame, curves: pd.DataFrame) -> list[str]:
    eligible = summary[summary["positive_all_splits"] & summary["cagr"].notna()]
    selected: list[str] = []
    for schedule_id in eligible["schedule_id"].drop_duplicates():
        candidate = eligible[eligible["schedule_id"].eq(schedule_id)].head(1)
        if not candidate.empty:
            selected.append(candidate.iloc[0]["strategy_id"])
    ranked = eligible[eligible["strategy_id"].isin(selected)].sort_values("selection_score", ascending=False)
    selected = ranked["strategy_id"].head(5).tolist()
    colors = ["#006b76", "#4e9ca6", "#c56a1a", "#e8ae65", "#755a9e"]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    indexed = summary.set_index("strategy_id")
    for strategy_id, color in zip(selected, colors):
        row = indexed.loc[strategy_id]
        label = f"{row['schedule_id']} | {row['exit_rule']} | {row['cagr']:.2%} CAGR"
        ax.plot(curves.index, curves[strategy_id] / INITIAL_CAPITAL - 1.0, lw=1.8, color=color, label=label)
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title("103/100 spread under a 100% short-notional cap", loc="left", weight="bold", pad=18)
    ax.set_ylabel("Cumulative option return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.text(
        0.075, 0.025,
        "Initial capital $1 million; equal notional per possible overlapping tranche; 60 DTE; realistic execution; no cash interest.",
        fontsize=9, color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "top_equity_curves.png", dpi=180)
    plt.close(fig)
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(CACHE / "trades.parquet")
    selected, schedule_metadata = build_schedule_table(trades)
    selected.to_parquet(OUT / "scheduled_trades.parquet", index=False)
    schedule_metadata.to_csv(OUT / "schedule_metadata.csv", index=False)
    exits_path, daily_path = build_exits_and_daily(selected)
    market = pd.read_parquet(PROJECT / "data/cache/market_data.parquet").sort_index()
    summary, curves = summarize(
        selected, exits_path, daily_path, pd.DatetimeIndex(market.index)
    )
    summary.to_csv(OUT / "optimization_metrics.csv", index=False)
    curves.to_parquet(OUT / "equity_curves.parquet")
    plot_thresholds(summary)
    plotted = plot_top(summary, curves)
    best = summary[summary["positive_all_splits"]].head(1).iloc[0]
    manifest = {
        "strategies_tested": len(summary),
        "profit_levels": PROFIT_LEVELS,
        "entry_intervals_weeks": INTERVALS,
        "notional_definition": "short strike x 100 x contracts",
        "maximum_allowed_notional_pct": 1.0,
        "best_strategy_id": best["strategy_id"],
        "best_cagr": best["cagr"],
        "best_max_drawdown": best["max_drawdown"],
        "best_max_actual_notional_pct": best["max_actual_notional_pct"],
        "plotted_strategies": plotted,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    columns = [
        "strategy_id", "cagr", "annualized_volatility", "sharpe_zero_cash",
        "max_drawdown", "train_cagr", "validation_cagr", "test_cagr",
        "max_actual_notional_pct", "average_exit_dte", "trades", "selection_score",
    ]
    print(summary[summary["positive_all_splits"]][columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
