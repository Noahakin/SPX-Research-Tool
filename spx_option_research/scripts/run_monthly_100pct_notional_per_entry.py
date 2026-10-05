from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / "data/cache/itm_structures"
ITM_RESULTS = PROJECT / "results/itm_structures"
OUT = PROJECT / "results/monthly_100pct_notional_per_entry"

INITIAL_CAPITAL = 1_000_000.0
PARAMETER_ID = "vertical_103/100_monthly_dte60_m103"
STORED_STRATEGY_ID = f"{PARAMETER_ID}_profit_25"
PROFIT_FRACTION = 0.25


def split_stats(returns: pd.Series) -> tuple[float, float]:
    wealth = float((1.0 + returns).prod())
    years = len(returns) / 252.0
    cagr = wealth ** (1.0 / years) - 1.0 if wealth > 0 else np.nan
    standard_deviation = float(returns.std(ddof=1))
    sharpe = (
        float(returns.mean() / standard_deviation * math.sqrt(252.0))
        if standard_deviation > 0
        else np.nan
    )
    return cagr, sharpe


def prepare_trade_paths() -> tuple[pd.DataFrame, pd.DataFrame]:
    marks = pd.read_parquet(
        CACHE / "trade_marks.parquet",
        filters=[("parameter_id", "==", PARAMETER_ID)],
    )
    marks["mark_date"] = pd.to_datetime(marks["mark_date"])
    trades = pd.read_parquet(CACHE / "trades.parquet")
    trades = trades.loc[trades["parameter_id"].eq(PARAMETER_ID)].copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["short_strike"] = trades["spot_entry"] * trades["short_strike_ratio"]

    path_rows: list[pd.DataFrame] = []
    exits: list[dict[str, object]] = []
    for trade_id, trade_marks in marks.groupby("trade_id", sort=False):
        trade_marks = trade_marks.sort_values("mark_date").copy()
        eligible = trade_marks[
            trade_marks["is_final"]
            | (
                trade_marks["mark_date"].gt(trade_marks["entry_date"])
                & trade_marks["pnl_realistic"].ge(
                    PROFIT_FRACTION * trade_marks["max_profit_per_unit"]
                )
            )
        ]
        exit_row = eligible.iloc[0]
        path = trade_marks.loc[
            trade_marks["mark_date"].le(exit_row["mark_date"])
        ].copy()
        path["cumulative_pnl_per_spread"] = path["pnl_mid"]
        on_exit = path["mark_date"].eq(exit_row["mark_date"])
        path.loc[on_exit, "cumulative_pnl_per_spread"] = exit_row["pnl_realistic"]
        path["daily_pnl_per_spread"] = path["cumulative_pnl_per_spread"].diff().fillna(
            path["cumulative_pnl_per_spread"]
        )
        path_rows.append(path[["trade_id", "mark_date", "daily_pnl_per_spread"]])
        exits.append(
            {
                "trade_id": trade_id,
                "exit_date": exit_row["mark_date"],
                "exit_dte": int(exit_row["current_dte"]),
                "realized_pnl_per_spread": float(exit_row["pnl_realistic"]),
            }
        )

    paths = pd.concat(path_rows, ignore_index=True)
    trades = trades.merge(pd.DataFrame(exits), on="trade_id", how="inner")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    return trades.sort_values("entry_date").reset_index(drop=True), paths


def simulate(
    trades: pd.DataFrame,
    paths: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    *,
    compound_notional: bool,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    path_by_date = {
        date: group[["trade_id", "daily_pnl_per_spread"]]
        for date, group in paths.groupby("mark_date")
    }
    entries_by_date = {
        date: group for date, group in trades.groupby("entry_date", sort=False)
    }
    contracts: dict[str, float] = {}
    trade_records: list[dict[str, object]] = []
    equity = INITIAL_CAPITAL
    equity_values: list[float] = []
    daily_pnl_values: list[float] = []

    for date in market_index:
        pre_day_equity = equity
        if date in entries_by_date:
            for trade in entries_by_date[date].itertuples():
                target_notional = pre_day_equity if compound_notional else INITIAL_CAPITAL
                trade_contracts = target_notional / (float(trade.short_strike) * 100.0)
                contracts[trade.trade_id] = trade_contracts
                trade_records.append(
                    {
                        "trade_id": trade.trade_id,
                        "entry_date": trade.entry_date,
                        "exit_date": trade.exit_date,
                        "expiration_date": trade.expiration_date,
                        "short_strike": trade.short_strike,
                        "contracts": trade_contracts,
                        "entry_equity": pre_day_equity,
                        "entry_notional": target_notional,
                        "entry_notional_pct_equity": target_notional / pre_day_equity,
                        "max_loss_dollars": trade_contracts * trade.max_loss_per_unit,
                        "realized_pnl": (
                            trade_contracts * trade.realized_pnl_per_spread
                        ),
                        "exit_dte": trade.exit_dte,
                    }
                )

        day_pnl = 0.0
        if date in path_by_date:
            for change in path_by_date[date].itertuples(index=False):
                day_pnl += (
                    float(change.daily_pnl_per_spread) * contracts[change.trade_id]
                )
        equity += day_pnl
        equity_values.append(equity)
        daily_pnl_values.append(day_pnl)

    equity_curve = pd.Series(equity_values, index=market_index, name="equity")
    daily_pnl = pd.Series(daily_pnl_values, index=market_index, name="daily_pnl")
    sized_trades = pd.DataFrame(trade_records)

    exposure_rows: list[dict[str, object]] = []
    for date, current_equity in equity_curve.items():
        # Exit is processed before a new entry on the same date, so an exited
        # position is excluded from the end-of-day exposure measure.
        active = sized_trades[
            sized_trades["entry_date"].le(date)
            & sized_trades["exit_date"].gt(date)
        ]
        notional = float(active["entry_notional"].sum())
        maximum_loss = float(active["max_loss_dollars"].sum())
        exposure_rows.append(
            {
                "date": date,
                "equity": current_equity,
                "open_positions": len(active),
                "aggregate_short_strike_notional": notional,
                "notional_pct_current_equity": (
                    notional / current_equity if current_equity else np.nan
                ),
                "aggregate_maximum_loss": maximum_loss,
                "maximum_loss_pct_current_equity": (
                    maximum_loss / current_equity if current_equity else np.nan
                ),
            }
        )
    exposures = pd.DataFrame(exposure_rows).set_index("date")
    return equity_curve, daily_pnl, sized_trades, exposures


def summarize(
    name: str,
    equity: pd.Series,
    daily_pnl: pd.Series,
    trades: pd.DataFrame,
    exposures: pd.DataFrame,
) -> dict[str, object]:
    returns = daily_pnl / equity.shift(1).fillna(INITIAL_CAPITAL)
    years = len(returns) / 252.0
    peak = pd.Series(
        np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity.to_numpy()])[1:],
        index=equity.index,
    )
    first = int(len(returns) * 0.60)
    second = int(len(returns) * 0.80)
    train_cagr, train_sharpe = split_stats(returns.iloc[:first])
    validation_cagr, validation_sharpe = split_stats(returns.iloc[first:second])
    test_cagr, test_sharpe = split_stats(returns.iloc[second:])
    invested = exposures["open_positions"].gt(0)
    return {
        "version": name,
        "cagr": (equity.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0,
        "annualized_volatility": float(returns.std(ddof=1) * math.sqrt(252.0)),
        "sharpe_zero_cash": float(
            returns.mean() / returns.std(ddof=1) * math.sqrt(252.0)
        ),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "ending_equity": float(equity.iloc[-1]),
        "trades": len(trades),
        "win_rate": float(trades["realized_pnl"].gt(0).mean()),
        "average_exit_dte": float(trades["exit_dte"].mean()),
        "train_cagr": train_cagr,
        "train_sharpe": train_sharpe,
        "validation_cagr": validation_cagr,
        "validation_sharpe": validation_sharpe,
        "test_cagr": test_cagr,
        "test_sharpe": test_sharpe,
        "average_notional_pct_equity_all_days": float(
            exposures["notional_pct_current_equity"].mean()
        ),
        "average_notional_pct_equity_while_invested": float(
            exposures.loc[invested, "notional_pct_current_equity"].mean()
        ),
        "max_notional_pct_equity": float(
            exposures["notional_pct_current_equity"].max()
        ),
        "max_open_positions": int(exposures["open_positions"].max()),
        "max_defined_loss_pct_equity": float(
            exposures["maximum_loss_pct_current_equity"].max()
        ),
        "pct_days_flat": float(exposures["open_positions"].eq(0).mean()),
        "pct_days_one_position": float(exposures["open_positions"].eq(1).mean()),
        "pct_days_two_positions": float(exposures["open_positions"].eq(2).mean()),
        "pct_days_three_positions": float(exposures["open_positions"].eq(3).mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades, paths = prepare_trade_paths()
    market_index = pd.read_parquet(
        CACHE / "equity_curves.parquet", columns=[STORED_STRATEGY_ID]
    ).index

    compounded = simulate(
        trades, paths, market_index, compound_notional=True
    )
    fixed = simulate(trades, paths, market_index, compound_notional=False)
    compounded_equity, compounded_pnl, compounded_trades, compounded_exposures = compounded
    fixed_equity, fixed_pnl, fixed_trades, fixed_exposures = fixed

    summary = pd.DataFrame(
        [
            summarize(
                "100% of current equity per monthly entry",
                compounded_equity,
                compounded_pnl,
                compounded_trades,
                compounded_exposures,
            ),
            summarize(
                "Fixed $1 million per monthly entry",
                fixed_equity,
                fixed_pnl,
                fixed_trades,
                fixed_exposures,
            ),
        ]
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    compounded_trades.to_csv(OUT / "trades_compounded.csv", index=False)
    compounded_exposures.to_csv(OUT / "exposure_compounded.csv", index_label="date")

    old_equity = pd.read_parquet(
        CACHE / "equity_curves.parquet", columns=[STORED_STRATEGY_ID]
    ).iloc[:, 0]
    curves = pd.DataFrame(
        {
            "100pct_current_equity_per_entry": compounded_equity,
            "fixed_1m_per_entry": fixed_equity,
            "earlier_5pct_max_loss": old_equity,
        }
    )
    curves.to_csv(OUT / "equity_curves.csv", index_label="date")

    requested = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("white")
    years = (market_index - market_index[0]).days / 365.2425
    ax.fill_between(
        market_index,
        1.03**years - 1.0,
        1.05**years - 1.0,
        color="#bbf7d0",
        alpha=0.32,
        label="3%–5% annual return path",
        zorder=1,
    )
    ax.plot(
        market_index,
        compounded_equity / INITIAL_CAPITAL - 1.0,
        color="#087f8c",
        linewidth=2.8,
        label=(
            f"100% of equity per entry  |  CAGR {requested['cagr']:.2%}  |  "
            f"DD {requested['max_drawdown']:.2%}"
        ),
        zorder=4,
    )
    fixed_row = summary.iloc[1]
    ax.plot(
        market_index,
        fixed_equity / INITIAL_CAPITAL - 1.0,
        color="#2563eb",
        linewidth=1.8,
        linestyle="--",
        label=(
            f"Fixed $1M per entry  |  CAGR {fixed_row['cagr']:.2%}  |  "
            f"DD {fixed_row['max_drawdown']:.2%}"
        ),
        zorder=3,
    )
    old_metrics = pd.read_csv(ITM_RESULTS / "strategy_metrics.csv")
    old = old_metrics.loc[old_metrics["strategy_id"].eq(STORED_STRATEGY_ID)].iloc[0]
    ax.plot(
        market_index,
        old_equity / INITIAL_CAPITAL - 1.0,
        color="#94a3b8",
        linewidth=1.5,
        alpha=0.8,
        label=(
            f"Earlier max-loss sizing  |  CAGR {old['cagr']:.2%}  |  "
            f"DD {old['max_drawdown']:.2%}"
        ),
        zorder=2,
    )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_title(
        "SPX 103/100 monthly: 100% notional for every new trade",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.01,
        "~60 DTE · 25% profit target · $1 million starting equity · option P&L only",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    ax.set_ylabel("Cumulative option return", fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.07,
        0.015,
        "Historical backtest: Sep 2016–Sep 2026; realistic execution costs; fractional contracts; "
        "no cash interest. Current-equity sizing uses equity from the previous close.",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.98])
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Monthly 100% notional per entry\n\nThe requested version sells the 103/100 SPX put spread monthly near 60 DTE, exits at 25% of entry credit, begins with $1 million, and targets short-strike notional equal to 100% of portfolio equity for every new entry. Sizing uses equity from the previous close.\n\n| Version | CAGR | Volatility | Sharpe | Max drawdown | Ending equity |\n|---|---:|---:|---:|---:|---:|\n| 100% of current equity per entry | {requested['cagr']:.2%} | {requested['annualized_volatility']:.2%} | {requested['sharpe_zero_cash']:.2f} | {requested['max_drawdown']:.2%} | ${requested['ending_equity']:,.0f} |\n| Fixed $1 million per entry | {fixed_row['cagr']:.2%} | {fixed_row['annualized_volatility']:.2%} | {fixed_row['sharpe_zero_cash']:.2f} | {fixed_row['max_drawdown']:.2%} | ${fixed_row['ending_equity']:,.0f} |\n| Earlier 5% max-loss sizing | {old['cagr']:.2%} | {old['annualized_volatility']:.2%} | {old['sharpe_zero_cash']:.2f} | {old['max_drawdown']:.2%} | ${old['ending_equity']:,.0f} |\n\nFor the requested current-equity version, aggregate short-strike notional averaged {requested['average_notional_pct_equity_all_days']:.1%} of equity across all trading days and {requested['average_notional_pct_equity_while_invested']:.1%} while at least one trade was open. It peaked at {requested['max_notional_pct_equity']:.1%}, with a maximum of {int(requested['max_open_positions'])} open positions. Defined maximum loss peaked at {requested['max_defined_loss_pct_equity']:.2%} of equity.\n\nThe portfolio was flat on {requested['pct_days_flat']:.1%} of days, held one position on {requested['pct_days_one_position']:.1%}, two on {requested['pct_days_two_positions']:.1%}, and three on {requested['pct_days_three_positions']:.1%}.\n"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
