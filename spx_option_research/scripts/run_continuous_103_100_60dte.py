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
CACHE = PROJECT / "data/cache/sequential_itm_hedges"
MONTHLY_RESULTS = PROJECT / "results/monthly_100pct_notional_per_entry"
OUT = PROJECT / "results/continuous_103_100_60dte"

INITIAL_CAPITAL = 1_000_000.0
PARAMETER_ID = "103/100_dte60"
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


def choose_exits(marks: pd.DataFrame) -> pd.DataFrame:
    exit_rows: list[pd.Series] = []
    for _, trade in marks.groupby("trade_id", sort=False):
        trade = trade.sort_values("mark_date")
        eligible = trade[
            trade["is_final"]
            | (
                trade["mark_date"].gt(trade["entry_date"])
                & trade["core_pnl_realistic"].ge(
                    PROFIT_FRACTION * trade["core_max_profit"]
                )
            )
        ]
        exit_rows.append(eligible.iloc[0])
    exits = pd.DataFrame(exit_rows).reset_index(drop=True)
    return exits.rename(
        columns={
            "mark_date": "exit_date",
            "current_dte": "exit_dte",
            "core_pnl_realistic": "realized_pnl_per_spread",
        }
    )


def select_sequence(exits: pd.DataFrame, *, same_day_reentry: bool) -> pd.DataFrame:
    exits = exits.sort_values("entry_date").copy()
    selected: list[int] = []
    available_after = pd.Timestamp.min
    for index, trade in exits.iterrows():
        entry_date = pd.Timestamp(trade["entry_date"])
        eligible = (
            entry_date >= available_after
            if same_day_reentry
            else entry_date > available_after
        )
        if eligible:
            selected.append(index)
            available_after = pd.Timestamp(trade["exit_date"])
    sequence = exits.loc[selected].sort_values("entry_date").reset_index(drop=True)
    sequence["sequence_number"] = np.arange(1, len(sequence) + 1)
    return sequence


def simulate(
    sequence: pd.DataFrame,
    marks: pd.DataFrame,
    market_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    mark_groups = {
        trade_id: group.sort_values("mark_date")
        for trade_id, group in marks.groupby("trade_id", sort=False)
    }
    equity = INITIAL_CAPITAL
    daily_pnl = pd.Series(0.0, index=market_index)
    detail_rows: list[dict[str, object]] = []

    for trade in sequence.itertuples():
        pre_trade_equity = equity
        contracts = pre_trade_equity / (float(trade.short_strike) * 100.0)
        path = mark_groups[trade.trade_id]
        path = path.loc[path["mark_date"].le(pd.Timestamp(trade.exit_date))].copy()
        cumulative = path.set_index("mark_date")["core_pnl_mid"].astype(float)
        cumulative.iloc[-1] = float(trade.realized_pnl_per_spread)
        scaled = cumulative * contracts
        changes = scaled.diff().fillna(scaled.iloc[0])
        daily_pnl.loc[changes.index] += changes
        dollar_pnl = contracts * float(trade.realized_pnl_per_spread)
        equity += dollar_pnl
        detail_rows.append(
            {
                "sequence_number": int(trade.sequence_number),
                "trade_id": trade.trade_id,
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "expiration_date": trade.expiration_date,
                "actual_dte": trade.actual_dte,
                "exit_dte": trade.exit_dte,
                "holding_calendar_days": (
                    pd.Timestamp(trade.exit_date) - pd.Timestamp(trade.entry_date)
                ).days,
                "spot_entry": trade.spot_entry,
                "short_strike": trade.short_strike,
                "long_strike": float(trade.short_strike)
                - (float(trade.core_max_profit) + float(trade.core_max_loss)) / 100.0,
                "entry_credit_per_spread": trade.core_entry_cash,
                "max_loss_per_spread": trade.core_max_loss,
                "contracts": contracts,
                "entry_equity": pre_trade_equity,
                "entry_notional": contracts * float(trade.short_strike) * 100.0,
                "defined_maximum_loss": contracts * float(trade.core_max_loss),
                "realized_pnl_per_spread": trade.realized_pnl_per_spread,
                "dollar_pnl": dollar_pnl,
                "trade_return": dollar_pnl / pre_trade_equity,
                "post_trade_equity": equity,
            }
        )

    equity_curve = INITIAL_CAPITAL + daily_pnl.cumsum()
    detail = pd.DataFrame(detail_rows)

    exposure_rows: list[dict[str, object]] = []
    for date, current_equity in equity_curve.items():
        active = detail[
            detail["entry_date"].le(date) & detail["exit_date"].gt(date)
        ]
        notional = float(active["entry_notional"].sum())
        maximum_loss = float(active["defined_maximum_loss"].sum())
        exposure_rows.append(
            {
                "date": date,
                "equity": current_equity,
                "open_positions": len(active),
                "aggregate_short_strike_notional": notional,
                "notional_pct_equity": notional / current_equity,
                "defined_maximum_loss": maximum_loss,
                "defined_maximum_loss_pct_equity": maximum_loss / current_equity,
            }
        )
    exposure = pd.DataFrame(exposure_rows).set_index("date")
    return equity_curve, daily_pnl, detail, exposure


def summarize(
    name: str,
    reentry_rule: str,
    equity: pd.Series,
    daily_pnl: pd.Series,
    trades: pd.DataFrame,
    exposure: pd.DataFrame,
) -> pd.DataFrame:
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
    invested = exposure["open_positions"].gt(0)
    row = {
        "strategy": name,
        "entry_sizing": "100% of current equity short-strike notional",
        "reentry_rule": reentry_rule,
        "trades": len(trades),
        "cagr": (equity.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0,
        "annualized_volatility": float(returns.std(ddof=1) * math.sqrt(252.0)),
        "sharpe_zero_cash": float(
            returns.mean() / returns.std(ddof=1) * math.sqrt(252.0)
        ),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "ending_equity": float(equity.iloc[-1]),
        "win_rate": float(trades["dollar_pnl"].gt(0).mean()),
        "average_holding_calendar_days": float(trades["holding_calendar_days"].mean()),
        "average_exit_dte": float(trades["exit_dte"].mean()),
        "train_cagr": train_cagr,
        "train_sharpe": train_sharpe,
        "validation_cagr": validation_cagr,
        "validation_sharpe": validation_sharpe,
        "test_cagr": test_cagr,
        "test_sharpe": test_sharpe,
        "average_notional_pct_equity_all_days": float(
            exposure["notional_pct_equity"].mean()
        ),
        "average_notional_pct_equity_while_invested": float(
            exposure.loc[invested, "notional_pct_equity"].mean()
        ),
        "max_notional_pct_equity": float(exposure["notional_pct_equity"].max()),
        "max_open_positions": int(exposure["open_positions"].max()),
        "max_defined_loss_pct_equity": float(
            exposure["defined_maximum_loss_pct_equity"].max()
        ),
        "pct_days_invested": float(invested.mean()),
    }
    return pd.DataFrame([row])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(
        CACHE / "core_candidates.parquet",
        filters=[("parameter_id", "==", PARAMETER_ID)],
    )
    marks = pd.read_parquet(
        CACHE / "core_marks.parquet",
        filters=[("parameter_id", "==", PARAMETER_ID)],
    )
    for column in ("entry_date", "expiration_date", "mark_date"):
        if column in marks:
            marks[column] = pd.to_datetime(marks[column])
    for column in ("entry_date", "expiration_date"):
        candidates[column] = pd.to_datetime(candidates[column])

    exits = choose_exits(marks)
    exit_columns = [
        "trade_id",
        "exit_date",
        "exit_dte",
        "realized_pnl_per_spread",
    ]
    exits = candidates.merge(exits[exit_columns], on="trade_id", how="inner")
    # Near the dataset boundary a newly selected contract can have only its
    # entry mark and be labeled final immediately. It is not a testable trade.
    exits = exits.loc[exits["exit_date"].gt(exits["entry_date"])].copy()
    same_day_sequence = select_sequence(exits, same_day_reentry=True)
    next_session_sequence = select_sequence(exits, same_day_reentry=False)

    monthly_curves = pd.read_csv(
        MONTHLY_RESULTS / "equity_curves.csv", parse_dates=["date"], index_col="date"
    )
    market_index = monthly_curves.index
    equity, daily_pnl, trades, exposure = simulate(
        same_day_sequence, marks, market_index
    )
    next_equity, next_daily_pnl, next_trades, next_exposure = simulate(
        next_session_sequence, marks, market_index
    )
    same_day_summary = summarize(
        "Continuous same-day re-entry",
        "close old and open new on the same closing surface",
        equity,
        daily_pnl,
        trades,
        exposure,
    )
    next_session_summary = summarize(
        "Continuous next-session re-entry",
        "next available option-surface date after exit",
        next_equity,
        next_daily_pnl,
        next_trades,
        next_exposure,
    )
    summary = pd.concat([same_day_summary, next_session_summary], ignore_index=True)

    summary.to_csv(OUT / "summary.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)
    exposure.to_csv(OUT / "exposure.csv", index_label="date")
    next_trades.to_csv(OUT / "trades_next_session.csv", index=False)
    next_exposure.to_csv(OUT / "exposure_next_session.csv", index_label="date")
    pd.DataFrame({"equity": equity, "daily_pnl": daily_pnl}).to_csv(
        OUT / "equity_curve.csv", index_label="date"
    )
    pd.DataFrame(
        {"equity": next_equity, "daily_pnl": next_daily_pnl}
    ).to_csv(OUT / "equity_curve_next_session.csv", index_label="date")

    monthly_summary = pd.read_csv(MONTHLY_RESULTS / "summary.csv").iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "version": "Continuous same-day re-entry",
                "cagr": summary.iloc[0]["cagr"],
                "annualized_volatility": summary.iloc[0]["annualized_volatility"],
                "sharpe_zero_cash": summary.iloc[0]["sharpe_zero_cash"],
                "max_drawdown": summary.iloc[0]["max_drawdown"],
                "ending_equity": summary.iloc[0]["ending_equity"],
                "trades": summary.iloc[0]["trades"],
            },
            {
                "version": "Continuous next-session re-entry",
                "cagr": summary.iloc[1]["cagr"],
                "annualized_volatility": summary.iloc[1]["annualized_volatility"],
                "sharpe_zero_cash": summary.iloc[1]["sharpe_zero_cash"],
                "max_drawdown": summary.iloc[1]["max_drawdown"],
                "ending_equity": summary.iloc[1]["ending_equity"],
                "trades": summary.iloc[1]["trades"],
            },
            {
                "version": "Monthly 100% per entry",
                "cagr": monthly_summary["cagr"],
                "annualized_volatility": monthly_summary["annualized_volatility"],
                "sharpe_zero_cash": monthly_summary["sharpe_zero_cash"],
                "max_drawdown": monthly_summary["max_drawdown"],
                "ending_equity": monthly_summary["ending_equity"],
                "trades": monthly_summary["trades"],
            },
        ]
    )
    comparison.to_csv(OUT / "comparison.csv", index=False)

    row = summary.iloc[0]
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
        equity / INITIAL_CAPITAL - 1.0,
        color="#087f8c",
        linewidth=2.8,
        label=(
            f"Same-day re-entry  |  CAGR {row['cagr']:.2%}  |  "
            f"Sharpe {row['sharpe_zero_cash']:.2f}  |  DD {row['max_drawdown']:.2%}"
        ),
        zorder=3,
    )
    next_row = summary.iloc[1]
    ax.plot(
        market_index,
        next_equity / INITIAL_CAPITAL - 1.0,
        color="#2563eb",
        linewidth=1.7,
        linestyle="--",
        label=(
            f"Next-session re-entry  |  CAGR {next_row['cagr']:.2%}  |  "
            f"Sharpe {next_row['sharpe_zero_cash']:.2f}  |  "
            f"DD {next_row['max_drawdown']:.2%}"
        ),
        zorder=2.5,
    )
    monthly_equity = monthly_curves["100pct_current_equity_per_entry"]
    ax.plot(
        market_index,
        monthly_equity / INITIAL_CAPITAL - 1.0,
        color="#64748b",
        linewidth=1.8,
        label=(
            f"Calendar-month entries  |  CAGR {monthly_summary['cagr']:.2%}  |  "
            f"Sharpe {monthly_summary['sharpe_zero_cash']:.2f}  |  "
            f"DD {monthly_summary['max_drawdown']:.2%}"
        ),
        zorder=2,
    )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_title(
        "SPX 103/100: reopen a fresh 60-DTE spread after every exit",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.01,
        "25% profit target · one position at a time · 100% current-equity notional · no cash interest",
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
        "Historical backtest: Sep 2016–Sep 2026; realistic execution costs; fractional contracts. "
        "The primary curve closes and reopens using the same day's closing surface.",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.98])
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Continuous 103/100 60-DTE sequence\n\nThis version holds one 103/100 SPX put spread at a time, targets 60 DTE, closes when 25% of the initial credit has been earned or at the final available mark, and immediately opens a new spread using the same day's closing option surface. Every entry is sized to 100% of current portfolio short-strike notional. Results include realistic execution costs and no cash interest. The next-session version is included as a conservative execution sensitivity.\n\n| Version | CAGR | Volatility | Sharpe | Max drawdown | Ending equity | Trades |\n|---|---:|---:|---:|---:|---:|---:|\n| Continuous same-day re-entry | {row['cagr']:.2%} | {row['annualized_volatility']:.2%} | {row['sharpe_zero_cash']:.2f} | {row['max_drawdown']:.2%} | ${row['ending_equity']:,.0f} | {int(row['trades'])} |\n| Continuous next-session re-entry | {next_row['cagr']:.2%} | {next_row['annualized_volatility']:.2%} | {next_row['sharpe_zero_cash']:.2f} | {next_row['max_drawdown']:.2%} | ${next_row['ending_equity']:,.0f} | {int(next_row['trades'])} |\n| Calendar-month entries | {monthly_summary['cagr']:.2%} | {monthly_summary['annualized_volatility']:.2%} | {monthly_summary['sharpe_zero_cash']:.2f} | {monthly_summary['max_drawdown']:.2%} | ${monthly_summary['ending_equity']:,.0f} | {int(monthly_summary['trades'])} |\n\nThe same-day sequence was invested on {row['pct_days_invested']:.1%} of trading days. Notional averaged {row['average_notional_pct_equity_while_invested']:.1%} of equity while invested and peaked at {row['max_notional_pct_equity']:.1%}; no more than {int(row['max_open_positions'])} position was open. Defined maximum loss peaked at {row['max_defined_loss_pct_equity']:.2%} of equity. Average holding time was {row['average_holding_calendar_days']:.1f} calendar days, and the average exit occurred with {row['average_exit_dte']:.1f} DTE remaining.\n\nTrain, validation, and test CAGRs for the same-day sequence were {row['train_cagr']:.2%}, {row['validation_cagr']:.2%}, and {row['test_cagr']:.2%}.\n"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    print(summary.to_string(index=False))
    print("\nComparison")
    print(comparison.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
