from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/leveraged_103_100_review"
ITM_RESULTS = PROJECT / "results/itm_structures"
ITM_CACHE = PROJECT / "data/cache/itm_structures"
NOTIONAL_RESULTS = ITM_RESULTS / "103_100_notional"

INITIAL_CAPITAL = 1_000_000.0
PARAMETER_ID = "vertical_103/100_monthly_dte60_m103"
STRATEGY_ID = f"{PARAMETER_ID}_profit_25"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    leveraged = pd.read_parquet(
        ITM_CACHE / "equity_curves.parquet", columns=[STRATEGY_ID]
    )[STRATEGY_ID]
    capped = pd.read_parquet(
        NOTIONAL_RESULTS / "equity_curves.parquet", columns=["monthly_profit_25"]
    )["monthly_profit_25"]
    equity = pd.concat(
        {
            "5% max-loss sizing": leveraged,
            "100% short-strike notional cap": capped,
        },
        axis=1,
    ).dropna()
    equity.to_csv(OUT / "equity_comparison.csv", index_label="date")

    old_metrics = pd.read_csv(ITM_RESULTS / "strategy_metrics.csv")
    old = old_metrics.loc[old_metrics["strategy_id"].eq(STRATEGY_ID)].iloc[0]
    new_metrics = pd.read_csv(NOTIONAL_RESULTS / "optimization_metrics.csv")
    new = new_metrics.loc[new_metrics["strategy_id"].eq("monthly_profit_25")].iloc[0]

    trades = pd.read_parquet(ITM_CACHE / "trades.parquet")
    trades = trades.loc[trades["parameter_id"].eq(PARAMETER_ID)].copy()
    exits = pd.read_parquet(ITM_CACHE / "trade_exits.parquet")
    exits = exits.loc[
        exits["parameter_id"].eq(PARAMETER_ID)
        & exits["exit_rule"].eq("profit_25"),
        ["trade_id", "exit_date"],
    ]
    positions = trades.merge(exits, on="trade_id", how="inner")
    positions["entry_date"] = pd.to_datetime(positions["entry_date"])
    positions["exit_date"] = pd.to_datetime(positions["exit_date"])
    positions["short_strike_notional"] = (
        positions["contracts"]
        * positions["spot_entry"]
        * positions["short_strike_ratio"]
        * 100.0
    )
    positions["maximum_loss"] = (
        positions["contracts"] * positions["max_loss_per_unit"]
    )

    aggregate_notional = pd.Series(0.0, index=equity.index)
    aggregate_max_loss = pd.Series(0.0, index=equity.index)
    open_positions = pd.Series(0, index=equity.index)
    for trade in positions.itertuples():
        # Treat an exit and a new entry on the same date in execution order: the
        # old position is closed before measuring the newly opened portfolio.
        active = (equity.index >= trade.entry_date) & (equity.index < trade.exit_date)
        aggregate_notional.loc[active] += trade.short_strike_notional
        aggregate_max_loss.loc[active] += trade.maximum_loss
        open_positions.loc[active] += 1

    active = aggregate_notional.gt(0)
    max_notional_initial = aggregate_notional.max() / INITIAL_CAPITAL
    average_active_notional_initial = (
        aggregate_notional.loc[active].mean() / INITIAL_CAPITAL
    )
    max_notional_equity = (
        aggregate_notional / equity["5% max-loss sizing"]
    ).max()
    max_concurrent_loss_initial = aggregate_max_loss.max() / INITIAL_CAPITAL

    summary = pd.DataFrame(
        [
            {
                "version": "Earlier leveraged result",
                "sizing": "5% maximum-loss budget split across expected overlapping positions",
                "cagr": old["cagr"],
                "annualized_volatility": old["annualized_volatility"],
                "sharpe_zero_cash": old["sharpe_zero_cash"],
                "max_drawdown": old["max_drawdown"],
                "ending_equity": old["ending_equity"],
                "trades": old["trades"],
                "win_rate": old["win_rate"],
                "average_exit_dte": old["average_exit_dte"],
                "train_cagr": old["train_cagr"],
                "validation_cagr": old["validation_cagr"],
                "test_cagr": old["test_cagr"],
                "max_short_strike_notional_pct": max_notional_initial,
                "average_active_short_strike_notional_pct": average_active_notional_initial,
                "max_concurrent_max_loss_pct": max_concurrent_loss_initial,
            },
            {
                "version": "Same rules, no-leverage cap",
                "sizing": "100% aggregate short-strike notional cap",
                "cagr": new["cagr"],
                "annualized_volatility": new["annualized_volatility"],
                "sharpe_zero_cash": new["sharpe_zero_cash"],
                "max_drawdown": new["max_drawdown"],
                "ending_equity": new["ending_equity"],
                "trades": new["trades"],
                "win_rate": new["win_rate"],
                "average_exit_dte": new["average_exit_dte"],
                "train_cagr": new["train_cagr"],
                "validation_cagr": new["validation_cagr"],
                "test_cagr": new["test_cagr"],
                "max_short_strike_notional_pct": new["max_actual_notional_pct"],
                "average_active_short_strike_notional_pct": float("nan"),
                "max_concurrent_max_loss_pct": float("nan"),
            },
        ]
    )
    summary.to_csv(OUT / "summary.csv", index=False)

    returns = equity / INITIAL_CAPITAL - 1.0
    years = (equity.index - equity.index[0]).days / 365.2425
    lower_target = 1.03**years - 1.0
    upper_target = 1.05**years - 1.0

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("white")
    ax.fill_between(
        equity.index,
        lower_target,
        upper_target,
        color="#bbf7d0",
        alpha=0.38,
        label="3%–5% annual return path",
        zorder=1,
    )
    ax.plot(
        returns.index,
        returns["5% max-loss sizing"],
        color="#087f8c",
        linewidth=2.8,
        label=(
            f"Earlier 5% max-loss sizing  |  CAGR {old['cagr']:.2%}  |  "
            f"max DD {old['max_drawdown']:.2%}"
        ),
        zorder=3,
    )
    ax.plot(
        returns.index,
        returns["100% short-strike notional cap"],
        color="#64748b",
        linewidth=2.2,
        label=(
            f"100% notional cap  |  CAGR {new['cagr']:.2%}  |  "
            f"max DD {new['max_drawdown']:.2%}"
        ),
        zorder=2,
    )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_title(
        "The earlier standout: SPX 103/100 monthly put spread",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.01,
        "~60 DTE · take profit at 25% of entry credit · option P&L only · no cash interest",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    ax.set_ylabel("Cumulative option return", fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.text(
        0.99,
        0.03,
        (
            f"Earlier sizing reached {max_notional_initial:.1%} short-strike notional "
            f"({max_notional_equity:.1%} of then-current equity)\n"
            f"and {max_concurrent_loss_initial:.2%} concurrent defined maximum loss."
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f1f5f9", "edgecolor": "#cbd5e1"},
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.07,
        0.015,
        "Historical backtest: Sep 2016–Sep 2026; realistic execution costs; continuous contracts. "
        "The green band compounds from the common start date.",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.98])
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = f"""# Earlier leveraged 103/100 result\n\nThe standout strategy was the monthly 103/100 SPX put spread entered near 60 DTE and closed when 25% of the initial credit had been earned. Results use option P&L only and no cash interest.\n\n| Version | CAGR | Volatility | Zero-cash Sharpe | Max drawdown | Ending equity |\n|---|---:|---:|---:|---:|---:|\n| Earlier 5% max-loss sizing | {old['cagr']:.2%} | {old['annualized_volatility']:.2%} | {old['sharpe_zero_cash']:.2f} | {old['max_drawdown']:.2%} | ${old['ending_equity']:,.0f} |\n| 100% short-strike notional cap | {new['cagr']:.2%} | {new['annualized_volatility']:.2%} | {new['sharpe_zero_cash']:.2f} | {new['max_drawdown']:.2%} | ${new['ending_equity']:,.0f} |\n\nThe earlier version used a configured 5% maximum-loss budget divided across the number of positions expected to overlap. Its aggregate short-strike notional averaged {average_active_notional_initial:.1%} of original capital while positions were open and peaked at {max_notional_initial:.1%}. Relative to then-current equity, the peak was {max_notional_equity:.1%}. Irregular monthly spacing caused concurrent defined maximum loss to peak at {max_concurrent_loss_initial:.2%} of original capital.\n\nThe earlier version completed {int(old['trades'])} trades, won {old['win_rate']:.1%}, and exited with {old['average_exit_dte']:.1f} DTE on average. Its train, validation, and test CAGRs were {old['train_cagr']:.2%}, {old['validation_cagr']:.2%}, and {old['test_cagr']:.2%}.\n"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
