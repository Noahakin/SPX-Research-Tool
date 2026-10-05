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
RESULTS = PROJECT / "results/itm_structures"
OUT = PROJECT / "results/leveraged_103_100_profit_targets"

INITIAL_CAPITAL = 1_000_000.0
PARAMETER_ID = "vertical_103/100_monthly_dte60_m103"
PROFIT_LEVELS = list(range(5, 100, 5))
REQUESTED_LEVELS = [15, 25, 40, 60, 80]


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


def build_path(
    marks: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    profit_level: int,
) -> tuple[pd.Series, pd.DataFrame]:
    profit_fraction = profit_level / 100.0
    daily_changes: list[pd.DataFrame] = []
    exit_rows: list[pd.Series] = []

    for _, trade in marks.groupby("trade_id", sort=False):
        trade = trade.sort_values("mark_date").copy()
        eligible = trade[
            trade["is_final"]
            | (
                trade["mark_date"].gt(trade["entry_date"])
                & trade["pnl_realistic"].ge(
                    profit_fraction * trade["max_profit_per_unit"]
                )
            )
        ]
        exit_row = eligible.iloc[0]
        exit_rows.append(exit_row)
        path = trade.loc[trade["mark_date"].le(exit_row["mark_date"])].copy()
        path["cumulative_trade_pnl"] = path["pnl_mid"] * path["contracts"]
        on_exit = path["mark_date"].eq(exit_row["mark_date"])
        path.loc[on_exit, "cumulative_trade_pnl"] = (
            exit_row["pnl_realistic"] * exit_row["contracts"]
        )
        path["daily_pnl"] = path["cumulative_trade_pnl"].diff().fillna(
            path["cumulative_trade_pnl"]
        )
        daily_changes.append(path[["mark_date", "daily_pnl"]])

    daily = (
        pd.concat(daily_changes)
        .groupby("mark_date")["daily_pnl"]
        .sum()
        .reindex(market_index, fill_value=0.0)
    )
    equity = INITIAL_CAPITAL + daily.cumsum()
    exits = pd.DataFrame(exit_rows).reset_index(drop=True)
    return equity, exits


def summarize(
    equity: pd.Series,
    exits: pd.DataFrame,
    first_split: int,
    second_split: int,
    profit_level: int,
) -> dict[str, float | int | bool]:
    daily_pnl = equity.diff().fillna(equity.iloc[0] - INITIAL_CAPITAL)
    prior_equity = equity.shift(1).fillna(INITIAL_CAPITAL)
    returns = daily_pnl / prior_equity
    years = len(returns) / 252.0
    ending_ratio = float(equity.iloc[-1] / INITIAL_CAPITAL)
    peak = pd.Series(
        np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity.to_numpy()])[1:],
        index=equity.index,
    )
    train_cagr, train_sharpe = split_stats(returns.iloc[:first_split])
    validation_cagr, validation_sharpe = split_stats(
        returns.iloc[first_split:second_split]
    )
    test_cagr, test_sharpe = split_stats(returns.iloc[second_split:])
    volatility = float(returns.std(ddof=1) * math.sqrt(252.0))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252.0))
        if returns.std(ddof=1) > 0
        else np.nan
    )
    return {
        "profit_target": profit_level / 100.0,
        "trades": int(len(exits)),
        "cagr": ending_ratio ** (1.0 / years) - 1.0,
        "annualized_volatility": volatility,
        "sharpe_zero_cash": sharpe,
        "max_drawdown": float((equity / peak - 1.0).min()),
        "ending_equity": float(equity.iloc[-1]),
        "win_rate": float(exits["pnl_realistic"].gt(0).mean()),
        "average_exit_dte": float(exits["current_dte"].mean()),
        "median_exit_dte": float(exits["current_dte"].median()),
        "train_cagr": train_cagr,
        "train_sharpe": train_sharpe,
        "validation_cagr": validation_cagr,
        "validation_sharpe": validation_sharpe,
        "test_cagr": test_cagr,
        "test_sharpe": test_sharpe,
        "positive_all_splits": bool(
            train_cagr > 0 and validation_cagr > 0 and test_cagr > 0
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    marks = pd.read_parquet(
        CACHE / "trade_marks.parquet",
        filters=[("parameter_id", "==", PARAMETER_ID)],
    )
    marks["mark_date"] = pd.to_datetime(marks["mark_date"])
    market_index = pd.read_parquet(
        CACHE / "equity_curves.parquet",
        columns=[f"{PARAMETER_ID}_profit_25"],
    ).index
    first_split = int(len(market_index) * 0.60)
    second_split = int(len(market_index) * 0.80)

    curves: dict[int, pd.Series] = {}
    rows: list[dict[str, float | int | bool]] = []
    for level in PROFIT_LEVELS:
        equity, exits = build_path(marks, market_index, level)
        curves[level] = equity
        rows.append(summarize(equity, exits, first_split, second_split, level))

    metrics = pd.DataFrame(rows).sort_values("profit_target").reset_index(drop=True)
    metrics.to_csv(OUT / "all_profit_target_metrics.csv", index=False)
    requested = metrics.loc[metrics["profit_target"].mul(100).isin(REQUESTED_LEVELS)]
    requested.to_csv(OUT / "requested_profit_target_metrics.csv", index=False)
    pd.DataFrame(
        {f"profit_{level}": curves[level] for level in REQUESTED_LEVELS}
    ).to_csv(OUT / "requested_equity_curves.csv", index_label="date")

    stored = pd.read_parquet(
        CACHE / "equity_curves.parquet",
        columns=[f"{PARAMETER_ID}_profit_25"],
    ).iloc[:, 0]
    validation_max_difference = float((curves[25] - stored).abs().max())
    if validation_max_difference > 1e-6:
        raise RuntimeError(
            f"Reconstructed 25% curve differs from stored curve by "
            f"${validation_max_difference:,.6f}"
        )

    colors = {
        15: "#7c3aed",
        25: "#087f8c",
        40: "#2563eb",
        60: "#d97706",
        80: "#dc2626",
    }
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
    for level in REQUESTED_LEVELS:
        row = metrics.loc[metrics["profit_target"].eq(level / 100.0)].iloc[0]
        ax.plot(
            market_index,
            curves[level] / INITIAL_CAPITAL - 1.0,
            color=colors[level],
            linewidth=2.8 if level == 25 else 1.8,
            alpha=1.0 if level == 25 else 0.82,
            label=(
                f"{level}% target  |  CAGR {row['cagr']:.2%}  |  "
                f"Sharpe {row['sharpe_zero_cash']:.2f}  |  DD {row['max_drawdown']:.2%}"
            ),
            zorder=3 if level == 25 else 2,
        )
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_title(
        "SPX 103/100 monthly put spread: profit-target comparison",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.01,
        "~60 DTE · earlier 5% maximum-loss sizing · option P&L only · no cash interest",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    ax.set_ylabel("Cumulative option return", fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.8)
    ax.legend(loc="upper left", frameon=False, fontsize=9.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.07,
        0.015,
        "Historical backtest: Sep 2016–Sep 2026; realistic execution costs; continuous contracts. "
        "Exit occurs at the first available close meeting the target.",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout(rect=[0.04, 0.05, 0.99, 0.98])
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    best_cagr = metrics.loc[metrics["cagr"].idxmax()]
    best_sharpe = metrics.loc[metrics["sharpe_zero_cash"].idxmax()]
    report_lines = [
        "# Leveraged 103/100 profit-target comparison",
        "",
        "This uses the earlier monthly 103/100 spread near 60 DTE with the original 5% maximum-loss sizing. Results are option-only with no cash interest.",
        "",
        f"The highest CAGR among 5%-95% targets was the {best_cagr['profit_target']:.0%} target at {best_cagr['cagr']:.2%}. The highest zero-cash Sharpe was the {best_sharpe['profit_target']:.0%} target at {best_sharpe['sharpe_zero_cash']:.2f}.",
        "",
        "| Profit target | CAGR | Volatility | Sharpe | Max drawdown | Win rate | Average exit DTE | Train CAGR | Validation CAGR | Test CAGR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in requested.itertuples():
        report_lines.append(
            f"| {row.profit_target:.0%} | {row.cagr:.2%} | {row.annualized_volatility:.2%} | "
            f"{row.sharpe_zero_cash:.2f} | {row.max_drawdown:.2%} | {row.win_rate:.1%} | "
            f"{row.average_exit_dte:.1f} | {row.train_cagr:.2%} | {row.validation_cagr:.2%} | "
            f"{row.test_cagr:.2%} |"
        )
    report_lines.extend(
        [
            "",
            f"Reconstruction check against the previously stored 25% curve: maximum absolute difference ${validation_max_difference:,.6f}.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(requested.to_string(index=False))
    print(
        f"\nBest CAGR: {best_cagr['profit_target']:.0%} at {best_cagr['cagr']:.4%}; "
        f"best Sharpe: {best_sharpe['profit_target']:.0%} at "
        f"{best_sharpe['sharpe_zero_cash']:.3f}"
    )
    print(f"25% curve reconstruction max difference: ${validation_max_difference:,.8f}")
    print(f"Saved results to {OUT}")


if __name__ == "__main__":
    main()
