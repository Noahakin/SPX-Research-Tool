from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import backtest_vix_30d_accrual_calls as base


OUT = base.PROJECT / "results/vix_30d_spike_exit"
SPIKE_THRESHOLDS = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def prepare_mark_values(entries: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    metadata = entries[
        [
            "trade_id",
            "target_delta",
            "entry_date",
            "expiration_date",
            "spot_entry",
            "contracts",
            "premium_spent",
        ]
    ]
    valued = marks.merge(metadata, on="trade_id", how="inner")
    valued = valued[
        valued["mark_date"].ge(valued["entry_date"])
        & valued["mark_date"].lt(valued["expiration_date"])
    ].copy()
    valued["vix_return_from_entry"] = valued["underlying_price"] / valued["spot_entry"] - 1.0
    valued["liquidation_price"] = np.maximum(
        0.0, valued["mid"] - base.SPREAD_FRACTION * (valued["ask"] - valued["bid"])
    )
    gross = valued["liquidation_price"] * base.MULTIPLIER
    valued["liquidation_value"] = (
        np.where(gross > 0, np.maximum(0.0, gross - base.COMMISSION), 0.0)
        * valued["contracts"]
    )
    valued["marked_option_pnl"] = valued["liquidation_value"] - valued["premium_spent"]
    return valued.sort_values(["trade_id", "mark_date"]).reset_index(drop=True)


def choose_exits(
    entries: pd.DataFrame,
    valued: pd.DataFrame,
    spike_threshold: float | None,
    require_profit: bool,
) -> pd.DataFrame:
    final_idx = valued.groupby("trade_id")["mark_date"].idxmax()
    selected = valued.loc[final_idx].copy()
    selected["trigger_hit"] = False

    if spike_threshold is not None:
        trigger = valued[
            valued["mark_date"].gt(valued["entry_date"])
            & valued["vix_return_from_entry"].ge(spike_threshold)
        ].copy()
        if require_profit:
            trigger = trigger[trigger["marked_option_pnl"].gt(0)]
        if not trigger.empty:
            trigger_idx = trigger.groupby("trade_id")["mark_date"].idxmin()
            first_trigger = trigger.loc[trigger_idx].copy()
            first_trigger["trigger_hit"] = True
            hit_ids = set(first_trigger["trade_id"])
            selected = pd.concat(
                [selected[~selected["trade_id"].isin(hit_ids)], first_trigger],
                ignore_index=True,
            )

    selected = selected.rename(
        columns={
            "mark_date": "exit_date",
            "underlying_price": "spot_exit",
            "dte": "exit_quote_dte",
            "bid": "exit_bid",
            "ask": "exit_ask",
            "mid": "exit_mid",
            "liquidation_price": "exit_modeled_price",
            "liquidation_value": "exit_proceeds",
            "marked_option_pnl": "option_pnl",
        }
    )
    cols = [
        "trade_id",
        "exit_date",
        "exit_quote_dte",
        "spot_exit",
        "vix_return_from_entry",
        "exit_bid",
        "exit_ask",
        "exit_mid",
        "exit_modeled_price",
        "exit_proceeds",
        "option_pnl",
        "trigger_hit",
    ]
    trades = entries.merge(selected[cols], on="trade_id", how="inner")
    trades["holding_days"] = (trades["exit_date"] - trades["entry_date"]).dt.days
    return trades.sort_values(["target_delta", "entry_date"]).reset_index(drop=True)


def strategy_label(target_delta: int, threshold: float | None, require_profit: bool) -> str:
    if threshold is None:
        return f"{target_delta}d hold"
    suffix = " + profitable" if require_profit else ""
    threshold_label = f"{threshold * 100:g}%"
    return f"{target_delta}d / VIX +{threshold_label}{suffix}"


def strategy_metrics(
    trades: pd.DataFrame,
    threshold: float | None,
    require_profit: bool,
) -> dict[str, object]:
    target_delta = int(trades["target_delta"].iloc[0])
    total_accrual = len(trades) * base.MONTHLY_ACCRUAL
    final_equity = (
        base.INITIAL_PRINCIPAL
        + total_accrual
        - trades["premium_spent"].sum()
        + trades["exit_proceeds"].sum()
    )
    years = (trades["exit_date"].max() - trades["entry_date"].min()).days / 365.25
    return {
        "strategy": strategy_label(target_delta, threshold, require_profit),
        "target_delta": target_delta,
        "spike_threshold": threshold,
        "require_profit": require_profit,
        "trades": len(trades),
        "triggered_exits": int(trades["trigger_hit"].sum()),
        "average_holding_days": trades["holding_days"].mean(),
        "winning_trades": int(trades["option_pnl"].gt(0).sum()),
        "win_rate": trades["option_pnl"].gt(0).mean(),
        "total_premium_spent": trades["premium_spent"].sum(),
        "total_exit_proceeds": trades["exit_proceeds"].sum(),
        "option_pnl": trades["option_pnl"].sum(),
        "proceeds_divided_by_premium": trades["exit_proceeds"].sum()
        / trades["premium_spent"].sum(),
        "final_equity": final_equity,
        "total_return": final_equity / base.INITIAL_PRINCIPAL - 1.0,
        "cagr": (final_equity / base.INITIAL_PRINCIPAL) ** (1.0 / years) - 1.0,
        "best_trade_pnl": trades["option_pnl"].max(),
        "worst_trade_pnl": trades["option_pnl"].min(),
    }


def run_grid(entries: pd.DataFrame, valued: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics: list[dict[str, object]] = []
    trade_sets: dict[str, pd.DataFrame] = {}
    for target_delta in base.DELTA_TARGETS:
        delta_entries = entries[entries["target_delta"].eq(target_delta)].copy()
        delta_values = valued[valued["target_delta"].eq(target_delta)].copy()
        hold = choose_exits(delta_entries, delta_values, None, False)
        label = strategy_label(target_delta, None, False)
        trade_sets[label] = hold
        metrics.append(strategy_metrics(hold, None, False))
        for require_profit in [False, True]:
            for threshold in SPIKE_THRESHOLDS:
                trades = choose_exits(delta_entries, delta_values, threshold, require_profit)
                label = strategy_label(target_delta, threshold, require_profit)
                trade_sets[label] = trades
                metrics.append(strategy_metrics(trades, threshold, require_profit))
    return pd.DataFrame(metrics), trade_sets


def build_curve(
    trades: pd.DataFrame,
    valued: pd.DataFrame,
    dates: pd.DatetimeIndex,
    strategy: str,
) -> pd.DataFrame:
    selected_marks = valued.merge(
        trades[["trade_id", "exit_date"]], on="trade_id", how="inner"
    )
    selected_marks = selected_marks[
        selected_marks["mark_date"].le(selected_marks["exit_date"])
    ]
    grid = (
        selected_marks.pivot_table(
            index="mark_date", columns="trade_id", values="liquidation_value", aggfunc="last"
        )
        .reindex(dates)
        .ffill()
    )
    for _, trade in trades.iterrows():
        if trade["trade_id"] not in grid.columns:
            continue
        # Exit proceeds enter cash on the exit date; do not also count the
        # option's liquidation value on that same date.
        inactive = (grid.index < trade["entry_date"]) | (grid.index >= trade["exit_date"])
        grid.loc[inactive, trade["trade_id"]] = 0.0
    open_value = grid.fillna(0.0).sum(axis=1)

    daily = pd.DataFrame({"date": dates})
    daily["accrual"] = daily["date"].isin(trades["entry_date"]).astype(float) * base.MONTHLY_ACCRUAL
    spent = trades.groupby("entry_date")["premium_spent"].sum()
    proceeds = trades.groupby("exit_date")["exit_proceeds"].sum()
    daily["premium_spent"] = daily["date"].map(spent).fillna(0.0)
    daily["exit_proceeds"] = daily["date"].map(proceeds).fillna(0.0)
    daily["cash"] = base.INITIAL_PRINCIPAL + (
        daily["accrual"] - daily["premium_spent"] + daily["exit_proceeds"]
    ).cumsum()
    daily["open_option_value"] = open_value.to_numpy()
    daily["equity"] = daily["cash"] + daily["open_option_value"]
    daily["strategy"] = strategy
    return daily[["date", "strategy", "equity"]]


def selected_curves(
    summary: pd.DataFrame,
    trade_sets: dict[str, pd.DataFrame],
    valued: pd.DataFrame,
    paths: dict[pd.Timestamp, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Treat a 10% VIX rise as the minimum actual spike. Smaller 5% and 7.5%
    # thresholds remain in the sensitivity grid but are excluded here.
    filtered = summary[summary["spike_threshold"].ge(0.10)].copy()
    best = filtered.loc[filtered.groupby("target_delta")["final_equity"].idxmax()].copy()
    overall = best.loc[best["final_equity"].idxmax()]
    selected_labels = best["strategy"].tolist()
    hold_label = strategy_label(int(overall["target_delta"]), None, False)
    if hold_label not in selected_labels:
        selected_labels.append(hold_label)
    dates = pd.DatetimeIndex(sorted(paths))
    start = min(trade_sets[label]["entry_date"].min() for label in selected_labels)
    end = max(trade_sets[label]["exit_date"].max() for label in selected_labels)
    dates = dates[(dates >= start) & (dates <= end)]

    curves: list[pd.DataFrame] = []
    baseline = pd.DataFrame({"date": dates})
    entry_dates = sorted(trade_sets[selected_labels[0]]["entry_date"].unique())
    baseline["equity"] = base.INITIAL_PRINCIPAL + baseline["date"].isin(entry_dates).cumsum() * base.MONTHLY_ACCRUAL
    baseline["strategy"] = "Retain 5% accrual"
    curves.append(baseline[["date", "strategy", "equity"]])
    for label in selected_labels:
        curves.append(build_curve(trade_sets[label], valued, dates, label))
    return pd.concat(curves, ignore_index=True), best.sort_values("target_delta")


def save_chart(curves: pd.DataFrame, best: pd.DataFrame, overall: pd.Series) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    fig.patch.set_facecolor("#f8fafc")
    colors = {
        "Retain 5% accrual": "#64748b",
        5: "#94a3b8",
        10: "#0f766e",
        15: "#2563eb",
        20: "#7c3aed",
        25: "#dc2626",
    }
    hold_label = strategy_label(int(overall["target_delta"]), None, False)
    for ax in axes:
        ax.set_facecolor("#ffffff")
        for strategy, group in curves.groupby("strategy", sort=False):
            if ax is axes[1] and strategy == "Retain 5% accrual":
                continue
            if strategy == "Retain 5% accrual":
                color, style, width = colors[strategy], "--", 2.8
            elif strategy == hold_label:
                color, style, width = "#0f172a", ":", 2.0
            else:
                delta = int(strategy.split("d")[0])
                color, style, width = colors[delta], "-", 2.1
            ax.plot(group["date"], group["equity"], label=strategy, color=color, linestyle=style, linewidth=width)
        ax.axhline(base.INITIAL_PRINCIPAL, color="#0f172a", linewidth=1.0, alpha=0.35)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value/1_000_000:.2f}M"))
        ax.set_ylabel("Portfolio equity")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title(
        "VIX-spike exits on monthly 30-DTE calls",
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    axes[0].text(
        0,
        1.01,
        r"\$1 million principal · 5% annual accrual · first close above trigger · modeled execution",
        transform=axes[0].transAxes,
        fontsize=11.5,
        color="#475569",
    )
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].set_title("Spike-exit portfolios (expanded scale)", fontsize=13, loc="left")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].text(
        0.995,
        0.015,
        "Source: local VIX option archive, Jan 2022–Jul 2026",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_heatmaps(summary: pd.DataFrame) -> None:
    grid = summary[summary["spike_threshold"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    for ax, require_profit in zip(axes, [False, True]):
        subset = grid[grid["require_profit"].eq(require_profit)]
        pivot = subset.pivot(index="target_delta", columns="spike_threshold", values="final_equity")
        values = (pivot.to_numpy() - base.INITIAL_PRINCIPAL) / 1_000.0
        im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(50, np.nanmax(values)))
        ax.set_xticks(range(len(pivot.columns)), [f"{v * 100:g}%" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{v}d" for v in pivot.index])
        ax.set_xlabel("VIX rise from entry")
        ax.set_title("VIX trigger only" if not require_profit else "VIX trigger + profitable call")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"${values[i, j]:.0f}k", ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Entry call delta")
    fig.suptitle("Ending equity above $1 million", fontsize=18, fontweight="bold", x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "exit_grid_heatmap.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, best: pd.DataFrame, overall: pd.Series) -> None:
    lines = [
        "# VIX spike exit test",
        "",
        "Calls are bought on the first trading day of each month with the $4,166.67 monthly accrual. The exit occurs on the first daily close where VIX has risen by the specified percentage from its entry level. The profitable variant also requires the executable call value to exceed its purchase cost. Calls that never trigger are sold at the final valid pre-expiration quote.",
        "",
        "## Best spike rule by call delta",
        "",
        "| Delta | VIX trigger | Profit condition | Triggered | Winners | Proceeds / premium | Final equity | CAGR |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| {row['target_delta']:.0f} | +{row['spike_threshold'] * 100:g}% | "
            f"{'yes' if row['require_profit'] else 'no'} | {row['triggered_exits']:.0f}/{row['trades']:.0f} | "
            f"{row['winning_trades']:.0f}/{row['trades']:.0f} | {row['proceeds_divided_by_premium']:.2f}x | "
            f"${row['final_equity']:,.0f} | {row['cagr']:.2%} |"
        )
    hold = summary[
        summary["target_delta"].eq(overall["target_delta"])
        & summary["spike_threshold"].isna()
    ].iloc[0]
    lines.extend(
        [
            "",
            f"The best in-sample rule was **{overall['strategy']}**, ending at **${overall['final_equity']:,.0f}** versus **${hold['final_equity']:,.0f}** for the same call held to its final tradable quote and **$1,225,000** if all accrual had been retained.",
            "",
            "The sensitivity grid also includes 5% and 7.5% moves, but the selected spike rules require at least a 10% VIX rise. This is an in-sample threshold search over a short period without the 2020 VIX shock. The selected threshold should be treated as a research candidate rather than a stable optimum.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = base.build_paths()
    entries, skips = base.select_entries(paths)
    raw_marks = base.load_marks(entries)
    valued = prepare_mark_values(entries, raw_marks)
    complete_ids = set(valued["trade_id"])
    entries = entries[entries["trade_id"].isin(complete_ids)].copy()
    summary, trade_sets = run_grid(entries, valued)
    curves, best = selected_curves(summary, trade_sets, valued, paths)
    filtered = summary[summary["spike_threshold"].ge(0.10)]
    overall = filtered.loc[filtered["final_equity"].idxmax()]

    all_trades = []
    for label, trades in trade_sets.items():
        export = trades.copy()
        export["strategy"] = label
        all_trades.append(export)
    pd.concat(all_trades, ignore_index=True).to_csv(OUT / "all_trades.csv", index=False)
    summary.sort_values(["target_delta", "require_profit", "spike_threshold"], na_position="first").to_csv(
        OUT / "summary.csv", index=False
    )
    best.to_csv(OUT / "best_by_delta.csv", index=False)
    curves.to_csv(OUT / "equity_curves.csv", index=False)
    skips.to_csv(OUT / "skips.csv", index=False)
    save_chart(curves, best, overall)
    save_heatmaps(summary)
    write_report(summary, best, overall)

    print("BEST BY DELTA")
    print(
        best[
            [
                "strategy",
                "triggered_exits",
                "winning_trades",
                "proceeds_divided_by_premium",
                "final_equity",
                "cagr",
            ]
        ].to_string(index=False)
    )
    print("\nOVERALL BEST")
    print(overall.to_string())
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
