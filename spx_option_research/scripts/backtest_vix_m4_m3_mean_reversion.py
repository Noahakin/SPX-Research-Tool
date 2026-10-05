from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import analyze_vix_curve_next_day_spx as curve_source


OUT = curve_source.PROJECT / "results/vix_m4_m3_mean_reversion_strategy"
OFFICIAL_EXTENSION = OUT / "official_monthly_vx_extension.csv"
MULTIPLIER = 1_000.0
ROLLING_WINDOW = 252
MIN_HISTORY = 126
LEG_SIDE_COST = 25.0
ENTRY_THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0]
MAX_HOLDS = [10, 15, 20, 30]
DIRECTION_MODES = ["both", "long_low", "short_high"]
TRAIN_END = pd.Timestamp("2024-12-31")


@dataclass
class Position:
    direction: int
    near_expiration: pd.Timestamp
    far_expiration: pd.Timestamp
    last_near: float
    last_far: float
    entry_date: pd.Timestamp
    entry_signal_date: pd.Timestamp
    entry_index: int
    entry_z: float
    entry_spread: float
    entry_rolling_mean: float
    gross_pnl: float = 0.0
    costs: float = 0.0
    rolls: int = 0


def load_futures() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(
        curve_source.FUTURES_SOURCE, parse_dates=["date", "expiration_date"]
    )
    if OFFICIAL_EXTENSION.exists():
        extension = pd.read_csv(
            OFFICIAL_EXTENSION, parse_dates=["date", "expiration_date"]
        )
        raw = pd.concat(
            [raw, extension[["date", "expiration_date", "future"]]],
            ignore_index=True,
        )
    raw = raw[
        raw["expiration_date"].gt(raw["date"])
        & raw["future"].notna()
        & raw["future"].gt(0)
    ].copy()
    raw = raw.sort_values(["date", "expiration_date"]).drop_duplicates(
        ["date", "expiration_date"], keep="last"
    )
    raw["term"] = raw.groupby("date").cumcount() + 1
    selected = raw[raw["term"].isin([3, 4])].copy()
    price = selected.pivot(index="date", columns="term", values="future")
    expiration = selected.pivot(
        index="date", columns="term", values="expiration_date"
    )
    curve = pd.DataFrame(index=price.index)
    curve["m3"] = price[3]
    curve["m4"] = price[4]
    curve["m3_expiration"] = expiration[3]
    curve["m4_expiration"] = expiration[4]
    curve = curve.dropna().reset_index().sort_values("date").reset_index(drop=True)
    curve["spread"] = curve["m4"] - curve["m3"]
    curve["rolling_mean"] = (
        curve["spread"]
        .rolling(ROLLING_WINDOW, min_periods=MIN_HISTORY)
        .mean()
        .shift(1)
    )
    curve["rolling_std"] = (
        curve["spread"]
        .rolling(ROLLING_WINDOW, min_periods=MIN_HISTORY)
        .std()
        .shift(1)
    )
    curve["z_score"] = (
        curve["spread"] - curve["rolling_mean"]
    ) / curve["rolling_std"]
    quotes = raw[["date", "expiration_date", "future"]].copy()
    return curve, quotes


def quote_lookup(quotes: pd.DataFrame) -> dict[tuple[pd.Timestamp, pd.Timestamp], float]:
    return {
        (row.date, row.expiration_date): float(row.future)
        for row in quotes.itertuples(index=False)
    }


def backtest(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    entry_threshold: float,
    max_hold: int,
    leg_side_cost: float = LEG_SIDE_COST,
    direction_mode: str = "both",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = quote_lookup(quotes)
    position: Position | None = None
    scheduled: dict[str, object] | None = None
    trades: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    cumulative_pnl = 0.0

    def close_position(
        current: Position,
        row: pd.Series,
        reason: str,
        daily_cost: float,
    ) -> tuple[dict[str, object], float]:
        exit_cost = 2.0 * leg_side_cost
        current.costs += exit_cost
        daily_cost += exit_cost
        net_pnl = current.gross_pnl - current.costs
        trade = {
            "entry_date": current.entry_date,
            "entry_signal_date": current.entry_signal_date,
            "exit_date": row["date"],
            "direction": current.direction,
            "position": (
                "long M4 / short M3"
                if current.direction == 1
                else "short M4 / long M3"
            ),
            "entry_z": current.entry_z,
            "entry_spread": current.entry_spread,
            "entry_rolling_mean": current.entry_rolling_mean,
            "exit_dynamic_spread": row["spread"],
            "exit_z": row["z_score"],
            "holding_sessions": int(row.name - current.entry_index),
            "rolls": current.rolls,
            "gross_pnl": current.gross_pnl,
            "costs": current.costs,
            "net_pnl": net_pnl,
            "exit_reason": reason,
        }
        return trade, daily_cost

    for index, row in curve.iterrows():
        date = pd.Timestamp(row["date"])
        daily_gross = 0.0
        daily_cost = 0.0
        action_executed = ""

        if position is not None:
            near_price = lookup.get((date, position.near_expiration))
            far_price = lookup.get((date, position.far_expiration))
            if near_price is None or far_price is None:
                raise RuntimeError(
                    f"Missing held-contract quote on {date:%Y-%m-%d}: "
                    f"{position.near_expiration:%Y-%m-%d}, {position.far_expiration:%Y-%m-%d}"
                )
            daily_gross = position.direction * (
                (far_price - position.last_far) - (near_price - position.last_near)
            ) * MULTIPLIER
            position.gross_pnl += daily_gross
            position.last_near = near_price
            position.last_far = far_price

        if scheduled is not None:
            if scheduled["action"] == "close" and position is not None:
                trade, daily_cost = close_position(
                    position, row, str(scheduled["reason"]), daily_cost
                )
                trades.append(trade)
                position = None
                action_executed = f"close:{scheduled['reason']}"
            elif scheduled["action"] == "open" and position is None:
                direction = int(scheduled["direction"])
                entry_cost = 2.0 * leg_side_cost
                daily_cost += entry_cost
                position = Position(
                    direction=direction,
                    near_expiration=pd.Timestamp(row["m3_expiration"]),
                    far_expiration=pd.Timestamp(row["m4_expiration"]),
                    last_near=float(row["m3"]),
                    last_far=float(row["m4"]),
                    entry_date=date,
                    entry_signal_date=pd.Timestamp(scheduled["signal_date"]),
                    entry_index=index,
                    entry_z=float(scheduled["signal_z"]),
                    entry_spread=float(row["spread"]),
                    entry_rolling_mean=float(row["rolling_mean"]),
                    costs=entry_cost,
                )
                action_executed = "open"
            scheduled = None

        if position is not None:
            current_near = pd.Timestamp(row["m3_expiration"])
            current_far = pd.Timestamp(row["m4_expiration"])
            if (
                position.near_expiration != current_near
                or position.far_expiration != current_far
            ):
                roll_cost = 4.0 * leg_side_cost
                daily_cost += roll_cost
                position.costs += roll_cost
                position.near_expiration = current_near
                position.far_expiration = current_far
                position.last_near = float(row["m3"])
                position.last_far = float(row["m4"])
                position.rolls += 1
                action_executed = f"{action_executed};roll" if action_executed else "roll"

        if index < len(curve) - 1 and np.isfinite(row["z_score"]):
            if position is None:
                if (
                    direction_mode in {"both", "short_high"}
                    and row["z_score"] >= entry_threshold
                ):
                    scheduled = {
                        "action": "open",
                        "direction": -1,
                        "signal_date": date,
                        "signal_z": row["z_score"],
                    }
                elif (
                    direction_mode in {"both", "long_low"}
                    and row["z_score"] <= -entry_threshold
                ):
                    scheduled = {
                        "action": "open",
                        "direction": 1,
                        "signal_date": date,
                        "signal_z": row["z_score"],
                    }
            else:
                holding_sessions = index - position.entry_index
                reverted = (
                    position.direction == -1 and row["z_score"] <= 0
                ) or (position.direction == 1 and row["z_score"] >= 0)
                if reverted:
                    scheduled = {"action": "close", "reason": "rolling_mean"}
                elif holding_sessions >= max_hold:
                    scheduled = {"action": "close", "reason": "max_hold"}

        if index == len(curve) - 1 and position is not None:
            trade, daily_cost = close_position(
                position, row, "end_of_sample", daily_cost
            )
            trades.append(trade)
            position = None
            action_executed = "close:end_of_sample"

        daily_net = daily_gross - daily_cost
        cumulative_pnl += daily_net
        daily_rows.append(
            {
                "date": date,
                "spread": row["spread"],
                "rolling_mean": row["rolling_mean"],
                "rolling_std": row["rolling_std"],
                "z_score": row["z_score"],
                "position": position.direction if position is not None else 0,
                "daily_gross_pnl": daily_gross,
                "daily_cost": daily_cost,
                "daily_net_pnl": daily_net,
                "cumulative_net_pnl": cumulative_pnl,
                "action": action_executed,
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(daily_rows)


def metrics(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    entry_threshold: float,
    max_hold: int,
    leg_side_cost: float,
    direction_mode: str,
) -> dict[str, object]:
    active = daily[daily["z_score"].notna()].copy()
    years = (active["date"].max() - active["date"].min()).days / 365.25
    daily_std = active["daily_net_pnl"].std()
    annualized_pnl = active["daily_net_pnl"].mean() * 252
    annualized_vol = daily_std * np.sqrt(252)
    curve = active["cumulative_net_pnl"]
    drawdown = curve - curve.cummax()
    train_trades = trades[trades["entry_date"].le(TRAIN_END)]
    test_trades = trades[trades["entry_date"].gt(TRAIN_END)]
    return {
        "direction_mode": direction_mode,
        "entry_threshold": entry_threshold,
        "max_hold": max_hold,
        "leg_side_cost": leg_side_cost,
        "two_leg_round_trip_cost": leg_side_cost * 4.0,
        "trades": len(trades),
        "long_low_spread_trades": int(trades["direction"].eq(1).sum()),
        "short_high_spread_trades": int(trades["direction"].eq(-1).sum()),
        "win_rate": trades["net_pnl"].gt(0).mean(),
        "average_holding_sessions": trades["holding_sessions"].mean(),
        "average_rolls_per_trade": trades["rolls"].mean(),
        "gross_pnl": trades["gross_pnl"].sum(),
        "total_costs": trades["costs"].sum(),
        "net_pnl": trades["net_pnl"].sum(),
        "average_net_pnl_per_trade": trades["net_pnl"].mean(),
        "median_net_pnl_per_trade": trades["net_pnl"].median(),
        "annualized_pnl_one_spread": annualized_pnl,
        "annualized_vol_one_spread": annualized_vol,
        "pnl_sharpe": annualized_pnl / annualized_vol if annualized_vol > 0 else np.nan,
        "maximum_drawdown": drawdown.min(),
        "train_2022_2024_net_pnl": train_trades["net_pnl"].sum(),
        "train_2022_2024_trades": len(train_trades),
        "test_2025_2026_net_pnl": test_trades["net_pnl"].sum(),
        "test_2025_2026_trades": len(test_trades),
        "test_2025_2026_win_rate": test_trades["net_pnl"].gt(0).mean(),
        "net_pnl_long_low_spread": trades.loc[
            trades["direction"].eq(1), "net_pnl"
        ].sum(),
        "net_pnl_short_high_spread": trades.loc[
            trades["direction"].eq(-1), "net_pnl"
        ].sum(),
        "sample_years": years,
    }


def run_grid(
    curve: pd.DataFrame, quotes: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, float, int], tuple[pd.DataFrame, pd.DataFrame]],
]:
    rows: list[dict[str, object]] = []
    results: dict[tuple[str, float, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for direction_mode in DIRECTION_MODES:
        for entry_threshold in ENTRY_THRESHOLDS:
            for max_hold in MAX_HOLDS:
                trades, daily = backtest(
                    curve,
                    quotes,
                    entry_threshold,
                    max_hold,
                    LEG_SIDE_COST,
                    direction_mode,
                )
                rows.append(
                    metrics(
                        trades,
                        daily,
                        entry_threshold,
                        max_hold,
                        LEG_SIDE_COST,
                        direction_mode,
                    )
                )
                results[(direction_mode, entry_threshold, max_hold)] = (trades, daily)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["train_2022_2024_net_pnl", "pnl_sharpe"], ascending=False
    ).reset_index(drop=True), results


def cost_sensitivity(
    curve: pd.DataFrame,
    quotes: pd.DataFrame,
    entry_threshold: float,
    max_hold: int,
    direction_mode: str,
) -> pd.DataFrame:
    rows = []
    for leg_side_cost in [0.0, 12.5, 25.0, 50.0]:
        trades, daily = backtest(
            curve,
            quotes,
            entry_threshold,
            max_hold,
            leg_side_cost,
            direction_mode,
        )
        rows.append(
            metrics(
                trades,
                daily,
                entry_threshold,
                max_hold,
                leg_side_cost,
                direction_mode,
            )
        )
    return pd.DataFrame(rows)


def save_chart(
    selected: pd.Series, trades: pd.DataFrame, daily: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.72, 1.0]},
    )
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22)

    axes[0].plot(daily["date"], daily["spread"], color="#0f172a", linewidth=1.1, label="M4-M3")
    axes[0].plot(daily["date"], daily["rolling_mean"], color="#f59e0b", linewidth=1.8, label="Causal rolling mean")
    high = daily["rolling_mean"] + selected["entry_threshold"] * daily["rolling_std"]
    low = daily["rolling_mean"] - selected["entry_threshold"] * daily["rolling_std"]
    axes[0].plot(daily["date"], high, color="#dc2626", linewidth=1.0, linestyle="--", alpha=0.8)
    axes[0].plot(daily["date"], low, color="#0f766e", linewidth=1.0, linestyle="--", alpha=0.8)
    axes[0].set_ylabel("Spread (VIX points)")
    axes[0].legend(frameon=False, ncol=2, loc="lower right")

    axes[1].plot(daily["date"], daily["z_score"], color="#7c3aed", linewidth=1.0)
    axes[1].axhline(selected["entry_threshold"], color="#dc2626", linestyle="--")
    axes[1].axhline(-selected["entry_threshold"], color="#0f766e", linestyle="--")
    axes[1].axhline(0, color="#334155", linewidth=0.8)
    axes[1].set_ylabel("Rolling z-score")

    axes[2].plot(
        daily["date"], daily["cumulative_net_pnl"], color="#2563eb", linewidth=2.0
    )
    axes[2].fill_between(
        daily["date"], 0, daily["cumulative_net_pnl"], color="#2563eb", alpha=0.12
    )
    axes[2].axhline(0, color="#334155", linewidth=0.8)
    axes[2].set_ylabel("Cumulative net P&L")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[0].set_title(
        "Tradeable M4-M3 VIX futures mean-reversion strategy",
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    axes[0].text(
        0,
        1.01,
        f"{selected['direction_mode'].replace('_', ' ')} · entry {selected['entry_threshold']:.2f}z · exit at rolling mean · max {int(selected['max_hold'])} sessions · one VX spread",
        transform=axes[0].transAxes,
        fontsize=11.5,
        color="#475569",
    )
    axes[2].text(
        0.995,
        0.015,
        f"Includes ${selected['two_leg_round_trip_cost']:.0f} base round-trip cost plus roll costs",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout()
    fig.savefig(OUT / "strategy_equity_curve.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(
    selected: pd.Series,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    largest_trade = trades.loc[trades["net_pnl"].idxmax()]
    largest_share = largest_trade["net_pnl"] / selected["net_pnl"]
    spreads_for_3pct = int(np.ceil(30_000 / selected["annualized_pnl_one_spread"]))
    spreads_for_5pct = int(np.ceil(50_000 / selected["annualized_pnl_one_spread"]))
    lines = [
        "# Tradeable M4-M3 VIX futures mean reversion",
        "",
        "A high spread is traded by buying M3 and selling M4. A low spread is traded by selling M3 and buying M4. Signals use a 252-session rolling mean and standard deviation, shifted one day, with a 126-session minimum history. Orders execute at the following session's settlement. The strategy holds one two-leg VX spread, rolls the fixed contracts when necessary, and exits at the rolling mean or the maximum holding period.",
        "",
        f"The configuration selected using 2022-2024 training P&L is **{selected['direction_mode'].replace('_', ' ')} at {selected['entry_threshold']:.2f} standard deviations with a {int(selected['max_hold'])}-session maximum hold**.",
        "",
        "| Metric | Result |",
        "|:---|---:|",
        f"| Trades | {selected['trades']:.0f} |",
        f"| Win rate | {selected['win_rate']:.1%} |",
        f"| Net P&L, one spread | ${selected['net_pnl']:,.0f} |",
        f"| Annualized P&L | ${selected['annualized_pnl_one_spread']:,.0f} |",
        f"| P&L Sharpe | {selected['pnl_sharpe']:.2f} |",
        f"| Maximum drawdown | ${selected['maximum_drawdown']:,.0f} |",
        f"| 2022-2024 training P&L | ${selected['train_2022_2024_net_pnl']:,.0f} |",
        f"| 2025-2026 test P&L | ${selected['test_2025_2026_net_pnl']:,.0f} |",
        f"| Long-low-spread P&L | ${selected['net_pnl_long_low_spread']:,.0f} |",
        f"| Short-high-spread P&L | ${selected['net_pnl_short_high_spread']:,.0f} |",
        "",
        "## Current implementation",
        "",
        "Enter only when M4-M3 is at or below the prior 252-session rolling mean minus 1.25 rolling standard deviations. Buy M4 and sell M3 in equal contract counts at the following session's settlement. Exit at the following settlement after the z-score reaches zero, or after 10 sessions. Do not trade the high-spread signal.",
        "",
    ]
    latest_signal_file = OUT / "latest_signal.csv"
    if latest_signal_file.exists():
        latest = pd.read_csv(latest_signal_file).iloc[0]
        lines.extend(
            [
                f"As of **{latest['as_of']}**, the spread was **{latest['m4_minus_m3']:.4f}** versus a lower entry level of **{latest['lower_entry_level']:.4f}** (z = {latest['z_score']:.2f}). The current instruction is **{latest['signal']}**.",
                "",
            ]
        )
    lines.extend(
        [
        "## Scaling and concentration",
        "",
        f"At the backtested annual P&L of ${selected['annualized_pnl_one_spread']:,.0f} per standard VX spread, a $1 million account would need about {spreads_for_3pct} spreads for 3% or {spreads_for_5pct} spreads for 5%. That scale would have produced historical drawdowns of roughly ${abs(selected['maximum_drawdown']) * spreads_for_3pct:,.0f} and ${abs(selected['maximum_drawdown']) * spreads_for_5pct:,.0f}, before any margin stress beyond the sample.",
        "",
        f"The largest trade earned ${largest_trade['net_pnl']:,.0f} on {pd.Timestamp(largest_trade['entry_date']):%Y-%m-%d}, equal to {largest_share:.1%} of total net P&L. This concentration and the sample of only {int(selected['trades'])} trades make full target sizing premature.",
        "",
        "## Transaction-cost sensitivity",
        "",
        "| Two-leg round trip | Net P&L | Sharpe | Maximum drawdown |",
        "|---:|---:|---:|---:|",
        ]
    )
    for _, row in sensitivity.iterrows():
        lines.append(
            f"| ${row['two_leg_round_trip_cost']:.0f} | ${row['net_pnl']:,.0f} | "
            f"{row['pnl_sharpe']:.2f} | ${row['maximum_drawdown']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "Futures prices are settlement-style marks from the local matched VX history, extended through the latest available official Cboe daily settlements. Margin, intraday slippage, and variation-margin funding are not modeled. One spread means one contract in each leg; each 1.00-point spread move is $1,000 of gross P&L.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curve, quotes = load_futures()
    summary, results = run_grid(curve, quotes)
    selected = summary.iloc[0]
    key = (
        str(selected["direction_mode"]),
        float(selected["entry_threshold"]),
        int(selected["max_hold"]),
    )
    trades, daily = results[key]
    sensitivity = cost_sensitivity(
        curve, quotes, key[1], key[2], key[0]
    )

    summary.to_csv(OUT / "parameter_grid.csv", index=False)
    trades.to_csv(OUT / "selected_trades.csv", index=False)
    daily.to_csv(OUT / "selected_daily_pnl.csv", index=False)
    sensitivity.to_csv(OUT / "cost_sensitivity.csv", index=False)
    pd.DataFrame([selected]).to_csv(OUT / "selected_summary.csv", index=False)
    yearly = trades.assign(year=pd.to_datetime(trades["entry_date"]).dt.year).groupby(
        "year", as_index=False
    ).agg(
        trades=("net_pnl", "size"),
        gross_pnl=("gross_pnl", "sum"),
        costs=("costs", "sum"),
        net_pnl=("net_pnl", "sum"),
    )
    yearly.to_csv(OUT / "yearly_performance.csv", index=False)
    save_chart(selected, trades, daily)
    write_report(selected, summary, sensitivity, trades)

    print("SELECTED")
    print(selected.to_string())
    print("\nGRID")
    print(
        summary[
            [
                "direction_mode",
                "entry_threshold",
                "max_hold",
                "trades",
                "win_rate",
                "net_pnl",
                "pnl_sharpe",
                "maximum_drawdown",
                "train_2022_2024_net_pnl",
                "test_2025_2026_net_pnl",
            ]
        ].to_string(index=False)
    )
    print("\nCOST SENSITIVITY")
    print(
        sensitivity[
            [
                "two_leg_round_trip_cost",
                "net_pnl",
                "pnl_sharpe",
                "maximum_drawdown",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
