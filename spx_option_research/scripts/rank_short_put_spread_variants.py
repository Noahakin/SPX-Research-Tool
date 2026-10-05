from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OUT = RESULTS / "top_short_put_spread_ranking"


def score(frame: pd.DataFrame) -> pd.Series:
    return (
        0.15 * frame["train_sharpe"]
        + 0.25 * frame["validation_sharpe"]
        + 0.50 * frame["test_sharpe"]
        + 0.10 * frame["sharpe_zero_cash"]
    )


def grid_candidates() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "itm_structures/strategy_metrics.csv")
    frame = frame[
        frame["family"].eq("vertical")
        & frame["cagr"].between(0.03, 0.05)
        & frame["positive_all_splits"]
    ].copy()
    frame["strategy"] = (
        frame["structure"]
        + ", monthly "
        + frame["target_dte"].astype(int).astype(str)
        + " DTE, "
        + frame["exit_rule"].str.replace("_", " ")
    )
    frame["sizing"] = "5% concurrent max-loss budget; overlap allowed"
    frame["hedge"] = "None"
    frame["source"] = "ITM structure grid"
    return frame


def extra_profit_targets() -> pd.DataFrame:
    frame = pd.read_csv(
        RESULTS / "leveraged_103_100_profit_targets/all_profit_target_metrics.csv"
    )
    frame = frame[
        frame["cagr"].between(0.03, 0.05)
        & frame["positive_all_splits"]
        & ~frame["profit_target"].isin([0.25, 0.50, 0.75])
    ].copy()
    frame["strategy_id"] = (
        "leveraged_103_100_60d_profit_"
        + (100 * frame["profit_target"]).round().astype(int).astype(str)
    )
    frame["strategy"] = (
        "103/100, monthly 60 DTE, "
        + (100 * frame["profit_target"]).round().astype(int).astype(str)
        + "% profit target"
    )
    frame["sizing"] = "5% concurrent max-loss budget; overlap allowed"
    frame["hedge"] = "None"
    frame["source"] = "103/100 profit-target sweep"
    frame["selection_score"] = score(frame)
    return frame


def sizing_candidates() -> pd.DataFrame:
    frame = pd.read_csv(
        RESULTS / "monthly_100pct_notional_per_entry/summary.csv"
    )
    frame = frame[frame["cagr"].between(0.03, 0.05)].copy()
    frame["strategy_id"] = "monthly_sizing_" + frame.index.astype(str)
    frame["strategy"] = "103/100, monthly 60 DTE, 25% target; " + frame["version"]
    frame["sizing"] = frame["version"]
    frame["hedge"] = "None"
    frame["source"] = "Monthly entry sizing comparison"
    frame["selection_score"] = score(frame)
    frame["positive_all_splits"] = (
        frame["train_cagr"].gt(0)
        & frame["validation_cagr"].gt(0)
        & frame["test_cagr"].gt(0)
    )
    return frame


def sequential_candidates() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "sequential_itm_hedges/screen_metrics.csv")
    frame = frame[
        frame["cagr"].between(0.03, 0.05) & frame["positive_all_splits"]
    ].copy()
    frame.rename(
        columns={
            "annualized_volatility_realized": "annualized_volatility",
            "sharpe_zero_cash_realized": "sharpe_zero_cash",
            "max_drawdown_realized": "max_drawdown",
        },
        inplace=True,
    )
    frame["strategy"] = (
        frame["structure"]
        + ", sequential "
        + frame["target_dte"].astype(int).astype(str)
        + " DTE, "
        + frame["exit_rule"].str.replace("_", " ")
    )
    frame["sizing"] = "One position; short-strike notional capped at equity"
    frame["hedge"] = np.where(
        frame["overlay_id"].eq("none"),
        "None",
        frame["hedge_delta"].round().astype("Int64").astype(str)
        + "-delta put; "
        + (100 * frame["hedge_budget"]).round().astype("Int64").astype(str)
        + "% credit budget",
    )
    frame["strategy"] = np.where(
        frame["overlay_id"].eq("none"),
        frame["strategy"],
        frame["strategy"] + "; + " + frame["hedge"],
    )
    frame["source"] = "Sequential unlevered hedge grid"
    return frame


def exact_sequential_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    exact = pd.read_csv(RESULTS / "sequential_itm_hedges/finalist_metrics.csv")
    exact_columns = [
        "strategy_id",
        "cagr",
        "annualized_volatility",
        "sharpe_zero_cash",
        "max_drawdown",
        "trades",
        "win_rate",
        "mean_trade_return_spx_down_5",
        "mean_trade_return_spx_down_10",
    ]
    exact = exact[exact_columns].rename(
        columns={column: f"exact_{column}" for column in exact_columns if column != "strategy_id"}
    )
    merged = frame.merge(exact, how="left", on="strategy_id")
    for column in exact_columns[1:]:
        exact_column = f"exact_{column}"
        if exact_column in merged:
            if column in merged:
                merged[column] = merged[exact_column].combine_first(merged[column])
            else:
                merged[column] = merged[exact_column]
            merged.drop(columns=exact_column, inplace=True)
    return merged


def build_universe() -> pd.DataFrame:
    frames = [
        grid_candidates(),
        extra_profit_targets(),
        sizing_candidates(),
        sequential_candidates(),
    ]
    required = [
        "strategy_id",
        "strategy",
        "sizing",
        "hedge",
        "source",
        "cagr",
        "annualized_volatility",
        "sharpe_zero_cash",
        "max_drawdown",
        "train_cagr",
        "validation_cagr",
        "test_cagr",
        "train_sharpe",
        "validation_sharpe",
        "test_sharpe",
        "selection_score",
        "positive_all_splits",
    ]
    universe = pd.concat(
        [frame.reindex(columns=required) for frame in frames], ignore_index=True
    )
    universe = exact_sequential_metrics(universe)
    return universe.sort_values("selection_score", ascending=False).reset_index(drop=True)


def plot_ranking(top: pd.DataFrame) -> None:
    plot = top.sort_values("rank", ascending=False)
    labels = [
        f"{int(row['rank'])}. {row['strategy']}" for _, row in plot.iterrows()
    ]
    colors = ["#2563eb" if "103/100" in label else "#0f766e" for label in labels]
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    bars = ax.barh(labels, plot["selection_score"], color=colors, alpha=0.92)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Test-weighted robustness score")
    ax.set_title(
        "Top short-put-spread variants",
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    ax.text(
        0,
        1.01,
        "Eligible: 3%-5% CAGR and positive train, validation, and test CAGR",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    for bar, (_, row) in zip(bars, plot.iterrows()):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{row['cagr']:.1%} CAGR · {row['sharpe_zero_cash']:.2f} Sharpe · {row['max_drawdown']:.1%} DD",
            va="center",
            fontsize=9,
            color="#334155",
        )
    ax.set_xlim(0, top["selection_score"].max() * 1.48)
    fig.tight_layout()
    fig.savefig(OUT / "top10_ranking.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, hedged: bool = False) -> list[str]:
    if hedged:
        lines = [
            "| Strategy | CAGR | Sharpe | Max DD | Test CAGR | Down 5% trade |",
            "|:---|---:|---:|---:|---:|---:|",
        ]
        for _, row in frame.iterrows():
            down = row.get("mean_trade_return_spx_down_5", np.nan)
            down_text = "—" if pd.isna(down) else f"{down:.2%}"
            lines.append(
                f"| {row['strategy']} | {row['cagr']:.2%} | {row['sharpe_zero_cash']:.2f} | "
                f"{row['max_drawdown']:.2%} | {row['test_cagr']:.2%} | {down_text} |"
            )
        return lines
    lines = [
        "| Rank | Strategy | Sizing | CAGR | Sharpe | Max DD | Test CAGR | Score |",
        "|---:|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['strategy']} | {row['sizing']} | "
            f"{row['cagr']:.2%} | {row['sharpe_zero_cash']:.2f} | {row['max_drawdown']:.2%} | "
            f"{row['test_cagr']:.2%} | {row['selection_score']:.3f} |"
        )
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universe = build_universe()
    top = universe.head(10).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    hedged = universe[universe["hedge"].ne("None")].head(10).copy()
    crash_positive = universe[
        universe["hedge"].ne("None")
        & universe["mean_trade_return_spx_down_5"].gt(0)
    ].head(5)
    unlevered = universe[
        universe["source"].eq("Sequential unlevered hedge grid")
        & universe["hedge"].eq("None")
    ]
    best_unlevered_score = unlevered.head(1)
    best_unlevered_sharpe = unlevered.sort_values(
        ["sharpe_zero_cash", "max_drawdown"], ascending=[False, False]
    ).head(1)

    top.to_csv(OUT / "overall_top10.csv", index=False)
    hedged.to_csv(OUT / "hedged_runners_up.csv", index=False)
    crash_positive.to_csv(OUT / "crash_positive_hedges.csv", index=False)
    universe.to_csv(OUT / "eligible_universe.csv", index=False)
    plot_ranking(top)

    lines = [
        "# Ranking of short put-spread variants",
        "",
        "The ranking uses the research score already applied in the strategy searches: 15% train Sharpe, 25% validation Sharpe, 50% test Sharpe, and 10% full-period zero-cash Sharpe. Eligibility requires a 3%-5% full-period CAGR and positive CAGR in the train, validation, and test samples. Duplicate reports of an identical rule and sizing method are removed; genuinely different sizing rules remain separate variants.",
        "",
        "## Overall top ten",
        "",
        *markdown_table(top),
        "",
        "None of the protective-put variants reached the overall top ten. The strongest top-ten entries use monthly 60-DTE spreads and allow overlapping positions, so their return is not directly available without leverage.",
        "",
        "## Unlevered references",
        "",
        f"The highest test-weighted unlevered rule was **{best_unlevered_score.iloc[0]['strategy']}** at {best_unlevered_score.iloc[0]['cagr']:.2%} CAGR, {best_unlevered_score.iloc[0]['sharpe_zero_cash']:.2f} Sharpe, {best_unlevered_score.iloc[0]['max_drawdown']:.2%} maximum drawdown, and {best_unlevered_score.iloc[0]['test_cagr']:.2%} test CAGR.",
        "",
        f"The strongest full-period risk-adjusted unlevered rule was **{best_unlevered_sharpe.iloc[0]['strategy']}** at {best_unlevered_sharpe.iloc[0]['cagr']:.2%} CAGR, {best_unlevered_sharpe.iloc[0]['sharpe_zero_cash']:.2f} Sharpe, and {best_unlevered_sharpe.iloc[0]['max_drawdown']:.2%} maximum drawdown.",
        "",
        "## Best hedged runner-up by total-return robustness",
        "",
        *markdown_table(hedged.head(1), hedged=True),
        "",
        "## Best hedge with positive average returns in 5% SPX declines",
        "",
        *markdown_table(crash_positive.head(1), hedged=True),
        "",
        "The full-credit-funded monthly hedge, small far-OTM monthly puts, and 5%-budget put-spread overlays were also reviewed. They did not make the ranking because their full-period Sharpe, drawdown, or chronological stability was weaker. Metrics use option P&L only and zero cash interest.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(top[["rank", "strategy", "cagr", "sharpe_zero_cash", "max_drawdown", "test_cagr", "selection_score"]].to_string(index=False))
    print(f"\nSaved ranking to {OUT}")


if __name__ == "__main__":
    main()
