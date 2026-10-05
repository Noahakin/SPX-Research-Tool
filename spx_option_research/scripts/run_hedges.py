from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.analysis import summarize_daily_finalists, walk_forward_core_selection
from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.hedges import (
    build_hedge_daily_mtm,
    build_hedge_entry_candidates,
    build_hedge_trade_outcomes,
    summarize_hedge_outcomes,
)
from spxresearch.metrics import drawdown_series, performance_metrics
from spxresearch.optimizer import load_config
from spxresearch.overlay import add_pareto_flags, evaluate_hedge_overlays
from spxresearch.reporting import generate_report
from spxresearch.robustness import (
    bootstrap_confidence_intervals,
    deflated_sharpe_ratio,
    leave_one_period_out,
    probabilistic_sharpe_ratio,
    rolling_walk_forward,
    stationary_block_bootstrap,
)
from spxresearch.visualization import (
    plot_core_heatmaps,
    plot_frontier,
    plot_metric_heatmap,
    plot_strategy_diagnostics,
)
from spxresearch.warehouse import ResearchWarehouse


STRESS_PERIODS = {
    "Volmageddon": ("2018-02-01", "2018-02-28"),
    "2018 Q4": ("2018-10-01", "2018-12-31"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2020 recovery": ("2020-03-24", "2020-12-31"),
    "2022 rate shock": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026 YTD": ("2026-01-01", "2026-09-22"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPX downside-hedge overlays")
    parser.add_argument("--config", type=Path, default=PROJECT / "config" / "base.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="12GB")
    return parser.parse_args()


def checkpoint_exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _series_maps(frame: pd.DataFrame, value: str) -> dict[tuple[str, str], pd.Series]:
    id_column = "base_strategy_id" if "base_strategy_id" in frame else "hedge_strategy_id"
    rule_column = "exit_rule"
    result: dict[tuple[str, str], pd.Series] = {}
    for key, group in frame.groupby([id_column, rule_column], sort=False):
        series = group.groupby("mark_date")[value].sum()
        series.index = pd.to_datetime(series.index)
        result[key] = series
    return result


def _strategy_returns(
    row: pd.Series,
    core_map: dict[tuple[str, str], pd.Series],
    hedge_map: dict[tuple[str, str], pd.Series],
    market: pd.DataFrame,
    capital: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    index = pd.DatetimeIndex(market.index)
    core = core_map[(row["base_strategy_id"], row["core_exit_rule"])].reindex(index, fill_value=0.0)
    hedge = pd.Series(0.0, index=index)
    if row["hedge_strategy_id"] != "none":
        hedge = hedge_map[(row["hedge_strategy_id"], row["hedge_exit_rule"])].reindex(index, fill_value=0.0)
        hedge = hedge * float(row["hedge_scale"])
    option = (core + hedge) / capital
    total = option + market["risk_free_daily"].reindex(index).fillna(0.0)
    return total, core / capital, hedge / capital


def _stress_table(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for name, (start, end) in STRESS_PERIODS.items():
        strategy = returns.loc[start:end]
        spx = benchmark.reindex(strategy.index).fillna(0.0)
        if strategy.empty:
            continue
        records.append(
            {
                "period": name,
                "start": start,
                "end": end,
                "strategy_return": float((1.0 + strategy).prod() - 1.0),
                "spx_return": float((1.0 + spx).prod() - 1.0),
                "strategy_max_drawdown": float(drawdown_series(strategy).min()),
                "worst_day": float(strategy.min()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = (WORKSPACE / config["data_root"]).resolve()
    cache = (WORKSPACE / config["cache_root"]).resolve()
    results = (WORKSPACE / config["results_root"]).resolve()
    charts = results / "charts"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(results / "research.log", encoding="utf-8"), logging.StreamHandler()],
    )
    log = logging.getLogger("spxresearch.hedges")
    candidates_path = cache / "hedge_candidates.parquet"
    outcomes_path = cache / "hedge_outcomes.parquet"
    selected_path = cache / "hedge_selected_outcomes.parquet"
    daily_path = cache / "hedge_selected_daily_mtm.parquet"

    archive = SPXSurfaceArchive(data_root)
    with ResearchWarehouse(
        cache / "research.duckdb", data_root, threads=args.threads, memory_limit=args.memory_limit
    ) as warehouse:
        if args.force or not checkpoint_exists(candidates_path):
            log.info("Selecting broad long-hedge universe")
            hedge_candidates = build_hedge_entry_candidates(archive, config, candidates_path)
        else:
            hedge_candidates = pd.read_parquet(candidates_path)
        if args.force or not checkpoint_exists(outcomes_path):
            log.info("Valuing hedge exits for %s candidates", f"{len(hedge_candidates):,}")
            build_hedge_trade_outcomes(warehouse.connection, candidates_path, outcomes_path)
        hedge_outcomes = pd.read_parquet(outcomes_path)
        hedge_summary = summarize_hedge_outcomes(hedge_outcomes)
        hedge_summary["tail_screen_score"] = (
            hedge_summary["down_5_payoff_to_cost"].fillna(0.0)
            + 2.0 * hedge_summary["down_10_payoff_to_cost"].fillna(0.0)
            + 0.1 * hedge_summary["payoff_to_cost"].fillna(0.0)
        )
        hedge_summary.to_parquet(results / "hedge_screen.parquet", index=False)
        hedge_summary.sort_values("tail_screen_score", ascending=False).to_csv(
            results / "hedge_screen.csv", index=False
        )
        ranked_hedges = hedge_summary.sort_values("tail_screen_score", ascending=False)
        selected_keys = pd.concat(
            [
                ranked_hedges.groupby("family", group_keys=False).head(15),
                ranked_hedges.groupby(["family", "target_dte"], group_keys=False).head(2),
            ],
            ignore_index=True,
        )[["hedge_strategy_id", "exit_rule"]].drop_duplicates()
        selected = hedge_outcomes.merge(selected_keys, on=["hedge_strategy_id", "exit_rule"])
        selected.to_parquet(selected_path, index=False)
        if args.force or not checkpoint_exists(daily_path):
            log.info("Building daily MTM for %s shortlisted hedge programs", len(selected_keys))
            build_hedge_daily_mtm(warehouse.connection, selected_path, daily_path)
        walk_forward_core_selection(
            warehouse.connection,
            cache / "core_fixed_outcomes.parquet",
            archive.populated_dates,
            initial_capital=float(config["initial_capital"]),
        ).to_csv(results / "walk_forward_core_selection.csv", index=False)

    market = pd.read_parquet(cache / "market_data.parquet")
    market.index = pd.to_datetime(market.index)
    core_daily = pd.read_parquet(cache / "core_finalist_daily_mtm.parquet")
    core_screen = pd.read_parquet(results / "core_screen.parquet")
    core_metrics = summarize_daily_finalists(
        core_daily, market, initial_capital=float(config["initial_capital"])
    )
    core_metrics.to_parquet(results / "core_finalist_metrics.parquet", index=False)
    core_metrics.to_csv(results / "core_finalist_metrics.csv", index=False)
    hedge_daily = pd.read_parquet(daily_path)
    overlays = evaluate_hedge_overlays(
        core_daily,
        core_metrics,
        core_screen,
        hedge_daily,
        hedge_summary[hedge_summary.set_index(["hedge_strategy_id", "exit_rule"]).index.isin(
            pd.MultiIndex.from_frame(selected_keys)
        )],
        market,
        config,
    )
    overlays = add_pareto_flags(overlays)
    overlays.to_parquet(results / "overlay_master.parquet", index=False)
    overlays.sort_values("robust_score", ascending=False).to_csv(results / "overlay_master.csv", index=False)

    return_targets: list[pd.DataFrame] = []
    for target in [0.03, 0.035, 0.04, 0.045, 0.05, 0.06]:
        subset = overlays[overlays["annualized_option_return"] >= target].copy()
        if subset.empty:
            continue
        chosen = pd.concat(
            [
                subset.nlargest(1, "robust_score"),
                subset.nsmallest(1, "annualized_volatility"),
                subset.nlargest(1, "max_drawdown"),
                subset.nlargest(1, "es_95"),
                subset.nlargest(1, "es_99"),
            ]
        ).drop_duplicates("strategy_id")
        chosen.insert(0, "net_option_return_target", target)
        return_targets.append(chosen)
    constrained = pd.concat(return_targets, ignore_index=True) if return_targets else pd.DataFrame()
    constrained.to_csv(results / "return_target_frontiers.csv", index=False)

    core_map = _series_maps(core_daily, "daily_option_pnl")
    core_one_lot_map = _series_maps(core_daily, "daily_option_pnl_one_lot")
    hedge_map = _series_maps(hedge_daily, "daily_hedge_pnl_one_lot")
    finalists = overlays.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["robust_score", "oos_sharpe"]
    ).sort_values("robust_score", ascending=False).head(int(config["finalist_count"]))
    detailed: list[dict[str, object]] = []
    returns_by_id: dict[str, pd.Series] = {}
    options_by_id: dict[str, pd.Series] = {}
    core_decomposition = (
        core_screen.sort_values("robust_score", ascending=False)
        .drop_duplicates("base_strategy_id")
        .set_index("base_strategy_id")
    )
    hedge_decomposition = hedge_summary.set_index(["hedge_strategy_id", "exit_rule"])
    for _, row in finalists.iterrows():
        total, core_option, hedge_option = _strategy_returns(
            row, core_map, hedge_map, market, float(config["initial_capital"])
        )
        metrics = performance_metrics(
            total,
            option_returns=core_option + hedge_option,
            collateral_returns=market["risk_free_daily"],
            benchmark_returns=market["spx_return"],
        )
        record = row.to_dict()
        record.update(metrics)
        record["annualized_core_option_return"] = float(core_option.mean() * 252.0)
        record["annualized_hedge_pnl"] = float(hedge_option.mean() * 252.0)
        core_detail = core_decomposition.loc[row["base_strategy_id"]]
        record["annualized_core_gross_premium"] = float(
            core_detail["annualized_gross_premium"]
        )
        record["annualized_core_losses_and_closing_cost"] = (
            record["annualized_core_gross_premium"]
            - record["annualized_core_option_return"]
        )
        record["annualized_core_execution_drag"] = float(
            core_detail["annualized_realistic_cost"]
        )
        record["annualized_hedge_payoff"] = (
            float(row["annual_hedge_cost"]) + record["annualized_hedge_pnl"]
        )
        record["annualized_hedge_execution_drag"] = 0.0
        if row["hedge_strategy_id"] != "none":
            hedge_detail = hedge_decomposition.loc[
                (row["hedge_strategy_id"], row["hedge_exit_rule"])
            ]
            record["annualized_hedge_execution_drag"] = (
                float(hedge_detail["annual_pnl_ideal_one_lot"])
                - float(hedge_detail["annual_pnl_one_lot"])
            ) * float(row["hedge_scale"]) / float(config["initial_capital"])
        record["annualized_total_execution_drag"] = (
            record["annualized_core_execution_drag"]
            + record["annualized_hedge_execution_drag"]
        )
        detailed.append(record)
        returns_by_id[str(row["strategy_id"])] = total
        options_by_id[str(row["strategy_id"])] = core_option + hedge_option
    finalist_table = pd.DataFrame(detailed).sort_values("robust_score", ascending=False)
    finalist_table.to_parquet(results / "finalists.parquet", index=False)
    finalist_table.to_csv(results / "finalists.csv", index=False)

    sizing_records: list[dict[str, object]] = []
    collateral = market["risk_free_daily"].fillna(0.0)
    gross_lookup = (
        core_screen.sort_values("robust_score", ascending=False)
        .drop_duplicates("base_strategy_id")
        .set_index("base_strategy_id")["annualized_gross_premium"]
    )
    for _, row in finalists.iterrows():
        strategy_id = str(row["strategy_id"])
        option = options_by_id[strategy_id]
        variants: list[tuple[str, float, pd.Series]] = [("max_loss_normalized", 1.0, option)]
        option_volatility = float(option.std(ddof=1) * math.sqrt(252.0))
        volatility_scale = (
            float(config["volatility_target"]) / option_volatility
            if option_volatility > 0
            else math.nan
        )
        if np.isfinite(volatility_scale):
            variants.append(("volatility_normalized", volatility_scale, option * volatility_scale))
        gross = float(gross_lookup.loc[row["base_strategy_id"]])
        if gross > 0:
            premium_scale = 0.04 / gross
            variants.append(("four_percent_premium_target", premium_scale, option * premium_scale))
        core_one = core_one_lot_map[(row["base_strategy_id"], row["core_exit_rule"])].reindex(
            market.index, fill_value=0.0
        )
        hedge_one = pd.Series(0.0, index=market.index)
        if row["hedge_strategy_id"] != "none":
            hedge_one = hedge_map[(row["hedge_strategy_id"], row["hedge_exit_rule"])].reindex(
                market.index, fill_value=0.0
            )
        variants.append(
            ("one_contract_each_week", 1.0, (core_one + hedge_one) / float(config["initial_capital"]))
        )
        for sizing_method, scale, sized_option in variants:
            metrics = performance_metrics(
                sized_option + collateral,
                option_returns=sized_option,
                collateral_returns=collateral,
                benchmark_returns=market["spx_return"],
            )
            metrics.update(
                {
                    "strategy_id": strategy_id,
                    "sizing_method": sizing_method,
                    "sizing_scale": scale,
                }
            )
            sizing_records.append(metrics)
    pd.DataFrame(sizing_records).to_csv(results / "position_sizing_comparison.csv", index=False)

    transaction_records: list[dict[str, object]] = []
    for _, row in finalist_table.iterrows():
        baseline_returns = returns_by_id[str(row["strategy_id"])]
        realistic_drag = float(row["annualized_total_execution_drag"])
        for cost_multiplier in [0.0, 1.0, 1.25, 1.5, 2.0]:
            stressed = baseline_returns - (cost_multiplier - 1.0) * realistic_drag / 252.0
            metrics = performance_metrics(
                stressed,
                collateral_returns=market["risk_free_daily"].reindex(stressed.index),
            )
            transaction_records.append(
                {
                    "strategy_id": row["strategy_id"],
                    "cost_multiplier_vs_realistic": cost_multiplier,
                    "annualized_return": metrics.get("annualized_return"),
                    "sharpe": metrics.get("sharpe"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "es_95": metrics.get("es_95"),
                }
            )
    pd.DataFrame(transaction_records).to_csv(
        results / "transaction_cost_stress.csv", index=False
    )

    robustness_records: list[dict[str, object]] = []
    trial_std = float(core_screen["oos_sharpe"].replace([np.inf, -np.inf], np.nan).std())
    number_trials = len(core_screen) + len(overlays)
    for strategy_id, returns in list(options_by_id.items())[:5]:
        psr = probabilistic_sharpe_ratio(returns)
        dsr, expected_maximum = deflated_sharpe_ratio(
            returns, number_of_trials=number_trials, trial_sharpe_std=trial_std
        )
        bootstrap = stationary_block_bootstrap(
            returns,
            samples=int(config["bootstrap_samples"]),
            block_length=int(config["block_length"]),
            seed=int(config["seed"]),
        )
        bootstrap.to_parquet(results / f"bootstrap_{strategy_id}.parquet", index=False)
        record: dict[str, object] = {
            "strategy_id": strategy_id,
            "probabilistic_sharpe_ratio": psr,
            "deflated_sharpe_ratio": dsr,
            "expected_max_sharpe_under_trials": expected_maximum,
            "number_of_trials": number_trials,
        }
        record.update(bootstrap_confidence_intervals(bootstrap))
        robustness_records.append(record)
        rolling_walk_forward(returns).to_csv(results / f"walk_forward_{strategy_id}.csv", index=False)
        annual_periods = {
            f"year_{year}": (f"{year}-01-01", f"{year}-12-31")
            for year in sorted(pd.Index(returns.index.year).unique())
        }
        leave_one_period_out(returns, {**STRESS_PERIODS, **annual_periods}).to_csv(
            results / f"leave_one_period_out_{strategy_id}.csv", index=False
        )
    robustness = pd.DataFrame(robustness_records)
    robustness.to_csv(results / "robustness_summary.csv", index=False)

    winner = finalist_table.iloc[0]
    winning_returns = returns_by_id[str(winner["strategy_id"])]
    stress = _stress_table(winning_returns, market["spx_return"])
    stress.to_csv(results / "stress_periods.csv", index=False)
    regime = pd.DataFrame(
        {
            "strategy_return": winning_returns,
            "spx_return": market["spx_return"],
            "vix_quartile": pd.qcut(market["vix"], 4, duplicates="drop"),
            "trend": np.where(market["spx_trend_200d"] >= 0, "above_200d", "below_200d"),
        }
    )
    regime.groupby(["vix_quartile", "trend"], observed=True).agg(
        observations=("strategy_return", "count"),
        strategy_return=("strategy_return", "mean"),
        spx_return=("spx_return", "mean"),
    ).reset_index().to_csv(results / "regime_analysis.csv", index=False)

    plot_strategy_diagnostics(winning_returns, market["spx_return"], charts / "winner", label=str(winner["strategy_id"]))
    plot_core_heatmaps(core_screen, charts / "core")
    plot_frontier(overlays, "annualized_return", "annualized_volatility", charts / "frontier_return_volatility.png", title="Return / volatility frontier")
    plot_frontier(overlays, "annual_hedge_cost", "es_95", charts / "frontier_cost_es95.png", title="Hedge cost / 95% expected shortfall")
    plot_frontier(overlays, "annualized_return", "max_drawdown", charts / "frontier_return_drawdown.png", title="Return / maximum drawdown")
    hedged_only = overlays[overlays["hedge_family"] != "none"]
    plot_metric_heatmap(hedged_only, "short_put_delta", "hedge_delta_1", "oos_sharpe", charts / "heatmap_short_delta_hedge_delta.png", title="Short-put delta / hedge delta")
    plot_metric_heatmap(hedged_only, "short_dte", "hedge_dte", "oos_sharpe", charts / "heatmap_short_dte_hedge_dte.png", title="Short DTE / hedge DTE")
    plot_metric_heatmap(hedged_only, "annual_hedge_cost", "hedge_delta_1", "es_95", charts / "heatmap_hedge_cost_delta.png", title="Hedge cost / hedge strike proxy (delta)")

    master = pd.concat(
        [
            core_screen.assign(record_type="core_screen"),
            hedge_summary.assign(record_type="hedge_screen"),
            overlays.assign(record_type="core_plus_hedge"),
        ],
        ignore_index=True,
        sort=False,
    )
    master.to_parquet(results / "master_strategy_results.parquet", index=False)
    master.to_csv(results / "master_strategy_results.csv", index=False)
    audit = json.loads((results / "archive_audit.json").read_text(encoding="utf-8"))
    generate_report(
        results / "research_report.md",
        audit=audit,
        core_screen=core_screen,
        overlay_results=overlays,
        finalists=finalist_table,
        robustness=robustness,
        stress=stress,
    )
    log.info("Hedge, robustness, chart, and report pipeline complete")


if __name__ == "__main__":
    main()
