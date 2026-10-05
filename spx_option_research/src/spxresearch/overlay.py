from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .metrics import drawdown_series, expected_shortfall, performance_metrics


def _fast_metrics(
    returns: pd.Series,
    *,
    excess_returns: pd.Series | None = None,
) -> dict[str, float]:
    standard_deviation = returns.std(ddof=1)
    excess = returns if excess_returns is None else excess_returns
    excess_deviation = excess.std(ddof=1)
    wealth = float((1.0 + returns).prod())
    years = len(returns) / 252.0
    return {
        "annualized_return": wealth ** (1.0 / years) - 1.0 if wealth > 0 and years > 0 else math.nan,
        "annualized_volatility": float(standard_deviation * math.sqrt(252.0)),
        "sharpe": float(excess.mean() / excess_deviation * math.sqrt(252.0))
        if excess_deviation > 0
        else math.nan,
        "max_drawdown": float(drawdown_series(returns).min()),
        "es_95": expected_shortfall(returns, 0.95),
        "es_99": expected_shortfall(returns, 0.99),
    }


def _period_sharpe(values: pd.Series) -> float:
    deviation = values.std(ddof=1)
    return float(values.mean() / deviation * math.sqrt(252.0)) if deviation > 0 else math.nan


def evaluate_hedge_overlays(
    core_daily: pd.DataFrame,
    core_metrics: pd.DataFrame,
    core_screen: pd.DataFrame,
    hedge_daily: pd.DataFrame,
    hedge_summary: pd.DataFrame,
    market: pd.DataFrame,
    config: dict[str, Any],
    *,
    top_cores: int = 20,
    top_hedges_per_family: int = 15,
) -> pd.DataFrame:
    """Budget-scale hedge ladders and combine them with daily core MTM."""
    capital = float(config["initial_capital"])
    index = pd.DatetimeIndex(market.index).sort_values()
    collateral = market["risk_free_daily"].reindex(index).fillna(0.0)
    split_1 = int(len(index) * config["split_fractions"][0])
    split_2 = split_1 + int(len(index) * config["split_fractions"][1])

    core_choice = core_metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=["sharpe"])
    ranking_column = "robust_score" if "robust_score" in core_choice else "sharpe"
    ranked_cores = core_choice.sort_values(ranking_column, ascending=False)
    core_parts = [ranked_cores.head(top_cores)]
    for return_target in [0.03, 0.035, 0.04, 0.045, 0.05, 0.06]:
        core_parts.append(
            ranked_cores[ranked_cores["annualized_option_return"] >= return_target].head(5)
        )
    core_choice = pd.concat(core_parts, ignore_index=True).drop_duplicates(
        ["base_strategy_id", "exit_rule"]
    )
    hedge_choice = hedge_summary.copy()
    hedge_choice["tail_screen_score"] = (
        hedge_choice["down_5_payoff_to_cost"].fillna(0.0)
        + 2.0 * hedge_choice["down_10_payoff_to_cost"].fillna(0.0)
        + 0.1 * hedge_choice["payoff_to_cost"].fillna(0.0)
    )
    ranked_hedges = hedge_choice.sort_values("tail_screen_score", ascending=False)
    hedge_choice = pd.concat(
        [
            ranked_hedges.groupby("family", group_keys=False).head(top_hedges_per_family),
            ranked_hedges.groupby(["family", "target_dte"], group_keys=False).head(2),
        ],
        ignore_index=True,
    ).drop_duplicates(["hedge_strategy_id", "exit_rule"])
    screen_lookup = (
        core_screen.sort_values("robust_score", ascending=False)
        .drop_duplicates("base_strategy_id")
        .set_index("base_strategy_id")
    )
    hedge_lookup = hedge_choice.set_index(["hedge_strategy_id", "exit_rule"])

    core_series = {
        key: group.groupby("mark_date")["daily_option_pnl"].sum().set_axis(
            pd.to_datetime(group.groupby("mark_date")["daily_option_pnl"].sum().index)
        ).reindex(index, fill_value=0.0)
        for key, group in core_daily.groupby(["base_strategy_id", "exit_rule"], sort=False)
    }
    hedge_series = {
        key: group.groupby("mark_date")["daily_hedge_pnl_one_lot"].sum().set_axis(
            pd.to_datetime(group.groupby("mark_date")["daily_hedge_pnl_one_lot"].sum().index)
        ).reindex(index, fill_value=0.0)
        for key, group in hedge_daily.groupby(["hedge_strategy_id", "exit_rule"], sort=False)
        if key in hedge_lookup.index
    }

    records: list[dict[str, Any]] = []
    percent_budgets = [float(value) for value in config["hedge_budgets"] if float(value) > 0]
    absolute_budgets = [float(value) for value in config["absolute_hedge_costs"]]
    for core in core_choice.itertuples(index=False):
        core_key = (core.base_strategy_id, core.exit_rule)
        if core_key not in core_series or core.base_strategy_id not in screen_lookup.index:
            continue
        core_pnl = core_series[core_key]
        core_option = core_pnl / capital
        gross_premium = float(screen_lookup.loc[core.base_strategy_id, "annualized_gross_premium"])
        short_dte = int(screen_lookup.loc[core.base_strategy_id, "short_dte"])

        baseline = _fast_metrics(core_option + collateral, excess_returns=core_option)
        baseline.update(
            {
                "strategy_id": f"{core.base_strategy_id}_{core.exit_rule}_unhedged",
                "base_strategy_id": core.base_strategy_id,
                "core_exit_rule": core.exit_rule,
                "hedge_strategy_id": "none",
                "hedge_exit_rule": "none",
                "hedge_family": "none",
                "overlay_type": "unhedged",
                "short_dte": short_dte,
                "short_put_delta": float(screen_lookup.loc[core.base_strategy_id, "short_put_delta"]),
                "width_method": screen_lookup.loc[core.base_strategy_id, "width_method"],
                "width_value": float(screen_lookup.loc[core.base_strategy_id, "width_value"]),
                "hedge_dte": math.nan,
                "hedge_delta_1": math.nan,
                "hedge_delta_2": math.nan,
                "hedge_width_pct": math.nan,
                "hedge_center_drawdown": math.nan,
                "budget_type": "none",
                "budget_value": 0.0,
                "hedge_scale": 0.0,
                "annual_hedge_cost": 0.0,
                "annualized_core_option_return": float(core_option.mean() * 252.0),
                "annualized_hedge_pnl": 0.0,
                "annualized_option_return": float(core_option.mean() * 252.0),
                "train_sharpe": _period_sharpe(core_option.iloc[:split_1]),
                "validation_sharpe": _period_sharpe(core_option.iloc[split_1:split_2]),
                "oos_sharpe": _period_sharpe(core_option.iloc[split_2:]),
                "sharpe_improvement_per_cost": math.nan,
                "es95_improvement_per_cost": math.nan,
            }
        )
        records.append(baseline)

        for hedge_key, hedge_pnl in hedge_series.items():
            hedge = hedge_lookup.loc[hedge_key]
            annual_one_lot_cost = float(hedge["annual_hedge_cost_one_lot"])
            if not np.isfinite(annual_one_lot_cost) or annual_one_lot_cost <= 0:
                continue
            budgets = [
                *(('premium_fraction', value, gross_premium * value * capital) for value in percent_budgets),
                *(('absolute_nav', value, value * capital) for value in absolute_budgets),
            ]
            for budget_type, budget_value, target_cost in budgets:
                scale = target_cost / annual_one_lot_cost
                option_returns = (core_pnl + hedge_pnl * scale) / capital
                total_returns = option_returns + collateral
                metrics = _fast_metrics(total_returns, excess_returns=option_returns)
                annual_cost = annual_one_lot_cost * scale / capital
                metrics.update(
                    {
                        "strategy_id": (
                            f"{core.base_strategy_id}_{core.exit_rule}__{hedge_key[0]}_"
                            f"{hedge_key[1]}__{budget_type}_{budget_value:g}"
                        ),
                        "base_strategy_id": core.base_strategy_id,
                        "core_exit_rule": core.exit_rule,
                        "hedge_strategy_id": hedge_key[0],
                        "hedge_exit_rule": hedge_key[1],
                        "hedge_family": hedge["family"],
                        "overlay_type": "calendar_diagonal"
                        if int(hedge["target_dte"]) > short_dte
                        else "same_expiration",
                        "short_dte": short_dte,
                        "short_put_delta": float(screen_lookup.loc[core.base_strategy_id, "short_put_delta"]),
                        "width_method": screen_lookup.loc[core.base_strategy_id, "width_method"],
                        "width_value": float(screen_lookup.loc[core.base_strategy_id, "width_value"]),
                        "hedge_dte": int(hedge["target_dte"]),
                        "hedge_delta_1": float(hedge["delta_1"]),
                        "hedge_delta_2": float(hedge["delta_2"]),
                        "hedge_width_pct": float(hedge["width_pct"]),
                        "hedge_center_drawdown": float(hedge["center_drawdown"]),
                        "budget_type": budget_type,
                        "budget_value": budget_value,
                        "hedge_scale": scale,
                        "annual_hedge_cost": annual_cost,
                        "annualized_core_option_return": float(core_option.mean() * 252.0),
                        "annualized_hedge_pnl": float((hedge_pnl * scale / capital).mean() * 252.0),
                        "annualized_option_return": float(option_returns.mean() * 252.0),
                        "train_sharpe": _period_sharpe(option_returns.iloc[:split_1]),
                        "validation_sharpe": _period_sharpe(option_returns.iloc[split_1:split_2]),
                        "oos_sharpe": _period_sharpe(option_returns.iloc[split_2:]),
                        "sharpe_improvement_per_cost": (metrics["sharpe"] - baseline["sharpe"]) / annual_cost
                        if annual_cost > 0
                        else math.nan,
                        "es95_improvement_per_cost": (metrics["es_95"] - baseline["es_95"]) / annual_cost
                        if annual_cost > 0
                        else math.nan,
                    }
                )
                records.append(metrics)
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["robust_score"] = (
        0.15 * result["train_sharpe"]
        + 0.25 * result["validation_sharpe"]
        + 0.50 * result["oos_sharpe"]
        + 0.10 * result["sharpe"]
        - 0.02 * result["hedge_family"].map(
            {"none": 0, "outright_put": 1, "long_put_spread": 2, "put_butterfly": 3, "bounded_2x1": 2}
        ).fillna(3)
    )
    return result


def add_pareto_flags(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["pareto_return_volatility"] = _pareto(
        frame[["annualized_return", "annualized_volatility"]].to_numpy(float),
        maximize=(True, False),
    )
    frame["pareto_return_drawdown"] = _pareto(
        np.column_stack([frame["annualized_return"], -frame["max_drawdown"]]),
        maximize=(True, False),
    )
    frame["pareto_cost_es95"] = _pareto(
        np.column_stack([frame["annual_hedge_cost"], -frame["es_95"]]),
        maximize=(False, False),
    )
    return frame


def _pareto(values: np.ndarray, maximize: tuple[bool, bool]) -> np.ndarray:
    transformed = values.copy()
    for column, should_maximize in enumerate(maximize):
        if should_maximize:
            transformed[:, column] *= -1.0
    finite = np.isfinite(transformed).all(axis=1)
    efficient = np.zeros(len(values), dtype=bool)
    for index in np.flatnonzero(finite):
        candidate = transformed[index]
        dominated = np.any(
            np.all(transformed[finite] <= candidate, axis=1)
            & np.any(transformed[finite] < candidate, axis=1)
        )
        efficient[index] = not dominated
    return efficient


def enrich_finalists(
    results: pd.DataFrame,
    return_series: dict[str, pd.Series],
    market: pd.DataFrame,
    count: int = 20,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in results.sort_values("robust_score", ascending=False).head(count).itertuples(index=False):
        returns = return_series[row.strategy_id]
        metrics = performance_metrics(
            returns,
            benchmark_returns=market["spx_return"].reindex(returns.index),
        )
        metrics.update(row._asdict())
        records.append(metrics)
    return pd.DataFrame(records)
