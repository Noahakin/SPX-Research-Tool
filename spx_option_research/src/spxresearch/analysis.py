from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .metrics import chronological_splits, performance_metrics


def screen_core_outcomes(
    connection: duckdb.DuckDBPyConnection,
    outcomes_path: str | Path,
    market_dates: pd.DatetimeIndex,
    *,
    initial_capital: float,
    output_path: str | Path,
) -> pd.DataFrame:
    """Fast broad screen using realized trade P&L before finalist daily MTM."""
    outcomes = Path(outcomes_path).resolve().as_posix().replace("'", "''")
    dates = pd.DatetimeIndex(market_dates).sort_values()
    train_end = dates[int(len(dates) * 0.60) - 1].date()
    validation_end = dates[int(len(dates) * 0.80) - 1].date()
    total_days = len(dates)
    train_days = int((dates <= pd.Timestamp(train_end)).sum())
    validation_days = int(
        ((dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(validation_end))).sum()
    )
    test_days = total_days - train_days - validation_days
    years = total_days / 252.0

    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW core_outcomes AS SELECT * FROM read_parquet('{outcomes}')"
    )
    result = connection.execute(
        f"""
        WITH daily AS (
            SELECT
                base_strategy_id,
                exit_dte,
                exit_date,
                min(target_dte) AS target_dte,
                min(short_delta_target) AS short_delta,
                min(width_method) AS width_method,
                min(width_value) AS width_value,
                sum(pnl_per_spread_ideal * contracts_ideal) / {initial_capital} AS r_ideal,
                sum(pnl_per_spread_realistic * contracts_realistic) / {initial_capital} AS r_realistic,
                sum(pnl_per_spread_conservative * contracts_conservative) / {initial_capital} AS r_conservative,
                sum(entry_credit_realistic * 100.0 * contracts_realistic) / {initial_capital} AS gross_premium
            FROM core_outcomes
            GROUP BY base_strategy_id, exit_dte, exit_date
        ),
        aggregate AS (
            SELECT
                base_strategy_id,
                exit_dte,
                min(target_dte) AS short_dte,
                min(short_delta) AS short_put_delta,
                min(width_method) AS width_method,
                min(width_value) AS width_value,
                count(*) AS active_exit_days,
                sum(r_ideal) / {years} AS annualized_option_return_ideal,
                sum(r_realistic) / {years} AS annualized_option_return_realistic,
                sum(r_conservative) / {years} AS annualized_option_return_conservative,
                sum(gross_premium) / {years} AS annualized_gross_premium,
                sum(r_realistic) AS sum_r,
                sum(r_realistic * r_realistic) AS sum_r2,
                sum(CASE WHEN exit_date <= DATE '{train_end}' THEN r_realistic ELSE 0 END) AS train_sum,
                sum(CASE WHEN exit_date <= DATE '{train_end}' THEN r_realistic * r_realistic ELSE 0 END) AS train_sum2,
                sum(CASE WHEN exit_date > DATE '{train_end}' AND exit_date <= DATE '{validation_end}' THEN r_realistic ELSE 0 END) AS validation_sum,
                sum(CASE WHEN exit_date > DATE '{train_end}' AND exit_date <= DATE '{validation_end}' THEN r_realistic * r_realistic ELSE 0 END) AS validation_sum2,
                sum(CASE WHEN exit_date > DATE '{validation_end}' THEN r_realistic ELSE 0 END) AS test_sum,
                sum(CASE WHEN exit_date > DATE '{validation_end}' THEN r_realistic * r_realistic ELSE 0 END) AS test_sum2,
                sum(r_ideal - r_realistic) / {years} AS annualized_realistic_cost,
                sum(r_realistic - r_conservative) / {years} AS annualized_conservative_incremental_cost
            FROM daily
            GROUP BY base_strategy_id, exit_dte
        )
        SELECT
            *,
            base_strategy_id || '_exit' || CAST(exit_dte AS VARCHAR) AS strategy_id,
            (sum_r / {total_days}) * 252.0 AS screen_annualized_return,
            sqrt(greatest((sum_r2 - sum_r * sum_r / {total_days}) / ({total_days} - 1), 0)) * sqrt(252.0) AS screen_volatility,
            CASE WHEN sum_r2 - sum_r * sum_r / {total_days} > 0
              THEN (sum_r / {total_days}) /
                   sqrt((sum_r2 - sum_r * sum_r / {total_days}) / ({total_days} - 1)) * sqrt(252.0)
            END AS screen_sharpe,
            CASE WHEN train_sum2 - train_sum * train_sum / {train_days} > 0
              THEN (train_sum / {train_days}) /
                   sqrt((train_sum2 - train_sum * train_sum / {train_days}) / ({train_days} - 1)) * sqrt(252.0)
            END AS train_sharpe,
            CASE WHEN validation_sum2 - validation_sum * validation_sum / {validation_days} > 0
              THEN (validation_sum / {validation_days}) /
                   sqrt((validation_sum2 - validation_sum * validation_sum / {validation_days}) / ({validation_days} - 1)) * sqrt(252.0)
            END AS validation_sharpe,
            CASE WHEN test_sum2 - test_sum * test_sum / {test_days} > 0
              THEN (test_sum / {test_days}) /
                   sqrt((test_sum2 - test_sum * test_sum / {test_days}) / ({test_days} - 1)) * sqrt(252.0)
            END AS oos_sharpe
        FROM aggregate
        """
    ).fetchdf()

    result = add_plateau_scores(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    result.sort_values("robust_score", ascending=False).to_csv(output.with_suffix(".csv"), index=False)
    return result


def add_plateau_scores(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["plateau_sharpe"] = np.nan
    for _, group in frame.groupby(["short_dte", "exit_dte", "width_method"], sort=False):
        short = group["short_put_delta"].to_numpy(float)
        width = group["width_value"].to_numpy(float)
        sharpe = group["oos_sharpe"].to_numpy(float)
        values = np.full(len(group), np.nan)
        for index in range(len(group)):
            width_tolerance = max(1.0, abs(width[index]) * 0.26)
            neighbor = (np.abs(short - short[index]) <= 5.0) & (
                np.abs(width - width[index]) <= width_tolerance
            )
            neighbor_values = sharpe[neighbor]
            neighbor_values = neighbor_values[np.isfinite(neighbor_values)]
            if len(neighbor_values) >= 2:
                values[index] = np.median(neighbor_values)
        frame.loc[group.index, "plateau_sharpe"] = values
    frame["stability_gap"] = frame["oos_sharpe"] - frame["plateau_sharpe"]
    collapse_penalty = (
        (frame["validation_sharpe"] < 0).astype(float)
        + (frame["oos_sharpe"] < 0).astype(float)
        + frame["stability_gap"].clip(lower=0).fillna(1.0)
    )
    frame["robust_score"] = (
        0.15 * frame["train_sharpe"]
        + 0.25 * frame["validation_sharpe"]
        + 0.45 * frame["oos_sharpe"]
        + 0.15 * frame["plateau_sharpe"]
        - 0.25 * collapse_penalty
    )
    return frame


def summarize_daily_finalists(
    daily_pnl: pd.DataFrame,
    market: pd.DataFrame,
    *,
    initial_capital: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    market = market.sort_index()
    for (base_id, exit_rule), group in daily_pnl.groupby(
        ["base_strategy_id", "exit_rule"], sort=False
    ):
        pnl = group.groupby("mark_date")["daily_option_pnl"].sum()
        pnl.index = pd.to_datetime(pnl.index)
        option_returns = pnl.reindex(market.index, fill_value=0.0) / initial_capital
        collateral = market["risk_free_daily"].fillna(0.0)
        total = option_returns + collateral
        metrics = performance_metrics(
            total,
            option_returns=option_returns,
            collateral_returns=collateral,
            benchmark_returns=market["spx_return"],
        )
        splits = chronological_splits(total.index, (0.60, 0.20, 0.20))
        for split_name, split_index in splits.items():
            split_metrics = performance_metrics(
                total.reindex(split_index),
                collateral_returns=collateral.reindex(split_index),
            )
            metrics[f"{split_name}_sharpe"] = split_metrics.get("sharpe", np.nan)
            metrics[f"{split_name}_annualized_return"] = split_metrics.get(
                "annualized_return", np.nan
            )
        metrics.update(
            {
                "base_strategy_id": base_id,
                "exit_rule": exit_rule,
                "strategy_id": f"{base_id}_{exit_rule}",
            }
        )
        records.append(metrics)
    result = pd.DataFrame(records)
    if not result.empty:
        result["robust_score"] = (
            0.15 * result["train_sharpe"]
            + 0.25 * result["validation_sharpe"]
            + 0.50 * result["test_sharpe"]
            + 0.10 * result["sharpe"]
        )
        result.rename(columns={"test_sharpe": "oos_sharpe"}, inplace=True)
    return result


def walk_forward_core_selection(
    connection: duckdb.DuckDBPyConnection,
    outcomes_path: str | Path,
    market_dates: pd.DatetimeIndex,
    *,
    initial_capital: float,
    first_test_year: int = 2018,
) -> pd.DataFrame:
    """Select on prior years only, then report the next calendar year's result.

    This broad stage books trade P&L on exit dates for speed. Strategies chosen
    here are subsequently assessed with genuine daily MTM.
    """
    path = Path(outcomes_path).resolve().as_posix().replace("'", "''")
    dates = pd.DatetimeIndex(market_dates).sort_values()
    years = [year for year in sorted(dates.year.unique()) if year >= first_test_year]
    counts = []
    for year in years:
        train_days = int((dates < pd.Timestamp(f"{year}-01-01")).sum())
        test_days = int((dates.year == year).sum())
        if train_days >= 252 and test_days:
            counts.append((year, train_days, test_days))
    if not counts:
        return pd.DataFrame()
    values = ",".join(f"({year},{train},{test})" for year, train, test in counts)
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW wf_outcomes AS SELECT * FROM read_parquet('{path}')"
    )
    return connection.execute(
        f"""
        WITH test_years(test_year, train_days, test_days) AS (VALUES {values}),
        daily AS (
            SELECT
                base_strategy_id, exit_dte, exit_date,
                min(target_dte) AS short_dte,
                min(short_delta_target) AS short_put_delta,
                min(width_method) AS width_method,
                min(width_value) AS width_value,
                sum(pnl_per_spread_realistic * contracts_realistic) / {float(initial_capital)} AS r
            FROM wf_outcomes
            GROUP BY base_strategy_id, exit_dte, exit_date
        ),
        moments AS (
            SELECT
                d.base_strategy_id, d.exit_dte,
                min(d.short_dte) AS short_dte,
                min(d.short_put_delta) AS short_put_delta,
                min(d.width_method) AS width_method,
                min(d.width_value) AS width_value,
                y.test_year, y.train_days, y.test_days,
                sum(CASE WHEN d.exit_date < make_date(y.test_year, 1, 1) THEN r ELSE 0 END) AS train_sum,
                sum(CASE WHEN d.exit_date < make_date(y.test_year, 1, 1) THEN r*r ELSE 0 END) AS train_sum2,
                sum(CASE WHEN year(d.exit_date) = y.test_year THEN r ELSE 0 END) AS test_sum,
                sum(CASE WHEN year(d.exit_date) = y.test_year THEN r*r ELSE 0 END) AS test_sum2
            FROM daily d CROSS JOIN test_years y
            GROUP BY d.base_strategy_id, d.exit_dte, y.test_year, y.train_days, y.test_days
        ),
        scored AS (
            SELECT *,
                CASE WHEN train_sum2 - train_sum*train_sum/train_days > 0
                  THEN (train_sum/train_days)
                    / sqrt((train_sum2 - train_sum*train_sum/train_days)/(train_days-1))
                    * sqrt(252.0) END AS train_sharpe,
                CASE WHEN test_sum2 - test_sum*test_sum/test_days > 0
                  THEN (test_sum/test_days)
                    / sqrt((test_sum2 - test_sum*test_sum/test_days)/(test_days-1))
                    * sqrt(252.0) END AS test_sharpe,
                test_sum * 252.0 / test_days AS test_annualized_option_return
            FROM moments
        )
        SELECT *
        FROM scored
        WHERE train_sharpe IS NOT NULL
        QUALIFY row_number() OVER (
            PARTITION BY test_year, width_method ORDER BY train_sharpe DESC
        ) <= 5
        ORDER BY test_year, width_method, train_sharpe DESC
        """
    ).fetchdf()
