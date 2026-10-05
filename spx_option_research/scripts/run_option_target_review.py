from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.core_engine import build_daily_mtm, build_dynamic_exit_outcomes
from spxresearch.warehouse import ResearchWarehouse


def qpath(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.astype(float)
    wealth = (1.0 + returns).cumprod()
    years = len(returns) / 252.0
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    drawdown = wealth.to_numpy() / peak - 1.0
    volatility = returns.std(ddof=1) * np.sqrt(252.0)
    return {
        "cagr": wealth.iloc[-1] ** (1.0 / years) - 1.0,
        "arithmetic_annual_return": returns.mean() * 252.0,
        "annualized_volatility": volatility,
        "sharpe_zero_cash": returns.mean() / returns.std(ddof=1) * np.sqrt(252.0),
        "max_drawdown": drawdown.min(),
        "ending_value_per_100": wealth.iloc[-1] * 100.0,
    }


def main() -> None:
    cache = PROJECT / "data/cache"
    results = PROJECT / "results"
    screen = pd.read_parquet(results / "core_screen.parquet")
    qualified = screen[
        screen["annualized_option_return_realistic"].between(0.03, 0.05)
        & screen["validation_sharpe"].gt(0)
        & screen["oos_sharpe"].gt(0)
    ].sort_values("robust_score", ascending=False)

    # Daily-mark a broad but tractable set: leaders within each spread family,
    # plus leaders near each half-point return hurdle.
    pieces = [qualified.groupby("width_method", group_keys=False).head(15)]
    for low in np.arange(0.03, 0.05, 0.005):
        band = qualified[qualified["annualized_option_return_realistic"].between(low, low + 0.005)]
        pieces.append(band.groupby("width_method", group_keys=False).head(5))
    selected = pd.concat(pieces).drop_duplicates(["base_strategy_id", "exit_dte"])
    selected[["base_strategy_id", "exit_dte"]].to_csv(
        results / "option_target_selected.csv", index=False
    )

    fixed_subset = cache / "option_target_fixed_outcomes.parquet"
    candidate_subset = cache / "option_target_candidates.parquet"
    fixed_daily = cache / "option_target_fixed_daily.parquet"
    dynamic_outcomes = cache / "option_target_dynamic_outcomes.parquet"
    dynamic_daily = cache / "option_target_dynamic_daily.parquet"

    con = duckdb.connect()
    con.register("chosen", selected[["base_strategy_id", "exit_dte"]])
    con.execute(
        f"COPY (SELECT o.* FROM read_parquet('{qpath(cache / 'core_fixed_outcomes.parquet')}') o "
        "JOIN chosen c USING (base_strategy_id, exit_dte)) "
        f"TO '{qpath(fixed_subset)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    base_ids = selected[["base_strategy_id"]].drop_duplicates()
    con.unregister("chosen")
    con.register("chosen_base", base_ids)
    con.execute(
        f"COPY (SELECT c.* FROM read_parquet('{qpath(cache / 'core_candidates.parquet')}') c "
        "JOIN chosen_base b USING (base_strategy_id)) "
        f"TO '{qpath(candidate_subset)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()

    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    with ResearchWarehouse(cache / "research.duckdb", data_root, threads=8, memory_limit="12GB") as wh:
        build_daily_mtm(wh.connection, fixed_subset, fixed_daily)
        build_dynamic_exit_outcomes(wh.connection, candidate_subset, dynamic_outcomes)
        build_daily_mtm(wh.connection, dynamic_outcomes, dynamic_daily)

    market = pd.read_parquet(cache / "market_data.parquet").sort_index()
    daily = pd.concat([pd.read_parquet(fixed_daily), pd.read_parquet(dynamic_daily)])
    records: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    for (base_id, exit_rule), group in daily.groupby(["base_strategy_id", "exit_rule"]):
        pnl = group.groupby("mark_date")["daily_option_pnl"].sum()
        pnl.index = pd.to_datetime(pnl.index)
        returns = pnl.reindex(market.index, fill_value=0.0) / 1_000_000.0
        row: dict[str, object] = {
            "base_strategy_id": base_id,
            "exit_rule": exit_rule,
            "strategy_id": f"{base_id}_{exit_rule}",
        }
        row.update(metrics(returns))
        split1, split2 = int(len(returns) * 0.60), int(len(returns) * 0.80)
        for name, values in (
            ("train", returns.iloc[:split1]),
            ("validation", returns.iloc[split1:split2]),
            ("test", returns.iloc[split2:]),
        ):
            split_metrics = metrics(values)
            row[f"{name}_cagr"] = split_metrics["cagr"]
            row[f"{name}_sharpe_zero_cash"] = split_metrics["sharpe_zero_cash"]
        records.append(row)
        curves[row["strategy_id"]] = (1.0 + returns).cumprod() * 100.0

    summary = pd.DataFrame(records)
    summary["positive_all_splits"] = (
        summary["train_cagr"].gt(0)
        & summary["validation_cagr"].gt(0)
        & summary["test_cagr"].gt(0)
    )
    summary["selection_score"] = (
        0.15 * summary["train_sharpe_zero_cash"]
        + 0.25 * summary["validation_sharpe_zero_cash"]
        + 0.50 * summary["test_sharpe_zero_cash"]
        + 0.10 * summary["sharpe_zero_cash"]
    )
    summary.sort_values("selection_score", ascending=False, inplace=True)
    summary.to_csv(results / "option_target_daily_metrics.csv", index=False)
    pd.DataFrame(curves, index=market.index).to_csv(results / "option_target_equity_curves.csv")
    qualifying_daily = summary[
        summary["cagr"].between(0.03, 0.05) & summary["positive_all_splits"]
    ]
    print(f"Daily variants tested: {len(summary)}")
    print(f"Daily variants meeting target: {len(qualifying_daily)}")
    print(qualifying_daily.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
