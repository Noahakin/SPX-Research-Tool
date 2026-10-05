from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.analysis import summarize_daily_finalists
from spxresearch.core_engine import (
    build_core_trade_outcomes,
    build_daily_mtm,
    build_dynamic_exit_outcomes,
)
from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.optimizer import (
    CoreParameter,
    build_core_entry_candidates,
    load_config,
    weekly_entry_dates,
)
from spxresearch.warehouse import ResearchWarehouse


def main() -> None:
    config = load_config(PROJECT / "config" / "base.json")
    data_root = (WORKSPACE / config["data_root"]).resolve()
    cache = (WORKSPACE / config["cache_root"]).resolve()
    results = (WORKSPACE / config["results_root"]).resolve()
    finalists = pd.read_parquet(results / "finalists.parquet")
    core_screen = pd.read_parquet(results / "core_screen.parquet")
    base_ids = finalists["base_strategy_id"].drop_duplicates().head(10)
    definitions = (
        core_screen[core_screen["base_strategy_id"].isin(base_ids)]
        .drop_duplicates("base_strategy_id")
        .set_index("base_strategy_id")
    )
    parameters = [
        CoreParameter(
            int(row.short_dte),
            int(row.short_put_delta),
            str(row.width_method),
            float(row.width_value),
        )
        for row in definitions.itertuples()
    ]
    archive = SPXSurfaceArchive(data_root)
    sessions = archive.populated_dates
    baseline_entries = weekly_entry_dates(sessions, int(config["entry_weekday"]))
    locations = sessions.get_indexer(baseline_entries)
    market = pd.read_parquet(cache / "market_data.parquet")
    market.index = pd.to_datetime(market.index)
    records: list[pd.DataFrame] = []

    with ResearchWarehouse(cache / "research.duckdb", data_root, threads=8, memory_limit="12GB") as warehouse:
        for shift in [-1, 1]:
            shifted_locations = locations + shift
            shifted_locations = shifted_locations[
                (shifted_locations >= 0) & (shifted_locations < len(sessions))
            ]
            shifted_dates = sessions[shifted_locations]
            prefix = cache / f"entry_shift_{shift:+d}"
            candidates_path = prefix.with_name(prefix.name + "_candidates.parquet")
            fixed_path = prefix.with_name(prefix.name + "_fixed.parquet")
            dynamic_path = prefix.with_name(prefix.name + "_dynamic.parquet")
            fixed_daily_path = prefix.with_name(prefix.name + "_fixed_daily.parquet")
            dynamic_daily_path = prefix.with_name(prefix.name + "_dynamic_daily.parquet")
            build_core_entry_candidates(
                archive,
                config,
                candidates_path,
                dates=shifted_dates,
                parameters=parameters,
            )
            build_core_trade_outcomes(warehouse.connection, candidates_path, fixed_path)
            build_dynamic_exit_outcomes(warehouse.connection, candidates_path, dynamic_path)
            build_daily_mtm(warehouse.connection, fixed_path, fixed_daily_path)
            build_daily_mtm(warehouse.connection, dynamic_path, dynamic_daily_path)
            daily = pd.concat(
                [pd.read_parquet(fixed_daily_path), pd.read_parquet(dynamic_daily_path)],
                ignore_index=True,
            )
            metrics = summarize_daily_finalists(
                daily, market, initial_capital=float(config["initial_capital"])
            )
            metrics.insert(0, "entry_shift_sessions", shift)
            records.append(metrics)
    result = pd.concat(records, ignore_index=True)
    result.to_parquet(results / "entry_shift_robustness.parquet", index=False)
    result.to_csv(results / "entry_shift_robustness.csv", index=False)


if __name__ == "__main__":
    main()
