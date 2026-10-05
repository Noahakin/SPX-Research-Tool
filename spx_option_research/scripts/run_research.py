from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.analysis import screen_core_outcomes, summarize_daily_finalists
from spxresearch.core_engine import (
    build_core_trade_outcomes,
    build_daily_mtm,
    build_dynamic_exit_outcomes,
)
from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.market_data import build_market_data
from spxresearch.optimizer import build_core_entry_candidates, load_config
from spxresearch.warehouse import ResearchWarehouse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPX option research pipeline")
    parser.add_argument("--config", type=Path, default=PROJECT / "config" / "base.json")
    parser.add_argument("--force", action="store_true", help="Rebuild existing checkpoints")
    parser.add_argument(
        "--force-finalists",
        action="store_true",
        help="Rebuild shortlists/dynamic exits/daily MTM while reusing the broad grid",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="12GB")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def checkpoint_exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = (WORKSPACE / config["data_root"]).resolve()
    cache = (WORKSPACE / config["cache_root"]).resolve()
    results = (WORKSPACE / config["results_root"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(results / "research.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("spxresearch")

    archive = SPXSurfaceArchive(data_root)
    candidates_path = cache / "core_candidates.parquet"
    outcomes_path = cache / "core_fixed_outcomes.parquet"
    screen_path = results / "core_screen.parquet"
    shortlist_path = cache / "core_shortlisted_candidates.parquet"
    fixed_shortlist_path = cache / "core_shortlisted_fixed_outcomes.parquet"
    dynamic_path = cache / "core_dynamic_outcomes.parquet"
    fixed_daily_path = cache / "core_fixed_daily_mtm.parquet"
    dynamic_daily_path = cache / "core_dynamic_daily_mtm.parquet"
    market_path = cache / "market_data.parquet"

    with ResearchWarehouse(
        cache / "research.duckdb",
        data_root,
        threads=args.threads,
        memory_limit=args.memory_limit,
    ) as warehouse:
        log.info("Auditing the complete option archive")
        audit = warehouse.audit()
        write_json(results / "archive_audit.json", asdict(audit))
        daily_market = warehouse.daily_market()
        daily_market.to_parquet(results / "archive_daily_coverage.parquet")
        log.info("Archive audit: %s", audit)

        if args.force or not checkpoint_exists(market_path):
            log.info("Downloading/caching FRED collateral and VIX series")
            market = build_market_data(daily_market["spx_spot"], WORKSPACE, market_path)
        else:
            market = pd.read_parquet(market_path)
            market.index = pd.to_datetime(market.index)

        if args.force or not checkpoint_exists(candidates_path):
            log.info("Selecting broad core entries")
            candidates = build_core_entry_candidates(archive, config, candidates_path)
        else:
            candidates = pd.read_parquet(candidates_path)
            log.info("Reusing %s core candidates", f"{len(candidates):,}")

        if args.force or not checkpoint_exists(outcomes_path):
            log.info("Valuing fixed-exit core outcomes")
            build_core_trade_outcomes(warehouse.connection, candidates_path, outcomes_path)

        if args.force or not checkpoint_exists(screen_path):
            log.info("Screening broad core grid")
            screen = screen_core_outcomes(
                warehouse.connection,
                outcomes_path,
                market.index,
                initial_capital=float(config["initial_capital"]),
                output_path=screen_path,
            )
        else:
            screen = pd.read_parquet(screen_path)

        eligible = screen.dropna(
            subset=["robust_score", "validation_sharpe", "oos_sharpe"]
        ).copy()
        eligible = eligible[
            (eligible["validation_sharpe"] > 0) & (eligible["oos_sharpe"] > 0)
        ]
        eligible.sort_values("robust_score", ascending=False, inplace=True)
        per_family = int(config["top_core_per_family"])
        shortlist_parts = [
            eligible.groupby("width_method", group_keys=False).head(per_family)
        ]
        # Preserve the highest-quality configurations at each explicit income
        # hurdle, even when lower-risk near-cash variants have higher Sharpe.
        for return_target in [0.03, 0.035, 0.04, 0.045, 0.05, 0.06]:
            income = eligible[
                eligible["annualized_option_return_realistic"] >= return_target
            ]
            shortlist_parts.append(
                income.groupby("width_method", group_keys=False).head(10)
            )
        # Retain delta diversity for the requested ATM and slightly-ITM
        # benchmarks and for parameter-neighborhood validation.
        shortlist_parts.append(
            eligible.groupby(["short_put_delta", "width_method"], group_keys=False).head(2)
        )
        shortlisted_screen = pd.concat(shortlist_parts, ignore_index=True).drop_duplicates(
            ["base_strategy_id", "exit_dte"]
        )
        shortlisted_screen.to_csv(results / "core_shortlist.csv", index=False)
        shortlist_ids = set(shortlisted_screen["base_strategy_id"])
        shortlisted_candidates = candidates[candidates["base_strategy_id"].isin(shortlist_ids)]
        shortlisted_candidates.to_parquet(shortlist_path, index=False)

        fixed = pd.read_parquet(outcomes_path)
        fixed = fixed[fixed["base_strategy_id"].isin(shortlist_ids)]
        fixed.to_parquet(fixed_shortlist_path, index=False)

        if args.force or args.force_finalists or not checkpoint_exists(dynamic_path):
            log.info("Evaluating dynamic exits for %s stable core families", len(shortlist_ids))
            build_dynamic_exit_outcomes(warehouse.connection, shortlist_path, dynamic_path)
        if args.force or args.force_finalists or not checkpoint_exists(fixed_daily_path):
            log.info("Building genuine daily MTM for fixed exits")
            build_daily_mtm(warehouse.connection, fixed_shortlist_path, fixed_daily_path)
        if args.force or args.force_finalists or not checkpoint_exists(dynamic_daily_path):
            log.info("Building genuine daily MTM for dynamic exits")
            build_daily_mtm(warehouse.connection, dynamic_path, dynamic_daily_path)

    fixed_daily = pd.read_parquet(fixed_daily_path)
    dynamic_daily = pd.read_parquet(dynamic_daily_path)
    daily = pd.concat([fixed_daily, dynamic_daily], ignore_index=True)
    daily.to_parquet(cache / "core_finalist_daily_mtm.parquet", index=False)
    finalist_metrics = summarize_daily_finalists(
        daily,
        market,
        initial_capital=float(config["initial_capital"]),
    ).sort_values("sharpe", ascending=False)
    finalist_metrics.to_parquet(results / "core_finalist_metrics.parquet", index=False)
    finalist_metrics.to_csv(results / "core_finalist_metrics.csv", index=False)
    write_json(
        results / "core_run_manifest.json",
        {
            "candidate_rows": len(candidates),
            "strategies_screened": len(screen),
            "shortlisted_base_strategies": len(shortlist_ids),
            "daily_mtm_rows": len(daily),
            "finalist_variants": len(finalist_metrics),
        },
    )
    log.info("Core pipeline complete; outputs are under %s", results)


if __name__ == "__main__":
    main()
