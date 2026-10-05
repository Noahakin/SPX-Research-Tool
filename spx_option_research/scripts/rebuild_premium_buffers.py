"""Replace rejected Sharpe-ranked buffers; retain the original long-put choices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from optimize_organized_spx_hedges import OUTPUT, BASE_OUTPUT, SOURCE, prepare_cycles, simulate_candidates
from premium_buffer_hedges import buffer_catalog, simulate_buffers, select_buffers


def reconstruct(output=OUTPUT):
    config = json.loads((output/"run.json").read_text(encoding="utf-8"))
    source = Path(config["source_quote_dir"])
    for name, expected in config["source_artifact_sha256"].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected, name
    dates, by_tenor = prepare_cycles(source)
    metrics = pd.read_csv(output/"strategy_metrics.csv", dtype={"folder": str}, low_memory=False)
    equity = np.empty((len(dates), len(metrics)))
    base = metrics[~metrics.category.eq("Both")]
    with np.load(BASE_OUTPUT/"curves.npz") as archive:
        np.testing.assert_array_equal(dates.to_numpy(), archive["dates"])
        np.testing.assert_array_equal(base.strategy_id, archive["strategy_ids"])
        equity[:, base.index] = archive["equity"]
    for tenor, cycles in by_tenor.items():
        for buffer in (False, True):
            subset = metrics[metrics.category.eq("Both") & metrics.tenor_days.eq(tenor)
                             & metrics.hedge_type.eq("Downside buffer" if buffer else "Long put")]
            if buffer:
                verified, nav = simulate_buffers(subset.reset_index(drop=True), cycles, dates, tenor,
                    minimum_quantity=config["minimum_buffer_quantity"])
            else:
                verified, nav = simulate_candidates(subset.reset_index(drop=True), cycles, dates, tenor)
            for key in ("cagr", "daily_sharpe", "annualized_volatility", "max_drawdown", "ending_equity"):
                np.testing.assert_allclose(verified[key], subset[key], rtol=1e-12, atol=1e-9)
            equity[:, subset.index] = nav
    return dates.to_numpy(dtype="datetime64[ns]"), metrics.strategy_id.to_numpy(str), equity


def preview():
    dates, by_tenor = prepare_cycles()
    spec = buffer_catalog().query("short_variation == '98-95' and premium_fraction == .20")
    for tenor in (3, 7):
        ledger = []
        metrics, _ = simulate_buffers(spec, by_tenor[tenor], dates, tenor, minimum_quantity=1, ledger_writer=ledger.append)
        trades = pd.concat(ledger, ignore_index=True)
        print(metrics[["tenor", "traded_cycles", "cash_cycles", "hedge_target_min", "hedge_target_max", "mean_premium_fraction_spent", "width_3_cycles", "width_4_cycles", "width_5_cycles"]].to_string(index=False), flush=True)
        print(trades.loc[trades.traded, ["entry_date", "net_short_credit_points", "hedge_primary_pct", "hedge_secondary_pct", "hedge_width_pct", "hedge_debit_points", "hedge_quantity_ratio"]].tail(5).to_string(index=False), flush=True)


def build():
    started = time.monotonic()
    archive_path = OUTPUT/"Superseded Sharpe buffer metadata.zip"
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ("run.json", "strategy_metrics.csv", "combination_metrics.csv", "validation.json",
                         "Rebuildable curves.json", "Rebuildable ledger.json", "Rebuildable candidates.json"):
                path = OUTPUT/name
                if path.exists():
                    archive.write(path, name)
    previous = pd.read_csv(OUTPUT/"strategy_metrics.csv", dtype={"folder": str})
    config = json.loads((OUTPUT/"run.json").read_text(encoding="utf-8"))
    dates, by_tenor = prepare_cycles()
    specifications = buffer_catalog()
    frames = []
    chunks = []
    writer = None
    ledger_rows = 0
    def flush():
        nonlocal writer, ledger_rows
        if not chunks:
            return
        frame = pd.concat(chunks, ignore_index=True)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT/"buffer_trade_ledger.parquet", table.schema, compression="zstd", compression_level=9)
        writer.write_table(table)
        ledger_rows += len(frame)
        chunks.clear()
    def ledger(frame):
        chunks.append(frame)
        if len(chunks) >= 80:
            flush()
    try:
        for tenor, cycles in by_tenor.items():
            metrics, nav = simulate_buffers(specifications, cycles, dates, tenor, minimum_quantity=1, ledger_writer=ledger)
            assert np.isfinite(nav).all() and (nav > 0).all()
            frames.append(metrics)
            print(f"Rebuilt {metrics.tenor.iloc[0]}: {len(metrics)} buffer strategies; {time.monotonic()-started:.0f}s", flush=True)
        flush()
    finally:
        if writer is not None:
            writer.close()
    buffers = pd.concat(frames, ignore_index=True)
    retained = previous[~(previous.category.eq("Both") & previous.hedge_type.eq("Downside buffer"))]
    metrics = pd.concat([retained, buffers], ignore_index=True)
    assert len(metrics) == 8820 and not metrics.strategy_id.duplicated().any()
    metrics.to_csv(OUTPUT/"strategy_metrics.csv", index=False)
    metrics[metrics.category.eq("Both")].to_csv(OUTPUT/"combination_metrics.csv", index=False)
    config.update(dynamic_buffer_selection=True, minimum_buffer_quantity=1,
        hedge_notional_cap_applies_to="Long puts only", buffer_hedge_notional_cap=None,
        buffer_selection_is_retrospective=False, buffer_widths_pct_points=[5, 4, 3],
        minimum_actual_buffer_width_pct=3, buffer_long_target_grid_pct=list(range(83, 101)),
        buffer_selection_rule="At each entry, highest affordable long-put target at/below 100%; at that target prefer width 5, then 4, then 3; actual listed width at least 3% of spot",
        buffer_sizing="At least one buffer per short contract; q = budget fraction x net short credit / all-in buffer debit; full budget spent; no upper quantity cap",
        buffer_missing_rule="Entire combined strategy stays in cash when base is ineligible or no valid buffer is affordable; no narrow or partial-notional fallback",
        selection_rule="Buffers: entry-date affordability and protection. Long puts: retained full-sample combined Sharpe selections.",
        selection_is_retrospective="Long puts only; buffers use entry quotes only",
        hedge_rule="Buffers: dynamic closest affordable 3-5 point spreads. Long puts: retained fixed historical selections.",
        combination_sizing="Buffers use full allocated net credit with q >= 1; original long-put positions retain q = min(1, fraction x credit / debit)",
        premium_budget_is_limit="Long puts only; buffers spend the exact allocation on every traded roll",
        hedge_widths_pct_points=[5, 4, 3],
        curve_storage="Reconstructed in memory from retained quotes and current selection rules; metrics reconciled on every load",
        bulk_artifact_rebuild_command="python -B spx_option_research/scripts/rebuild_premium_buffers.py --build")
    config["categories"]["Both"] = "Premium-funded short puts: historical long-put choices and closest affordable 3-5 point buffers"
    for key in ("automated_tests_passed", "candidate_count"):
        config.pop(key, None)
    (OUTPUT/"run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = dict(status="buffer accounting checks passed; independent audit pending", buffer_strategies=len(buffers),
        buffer_ledger_rows=ledger_rows, traded_buffer_cycles=int(buffers.traded_cycles.sum()),
        cash_buffer_cycles=int(buffers.cash_cycles.sum()),
        no_affordable_buffer_cycles=int(buffers.no_affordable_buffer_cycles.sum()),
        full_budget_and_entry_cost_checks="All traded buffer cycles passed", actual_minimum_width=3,
        retained_long_put_strategies=int(retained.category.eq("Both").sum()),
        preserved_baseline_strategies=int((~retained.category.eq("Both")).sum()))
    (OUTPUT/"validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    build() if args.build else preview()
