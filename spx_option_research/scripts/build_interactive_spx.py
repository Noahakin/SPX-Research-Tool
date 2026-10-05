"""Build a portable, offline SPX comparison site with exact daily NAV arrays."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote

import numpy as np
import pandas as pd

from organized_spx_engine import INITIAL
from organized_spx_data import load_cash, CASH_PATH
from optimize_organized_spx_hedges import OUTPUT as RESEARCH, BASE_OUTPUT, SOURCE, prepare_cycles, simulate_candidates
from premium_buffer_hedges import simulate_buffers

PROJECT = Path(__file__).resolve().parents[1]
SITE = PROJECT.parent / "SPX Research Interactive"
ASSETS = PROJECT / "web"


def overlay_spx(option_nav, cycles, entries, cash_values, initial=INITIAL):
    """Add 100% SPX to fixed option notionals and compound at each option roll.

    Shared roll dates in saved NAV already contain the NEXT trade's costs.
    Restore the previous trade's settlement return from the ledger before
    adding SPX and then applying the next trade's costs to combined equity.
    Underlying exposure continues during option cash cycles.
    """
    if len(cycles) != len(entries):
        raise ValueError("Every roll needs its pre-entry equity and settlement return")
    result = np.empty_like(option_nav)
    equity = np.full(option_nav.shape[1], initial)
    for cycle, (option_before, terminal_return) in zip(cycles, entries):
        ix = cycle["ix"]
        cumulative = option_nav[ix] / option_before - 1
        cumulative[-1] = terminal_return
        spx = cash_values[ix] / cash_values[ix[0]] - 1
        result[ix] = equity * (1 + cumulative + spx[:, None])
        equity = result[ix[-1]].copy()
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("Overlay NAV must remain finite and positive")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def catalog_row(row, chunk, slot):
    high, width = int(row["primary_pct"]), int(row["width_pct"])
    variation = f"{high}/{high-width}" if width else str(high)
    category = row["category"]
    name = ("Sell " if category != "Put buying" else "Buy ")+variation
    premium = clean(row.get("premium_fraction"))
    hedge = row.get("hedge_type") if category == "Both" else None
    dynamic = hedge == "Downside buffer"
    if category == "Both":
        name += " + buffer" if dynamic else " + long put"
        subtitle = f"{premium:.0%} premium · {row['tenor']}"
    else:
        subtitle = f"{'Put spread' if width else 'Single put'} · {row['tenor']}"
    keys = ("cagr", "annualized_volatility", "daily_sharpe", "max_drawdown", "total_return", "ending_equity")
    return dict(id=row["strategy_id"], name=name, subtitle=subtitle, category=category,
        primary=high, width=width, variation=variation, tenor=row["tenor"], days=int(row["tenor_days"]),
        premium=premium, hedge=hedge, dynamic=dynamic,
        hedgeHigh=clean(row.get("hedge_primary_pct")), hedgeLow=clean(row.get("hedge_secondary_pct")),
        targetMin=clean(row.get("hedge_target_min")), targetMax=clean(row.get("hedge_target_max")),
        cashCycles=int(row["cash_cycles"]), cycles=int(row["cycles"]),
        meanSpend=clean(row.get("mean_premium_fraction_spent")),
        chunk=chunk, slot=slot,
        metrics={key:clean(row[key]) for key in keys})


def copy_assets(site=SITE):
    site.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "core.js", "profit-core.js", "app.js"):
        shutil.copy2(ASSETS/name, site/name)
    library = site.parent / "SPX Research Organized Charts" / "index.html"
    if library.exists():
        html = library.read_text(encoding="utf-8")
        link = f'<a href="../{quote(site.name)}/index.html">Interactive comparison ↗</a>'
        marker = '<div class="document-links">'
        if link not in html and marker in html:
            library.write_text(html.replace(marker, marker+link, 1), encoding="utf-8")


def build(site=SITE):
    started = time.monotonic()
    if shutil.disk_usage(site.parent).free < 600*1024*1024:
        raise RuntimeError("The interactive dataset needs at least 600 MiB of free working space")
    site.mkdir(parents=True, exist_ok=True)
    (site/"data").mkdir(exist_ok=True)
    config = json.loads((RESEARCH/"run.json").read_text(encoding="utf-8"))
    for name, expected in config["source_artifact_sha256"].items():
        if sha256(SOURCE/name) != expected:
            raise ValueError(f"Audited source changed: {name}")
    metrics = pd.read_csv(RESEARCH/"strategy_metrics.csv", dtype={"folder":str}, low_memory=False)
    dates, cycles_by_tenor = prepare_cycles()
    cash = load_cash().reindex(dates).to_numpy(float)
    rows, chunks, chunk_number = [], {}, 0

    def write_group(frame, plain, combined):
        nonlocal chunk_number
        assert plain.shape == combined.shape == (len(dates), len(frame))
        assert np.isfinite(plain).all() and np.isfinite(combined).all()
        for start in range(0, len(frame), 64):
            part = frame.iloc[start:start+64]
            key = f"c{chunk_number:03d}"
            values = np.stack([plain[:, start:start+64].T, combined[:, start:start+64].T], axis=1).astype("<f8")
            raw = values.tobytes(order="C")
            packed = gzip.compress(raw, compresslevel=6, mtime=0)
            assert gzip.decompress(packed) == raw
            encoded = base64.b64encode(packed).decode("ascii")
            target = site/"data"/(key+".js")
            target.write_text(f'window.__SPX_CHUNK__("{key}","{encoded}");\n', encoding="ascii")
            chunks[key] = dict(file="data/"+key+".js", strategies=len(part), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            rows.extend(catalog_row(row, key, slot) for slot, row in enumerate(part.to_dict("records")))
            chunk_number += 1

    # Validate roll-aware overlay reconstruction against ALL existing SPX lines.
    base_metrics = pd.read_csv(BASE_OUTPUT/"strategy_metrics.csv", low_memory=False)
    base_ledger = pd.read_parquet(BASE_OUTPUT/"trade_ledger.parquet",
        columns=["strategy_id", "roll_id", "entry_equity", "terminal_return"])
    ledger_groups = base_ledger.groupby("roll_id", sort=False).indices
    with np.load(BASE_OUTPUT/"curves.npz") as archive:
        base_nav = archive["equity"]
        np.testing.assert_array_equal(archive["dates"], dates.to_numpy())
        base_lookup = {v:i for i,v in enumerate(archive["strategy_ids"])}
    overlay_error = 0.
    for tenor, cycles in cycles_by_tenor.items():
        frame = base_metrics[base_metrics.tenor_days.eq(tenor) & base_metrics.underlying.eq(0)]
        ids = frame.strategy_id.to_list()
        plain = base_nav[:, [base_lookup[v] for v in ids]]
        combined = base_nav[:, [base_lookup["spx_"+v] for v in ids]]
        entries = []
        for cycle in cycles:
            ledger = base_ledger.iloc[ledger_groups[cycle["roll_id"]]].set_index("strategy_id").loc[ids]
            entries.append((ledger.entry_equity.to_numpy(), ledger.terminal_return.to_numpy()))
        verified = overlay_spx(plain, cycles, entries, cash)
        np.testing.assert_allclose(verified, combined, rtol=1e-11, atol=1e-6)
        overlay_error = max(overlay_error, float(np.max(np.abs(verified-combined))))
        write_group(frame, plain, combined)
    del base_ledger, base_nav, ledger_groups
    print(f"Preserved 1,470 option curves and reconciled 1,470 existing SPX overlays ({time.monotonic()-started:.0f}s)", flush=True)
    for tenor, cycles in cycles_by_tenor.items():
        for hedge in ("Long put", "Downside buffer"):
            frame = metrics[metrics.category.eq("Both") & metrics.tenor_days.eq(tenor) & metrics.hedge_type.eq(hedge)].reset_index(drop=True)
            entries = []
            def observe(ledger):
                entries.append((ledger.entry_equity.to_numpy(copy=True), ledger.terminal_return.to_numpy(copy=True)))
            if hedge == "Long put":
                stats, plain = simulate_candidates(frame, cycles, dates, tenor, observe)
            else:
                stats, plain = simulate_buffers(frame, cycles, dates, tenor,
                    minimum_quantity=config["minimum_buffer_quantity"], ledger_writer=observe)
            for key in ("cagr", "annualized_volatility", "daily_sharpe", "max_drawdown", "ending_equity"):
                np.testing.assert_allclose(stats[key], frame[key], rtol=1e-12, atol=1e-9)
            combined = overlay_spx(plain, cycles, entries, cash)
            write_group(frame, plain, combined)
        print(f"Built {tenor}-day comparisons; {len(rows):,} strategies ({time.monotonic()-started:.0f}s)", flush=True)
    assert len(rows) == 7350 and len({row["id"] for row in rows}) == 7350
    catalog = dict(version=1, title="SPX Research · Compare", initial=INITIAL,
        dates=[day.strftime("%Y-%m-%d") for day in dates], spx=cash.tolist(),
        strategies=rows, chunks=chunks, originalCharts=8820, availableLines=14700,
        rules=dict(spx="100% SPX price exposure plus 100% option notional, resized at each option roll; no dividends or financing return",
            buffers=config["buffer_selection_rule"], bufferSizing=config["buffer_sizing"],
            longPuts="Retained full-period combined-Sharpe selections; retrospective",
            costs=config["cost_label"]))
    (site/"catalog.js").write_text("window.SPX_CATALOG="+json.dumps(catalog, separators=(",",":"), allow_nan=False)+";\n", encoding="utf-8")
    copy_assets(site)
    report = dict(status="passed", original_curves_covered=8820, option_strategies=7350,
        existing_spx_overlays_reconciled=1470, new_spx_overlays=5880,
        maximum_existing_overlay_difference_dollars=overlay_error,
        dates=len(dates), daily_values=7350*2*len(dates),
        encoding="gzip + base64, little-endian Float64; exact round-trip checked for every shard",
        data_files=len(chunks), source_artifact_sha256=config["source_artifact_sha256"],
        source_cash_sha256=sha256(CASH_PATH), source_base_curves_sha256=sha256(BASE_OUTPUT/"curves.npz"),
        total_bytes=sum(path.stat().st_size for path in site.rglob("*") if path.is_file()))
    (site/"Data validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (site/"README.md").write_text("# SPX Research Interactive\n\nOpen **index.html** in current Chrome or Edge. No server or internet connection is needed. Keep the entire folder together.\n\nThe website opens with only the SPX benchmark. Load a saved setup explicitly to restore a comparison. The chart’s upper-left panel shows CAGR, annualized volatility, maximum drawdown and Sharpe for the visible portfolios; with-SPX lines also show differences versus SPX over the same dates.\n\nSelect any number of option strategies, choose Options only / With SPX / Both for each, and adjust the date range. Option-expiry filters are separate from the historical date window. All lines are independent portfolios.\n\nThe site includes 7,350 distinct strategies and 14,700 options-only/SPX combinations, plus SPX as a benchmark. All 8,820 existing chart curves are represented. Data through September 18, 2026.\n\nSPX overlays use fixed quantities between option rolls and resize the complete portfolio at the next roll. SPX remains invested during option cash cycles. These are 100% SPX price exposure plus 100% options notional, not a 50/50 mix or a sum of compounded equity curves. No dividends, financing, taxes or interest are modeled.\n\nTime-window metrics include returns from the close before the first selected session; the original first session uses initial capital. Existing positions are not restarted when you change the date range. Buffers retain entry-date affordability selection; long-put targets remain retrospective research winners.\n\nRebuild: `python -B spx_option_research/scripts/build_interactive_spx.py`. Refresh only the interface: add `--assets-only`.\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--output", type=Path, default=SITE)
    args = parser.parse_args()
    copy_assets(args.output) if args.assets_only else build(args.output)
