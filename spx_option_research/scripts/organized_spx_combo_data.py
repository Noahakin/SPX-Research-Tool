"""Extend the existing SPX research inputs with observed 80%–89% put legs.

The original schedules, contracts, and raw quotes are reused without changes.
Only supplemental contracts and missing daily date/symbol requirements are read
from the archive. This module neither repairs quotes nor estimates prices.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from organized_spx_data import (
    DATA_ROOT, DEFAULT_OUTPUT as BASE_OUTPUT, ENTRY_COLUMNS, ISSUE_COLUMNS,
    QUOTE_COLUMNS, SELECTED_COLUMNS, SPXSurfaceArchive, _issue, _normalize,
    _quote_valid, _select_entries, add_strike_range_metadata, load_cash,
    quote_requirements,
)


DEFAULT_OUTPUT = BASE_OUTPUT.parent / "organized_spx_combinations"
SUPPLEMENTAL_RATIOS = tuple(value / 100 for value in range(80, 90))
KEYS = ["snapshot_date", "option_symbol"]
FORMAT_VERSION = 1


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supplemental_entries(archive: SPXSurfaceArchive, schedule: pd.DataFrame,
                         cash: pd.Series, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    grouped = list(schedule.loc[schedule.has_expiry].groupby("entry_date", sort=True))

    def read_one(item: tuple[pd.Timestamp, pd.DataFrame]) -> tuple[list[dict], list[dict]]:
        day, cycles = item
        if day not in archive.paths:
            raise LookupError(f"Original entry session absent from archive: {day.date()}")
        frame = _normalize(pq.ParquetFile(archive.path_for(day)).read(columns=ENTRY_COLUMNS).to_pandas())
        puts = frame.loc[frame.option_symbol.str.startswith("SPXW") & frame.option_type.astype(str).str.lower().eq("put")]
        if puts.snapshot_date.ne(day).any():
            raise ValueError(f"Archive entry snapshot date mismatch: {day.date()}")
        if puts.option_symbol.duplicated().any():
            raise ValueError(f"Duplicate source entry symbols: {day.date()}")
        selected, issues = [], []
        for row in cycles.to_dict("records"):
            chosen, problems = _select_entries(puts, row, float(cash.loc[day]), SUPPLEMENTAL_RATIOS)
            selected.extend(chosen)
            issues.extend(problems)
        return selected, issues

    records, issues = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (chosen, problems) in enumerate(pool.map(read_one, grouped), 1):
            records.extend(chosen)
            issues.extend(problems)
            if count % 300 == 0:
                print(f"Supplemental entries: read {count:,}/{len(grouped):,} snapshots", flush=True)
    entries = pd.DataFrame(records, columns=SELECTED_COLUMNS).sort_values(["target_dte", "entry_date", "ratio"]).reset_index(drop=True)
    ranges = entries[["entry_date", "expiration_date", "min_listed_strike", "max_listed_strike"]].drop_duplicates()
    entries, range_issues = add_strike_range_metadata(entries, ranges)
    issues = pd.concat([pd.DataFrame(issues, columns=ISSUE_COLUMNS), range_issues], ignore_index=True)
    if len(entries) != int(schedule.has_expiry.sum()) * len(SUPPLEMENTAL_RATIOS):
        raise ValueError("Supplemental entry grid is incomplete")
    return entries, issues, len(grouped)


def load_missing_quotes(archive: SPXSurfaceArchive, wanted: dict[pd.Timestamp, set[str]],
                        existing: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    if existing.duplicated(KEYS).any():
        raise ValueError("Original quote cache contains duplicate date/symbol keys")
    have = {day: set(group.option_symbol) for day, group in existing.groupby("snapshot_date", sort=False)}
    missing = {day: symbols - have.get(day, set()) for day, symbols in wanted.items()}
    missing = {day: symbols for day, symbols in missing.items() if symbols}

    def read_one(item: tuple[pd.Timestamp, set[str]]) -> tuple[pd.DataFrame, list[dict]]:
        day, symbols = item
        if day not in archive.paths:
            return pd.DataFrame(columns=[*QUOTE_COLUMNS, "source_file"]), [
                _issue("daily", day, "required exact quote absent", symbol=symbol, detail="archive date absent")
                for symbol in sorted(symbols)]
        path = archive.path_for(day)
        frame = pq.ParquetFile(path).read(columns=QUOTE_COLUMNS).to_pandas()
        frame["option_symbol"] = frame.option_symbol.astype(str).str.strip()
        frame = _normalize(frame.loc[frame.option_symbol.isin(symbols)])
        frame["source_file"] = path.relative_to(archive.root).as_posix()
        issues = [_issue("daily", day, "required exact quote absent", symbol=symbol)
                  for symbol in sorted(symbols - set(frame.option_symbol))]
        if frame.snapshot_date.ne(day).any():
            raise ValueError(f"Archive daily snapshot date mismatch: {day.date()}")
        if frame.option_symbol.duplicated().any():
            raise ValueError(f"Duplicate supplemental daily symbols: {day.date()}")
        for quote in frame.loc[~_quote_valid(frame)].itertuples(index=False):
            issues.append(_issue("daily", day, "invalid bid/ask/strike", symbol=quote.option_symbol))
        return frame, issues

    pieces, issues = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, (frame, problems) in enumerate(pool.map(read_one, missing.items()), 1):
            pieces.append(frame)
            issues.extend(problems)
            if count % 400 == 0:
                print(f"Supplemental daily quotes: read {count:,}/{len(missing):,} snapshots", flush=True)
    quotes = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=[*QUOTE_COLUMNS, "source_file"])
    quotes = quotes.sort_values(KEYS).reset_index(drop=True)
    return quotes, pd.DataFrame(issues, columns=ISSUE_COLUMNS), len(missing), sum(map(len, missing.values()))


def validate_inputs(entries: pd.DataFrame, quotes: pd.DataFrame, schedule: pd.DataFrame,
                    base_entries: pd.DataFrame, base_quotes: pd.DataFrame,
                    cash: pd.Series) -> dict:
    if entries.duplicated(["roll_id", "ratio"]).any() or quotes.duplicated(KEYS).any():
        raise ValueError("Combined inputs contain duplicate keys")
    original_entries = entries.loc[entries.ratio.ge(.90)].sort_values(["roll_id", "ratio"]).reset_index(drop=True)
    expected_entries = base_entries.sort_values(["roll_id", "ratio"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(original_entries[base_entries.columns], expected_entries, check_exact=True)
    old_index = pd.MultiIndex.from_frame(base_quotes[KEYS])
    current_index = pd.MultiIndex.from_frame(quotes[KEYS])
    retained = quotes.set_index(KEYS).loc[old_index].reset_index()
    pd.testing.assert_frame_equal(retained[base_quotes.columns], base_quotes.reset_index(drop=True), check_exact=True)
    if len(entries) != int(schedule.has_expiry.sum()) * 31:
        raise ValueError("Combined inputs must have 31 strike targets per quoted cycle")
    if not entries.groupby("roll_id").size().eq(31).all():
        raise ValueError("Combined strike grid is incomplete")
    if not entries.entry_date.isin(cash.index).all() or not quotes.snapshot_date.isin(cash.index).all():
        raise ValueError("Option dates differ from the original cash calendar")
    if not quotes.snapshot_date.lt(quotes.expiration_date).all():
        raise ValueError("Expiration or post-expiration option quotes were loaded")
    cash_entry = entries.entry_date.map(cash)
    if not np.allclose(cash_entry, entries.spot_entry, rtol=0, atol=0):
        raise ValueError("Combined entries do not use the original cash closes")
    if entries[["min_listed_strike", "max_listed_strike", "target_strike", "selected_strike_distance_points"]].isna().any().any():
        raise ValueError("Missing strike provenance")
    wanted = quote_requirements(entries, cash.index, cash.index[-1])
    required = pd.MultiIndex.from_tuples([(day, symbol) for day, symbols in wanted.items() for symbol in symbols], names=KEYS)
    absent, extra = required.difference(current_index), current_index.difference(required)
    if len(extra):
        raise ValueError("Unrequested date/symbol pairs found in combined quotes")
    entry_quotes = entries.merge(quotes, left_on=["entry_date", "option_symbol"], right_on=KEYS,
                                  how="left", validate="many_to_one", suffixes=("_entry", "_daily"))
    if entry_quotes.snapshot_date.isna().any():
        raise ValueError("A selected entry quote is missing from daily observations")
    for field in ("strike", "bid", "ask"):
        if not np.allclose(entry_quotes[f"{field}_entry"], entry_quotes[f"{field}_daily"], equal_nan=True, rtol=0, atol=0):
            raise ValueError(f"Selected entry {field} differs from its exact daily quote")
    if not entry_quotes.expiration_date_entry.eq(entry_quotes.expiration_date_daily).all():
        raise ValueError("Selected entry expiry differs from its exact daily quote")
    supplemental = entries.loc[entries.ratio.lt(.90)]
    return dict(required_daily_quotes=len(required), observed_daily_quotes=len(quotes),
                missing_daily_quotes=len(absent), duplicate_daily_quotes=0,
                preserved_original_entries=len(base_entries), preserved_original_quotes=len(base_quotes),
                supplemental_outside_listed_range=int(supplemental.target_outside_listed_range.sum()),
                supplemental_invalid_entry_quotes=int((~supplemental.quote_valid).sum()),
                combined_invalid_daily_quotes=int((~_quote_valid(quotes)).sum()),
                combined_zero_ask_quotes=int(quotes.ask.eq(0).sum()))


def build_combo_extraction(output: Path = DEFAULT_OUTPUT, *, base: Path = BASE_OUTPUT,
                           workers: int = 6, force: bool = False) -> dict:
    output, base = Path(output).resolve(), Path(base).resolve()
    if output == base or base in output.parents:
        raise ValueError("Supplemental output must be separate from original research inputs")
    if workers < 1:
        raise ValueError("workers must be positive")
    base_files = ("entries.parquet", "daily_quotes.parquet", "schedule.parquet", "issues.csv", "manifest.json")
    base_hashes = {name: _hash_file(base / name) for name in base_files}
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest.get("stage") != "complete":
        raise ValueError("Original extraction must be complete")
    settings = base_manifest["settings"]
    full_cash = load_cash(Path(settings["cash_source"]))
    cash_hash = hashlib.sha256(pd.util.hash_pandas_object(full_cash, index=True).values.tobytes()).hexdigest()
    if cash_hash != settings["cash_sha256"]:
        raise ValueError("Cash source changed since the original extraction")
    cash = full_cash.loc[pd.Timestamp(settings["start"]):pd.Timestamp(settings["end"])]
    archive = SPXSurfaceArchive(Path(settings.get("archive_root", DATA_ROOT)), cache_size=1)
    index = []
    for day in cash.index:
        path = archive.paths.get(day)
        if path is None:
            index.append(dict(date=str(day.date()), path=None))
        else:
            stat = path.stat()
            index.append(dict(date=str(day.date()), path=path.relative_to(archive.root).as_posix(), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    archive_hash = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
    if archive_hash != base_manifest["source_index_sha256"]:
        raise ValueError("Raw archive changed since original extraction; mixed source vintages are prohibited")
    inputs = dict(format_version=FORMAT_VERSION, source_directory=str(base), source_input_sha256=base_manifest["input_sha256"],
                  source_artifact_sha256=base_hashes, source_index_sha256=archive_hash, cash_sha256=cash_hash,
                  supplemental_ratios=SUPPLEMENTAL_RATIOS, combined_ratios=tuple(i / 100 for i in range(80, 111)))
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    hit = not force and prior.get("input_sha256") == fingerprint
    base_entries = pd.read_parquet(base / "entries.parquet")
    base_quotes = pd.read_parquet(base / "daily_quotes.parquet")
    schedule = pd.read_parquet(base / "schedule.parquet")
    if hit and prior.get("stage") == "complete" and all((output / name).exists() for name in ("entries.parquet", "daily_quotes.parquet", "schedule.parquet", "issues.csv")):
        print("Combined extraction cache hit; no archive chains reread.", flush=True)
        return dict(entries=pd.read_parquet(output / "entries.parquet"), daily_quotes=pd.read_parquet(output / "daily_quotes.parquet"),
                    schedule=pd.read_parquet(output / "schedule.parquet"), cash=cash,
                    issues=pd.read_csv(output / "issues.csv", parse_dates=["date"]), manifest=prior)
    if prior and not hit:
        print("Combined-input cache invalidated; rebuilding only derived supplemental inputs.", flush=True)
    if hit and (output / "supplemental_entries.parquet").exists() and (output / "supplemental_issues.csv").exists():
        additions = pd.read_parquet(output / "supplemental_entries.parquet")
        added_issues = pd.read_csv(output / "supplemental_issues.csv", parse_dates=["date"])
        added_issues = added_issues.loc[added_issues.stage.ne("daily")]
        entry_files = int(prior["supplemental_entry_snapshot_files"])
        print(f"Supplemental entry cache: {len(additions):,} records", flush=True)
    else:
        additions, added_issues, entry_files = supplemental_entries(archive, schedule, cash, workers)
        additions.to_parquet(output / "supplemental_entries.parquet", index=False)
        added_issues.to_csv(output / "supplemental_issues.csv", index=False, date_format="%Y-%m-%d")
    entries = pd.concat([base_entries, additions], ignore_index=True).sort_values(["target_dte", "entry_date", "ratio"]).reset_index(drop=True)
    entries.to_parquet(output / "entries.parquet", index=False)
    # Byte-identical schedule copy makes original date/expiration preservation
    # independently checkable without relying on a fresh schedule calculation.
    (output / "schedule.parquet").write_bytes((base / "schedule.parquet").read_bytes())
    manifest = dict(settings=dict(settings, ratios=list(inputs["combined_ratios"])), input_sha256=fingerprint,
                    source_input_sha256=base_manifest["input_sha256"], inputs=inputs, stage="entries",
                    source_index_sha256=archive_hash, source_cash_sessions=len(cash),
                    supplemental_entry_snapshot_files=entry_files, original_entries=len(base_entries),
                    supplemental_entries=len(additions), entries=len(entries), availability=base_manifest["availability"],
                    original_data_modified=False, interpolation=False, missing_quote_fill=False,
                    invalid_quote_substitution=False, expiration_valuation=base_manifest["expiration_valuation"],
                    final_open_positions=base_manifest["final_open_positions"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    wanted = quote_requirements(entries, cash.index, cash.index[-1])
    extra_quotes, daily_issues, quote_files, missing_requirements = load_missing_quotes(archive, wanted, base_quotes, workers)
    quotes = pd.concat([base_quotes, extra_quotes], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    qa = validate_inputs(entries, quotes, schedule, base_entries, base_quotes, cash)
    added_issues = pd.concat([added_issues, daily_issues], ignore_index=True)
    old_issues = pd.read_csv(base / "issues.csv", parse_dates=["date"])
    issues = pd.concat([old_issues, added_issues], ignore_index=True)
    extra_quotes.to_parquet(output / "supplemental_daily_quotes.parquet", index=False)
    quotes.to_parquet(output / "daily_quotes.parquet", index=False)
    added_issues.to_csv(output / "supplemental_issues.csv", index=False, date_format="%Y-%m-%d")
    issues.to_csv(output / "issues.csv", index=False, date_format="%Y-%m-%d")
    if {name: _hash_file(base / name) for name in base_files} != base_hashes:
        raise ValueError("Original inputs changed during extraction")
    if _hash_file(output / "schedule.parquet") != base_hashes["schedule.parquet"]:
        raise ValueError("Copied schedule does not match original bytes")
    manifest.update(stage="complete", reused_daily_quotes=len(base_quotes), supplemental_daily_quotes=len(extra_quotes),
                    supplemental_daily_snapshot_files=quote_files, supplemental_daily_requirements=missing_requirements,
                    daily_quotes=len(quotes), qa=qa,
                    issue_counts={str(key): int(value) for key, value in issues.reason.value_counts().items()})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Combined extraction: {len(entries):,} entries; {len(quotes):,} quotes ({len(base_quotes):,} reused, {len(extra_quotes):,} supplemental).", flush=True)
    print("QA: " + json.dumps(qa, sort_keys=True), flush=True)
    return dict(entries=entries, daily_quotes=quotes, schedule=schedule, cash=cash, issues=issues, manifest=manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_combo_extraction(args.output, base=args.base, workers=args.workers, force=args.force)
