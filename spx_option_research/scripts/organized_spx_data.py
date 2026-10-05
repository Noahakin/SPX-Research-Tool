"""Extract fixed SPX put contracts and their observed daily quotes.

The cash index determines strikes and the trading calendar. Source underlying
prices, Greeks, and supplied midpoints are deliberately not loaded. Invalid or
missing quotes are reported without substituting contracts or estimating prices.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DATA_ROOT = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
CASH_PATH = PROJECT / "results/weekly_dynamic_research/market_spx.csv"
DEFAULT_OUTPUT = PROJECT / "results/organized_spx_research"
START = pd.Timestamp("2016-09-23")
END = pd.Timestamp("2026-09-18")
TARGET_DTES = (3, 7, 14, 21, 28, 42, 56)
RATIOS = tuple(i / 100 for i in range(90, 111))
QUOTE_COLUMNS = ["snapshot_date", "expiration_date", "option_symbol", "strike", "bid", "ask"]
ENTRY_COLUMNS = [*QUOTE_COLUMNS, "option_type"]
SCHEDULE_COLUMNS = ["roll_id", "target_dte", "entry_date", "expiration_date", "valuation_end", "actual_dte", "has_expiry", "reason"]
SELECTED_COLUMNS = ["roll_id", "target_dte", "entry_date", "expiration_date", "actual_dte", "spot_entry", "ratio", "option_symbol", "strike", "bid", "ask", "mid", "quote_valid",
                    "min_listed_strike", "max_listed_strike", "target_strike", "selected_strike_distance_points", "target_outside_listed_range"]
ISSUE_COLUMNS = ["stage", "date", "roll_id", "target_dte", "option_symbol", "reason", "detail"]
FORMAT_VERSION = 2
sys.path.insert(0, str(PROJECT / "src"))
from spxresearch.data_loader import SPXSurfaceArchive  # noqa: E402


def load_cash(cash_path: Path = CASH_PATH) -> pd.Series:
    frame = pd.read_csv(cash_path, usecols=["date", "close"], parse_dates=["date"])
    frame["date"] = frame.date.dt.normalize()
    frame["close"] = pd.to_numeric(frame.close, errors="coerce")
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError("Cash dates must be nonmissing and unique")
    if not np.isfinite(frame.close).all() or frame.close.le(0).any():
        raise ValueError("Cash closes must be finite and positive")
    return frame.set_index("date").close.sort_index()


def _issue(stage: str, day: pd.Timestamp, reason: str, *, roll_id: str = "", target_dte: int | None = None, symbol: str = "", detail: str = "") -> dict:
    return dict(stage=stage, date=day, roll_id=roll_id, target_dte=target_dte, option_symbol=symbol, reason=reason, detail=detail)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("snapshot_date", "expiration_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    frame["option_symbol"] = frame.option_symbol.astype(str).str.strip()
    for column in ("strike", "bid", "ask"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _quote_valid(frame: pd.DataFrame) -> pd.Series:
    return (pd.Series(np.isfinite(frame[["strike", "bid", "ask"]]).all(axis=1), index=frame.index)
            & frame.strike.gt(0) & frame.bid.ge(0) & frame.ask.ge(frame.bid))


def _holiday_adjust(anchor: pd.Timestamp, full_cash: pd.Series) -> pd.Timestamp:
    """Use the prior observed session; future listed Fridays remain Fridays."""
    if anchor > full_cash.index.max():
        if anchor.weekday() != 4:
            raise ValueError("Only future calendar-Friday anchors are supported")
        return anchor
    prior = full_cash.index[full_cash.index <= anchor]
    if len(prior) == 0 or (anchor - prior[-1]).days > 4:
        raise ValueError(f"No cash session within four days of {anchor.date()}")
    return prior[-1]


def _schedule_row(target: int, entry: pd.Timestamp, expiry: pd.Timestamp, end: pd.Timestamp, *, has_expiry: bool = False, reason: str = "") -> dict:
    return dict(roll_id=f"dte{target:03d}_{entry:%Y%m%d}", target_dte=target,
                entry_date=entry, expiration_date=expiry, valuation_end=min(expiry, end),
                actual_dte=int((expiry - entry).days), has_expiry=has_expiry, reason=reason)


def anchored_schedule(full_cash: pd.Series, start: pd.Timestamp, end: pd.Timestamp, targets: Sequence[int]) -> pd.DataFrame:
    if start.weekday() != 4:
        raise ValueError("Weekly cycles require a calendar-Friday starting anchor")
    rows = []
    for target in targets:
        if target == 3:
            continue
        if target <= 0 or target % 7:
            raise ValueError(f"Unsupported anchored calendar tenor: {target}")
        for anchor in pd.date_range(start, end, freq=f"{target}D"):
            entry = _holiday_adjust(anchor, full_cash)
            if entry < start or entry >= end:
                continue
            expiry = _holiday_adjust(anchor + pd.Timedelta(days=target), full_cash)
            rows.append(_schedule_row(target, entry, expiry, end))
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def _eligible_expirations(puts: pd.DataFrame, day: pd.Timestamp, full_cash: pd.Series) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    listed = pd.DatetimeIndex(puts.expiration_date.dropna().unique()).sort_values()
    future = listed[listed > day]
    eligible = future[(future.weekday < 5) & ((future > full_cash.index.max()) | future.isin(full_cash.index))]
    return eligible, future.difference(eligible)


def _select_entries(puts: pd.DataFrame, row: dict, spot: float, ratios: Sequence[float]) -> tuple[list[dict], list[dict]]:
    same_expiry = puts.loc[puts.expiration_date.eq(row["expiration_date"])].copy()
    listed = same_expiry.loc[np.isfinite(same_expiry.strike) & same_expiry.strike.gt(0)]
    listed = listed.sort_values(["strike", "option_symbol"], kind="stable")
    if listed.empty:
        raise ValueError(f"No usable listed strike for {row['roll_id']}")
    selected, issues = [], []
    strikes = listed.strike.to_numpy(dtype=float)
    for ratio in ratios:
        # Sorting first makes exact-distance ties choose the lower listed strike.
        quote = listed.iloc[int(np.argmin(np.abs(strikes - spot * ratio)))]
        valid = bool(np.isfinite([quote.strike, quote.bid, quote.ask]).all() and quote.bid >= 0 and quote.ask >= quote.bid)
        selected.append({key: row[key] for key in ("roll_id", "target_dte", "entry_date", "expiration_date", "actual_dte")} | dict(
            spot_entry=float(spot), ratio=float(ratio), option_symbol=quote.option_symbol,
            strike=float(quote.strike), bid=float(quote.bid), ask=float(quote.ask),
            mid=float((quote.bid + quote.ask) / 2), quote_valid=valid,
            min_listed_strike=float(strikes.min()), max_listed_strike=float(strikes.max()), target_strike=float(spot * ratio),
            selected_strike_distance_points=float(abs(quote.strike - spot * ratio)),
            target_outside_listed_range=bool(spot * ratio < strikes.min() or spot * ratio > strikes.max())))
        if not valid:
            issues.append(_issue("entry", row["entry_date"], "invalid selected entry quote", roll_id=row["roll_id"], target_dte=row["target_dte"], symbol=quote.option_symbol, detail=f"ratio={ratio:.2f}; nearest listed strike retained"))
    return selected, issues


def build_entries(archive: SPXSurfaceArchive, full_cash: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                  targets: Sequence[int], ratios: Sequence[float], renewal_targets: Sequence[int] = (56,)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    schedule = anchored_schedule(full_cash, start, end, [target for target in targets if target not in renewal_targets])
    scheduled = {day: group.to_dict("records") for day, group in schedule.groupby("entry_date", sort=True)}
    days = full_cash.index[(full_cash.index >= start) & (full_cash.index <= end)]
    next_renewal_entries = {target: start for target in targets if target == 3 or target in renewal_targets}
    schedules, entries, issues = [], [], []
    read_count = 0
    for day_number, day in enumerate(days[:-1]):
        day_rows = scheduled.get(day, [])
        renewals_due = [target for target, due in next_renewal_entries.items() if day == due]
        if not day_rows and not renewals_due:
            continue
        if day not in archive.paths:
            puts = pd.DataFrame(columns=ENTRY_COLUMNS)
            issues.append(_issue("entry", day, "archive session absent"))
        else:
            frame = _normalize(pq.ParquetFile(archive.path_for(day)).read(columns=ENTRY_COLUMNS).to_pandas())
            puts = frame.loc[frame.option_symbol.str.startswith("SPXW") & frame.option_type.astype(str).str.lower().eq("put")].copy()
            bad_dates = puts.snapshot_date.ne(day)
            if bad_dates.any():
                raise ValueError(f"Source snapshot date disagrees with archive path: {day.date()}")
            duplicates = puts[puts.option_symbol.duplicated(keep=False)]
            for symbol in duplicates.option_symbol.unique():
                issues.append(_issue("entry", day, "duplicate source entry symbol", symbol=symbol))
            read_count += 1
            if read_count % 300 == 0:
                print(f"Pass 1: read {read_count:,} entry snapshots", flush=True)
        eligible, excluded = _eligible_expirations(puts, day, full_cash)
        for expiry in excluded:
            issues.append(_issue("calendar", day, "listed expiration is not a cash session", detail=str(expiry.date())))
        for target in renewals_due:
            if target == 3:
                choices = eligible[((eligible - day).days >= 1) & ((eligible - day).days <= 5)]
                policy = "nearest listed PM expiration within 1–5 calendar days; ties prefer longer"
                absent = "no listed PM expiration within 1–5 calendar days; cash until next session"
            else:
                anchors = pd.date_range(day + pd.Timedelta(days=1), eligible.max() + pd.Timedelta(days=6), freq="W-FRI") if len(eligible) else []
                adjusted_fridays = {_holiday_adjust(anchor, full_cash) for anchor in anchors}
                choices = eligible[eligible.isin(adjusted_fridays)]
                policy = f"nearest listed Friday or adjusted-Friday PM expiration to {target} calendar days; ties prefer longer; renew at expiry"
                absent = "no listed Friday or adjusted-Friday PM expiration; cash until next session"
            if len(choices):
                expiry = min(choices, key=lambda value: (abs((value - day).days - target), -(value - day).days))
                renewal_row = _schedule_row(target, day, expiry, end, has_expiry=True, reason=policy)
            else:
                expiry = days[day_number + 1]
                renewal_row = _schedule_row(target, day, expiry, end, reason=absent)
            next_renewal_entries[target] = expiry
            day_rows = [*day_rows, renewal_row]
        for row in day_rows:
            if row["target_dte"] not in next_renewal_entries:
                row["has_expiry"] = row["expiration_date"] in eligible
                row["reason"] = ("exact PM expiration on adjusted calendar-Friday anchor" if row["has_expiry"]
                                 else "exact PM expiration absent; explicit cash cycle")
            schedules.append(row)
            if not row["has_expiry"]:
                issues.append(_issue("schedule", day, row["reason"], roll_id=row["roll_id"], target_dte=row["target_dte"], detail=str(row["expiration_date"].date())))
                continue
            chosen, selected_issues = _select_entries(puts, row, float(full_cash.loc[day]), ratios)
            entries.extend(chosen)
            issues.extend(selected_issues)
    schedule = pd.DataFrame(schedules, columns=SCHEDULE_COLUMNS).sort_values(["target_dte", "entry_date"]).reset_index(drop=True)
    entries = pd.DataFrame(entries, columns=SELECTED_COLUMNS).sort_values(["target_dte", "entry_date", "ratio"]).reset_index(drop=True)
    for target, group in schedule.groupby("target_dte"):
        if group.iloc[0].entry_date != start or group.iloc[-1].valuation_end != end:
            raise ValueError(f"Incomplete calendar coverage for target {target}")
        if not np.array_equal(group.entry_date.iloc[1:].to_numpy(), group.valuation_end.iloc[:-1].to_numpy()):
            raise ValueError(f"Overlapping or missing roll intervals for target {target}")
    expected = int(schedule.has_expiry.sum()) * len(ratios)
    if len(entries) != expected or entries.duplicated(["roll_id", "ratio"]).any():
        raise ValueError("Selected entry grid is incomplete or duplicated")
    return entries, schedule, pd.DataFrame(issues, columns=ISSUE_COLUMNS), read_count


def quote_requirements(entries: pd.DataFrame, cash_dates: pd.DatetimeIndex, end: pd.Timestamp) -> dict[pd.Timestamp, set[str]]:
    wanted: dict[pd.Timestamp, set[str]] = {}
    for (entry, expiry), group in entries.groupby(["entry_date", "expiration_date"], sort=True):
        symbols = set(group.option_symbol)
        for day in cash_dates[(cash_dates >= entry) & (cash_dates < expiry) & (cash_dates <= end)]:
            wanted.setdefault(day, set()).update(symbols)
    return dict(sorted(wanted.items()))


def build_daily_quotes(archive: SPXSurfaceArchive, entries: pd.DataFrame, cash_dates: pd.DatetimeIndex,
                       end: pd.Timestamp, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    wanted = quote_requirements(entries, cash_dates, end)
    range_expiries = {day: set(group.expiration_date) for day, group in entries.groupby("entry_date")}

    def read_one(item: tuple[pd.Timestamp, set[str]]) -> tuple[pd.DataFrame, list[dict], list[dict]]:
        day, symbols = item
        if day not in archive.paths:
            return pd.DataFrame(columns=[*QUOTE_COLUMNS, "source_file"]), [
                _issue("daily", day, "archive session absent"),
                *[_issue("daily", day, "required exact quote absent", symbol=symbol) for symbol in sorted(symbols)]], []
        path = archive.path_for(day)
        frame = pq.ParquetFile(path).read(columns=QUOTE_COLUMNS).to_pandas()
        frame["option_symbol"] = frame.option_symbol.astype(str).str.strip()
        ranges = []
        if day in range_expiries:
            expiry = pd.to_datetime(frame.expiration_date).dt.normalize()
            pm_put = frame.option_symbol.str.startswith("SPXW") & frame.option_symbol.str.contains(r"\d{6}P", regex=True)
            listed = frame.loc[pm_put & expiry.isin(range_expiries[day]), ["expiration_date", "strike"]].copy()
            listed["expiration_date"] = pd.to_datetime(listed.expiration_date).dt.normalize()
            listed["strike"] = pd.to_numeric(listed.strike, errors="coerce")
            listed = listed.loc[np.isfinite(listed.strike) & listed.strike.gt(0)]
            for expiration, group in listed.groupby("expiration_date"):
                ranges.append(dict(entry_date=day, expiration_date=expiration,
                                   min_listed_strike=float(group.strike.min()), max_listed_strike=float(group.strike.max())))
        frame = _normalize(frame.loc[frame.option_symbol.isin(symbols)])
        frame["source_file"] = path.relative_to(archive.root).as_posix()
        problems = []
        for symbol in sorted(symbols - set(frame.option_symbol)):
            problems.append(_issue("daily", day, "required exact quote absent", symbol=symbol))
        for symbol in frame.loc[frame.option_symbol.duplicated(keep=False), "option_symbol"].unique():
            problems.append(_issue("daily", day, "duplicate exact quote", symbol=symbol))
        for quote in frame.loc[frame.snapshot_date.ne(day)].itertuples(index=False):
            problems.append(_issue("daily", day, "snapshot date disagrees with archive path", symbol=quote.option_symbol))
        for quote in frame.loc[~_quote_valid(frame)].itertuples(index=False):
            problems.append(_issue("daily", day, "invalid bid/ask/strike", symbol=quote.option_symbol))
        return frame, problems, ranges

    pieces, issues, ranges = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (frame, problems, entry_ranges) in enumerate(pool.map(read_one, wanted.items()), start=1):
            pieces.append(frame)
            issues.extend(problems)
            ranges.extend(entry_ranges)
            if number % 400 == 0:
                print(f"Pass 2: read {number:,}/{len(wanted):,} daily snapshots", flush=True)
    quotes = (pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=[*QUOTE_COLUMNS, "source_file"]))
    quotes = quotes.sort_values(["snapshot_date", "option_symbol"]).reset_index(drop=True)
    return quotes, pd.DataFrame(issues, columns=ISSUE_COLUMNS), len(wanted), pd.DataFrame(ranges)


def add_strike_range_metadata(entries: pd.DataFrame, ranges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries = entries.drop(columns=["min_listed_strike", "max_listed_strike"], errors="ignore").merge(
        ranges, on=["entry_date", "expiration_date"], how="left", validate="many_to_one")
    if entries[["min_listed_strike", "max_listed_strike"]].isna().any().any():
        raise ValueError("Missing observed entry strike range")
    entries["target_strike"] = entries.spot_entry * entries.ratio
    entries["selected_strike_distance_points"] = (entries.strike - entries.target_strike).abs()
    entries["target_outside_listed_range"] = entries.target_strike.lt(entries.min_listed_strike) | entries.target_strike.gt(entries.max_listed_strike)
    issues = [_issue("entry_range", row.entry_date, "target strike outside listed range", roll_id=row.roll_id,
                     target_dte=row.target_dte, symbol=row.option_symbol,
                     detail=f"ratio={row.ratio:.2f}; target={row.target_strike:.8f}; listed_range={row.min_listed_strike:g}..{row.max_listed_strike:g}; selected_distance={row.selected_strike_distance_points:.8f}")
              for row in entries.loc[entries.target_outside_listed_range].itertuples(index=False)]
    return entries[SELECTED_COLUMNS], pd.DataFrame(issues, columns=ISSUE_COLUMNS)


def _availability(schedule: pd.DataFrame) -> list[dict]:
    rows = []
    for target, group in schedule.groupby("target_dte", sort=True):
        traded = group[group.has_expiry]
        days = (group.valuation_end - group.entry_date).dt.days
        rows.append(dict(target_dte=int(target), cycles=len(group), quoted_cycles=len(traded),
                         cash_cycles=int((~group.has_expiry).sum()),
                         quoted_calendar_day_fraction=float(days[group.has_expiry].sum() / days.sum()),
                         actual_dte_counts={str(int(k)): int(v) for k, v in traded.actual_dte.value_counts().sort_index().items()}))
    return rows


def build_extraction(output: Path = DEFAULT_OUTPUT, *, start=START, end=END, targets: Sequence[int] = TARGET_DTES,
                     ratios: Sequence[float] = RATIOS, archive_root: Path = DATA_ROOT, cash_path: Path = CASH_PATH,
                     workers: int = 6, force: bool = False, entries_only: bool = False, renewal_targets: Sequence[int] = (56,)) -> dict:
    output = Path(output).resolve()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    targets, ratios = tuple(sorted(set(map(int, targets)))), tuple(sorted(set(map(float, ratios))))
    renewal_targets = tuple(sorted(set(map(int, renewal_targets)) & set(targets)))
    if start >= end or not targets or not ratios or not np.isfinite(ratios).all() or min(ratios) <= 0 or workers < 1:
        raise ValueError("Invalid extraction interval or grid")
    full_cash = load_cash(Path(cash_path))
    if start not in full_cash.index or end not in full_cash.index:
        raise ValueError("Both extraction endpoints must have observed cash closes")
    cash = full_cash.loc[start:end]
    archive = SPXSurfaceArchive(archive_root, cache_size=1)
    source_index = []
    for day in cash.index:
        path = archive.paths.get(day)
        if path is None:
            source_index.append(dict(date=str(day.date()), path=None))
        else:
            stat = path.stat()
            source_index.append(dict(date=str(day.date()), path=path.relative_to(archive.root).as_posix(), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    settings = dict(format_version=FORMAT_VERSION, start=str(start.date()), end=str(end.date()), targets=targets, ratios=ratios,
                    weekly_expiration_policy="anchored_exact_except_renewal_targets", renewal_targets=renewal_targets,
                    renewal_expiration_policy="closest_listed_Friday_or_adjusted_Friday_ties_longer",
                    short_expiration_policy="closest_1_to_5_days_ties_longer",
                    cash_source=str(Path(cash_path).resolve()), archive_root=str(archive.root),
                    cash_sha256=hashlib.sha256(pd.util.hash_pandas_object(full_cash, index=True).values.tobytes()).hexdigest())
    fingerprint = hashlib.sha256(json.dumps(dict(settings=settings, source_index=source_index), sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    archive_index_hash = hashlib.sha256(json.dumps(source_index, sort_keys=True).encode()).hexdigest()
    cache_hit = (not force and prior.get("input_sha256") == fingerprint
                 and all((output / name).exists() for name in ("entries.parquet", "schedule.parquet", "issues.csv")))
    if prior and not cache_hit:
        print("Extraction cache invalidated by changed inputs, protocol, or --force; rebuilding derived artifacts.", flush=True)
    if cache_hit:
        entries = pd.read_parquet(output / "entries.parquet")
        schedule = pd.read_parquet(output / "schedule.parquet")
        issues = pd.read_csv(output / "issues.csv", parse_dates=["date"], keep_default_na=False)
        first_files = int(prior["entry_snapshot_files"])
        print(f"Pass 1 cache: {len(entries):,} selected entry records", flush=True)
        if prior.get("stage") == "complete" and (output / "daily_quotes.parquet").exists():
            quotes = pd.read_parquet(output / "daily_quotes.parquet")
            print(f"Pass 2 cache: {len(quotes):,} observed daily quotes", flush=True)
            return dict(entries=entries, schedule=schedule, daily_quotes=quotes, cash=cash, issues=issues, manifest=prior)
    else:
        previous_settings = prior.get("settings", {})
        compatible_settings = ("start", "end", "targets", "ratios", "cash_source", "archive_root", "cash_sha256", "short_expiration_policy")
        reuse_anchored = (not force and previous_settings.get("format_version") == 1
                          and previous_settings.get("weekly_expiration_policy") == "anchored_exact"
                          and prior.get("source_index_sha256") == archive_index_hash
                          and all(json.dumps(previous_settings.get(key), sort_keys=True) == json.dumps(settings.get(key), sort_keys=True) for key in compatible_settings)
                          and all((output / name).exists() for name in ("entries.parquet", "schedule.parquet", "issues.csv")))
        if reuse_anchored and renewal_targets:
            print(f"Reusing unchanged tenor selections; extracting supplemental renewal targets {renewal_targets}.", flush=True)
            old_entries = pd.read_parquet(output / "entries.parquet")
            old_schedule = pd.read_parquet(output / "schedule.parquet")
            old_issues = pd.read_csv(output / "issues.csv", parse_dates=["date"])
            supplemental_entries, supplemental_schedule, supplemental_issues, supplemental_files = build_entries(
                archive, full_cash, start, end, renewal_targets, ratios, renewal_targets)
            entries = pd.concat([old_entries.loc[~old_entries.target_dte.isin(renewal_targets)], supplemental_entries], ignore_index=True)
            schedule = pd.concat([old_schedule.loc[~old_schedule.target_dte.isin(renewal_targets)], supplemental_schedule], ignore_index=True)
            old_issues = old_issues.loc[~old_issues.target_dte.isin(renewal_targets) & ~old_issues.stage.isin(["daily", "entry_range"])]
            old_issues = old_issues.loc[old_issues.stage.ne("calendar") | old_issues.date.isin(schedule.entry_date)]
            issues = pd.concat([old_issues, supplemental_issues], ignore_index=True).fillna("").drop_duplicates()
            entries = entries.sort_values(["target_dte", "entry_date", "ratio"]).reset_index(drop=True)
            schedule = schedule.sort_values(["target_dte", "entry_date"]).reset_index(drop=True)
            first_files = int(schedule.entry_date.nunique())
            print(f"Read {supplemental_files} supplemental entry snapshots.", flush=True)
        else:
            entries, schedule, issues, first_files = build_entries(archive, full_cash, start, end, targets, ratios, renewal_targets)
        entries.to_parquet(output / "entries.parquet", index=False)
        schedule.to_parquet(output / "schedule.parquet", index=False)
    availability = _availability(schedule)
    print("Expiration availability: " + json.dumps(availability), flush=True)
    manifest = dict(settings=settings, input_sha256=fingerprint, stage="entries", source_cash_sessions=len(cash),
                    source_archive_sessions=sum(row["path"] is not None for row in source_index),
                    entry_snapshot_files=first_files, entries=len(entries), availability=availability,
                    interpolation=False, missing_quote_fill=False, invalid_quote_substitution=False,
                    final_open_positions="Observed midpoint on the final date; no new entry on the final date",
                    expiration_valuation="PM cash-index intrinsic; no expiration-date quote required",
                    source_index_sha256=archive_index_hash)
    issues = issues.loc[~issues.stage.isin(["daily", "entry_range"])].copy()
    issues.to_csv(output / "issues.csv", index=False, date_format="%Y-%m-%d")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    quotes = pd.DataFrame(columns=[*QUOTE_COLUMNS, "source_file"])
    if not entries_only:
        quotes, daily_issues, second_files, ranges = build_daily_quotes(archive, entries, cash.index, end, workers)
        entries, range_issues = add_strike_range_metadata(entries, ranges)
        issues = pd.concat([issues, daily_issues, range_issues], ignore_index=True)
        entries.to_parquet(output / "entries.parquet", index=False)
        quotes.to_parquet(output / "daily_quotes.parquet", index=False)
        issues.to_csv(output / "issues.csv", index=False, date_format="%Y-%m-%d")
        manifest.update(stage="complete", daily_snapshot_files=second_files, daily_quotes=len(quotes),
                        required_daily_quotes=sum(map(len, quote_requirements(entries, cash.index, end).values())),
                        issue_counts={str(k): int(v) for k, v in issues.reason.value_counts().items()})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Extracted {len(entries):,} entry records and {len(quotes):,} raw daily quotes; {len(daily_issues):,} daily quote issues", flush=True)
    return dict(entries=entries, schedule=schedule, daily_quotes=quotes, cash=cash, issues=issues, manifest=manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--entries-only", action="store_true")
    args = parser.parse_args()
    build_extraction(output=args.output, workers=args.workers, force=args.force, entries_only=args.entries_only)


if __name__ == "__main__":
    main()
