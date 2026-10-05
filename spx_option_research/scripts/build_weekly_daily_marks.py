"""Mark every weekly candidate daily in units of its entry SPX cash notional."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DATA_ROOT = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
DEFAULT_OUTPUT = PROJECT / "results/weekly_dynamic_research"
sys.path.insert(0, str(PROJECT / "src"))
from spxresearch.data_loader import SPXSurfaceArchive  # noqa: E402

QUOTE_COLUMNS = ["snapshot_date", "expiration_date", "option_symbol", "strike", "bid", "ask"]
FILLS = ("mid", "realistic", "natural")


def _prepare(candidates: pd.DataFrame, cash_closes: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    required = {"candidate_id", "entry_date", "expiration_date", "spot_entry", "short_symbol", "long_symbol", "short_strike", "long_strike"}
    required.update(f"{measure}_{fill}_pct_spot_notional" for measure in ("premium", "pnl") for fill in FILLS)
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate columns missing: {sorted(missing)}")
    if candidates.empty:
        raise ValueError("No candidate trades")
    data = candidates.copy()
    if data.candidate_id.isna().any() or data.candidate_id.duplicated().any():
        raise ValueError("candidate_id must be nonmissing and unique")
    for column in ("entry_date", "expiration_date"):
        data[column] = pd.to_datetime(data[column]).dt.normalize()
    if data.entry_date.isna().any() or data.expiration_date.isna().any() or data.expiration_date.le(data.entry_date).any():
        raise ValueError("Invalid candidate dates")
    for column in ("short_symbol", "long_symbol"):
        if data[column].isna().any():
            raise ValueError("Missing candidate symbol")
        data[column] = data[column].astype(str).str.strip()
    numeric = ["spot_entry", "short_strike", "long_strike", *[f"{measure}_{fill}_pct_spot_notional" for measure in ("premium", "pnl") for fill in FILLS]]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite candidate economics")
    if data.spot_entry.le(0).any() or data.long_strike.le(0).any() or data.short_strike.le(data.long_strike).any():
        raise ValueError("Invalid candidate spot or strike order")
    cash = cash_closes.astype(float).copy()
    cash.index = pd.DatetimeIndex(cash.index).normalize()
    if cash.index.has_duplicates:
        raise ValueError("Cash-index dates must be unique")
    cash = cash.sort_index()
    if not np.isfinite(cash).all() or cash.le(0).any():
        raise ValueError("Invalid cash-index closes")
    for column in ("entry_date", "expiration_date"):
        if not data[column].isin(cash.index).all():
            raise ValueError(f"Missing cash-index {column} session")
    data.sort_values(["entry_date", "candidate_id"], inplace=True)
    return data.reset_index(drop=True), cash


def quote_requirements(candidates: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[pd.Timestamp, set[str]]:
    """Exact symbols held from entry through the final pre-expiration session."""
    wanted: dict[pd.Timestamp, set[str]] = {}
    for (entry, expiry), group in candidates.groupby(["entry_date", "expiration_date"], sort=True):
        symbols = set(group.short_symbol) | set(group.long_symbol)
        for day in dates[(dates >= entry) & (dates < expiry)]:
            wanted.setdefault(day, set()).update(symbols)
    return dict(sorted(wanted.items()))


def _quote_issues(quotes: pd.DataFrame, wanted: dict[pd.Timestamp, set[str]]) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for day, group in quotes.groupby("snapshot_date", sort=False):
        if group.option_symbol.duplicated().any():
            issues.append({"date": day, "option_symbol": "", "reason": "duplicate selected symbol"})
    seen = set(zip(quotes.snapshot_date, quotes.option_symbol))
    for day, symbols in wanted.items():
        for symbol in sorted(symbols):
            if (day, symbol) not in seen:
                issues.append({"date": day, "option_symbol": symbol, "reason": "required quote absent"})
    bad = (~np.isfinite(quotes[["bid", "ask", "strike"]].to_numpy(dtype=float)).all(axis=1)
           | quotes.bid.lt(0) | quotes.ask.lt(quotes.bid) | quotes.strike.le(0))
    for row in quotes.loc[bad].itertuples(index=False):
        issues.append({"date": row.snapshot_date, "option_symbol": row.option_symbol, "reason": "invalid bid/ask/strike"})
    return issues


def load_required_quotes(candidates: pd.DataFrame, cash: pd.Series, output: Path, archive: SPXSurfaceArchive) -> pd.DataFrame:
    """Read minimal columns, cache by required symbols and archive file index."""
    wanted = quote_requirements(candidates, cash.index)
    source_index = []
    missing_dates = []
    for day, symbols in wanted.items():
        if day not in archive.paths:
            missing_dates.append({"date": day, "option_symbol": "", "reason": "archive date missing"})
            continue
        path = archive.path_for(day)
        stat = path.stat()
        source_index.append({"date": str(day.date()), "path": str(path.relative_to(archive.root)),
                             "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "symbols": sorted(symbols)})
    issue_path = output / "weekly_daily_quote_issues.csv"
    if missing_dates:
        pd.DataFrame(missing_dates).to_csv(issue_path, index=False)
        raise LookupError(f"Missing archive sessions; see {issue_path}")
    source_hash = hashlib.sha256(json.dumps(source_index, sort_keys=True).encode()).hexdigest()
    quote_path = output / "weekly_daily_option_quotes.parquet"
    manifest_path = output / "weekly_daily_quotes_manifest.json"
    cache_hit = False
    if quote_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_hit = manifest.get("source_index_sha256") == source_hash
    if cache_hit:
        quotes = pd.read_parquet(quote_path)
    else:
        def read_one(item: tuple[pd.Timestamp, set[str]]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
            day, symbols = item
            path = archive.path_for(day)
            frame = pq.ParquetFile(path).read(columns=QUOTE_COLUMNS).to_pandas()
            frame["option_symbol"] = frame.option_symbol.astype(str).str.strip()
            frame = frame.loc[frame.option_symbol.isin(symbols)].copy()
            frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date).dt.normalize()
            frame["expiration_date"] = pd.to_datetime(frame.expiration_date).dt.normalize()
            errors = [{"date": day, "option_symbol": row.option_symbol, "reason": "snapshot date disagrees with file date"}
                      for row in frame.loc[frame.snapshot_date.ne(day)].itertuples(index=False)]
            frame["source_file"] = str(path.relative_to(archive.root))
            return frame, errors

        pieces, read_errors = [], []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for number, (frame, errors) in enumerate(pool.map(read_one, wanted.items()), start=1):
                pieces.append(frame)
                read_errors.extend(errors)
                if number % 400 == 0:
                    print(f"Read {number:,}/{len(wanted):,} daily option files", flush=True)
        quotes = pd.concat(pieces, ignore_index=True)
        if read_errors:
            pd.DataFrame(read_errors).to_csv(issue_path, index=False)
            raise ValueError(f"Invalid source snapshot dates; see {issue_path}")
    quotes["snapshot_date"] = pd.to_datetime(quotes.snapshot_date).dt.normalize()
    quotes["expiration_date"] = pd.to_datetime(quotes.expiration_date).dt.normalize()
    quotes["option_symbol"] = quotes.option_symbol.astype(str).str.strip()
    issues = _quote_issues(quotes, wanted)
    pd.DataFrame(issues, columns=["date", "option_symbol", "reason"]).to_csv(issue_path, index=False)
    if issues:
        raise LookupError(f"{len(issues)} missing or invalid quote issues; see {issue_path}")
    if len(quotes) != sum(map(len, wanted.values())):
        raise ValueError("Cached quote count differs from required exact date/symbol pairs")
    quotes["mid"] = (quotes.bid + quotes.ask) / 2.0
    quotes.sort_values(["snapshot_date", "option_symbol"], inplace=True)
    quotes.reset_index(drop=True, inplace=True)
    if not cache_hit:
        quotes.to_parquet(quote_path, index=False)
        manifest_path.write_text(json.dumps({"source_root": str(archive.root), "source_index_sha256": source_hash,
                                            "source_files": len(source_index), "quotes": len(quotes),
                                            "cache_key": "Required dates/symbols, relative archive paths, file sizes and nanosecond modification times",
                                            "fill_missing_quotes": False}, indent=2), encoding="utf-8")
    print(f"Validated {len(quotes):,} exact daily quotes ({'cache' if cache_hit else 'archive'}); no missing or invalid quotes", flush=True)
    return quotes


def _close(actual: float, expected: float, description: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{description}: {actual} versus {expected}")


def unit_marks_from_quotes(candidates: pd.DataFrame, quotes: pd.DataFrame, cash_closes: pd.Series) -> pd.DataFrame:
    """Pure ledger: entry premium less current liability, with expiry intrinsic.

    Returns are cumulative P&L divided by cash SPX at entry, not returns on
    margin, and not daily portfolio percentage returns.  Entry friction is
    recognized once.  No option quotes are needed on expiration itself.
    """
    data, cash = _prepare(candidates, cash_closes)
    raw = quotes.copy()
    if not set(QUOTE_COLUMNS).issubset(raw.columns):
        raise ValueError("Quote columns missing")
    raw["snapshot_date"] = pd.to_datetime(raw.snapshot_date).dt.normalize()
    raw["expiration_date"] = pd.to_datetime(raw.expiration_date).dt.normalize()
    raw["option_symbol"] = raw.option_symbol.astype(str).str.strip()
    if raw.duplicated(["snapshot_date", "option_symbol"]).any():
        raise ValueError("Daily option date/symbol must be unique")
    issues = _quote_issues(raw, quote_requirements(data, cash.index))
    if issues:
        raise LookupError(f"Missing or invalid required quotes: {issues[:3]}")
    lookup = {(row.snapshot_date, row.option_symbol): row for row in raw.itertuples(index=False)}
    rows = []
    for candidate in data.itertuples(index=False):
        spot = float(candidate.spot_entry)
        _close(spot, cash.loc[candidate.entry_date], "Candidate entry cash spot")
        days = cash.index[(cash.index >= candidate.entry_date) & (cash.index <= candidate.expiration_date)]
        width = float(candidate.short_strike - candidate.long_strike)
        for day in days:
            expiry = day == candidate.expiration_date
            if expiry:
                liability = max(candidate.short_strike - cash.loc[day], 0.0) - max(candidate.long_strike - cash.loc[day], 0.0)
            else:
                short = lookup[(day, candidate.short_symbol)]
                long = lookup[(day, candidate.long_symbol)]
                for leg, strike in ((short, candidate.short_strike), (long, candidate.long_strike)):
                    _close(float(leg.strike), float(strike), "Held leg strike")
                    if leg.expiration_date != candidate.expiration_date:
                        raise ValueError("Held leg expiration does not match quote")
                liability = (short.bid + short.ask - long.bid - long.ask) / 2.0
            result = {"candidate_id": candidate.candidate_id, "date": day,
                      "entry_date": candidate.entry_date, "expiration_date": candidate.expiration_date,
                      "spread_mid_points": float(liability), "width_points": width,
                      "bounded_spread_mid_points": float(np.clip(liability, 0.0, width)),
                      "mid_outside_payoff_bounds": bool(not expiry and (liability < -0.01 or liability > width + 0.01)),
                      "is_expiration": expiry}
            for fill in FILLS:
                premium = float(getattr(candidate, f"premium_{fill}_pct_spot_notional"))
                result[f"cum_return_{fill}"] = premium - float(liability) / spot
                if expiry:
                    expected = float(getattr(candidate, f"pnl_{fill}_pct_spot_notional"))
                    _close(result[f"cum_return_{fill}"], expected, f"{candidate.candidate_id} {fill} settlement")
            if day == candidate.entry_date:
                _close(result["cum_return_mid"], 0.0, "Entry premium offset by midpoint liability")
                if result["cum_return_realistic"] > 1e-10 or result["cum_return_natural"] > 1e-10:
                    raise ValueError("Entry friction cannot create a positive midpoint mark")
            rows.append(result)
    marks = pd.DataFrame(rows).sort_values(["candidate_id", "date"]).reset_index(drop=True)
    if marks.duplicated(["candidate_id", "date"]).any():
        raise ValueError("Duplicate candidate date unit marks")
    return marks


def build_unit_marks(candidates: pd.DataFrame, cash_closes: pd.Series, output: str | Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data, cash = _prepare(candidates, cash_closes)
    archive = SPXSurfaceArchive(DATA_ROOT, cache_size=1)
    quotes = load_required_quotes(data, cash, output, archive)
    marks = unit_marks_from_quotes(data, quotes, cash)
    marks.to_parquet(output / "unit_marks.parquet", index=False)
    marks.to_csv(output / "unit_marks.csv", index=False)
    flags = marks.loc[marks.mid_outside_payoff_bounds].copy()
    flags.to_csv(output / "weekly_midpoint_bound_flags.csv", index=False)
    settlements = marks.loc[marks.is_expiration, ["candidate_id", "date", *[f"cum_return_{fill}" for fill in FILLS]]].copy()
    expected = data[["candidate_id", *[f"pnl_{fill}_pct_spot_notional" for fill in FILLS]]]
    settlements = settlements.merge(expected, on="candidate_id", how="left", validate="one_to_one")
    for fill in FILLS:
        settlements[f"reconciliation_error_{fill}"] = settlements[f"cum_return_{fill}"] - settlements[f"pnl_{fill}_pct_spot_notional"]
    settlements.to_csv(output / "weekly_unit_settlement_reconciliation.csv", index=False)
    candidate_hash = hashlib.sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes()).hexdigest()
    cash_hash = hashlib.sha256(pd.util.hash_pandas_object(cash, index=True).values.tobytes()).hexdigest()
    manifest = {"candidate_count": len(data), "candidate_rows_sha256": candidate_hash, "cash_rows_sha256": cash_hash,
                "unit_mark_rows": len(marks), "quote_rows": len(quotes), "settlements_reconciled": len(settlements),
                "midpoint_bound_flags": len(flags), "first_entry": str(data.entry_date.min().date()),
                "last_expiration": str(data.expiration_date.max().date()),
                "formula": "cum_return_fill = entry premium per SPX cash notional - spread midpoint liability / entry SPX cash",
                "expiry": "cash SPX intrinsic spread liability, no expiration quote required",
                "costs": "Entry transaction costs once, carried in premium; no additional daily costs",
                "missing_marks": "Fail; never forward-fill", "primary_marks": "Raw actual bid/ask midpoints, without clipping"}
    (output / "unit_marks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(marks):,} unit marks; reconciled all {len(settlements):,} expirations; {len(flags)} midpoint bound flags", flush=True)
    return marks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    candidates = pd.read_parquet(args.output / "candidate_trades.parquet")
    cash = pd.read_csv(args.output / "market_spx.csv", parse_dates=["date"]).set_index("date").close
    build_unit_marks(candidates, cash, args.output)


if __name__ == "__main__":
    main()
