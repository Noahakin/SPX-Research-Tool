"""Cash-index features for the weekly option study, lagged one SPX session.

SPX prices are actual Yahoo cash-index closes.  Official Cboe daily index
histories supply VIX, VVIX and VIX3M.  The option archive's underlying_price is
never read here.  Run directly to materialize auditable market source files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DEFAULT_OUTPUT = PROJECT / "results/weekly_dynamic_research"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2026-09-22"
CASH_CACHE = PROJECT / "results/requested_premium_study/spx_cash_close.csv"
LOCAL_CASH_ROOT = (
    WORKSPACE
    / "05 - Underlying Prices and Events/Cleaned Data/Yahoo Finance/FX External/processed"
    / "provider=yfinance/group=equities/series=sp500"
)
CBOE_BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_source_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate source dates in {path}")
    if not np.isfinite(frame["close"]).all() or frame["close"].le(0).any():
        raise ValueError(f"Invalid source closes in {path}")
    return frame.sort_values("date").reset_index(drop=True)


def _official_index_source(name: str, output: Path, refresh: bool) -> pd.DataFrame:
    path = output / f"market_{name}.csv"
    if path.exists() and not refresh:
        return _read_source_file(path)
    url = f"{CBOE_BASE}/{name.upper()}_History.csv"
    audited = output / f"audit_cboe_{name}_history.csv"
    if audited.exists() and not refresh:
        raw = pd.read_csv(audited)
        retrieved_at = datetime.fromtimestamp(audited.stat().st_mtime, timezone.utc).isoformat()
    else:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        raw = pd.read_csv(io.StringIO(response.text))
        retrieved_at = _utc_now()
    value_column = "VVIX" if name == "vvix" else "CLOSE"
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["DATE"], format="%m/%d/%Y"),
            "close": pd.to_numeric(raw[value_column], errors="raise"),
            "source": url,
            "retrieved_at_utc": retrieved_at,
        }
    ).sort_values("date")
    if result.date.duplicated().any():
        raise ValueError(f"Duplicate dates in official {name} history")
    return result.reset_index(drop=True)


def _cash_source(output: Path, refresh: bool) -> pd.DataFrame:
    path = output / "market_spx.csv"
    if path.exists() and not refresh:
        return _read_source_file(path)
    pieces: list[pd.DataFrame] = []
    for source_path in sorted(LOCAL_CASH_ROOT.glob("year=*/*.parquet")):
        frame = pd.read_parquet(source_path)
        piece = frame[["observation_date", "close", "retrieved_at_utc"]].rename(
            columns={"observation_date": "date"}
        )
        piece["date"] = pd.to_datetime(piece["date"]).dt.normalize()
        # A quote retrieved on its own date before the NY cash close is not a
        # completed daily close.  Historical observations remain usable.
        retrieved = pd.to_datetime(piece["retrieved_at_utc"], utc=True)
        close_time = (piece["date"] + pd.Timedelta(hours=16)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        piece = piece.loc[retrieved.ge(close_time)].copy()
        piece["source"] = f"Yahoo ^GSPC local: {source_path.relative_to(WORKSPACE).as_posix()}"
        pieces.append(piece)
    if not CASH_CACHE.exists():
        raise FileNotFoundError(f"Corrected cash-index study cache is required: {CASH_CACHE}")
    cached = pd.read_csv(CASH_CACHE, parse_dates=["date"]).rename(columns={"spx_close": "close"})
    cached["source"] = "Yahoo ^GSPC: results/requested_premium_study/spx_cash_close.csv"
    cached["retrieved_at_utc"] = pd.NA  # Original download time was not retained by the earlier study.
    pieces.append(cached)
    result = pd.concat(pieces, ignore_index=True, sort=False)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["close"] = pd.to_numeric(result["close"], errors="raise")
    return result.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def calculate_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return features available before each entry date's option snapshot.

    ``prices`` must use the SPX trading-session calendar and columns spx, vix,
    vvix and vix3m.  No missing source observation is forward- or back-filled.
    Features are calculated first, then shifted by exactly one SPX session.
    All rolling standard deviations use ddof=1 and return annualized decimal
    volatility.  Momentum/change columns are fractional price changes.
    """
    required = {"spx", "vix", "vvix", "vix3m"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Missing market columns: {sorted(required - set(prices.columns))}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must use a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("Duplicate market-session dates")
    daily = prices.sort_index().copy()
    if daily.spx.isna().any() or daily.spx.le(0).any():
        raise ValueError("SPX calendar must contain only valid positive cash closes")
    returns = daily.spx.pct_change(fill_method=None)
    features = pd.DataFrame(index=daily.index)
    features["spx_close_lag1"] = daily.spx
    for window in (1, 5, 21, 63, 252):
        features[f"spx_momentum_{window}d"] = daily.spx.pct_change(window, fill_method=None)
    for window in (5, 21, 63):
        features[f"spx_rv_{window}d"] = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252.0)
    features["spx_drawdown_252d"] = daily.spx / daily.spx.rolling(252, min_periods=252).max() - 1.0
    for name in ("vix", "vvix", "vix3m"):
        series = daily[name].astype(float)
        features[name] = series
        for window in (5, 21):
            features[f"{name}_change_{window}d"] = series.pct_change(window, fill_method=None)
        prior = series.shift(1)
        mean = prior.rolling(252, min_periods=252).mean()
        stdev = prior.rolling(252, min_periods=252).std(ddof=1)
        features[f"{name}_zscore_252d"] = (series - mean) / stdev.where(stdev.gt(1e-12))
    features["vix_to_vix3m"] = daily.vix / daily.vix3m
    features["vvix_to_vix"] = daily.vvix / daily.vix
    features["vix_to_spx_rv_21d"] = (daily.vix / 100.0) / features.spx_rv_21d.where(features.spx_rv_21d.gt(1e-12))
    lagged = features.shift(1)
    lagged["market_source_date"] = pd.Series(daily.index, index=daily.index).shift(1)
    lagged.index.name = "date"
    return lagged


def load_market_features(
    output: str | Path = DEFAULT_OUTPUT,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp = DEFAULT_END,
    refresh: bool = False,
) -> pd.DataFrame:
    """Materialize source CSVs and the primary lagged ``market_data.parquet``."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if start > end:
        raise ValueError("Market data start is after end")
    sources: dict[str, pd.DataFrame] = {"spx": _cash_source(output, refresh)}
    sources.update({name: _official_index_source(name, output, refresh) for name in ("vix", "vvix", "vix3m")})
    coverage: dict[str, dict[str, object]] = {}
    series = {}
    for name, frame in sources.items():
        frame = frame.loc[frame.date.between(start, end)].copy()
        if frame.empty:
            raise ValueError(f"No {name} observations inside requested period")
        if not np.isfinite(frame.close.astype(float)).all() or frame.close.le(0).any():
            raise ValueError(f"Invalid {name} source closes")
        path = output / f"market_{name}.csv"
        frame.to_csv(path, index=False)
        series[name] = frame.set_index("date").close.astype(float)
        coverage[name] = {
            "first_date": str(frame.date.min().date()), "last_date": str(frame.date.max().date()),
            "observations": int(len(frame)), "sources": sorted(frame.source.unique().tolist()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    prices = pd.DataFrame(index=series["spx"].index)
    for name, values in series.items():
        prices[name] = values.reindex(prices.index)
        missing = prices.index[prices[name].isna()]
        coverage[name]["missing_on_spx_calendar"] = int(len(missing))
        coverage[name]["missing_dates"] = [str(date.date()) for date in missing]
    prices.index.name = "date"
    features = calculate_market_features(prices)
    features.to_parquet(output / "market_data.parquet")
    features.to_csv(output / "market_data.csv")
    numeric = features.select_dtypes(include="number")
    study_rows = features.loc["2016-09-22":]
    manifest = {
        "generated_at_utc": _utc_now(),
        "start": str(start.date()), "end": str(end.date()),
        "sources": coverage,
        "feature_timing": "All features shifted one SPX trading session after alignment; market_source_date records the latest source close.",
        "missing_policy": "No forward-fill or backfill; rolling windows require all observations.",
        "rv_definition": "Sample standard deviation of simple daily cash SPX returns times sqrt(252).",
        "zscore_definition": "Level minus preceding 252-session mean, divided by preceding 252-session sample std; result then lagged one session.",
        "cash_cache_download_time": "Earlier corrected Yahoo cache did not retain its source retrieval timestamp; its hash is recorded.",
        "cash_cache_sha256": hashlib.sha256(CASH_CACHE.read_bytes()).hexdigest(),
        "feature_columns": numeric.columns.tolist(),
        "missing_by_feature_entire_file": features.isna().sum().astype(int).to_dict(),
        "missing_by_feature_study_period": study_rows.isna().sum().astype(int).to_dict(),
    }
    (output / "market_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    frame = load_market_features(args.output, args.start, args.end, args.refresh)
    print(f"Saved {len(frame):,} daily rows, {frame.index.min().date()} through {frame.index.max().date()}")
    print(f"Features: {len(frame.columns) - 1}; all shifted one SPX trading session")
    print("Missing values during option study:", int(frame.loc["2016-09-22":].isna().sum().sum()))


if __name__ == "__main__":
    main()
