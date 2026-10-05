from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "snapshot_date",
    "underlying_price",
    "expiration_date",
    "dte",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "option_symbol",
}

RESEARCH_COLUMNS = sorted(REQUIRED_COLUMNS | {"last", "is_settlement"})


def date_from_path(path: Path) -> pd.Timestamp:
    match = re.search(r"date=(\d{4}-\d{2}-\d{2})", str(path))
    if not match:
        raise ValueError(f"No date partition in {path}")
    return pd.Timestamp(match.group(1)).normalize()


class SPXSurfaceArchive:
    """Indexed, cached access to one EOD SPX chain per trading date."""

    def __init__(self, root: str | Path, cache_size: int = 12) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        files = sorted(self.root.rglob("eod.parquet"))
        self.paths = {date_from_path(path): path for path in files}
        if not self.paths:
            raise ValueError(f"No EOD parquet files under {self.root}")
        self._read_cached = lru_cache(maxsize=cache_size)(self._read_uncached)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.paths))

    @property
    def populated_dates(self) -> pd.DatetimeIndex:
        """Dates with at least one quote row (weekday holiday files are empty)."""
        return pd.DatetimeIndex(
            sorted(
                date
                for date, path in self.paths.items()
                if pq.ParquetFile(path).metadata.num_rows > 0
            )
        )

    def path_for(self, value: pd.Timestamp | str) -> Path:
        key = pd.Timestamp(value).normalize()
        return self.paths[key]

    def read(self, value: pd.Timestamp | str) -> pd.DataFrame:
        return self._read_cached(pd.Timestamp(value).normalize()).copy(deep=False)

    def _read_uncached(self, value: pd.Timestamp) -> pd.DataFrame:
        path = self.paths[value]
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        missing = REQUIRED_COLUMNS - schema
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        columns = [column for column in RESEARCH_COLUMNS if column in schema]
        frame = pq.read_table(path, columns=columns).to_pandas()
        if frame.empty:
            return frame
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
        frame["expiration_date"] = pd.to_datetime(frame["expiration_date"]).dt.normalize()
        frame["option_symbol"] = frame["option_symbol"].astype("string").str.strip()
        frame["option_type"] = frame["option_type"].astype("string").str.lower()
        frame["settlement"] = frame["option_symbol"].map(contract_settlement)
        return frame

    def quote_panel(self, symbols: Iterable[str]) -> pd.DataFrame:
        wanted = {str(symbol).strip() for symbol in symbols}
        records: list[pd.DataFrame] = []
        for day in self.dates:
            chain = self.read(day)
            if chain.empty:
                continue
            selected = chain[chain["option_symbol"].isin(wanted)]
            if not selected.empty:
                records.append(selected)
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    def spot_series(self) -> pd.Series:
        values: dict[pd.Timestamp, float] = {}
        for day in self.dates:
            chain = self.read(day)
            if not chain.empty:
                values[day] = float(chain["underlying_price"].dropna().median())
        return pd.Series(values, name="spx_spot").sort_index()


def contract_settlement(symbol: object) -> str:
    text = str(symbol).strip().upper()
    if text.startswith("SPXW"):
        return "PM"
    if text.startswith("SPX"):
        return "AM"
    return "UNKNOWN"
