from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests


FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def _load_env_value(name: str, workspace: Path) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    env_path = workspace / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    return value
    raise RuntimeError(f"{name} is unavailable")


def fred_series(
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    api_key: str,
) -> pd.Series:
    response = requests.get(
        FRED_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.date().isoformat(),
            "observation_end": end.date().isoformat(),
        },
        timeout=60,
    )
    response.raise_for_status()
    observations = response.json().get("observations", [])
    values = {
        pd.Timestamp(item["date"]): float(item["value"])
        for item in observations
        if item.get("value") not in {None, "."}
    }
    return pd.Series(values, name=series_id, dtype="float64").sort_index()


def build_market_data(
    spot: pd.Series,
    workspace: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    workspace = Path(workspace).resolve()
    key = _load_env_value("FRED_API_KEY", workspace)
    index = pd.DatetimeIndex(spot.index).normalize()
    start, end = index.min(), index.max()
    vix = fred_series("VIXCLS", start, end, api_key=key).reindex(index).ffill()
    rate = fred_series("DGS3MO", start, end, api_key=key).reindex(index).ffill()
    frame = pd.DataFrame(index=index)
    frame.index.name = "date"
    frame["spx_spot"] = spot.reindex(index)
    frame["spx_return"] = frame["spx_spot"].pct_change(fill_method=None).fillna(0.0)
    frame["vix"] = vix
    frame["risk_free_annual"] = rate / 100.0
    frame["risk_free_daily"] = (1.0 + frame["risk_free_annual"]) ** (1.0 / 252.0) - 1.0
    frame["realized_vol_21d"] = frame["spx_return"].rolling(21).std() * (252.0**0.5)
    frame["spx_drawdown"] = frame["spx_spot"] / frame["spx_spot"].cummax() - 1.0
    frame["spx_trend_200d"] = frame["spx_spot"] / frame["spx_spot"].rolling(200).mean() - 1.0
    frame["vix_percentile_252d"] = frame["vix"].rolling(252).rank(pct=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    return frame

