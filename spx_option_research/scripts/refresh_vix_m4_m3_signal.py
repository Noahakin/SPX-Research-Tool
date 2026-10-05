from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

import analyze_vix_curve_next_day_spx as curve_source


OUT = curve_source.PROJECT / "results/vix_m4_m3_mean_reversion_strategy"
EXTENSION = OUT / "official_monthly_vx_extension.csv"
LATEST_SIGNAL = OUT / "latest_signal.csv"
API = "https://www-api.cboe.com/us/futures/market_statistics/settlement/csv"
ENTRY_Z = 1.25
ROLLING_WINDOW = 252
MIN_HISTORY = 126


def download_monthly_settlements(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    for date in pd.date_range(start, end, freq="B"):
        response = session.get(API, params={"dt": f"{date:%Y-%m-%d}"}, timeout=20)
        if response.status_code != 200 or not response.text.strip():
            continue
        try:
            frame = pd.read_csv(StringIO(response.text))
        except pd.errors.EmptyDataError:
            continue
        required = {"Product", "Symbol", "Expiration Date", "Price"}
        if not required.issubset(frame.columns):
            continue
        monthly = frame[
            frame["Product"].eq("VX")
            & frame["Symbol"].astype(str).str.match(r"^VX/[A-Z][0-9]$")
        ].copy()
        if monthly.empty:
            continue
        monthly["date"] = date
        monthly["expiration_date"] = pd.to_datetime(monthly["Expiration Date"])
        monthly["future"] = pd.to_numeric(monthly["Price"], errors="coerce")
        rows.append(monthly[["date", "expiration_date", "future", "Symbol"]])
    if not rows:
        return pd.DataFrame(columns=["date", "expiration_date", "future", "Symbol"])
    return pd.concat(rows, ignore_index=True)


def build_curve(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[
        raw["expiration_date"].gt(raw["date"])
        & raw["future"].notna()
        & raw["future"].gt(0)
    ].copy()
    raw = raw.sort_values(["date", "expiration_date"]).drop_duplicates(
        ["date", "expiration_date"], keep="last"
    )
    raw["term"] = raw.groupby("date").cumcount() + 1
    selected = raw[raw["term"].isin([3, 4])]
    prices = selected.pivot(index="date", columns="term", values="future")
    expirations = selected.pivot(index="date", columns="term", values="expiration_date")
    curve = pd.DataFrame(index=prices.index)
    curve["m3"] = prices[3]
    curve["m4"] = prices[4]
    curve["m3_expiration"] = expirations[3]
    curve["m4_expiration"] = expirations[4]
    curve = curve.dropna().sort_index()
    curve["spread"] = curve["m4"] - curve["m3"]
    curve["rolling_mean"] = (
        curve["spread"].rolling(ROLLING_WINDOW, min_periods=MIN_HISTORY).mean().shift(1)
    )
    curve["rolling_std"] = (
        curve["spread"].rolling(ROLLING_WINDOW, min_periods=MIN_HISTORY).std().shift(1)
    )
    curve["z_score"] = (
        (curve["spread"] - curve["rolling_mean"]) / curve["rolling_std"]
    )
    return curve.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    historical = pd.read_csv(
        curve_source.FUTURES_SOURCE, parse_dates=["date", "expiration_date"]
    )[["date", "expiration_date", "future"]]
    start = historical["date"].max() + pd.Timedelta(days=1)
    end = pd.Timestamp.today().normalize()
    extension = download_monthly_settlements(start, end)
    extension.to_csv(EXTENSION, index=False)

    combined = pd.concat(
        [historical, extension[["date", "expiration_date", "future"]]],
        ignore_index=True,
    )
    curve = build_curve(combined)
    latest = curve.iloc[-1].copy()
    lower_entry = latest["rolling_mean"] - ENTRY_Z * latest["rolling_std"]
    if latest["z_score"] <= -ENTRY_Z:
        signal = "ENTER NEXT SETTLEMENT: BUY M4 / SELL M3"
    else:
        signal = "NO NEW TRADE"
    output = pd.DataFrame(
        [
            {
                "as_of": latest["date"],
                "m3_expiration": latest["m3_expiration"],
                "m3_settlement": latest["m3"],
                "m4_expiration": latest["m4_expiration"],
                "m4_settlement": latest["m4"],
                "m4_minus_m3": latest["spread"],
                "rolling_mean": latest["rolling_mean"],
                "rolling_std": latest["rolling_std"],
                "lower_entry_level": lower_entry,
                "z_score": latest["z_score"],
                "signal": signal,
                "source": "Cboe daily settlement API",
            }
        ]
    )
    output.to_csv(LATEST_SIGNAL, index=False)
    print(output.to_string(index=False))
    print(f"Saved {len(extension):,} extension rows to {EXTENSION}")


if __name__ == "__main__":
    main()
