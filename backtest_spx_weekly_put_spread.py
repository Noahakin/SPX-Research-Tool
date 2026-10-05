from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA_ROOT = (
    ROOT
    / "04 - Options and Volatility"
    / "Raw Downloads"
    / "IVolatility"
    / "data"
    / "raw"
    / "spx_options"
)
OUT = ROOT / "spx_weekly_put_spread"
MULTIPLIER = 100.0

ENTRY_COLUMNS = [
    "snapshot_date",
    "underlying_price",
    "expiration_date",
    "dte",
    "strike",
    "option_type",
    "bid",
    "ask",
    "delta",
    "option_symbol",
]


def date_from_path(path: Path) -> pd.Timestamp:
    match = re.search(r"date=(\d{4}-\d{2}-\d{2})", str(path))
    if not match:
        raise ValueError(f"No date partition in {path}")
    return pd.Timestamp(match.group(1))


def read_chain(path: Path) -> pd.DataFrame:
    return pq.read_table(path, columns=ENTRY_COLUMNS).to_pandas()


def clean_weekly_puts(chain: pd.DataFrame, expiration: pd.Timestamp, dte: int) -> pd.DataFrame:
    symbol = chain["option_symbol"].fillna("").astype(str).str.strip()
    expiration_col = pd.to_datetime(chain["expiration_date"])
    mask = (
        chain["option_type"].astype(str).str.lower().eq("put")
        & chain["dte"].eq(dte)
        & expiration_col.eq(expiration)
        & symbol.str.startswith("SPXW")
        & chain["strike"].gt(0)
        & chain["bid"].ge(0)
        & chain["ask"].gt(0)
        & chain["ask"].ge(chain["bid"])
    )
    return chain.loc[mask].copy()


def nearest_strike(chain: pd.DataFrame, target: float) -> pd.Series:
    idx = (chain["strike"] - target).abs().idxmin()
    return chain.loc[idx]


def max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.cumsum()
    return float((equity - equity.cummax()).min()) if len(equity) else math.nan


def summarize(trades: pd.DataFrame, pnl_col: str, return_col: str) -> dict[str, float | int]:
    pnl = trades[pnl_col].astype(float)
    returns = trades[return_col].astype(float)
    std = returns.std(ddof=1)
    sharpe = math.sqrt(52) * returns.mean() / std if std and np.isfinite(std) else math.nan
    losses = pnl[pnl <= pnl.quantile(0.05)]
    return {
        "trades": int(len(trades)),
        "total_pnl_dollars": float(pnl.sum()),
        "mean_pnl_dollars": float(pnl.mean()),
        "median_pnl_dollars": float(pnl.median()),
        "win_rate_pct": float(100 * (pnl > 0).mean()),
        "weekly_return_mean_pct": float(100 * returns.mean()),
        "annualized_sharpe": float(sharpe),
        "worst_trade_dollars": float(pnl.min()),
        "five_pct_trade_dollars": float(pnl.quantile(0.05)),
        "expected_shortfall_5pct_dollars": float(losses.mean()),
        "max_drawdown_dollars": max_drawdown(pnl),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_ROOT.rglob("eod.parquet"))
    by_date = {date_from_path(path): path for path in files if path.stat().st_size > 10_000}

    # A weekly trade is entered at Friday's EOD and expires the following Friday.
    # Exact 7-DTE keeps the test faithful to the requested holding period. Weeks
    # disrupted by Friday holidays or missing archive dates are skipped.
    entry_dates = [
        date
        for date in sorted(by_date)
        if date.weekday() == 4 and date + pd.Timedelta(days=7) in by_date
    ]

    records: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for number, entry_date in enumerate(entry_dates, start=1):
        expiration = entry_date + pd.Timedelta(days=7)
        one_dte_date = expiration - pd.Timedelta(days=1)
        if one_dte_date not in by_date:
            skips.append({"entry_date": entry_date, "reason": "missing_1dte_file"})
            continue

        entry_chain = read_chain(by_date[entry_date])
        entry_puts = clean_weekly_puts(entry_chain, expiration, 7)
        if entry_puts.empty:
            skips.append({"entry_date": entry_date, "reason": "no_7dte_spxw_puts"})
            continue

        spot_entry = float(entry_puts["underlying_price"].dropna().median())
        short = nearest_strike(entry_puts, spot_entry * 0.99)
        long = nearest_strike(entry_puts, spot_entry * 0.97)
        if float(short["strike"]) <= float(long["strike"]):
            skips.append({"entry_date": entry_date, "reason": "invalid_strike_order"})
            continue

        entry_credit = float(short["bid"] - long["ask"])
        width = float(short["strike"] - long["strike"])
        max_risk_dollars = (width - entry_credit) * MULTIPLIER
        if entry_credit <= 0 or max_risk_dollars <= 0:
            skips.append({"entry_date": entry_date, "reason": "nonpositive_credit_or_risk"})
            continue

        one_dte_chain = read_chain(by_date[one_dte_date])
        one_dte_puts = clean_weekly_puts(one_dte_chain, expiration, 1)
        long_1dte = one_dte_puts[np.isclose(one_dte_puts["strike"], float(long["strike"]))]
        if long_1dte.empty:
            skips.append({"entry_date": entry_date, "reason": "long_missing_at_1dte"})
            continue
        long_1dte = long_1dte.iloc[0]
        spot_1dte = float(long_1dte["underlying_price"])
        long_itm_1dte = spot_1dte < float(long["strike"])

        expiration_chain = read_chain(by_date[expiration])
        expiration_puts = clean_weekly_puts(expiration_chain, expiration, 0)
        short_exp = expiration_puts[np.isclose(expiration_puts["strike"], float(short["strike"]))]
        long_exp = expiration_puts[np.isclose(expiration_puts["strike"], float(long["strike"]))]
        if short_exp.empty or long_exp.empty:
            skips.append({"entry_date": entry_date, "reason": "legs_missing_at_expiration"})
            continue

        spot_expiration = float(short_exp.iloc[0]["underlying_price"])
        short_intrinsic = max(float(short["strike"]) - spot_expiration, 0.0)
        long_intrinsic = max(float(long["strike"]) - spot_expiration, 0.0)

        hold_pnl = (entry_credit - short_intrinsic + long_intrinsic) * MULTIPLIER
        early_long_sale = float(long_1dte["bid"]) if long_itm_1dte else math.nan
        early_pnl = (
            (entry_credit + early_long_sale - short_intrinsic) * MULTIPLIER
            if long_itm_1dte
            else hold_pnl
        )

        records.append(
            {
                "entry_date": entry_date,
                "one_dte_date": one_dte_date,
                "expiration_date": expiration,
                "spot_entry": spot_entry,
                "short_strike": float(short["strike"]),
                "long_strike": float(long["strike"]),
                "short_otm_pct_actual": 100 * (1 - float(short["strike"]) / spot_entry),
                "long_otm_pct_actual": 100 * (1 - float(long["strike"]) / spot_entry),
                "short_entry_bid": float(short["bid"]),
                "long_entry_ask": float(long["ask"]),
                "entry_credit_points": entry_credit,
                "spread_width_points": width,
                "max_risk_dollars": max_risk_dollars,
                "short_entry_delta": float(short["delta"]) if pd.notna(short["delta"]) else math.nan,
                "long_entry_delta": float(long["delta"]) if pd.notna(long["delta"]) else math.nan,
                "spot_1dte": spot_1dte,
                "long_itm_1dte": bool(long_itm_1dte),
                "long_1dte_bid": float(long_1dte["bid"]),
                "spot_expiration": spot_expiration,
                "short_intrinsic_expiration": short_intrinsic,
                "long_intrinsic_expiration": long_intrinsic,
                "hold_both_pnl_dollars": hold_pnl,
                "sell_long_1dte_pnl_dollars": early_pnl,
                "pnl_difference_dollars": early_pnl - hold_pnl,
                "hold_both_return_on_risk": hold_pnl / max_risk_dollars,
                "sell_long_1dte_return_on_risk": early_pnl / max_risk_dollars,
            }
        )
        if number % 25 == 0:
            print(f"Processed {number}/{len(entry_dates)} candidate weeks", flush=True)

    trades = pd.DataFrame(records).sort_values("entry_date").reset_index(drop=True)
    skip_df = pd.DataFrame(skips)
    if trades.empty:
        raise RuntimeError("No valid weekly trades were constructed")

    summary = pd.DataFrame(
        {
            "hold_both_to_expiration": summarize(
                trades, "hold_both_pnl_dollars", "hold_both_return_on_risk"
            ),
            "sell_long_if_itm_at_1dte": summarize(
                trades, "sell_long_1dte_pnl_dollars", "sell_long_1dte_return_on_risk"
            ),
        }
    ).T
    triggered = trades[trades["long_itm_1dte"]]
    comparison = {
        "archive_start": str(min(by_date).date()),
        "archive_end": str(max(by_date).date()),
        "first_trade": str(trades["entry_date"].min().date()),
        "last_trade": str(trades["entry_date"].max().date()),
        "valid_trades": int(len(trades)),
        "skipped_candidate_weeks": int(len(skip_df)),
        "early_exit_triggers": int(len(triggered)),
        "early_exit_trigger_rate_pct": float(100 * len(triggered) / len(trades)),
        "early_exit_total_pnl_difference_dollars": float(trades["pnl_difference_dollars"].sum()),
        "early_exit_mean_difference_when_triggered_dollars": (
            float(triggered["pnl_difference_dollars"].mean()) if len(triggered) else math.nan
        ),
        "early_exit_wins_vs_hold": int((triggered["pnl_difference_dollars"] > 0).sum()),
        "early_exit_losses_vs_hold": int((triggered["pnl_difference_dollars"] < 0).sum()),
        "early_exit_ties_vs_hold": int((triggered["pnl_difference_dollars"] == 0).sum()),
    }

    trades.to_csv(OUT / "weekly_put_spread_trades.csv", index=False)
    summary.to_csv(OUT / "weekly_put_spread_summary.csv", index_label="strategy")
    skip_df.to_csv(OUT / "weekly_put_spread_skips.csv", index=False)
    with (OUT / "weekly_put_spread_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    x = trades["expiration_date"]
    hold_equity = trades["hold_both_pnl_dollars"].cumsum()
    early_equity = trades["sell_long_1dte_pnl_dollars"].cumsum()
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(x, hold_equity, label="Hold both legs to expiration", linewidth=2)
    ax.plot(x, early_equity, label="Sell long put at 1 DTE if ITM", linewidth=2)
    trigger_rows = trades["long_itm_1dte"]
    ax.scatter(
        trades.loc[trigger_rows, "expiration_date"],
        early_equity.loc[trigger_rows],
        s=22,
        color="crimson",
        alpha=0.8,
        label="Early-sale trigger",
        zorder=3,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Weekly SPXW 1%/3% OTM Put Credit Spread — 1 Contract")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_xlabel("Expiration date")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "weekly_put_spread_equity_curves.png", dpi=180)
    plt.close(fig)

    print("\nSUMMARY")
    print(summary.round(4).to_string())
    print("\nCOMPARISON")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
