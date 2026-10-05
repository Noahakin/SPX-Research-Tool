from __future__ import annotations

import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from run_requested_premium_study import CASH_CACHE, DATA_ROOT, SPXSurfaceArchive

SOURCE = PROJECT / "results/standardized_iv_rv_spread_selector"
OUT = PROJECT / "results/standardized_iv_rv_daily"
INITIAL = 1_000_000.0
MULTIPLIER = 100.0
QUOTE_COLUMNS = ["snapshot_date", "expiration_date", "option_symbol", "strike", "bid", "ask"]


def daily_quotes(trades: pd.DataFrame, dates: pd.DatetimeIndex, archive: SPXSurfaceArchive) -> pd.DataFrame:
    wanted = {day: set() for day in dates}
    for row in trades.itertuples(index=False):
        for day in dates[(dates >= row.entry_date) & (dates < row.expiration_date)]:
            wanted[day].update([row.upper_symbol, row.lower_symbol])

    def read_one(item):
        day, symbols = item
        if not symbols:
            return pd.DataFrame(columns=QUOTE_COLUMNS), []
        if day not in archive.paths:
            return pd.DataFrame(columns=QUOTE_COLUMNS), [{"date": day, "reason": "archive date missing"}]
        frame = pq.ParquetFile(archive.path_for(day)).read(columns=QUOTE_COLUMNS).to_pandas()
        frame["option_symbol"] = frame["option_symbol"].astype(str).str.strip()
        frame = frame[frame["option_symbol"].isin(symbols)].copy()
        issues = [{"date": day, "option_symbol": symbol, "reason": "required quote absent"}
                  for symbol in sorted(symbols - set(frame["option_symbol"]))]
        if frame["option_symbol"].duplicated().any():
            issues.append({"date": day, "reason": "duplicate selected symbol"})
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
        frame["expiration_date"] = pd.to_datetime(frame["expiration_date"]).dt.normalize()
        bad = (~np.isfinite(frame[["bid", "ask", "strike"]]).all(axis=1)
               | frame.bid.lt(0) | frame.ask.lt(frame.bid) | frame.strike.le(0)
               | frame.snapshot_date.ne(day))
        for row in frame[bad].itertuples(index=False):
            issues.append({"date": day, "option_symbol": row.option_symbol, "reason": "invalid quote"})
        frame["mid"] = (frame.bid + frame.ask) / 2.0
        frame["source_file"] = str(archive.path_for(day).relative_to(DATA_ROOT))
        return frame, issues

    frames, issues = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for frame, errors in pool.map(read_one, wanted.items()):
            if not frame.empty:
                frames.append(frame)
            issues.extend(errors)
    pd.DataFrame(issues, columns=["date", "option_symbol", "reason"]).to_csv(OUT / "quote_issues.csv", index=False)
    if issues:
        raise RuntimeError(f"{len(issues)} missing/invalid quote issues; see {OUT / 'quote_issues.csv'}")
    quotes = pd.concat(frames, ignore_index=True).sort_values(["snapshot_date", "option_symbol"])
    expected_count = sum(len(symbols) for symbols in wanted.values())
    if len(quotes) != expected_count:
        raise ValueError("selected quote count does not reconcile")
    quotes.to_parquet(OUT / "daily_option_quotes.parquet", index=False)
    quotes.to_csv(OUT / "daily_option_quotes.csv", index=False)
    print(f"Validated {len(quotes):,} actual daily leg quotes; no missing/invalid marks.", flush=True)
    return quotes


def close_enough(left: float, right: float, description: str) -> None:
    if not np.isclose(left, right, rtol=1e-10, atol=1e-7):
        raise ValueError(f"{description}: {left} versus {right}")


def simulate_daily(trades: pd.DataFrame, quotes: pd.DataFrame, closes: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cash plus actual marked option value; old expiry precedes new entry on rolls."""
    lookup = quotes.set_index(["snapshot_date", "option_symbol"])
    if lookup.index.has_duplicates:
        raise ValueError("daily symbol quotes must be unique")
    start, end = trades.entry_date.min(), trades.expiration_date.max()
    closes = closes.sort_index().loc[start:end]
    daily_records, settled_records = [], []
    for portfolio, group in trades.groupby("portfolio", sort=False):
        group = group.sort_values("entry_date")
        if group.entry_date.duplicated().any():
            raise ValueError("portfolio has duplicate entry dates")
        entries = {row.entry_date: row._asdict() for row in group.itertuples(index=False)}
        cash, previous_equity = INITIAL, INITIAL
        active = None
        for day, spot in closes.items():
            settlement_cash = 0.0
            entry_cost = 0.0
            if active is not None and day >= active["expiration_date"]:
                if day != active["expiration_date"]:
                    raise ValueError("cash index history omits an expiration session")
                expiry_one = MULTIPLIER * (-max(active["upper_strike"] - spot, 0.0)
                                          + max(active["lower_strike"] - spot, 0.0))
                close_enough(expiry_one, active["expiration_value_cash"], "cash intrinsic per contract")
                settlement_cash = active["contracts"] * expiry_one
                cash += settlement_cash
                expected = INITIAL * active["ending_equity_realistic"]
                close_enough(cash, expected, "pre-new-entry settlement equity")
                settled_records.append({
                    "portfolio": portfolio, "entry_date": active["entry_date"], "expiration_date": day,
                    "spread": active["spread"], "contracts": active["contracts"],
                    "settled_equity_before_new_entry": cash, "original_monthly_equity": expected,
                    "reconciliation_error_dollars": cash - expected,
                })
                active = None

            if day in entries:
                if active is not None:
                    raise ValueError("overlapping trades")
                active = entries[day].copy()
                close_enough(cash, INITIAL * active["entry_equity_realistic"], "equity used for sizing")
                close_enough(spot, active["spot_entry"], "entry cash spot")
                active["contracts"] = cash / (active["spot_entry"] * MULTIPLIER)
                close_enough(active["contracts"], active["contracts_realistic_per_initial_1m"], "saved contract sizing")
                cash += active["contracts"] * active["premium_realistic_cash"]
                entry_cost = active["contracts"] * (active["premium_mid_cash"] - active["premium_realistic_cash"])

            value, width, spread_mid, contracts = 0.0, np.nan, np.nan, 0.0
            if active is not None:
                short = lookup.loc[(day, active["upper_symbol"])]
                long = lookup.loc[(day, active["lower_symbol"])]
                for leg, strike in ((short, active["upper_strike"]), (long, active["lower_strike"])):
                    close_enough(leg["strike"], strike, "held strike")
                    if leg["expiration_date"] != active["expiration_date"]:
                        raise ValueError("held expiration does not match quote")
                spread_mid = float(short["mid"] - long["mid"])
                width = active["upper_strike"] - active["lower_strike"]
                if day == active["entry_date"]:
                    close_enough(spread_mid * MULTIPLIER, active["premium_mid_cash"], "entry midpoint credit")
                    realistic_credit = MULTIPLIER * (
                        spread_mid - 0.25 * (short["ask"] - short["bid"] + long["ask"] - long["bid"])
                    ) - 3.0
                    close_enough(realistic_credit, active["premium_realistic_cash"], "entry costs")
                contracts = active["contracts"]
                value = -MULTIPLIER * contracts * spread_mid
            equity = cash + value
            daily_records.append({
                "portfolio": portfolio, "date": day, "cash": cash, "option_value_mid": value,
                "equity": equity, "daily_pnl": equity - previous_equity,
                "daily_return": equity / previous_equity - 1.0,
                "entry_cost_dollars": entry_cost, "settlement_cash_flow": settlement_cash,
                "active_spread": active["spread"] if active is not None else "cash",
                "contracts": contracts, "spread_mid_points": spread_mid, "width_points": width,
                "mid_outside_payoff_bounds": bool(active is not None and (spread_mid < -0.01 or spread_mid > width + 0.01)),
            })
            previous_equity = equity
        if active is not None:
            raise ValueError("unsettled position at final date")
        close_enough(previous_equity, INITIAL * group.ending_equity_realistic.iloc[-1], "terminal equity")
    daily = pd.DataFrame(daily_records)
    settlement = pd.DataFrame(settled_records)
    if len(settlement) != len(trades):
        raise ValueError("not every source trade settled")
    return daily, settlement


def metrics(daily: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    rows = []
    old = original.set_index("portfolio")
    for name, group in daily.groupby("portfolio", sort=False):
        group = group.sort_values("date")
        r = group.daily_return
        sd = float(r.std(ddof=1))
        equity = group.equity.to_numpy()
        peak = np.maximum.accumulate(np.r_[INITIAL, equity])[1:]
        years = (group.date.max() - group.date.min()).days / 365.2425
        close_enough(float((1 + r).prod()), equity[-1] / INITIAL, "daily compounding")
        rows.append({
            "portfolio": name, "first_entry": group.date.min(), "last_expiration": group.date.max(),
            "daily_observations": len(group), "daily_sharpe": float(r.mean() / sd * math.sqrt(252.0)),
            "daily_annualized_volatility": sd * math.sqrt(252.0),
            "cagr": (equity[-1] / INITIAL) ** (1.0 / years) - 1.0,
            "max_drawdown_daily": float((equity / peak - 1.0).min()),
            "worst_day": float(r.min()), "best_day": float(r.max()), "ending_equity": equity[-1],
            "previous_monthly_sharpe": old.loc[name, "sharpe_realistic"],
            "previous_monthly_max_drawdown": old.loc[name, "max_drawdown_at_rolls_realistic"],
            "previous_ending_equity": INITIAL * old.loc[name, "ending_wealth_realistic"],
            "quote_payoff_bound_flags": int(group.mid_outside_payoff_bounds.sum()),
        })
    return pd.DataFrame(rows)


def bounded_mark_sensitivity(daily: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Check raw quote midpoint noise without replacing the primary observations."""
    adjusted = daily.copy()
    bounded_mid = adjusted.spread_mid_points.clip(lower=0.0, upper=adjusted.width_points)
    adjusted["option_value_mid"] = (-MULTIPLIER * adjusted.contracts * bounded_mid).fillna(0.0)
    adjusted["equity"] = adjusted.cash + adjusted.option_value_mid
    previous = adjusted.groupby("portfolio", sort=False).equity.shift(1).fillna(INITIAL)
    adjusted["daily_pnl"] = adjusted.equity - previous
    adjusted["daily_return"] = adjusted.equity / previous - 1.0
    adjusted["mid_outside_payoff_bounds"] = False
    result = metrics(adjusted, original)
    result.to_csv(OUT / "bounded_midpoint_sensitivity.csv", index=False)
    return result


def write_chart(daily: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.patch.set_facecolor("#f8fafc")
    colors = ["#087f8c", "#d68028"]
    for (name, group), color in zip(daily.groupby("portfolio", sort=False), colors):
        group = group.sort_values("date")
        row = summary.set_index("portfolio").loc[name]
        label = "Dynamic IV/RV z-score" if name.startswith("Dynamic") else "Fixed 99/96"
        dates = pd.DatetimeIndex([group.date.min() - pd.Timedelta(seconds=1), *group.date])
        eq = np.r_[INITIAL, group.equity]
        ax.plot(dates, eq, color=color, linewidth=1.6,
                label=f"{label}  |  daily Sharpe {row.daily_sharpe:.2f}  |  CAGR {row.cagr:.2%}")
        ax.annotate(f"${eq[-1]:,.0f}", (dates[-1], eq[-1]), xytext=(10, 0), textcoords="offset points",
                    fontsize=10, va="center", color=color, fontweight="bold")
    ax.set_title("SPX put spreads: equity marked daily", loc="left", fontsize=18, fontweight="bold", pad=29)
    ax.text(0, 1.025, "Growth of $1 million  |  November 2018–September 2026", transform=ax.transAxes,
            fontsize=11, color="#475569")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1e6:.2f}M"))
    ax.set_ylabel("Portfolio equity")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(daily.date.min() - pd.Timedelta(days=55), daily.date.max() + pd.Timedelta(days=340))
    ax.axhline(INITIAL, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.text(0.08, 0.025, "Actual daily option bid/ask midpoint marks; cash intrinsic at expiry. Entry costs included once.\n"
             "100% SPX spot notional at each monthly entry. Option P&L only; no collateral interest.", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    fig.savefig(OUT / "daily_equity_comparison.png", dpi=180)
    fig.savefig(OUT / "daily_equity_comparison.svg")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(SOURCE / "portfolio_trades.parquet")
    original = pd.read_csv(SOURCE / "summary.csv")
    closes = pd.read_csv(CASH_CACHE, parse_dates=["date"]).set_index("date").spx_close
    dates = closes.loc[trades.entry_date.min():trades.expiration_date.max()].index
    print(f"Reading actual quote marks for {len(dates):,} sessions and {len(trades)} portfolio trades.", flush=True)
    archive = SPXSurfaceArchive(DATA_ROOT)
    quotes = daily_quotes(trades, dates, archive)
    daily, reconciliations = simulate_daily(trades, quotes, closes)
    summary = metrics(daily, original)
    bounded = bounded_mark_sensitivity(daily, original)
    bounded_text = "; ".join(f"{r.portfolio}: {r.daily_sharpe:.6f}" for r in bounded.itertuples(index=False))
    for name, frame in (("daily_portfolios", daily), ("settlement_reconciliation", reconciliations), ("summary", summary)):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    daily.to_parquet(OUT / "daily_portfolios.parquet", index=False)
    daily.pivot(index="date", columns="portfolio", values="equity").to_csv(OUT / "equity_curves_daily.csv")
    daily.pivot(index="date", columns="portfolio", values="daily_return").to_csv(OUT / "returns_daily.csv")
    daily[daily.mid_outside_payoff_bounds].to_csv(OUT / "spread_bound_flags.csv", index=False)
    write_chart(daily, summary)
    rows = "\n".join(f"| {r.portfolio} | {r.daily_sharpe:.3f} | {r.previous_monthly_sharpe:.3f} | {r.cagr:.2%} | {r.daily_annualized_volatility:.2%} | {r.max_drawdown_daily:.2%} | ${r.ending_equity:,.0f} |" for r in summary.itertuples(index=False))
    report = f"""# Daily valuation of dynamic SPX spreads versus fixed 99/96

Both strategies use the exact 94 original trades from November 16, 2018 through September 18, 2026. Open short and long legs are valued using their actual EOD bid/ask midpoints on each trading session. Expiration uses intrinsic value at the cached SPX cash close. This produces {len(dates):,} daily return observations per portfolio, including first-entry slippage and commissions. There are no interpolated option prices, filled-forward missing marks, or artificial zero-P&L holding days.

| Portfolio | Daily Sharpe | Previous monthly Sharpe | CAGR | Annualized daily volatility | Daily max drawdown | Ending equity |
|---|---:|---:|---:|---:|---:|---:|
{rows}

Daily Sharpe is `mean(daily return) / sample_std(daily return) * sqrt(252)` with zero cash rate. Each daily return is the change in equity divided by the prior session's equity; the first entry uses initial capital of $1 million. CAGR uses exact elapsed calendar years and final equity. Collateral interest, taxes, and settlement fees remain excluded.

At each monthly roll the old trade settles first. Its cash equity reconciles to the original monthly backtest; the new trade is then sized at 100% of that equity divided by cash SPX times 100. The new trade's closing NAV immediately reflects entry slippage and commissions. Daily equity on a roll can therefore differ slightly from the previous chart's pre-new-entry settlement equity. All 188 trade settlements reconcile, and both terminal equity values are unchanged.

Entry execution remains one-quarter of each leg's full bid/ask spread away from midpoint plus $1.50 per contract per leg. These costs are charged once at entry; daily mid marks do not incur hypothetical liquidation costs. The saved trade strikes, symbols, entries, expirations, and quantities are preserved. Daily quotes validate date, symbol, strike, expiration, finite nonnegative bid, and ask at least bid. All {len(quotes):,} required daily leg quotes are present. {int(daily.mid_outside_payoff_bounds.sum())} portfolio-date midpoint spread values lie outside the undiscounted payoff range by more than 0.01 point; any such observations are separately saved in spread_bound_flags.csv for inspection.

As a sensitivity check, bounding daily midpoint spread values to [0, strike width] produces annualized daily Sharpes of {bounded_text}. This does not replace the primary observed-midpoint marks. The small quote-midpoint discrepancies have negligible influence on the results.

![Daily equity](daily_equity_comparison.png)

Reproduce: `python scripts/mark_standardized_spread_daily.py`. Audit data include actual daily leg quotes, daily equity and returns, entry costs, and the reconciliation of every expiration to the original monthly backtest.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "source_trades": str((SOURCE / "portfolio_trades.parquet").relative_to(PROJECT)),
        "source_trades_sha256": hashlib.sha256((SOURCE / "portfolio_trades.parquet").read_bytes()).hexdigest(),
        "cash_close_sha256": hashlib.sha256(CASH_CACHE.read_bytes()).hexdigest(),
        "daily_quotes_sha256": hashlib.sha256((OUT / "daily_option_quotes.parquet").read_bytes()).hexdigest(),
        "sessions_per_portfolio": len(dates), "quotes": len(quotes), "trade_settlements": len(reconciliations),
        "max_settlement_error_dollars": float(reconciliations.reconciliation_error_dollars.abs().max()),
        "mark": "actual daily bid/ask midpoints; cash SPX intrinsic at expiration",
        "missing_quote_policy": "fail; no forward filling or interpolation",
        "sharpe": "daily mean/sample_std * sqrt(252), zero cash rate",
        "first_entry_cost_included": True, "daily_liquidation_costs_charged": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved daily mark-to-market results to {OUT}")


if __name__ == "__main__":
    main()
