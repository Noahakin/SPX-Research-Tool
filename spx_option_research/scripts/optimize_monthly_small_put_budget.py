from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUT = PROJECT / "results/monthly_small_put_budget"
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))

from compare_monthly_credit_funded_hedge import (
    INITIAL_CAPITAL,
    MODEL,
    STRUCTURES,
    add_expiration_and_size,
    choose_vertical,
    daily_paths,
    metrics,
    monthly_roll_dates,
)
from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.option_selector import clean_puts
from spxresearch.warehouse import ResearchWarehouse


BUDGET_PCTS = [0.01, 0.025, 0.05, 0.075, 0.10]
BASELINE = "99/96 monthly, no hedge"
UNHEDGED_ITM = "103/100 monthly, no hedge"


def budget_label(budget: float) -> str:
    return f"103/100 + far OTM put ({budget:.1%} credit budget)"


def build_entries(archive: SPXSurfaceArchive) -> tuple[pd.DataFrame, pd.DataFrame]:
    roll_dates = monthly_roll_dates(archive.populated_dates)
    rows: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for entry_date, expiration_date in zip(roll_dates[:-1], roll_dates[1:]):
        chain = archive.read(entry_date)
        puts = clean_puts(chain, pm_only=True)
        puts = puts[puts["expiration_date"].eq(expiration_date)].copy()
        if puts.empty:
            skips.append({"entry_date": entry_date, "strategy": "all", "reason": "no matching PM expiration"})
            continue
        spot = float(puts["underlying_price"].dropna().median())
        try:
            base = choose_vertical(puts, spot, *STRUCTURES["99/96"])
            itm = choose_vertical(puts, spot, *STRUCTURES["103/100"])
        except LookupError as error:
            skips.append({"entry_date": entry_date, "strategy": "all", "reason": str(error)})
            continue

        common = {
            "entry_date": entry_date,
            "expiration_date": expiration_date,
            "entry_dte": int((expiration_date - entry_date).days),
            "spot_entry": spot,
        }
        rows.append(
            {
                **common,
                "trade_id": f"baseline_{entry_date:%Y%m%d}",
                "strategy": BASELINE,
                "structure": "99/96",
                "short_strike": base["short_strike"],
                "long_strike": base["long_strike"],
                "short_symbol": base["short_symbol"],
                "long_symbol": base["long_symbol"],
                "entry_credit_cash": base["entry_credit_cash"],
                "entry_credit_points": base["entry_credit_points"],
                "package_entry_cash": base["entry_credit_cash"],
                "has_hedge": False,
                "budget_pct": 0.0,
                "hedge_strike": np.nan,
                "hedge_symbol": None,
                "hedge_abs_delta": np.nan,
                "hedge_otm_pct": np.nan,
                "hedge_debit_cash": 0.0,
                "budget_utilization": np.nan,
            }
        )
        rows.append(
            {
                **common,
                "trade_id": f"itm_unhedged_{entry_date:%Y%m%d}",
                "strategy": UNHEDGED_ITM,
                "structure": "103/100",
                "short_strike": itm["short_strike"],
                "long_strike": itm["long_strike"],
                "short_symbol": itm["short_symbol"],
                "long_symbol": itm["long_symbol"],
                "entry_credit_cash": itm["entry_credit_cash"],
                "entry_credit_points": itm["entry_credit_points"],
                "package_entry_cash": itm["entry_credit_cash"],
                "has_hedge": False,
                "budget_pct": 0.0,
                "hedge_strike": np.nan,
                "hedge_symbol": None,
                "hedge_abs_delta": np.nan,
                "hedge_otm_pct": np.nan,
                "hedge_debit_cash": 0.0,
                "budget_utilization": np.nan,
            }
        )

        eligible = puts[puts["strike"].astype(float) < float(itm["long_strike"])].copy()
        eligible["hedge_debit_cash"] = eligible.apply(lambda quote: -MODEL.cash_flow(quote, 1), axis=1)
        eligible = eligible[eligible["hedge_debit_cash"].gt(0)]
        for budget in BUDGET_PCTS:
            budget_cash = budget * float(itm["entry_credit_cash"])
            affordable = eligible[eligible["hedge_debit_cash"].le(budget_cash)]
            strategy = budget_label(budget)
            if affordable.empty:
                skips.append({
                    "entry_date": entry_date,
                    "strategy": strategy,
                    "reason": "no full protective put fits budget",
                })
                continue
            hedge = affordable.sort_values(["strike", "hedge_debit_cash"], ascending=[False, False]).iloc[0]
            debit = float(hedge["hedge_debit_cash"])
            rows.append(
                {
                    **common,
                    "trade_id": f"budget_{budget:.4f}_{entry_date:%Y%m%d}",
                    "strategy": strategy,
                    "structure": "103/100 + put",
                    "short_strike": itm["short_strike"],
                    "long_strike": itm["long_strike"],
                    "short_symbol": itm["short_symbol"],
                    "long_symbol": itm["long_symbol"],
                    "entry_credit_cash": itm["entry_credit_cash"],
                    "entry_credit_points": itm["entry_credit_points"],
                    "package_entry_cash": float(itm["entry_credit_cash"]) - debit,
                    "has_hedge": True,
                    "budget_pct": budget,
                    "hedge_strike": float(hedge["strike"]),
                    "hedge_symbol": str(hedge["option_symbol"]).strip(),
                    "hedge_abs_delta": abs(float(hedge["delta"])),
                    "hedge_otm_pct": 100.0 * (1.0 - float(hedge["strike"]) / spot),
                    "hedge_debit_cash": debit,
                    "budget_utilization": debit / budget_cash,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(skips)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    archive = SPXSurfaceArchive(data_root)
    entries, skips = build_entries(archive)
    if entries.empty:
        raise RuntimeError("No monthly trades selected")

    expected = [BASELINE, UNHEDGED_ITM] + [budget_label(value) for value in BUDGET_PCTS]
    counts = entries.groupby("strategy")["trade_id"].size()
    common_count = int(counts.min())
    incomplete = counts[counts.ne(common_count)]
    if not incomplete.empty:
        valid_dates = set(entries.groupby("entry_date")["strategy"].nunique().loc[lambda x: x.eq(len(expected))].index)
        entries = entries[entries["entry_date"].isin(valid_dates)].copy()

    database = PROJECT / "data/cache/monthly_small_put_budget.duckdb"
    with ResearchWarehouse(database, data_root, threads=8, memory_limit="12GB") as warehouse:
        entries = add_expiration_and_size(entries, warehouse)
        paths = daily_paths(entries, warehouse)

    entries.to_csv(OUT / "trades.csv", index=False)
    entries.to_parquet(OUT / "trades.parquet", index=False)
    skips.to_csv(OUT / "skips.csv", index=False)
    paths["mark_date"] = pd.to_datetime(paths["mark_date"])
    paths.to_parquet(OUT / "daily_paths.parquet", index=False)

    all_dates = pd.DatetimeIndex(archive.populated_dates)
    curves: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for strategy in expected:
        strategy_paths = paths[paths["strategy"].eq(strategy)]
        daily = strategy_paths.groupby("mark_date")["daily_pnl"].sum().reindex(all_dates, fill_value=0.0)
        equity = INITIAL_CAPITAL + daily.cumsum()
        trades = entries[entries["strategy"].eq(strategy)].copy()
        trades["spot_return"] = trades["spot_expiration"] / trades["spot_entry"] - 1.0
        trades["trade_return"] = trades["dollar_pnl"] / trades["pre_trade_equity"]
        row: dict[str, object] = {"strategy": strategy}
        row.update(metrics(equity, daily, trades))
        row.update(
            {
                "budget_pct": float(trades["budget_pct"].iloc[0]),
                "average_entry_credit_points": float(trades["entry_credit_points"].mean()),
                "average_package_credit_points": float(trades["package_entry_cash"].mean() / 100.0),
                "average_hedge_debit_points": float(trades["hedge_debit_cash"].mean() / 100.0),
                "average_budget_utilization": float(trades["budget_utilization"].mean()),
                "average_hedge_otm_pct": float(trades["hedge_otm_pct"].mean()),
                "average_hedge_abs_delta": float(trades["hedge_abs_delta"].mean()),
                "hedge_profitable_rate": float(
                    ((trades["hedge_expiration_value_cash"] - trades["hedge_debit_cash"]) > 0).mean()
                ) if trades["has_hedge"].any() else np.nan,
                "mean_return_spx_down_5": float(trades.loc[trades["spot_return"].le(-0.05), "trade_return"].mean()),
                "observations_spx_down_5": int(trades["spot_return"].le(-0.05).sum()),
                "mean_return_spx_down_10": float(trades.loc[trades["spot_return"].le(-0.10), "trade_return"].mean()),
                "observations_spx_down_10": int(trades["spot_return"].le(-0.10).sum()),
            }
        )
        rows.append(row)
        curves[strategy] = equity

    summary = pd.DataFrame(rows).sort_values("cagr", ascending=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    curve_frame = pd.DataFrame(curves, index=all_dates)
    curve_frame.index.name = "date"
    curve_frame.to_csv(OUT / "equity_curves.csv")
    curve_frame.to_parquet(OUT / "equity_curves.parquet")

    plotted = [BASELINE, UNHEDGED_ITM, budget_label(0.025), budget_label(0.05), budget_label(0.10)]
    colors = ["#006b76", "#55a7b0", "#e8ae65", "#c56a1a", "#755a9e"]
    indexed = summary.set_index("strategy")
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for strategy, color in zip(plotted, colors):
        short_label = strategy.replace(" monthly, no hedge", " unhedged").replace(" monthly,", ",")
        label = f"{short_label} ({indexed.loc[strategy, 'cagr']:.2%})"
        ax.plot(curve_frame.index, curve_frame[strategy] / INITIAL_CAPITAL - 1.0, lw=1.8, color=color, label=label)
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title("Monthly SPX put spreads with small far-OTM hedge budgets", loc="left", weight="bold", pad=20)
    ax.set_ylabel("Cumulative option return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(
        0.075,
        0.025,
        "One full hedge put per vertical; highest affordable strike; third-Friday rolls; held to expiration; 100% short-strike notional; no cash interest.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=180)
    plt.close(fig)

    best_hedged = summary[summary["budget_pct"].gt(0)].iloc[0]
    manifest = {
        "archive_start": str(all_dates.min().date()),
        "archive_end": str(all_dates.max().date()),
        "trades_per_strategy": int(entries.groupby("strategy")["trade_id"].size().min()),
        "budgets_tested": BUDGET_PCTS,
        "best_hedged_strategy": best_hedged["strategy"],
        "best_hedged_cagr": float(best_hedged["cagr"]),
        "plotted_strategies": plotted,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
