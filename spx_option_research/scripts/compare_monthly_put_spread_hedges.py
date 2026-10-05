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
OUT = PROJECT / "results/monthly_put_spread_hedges"
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))

from compare_monthly_credit_funded_hedge import (
    INITIAL_CAPITAL,
    MODEL,
    choose_vertical,
    metrics,
    monthly_roll_dates,
)
from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.option_selector import clean_puts
from spxresearch.warehouse import ResearchWarehouse


CORE_STRUCTURES = {"99/96": (0.99, 0.96), "101/98": (1.01, 0.98)}
HEDGE_BUDGET_PCT = 0.05
HEDGE_WIDTH_SPOT_PCT = 0.03


def strategy_label(structure: str, hedged: bool) -> str:
    return f"{structure} monthly, {'5% credit put-spread hedge' if hedged else 'no hedge'}"


def protective_spread_candidates(puts: pd.DataFrame, spot: float, strike_cap: float) -> pd.DataFrame:
    uppers = puts[puts["strike"].astype(float) < strike_cap].copy()
    records: list[dict[str, object]] = []
    for _, upper in uppers.iterrows():
        lower_pool = puts[puts["strike"].astype(float) < float(upper["strike"])]
        if lower_pool.empty:
            continue
        target_lower = float(upper["strike"]) - HEDGE_WIDTH_SPOT_PCT * spot
        lower = lower_pool.loc[(lower_pool["strike"].astype(float) - target_lower).abs().idxmin()]
        debit_cash = -(MODEL.cash_flow(upper, 1) + MODEL.cash_flow(lower, -1))
        if debit_cash <= 0:
            continue
        records.append(
            {
                "hedge_upper_strike": float(upper["strike"]),
                "hedge_lower_strike": float(lower["strike"]),
                "hedge_upper_symbol": str(upper["option_symbol"]).strip(),
                "hedge_lower_symbol": str(lower["option_symbol"]).strip(),
                "hedge_upper_abs_delta": abs(float(upper["delta"])),
                "hedge_lower_abs_delta": abs(float(lower["delta"])),
                "hedge_debit_cash": float(debit_cash),
            }
        )
    return pd.DataFrame(records)


def build_entries(archive: SPXSurfaceArchive) -> tuple[pd.DataFrame, pd.DataFrame]:
    roll_dates = monthly_roll_dates(archive.populated_dates)
    rows: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for entry_date, expiration_date in zip(roll_dates[:-1], roll_dates[1:]):
        chain = archive.read(entry_date)
        puts = clean_puts(chain, pm_only=True)
        puts = puts[puts["expiration_date"].eq(expiration_date)].copy()
        if puts.empty:
            skips.append({"entry_date": entry_date, "structure": "all", "reason": "no matching PM expiration"})
            continue
        spot = float(puts["underlying_price"].dropna().median())
        for structure, ratios in CORE_STRUCTURES.items():
            try:
                core = choose_vertical(puts, spot, *ratios)
            except LookupError as error:
                skips.append({"entry_date": entry_date, "structure": structure, "reason": str(error)})
                continue

            common = {
                "entry_date": entry_date,
                "expiration_date": expiration_date,
                "entry_dte": int((expiration_date - entry_date).days),
                "spot_entry": spot,
                "structure": structure,
                "short_strike": core["short_strike"],
                "long_strike": core["long_strike"],
                "short_symbol": core["short_symbol"],
                "long_symbol": core["long_symbol"],
                "entry_credit_cash": core["entry_credit_cash"],
                "entry_credit_points": core["entry_credit_points"],
            }
            rows.append(
                {
                    **common,
                    "trade_id": f"{structure.replace('/', '_')}_none_{entry_date:%Y%m%d}",
                    "strategy": strategy_label(structure, False),
                    "has_hedge": False,
                    "package_entry_cash": core["entry_credit_cash"],
                    "hedge_budget_cash": 0.0,
                    "hedge_debit_cash": 0.0,
                    "budget_utilization": np.nan,
                    "hedge_upper_strike": np.nan,
                    "hedge_lower_strike": np.nan,
                    "hedge_upper_symbol": None,
                    "hedge_lower_symbol": None,
                    "hedge_upper_abs_delta": np.nan,
                    "hedge_lower_abs_delta": np.nan,
                    "hedge_upper_otm_pct": np.nan,
                    "hedge_width_points": np.nan,
                }
            )

            budget_cash = HEDGE_BUDGET_PCT * float(core["entry_credit_cash"])
            candidates = protective_spread_candidates(puts, spot, float(core["long_strike"]))
            affordable = candidates[candidates["hedge_debit_cash"].le(budget_cash)] if not candidates.empty else candidates
            if affordable.empty:
                skips.append({
                    "entry_date": entry_date,
                    "structure": structure,
                    "reason": "no full protective put spread fits 5% credit budget",
                })
                continue
            hedge = affordable.sort_values(
                ["hedge_upper_strike", "hedge_debit_cash"], ascending=[False, False]
            ).iloc[0]
            debit = float(hedge["hedge_debit_cash"])
            rows.append(
                {
                    **common,
                    "trade_id": f"{structure.replace('/', '_')}_hedged_{entry_date:%Y%m%d}",
                    "strategy": strategy_label(structure, True),
                    "has_hedge": True,
                    "package_entry_cash": float(core["entry_credit_cash"]) - debit,
                    "hedge_budget_cash": budget_cash,
                    "hedge_debit_cash": debit,
                    "budget_utilization": debit / budget_cash,
                    "hedge_upper_strike": float(hedge["hedge_upper_strike"]),
                    "hedge_lower_strike": float(hedge["hedge_lower_strike"]),
                    "hedge_upper_symbol": hedge["hedge_upper_symbol"],
                    "hedge_lower_symbol": hedge["hedge_lower_symbol"],
                    "hedge_upper_abs_delta": float(hedge["hedge_upper_abs_delta"]),
                    "hedge_lower_abs_delta": float(hedge["hedge_lower_abs_delta"]),
                    "hedge_upper_otm_pct": 100.0 * (1.0 - float(hedge["hedge_upper_strike"]) / spot),
                    "hedge_width_points": float(hedge["hedge_upper_strike"] - hedge["hedge_lower_strike"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(skips)


def add_expiration_and_size(entries: pd.DataFrame, warehouse: ResearchWarehouse) -> pd.DataFrame:
    warehouse.connection.register("monthly_spread_entries", entries)
    expiration = warehouse.execute(
        """
        SELECT e.trade_id, median(s.spot) AS spot_expiration
        FROM monthly_spread_entries e
        JOIN surface s
          ON s.trade_date = CAST(e.expiration_date AS DATE)
         AND s.expiration_date = CAST(e.expiration_date AS DATE)
         AND s.settlement = 'PM'
        GROUP BY e.trade_id
        """
    ).fetchdf()
    result = entries.merge(expiration, on="trade_id", how="inner")
    result["core_expiration_debit_cash"] = (
        np.maximum(result["short_strike"] - result["spot_expiration"], 0.0)
        - np.maximum(result["long_strike"] - result["spot_expiration"], 0.0)
    ) * 100.0
    result["hedge_expiration_value_cash"] = np.where(
        result["has_hedge"],
        (
            np.maximum(result["hedge_upper_strike"] - result["spot_expiration"], 0.0)
            - np.maximum(result["hedge_lower_strike"] - result["spot_expiration"], 0.0)
        ) * 100.0,
        0.0,
    )
    result["pnl_per_spread"] = (
        result["package_entry_cash"]
        - result["core_expiration_debit_cash"]
        + result["hedge_expiration_value_cash"]
    )

    sized: list[pd.DataFrame] = []
    for strategy, group in result.groupby("strategy", sort=False):
        equity = INITIAL_CAPITAL
        records: list[pd.Series] = []
        for _, row in group.sort_values("entry_date").iterrows():
            row = row.copy()
            row["pre_trade_equity"] = equity
            row["contracts"] = equity / (float(row["short_strike"]) * 100.0)
            row["dollar_pnl"] = float(row["pnl_per_spread"]) * float(row["contracts"])
            equity += float(row["dollar_pnl"])
            row["post_trade_equity"] = equity
            records.append(row)
        sized.append(pd.DataFrame(records))
    return pd.concat(sized, ignore_index=True)


def daily_paths(entries: pd.DataFrame, warehouse: ResearchWarehouse) -> pd.DataFrame:
    warehouse.connection.unregister("monthly_spread_entries")
    warehouse.connection.register("monthly_spread_entries", entries)
    paths = warehouse.execute(
        """
        SELECT
            e.trade_id,
            e.strategy,
            e.entry_date,
            e.expiration_date,
            e.contracts,
            e.package_entry_cash,
            e.has_hedge,
            s.trade_date AS mark_date,
            CASE
                WHEN s.trade_date = CAST(e.expiration_date AS DATE) OR s.dte = 0 THEN
                    (greatest(e.short_strike - s.spot, 0.0)
                     - greatest(e.long_strike - s.spot, 0.0)) * 100.0
                ELSE (s.mid - l.mid) * 100.0
            END AS core_mark_debit_cash,
            CASE
                WHEN NOT e.has_hedge THEN 0.0
                WHEN s.trade_date = CAST(e.expiration_date AS DATE) OR s.dte = 0 THEN
                    (greatest(e.hedge_upper_strike - s.spot, 0.0)
                     - greatest(e.hedge_lower_strike - s.spot, 0.0)) * 100.0
                ELSE (hu.mid - hl.mid) * 100.0
            END AS hedge_mark_value_cash
        FROM monthly_spread_entries e
        JOIN surface s
          ON s.option_symbol = e.short_symbol
         AND s.trade_date BETWEEN CAST(e.entry_date AS DATE) AND CAST(e.expiration_date AS DATE)
        JOIN surface l
          ON l.option_symbol = e.long_symbol
         AND l.trade_date = s.trade_date
        LEFT JOIN surface hu
          ON e.has_hedge
         AND hu.option_symbol = e.hedge_upper_symbol
         AND hu.trade_date = s.trade_date
        LEFT JOIN surface hl
          ON e.has_hedge
         AND hl.option_symbol = e.hedge_lower_symbol
         AND hl.trade_date = s.trade_date
        ORDER BY strategy, trade_id, mark_date
        """
    ).fetchdf()
    paths["hedge_mark_value_cash"] = (
        paths.groupby("trade_id")["hedge_mark_value_cash"].ffill().fillna(0.0)
    )
    paths["cumulative_pnl"] = (
        paths["package_entry_cash"]
        - paths["core_mark_debit_cash"]
        + paths["hedge_mark_value_cash"]
    ) * paths["contracts"]
    paths["daily_pnl"] = paths.groupby("trade_id")["cumulative_pnl"].diff()
    paths["daily_pnl"] = paths["daily_pnl"].fillna(paths["cumulative_pnl"])
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    archive = SPXSurfaceArchive(data_root)
    entries, skips = build_entries(archive)
    expected = [
        strategy_label("99/96", False),
        strategy_label("99/96", True),
        strategy_label("101/98", False),
        strategy_label("101/98", True),
    ]
    valid_dates = set(
        entries.groupby("entry_date")["strategy"].nunique().loc[lambda x: x.eq(len(expected))].index
    )
    entries = entries[entries["entry_date"].isin(valid_dates)].copy()
    if entries.empty:
        raise RuntimeError("No complete matched monthly entries")

    database = PROJECT / "data/cache/monthly_put_spread_hedges.duckdb"
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
        row: dict[str, object] = {"strategy": strategy, "structure": trades["structure"].iloc[0], "hedged": bool(trades["has_hedge"].iloc[0])}
        row.update(metrics(equity, daily, trades))
        row.update(
            {
                "average_core_credit_points": float(trades["entry_credit_points"].mean()),
                "average_package_credit_points": float(trades["package_entry_cash"].mean() / 100.0),
                "average_hedge_debit_points": float(trades["hedge_debit_cash"].mean() / 100.0),
                "average_budget_utilization": float(trades["budget_utilization"].mean()),
                "average_hedge_upper_otm_pct": float(trades["hedge_upper_otm_pct"].mean()),
                "average_hedge_upper_abs_delta": float(trades["hedge_upper_abs_delta"].mean()),
                "average_hedge_width_points": float(trades["hedge_width_points"].mean()),
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

    colors = ["#006b76", "#55a7b0", "#c56a1a", "#e8ae65"]
    indexed = summary.set_index("strategy")
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for strategy, color in zip(expected, colors):
        row = indexed.loc[strategy]
        label = f"{strategy.replace(' monthly,', ',')} ({row['cagr']:.2%} CAGR)"
        ax.plot(curve_frame.index, curve_frame[strategy] / INITIAL_CAPITAL - 1.0, lw=2.0, color=color, label=label)
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title("Monthly 99/96 and 101/98 SPX put spreads with 5% tail-spread hedges", loc="left", weight="bold", pad=20)
    ax.set_ylabel("Cumulative option return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(
        0.075,
        0.025,
        "Protective spread: one 3%-of-spot-wide debit put spread per core vertical, highest affordable strike below the core long; held to expiration.",
        fontsize=9,
        color="#475569",
    )
    fig.text(
        0.075,
        0.008,
        "One position at a time; short-strike notional equals current equity; realistic execution; no cash interest.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=180)
    plt.close(fig)

    manifest = {
        "archive_start": str(all_dates.min().date()),
        "archive_end": str(all_dates.max().date()),
        "trades_per_strategy": int(entries.groupby("strategy")["trade_id"].size().min()),
        "hedge_budget_pct_of_core_credit": HEDGE_BUDGET_PCT,
        "target_hedge_width_pct_of_spot": HEDGE_WIDTH_SPOT_PCT,
        "strategies": expected,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
