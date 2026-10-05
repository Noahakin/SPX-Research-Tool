from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUT = PROJECT / "results/sequential_itm_hedges"
CACHE = PROJECT / "data/cache/sequential_itm_hedges"
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.warehouse import ResearchWarehouse


TARGET_OTM_PCTS = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 15.0]


def main() -> None:
    core = pd.read_parquet(CACHE / "core_candidates.parquet")
    core = core[core["structure"].isin(["103/100", "104/101"])].copy()
    core["width_points"] = core["short_strike"] - core["long_strike"]
    core["credit_points"] = core["core_entry_cash"] / 100.0
    core["credit_pct_width"] = core["core_entry_cash"] / (core["width_points"] * 100.0)
    core["credit_pct_short_notional"] = core["core_entry_cash"] / (core["short_strike"] * 100.0)

    spread_summary = (
        core.groupby(["structure", "target_dte"], as_index=False)
        .agg(
            observations=("trade_id", "size"),
            average_actual_dte=("actual_dte", "mean"),
            average_width_points=("width_points", "mean"),
            average_credit_points=("credit_points", "mean"),
            median_credit_points=("credit_points", "median"),
            average_credit_pct_width=("credit_pct_width", "mean"),
            median_credit_pct_width=("credit_pct_width", "median"),
            average_credit_pct_short_notional=("credit_pct_short_notional", "mean"),
        )
    )
    spread_summary.to_csv(OUT / "entry_credit_summary.csv", index=False)

    entries = core[core["structure"].eq("103/100")][
        ["trade_id", "entry_date", "expiration_date", "target_dte", "spot_entry"]
    ].rename(columns={"spot_entry": "spot"})
    targets = pd.DataFrame({"target_otm_pct": TARGET_OTM_PCTS})
    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    database = CACHE / "premium_analysis.duckdb"
    with ResearchWarehouse(database, data_root) as warehouse:
        warehouse.connection.register("premium_entries", entries)
        warehouse.connection.register("otm_targets", targets)
        protection = warehouse.execute(
            """
            SELECT
                e.trade_id,
                e.entry_date,
                e.expiration_date,
                e.target_dte,
                e.spot,
                t.target_otm_pct,
                s.strike AS hedge_strike,
                100.0 * (1.0 - s.strike / e.spot) AS actual_otm_pct,
                abs(s.delta) AS absolute_delta,
                s.bid,
                s.ask,
                ((s.bid + s.ask) / 2.0 + 0.25 * (s.ask - s.bid) + 0.015) AS debit_points
            FROM premium_entries e
            CROSS JOIN otm_targets t
            JOIN surface s
              ON s.trade_date = CAST(e.entry_date AS DATE)
             AND s.expiration_date = CAST(e.expiration_date AS DATE)
            WHERE s.option_type = 'put'
              AND s.settlement = 'PM'
              AND s.strike > 0
              AND s.strike < e.spot
              AND s.bid >= 0
              AND s.ask >= s.bid
              AND s.delta BETWEEN -1.0 AND 0.0
            QUALIFY row_number() OVER (
                PARTITION BY e.trade_id, t.target_otm_pct
                ORDER BY abs(s.strike / e.spot - (1.0 - t.target_otm_pct / 100.0))
            ) = 1
            """
        ).fetchdf()

    join_keys = ["entry_date", "expiration_date", "target_dte"]
    protection["entry_date"] = pd.to_datetime(protection["entry_date"])
    protection["expiration_date"] = pd.to_datetime(protection["expiration_date"])
    combined = protection.merge(
        core[join_keys + ["structure", "core_entry_cash"]],
        on=join_keys,
        how="inner",
    )
    combined["debit_cash"] = combined["debit_points"] * 100.0
    combined["cost_pct_spot_notional"] = combined["debit_points"] / combined["spot"]
    combined["cost_pct_core_credit"] = combined["debit_cash"] / combined["core_entry_cash"]
    combined.to_parquet(CACHE / "protective_put_costs_by_otm.parquet", index=False)

    protection_summary = (
        combined.groupby(["structure", "target_dte", "target_otm_pct"], as_index=False)
        .agg(
            observations=("trade_id", "size"),
            average_actual_otm_pct=("actual_otm_pct", "mean"),
            average_absolute_delta=("absolute_delta", "mean"),
            average_debit_points=("debit_points", "mean"),
            median_debit_points=("debit_points", "median"),
            average_cost_pct_spot_notional=("cost_pct_spot_notional", "mean"),
            average_cost_pct_core_credit=("cost_pct_core_credit", "mean"),
            median_cost_pct_core_credit=("cost_pct_core_credit", "median"),
        )
    )
    protection_summary.to_csv(OUT / "protective_put_cost_by_otm.csv", index=False)

    print("SPREAD ENTRY CREDIT")
    print(spread_summary.to_string(index=False))
    print("\nPROTECTIVE PUT COST — 21 AND 45 DTE")
    print(
        protection_summary[protection_summary["target_dte"].isin([21, 45])].to_string(index=False)
    )


if __name__ == "__main__":
    main()
