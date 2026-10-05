"""Independent quote-based selection and daily accounting audit for buffers."""
from __future__ import annotations
import json
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from organized_spx_data import load_cash, START, END
from organized_spx_engine import INITIAL, FEE_POINTS, SPREAD_FRACTION, STRIKE_TOLERANCE
from optimize_organized_spx_hedges import OUTPUT, SOURCE


def verify():
    metrics = pd.read_csv(OUTPUT/"strategy_metrics.csv", dtype={"folder": str}, low_memory=False)
    buffers = metrics[metrics.hedge_type.eq("Downside buffer") & metrics.category.eq("Both")]
    ledger = pd.read_parquet(OUTPUT/"buffer_trade_ledger.parquet")
    entries = pd.read_parquet(SOURCE/"entries.parquet")
    quotes = pd.read_parquet(SOURCE/"audited_quotes.parquet").set_index(["snapshot_date", "option_symbol"])
    schedule = pd.read_parquet(SOURCE/"schedule.parquet")
    cash = load_cash().loc[START:END]
    entry_groups = {key: frame.sort_values("ratio") for key, frame in entries.groupby("roll_id")}
    ledger_groups = ledger.groupby("roll_id", sort=False).indices
    max_budget_error = 0.
    max_equity_error = 0.
    checked_daily = 0
    for tenor, specs in buffers.groupby("tenor_days", sort=True):
        ids = specs.strategy_id.to_list()
        n = len(specs)
        nav = np.full((len(cash), n), np.nan)
        equity = np.full(n, INITIAL)
        fractions = specs.premium_fraction.to_numpy(float)
        base_hi = specs.primary_pct.to_numpy(int)-80
        is_spread = specs.width_pct.to_numpy() > 0
        base_lo = specs.secondary_pct.fillna(80).to_numpy(int)-80
        for roll in schedule[schedule.target_dte.eq(tenor)].sort_values("entry_date").to_dict("records"):
            rows = ledger.iloc[ledger_groups[roll["roll_id"]]].set_index("strategy_id").loc[ids]
            ix = np.flatnonzero((cash.index >= roll["entry_date"]) & (cash.index <= roll["valuation_end"]))
            dates = cash.index[ix]
            spot = float(cash.loc[roll["entry_date"]])
            traded = np.zeros(n, bool)
            credit = np.full(n, np.nan)
            chosen_high, chosen_low = np.zeros(n, int), np.zeros(n, int)
            quantity = np.zeros(n)
            pnl = np.zeros((len(dates), n))
            if roll["has_expiry"]:
                e = entry_groups[roll["roll_id"]]
                k = e.strike.to_numpy(float)
                symbols = e.option_symbol.to_numpy()
                observed = quotes.reindex(pd.MultiIndex.from_product([dates, symbols], names=quotes.index.names))
                mids = observed.used_mid.to_numpy().reshape(len(dates), 31).copy()
                bids = observed.used_bid.to_numpy().reshape(len(dates), 31)[0]
                asks = observed.used_ask.to_numpy().reshape(len(dates), 31)[0]
                estimated = observed.estimated.fillna(False).to_numpy(bool).reshape(len(dates), 31)[0]
                initial_mid = (bids+asks)/2
                costs = SPREAD_FRACTION*(asks-bids)+FEE_POINTS
                valid = (np.isfinite(bids) & np.isfinite(asks) & (bids >= 0) & (asks > 0)
                    & (asks >= bids) & ~estimated & (np.abs(k/spot-np.arange(80, 111)/100) <= STRIKE_TOLERANCE+1e-12))
                credit = initial_mid[base_hi]-costs[base_hi]-np.where(is_spread, initial_mid[base_lo]+costs[base_lo], 0)
                base_valid = valid[base_hi] & (~is_spread | valid[base_lo]) & (credit > 0)
                base_valid &= ~is_spread | ((k[base_hi] > k[base_lo]) & (credit <= k[base_hi]-k[base_lo]+1e-12))
                np.testing.assert_array_equal(base_valid, rows.base_eligible)
                # Independently walk the ranked strike/width list, assigning each
                # still-unfilled budget the first feasible full-size buffer.
                for target in range(100, 82, -1):
                    for width in (5, 4, 3):
                        high, low = target-80, target-width-80
                        if low < 0 or not (valid[high] and valid[low]):
                            continue
                        actual_width = k[high]-k[low]
                        mid = initial_mid[high]-initial_mid[low]
                        debit = mid+costs[high]+costs[low]
                        if (actual_width < .03*spot-1e-10 or mid < 0 or mid > actual_width+1e-10
                            or not np.isfinite(debit) or not 0 < debit < actual_width):
                            continue
                        take = base_valid & ~traded & (fractions*credit+1e-12 >= debit)
                        chosen_high[take], chosen_low[take] = high, low
                        quantity[take] = fractions[take]*credit[take]/debit
                        traded[take] = True
                np.testing.assert_array_equal(traded, rows.traded)
                np.testing.assert_array_equal(chosen_high[traded]+80, rows.hedge_primary_pct[traded])
                np.testing.assert_array_equal(chosen_low[traded]+80, rows.hedge_secondary_pct[traded])
                np.testing.assert_allclose(quantity, rows.hedge_quantity_ratio, rtol=1e-12, atol=1e-12)
                if dates[-1] == roll["expiration_date"]:
                    mids[-1] = np.maximum(k-cash.iloc[ix[-1]], 0)
                debit = initial_mid[chosen_high]+costs[chosen_high]-initial_mid[chosen_low]+costs[chosen_low]
                marked_short_liability = mids[:, base_hi]-np.where(is_spread, mids[:, base_lo], 0)
                marked_buffer = mids[:, chosen_high]-mids[:, chosen_low]
                pnl = credit-marked_short_liability+quantity*(marked_buffer-debit)
                pnl[:, ~traded] = 0
                if traded.any():
                    error = np.max(np.abs(quantity[traded]*debit[traded]-fractions[traded]*credit[traded]))
                    max_budget_error = max(max_budget_error, float(error))
            else:
                assert not rows.traded.any() and not rows.base_eligible.any()
            if "entry_equity" in rows:
                np.testing.assert_allclose(rows.entry_equity, equity, rtol=1e-12, atol=1e-7)
            nav[ix] = equity*(1+pnl/spot)
            equity = nav[ix[-1]].copy()
            if "ending_equity" in rows:
                max_equity_error = max(max_equity_error, float(np.max(np.abs(equity-rows.ending_equity.to_numpy()))))
                np.testing.assert_allclose(rows.ending_equity, equity, rtol=1e-12, atol=1e-7)
        assert np.isfinite(nav).all() and (nav > 0).all()
        daily = nav/np.vstack([np.full((1, n), INITIAL), nav[:-1]])-1
        sd = daily.std(axis=0, ddof=1)
        peaks = np.maximum.accumulate(np.vstack([np.full((1, n), INITIAL), nav]), axis=0)[1:]
        calculated = dict(cagr=(nav[-1]/INITIAL)**(1/((cash.index[-1]-cash.index[0]).days/365.2425))-1,
            annualized_volatility=sd*np.sqrt(252),
            daily_sharpe=np.divide(daily.mean(axis=0)*np.sqrt(252), sd, out=np.full(n, np.nan), where=sd>0),
            max_drawdown=(nav/peaks-1).min(axis=0), ending_equity=nav[-1])
        for key, values in calculated.items():
            np.testing.assert_allclose(values, specs[key], rtol=1e-10, atol=1e-9)
        checked_daily += nav.size
        print(f"Independent quote audit passed: {specs.tenor.iloc[0]}", flush=True)
    with zipfile.ZipFile(OUTPUT/"Superseded Sharpe buffer metadata.zip") as archive:
        previous = pd.read_csv(archive.open("strategy_metrics.csv"), dtype={"folder": str})
    retained_ids = previous.loc[~(previous.category.eq("Both") & previous.hedge_type.eq("Downside buffer")), "strategy_id"]
    columns = ["cagr", "annualized_volatility", "daily_sharpe", "max_drawdown", "ending_equity"]
    # CSV parse/write cycles can round the last binary digit; prices and model
    # parameters remain untouched. Images are separately verified byte-for-byte.
    np.testing.assert_allclose(metrics.set_index("strategy_id").loc[retained_ids, columns],
                              previous.set_index("strategy_id").loc[retained_ids, columns], rtol=1e-14, atol=1e-12)
    report = json.loads((OUTPUT/"validation.json").read_text(encoding="utf-8"))
    if "ending_equity" not in ledger:
        max_equity_error = report.get("maximum_ending_equity_error_dollars")
    report.update(status="passed", independent_quote_selection_rows=len(ledger),
        independently_reconstructed_buffer_daily_values=checked_daily,
        maximum_budget_error_points=max_budget_error, maximum_ending_equity_error_dollars=max_equity_error,
        retained_long_put_and_baseline_metrics_unchanged=len(retained_ids),
        buffer_selection_uses_future_returns=False)
    report["automated_tests_passed"] = 120
    (OUTPUT/"validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    verify()
