"""Independently reconcile selected hedges against observed quotes and saved NAV."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from organized_spx_data import load_cash
from verify_spx_combinations import verify_statistics

ROOT = Path(__file__).resolve().parents[1]/"results"
OUTPUT = ROOT/"organized_spx_best_hedges"
SOURCE = ROOT/"organized_spx_combinations"
BASE = ROOT/"organized_spx_research"


def verify():
    started = time.monotonic()
    metrics = pd.read_csv(OUTPUT/"strategy_metrics.csv", dtype={"folder": str})
    selected = metrics[metrics.category.eq("Both")].reset_index(drop=True)
    with np.load(OUTPUT/"curves.npz") as archive:
        dates = pd.DatetimeIndex(archive["dates"])
        ids, all_nav = archive["strategy_ids"], archive["equity"]
    with np.load(BASE/"curves.npz") as archive:
        np.testing.assert_array_equal(archive["dates"], dates.to_numpy())
        np.testing.assert_array_equal(archive["strategy_ids"], ids[:2940])
        np.testing.assert_array_equal(archive["equity"], all_nav[:, :2940])
    np.testing.assert_array_equal(ids, metrics.strategy_id)
    assert len(metrics) == 8820 and len(selected) == 5880
    assert all_nav.shape == (2510, 8820)
    stat_error = verify_statistics(all_nav, metrics, dates, 1e6)
    candidates = pd.read_parquet(OUTPUT/"candidate_metrics.parquet", columns=[
        "strategy_id", "premium_fraction", "short_variation", "tenor_days", "hedge_type",
        "daily_sharpe", "cagr", "max_drawdown", "relative_base_coverage", "traded_cycles",
        "status", "selection_eligible", "selected"])
    assert len(candidates) == 283332 and not candidates.strategy_id.duplicated().any()
    independently_eligible = (candidates.status.eq("complete") & candidates.traded_cycles.gt(0)
        & candidates.relative_base_coverage.ge(.90) & np.isfinite(candidates.daily_sharpe))
    np.testing.assert_array_equal(independently_eligible, candidates.selection_eligible)
    keys = ["premium_fraction", "short_variation", "tenor_days", "hedge_type"]
    ranked = candidates[independently_eligible].sort_values(
        ["daily_sharpe", "cagr", "max_drawdown", "strategy_id"], ascending=[False, False, False, True])
    expected = ranked.drop_duplicates(keys)
    assert set(expected.strategy_id) == set(selected.strategy_id) == set(candidates[candidates.selected].strategy_id)
    assert selected.groupby(keys).size().eq(1).all()
    assert set(selected.premium_fraction) == {.05, .10, .15, .20}
    for fraction, group in selected.groupby("premium_fraction"):
        assert len(group) == 1470 and group.short_variation.nunique() == 105
    print("All candidate rankings, selected groups, original curves and daily metrics verified.", flush=True)
    nav = all_nav[:, 2940:]
    strategy_index = pd.Index(selected.strategy_id)
    quotes = pd.read_parquet(SOURCE/"audited_quotes.parquet").set_index(["snapshot_date", "option_symbol"])
    cash = load_cash().reindex(dates)
    schedule = pd.read_parquet(SOURCE/"schedule.parquet").set_index("roll_id")
    state = np.full(len(selected), 1e6)
    count = np.zeros(len(selected), int)
    trade_count = np.zeros(len(selected), int)
    seen_rolls = set()
    maximum_daily_error, maximum_budget_excess = 0., 0.
    rows = 0
    for batch in pq.ParquetFile(OUTPUT/"trade_ledger.parquet").iter_batches(batch_size=33600):
        # Every writer group contains complete cycles, but iterator batches may split
        # one cycle. Accumulate below so verification remains independent of layout.
        frame = batch.to_pandas()
        for roll, ledger in frame.groupby("roll_id", sort=False):
            spec = schedule.loc[roll]
            ci = strategy_index.get_indexer(ledger.strategy_id)
            assert (ci >= 0).all() and len(ci) == len(set(ci))
            for key in zip([roll]*len(ci), ci):
                assert key not in seen_rolls
                seen_rolls.add(key)
            strategy = selected.iloc[ci]
            assert strategy.tenor_days.eq(spec.target_dte).all()
            ix = np.flatnonzero((dates >= spec.entry_date) & (dates <= spec.valuation_end))
            days = dates[ix]
            spot = float(cash.loc[spec.entry_date])
            np.testing.assert_allclose(ledger.entry_equity, state[ci], rtol=1e-12, atol=1e-7)
            np.testing.assert_allclose(ledger.spot_entry, spot, rtol=0, atol=0)
            n = len(ledger)
            cumulative = np.zeros((len(days), n))
            spread = strategy.width_pct.to_numpy() > 0
            buffer = strategy.hedge_width_pct.to_numpy() > 0
            if spec.has_expiry:
                names = ["short", "base_long", "hedge_long", "hedge_short"]
                columns = [ledger[name+"_symbol"].fillna("").to_numpy(str) for name in names]
                symbols = pd.Index(sorted(set(np.concatenate(columns)) - {""}))
                observed = quotes.reindex(pd.MultiIndex.from_product([days, symbols], names=quotes.index.names))
                marks = observed.used_mid.to_numpy().reshape(len(days), len(symbols)).copy()
                entry = observed.iloc[:len(symbols)]
                bid, ask, strikes = (entry[col].to_numpy() for col in ("used_bid", "used_ask", "strike"))
                if days[-1] == spec.expiration_date:
                    marks[-1] = np.maximum(strikes - cash.iloc[ix[-1]], 0)
                mid, costs = (bid+ask)/2, .25*(ask-bid)+.015
                observed_valid = (np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask > 0)
                    & (ask >= bid) & ~entry.estimated.to_numpy(bool))
                indices = [np.maximum(symbols.get_indexer(values), 0) for values in columns]
                a, b, c, d = indices
                active = [np.ones(n, bool), spread, np.ones(n, bool), buffer]
                target_names = ["primary_pct", "secondary_pct", "hedge_primary_pct", "hedge_secondary_pct"]
                valid = []
                for name, positions, on, target in zip(names, indices, active, target_names):
                    np.testing.assert_allclose(ledger.loc[on, name+"_strike"], strikes[positions[on]], rtol=0, atol=1e-10)
                    valid.append(observed_valid[positions] & (np.abs(strikes[positions]/spot - strategy[target].to_numpy()/100) <= .005+1e-12))
                credit = mid[a]-costs[a]-np.where(spread, mid[b]+costs[b], 0)
                debit = mid[c]+costs[c]-np.where(buffer, mid[d]-costs[d], 0)
                base_ok = valid[0] & (~spread | valid[1]) & (credit > 0) & (~spread | ((strikes[a] > strikes[b]) & (credit <= strikes[a]-strikes[b]+1e-12)))
                hedge_mid = mid[c]-np.where(buffer, mid[d], 0)
                eligible = (base_ok & valid[2] & (~buffer | valid[3]) & (debit > 0) & (hedge_mid >= 0)
                    & (~buffer | ((strikes[c] > strikes[d]) & (hedge_mid <= strikes[c]-strikes[d]+1e-12))))
                q = np.zeros(n)
                fraction = strategy.premium_fraction.to_numpy()
                q[eligible] = np.minimum(1, fraction[eligible]*credit[eligible]/debit[eligible])
                np.testing.assert_array_equal(ledger.base_eligible, base_ok)
                np.testing.assert_array_equal(ledger.traded, eligible)
                np.testing.assert_allclose(ledger.net_short_credit_points, credit, rtol=0, atol=1e-10)
                np.testing.assert_allclose(ledger.hedge_debit_points, debit, rtol=0, atol=1e-10)
                np.testing.assert_allclose(ledger.hedge_quantity_ratio, q, rtol=0, atol=1e-12)
                np.testing.assert_allclose(ledger.net_credit_retained_points, np.where(eligible, credit-q*debit, 0), rtol=0, atol=1e-10)
                maximum_budget_excess = max(maximum_budget_excess, float(np.max(np.where(eligible, q*debit-fraction*credit, 0))))
                assert maximum_budget_excess < 1e-10
                # Four signed contract holdings independently valued in cash points.
                cumulative = ((mid[a]-costs[a]-marks[:, a])
                    + np.where(spread, marks[:, b]-mid[b]-costs[b], 0)
                    + q*(marks[:, c]-mid[c]-costs[c]
                    + np.where(buffer, mid[d]-costs[d]-marks[:, d], 0)))/spot
                cumulative[:, ~eligible] = 0
                trade_count[ci] += eligible
            else:
                assert not ledger.traded.any() and ledger.hedge_quantity_ratio.eq(0).all()
            reconstructed = state[ci]*(1+cumulative)
            np.testing.assert_allclose(ledger.ending_equity, reconstructed[-1], rtol=1e-12, atol=1e-7)
            state[ci] = reconstructed[-1]
            # Next entry friction overwrites a settlement-date value in saved NAV.
            keep = np.arange(len(ix)) if ix[-1] == len(dates)-1 else np.arange(len(ix)-1)
            difference = np.abs(reconstructed[keep] - nav[np.ix_(ix[keep], ci)])
            maximum_daily_error = max(maximum_daily_error, float(difference.max(initial=0)))
            np.testing.assert_allclose(reconstructed[keep], nav[np.ix_(ix[keep], ci)], rtol=1e-11, atol=1e-6)
            count[ci] += 1
            rows += len(ledger)
    np.testing.assert_array_equal(count, selected.cycles)
    np.testing.assert_array_equal(trade_count, selected.traded_cycles)
    np.testing.assert_allclose(state, nav[-1], rtol=1e-11, atol=1e-6)
    report = dict(status="passed", original_curves_bitwise_unchanged=2940,
        selected_hedges=len(selected), candidates_ranked=len(candidates),
        daily_values_verified=int(all_nav.size), ledger_rows_verified=rows,
        maximum_daily_nav_error_dollars=maximum_daily_error,
        maximum_metric_error=stat_error, maximum_budget_excess_points=maximum_budget_excess,
        premium_fractions=[.05, .10, .15, .20], short_variations=105,
        selection="Highest full-sample combined daily Sharpe; retrospective",
        minimum_relative_base_coverage=.90, elapsed_seconds=round(time.monotonic()-started, 2))
    (OUTPUT/"validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    verify()
