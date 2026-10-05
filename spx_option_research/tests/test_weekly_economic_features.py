from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/weekly_economic_features.py"
SPEC = importlib.util.spec_from_file_location("weekly_economic_features", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture(weeks: int = 174) -> tuple[pd.DataFrame, pd.Series]:
    entries = pd.date_range("2013-01-04", periods=weeks + 1, freq="W-FRI")
    dates = pd.bdate_range("2012-01-02", entries[-1])
    t = np.arange(len(dates))
    daily_returns = 0.0001 + 0.009 * np.sin(t * 0.47) + 0.004 * np.cos(t * 0.13)
    closes = pd.Series(4000 * np.exp(np.cumsum(daily_returns)), index=dates)
    rows = []
    for entry, expiration in zip(entries[:-1], entries[1:]):
        spot = closes.loc[entry]
        for upper, lower in ((0.99, 0.96), (1.02, 0.99)):
            short_fill = 0.015 * spot
            long_fill = 0.009 * spot
            credit = (short_fill - long_fill - 0.03) / spot
            rows.append({
                "entry_date": entry, "expiration_date": expiration,
                "spot_entry": spot, "spot_expiration": closes.loc[expiration],
                "upper_strike": upper * spot, "lower_strike": lower * spot,
                "dte": 7, "short_sell_fill_realistic": short_fill,
                "long_buy_fill_realistic": long_fill,
                "premium_realistic_pct_spot_notional": credit,
                "premium_realistic_cash": credit * spot * 100,
                "width_pct": upper - lower,
                "max_loss_realistic_pct": upper - lower - credit,
            })
    return pd.DataFrame(rows), closes


class WeeklyEconomicFeatureTests(unittest.TestCase):
    def test_history_uses_weeks_and_strictly_matured_returns(self):
        frame, closes = fixture()
        features = MODULE.economic_features(frame, closes)
        counts = features.drop_duplicates("entry_date").econ_history_count
        self.assertEqual(counts.iloc[0], 0)
        self.assertEqual(counts.iloc[1], 0)
        self.assertEqual(counts.iloc[2], 1)
        self.assertEqual(counts.iloc[105], 104)
        self.assertEqual(counts.iloc[-1], 156)
        self.assertTrue(features.iloc[:210].econ_expected_edge.isna().all())
        self.assertTrue(features.iloc[210:].econ_expected_edge.notna().all())
        history = features.dropna(subset=["econ_history_latest_expiration"])
        self.assertTrue(history.econ_history_latest_expiration.lt(history.entry_date).all())

    def test_edge_decomposes_into_rich_short_cheap_hedge_and_fees(self):
        frame, closes = fixture()
        features = MODULE.economic_features(frame, closes).dropna(subset=["econ_expected_edge"])
        for prefix in ("econ_", "econ_normal_"):
            reconstructed = (features[prefix + "short_fair_edge"]
                             + features[prefix + "long_fair_edge"]
                             - features.econ_commission)
            np.testing.assert_allclose(reconstructed, features[prefix + "expected_edge"], atol=1e-14)
            np.testing.assert_allclose(features.econ_credit - features[prefix + "expected_liability"],
                                       features[prefix + "expected_edge"], atol=1e-14)
        self.assertTrue(features.econ_expected_liability.between(0, features.width_pct + 1e-14).all())
        self.assertTrue(features.econ_tail_liability_95.le(features.width_pct + 1e-14).all())
        self.assertTrue(features.econ_risk_denominator.ge(0.1 * features.width_pct - 1e-14).all())

    def test_appending_and_changing_future_data_cannot_change_earlier_features(self):
        frame, closes = fixture()
        cutoff = frame.entry_date.unique()[135]
        earlier = frame.loc[frame.entry_date.le(cutoff)]
        before = MODULE.economic_features(earlier, closes.loc[:cutoff])
        changed = frame.copy()
        changed.loc[changed.entry_date.ge(cutoff), "spot_expiration"] *= 2.0
        future_closes = closes.copy()
        future_closes.loc[future_closes.index >= cutoff] *= 3.0
        after = MODULE.economic_features(changed, future_closes)
        pd.testing.assert_frame_equal(before.filter(regex="^econ_"),
                                      after.loc[earlier.index].filter(regex="^econ_"))

    def test_expiring_at_current_entry_is_excluded(self):
        frame, closes = fixture()
        cutoff = frame.entry_date.max()
        before = MODULE.economic_features(frame, closes)
        frame.loc[frame.expiration_date.eq(cutoff), "spot_expiration"] *= 0.1
        after = MODULE.economic_features(frame, closes)
        pd.testing.assert_frame_equal(
            before.loc[before.entry_date.eq(cutoff)].filter(regex="^econ_"),
            after.loc[after.entry_date.eq(cutoff)].filter(regex="^econ_"),
        )

    def test_cheaper_hedge_increases_edge_by_exact_execution_saving(self):
        frame, closes = fixture()
        cutoff = frame.entry_date.max()
        before = MODULE.economic_features(frame, closes)
        selected = frame.entry_date.eq(cutoff)
        saving = 0.0004
        frame.loc[selected, "long_buy_fill_realistic"] -= saving * frame.loc[selected, "spot_entry"]
        frame.loc[selected, "premium_realistic_pct_spot_notional"] += saving
        frame.loc[selected, "premium_realistic_cash"] += saving * frame.loc[selected, "spot_entry"] * 100
        frame.loc[selected, "max_loss_realistic_pct"] -= saving
        after = MODULE.economic_features(frame, closes)
        for column in ("econ_long_fair_edge", "econ_expected_edge", "econ_normal_expected_edge"):
            np.testing.assert_allclose(after.loc[selected, column] - before.loc[selected, column], saving)
        np.testing.assert_array_equal(after.loc[selected, "econ_short_fair_edge"],
                                      before.loc[selected, "econ_short_fair_edge"])

    def test_order_index_and_candidate_count_do_not_change_distribution(self):
        frame, closes = fixture()
        before = MODULE.economic_features(frame, closes)
        shuffled = frame.sample(frac=1, random_state=42)
        after = MODULE.economic_features(shuffled, closes)
        pd.testing.assert_index_equal(after.index, shuffled.index)
        pd.testing.assert_frame_equal(before.filter(regex="^econ_"),
                                      after.sort_index().filter(regex="^econ_"))
        one_pair = frame.iloc[::2]
        fewer = MODULE.economic_features(one_pair, closes)
        pd.testing.assert_frame_equal(before.loc[one_pair.index].filter(regex="^econ_"),
                                      fewer.filter(regex="^econ_"))

    def test_tail_probability_mass_and_flat_normal_reference(self):
        values = np.array([[0.0], [0.2], [0.6], [0.8]])
        np.testing.assert_allclose(MODULE.upper_tail_mean(values, 0.375), [(0.8 + 0.5 * 0.6) / 1.5])
        from scipy.stats import norm
        total_vol = np.array([0.03])
        expected_atm_put = norm.cdf(0.015) - norm.cdf(-0.015)
        np.testing.assert_allclose(MODULE._normal_put_payoff(np.array([1.0]), total_vol), expected_atm_put)

    def test_bad_accounting_is_rejected(self):
        frame, closes = fixture(3)
        frame.loc[0, "premium_realistic_cash"] += 3.0
        with self.assertRaisesRegex(ValueError, "does not reconcile"):
            MODULE.economic_features(frame, closes)


if __name__ == "__main__":
    unittest.main()
