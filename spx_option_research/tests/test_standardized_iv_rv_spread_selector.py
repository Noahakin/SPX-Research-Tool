from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_standardized_iv_rv_spread_selector.py"
SPEC = importlib.util.spec_from_file_location("standardized_iv_rv_spread_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidates() -> pd.DataFrame:
    rows = []
    for j, date in enumerate(pd.date_range("2024-01-01", periods=4, freq="MS")):
        for rank, spread in enumerate(MODULE.base.SPREADS):
            ivs = ([0.15, 0.20, 0.25, 0.30] if rank == 0 else
                   [0.09, 0.10, 0.11, 0.20] if rank == 1 else
                   [0.08, 0.10, 0.12, 0.01])
            rows.append({
                "entry_date": date, "expiration_date": date + pd.DateOffset(months=1),
                "spread": spread, "target_upper_ratio": 0.98 + rank / 100,
                "short_leg_implied_vol": ivs[j], "long_leg_implied_vol": ivs[j] + 0.01,
                "realized_vol_21d": 0.10,
            })
    return pd.DataFrame(rows)


class StandardizedSignalTests(unittest.TestCase):
    def test_mean_and_sample_std_use_only_previous_observations(self):
        scored = MODULE.standardized_scores(candidates(), lookback=3)
        first = scored[scored.spread.eq("98/95")]
        self.assertTrue(first.richness_score.iloc[:3].isna().all())
        final = first.iloc[-1]
        self.assertAlmostEqual(final.historical_mean_iv_rv, 2.0)
        self.assertAlmostEqual(final.historical_std_iv_rv, 0.5)
        self.assertAlmostEqual(final.richness_score, 2.0)
        self.assertLess(final.history_end_entry, final.entry_date)

    def test_current_realized_volatility_can_change_cross_spread_ranking(self):
        low_rv = candidates()
        high_rv = low_rv.copy()
        high_rv.loc[high_rv.entry_date.eq(high_rv.entry_date.max()), "realized_vol_21d"] = 0.30
        low_scores = MODULE.standardized_scores(low_rv, lookback=3)
        high_scores = MODULE.standardized_scores(high_rv, lookback=3)
        left = MODULE.base.select_trades(low_scores, MODULE.DYNAMIC)
        right = MODULE.base.select_trades(high_scores, MODULE.DYNAMIC)
        self.assertEqual(left[left.portfolio.eq(MODULE.DYNAMIC)].spread.tolist(), ["99/96"])
        self.assertEqual(right[right.portfolio.eq(MODULE.DYNAMIC)].spread.tolist(), ["98/95"])
        for column in ["historical_mean_iv_rv", "historical_std_iv_rv"]:
            pd.testing.assert_series_equal(low_scores[column], high_scores[column])

    def test_appending_future_data_cannot_change_earlier_scores(self):
        original = candidates()
        future = original[original.entry_date.eq(original.entry_date.max())].copy()
        future.entry_date += pd.DateOffset(months=1)
        future.expiration_date += pd.DateOffset(months=1)
        future.short_leg_implied_vol *= 10
        extended = pd.concat([original, future], ignore_index=True)
        left = MODULE.standardized_scores(original, lookback=3)
        right = MODULE.standardized_scores(extended, lookback=3)
        right = right[right.entry_date.le(original.entry_date.max())]
        columns = ["entry_date", "spread", "historical_mean_iv_rv", "historical_std_iv_rv", "richness_score"]
        pd.testing.assert_frame_equal(left[columns].reset_index(drop=True), right[columns].reset_index(drop=True))

    def test_zero_historical_variance_is_not_an_infinite_trade_signal(self):
        frame = candidates()
        frame.short_leg_implied_vol = 0.20
        scored = MODULE.standardized_scores(frame, lookback=3)
        self.assertTrue(scored.richness_score.isna().all())
        with self.assertRaisesRegex(ValueError, "no dates"):
            MODULE.base.select_trades(scored, MODULE.DYNAMIC)


if __name__ == "__main__":
    unittest.main()
