from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_relative_value_spread_selector.py"
SPEC = importlib.util.spec_from_file_location("relative_value_spread_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidates() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    for index, date in enumerate(dates):
        for rank, spread in enumerate(MODULE.SPREADS):
            # Every candidate normally has a different IV/RV ratio. In the
            # final period the lowest-IV candidate becomes relatively rich.
            iv = 0.30 - 0.02 * rank
            if index == 5 and rank == 5:
                iv *= 1.20
            rows.append({
                "entry_date": date, "expiration_date": date + pd.DateOffset(months=1),
                "spread": spread, "target_upper_ratio": 0.98 + rank * 0.01,
                "short_leg_implied_vol": iv, "long_leg_implied_vol": iv + 0.03,
                "realized_vol_21d": 0.15, "spot_entry": 5000.0,
                **{f"pnl_{fill}_pct_spot_notional": 0.01 for fill in MODULE.FILLS},
                **{f"premium_{fill}_pct_spot_notional": 0.015 for fill in MODULE.FILLS},
            })
    return pd.DataFrame(rows)


class RelativeValueSelectorTests(unittest.TestCase):
    def test_history_excludes_current_and_requires_complete_window(self):
        scored = MODULE.history_scores(candidates(), lookback=3)
        first_spread = scored[scored.spread.eq("98/95")]
        self.assertTrue(first_spread.historical_median_iv_rv.iloc[:3].isna().all())
        self.assertEqual(first_spread.historical_median_iv_rv.iloc[3], 2.0)
        final = scored[scored.spread.eq("103/100")].iloc[-1]
        self.assertAlmostEqual(final.historical_median_iv_rv, 0.20 / 0.15)
        self.assertAlmostEqual(final.richness_score, 0.20)
        self.assertLess(final.history_end_entry, final.entry_date)

    def test_relative_richness_can_select_lower_raw_iv_and_matches_benchmark_dates(self):
        scored = MODULE.history_scores(candidates(), lookback=3)
        selected = MODULE.select_trades(scored)
        dynamic = selected[selected.portfolio.eq(MODULE.DYNAMIC)]
        benchmark = selected[selected.portfolio.eq(MODULE.BENCHMARK)]
        self.assertEqual(dynamic.spread.iloc[-1], "103/100")
        self.assertEqual(dynamic.entry_date.tolist(), benchmark.entry_date.tolist())
        self.assertEqual(len(dynamic), 3)
        self.assertTrue(benchmark.spread.eq("99/96").all())

    def test_future_prices_and_outcomes_cannot_change_past_scores_or_choices(self):
        original = candidates()
        changed = original.copy()
        cutoff = pd.Timestamp("2024-05-01")
        changed.loc[changed.entry_date.gt(cutoff), "short_leg_implied_vol"] *= 20
        changed.loc[changed.entry_date.gt(cutoff), "realized_vol_21d"] *= 4
        changed["pnl_realistic_pct_spot_notional"] = -0.5
        left = MODULE.history_scores(original, lookback=3)
        right = MODULE.history_scores(changed, lookback=3)
        columns = ["entry_date", "spread", "historical_median_iv_rv", "richness_score"]
        pd.testing.assert_frame_equal(left[left.entry_date.le(cutoff)][columns], right[right.entry_date.le(cutoff)][columns])
        left_choice = MODULE.select_trades(left)
        right_choice = MODULE.select_trades(right)
        keys = ["portfolio", "entry_date", "spread"]
        pd.testing.assert_frame_equal(left_choice[left_choice.entry_date.le(cutoff)][keys].reset_index(drop=True),
                                      right_choice[right_choice.entry_date.le(cutoff)][keys].reset_index(drop=True))

    def test_monthly_position_sizing_and_risk_metrics_use_current_equity(self):
        selected = MODULE.select_trades(MODULE.history_scores(candidates(), lookback=3))
        for name in (MODULE.DYNAMIC, MODULE.BENCHMARK):
            indices = selected.index[selected.portfolio.eq(name)]
            for fill in MODULE.FILLS:
                selected.loc[indices, f"pnl_{fill}_pct_spot_notional"] = [0.01, -0.02, 0.03]
        paths = MODULE.build_paths(selected)
        dynamic = paths[paths.portfolio.eq(MODULE.DYNAMIC)]
        self.assertAlmostEqual(dynamic.ending_equity_realistic.iloc[-1], 1.01 * 0.98 * 1.03)
        self.assertAlmostEqual(dynamic.contracts_realistic_per_initial_1m.iloc[1], 1.01 * 1_000_000 / 500_000)
        stats = MODULE.performance(paths).set_index("portfolio").loc[MODULE.DYNAMIC]
        returns = np.array([0.01, -0.02, 0.03])
        self.assertAlmostEqual(stats.sharpe_realistic, returns.mean() / returns.std(ddof=1) * math.sqrt(12))
        self.assertAlmostEqual(stats.cagr_realistic, (1.01 * 0.98 * 1.03) ** (1 / stats.elapsed_years) - 1)
        self.assertAlmostEqual(stats.max_drawdown_at_rolls_realistic, -0.02)

    def test_incomplete_candidate_dates_fail_instead_of_silently_biasing_selection(self):
        scored = MODULE.history_scores(candidates(), lookback=3)
        scored = scored.drop(scored.index[-1])
        with self.assertRaisesRegex(ValueError, "all six"):
            MODULE.select_trades(scored)

    def test_simple_cash_return_rv_and_no_future_close_leakage(self):
        daily_dates = pd.date_range("2023-01-02", periods=30, freq="B")
        returns = np.linspace(-0.02, 0.025, 29)
        closes = pd.Series(100 * np.r_[1.0, np.cumprod(1.0 + returns)], index=daily_dates)
        frame = candidates().iloc[:6].copy()
        frame["entry_date"] = daily_dates[24]
        frame["expiration_date"] = daily_dates[29]
        prepared = MODULE.prepare_candidates(frame, closes)
        expected = np.std(returns[3:24], ddof=1) * math.sqrt(252)
        self.assertAlmostEqual(prepared.realized_vol_21d.iloc[0], expected)
        closes.iloc[25:] *= 2
        again = MODULE.prepare_candidates(frame, closes)
        pd.testing.assert_series_equal(prepared.realized_vol_21d, again.realized_vol_21d)


if __name__ == "__main__":
    unittest.main()
