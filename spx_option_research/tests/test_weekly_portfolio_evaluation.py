from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/weekly_portfolio_evaluation.py"
SPEC = importlib.util.spec_from_file_location("weekly_portfolio_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def fixture():
    calendar = pd.bdate_range("2024-01-05", "2024-01-26")
    entries = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    records, marks = [], []
    for i, (entry, expiry) in enumerate(zip(entries[:-1], entries[1:])):
        pnl = [0.01, 0.02, -0.005][i]
        cost = [0.001, 0.002, 0.002][i]
        candidate_id = f"candidate_{i}"
        record = {
            "candidate_id": candidate_id, "entry_date": entry, "expiration_date": expiry,
            "width_pct": 0.03, "short_delta": -0.5, "long_delta": -0.2,
        }
        days = calendar[(calendar >= entry) & (calendar <= expiry)]
        realistic = np.linspace(-cost, pnl, len(days))
        for fill, adjustment in (("mid", cost), ("realistic", 0.0), ("natural", -cost)):
            record[f"pnl_{fill}_pct_spot_notional"] = pnl + adjustment
            record[f"max_loss_{fill}_pct"] = 0.02 - adjustment
        records.append(record)
        for j, day in enumerate(days):
            marks.append({"candidate_id": candidate_id, "date": day,
                          "cum_return_mid": realistic[j] + cost,
                          "cum_return_realistic": realistic[j],
                          "cum_return_natural": realistic[j] - cost})
    candidates = pd.DataFrame(records)
    selection = candidates[["entry_date", "expiration_date", "candidate_id"]].copy()
    return selection, candidates, pd.DataFrame(marks), calendar


class WeeklyPortfolioTests(unittest.TestCase):
    def test_double_notional_resets_weekly_and_scales_friction_and_exposure(self):
        selection, candidates, marks, calendar = fixture()
        base, _ = MODULE.simulate_policy(selection, candidates, marks, calendar)
        daily, weekly = MODULE.simulate_policy(selection, candidates, marks, calendar, notional_multiple=2.0)
        self.assertAlmostEqual(daily.equity.iloc[0], 998_000)
        self.assertAlmostEqual(weekly.entry_equity.iloc[1], 1_020_000)
        self.assertAlmostEqual(daily.set_index("date").loc[pd.Timestamp("2024-01-12"), "equity"], 1_020_000 * 0.996)
        self.assertAlmostEqual(daily.equity.iloc[-1], 1_000_000 * 1.02 * 1.04 * 0.99)
        np.testing.assert_allclose(weekly.weekly_return, [0.02, 0.04, -0.01])
        np.testing.assert_allclose(weekly.max_loss_pct, 0.04)
        np.testing.assert_allclose(weekly.net_delta, 0.6)
        np.testing.assert_allclose(weekly.width_pct, 0.03)
        self.assertGreater(float((daily.daily_return - 2 * base.daily_return).abs().max()), 1e-6)
        np.testing.assert_allclose(daily.position_cum_return, 2 * daily.unit_cum_return)
        self.assertTrue(weekly.source_pnl_reconciliation_error.eq(0).all())

    def test_notional_multiple_rejects_invalid_inputs_and_nonpositive_nav(self):
        selection, candidates, marks, calendar = fixture()
        for multiple in (0, -1, np.inf, np.nan):
            with self.assertRaisesRegex(ValueError, "notional multiple"):
                MODULE.simulate_policy(selection, candidates, marks, calendar, notional_multiple=multiple)
        marks.loc[1, "cum_return_realistic"] = -0.6
        with self.assertRaisesRegex(ValueError, "nonpositive marked equity"):
            MODULE.simulate_policy(selection, candidates, marks, calendar, notional_multiple=2.0)

    def test_roll_settles_old_position_then_charges_new_entry(self):
        selection, candidates, marks, calendar = fixture()
        daily, weekly = MODULE.simulate_policy(selection, candidates, marks, calendar)
        self.assertEqual(len(daily), len(calendar))
        self.assertAlmostEqual(daily.equity.iloc[0], 999_000)
        self.assertAlmostEqual(daily.daily_return.iloc[0], -0.001)
        self.assertAlmostEqual(weekly.ending_equity.iloc[0], 1_010_000)
        self.assertAlmostEqual(weekly.entry_equity.iloc[1], 1_010_000)
        self.assertAlmostEqual(daily.set_index("date").loc[pd.Timestamp("2024-01-12"), "equity"],
                               1_010_000 * (1 - 0.002))
        terminal = 1_000_000 * 1.01 * 1.02 * 0.995
        self.assertAlmostEqual(daily.equity.iloc[-1], terminal)
        self.assertAlmostEqual((1 + daily.daily_return).prod(), terminal / 1_000_000)
        self.assertTrue(weekly.source_pnl_reconciliation_error.eq(0).all())

    def test_cash_week_keeps_sessions_and_sizing_equity(self):
        selection, candidates, marks, calendar = fixture()
        selection.loc[1, "candidate_id"] = None
        daily, weekly = MODULE.simulate_policy(selection.set_index("entry_date"), candidates, marks, calendar)
        self.assertEqual(len(daily), len(calendar))
        self.assertEqual(weekly.weekly_return.iloc[1], 0)
        self.assertAlmostEqual(weekly.entry_equity.iloc[2], 1_010_000)
        interior = daily.date.between("2024-01-15", "2024-01-18")
        self.assertTrue(daily.loc[interior, "daily_return"].eq(0).all())
        stats = MODULE.summarize(daily, weekly)
        self.assertAlmostEqual(stats["trade_fraction"], 2 / 3)
        self.assertAlmostEqual(stats["mean_net_delta"], 0.3)
        self.assertAlmostEqual(stats["mean_net_delta_all_weeks"], 0.2)

    def test_all_cash_policy_is_flat_with_undefined_sharpe(self):
        selection, candidates, marks, calendar = fixture()
        selection["candidate_id"] = "CASH"
        daily, weekly = MODULE.simulate_policy(selection, candidates, marks, calendar, initial_equity=2000)
        self.assertTrue(daily.equity.eq(2000).all())
        stats = MODULE.summarize(daily, weekly)
        self.assertEqual(stats["cagr"], 0)
        self.assertEqual(stats["max_drawdown_daily"], 0)
        self.assertEqual(stats["trade_fraction"], 0)
        self.assertTrue(np.isnan(stats["daily_sharpe"]))

    def test_each_fill_scenario_reconciles_and_retains_initial_cost(self):
        selection, candidates, marks, calendar = fixture()
        ending = {}
        for fill in MODULE.FILLS:
            daily, weekly = MODULE.simulate_policy(selection, candidates, marks, calendar, fill)
            expected = 1_000_000 * np.prod(1 + candidates[f"pnl_{fill}_pct_spot_notional"])
            self.assertAlmostEqual(daily.equity.iloc[-1], expected)
            ending[fill] = expected
        self.assertGreater(ending["mid"], ending["realistic"])
        self.assertGreater(ending["realistic"], ending["natural"])

    def test_metrics_use_daily_marks_initial_peak_and_exact_elapsed_time(self):
        selection, candidates, marks, calendar = fixture()
        daily, weekly = MODULE.simulate_policy(selection, candidates, marks, calendar)
        stats = MODULE.summarize(daily, weekly)
        expected_sharpe = daily.daily_return.mean() / daily.daily_return.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(stats["daily_sharpe"], expected_sharpe)
        expected_cagr = (daily.equity.iloc[-1] / 1_000_000) ** (365.2425 / 21) - 1
        self.assertAlmostEqual(stats["cagr"], expected_cagr)
        peaks = np.maximum.accumulate(np.r_[1_000_000, daily.equity])[1:]
        self.assertAlmostEqual(stats["max_drawdown_daily"], np.min(daily.equity / peaks - 1))
        self.assertLessEqual(stats["max_drawdown_daily"], -0.001)

    def test_missing_mark_and_terminal_mismatch_fail(self):
        selection, candidates, marks, calendar = fixture()
        with self.assertRaisesRegex(ValueError, "missing unit marks"):
            MODULE.simulate_policy(selection, candidates, marks.drop(index=1), calendar)
        candidates.loc[0, "pnl_realistic_pct_spot_notional"] += 0.001
        with self.assertRaisesRegex(ValueError, "terminal source P&L"):
            MODULE.simulate_policy(selection, candidates, marks, calendar)

    def test_omitted_week_or_wrong_candidate_dates_fail(self):
        selection, candidates, marks, calendar = fixture()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            MODULE.simulate_policy(selection.drop(index=1), candidates, marks, calendar)
        selection.loc[0, "candidate_id"] = "candidate_1"
        with self.assertRaisesRegex(ValueError, "different trade dates"):
            MODULE.simulate_policy(selection, candidates, marks, calendar)


class WeeklyBootstrapTests(unittest.TestCase):
    def test_paired_returns_preserve_constant_difference_and_reproducibility(self):
        dates = pd.bdate_range("2020-01-02", periods=360).delete([14, 28, 90])
        t = np.arange(len(dates))
        a = 0.0003 + 0.001 * np.sin(t * 0.4)
        left = pd.DataFrame({"date": dates, "daily_return": a})
        right = pd.DataFrame({"date": dates, "daily_return": a - 0.0001})
        first = MODULE.paired_block_bootstrap(left, right)
        second = MODULE.paired_block_bootstrap(left, right)
        self.assertEqual(first, second)
        for key in ("annual_arithmetic_return_difference", "annual_arithmetic_return_difference_ci_low",
                    "annual_arithmetic_return_difference_ci_high"):
            self.assertAlmostEqual(first[key], 0.0252)
        self.assertGreater(first["sharpe_difference_ci_low"], 0)
        self.assertEqual(first["finite_sharpe_draws"], 1000)

    def test_identical_policies_have_zero_difference(self):
        dates = pd.bdate_range("2024-01-02", periods=55)
        daily = pd.DataFrame({"date": dates, "daily_return": np.sin(np.arange(55)) / 1000})
        stats = MODULE.paired_block_bootstrap(daily, daily, draws=100)
        self.assertEqual(stats["sharpe_difference_ci_low"], 0)
        self.assertEqual(stats["sharpe_difference_ci_high"], 0)
        self.assertEqual(stats["annual_arithmetic_return_difference_ci_low"], 0)
        self.assertEqual(stats["annual_arithmetic_return_difference_ci_high"], 0)
        with self.assertRaisesRegex(ValueError, "exactly matching"):
            MODULE.paired_block_bootstrap(daily, daily.iloc[1:])


if __name__ == "__main__":
    unittest.main()
