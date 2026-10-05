from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import optimize_organized_spx_hedges as model


class BestHedgeTests(unittest.TestCase):
    def cycle(self, entry, terminal, spot=1000., end_spot=920.):
        k = np.arange(80, 111)*spot/100
        mid = np.maximum(k-spot, 0) + np.maximum(.10, 8-np.abs(np.arange(80, 111)-100))
        bid, ask = mid-.025, mid+.025
        friction = np.full(31, .0275)
        marks = np.array([mid, np.maximum(k-end_spot, 0)])
        return dict(ix=np.array([entry, terminal]), spot=spot, k=k, bid=bid, ask=ask,
            mid=mid, friction=friction, valid=np.ones(31, bool), error=np.zeros(31),
            long=(marks-mid-friction)/spot, short=(mid-marks-friction)/spot,
            estimated=np.zeros(31, int), actual_dte=7)

    def spec(self, width=3, hedge=94, hedge_width=0):
        frame = model.candidate_catalog(dict(primary_pct=98, width_pct=width))
        return frame[(frame.premium_fraction == .10) & (frame.hedge_primary_pct == hedge)
                     & (frame.hedge_width_pct == hedge_width)].reset_index(drop=True)

    def test_distinct_long_hedge_price_and_settlement(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-12"])
        metrics, nav = model.simulate_candidates(self.spec(), [self.cycle(0, 1)], dates, 7)
        credit, debit = 6-3-.055, 2+.0275
        q = .10*credit/debit
        self.assertAlmostEqual(nav[0, 0], 1e6*(1-(.055+q*.0275)/1000))
        self.assertAlmostEqual(nav[1, 0], 1e6*(1+(credit-30+q*(20-debit))/1000))
        self.assertAlmostEqual(metrics.mean_premium_fraction_spent.iloc[0], .10)

    def test_single_short_and_two_leg_buffer(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-12"])
        _, nav = model.simulate_candidates(self.spec(width=0, hedge_width=5), [self.cycle(0, 1)], dates, 7)
        credit, debit = 6-.0275, 2-.1+.055
        q = .10*credit/debit
        self.assertAlmostEqual(nav[1, 0], 1e6*(1+(credit-60+q*(20-debit))/1000))

    def test_roll_costs_apply_to_settled_equity(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
        first, second = self.cycle(0, 1), self.cycle(1, 2)
        _, nav = model.simulate_candidates(self.spec(), [first, second], dates, 7)
        q = .1*(3-.055)/(2+.0275)
        settled = 1e6*(1+(3-.055-30+q*(20-2-.0275))/1000)
        self.assertAlmostEqual(nav[1, 0], settled*(1-(.055+q*.0275)/1000))

    def test_bad_hedge_quote_makes_cash_cycle_and_not_selection_eligible(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-12"])
        cycle = self.cycle(0, 1)
        cycle["valid"][14] = False
        metrics, nav = model.simulate_candidates(self.spec(), [cycle], dates, 7)
        np.testing.assert_array_equal(nav, np.full((2, 1), 1e6))
        self.assertEqual(metrics.base_eligible_cycles.iloc[0], 1)
        self.assertFalse(metrics.selection_eligible.iloc[0])

    def test_highest_eligible_sharpe_and_stable_ties(self):
        frame = pd.DataFrame(dict(premium_fraction=[.05]*4, short_variation=["98-95"]*4,
            tenor_days=[7]*4, hedge_type=["Long put"]*4, daily_sharpe=[9, 2, 2, 1],
            cagr=[1, .03, .04, .09], max_drawdown=[-.01]*4,
            strategy_id=["bad_coverage", "a", "b", "c"], selection_eligible=[False, True, True, True]))
        self.assertEqual(model.choose_winners(frame).strategy_id.tolist(), ["b"])

    def test_catalog_contains_all_requested_budgets_and_downside_strikes(self):
        frame = model.candidate_catalog(dict(primary_pct=98, width_pct=3))
        self.assertEqual(set(frame.premium_fraction), {.05, .10, .15, .20})
        self.assertEqual(set(frame.hedge_width_pct), {0, 1, 2, 3, 5, 10})
        self.assertTrue(frame.hedge_primary_pct.between(80, 95).all())
        self.assertTrue(frame.hedge_secondary_pct.dropna().ge(80).all())
        self.assertTrue(frame.folder.str.match(r"(05|10|15|20) percent premium/98-95/Best (long put|put buffer)").all())


if __name__ == "__main__":
    unittest.main()
