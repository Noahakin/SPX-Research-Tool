from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from premium_buffer_hedges import buffer_catalog, select_buffers, simulate_buffers


class BufferProtectionTests(unittest.TestCase):
    def cycle(self, first=0, last=1, spot=1000., end_spot=940.):
        k = np.arange(80, 111)*spot/100
        mid = 1.3**(np.arange(80, 111)-90)
        cost = np.full(31, .025)
        marks = np.array([mid, np.maximum(k-end_spot, 0)])
        return dict(ix=np.array([first, last]), spot=spot, k=k, mid=mid, friction=cost,
            valid=np.ones(31, bool), error=np.zeros(31), estimated=np.zeros(31, int),
            long=(marks-mid-cost)/spot, short=(mid-marks-cost)/spot,
            actual_dte=7, roll_id=f"r{first}", entry_date=pd.Timestamp("2024-01-05")+pd.Timedelta(days=7*first),
            expiration_date=pd.Timestamp("2024-01-12")+pd.Timedelta(days=7*first),
            valuation_end=pd.Timestamp("2024-01-12")+pd.Timedelta(days=7*first))

    def test_closest_affordable_and_preferred_five_point_width(self):
        c = self.cycle()
        debit = c["mid"][20]-c["mid"][15]+.05
        pick = select_buffers(c, [debit], minimum_quantity=1)
        self.assertEqual(pick["high"].item()+80, 100)
        self.assertEqual(pick["width"].item(), 5)
        self.assertAlmostEqual(pick["quantity"].item(), 1)

    def test_allow_four_or_three_to_move_closer(self):
        c = self.cycle()
        for width in (3, 4):
            debit = c["mid"][20]-c["mid"][20-width]+.05
            pick = select_buffers(c, [debit], minimum_quantity=1)
            self.assertEqual(pick["high"].item()+80, 100)
            self.assertEqual(pick["width"].item(), width)

    def test_entire_budget_spent_without_one_contract_cap(self):
        c = self.cycle()
        pick = select_buffers(c, [30.], minimum_quantity=1)
        self.assertEqual(pick["high"].item()+80, 100)
        self.assertEqual(pick["width"].item(), 5)
        self.assertGreater(pick["quantity"].item(), 1)
        self.assertAlmostEqual((pick["quantity"]*pick["debit"]).item(), 30.)

    def test_no_narrow_tail_buffer_for_unaffordable_budget(self):
        pick = select_buffers(self.cycle(), [.001], minimum_quantity=1)
        self.assertFalse(pick["traded"].item())
        self.assertEqual(pick["quantity"].item(), 0)

    def test_actual_width_and_entry_quote_screen(self):
        c = self.cycle()
        c["valid"][20] = False
        c["k"][16] = c["k"][19]-29.9
        budget = c["mid"][19]-c["mid"][16]+.05
        pick = select_buffers(c, [budget], minimum_quantity=1)
        self.assertLess(pick["high"].item()+80, 99)
        self.assertGreaterEqual(pick["actual_width_pct"].item(), 3)

    def test_future_returns_do_not_change_selection(self):
        c = self.cycle()
        expected = select_buffers(c, [1., 3., 8.], minimum_quantity=1)
        c["long"][1] = np.linspace(-1000, 1000, 31)
        c["short"][1] *= -100
        actual = select_buffers(c, [1., 3., 8.], minimum_quantity=1)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])

    def test_twenty_percent_spend_expiry_payoff_and_roll_costs(self):
        specs = buffer_catalog().query("primary_pct == 98 and width_pct == 3 and premium_fraction == .20").reset_index(drop=True)
        dates = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
        c1, c2 = self.cycle(), self.cycle(1, 2)
        ledger = []
        metrics, nav = simulate_buffers(specs, [c1, c2], dates, 7, minimum_quantity=1, ledger_writer=ledger.append)
        row = ledger[0].iloc[0]
        self.assertTrue(row.traded)
        self.assertAlmostEqual(row.hedge_debit_points*row.hedge_quantity_ratio, .2*row.net_short_credit_points)
        short_payoff = max(row.short_strike-940, 0)-max(row.base_long_strike-940, 0)
        hedge_payoff = max(row.hedge_long_strike-940, 0)-max(row.hedge_short_strike-940, 0)
        net_payoff = row.net_short_credit_points-short_payoff+row.hedge_quantity_ratio*(hedge_payoff-row.hedge_debit_points)
        expected_settlement = 1e6*(1+net_payoff/1000)
        self.assertAlmostEqual(row.ending_equity, expected_settlement)
        second_cost = .05*(1+ledger[1].hedge_quantity_ratio.iloc[0])/1000
        self.assertAlmostEqual(nav[1, 0], expected_settlement*(1-second_cost))
        self.assertAlmostEqual(metrics.mean_premium_fraction_spent.item(), .2)

    def test_invalid_or_unaffordable_roll_is_cash(self):
        specs = buffer_catalog().query("primary_pct == 90 and width_pct == 0 and premium_fraction == .05").reset_index(drop=True)
        c = self.cycle()
        c["valid"][:] = False
        metrics, nav = simulate_buffers(specs, [c], pd.to_datetime(["2024-01-05", "2024-01-12"]), 7, minimum_quantity=1)
        np.testing.assert_array_equal(nav, [[1e6], [1e6]])
        self.assertEqual(metrics.cash_cycles.item(), 1)


if __name__ == "__main__":
    unittest.main()
