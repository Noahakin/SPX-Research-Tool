from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_weekly_daily_marks.py"
SPEC = importlib.util.spec_from_file_location("build_weekly_daily_marks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample():
    entry, middle, expiry = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-12"])
    cash = pd.Series([1000.0, 995.0, 980.0], index=[entry, middle, expiry])
    candidate = {"candidate_id": "week1_100_97", "entry_date": entry, "expiration_date": expiry,
                 "spot_entry": 1000.0, "short_strike": 1000.0, "long_strike": 970.0,
                 "short_symbol": "SHORT1", "long_symbol": "LONG1"}
    for fill, credit in (("mid", 15.0), ("realistic", 14.47), ("natural", 13.97)):
        candidate[f"premium_{fill}_pct_spot_notional"] = credit / 1000
        candidate[f"pnl_{fill}_pct_spot_notional"] = (credit - 20) / 1000
    rows = []
    for day, symbol, strike, mid in [(entry, "SHORT1", 1000, 20), (entry, "LONG1", 970, 5),
                                     (middle, "SHORT1", 1000, 24), (middle, "LONG1", 970, 7)]:
        rows.append({"snapshot_date": day, "expiration_date": expiry, "option_symbol": symbol,
                     "strike": strike, "bid": mid - .5, "ask": mid + .5})
    return pd.DataFrame([candidate]), pd.DataFrame(rows), cash


class WeeklyDailyUnitMarkTests(unittest.TestCase):
    def test_entry_credit_offsets_liability_and_friction_is_immediate(self):
        candidates, quotes, cash = sample()
        marks = MODULE.unit_marks_from_quotes(candidates, quotes, cash)
        first = marks.iloc[0]
        self.assertAlmostEqual(first.cum_return_mid, 0)
        self.assertAlmostEqual(first.cum_return_realistic, -.53 / 1000)
        self.assertAlmostEqual(first.cum_return_natural, -1.03 / 1000)
        self.assertEqual(len(marks), 3)

    def test_daily_change_counts_option_pnl_without_recharging_cost(self):
        candidates, quotes, cash = sample()
        marks = MODULE.unit_marks_from_quotes(candidates, quotes, cash)
        self.assertAlmostEqual(marks.cum_return_realistic.iloc[1], (14.47 - 17) / 1000)
        self.assertAlmostEqual(marks.cum_return_realistic.iloc[1] - marks.cum_return_realistic.iloc[0], -2 / 1000)

    def test_expiry_uses_cash_intrinsic_without_expiry_option_quote(self):
        candidates, quotes, cash = sample()
        marks = MODULE.unit_marks_from_quotes(candidates, quotes, cash)
        self.assertTrue(marks.is_expiration.iloc[-1])
        self.assertEqual(marks.spread_mid_points.iloc[-1], 20.0)
        for fill in MODULE.FILLS:
            self.assertAlmostEqual(marks[f"cum_return_{fill}"].iloc[-1], candidates[f"pnl_{fill}_pct_spot_notional"].iloc[0])
        self.assertFalse(marks.mid_outside_payoff_bounds.any())

    def test_wrong_expected_settlement_fails(self):
        candidates, quotes, cash = sample()
        candidates.loc[0, "pnl_realistic_pct_spot_notional"] += .001
        with self.assertRaisesRegex(ValueError, "settlement"):
            MODULE.unit_marks_from_quotes(candidates, quotes, cash)

    def test_missing_duplicate_or_bad_quotes_fail_without_filling(self):
        candidates, quotes, cash = sample()
        with self.assertRaisesRegex(LookupError, "required quote absent"):
            MODULE.unit_marks_from_quotes(candidates, quotes.drop(index=2), cash)
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.unit_marks_from_quotes(candidates, pd.concat([quotes, quotes.iloc[[0]]]), cash)
        bad = quotes.copy()
        bad.loc[2, "ask"] = bad.loc[2, "bid"] - 1
        with self.assertRaisesRegex(LookupError, "invalid"):
            MODULE.unit_marks_from_quotes(candidates, bad, cash)

    def test_quote_contract_identity_is_checked(self):
        candidates, quotes, cash = sample()
        wrong_strike = quotes.copy()
        wrong_strike.loc[2, "strike"] += 5
        with self.assertRaisesRegex(ValueError, "strike"):
            MODULE.unit_marks_from_quotes(candidates, wrong_strike, cash)
        wrong_expiry = quotes.copy()
        wrong_expiry.loc[2, "expiration_date"] += pd.Timedelta(days=7)
        with self.assertRaisesRegex(ValueError, "expiration"):
            MODULE.unit_marks_from_quotes(candidates, wrong_expiry, cash)

    def test_out_of_bounds_midpoint_is_flagged_and_not_silently_clipped(self):
        candidates, quotes, cash = sample()
        quotes.loc[2, ["bid", "ask"]] = [37.5, 38.5]  # Short mid 38 minus long mid 7 = 31; width 30.
        marks = MODULE.unit_marks_from_quotes(candidates, quotes, cash)
        middle = marks.iloc[1]
        self.assertTrue(middle.mid_outside_payoff_bounds)
        self.assertEqual(middle.spread_mid_points, 31)
        self.assertEqual(middle.bounded_spread_mid_points, 30)
        self.assertAlmostEqual(middle.cum_return_mid, (15 - 31) / 1000)

    def test_quote_requirements_share_legs_and_exclude_expiration(self):
        candidates, _, cash = sample()
        extra = candidates.copy()
        extra["candidate_id"] = "second_candidate_same_legs"
        wanted = MODULE.quote_requirements(pd.concat([candidates, extra]), cash.index)
        self.assertEqual(len(wanted), 2)
        self.assertNotIn(candidates.expiration_date.iloc[0], wanted)
        self.assertEqual(sum(map(len, wanted.values())), 4)

    def test_missing_expiration_cash_price_fails(self):
        candidates, quotes, cash = sample()
        with self.assertRaisesRegex(ValueError, "expiration_date"):
            MODULE.unit_marks_from_quotes(candidates, quotes, cash.iloc[:-1])


if __name__ == "__main__":
    unittest.main()
