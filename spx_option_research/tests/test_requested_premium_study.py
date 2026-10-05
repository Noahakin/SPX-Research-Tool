from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_requested_premium_study.py"
SPEC = importlib.util.spec_from_file_location("requested_premium_study", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def quote(strike: float, bid: float, ask: float, symbol: str) -> pd.Series:
    return pd.Series(
        {
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "option_symbol": symbol,
        }
    )


class RequestedPremiumAccountingTests(unittest.TestCase):
    def test_short_spread_entry_and_expiration_accounting(self) -> None:
        legs = [
            (quote(101.0, 4.8, 5.2, "SHORT"), -1, "short"),
            (quote(98.0, 2.8, 3.2, "LONG"), 1, "long"),
        ]
        self.assertAlmostEqual(MODULE.entry_cash(legs, MODULE.MID_MODEL), 200.0)
        self.assertLess(
            MODULE.entry_cash(legs, MODULE.NATURAL_MODEL),
            MODULE.entry_cash(legs, MODULE.REALISTIC_MODEL),
        )
        self.assertAlmostEqual(MODULE.expiration_value(legs, 105.0), 0.0)
        self.assertAlmostEqual(MODULE.expiration_value(legs, 99.0), -200.0)
        self.assertAlmostEqual(MODULE.expiration_value(legs, 90.0), -300.0)

    def test_buffer_has_debit_and_bounded_positive_expiry_value(self) -> None:
        legs = [
            (quote(95.0, 3.8, 4.2, "LONG"), 1, "long"),
            (quote(85.0, 0.8, 1.2, "SHORT"), -1, "short"),
        ]
        self.assertAlmostEqual(MODULE.entry_cash(legs, MODULE.MID_MODEL), -300.0)
        self.assertAlmostEqual(MODULE.expiration_value(legs, 100.0), 0.0)
        self.assertAlmostEqual(MODULE.expiration_value(legs, 90.0), 500.0)
        self.assertAlmostEqual(MODULE.expiration_value(legs, 80.0), 1000.0)

    def test_annualization_uses_contract_years(self) -> None:
        frame = pd.DataFrame(
            {
                "dte": [30, 60],
                "premium": [0.01, 0.02],
            }
        )
        expected = 0.03 / (90.0 / 365.2425)
        self.assertAlmostEqual(MODULE.annualized_rate(frame, "premium"), expected)

    def test_strike_selection_uses_supplied_cash_spot(self) -> None:
        puts = pd.DataFrame(
            [
                {"strike": 98.0, "bid": 1.0, "ask": 1.2, "option_symbol": "P98"},
                {"strike": 101.0, "bid": 4.0, "ask": 4.2, "option_symbol": "P101"},
                {"strike": 111.0, "bid": 13.0, "ask": 13.2, "option_symbol": "P111"},
            ]
        )
        structure = MODULE.Structure("test", 1, "short_put_spread", 1.01, 0.98)
        selected = MODULE.signed_legs(puts, 100.0, structure)
        self.assertEqual([float(leg[0]["strike"]) for leg in selected], [101.0, 98.0])

    def test_three_month_cohorts_are_non_overlapping(self) -> None:
        entries = pd.date_range("2024-01-19", periods=6, freq="MS")
        frame = pd.DataFrame(
            {
                "entry_date": entries,
                "expiration_date": entries + pd.DateOffset(months=3),
                "dte": [91] * 6,
                "pnl_mid_cash": [1.0] * 6,
                "pnl_realistic_cash": [1.0] * 6,
                "pnl_natural_cash": [1.0] * 6,
                "premium_mid_pct_spot_notional": [0.01] * 6,
                "premium_realistic_pct_spot_notional": [0.01] * 6,
                "premium_natural_pct_spot_notional": [0.01] * 6,
                "pnl_mid_pct_spot_notional": [-0.005] * 6,
                "pnl_realistic_pct_spot_notional": [-0.005] * 6,
                "pnl_natural_pct_spot_notional": [-0.005] * 6,
            }
        )
        cohorts = MODULE.three_month_roll_cohorts(frame)
        self.assertEqual(cohorts["observations"].tolist(), [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
