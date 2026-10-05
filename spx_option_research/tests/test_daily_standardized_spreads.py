from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mark_standardized_spread_daily.py"
SPEC = importlib.util.spec_from_file_location("mark_standardized_spread_daily", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample():
    start, middle, roll, end = pd.to_datetime(["2024-01-19", "2024-01-22", "2024-02-16", "2024-03-15"])
    closes = pd.Series([1000.0, 995.0, 980.0, 1000.0], index=[start, middle, roll, end])
    first_contracts = 10.0
    first_ending = 1_000_000 + first_contracts * (1447 - 2000)
    second_contracts = first_ending / (980 * 100)
    second_ending = first_ending + second_contracts * 947
    trades = pd.DataFrame([
        {"portfolio": "test", "entry_date": start, "expiration_date": roll,
         "spot_entry": 1000.0, "upper_strike": 1000.0, "lower_strike": 970.0,
         "upper_symbol": "S1", "lower_symbol": "L1", "spread": "100/97",
         "premium_mid_cash": 1500.0, "premium_realistic_cash": 1447.0,
         "expiration_value_cash": -2000.0, "entry_equity_realistic": 1.0,
         "ending_equity_realistic": first_ending / 1_000_000,
         "contracts_realistic_per_initial_1m": first_contracts},
        {"portfolio": "test", "entry_date": roll, "expiration_date": end,
         "spot_entry": 980.0, "upper_strike": 980.0, "lower_strike": 950.0,
         "upper_symbol": "S2", "lower_symbol": "L2", "spread": "100/97",
         "premium_mid_cash": 1000.0, "premium_realistic_cash": 947.0,
         "expiration_value_cash": 0.0, "entry_equity_realistic": first_ending / 1_000_000,
         "ending_equity_realistic": second_ending / 1_000_000,
         "contracts_realistic_per_initial_1m": second_contracts},
    ])
    rows = []
    for day, expiry, symbol, strike, mid in [
        (start, roll, "S1", 1000, 20), (start, roll, "L1", 970, 5),
        (middle, roll, "S1", 1000, 24), (middle, roll, "L1", 970, 7),
        (roll, end, "S2", 980, 15), (roll, end, "L2", 950, 5),
    ]:
        rows.append({"snapshot_date": day, "expiration_date": expiry, "option_symbol": symbol,
                     "strike": strike, "bid": mid - 0.5, "ask": mid + 0.5, "mid": float(mid)})
    return trades, pd.DataFrame(rows), closes


class DailySpreadLedgerTests(unittest.TestCase):
    def test_entry_premium_is_offset_by_option_liability_and_costs_are_immediate(self):
        trades, quotes, closes = sample()
        daily, _ = MODULE.simulate_daily(trades, quotes, closes)
        self.assertAlmostEqual(daily.equity.iloc[0], 999_470.0)
        self.assertAlmostEqual(daily.entry_cost_dollars.iloc[0], 530.0)
        self.assertAlmostEqual(daily.daily_return.iloc[0], -530 / 1_000_000)
        self.assertAlmostEqual(daily.option_value_mid.iloc[0], -15_000)

    def test_daily_option_change_generates_pnl_without_recharging_costs(self):
        trades, quotes, closes = sample()
        daily, _ = MODULE.simulate_daily(trades, quotes, closes)
        self.assertAlmostEqual(daily.equity.iloc[1], 997_470.0)
        self.assertAlmostEqual(daily.daily_pnl.iloc[1], -2_000.0)
        self.assertAlmostEqual(daily.daily_return.iloc[1], -2000 / 999_470)
        self.assertEqual(daily.entry_cost_dollars.iloc[1], 0.0)

    def test_expiry_precedes_roll_sizing_and_every_settlement_reconciles(self):
        trades, quotes, closes = sample()
        daily, settlements = MODULE.simulate_daily(trades, quotes, closes)
        settlement_equity = 994_470.0
        q = settlement_equity / 98_000
        self.assertAlmostEqual(daily.contracts.iloc[2], q)
        self.assertAlmostEqual(daily.equity.iloc[2], settlement_equity - q * 53)
        self.assertAlmostEqual(daily.equity.iloc[-1], settlement_equity + q * 947)
        self.assertAlmostEqual(daily.option_value_mid.iloc[-1], 0.0)
        self.assertEqual(len(settlements), 2)
        np.testing.assert_allclose(settlements.reconciliation_error_dollars, 0.0, atol=1e-7)
        self.assertAlmostEqual((1 + daily.daily_return).prod(), daily.equity.iloc[-1] / 1_000_000)

    def test_daily_sharpe_uses_daily_returns_and_sqrt_252(self):
        trades, quotes, closes = sample()
        daily, _ = MODULE.simulate_daily(trades, quotes, closes)
        old = pd.DataFrame([{"portfolio": "test", "sharpe_realistic": 0.5,
                             "max_drawdown_at_rolls_realistic": -0.1,
                             "ending_wealth_realistic": trades.ending_equity_realistic.iloc[-1]}])
        stats = MODULE.metrics(daily, old).iloc[0]
        expected = daily.daily_return.mean() / daily.daily_return.std(ddof=1) * math.sqrt(252)
        self.assertAlmostEqual(stats.daily_sharpe, expected)
        self.assertEqual(stats.daily_observations, 4)

    def test_missing_or_duplicate_quotes_fail_without_forward_filling(self):
        trades, quotes, closes = sample()
        with self.assertRaises(KeyError):
            MODULE.simulate_daily(trades, quotes.drop(index=2), closes)
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.simulate_daily(trades, pd.concat([quotes, quotes.iloc[[0]]]), closes)


if __name__ == "__main__":
    unittest.main()
