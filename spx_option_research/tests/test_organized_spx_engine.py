from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import organized_spx_engine as engine


class OrganizedAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.days = pd.bdate_range("2024-01-05", "2024-01-23")
        cls.cash = pd.Series([1000, 1000, 970, 980, 995, 1005, 1008, 1010, 990, 1005, 1020, 1010, 1030], index=cls.days, dtype=float)
        rows, entries, quotes = [], [], []
        for number, (entry, expiry, traded) in enumerate((("2024-01-05", "2024-01-12", True), ("2024-01-12", "2024-01-19", False), ("2024-01-19", "2024-01-26", True))):
            entry, expiry = pd.Timestamp(entry), pd.Timestamp(expiry)
            roll_id = f"fixture{number}"
            rows.append(dict(roll_id=roll_id, entry_date=entry, expiration_date=expiry,
                             valuation_end=min(expiry, cls.days[-1]), target_dte=7, actual_dte=7, has_expiry=traded))
            if not traded:
                continue
            spot = cls.cash.loc[entry]
            for ratio in range(90, 111):
                strike = ratio / 100 * spot
                symbol = f"{roll_id}_put{ratio}"
                mid = max(strike - spot, 0) + 1
                entries.append(dict(roll_id=roll_id, ratio=ratio/100, strike=strike, option_symbol=symbol))
                for day in cls.days[(cls.days >= entry) & (cls.days < expiry)]:
                    value = max(strike - cls.cash.loc[day], 0) + (expiry-day).days / 7
                    quotes.append(dict(snapshot_date=day, option_symbol=symbol, used_mid=value,
                                       used_bid=value-.10, used_ask=value+.10, estimated=False))
        pd.DataFrame(rows).to_parquet(cls.output / "schedule.parquet")
        pd.DataFrame(entries).to_parquet(cls.output / "entries.parquet")
        with patch.object(engine, "START", cls.days[0]), patch.object(engine, "END", cls.days[-1]), patch.object(engine, "load_cash", return_value=cls.cash), patch.object(engine, "write_manifest"):
            engine.simulate(cls.output, pd.DataFrame(quotes))
        cls.metrics = pd.read_csv(cls.output / "strategy_metrics.csv").set_index("strategy_id")
        cls.ledger = pd.read_parquet(cls.output / "trade_ledger.parquet")
        z = np.load(cls.output / "curves.npz")
        cls.curves = pd.DataFrame(z["equity"], index=cls.days, columns=z["strategy_ids"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cost_accounting_and_negative_credit_week_retained(self):
        # Both OTM leg premiums are 1 point. Flat gross spread still trades,
        # paying .05 points of slippage and .015 fee on each leg.
        curve = self.curves["short_99-96_d07"]
        self.assertAlmostEqual(curve.iloc[0], 1e6 * (1 - .13/1000))
        ledger = self.ledger[self.ledger.strategy_id.eq("short_99-96_d07")]
        self.assertTrue(ledger.traded.iloc[0])
        self.assertAlmostEqual(ledger.terminal_return.iloc[0], -.13/1000)

    def test_expiry_settlement_and_cash_cycle_preserve_equity(self):
        # Sell1100/buy1000 at spot1000 receives99.87 net. At expiry1005,
        # payoff95 leaves4.87points. No cost on the following cash cycle.
        curve = self.curves["short_110-100_d07"]
        settled = 1e6 * (1 + 4.87/1000)
        self.assertAlmostEqual(curve.loc["2024-01-12"], settled)
        self.assertAlmostEqual(curve.loc["2024-01-18"], settled)
        self.assertEqual(self.metrics.loc["short_110-100_d07", "cash_cycles"], 1)

    def test_roll_sizing_uses_prior_settlement_before_new_friction(self):
        curve = self.curves["short_110-100_d07"]
        settled = 1e6 * (1 + 4.87/1000)
        expected = settled * (1 - .13/1020)
        self.assertAlmostEqual(curve.loc["2024-01-19"], expected)
        ledger = self.ledger[self.ledger.strategy_id.eq("short_110-100_d07")]
        self.assertAlmostEqual(ledger.entry_equity.iloc[-1], settled)

    def test_open_final_put_uses_mark_not_intrinsic(self):
        # Final90put remains OTM but has3/7points premium before Jan26.
        curve = self.curves["long_90_d07"]
        first = 1e6 * (1 - 1.065/1000)
        expected = first * (1 + (3/7 - 1.065)/1020)
        self.assertAlmostEqual(curve.iloc[-1], expected)
        ledger = self.ledger[self.ledger.strategy_id.eq("long_90_d07")]
        self.assertTrue(ledger.final_position_open.iloc[-1])

    def test_spx_remains_exposed_during_option_cash_cycle(self):
        curve = self.curves["spx_short_110-100_d07"]
        first = 1e6 * (1 + 4.87/1000 + 5/1000)
        self.assertAlmostEqual(curve.loc["2024-01-12"], first)
        self.assertAlmostEqual(curve.loc["2024-01-17"], first * 990/1005)

    def test_missing_daily_marks_do_not_produce_false_sharpe(self):
        values = np.array([999, np.nan, 1010, 1015.0])
        stats = engine.curve_statistics(values, pd.date_range("2024-01-01", periods=4), initial=1000)
        self.assertTrue(np.isnan(stats["daily_sharpe"]))
        self.assertTrue(np.isnan(stats["annualized_volatility"]))
        self.assertTrue(np.isfinite(stats["cagr"]))
        self.assertEqual(stats["missing_daily_marks"], 1)

    def test_sharpe_includes_initial_cost_and_daily_sample_sd(self):
        curve = self.curves["short_110-100_d07"].to_numpy()
        returns = curve / np.r_[1e6, curve[:-1]] - 1
        expected = returns.mean()/returns.std(ddof=1)*np.sqrt(252)
        self.assertAlmostEqual(self.metrics.loc["short_110-100_d07", "daily_sharpe"], expected)
        self.assertAlmostEqual(np.prod(1+returns), curve[-1]/1e6)


class OrganizedQuoteTests(unittest.TestCase):
    def test_adjacent_estimate_repairs_known_crash_day_quote(self):
        chain = pd.DataFrame(dict(strike=[3235,3240,3245], bid=[262.6,.05,272.6], ask=[283.4,288.4,293.4], option_symbol=["low","bad","high"]))
        result = engine.adjacent_estimate(chain,3240)
        self.assertEqual(result["used_mid"],278)
        self.assertEqual(result["lower_symbol"],"low")

    def test_no_extrapolation_or_interpolation_across_large_gaps(self):
        chain = pd.DataFrame(dict(strike=[2800,3500],bid=[1,200],ask=[2,202],option_symbol=["low","high"]))
        self.assertIsNone(engine.adjacent_estimate(chain,3240))
        self.assertIsNone(engine.adjacent_estimate(chain,2700))

    def test_repeated_wide_plateau_is_not_used_as_its_own_donor(self):
        chain = pd.DataFrame(dict(strike=[3400,3405,3410,3415,3420,3425],
                                  bid=[254.5,200.1,200.1,200.1,200.1,276.5],
                                  ask=[269.6,392.1,392.1,392.1,392.1,291.7],
                                  option_symbol=["low","plateau1","target","plateau2","plateau3","high"]))
        result = engine.adjacent_estimate(chain,3410)
        self.assertAlmostEqual(result["used_mid"],270.87)
        self.assertEqual(result["lower_symbol"],"low")
        self.assertEqual(result["upper_symbol"],"high")

    def test_penny_puts_are_not_flagged_as_corrupt(self):
        chain = pd.DataFrame(dict(strike=[900,1000,1100], bid=[0,.05,38], ask=[.05,288.4,0]))
        self.assertEqual(engine.suspect_quotes(chain).tolist(),[False,True,True])

    def test_penny_quote_estimate_allows_observed_coarse_strike_bracket(self):
        chain=pd.DataFrame(dict(strike=[3200,3250,3300],bid=[0,0,0],ask=[.05,0,.05],option_symbol=["low","bad","high"]))
        self.assertAlmostEqual(engine.adjacent_estimate(chain,3250)["used_mid"],.025)
        chain.loc[[0,2],"bid"]=[2,3]
        chain.loc[[0,2],"ask"]=[3,4]
        self.assertIsNone(engine.adjacent_estimate(chain,3250))


if __name__ == "__main__":
    unittest.main()
