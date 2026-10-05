from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_weekly_spread_candidates.py"
SPEC = importlib.util.spec_from_file_location("build_weekly_spread_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def quote(
    strike: float,
    bid: float,
    ask: float,
    symbol: str,
    *,
    expiration: str = "2024-01-12",
    option_type: str = "put",
) -> dict[str, object]:
    return {
        "snapshot_date": pd.Timestamp("2024-01-05"),
        "expiration_date": pd.Timestamp(expiration),
        "settlement": "PM",
        "option_type": option_type,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2.0,
        "implied_volatility": 0.20 + (100.0 - strike) / 1000.0,
        "delta": -0.50 + (100.0 - strike) / 100.0,
        "gamma": 0.01,
        "theta": -0.50,
        "vega": 1.25,
        "volume": 100.0,
        "open_interest": 500.0,
        "option_symbol": symbol,
    }


class FakeArchive:
    def __init__(self, dates: list[str], chain: pd.DataFrame) -> None:
        self.populated_dates = pd.DatetimeIndex(dates)
        self.chain = chain

    def read(self, date: pd.Timestamp) -> pd.DataFrame:
        return self.chain.copy()


class WeeklyCandidateTests(unittest.TestCase):
    def test_fixed_policy_can_retain_trade_when_fees_exceed_premium(self) -> None:
        puts = pd.DataFrame([quote(100, .05, .15, "SHORT"), quote(97, 0, .10, "LONG")])
        kwargs = dict(entry_date=pd.Timestamp("2024-01-05"), scheduled_entry_friday=pd.Timestamp("2024-01-05"),
                      expiration_date=pd.Timestamp("2024-01-12"), scheduled_expiration_friday=pd.Timestamp("2024-01-12"),
                      spot_entry=100.0, spot_expiration=100.0, short_ratio=1.0, width=.03, puts=puts)
        with self.assertRaises(MODULE.CandidateError):
            MODULE.build_candidate_record(**kwargs)
        record = MODULE.build_candidate_record(**kwargs, require_positive_credit=False)
        self.assertAlmostEqual(record["premium_realistic_cash"], -3.0)
        self.assertAlmostEqual(record["pnl_realistic_cash"], -3.0)
        self.assertAlmostEqual(record["max_loss_realistic_pct"], .0303)

    def test_weekly_schedule_uses_prior_session_for_friday_holiday(self) -> None:
        dates = pd.to_datetime(
            ["2024-03-22", "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-04-05"]
        )
        schedule = MODULE.weekly_roll_schedule(dates)
        self.assertEqual(
            schedule["scheduled_friday"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-03-22", "2024-03-29", "2024-04-05"],
        )
        self.assertEqual(
            schedule["roll_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-03-22", "2024-03-28", "2024-04-05"],
        )

    def test_candidate_accounting_and_cash_spot_normalization(self) -> None:
        puts = pd.DataFrame(
            [quote(100.0, 5.0, 5.4, "SHORT"), quote(97.0, 2.4, 2.6, "LONG")]
        )
        record = MODULE.build_candidate_record(
            entry_date=pd.Timestamp("2024-01-05"),
            scheduled_entry_friday=pd.Timestamp("2024-01-05"),
            expiration_date=pd.Timestamp("2024-01-12"),
            scheduled_expiration_friday=pd.Timestamp("2024-01-12"),
            spot_entry=100.0,
            spot_expiration=98.0,
            short_ratio=1.0,
            width=0.03,
            puts=puts,
        )
        self.assertAlmostEqual(record["premium_mid_cash"], 270.0)
        self.assertAlmostEqual(record["premium_realistic_cash"], 252.0)
        self.assertAlmostEqual(record["premium_natural_cash"], 237.0)
        self.assertAlmostEqual(record["short_terminal_intrinsic_cash"], 200.0)
        self.assertAlmostEqual(record["long_terminal_intrinsic_cash"], 0.0)
        self.assertAlmostEqual(record["short_terminal_payoff_cash"], -200.0)
        self.assertAlmostEqual(record["long_terminal_payoff_cash"], 0.0)
        self.assertAlmostEqual(record["pnl_realistic_cash"], 52.0)
        self.assertAlmostEqual(record["pnl_realistic_pct_spot_notional"], 0.0052)
        self.assertAlmostEqual(record["max_loss_realistic_pct"], 0.0048)
        self.assertAlmostEqual(record["short_sell_fill_realistic"], 5.1)
        self.assertAlmostEqual(record["long_buy_fill_realistic"], 2.55)
        self.assertAlmostEqual(record["gross_credit_realistic_cash"], 255.0)
        self.assertAlmostEqual(record["commission_realistic_cash"], 3.0)
        self.assertEqual(record["upper_bid"], record["short_bid"])
        self.assertEqual(record["lower_bid"], record["long_bid"])

    def test_credit_must_be_positive_and_less_than_width(self) -> None:
        puts = pd.DataFrame(
            [quote(100.0, 1.0, 1.2, "SHORT"), quote(97.0, 1.1, 1.3, "LONG")]
        )
        with self.assertRaisesRegex(MODULE.CandidateError, "credit"):
            MODULE.build_candidate_record(
                entry_date=pd.Timestamp("2024-01-05"),
                scheduled_entry_friday=pd.Timestamp("2024-01-05"),
                expiration_date=pd.Timestamp("2024-01-12"),
                scheduled_expiration_friday=pd.Timestamp("2024-01-12"),
                spot_entry=100.0,
                spot_expiration=100.0,
                short_ratio=1.0,
                width=0.03,
                puts=puts,
            )

    def test_missing_grid_point_does_not_remove_valid_candidate(self) -> None:
        chain = pd.DataFrame(
            [
                quote(100.0, 5.0, 5.4, "P100"),
                quote(90.0, 1.0, 1.2, "P90"),
                quote(100.0, 5.1, 5.5, "C100", option_type="call"),
            ]
        )
        archive = FakeArchive(["2024-01-05", "2024-01-12"], chain)
        cash = pd.Series(
            [100.0, 98.0], index=pd.to_datetime(["2024-01-05", "2024-01-12"])
        )
        candidates, missing, weekly, skipped = MODULE.build_weekly_candidate_panel(
            archive,
            cash,
            market_features=pd.DataFrame(index=cash.index),
            candidate_grid=((0.99, 0.03), (0.99, 0.09)),
            progress_every=0,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(float(candidates.iloc[0]["target_width_pct"]), 0.09)
        self.assertEqual(len(missing), 1)
        self.assertIn("not descending", str(missing.iloc[0]["reason"]))
        self.assertEqual(int(weekly.iloc[0]["available_candidates"]), 1)
        self.assertFalse(bool(weekly.iloc[0]["baseline_99_96_available"]))
        self.assertTrue(skipped.empty)

    def test_chain_features_include_skew_and_volume_without_future_data(self) -> None:
        chain = pd.DataFrame(
            [
                quote(100.0, 5.0, 5.2, "P100"),
                quote(95.0, 2.0, 2.2, "P95"),
                quote(100.0, 5.1, 5.3, "C100", option_type="call"),
                quote(100.0, 8.0, 8.2, "P100M", expiration="2024-02-02"),
            ]
        )
        weekly_puts = chain[
            chain["option_type"].eq("put")
            & chain["expiration_date"].eq(pd.Timestamp("2024-01-12"))
        ]
        features = MODULE.chain_features(
            chain,
            weekly_puts,
            entry_date=pd.Timestamp("2024-01-05"),
            expiration_date=pd.Timestamp("2024-01-12"),
            spot=100.0,
        )
        self.assertAlmostEqual(features["same_expiry_atm_put_iv"], 0.20)
        self.assertAlmostEqual(features["same_expiry_95_put_iv"], 0.205)
        self.assertAlmostEqual(features["same_expiry_put_skew_95_minus_atm"], 0.005)
        self.assertAlmostEqual(features["one_month_atm_put_iv"], 0.20)
        self.assertEqual(features["one_month_dte"], 28)
        self.assertEqual(features["chain_put_volume"], 300.0)
        self.assertEqual(features["chain_call_volume"], 100.0)


if __name__ == "__main__":
    unittest.main()
