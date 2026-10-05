from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spxresearch.execution import ExecutionModel
from spxresearch.core_engine import build_daily_mtm, build_dynamic_exit_outcomes
from spxresearch.hedges import hedge_parameter_grid
from spxresearch.option_selector import select_expiration, select_put_spread
from spxresearch.optimizer import weekly_entry_dates
from spxresearch.portfolio import MarkedPosition, OptionLeg, put_spread_max_loss
from spxresearch.pricing import black_scholes_european, intrinsic_value
from spxresearch.robustness import probabilistic_sharpe_ratio, stationary_block_bootstrap


def toy_chain() -> pd.DataFrame:
    rows = []
    for dte, expiry in [(29, "2024-02-01"), (31, "2024-02-03")]:
        for strike, delta, bid, ask in [
            (100.0, -0.50, 5.0, 5.2),
            (95.0, -0.30, 2.8, 3.0),
            (90.0, -0.15, 1.1, 1.3),
            (85.0, -0.07, 0.4, 0.6),
        ]:
            rows.append(
                {
                    "option_type": "put",
                    "strike": strike,
                    "delta": delta,
                    "bid": bid,
                    "ask": ask,
                    "dte": dte,
                    "expiration_date": pd.Timestamp(expiry),
                    "settlement": "PM",
                }
            )
    return pd.DataFrame(rows)


class SelectionTests(unittest.TestCase):
    def test_nearest_expiration(self) -> None:
        expiration, dte = select_expiration(toy_chain(), 30)
        self.assertEqual(dte, 29)
        self.assertEqual(expiration, pd.Timestamp("2024-02-01"))

    def test_delta_width_and_strike_order(self) -> None:
        spread = select_put_spread(
            toy_chain(),
            target_dte=30,
            short_abs_delta=0.30,
            width_method="delta",
            width_value=15,
        )
        self.assertEqual(float(spread.short["strike"]), 95.0)
        self.assertEqual(float(spread.long["strike"]), 90.0)

    def test_weekly_entries_use_first_available_session(self) -> None:
        dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-08"])
        selected = weekly_entry_dates(dates, preferred_weekday=0)
        self.assertEqual(list(selected), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-08")])


class ExecutionTests(unittest.TestCase):
    def test_mid_fill_and_round_trip(self) -> None:
        quote = pd.Series({"bid": 2.0, "ask": 2.4})
        model = ExecutionModel(0.0, 0.0)
        self.assertAlmostEqual(model.cash_flow(quote, -1), 220.0)
        self.assertAlmostEqual(model.liquidation_value(quote, -1), -220.0)

    def test_realistic_fill_is_worse_than_mid(self) -> None:
        quote = pd.Series({"bid": 2.0, "ask": 2.4})
        ideal = ExecutionModel(0.0, 0.0)
        realistic = ExecutionModel(0.25, 1.5)
        self.assertLess(realistic.cash_flow(quote, -1), ideal.cash_flow(quote, -1))
        self.assertLess(realistic.cash_flow(quote, 1), ideal.cash_flow(quote, 1))


class PricingAndAccountingTests(unittest.TestCase):
    def test_put_call_parity(self) -> None:
        call = black_scholes_european(100, 100, 1, 0.04, 0.01, 0.20, "call")
        put = black_scholes_european(100, 100, 1, 0.04, 0.01, 0.20, "put")
        parity = 100 * np.exp(-0.01) - 100 * np.exp(-0.04)
        self.assertAlmostEqual(call.price - put.price, parity, places=8)

    def test_defined_spread_payoff(self) -> None:
        self.assertEqual(intrinsic_value(80, 100, "put"), 20)
        self.assertEqual(intrinsic_value(80, 90, "put"), 10)
        self.assertAlmostEqual(put_spread_max_loss(100, 90, 2), 800)

    def test_daily_mark_and_settlement(self) -> None:
        position = MarkedPosition(
            pd.Timestamp("2024-01-02"),
            (
                OptionLeg("SHORT", -1, 100, "put", pd.Timestamp("2024-02-02")),
                OptionLeg("LONG", 1, 90, "put", pd.Timestamp("2024-02-02")),
            ),
            entry_cash_flow=200.0,
        )
        marks = pd.DataFrame(
            [
                {"option_symbol": "SHORT", "mid": 5.0},
                {"option_symbol": "LONG", "mid": 2.0},
            ]
        )
        self.assertAlmostEqual(position.open_pnl(marks), -100.0)
        self.assertAlmostEqual(position.entry_cash_flow + position.settlement_cash_flow(80), -800.0)


class EngineTests(unittest.TestCase):
    def test_dynamic_exits_and_daily_marks_have_separate_outputs(self) -> None:
        connection = duckdb.connect()
        connection.execute(
            """
            CREATE TABLE surface (
                option_symbol VARCHAR, trade_date DATE, dte INTEGER, spot DOUBLE,
                delta DOUBLE, bid DOUBLE, ask DOUBLE, mid DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO surface VALUES
              ('S', '2024-01-02', 2, 100, -0.30, 3.8, 4.2, 4.0),
              ('L', '2024-01-02', 2, 100, -0.10, 0.8, 1.2, 1.0),
              ('S', '2024-01-03', 1, 102, -0.40, 1.8, 2.2, 2.0),
              ('L', '2024-01-03', 1, 102, -0.08, 0.4, 0.6, 0.5),
              ('S', '2024-01-04', 0, 105, -0.80, 0.0, 0.1, 0.05),
              ('L', '2024-01-04', 0, 105,  0.00, 0.0, 0.1, 0.05)
            """
        )
        candidate = pd.DataFrame(
            [{
                "trade_id": "t1", "base_strategy_id": "core", "entry_date": pd.Timestamp("2024-01-02"),
                "expiration_date": pd.Timestamp("2024-01-04"), "short_symbol": "S", "long_symbol": "L",
                "short_strike": 100.0, "long_strike": 90.0, "entry_credit_realistic": 3.5,
                "max_loss_realistic": 650.0, "contracts_realistic": 1,
            }]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.parquet"
            dynamic = root / "dynamic.parquet"
            daily = root / "daily.parquet"
            candidate.to_parquet(candidates, index=False)
            build_dynamic_exit_outcomes(connection, candidates, dynamic)
            outcomes = pd.read_parquet(dynamic)
            exits = outcomes.set_index("exit_rule")["exit_date"].astype("datetime64[ns]")
            self.assertEqual(exits["capture_50"], pd.Timestamp("2024-01-03"))
            self.assertEqual(exits["capture_75"], pd.Timestamp("2024-01-04"))
            self.assertEqual(exits["short_delta_75"], pd.Timestamp("2024-01-04"))
            build_daily_mtm(connection, dynamic, daily)
            marks = pd.read_parquet(daily)
            self.assertEqual(set(marks["exit_rule"]), {"capture_50", "capture_75", "short_delta_75"})
            self.assertEqual(len(marks), 8)
        connection.close()


class ResearchDesignTests(unittest.TestCase):
    def test_hedge_grid_respects_archive_maturity(self) -> None:
        config = {
            "hedge_dtes": [14, 90, 180, 270, 365],
            "hedge_deltas": [5, 25, 50],
        }
        grid = hedge_parameter_grid(config, maximum_dte=184)
        self.assertTrue(grid)
        self.assertLessEqual(max(int(row["target_dte"]) for row in grid), 184)
        self.assertFalse(any(int(row["target_dte"]) in {270, 365} for row in grid))

    def test_bootstrap_is_deterministic_and_psr_rewards_positive_returns(self) -> None:
        returns = pd.Series(
            np.tile([0.001, -0.0005, 0.0008, 0.0002], 100),
            index=pd.bdate_range("2020-01-01", periods=400),
        )
        first = stationary_block_bootstrap(returns, samples=20, block_length=10, seed=1729)
        second = stationary_block_bootstrap(returns, samples=20, block_length=10, seed=1729)
        pd.testing.assert_frame_equal(first, second)
        self.assertGreater(probabilistic_sharpe_ratio(returns), 0.99)


if __name__ == "__main__":
    unittest.main()
