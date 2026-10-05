from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_dynamic_iv_spread_selector.py"
SPEC = importlib.util.spec_from_file_location("dynamic_iv_spread_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DynamicIvSpreadSelectorTests(unittest.TestCase):
    def test_realized_volatility_uses_trailing_log_returns(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.03, -0.01])
        closes = pd.Series(
            100.0 * np.exp(np.r_[0.0, returns.cumsum()]),
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
        )
        result = MODULE.trailing_realized_volatility(closes, window=4)
        expected = float(returns.std(ddof=1) * math.sqrt(252.0))
        self.assertTrue(result.iloc[:-1].isna().all())
        self.assertAlmostEqual(float(result.iloc[-1]), expected)

    def test_selector_chooses_highest_iv_rv_gap_and_fixed_benchmark(self) -> None:
        dates = pd.to_datetime(["2024-01-19", "2024-02-16"])
        rows = []
        for date_index, date in enumerate(dates):
            for ratio, gap in ((0.98, 0.25), (0.99, 0.15), (1.00, 0.20)):
                rows.append(
                    {
                        "entry_date": date,
                        "expiration_date": date + pd.DateOffset(months=1),
                        "target_upper_ratio": ratio,
                        "iv_rv_pct_deviation": gap + date_index * 10.0 * (ratio - 0.98),
                        "pnl_realistic_pct_spot_notional": 0.01,
                        "spread": f"{ratio:.2f}",
                    }
                )
        candidates = pd.DataFrame(rows)
        selected = MODULE.select_portfolio_trades(candidates)
        dynamic = selected[selected["portfolio"].eq(MODULE.DYNAMIC_NAME)]
        benchmark = selected[selected["portfolio"].eq(MODULE.BENCHMARK_NAME)]
        self.assertEqual(dynamic["target_upper_ratio"].tolist(), [0.98, 1.00])
        self.assertEqual(benchmark["target_upper_ratio"].tolist(), [0.99, 0.99])

    def test_portfolio_paths_compound_100_percent_notional_returns(self) -> None:
        selected = pd.DataFrame(
            {
                "portfolio": [MODULE.DYNAMIC_NAME, MODULE.DYNAMIC_NAME],
                "entry_date": pd.to_datetime(["2024-01-19", "2024-02-16"]),
                "expiration_date": pd.to_datetime(["2024-02-16", "2024-03-15"]),
                "pnl_mid_pct_spot_notional": [0.11, -0.04],
                "pnl_realistic_pct_spot_notional": [0.10, -0.05],
            }
        )
        path = MODULE.portfolio_paths(selected)
        self.assertEqual(path["entry_equity"].tolist(), [1.0, 1.1])
        self.assertAlmostEqual(float(path["ending_equity"].iloc[-1]), 1.045)
        self.assertAlmostEqual(
            float(path["dollar_pnl_per_initial_1m"].sum()), 45_000.0
        )

    def test_percentage_gap_order_is_iv_order_with_common_realized_vol(self) -> None:
        realized = 0.16
        implied = pd.Series([0.18, 0.22, 0.19])
        gaps = implied / realized - 1.0
        self.assertEqual(implied.sort_values().index.tolist(), gaps.sort_values().index.tolist())


if __name__ == "__main__":
    unittest.main()
