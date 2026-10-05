from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from organized_spx_parity import fit_parity, parity_estimate


def synthetic_chain() -> tuple[pd.DataFrame, dict[float, float]]:
    """Observed pairs with known carry, then a contiguous block of broken puts."""
    discount, forward = .995, 3000.0
    rows, true_puts = [], {}
    for strike in np.arange(2700.0, 3301.0, 10.0):
        time_value = 40.0 * np.exp(-abs(strike - forward) / 200.0)
        call = max(discount * (forward - strike), 0.0) + time_value
        put = call + discount * (strike - forward)
        true_puts[float(strike)] = put
        for option_type, midpoint in (("put", put), ("call", call)):
            rows.append(dict(
                snapshot_date=pd.Timestamp("2024-11-29"),
                expiration_date=pd.Timestamp("2024-12-31"),
                option_symbol=f"SPXW 241231{option_type[0].upper()}{int(strike * 1000):08d}",
                option_type=option_type, strike=strike,
                bid=midpoint - .5, ask=midpoint + .5,
                underlying_price=9999.0,
            ))
    chain = pd.DataFrame(rows)
    chain.loc[chain.option_type.eq("put") & chain.strike.ge(3130), "bid"] = 0.0
    return chain, true_puts


class OrganizedParityTests(unittest.TestCase):
    def test_recovers_known_carry_and_puts_beyond_missing_block(self):
        chain, true_puts = synthetic_chain()
        original = chain.copy(deep=True)
        model = fit_parity(chain)
        self.assertIsNotNone(model)
        self.assertAlmostEqual(model["discount_factor"], .995, places=10)
        self.assertAlmostEqual(model["implied_forward"], 3000.0, places=8)
        self.assertAlmostEqual(model["prepaid_forward"], 2985.0, places=8)
        for strike in (3130.0, 3150.0, 3300.0):
            with self.subTest(strike=strike):
                estimate = parity_estimate(model, strike)
                self.assertIsNotNone(estimate)
                self.assertAlmostEqual(estimate["used_mid"], true_puts[strike], places=8)
                self.assertLessEqual(estimate["parity_feasible_lower"], true_puts[strike])
                self.assertGreaterEqual(estimate["parity_feasible_upper"], true_puts[strike])
                self.assertIn("entry fills prohibited", estimate["parity_bid_ask_note"])
        pd.testing.assert_frame_equal(chain, original)

    def test_ignores_expiration_specific_underlying_and_input_order(self):
        chain, _ = synthetic_chain()
        baseline = parity_estimate(chain, 3150.0)
        changed = chain.sample(frac=1.0, random_state=42).copy()
        changed["underlying_price"] = 1.0
        estimate = parity_estimate(changed, 3150.0)
        self.assertAlmostEqual(estimate["used_mid"], baseline["used_mid"], places=9)

    def test_rejects_invalid_or_missing_matching_call(self):
        chain, _ = synthetic_chain()
        target = chain.option_type.eq("call") & chain.strike.eq(3150.0)
        invalid = chain.copy()
        invalid.loc[target, "ask"] = 0.0
        self.assertIsNone(parity_estimate(invalid, 3150.0))
        self.assertIsNone(parity_estimate(chain.loc[~target], 3150.0))

    def test_rejects_too_little_independent_calibration(self):
        chain, _ = synthetic_chain()
        self.assertIsNone(fit_parity(chain.loc[chain.strike.le(2780.0)]))

    def test_rejects_mixed_snapshot_or_expiration(self):
        chain, _ = synthetic_chain()
        for column, other in (("snapshot_date", "2024-11-27"), ("expiration_date", "2025-01-02")):
            with self.subTest(column=column):
                mixed = chain.copy()
                mixed.loc[0, column] = pd.Timestamp(other)
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    fit_parity(mixed)

    def test_rejects_ambiguous_duplicate_contract(self):
        chain, _ = synthetic_chain()
        ambiguous = pd.concat([chain, chain.iloc[[0]]], ignore_index=True)
        self.assertIsNone(fit_parity(ambiguous))

    def test_rejects_pairs_that_cannot_share_a_parity_relationship(self):
        chain, _ = synthetic_chain()
        contradictory = chain.copy()
        bad = contradictory.option_type.eq("put") & contradictory.strike.between(2800.0, 3100.0)
        shifts = np.where((contradictory.loc[bad, "strike"] / 10).astype(int) % 2, 30.0, 0.0)
        contradictory.loc[bad, "bid"] += shifts
        contradictory.loc[bad, "ask"] += shifts
        self.assertIsNone(fit_parity(contradictory))


if __name__ == "__main__":
    unittest.main()
