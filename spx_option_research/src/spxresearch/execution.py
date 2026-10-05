from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExecutionModel:
    """Quote-aware execution model for a single option contract."""

    spread_fraction: float
    commission_per_contract: float
    multiplier: float = 100.0

    def fill(self, quote: pd.Series, quantity: int) -> float:
        if quantity == 0:
            return 0.0
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        if bid < 0 or ask < bid:
            raise ValueError("Invalid bid/ask quote")
        mid = (bid + ask) / 2.0
        half_spread = (ask - bid) / 2.0
        direction = 1.0 if quantity > 0 else -1.0
        return mid + direction * self.spread_fraction * 2.0 * half_spread

    def cash_flow(self, quote: pd.Series, quantity: int) -> float:
        """Cash received (positive) or paid (negative), including commission."""
        price = self.fill(quote, quantity)
        gross = -quantity * price * self.multiplier
        costs = abs(quantity) * self.commission_per_contract
        return gross - costs

    def liquidation_value(self, quote: pd.Series, quantity: int) -> float:
        """Cash flow from closing an existing signed quantity."""
        return self.cash_flow(quote, -quantity)

