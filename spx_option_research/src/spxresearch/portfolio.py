from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .execution import ExecutionModel
from .pricing import intrinsic_value


MULTIPLIER = 100.0


@dataclass(frozen=True)
class OptionLeg:
    option_symbol: str
    quantity: int
    strike: float
    option_type: str
    expiration: pd.Timestamp


@dataclass
class MarkedPosition:
    entry_date: pd.Timestamp
    legs: tuple[OptionLeg, ...]
    entry_cash_flow: float
    contracts: int = 1

    def mark_value(self, quotes: pd.DataFrame) -> float:
        by_symbol = quotes.set_index("option_symbol")
        value = 0.0
        for leg in self.legs:
            if leg.option_symbol not in by_symbol.index:
                raise LookupError(f"Missing daily quote for {leg.option_symbol}")
            quote = by_symbol.loc[leg.option_symbol]
            if isinstance(quote, pd.DataFrame):
                quote = quote.iloc[0]
            value += leg.quantity * self.contracts * float(quote["mid"]) * MULTIPLIER
        return value

    def liquidation_cash_flow(self, quotes: pd.DataFrame, execution: ExecutionModel) -> float:
        by_symbol = quotes.set_index("option_symbol")
        cash = 0.0
        for leg in self.legs:
            quote = by_symbol.loc[leg.option_symbol]
            if isinstance(quote, pd.DataFrame):
                quote = quote.iloc[0]
            cash += execution.liquidation_value(quote, leg.quantity * self.contracts)
        return cash

    def settlement_cash_flow(self, spot: float) -> float:
        cash = 0.0
        for leg in self.legs:
            cash += (
                leg.quantity
                * self.contracts
                * intrinsic_value(spot, leg.strike, leg.option_type)
                * MULTIPLIER
            )
        return cash

    def open_pnl(self, quotes: pd.DataFrame) -> float:
        return self.entry_cash_flow + self.mark_value(quotes)


def put_spread_max_loss(
    short_strike: float,
    long_strike: float,
    entry_credit: float,
    *,
    multiplier: float = MULTIPLIER,
) -> float:
    if long_strike >= short_strike:
        raise ValueError("long put strike must be below short put strike")
    width = short_strike - long_strike
    if entry_credit >= width:
        raise ValueError("entry credit cannot equal or exceed spread width")
    return (width - entry_credit) * multiplier


def position_from_spread(
    selected: object,
    execution: ExecutionModel,
    entry_date: pd.Timestamp,
    contracts: int,
) -> MarkedPosition:
    short = selected.short
    long = selected.long
    entry_cash = execution.cash_flow(short, -contracts) + execution.cash_flow(long, contracts)
    legs = (
        OptionLeg(
            str(short["option_symbol"]),
            -1,
            float(short["strike"]),
            "put",
            pd.Timestamp(short["expiration_date"]),
        ),
        OptionLeg(
            str(long["option_symbol"]),
            1,
            float(long["strike"]),
            "put",
            pd.Timestamp(long["expiration_date"]),
        ),
    )
    return MarkedPosition(entry_date, legs, entry_cash, contracts)

