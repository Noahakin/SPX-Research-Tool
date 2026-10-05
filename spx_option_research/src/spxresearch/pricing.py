from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class EuropeanGreeks:
    price: float
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_vol_point: float
    rho_per_rate_point: float


def black_scholes_european(
    spot: float,
    strike: float,
    years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    option_type: str,
) -> EuropeanGreeks:
    """European Black-Scholes value and Greeks with continuous dividends."""
    kind = option_type.lower()
    if kind not in {"call", "put", "c", "p"}:
        raise ValueError("option_type must be call or put")
    if spot <= 0 or strike <= 0 or volatility <= 0 or years <= 0:
        raise ValueError("spot, strike, volatility, and years must be positive")

    root_t = math.sqrt(years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * years
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    discount_r = math.exp(-rate * years)
    discount_q = math.exp(-dividend_yield * years)

    if kind in {"call", "c"}:
        price = spot * discount_q * norm.cdf(d1) - strike * discount_r * norm.cdf(d2)
        delta = discount_q * norm.cdf(d1)
        rho = strike * years * discount_r * norm.cdf(d2) / 100.0
        theta = (
            -(spot * discount_q * norm.pdf(d1) * volatility) / (2.0 * root_t)
            - rate * strike * discount_r * norm.cdf(d2)
            + dividend_yield * spot * discount_q * norm.cdf(d1)
        ) / 365.0
    else:
        price = strike * discount_r * norm.cdf(-d2) - spot * discount_q * norm.cdf(-d1)
        delta = discount_q * (norm.cdf(d1) - 1.0)
        rho = -strike * years * discount_r * norm.cdf(-d2) / 100.0
        theta = (
            -(spot * discount_q * norm.pdf(d1) * volatility) / (2.0 * root_t)
            + rate * strike * discount_r * norm.cdf(-d2)
            - dividend_yield * spot * discount_q * norm.cdf(-d1)
        ) / 365.0

    gamma = discount_q * norm.pdf(d1) / (spot * volatility * root_t)
    vega = spot * discount_q * norm.pdf(d1) * root_t / 100.0
    return EuropeanGreeks(price, delta, gamma, theta, vega, rho)


def intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    if option_type.lower() in {"put", "p"}:
        return max(strike - spot, 0.0)
    if option_type.lower() in {"call", "c"}:
        return max(spot - strike, 0.0)
    raise ValueError("option_type must be call or put")

