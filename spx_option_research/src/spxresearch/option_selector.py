from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectedSpread:
    expiration: pd.Timestamp
    short: pd.Series
    long: pd.Series
    target_dte: int
    actual_dte: int
    width_method: str
    width_value: float


def clean_puts(chain: pd.DataFrame, *, pm_only: bool = False) -> pd.DataFrame:
    if chain.empty:
        return chain.copy()
    valid = (
        chain["option_type"].eq("put")
        & chain["strike"].gt(0)
        & chain["bid"].ge(0)
        & chain["ask"].ge(chain["bid"])
        & chain["delta"].between(-1.0, 0.0, inclusive="both")
    )
    if pm_only:
        valid &= chain["settlement"].eq("PM")
    return chain.loc[valid].copy()


def select_expiration(
    chain: pd.DataFrame,
    target_dte: int,
    *,
    tolerance: int | None = None,
    pm_only: bool = False,
) -> tuple[pd.Timestamp, int]:
    puts = clean_puts(chain, pm_only=pm_only)
    if puts.empty:
        raise LookupError("No valid puts")
    dtes = np.sort(puts["dte"].dropna().astype(int).unique())
    actual = int(dtes[np.argmin(np.abs(dtes - target_dte))])
    if tolerance is not None and abs(actual - target_dte) > tolerance:
        raise LookupError(f"No expiration within {tolerance} days of {target_dte} DTE")
    expiration = pd.Timestamp(
        puts.loc[puts["dte"].eq(actual), "expiration_date"].iloc[0]
    ).normalize()
    return expiration, actual


def select_put_by_delta(
    puts: pd.DataFrame,
    target_abs_delta: float,
    *,
    below_strike: float | None = None,
) -> pd.Series:
    candidates = puts.copy()
    if below_strike is not None:
        candidates = candidates[candidates["strike"] < below_strike]
    candidates = candidates[candidates["delta"].notna()]
    if candidates.empty:
        raise LookupError("No put satisfies strike constraint")
    target_value = abs(float(target_abs_delta))
    if target_value > 1.0:
        target_value /= 100.0
    target = -target_value
    distance = (candidates["delta"] - target).abs()
    return candidates.loc[distance.idxmin()]


def select_put_spread(
    chain: pd.DataFrame,
    *,
    target_dte: int,
    short_abs_delta: float,
    width_method: str,
    width_value: float,
    dte_tolerance: int | None = None,
    pm_only: bool = False,
) -> SelectedSpread:
    expiration, actual_dte = select_expiration(
        chain, target_dte, tolerance=dte_tolerance, pm_only=pm_only
    )
    puts = clean_puts(chain, pm_only=pm_only)
    puts = puts[puts["expiration_date"].eq(expiration)]
    short = select_put_by_delta(puts, short_abs_delta)

    if width_method == "delta":
        short_delta = float(short_abs_delta) / 100.0 if float(short_abs_delta) > 1 else float(short_abs_delta)
        delta_width = float(width_value) / 100.0 if float(width_value) > 1 else float(width_value)
        target_long_delta = max(short_delta - delta_width, 0.005)
        long = select_put_by_delta(puts, target_long_delta, below_strike=float(short["strike"]))
    elif width_method == "percent":
        target_strike = float(short["strike"]) * (1.0 - float(width_value) / 100.0)
        eligible = puts[puts["strike"] < float(short["strike"])]
        if eligible.empty:
            raise LookupError("No lower strike for percent-width spread")
        long = eligible.loc[(eligible["strike"] - target_strike).abs().idxmin()]
    elif width_method == "points":
        target_strike = float(short["strike"]) - float(width_value)
        eligible = puts[puts["strike"] < float(short["strike"])]
        if eligible.empty:
            raise LookupError("No lower strike for point-width spread")
        long = eligible.loc[(eligible["strike"] - target_strike).abs().idxmin()]
    else:
        raise ValueError(f"Unknown width method: {width_method}")

    if float(long["strike"]) >= float(short["strike"]):
        raise LookupError("Selected long strike is not below short strike")
    return SelectedSpread(
        expiration=expiration,
        short=short,
        long=long,
        target_dte=target_dte,
        actual_dte=actual_dte,
        width_method=width_method,
        width_value=float(width_value),
    )
