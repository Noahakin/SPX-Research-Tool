"""Same-snapshot put-call parity estimates for documented daily-mark repairs.

P(K) = C(K) + discount_factor * K - prepaid_forward. Both coefficients are
calibrated from observed paired quotes in the same expiration. Neither cash
SPX nor the source's expiration-specific underlying field is an input.

Synthetic bid/ask sides are call sides plus fitted carry. They are not observed
put quotes and must never authorize an entry fill. Feasible price bounds and
calibration-window sensitivity are reported separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linprog

from organized_spx_data import DATA_ROOT, DEFAULT_OUTPUT, SPXSurfaceArchive


METHOD = "Same-day same-expiry calibrated put-call parity estimate; daily mark only"


def good_quotes(frame: pd.DataFrame) -> pd.Series:
    finite = np.isfinite(frame[["strike", "bid", "ask"]]).all(axis=1)
    valid = finite & frame.strike.gt(0) & frame.bid.ge(0) & frame.ask.gt(0) & frame.ask.ge(frame.bid)
    implausible = frame.ask.ge(25) & (frame.bid.le(.5) | frame.bid.lt(frame.ask * .02))
    return valid & ~implausible


def _fit(frame: pd.DataFrame) -> dict | None:
    if len(frame) < 12 or frame.strike.max() - frame.strike.min() < 50:
        return None
    x = frame.strike.to_numpy(float)
    y = frame.parity_mid.to_numpy(float)
    weights = 1 / np.maximum(frame.parity_half_spread.to_numpy(float), .05) ** 2
    center = float(np.average(x, weights=weights))
    design = np.column_stack([np.ones(len(frame)), x - center])
    coefficients = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    intercept = float(coefficients[0] - coefficients[1] * center)
    slope = float(coefficients[1])
    residual = y - (intercept + slope * x)
    if not np.isfinite([intercept, slope]).all() or not .95 <= slope <= 1.02:
        return None
    return dict(intercept=intercept, discount_factor=slope, prepaid_forward=-intercept,
                implied_forward=-intercept / slope, residual_rms=float(np.sqrt(np.mean(residual ** 2))),
                weighted_residual_rms=float(np.sqrt(np.average(residual ** 2, weights=weights))),
                max_abs_residual=float(np.abs(residual).max()))


def fit_parity(chain: pd.DataFrame) -> dict[str, Any] | None:
    """Fit one expiration; return None when its independent evidence is weak."""
    required = {"option_symbol", "option_type", "strike", "bid", "ask"}
    if not required.issubset(chain.columns):
        raise ValueError(f"Parity source lacks {sorted(required - set(chain.columns))}")
    if "expiration_date" in chain and chain.expiration_date.nunique() > 1:
        raise ValueError("Parity calibration must use exactly one expiration")
    if "snapshot_date" in chain and chain.snapshot_date.nunique() > 1:
        raise ValueError("Parity calibration must use exactly one snapshot")
    source = chain.copy()
    source["option_symbol"] = source.option_symbol.astype(str).str.strip()
    source["option_type"] = source.option_type.astype(str).str.lower()
    for column in ("strike", "bid", "ask"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.loc[source.option_symbol.str.startswith("SPXW")]
    if source.duplicated(["strike", "option_type"]).any():
        return None
    source["mid"] = (source.bid + source.ask) / 2
    source["good"] = good_quotes(source)
    puts = source.loc[source.option_type.eq("put")]
    calls = source.loc[source.option_type.eq("call")]
    pairs = puts.merge(calls, on="strike", suffixes=("_put", "_call"), validate="one_to_one")
    pairs = pairs.loc[pairs.good_put & pairs.good_call & pairs.bid_put.gt(0) & pairs.bid_call.gt(0)].copy()
    if len(pairs) < 12:
        return None
    pairs["parity_mid"] = pairs.mid_put - pairs.mid_call
    pairs["parity_low"] = pairs.bid_put - pairs.ask_call
    pairs["parity_high"] = pairs.ask_put - pairs.bid_call
    pairs["parity_half_spread"] = (pairs.parity_high - pairs.parity_low) / 2
    # DF=1 is used solely as a centering proxy; the fitted valuation uses a
    # free discount factor and a free prepaid forward.
    proxy = float((pairs.strike - pairs.parity_mid).median())
    if not np.isfinite(proxy) or proxy <= 0:
        return None
    primary_mask = pairs.strike.between(.9 * proxy, 1.1 * proxy)
    primary = pairs.loc[primary_mask]
    fitted = _fit(primary)
    if fitted is None or fitted["residual_rms"] > 2:
        return None
    fitted_parity = fitted["intercept"] + fitted["discount_factor"] * pairs.strike
    if (fitted_parity.lt(pairs.parity_low - .10) | fitted_parity.gt(pairs.parity_high + .10)).any():
        return None
    variants = []
    for label, subset in (("all_valid_pairs", pairs), ("seven_percent_window", pairs.loc[pairs.strike.between(.93 * proxy, 1.07 * proxy)])):
        variant = _fit(subset)
        if variant is not None:
            variants.append(dict(label=label, **variant))
    if len(variants) < 2:
        return None
    pairs["used_in_primary_fit"] = primary_mask
    pairs["fitted_parity"] = fitted_parity
    pairs["residual"] = pairs.parity_mid - pairs.fitted_parity
    matrix = np.column_stack([pairs.strike.to_numpy(float), -np.ones(len(pairs))])
    constraints = np.r_[matrix, -matrix]
    limits = np.r_[pairs.parity_high.to_numpy(float), -pairs.parity_low.to_numpy(float)]
    feasible = linprog([0., 0.], A_ub=constraints, b_ub=limits, bounds=[(0, None), (None, None)], method="highs")
    if not feasible.success:
        return None
    return dict(**fitted, pairs=pairs, calls=calls.set_index("strike"), variants=variants,
                constraints=constraints, limits=limits, forward_proxy=proxy,
                calibration_pairs=len(primary), valid_pairs=len(pairs),
                calibration_min_strike=float(primary.strike.min()), calibration_max_strike=float(primary.strike.max()))


def parity_estimate(chain_or_fit: pd.DataFrame | dict, strike: float) -> dict | None:
    """Return an explicitly estimated daily mark and supporting evidence."""
    fitted = fit_parity(chain_or_fit) if isinstance(chain_or_fit, pd.DataFrame) else chain_or_fit
    if fitted is None or float(strike) not in fitted["calls"].index:
        return None
    call = fitted["calls"].loc[float(strike)]
    if not bool(call.good):
        return None
    carry = fitted["intercept"] + fitted["discount_factor"] * float(strike)
    bid, ask = float(call.bid + carry), float(call.ask + carry)
    mid = (bid + ask) / 2
    alternatives = [mid, *[float(call.mid + fit["intercept"] + fit["discount_factor"] * strike) for fit in fitted["variants"]]]
    if not np.isfinite([bid, ask, *alternatives]).all() or bid < 0 or ask < bid:
        return None
    if max(alternatives) - min(alternatives) > max(2., .02 * mid):
        return None
    low = linprog([strike, -1.], A_ub=fitted["constraints"], b_ub=fitted["limits"], bounds=[(0, None), (None, None)], method="highs")
    high = linprog([-strike, 1.], A_ub=fitted["constraints"], b_ub=fitted["limits"], bounds=[(0, None), (None, None)], method="highs")
    if not low.success or not high.success:
        return None
    return dict(used_bid=bid, used_ask=ask, used_mid=mid, method=METHOD,
                parity_call_symbol=str(call.option_symbol), parity_call_bid=float(call.bid), parity_call_ask=float(call.ask),
                parity_discount_factor=fitted["discount_factor"], parity_prepaid_forward=fitted["prepaid_forward"],
                parity_implied_forward=fitted["implied_forward"], parity_calibration_pairs=fitted["calibration_pairs"],
                parity_all_valid_pairs=fitted["valid_pairs"], parity_calibration_min_strike=fitted["calibration_min_strike"],
                parity_calibration_max_strike=fitted["calibration_max_strike"], parity_fit_rms=fitted["residual_rms"],
                parity_fit_max_abs_residual=fitted["max_abs_residual"],
                parity_sensitivity_low=min(alternatives), parity_sensitivity_high=max(alternatives),
                parity_feasible_lower=max(0., float(call.bid + low.fun)),
                parity_feasible_upper=float(call.ask - high.fun),
                parity_bid_ask_note="Synthetic call sides plus fitted carry; not observed put quotes; entry fills prohibited")


def write_evidence(output: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Audit suspect held quotes without changing original or engine artifacts."""
    output = Path(output)
    quotes = pd.read_parquet(output / "daily_quotes.parquet")
    suspects = quotes.loc[~good_quotes(quotes)]
    archive = SPXSurfaceArchive(DATA_ROOT, cache_size=1)
    evidence, pair_frames, summaries = [], [], []
    for day, daily in suspects.groupby("snapshot_date", sort=True):
        path = archive.path_for(day)
        source = pq.ParquetFile(path).read(columns=["snapshot_date", "expiration_date", "option_symbol", "option_type", "strike", "bid", "ask"]).to_pandas()
        source["expiration_date"] = pd.to_datetime(source.expiration_date).dt.normalize()
        for expiry, group in daily.groupby("expiration_date", sort=True):
            fitted = fit_parity(source.loc[source.expiration_date.eq(expiry)])
            if fitted is not None:
                pairs = fitted["pairs"].copy()
                pairs.insert(0, "calibration_date", day)
                pairs.insert(1, "calibration_expiration", expiry)
                pair_frames.append(pairs)
                summaries.append(dict(snapshot_date=day, expiration_date=expiry,
                                      **{key: value for key, value in fitted.items() if isinstance(value, (float, int))}))
            for row in group.itertuples(index=False):
                estimate = parity_estimate(fitted, float(row.strike)) if fitted is not None else None
                evidence.append(dict(snapshot_date=day, expiration_date=expiry, option_symbol=row.option_symbol,
                                     strike=row.strike, raw_bid=row.bid, raw_ask=row.ask, source_file=str(path),
                                     status="estimated" if estimate else "unresolved") | (estimate or {}))
    result = pd.DataFrame(evidence)
    result.to_csv(output / "parity_estimates.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "parity_calibration_summary.csv", index=False)
    if pair_frames:
        pd.concat(pair_frames, ignore_index=True).to_csv(output / "parity_calibration_pairs.csv", index=False)
    print(f"Parity evidence: {result.status.eq('estimated').sum()} estimates, {result.status.eq('unresolved').sum()} unresolved", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    write_evidence(parser.parse_args().output)
