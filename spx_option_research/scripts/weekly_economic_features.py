"""Past-only economic compensation features for weekly SPX put spreads.

Prices and fills are option points; all payoff, edge and loss outputs are
fractions of entry cash-SPX notional.  A "fair" value here is an undiscounted
physical expected terminal payoff, not an arbitrage price or an IV estimate.
The helper does not use vendor Greeks, whose unit conventions belong to the
upstream extractor rather than to this payoff-based calculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


CALENDAR_DAYS_PER_YEAR = 365.2425
TRADING_DAYS_PER_YEAR = 252.0
MULTIPLIER = 100.0
COMMISSION_PER_LEG = 1.50
NUMERIC_COLUMNS = (
    "econ_rv21_lag1", "econ_rv63_lag1", "econ_forecast_vol",
    "econ_credit", "econ_commission", "econ_max_loss",
    "econ_expected_short_payoff", "econ_expected_long_payoff",
    "econ_expected_liability", "econ_short_fair_edge", "econ_long_fair_edge",
    "econ_expected_edge", "econ_std_liability", "econ_tail_liability_95",
    "econ_tail_loss_95", "econ_probability_loss", "econ_probability_max_loss",
    "econ_risk_denominator", "econ_edge_per_max_loss", "econ_edge_per_risk",
    "econ_edge_per_tail_risk", "econ_simulated_mean_log_return",
    "econ_normal_expected_short_payoff", "econ_normal_expected_long_payoff",
    "econ_normal_expected_liability", "econ_normal_short_fair_edge",
    "econ_normal_long_fair_edge", "econ_normal_expected_edge",
    "econ_normal_edge_per_max_loss",
)


def upper_tail_mean(values: np.ndarray, tail_probability: float = 0.05) -> np.ndarray:
    """Exact empirical upper-tail mean, including a fractional boundary mass.

    Rows are scenarios; columns are candidates.  Unlike selecting all values
    above an empirical quantile, this always averages exactly the requested
    probability mass, including when bounded spread payoffs have ties.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("tail values must be a nonempty finite scenario matrix")
    if not 0 < tail_probability <= 1:
        raise ValueError("tail probability must be in (0, 1]")
    ordered = np.sort(values, axis=0)[::-1]
    mass = len(values) * tail_probability
    whole = int(np.floor(mass))
    total = ordered[:whole].sum(axis=0)
    fraction = mass - whole
    if fraction > 1e-12 and whole < len(values):
        total = total + fraction * ordered[whole]
    return total / mass


def _normal_put_payoff(strike_ratio: np.ndarray, total_vol: np.ndarray) -> np.ndarray:
    """E[(K/S - S_T/S)+] under flat-RV lognormal, zero arithmetic drift."""
    d1 = (-np.log(strike_ratio) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol
    return strike_ratio * ndtr(-d2) - ndtr(-d1)


def economic_features(
    candidates: pd.DataFrame,
    cash_closes: pd.Series,
    *,
    history_window: int = 156,
    min_history: int = 104,
) -> pd.DataFrame:
    """Append economic features without changing candidate index or row order.

    Required fields are entry_date, expiration_date, spot_entry,
    spot_expiration, upper_strike, lower_strike and dte.  For execution, pass
    short_sell_fill_realistic/long_buy_fill_realistic (points, before fees),
    or short_bid/short_ask/long_bid/long_ask.  Optional saved premium/width/loss
    fields are checked against the actual fills to catch accounting errors.

    Forecast volatility is sqrt((lagged RV21**2 + lagged RV63**2) / 2), with
    RV computed from sample standard deviations of cash-SPX log returns.
    All RV observations stop before entry.  Historical log weekly returns
    are divided by their own entry's lagged RV21 * sqrt(DTE/365.2425), then
    rescaled by the current forecast volatility and current tenor.  Retain
    the last 156 usable weekly observations with expiration strictly before
    the current entry; at least 104 are needed.  Each entry week contributes
    one observation regardless of its number of spread candidates.

    Historical residuals are uncentered and are not clipped: their historical
    drift and asymmetry remain in the scenario distribution.  The normal
    comparison instead uses a zero-arithmetic-drift lognormal distribution.
    Neither method discounts the liability; both compare cash received now
    with expected terminal cash paid, matching the zero-interest backtest.

    Risk ratios use positive maximum loss, and a standard-deviation/tail
    denominator floored at max(10% of width, 5bp of spot notional).  Initial
    observations retain available normal-model features but have NaN in
    empirical features until the history requirement is satisfied.
    """
    required = {
        "entry_date", "expiration_date", "spot_entry", "spot_expiration",
        "upper_strike", "lower_strike", "dte",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidate table lacks columns: {sorted(missing)}")
    if min_history < 2 or history_window < min_history:
        raise ValueError("require 2 <= min_history <= history_window")
    result = candidates.copy()
    count = len(result)
    if not count:
        for column in NUMERIC_COLUMNS:
            result[column] = pd.Series(index=result.index, dtype=float)
        result["econ_history_count"] = pd.Series(index=result.index, dtype=int)
        result["econ_history_latest_expiration"] = pd.NaT
        return result

    # Work with positions so a caller's arbitrary or duplicated index survives.
    frame = candidates.reset_index(drop=True).copy()
    for column in ("entry_date", "expiration_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    closes = cash_closes.astype(float).copy()
    closes.index = pd.DatetimeIndex(closes.index).normalize()
    closes = closes.sort_index()
    if closes.index.has_duplicates or not np.isfinite(closes).all() or closes.le(0).any():
        raise ValueError("cash closes must have unique dates and finite positive values")
    if frame[["entry_date", "expiration_date"]].isna().any().any():
        raise ValueError("candidate dates must be present")
    inputs = frame[["spot_entry", "upper_strike", "lower_strike", "dte"]].to_numpy(float)
    if not np.isfinite(inputs).all() or (inputs <= 0).any():
        raise ValueError("entry spots, strikes and DTE must be finite and positive")
    if frame.upper_strike.le(frame.lower_strike).any():
        raise ValueError("short strike must exceed long strike")
    actual_dte = (frame.expiration_date - frame.entry_date).dt.days.to_numpy()
    if not np.array_equal(actual_dte, frame.dte.to_numpy(float)):
        raise ValueError("DTE must equal the calendar interval to expiration")

    spot = frame.spot_entry.to_numpy(float)
    short_ratio = frame.upper_strike.to_numpy(float) / spot
    long_ratio = frame.lower_strike.to_numpy(float) / spot
    width = short_ratio - long_ratio
    if {"short_sell_fill_realistic", "long_buy_fill_realistic"}.issubset(frame.columns):
        short_sale = frame.short_sell_fill_realistic.to_numpy(float) / spot
        long_cost = frame.long_buy_fill_realistic.to_numpy(float) / spot
    else:
        quote_columns = {"short_bid", "short_ask", "long_bid", "long_ask"}
        missing_quotes = quote_columns.difference(frame.columns)
        if missing_quotes:
            raise ValueError(f"missing execution fills and quote columns: {sorted(missing_quotes)}")
        short_sale = (0.75 * frame.short_bid + 0.25 * frame.short_ask).to_numpy(float) / spot
        long_cost = (0.25 * frame.long_bid + 0.75 * frame.long_ask).to_numpy(float) / spot
    commission = 2 * COMMISSION_PER_LEG / (MULTIPLIER * spot)
    credit = short_sale - long_cost - commission
    max_loss = width - credit
    if not np.isfinite(np.r_[short_sale, long_cost]).all():
        raise ValueError("execution fills must be finite")
    if (credit <= 0).any() or (max_loss <= 0).any():
        raise ValueError("spreads require positive credit below the strike width")
    comparisons = {
        "premium_realistic_pct_spot_notional": credit,
        "premium_realistic_cash": credit * spot * MULTIPLIER,
        "width_pct": width,
        "max_loss_realistic_pct": max_loss,
    }
    for column, expected in comparisons.items():
        if column in frame and not np.allclose(frame[column].to_numpy(float), expected,
                                               rtol=1e-9, atol=1e-10):
            raise ValueError(f"{column} does not reconcile to strikes, fills and commissions")

    log_returns = np.log(closes / closes.shift(1))
    rv21 = log_returns.rolling(21, min_periods=21).std(ddof=1).shift(1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    rv63 = log_returns.rolling(63, min_periods=63).std(ddof=1).shift(1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    frame["_rv21"] = frame.entry_date.map(rv21)
    frame["_rv63"] = frame.entry_date.map(rv63)
    for column in ("expiration_date", "spot_entry", "spot_expiration", "dte"):
        if frame.groupby("entry_date")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"candidate rows disagree on weekly {column}")
    weeks = frame.drop_duplicates("entry_date").sort_values("entry_date").copy()
    denominator = weeks._rv21 * np.sqrt(weeks.dte / CALENDAR_DAYS_PER_YEAR)
    # A future expiration can be unavailable; it cannot become a history input.
    historical_end = pd.to_numeric(weeks.spot_expiration, errors="coerce")
    weeks["_residual"] = np.log(historical_end.where(historical_end.gt(0)) / weeks.spot_entry) / denominator
    weeks = weeks.loc[np.isfinite(weeks._residual) & weeks._rv21.gt(1e-8)].copy()

    values = {column: np.full(count, np.nan) for column in NUMERIC_COLUMNS}
    values["econ_rv21_lag1"] = frame._rv21.to_numpy(float)
    values["econ_rv63_lag1"] = frame._rv63.to_numpy(float)
    forecast = np.sqrt(0.5 * (frame._rv21.to_numpy(float)**2 + frame._rv63.to_numpy(float)**2))
    values["econ_forecast_vol"] = forecast
    values["econ_credit"], values["econ_commission"], values["econ_max_loss"] = credit, commission, max_loss
    counts = np.zeros(count, dtype=int)
    latest = np.full(count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    risk_floor = np.maximum(0.1 * width, 0.0005)
    total_vol = forecast * np.sqrt(frame.dte.to_numpy(float) / CALENDAR_DAYS_PER_YEAR)
    usable_normal = np.isfinite(total_vol) & (total_vol > 1e-8)
    if usable_normal.any():
        pos = np.flatnonzero(usable_normal)
        short_expected = _normal_put_payoff(short_ratio[pos], total_vol[pos])
        long_expected = _normal_put_payoff(long_ratio[pos], total_vol[pos])
        liability = np.maximum(short_expected - long_expected, 0.0)
        normal = {
            "econ_normal_expected_short_payoff": short_expected,
            "econ_normal_expected_long_payoff": long_expected,
            "econ_normal_expected_liability": liability,
            "econ_normal_short_fair_edge": short_sale[pos] - short_expected,
            "econ_normal_long_fair_edge": long_expected - long_cost[pos],
            "econ_normal_expected_edge": credit[pos] - liability,
            "econ_normal_edge_per_max_loss": (credit[pos] - liability) / max_loss[pos],
        }
        for column, array in normal.items():
            values[column][pos] = array

    for entry, indices in frame.groupby("entry_date", sort=True).groups.items():
        pos = np.asarray(indices, dtype=int)
        history = weeks.loc[weeks.expiration_date.lt(entry)].tail(history_window)
        counts[pos] = len(history)
        if len(history):
            latest[pos] = history.expiration_date.max().to_datetime64()
        if len(history) < min_history or not usable_normal[pos].all():
            continue
        scenario_log_returns = history._residual.to_numpy(float)[:, None] * total_vol[pos][None, :]
        simulated_spot_ratios = np.exp(scenario_log_returns)
        if not np.isfinite(simulated_spot_ratios).all():
            raise ValueError(f"nonfinite filtered historical scenarios at {entry}")
        short_payoffs = np.maximum(short_ratio[pos][None, :] - simulated_spot_ratios, 0.0)
        long_payoffs = np.maximum(long_ratio[pos][None, :] - simulated_spot_ratios, 0.0)
        liability = short_payoffs - long_payoffs
        short_expected, long_expected = short_payoffs.mean(axis=0), long_payoffs.mean(axis=0)
        expected = liability.mean(axis=0)
        edge = credit[pos] - expected
        sd = liability.std(axis=0, ddof=1)
        tail_liability = upper_tail_mean(liability)
        tail_loss = np.maximum(tail_liability - credit[pos], 0.0)
        risk = np.maximum(sd, risk_floor[pos])
        empirical = {
            "econ_expected_short_payoff": short_expected,
            "econ_expected_long_payoff": long_expected,
            "econ_expected_liability": expected,
            "econ_short_fair_edge": short_sale[pos] - short_expected,
            "econ_long_fair_edge": long_expected - long_cost[pos],
            "econ_expected_edge": edge,
            "econ_std_liability": sd,
            "econ_tail_liability_95": tail_liability,
            "econ_tail_loss_95": tail_loss,
            "econ_probability_loss": (liability > credit[pos][None, :]).mean(axis=0),
            "econ_probability_max_loss": (simulated_spot_ratios <= long_ratio[pos][None, :]).mean(axis=0),
            "econ_risk_denominator": risk,
            "econ_edge_per_max_loss": edge / max_loss[pos],
            "econ_edge_per_risk": edge / risk,
            "econ_edge_per_tail_risk": edge / np.maximum(tail_loss, risk_floor[pos]),
            "econ_simulated_mean_log_return": scenario_log_returns.mean(axis=0),
        }
        for column, array in empirical.items():
            values[column][pos] = array
    for column, array in values.items():
        result[column] = array
    result["econ_history_count"] = counts
    result["econ_history_latest_expiration"] = latest
    return result
