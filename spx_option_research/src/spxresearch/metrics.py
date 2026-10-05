from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def expected_shortfall(returns: pd.Series, confidence: float) -> float:
    values = returns.dropna().astype(float)
    if values.empty:
        return math.nan
    cutoff = values.quantile(1.0 - confidence)
    tail = values[values <= cutoff]
    return float(tail.mean()) if not tail.empty else math.nan


def performance_metrics(
    returns: pd.Series,
    *,
    option_returns: pd.Series | None = None,
    collateral_returns: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {}
    years = len(r) / TRADING_DAYS
    total = float((1.0 + r).prod())
    annual_return = total ** (1.0 / years) - 1.0 if years > 0 and total > 0 else math.nan
    annual_vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    if collateral_returns is not None:
        excess = r - collateral_returns.reindex(r.index).fillna(0.0)
    else:
        excess = r
    excess_volatility = excess.std(ddof=1)
    sharpe = (
        float(excess.mean() / excess_volatility * math.sqrt(TRADING_DAYS))
        if excess_volatility
        else math.nan
    )
    downside = excess[excess < 0].std(ddof=1)
    sortino = float(excess.mean() / downside * math.sqrt(TRADING_DAYS)) if downside else math.nan
    dd = drawdown_series(r)
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    weekly = (1.0 + r).resample("W-FRI").prod() - 1.0
    metrics: dict[str, Any] = {
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()),
        "calmar": annual_return / abs(float(dd.min())) if dd.min() < 0 else math.nan,
        "worst_day": float(r.min()),
        "worst_week": float(weekly.min()),
        "worst_month": float(monthly.min()),
        "best_month": float(monthly.max()),
        "win_rate": float((r > 0).mean()),
        "average_monthly_return": float(monthly.mean()),
        "median_monthly_return": float(monthly.median()),
        "skewness": float(stats.skew(r, bias=False)),
        "kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
        "var_95": float(r.quantile(0.05)),
        "var_99": float(r.quantile(0.01)),
        "es_95": expected_shortfall(r, 0.95),
        "es_99": expected_shortfall(r, 0.99),
    }
    if option_returns is not None:
        metrics["annualized_option_return"] = float(option_returns.reindex(r.index).fillna(0).mean() * TRADING_DAYS)
    if collateral_returns is not None:
        metrics["annualized_collateral_return"] = float(collateral_returns.reindex(r.index).fillna(0).mean() * TRADING_DAYS)
    if benchmark_returns is not None:
        aligned = pd.concat([r, benchmark_returns], axis=1).dropna()
        if len(aligned) > 2 and aligned.iloc[:, 1].var() > 0:
            covariance = aligned.cov().iloc[0, 1]
            metrics["beta_to_spx"] = float(covariance / aligned.iloc[:, 1].var())
            metrics["correlation_to_spx"] = float(aligned.corr().iloc[0, 1])
            up = aligned.iloc[:, 1] > 0
            down = aligned.iloc[:, 1] < 0
            metrics["upside_capture"] = float(aligned.loc[up].iloc[:, 0].mean() / aligned.loc[up].iloc[:, 1].mean()) if up.any() else math.nan
            metrics["downside_capture"] = float(aligned.loc[down].iloc[:, 0].mean() / aligned.loc[down].iloc[:, 1].mean()) if down.any() else math.nan
            benchmark = aligned.iloc[:, 1]
            strategy = aligned.iloc[:, 0]
            buckets = {
                "spx_positive": benchmark > 0,
                "spx_0_to_minus_5": (benchmark <= 0) & (benchmark > -0.05),
                "spx_minus_5_to_minus_10": (benchmark <= -0.05) & (benchmark > -0.10),
                "spx_minus_10_to_minus_20": (benchmark <= -0.10) & (benchmark > -0.20),
                "spx_below_minus_20": benchmark <= -0.20,
            }
            for name, mask in buckets.items():
                metrics[f"mean_return_when_{name}"] = (
                    float(strategy[mask].mean()) if mask.any() else math.nan
                )
                metrics[f"observations_when_{name}"] = int(mask.sum())
    return metrics


def chronological_splits(index: pd.Index, fractions: tuple[float, float, float]) -> dict[str, pd.Index]:
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to one")
    ordered = pd.Index(index).sort_values()
    first = int(len(ordered) * fractions[0])
    second = first + int(len(ordered) * fractions[1])
    return {
        "train": ordered[:first],
        "validation": ordered[first:second],
        "test": ordered[second:],
    }
