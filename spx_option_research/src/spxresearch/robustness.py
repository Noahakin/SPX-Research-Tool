from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import drawdown_series, expected_shortfall


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Bailey and Lopez de Prado PSR using daily observations."""
    values = returns.dropna().to_numpy(float)
    if len(values) < 3 or np.std(values, ddof=1) == 0:
        return math.nan
    observed = float(np.mean(values) / np.std(values, ddof=1))
    benchmark = float(benchmark_sharpe) / math.sqrt(252.0)
    skew = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    denominator = math.sqrt(
        max(1.0 - skew * observed + ((kurtosis - 1.0) / 4.0) * observed**2, 1e-12)
    )
    z_score = (observed - benchmark) * math.sqrt(len(values) - 1.0) / denominator
    return float(stats.norm.cdf(z_score))


def expected_max_sharpe(number_of_trials: int, sharpe_std: float) -> float:
    if number_of_trials <= 1 or not np.isfinite(sharpe_std) or sharpe_std <= 0:
        return 0.0
    euler_gamma = 0.5772156649015329
    n = float(number_of_trials)
    return float(
        sharpe_std
        * (
            (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / n)
            + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (n * math.e))
        )
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    number_of_trials: int,
    trial_sharpe_std: float,
) -> tuple[float, float]:
    benchmark = expected_max_sharpe(number_of_trials, trial_sharpe_std)
    return probabilistic_sharpe_ratio(returns, benchmark), benchmark


def stationary_block_bootstrap(
    returns: pd.Series,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    """Circular fixed-block bootstrap preserving short-horizon dependence."""
    clean = returns.dropna().astype(float)
    values = clean.to_numpy()
    n = len(values)
    if n == 0:
        return pd.DataFrame()
    block = max(1, min(int(block_length), n))
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    for _ in range(int(samples)):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        indexes = np.concatenate(
            [(np.arange(start, start + block) % n) for start in starts]
        )[:n]
        draw = pd.Series(values[indexes])
        standard_deviation = draw.std(ddof=1)
        wealth = float((1.0 + draw).prod())
        annual_return = wealth ** (252.0 / n) - 1.0 if wealth > 0 else math.nan
        records.append(
            {
                "annualized_return": annual_return,
                "sharpe": float(draw.mean() / standard_deviation * math.sqrt(252.0))
                if standard_deviation > 0
                else math.nan,
                "max_drawdown": float(drawdown_series(draw).min()),
                "es_95": expected_shortfall(draw, 0.95),
                "es_99": expected_shortfall(draw, 0.99),
            }
        )
    return pd.DataFrame(records)


def bootstrap_confidence_intervals(samples: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in samples.columns:
        values = samples[column].dropna()
        result[f"{column}_ci_2p5"] = float(values.quantile(0.025))
        result[f"{column}_ci_50"] = float(values.quantile(0.5))
        result[f"{column}_ci_97p5"] = float(values.quantile(0.975))
    return result


def leave_one_period_out(returns: pd.Series, periods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    returns = returns.sort_index()
    for name, (start, end) in periods.items():
        kept = returns.loc[~returns.index.to_series().between(start, end).to_numpy()]
        volatility = kept.std(ddof=1)
        records.append(
            {
                "excluded_period": name,
                "observations": len(kept),
                "annualized_return": float(kept.mean() * 252.0),
                "sharpe": float(kept.mean() / volatility * math.sqrt(252.0))
                if volatility > 0
                else math.nan,
                "max_drawdown": float(drawdown_series(kept).min()),
            }
        )
    return pd.DataFrame(records)


def rolling_walk_forward(returns: pd.Series, test_years: int = 1) -> pd.DataFrame:
    returns = returns.sort_index()
    records: list[dict[str, float | int]] = []
    years = sorted(pd.Index(returns.index.year).unique())
    for year in years[1:]:
        test = returns[returns.index.year == year]
        if test.empty:
            continue
        volatility = test.std(ddof=1)
        records.append(
            {
                "test_year": int(year),
                "observations": len(test),
                "annualized_return": float(test.mean() * 252.0),
                "sharpe": float(test.mean() / volatility * math.sqrt(252.0))
                if volatility > 0
                else math.nan,
                "max_drawdown": float(drawdown_series(test).min()),
            }
        )
    return pd.DataFrame(records)
