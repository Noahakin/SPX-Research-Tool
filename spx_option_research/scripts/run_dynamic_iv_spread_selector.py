from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_requested_premium_study import (  # noqa: E402
    DATA_ROOT,
    MULTIPLIER,
    SPXSurfaceArchive,
    Structure,
    annualized_rate,
    expiration_value,
    load_spx_cash_closes,
    monthly_roll_dates,
    structure_record,
    valid_pm_puts,
)


OUT = PROJECT / "results/dynamic_iv_spread_selector"
REALIZED_WINDOW = 21
TRADING_DAYS_PER_YEAR = 252.0
BENCHMARK_UPPER_RATIO = 0.99

CANDIDATE_STRUCTURES = tuple(
    Structure(
        f"1M {int(round(upper_ratio * 100))}/{int(round((upper_ratio - 0.03) * 100))} short put spread",
        1,
        "short_put_spread",
        upper_ratio,
        upper_ratio - 0.03,
    )
    for upper_ratio in (0.98, 0.99, 1.00, 1.01, 1.02, 1.03)
)

DYNAMIC_NAME = "Highest short-leg IV/RV deviation"
BENCHMARK_NAME = "Fixed 99/96"


def trailing_realized_volatility(
    cash_closes: pd.Series,
    window: int = REALIZED_WINDOW,
) -> pd.Series:
    """Annualized rolling volatility of the latest `window` close-to-close log returns."""
    if window < 2:
        raise ValueError("realized-volatility window must be at least two returns")
    closes = cash_closes.astype(float).sort_index()
    if closes.index.has_duplicates:
        closes = closes[~closes.index.duplicated(keep="last")]
    if closes.le(0).any():
        raise ValueError("cash closes must be positive")
    log_returns = np.log(closes / closes.shift(1))
    return log_returns.rolling(window=window, min_periods=window).std(ddof=1) * math.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def _valid_implied_volatility(value: object, description: str) -> float:
    implied_volatility = float(value)
    if not np.isfinite(implied_volatility) or implied_volatility <= 0:
        raise LookupError(f"invalid implied volatility for {description}")
    return implied_volatility


def build_candidate_trades(
    archive: SPXSurfaceArchive,
    cash_closes: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build six comparable candidates on every complete monthly roll.

    The option archive supplies an IV for each leg rather than a unique IV for a
    vertical.  The selection signal is therefore the IV of the short put, which
    is the volatility being sold.  Long-leg IV is retained in the output for
    audit but is not part of the ranking rule.
    """
    rolls = monthly_roll_dates(archive.populated_dates)
    realized_volatility = trailing_realized_volatility(cash_closes)
    records: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []

    for index, entry_date in enumerate(rolls[:-1]):
        expiration_date = rolls[index + 1]
        if entry_date not in cash_closes.index:
            skips.append(
                {
                    "entry_date": entry_date,
                    "expiration_date": expiration_date,
                    "reason": "missing cash SPX entry close",
                }
            )
            continue
        realized = float(realized_volatility.get(entry_date, np.nan))
        if not np.isfinite(realized) or realized <= 0:
            skips.append(
                {
                    "entry_date": entry_date,
                    "expiration_date": expiration_date,
                    "reason": f"fewer than {REALIZED_WINDOW} usable trailing cash-index returns",
                }
            )
            continue

        chain = archive.read(entry_date)
        puts = valid_pm_puts(chain, expiration_date)
        monthly_records: list[dict[str, object]] = []
        try:
            if puts.empty:
                raise LookupError("no valid PM-settled puts for the requested expiration")
            spot = float(cash_closes.loc[entry_date])
            for structure in CANDIDATE_STRUCTURES:
                record = structure_record(
                    entry_date=entry_date,
                    expiration_date=expiration_date,
                    spot=spot,
                    structure=structure,
                    puts=puts,
                )
                short_quote = record["_legs"][0][0]
                long_quote = record["_legs"][1][0]
                short_iv = _valid_implied_volatility(
                    short_quote["implied_volatility"], f"{structure.label} short leg"
                )
                long_iv = _valid_implied_volatility(
                    long_quote["implied_volatility"], f"{structure.label} long leg"
                )
                record.update(
                    {
                        "spread": structure.label.split()[1],
                        "short_leg_implied_vol": short_iv,
                        "long_leg_implied_vol": long_iv,
                        "realized_vol_21d": realized,
                        "iv_minus_realized_vol": short_iv - realized,
                        "iv_rv_pct_deviation": short_iv / realized - 1.0,
                    }
                )
                monthly_records.append(record)
        except (LookupError, ValueError, KeyError, TypeError) as error:
            skips.append(
                {
                    "entry_date": entry_date,
                    "expiration_date": expiration_date,
                    "reason": str(error),
                }
            )
        else:
            records.extend(monthly_records)

    candidates = pd.DataFrame(records)
    if candidates.empty:
        raise RuntimeError("the archive produced no complete candidate months")
    if candidates.groupby("entry_date").size().ne(len(CANDIDATE_STRUCTURES)).any():
        raise RuntimeError("candidate months must contain all six spreads")

    candidates["spot_expiration"] = candidates["expiration_date"].map(cash_closes)
    if candidates["spot_expiration"].isna().any():
        missing = sorted(
            candidates.loc[candidates["spot_expiration"].isna(), "expiration_date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
        )
        raise RuntimeError(f"missing expiration cash SPX close for {missing}")

    expiration_values: list[float] = []
    pnl_mid: list[float] = []
    pnl_realistic: list[float] = []
    pnl_natural: list[float] = []
    for row in candidates.to_dict("records"):
        value = expiration_value(row["_legs"], float(row["spot_expiration"]))
        expiration_values.append(value)
        pnl_mid.append(float(row["_entry_cash_mid"]) + value)
        pnl_realistic.append(float(row["_entry_cash_realistic"]) + value)
        pnl_natural.append(float(row["_entry_cash_natural"]) + value)
    candidates["expiration_value_cash"] = expiration_values
    candidates["pnl_mid_cash"] = pnl_mid
    candidates["pnl_realistic_cash"] = pnl_realistic
    candidates["pnl_natural_cash"] = pnl_natural
    for fill in ("mid", "realistic", "natural"):
        candidates[f"pnl_{fill}_pct_spot_notional"] = candidates[f"pnl_{fill}_cash"] / (
            candidates["spot_entry"] * MULTIPLIER
        )

    candidates["iv_rank"] = candidates.groupby("entry_date")[
        "iv_rv_pct_deviation"
    ].rank(method="first", ascending=False)
    candidates.drop(
        columns=[
            "_legs",
            "_entry_cash_mid",
            "_entry_cash_realistic",
            "_entry_cash_natural",
        ],
        inplace=True,
    )
    candidates.sort_values(["entry_date", "target_upper_ratio"], inplace=True)
    candidates.reset_index(drop=True, inplace=True)
    return candidates, pd.DataFrame(skips)


def select_portfolio_trades(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "entry_date",
        "expiration_date",
        "target_upper_ratio",
        "iv_rv_pct_deviation",
        "pnl_realistic_pct_spot_notional",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate frame lacks columns: {sorted(missing)}")

    ordered = candidates.sort_values(
        ["entry_date", "iv_rv_pct_deviation", "target_upper_ratio"],
        ascending=[True, False, True],
        kind="stable",
    )
    dynamic = ordered.groupby("entry_date", sort=True, as_index=False).head(1).copy()
    dynamic["portfolio"] = DYNAMIC_NAME

    benchmark = candidates[
        np.isclose(candidates["target_upper_ratio"], BENCHMARK_UPPER_RATIO)
    ].copy()
    if benchmark["entry_date"].duplicated().any():
        raise ValueError("benchmark has duplicate entry dates")
    benchmark["portfolio"] = BENCHMARK_NAME

    dynamic_dates = pd.Index(dynamic["entry_date"])
    benchmark_dates = pd.Index(benchmark["entry_date"])
    if not dynamic_dates.equals(benchmark_dates):
        raise ValueError("dynamic and benchmark entry dates do not match")
    selected = pd.concat([dynamic, benchmark], ignore_index=True)
    return selected.sort_values(["portfolio", "entry_date"]).reset_index(drop=True)


def portfolio_paths(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for portfolio, group in selected.groupby("portfolio", sort=False):
        path = group.sort_values("entry_date").copy()
        for fill in ("mid", "realistic"):
            return_column = f"monthly_return_{fill}"
            entry_column = f"entry_equity_{fill}"
            ending_column = f"ending_equity_{fill}"
            path[return_column] = path[f"pnl_{fill}_pct_spot_notional"].astype(float)
            path[entry_column] = (1.0 + path[return_column]).cumprod().shift(
                fill_value=1.0
            )
            path[ending_column] = path[entry_column] * (1.0 + path[return_column])
            path[f"dollar_pnl_{fill}_per_initial_1m"] = (
                path[ending_column] - path[entry_column]
            ) * 1_000_000.0
        # Unsuffixed columns are realistic-fill aliases for convenient downstream use.
        path["monthly_return"] = path["monthly_return_realistic"]
        path["entry_equity"] = path["entry_equity_realistic"]
        path["ending_equity"] = path["ending_equity_realistic"]
        path["dollar_pnl_per_initial_1m"] = path[
            "dollar_pnl_realistic_per_initial_1m"
        ]
        rows.append(path)
    return pd.concat(rows, ignore_index=True)


def _fill_metrics(
    ordered: pd.DataFrame,
    fill: str,
    elapsed_years: float,
) -> dict[str, float]:
    returns = ordered[f"monthly_return_{fill}"].astype(float)
    ending_wealth = float((1.0 + returns).prod())
    standard_deviation = float(returns.std(ddof=1))
    volatility = standard_deviation * math.sqrt(12.0)
    sharpe = (
        float(returns.mean() / standard_deviation * math.sqrt(12.0))
        if standard_deviation > 0
        else np.nan
    )
    wealth = pd.concat(
        [pd.Series([1.0]), (1.0 + returns.reset_index(drop=True)).cumprod()],
        ignore_index=True,
    )
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        f"annualized_premium_{fill}_pct_spot_notional": annualized_rate(
            ordered, f"premium_{fill}_pct_spot_notional"
        ),
        f"annualized_arithmetic_pnl_{fill}_pct_spot_notional": annualized_rate(
            ordered, f"pnl_{fill}_pct_spot_notional"
        ),
        f"cagr_{fill}": ending_wealth ** (1.0 / elapsed_years) - 1.0,
        f"annualized_volatility_{fill}": volatility,
        f"sharpe_zero_cash_monthly_{fill}": sharpe,
        f"maximum_drawdown_{fill}": float(drawdown.min()),
        f"ending_wealth_per_dollar_{fill}": ending_wealth,
        f"win_rate_{fill}": float(returns.gt(0).mean()),
    }


def summarize_portfolios(paths: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for portfolio, group in paths.groupby("portfolio", sort=False):
        ordered = group.sort_values("entry_date")
        elapsed_years = (
            pd.Timestamp(ordered["expiration_date"].max())
            - pd.Timestamp(ordered["entry_date"].min())
        ).days / 365.2425
        if elapsed_years <= 0:
            raise ValueError("portfolio history must span a positive period")
        row: dict[str, object] = {
            "portfolio": portfolio,
            "observations": len(ordered),
            "first_entry": ordered["entry_date"].min(),
            "last_expiration": ordered["expiration_date"].max(),
            "elapsed_years": elapsed_years,
            "average_short_leg_implied_vol": float(
                ordered["short_leg_implied_vol"].mean()
            ),
            "average_realized_vol_21d": float(ordered["realized_vol_21d"].mean()),
            "average_iv_rv_pct_deviation": float(
                ordered["iv_rv_pct_deviation"].mean()
            ),
        }
        for fill in ("mid", "realistic"):
            row.update(_fill_metrics(ordered, fill, elapsed_years))
        row.update(
            {
                # Unsuffixed fields are realistic-fill aliases.
                "annualized_arithmetic_pnl_pct_spot_notional": row[
                    "annualized_arithmetic_pnl_realistic_pct_spot_notional"
                ],
                "cagr": row["cagr_realistic"],
                "annualized_volatility": row["annualized_volatility_realistic"],
                "sharpe_zero_cash_monthly": row[
                    "sharpe_zero_cash_monthly_realistic"
                ],
                "maximum_drawdown": row["maximum_drawdown_realistic"],
                "ending_wealth_per_dollar": row[
                    "ending_wealth_per_dollar_realistic"
                ],
                "win_rate": row["win_rate_realistic"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_candidate_diagnostics(
    candidates: pd.DataFrame,
    selection_counts: pd.DataFrame,
) -> pd.DataFrame:
    fixed = candidates.copy()
    fixed["portfolio"] = "Fixed " + fixed["spread"].astype(str)
    diagnostics = summarize_portfolios(portfolio_paths(fixed))
    diagnostics["spread"] = diagnostics["portfolio"].str.removeprefix("Fixed ")
    diagnostics = diagnostics.merge(selection_counts, on="spread", how="left")
    diagnostics["months"] = diagnostics["months"].fillna(0).astype(int)
    columns = [
        "spread",
        "observations",
        "months",
        "cagr_mid",
        "sharpe_zero_cash_monthly_mid",
        "cagr_realistic",
        "sharpe_zero_cash_monthly_realistic",
        "annualized_arithmetic_pnl_mid_pct_spot_notional",
        "annualized_arithmetic_pnl_realistic_pct_spot_notional",
        "annualized_premium_mid_pct_spot_notional",
        "annualized_premium_realistic_pct_spot_notional",
        "maximum_drawdown_realistic",
        "win_rate_realistic",
        "average_short_leg_implied_vol",
        "average_iv_rv_pct_deviation",
    ]
    spread_order = {
        structure.label.split()[1]: index
        for index, structure in enumerate(CANDIDATE_STRUCTURES)
    }
    diagnostics["_spread_order"] = diagnostics["spread"].map(spread_order)
    return (
        diagnostics.sort_values("_spread_order")[columns]
        .reset_index(drop=True)
    )


def yearly_results(paths: pd.DataFrame) -> pd.DataFrame:
    frame = paths.copy()
    frame["year"] = pd.to_datetime(frame["expiration_date"]).dt.year
    return (
        frame.groupby(["portfolio", "year"], as_index=False)
        .agg(
            trades=("monthly_return", "size"),
            premium_realistic_pct_spot_notional=(
                "premium_realistic_pct_spot_notional",
                "sum",
            ),
            pnl_realistic_pct_spot_notional=("monthly_return", "sum"),
            compounded_return=("monthly_return", lambda values: (1.0 + values).prod() - 1.0),
            wins=("monthly_return", lambda values: int((values > 0).sum())),
        )
        .sort_values(["portfolio", "year"])
    )


def write_report(
    summary: pd.DataFrame,
    paths: pd.DataFrame,
    selection_counts: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    indexed = summary.set_index("portfolio")
    dynamic = indexed.loc[DYNAMIC_NAME]
    benchmark = indexed.loc[BENCHMARK_NAME]
    difference_cagr = float(dynamic["cagr"] - benchmark["cagr"])
    difference_sharpe = float(
        dynamic["sharpe_zero_cash_monthly"] - benchmark["sharpe_zero_cash_monthly"]
    )
    counts = ", ".join(
        f"{row.spread}: {int(row.months)}"
        for row in selection_counts.itertuples(index=False)
    )
    selected_dynamic = paths[paths["portfolio"].eq(DYNAMIC_NAME)]
    positive_signal = float(selected_dynamic["iv_rv_pct_deviation"].gt(0).mean())
    diagnostic_lines = [
        f"| {row.spread} | {int(row.months)} | {row.cagr_mid:.2%} | "
        f"{row.sharpe_zero_cash_monthly_mid:.2f} | {row.cagr_realistic:.2%} | "
        f"{row.sharpe_zero_cash_monthly_realistic:.2f} |"
        for row in diagnostics.itertuples(index=False)
    ]

    report = f"""# Dynamic 1M SPX put-spread IV/RV selector

This test sells one of six three-percentage-point-wide put spreads each month: 98/95, 99/96, 100/97, 101/98, 102/99, or 103/100. Here “3-wide” means the target strikes differ by 3% of entry SPX spot, not three SPX index points. It uses the same matched third-Friday PM-settled SPXW rolls, cash-index strikes and settlement, expiration holding period, and realistic execution assumption as the requested premium study. Every entry targets SPX cash-spot notional equal to 100% of current portfolio equity; fractional contracts make the comparison exact.

The signal defines a spread's current IV as the option archive's implied volatility for its short put. Trailing realized volatility is the annualized sample standard deviation of the latest 21 SPX cash close-to-close log returns, including the entry-date close. Each candidate receives `short IV / 21-day realized vol - 1`, and the candidate with the largest value is sold. The long-leg IV is retained in the candidate file but does not enter the signal.

| Portfolio | CAGR | Sharpe | Annualized volatility | Maximum drawdown | Annual arithmetic P&L | Annual premium | Ending value of $1 | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dynamic highest IV/RV deviation | {dynamic['cagr']:.2%} | {dynamic['sharpe_zero_cash_monthly']:.2f} | {dynamic['annualized_volatility']:.2%} | {dynamic['maximum_drawdown']:.2%} | {dynamic['annualized_arithmetic_pnl_pct_spot_notional']:.2%} | {dynamic['annualized_premium_realistic_pct_spot_notional']:.2%} | ${dynamic['ending_wealth_per_dollar']:.3f} | {dynamic['win_rate']:.1%} |
| Fixed 99/96 every month | {benchmark['cagr']:.2%} | {benchmark['sharpe_zero_cash_monthly']:.2f} | {benchmark['annualized_volatility']:.2%} | {benchmark['maximum_drawdown']:.2%} | {benchmark['annualized_arithmetic_pnl_pct_spot_notional']:.2%} | {benchmark['annualized_premium_realistic_pct_spot_notional']:.2%} | ${benchmark['ending_wealth_per_dollar']:.3f} | {benchmark['win_rate']:.1%} |

The dynamic rule changed CAGR by {difference_cagr:+.2%} per year and Sharpe by {difference_sharpe:+.2f} relative to fixed 99/96. It selected: {counts}. Its selected short-leg IV exceeded trailing realized volatility in {positive_signal:.1%} of months.

## Fixed-candidate diagnostic

| Spread | Selected months | Mid CAGR | Mid Sharpe | Realistic CAGR | Realistic Sharpe |
|---|---:|---:|---:|---:|---:|
{chr(10).join(diagnostic_lines)}

Because all six candidates share the same realized-volatility denominator on a given entry date, ranking by percentage deviation is exactly the same as ranking by short-leg IV. Realized volatility shows whether the selected option was rich relative to the recent SPX path, but it cannot change which candidate wins this cross-sectional rule. Put skew therefore has a large influence on the chosen spread.

Sharpe uses the non-overlapping monthly option returns, a zero cash rate, and a square-root-of-12 annualization. CAGR compounds each realized monthly return over the calendar time from the first entry through the last expiration. Results are option-only and exclude collateral interest, taxes, and settlement fees.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = SPXSurfaceArchive(DATA_ROOT)
    cash_closes = load_spx_cash_closes(
        archive.populated_dates.min(), archive.populated_dates.max()
    )
    candidates, skips = build_candidate_trades(archive, cash_closes)
    selected = select_portfolio_trades(candidates)
    paths = portfolio_paths(selected)
    summary = summarize_portfolios(paths)
    yearly = yearly_results(paths)
    dynamic = paths[paths["portfolio"].eq(DYNAMIC_NAME)].copy()
    selection_counts = (
        pd.DataFrame({"spread": [structure.label.split()[1] for structure in CANDIDATE_STRUCTURES]})
        .merge(
            dynamic.groupby("spread", as_index=False)
        .agg(months=("entry_date", "size"))
            , on="spread", how="left"
        )
        .fillna({"months": 0})
        .astype({"months": int})
    )
    diagnostics = fixed_candidate_diagnostics(candidates, selection_counts)

    candidates.to_csv(OUT / "candidate_trades.csv", index=False)
    candidates.to_parquet(OUT / "candidate_trades.parquet", index=False)
    paths.to_csv(OUT / "portfolio_trades.csv", index=False)
    paths.to_parquet(OUT / "portfolio_trades.parquet", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    yearly.to_csv(OUT / "yearly_results.csv", index=False)
    selection_counts.to_csv(OUT / "selection_counts.csv", index=False)
    diagnostics.to_csv(OUT / "fixed_candidate_diagnostics.csv", index=False)
    skips.to_csv(OUT / "skips.csv", index=False)
    write_report(summary, paths, selection_counts, diagnostics)

    manifest = {
        "archive_start": str(archive.populated_dates.min().date()),
        "archive_end": str(archive.populated_dates.max().date()),
        "complete_monthly_cycles": int(candidates["entry_date"].nunique()),
        "candidate_spreads": [structure.label for structure in CANDIDATE_STRUCTURES],
        "dynamic_signal": "short-leg supplied IV / trailing 21-session realized SPX volatility - 1",
        "spread_iv_definition": "supplied implied volatility of the short put",
        "realized_volatility_definition": "sample standard deviation of 21 close-to-close log returns through entry close, annualized by sqrt(252)",
        "selection_tie_break": "lower short-strike ratio",
        "benchmark": BENCHMARK_NAME,
        "sizing": "SPX cash-spot notional equals 100% of current equity; fractional contracts",
        "spread_width": "target strike difference is 3% of entry SPX spot",
        "holding_period": "entry to expiration; non-overlapping monthly rolls",
        "sharpe": "mean monthly option return / sample standard deviation times sqrt(12); zero cash rate",
        "spot_and_settlement_source": "Yahoo Finance ^GSPC unadjusted cash close",
        "option_settlement": "PM-settled SPXW; intrinsic value at expiration-day cash SPX close",
        "realistic_execution": {
            "bid_ask_spread_fraction_away_from_mid": 0.25,
            "commission_per_contract_per_leg": 1.50,
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nDynamic selection counts")
    print(selection_counts.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
