from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from run_dynamic_iv_spread_selector import CANDIDATE_STRUCTURES
from run_requested_premium_study import CASH_CACHE

SOURCE = PROJECT / "results/dynamic_iv_spread_selector/candidate_trades.parquet"
OUT = PROJECT / "results/relative_value_spread_selector"
SPREADS = [structure.label.split()[1] for structure in CANDIDATE_STRUCTURES]
DYNAMIC = "Dynamic: own 24-month IV/RV history"
BENCHMARK = "Fixed 99/96"
LOOKBACK = 24
FILLS = ("mid", "realistic", "natural")


def prepare_candidates(candidates: pd.DataFrame, closes: pd.Series) -> pd.DataFrame:
    """Retain observed option prices; recompute the trailing cash-index RV explicitly."""
    frame = candidates.drop(columns=["iv_rank"], errors="ignore").copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"])
    if frame.duplicated(["entry_date", "spread"]).any():
        raise ValueError("duplicate spread/date observations")
    for _, group in frame.groupby("entry_date"):
        if set(group["spread"]) != set(SPREADS):
            raise ValueError("every candidate date must have all six spreads")
    closes = closes.astype(float).sort_index()
    if closes.index.has_duplicates or not np.isfinite(closes).all() or closes.le(0).any():
        raise ValueError("cash closes must be finite, positive, and unique by date")
    rv = closes.pct_change(fill_method=None).rolling(21, min_periods=21).std(ddof=1)
    frame["realized_vol_21d"] = frame["entry_date"].map(rv * math.sqrt(252.0))
    for column in ("realized_vol_21d", "short_leg_implied_vol", "long_leg_implied_vol"):
        if not np.isfinite(frame[column]).all() or frame[column].le(0).any():
            raise ValueError(f"invalid {column}")
    frame["iv_rv_pct_deviation"] = frame["short_leg_implied_vol"] / frame["realized_vol_21d"] - 1.0
    frame["iv_minus_realized_vol"] = frame["short_leg_implied_vol"] - frame["realized_vol_21d"]
    return frame.sort_values(["entry_date", "target_upper_ratio"]).reset_index(drop=True)


def history_scores(
    candidates: pd.DataFrame, lookback: int = LOOKBACK, *, mean_leg_iv: bool = False
) -> pd.DataFrame:
    """Normalize each spread by its preceding complete monthly observations only.

    The current IV/RV ratio is divided by the median of this same spread's
    previous `lookback` IV/RV ratios. Current and future observations cannot
    enter that median. The history window counts available monthly rolls.
    """
    if lookback < 2:
        raise ValueError("history needs at least two prior observations")
    scored = candidates.sort_values(["spread", "entry_date"]).copy()
    if scored.duplicated(["spread", "entry_date"]).any():
        raise ValueError("duplicate spread/date observations")
    scored["signal_iv"] = (
        (scored["short_leg_implied_vol"] + scored["long_leg_implied_vol"]) / 2.0
        if mean_leg_iv else scored["short_leg_implied_vol"]
    )
    scored["current_iv_rv"] = scored["signal_iv"] / scored["realized_vol_21d"]
    if not np.isfinite(scored["current_iv_rv"]).all() or scored["current_iv_rv"].le(0).any():
        raise ValueError("IV/RV ratios must be finite and positive")
    grouped = scored.groupby("spread", sort=False)
    scored["historical_median_iv_rv"] = grouped["current_iv_rv"].transform(
        lambda values: values.shift(1).rolling(lookback, min_periods=lookback).median()
    )
    scored["history_start_entry"] = grouped["entry_date"].shift(lookback)
    scored["history_end_entry"] = grouped["entry_date"].shift(1)
    scored["history_observations"] = grouped.cumcount().clip(upper=lookback)
    scored["richness_score"] = scored["current_iv_rv"] / scored["historical_median_iv_rv"] - 1.0
    scored["history_window"] = lookback
    scored["iv_definition"] = "mean of both legs" if mean_leg_iv else "short leg"
    return scored.sort_values(["entry_date", "target_upper_ratio"]).reset_index(drop=True)


def select_trades(scored: pd.DataFrame, name: str = DYNAMIC) -> pd.DataFrame:
    eligible = scored.loc[np.isfinite(scored["richness_score"])].copy()
    if eligible.empty:
        raise ValueError("no dates have a complete trailing history")
    for _, group in eligible.groupby("entry_date"):
        if set(group["spread"]) != set(SPREADS) or len(group) != len(SPREADS):
            raise ValueError("all six candidates must have scores on each eligible date")
    ordered = eligible.sort_values(
        ["entry_date", "richness_score", "target_upper_ratio"],
        ascending=[True, False, True], kind="stable",
    )
    dynamic = ordered.groupby("entry_date", sort=True).head(1).copy()
    dynamic["portfolio"] = name
    benchmark = eligible[eligible["spread"].eq("99/96")].copy()
    benchmark["portfolio"] = BENCHMARK
    return pd.concat([dynamic, benchmark], ignore_index=True).sort_values(["portfolio", "entry_date"])


def build_paths(selected: pd.DataFrame) -> pd.DataFrame:
    result = []
    for _, group in selected.groupby("portfolio", sort=False):
        path = group.sort_values("entry_date").copy()
        if path["expiration_date"].shift(1).gt(path["entry_date"]).any():
            raise ValueError("monthly trades overlap")
        for fill in FILLS:
            returns = path[f"pnl_{fill}_pct_spot_notional"]
            if not np.isfinite(returns).all() or returns.le(-1).any():
                raise ValueError("invalid normalized return")
            path[f"ending_equity_{fill}"] = (1.0 + returns).cumprod()
            path[f"entry_equity_{fill}"] = path[f"ending_equity_{fill}"].shift(1, fill_value=1.0)
            path[f"contracts_{fill}_per_initial_1m"] = (
                path[f"entry_equity_{fill}"] * 1_000_000.0 / (path["spot_entry"] * 100.0)
            )
        result.append(path)
    return pd.concat(result, ignore_index=True)


def performance(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, group in paths.groupby("portfolio", sort=False):
        group = group.sort_values("entry_date")
        first, last = group["entry_date"].min(), group["expiration_date"].max()
        years = (last - first).days / 365.2425
        # Every tested post-warmup month is present. Fail rather than silently
        # omitting missing months from a monthly Sharpe calculation.
        expected = pd.period_range(first, group["entry_date"].max(), freq="M")
        if not pd.PeriodIndex(group["entry_date"], freq="M").equals(expected):
            raise ValueError("performance sample has missing monthly rolls")
        row = {
            "portfolio": portfolio, "trades": len(group), "first_entry": first,
            "last_expiration": last, "elapsed_years": years,
            "spread_changes": int(group["spread"].ne(group["spread"].shift()).sum() - 1),
        }
        for fill in FILLS:
            r = group[f"pnl_{fill}_pct_spot_notional"]
            wealth = np.r_[1.0, (1.0 + r).cumprod()]
            sd = float(r.std(ddof=1))
            row.update({
                f"cagr_{fill}": float(wealth[-1] ** (1.0 / years) - 1.0),
                f"sharpe_{fill}": float(r.mean() / sd * math.sqrt(12.0)) if sd > 0 else np.nan,
                f"annual_volatility_{fill}": sd * math.sqrt(12.0),
                f"max_drawdown_at_rolls_{fill}": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
                f"ending_wealth_{fill}": float(wealth[-1]),
                f"win_rate_{fill}": float(r.gt(0).mean()),
                f"worst_month_{fill}": float(r.min()),
                f"annualized_premium_{fill}": float(group[f"premium_{fill}_pct_spot_notional"].sum() / years),
                f"annualized_arithmetic_pnl_{fill}": float(r.sum() / years),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def common_window_sensitivity(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = {window: select_trades(history_scores(candidates, window), f"Dynamic: {window}-month history")
                  for window in (12, 24, 36)}
    common_dates = set.intersection(*(set(frame["entry_date"]) for frame in selections.values()))
    rows, choices = [], []
    for window, frame in selections.items():
        matched = frame[frame["entry_date"].isin(common_dates)].copy()
        stats = performance(build_paths(matched))
        stats["history_window"] = window
        rows.append(stats)
        chosen = matched[matched["portfolio"].ne(BENCHMARK)].copy()
        chosen["history_window"] = window
        choices.append(chosen)
    return pd.concat(rows, ignore_index=True), pd.concat(choices, ignore_index=True)


def chart(paths: pd.DataFrame, counts: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), gridspec_kw={"height_ratios": [2.2, 1]})
    colors = {DYNAMIC: "#117a75", BENCHMARK: "#db8522"}
    for name, group in paths.groupby("portfolio", sort=False):
        group = group.sort_values("entry_date")
        dates = [group["entry_date"].min(), *group["expiration_date"]]
        values = np.r_[0.0, group["ending_equity_realistic"].to_numpy() - 1.0]
        axes[0].plot(dates, values, label=name, color=colors[name], linewidth=2)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Compounded option return")
    axes[0].set_title("SPX monthly relative-value selector vs. fixed 99/96", loc="left", fontsize=15)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(alpha=0.2)
    bars = axes[1].bar(counts["spread"], counts["months"], color="#117a75")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylabel("Months selected")
    axes[1].set_ylim(0, counts["months"].max() * 1.2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.08, 0.02, "100% spot notional per entry | Realistic fills | Option P&L only | Values measured at monthly expirations", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT / "comparison.png", dpi=170)
    plt.close(fig)


def write_report(summary, paths, counts, sensitivity, mean_leg_summary, fixed_summary) -> None:
    indexed = summary.set_index("portfolio")
    dynamic, benchmark = indexed.loc[DYNAMIC], indexed.loc[BENCHMARK]
    main_rows = "\n".join(
        f"| {row.portfolio} | {row.cagr_realistic:.2%} | {row.sharpe_realistic:.3f} | "
        f"{row.annual_volatility_realistic:.2%} | {row.max_drawdown_at_rolls_realistic:.2%} | {row.win_rate_realistic:.1%} |"
        for row in summary.itertuples(index=False)
    )
    sensitivity_rows = "\n".join(
        f"| {int(row.history_window)} | {row.portfolio} | {row.cagr_realistic:.2%} | {row.sharpe_realistic:.3f} |"
        for row in sensitivity.itertuples(index=False)
    )
    fixed_rows = "\n".join(
        f"| {row.portfolio} | {row.cagr_realistic:.2%} | {row.sharpe_realistic:.3f} |"
        for row in fixed_summary.itertuples(index=False)
    )
    selected = paths[paths["portfolio"].eq(DYNAMIC)]
    count_text = "; ".join(f"{row.spread}: {row.months}" for row in counts.itertuples(index=False))
    average = mean_leg_summary[mean_leg_summary["portfolio"].ne(BENCHMARK)].iloc[0]
    text = f"""# Relative value of monthly SPX put spreads

The 24-month historical normalization produces a changing allocation across all six spreads. It returns **{dynamic.cagr_realistic:.2%} CAGR and {dynamic.sharpe_realistic:.3f} Sharpe**, compared with **{benchmark.cagr_realistic:.2%} and {benchmark.sharpe_realistic:.3f}** for fixed 99/96 on identical dates. The CAGR difference is {(dynamic.cagr_realistic-benchmark.cagr_realistic)*100:+.2f} percentage points per year; the Sharpe difference is {dynamic.sharpe_realistic-benchmark.sharpe_realistic:+.3f}.

For candidate spread s at monthly entry t:

```
current_ratio(s,t) = short_put_IV(s,t) / trailing_21_session_SPX_RV(t)
usual_ratio(s,t) = median of current_ratio(s,u) over its previous 24 available monthly entries u < t
richness_score(s,t) = current_ratio(s,t) / usual_ratio(s,t) - 1
```

Sell the spread with the highest signed score among 98/95, 99/96, 100/97, 101/98, 102/99 and 103/100. A score of +25% means its IV/RV ratio is 25% above its own trailing median. Each spread has its own changing historical denominator; the current observation is excluded from that denominator. Exact ties favor the lower short strike. Trade every eligible month, including months in which every candidate scores below zero. The 24-month window is the primary specification; 12 and 36 months are sensitivity checks rather than a search for the best window.

There is a useful algebraic limit: current RV is still common to all spreads. At a particular entry, it cancels from the ranking, which becomes `short_IV(s,t) / usual_ratio(s,t)`. The historical denominator makes this a dynamic measure of relative richness on the volatility surface; current realized volatility alone does not drive the choice. This corrects the fixed preference for high-IV downside strikes, but does not establish that a whole vertical is mispriced. A vertical has two legs and no unique quoted IV. The primary score uses the short leg; a mean-of-both-legs sensitivity is provided below. This is an unhedged put-spread strategy whose returns also reflect equity exposure.

The source option archive covers September 22, 2016 through September 22, 2026. The first 24 available monthly entries establish history. There is no PM contract for the November-December 2016 cycle. The executable comparison therefore covers **{dynamic.first_entry:%B %d, %Y} through {dynamic.last_expiration:%B %d, %Y}, {int(dynamic.trades)} consecutive monthly trades**. The 10-year archive supplies the history and subsequent evaluation; a 10-year trading result cannot be claimed for this rule without earlier training data. Both strategies begin and end on the same dates.

| Portfolio | CAGR | Annualized monthly Sharpe | Annualized monthly volatility | Max drawdown at rolls | Winning months |
|---|---:|---:|---:|---:|---:|
{main_rows}

Selections: {count_text}. The spread changes {int(dynamic.spread_changes)} times between consecutive entries. In {int(selected.richness_score.lt(0).sum())} months the selected score is below zero. Full scores, historical window boundaries, strikes, leg quotes, IVs, position sizing, and cash-settlement returns are saved in the CSV/parquet artifacts.

The mean-of-both-leg-IV version with 24 prior entries produces {average.cagr_realistic:.2%} CAGR and {average.sharpe_realistic:.3f} Sharpe on the same primary dates. This tests a different spread-IV convention; neither convention is a unique implied volatility for the net spread.

The lookback checks below all use the same {int(sensitivity.trades.iloc[0])} months, from {sensitivity.first_entry.min():%B %d, %Y} to {sensitivity.last_expiration.max():%B %d, %Y}, after the longest warmup. Comparing windows on the same dates avoids mixing window effects with sample effects.

| History observations | Portfolio | CAGR | Sharpe |
|---|---|---:|---:|
{sensitivity_rows}

None of these windows beats the fixed benchmark on Sharpe over the common period. The primary CAGR advantage is small and depends on the reference window and sample dates; these results do not establish a reliable performance improvement.

For context, these fixed candidates use the same primary evaluation dates:

| Fixed strategy | CAGR | Sharpe |
|---|---:|---:|
{fixed_rows}

Execution and accounting follow the preceding research: third-Friday monthly PM-settled SPXW spreads, next monthly expiration, nearest listed strikes to the cash-SPX percentages, held to expiry. Three-wide denotes three percentage points of SPX spot. Notional equals 100% of current equity at each entry, with fractional contracts `equity / (100 * entry SPX)`. Realistic entry fills are one-quarter of the full bid/ask spread away from midpoint plus $1.50 per contract per leg; no exit spread is charged on cash settlement. Midpoint and natural bid/ask sensitivities are also in summary.csv. Option-only P&L excludes interest on collateral, taxes, and settlement fees.

Realized volatility is the sample standard deviation of 21 simple cash-SPX close-to-close returns through the entry close, annualized by sqrt(252). The cash close and the EOD option snapshot share the same-date convention from the preceding study; it assumes execution at that snapshot and is not a tested next-session fill. The archive's expiration-specific underlying_price field is never used for cash spot or settlement. CAGR compounds net monthly returns over exact elapsed calendar years. Sharpe is mean monthly option return divided by sample standard deviation times sqrt(12), with zero cash rate. Drawdown is measured at monthly rolls, not daily marked-to-market. The candidate prices reconcile to the earlier research; the selection uses only entry data and prior history, never the subsequent payoff.

![Cumulative returns and selection counts](comparison.png)

Reproduce from the existing candidate archive: `python scripts/run_relative_value_spread_selector.py`. Rebuild the candidate archive with `python scripts/run_dynamic_iv_spread_selector.py` if needed. Run checks using `python -m unittest discover -s tests -v` from the project directory.
"""
    (OUT / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Rebuild candidate archive first: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    closes = pd.read_csv(CASH_CACHE, parse_dates=["date"]).set_index("date")["spx_close"]
    candidates = prepare_candidates(pd.read_parquet(SOURCE), closes)
    scored = history_scores(candidates)
    paths = build_paths(select_trades(scored))
    summary = performance(paths)
    chosen = paths[paths["portfolio"].eq(DYNAMIC)]
    counts = chosen["spread"].value_counts().reindex(SPREADS, fill_value=0).rename_axis("spread").reset_index(name="months")
    sensitivity, sensitivity_choices = common_window_sensitivity(candidates)
    mean_leg_paths = build_paths(select_trades(history_scores(candidates, mean_leg_iv=True), "Dynamic: mean-leg IV, 24-month history"))
    mean_leg_summary = performance(mean_leg_paths)
    dates = set(paths["entry_date"])
    fixed = candidates[candidates["entry_date"].isin(dates)].copy()
    fixed["portfolio"] = "Fixed " + fixed["spread"]
    fixed_summary = performance(build_paths(fixed))
    yearly = paths.assign(year=paths["expiration_date"].dt.year).groupby(["portfolio", "year"], as_index=False).agg(
        trades=("spread", "size"),
        compounded_return=("pnl_realistic_pct_spot_notional", lambda r: float((1 + r).prod() - 1)),
        arithmetic_pnl=("pnl_realistic_pct_spot_notional", "sum"),
    )
    for name, frame in {
        "candidate_scores": scored, "portfolio_trades": paths, "summary": summary,
        "selection_counts": counts, "lookback_sensitivity_common_dates": sensitivity,
        "lookback_sensitivity_choices": sensitivity_choices, "mean_leg_iv_summary": mean_leg_summary,
        "mean_leg_iv_trades": mean_leg_paths, "fixed_candidate_summary": fixed_summary,
        "yearly_results": yearly,
    }.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    scored.to_parquet(OUT / "candidate_scores.parquet", index=False)
    paths.to_parquet(OUT / "portfolio_trades.parquet", index=False)
    chart(paths, counts)
    write_report(summary, paths, counts, sensitivity, mean_leg_summary, fixed_summary)
    manifest = {
        "candidate_source": str(SOURCE.relative_to(PROJECT)),
        "candidate_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "cash_close_source": str(CASH_CACHE.relative_to(PROJECT)),
        "cash_close_source_sha256": hashlib.sha256(CASH_CACHE.read_bytes()).hexdigest(),
        "candidates": SPREADS,
        "signal": "(short IV / RV21) / own previous 24 available monthly IV/RV median - 1",
        "lookback": LOOKBACK, "history_includes_current": False,
        "rv": "21 simple cash-close returns, sample std * sqrt(252), including entry close",
        "evaluation_first_entry": str(paths.entry_date.min().date()),
        "evaluation_last_expiration": str(paths.expiration_date.max().date()),
        "evaluation_trades_per_strategy": int(paths.entry_date.nunique()),
        "warmup_observations_per_spread": LOOKBACK,
        "notional": "100% of current equity divided by cash spot times 100; fractional contracts",
        "realistic_fill": "midpoint +/- 25% of full leg bid/ask plus $1.50 per contract per leg",
        "sharpe": "sample mean/std of monthly option returns * sqrt(12); zero cash rate",
        "cagr": "compounded wealth ^ (365.2425 / elapsed calendar days) - 1",
        "selection_uses_expiration_outcome": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[["portfolio", "trades", "cagr_realistic", "sharpe_realistic", "max_drawdown_at_rolls_realistic"]].to_string(index=False))
    print(counts.to_string(index=False))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
