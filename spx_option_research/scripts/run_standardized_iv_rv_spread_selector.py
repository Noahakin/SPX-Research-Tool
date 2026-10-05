from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import run_relative_value_spread_selector as base

OUT = PROJECT / "results/standardized_iv_rv_spread_selector"
DYNAMIC = "Dynamic: 24-month IV/RV z-score"


def standardized_scores(candidates: pd.DataFrame, lookback: int = 24, *, mean_leg_iv: bool = False) -> pd.DataFrame:
    """Current IV/RV standardized by that spread's own strictly prior history."""
    scored = base.history_scores(candidates, lookback, mean_leg_iv=mean_leg_iv)
    scored = scored.sort_values(["spread", "entry_date"])
    grouped = scored.groupby("spread", sort=False)["current_iv_rv"]
    scored["historical_mean_iv_rv"] = grouped.transform(
        lambda x: x.shift(1).rolling(lookback, min_periods=lookback).mean()
    )
    scored["historical_std_iv_rv"] = grouped.transform(
        lambda x: x.shift(1).rolling(lookback, min_periods=lookback).std(ddof=1)
    )
    scored["relative_ratio_score"] = scored["richness_score"]
    scored["richness_score"] = (
        (scored["current_iv_rv"] - scored["historical_mean_iv_rv"])
        / scored["historical_std_iv_rv"].where(scored["historical_std_iv_rv"].gt(1e-12))
    )
    scored["score_definition"] = "own-history IV/RV z-score"
    return scored.sort_values(["entry_date", "target_upper_ratio"]).reset_index(drop=True)


def sensitivity(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = {w: base.select_trades(standardized_scores(candidates, w), f"Dynamic: {w}-month IV/RV z-score") for w in (12, 24, 36)}
    common = set.intersection(*(set(x.entry_date) for x in runs.values()))
    rows, trades = [], []
    for w, selected in runs.items():
        paths = base.build_paths(selected[selected.entry_date.isin(common)])
        stats = base.performance(paths)
        stats["history_window"] = w
        rows.append(stats)
        paths["history_window"] = w
        trades.append(paths)
    return pd.concat(rows, ignore_index=True), pd.concat(trades, ignore_index=True)


def comparison_chart(paths: pd.DataFrame, counts: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})
    for name, group in paths.groupby("portfolio", sort=False):
        group = group.sort_values("entry_date")
        axes[0].plot([group.entry_date.min(), *group.expiration_date],
                     np.r_[0, group.ending_equity_realistic - 1], label=name, linewidth=2)
    axes[0].set_title("SPX monthly spread selection: IV/RV relative to its own history", loc="left", fontsize=14)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Compounded option return")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    bars = axes[1].bar(counts.spread, counts.months, color="#257e92")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylabel("Months selected")
    axes[1].set_ylim(0, counts.months.max() * 1.2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.08, 0.02, "100% spot notional | Realistic fills | Monthly expiration values | Option P&L only", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)


def report(summary, paths, counts, checks, mean_stats, ratio_stats) -> None:
    indexed = summary.set_index("portfolio")
    d, b = indexed.loc[DYNAMIC], indexed.loc[base.BENCHMARK]
    rows = "\n".join(f"| {r.portfolio} | {r.cagr_realistic:.2%} | {r.sharpe_realistic:.3f} | {r.annual_volatility_realistic:.2%} | {r.max_drawdown_at_rolls_realistic:.2%} |" for r in summary.itertuples(index=False))
    check_rows = "\n".join(f"| {r.history_window} | {r.portfolio} | {r.cagr_realistic:.2%} | {r.sharpe_realistic:.3f} |" for r in checks.itertuples(index=False))
    choices = "; ".join(f"{r.spread}: {r.months}" for r in counts.itertuples(index=False))
    dynamic_trades = paths[paths.portfolio.eq(DYNAMIC)]
    mean = mean_stats[mean_stats.portfolio.ne(base.BENCHMARK)].iloc[0]
    ratio = ratio_stats[ratio_stats.portfolio.ne(base.BENCHMARK)].iloc[0]
    text = f"""# Monthly SPX spread selection by historical IV/RV z-score

This rule measures how unusually high each candidate's current IV/RV ratio is relative to its own history. It has a changing numerator (distance from its own usual ratio) and a changing, spread-specific denominator (the usual variability of that ratio). Current realized volatility can affect the ranking, because each spread has a different historical mean and standard deviation.

For each of 98/95, 99/96, 100/97, 101/98, 102/99 and 103/100:

```
X(s,t) = short_put_IV(s,t) / trailing_21_session_SPX_RV(t)
mu(s,t) = mean of X for this spread's previous 24 available monthly entries
sd(s,t) = sample standard deviation of those same 24 previous X observations
score(s,t) = (X(s,t) - mu(s,t)) / sd(s,t)
```

Sell the spread with the highest signed score. A score of +2 means its IV/RV ratio is two historical standard deviations above its own previous average. All baseline observations strictly precede the current entry. Twenty-four observations is the primary window, with 12 and 36 as prespecified sensitivity checks. This window was retained before running the z-score results. No window or score is selected based on subsequent trading performance. Exact ties favor the lower short strike. A zero historical standard deviation invalidates the score; it does not arise in the actual sample. Every eligible month has all six candidates, and the strategy always trades even if all scores are negative.

The rule earns **{d.cagr_realistic:.2%} CAGR with {d.sharpe_realistic:.3f} Sharpe**, compared with **{b.cagr_realistic:.2%} and {b.sharpe_realistic:.3f}** for fixed 99/96. It changes spreads {int(d.spread_changes)} times across **{int(d.trades)} monthly trades, {d.first_entry:%B %d, %Y} through {d.last_expiration:%B %d, %Y}**. The archive covers September 2016-September 2026; the first 24 available monthly entries supply the historical reference, so this is about eight years of trading evaluation using ten years of data. The unavailable November-December 2016 PM contract lies in the history-building period. Every evaluation month is present.

| Strategy | CAGR | Annualized monthly Sharpe | Annualized monthly volatility | Max drawdown at rolls |
|---|---:|---:|---:|---:|
{rows}

Selections: {choices}. The selected score is negative in {int(dynamic_trades.richness_score.lt(0).sum())} months. These are the richest available candidates under the rule, not evidence of absolute overpricing. A vertical has no unique quoted IV; this specification ranks the short-leg IV and realizes P&L on both legs. It measures relative volatility richness rather than estimating a full spread's economic fair value. A two-leg fair-value residual using contemporaneous forward, discount rate, and a realized-volatility payoff model would test a different hypothesis.

The 12-, 24-, and 36-observation checks use identical dates after the longest history requirement: {checks.first_entry.min():%B %d, %Y} through {checks.last_expiration.max():%B %d, %Y}, {int(checks.trades.iloc[0])} months.

| History observations | Strategy | CAGR | Sharpe |
|---|---|---:|---:|
{check_rows}

The 24- and 36-observation versions beat the benchmark on CAGR and Sharpe over the common period. The 12-observation version loses on Sharpe, so the improvement is sensitive to the reference window. Much of the primary strategy's advantage comes from its 2023 trades. These are exploratory results, with the score family refined during this research; they are not an independent holdout validation or proof of a persistent edge.

Using the mean of the two leg IVs in X, with the same 24-prior-observation z-score, gives {mean.cagr_realistic:.2%} CAGR and {mean.sharpe_realistic:.3f} Sharpe over the primary period. The simpler own-median ratio normalization previously examined gives {ratio.cagr_realistic:.2%} CAGR and {ratio.sharpe_realistic:.3f} Sharpe. Unlike the z-score, that simpler ratio makes current RV cancel from the monthly ranking. These alternatives are diagnostics; none is evidence that a chosen spread is necessarily overpriced.

All comparisons use 100% of current equity as SPX cash-spot notional each month, with contracts = equity / (cash SPX * 100), fractional contracts, and compounding. Three-wide means three percentage points of cash SPX, rounded to nearest listed strikes. Positions are PM-settled SPXW put spreads entered at the third-Friday monthly EOD snapshot and held to the next monthly expiration, with cash intrinsic settlement. Realistic fills are 25% of the full bid/ask spread away from midpoint plus $1.50 per contract per leg. Midpoint and natural bid/ask sensitivity results are also saved. Cash spot and settlement come from cached Yahoo ^GSPC closes, not the archive's expiration-specific underlying_price field.

RV is the sample standard deviation of the latest 21 simple close-to-close SPX returns through the entry close times sqrt(252). The entry-close and EOD quote convention assumes execution at the same snapshot. Future prices and future option payoffs never enter the selection or historical reference. CAGR compounds normalized monthly P&L over exact elapsed calendar years. Sharpe uses mean monthly option return / sample standard deviation * sqrt(12), with zero cash rate. No collateral interest, taxes, or settlement fees are included. Drawdown is observed at monthly rolls and does not measure intramonth marked-to-market losses.

![Performance and spread choices](comparison.png)

Reproduce: `python scripts/run_standardized_iv_rv_spread_selector.py` from the project directory. Source candidates are built by `python scripts/run_dynamic_iv_spread_selector.py`. Tests: `python -m unittest discover -s tests -v`. Entry-level score histories and all six candidate outcomes are saved for audit.
"""
    (OUT / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    closes = pd.read_csv(base.CASH_CACHE, parse_dates=["date"]).set_index("date").spx_close
    candidates = base.prepare_candidates(pd.read_parquet(base.SOURCE), closes)
    scored = standardized_scores(candidates)
    paths = base.build_paths(base.select_trades(scored, DYNAMIC))
    summary = base.performance(paths)
    counts = paths[paths.portfolio.eq(DYNAMIC)].spread.value_counts().reindex(base.SPREADS, fill_value=0).rename_axis("spread").reset_index(name="months")
    checks, check_trades = sensitivity(candidates)
    mean_paths = base.build_paths(base.select_trades(standardized_scores(candidates, mean_leg_iv=True), "Dynamic: mean-leg IV/RV z-score"))
    mean_stats = base.performance(mean_paths)
    ratio_paths = base.build_paths(base.select_trades(base.history_scores(candidates)))
    ratio_stats = base.performance(ratio_paths)
    yearly = paths.assign(year=paths.expiration_date.dt.year).groupby(["portfolio", "year"], as_index=False).agg(
        trades=("spread", "size"),
        compounded_return=("pnl_realistic_pct_spot_notional", lambda r: float((1 + r).prod() - 1)),
        arithmetic_pnl=("pnl_realistic_pct_spot_notional", "sum"),
    )
    for name, frame in {
        "candidate_scores": scored, "portfolio_trades": paths, "summary": summary,
        "selection_counts": counts, "lookback_sensitivity_common_dates": checks,
        "lookback_sensitivity_trades": check_trades, "mean_leg_iv_summary": mean_stats,
        "mean_leg_iv_trades": mean_paths, "ratio_normalization_summary": ratio_stats,
        "yearly_results": yearly,
    }.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    paths.to_parquet(OUT / "portfolio_trades.parquet", index=False)
    scored.to_parquet(OUT / "candidate_scores.parquet", index=False)
    comparison_chart(paths, counts)
    report(summary, paths, counts, checks, mean_stats, ratio_stats)
    import hashlib
    manifest = {
        "signal": "z-score of IV/RV relative to same spread's previous 24 monthly observations",
        "historical_mean_and_sample_std_exclude_current": True,
        "candidate_source": str(base.SOURCE.relative_to(PROJECT)),
        "candidate_source_sha256": hashlib.sha256(base.SOURCE.read_bytes()).hexdigest(),
        "cash_source_sha256": hashlib.sha256(base.CASH_CACHE.read_bytes()).hexdigest(),
        "rv": "21 simple cash-SPX returns through entry close; sample std * sqrt(252)",
        "primary_history": 24, "sensitivity_windows_common_dates": [12, 24, 36],
        "first_entry": str(paths.entry_date.min().date()), "last_expiration": str(paths.expiration_date.max().date()),
        "trades_per_strategy": int(paths.entry_date.nunique()),
        "notional": "100% of current equity using cash spot times 100",
        "sharpe": "monthly, annualized by sqrt(12), zero cash rate",
        "script": "scripts/run_standardized_iv_rv_spread_selector.py",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[["portfolio", "trades", "cagr_realistic", "sharpe_realistic"]].to_string(index=False))
    print(counts.to_string(index=False))
    print(checks[["history_window", "portfolio", "cagr_realistic", "sharpe_realistic"]].to_string(index=False))


if __name__ == "__main__":
    main()
