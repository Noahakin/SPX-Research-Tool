"""Compare fixed weekly 97/94 at 100% and 200% equity notional on identical dates."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weekly_portfolio_evaluation import simulate_policy, summarize

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/weekly_dynamic_research"
OUT = PROJECT / "results/weekly_97_94_notional_comparison"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(SOURCE / "candidate_trades.parquet")
    marks = pd.read_parquet(SOURCE / "unit_marks.parquet")
    selected = pd.read_csv(SOURCE / "holdout_selections.csv", parse_dates=["entry_date", "expiration_date"])
    calendar = pd.DatetimeIndex(pd.read_csv(SOURCE / "market_spx.csv", parse_dates=["date"]).date)
    specs = [
        ("97/94 at 100%", "Validation-best fixed spread", 0.97, 1.0),
        ("97/94 at 200%", "Validation-best fixed spread", 0.97, 2.0),
        ("99/96 at 100%", "Fixed weekly 99/96", 0.99, 1.0),
    ]
    summaries, daily_frames, weekly_frames, annual_rows = [], [], [], []
    for name, source_name, short_ratio, multiple in specs:
        policy = selected[selected.portfolio.eq(source_name)].copy()
        chosen = candidates[candidates.candidate_id.isin(policy.candidate_id)]
        if len(chosen) != len(policy) or not np.isclose(chosen.target_short_ratio, short_ratio).all():
            raise ValueError("fixed spread selections do not match the requested strike")
        if not np.isclose(chosen.target_width_pct, 0.03).all():
            raise ValueError("fixed spread width differs from 3%")
        unit_marks = marks[marks.candidate_id.isin(policy.candidate_id)]
        for fill in ("realistic", "natural"):
            daily, weekly = simulate_policy(policy, candidates, unit_marks, calendar,
                                             fill=fill, notional_multiple=multiple)
            stats = summarize(daily, weekly)
            stats.update(portfolio=name, fill=fill, notional_multiple=multiple,
                         total_pnl=stats["ending_equity"] - stats["initial_equity"],
                         largest_entry_max_loss_pct=float(weekly.max_loss_pct.max()))
            summaries.append(stats)
            daily["portfolio"], weekly["portfolio"] = name, name
            daily_frames.append(daily)
            weekly_frames.append(weekly)
            for year, group in daily.groupby(daily.date.dt.year):
                annual_rows.append(dict(portfolio=name, fill=fill, year=year,
                                        first_date=group.date.min(), last_date=group.date.max(),
                                        return_in_period=float((1 + group.daily_return).prod() - 1)))
    summary = pd.DataFrame(summaries)
    audit_path = OUT / "independent_notional_summary.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        for row in audit.itertuples(index=False):
            name = f"97/94 at {row.notional_multiplier:.0%}"
            actual = summary[(summary.portfolio == name) & (summary.fill == row.fill)].iloc[0]
            for field in ("ending_equity", "cagr", "daily_sharpe", "max_drawdown_daily", "worst_day", "worst_week"):
                if not np.isclose(actual[field], getattr(row, field), rtol=1e-10, atol=1e-10):
                    raise ValueError(f"independent reconstruction disagrees: {name} {row.fill} {field}")
    daily_all, weekly_all = pd.concat(daily_frames), pd.concat(weekly_frames)
    summary.to_csv(OUT / "summary.csv", index=False)
    daily_all.to_csv(OUT / "daily_portfolios.csv", index=False)
    weekly_all.to_csv(OUT / "weekly_portfolios.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(OUT / "calendar_returns.csv", index=False)

    # Confirm default sizing reproduces the earlier study exactly.
    previous = pd.read_csv(SOURCE / "holdout_summary.csv")
    for name, source_name, _, multiple in specs:
        if multiple != 1:
            continue
        for fill in ("realistic", "natural"):
            actual = summary[(summary.portfolio == name) & (summary.fill == fill)].iloc[0]
            expected = previous[(previous.portfolio == source_name) & (previous.fill == fill)].iloc[0]
            for field in ("ending_equity", "cagr", "daily_sharpe", "max_drawdown_daily"):
                if not np.isclose(actual[field], expected[field], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"100% benchmark changed: {name} {fill} {field}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    colors = ["#64748b", "#0f766e", "#d97706"]
    for (name, _, _, _), color in zip(specs, colors):
        path = daily_all[(daily_all.portfolio == name) & (daily_all.fill == "realistic")].sort_values("date")
        nav = path.equity.to_numpy()
        peak = np.maximum.accumulate(np.r_[1_000_000, nav])[1:]
        axes[0].plot(path.date, nav / 1_000_000, color=color, label=name, lw=1.8)
        axes[1].plot(path.date, 100 * (nav / peak - 1), color=color, lw=1.3)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Fixed weekly SPX put spreads: 100% vs 200% notional", fontsize=15)
    axes[0].set_title("Daily option marks; weekly position sizing; slippage and commissions included", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown.png", dpi=200)
    plt.close(fig)

    shown = summary[summary.fill.eq("realistic")]
    lines = ["# Weekly 97/94: 200% notional", "",
             "January 5, 2024–September 18, 2026: 141 weekly trades, 678 daily marks. Contracts are set to 100% or 200% of current equity in cash-SPX notional at each weekly entry, then held fixed until expiry. Strike targets remain 97%/94%. Friday-to-Friday maturities are normally seven calendar days, with holiday adjustments.", "",
             "| Portfolio | CAGR | Daily Sharpe | Max drawdown | Worst day | Worst week | Ending $1M |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in shown.itertuples(index=False):
        lines.append(f"| {row.portfolio} | {row.cagr:.2%} | {row.daily_sharpe:.4f} | {row.max_drawdown_daily:.2%} | {row.worst_day:.2%} | {row.worst_week:.2%} | ${row.ending_equity:,.0f} |")
    natural = summary[(summary.portfolio == "97/94 at 200%") & (summary.fill == "natural")].iloc[0]
    lines += ["", "[Daily equity and drawdown chart](equity_drawdown.png)", "",
              "Execution assumes each leg fills 25% of its full bid/ask spread away from midpoint plus $1.50 per contract per leg. All costs scale with position size. Cash earns zero, and no additional borrowing or market-impact charge is modeled. This is SPX notional exposure; the theoretical maximum loss per 200%-notional 3%-wide spread is approximately 6% of entry equity less net premium.", "",
              f"At full bid/ask execution, 200% notional produces {natural.cagr:.2%} CAGR, {natural.daily_sharpe:.4f} daily Sharpe and {natural.max_drawdown_daily:.2%} maximum drawdown.", "",
              "The 100% results reconcile to the previous study. Increasing notional does not double Sharpe: both expected P&L and fluctuations scale, while weekly compounding and intraperiod changes in equity produce small deviations from exact proportionality.", "",
              "Data: [summary](summary.csv), [daily NAV](daily_portfolios.csv), [weekly trades](weekly_portfolios.csv), [calendar returns](calendar_returns.csv)."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary[["portfolio", "fill", "cagr", "daily_sharpe", "max_drawdown_daily", "worst_day", "worst_week", "ending_equity"]].to_string(index=False))


if __name__ == "__main__":
    main()
