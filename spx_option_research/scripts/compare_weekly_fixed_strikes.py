"""Weekly 96/93, 97/94 and 98/95 at 100%/200% notional, with daily marks."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_weekly_spread_candidates import LEG_FIELDS, build_candidate_record
from build_weekly_daily_marks import quote_requirements, unit_marks_from_quotes
from weekly_portfolio_evaluation import simulate_policy, summarize

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/weekly_dynamic_research"
OUT = PROJECT / "results/weekly_fixed_strike_comparison"


def build_96_93(existing, schedule):
    """Reuse exact quoted puts already selected at 96% and 93% of the same cash spot."""
    references = existing[np.isclose(existing.target_short_ratio, 0.98)]
    short_sources = references[np.isclose(references.target_width_pct, 0.02)].set_index("entry_date")
    long_sources = references[np.isclose(references.target_width_pct, 0.05)].set_index("entry_date")
    records, provenance = [], []
    for week in schedule.itertuples(index=False):
        short_source = short_sources.loc[week.entry_date]
        long_source = long_sources.loc[week.entry_date]
        for field in ("expiration_date", "spot_entry", "spot_expiration"):
            if short_source[field] != long_source[field]:
                raise ValueError("source legs disagree on trade dates or cash prices")
        if short_source.expiration_date != week.expiration_date:
            raise ValueError("source legs disagree with the fixed benchmark schedule")
        puts = []
        for source, target in ((short_source, 0.96), (long_source, 0.93)):
            if not np.isclose(source.target_long_ratio, target):
                raise ValueError("source leg has a different cash-spot strike target")
            quote = {field: source[f"long_{field}"] for field in LEG_FIELDS if field != "symbol"}
            quote.update(option_symbol=source.long_symbol, strike=source.long_strike)
            puts.append(quote)
        record = build_candidate_record(
            entry_date=week.entry_date, expiration_date=week.expiration_date,
            scheduled_entry_friday=short_source.scheduled_entry_friday,
            scheduled_expiration_friday=short_source.scheduled_expiration_friday,
            spot_entry=float(short_source.spot_entry), spot_expiration=float(short_source.spot_expiration),
            short_ratio=0.96, width=0.03, puts=pd.DataFrame(puts))
        records.append(record)
        provenance.append(dict(candidate_id=record["candidate_id"],
                               short_leg_source_candidate=short_source.candidate_id,
                               long_leg_source_candidate=long_source.candidate_id,
                               short_symbol=record["short_symbol"], long_symbol=record["long_symbol"]))
    return pd.DataFrame(records), pd.DataFrame(provenance)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    existing = pd.read_parquet(SOURCE / "candidate_trades.parquet")
    choices = pd.read_csv(SOURCE / "holdout_selections.csv", parse_dates=["entry_date", "expiration_date"])
    schedule = choices[choices.portfolio.eq("Fixed weekly 99/96")][["entry_date", "expiration_date"]].sort_values("entry_date")
    cash = pd.read_csv(SOURCE / "market_spx.csv", parse_dates=["date"]).set_index("date").close
    added, provenance = build_96_93(existing, schedule)
    added.to_parquet(OUT / "candidate_96_93.parquet", index=False)
    added.to_csv(OUT / "candidate_96_93.csv", index=False)
    provenance.to_csv(OUT / "candidate_96_93_provenance.csv", index=False)

    quotes = pd.read_parquet(SOURCE / "weekly_daily_option_quotes.parquet")
    wanted = quote_requirements(added, cash.index)
    keys = pd.MultiIndex.from_tuples([(day, symbol) for day, symbols in wanted.items() for symbol in sorted(symbols)],
                                     names=["snapshot_date", "option_symbol"])
    indexed = quotes.set_index(["snapshot_date", "option_symbol"])
    if len(keys.difference(indexed.index)):
        raise ValueError("new 96/93 requires an uncached exact daily option quote")
    selected_quotes = indexed.loc[keys].reset_index()
    added_marks = unit_marks_from_quotes(added, selected_quotes, cash)
    added_marks.to_parquet(OUT / "unit_marks_96_93.parquet", index=False)
    existing = existing[existing.entry_date.isin(schedule.entry_date) & np.isclose(existing.target_width_pct, 0.03)
                        & existing.target_short_ratio.isin([0.97, 0.98, 0.99])].copy()
    candidates = pd.concat([existing, added], ignore_index=True)
    marks = pd.read_parquet(SOURCE / "unit_marks.parquet")
    marks = pd.concat([marks[marks.candidate_id.isin(existing.candidate_id)], added_marks], ignore_index=True)
    candidates.to_parquet(OUT / "comparison_candidates.parquet", index=False)
    marks.to_parquet(OUT / "comparison_unit_marks.parquet", index=False)
    marks[marks.mid_outside_payoff_bounds].to_csv(OUT / "mark_bound_flags.csv", index=False)

    specs = [(ratio, multiple) for ratio in (0.96, 0.97, 0.98) for multiple in (1.0, 2.0)] + [(0.99, 1.0)]
    summaries, daily_frames, weekly_frames, annual = [], [], [], []
    for ratio, multiple in specs:
        spread = f"{ratio:.0%}/{ratio - .03:.0%}".replace("%", "")
        name = f"{spread} at {multiple:.0%}"
        chosen = candidates[np.isclose(candidates.target_short_ratio, ratio)].copy()
        selection = schedule.merge(chosen[["entry_date", "candidate_id"]], on="entry_date", validate="one_to_one", how="left")
        if selection.candidate_id.isna().any() or len(chosen) != len(schedule):
            raise ValueError(f"incomplete fixed spread history for {spread}")
        units = marks[marks.candidate_id.isin(selection.candidate_id)]
        for fill in ("realistic", "natural"):
            daily, weekly = simulate_policy(selection, candidates, units, cash.index, fill=fill, notional_multiple=multiple)
            stats = summarize(daily, weekly)
            stats.update(portfolio=name, spread=spread, target_short_ratio=ratio, notional_multiple=multiple,
                         fill=fill, total_pnl=stats["ending_equity"] - stats["initial_equity"],
                         entry_max_loss_pct=float(weekly.max_loss_pct.max()),
                         quote_bound_flags=int(units.mid_outside_payoff_bounds.sum()))
            summaries.append(stats)
            daily["portfolio"], weekly["portfolio"] = name, name
            daily_frames.append(daily)
            weekly_frames.append(weekly)
            for year, group in daily.groupby(daily.date.dt.year):
                annual.append(dict(portfolio=name, fill=fill, year=year, first_date=group.date.min(),
                                   last_date=group.date.max(), return_in_period=float((1 + group.daily_return).prod() - 1)))
    summary = pd.DataFrame(summaries)
    daily_all, weekly_all = pd.concat(daily_frames), pd.concat(weekly_frames)
    summary.to_csv(OUT / "summary.csv", index=False)
    daily_all.to_csv(OUT / "daily_portfolios.csv", index=False)
    weekly_all.to_csv(OUT / "weekly_portfolios.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT / "calendar_returns.csv", index=False)

    previous = pd.read_csv(PROJECT / "results/weekly_97_94_notional_comparison/summary.csv")
    for row in previous.itertuples(index=False):
        current = summary[(summary.portfolio == row.portfolio) & (summary.fill == row.fill)].iloc[0]
        for field in ("ending_equity", "cagr", "daily_sharpe", "max_drawdown_daily", "worst_day", "worst_week"):
            if not np.isclose(current[field], getattr(row, field), rtol=1e-10, atol=1e-10):
                raise ValueError(f"previous benchmark changed: {row.portfolio} {field}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    portfolios = ["96/93 at 200%", "97/94 at 200%", "98/95 at 200%", "99/96 at 100%"]
    for name, color in zip(portfolios, ["#2563eb", "#0f766e", "#dc2626", "#a16207"]):
        path = daily_all[(daily_all.portfolio == name) & (daily_all.fill == "realistic")].sort_values("date")
        nav = path.equity.to_numpy()
        peak = np.maximum.accumulate(np.r_[1_000_000, nav])[1:]
        axes[0].plot(path.date, nav / 1_000_000, label=name, color=color, lw=1.7)
        axes[1].plot(path.date, 100 * (nav / peak - 1), color=color, lw=1.2)
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_ylabel("Drawdown (%)")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Fixed weekly SPX put spreads at 200% notional", fontsize=15)
    axes[0].set_title("Daily option marks; weekly resizing; slippage and commissions included", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown.png", dpi=200)
    plt.close(fig)

    lines = ["# Fixed weekly 96/93, 97/94 and 98/95", "",
             "January 5, 2024–September 18, 2026; 141 weekly trades, 678 daily marks. Short and long strikes are nearest listed puts at the specified ratios of cash SPX. Maturities are normally seven calendar days, holiday adjusted. Positions are resized weekly to 100% or 200% of current equity in SPX spot notional and held to PM cash settlement.", "",
             "| Spread | Notional | CAGR | Daily Sharpe | Max drawdown | Worst day | Worst week | Ending $1M |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary[summary.fill.eq("realistic")].itertuples(index=False):
        lines.append(f"| {row.spread} | {row.notional_multiple:.0%} | {row.cagr:.2%} | {row.daily_sharpe:.4f} | {row.max_drawdown_daily:.2%} | {row.worst_day:.2%} | {row.worst_week:.2%} | ${row.ending_equity:,.0f} |")
    lines += ["", "[Equity and drawdown chart](equity_drawdown.png)", "",
              "Execution assumes 25% of each full bid/ask spread away from midpoint plus $1.50 per contract per leg. Costs scale with contracts. Cash earns zero; no additional financing or market-impact charge is modeled. The full bid/ask execution sensitivity is in summary.csv.", "",
              "The 96/93 spread is a new requested comparison after the earlier research. Its quoted 96% put was already held as the hedge in 98/96, and its quoted 93% put as the hedge in 98/93. These exact entry contracts and daily quotes were reused to form and mark the new spread. No option prices were interpolated, and all 141 expirations reconcile to cash intrinsic. Provenance is retained by contract symbol and source candidate.", "",
              "The 97/94 and 99/96 results reproduce the previous study. These are descriptive comparisons on the same previously examined period, not a new independent evaluation period.", "",
              "[Summary including bid/ask sensitivity](summary.csv) · [Daily NAV](daily_portfolios.csv) · [Weekly trades](weekly_portfolios.csv) · [Calendar returns](calendar_returns.csv)"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(dict(first_entry=str(schedule.entry_date.min().date()),
        last_expiration=str(schedule.expiration_date.max().date()), weeks=len(schedule), new_96_93_candidates=len(added),
        new_exact_daily_leg_quotes=len(selected_quotes), new_96_93_unit_marks=len(added_marks),
        new_96_93_bound_flags=int(added_marks.mid_outside_payoff_bounds.sum()),
        notional_multiples=[1, 2], quote_source=str(SOURCE / "weekly_daily_option_quotes.parquet")), indent=2), encoding="utf-8")
    print(summary[["portfolio", "fill", "cagr", "daily_sharpe", "max_drawdown_daily", "worst_week", "ending_equity", "quote_bound_flags"]].to_string(index=False))


if __name__ == "__main__":
    main()
