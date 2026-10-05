"""Full-archive fixed weekly spreads, including quiet weeks and daily mark sensitivities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from build_weekly_spread_candidates import DATA_ROOT, LEG_FIELDS, SPXSurfaceArchive, build_candidate_record, valid_pm_puts
from build_weekly_daily_marks import QUOTE_COLUMNS, quote_requirements, unit_marks_from_quotes
from weekly_portfolio_evaluation import simulate_policy, summarize

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/weekly_dynamic_research"
OUT = PROJECT / "results/weekly_fixed_strikes_10y"
RATIOS = (.96, .97, .98, .99)
PRIMARY = "adjacent_put_estimate"


def build_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    original = pd.read_parquet(SOURCE / "candidate_trades.parquet")
    schedule = pd.read_csv(SOURCE / "weekly_availability.csv", parse_dates=["entry_date", "expiration_date", "scheduled_entry_friday", "scheduled_expiration_friday"])
    schedule.to_csv(OUT / "schedule.csv", index=False)
    cash = pd.read_csv(SOURCE / "market_spx.csv", parse_dates=["date"]).set_index("date").close
    existing = original[np.isclose(original.target_width_pct, .03)].set_index(["entry_date", "target_short_ratio"])
    short_refs = original[np.isclose(original.target_short_ratio, .98) & np.isclose(original.target_width_pct, .02)].set_index("entry_date")
    long_refs = original[np.isclose(original.target_short_ratio, .98) & np.isclose(original.target_width_pct, .05)].set_index("entry_date")
    archive = None
    records, provenance, missing = [], [], []
    for week in schedule.itertuples(index=False):
        if week.available_candidates == 0:
            missing.append(dict(entry_date=week.entry_date, expiration_date=week.expiration_date,
                                reason="Required weekly PM expiration absent from archive; explicit cash week for every strategy"))
            continue
        for ratio in RATIOS:
            key = (week.entry_date, ratio)
            if key in existing.index:
                record = existing.loc[key].to_dict()
                record.update(entry_date=week.entry_date, target_short_ratio=ratio)
                records.append(record)
                provenance.append(dict(candidate_id=record["candidate_id"], source="existing exact candidate"))
                continue
            source_name = "entry archive, fixed rule retains nonpositive net credit"
            if ratio == .96 and week.entry_date in short_refs.index and week.entry_date in long_refs.index:
                legs = []
                for reference in (short_refs.loc[week.entry_date], long_refs.loc[week.entry_date]):
                    if reference.expiration_date != week.expiration_date or reference.spot_entry != cash.loc[week.entry_date]:
                        raise ValueError("cached source-leg timing/spot mismatch")
                    leg = {field: reference[f"long_{field}"] for field in LEG_FIELDS if field != "symbol"}
                    leg.update(strike=reference.long_strike, option_symbol=reference.long_symbol)
                    legs.append(leg)
                puts = pd.DataFrame(legs)
                source_name = "exact 96% and 93% puts previously held as long legs"
            else:
                if archive is None:
                    archive = SPXSurfaceArchive(DATA_ROOT, cache_size=8)
                puts = valid_pm_puts(archive.read(week.entry_date), week.expiration_date)
                if puts.empty:
                    raise ValueError(f"unexpected missing entry quotes: {week.entry_date} {ratio}")
            record = build_candidate_record(entry_date=week.entry_date, expiration_date=week.expiration_date,
                scheduled_entry_friday=week.scheduled_entry_friday, scheduled_expiration_friday=week.scheduled_expiration_friday,
                spot_entry=float(cash.loc[week.entry_date]), spot_expiration=float(cash.loc[week.expiration_date]),
                short_ratio=ratio, width=.03, puts=puts, require_positive_credit=False)
            records.append(record)
            provenance.append(dict(candidate_id=record["candidate_id"], source=source_name))
    candidates = pd.DataFrame(records).sort_values(["entry_date", "target_short_ratio"]).reset_index(drop=True)
    if candidates.groupby("target_short_ratio").size().ne(520).any():
        raise ValueError("expected 520 quoted cycles per fixed strike")
    candidates.to_parquet(OUT / "comparison_candidates.parquet", index=False)
    candidates.to_csv(OUT / "comparison_candidates.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUT / "candidate_provenance.csv", index=False)
    pd.DataFrame(missing).to_csv(OUT / "cash_weeks.csv", index=False)
    selections = []
    for ratio in RATIOS:
        selected = schedule[["entry_date", "expiration_date"]].merge(
            candidates[np.isclose(candidates.target_short_ratio, ratio)][["entry_date", "candidate_id"]],
            how="left", on="entry_date", validate="one_to_one")
        selected["target_short_ratio"] = ratio
        if selected.candidate_id.isna().sum() != 1:
            raise ValueError("unexpected missing trade beyond common archive gap")
        selections.append(selected)
    pd.concat(selections, ignore_index=True).to_csv(OUT / "selections.csv", index=False)
    print(f"Prepared {len(candidates):,} exact fixed-spread records; {len(schedule)} scheduled cycles, one cash week.", flush=True)

    wanted = quote_requirements(candidates, cash.index)
    keys = pd.MultiIndex.from_tuples([(day, symbol) for day, symbols in wanted.items() for symbol in sorted(symbols)], names=["snapshot_date", "option_symbol"])
    cached = pd.read_parquet(SOURCE / "weekly_daily_option_quotes.parquet").set_index(["snapshot_date", "option_symbol"])
    absent = keys.difference(cached.index)
    quotes = cached.loc[keys.intersection(cached.index)].reset_index()
    extra, extra_files = [], []
    if len(absent):
        if archive is None:
            archive = SPXSurfaceArchive(DATA_ROOT, cache_size=1)
        for day, group in absent.to_frame(index=False).groupby("snapshot_date"):
            path = archive.path_for(day)
            frame = pq.ParquetFile(path).read(columns=QUOTE_COLUMNS).to_pandas()
            frame["option_symbol"] = frame.option_symbol.astype(str).str.strip()
            frame = frame[frame.option_symbol.isin(group.option_symbol)].copy()
            frame["source_file"] = str(path.relative_to(archive.root))
            frame["mid"] = (frame.bid + frame.ask) / 2
            extra.append(frame)
            extra_files.append(dict(date=str(day.date()), path=str(path), bytes=path.stat().st_size))
        quotes = pd.concat([quotes, *extra], ignore_index=True)
    if len(quotes) != len(keys):
        raise ValueError("missing or duplicate requested daily quotes")
    for column in ("snapshot_date", "expiration_date"):
        quotes[column] = pd.to_datetime(quotes[column]).dt.normalize()
    quotes.to_parquet(OUT / "daily_option_quotes.parquet", index=False)
    marks = unit_marks_from_quotes(candidates, quotes, cash)
    marks.to_parquet(OUT / "unit_marks_raw.parquet", index=False)
    marks[marks.mid_outside_payoff_bounds].to_csv(OUT / "raw_mark_bound_flags.csv", index=False)
    candidates[["candidate_id", "entry_date", "target_short_ratio", *[f"premium_{fill}_pct_spot_notional" for fill in ("mid", "realistic", "natural")]]].loc[
        candidates.premium_realistic_pct_spot_notional.le(0) | candidates.premium_natural_pct_spot_notional.le(0)
    ].to_csv(OUT / "nonpositive_net_credit_trades.csv", index=False)
    manifest = dict(first_entry=str(schedule.entry_date.min().date()), last_expiration=str(schedule.expiration_date.max().date()),
                    scheduled_cycles=len(schedule), quoted_cycles=520, cash_cycles=1, candidates=len(candidates), unit_marks=len(marks),
                    exact_daily_leg_quotes=len(quotes), supplemental_quotes=len(absent), supplemental_files=extra_files,
                    raw_mark_bound_flags=int(marks.mid_outside_payoff_bounds.sum()),
                    original_quote_manifest_sha256=hashlib.sha256((SOURCE / "weekly_daily_quotes_manifest.json").read_bytes()).hexdigest(),
                    fixed_rule="Every quoted weekly cycle, including net debits after execution costs; no premium-sign gate")
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validated {len(quotes):,} daily leg quotes ({len(absent)} supplemental); {len(marks):,} marks; all settlements reconcile.", flush=True)


def adjusted_marks(raw, candidates, convention, evidence):
    result = raw.copy()
    spot = result.candidate_id.map(candidates.set_index("candidate_id").spot_entry)
    new_liability = result.spread_mid_points.copy()
    if convention != "observed_raw":
        for row in evidence.itertuples(index=False):
            mask = result.candidate_id.eq(row.candidate_id) & result.date.eq(row.date)
            if mask.sum() != 1:
                raise ValueError("adjustment must identify exactly one held candidate/day")
            field = "call_parity_spread" if convention == "call_parity_estimate" else "adjacent_put_spread"
            new_liability.loc[mask] = getattr(row, field)
        if convention == "bounded_adjacent_estimate":
            new_liability = new_liability.clip(lower=0, upper=result.width_points)
    adjustment = (result.spread_mid_points - new_liability) / spot
    if adjustment[result.is_expiration | result.date.eq(result.entry_date)].abs().gt(1e-12).any():
        raise ValueError("daily mark estimates cannot change entry or settlement")
    for fill in ("mid", "realistic", "natural"):
        result[f"cum_return_{fill}"] += adjustment
    result["valuation_spread_points"] = new_liability
    return result


def evaluate():
    candidates = pd.read_parquet(OUT / "comparison_candidates.parquet")
    raw = pd.read_parquet(OUT / "unit_marks_raw.parquet")
    selections = pd.read_csv(OUT / "selections.csv", parse_dates=["entry_date", "expiration_date"])
    cash = pd.read_csv(SOURCE / "market_spx.csv", parse_dates=["date"]).set_index("date").close
    evidence = pd.read_csv(OUT / "mark_adjustments.csv", parse_dates=["date"])
    specs = [(r, n) for r in (.96, .97, .98) for n in (1., 2.)] + [(.99, 1.)]
    summaries, daily_frames, weekly_frames, annual = [], [], [], []
    for convention in ("observed_raw", PRIMARY, "call_parity_estimate", "bounded_adjacent_estimate"):
        marks = adjusted_marks(raw, candidates, convention, evidence)
        if convention == PRIMARY:
            marks.to_parquet(OUT / "unit_marks_primary.parquet", index=False)
        for ratio, multiple in specs:
            spread = f"{ratio:.0%}/{ratio-.03:.0%}".replace("%", "")
            name = f"{spread} at {multiple:.0%}"
            policy = selections[np.isclose(selections.target_short_ratio, ratio)]
            units = marks[marks.candidate_id.isin(policy.candidate_id.dropna())]
            for fill in ("realistic", "natural"):
                daily, weekly = simulate_policy(policy, candidates, units, cash.index, fill=fill, notional_multiple=multiple)
                stats = summarize(daily, weekly)
                active = weekly[weekly.traded]
                stats.update(portfolio=name, spread=spread, notional_multiple=multiple, fill=fill, marking_convention=convention,
                             total_pnl=stats["ending_equity"]-stats["initial_equity"], win_rate=float(active.weekly_return.gt(0).mean()))
                summaries.append(stats)
                if convention == PRIMARY:
                    daily["portfolio"], weekly["portfolio"] = name, name
                    daily_frames.append(daily)
                    weekly_frames.append(weekly)
                    for year, group in daily.groupby(daily.date.dt.year):
                        annual.append(dict(portfolio=name, fill=fill, year=year, first_date=group.date.min(), last_date=group.date.max(),
                                           return_in_period=float((1+group.daily_return).prod()-1)))
    sensitivity = pd.DataFrame(summaries)
    sensitivity.to_csv(OUT / "marking_sensitivity.csv", index=False)
    summary = sensitivity[sensitivity.marking_convention.eq(PRIMARY)].copy()
    summary.to_csv(OUT / "summary.csv", index=False)
    daily_all, weekly_all = pd.concat(daily_frames), pd.concat(weekly_frames)
    daily_all.to_csv(OUT / "daily_portfolios.csv", index=False)
    weekly_all.to_csv(OUT / "weekly_portfolios.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT / "calendar_returns.csv", index=False)
    for (_, _), group in sensitivity.groupby(["portfolio", "fill"]):
        if group.cagr.max()-group.cagr.min() > 1e-12:
            raise ValueError("daily mark conventions changed terminal return")
    plot_curves(daily_all)
    write_report(summary, sensitivity, pd.DataFrame(annual))
    print(summary[["portfolio", "fill", "cagr", "daily_sharpe", "max_drawdown_daily", "worst_week", "ending_equity"]].to_string(index=False))


def plot_curves(daily):
    for multiple in (1, 2):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for spread, color in zip(("96/93", "97/94", "98/95"), ("#2563eb", "#0f766e", "#dc2626")):
            name = f"{spread} at {multiple:.0%}"
            path = daily[(daily.portfolio == name) & daily.fill.eq("realistic")].sort_values("date")
            nav = path.equity.to_numpy()
            peak = np.maximum.accumulate(np.r_[1_000_000, nav])[1:]
            axes[0].plot(path.date, nav/1_000_000, color=color, label=name, lw=1.6)
            axes[1].plot(path.date, 100*(nav/peak-1), color=color, lw=1.2)
        for axis in axes:
            axis.grid(alpha=.2)
        axes[0].legend(frameon=False)
        axes[0].set_ylabel("Growth of $1")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.suptitle(f"Weekly SPX put spreads at {multiple:.0%} notional: full ten-year archive", fontsize=15)
        axes[0].set_title("Daily marks; weekly resizing; costs included; one corrupt 2020 mark estimated from adjacent puts", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"equity_drawdown_{multiple}x.png", dpi=200)
        plt.close(fig)


def write_report(summary, sensitivity, annual):
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    lines = ["# Full ten-year fixed weekly SPX spread comparison", "",
        f"{manifest['first_entry']}–{manifest['last_expiration']}: 521 scheduled weekly cycles, 520 traded and one common cash week. This is the full available archive, rather than the previous 2024–2026 comparison. Maturities are normally seven calendar days, holiday adjusted. Each entry uses the nearest listed short/long strikes at the stated cash-SPX ratios, and contracts are resized to 100% or 200% of current equity in SPX notional. Positions remain fixed through PM cash expiry.", "",
        "| Spread | Notional | CAGR | Daily Sharpe* | Max drawdown* | Worst week | Ending $1M |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary[summary.fill.eq("realistic")].itertuples(index=False):
        lines.append(f"| {row.spread} | {row.notional_multiple:.0%} | {row.cagr:.2%} | {row.daily_sharpe:.4f} | {row.max_drawdown_daily:.2%} | {row.worst_week:.2%} | ${row.ending_equity:,.0f} |")
    lines += ["", "[200% equity/drawdown chart](equity_drawdown_2x.png) · [100% chart](equity_drawdown_1x.png)", "",
        "Costs: each leg fills 25% of its full bid/ask spread away from midpoint, plus $1.50 per contract per leg. Costs scale with contracts. Cash earns zero; no additional financing or market-impact charge is modeled. Sharpe uses daily marked returns and sqrt(252), while CAGR uses exact elapsed calendar time.", "",
        "*One February 27, 2020 quote for the February 28 3240 put is demonstrably corrupt. For the held 97/94 spread, it creates a negative spread liability even though both puts are deeply in the money. The primary daily series estimates that put from the adjacent same-expiry 3235/3245 put midpoints. This is a disclosed estimate, not a recovered observed quote. The raw marks are preserved, and same-expiry call-put parity with a one-day zero-discount assumption provides a separate sensitivity. Cash expiration P&L, total return and CAGR do not depend on this interim estimate.", "",
        "| Portfolio | Primary daily Sharpe | Raw-quote Sharpe | Call-parity sensitivity Sharpe |",
        "|---|---:|---:|---:|"]
    for name, group in sensitivity[sensitivity.fill.eq("realistic")].groupby("portfolio", sort=False):
        value = group.set_index("marking_convention").daily_sharpe
        lines.append(f"| {name} | {value[PRIMARY]:.4f} | {value['observed_raw']:.4f} | {value['call_parity_estimate']:.4f} |")
    lines += ["", "The December 9–16, 2016 weekly PM expiry is absent from the archive and is held as cash for all strategies; its sessions remain in every metric. All other quoted weeks are traded, including quiet weeks when execution costs exceed premium. The original exploratory candidate file had removed some such weeks because it required positive credit under every fill assumption; the fixed ten-year study restores their exact quotes and charges the resulting losses.", "",
        "## Calendar returns at 200% notional", "",
        "| Year | 96/93 | 97/94 | 98/95 |", "|---|---:|---:|---:|"]
    yearly = annual[annual.fill.eq("realistic") & annual.portfolio.str.endswith("200%")].pivot(index="year", columns="portfolio", values="return_in_period")
    for year, row in yearly.iterrows():
        lines.append(f"| {year} | {row['96/93 at 200%']:.2%} | {row['97/94 at 200%']:.2%} | {row['98/95 at 200%']:.2%} |")
    lines += ["", "2016 begins September 23; 2026 ends September 18. Other rows are complete calendar years.", "",
        "[Summary and full bid/ask execution sensitivity](summary.csv) · [All marking sensitivities](marking_sensitivity.csv) · [Daily NAV](daily_portfolios.csv) · [Weekly trades](weekly_portfolios.csv) · [Calendar returns](calendar_returns.csv) · [Quote audit](data_quality_audit.md)"]
    (OUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inputs", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("inputs", "all"):
        build_inputs()
    if args.stage in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
