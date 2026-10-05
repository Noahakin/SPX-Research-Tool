"""Auditable daily accounting for the organized SPX research chart library.

Quote extraction is independent. This module retains source observations, logs
isolated same-snapshot quote estimates, and produces curves and trade ledgers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from organized_spx_data import DEFAULT_OUTPUT, DATA_ROOT, START, END, load_cash, SPXSurfaceArchive
from organized_spx_parity import fit_parity, parity_estimate

PROJECT = Path(__file__).resolve().parents[1]
INITIAL = 1_000_000.0
FEE_POINTS = 1.50 / 100.0
SPREAD_FRACTION = 0.25
STRIKE_TOLERANCE = 0.005  # Half the one-percentage-point requested strike grid.
TENORS = {3: "3 days", 7: "1 week", 14: "2 weeks", 21: "3 weeks", 28: "4 weeks", 42: "6 weeks", 56: "8 weeks"}
WIDTHS = (1, 2, 3, 5, 10)
QUOTE_KEY = ["snapshot_date", "option_symbol"]


def catalog() -> pd.DataFrame:
    rows = []
    for primary in range(90, 111):
        for width in (0, *WIDTHS):
            secondary = primary - width if width else None
            if secondary is not None and secondary < 90:
                continue
            label = str(primary) if not width else f"{primary}-{secondary}"
            for category, side, underlying in (("Put selling", -1, 0), ("Put buying", 1, 0), ("SPX with overlay", -1, 1), ("SPX with overlay", 1, 1)):
                direction = "long" if side == 1 else "short"
                folder = label if not underlying else f"SPX plus {direction} {label}"
                action = "Buy" if side == 1 else "Sell"
                opposite = "sell" if side == 1 else "buy"
                title = f"{action} {primary}% put"
                if width:
                    title += f" / {opposite} {secondary}% put"
                if underlying:
                    title = "SPX + " + title.lower()
                rows.append(dict(category=category, folder=folder, structure="Single put" if not width else "Put spread",
                                 primary_pct=primary, secondary_pct=secondary, width_pct=width,
                                 side=side, underlying=underlying, title=title,
                                 structure_id=f"{'spx_' if underlying else ''}{direction}_{label}"))
    result = pd.DataFrame(rows)
    if len(result) != 420 or result.structure_id.duplicated().any():
        raise AssertionError("Strategy catalog does not match the stated grid")
    return result


def suspect_quotes(frame: pd.DataFrame) -> pd.Series:
    finite = np.isfinite(frame[["strike", "bid", "ask"]]).all(axis=1)
    invalid = ~finite | frame.strike.le(0) | frame.bid.lt(0) | frame.ask.le(0) | frame.ask.lt(frame.bid)
    # This catches source defects such as a 0.05 bid against a 288.40 ask,
    # without labelling ordinary penny options as corrupt.
    implausible_bid = frame.ask.ge(25) & (frame.bid.le(0.5) | frame.bid.lt(frame.ask * .02))
    return invalid | implausible_bid


def very_wide_quotes(frame: pd.DataFrame) -> pd.Series:
    mid = (frame.bid + frame.ask) / 2
    return frame.ask.ge(25) & ((frame.bid.lt(.5 * frame.ask))
                              | ((frame.ask - frame.bid) > np.maximum(25, .30 * mid)))


def adjacent_estimate(chain: pd.DataFrame, strike: float) -> dict | None:
    """Bracket with valid nearby puts, never extrapolate from other dates.

    The estimate is explicitly not an observed executable quote. Require
    locally sensible monotonicity, width bounds, and nearby strikes.
    """
    broad_donor_spread = very_wide_quotes(chain)
    good = chain.loc[~suspect_quotes(chain) & ~broad_donor_spread].sort_values("strike").drop_duplicates("strike")
    below, above = good[good.strike < strike], good[good.strike > strike]
    if below.empty or above.empty:
        return None
    low, high = below.iloc[-1], above.iloc[0]
    max_gap = max(25.0, strike * .005)
    # Far downside listings often use 50-point increments. For penny puts,
    # allow a wider observed bracket; never use it for material-price options.
    coarse_penny_bracket = max(low.ask, high.ask) <= .25
    if coarse_penny_bracket:
        max_gap = max(max_gap, 50.0)
    if strike - low.strike > max_gap or high.strike - strike > max_gap:
        return None
    low_mid, high_mid = (low.bid + low.ask) / 2, (high.bid + high.ask) / 2
    width = high.strike - low.strike
    if high_mid < low_mid - .10 or high_mid - low_mid > width + .50:
        return None
    fraction = (strike - low.strike) / width
    bid = float(low.bid + fraction * (high.bid - low.bid))
    ask = float(low.ask + fraction * (high.ask - low.ask))
    if bid < 0 or ask < bid:
        return None
    return dict(used_bid=bid, used_ask=ask, used_mid=(bid + ask) / 2,
                lower_symbol=low.option_symbol, upper_symbol=high.option_symbol,
                lower_strike=float(low.strike), upper_strike=float(high.strike),
                lower_bid=float(low.bid), lower_ask=float(low.ask),
                upper_bid=float(high.bid), upper_ask=float(high.ask),
                method="Same-day same-expiry adjacent-put linear estimate" +
                       ("; penny quotes, up to 50-point donor gap" if coarse_penny_bracket else ""))


def audit_quotes(output: Path) -> pd.DataFrame:
    entries = pd.read_parquet(output / "entries.parquet")
    quotes = pd.read_parquet(output / "daily_quotes.parquet")
    if quotes.duplicated(QUOTE_KEY).any():
        raise ValueError("Duplicate source quote keys must be resolved before simulation")
    quotes["used_bid"], quotes["used_ask"] = quotes.bid, quotes.ask
    quotes["used_mid"] = (quotes.bid + quotes.ask) / 2
    quotes["estimated"] = False
    quotes["unresolved"] = False
    flagged = suspect_quotes(quotes)
    # Audit large violations of same-expiry vertical payoff bounds. Small
    # bid/ask midpoint noise is retained and reported, not clipped away.
    bound_flags = []
    for (day, expiry), group in quotes.groupby(["snapshot_date", "expiration_date"], sort=False):
        group = group.sort_values("strike").drop_duplicates("strike")
        mids = group.used_mid.to_numpy()
        ks = group.strike.to_numpy()
        changes, widths = np.diff(mids), np.diff(ks)
        bad = np.flatnonzero((changes < -1.0) | (changes > widths + 1.0))
        for pos in bad:
            low, high = group.iloc[pos], group.iloc[pos + 1]
            bound_flags.append(dict(date=day, expiration_date=expiry, lower_symbol=low.option_symbol,
                                    upper_symbol=high.option_symbol, width=widths[pos], midpoint_difference=changes[pos]))
            flagged.loc[[low.name, high.name]] = True
    evidence = []
    if flagged.any():
        archive = SPXSurfaceArchive(DATA_ROOT, cache_size=1)
        for day, group in quotes.loc[flagged].groupby("snapshot_date", sort=True):
            source = pq.ParquetFile(archive.path_for(day)).read(columns=["option_symbol", "option_type", "expiration_date", "strike", "bid", "ask"]).to_pandas()
            source.option_symbol = source.option_symbol.astype(str).str.strip()
            source.expiration_date = pd.to_datetime(source.expiration_date).dt.normalize()
            source = source[source.option_symbol.str.startswith("SPXW")]
            parity_models = {}
            for idx, row in group.iterrows():
                chain = source[source.expiration_date.eq(row.expiration_date)]
                estimate = adjacent_estimate(chain[chain.option_type.eq("put")], float(row.strike))
                raw_bad = bool(suspect_quotes(pd.DataFrame([row])).iloc[0])
                # Bound-pair flags identify both neighbors. Replace only an
                # isolated point that materially disagrees with its own bracket.
                relative_threshold = .05 if bool(very_wide_quotes(pd.DataFrame([row])).iloc[0]) else .20
                mismatch = estimate is not None and abs(row.used_mid - estimate["used_mid"]) > max(5.0, relative_threshold * max(estimate["used_mid"], 1.0))
                if raw_bad or mismatch:
                    if estimate is None:
                        if row.expiration_date not in parity_models:
                            parity_models[row.expiration_date] = fit_parity(chain)
                        fitted = parity_models[row.expiration_date]
                        estimate = parity_estimate(fitted, float(row.strike)) if fitted is not None else None
                    record = dict(snapshot_date=day, expiration_date=row.expiration_date, option_symbol=row.option_symbol,
                                  strike=row.strike, raw_bid=row.bid, raw_ask=row.ask, raw_mid=row.used_mid,
                                  source_file=str(archive.path_for(day)), entry_quote=bool(((entries.entry_date == day) & (entries.option_symbol == row.option_symbol)).any()))
                    if estimate is not None:
                        for key in ("used_bid", "used_ask", "used_mid"):
                            quotes.loc[idx, key] = estimate[key]
                        quotes.loc[idx, "estimated"] = True
                        evidence.append(record | estimate | dict(status="estimated"))
                    else:
                        quotes.loc[idx, ["used_bid", "used_ask", "used_mid"]] = np.nan
                        quotes.loc[idx, "unresolved"] = True
                        evidence.append(record | dict(status="unresolved", method="No defensible nearby bracket"))
    pd.DataFrame(bound_flags, columns=["date", "expiration_date", "lower_symbol", "upper_symbol", "width", "midpoint_difference"]).to_csv(output / "quote_bound_flags.csv", index=False)
    evidence_columns = ["snapshot_date", "expiration_date", "option_symbol", "strike", "raw_bid", "raw_ask", "raw_mid", "used_bid", "used_ask", "used_mid", "lower_symbol", "upper_symbol", "lower_strike", "upper_strike", "lower_bid", "lower_ask", "upper_bid", "upper_ask", "method", "status", "entry_quote", "source_file"]
    evidence_frame = pd.DataFrame(evidence)
    extra_columns = [key for key in evidence_frame.columns if key not in evidence_columns]
    evidence_frame.reindex(columns=evidence_columns + extra_columns).to_csv(output / "quote_repair_evidence.csv", index=False)
    quotes.to_parquet(output / "audited_quotes.parquet", index=False)
    print(f"Quote audit: {len(quotes):,} observations, {int(quotes.estimated.sum())} estimates, {int(quotes.unresolved.sum())} unresolved, {len(bound_flags)} bound flags", flush=True)
    return quotes


def leg_returns(marks: np.ndarray, bid: np.ndarray, ask: np.ndarray, spot: float) -> tuple[np.ndarray, np.ndarray]:
    mid = (bid + ask) / 2
    friction = SPREAD_FRACTION * (ask - bid) + FEE_POINTS
    long = (marks - mid - friction) / spot
    short = (mid - marks - friction) / spot
    return long, short


def curve_statistics(curve: np.ndarray, dates: pd.DatetimeIndex, initial: float = INITIAL) -> dict:
    if np.any(curve[np.isfinite(curve)] <= 0):
        raise ValueError("Nonpositive portfolio equity requires an explicit insolvency protocol")
    elapsed = (dates[-1] - dates[0]).days / 365.2425
    returns = curve / np.r_[initial, curve[:-1]] - 1
    complete = np.isfinite(returns).all()
    sd = float(np.std(returns, ddof=1)) if complete else np.nan
    mean = float(np.mean(returns)) if complete else np.nan
    cagr = float((curve[-1] / initial) ** (1 / elapsed) - 1) if np.isfinite(curve[-1]) else np.nan
    peaks = np.maximum.accumulate(np.r_[initial, curve])[1:] if complete else np.full(len(curve), np.nan)
    return dict(cagr=cagr, annualized_volatility=sd * np.sqrt(252),
                daily_sharpe=mean / sd * np.sqrt(252) if sd > 0 else np.nan,
                annualized_arithmetic_return=mean * 252, ending_equity=float(curve[-1]),
                total_return=float(curve[-1] / initial - 1),
                max_drawdown=float(np.min(curve / peaks - 1)) if complete else np.nan,
                missing_daily_marks=int((~np.isfinite(curve)).sum()), daily_observations=len(curve))


def simulate(output: Path, quotes: pd.DataFrame | None = None) -> None:
    entries = pd.read_parquet(output / "entries.parquet")
    schedule = pd.read_parquet(output / "schedule.parquet")
    quotes = pd.read_parquet(output / "audited_quotes.parquet") if quotes is None else quotes
    quote_lookup = quotes.set_index(QUOTE_KEY)
    if quote_lookup.index.duplicated().any():
        raise ValueError("Duplicate audited quotes")
    cash = load_cash().loc[START:END]
    dates = cash.index
    specs = catalog()
    primary_idx = specs.primary_pct.to_numpy(int) - 90
    secondary_idx = specs.secondary_pct.fillna(90).to_numpy(int) - 90
    spread = specs.width_pct.to_numpy() > 0
    side = specs.side.to_numpy()
    underlying = specs.underlying.to_numpy()
    curve_list, metrics, ledger, bounds = [], [], [], []
    entry_groups = {key: group.sort_values("ratio") for key, group in entries.groupby("roll_id", sort=False)}
    for tenor, cycles in schedule.groupby("target_dte", sort=True):
        cycles = cycles.sort_values("entry_date").reset_index(drop=True)
        if not np.array_equal(cycles.expiration_date.iloc[:-1].to_numpy(), cycles.entry_date.iloc[1:].to_numpy()):
            raise ValueError(f"Discontinuous {tenor}D schedule")
        if cycles.entry_date.iloc[0] != dates[0] or cycles.valuation_end.iloc[-1] != dates[-1]:
            raise ValueError("Schedules must cover the entire common period")
        nav = np.full((len(dates), len(specs)), np.nan)
        equity = np.full(len(specs), INITIAL)
        counts = np.zeros(len(specs), dtype=int)
        cash_counts = np.zeros(len(specs), dtype=int)
        estimates = np.zeros(len(specs), dtype=int)
        actual_min = np.full(len(specs), np.inf)
        actual_max = np.full(len(specs), -np.inf)
        max_strike_error = np.zeros(len(specs))
        for cycle in cycles.itertuples(index=False):
            positions = np.flatnonzero((dates >= cycle.entry_date) & (dates <= cycle.valuation_end))
            days = dates[positions]
            spot = float(cash.loc[cycle.entry_date])
            option = np.zeros((len(days), len(specs)))
            available = np.zeros(len(specs), dtype=bool)
            reason = np.full(len(specs), "Required expiration unavailable", dtype=object)
            est_count = np.zeros(len(specs), dtype=int)
            k = np.full(21, np.nan)
            entry_bid, entry_ask = np.full(21, np.nan), np.full(21, np.nan)
            strike_error = np.full(21, np.nan)
            if cycle.has_expiry:
                selected = entry_groups[cycle.roll_id]
                if len(selected) != 21 or not np.array_equal(np.rint(selected.ratio.to_numpy() * 100), np.arange(90, 111)):
                    raise ValueError("Every quoted cycle needs the same complete requested leg grid")
                symbols, k = selected.option_symbol.to_numpy(), selected.strike.to_numpy(float)
                strike_error = np.abs(k / spot - selected.ratio.to_numpy())
                keys = pd.MultiIndex.from_product([days, symbols], names=QUOTE_KEY)
                used = quote_lookup.reindex(keys)
                marks = used.used_mid.to_numpy(copy=True).reshape(len(days), 21)
                entry_bid = used.used_bid.to_numpy().reshape(len(days), 21)[0]
                entry_ask = used.used_ask.to_numpy().reshape(len(days), 21)[0]
                is_estimated = used.estimated.fillna(False).to_numpy(dtype=bool, copy=True).reshape(len(days), 21)
                if days[-1] == cycle.expiration_date:
                    marks[-1] = np.maximum(k - cash.loc[cycle.expiration_date], 0)
                    is_estimated[-1] = False
                long, short = leg_returns(marks, entry_bid, entry_ask, spot)
                legs_valid = ((strike_error <= STRIKE_TOLERANCE + 1e-12) & np.isfinite(entry_bid)
                              & np.isfinite(entry_ask) & ~is_estimated[0])
                available = legs_valid[primary_idx] & (~spread | legs_valid[secondary_idx])
                collapsed = spread & (k[primary_idx] <= k[secondary_idx])
                available &= ~collapsed
                reason[:] = "Traded"
                reason[~available] = "Entry strike or quote unavailable"
                reason[collapsed] = "Requested spread collapsed to one listed strike"
                a = np.where(side[None, :] == 1, long[:, primary_idx], short[:, primary_idx])
                b = np.where(side[None, :] == 1, short[:, secondary_idx], long[:, secondary_idx])
                option = a + np.where(spread[None, :], b, 0)
                option[:, ~available] = 0
                leg_estimates = is_estimated.sum(axis=0)
                est_count = (leg_estimates[primary_idx] + np.where(spread, leg_estimates[secondary_idx], 0)) * available
                err = np.maximum(strike_error[primary_idx], np.where(spread, strike_error[secondary_idx], 0))
                max_strike_error = np.maximum(max_strike_error, np.where(available, err, 0))
                # Retain minor quote noise (including negative .025-point
                # vertical midpoints). Record meaningful bounds violations.
                vertical = marks[:, primary_idx] - marks[:, secondary_idx]
                width_points = k[primary_idx] - k[secondary_idx]
                bad = spread[None, :] & available[None, :] & ((vertical < -1) | (vertical > width_points[None, :] + 1))
                # One record per structure/day, rather than per direction.
                for day_i, spec_i in zip(*np.where(bad & (side[None, :] == -1) & (underlying[None, :] == 0))):
                    bounds.append(dict(roll_id=cycle.roll_id, date=days[day_i], primary_pct=int(specs.iloc[spec_i].primary_pct),
                                       secondary_pct=int(specs.iloc[spec_i].secondary_pct), spread_mark=float(vertical[day_i, spec_i]),
                                       width_points=float(width_points[spec_i])))
            cum_return = option + (cash.loc[days].to_numpy() / spot - 1)[:, None] * underlying[None, :]
            before = equity.copy()
            nav[positions] = before[None, :] * (1 + cum_return)
            equity = nav[positions[-1]].copy()
            if not np.isfinite(equity).all():
                raise ValueError(f"Unresolved terminal quote for {cycle.roll_id}; cannot compound a fabricated value")
            if np.any(equity <= 0):
                raise ValueError("Portfolio insolvency requires explicit handling")
            counts += available
            cash_counts += ~available
            estimates += est_count
            actual_min = np.minimum(actual_min, np.where(available, cycle.actual_dte, np.inf))
            actual_max = np.maximum(actual_max, np.where(available, cycle.actual_dte, -np.inf))
            for i, spec in enumerate(specs.itertuples(index=False)):
                ledger.append(dict(strategy_id=f"{spec.structure_id}_d{tenor:02d}", roll_id=cycle.roll_id,
                                   entry_date=cycle.entry_date, expiration_date=cycle.expiration_date, valuation_end=cycle.valuation_end,
                                   actual_dte=cycle.actual_dte, traded=bool(available[i]), reason=reason[i],
                                   primary_strike=k[primary_idx[i]], secondary_strike=k[secondary_idx[i]] if spread[i] else np.nan,
                                   primary_bid=entry_bid[primary_idx[i]], primary_ask=entry_ask[primary_idx[i]],
                                   secondary_bid=entry_bid[secondary_idx[i]] if spread[i] else np.nan,
                                   secondary_ask=entry_ask[secondary_idx[i]] if spread[i] else np.nan,
                                   spot_entry=spot, entry_equity=before[i], ending_equity=equity[i],
                                   terminal_return=cum_return[-1, i], estimated_quote_count=int(est_count[i]),
                                   final_position_open=bool(cycle.expiration_date > END)))
        for i, spec in enumerate(specs.to_dict("records")):
            stats = curve_statistics(nav[:, i], dates)
            strategy_id = f"{spec['structure_id']}_d{tenor:02d}"
            coverage = f"{int(cash_counts[i])}/{len(cycles)} option cycles in cash" if cash_counts[i] else "complete"
            status = "complete" if not stats["missing_daily_marks"] else "incomplete daily quotes"
            metrics.append(spec | dict(strategy_id=strategy_id, tenor=TENORS[int(tenor)], tenor_days=int(tenor),
                                       notional_multiple=1.0, cycles=len(cycles), traded_cycles=int(counts[i]), cash_cycles=int(cash_counts[i]),
                                       trade_fraction=float(counts[i] / len(cycles)), coverage_status=coverage, status=status,
                                       actual_dte_min=float(actual_min[i]) if counts[i] else np.nan,
                                       actual_dte_max=float(actual_max[i]) if counts[i] else np.nan,
                                       max_strike_error_pct_points=float(max_strike_error[i] * 100),
                                       estimated_quote_count=int(estimates[i])) | stats)
            curve_list.append(nav[:, i])
        print(f"Simulated {TENORS[int(tenor)]}: {len(cycles)} cycles × {len(specs)} portfolios", flush=True)
    frame = pd.DataFrame(metrics)
    curves = np.column_stack(curve_list)
    frame.to_csv(output / "strategy_metrics.csv", index=False)
    np.savez_compressed(output / "curves.npz", dates=dates.to_numpy(dtype="datetime64[ns]"),
                        strategy_ids=frame.strategy_id.to_numpy(dtype=str), equity=curves)
    pd.DataFrame(ledger).to_parquet(output / "trade_ledger.parquet", index=False)
    pd.DataFrame(bounds, columns=["roll_id", "date", "primary_pct", "secondary_pct", "spread_mark", "width_points"]).to_csv(output / "remaining_bound_flags.csv", index=False)
    write_manifest(output, frame)
    print(f"Saved {curves.shape[1]:,} curves × {curves.shape[0]:,} daily observations; {len(ledger):,} ledger rows; {len(bounds)} remaining substantial vertical-bound flags", flush=True)


def write_manifest(output: Path, metrics: pd.DataFrame) -> None:
    config = dict(start_date=str(START.date()), end_date=str(END.date()), initial_equity=INITIAL,
                  notional_label="100% current-equity SPX notional at each roll; cash earns zero",
                  cost_label="25% of each leg's full bid/ask spread from mid + $1.50/contract/leg at entry",
                  equity_label="Daily marked portfolio equity", sharpe_label="Daily Sharpe, 252 sessions, 0% cash rate",
                  categories={"Put selling": "Short single puts and credit put spreads", "Put buying": "Long single puts and debit put spreads",
                              "SPX with overlay": "100% SPX price exposure plus each long or short put strategy"},
                  strike_grid_pct=list(range(90, 111)), spread_widths_pct_points=list(WIDTHS),
                  strike_tolerance_pct_points=STRIKE_TOLERANCE * 100, tenor_days=list(TENORS),
                  strategies=len(metrics), daily_sessions=int(metrics.daily_observations.iloc[0]),
                  expiration_rules={"3": "Nearest listed PM expiry in 1–5 calendar days, ties longer, renew at expiry",
                                    "7/14/21/28/42": "Fixed calendar-Friday anchors, holidays previous cash session; exact listed PM expiry, otherwise cash",
                                    "56": "Nearest listed Friday or holiday-adjusted-Friday PM expiry, ties longer, renew at expiry"},
                  daily_marking="Observed same-contract midpoint; PM expiry cash intrinsic; isolated quote estimates documented; no forward fill",
                  spx_overlay_assumption="SPX price returns, no dividends, financing or interest; 100% SPX and 100% option notional, fixed quantities between rolls",
                  entry_rules="Closest listed strike before screening, ties lower. No trade if either leg differs from target by more than 0.5 percentage point, entry quote unresolved/estimated, or strikes collapse. No premium-sign gate.",
                  final_positions="Unexpired positions marked at end-date observed midpoint, without hypothetical closing costs",
                  execution_timing="EOD quote execution and simultaneous cash-close sizing assumed; archive 16:00 label is generated, not exchange-verified",
                  elapsed_years=(END-START).days/365.2425,
                  source_fingerprint=json.loads((output / "manifest.json").read_text())["input_sha256"])
    config["input_artifact_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("entries.parquet", "daily_quotes.parquet", "audited_quotes.parquet", "schedule.parquet")}
    (output / "run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reuse-audit", action="store_true")
    args = parser.parse_args()
    quotes = None if args.reuse_audit else audit_quotes(args.output)
    if not args.audit_only:
        simulate(args.output, quotes)


if __name__ == "__main__":
    main()
