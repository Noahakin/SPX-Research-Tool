"""Select historical best premium-funded hedges, separately by short leg and tenor.

Source observations and the original standalone portfolios remain untouched.
The saved winners are retrospective full-sample selections, not live signals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from organized_spx_data import START, END, load_cash
from organized_spx_engine import (INITIAL, FEE_POINTS, SPREAD_FRACTION,
    STRIKE_TOLERANCE, TENORS, WIDTHS, QUOTE_KEY, catalog, leg_returns)
from organized_spx_combinations import OUTPUT as SOURCE, BASE_OUTPUT, premium_quantity

OUTPUT = SOURCE.parent / "organized_spx_best_hedges"
FRACTIONS = (.05, .10, .15, .20)
MIN_RELATIVE_COVERAGE = .90
CANDIDATE_COLUMNS = ["strategy_id", "short_variation", "primary_pct", "secondary_pct",
    "width_pct", "hedge_type", "hedge_primary_pct", "hedge_secondary_pct", "hedge_width_pct",
    "premium_fraction", "tenor_days", "cagr", "annualized_volatility", "daily_sharpe",
    "max_drawdown", "relative_base_coverage", "traded_cycles", "base_eligible_cycles",
    "cycles", "status", "selection_eligible", "selected"]


def candidate_catalog(base: dict) -> pd.DataFrame:
    high = int(base["primary_pct"])
    width = int(base["width_pct"])
    anchor = high - width
    name = f"{high}-{anchor}" if width else str(high)
    rows = []
    for fraction in FRACTIONS:
        pct = round(fraction * 100)
        for hedge_high in range(80, anchor + 1):
            for hedge_width in (0, *WIDTHS):
                hedge_low = hedge_high - hedge_width
                if hedge_low < 80:
                    continue
                kind = "Long put" if not hedge_width else "Downside buffer"
                hedge_name = str(hedge_high) if not hedge_width else f"{hedge_high}/{hedge_low}"
                leaf = "Best long put" if not hedge_width else "Best put buffer"
                rows.append(dict(category="Both", primary_pct=high,
                    secondary_pct=float(anchor) if width else np.nan, width_pct=width,
                    short_variation=name, hedge_type=kind, hedge_primary_pct=hedge_high,
                    hedge_secondary_pct=hedge_low if hedge_width else np.nan,
                    hedge_width_pct=hedge_width, premium_fraction=fraction,
                    notional_multiple=1., side=-1, underlying=0,
                    structure=("Short put" if not width else "Short spread") + " + " + kind.lower(),
                    folder=f"{pct:02d} percent premium/{name}/{leaf}",
                    title=f"Sell {name.replace('-', '/')} + buy {hedge_name} {'put' if not hedge_width else 'buffer'} (up to {pct}% premium)",
                    comparison_title=f"Sell {name.replace('-', '/')} + {leaf.lower()} (up to {pct}% premium)",
                    structure_id=f"best_{name}_h{hedge_name.replace('/', '-')}_p{pct:02d}"))
    return pd.DataFrame(rows)


def prepare_cycles(source=SOURCE, cash=None, quotes=None):
    cash = load_cash().loc[START:END] if cash is None else cash
    dates = cash.index
    schedule = pd.read_parquet(source / "schedule.parquet")
    entries = pd.read_parquet(source / "entries.parquet")
    quotes = pd.read_parquet(source / "audited_quotes.parquet") if quotes is None else quotes
    lookup = quotes.set_index(QUOTE_KEY)
    if lookup.index.duplicated().any():
        raise ValueError("Duplicate quote keys")
    entry_groups = {key: frame.sort_values("ratio") for key, frame in entries.groupby("roll_id", sort=False)}
    by_tenor = {}
    for tenor, group in schedule.groupby("target_dte", sort=True):
        group = group.sort_values("entry_date")
        if (group.entry_date.iloc[0] != dates[0] or group.valuation_end.iloc[-1] != dates[-1]
            or not np.array_equal(group.expiration_date.iloc[:-1], group.entry_date.iloc[1:])):
            raise ValueError("Incomplete or discontinuous roll schedule")
        cycles = []
        for row in group.to_dict("records"):
            ix = np.flatnonzero((dates >= row["entry_date"]) & (dates <= row["valuation_end"]))
            days = dates[ix]
            spot = float(cash.loc[row["entry_date"]])
            k, bid, ask = (np.full(31, np.nan) for _ in range(3))
            estimated = np.zeros((len(days), 31), bool)
            error = np.full(31, np.inf)
            valid = np.zeros(31, bool)
            long, short = np.zeros_like(estimated, float), np.zeros_like(estimated, float)
            symbols = np.full(31, "", object)
            if row["has_expiry"]:
                selected = entry_groups[row["roll_id"]]
                np.testing.assert_array_equal(np.rint(selected.ratio.to_numpy()*100), np.arange(80, 111))
                k = selected.strike.to_numpy(float)
                symbols = selected.option_symbol.to_numpy()
                keys = pd.MultiIndex.from_product([days, symbols], names=QUOTE_KEY)
                observed = lookup.reindex(keys)
                marks = observed.used_mid.to_numpy(copy=True).reshape(len(days), 31)
                bid = observed.used_bid.to_numpy().reshape(len(days), 31)[0]
                ask = observed.used_ask.to_numpy().reshape(len(days), 31)[0]
                estimated = observed.estimated.fillna(False).to_numpy(bool, copy=True).reshape(len(days), 31)
                if days[-1] == row["expiration_date"]:
                    marks[-1] = np.maximum(k-cash.loc[days[-1]], 0)
                    estimated[-1] = False
                error = np.abs(k/spot - np.arange(80, 111)/100)
                valid = (np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask > 0)
                    & (ask >= bid) & ~estimated[0] & (error <= STRIKE_TOLERANCE + 1e-12))
                long, short = leg_returns(marks, bid, ask, spot)
            cycles.append(row | dict(ix=ix, spot=spot, k=k, bid=bid, ask=ask,
                mid=(bid+ask)/2, friction=SPREAD_FRACTION*(ask-bid)+FEE_POINTS,
                valid=valid, error=error, long=long, short=short,
                estimated=estimated.sum(axis=0), symbols=symbols))
        by_tenor[int(tenor)] = cycles
    return dates, by_tenor


def simulate_candidates(specs, cycles, dates, tenor, ledger_writer=None):
    """Vectorized portfolios, with settlement before costs on a shared roll date."""
    n = len(specs)
    hi = specs.primary_pct.to_numpy(int)-80
    lo = specs.secondary_pct.fillna(80).to_numpy(int)-80
    hh = specs.hedge_primary_pct.to_numpy(int)-80
    hl = specs.hedge_secondary_pct.fillna(80).to_numpy(int)-80
    spread = specs.width_pct.to_numpy() > 0
    buffer = specs.hedge_width_pct.to_numpy() > 0
    fractions = specs.premium_fraction.to_numpy()
    nav = np.full((len(dates), n), np.nan)
    equity = np.full(n, INITIAL)
    count, base_count, estimates, capped = (np.zeros(n, int) for _ in range(4))
    quantity_sum, quantity_max, spend_sum, credit_sum, debit_sum, error_max = (np.zeros(n) for _ in range(6))
    dte_min, dte_max = np.full(n, np.inf), np.full(n, -np.inf)
    ids = np.char.add(specs.structure_id.to_numpy(str), f"_d{tenor:02d}")
    for cycle in cycles:
        k, mid, cost, valid = (cycle[key] for key in ("k", "mid", "friction", "valid"))
        credit = mid[hi]-cost[hi]-np.where(spread, mid[lo]+cost[lo], 0)
        debit = mid[hh]+cost[hh]-np.where(buffer, mid[hl]-cost[hl], 0)
        base_valid = (valid[hi] & (~spread | valid[lo]) & (credit > 0)
            & (~spread | ((k[hi] > k[lo]) & (credit <= k[hi]-k[lo]+1e-12))))
        hedge_mid = mid[hh]-np.where(buffer, mid[hl], 0)
        eligible = (base_valid & valid[hh] & (~buffer | valid[hl]) & (debit > 0)
            & (hedge_mid >= 0) & (~buffer | ((k[hh] > k[hl]) & (hedge_mid <= k[hh]-k[hl]+1e-12))))
        q, _ = premium_quantity(credit, debit, fractions)
        q[~eligible] = 0
        base_return = cycle["short"][:, hi] + np.where(spread, cycle["long"][:, lo], 0)
        hedge_return = cycle["long"][:, hh] + np.where(buffer, cycle["short"][:, hl], 0)
        cumulative = base_return + q * hedge_return
        cumulative[:, ~eligible] = 0
        before = equity.copy()
        nav[cycle["ix"]] = before * (1+cumulative)
        equity = nav[cycle["ix"][-1]].copy()
        # Count distinct held contract/date estimates, including shared strikes once.
        used = cycle["estimated"][hi].copy()
        used += np.where(spread & (lo != hi), cycle["estimated"][lo], 0)
        used += np.where((hh != hi) & (~spread | (hh != lo)), cycle["estimated"][hh], 0)
        used += np.where(buffer & (hl != hi) & (~spread | (hl != lo)) & (hl != hh), cycle["estimated"][hl], 0)
        count += eligible
        base_count += base_valid
        estimates += used * eligible
        capped += eligible & np.isclose(q, 1, rtol=0, atol=1e-12)
        quantity_sum += q
        quantity_max = np.maximum(quantity_max, q)
        paid_fraction = np.zeros(n)
        np.divide(q*debit, credit, out=paid_fraction, where=eligible)
        spend_sum += paid_fraction
        credit_sum += np.where(eligible, credit/cycle["spot"], 0)
        debit_sum += np.where(eligible, q*debit/cycle["spot"], 0)
        error = np.maximum(cycle["error"][hi], cycle["error"][hh])
        error = np.maximum(error, np.where(spread, cycle["error"][lo], 0))
        error = np.maximum(error, np.where(buffer, cycle["error"][hl], 0))
        error_max = np.maximum(error_max, np.where(eligible, error, 0))
        dte_min = np.minimum(dte_min, np.where(eligible, cycle["actual_dte"], np.inf))
        dte_max = np.maximum(dte_max, np.where(eligible, cycle["actual_dte"], -np.inf))
        if ledger_writer is not None:
            ledger = pd.DataFrame(dict(strategy_id=ids, roll_id=cycle["roll_id"],
                target_dte=tenor, entry_date=cycle["entry_date"], expiration_date=cycle["expiration_date"],
                valuation_end=cycle["valuation_end"], traded=eligible, base_eligible=base_valid,
                net_short_credit_points=credit, hedge_debit_points=debit,
                premium_fraction=fractions, hedge_quantity_ratio=q,
                net_credit_retained_points=np.where(eligible, credit-q*debit, 0),
                spot_entry=cycle["spot"], entry_equity=before, ending_equity=equity,
                terminal_return=cumulative[-1]))
            for name, indices, active in (("short", hi, np.ones(n, bool)), ("base_long", lo, spread),
                                         ("hedge_long", hh, np.ones(n, bool)), ("hedge_short", hl, buffer)):
                ledger[name+"_symbol"] = np.where(active, cycle["symbols"][indices], "")
                ledger[name+"_strike"] = np.where(active, k[indices], np.nan)
            ledger_writer(ledger)
    finite = np.isfinite(nav).all(axis=0) & (nav > 0).all(axis=0)
    daily = nav / np.vstack([np.full((1, n), INITIAL), nav[:-1]])-1
    sd, mean = daily.std(axis=0, ddof=1), daily.mean(axis=0)
    peaks = np.maximum.accumulate(np.vstack([np.full((1, n), INITIAL), nav]), axis=0)[1:]
    elapsed = (dates[-1]-dates[0]).days/365.2425
    result = specs.copy()
    result["strategy_id"] = ids
    result["tenor"], result["tenor_days"] = TENORS[tenor], tenor
    fields = dict(cagr=(nav[-1]/INITIAL)**(1/elapsed)-1,
        annualized_volatility=sd*np.sqrt(252),
        daily_sharpe=np.divide(mean*np.sqrt(252), sd, out=np.full(n, np.nan), where=sd>0),
        annualized_arithmetic_return=mean*252, ending_equity=nav[-1], total_return=nav[-1]/INITIAL-1,
        max_drawdown=(nav/peaks-1).min(axis=0), missing_daily_marks=(~np.isfinite(nav)).sum(axis=0),
        daily_observations=len(dates), cycles=len(cycles), traded_cycles=count,
        base_eligible_cycles=base_count, cash_cycles=len(cycles)-count, trade_fraction=count/len(cycles),
        relative_base_coverage=np.divide(count, base_count, out=np.zeros(n), where=base_count>0),
        status=np.where(finite, "complete", "invalid"), estimated_quote_count=estimates,
        actual_dte_min=np.where(count>0, dte_min, np.nan), actual_dte_max=np.where(count>0, dte_max, np.nan),
        max_strike_error_pct_points=error_max*100, hedge_notional_cap=1., hedge_cap_binding_cycles=capped,
        mean_hedge_notional_multiple=quantity_sum/np.maximum(count, 1), max_hedge_notional_multiple=quantity_max,
        mean_premium_fraction_spent=spend_sum/np.maximum(count, 1),
        mean_short_credit_pct_notional=credit_sum/np.maximum(count, 1),
        mean_hedge_spend_pct_notional=debit_sum/np.maximum(count, 1))
    for key, value in fields.items():
        result[key] = value
    result["selection_eligible"] = finite & (count > 0) & (result.relative_base_coverage >= MIN_RELATIVE_COVERAGE) & np.isfinite(result.daily_sharpe)
    return result, nav


def choose_winners(metrics):
    keys = ["premium_fraction", "short_variation", "tenor_days", "hedge_type"]
    eligible = metrics[metrics.selection_eligible]
    # Stable explicit tie-breaks: Sharpe, then CAGR, then smaller drawdown, then ID.
    winners = eligible.sort_values(["daily_sharpe", "cagr", "max_drawdown", "strategy_id"],
        ascending=[False, False, False, True]).groupby(keys, sort=False).head(1).copy()
    if len(winners) != metrics.groupby(keys).ngroups:
        raise ValueError("At least one budget/short/expiry/hedge group has no eligible candidate")
    return winners


def rebuild_selected_library(output=OUTPUT):
    """Reconstruct audited daily curves in memory without a duplicate disk cache."""
    config = json.loads((output/"run.json").read_text(encoding="utf-8"))
    if config.get("dynamic_buffer_selection"):
        from rebuild_premium_buffers import reconstruct
        return reconstruct(output)
    source = Path(config["source_quote_dir"])
    for name, expected in config["source_artifact_sha256"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source artifact changed: {name}")
    metrics = pd.read_csv(output/"strategy_metrics.csv", dtype={"folder": str})
    dates, by_tenor = prepare_cycles(source)
    equity = np.empty((len(dates), len(metrics)))
    base = metrics[~metrics.category.eq("Both")]
    with np.load(BASE_OUTPUT/"curves.npz") as archive:
        np.testing.assert_array_equal(dates.to_numpy(), archive["dates"])
        np.testing.assert_array_equal(base.strategy_id, archive["strategy_ids"])
        equity[:, base.index] = archive["equity"]
    for tenor, cycles in by_tenor.items():
        subset = metrics[metrics.category.eq("Both") & metrics.tenor_days.eq(tenor)]
        verified, nav = simulate_candidates(subset.reset_index(drop=True), cycles, dates, tenor)
        for key in ("cagr", "daily_sharpe", "annualized_volatility", "max_drawdown", "ending_equity"):
            np.testing.assert_allclose(verified[key], subset[key], rtol=1e-12, atol=1e-9)
        equity[:, subset.index] = nav
    return dates.to_numpy(dtype="datetime64[ns]"), metrics.strategy_id.to_numpy(str), equity


def run(output=OUTPUT, source=SOURCE, candidate_only=False):
    if (output/"run.json").exists() and json.loads((output/"run.json").read_text(encoding="utf-8")).get("dynamic_buffer_selection"):
        raise RuntimeError("Sharpe buffer selection has been superseded. Run rebuild_premium_buffers.py --build instead.")
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    dates, by_tenor = prepare_cycles(source)
    bases = catalog().query("category == 'Put selling'").to_dict("records")
    selected, selected_curves = [], []
    candidate_writer = None
    total = 0
    try:
        for tenor, cycles in by_tenor.items():
            for base in bases:
                specs = candidate_catalog(base)
                metrics, nav = simulate_candidates(specs, cycles, dates, tenor)
                winners = choose_winners(metrics)
                selected.append(winners)
                if not candidate_only:
                    selected_curves.append(nav[:, winners.index])
                metrics["selected"] = metrics.index.isin(winners.index)
                table = pa.Table.from_pandas(metrics[CANDIDATE_COLUMNS], preserve_index=False)
                if candidate_writer is None:
                    candidate_writer = pq.ParquetWriter(output/"candidate_metrics.parquet", table.schema, compression="zstd")
                candidate_writer.write_table(table)
                total += len(metrics)
            print(f"Optimized {TENORS[tenor]}: {len(bases)} short variations; {total:,} candidates so far; {time.monotonic()-started:.0f}s", flush=True)
    finally:
        if candidate_writer:
            candidate_writer.close()
    winners = pd.concat(selected, ignore_index=True)
    if candidate_only:
        prior = pd.read_csv(output/"combination_metrics.csv")
        np.testing.assert_array_equal(winners.strategy_id, prior.strategy_id)
        print(f"Rebuilt compact candidate report; all {len(winners):,} selections unchanged.", flush=True)
        return
    nav = np.column_stack(selected_curves)
    winners["selection_basis"] = "Highest full-sample combined daily Sharpe; retrospective selection"
    winners.to_csv(output/"combination_metrics.csv", index=False)
    # Re-run only the selected strategies, retaining the full entry ledger.
    chunks, writer = [], None
    def flush():
        nonlocal writer
        if chunks:
            table = pa.Table.from_pandas(pd.concat(chunks, ignore_index=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output/"trade_ledger.parquet", table.schema, compression="zstd")
            writer.write_table(table)
            chunks.clear()
    def ledger(frame):
        chunks.append(frame)
        if len(chunks) >= 40:
            flush()
    try:
        for tenor, cycles in by_tenor.items():
            subset = winners[winners.tenor_days.eq(tenor)]
            _, saved_nav = simulate_candidates(subset.reset_index(drop=True), cycles, dates, tenor, ledger)
            np.testing.assert_allclose(saved_nav, nav[:, subset.index], rtol=0, atol=0)
        flush()
    finally:
        if writer:
            writer.close()
    old = pd.read_csv(BASE_OUTPUT/"strategy_metrics.csv", dtype={"folder": str})
    with np.load(BASE_OUTPUT/"curves.npz") as archive:
        np.testing.assert_array_equal(dates.to_numpy(), archive["dates"])
        np.testing.assert_array_equal(old.strategy_id.to_numpy(str), archive["strategy_ids"])
        all_nav = np.column_stack([archive["equity"], nav])
    metrics = pd.concat([old, winners], ignore_index=True)
    metrics.to_csv(output/"strategy_metrics.csv", index=False)
    np.savez_compressed(output/"curves.npz", dates=dates.to_numpy(dtype="datetime64[ns]"),
        strategy_ids=metrics.strategy_id.to_numpy(str), equity=all_nav)
    config = json.loads((source/"run.json").read_text(encoding="utf-8"))
    config.update(strategies=len(metrics), combination_strategies=len(winners),
        premium_fractions=list(FRACTIONS), hedge_widths_pct_points=list(WIDTHS),
        best_hedge_selection=True, candidate_count=total, short_variations=len(bases),
        minimum_relative_base_coverage=MIN_RELATIVE_COVERAGE,
        selection_rule="Maximum full-sample combined-strategy daily Sharpe per budget, short variation, expiration and hedge type; ties CAGR, drawdown, ID",
        selection_is_retrospective=True, hedge_strike_grid_pct=list(range(80, 111)),
        folder_order="premium budget / short-put variation / best hedge type / expiration chart",
        hedge_rule="Same-expiry put at/below base spread lower strike (single short: its strike), or debit buffer starting there or lower; search one-point target grid down to 80%, widths 1/2/3/5/10 points",
        combination_notional_label="Short option strategy: 100% SPX notional; hedge uses up to the stated share of net short premium after costs, capped at 100% SPX notional; no SPX holding",
        premium_budget_basis="net short-strategy premium after slippage and commissions",
        combination_sizing="Hedge contracts per short contract = min(1, premium fraction x net short premium / all-in hedge debit); fixed until roll",
        combination_result_dir=str(output), source_quote_dir=str(source))
    config["categories"]["Both"] = "Short puts and short put spreads with the best historical long put or put buffer at each premium budget"
    config["source_artifact_sha256"] = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
        for name in ("entries.parquet", "schedule.parquet", "audited_quotes.parquet")}
    (output/"run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {len(winners):,} winners from {total:,} candidates; elapsed {time.monotonic()-started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()
    run(args.output, candidate_only=args.candidates_only)
