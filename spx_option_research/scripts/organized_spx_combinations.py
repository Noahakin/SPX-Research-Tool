"""Combine short put spreads with downside hedges funded by net entry credit.

All position sizes use the current roll's observed quotes. Up to a fixed fraction
of net short-spread credit buys a same-expiry long put or debit put spread, capped
at 100% SPX notional. The portfolio holds no SPX position and never sizes hedges
using future outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from organized_spx_data import DEFAULT_OUTPUT as BASE_OUTPUT, START, END, load_cash
from organized_spx_engine import (INITIAL, FEE_POINTS, SPREAD_FRACTION, STRIKE_TOLERANCE,
                                  TENORS, QUOTE_KEY, catalog, leg_returns, curve_statistics, audit_quotes)

OUTPUT = BASE_OUTPUT.parent / "organized_spx_combinations"
FRACTIONS = (.10, .25, .50)
HEDGE_WIDTHS = (0, 5, 10)
HEDGE_CAP = 1.0


def rename_existing_category(base_output: Path = BASE_OUTPUT) -> None:
    """Change labels only; preserve all IDs, curves, metrics, and quote data."""
    path = base_output / "strategy_metrics.csv"
    metrics = pd.read_csv(path, dtype={"folder":str})
    if metrics.category.eq("Both").any():
        metrics.loc[metrics.category.eq("Both"), "category"] = "SPX with overlay"
        metrics.to_csv(path, index=False)
    path = base_output / "run.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if "Both" in config["categories"]:
        config["categories"]["SPX with overlay"] = config["categories"].pop("Both")
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def combination_catalog(fractions=FRACTIONS) -> pd.DataFrame:
    if not fractions or any(not 0 < f < 1 for f in fractions):
        raise ValueError("Premium fractions must lie strictly between zero and one")
    base = catalog()
    base = base[(base.category == "Put selling") & base.width_pct.gt(0)]
    rows = []
    for spec in base.itertuples(index=False):
        high, low = int(spec.primary_pct), int(spec.secondary_pct)
        for hedge_width in HEDGE_WIDTHS:
            kind = "Long put" if hedge_width == 0 else "Downside buffer"
            hedge_folder = f"Long put {low}" if not hedge_width else f"Buffer {low}-{low-hedge_width}"
            hedge_label = f"{low} put" if not hedge_width else f"{low}/{low-hedge_width} buffer"
            for fraction in fractions:
                pct = int(round(fraction*100))
                rows.append(dict(category="Both", folder=f"{high}-{low}/{hedge_folder}/{pct} percent premium",
                                 structure="Short spread + long put" if not hedge_width else "Short spread + downside buffer",
                                 primary_pct=high, secondary_pct=low, width_pct=high-low,
                                 hedge_type=kind, hedge_primary_pct=low,
                                 hedge_secondary_pct=low-hedge_width if hedge_width else np.nan,
                                 hedge_width_pct=hedge_width, premium_fraction=float(fraction),
                                 notional_multiple=1.0, side=-1, underlying=0,
                                 title=f"Sell {high}/{low} + buy {hedge_label} (up to {pct}% premium)",
                                 structure_id=f"combo_{high}-{low}_h{low}" + (f"-{low-hedge_width}" if hedge_width else "") + f"_p{pct:02d}"))
    result = pd.DataFrame(rows)
    if result.structure_id.duplicated().any() or len(result) != 84*3*len(fractions):
        raise AssertionError("Unexpected combination grid")
    return result


def premium_quantity(net_credit, hedge_debit, fraction, cap=HEDGE_CAP):
    """A budget includes slippage and per-contract fees on both transactions."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Hedge notional cap must be finite and positive")
    c, d, f = np.broadcast_arrays(np.asarray(net_credit, float), np.asarray(hedge_debit, float), np.asarray(fraction, float))
    eligible = np.isfinite(c) & np.isfinite(d) & (c > 0) & (d > 0)
    q = np.zeros(c.shape, dtype=float)
    np.divide(f*c, d, out=q, where=eligible)
    np.minimum(q, cap, out=q)
    return q, eligible


def audit_inputs(output: Path = OUTPUT) -> pd.DataFrame:
    quotes = audit_quotes(output)
    prior = pd.read_parquet(BASE_OUTPUT / "audited_quotes.parquet").set_index(QUOTE_KEY)
    current = quotes.set_index(QUOTE_KEY).reindex(prior.index)
    for key in ("bid", "ask", "strike", "used_bid", "used_ask", "used_mid", "estimated", "unresolved"):
        if key in ("estimated", "unresolved"):
            np.testing.assert_array_equal(current[key].to_numpy(), prior[key].to_numpy())
        else:
            np.testing.assert_allclose(current[key].to_numpy(), prior[key].to_numpy(), rtol=0, atol=1e-10)
    print("Existing audited prices reconcile exactly to the original library.", flush=True)
    return quotes


def simulate(output: Path = OUTPUT, *, fractions=FRACTIONS, quotes=None, cash=None,
             write_combined=True, base_output: Path = BASE_OUTPUT, hedge_cap=HEDGE_CAP) -> None:
    output = Path(output)
    entries = pd.read_parquet(output / "entries.parquet")
    schedule = pd.read_parquet(output / "schedule.parquet")
    quotes = pd.read_parquet(output / "audited_quotes.parquet") if quotes is None else quotes
    lookup = quotes.set_index(QUOTE_KEY)
    if lookup.index.duplicated().any():
        raise ValueError("Duplicate date/contract quote keys")
    cash = load_cash().loc[START:END] if cash is None else cash
    dates = cash.index
    specs = combination_catalog(fractions)
    ids0 = specs.structure_id.to_numpy(str)
    hi = specs.primary_pct.to_numpy(int)-80
    lo = specs.secondary_pct.to_numpy(int)-80
    hedge_lo = specs.hedge_secondary_pct.fillna(80).to_numpy(int)-80
    buffer = specs.hedge_width_pct.to_numpy() > 0
    frac = specs.premium_fraction.to_numpy()
    entry_groups = {key: frame.sort_values("ratio") for key,frame in entries.groupby("roll_id", sort=False)}
    metrics, all_curves = [], []
    writer = None
    ledger_chunks = []
    ledger_rows = 0

    def flush():
        nonlocal writer
        if not ledger_chunks:
            return
        table = pa.Table.from_pandas(pd.concat(ledger_chunks, ignore_index=True), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output / "trade_ledger.parquet", table.schema, compression="zstd")
        writer.write_table(table)
        ledger_chunks.clear()

    try:
        for tenor, cycles in schedule.groupby("target_dte", sort=True):
            cycles = cycles.sort_values("entry_date").reset_index(drop=True)
            if not np.array_equal(cycles.expiration_date.iloc[:-1].to_numpy(), cycles.entry_date.iloc[1:].to_numpy()):
                raise ValueError("Roll schedule is not continuous")
            if cycles.entry_date.iloc[0] != dates[0] or cycles.valuation_end.iloc[-1] != dates[-1]:
                raise ValueError("Every tenor must cover the same full period")
            ids = np.char.add(ids0, f"_d{int(tenor):02d}")
            nav = np.full((len(dates),len(specs)),np.nan)
            equity = np.full(len(specs),INITIAL)
            count = np.zeros(len(specs),int)
            estimates = np.zeros(len(specs),int)
            dte_min = np.full(len(specs),np.inf)
            dte_max = np.full(len(specs),-np.inf)
            q_sum = np.zeros(len(specs))
            q_max = np.zeros(len(specs))
            credit_sum = np.zeros(len(specs))
            debit_sum = np.zeros(len(specs))
            capped_count = np.zeros(len(specs),int)
            spend_fraction_sum = np.zeros(len(specs))
            largest_error = np.zeros(len(specs))
            for cycle in cycles.itertuples(index=False):
                ix = np.flatnonzero((dates >= cycle.entry_date) & (dates <= cycle.valuation_end))
                days = dates[ix]
                spot = float(cash.loc[cycle.entry_date])
                traded = np.zeros(len(specs),bool)
                reason = np.full(len(specs),"Required expiration unavailable",object)
                cumulative = np.zeros((len(days),len(specs)))
                est = np.zeros(len(specs),int)
                k, bid, ask = (np.full(31,np.nan) for _ in range(3))
                q = np.zeros(len(specs))
                credit, debit = np.full(len(specs),np.nan),np.full(len(specs),np.nan)
                if cycle.has_expiry:
                    selected = entry_groups[cycle.roll_id]
                    if len(selected) != 31 or not np.array_equal(np.rint(selected.ratio.to_numpy()*100),np.arange(80,111)):
                        raise ValueError("The full 80%–110% leg grid is required on every quoted cycle")
                    k = selected.strike.to_numpy(float)
                    keys = pd.MultiIndex.from_product([days,selected.option_symbol.to_numpy()],names=QUOTE_KEY)
                    observed = lookup.reindex(keys)
                    marks = observed.used_mid.to_numpy(copy=True).reshape(len(days),31)
                    bid = observed.used_bid.to_numpy().reshape(len(days),31)[0]
                    ask = observed.used_ask.to_numpy().reshape(len(days),31)[0]
                    estimated = observed.estimated.fillna(False).to_numpy(dtype=bool,copy=True).reshape(len(days),31)
                    if days[-1] == cycle.expiration_date:
                        marks[-1] = np.maximum(k-cash.loc[days[-1]],0)
                        estimated[-1] = False
                    mid = (bid+ask)/2
                    friction = SPREAD_FRACTION*(ask-bid)+FEE_POINTS
                    # Each price is per one base contract, in SPX points.
                    credit = mid[hi]-mid[lo]-friction[hi]-friction[lo]
                    debit = mid[lo]+friction[lo] + np.where(buffer,-mid[hedge_lo]+friction[hedge_lo],0)
                    q, paid = premium_quantity(credit,debit,frac,cap=hedge_cap)
                    error = np.abs(k/spot-selected.ratio.to_numpy())
                    valid = np.isfinite(bid) & np.isfinite(ask) & (ask>0) & (bid>=0) & (ask>=bid) & ~estimated[0] & (error <= STRIKE_TOLERANCE+1e-12)
                    legs_valid = valid[hi] & valid[lo] & (~buffer | valid[hedge_lo])
                    distinct = (k[hi]>k[lo]) & (~buffer | (k[lo]>k[hedge_lo]))
                    hedge_mid = mid[lo]-np.where(buffer,mid[hedge_lo],0)
                    price_bounds = (credit <= k[hi]-k[lo]+1e-12) & (hedge_mid >= 0) & (~buffer | (hedge_mid <= k[lo]-k[hedge_lo]+1e-12))
                    traded = paid & legs_valid & distinct & price_bounds
                    reason[:] = "Traded"
                    reason[~paid] = "No positive net premium or positive hedge debit"
                    reason[~price_bounds] = "Entry spread quote violates payoff bounds"
                    reason[~legs_valid] = "Entry strike or observed quote unavailable"
                    reason[~distinct] = "Requested spread collapsed to one listed strike"
                    q[~traded] = 0
                    long, short = leg_returns(marks,bid,ask,spot)
                    base_return = short[:,hi]+long[:,lo]
                    hedge_return = long[:,lo]+np.where(buffer[None,:],short[:,hedge_lo],0)
                    cumulative = base_return+q[None,:]*hedge_return
                    cumulative[:,~traded] = 0
                    # Count unique estimated held contract/date marks. The
                    # shared middle put is one contract with larger quantity.
                    est_by_leg = estimated.sum(axis=0)
                    est = (est_by_leg[hi]+est_by_leg[lo]+np.where(buffer,est_by_leg[hedge_lo],0))*traded
                    err = np.maximum(error[hi],np.maximum(error[lo],np.where(buffer,error[hedge_lo],0)))
                    largest_error = np.maximum(largest_error,np.where(traded,err,0))
                before = equity.copy()
                nav[ix] = before[None,:]*(1+cumulative)
                equity = nav[ix[-1]].copy()
                if not np.isfinite(equity).all() or (equity<=0).any():
                    raise ValueError(f"Unresolved terminal valuation or insolvency in {cycle.roll_id}")
                count += traded
                estimates += est
                q_sum += q
                q_max = np.maximum(q_max,q)
                credit_sum += np.where(traded,credit/spot,0)
                debit_sum += np.where(traded,q*debit/spot,0)
                capped_count += traded & np.isclose(q,hedge_cap,rtol=0,atol=1e-12)
                actual_spend_fraction=np.zeros(len(specs))
                np.divide(q*debit,credit,out=actual_spend_fraction,where=traded)
                spend_fraction_sum += actual_spend_fraction
                dte_min = np.minimum(dte_min,np.where(traded,cycle.actual_dte,np.inf))
                dte_max = np.maximum(dte_max,np.where(traded,cycle.actual_dte,-np.inf))
                ledger_chunks.append(pd.DataFrame(dict(
                    strategy_id=ids,roll_id=cycle.roll_id,target_dte=int(tenor),entry_date=cycle.entry_date,
                    expiration_date=cycle.expiration_date,valuation_end=cycle.valuation_end,actual_dte=int(cycle.actual_dte),
                    traded=traded,reason=reason,primary_strike=k[hi],secondary_strike=k[lo],
                    hedge_secondary_strike=np.where(buffer,k[hedge_lo],np.nan),
                    primary_bid=bid[hi],primary_ask=ask[hi],secondary_bid=bid[lo],secondary_ask=ask[lo],
                    hedge_secondary_bid=np.where(buffer,bid[hedge_lo],np.nan),
                    hedge_secondary_ask=np.where(buffer,ask[hedge_lo],np.nan),
                    net_short_credit_points=credit,hedge_debit_points=debit,premium_fraction=frac,
                    hedge_quantity_ratio=q,net_credit_retained_points=np.where(traded,credit-q*debit,0),
                    hedge_notional_cap=float(hedge_cap),
                    spot_entry=spot,entry_equity=before,ending_equity=equity,terminal_return=cumulative[-1],
                    estimated_quote_count=est,final_position_open=cycle.expiration_date>dates[-1])))
                ledger_rows += len(specs)
                if len(ledger_chunks)>=100:
                    flush()
            for i,spec in enumerate(specs.to_dict("records")):
                stats = curve_statistics(nav[:,i],dates)
                if stats["missing_daily_marks"]:
                    raise ValueError("Combination quotes contain unresolved daily values")
                metrics.append(spec | dict(strategy_id=ids[i],tenor=TENORS[int(tenor)],tenor_days=int(tenor),
                    cycles=len(cycles),traded_cycles=int(count[i]),cash_cycles=int(len(cycles)-count[i]),
                    trade_fraction=float(count[i]/len(cycles)),coverage_status=f"{len(cycles)-count[i]}/{len(cycles)} option cycles in cash",
                    status="complete",actual_dte_min=float(dte_min[i]) if count[i] else np.nan,
                    actual_dte_max=float(dte_max[i]) if count[i] else np.nan,estimated_quote_count=int(estimates[i]),
                    max_strike_error_pct_points=float(largest_error[i]*100),
                    mean_hedge_notional_multiple=float(q_sum[i]/count[i]) if count[i] else 0,
                    max_hedge_notional_multiple=float(q_max[i]),
                    hedge_notional_cap=float(hedge_cap),hedge_cap_binding_cycles=int(capped_count[i]),
                    mean_premium_fraction_spent=float(spend_fraction_sum[i]/count[i]) if count[i] else 0,
                    mean_short_credit_pct_notional=float(credit_sum[i]/count[i]) if count[i] else 0,
                    mean_hedge_spend_pct_notional=float(debit_sum[i]/count[i]) if count[i] else 0) | stats)
                all_curves.append(nav[:,i])
            print(f"Combined {TENORS[int(tenor)]}: {len(cycles)} cycles x {len(specs)} portfolios",flush=True)
        flush()
    finally:
        if writer is not None:
            writer.close()
    frame = pd.DataFrame(metrics)
    curves = np.column_stack(all_curves)
    frame.to_csv(output / "combination_metrics.csv",index=False)
    np.savez_compressed(output / "combination_curves.npz",dates=dates.to_numpy(dtype="datetime64[ns]"),
                        strategy_ids=frame.strategy_id.to_numpy(str),equity=curves)
    if write_combined:
        merge_library(output,base_output=base_output,fractions=fractions,hedge_cap=hedge_cap)
    print(f"Saved {len(frame):,} combinations and {ledger_rows:,} roll records.",flush=True)


def merge_library(output: Path=OUTPUT, *, base_output:Path=BASE_OUTPUT, fractions=FRACTIONS,hedge_cap=HEDGE_CAP) -> None:
    rename_existing_category(base_output)
    old = pd.read_csv(base_output/"strategy_metrics.csv",dtype={"folder":str})
    new = pd.read_csv(output/"combination_metrics.csv",dtype={"folder":str})
    a,b = np.load(base_output/"curves.npz"),np.load(output/"combination_curves.npz")
    np.testing.assert_array_equal(a["dates"],b["dates"])
    np.testing.assert_array_equal(a["strategy_ids"],old.strategy_id.to_numpy(str))
    np.testing.assert_array_equal(b["strategy_ids"],new.strategy_id.to_numpy(str))
    metrics = pd.concat([old,new],ignore_index=True)
    if metrics.strategy_id.duplicated().any():
        raise ValueError("Duplicate merged strategy IDs")
    metrics.to_csv(output/"strategy_metrics.csv",index=False)
    np.savez_compressed(output/"curves.npz",dates=a["dates"],strategy_ids=metrics.strategy_id.to_numpy(str),
                        equity=np.column_stack([a["equity"],b["equity"]]))
    config=json.loads((base_output/"run.json").read_text(encoding="utf-8"))
    config.update(strategies=len(metrics),base_strategies=len(old),combination_strategies=len(new),
                  premium_budget_basis="net short-spread credit after execution costs",
                  premium_fractions=list(fractions),hedge_widths_pct_points=[5,10],
                  hedge_notional_cap=float(hedge_cap),premium_budget_is_limit=True,
                  hedge_rule="Buy a put at the short spread's lower strike, or a 5/10-point downside debit spread starting there; same expiry",
                  combination_notional_label=f"Short spread: 100% SPX notional; hedge uses up to the stated share of net premium, capped at {hedge_cap:.0%} SPX notional; no SPX holding",
                  combination_entry_rule="Skip if net credit<=0 or greater than short width, hedge debit<=0, hedge midpoint outside payoff bounds, any leg unavailable/estimated, or either spread collapses",
                  combination_sizing=f"Hedge contracts per short spread = min({hedge_cap:g}, premium fraction x net short credit / all-in hedge debit); fixed until roll",
                  base_result_dir=str(base_output),combination_result_dir=str(output),combined_library=True)
    config["categories"]["Both"]="Short put spread plus a long downside put or buffer, funded from part of its net premium"
    config["source_fingerprint"]=json.loads((output/"manifest.json").read_text())["input_sha256"]
    config["input_artifact_sha256"]={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                                      for name in ("entries.parquet","schedule.parquet","daily_quotes.parquet","audited_quotes.parquet")}
    (output/"run.json").write_text(json.dumps(config,indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--fractions",type=float,nargs="+",default=FRACTIONS)
    parser.add_argument("--reuse-audit",action="store_true")
    parser.add_argument("--hedge-cap",type=float,default=HEDGE_CAP)
    parser.add_argument("--rename-only",action="store_true")
    args=parser.parse_args()
    rename_existing_category()
    if args.rename_only:
        return
    quotes=None if args.reuse_audit else audit_inputs(args.output)
    simulate(args.output,fractions=tuple(args.fractions),quotes=quotes,hedge_cap=args.hedge_cap)


if __name__=="__main__":
    main()
