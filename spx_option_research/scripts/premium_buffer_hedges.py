"""Choose premium-funded buffers from entry quotes, without return ranking.

The selector accepts a minimum quantity explicitly. Production sizing is set in
run.json by the caller; it must never be inferred from historical performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from organized_spx_engine import INITIAL, TENORS, catalog
from optimize_organized_spx_hedges import FRACTIONS


def buffer_catalog():
    rows = []
    for base in catalog().query("category == 'Put selling'").to_dict("records"):
        high, width = int(base["primary_pct"]), int(base["width_pct"])
        name = f"{high}-{high-width}" if width else str(high)
        for fraction in FRACTIONS:
            pct = round(fraction*100)
            title = f"Sell {name.replace('-', '/')} + closest affordable buffer ({pct}% premium)"
            rows.append(dict(category="Both", primary_pct=high,
                secondary_pct=high-width if width else np.nan, width_pct=width,
                short_variation=name, hedge_type="Downside buffer",
                hedge_primary_pct=np.nan, hedge_secondary_pct=np.nan,
                hedge_width_pct=np.nan, premium_fraction=fraction,
                notional_multiple=1., side=-1, underlying=0,
                structure=("Short spread" if width else "Short put") + " + downside buffer",
                folder=f"{pct:02d} percent premium/{name}/Best put buffer",
                title=title, comparison_title=title,
                structure_id=f"buffer_{name}_p{pct:02d}",
                selection_basis="Entry-date closest affordable buffer; prefer 5 points, allow 4 or 3",
                dynamic_buffer=True, hedge_notional_cap=np.nan))
    return pd.DataFrame(rows)


def select_buffers(cycle, budgets, *, minimum_quantity):
    """Highest target <= ATM; at the same target, prefer width 5, then 4, then 3.

    Targets use the existing one-percentage-point grid. Actual listed widths
    must cover at least 3% of entry SPX. All-in costs count against the budget.
    q = budget / debit uses the entire allocation, including fractional units.
    """
    if minimum_quantity not in (0, 1):
        raise ValueError("minimum_quantity must explicitly be 0 or 1")
    candidates = [(high-80, high-width-80, width)
                  for high in range(100, 82, -1) for width in (5, 4, 3)
                  if high-width >= 80]
    hh, hl, widths = np.array(candidates).T
    k, mid, cost, valid = (cycle[key] for key in ("k", "mid", "friction", "valid"))
    spread_mid = mid[hh]-mid[hl]
    debit = spread_mid+cost[hh]+cost[hl]
    actual_width = k[hh]-k[hl]
    usable = (valid[hh] & valid[hl] & (actual_width >= .03*cycle["spot"]-1e-10)
              & (spread_mid >= 0) & (spread_mid <= actual_width+1e-10)
              & np.isfinite(debit) & (debit > 0) & (debit < actual_width))
    budgets = np.asarray(budgets, float)
    feasible = (usable[:, None] & (budgets[None, :] > 0)
                & (minimum_quantity*debit[:, None] <= budgets[None, :]+1e-12))
    traded = feasible.any(axis=0)
    choice = feasible.argmax(axis=0)
    paid = debit[choice]
    quantity = np.divide(budgets, paid, out=np.zeros_like(budgets), where=traded)
    return dict(traded=traded, high=hh[choice], low=hl[choice], width=widths[choice],
                debit=paid, quantity=quantity,
                actual_width_pct=actual_width[choice]/cycle["spot"]*100)


def simulate_buffers(specs, cycles, dates, tenor, *, minimum_quantity, ledger_writer=None):
    n = len(specs)
    hi = specs.primary_pct.to_numpy(int)-80
    spread = specs.width_pct.to_numpy() > 0
    lo = specs.secondary_pct.fillna(80).to_numpy(int)-80
    fractions = specs.premium_fraction.to_numpy(float)
    nav = np.full((len(dates), n), np.nan)
    equity = np.full(n, INITIAL)
    counts, base_counts, estimates = (np.zeros(n, int) for _ in range(3))
    q_sum, q_max, spend_sum, credit_sum, error_max = (np.zeros(n) for _ in range(5))
    target_min, width_min, actual_min, dte_min = (np.full(n, np.inf) for _ in range(4))
    target_max, width_max, actual_max, dte_max = (np.full(n, -np.inf) for _ in range(4))
    target_sum = np.zeros(n)
    width_counts = {w: np.zeros(n, int) for w in (3, 4, 5)}
    ids = np.char.add(specs.structure_id.to_numpy(str), f"_d{tenor:02d}")
    for cycle in cycles:
        k, mid, cost, valid = (cycle[key] for key in ("k", "mid", "friction", "valid"))
        credit = mid[hi]-cost[hi]-np.where(spread, mid[lo]+cost[lo], 0)
        base_valid = (valid[hi] & (~spread | valid[lo]) & (credit > 0)
            & (~spread | ((k[hi] > k[lo]) & (credit <= k[hi]-k[lo]+1e-12))))
        selected = select_buffers(cycle, np.where(base_valid, credit*fractions, 0),
                                  minimum_quantity=minimum_quantity)
        eligible, hh, hl, q, debit = (selected[key] for key in ("traded", "high", "low", "quantity", "debit"))
        base_return = cycle["short"][:, hi] + np.where(spread, cycle["long"][:, lo], 0)
        hedge_return = cycle["long"][:, hh]+cycle["short"][:, hl]
        cumulative = base_return+q*hedge_return
        cumulative[:, ~eligible] = 0
        before = equity.copy()
        nav[cycle["ix"]] = before*(1+cumulative)
        equity = nav[cycle["ix"][-1]].copy()
        # Independent entry-cost and expiry-payoff invariants for every trade.
        np.testing.assert_allclose(q[eligible]*debit[eligible], fractions[eligible]*credit[eligible], rtol=1e-12, atol=1e-10)
        entry_cost = cost[hi]+np.where(spread, cost[lo], 0)+q*(cost[hh]+cost[hl])
        np.testing.assert_allclose(cumulative[0, eligible], -entry_cost[eligible]/cycle["spot"], rtol=1e-10, atol=1e-12)
        if minimum_quantity:
            assert np.all(q[eligible] >= 1-1e-10)
        used = cycle["estimated"][hi].copy()
        used += np.where(spread & (lo != hi), cycle["estimated"][lo], 0)
        used += np.where((hh != hi) & (~spread | (hh != lo)), cycle["estimated"][hh], 0)
        used += np.where((hl != hi) & (~spread | (hl != lo)) & (hl != hh), cycle["estimated"][hl], 0)
        estimates += used*eligible
        counts += eligible
        base_counts += base_valid
        q_sum += q
        q_max = np.maximum(q_max, q)
        spend_sum += np.where(eligible, fractions, 0)
        credit_sum += np.where(eligible, credit/cycle["spot"], 0)
        error = np.maximum.reduce([cycle["error"][hi], np.where(spread, cycle["error"][lo], 0), cycle["error"][hh], cycle["error"][hl]])
        error_max = np.maximum(error_max, np.where(eligible, error, 0))
        target = hh+80
        width, actual = selected["width"], selected["actual_width_pct"]
        target_sum += np.where(eligible, target, 0)
        for value, vmin, vmax in ((target, target_min, target_max), (width, width_min, width_max),
                                 (actual, actual_min, actual_max), (cycle["actual_dte"], dte_min, dte_max)):
            vmin[:] = np.minimum(vmin, np.where(eligible, value, np.inf))
            vmax[:] = np.maximum(vmax, np.where(eligible, value, -np.inf))
        for w in width_counts:
            width_counts[w] += eligible & (width == w)
        if ledger_writer is not None:
            ledger_writer(pd.DataFrame(dict(strategy_id=ids, roll_id=cycle["roll_id"],
                target_dte=tenor, entry_date=cycle["entry_date"], expiration_date=cycle["expiration_date"],
                valuation_end=cycle["valuation_end"], traded=eligible, base_eligible=base_valid,
                net_short_credit_points=credit, premium_fraction=fractions,
                hedge_debit_points=np.where(eligible, debit, np.nan), hedge_quantity_ratio=q,
                hedge_primary_pct=np.where(eligible, target, np.nan),
                hedge_secondary_pct=np.where(eligible, hl+80, np.nan),
                hedge_width_pct=np.where(eligible, width, np.nan),
                actual_buffer_width_pct=np.where(eligible, actual, np.nan),
                short_strike=k[hi], base_long_strike=np.where(spread, k[lo], np.nan),
                hedge_long_strike=np.where(eligible, k[hh], np.nan),
                hedge_short_strike=np.where(eligible, k[hl], np.nan),
                net_credit_retained_points=np.where(eligible, credit-q*debit, 0),
                spot_entry=cycle["spot"], entry_equity=before, ending_equity=equity,
                terminal_return=cumulative[-1],
                skip_reason=np.where(eligible, "", np.where(base_valid, "No affordable valid 3-5 point buffer", "Base short ineligible")))))
    result = specs.copy()
    result["strategy_id"], result["tenor"], result["tenor_days"] = ids, TENORS[tenor], tenor
    daily = nav/np.vstack([np.full((1, n), INITIAL), nav[:-1]])-1
    sd, mean = daily.std(axis=0, ddof=1), daily.mean(axis=0)
    peaks = np.maximum.accumulate(np.vstack([np.full((1, n), INITIAL), nav]), axis=0)[1:]
    finite = np.isfinite(nav).all(axis=0) & (nav > 0).all(axis=0)
    elapsed = (dates[-1]-dates[0]).days/365.2425
    fields = dict(cagr=(nav[-1]/INITIAL)**(1/elapsed)-1,
        annualized_volatility=sd*np.sqrt(252),
        daily_sharpe=np.divide(mean*np.sqrt(252), sd, out=np.full(n, np.nan), where=sd>0),
        annualized_arithmetic_return=mean*252, ending_equity=nav[-1], total_return=nav[-1]/INITIAL-1,
        max_drawdown=(nav/peaks-1).min(axis=0), missing_daily_marks=(~np.isfinite(nav)).sum(axis=0),
        daily_observations=len(dates), cycles=len(cycles), traded_cycles=counts,
        base_eligible_cycles=base_counts, cash_cycles=len(cycles)-counts,
        no_affordable_buffer_cycles=base_counts-counts, trade_fraction=counts/len(cycles),
        relative_base_coverage=np.divide(counts, base_counts, out=np.zeros(n), where=base_counts>0),
        status=np.where(finite, "complete", "invalid"), estimated_quote_count=estimates,
        actual_dte_min=np.where(counts>0, dte_min, np.nan), actual_dte_max=np.where(counts>0, dte_max, np.nan),
        max_strike_error_pct_points=error_max*100,
        mean_hedge_notional_multiple=q_sum/np.maximum(counts, 1), max_hedge_notional_multiple=q_max,
        mean_premium_fraction_spent=spend_sum/np.maximum(counts, 1),
        mean_short_credit_pct_notional=credit_sum/np.maximum(counts, 1),
        mean_hedge_spend_pct_notional=fractions*credit_sum/np.maximum(counts, 1),
        hedge_target_min=np.where(counts>0, target_min, np.nan),
        hedge_target_max=np.where(counts>0, target_max, np.nan),
        mean_hedge_target=target_sum/np.maximum(counts, 1),
        buffer_width_min=np.where(counts>0, width_min, np.nan),
        buffer_width_max=np.where(counts>0, width_max, np.nan),
        actual_buffer_width_min=np.where(counts>0, actual_min, np.nan),
        actual_buffer_width_max=np.where(counts>0, actual_max, np.nan),
        minimum_buffer_quantity=minimum_quantity)
    for w, values in width_counts.items():
        fields[f"width_{w}_cycles"] = values
    for key, value in fields.items():
        result[key] = value
    return result, nav
