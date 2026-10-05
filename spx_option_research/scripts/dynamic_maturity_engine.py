"""Portfolio accounting for entry-by-entry choices across expirations and strikes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from maturity_grid_data import CADENCES, CADENCE_DAYS, SIZINGS, SHORTS, WIDTHS, TENORS, RATIOS
from maturity_grid_engine import INITIAL, statistics


def profiles(selector_count):
    ids = np.indices((selector_count,4,2,2)).reshape(4,-1)
    return pd.DataFrame(dict(selector_index=ids[0],cadence_index=ids[1],sizing_index=ids[2],mode_index=ids[3],
        cadence=np.asarray(CADENCES)[ids[1]],sizing=np.asarray(SIZINGS)[ids[2]],
        position_mode=np.asarray(['ladder','one_at_a_time'])[ids[3]]))


def chosen_increments(cycle, path, geometries):
    """Full-precision marks for only the geometries selected on this entry."""
    unique,inverse = np.unique(geometries,return_inverse=True)
    if 'increments' in path:
        return path['increments'][geometries,0,:]
    high = np.repeat(SHORTS,len(WIDTHS))[unique]
    width = np.tile(WIDTHS,len(SHORTS))[unique]
    hi = np.rint((high-RATIOS[0])*2).astype(int)
    lo = np.rint((high-width-RATIOS[0])*2).astype(int)
    raw = cycle.mid[:,hi]-cycle.mid[:,lo]
    friction = cycle.cost[:,hi]+cycle.cost[:,lo]
    actual_width = cycle.strikes[hi]-cycle.strikes[lo]
    half_spread = 2*np.maximum(friction-.03,0)
    feasible = np.isfinite(raw)&(raw+half_spread>=-.011)&(raw-half_spread<=actual_width[None,:]+.011)
    bad = ~feasible|(cycle.quality[:,hi]==0)|(cycle.quality[:,lo]==0)
    for day in range(len(raw)):
        raw[day,bad[day]] = raw[day-1,bad[day]] if day else 0
    raw = np.nan_to_num(raw,nan=0)
    mid_pnl = path['credit'][unique,None]-raw.T*100
    offsets = path['offsets'][unique,0]
    cumulative = np.where(np.arange(len(raw))[None,:]>=offsets[:,None],path['terminal'][unique,0,None],mid_pnl)
    increments = np.nan_to_num(np.diff(cumulative,axis=1,prepend=0),nan=0,posinf=0,neginf=0)
    return increments[inverse]


def simulate(data, selections, profile=None, ledger=False, scores=None, candidate_counts=None):
    if profile is None:
        profile = profiles(len(selections))
    profile = profile.reset_index(drop=True)
    total, n = len(profile), len(data.dates)
    geometries = len(SHORTS)*len(WIDTHS)
    selector = profile.selector_index.to_numpy(int)
    cadence = profile.cadence_index.to_numpy(int)
    sizing = profile.sizing_index.to_numpy(int)
    serial = profile.mode_index.to_numpy(int) == 1
    daily = np.zeros((total,n))
    release_risk = np.zeros_like(daily)
    release_notional = np.zeros_like(daily)
    equity = np.full(total,INITIAL)
    active_risk = np.zeros(total)
    active_notional = np.zeros(total)
    busy_until = np.full(total,-1,dtype=np.int32)
    maximum_risk = np.zeros(total)
    maximum_notional = np.zeros(total)
    counters = {k:np.zeros(total,dtype=np.int32) for k in (
        'trades','wins','profit_hits','unresolved_marks','estimated_marks','bound_flags',
        'selection_unresolved','selection_trades','available_entries','selection_available_entries',
        'positive_gate_cash_entries','occupied_entries','choice_changes')}
    accumulators = {k:np.zeros(total) for k in ('holding_days','actual_dte','notional_days','entry_scores')}
    previous_choice = np.full(total,-1,dtype=np.int32)
    maturity_bits = np.zeros(total,dtype=np.uint32)
    short_bits = np.zeros(total,dtype=np.uint64)
    width_bits = np.zeros(total,dtype=np.uint16)
    records = []
    previous = 0
    selection_end = data.start+int((n-data.start)*.8)-1
    for row, entry in enumerate(data.entries):
        if entry < data.start:
            continue
        equity += daily[:,previous:entry].sum(axis=1)
        active_risk -= release_risk[:,previous+1:entry+1].sum(axis=1)
        active_notional -= release_notional[:,previous+1:entry+1].sum(axis=1)
        active_risk = np.maximum(active_risk,0)
        active_notional = np.maximum(active_notional,0)
        previous = int(entry)
        scheduled = data.masks[entry,cadence]
        chosen = selections[selector,row]
        available = (candidate_counts[selector,row]>0) if candidate_counts is not None else chosen>=0
        counters['available_entries'] += scheduled&available
        if entry <= selection_end:
            counters['selection_available_entries'] += scheduled&available
        counters['positive_gate_cash_entries'] += scheduled&available&(chosen<0)
        blocked = serial&(busy_until>entry)
        counters['occupied_entries'] += scheduled&(chosen>=0)&blocked
        requested = scheduled&(chosen>=0)&~blocked&(equity>0)
        for ti in range(len(TENORS)):
            ix = np.flatnonzero(requested&(chosen//geometries==ti))
            if not len(ix):
                continue
            path = data.paths.get((row,ti))
            if path is None:
                raise AssertionError('Selector chose an unavailable expiration')
            cycle = data.cycles[(row,ti)]
            g = chosen[ix]%geometries
            assert data.features['valid'][row,chosen[ix]].all()
            unit_risk = path['max_loss'][g]
            tranches = np.where(serial[ix],1,np.maximum(1,np.ceil(cycle.dte/np.asarray(CADENCE_DAYS)[cadence[ix]])))
            risk_q = np.minimum(np.minimum(INITIAL*.05/tranches,np.maximum(INITIAL*.05-active_risk[ix],0))/unit_risk,
                np.maximum(INITIAL*5-active_notional[ix],0)/(cycle.spot*100))
            notional_q = np.minimum(np.maximum(equity[ix],0)/tranches,np.maximum(equity[ix]-active_notional[ix],0))/(cycle.spot*100)
            quantity = np.where(sizing[ix]==0,risk_q,notional_q)
            traded = quantity>1e-12
            ix, g, quantity, unit_risk = ix[traded], g[traded], quantity[traded], unit_risk[traded]
            if not len(ix):
                continue
            offsets = path['offsets'][g,0]
            exits = entry+offsets
            increments = chosen_increments(cycle,path,g)
            duration = increments.shape[1]
            daily[ix,entry:entry+duration] += quantity[:,None]*increments
            added_risk = quantity*unit_risk
            added_notional = quantity*cycle.spot*100
            release_risk[ix,exits] += added_risk
            release_notional[ix,exits] += added_notional
            active_risk[ix] += added_risk
            active_notional[ix] += added_notional
            maximum_risk[ix] = np.maximum(maximum_risk[ix],active_risk[ix])
            maximum_notional[ix] = np.maximum(maximum_notional[ix],active_notional[ix])
            busy_until[ix] = np.maximum(busy_until[ix],exits)
            holding = (data.dates.to_numpy()[exits]-data.dates.to_numpy()[entry])/np.timedelta64(1,'D')
            counters['trades'][ix] += 1
            counters['wins'][ix] += path['terminal'][g,0]>0
            counters['profit_hits'][ix] += path['profit_hit'][g,0]
            counters['unresolved_marks'][ix] += path['unresolved'][g,0]
            counters['estimated_marks'][ix] += path['estimated'][g,0]
            counters['bound_flags'][ix] += path['bound_flags'][g,0]
            counters['selection_unresolved'][ix] += path['selection_unresolved'][g]
            counters['selection_trades'][ix] += entry<=selection_end
            counters['choice_changes'][ix] += (previous_choice[ix]>=0)&(previous_choice[ix]!=chosen[ix])
            previous_choice[ix] = chosen[ix]
            maturity_bits[ix] |= np.uint32(1<<ti)
            short_bits[ix] |= np.left_shift(np.uint64(1),(g//len(WIDTHS)).astype(np.uint64))
            width_bits[ix] |= np.left_shift(np.uint16(1),(g%len(WIDTHS)).astype(np.uint16))
            accumulators['holding_days'][ix] += holding
            accumulators['actual_dte'][ix] += cycle.dte
            accumulators['notional_days'][ix] += added_notional*offsets
            if scores is not None:
                accumulators['entry_scores'][ix] += scores[selector[ix],row]
            if ledger:
                for j, index in enumerate(ix):
                    candidate = int(chosen[index])
                    parameter = data.catalog.iloc[candidate]
                    records.append(dict(index=int(index),entry=int(entry),exit=int(exits[j]),candidate=candidate,
                        entry_row=row,tenor_index=ti,geometry=int(g[j]),expiry=str(cycle.expiry.date()),
                        target_dte=int(parameter.target_dte),actual_dte=cycle.dte,
                        short_target=float(parameter.short_target),width=float(parameter.width),
                        short_strike=float(path['short_strike'][g[j]]),long_strike=float(path['long_strike'][g[j]]),
                        spot=cycle.spot,quantity=float(quantity[j]),entry_equity=float(equity[index]),
                        entry_credit=float(path['credit'][g[j]]),unit_max_loss=float(unit_risk[j]),
                        closing_cost_at_entry=float(data.features['closing_cost'][row,candidate]),
                        entry_score=float(scores[selector[index],row]) if scores is not None else np.nan,
                        candidates_considered=int(candidate_counts[selector[index],row]) if candidate_counts is not None else 0,
                        realized_pnl=float(path['terminal'][g[j],0]*quantity[j]),profit_hit=bool(path['profit_hit'][g[j],0]),
                        unresolved=int(path['unresolved'][g[j],0]),estimated=int(path['estimated'][g[j],0])))
    curves = INITIAL+daily.cumsum(axis=1)
    planned = data.masks[data.start:].sum(axis=0)[cadence]
    selection_planned = data.masks[data.start:selection_end+1].sum(axis=0)[cadence]
    count = counters['trades']
    divide = lambda values: np.divide(values,count,out=np.zeros(total),where=count>0)
    meta = {**counters, 'win_rate':divide(counters['wins']), 'profit_hit_rate':divide(counters['profit_hits']),
        'entry_coverage':counters['available_entries']/planned,
        'selection_entry_coverage':counters['selection_available_entries']/selection_planned,
        'mean_actual_dte':divide(accumulators['actual_dte']), 'mean_holding_days':divide(accumulators['holding_days']),
        'mean_entry_score':divide(accumulators['entry_scores']),
        'mean_notional_pct_initial':accumulators['notional_days']/((n-data.start)*INITIAL),
        'max_risk_pct_initial':maximum_risk/INITIAL,'max_notional_pct_initial':maximum_notional/INITIAL,
        'unique_maturity_targets':np.array([int(x).bit_count() for x in maturity_bits]),
        'unique_short_targets':np.array([int(x).bit_count() for x in short_bits]),
        'unique_widths':np.array([int(x).bit_count() for x in width_bits]),
        'planned_entries':planned, 'selection_planned_entries':selection_planned,
        **statistics(curves[:,data.start:],data.dates[data.start:])}
    return curves, meta, records
