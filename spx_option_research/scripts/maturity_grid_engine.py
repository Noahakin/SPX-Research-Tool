"""Vectorized profit-taking and overlapping portfolio simulation.

The original fixed 5% loss-budget convention is retained as one sizing mode.
The second mode splits 100% of prior-close equity notional across potential
overlaps. Entries observe risk/notional caps; existing trades keep their size.
"""
from __future__ import annotations
import numpy as np

from maturity_grid_data import RATIOS, WIDTHS, PROFITS, CADENCES, CADENCE_DAYS, SIZINGS

INITIAL=1_000_000.


def trade_paths(cycle,shorts,widths=WIDTHS,profits=PROFITS):
    high=np.repeat(np.asarray(shorts,float),len(widths)); width=np.tile(widths,len(shorts))
    hi=np.rint((high-RATIOS[0])*2).astype(int); lo=np.rint((high-width-RATIOS[0])*2).astype(int)
    raw=cycle.mid[:,hi]-cycle.mid[:,lo]
    friction=cycle.cost[:,hi]+cycle.cost[:,lo]
    actual_width=cycle.strikes[hi]-cycle.strikes[lo]
    quality=np.minimum(cycle.quality[:,hi],cycle.quality[:,lo])
    # Both legs must be observed to execute; quality=2 is valuation only.
    execution_observed=(cycle.quality[:,hi]==1)&(cycle.quality[:,lo]==1)
    bounded=np.isfinite(raw)&(raw>=-.011)&(raw<=actual_width[None,:]+.011)
    # Individual observed midpoints can differ by slightly more than the
    # payoff width when the two legs have wide markets. Preserve raw marks
    # and flag them; reject inconsistent markets whose entire bid/ask interval
    # lies outside the payoff bounds. Never let an unbounded mid trigger profit.
    half_spread=2*np.maximum(friction-.03,0)
    feasible=np.isfinite(raw)&(raw+half_spread>=-.011)&(raw-half_spread<=actual_width[None,:]+.011)
    execution_observed &= feasible
    observed=execution_observed&bounded
    unresolved=~feasible|(quality==0)
    credit=(raw[0]-friction[0])*100
    maximum_loss=actual_width*100-credit
    entry_valid=observed[0]&(credit>0)&(maximum_loss>0)&(actual_width>0)&(cycle.distances[hi]<=.005)&(cycle.distances[lo]<=.005)
    # Unresolved interim marks use the last supported mark solely as a
    # diagnostic. Any portfolio holding one is ineligible for the rankings.
    marked=raw.copy()
    for t in range(len(marked)):
        bad=unresolved[t]
        marked[t,bad]=marked[t-1,bad] if t else 0
    marked=np.nan_to_num(marked,nan=0)
    liquidation=credit[None,:]-(marked+np.nan_to_num(friction,nan=0))*100
    pnl_mid=credit[None,:]-marked*100
    n=len(marked); p=np.asarray(profits,float)
    triggers=(liquidation[:,:,None]>=credit[None,:,None]*p[None,None,:])&observed[:,:,None]
    triggers[0]=False
    # Expiry cash settlement and the common research-end close terminate
    # every position, even if its profit target was not reached.
    hit=triggers.any(axis=0)
    offsets=np.where(hit,triggers.argmax(axis=0),n-1)
    terminal=liquidation[offsets,np.arange(len(high))[:,None]]
    times=np.arange(n)
    cumulative=np.where(times[None,None,:]>=offsets[:,:,None],terminal[:,:,None],pnl_mid.T[:,None,:])
    increments=np.diff(cumulative,axis=2,prepend=0)
    increments=np.nan_to_num(increments,nan=0,posinf=0,neginf=0)
    unresolved_held=np.cumsum(unresolved,axis=0)[offsets,np.arange(len(high))[:,None]]
    unresolved_held+=(~execution_observed[offsets,np.arange(len(high))[:,None]])
    estimated_held=np.cumsum(((cycle.quality[:,hi]==2)|(cycle.quality[:,lo]==2))&(~unresolved),axis=0)[offsets,np.arange(len(high))[:,None]]
    bound_flags=np.cumsum(~bounded,axis=0)[offsets,np.arange(len(high))[:,None]]
    return dict(high=high,width=width,actual_width=actual_width,credit=credit,max_loss=maximum_loss,entry_valid=entry_valid,
        offsets=offsets,terminal=terminal,increments=increments,unresolved=unresolved_held,estimated=estimated_held,bound_flags=bound_flags,
        profit_hit=hit,short_strike=cycle.strikes[hi],long_strike=cycle.strikes[lo])


def simulate(cycles,shorts,dates,masks,widths=WIDTHS,profits=PROFITS,ledger=False):
    """Return curves and audit statistics for geometry × profit × cadence × sizing."""
    geometries=len(shorts)*len(widths); p=len(profits); n=len(dates)
    shape=(geometries,p,4,2); total=int(np.prod(shape))
    indices=np.arange(total).reshape(shape)
    daily=np.zeros((total,n)); release_risk=np.zeros_like(daily); release_notional=np.zeros_like(daily)
    equity=np.full(total,INITIAL); active_risk=np.zeros(total); active_notional=np.zeros(total)
    counts=np.zeros(total,dtype=np.int32); wins=counts.copy(); hits=counts.copy(); unresolved=counts.copy(); estimated=counts.copy()
    bound_flags=counts.copy()
    active_days=np.zeros(total); exposure_days=np.zeros(total); risk_days=np.zeros(total)
    max_risk=np.zeros(total);max_notional=np.zeros(total); holding_sum=np.zeros(total);credit_sum=np.zeros(total)
    available=np.zeros((geometries,4),dtype=int); dte_sum=np.zeros_like(available,dtype=float)
    previous=0; records=[]
    for cycle in cycles:
        entry=cycle.entry
        if entry>previous:
            equity+=daily[:,previous:entry].sum(axis=1)
            active_risk-=release_risk[:,previous+1:entry+1].sum(axis=1)
            active_notional-=release_notional[:,previous+1:entry+1].sum(axis=1)
        previous=entry
        active_risk=np.maximum(active_risk,0);active_notional=np.maximum(active_notional,0)
        trade=trade_paths(cycle,shorts,widths,profits)
        duration=trade['increments'].shape[-1]
        unit_risk=np.nan_to_num(np.broadcast_to(trade['max_loss'][:,None],(geometries,p)).reshape(-1),nan=0)
        unit_credit=np.nan_to_num(np.broadcast_to(trade['credit'][:,None],(geometries,p)).reshape(-1),nan=0)
        valid=np.broadcast_to(trade['entry_valid'][:,None],(geometries,p)).reshape(-1)
        increments=trade['increments'].reshape(geometries*p,duration)
        exits=entry+trade['offsets'].reshape(-1)
        calendar_holding=(dates.to_numpy()[exits]-dates.to_numpy()[entry])/np.timedelta64(1,'D')
        for cadence in np.flatnonzero(cycle.cadence):
            available[:,cadence]+=trade['entry_valid']; dte_sum[:,cadence]+=trade['entry_valid']*cycle.dte
            tranches=max(1,int(np.ceil(cycle.dte/CADENCE_DAYS[cadence])))
            for sizing in range(2):
                ix=indices[:,:,cadence,sizing].reshape(-1)
                risk_cap=np.maximum(INITIAL*.05-active_risk[ix],0)
                if sizing==0:
                    allocation=np.minimum(INITIAL*.05/tranches,risk_cap)
                    quantity=np.divide(allocation,unit_risk,out=np.zeros_like(allocation),where=unit_risk>0)
                    # A disclosed five-times notional ceiling prevents nearly
                    # fully intrinsic spreads from implying unlimited leverage.
                    quantity=np.minimum(quantity,np.maximum(INITIAL*5-active_notional[ix],0)/(cycle.spot*100))
                else:
                    notional_cap=np.maximum(equity[ix]-active_notional[ix],0)
                    quantity=np.minimum(np.maximum(equity[ix],0)/tranches,notional_cap)/(cycle.spot*100)
                quantity=np.where(valid&(equity[ix]>0),quantity,0)
                traded=quantity>1e-12
                daily[ix,entry:entry+duration]+=quantity[:,None]*increments
                added_risk=quantity*np.maximum(unit_risk,0); added_notional=quantity*cycle.spot*100
                release_risk[ix,exits]+=added_risk;release_notional[ix,exits]+=added_notional
                active_risk[ix]+=added_risk;active_notional[ix]+=added_notional
                max_risk[ix]=np.maximum(max_risk[ix],active_risk[ix]);max_notional[ix]=np.maximum(max_notional[ix],active_notional[ix])
                counts[ix]+=traded;wins[ix]+=traded&(trade['terminal'].reshape(-1)>0);hits[ix]+=traded&trade['profit_hit'].reshape(-1)
                unresolved[ix]+=traded*trade['unresolved'].reshape(-1);estimated[ix]+=traded*trade['estimated'].reshape(-1)
                bound_flags[ix]+=traded*trade['bound_flags'].reshape(-1)
                active_days[ix]+=traded*(exits-entry)
                exposure_days[ix]+=added_notional*(exits-entry);risk_days[ix]+=added_risk*(exits-entry)
                holding_sum[ix]+=traded*calendar_holding;credit_sum[ix]+=quantity*unit_credit
                if ledger:
                    for j in np.flatnonzero(traded):
                        g,pr=divmod(int(j),p)
                        records.append(dict(index=int(ix[j]),entry=entry,exit=int(exits[j]),expiry=str(cycle.expiry.date()),
                            actual_dte=cycle.dte,entry_equity=float(equity[ix[j]]),quantity=float(quantity[j]),
                            short_strike=float(trade['short_strike'][g]),long_strike=float(trade['long_strike'][g]),
                            entry_credit=float(unit_credit[j]),unit_max_loss=float(unit_risk[j]),
                            realized_pnl=float(trade['terminal'][g,pr]*quantity[j]),profit_hit=bool(trade['profit_hit'][g,pr]),
                            unresolved=int(trade['unresolved'][g,pr]),estimated=int(trade['estimated'][g,pr])))
    curves=INITIAL+daily.cumsum(axis=1)
    planned=np.broadcast_to(masks.sum(axis=0)[None,None,:,None],shape).reshape(-1)
    coverage=np.broadcast_to(available[:,None,:,None],shape).reshape(-1)/planned
    actual_dte=np.divide(dte_sum,available,out=np.zeros_like(dte_sum),where=available>0)
    meta=dict(trades=counts,win_rate=np.divide(wins,counts,out=np.zeros(total),where=counts>0),
        profit_hit_rate=np.divide(hits,counts,out=np.zeros(total),where=counts>0),entry_coverage=coverage,
        mean_actual_dte=np.broadcast_to(actual_dte[:,None,:,None],shape).reshape(-1),
        unresolved_marks=unresolved,estimated_marks=estimated,out_of_bounds_midpoint_marks=bound_flags,
        mean_holding_days=np.divide(holding_sum,counts,out=np.zeros(total),where=counts>0),
        max_risk_pct_initial=max_risk/INITIAL,max_notional_pct_initial=max_notional/INITIAL,
        mean_notional_pct_initial=exposure_days/(n*INITIAL),gross_credit_pct_initial=credit_sum/INITIAL,
        planned_entries=planned)
    return curves,meta,records


def statistics(curves,dates):
    """All ratios use actual daily portfolio returns and calendar-year CAGR."""
    prior=np.concatenate([np.full((len(curves),1),INITIAL),curves[:,:-1]],axis=1)
    returns=curves/prior-1
    output={}
    boundaries=[0,int(len(dates)*.6),int(len(dates)*.8),len(dates)]
    intervals=[('',0,len(dates))]+[(name,a,b) for name,a,b in zip(('train_','validation_','test_'),boundaries[:-1],boundaries[1:])]
    for prefix,start,end in intervals:
        r=returns[:,start:end];nav=curves[:,start:end];base=prior[:,start]
        years=(dates[end-1]-dates[start]).days/365.2425
        mean=r.mean(axis=1);sd=r.std(axis=1,ddof=1)
        valid=(nav>0).all(axis=1)&(base>0)&np.isfinite(r).all(axis=1)
        ratio=np.divide(nav[:,-1],base,out=np.ones(len(curves)),where=base>0)
        cagr=np.power(np.maximum(ratio,0),1/years)-1
        peaks=np.maximum.accumulate(np.concatenate([base[:,None],nav],axis=1),axis=1)[:,1:]
        dd=(nav/peaks-1).min(axis=1)
        sharpe=np.divide(mean*np.sqrt(252),sd,out=np.full(len(curves),np.nan),where=sd>1e-14)
        for key,value in [('cagr',cagr),('vol',sd*np.sqrt(252)),('sharpe',sharpe),('max_drawdown',dd)]:
            output[prefix+key]=np.where(valid,value,np.nan)
    output['ending_equity']=curves[:,-1]
    output['insolvent']=(curves<=0).any(axis=1)
    return output
