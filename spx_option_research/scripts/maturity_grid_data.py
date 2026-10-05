"""Read the option archive twice; retain compact fixed-contract paths in RAM.

Only same-day observed quotes authorize entries and profit-taking. Explicit
same-day estimates can value a held contract but cannot trigger execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from organized_spx_data import DATA_ROOT, load_cash, SPXSurfaceArchive
from organized_spx_parity import fit_parity, parity_estimate

START, END = pd.Timestamp('2016-09-23'), pd.Timestamp('2026-09-18')
TENORS = (3,7,10,14,21,28,35,42,45,56,60,75,90,120,150,180)
SHORTS = np.arange(90.,110.01,.5)
WIDTHS = np.array([1.,2.,3.,4.,5.,7.5,10.])
RATIOS = np.arange(80.,110.01,.5)
PROFITS = np.r_[np.arange(.05,.951,.05),np.inf]
CADENCES = ('weekly','every_2_weeks','every_4_weeks','monthly')
CADENCE_DAYS = (7,14,28,30)
SIZINGS = ('risk_5pct_initial','notional_100pct_equity')
COLUMNS = ['snapshot_date','expiration_date','option_symbol','option_type','strike','bid','ask']


@dataclass
class Cycle:
    tenor: int
    entry: int
    end: int
    expiry: pd.Timestamp
    dte: int
    spot: float
    cadence: np.ndarray
    strikes: np.ndarray
    symbols: np.ndarray
    distances: np.ndarray
    mid: np.ndarray
    cost: np.ndarray
    quality: np.ndarray
    settled: bool


def entry_calendar(dates: pd.DatetimeIndex):
    """Holiday-adjusted Friday anchors; monthly uses the third Friday."""
    masks = np.zeros((len(dates),4),dtype=bool)
    for j, anchor in enumerate(pd.date_range(START,END,freq='W-FRI')):
        i = int(dates.searchsorted(anchor,side='right')-1)
        if i < 0 or i >= len(dates)-1 or (anchor-dates[i]).days>4:
            continue
        masks[i,:3] = [True,j%2==0,j%4==0]
    for month in pd.period_range(START,END,freq='M'):
        friday = pd.date_range(month.start_time,month.end_time,freq='W-FRI')[2]
        i = int(dates.searchsorted(friday,side='right')-1)
        if 0 <= i < len(dates)-1 and (friday-dates[i]).days<=4:
            masks[i,3] = True
    return masks


def valid_quotes(bid,ask,strike):
    valid=np.isfinite(bid)&np.isfinite(ask)&np.isfinite(strike)&(bid>=0)&(ask>=bid)&(ask>0)&(strike>0)
    suspicious=(ask>=25)&((bid<=.5)|(bid<ask*.02))
    return valid&~suspicious


def nearest_indices(strikes,targets):
    """Exact distance ties choose the lower strike, before quote screening."""
    hi=np.minimum(np.searchsorted(strikes,targets),len(strikes)-1)
    lo=np.maximum(hi-1,0)
    return np.where(np.abs(strikes[lo]-targets)<=np.abs(strikes[hi]-targets),lo,hi)


def read_snapshot(path: Path, fingerprints: dict, day: pd.Timestamp):
    raw=path.read_bytes()
    fingerprints[day.strftime('%Y-%m-%d')]=hashlib.sha256(raw).hexdigest()
    frame=pq.read_table(pa.BufferReader(raw),columns=COLUMNS).to_pandas()
    if frame.empty:
        return frame
    frame['snapshot_date']=pd.to_datetime(frame.snapshot_date).dt.normalize()
    frame['expiration_date']=pd.to_datetime(frame.expiration_date).dt.normalize()
    frame['option_symbol']=frame.option_symbol.astype(str).str.strip()
    frame['option_type']=frame.option_type.astype(str).str.lower()
    if not frame.snapshot_date.eq(day).all():
        raise ValueError(f'Snapshot date mismatch: {day}')
    return frame.loc[frame.option_symbol.str.startswith('SPXW')].copy()


def estimate_quote(chain, strike, fit_cache, expiry):
    """Return an explicit daily valuation estimate using only this snapshot."""
    if expiry not in fit_cache:
        fit_cache[expiry]=fit_parity(chain)
    fitted=fit_cache[expiry]
    estimate=parity_estimate(fitted,float(strike)) if fitted is not None else None
    if estimate is not None:
        return estimate['used_mid'],(estimate['used_ask']-estimate['used_bid'])*.25+.015,'calibrated_same_day_parity'
    puts=chain.loc[chain.option_type.eq('put')].sort_values('strike')
    good=valid_quotes(puts.bid.to_numpy(float),puts.ask.to_numpy(float),puts.strike.to_numpy(float))
    puts=puts.loc[good]
    if puts.strike.duplicated().any() or len(puts)<2:
        return None
    strikes=puts.strike.to_numpy(float)
    j=int(np.searchsorted(strikes,strike))
    if j==0 or j==len(strikes) or strikes[j]==strike:
        return None
    lower,upper=puts.iloc[j-1],puts.iloc[j]
    limit=50 if max(lower.ask,upper.ask)<=.25 else 25
    if strike-lower.strike>limit or upper.strike-strike>limit:
        return None
    weight=(strike-lower.strike)/(upper.strike-lower.strike)
    bid=float(lower.bid*(1-weight)+upper.bid*weight)
    ask=float(lower.ask*(1-weight)+upper.ask*weight)
    return (bid+ask)/2,(ask-bid)*.25+.015,'same_day_adjacent_strike_interpolation'


def prepare_paths(tenors=TENORS):
    started=time.monotonic()
    cash=load_cash(); cash=cash.loc[START:END]
    dates=cash.index; masks=entry_calendar(dates)
    archive=SPXSurfaceArchive(DATA_ROOT,cache_size=1)
    fingerprints={}; cycles={t:[] for t in tenors}; omissions=[]; repairs=[]
    entry_days=np.flatnonzero(masks.any(axis=1))
    for number,i in enumerate(entry_days):
        day=dates[i]
        frame=read_snapshot(archive.path_for(day),fingerprints,day)
        puts=frame.loc[frame.option_type.eq('put')]
        expiries=pd.DatetimeIndex(puts.expiration_date.unique()).sort_values()
        expiries=expiries[(expiries>day)&(expiries.weekday<5)]
        expiries=expiries[(expiries>dates[-1])|expiries.isin(dates)]
        for tenor in tenors:
            tolerance=3 if tenor<=14 else 7
            eligible=expiries[np.abs((expiries-day).days-tenor)<=tolerance]
            if not len(eligible):
                omissions.append(dict(entry_date=str(day.date()),tenor=tenor,reason='No listed weekday PM expiry within tolerance'))
                continue
            expiry=min(eligible,key=lambda e:(abs((e-day).days-tenor),-(e-day).days))
            chain=puts.loc[puts.expiration_date.eq(expiry)&puts.strike.gt(0)].sort_values(['strike','option_symbol'])
            if chain.empty or chain.strike.duplicated().any() or chain.option_symbol.duplicated().any():
                omissions.append(dict(entry_date=str(day.date()),tenor=tenor,reason='Ambiguous or missing contract grid'))
                continue
            spot=float(cash.iloc[i]); target=spot*RATIOS/100
            selection=chain.iloc[nearest_indices(chain.strike.to_numpy(float),target)]
            end=min(int(dates.searchsorted(expiry)),len(dates)-1)
            shape=(end-i+1,len(RATIOS))
            cycles[tenor].append(Cycle(tenor,int(i),end,expiry,int((expiry-day).days),spot,masks[i].copy(),
                selection.strike.to_numpy(float),selection.option_symbol.to_numpy(str),np.abs(selection.strike.to_numpy(float)-target)/spot,
                np.full(shape,np.nan),np.full(shape,np.nan),np.zeros(shape,dtype=np.uint8),expiry<=dates[-1]))
        if (number+1)%100==0:
            print(f'Entry pass: {number+1}/{len(entry_days)} snapshots ({time.monotonic()-started:.0f}s)',flush=True)
    all_cycles=sorted([c for group in cycles.values() for c in group],key=lambda c:c.entry)
    next_cycle=0; active=[]
    for i,day in enumerate(dates):
        active=[c for c in active if c.end>=i]
        while next_cycle<len(all_cycles) and all_cycles[next_cycle].entry==i:
            active.append(all_cycles[next_cycle]); next_cycle+=1
        if not active:
            continue
        frame=read_snapshot(archive.path_for(day),fingerprints,day)
        puts=frame.loc[frame.option_type.eq('put')].copy()
        duplicates=puts.option_symbol.duplicated(keep=False)
        puts=puts.loc[~duplicates].set_index('option_symbol',drop=False)
        wanted=np.concatenate([c.symbols for c in active]); locations=puts.index.get_indexer(wanted)
        present=locations>=0
        values=np.full((len(wanted),3),np.nan)
        if present.any():
            values[present]=puts.iloc[locations[present]][['bid','ask','strike']].to_numpy(float)
        bid,ask,strike=values.T
        good=valid_quotes(bid,ask,strike)
        mids=np.where(good,(bid+ask)/2,np.nan); costs=np.where(good,(ask-bid)*.25+.015,np.nan)
        qualities=good.astype(np.uint8)
        offset=0; fit_cache={}; repaired={}
        for cycle in active:
            n=len(RATIOS); k=i-cycle.entry
            if cycle.settled and i==cycle.end:
                cycle.mid[k]=np.maximum(cycle.strikes-float(cash.iloc[i]),0)
                cycle.cost[k]=0;cycle.quality[k]=1
                offset+=n;continue
            for j in np.flatnonzero(~good[offset:offset+n]):
                symbol=cycle.symbols[j]
                if symbol not in repaired:
                    chain=frame.loc[frame.expiration_date.eq(cycle.expiry)]
                    value=estimate_quote(chain,cycle.strikes[j],fit_cache,cycle.expiry)
                    repaired[symbol]=value
                    repairs.append(dict(date=str(day.date()),symbol=symbol,strike=float(cycle.strikes[j]),expiry=str(cycle.expiry.date()),
                        method=value[2] if value else 'unresolved',estimated_mid=value[0] if value else None))
                value=repaired[symbol]
                if value:
                    mids[offset+j],costs[offset+j]=value[:2];qualities[offset+j]=2
            cycle.mid[k]=mids[offset:offset+n]
            cycle.cost[k]=costs[offset:offset+n]
            cycle.quality[k]=qualities[offset:offset+n]
            offset+=n
        if i%250==0:
            print(f'Mark pass: {i+1}/{len(dates)} sessions ({time.monotonic()-started:.0f}s)',flush=True)
    fingerprint=hashlib.sha256(''.join(k+v for k,v in sorted(fingerprints.items())).encode()).hexdigest()
    report=dict(quote_source_fingerprint=fingerprint,source_snapshots=len(fingerprints),cycles=len(all_cycles),
        path_values=sum(c.mid.size for c in all_cycles),repair_records=len(repairs),unresolved_records=sum(r['method']=='unresolved' for r in repairs),
        preparation_seconds=time.monotonic()-started)
    return dates,cash,masks,cycles,report,pd.DataFrame(omissions),pd.DataFrame(repairs)
