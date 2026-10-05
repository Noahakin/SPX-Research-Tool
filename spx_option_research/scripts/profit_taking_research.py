"""Daily, whole-package profit exits followed by immediate fresh entry.

Stores observed contract paths and compact trade records instead of duplicating
every equity curve. Both equity exposures replay the same option trades, with
their own compounded capital. Existing hold-to-expiration data is untouched.
"""
from __future__ import annotations

import argparse
from array import array
import base64
import gzip
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import pandas as pd

from organized_spx_data import SPXSurfaceArchive, load_cash, START, END
from organized_spx_engine import adjacent_estimate, suspect_quotes, very_wide_quotes
from maturity_grid_data import read_snapshot, nearest_indices, estimate_quote
from premium_buffer_hedges import select_buffers
from organized_spx_parity import fit_parity, parity_estimate

PROJECT = Path(__file__).resolve().parents[1]
SITE = PROJECT.parent / 'SPX Research Interactive'
INITIAL = 1_000_000.0
TARGETS = (.25, .50, .75)
TENORS = (3, 7, 14, 21, 28, 42, 56)
RATIOS = np.arange(80, 111)
PRICE_GROUP = 2048
TRADE_GROUP = 48
EPOCH = pd.Timestamp('1970-01-01')


def read_catalog(path=SITE / 'catalog.js'):
    text = path.read_text(encoding='utf-8')
    return json.loads(text.removeprefix('window.SPX_CATALOG=').strip().removesuffix(';'))


def choose_expiry(expiries, day, tenor, calendar):
    """Closest PM maturity within disclosed tolerance; equal distance goes longer."""
    eligible = pd.DatetimeIndex(expiries).sort_values()
    dte = (eligible - day).days
    valid = (dte > 0) & (eligible.weekday < 5) & ((eligible > calendar[-1]) | eligible.isin(calendar))
    valid &= ((dte >= 1) & (dte <= 5)) if tenor == 3 else (np.abs(dte - tenor) <= (3 if tenor <= 14 else 7))
    eligible = eligible[valid]
    return min(eligible, key=lambda e: (abs((e-day).days-tenor), -(e-day).days)) if len(eligible) else None


def package_quantities(category, credit, debit, fraction=0., buffer=False):
    if category != 'Both':
        return 0.
    return fraction * credit / debit if buffer else min(1., fraction * credit / debit)


def missing_ask_estimate(chain, strike, observed_bid):
    """Retain an observed positive bid; estimate only its missing ask spread.

    Donors must bracket the strike in the same expiry/snapshot. This is a
    valuation estimate, never an executable two-sided quote.
    """
    if not np.isfinite(observed_bid) or observed_bid<=0:return None
    puts=chain.loc[chain.option_type.eq('put')]
    good=puts.loc[~suspect_quotes(puts)&~very_wide_quotes(puts)].sort_values('strike').drop_duplicates('strike')
    lower,upper=good.loc[good.strike<strike],good.loc[good.strike>strike]
    if lower.empty or upper.empty:return None
    lo,hi=lower.iloc[-1],upper.iloc[0]
    if max(strike-lo.strike,hi.strike-strike)>max(50.,.05*strike):return None
    weight=(strike-lo.strike)/(hi.strike-lo.strike)
    spread=(lo.ask-lo.bid)*(1-weight)+(hi.ask-hi.bid)*weight
    ask=observed_bid+spread;mid=(observed_bid+ask)/2
    if spread<=0 or not lo.bid<=observed_bid<=ask<=hi.ask:return None
    if ask-lo.bid>strike-lo.strike+.5 or hi.bid-observed_bid>hi.strike-strike+.5:return None
    return dict(mid=float(mid),cost=float(.25*spread+.015),lower_strike=float(lo.strike),upper_strike=float(hi.strike),
                lower_bid=float(lo.bid),lower_ask=float(lo.ask),upper_bid=float(hi.bid),upper_ask=float(hi.ask),estimated_ask=float(ask))


class QuoteStore:
    """One shared history per contract; missing unused days remain explicit NaNs."""
    def __init__(self):
        self.lookup = {}
        self.symbols, self.strikes, self.expiries = [], [], []
        self.first, self.last, self.paths = [], [], []

    def identify(self, rows):
        ids = []
        for row in rows.itertuples(index=False):
            symbol = row.option_symbol
            if symbol not in self.lookup:
                self.lookup[symbol] = len(self.symbols)
                self.symbols.append(symbol)
                self.strikes.append(float(row.strike))
                self.expiries.append(pd.Timestamp(row.expiration_date))
                self.first.append(-1); self.last.append(-1); self.paths.append(array('d'))
            i = self.lookup[symbol]
            if self.strikes[i] != row.strike or self.expiries[i] != row.expiration_date:
                raise ValueError(f'Contract identity changed: {symbol}')
            ids.append(i)
        return np.asarray(ids, dtype=np.int32)

    def add(self, ids, day_index, mid, cost, quality):
        for cid, m, c, q in zip(ids, mid, cost, quality):
            cid = int(cid)
            if self.last[cid] == day_index:
                np.testing.assert_allclose(self.paths[cid][-3:], [m,c,q], rtol=0, atol=0)
                continue
            if self.first[cid] < 0:
                self.first[cid] = day_index
            elif day_index > self.last[cid]+1:
                self.paths[cid].extend([np.nan, np.nan, 0.] * (day_index-self.last[cid]-1))
            self.paths[cid].extend([float(m),float(c),float(q)])
            self.last[cid] = day_index


class Snapshot:
    def __init__(self, frame, day, audit, known_repairs):
        self.day, self.frame, self.audit = day, frame, audit
        puts = frame.loc[frame.option_type.eq('put')].copy()
        if puts.option_symbol.duplicated().any():
            raise ValueError(f'Duplicate contracts on {day}')
        self.puts = puts.set_index('option_symbol', drop=False)
        self.puts.index.name = '_symbol'
        self.chains = {expiry: rows for expiry,rows in frame.groupby('expiration_date',sort=False)}
        self.values, self.fit_cache, self.tight_fit_cache = {}, {}, {}
        self.known = known_repairs
        self.bound_flags = set()
        for _, chain in puts.groupby('expiration_date',sort=False):
            chain = chain.sort_values('strike').drop_duplicates('strike')
            mids = (chain.bid.to_numpy()+chain.ask.to_numpy())/2
            diff, width = np.diff(mids), np.diff(chain.strike.to_numpy())
            for j in np.flatnonzero((diff < -1.) | (diff > width+1.)):
                self.bound_flags.update(chain.option_symbol.iloc[[j,j+1]])
        # Most quotes are unexceptional. Avoid constructing a pandas Series
        # every time thousands of strategies request the same observed marks.
        bid,ask,strike=puts.bid.to_numpy(float),puts.ask.to_numpy(float),puts.strike.to_numpy(float)
        normal=np.isfinite(bid)&np.isfinite(ask)&np.isfinite(strike)&(bid>=0)&(ask>0)&(ask>=bid)&(strike>0)
        normal &= ~((ask>=25)&((bid<=.5)|(bid<ask*.02)))
        special=self.bound_flags|{symbol for (date,symbol) in known_repairs if date==day}
        symbols=puts.option_symbol.to_numpy()
        normal &= ~np.isin(symbols,list(special))
        mids=(bid+ask)/2;costs=.25*(ask-bid)+.015
        self.values.update((str(symbols[i]),(float(mids[i]),float(costs[i]),1)) for i in np.flatnonzero(normal))

    def quote(self, symbol, strike, expiry):
        if symbol in self.values:
            return self.values[symbol]
        if symbol not in self.puts.index:
            raw = None
        else:
            raw = self.puts.loc[symbol]
            if raw.strike != strike or raw.expiration_date != expiry:
                raise ValueError(f'Quote identity mismatch: {self.day} {symbol}')
        chain = self.chains.get(expiry)
        replacement = self.known.get((self.day,symbol));evidence={}
        reason = 'previously audited same-day estimate'
        if raw is None:
            bad = True
        else:
            bad = (not np.isfinite([raw.strike,raw.bid,raw.ask]).all() or raw.strike<=0 or raw.bid<0 or raw.ask<=0 or raw.ask<raw.bid
                   or (raw.ask>=25 and (raw.bid<=.5 or raw.bid<raw.ask*.02)))
        if replacement is None and chain is not None and (bad or symbol in self.bound_flags):
            estimate = adjacent_estimate(chain.loc[chain.option_type.eq('put')], strike)
            mismatch = False
            if estimate is not None and raw is not None:
                wide = raw.ask>=25 and (raw.bid<.5*raw.ask or raw.ask-raw.bid>max(25.,.30*(raw.bid+raw.ask)/2))
                threshold = .05 if wide else .20
                mismatch = abs((raw.bid+raw.ask)/2-estimate['used_mid']) > max(5.,threshold*max(estimate['used_mid'],1.))
            if bad or mismatch:
                if estimate is not None:
                    replacement = (estimate['used_mid'],.25*(estimate['used_ask']-estimate['used_bid'])+.015)
                    reason = estimate['method']
                else:
                    result = estimate_quote(chain,strike,self.fit_cache,expiry)
                    if result is not None:
                        replacement, reason = result[:2], result[2]
                    else:
                        # Extremely wide but technically valid donor markets
                        # can dominate the unweighted fit diagnostic. Exclude
                        # those donors, retaining every other parity safeguard.
                        if expiry not in self.tight_fit_cache:
                            self.tight_fit_cache[expiry]=fit_parity(chain.loc[~very_wide_quotes(chain)])
                        fitted=self.tight_fit_cache[expiry]
                        estimate=parity_estimate(fitted,strike) if fitted is not None else None
                        if estimate is not None:
                            replacement=(estimate['used_mid'],.25*(estimate['used_ask']-estimate['used_bid'])+.015)
                            reason='Same-day calibrated put-call parity; broad donor markets excluded; valuation only'
        if replacement is None and bad and raw is not None and (not np.isfinite(raw.ask) or raw.ask<=0) and chain is not None:
            evidence=missing_ask_estimate(chain,strike,raw.bid) or {}
            if evidence:
                replacement=(evidence['mid'],evidence['cost'])
                reason='Observed bid with missing ask; same-day same-expiry bracketing bid/ask spread estimate; valuation only'
        if replacement is not None:
            result = (*replacement,2)
            self.audit.append(dict(date=str(self.day.date()),symbol=symbol,reason=reason,
                raw_bid=None if raw is None else float(raw.bid),raw_ask=None if raw is None else float(raw.ask),
                used_mid=float(result[0]),used_cost=float(result[1]),**evidence))
        elif bad:
            result = (np.nan,np.nan,0)
        else:
            result = (float((raw.bid+raw.ask)/2),float(.25*(raw.ask-raw.bid)+.015),1)
        self.values[symbol] = result
        return result

    def quotes(self, store, ids):
        return np.array([self.quote(store.symbols[c],store.strikes[c],store.expiries[c]) for c in ids],dtype=float)


def entry_packages(snapshot, store, spot, calendar, records):
    """All decisions below use the current snapshot and retained hedge rules."""
    n=len(records)
    contracts=np.full((n,4),-1,np.int32); quantities=np.zeros((n,4)); credit=np.zeros(n)
    eligible=np.zeros(n,bool); entry_mid=np.zeros((n,4)); entry_cost=np.zeros((n,4)); expiries=np.full(n,-1,np.int32)
    for tenor in TENORS:
        indices=np.array([i for i,r in enumerate(records) if r['days']==tenor],dtype=int)
        if not len(indices):continue
        expiry=choose_expiry(snapshot.chains.keys(),snapshot.day,tenor,calendar)
        if expiry is None:continue
        chain=snapshot.puts.loc[snapshot.puts.expiration_date.eq(expiry)&snapshot.puts.strike.gt(0)].sort_values(['strike','option_symbol'])
        if chain.empty or chain.strike.duplicated().any():continue
        selected=chain.iloc[nearest_indices(chain.strike.to_numpy(),spot*RATIOS/100)]
        ids=store.identify(selected.reset_index(drop=True))
        quoted=snapshot.quotes(store,ids);mid,cost,quality=quoted.T
        k=selected.strike.to_numpy(float)
        valid=(quality==1)&(np.abs(k/spot-RATIOS/100)<=.005+1e-12)
        raw_mid=(selected.bid.to_numpy()+selected.ask.to_numpy())/2
        raw_cost=.25*(selected.ask.to_numpy()-selected.bid.to_numpy())+.015
        cycle=dict(k=k,mid=raw_mid,friction=raw_cost,valid=valid,spot=spot)
        buffer_rows=[i for i in indices if records[i]['dynamic']]
        buffer_lookup={i:j for j,i in enumerate(buffer_rows)}
        buffer_choices=None
        if buffer_rows:
            budgets=[]
            for i in buffer_rows:
                r=records[i];hi=r['primary']-80;lo=hi-r['width'];spread=r['width']>0
                amount=raw_mid[hi]-raw_cost[hi]-(raw_mid[lo]+raw_cost[lo] if spread else 0.)
                budgets.append(r['premium']*max(amount,0.))
            buffer_choices=select_buffers(cycle,np.asarray(budgets),minimum_quantity=1)
        for i in indices:
            r=records[i];hi=r['primary']-80;lo=hi-r['width'];spread=r['width']>0
            if not valid[hi] or (spread and (not valid[lo] or k[hi]<=k[lo])):continue
            base_mid=mid[hi]-(mid[lo] if spread else 0.)
            if base_mid<0 or (spread and base_mid>k[hi]-k[lo]+1e-10):continue
            base_cost=cost[hi]+(cost[lo] if spread else 0.)
            side=1. if r['category']=='Put buying' else -1.
            legs=[hi,lo if spread else -1,-1,-1];qty=[side,-side if spread else 0.,0.,0.]
            base_credit=base_mid-base_cost
            if side>0 and base_mid+base_cost<=0:continue
            if side<0 and base_credit<=0:continue
            if r['category']=='Both':
                budget=r['premium']*base_credit
                if r['dynamic']:
                    hedge=buffer_choices;j=buffer_lookup[i]
                    if not hedge['traded'][j]:continue
                    hh,hl=int(hedge['high'][j]),int(hedge['low'][j]);q=float(hedge['quantity'][j])
                else:
                    hh=int(r['hedgeHigh'])-80;hl=-1
                    if not valid[hh] or mid[hh]+cost[hh]<=0:continue
                    q=min(1.,budget/(mid[hh]+cost[hh]))
                legs[2:]=[hh,hl];qty[2:]=[q,-q if hl>=0 else 0.]
            active=np.array(legs)>=0
            leg_indices=np.maximum(legs,0)
            quantity=np.asarray(qty)
            safe_mid=np.where(active,mid[leg_indices],0.);safe_cost=np.where(active,cost[leg_indices],0.)
            net_credit=-float(np.sum(quantity*safe_mid+np.abs(quantity)*safe_cost))
            if r['category']!='Put buying' and net_credit<=0:continue
            contracts[i]=np.where(active,ids[leg_indices],-1)
            quantities[i]=quantity
            entry_mid[i]=np.where(active,mid[leg_indices],0.)
            entry_cost[i]=np.where(active,cost[leg_indices],0.)
            credit[i]=net_credit;eligible[i]=True;expiries[i]=(expiry-EPOCH).days
    return dict(contracts=contracts,quantities=quantities,credit=credit,eligible=eligible,
                mid=entry_mid,cost=entry_cost,expiry=expiries)


def simulate(records, dates, cash, frames, known_repairs=None, progress=None, repair_file=None):
    """One live package per strategy. Exit, resize, and re-enter in that order."""
    n=len(records);store=QuoteStore();audit=[]
    targets=np.array([r['profitTarget'] for r in records])
    capital=np.full((2,n),INITIAL);spot_entry=np.full(n,cash[0]);credit=np.zeros(n)
    contracts=np.full((n,4),-1,np.int32);quantities=np.zeros((n,4));expiry=np.full(n,-1,np.int32)
    active=np.zeros(n,bool);entered=np.full(n,-1,np.int32)
    histories=[array('I') for _ in records]
    curves=np.full((2,n,len(dates)),np.nan)
    metrics={key:np.zeros(n,dtype=np.int32) for key in ['trades','profitExits','expiryExits','cashDays','estimatedDays','unavailableEntries']}
    # Entry geometry is identical across targets; select it once per base row.
    base_map={};unique=[];base_indices=[]
    for r in records:
        base=r['baseId']
        if base not in base_map:base_map[base]=len(unique);unique.append(r)
        base_indices.append(base_map[base])
    base_indices=np.asarray(base_indices)
    for day_index,(day,frame) in enumerate(zip(dates,frames)):
        snapshot=Snapshot(frame,day,audit,known_repairs or {})
        flat=~active
        if day_index:capital[1,flat]*=cash[day_index]/cash[day_index-1]
        today=(day-EPOCH).days
        pending=np.flatnonzero(active)
        if len(pending):
            stored_strikes=np.asarray(store.strikes)
            held=contracts[pending];qty=quantities[pending];mids=np.zeros_like(qty);costs=np.zeros_like(qty);quality=np.ones_like(qty)
            expired=expiry[pending]==today
            if np.any(expiry[pending]<today):raise ValueError('Held contract passed expiration without settlement')
            needed=np.unique(held[(held>=0)&~expired[:,None]])
            if len(needed):
                values=snapshot.quotes(store,needed)
                if np.any(values[:,2]==0):
                    bad=needed[values[:,2]==0]
                    if repair_file is None:
                        raise ValueError(f'Unresolved held daily quotes on {day.date()}: {[store.symbols[c] for c in bad[:8]]}')
                    request=repair_file.with_name(repair_file.stem+'-needed.json')
                    request.parent.mkdir(parents=True,exist_ok=True)
                    request.write_text(json.dumps(dict(date=str(day.date()),contracts=[dict(symbol=store.symbols[c],strike=store.strikes[c],expiry=str(store.expiries[c].date())) for c in bad]),indent=2),encoding='utf-8')
                    print(f'PAUSED for same-day quote evidence: {request}; simulation state retained in memory.',flush=True)
                    while np.any(values[:,2]==0):
                        if repair_file.exists():
                            repairs=json.loads(repair_file.read_text(encoding='utf-8'))
                            for row in repairs:
                                if row.get('date')!=str(day.date()) or row.get('symbol') not in {store.symbols[c] for c in bad}:continue
                                mid,cost=float(row['mid']),float(row['cost'])
                                if not np.isfinite([mid,cost]).all() or mid<0 or cost<.015 or not row.get('evidence'):raise ValueError('Invalid documented quote repair')
                                snapshot.values[row['symbol']]=(mid,cost,2)
                                audit.append(row|dict(reason='Documented same-day valuation repair; execution prohibited',used_mid=mid,used_cost=cost))
                            values=snapshot.quotes(store,needed)
                        if np.any(values[:,2]==0):time.sleep(1)
                    print(f'Resumed after documented same-day valuation repair: {day.date()}',flush=True)
                store.add(needed,day_index,*values.T)
                for leg in range(4):
                    use=(held[:,leg]>=0)&~expired
                    locations=np.searchsorted(needed,held[use,leg])
                    mids[use,leg],costs[use,leg],quality[use,leg]=values[locations].T
            for leg in range(4):
                use=expired&(held[:,leg]>=0)
                mids[use,leg]=np.maximum(stored_strikes[held[use,leg]]-cash[day_index],0.)
            marked=credit[pending]+np.sum(qty*mids,axis=1)
            closing=marked-np.sum(np.abs(qty)*costs,axis=1)
            observed=np.all((quality==1)|(qty==0),axis=1)
            # A vertical outside its payoff bounds can be valued but cannot exit.
            bounded=np.ones(len(pending),bool)
            for hi,lo in [(0,1),(2,3)]:
                pair=(held[:,lo]>=0)&(qty[:,lo]!=0)
                width=np.zeros(len(pending));width[pair]=stored_strikes[held[pair,hi]]-stored_strikes[held[pair,lo]]
                spread_mid=mids[:,hi]-mids[:,lo]
                bounded &= ~pair|((spread_mid>=-1e-10)&(spread_mid<=width+1e-10))
            hit=(closing>=np.abs(credit[pending])*targets[pending])&observed&bounded&~expired
            done=hit|expired
            result=np.where(done,closing,marked)
            current=capital[:,pending]*(1+result[None,:]/spot_entry[pending][None,:])
            current[1]+=capital[1,pending]*(cash[day_index]/spot_entry[pending]-1)
            curves[:,pending,day_index]=current
            metrics['estimatedDays'][pending]+=np.any((quality==2)&(qty!=0),axis=1)
            closed=pending[done];capital[:,closed]=current[:,done];active[closed]=False
            metrics['profitExits'][pending[hit]]+=1;metrics['expiryExits'][pending[expired]]+=1
            for sid,reason in zip(closed,np.where(expired[done],2,1)):
                histories[sid][-6]=day_index;histories[sid][-1]=int(reason)
        # Flat portfolios keep cash / continuous SPX until a valid entry exists.
        due=np.flatnonzero(~active)
        if day_index<len(dates)-1 and len(due):
            wanted_bases=np.unique(base_indices[due])
            new=entry_packages(snapshot,store,cash[day_index],dates,[unique[j] for j in wanted_bases])
            reverse={int(base):j for j,base in enumerate(wanted_bases)}
            locations=np.array([reverse[int(base_indices[i])] for i in due])
            allowed=new['eligible'][locations];selected=due[allowed];chosen=locations[allowed]
            metrics['unavailableEntries'][due[~allowed]]+=1
            if len(selected):
                contracts[selected]=new['contracts'][chosen];quantities[selected]=new['quantities'][chosen]
                credit[selected]=new['credit'][chosen];spot_entry[selected]=cash[day_index];expiry[selected]=new['expiry'][chosen]
                active[selected]=True;entered[selected]=day_index;metrics['trades'][selected]+=1
                cost=np.sum(np.abs(quantities[selected])*new['cost'][chosen],axis=1)
                curves[:,selected,day_index]=capital[:,selected]*(1-cost[None,:]/cash[day_index])
                used=np.unique(contracts[selected][contracts[selected]>=0]);values=snapshot.quotes(store,used)
                store.add(used,day_index,*values.T)
                for sid in selected:
                    histories[sid].extend([day_index,len(dates)-1,*[int(c)+1 for c in contracts[sid]],0])
        flat=~active;curves[:,flat,day_index]=capital[:,flat];metrics['cashDays']+=flat
        if not np.isfinite(curves[:,:,day_index]).all():raise ValueError('Nonfinite daily equity')
        if np.any(curves[:,:,day_index]<=0):raise ValueError('Portfolio insolvency needs an explicit handling rule')
        if progress and ((day_index+1)%100==0 or day_index==len(dates)-1):progress(day_index+1,store,metrics)
    return curves,histories,store,metrics,audit


def pack_shard(path, key, parts):
    """Lossless byte shuffle before gzip preserves every original float bit."""
    raw=b''.join(parts)
    if len(raw)%8:raise ValueError('Shard sections must be eight-byte aligned')
    shuffled=np.frombuffer(raw,np.uint8).reshape(-1,8).T.copy().tobytes()
    packed=gzip.compress(shuffled,compresslevel=6,mtime=0)
    encoded=base64.b64encode(packed).decode('ascii')
    path.write_text(f'window.__SPX_CHUNK__("{key}","{encoded}");\n',encoding='ascii')
    return dict(file='profit-data/'+path.name,bytes=len(raw),sha256=hashlib.sha256(shuffled).hexdigest(),encoding='shuffle8')


def export(output, records, curves, histories, store, metrics, audit, dates, cash, fingerprints):
    output.mkdir(parents=True,exist_ok=True)
    chunks={};price_chunks={}
    for start in range(0,len(store.symbols),PRICE_GROUP):
        stop=min(start+PRICE_GROUP,len(store.symbols));key=f'p{start//PRICE_GROUP:03d}'
        descriptors=[];values=array('d');offset=0
        for c in range(start,stop):
            path=store.paths[c]
            descriptors.extend([store.first[c],len(path)//3,offset,store.strikes[c],(store.expiries[c]-EPOCH).days])
            values.extend(path);offset+=len(path)
        # Header and descriptors are doubles so a single byte shuffle suffices.
        header=np.array([1,start,stop-start,len(values)],dtype='<f8')
        price_chunks[key]=pack_shard(output/(key+'.js'),key,[header.tobytes(),np.array(descriptors,dtype='<f8').tobytes(),np.array(values,dtype='<f8').tobytes()])
        price_chunks[key]['kind']='prices'
    catalog_rows=[];reference=[]
    for start in range(0,len(records),TRADE_GROUP):
        stop=min(start+TRADE_GROUP,len(records));key=f't{start//TRADE_GROUP:03d}'
        offsets=[0];values=array('I');deps=set()
        for h in histories[start:stop]:
            values.extend(h);offsets.append(len(values)//7)
            matrix=np.asarray(h,dtype=np.uint32).reshape(-1,7)
            deps.update(f'p{int(c-1)//PRICE_GROUP:03d}' for c in np.unique(matrix[:,2:6]) if c)
        # Integer metadata is exactly represented as doubles and compresses well.
        raw=np.r_[np.array([1,stop-start,len(values)//7],float),np.asarray(offsets,float),np.asarray(values,float)].astype('<f8')
        chunks[key]=pack_shard(output/(key+'.js'),key,[raw.tobytes()])
        chunks[key].update(kind='trades',prices=sorted(deps),strategies=stop-start)
        for slot,i in enumerate(range(start,stop)):
            r=records[i]
            catalog_rows.append(dict(id=r['id'],base=r['baseId'],profitTarget=r['profitTarget'],chunk=key,slot=slot,
                **{name:int(value[i]) for name,value in metrics.items()}))
            points=np.unique(np.r_[np.linspace(0,len(dates)-1,19,dtype=int),np.asarray(histories[i],dtype=np.uint32).reshape(-1,7)[:4,:2].ravel()]).astype(int)
            nav=curves[:,i];returns=nav/np.c_[np.full(2,INITIAL),nav[:,:-1]]-1
            sd=returns.std(axis=1,ddof=1);mean=returns.mean(axis=1)
            years=(dates[-1]-dates[0]).days/365.2425
            peaks=np.maximum.accumulate(np.c_[np.full(2,INITIAL),nav],axis=1)[:,1:]
            stats=dict(cagr=((nav[:,-1]/INITIAL)**(1/years)-1).tolist(),vol=(sd*np.sqrt(252)).tolist(),
                       sharpe=np.divide(mean*np.sqrt(252),sd,out=np.full(2,np.nan),where=sd>1e-14).tolist(),maxDD=(nav/peaks-1).min(axis=1).tolist())
            stats={k:[float(v) if np.isfinite(v) else None for v in a] for k,a in stats.items()}
            reference.append(dict(id=r['id'],indices=points.tolist(),nav=curves[:,i,points].tolist(),stats=stats))
    protocol=dict(version=1,targets=TARGETS,start=str(dates[0].date()),end=str(dates[-1].date()),strategies=len(records),daily_sessions=len(dates),
        entry='At initial close, and immediately after each full-package profit exit or expiry. Retry each following session when a valid replacement is unavailable. No new entry at the final close.',
        expiration='Closest listed PM cash-session expiry to target; 3D within 1–5 days, 7/14D within 3 days, longer tenors within 7 days. Ties prefer longer.',
        trigger='Whole option package net executable P&L, after all entry and closing costs, reaches target times initial net credit (selling/Both) or debit (buying). SPX P&L does not trigger option exits.',
        costs='25% of each full bid/ask spread plus $1.50 per contract per leg, at entry and early exit. PM expiry settles to intrinsic without closing fees.',
        timing='First qualifying EOD snapshot, then fresh strikes and expiration at the same snapshot. One entry per strategy per session. Intraday hits are not observed.',
        marks='Held contracts use audited daily midpoints; expiry uses cash intrinsic. Explicit same-day estimates cannot authorize entry or profit exits. No forward filling.',
        final='An unexpired final position stays marked at the final midpoint, without hypothetical closing costs.',
        spx='Continuous 100% SPX price exposure; options use 100% entry-equity SPX notional. Both sizes reset at each option exit/new entry. No interest, dividends or financing.',
        hedge='Original fixed long-put targets retained; buffers reselect closest affordable protection, prefer 5 points, allow 4 or 3, minimum actual width 3% of entry SPX, use full budget with quantity at least 1.',
        estimates=len(audit),contracts=len(store.symbols),trades=int(metrics['trades'].sum()),source_sha256=fingerprints)
    metadata=dict(version=1,priceGroup=PRICE_GROUP,records=catalog_rows,chunks=chunks,priceChunks=price_chunks,protocol=protocol)
    (output/'metadata.json').write_text(json.dumps(metadata,separators=(',',':')),encoding='utf-8')
    (output/'reference.json.gz').write_bytes(gzip.compress(json.dumps(reference,separators=(',',':')).encode(),mtime=0))
    (output/'quote_estimates.json.gz').write_bytes(gzip.compress(json.dumps(audit,separators=(',',':')).encode(),mtime=0))
    print(f'Exported {len(records):,} variants, {int(metrics["trades"].sum()):,} trades, {len(store.symbols):,} contracts; {sum(p.stat().st_size for p in output.iterdir())/1024**2:.1f} MiB',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=PROJECT.parent/'.publish/profit-taking-stage')
    parser.add_argument('--sessions',type=int)
    parser.add_argument('--limit-bases',type=int)
    parser.add_argument('--repair-file',type=Path,help='Optional documented same-day valuation repairs; unresolved held quotes pause for evidence')
    args=parser.parse_args()
    catalog=read_catalog();records=[]
    bases=catalog['strategies'][:args.limit_bases] if args.limit_bases else catalog['strategies']
    for base in bases:
        for target in TARGETS:
            records.append(base|dict(id=base['id']+f'_tp{round(target*100):02d}',baseId=base['id'],profitTarget=target))
    full_cash=load_cash().loc[START:END]
    if args.sessions:full_cash=full_cash.iloc[:args.sessions]
    dates=full_cash.index;cash=full_cash.to_numpy()
    archive=SPXSurfaceArchive(args.source_root/'04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m',cache_size=1)
    known={}
    audited=args.source_root/'spx_option_research/results/organized_spx_combinations/audited_quotes.parquet'
    if audited.exists():
        q=pd.read_parquet(audited);q=q.loc[q.estimated]
        for r in q.itertuples(index=False):known[(r.snapshot_date,r.option_symbol)]=(r.used_mid,.25*(r.used_ask-r.used_bid)+.015)
    fingerprints={};started=time.monotonic()
    def frames():
        for day in dates:yield read_snapshot(archive.path_for(day),fingerprints,day)
    def progress(number,store,metrics):
        print(f'Backtest {number}/{len(dates)} sessions; {int(metrics["trades"].sum()):,} trades; {len(store.symbols):,} contracts; {time.monotonic()-started:.0f}s',flush=True)
    repair_file=args.repair_file or args.output.parent/'profit-quote-repairs.json'
    result=simulate(records,dates,cash,frames(),known,progress,repair_file)
    export(args.output,records,*result,dates,cash,fingerprints)


if __name__=='__main__':main()
