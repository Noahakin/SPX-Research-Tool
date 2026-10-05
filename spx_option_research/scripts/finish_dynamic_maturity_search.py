"""Retain policy decisions, matched fixed controls and independent reconciliations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dynamic_maturity_engine import simulate, profiles
from dynamic_maturity_scores import score_rules, universes
from maturity_grid_data import CADENCES, CADENCE_DAYS, SIZINGS, RATIOS
from maturity_grid_engine import INITIAL
from run_dynamic_maturity_search import OUT, policy_id, write_json


def independent_ledger_check(data, rows, ledger, curves):
    checks = []
    for strategy, trades in ledger.groupby('strategy_id',sort=False):
        params = rows.loc[strategy]
        rebuilt = np.full(len(data.dates),INITIAL)
        positions = []
        credit_error = exit_error = 0.
        for t in trades.itertuples():
            cycle = data.cycles[(int(t.entry_row),int(t.tenor_index))]
            entry, exit_day = int(t.entry), int(t.exit)
            offset = exit_day-entry
            hi = int(np.flatnonzero(cycle.strikes==t.short_strike)[0])
            lo = int(np.flatnonzero(cycle.strikes==t.long_strike)[0])
            raw = cycle.mid[:offset+1,hi]-cycle.mid[:offset+1,lo]
            friction = cycle.cost[:offset+1,hi]+cycle.cost[:offset+1,lo]
            width = t.short_strike-t.long_strike
            half_spread = 2*np.maximum(friction-.03,0)
            feasible = np.isfinite(raw)&(raw+half_spread>=-.011)&(raw-half_spread<=width+.011)
            bad = ~feasible|(cycle.quality[:offset+1,hi]==0)|(cycle.quality[:offset+1,lo]==0)
            marked = raw.copy()
            for day in range(len(marked)):
                if bad[day]:
                    marked[day] = marked[day-1] if day else 0.
            credit = (raw[0]-friction[0])*100
            credit_error = max(credit_error,abs(credit-t.entry_credit))
            per_contract = credit-marked*100
            per_contract[-1] = credit-(marked[-1]+np.nan_to_num(friction[-1],nan=0))*100
            pnl = t.quantity*per_contract
            exit_error = max(exit_error,abs(pnl[-1]-t.realized_pnl))
            rebuilt[entry:exit_day+1] += pnl
            rebuilt[exit_day+1:] += pnl[-1]
            observed = (cycle.quality[:offset+1,hi]==1)&(cycle.quality[:offset+1,lo]==1)
            supported = observed&feasible&(raw>=-.011)&(raw<=width+.011)
            reached = supported&(credit-(raw+friction)*100>=.25*credit)
            reached[0] = False
            assert not reached[:-1].any(), (strategy,'missed earlier executable 25% exit')
            assert exit_day==cycle.end or reached[-1], (strategy,'unsupported early exit')
            assert observed[0] and not bad[0], (strategy,'unsupported entry')
            positions.append(dict(entry=entry,exit=exit_day,quantity=t.quantity,
                risk=(width*100-credit)*t.quantity,notional=cycle.spot*100*t.quantity,
                unit_risk=width*100-credit,spot=cycle.spot,dte=cycle.dte,entry_equity=t.entry_equity))
        actual_curve = curves[strategy].to_numpy()
        nav_error = float(np.max(np.abs(rebuilt[data.start:]-actual_curve)))
        assert max(nav_error,credit_error,exit_error)<1e-5, (strategy,nav_error,credit_error,exit_error)
        quantity_error = equity_error = 0.
        for position in positions:
            entry = position['entry']
            prior = INITIAL if entry==0 else rebuilt[entry-1]
            held = [p for p in positions if p['entry']<entry<p['exit']]
            serial = params.position_mode=='one_at_a_time'
            if serial:
                assert not held, (strategy,'serial overlap')
            tranches = 1 if serial else max(1,int(np.ceil(position['dte']/CADENCE_DAYS[CADENCES.index(params.cadence)])))
            active_risk = sum(p['risk'] for p in held)
            active_notional = sum(p['notional'] for p in held)
            if params.sizing=='risk_5pct_initial':
                quantity = min(min(INITIAL*.05/tranches,max(INITIAL*.05-active_risk,0))/position['unit_risk'],
                    max(INITIAL*5-active_notional,0)/(position['spot']*100))
            else:
                quantity = min(max(prior,0)/tranches,max(prior-active_notional,0))/(position['spot']*100)
            quantity_error = max(quantity_error,abs(quantity-position['quantity']))
            equity_error = max(equity_error,abs(prior-position['entry_equity']))
        assert quantity_error<1e-8 and equity_error<1e-5, (strategy,quantity_error,equity_error)
        nav = rebuilt[data.start:]
        returns = nav/np.r_[INITIAL,nav[:-1]]-1
        sd = returns.std(ddof=1)
        metrics = dict(cagr=(nav[-1]/INITIAL)**(365.2425/(data.dates[-1]-data.dates[data.start]).days)-1,
            vol=sd*np.sqrt(252),sharpe=returns.mean()/sd*np.sqrt(252) if sd>1e-14 else np.nan,
            max_drawdown=float(np.min(nav/np.maximum.accumulate(np.r_[INITIAL,nav])[1:]-1)))
        for name,value in metrics.items():
            assert np.isclose(value,params[name],atol=1e-9,rtol=0,equal_nan=True),(strategy,name)
        checks.append(dict(strategy_id=strategy,trades=len(trades),max_daily_nav_difference=nav_error,
            max_entry_credit_difference=credit_error,max_realized_pnl_difference=exit_error,
            max_quantity_difference=quantity_error,max_prior_equity_difference=equity_error,
            terminal_cash_difference=abs(INITIAL+trades.realized_pnl.sum()-nav[-1]),
            unresolved_marks=int(params.unresolved_marks)))
    return checks


def verify_selectors(data, chosen):
    _, masks = universes(data.catalog)
    wanted = {}
    for key,value in chosen.items():
        if value['row'].get('family')=='fixed_control':
            continue
        wanted.setdefault(int(value['row']['rule_index']),[]).append((key,value))
    checked = 0
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        for ri,rule in enumerate(score_rules(data.features,data.market,data.cash.iloc[data.entries].to_numpy())):
            if ri not in wanted:
                continue
            for key,value in wanted[ri]:
                params = value['row']
                candidates = np.flatnonzero(masks[int(params['universe_index'])])
                for row,entry in enumerate(data.entries):
                    if entry<data.start or not data.masks[entry,CADENCES.index(params['cadence'])]:
                        continue
                    valid = data.features['valid'][row,candidates]&np.isfinite(rule['score'][row,candidates])
                    valid_candidates = candidates[valid]
                    expected = -1
                    best_score = -np.inf
                    if len(valid_candidates):
                        scores = rule['score'][row,valid_candidates]
                        best_score = float(np.max(scores))
                        if not params['positive_only'] or best_score>0:
                            expected = int(valid_candidates[np.flatnonzero(scores==best_score)[0]])
                    assert value['selections'][row]==expected,(key,str(data.dates[entry]),'ranking mismatch')
                    assert value['candidate_counts'][row]==len(valid_candidates),(key,'candidate count')
                    assert np.isclose(value['entry_scores'][row],best_score,atol=1e-10,rtol=0),(key,'score mismatch')
                    checked += 1
    return checked


def finish(data,state,output=OUT):
    output = Path(output)
    assert state['completed_rules']==88
    rows = sum(pq.read_metadata(path).num_rows for path in (output/'grid').glob('*.parquet'))
    assert rows==state['protocol']['expected_combinations']
    chosen = {}
    leader_rows = []
    for group,value in state['leaders'].items():
        key = policy_id(value['row'])
        chosen[key] = value
        leader_rows.append(dict(strategy_id=key,ranking_basis=group[0],**value['row']))
    leaders = pd.DataFrame(leader_rows)
    for basis in ('validation','historical'):
        metric = 'validation_score' if basis=='validation' else 'sharpe'
        leaders.loc[leaders.ranking_basis.eq(basis)].sort_values(metric,ascending=False).to_csv(output/f'{basis.title()} leaders by family.csv',index=False)
    global_rows = []
    for basis in ('validation','historical'):
        metric = 'validation_score' if basis=='validation' else 'sharpe'
        global_rows.append(leaders.loc[leaders.ranking_basis.eq(basis)].sort_values(metric,ascending=False).groupby(['sizing','position_mode'],sort=False).head(1))
    globals_frame = pd.concat(global_rows)
    globals_frame.to_csv(output/'Overall leaders.csv',index=False)
    curve_dict = {}
    ledger_rows = []
    decision_rows = []
    metric_rows = []
    replay_checks = []
    for strategy,value in chosen.items():
        r = value['row']
        profile = profiles(1)
        profile = profile.loc[(profile.cadence==r['cadence'])&(profile.sizing==r['sizing'])&(profile.position_mode==r['position_mode'])]
        curve,meta,trades = simulate(data,value['selections'][None,:],profile,True,
            value['entry_scores'][None,:],value['candidate_counts'][None,:])
        difference = float(np.max(np.abs(curve[0]-value['curve'])))
        assert difference<1e-5,(strategy,'single-policy replay')
        replay_checks.append(dict(strategy_id=strategy,max_difference=difference))
        curve_dict[strategy] = curve[0,data.start:]
        metric_rows.append(dict(strategy_id=strategy,**r))
        entered = {t['entry'] for t in trades}
        for t in trades:
            t.pop('index')
            t['strategy_id'] = strategy
            ledger_rows.append(t)
        for row,entry in enumerate(data.entries):
            if entry<data.start or not data.masks[entry,CADENCES.index(r['cadence'])]:
                continue
            candidate = int(value['selections'][row])
            count = int(value['candidate_counts'][row])
            action = 'entered' if entry in entered else ('no_scorable_candidates' if count==0 else 'score_not_positive' if candidate<0 else 'occupied_or_capacity')
            decision_rows.append(dict(strategy_id=strategy,entry_date=str(data.dates[entry].date()),candidate=candidate,
                score=float(value['entry_scores'][row]) if np.isfinite(value['entry_scores'][row]) else np.nan,
                candidates_considered=count,action=action))
    # Controls use the same entry schedules, allocation modes, time window and
    # liquidity rules as the selectors. They are comparisons, not new selectors.
    controls = []
    for target,short,width in [(60,103.,3.),(42,103.,5.),(56,104.,5.),(28,99.,3.)]:
        candidate = int(data.catalog.index[(data.catalog.target_dte==target)&(data.catalog.short_target==short)&(data.catalog.width==width)][0])
        valid = data.features['valid'][:,candidate]
        selection = np.where(valid,candidate,-1)[None,:]
        counts = valid.astype(np.int16)[None,:]
        profile = profiles(1)
        curves,meta,records = simulate(data,selection,profile,True,candidate_counts=counts)
        for index,p in profile.iterrows():
            key = f"fixed_d{target}_s{short:g}_w{width:g}_{p.cadence}_{p.sizing}_{p.position_mode}"
            row = dict(strategy_id=key,rule='fixed_control',family='fixed_control',target_dte=target,
                short_target=short,width=width,**p.to_dict(),**{k:v[index] for k,v in meta.items()})
            controls.append(row)
            metric_rows.append(row)
            curve_dict[key] = curves[index,data.start:]
            chosen[key] = dict(row=row,selections=selection[0])
            for record in records:
                if record['index']==index:
                    t=dict(record);t.pop('index');t['strategy_id']=key;ledger_rows.append(t)
    controls_frame = pd.DataFrame(controls)
    controls_frame.to_csv(output/'Matched fixed controls.csv',index=False)
    curves_frame = pd.DataFrame(curve_dict,index=data.dates[data.start:])
    curves_frame['SPX benchmark'] = INITIAL*data.cash.iloc[data.start:]/data.cash.iloc[data.start]
    curves_frame.to_parquet(output/'Retained daily curves.parquet',compression='zstd')
    ledger = pd.DataFrame(ledger_rows)
    ledger['entry_date'] = [str(data.dates[int(i)].date()) for i in ledger.entry]
    ledger['exit_date'] = [str(data.dates[int(i)].date()) for i in ledger.exit]
    ledger.to_parquet(output/'Retained trade ledger.parquet',index=False,compression='zstd')
    decisions = pd.DataFrame(decision_rows)
    decisions.to_parquet(output/'Entry decision audit.parquet',index=False,compression='zstd')
    metrics = pd.DataFrame(metric_rows).drop_duplicates('strategy_id').set_index('strategy_id')
    metrics.to_csv(output/'Retained strategy metrics.csv')
    daily_returns = curves_frame/curves_frame.shift(1).fillna(INITIAL)-1
    daily_returns.groupby(daily_returns.index.year).apply(lambda f:(1+f).prod()-1).to_csv(output/'Calendar year returns.csv',index_label='year')
    checks = independent_ledger_check(data,metrics,ledger,curves_frame)
    selection_checks = verify_selectors(data,chosen)
    validation = dict(status='passed',grid_rows=rows,grid_parts=len(list((output/'grid').glob('*.parquet'))),
        retained_strategies=len(metrics),retained_trades=len(ledger),scheduled_entry_decisions_checked=selection_checks,
        max_daily_nav_difference=max(c['max_daily_nav_difference'] for c in checks),
        method='Single-policy replays; independent position-by-position daily NAV, realized cashflows, prior-close allocation and first executable 25%-profit exits; direct entry-only re-ranking of every retained dynamic policy at every scheduled entry. Accounting checks do not erase separately disclosed data-quality flags.',
        independent_checks=checks,replay_checks=replay_checks,
        audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output/'Validation.json',validation)
    protocol=state['protocol'];protocol.update(status='passed',retained_strategies=len(metrics),retained_trades=len(ledger))
    write_json(output/'Protocol.json',protocol)
    summary = ledger.groupby(['strategy_id','target_dte','short_target','width'],as_index=False).agg(
        trades=('quantity','size'),mean_actual_dte=('actual_dte','mean'),realized_pnl=('realized_pnl','sum'))
    summary.to_csv(output/'Selection distribution.csv',index=False)
    print(f"Dynamic study passed: {rows:,} policy configurations; {len(metrics)} retained strategies; {len(ledger):,} audited trades; {selection_checks:,} entry decisions re-ranked.",flush=True)
    return globals_frame,metrics,ledger
