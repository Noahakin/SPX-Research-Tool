"""Compare each historical dynamic leader with fixed candidates in its own universe."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from dynamic_maturity_scores import score_rules, universes
from dynamic_maturity_engine import simulate, profiles
from finish_dynamic_maturity_search import independent_ledger_check
from run_dynamic_maturity_search import OUT, write_json


def compare(data,output=OUT):
    leaders=pd.read_csv(output/'Overall leaders.csv')
    leaders=leaders.loc[leaders.ranking_basis.eq('historical')]
    _,masks=universes(data.catalog)
    curves=pd.read_parquet(output/'Retained daily curves.parquet')
    ledger=pd.read_parquet(output/'Retained trade ledger.parquet')
    metrics=pd.read_csv(output/'Retained strategy metrics.csv').set_index('strategy_id')
    grid=[];winners=[];new_ledger=[];new_metrics=[]
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        for ri,rule in enumerate(score_rules(data.features,data.market,data.cash.iloc[data.entries].to_numpy())):
            group=leaders.loc[leaders.rule_index.eq(ri)]
            for leader in group.itertuples():
                candidates=np.flatnonzero(masks[int(leader.universe_index)])
                base=profiles(1)
                base=base.loc[base.cadence.eq(leader.cadence)&base.sizing.eq(leader.sizing)&base.position_mode.eq(leader.position_mode)].iloc[0]
                best=None;diagnostic=None
                for start in range(0,len(candidates),128):
                    ids=candidates[start:start+128]
                    scores=rule['score'][:,ids].T
                    available=data.features['valid'][:,ids].T&np.isfinite(scores)
                    allowed=available&((scores>0) if leader.positive_only else True)
                    selections=np.where(allowed,ids[:,None],-1)
                    profile=pd.DataFrame([base.to_dict() for _ in ids]);profile['selector_index']=np.arange(len(ids))
                    nav,meta,_=simulate(data,selections,profile,scores=scores,candidate_counts=available.astype(np.int16))
                    frame=data.catalog.iloc[ids].reset_index().rename(columns={'index':'candidate'})
                    for key,value in meta.items():frame[key]=value
                    frame['dynamic_strategy_id']=leader.strategy_id
                    frame['eligible']=(frame.entry_coverage.ge(.8)&frame.trades.ge(50)&frame.unresolved_marks.eq(0)&~frame.insolvent&frame.train_cagr.gt(0)&frame.validation_cagr.gt(0)&np.isfinite(frame.sharpe))
                    grid.append(frame)
                    diagnostic_rows=frame.loc[frame.trades.ge(20)&frame.unresolved_marks.eq(0)&~frame.insolvent&np.isfinite(frame.sharpe)]
                    if not diagnostic_rows.empty:
                        d=diagnostic_rows.nlargest(1,'sharpe').iloc[0]
                        if diagnostic is None or d.sharpe>diagnostic['row']['sharpe']:
                            j=int(d.name)
                            diagnostic=dict(row=d.to_dict(),selection=selections[j].copy(),scores=scores[j].copy(),
                                counts=available[j].astype(np.int16),curve=nav[j].copy())
                    eligible=frame.loc[frame.eligible]
                    if not eligible.empty:
                        r=eligible.nlargest(1,'sharpe').iloc[0]
                        if best is None or r.sharpe>best['row']['sharpe']:
                            j=int(r.name)
                            best=dict(row=r.to_dict(),selection=selections[j].copy(),scores=scores[j].copy(),
                                counts=available[j].astype(np.int16),curve=nav[j].copy())
                if best is None:
                    best=diagnostic
                if best is None:
                    raise AssertionError(f'No supported fixed diagnostic with at least 20 trades for {leader.strategy_id}')
                r=best['row'];candidate=int(r['candidate'])
                key='within_universe_fixed_for_'+leader.strategy_id
                profile=pd.DataFrame([base.to_dict()]);profile['selector_index']=0
                nav,meta,trades=simulate(data,best['selection'][None,:],profile,True,best['scores'][None,:],best['counts'][None,:])
                assert np.max(np.abs(nav[0]-best['curve']))<1e-5
                row={**r,**base.to_dict(),'strategy_id':key,'family':'fixed_control','rule':'fixed_geometry_same_score_gate',
                    'positive_only':bool(leader.positive_only),'dynamic_strategy_id':leader.strategy_id,
                    'fixed_meets_primary_screens':bool(r['eligible'])}
                winners.append(row);new_metrics.append(row);curves[key]=nav[0,data.start:]
                for trade in trades:
                    trade.pop('index');trade['strategy_id']=key
                    trade['entry_date']=str(data.dates[trade['entry']].date());trade['exit_date']=str(data.dates[trade['exit']].date())
                    new_ledger.append(trade)
                print(f"Within-universe control for {leader.strategy_id}: {r['short_target']:g}/{r['short_target']-r['width']:g}, {r['target_dte']:g} DTE, Sharpe {r['sharpe']:.3f}, CAGR {r['cagr']:.2%}; {len(candidates)} fixed candidates",flush=True)
    exhaustive=pd.concat(grid,ignore_index=True)
    for col in exhaustive.select_dtypes(include=['float64']).columns:exhaustive[col]=exhaustive[col].astype('float32')
    exhaustive.to_parquet(output/'Within universe fixed grid.parquet',index=False,compression='zstd')
    pd.DataFrame(winners).to_csv(output/'Within universe fixed winners.csv',index=False)
    extra_ledger=pd.DataFrame(new_ledger)
    extra_metrics=pd.DataFrame(new_metrics).set_index('strategy_id')
    checks=independent_ledger_check(data,extra_metrics,extra_ledger,curves)
    ledger=pd.concat([ledger.loc[~ledger.strategy_id.isin(extra_metrics.index)],extra_ledger],ignore_index=True)
    metrics=pd.concat([metrics.loc[~metrics.index.isin(extra_metrics.index)],extra_metrics])
    ledger.to_parquet(output/'Retained trade ledger.parquet',index=False,compression='zstd')
    curves.to_parquet(output/'Retained daily curves.parquet',compression='zstd')
    metrics.to_csv(output/'Retained strategy metrics.csv')
    daily=curves/curves.shift(1).fillna(1_000_000)-1
    daily.groupby(daily.index.year).apply(lambda f:(1+f).prod()-1).to_csv(output/'Calendar year returns.csv',index_label='year')
    validation=json.loads((output/'Validation.json').read_text())
    validation['independent_checks']=[x for x in validation['independent_checks'] if x['strategy_id'] not in extra_metrics.index]+checks
    validation.update(retained_strategies=len(metrics),retained_trades=len(ledger),additional_fixed_candidate_tests=len(exhaustive),
        within_universe_controls='Highest full-period Sharpe under identical universe bounds, scoring availability, positive-score gate, cadence, sizing and overlap mode. Prefer the identical primary eligibility screens. If none pass, clearly flag a diagnostic comparator requiring at least 20 trades and no unresolved marks instead. Both dynamic and fixed historical leaders use hindsight.',
        fixed_control_code_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    validation['max_daily_nav_difference']=max(x['max_daily_nav_difference'] for x in validation['independent_checks'])
    write_json(output/'Validation.json',validation)
    protocol=json.loads((output/'Protocol.json').read_text())
    protocol.update(additional_fixed_candidate_tests=len(exhaustive),retained_strategies=len(metrics),retained_trades=len(ledger))
    write_json(output/'Protocol.json',protocol)
    return pd.DataFrame(winners)
