"""Massive grid over genuine, entry-by-entry SPX spread-selection policies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dynamic_maturity_scores import score_rules, choose, universes
from dynamic_maturity_engine import profiles, simulate
from run_maturity_profit_grid import safe_json

OUT = Path(__file__).resolve().parents[1]/'results/dynamic_maturity_search'


def write_json(path, value):
    path.write_text(json.dumps(safe_json(value),indent=2),encoding='utf-8')


def policy_id(row):
    return f"r{int(row['rule_index']):03d}_u{int(row['universe_index']):02d}_g{int(row['positive_only'])}_{row['cadence']}_{row['sizing']}_{row['position_mode']}"


def run(data, state=None, output=OUT, limit=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'grid').mkdir(exist_ok=True)
    if state is None:
        state = {}
    universe_rows, universe_masks = universes(data.catalog)
    selector_rows = pd.DataFrame([{**r,'universe_index':ui,'positive_only':positive}
        for ui,r in enumerate(universe_rows) for positive in (False,True)])
    profile = profiles(len(selector_rows))
    base_parameters = pd.concat([profile,selector_rows.iloc[profile.selector_index.to_numpy()].reset_index(drop=True)],axis=1)
    total = 88*len(profile)
    n = len(data.dates)-data.start
    state.setdefault('leaders',{})
    state.setdefault('rule_definitions',[])
    state.setdefault('completed_rules',0)
    state.setdefault('eligible_historical',0)
    state.setdefault('eligible_validation',0)
    state.setdefault('started',time.monotonic())
    protocol = dict(status='running',expected_combinations=total,rule_count=88,universe_count=len(universe_rows),
        candidate_geometries_per_entry=len(data.catalog),entry_opportunities=len(data.entries),
        evaluation_start=str(data.dates[data.start].date()),evaluation_end=str(data.dates[-1].date()),
        train_end=str(data.dates[data.start+int(n*.6)-1].date()),validation_end=str(data.dates[data.start+int(n*.8)-1].date()),
        profit_target=.25,
        objective='At each scheduled entry choose one spread jointly across available short strikes, widths and expirations, using only that entry snapshot and prior history.',
        scores='72 lognormal physical-payoff specifications: RV 21/63/126, volatility scales 0.8/1/1.2, annual drift 0/5%, edge/max-loss or expected payoff Sharpe, each with/without annualization; 4 credit-based scores; 8 own-history short/mean-leg IV/RV z-scores; 4 own-history whole-spread edge z-scores.',
        score_caveat='Terminal expected-payoff scores are entry proxies, not a model of the expected first-passage 25%-profit exit. Historical z-scores measure relative richness, not guaranteed positive economic edge. Credit scores reward premium without estimating expected loss.',
        data_timing='RV and drift features use cash closes strictly before the entry session. IV/quotes use the entry EOD snapshot. Own-history scores shift by one weekly observation and require 80% of their 52/104-week window. No realized candidate P&L, exit date or future quote completeness can enter the selector.',
        universe='16 maturity targets 3-180 DTE; short strikes 90-110% of cash SPX in 0.5-point steps; widths 1/2/3/4/5/7.5/10 percentage points. 72 combinations of maturity, strike and width bounds. Listed weekday PM expiries nearest each target, with the prior grid tolerances. Not every listed calendar day is a separate target.',
        ties='Stable order: lower target DTE, then lower short-strike target, then narrower target width. Different target combinations can map to the same listed contracts.',
        score_gate='Compare always choosing the highest eligible score with requiring that highest score to be positive. Negative scores can be chosen by the unrestricted policy.',
        liquidity='Both legs must have valid observed quotes; net credit and maximum loss must be positive; present closing slippage/commissions cannot exceed 12.5% of net credit (half the 25% profit target).',
        entries='Weekly, every two weeks, every four weeks, monthly third-Friday anchors, holiday-adjusted. There is no immediate off-schedule reentry after a profit exit.',
        overlap='One-at-a-time policies stay flat until the next scheduled entry after closing. Ladder policies allow overlaps and divide each new allocation by ceil(actual DTE / cadence days), enforcing aggregate capacity.',
        sizing='5% of initial capital maximum-loss budget with a 500% initial-capital entry-basis notional cap, or 100% of prior-close equity entry-basis notional. Serial policies allocate one tranche; ladders divide across anticipated overlaps. Existing holdings retain their quantities; fractional contracts are allowed.',
        accounting='Initial capital $1m; actual daily leg midpoints while held. Entry/early exit pay 25% of each full bid-ask spread plus $1.50/leg. A 25% profit trigger uses net executable P&L after all costs, never on entry day. PM expiry settles to cash intrinsic; remaining positions close at the common endpoint.',
        quality='Same-day estimated marks cannot authorize entries or profit exits. Unresolved held marks disqualify historical rankings. Validation eligibility separately checks only unresolved marks through validation end, so later quality cannot change that choice.',
        ranking='Validation: at least 80% scored candidate coverage, 40 entries and no unresolved held marks through validation end; positive training and validation CAGR. Rank validation Sharpe minus 0.1*absolute(train-validation Sharpe). Historical: separately rank full-period Sharpe after full-period coverage/trade/data screens.',
        limitations='Previously studied archive; retrospective grid with selection bias, not a fresh untouched holdout. Zero cash interest/risk-free rate; no SPX stock position, dividends, financing, taxes or market impact. EOD execution is a snapshot assumption, not a next-session-fill test.',
        storage='Entry features and future paths remain in RAM. Exhaustive scalar summaries use Float32 compressed Parquet; shortlisted simulations and metrics retain Float64.',
        source_audit=data.audit,
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*[Path(__file__).with_name(name) for name in ['dynamic_maturity_data.py','dynamic_maturity_scores.py','dynamic_maturity_engine.py']]]})
    state['protocol'] = protocol
    write_json(output/'Protocol.json',protocol)
    data.catalog.to_csv(output/'Candidate target catalog.csv',index=False)
    data.market.to_csv(output/'Prior market features.csv',index_label='entry_date')
    print(f"Dynamic policy grid: {total:,} portfolios, {len(data.catalog):,} candidate targets per entry; evaluation {protocol['evaluation_start']} to {protocol['evaluation_end']}",flush=True)
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        for rule_index, rule in enumerate(score_rules(data.features,data.market,data.cash.iloc[data.entries].to_numpy())):
            if rule_index < state['completed_rules']:
                continue
            if limit is not None and rule_index >= limit:
                break
            score = rule['score']
            selections, entry_scores, counts = choose(score,data.features['valid'],universe_masks)
            curves, metrics, _ = simulate(data,selections,profile,scores=entry_scores,candidate_counts=counts)
            frame = base_parameters.copy()
            metadata = {k:v for k,v in rule.items() if k!='score'}
            frame['rule_index'] = rule_index
            for key,value in metadata.items():
                frame[key] = value
            for key,value in metrics.items():
                frame[key] = value
            frame['validation_score'] = frame.validation_sharpe-.1*(frame.validation_sharpe-frame.train_sharpe).abs()
            frame['eligible_validation'] = (frame.selection_entry_coverage.ge(.8)&frame.selection_trades.ge(40)&frame.selection_unresolved.eq(0)&frame.train_cagr.gt(0)&frame.validation_cagr.gt(0)&np.isfinite(frame.validation_sharpe))
            frame['eligible_historical'] = (frame.entry_coverage.ge(.8)&frame.trades.ge(50)&frame.unresolved_marks.eq(0)&~frame.insolvent&frame.train_cagr.gt(0)&frame.validation_cagr.gt(0)&np.isfinite(frame.sharpe))
            compact = frame.drop(columns=['selector_index','cadence_index','sizing_index','mode_index','wins','profit_hits','available_entries','selection_available_entries','selection_planned_entries'])
            compact = compact.copy()
            for column in compact.select_dtypes(include=['float64']).columns:
                compact[column] = compact[column].astype('float32')
            buffer = pa.BufferOutputStream()
            pq.write_table(pa.Table.from_pandas(compact,preserve_index=False),buffer,compression='zstd',compression_level=12)
            blob = buffer.getvalue()
            if shutil.disk_usage(output).free < len(blob)+5*1024*1024:
                raise RuntimeError('Insufficient disk space for the next compact grid part; completed rules and the live in-memory state are preserved.')
            (output/'grid'/f'rule_{rule_index:03d}.parquet').write_bytes(blob.to_pybytes())
            for basis,metric in [('validation','validation_score'),('historical','sharpe')]:
                eligible = frame.loc[frame[f'eligible_{basis}']]
                for group_key, group in eligible.groupby(['sizing','position_mode','family']):
                    best = group.nlargest(1,metric).iloc[0]
                    key = (basis,*group_key)
                    if key not in state['leaders'] or best[metric]>state['leaders'][key]['row'][metric]:
                        si = int(best.selector_index)
                        state['leaders'][key] = dict(row=best.to_dict(),curve=curves[int(best.name)].copy(),
                            selections=selections[si].copy(),entry_scores=entry_scores[si].copy(),candidate_counts=counts[si].copy())
            state['rule_definitions'].append(dict(rule_index=rule_index,**metadata))
            state['completed_rules'] = rule_index+1
            state['eligible_historical'] += int(frame.eligible_historical.sum())
            state['eligible_validation'] += int(frame.eligible_validation.sum())
            progress = dict(completed_rules=state['completed_rules'],completed_combinations=state['completed_rules']*len(profile),
                expected_combinations=total,eligible_historical=state['eligible_historical'],eligible_validation=state['eligible_validation'],
                elapsed_seconds=time.monotonic()-state['started'],free_bytes=shutil.disk_usage(output).free)
            write_json(output/'Progress.json',progress)
            print(f"Dynamic grid {progress['completed_combinations']:,}/{total:,}: {rule['rule']} ({progress['elapsed_seconds']:.0f}s; {progress['free_bytes']/1048576:.1f} MiB free)",flush=True)
    if state['completed_rules'] == 88:
        protocol.update(status='grid_complete',completed_combinations=total,
            eligible_historical=state['eligible_historical'],eligible_validation=state['eligible_validation'],
            elapsed_seconds=time.monotonic()-state['started'])
        write_json(output/'Protocol.json',protocol)
        pd.DataFrame(state['rule_definitions']).to_csv(output/'Scoring rule definitions.csv',index=False)
    return state
