"""Exhaustive multi-maturity SPX profit-taking study, with compact outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from maturity_grid_data import TENORS,SHORTS,WIDTHS,PROFITS,CADENCES,SIZINGS,START,END,prepare_paths
from maturity_grid_engine import INITIAL,simulate,statistics

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'results/maturity_profit_grid'


def safe_json(value):
    if isinstance(value,dict):return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    if isinstance(value,np.generic):return safe_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_json(path,value):
    path.write_text(json.dumps(safe_json(value),indent=2),encoding='utf-8')


def legacy_control():
    """Reproduce the saved earlier winner without reinterpreting its inputs."""
    cache=PROJECT/'data/cache/itm_structures'
    parameter='vertical_103/100_monthly_dte60_m103';strategy=parameter+'_profit_25'
    curve=pd.read_parquet(cache/'equity_curves.parquet',columns=[strategy])[strategy]
    marks=pd.read_parquet(cache/'trade_marks.parquet',filters=[('parameter_id','==',parameter)])
    daily=pd.Series(0.,index=curve.index);trades=0
    for _,group in marks.groupby('trade_id',sort=False):
        group=group.sort_values('mark_date')
        eligible=group.is_final|(group.mark_date.gt(group.entry_date)&group.pnl_realistic.ge(.25*group.max_profit_per_unit))
        exit_row=group.loc[eligible].iloc[0]
        path=group.loc[group.mark_date.le(exit_row.mark_date)].set_index('mark_date').pnl_mid.copy()
        path.iloc[-1]=exit_row.pnl_realistic
        cumulative=path*exit_row.contracts
        daily.loc[cumulative.index]+=cumulative.diff().fillna(cumulative.iloc[0])
        trades+=1
    rebuilt=INITIAL+daily.cumsum();difference=float(np.max(np.abs(rebuilt-curve)))
    if difference>1e-6:raise AssertionError(f'Legacy reconstruction mismatch: {difference}')
    old=pd.read_csv(PROJECT/'results/itm_structures/strategy_metrics.csv')
    row=old.loc[old.strategy_id.eq(strategy)].iloc[0].to_dict()
    positions=marks.drop_duplicates('trade_id')
    return dict(status='passed',cached_curve_max_difference_dollars=difference,trades=trades,
        weekend_dated_legacy_expirations=int((positions.expiration_date.dt.weekday>=5).sum()),
        cagr=row['cagr'],sharpe=row['sharpe_zero_cash'],vol=row['annualized_volatility'],max_drawdown=row['max_drawdown'],
        note='Reconciles the original cached trades and valuation convention; the expanded grid independently uses cash-index strike targets and weekday PM expiries.'),curve


def parameter_frame(tenor,shorts,widths,profits):
    ids=np.indices((len(shorts),len(widths),len(profits),4,2)).reshape(5,-1)
    target=np.asarray(profits)[ids[2]]
    return pd.DataFrame(dict(target_dte=tenor,short_target=np.asarray(shorts)[ids[0]],width=np.asarray(widths)[ids[1]],
        profit_target=np.where(np.isfinite(target),target,1.),cadence=np.asarray(CADENCES)[ids[3]],sizing=np.asarray(SIZINGS)[ids[4]]))


def strategy_id(row):
    profit='hold' if row['profit_target']==1 else f"p{round(row['profit_target']*100):02d}"
    return f"d{int(row['target_dte']):03d}_s{round(row['short_target']*2):03d}_w{round(row['width']*2):02d}_{profit}_{row['cadence']}_{row['sizing']}"


def label(row):
    spread=f"{row['short_target']:g}/{row['short_target']-row['width']:g}"
    profit='hold' if row['profit_target']==1 else f"{row['profit_target']:.0%} profit"
    return f"{spread} · {int(row['target_dte'])} DTE · {row['cadence']} · {profit}"


def run(prepared=None,shorts=SHORTS,widths=WIDTHS,profits=PROFITS,output=OUT,batch_shorts=2):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'grid').mkdir(exist_ok=True)
    started=time.monotonic();legacy,legacy_curve=legacy_control()
    write_json(output/'Legacy baseline validation.json',legacy)
    if prepared is None:prepared=prepare_paths()
    dates,cash,masks,cycles,audit,omissions,repairs=prepared
    total=len(cycles)*len(shorts)*len(widths)*len(profits)*4*2
    protocol=dict(status='running',combinations=total,start=str(dates[0].date()),end=str(dates[-1].date()),sessions=len(dates),
        maturity_targets=list(cycles),short_targets=list(shorts),widths=list(widths),profit_targets=[float(p) if np.isfinite(p) else 'hold' for p in profits],
        cadences=CADENCES,sizings=SIZINGS,initial=INITIAL,
        strikes='Percentages of cash SPX at entry; nearest listed strike, ties lower; maximum error 0.5 percentage point per leg',
        expiries='Listed weekday PM contracts only, nearest target; ties prefer longer. Tolerance 3 days through 14 DTE, otherwise 7.',
        entry='Holiday-adjusted Friday anchors; monthly third Friday. Both observed legs, positive net credit less than spread width required.',
        execution='Midpoint plus/minus 25% of each full bid/ask spread and $1.50 per leg, on both entry and early exit. Profit threshold uses executable net liquidation P&L after costs. No same-day profit exits.',
        settlement='Actual cash-index intrinsic value on listed PM expiry. Any position still open at the common data endpoint is closed at its final quote.',
        risk_sizing='Fixed 5% initial-capital max-loss budget, split by ceil(actual DTE / cadence days); enforce remaining 5% risk and 500% initial-capital cash-notional headroom at each entry.',
        notional_sizing='100% prior-close-equity cash-SPX notional divided across potential overlaps; entries cannot exceed remaining notional headroom. Existing holdings keep fixed quantities; no forced deleveraging.',
        tranches='Cadence days 7/14/28/30; available capacity left in cash until the next scheduled entry. Different profit targets do not cause off-schedule re-entry.',
        cash_return=0,spx_included=False,financing_return=0,
        data_quality='Same-day parity/adjacent-strike estimates can mark held positions but cannot authorize an entry or profit exit. Unresolved held marks use stale diagnostic values and disqualify a portfolio from rankings. No trades are removed based on future quote availability. Raw midpoint differences outside payoff bounds are flagged and cannot trigger profits; they remain valuation marks if the executable bid/ask interval intersects the payoff bounds.',
        ranking='At least 80% entry coverage, 50 trades, positive train and validation CAGR, no insolvency/unresolved held marks; validation Sharpe minus 0.1*absolute(train Sharpe - validation Sharpe). Test/full-period returns do not choose the validation leader.',
        chronological_splits={'train':str(dates[int(len(dates)*.6)-1].date()),'validation':str(dates[int(len(dates)*.8)-1].date()),'test':str(dates[-1].date())},
        limitations='This archive has already been studied. The historical test segment is reused, not a new untouched out-of-sample result. EOD execution timestamps are an archive assumption; no intraday fills, dividends, cash interest, taxes, financing or market impact are modeled.',
        storage='Prices, simulation and shortlisted results use Float64. Exhaustive-grid scalar metrics use compressed Float32 to conserve disk; metadata/trade counts remain exact.',
        source_audit=audit,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('maturity_grid_data.py'),Path(__file__).with_name('maturity_grid_engine.py')]})
    write_json(output/'Protocol.json',protocol);write_json(output/'Data audit.json',audit)
    omissions.to_csv(output/'Omitted entries.csv.gz',index=False,compression='gzip')
    repairs.to_csv(output/'Quote estimates and unresolved marks.csv.gz',index=False,compression='gzip')
    print(f"Legacy control passed: {legacy['cagr']:.2%} CAGR, {legacy['sharpe']:.3f} Sharpe; {total:,} grid rows",flush=True)
    leaders={};candidates=[];baseline=[];progress=[];qualified=0;done=0
    for tenor,group in cycles.items():
        for offset in range(0,len(shorts),batch_shorts):
            selected_shorts=np.asarray(shorts)[offset:offset+batch_shorts]
            curves,meta,_=simulate(group,selected_shorts,dates,masks,widths,profits)
            stats=statistics(curves,dates)
            frame=parameter_frame(tenor,selected_shorts,widths,profits)
            for key,value in {**meta,**stats}.items():frame[key]=value
            frame['eligible']=(frame.entry_coverage.ge(.8)&frame.trades.ge(50)&frame.unresolved_marks.eq(0)&~frame.insolvent&frame.train_cagr.gt(0)&frame.validation_cagr.gt(0)&np.isfinite(frame.validation_sharpe)&np.isfinite(frame.sharpe))
            frame['validation_score']=frame.validation_sharpe-.1*(frame.validation_sharpe-frame.train_sharpe).abs()
            qualified+=int(frame.eligible.sum())
            primary=frame.loc[frame.eligible&np.isclose(frame.profit_target,.25)]
            for sizing,sized in primary.groupby('sizing'):
                best=sized.sort_values('validation_score',ascending=False).iloc[0]
                key=(tenor,sizing)
                if key not in leaders or best.validation_score>leaders[key]['row']['validation_score']:
                    leaders[key]=dict(row=best.to_dict(),curve=curves[int(best.name)].copy())
            for sizing,sized in frame.loc[frame.eligible].groupby('sizing'):
                for idx,row in sized.nlargest(5,'validation_score').iterrows():
                    candidates.append(dict(row=row.to_dict(),curve=curves[int(idx)].copy()))
            candidates=sorted(candidates,key=lambda x:x['row']['validation_score'],reverse=True)[:30]
            reference=frame.loc[np.isclose(frame.short_target,103)&np.isclose(frame.width,3)&np.isclose(frame.profit_target,.25)]
            for idx,row in reference.iterrows():baseline.append(dict(row=row.to_dict(),curve=curves[int(idx)].copy()))
            compact=frame.copy()
            for column in compact.select_dtypes(include=['float64']).columns:compact[column]=compact[column].astype('float32')
            target=output/'grid'/f'd{tenor:03d}_batch{offset:02d}.parquet'
            pq.write_table(pa.Table.from_pandas(compact,preserve_index=False),target,compression='zstd',compression_level=9)
            done+=len(frame)
            if offset==0 or offset+batch_shorts>=len(shorts):
                print(f'Grid {done:,}/{total:,}: {tenor} DTE, short {selected_shorts[-1]:g}%; eligible {qualified:,}; {time.monotonic()-started:.0f}s',flush=True)
            if shutil.disk_usage(output).free<15*1024*1024:
                raise RuntimeError('Less than 15 MiB free; completed grid parts retained. More working space is required.')
        progress.append(dict(tenor=tenor,completed=done,eligible=qualified,seconds=time.monotonic()-started))
        write_json(output/'Progress.json',progress)
    assert done==total
    protocol.update(status='passed',eligible_combinations=qualified,elapsed_seconds=time.monotonic()-started)
    write_json(output/'Protocol.json',protocol)
    finish_report(output,dates,cash,masks,cycles,leaders,candidates,baseline,legacy,protocol)
    print(f'Completed {done:,} combinations; {qualified:,} met the screening rules. Report: {output/"Report.md"}',flush=True)
    return protocol


def finish_report(output,dates,cash,masks,cycles,leaders,candidates,baseline,legacy,protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    primary=pd.DataFrame([v['row'] for v in leaders.values()]).sort_values(['sizing','target_dte'])
    primary['strategy_id']=[strategy_id(r) for r in primary.to_dict('records')]
    primary.to_csv(output/'25 percent winners by maturity.csv',index=False)
    all_best=pd.DataFrame([v['row'] for v in candidates])
    all_best['strategy_id']=[strategy_id(r) for r in all_best.to_dict('records')]
    all_best.to_csv(output/'Validation shortlist all profit targets.csv',index=False)
    references=pd.DataFrame([v['row'] for v in baseline])
    references.to_csv(output/'103 100 maturity comparison.csv',index=False)
    chosen={strategy_id(v['row']):v for v in [*leaders.values(),*candidates[:10],*[b for b in baseline if b['row']['target_dte']==60 and b['row']['cadence']=='monthly']]}
    curve_frame=pd.DataFrame({key:v['curve'] for key,v in chosen.items()},index=dates)
    curve_frame['SPX benchmark']=INITIAL*cash/cash.iloc[0]
    curve_frame.to_parquet(output/'Shortlisted daily curves.parquet',compression='zstd')
    audits=[];ledger_rows=[]
    for key,value in chosen.items():
        r=value['row'];profits=np.array([np.inf if r['profit_target']==1 else r['profit_target']])
        replay,meta,records=simulate(cycles[int(r['target_dte'])],[r['short_target']],dates,masks,np.array([r['width']]),profits,True)
        index=CADENCES.index(r['cadence'])*2+SIZINGS.index(r['sizing'])
        difference=float(np.max(np.abs(replay[index]-value['curve'])))
        actual=[record for record in records if record['index']==index]
        terminal_difference=abs(INITIAL+sum(t['realized_pnl'] for t in actual)-replay[index,-1])
        if difference>1e-5 or terminal_difference>1e-5:raise AssertionError(f'Ledger validation failed: {key}')
        audits.append(dict(strategy_id=key,curve_replay_max_difference=difference,terminal_ledger_difference=terminal_difference,trades=len(actual)))
        for t in actual:
            t=dict(t);t.pop('index');t['entry_date']=str(dates[t.pop('entry')].date());t['exit_date']=str(dates[t.pop('exit')].date());t['strategy_id']=key;ledger_rows.append(t)
    pd.DataFrame(ledger_rows).to_parquet(output/'Shortlisted trade ledger.parquet',compression='zstd')
    write_json(output/'Validation.json',dict(status='passed',combinations=protocol['combinations'],legacy_control=legacy,shortlist_replays=audits,
        note='Exact single-configuration replays and independent sums of realized trade P&L reconcile the retained shortlisted curves.'))
    returns=curve_frame/curve_frame.shift(1).fillna(INITIAL)-1
    yearly=returns.groupby(returns.index.year).apply(lambda f:(1+f).prod()-1)
    yearly.to_csv(output/'Shortlisted calendar year returns.csv',index_label='year')
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for sizing,color in zip(SIZINGS,['#356ad1','#189187']):
        block=primary.loc[primary.sizing.eq(sizing)].sort_values('target_dte')
        for ax,metric,title in zip(axes.flat,['cagr','vol','max_drawdown','sharpe'],['CAGR','Annualized volatility','Maximum drawdown','Daily Sharpe']):
            ax.plot(block.target_dte,block[metric],marker='o',ms=3,color=color,label='5% risk budget' if sizing==SIZINGS[0] else '100% equity notional')
            ax.set_title(title,loc='left',fontsize=11);ax.grid(alpha=.18)
            if metric!='sharpe':ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].legend(fontsize=8);axes[1,0].set_xlabel('Target days to expiration');axes[1,1].set_xlabel('Target days to expiration')
    fig.suptitle('25% profit target · validation-selected spread and cadence at each maturity',fontsize=13)
    fig.tight_layout();fig.savefig(output/'25 percent maturity comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for sizing in SIZINGS:
        group=primary.loc[primary.sizing.eq(sizing)]
        if group.empty:continue
        row=group.nlargest(1,'validation_score').iloc[0].to_dict();key=strategy_id(row)
        nav=curve_frame[key]/INITIAL*100
        axes[0].plot(dates,nav,label=label(row)+' · '+('5% risk' if sizing==SIZINGS[0] else '100% notional'),lw=1.3)
        axes[1].plot(dates,nav/np.maximum.accumulate(np.r_[100,nav])[1:]-1,lw=1.2)
    axes[0].set_title('Leading 25%-profit strategies selected using validation data',loc='left');axes[0].set_ylabel('Growth of $100');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Drawdown');axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:ax.grid(alpha=.18)
    fig.tight_layout();fig.savefig(output/'Leading strategies.png',dpi=150);plt.close(fig)
    def percent(v):return f'{v:.2%}' if np.isfinite(v) else 'n/a'
    lines=['# SPX maturity and profit-taking grid','',f"Completed **{protocol['combinations']:,} parameter combinations**, with **{protocol['eligible_combinations']:,}** meeting the disclosed coverage, data-quality and train/validation screens.",'',
        f"The study covers {dates[0]:%B %d, %Y}–{dates[-1]:%B %d, %Y}, {len(dates):,} daily sessions. The original cached monthly 103/100, 60-DTE, 25%-profit winner was reproduced at {legacy['cagr']:.2%} CAGR and {legacy['sharpe']:.2f} Sharpe, within ${legacy['cached_curve_max_difference_dollars']:.8f} of its saved curve.",'',
        '## The 25% profit target','',
        'Each row chooses a spread and entry schedule within its maturity and sizing group using validation data. The historical test segment has already been used in earlier research; it is a reused diagnostic, not a new untouched holdout.','',
        '| DTE | Sizing | Spread | Entry schedule | CAGR | Volatility | Max DD | Sharpe | Validation Sharpe | Test Sharpe | Entry coverage |',
        '|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|']
    for r in primary.to_dict('records'):
        lines.append(f"| {r['target_dte']} | {'5% risk budget' if r['sizing']==SIZINGS[0] else '100% notional'} | {r['short_target']:g}/{r['short_target']-r['width']:g} | {r['cadence']} | {percent(r['cagr'])} | {percent(r['vol'])} | {percent(r['max_drawdown'])} | {r['sharpe']:.2f} | {r['validation_sharpe']:.2f} | {r['test_sharpe']:.2f} | {r['entry_coverage']:.1%} |")
    lines+=['','## Search and accounting','',
        f"- Maturities: {', '.join(map(str,protocol['maturity_targets']))} calendar days.",
        '- Short strikes: 90%–110% of SPX in 0.5-percentage-point steps. Widths: 1, 2, 3, 4, 5, 7.5 and 10 percentage points.',
        '- Entries: weekly, every two weeks, every four weeks, or monthly. Profit targets: 5% through 95% in five-point steps, plus hold to expiration. Early exits do not cause off-schedule re-entry.',
        '- The primary comparison fixes the profit target at 25%; the all-target shortlist is a separate search.',
        '- The 5% risk-budget mode uses original capital, allows leverage and imposes a 500% initial-capital notional ceiling. The 100% notional mode sizes from prior-close equity. Different maturity/cadence combinations split capital across expected overlaps. Both leave unused capacity in cash.',
        '- Entry and early-exit fills pay one quarter of each full bid/ask spread from midpoint plus $1.50 per leg. Profit triggers use net executable P&L after those costs. Cash earns zero; no dividends, financing, tax or market-impact returns are added.',
        '- Holdings retain their fixed contract quantities; surviving positions are marked every session. Actual PM expiry settles at cash intrinsic; the common data endpoint liquidates remaining positions.',
        '', '## Data and interpretation','',
        f"The earlier cached winner includes {legacy['weekend_dated_legacy_expirations']} weekend-dated archive expirations and uses the archive's underlying field. This grid selects listed weekday PM expirations and cash-index strike targets. Its comparable 103/100 control is saved separately across every maturity and cadence; it need not reproduce the old quote/calendar convention.",
        '', 'The archive supplies EOD snapshots with generated 16:00 labels. Observed quotes can authorize execution; explicitly estimated same-day marks cannot trigger profit-taking. Portfolios holding unresolved marks are retained for diagnostics but excluded from rankings. Raw midpoint differences outside payoff bounds are explicitly counted and cannot trigger profit-taking. Where the bid/ask interval intersects the payoff range, those observed midpoint marks are preserved; the code does not clip them into manufactured prices. Nearest-listed strike or expiry choices can make different parameter rows produce identical trades.',
        '', 'This is a large retrospective search. A high historical score alone does not establish future performance; compare validation/test consistency, neighboring maturities and widths, costs, coverage, and actual leverage. The test segment never enters the new validation-selection score, but the archive has been studied before.',
        '', '## Files','',
        '- `grid/`: every parameter row, as compressed Parquet. Scalar summary metrics use Float32; simulations and shortlisted curves/CSV results retain Float64 precision.',
        '- `25 percent winners by maturity.csv`: primary validation-selected comparison.',
        '- `103 100 maturity comparison.csv`: the original spread and 25% target, with each maturity and cadence.',
        '- `Validation shortlist all profit targets.csv`: the separate all-profit-target shortlist.',
        '- `Shortlisted daily curves.parquet`, `Shortlisted trade ledger.parquet`, and `Shortlisted calendar year returns.csv`: numerical paths and trade evidence.',
        '- `Protocol.json`, `Data audit.json`, `Validation.json`, and the compressed omission/quote-estimate logs: assumptions, source fingerprint, coverage and checks.',
        '', 'Reproduce from Raw Data: `python -B spx_option_research/scripts/run_maturity_profit_grid.py`.', '']
    (output/'Report.md').write_text('\n'.join(lines),encoding='utf-8')
    import html
    table=primary[['target_dte','sizing','short_target','width','cadence','cagr','vol','max_drawdown','sharpe','test_sharpe']].to_html(index=False,float_format=lambda x:f'{x:.4f}')
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SPX maturity grid</title><style>body{font:14px Segoe UI,sans-serif;color:#20344e;background:#f4f6fa;margin:30px auto;max-width:1200px;padding:0 20px}img{max-width:100%;background:white;border:1px solid #dae1ec;border-radius:10px;margin:15px 0}a{color:#3167d5}table{border-collapse:collapse;background:white;font-size:12px}td,th{padding:9px;border:1px solid #e0e5ee}.table{overflow:auto}p{line-height:1.7}</style><h1>SPX maturity and profit-taking grid</h1><p>'+html.escape(f"{protocol['combinations']:,} combinations · {len(dates):,} daily sessions · 25% profit target primary comparison")+'</p><p><a href="Report.md">Full research report</a> · <a href="25%20percent%20winners%20by%20maturity.csv">Download primary comparison</a> · <a href="Protocol.json">Protocol</a> · <a href="Validation.json">Validation</a></p><img src="25%20percent%20maturity%20comparison.png" alt="Performance by maturity"><img src="Leading%20strategies.png" alt="Daily equity and drawdown"><h2>Validation-selected winners at the 25% profit target</h2><p>Returns, volatility and drawdowns below are decimal ratios. Sizing differs between the two groups. This is retrospective research on an already-studied archive.</p><div class="table">'+table+'</div></html>',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.smoke:run(prepare_paths((7,60)),shorts=np.array([102.,103.]),widths=np.array([3.]),profits=np.array([.25,np.inf]),output=PROJECT/'results/maturity_profit_grid_smoke')
    else:run()
