"""Readable charts and results for the dynamic entry-selection experiment."""
from __future__ import annotations

import gzip
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from run_dynamic_maturity_search import OUT, write_json

SIZING = {'risk_5pct_initial':'5% initial-capital risk budget','notional_100pct_equity':'100% equity notional'}
MODE = {'ladder':'Overlapping positions','one_at_a_time':'One position at a time'}
CADENCE = {'weekly':'Weekly','every_2_weeks':'Every 2 weeks','every_4_weeks':'Every 4 weeks','monthly':'Monthly'}
FAMILY = {'edge_per_risk':'Expected edge / maximum loss','annual_edge_per_risk':'Annualized expected edge / maximum loss',
    'payoff_sharpe':'Expected payoff Sharpe','annual_payoff_sharpe':'Annualized expected payoff Sharpe',
    'credit_per_risk':'Credit / maximum loss','annual_credit_per_risk':'Annualized credit / maximum loss',
    'credit_per_width':'Credit / spread width','annual_credit_per_notional':'Annualized credit / SPX notional',
    'iv_rv_history_z':'IV/RV richness versus own history','whole_spread_history_z':'Whole-spread edge versus own history'}


def pct(v):
    return f'{v:.2%}' if np.isfinite(v) else '—'


def num(v):
    return f'{v:.2f}' if np.isfinite(v) else '—'


def rule_description(r):
    description = FAMILY.get(r.family,r.family)
    if r.rv_window:
        description += f'; {int(r.rv_window)}-session RV'
    if r.history_window:
        description += f'; {int(r.history_window)}-week history'
        if str(r.rule).startswith('mean_leg'):
            description += '; mean leg IV'
        elif r.family=='iv_rv_history_z':
            description += '; short-leg IV'
    elif r.rv_window:
        description += f'; volatility ×{r.vol_multiplier:g}; drift {r.assumed_drift:.0%}'
    description += '; positive score required' if r.positive_only else '; always choose the highest score'
    return description


def table(frame, include_rule=True):
    rows=[]
    for r in frame.itertuples():
        row={}
        if include_rule:
            row['Entry ranking']=rule_description(r)
            row['Allowed DTE']=f'{int(r.min_dte)}–{int(r.max_dte)}'
        else:
            row['Fixed spread']=f'{r.short_target:g}/{r.short_target-r.width:g}'
            row['Target DTE']=int(r.target_dte)
        row.update({'Entries':CADENCE[r.cadence],'CAGR':pct(r.cagr),'Annual vol':pct(r.vol),
            'Max drawdown':pct(r.max_drawdown),'Sharpe':num(r.sharpe),
            'Validation Sharpe':num(r.validation_sharpe),'Test CAGR':pct(r.test_cagr),'Test Sharpe':num(r.test_sharpe),
            'Trades':int(r.trades),'Different DTE targets':int(r.unique_maturity_targets),
            'Different short strikes':int(r.unique_short_targets),'Different widths':int(r.unique_widths),
            'Unresolved marks':int(r.unresolved_marks)})
        rows.append(row)
    return '<div class="table">'+pd.DataFrame(rows).to_html(index=False,border=0)+'</div>'


def present(output=OUT):
    output=Path(output)
    protocol=json.loads((output/'Protocol.json').read_text())
    validation=json.loads((output/'Validation.json').read_text())
    assert protocol['status']==validation['status']=='passed'
    overall=pd.read_csv(output/'Overall leaders.csv')
    controls=pd.read_csv(output/'Matched fixed controls.csv')
    within=pd.read_csv(output/'Within universe fixed winners.csv')
    controls=pd.concat([controls,within],ignore_index=True)
    curves=pd.read_parquet(output/'Retained daily curves.parquet')
    ledger=pd.read_parquet(output/'Retained trade ledger.parquet')
    historical=overall.loc[overall.ranking_basis.eq('historical')]
    selected_validation=overall.loc[overall.ranking_basis.eq('validation')]
    comparisons=[]
    for r in historical.itertuples():
        matched=within.loc[within.dynamic_strategy_id.eq(r.strategy_id)]
        if matched.empty:
            continue
        fixed=matched.nlargest(1,'sharpe').iloc[0]
        comparisons.append(dict(strategy_id=r.strategy_id,sizing=r.sizing,position_mode=r.position_mode,
            cadence=r.cadence,dynamic_cagr=r.cagr,dynamic_sharpe=r.sharpe,dynamic_max_drawdown=r.max_drawdown,
            fixed_strategy_id=fixed.strategy_id,fixed_spread=f'{fixed.short_target:g}/{fixed.short_target-fixed.width:g}',
            fixed_target_dte=int(fixed.target_dte),fixed_cagr=fixed.cagr,fixed_sharpe=fixed.sharpe,fixed_max_drawdown=fixed.max_drawdown,
            fixed_trades=int(fixed.trades),fixed_entry_coverage=fixed.entry_coverage,
            fixed_meets_primary_screens=bool(fixed.fixed_meets_primary_screens),
            sharpe_difference=r.sharpe-fixed.sharpe,cagr_difference=r.cagr-fixed.cagr))
    comparisons=pd.DataFrame(comparisons)
    comparisons.to_csv(output/'Dynamic versus matched fixed controls.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    groups=[(s,m) for s in SIZING for m in MODE]
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True)
    for ax,(sizing,mode) in zip(axes.flat,groups):
        r=historical.loc[historical.sizing.eq(sizing)&historical.position_mode.eq(mode)].iloc[0]
        comp=comparisons.loc[comparisons.strategy_id.eq(r.strategy_id)].iloc[0]
        fixed=controls.loc[controls.strategy_id.eq(comp.fixed_strategy_id)].iloc[0]
        fixed_label=f'Fixed {comp.fixed_spread}, {comp.fixed_target_dte} DTE'+(' (diagnostic)' if not comp.fixed_meets_primary_screens else '')
        for key,label,color,row in [(r.strategy_id,'Dynamic','#2667cc',r),(fixed.strategy_id,fixed_label,'#d48629',fixed)]:
            nav=curves[key]/1_000_000*100
            ax.plot(curves.index,nav,color=color,lw=1.35,label=label)
        ax.set_title(SIZING[sizing]+'\n'+MODE[mode]+' · '+CADENCE[r.cadence],loc='left',fontsize=10)
        ax.text(.025,.98,f"Dynamic: CAGR {pct(r.cagr)} | Vol {pct(r.vol)}\nMax DD {pct(r.max_drawdown)} | Sharpe {num(r.sharpe)}",transform=ax.transAxes,va='top',fontsize=8,
            bbox=dict(facecolor='white',alpha=.9,edgecolor='#d9e0e8',boxstyle='round,pad=.35'))
        ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.18);ax.set_ylabel('Growth of $100')
    fig.suptitle('Dynamic leaders and fixed comparisons within the same candidate universe',fontsize=13)
    fig.tight_layout();fig.savefig(output/'Dynamic versus fixed.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,7),sharex=True)
    for ax,(sizing,mode) in zip(axes.flat,groups):
        r=historical.loc[historical.sizing.eq(sizing)&historical.position_mode.eq(mode)].iloc[0]
        comp=comparisons.loc[comparisons.strategy_id.eq(r.strategy_id)].iloc[0]
        for key,label,color in [(r.strategy_id,'Dynamic','#2667cc'),(comp.fixed_strategy_id,'Fixed control','#d48629')]:
            nav=curves[key].to_numpy()
            ax.plot(curves.index,nav/np.maximum.accumulate(np.r_[1_000_000,nav])[1:]-1,label=label,color=color,lw=1.1)
        ax.set_title(SIZING[sizing]+'\n'+MODE[mode],loc='left',fontsize=10)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Daily drawdowns for the same comparisons',fontsize=13)
    fig.tight_layout();fig.savefig(output/'Dynamic drawdowns.png',dpi=140);plt.close(fig)
    focus=historical.loc[historical.sizing.eq('risk_5pct_initial')&historical.position_mode.eq('one_at_a_time')].iloc[0]
    trades=ledger.loc[ledger.strategy_id.eq(focus.strategy_id)].sort_values('entry')
    trade_dates=pd.to_datetime(trades.entry_date)
    fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True)
    axes[0].scatter(trade_dates,trades.actual_dte,s=11,color='#2667cc');axes[0].set_ylabel('Actual entry DTE')
    axes[1].plot(trade_dates,trades.short_target,'o-',ms=2,lw=.6,label='Short put',color='#2667cc')
    axes[1].plot(trade_dates,trades.short_target-trades.width,'o-',ms=2,lw=.6,label='Long put',color='#199584')
    axes[1].set_ylabel('Strike target (% SPX)');axes[1].legend(loc='upper left',fontsize=8)
    axes[2].scatter(trade_dates,trades.width,s=11,color='#a169b9');axes[2].set_ylabel('Width (% points)')
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle('What the selector actually bought and sold at each entry\n5% risk budget · one position at a time · historical leader',fontsize=12)
    fig.tight_layout();fig.savefig(output/'Entry choices over time.png',dpi=140);plt.close(fig)
    # A compressed CSV makes the complete grid available without requiring Parquet tools.
    with gzip.open(output/'All policy results.csv.gz','wt',encoding='utf-8',newline='') as handle:
        for i,path in enumerate(sorted((output/'grid').glob('*.parquet'))):
            pd.read_parquet(path).to_csv(handle,index=False,header=i==0)
    findings=[]
    for r in historical.itertuples():
        comp=comparisons.loc[comparisons.strategy_id.eq(r.strategy_id)].iloc[0]
        control_note=(f" No fixed candidate passed every primary eligibility screen. The displayed fixed comparison is a diagnostic with {comp.fixed_trades} trades and {comp.fixed_entry_coverage:.1%} score coverage; it uses a lower minimum trade count and can have lower coverage or nonpositive train/validation returns." if not comp.fixed_meets_primary_screens else '')
        findings.append(f"{SIZING[r.sizing]}, {MODE[r.position_mode].lower()}, {CADENCE[r.cadence].lower()} entries: {rule_description(r)}. Allowed maturities {r.min_dte:g}–{r.max_dte:g} DTE, short strikes {r.min_short:g}–{r.max_short:g}% and widths {r.min_width:g}–{r.max_width:g} percentage points. CAGR {pct(r.cagr)}, volatility {pct(r.vol)}, maximum drawdown {pct(r.max_drawdown)}, full-period Sharpe {num(r.sharpe)}; reused-test Sharpe {num(r.test_sharpe)}. Executed {int(r.trades)} trades across {int(r.unique_maturity_targets)} maturity targets, {int(r.unique_short_targets)} short-strike targets and {int(r.unique_widths)} widths. Fixed comparison within the identical universe, with the same score gate, schedule and sizing: {comp.fixed_spread} at {comp.fixed_target_dte} DTE, CAGR {pct(comp.fixed_cagr)}, Sharpe {num(comp.fixed_sharpe)}."+control_note)
    explanation='At every scheduled entry, each policy scores the then-available candidates and chooses the highest-scoring spread jointly across strike, width and expiration. The chosen contracts are held until the first supported EOD quote reaches 25% net profit, expiry, or the common data endpoint. The position is not switched merely because a different spread later scores better.'
    caution='Historical leaders were selected after looking across the entire period. Validation leaders are listed separately and are selected using training/validation data only. The archive has been studied before, so the later segment is reused historical evidence, not a fresh untouched out-of-sample test. Comparing hundreds of thousands of configurations creates substantial selection bias.'
    accounting='All returns are option P&L plus idle cash, without an SPX stock position. The 5% loss budget uses initial capital and can create up to 5 times initial-capital notional exposure. The 100% notional mode uses prior-close equity. Entries and early exits pay 25% of each leg’s full bid/ask spread from midpoint plus $1.50 per leg. Fractional contracts are used. Cash interest, financing, dividends, taxes and market impact are excluded.'
    report=['# Dynamic SPX spread selection across strikes, widths and expirations','',
        f"Completed **{protocol['completed_combinations']:,} dynamic policy configurations**, evaluating up to **{protocol['candidate_geometries_per_entry']:,} candidate targets at each entry**, plus **{protocol['additional_fixed_candidate_tests']:,} fixed-candidate controls**. The archive supplied {protocol['source_audit']['valid_entry_candidates']:,} valid candidate-entry combinations after the liquidity filter.",'',
        explanation,'',f"Common trading window: {protocol['evaluation_start']} to {protocol['evaluation_end']}. The preceding 104 weekly opportunities warm up history-dependent rules. Training ends {protocol['train_end']}; validation ends {protocol['validation_end']}.",'',
        '## Historical leaders','',*sum(([f,''] for f in findings),[]),
        '## Interpretation','',caution,'',accounting,'',
        'Expected-payoff rules use a physical lognormal terminal-payoff model with trailing 21/63/126-session realized volatility, volatility multipliers 0.8/1/1.2, and annual drift assumptions of 0%/5%. They rank edge divided by maximum loss or payoff standard deviation, with or without annualization. These are entry proxies: terminal expected payoff is not a model of the expected 25%-profit exit. Credit-based scores do not estimate expected losses. IV/RV and whole-spread history z-scores measure relative richness against strictly prior observations.','',
        'Every policy checks six maturity universes, three short-strike universes and four width universes, with and without a positive-score gate. Four entry schedules, two sizing methods and two overlap modes produce the full grid. Maturity targets cover 3–180 days; the search selects the nearest listed weekday PM expiration to each of 16 targets. It does not enumerate every calendar-day target. Different parameters can produce identical decisions.','',
        'A candidate must have valid observed quotes on both legs, positive net credit and maximum loss, and estimated closing friction no greater than half the 25% profit objective. Realized-volatility inputs end on the prior session. Entry quotes and supplied IV use the current EOD snapshot; fills at that snapshot remain an assumption. No future P&L, exit date or missing-quote status enters selection.','',
        'The one-at-a-time policy waits until a position closes and then enters at its next scheduled opportunity. It does not immediately reopen off schedule. Overlapping policies divide allocations across anticipated overlaps and enforce aggregate capacity. Original strikes and quantities remain fixed during each trade.','',
        'The main charts compare each historical leader with fixed candidates inside its own strike, width and maturity universe, using identical score availability, positive-score gating, entry cadence, sizing and overlap mode. The highest-Sharpe fixed candidate passing the same primary screens is preferred. When none pass, the chart explicitly labels a diagnostic comparator requiring at least 20 trades and no unresolved marks; its coverage and trade count are disclosed. Both sides are chosen with hindsight over the same period. Earlier fixed-spread controls are also retained in the matched-controls file.','',
        'Estimated same-day marks cannot authorize entries or early profit exits. Unresolved marks are counted. Historical rankings exclude affected portfolios; validation selection checks quality only through validation end so future quote quality cannot influence the chosen policy. Any later quality flags remain visible in the validation table.','',
        f"Independent reconciliation passed for {validation['retained_strategies']} retained strategies, {validation['retained_trades']:,} trades and {validation['scheduled_entry_decisions_checked']:,} scheduled entry decisions. Maximum difference in independently rebuilt daily NAV: ${validation['max_daily_nav_difference']:.10f}.",'',
        '## Files','',
        '- `All policy results.csv.gz` and `grid/`: the complete search.','- `Overall leaders.csv`: historical and validation leaders by sizing and overlap mode.',
        '- `Historical leaders by family.csv`, `Validation leaders by family.csv`: best policies within each scoring family.',
        '- `Matched fixed controls.csv`, `Dynamic versus matched fixed controls.csv`: comparable fixed-spread results.',
        '- `Within universe fixed grid.parquet`, `Within universe fixed winners.csv`: exhaustive fixed candidates within each illustrated dynamic leader’s allowed universe.',
        '- `Retained daily curves.parquet`, `Retained trade ledger.parquet`, `Entry decision audit.parquet`: actual positions, daily curves, score and cash/entry decisions.',
        '- `Selection distribution.csv`, `Calendar year returns.csv`: the changing chosen spreads and return history.',
        '- `Protocol.json`, `Validation.json`, `Scoring rule definitions.csv`: assumptions, fingerprints and checks.','',
        'Reproduce: `python -B spx_option_research/scripts/run_dynamic_maturity_study.py` from Raw Data.','']
    (output/'Report.md').write_text('\n'.join(report),encoding='utf-8')
    e=html.escape
    links=[('Report.md','Full method and findings'),('All policy results.csv.gz','Complete grid CSV (gzip)'),
        ('Overall leaders.csv','Overall leaders CSV'),('Selection distribution.csv','Chosen-spread distribution'),
        ('Calendar year returns.csv','Calendar-year returns'),('Validation.json','Accounting and selection audit')]
    doc=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dynamic SPX spread-selection research</title>',
        '<style>body{font:15px Segoe UI,Arial,sans-serif;color:#16354e;background:#f3f6fa;max-width:1320px;margin:35px auto;padding:0 22px;line-height:1.65}h1{font-size:33px;line-height:1.2}h2{margin-top:32px}h3{margin-top:5px}.box,section{background:white;border:1px solid #d6e0ec;border-radius:10px;padding:18px 22px;margin:20px 0}p{max-width:1150px}.links{display:flex;gap:8px 22px;flex-wrap:wrap}a{color:#235fbb}.table{overflow:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}th,td{padding:9px;border-bottom:1px solid #dce4ed;text-align:right}th{background:#eaf1f8}td:first-child,th:first-child{text-align:left;white-space:normal;min-width:220px;max-width:450px}img{max-width:100%;border-radius:8px}summary{cursor:pointer;font-weight:600}.muted{color:#526e82}.tag{text-transform:uppercase;letter-spacing:.08em;color:#3c729a;font-size:12px}</style></head><body>',
        '<div class="tag">Completed research · Dynamic entry selection</div><h1>Choose the spread anew at each entry</h1>',
        f"<p><strong>{protocol['completed_combinations']:,} policy configurations</strong> · up to {protocol['candidate_geometries_per_entry']:,} strike/width/maturity targets per entry · 25% profit-taking</p>",
        '<p>'+e(explanation)+'</p><nav class="links">',
        ' '.join(f'<a href="{quote(name)}">{e(label)}</a>' for name,label in links),'</nav>',
        '<div class="box"><h2 style="margin-top:0">What the completed search found</h2>',
        *['<p>'+e(f)+'</p>' for f in findings],'<p class="muted">'+e(caution)+'</p></div>',
        '<p>The charts compare each dynamic leader with a fixed candidate inside its own permitted strike, width and maturity range. Both use the same scoring availability, positive-score gate, schedule, sizing and position mode. Primary eligibility screens are matched where possible; when no fixed candidate qualifies, a lower-screened diagnostic comparator is explicitly labeled. Both sides are selected retrospectively.</p>',
        '<img src="Dynamic%20versus%20fixed.png" alt="Dynamic leaders compared with the best fixed candidates inside the same permitted universe and under identical gates, schedules and sizing">',
        '<img src="Dynamic%20drawdowns.png" alt="Daily drawdowns for the same comparisons">',
        '<h2>Actual entry choices</h2><p>The following chart shows the historical leader with a 5% risk budget and one position at a time. Each point is an executed trade, showing the expiration, strikes and width selected on that entry.</p>',
        '<img src="Entry%20choices%20over%20time.png" alt="Selected expirations, short and long strikes, and widths changing at each actual entry">',
        '<h2>Historical leaders by sizing and position mode</h2><p>Full-period performance selected these rows. The test columns are a consistency check and were part of the history used for that selection.</p>']
    for sizing,mode in groups:
        doc+=['<section><h3>'+e(SIZING[sizing]+' · '+MODE[mode])+'</h3>',table(historical.loc[historical.sizing.eq(sizing)&historical.position_mode.eq(mode)]),'</section>']
    doc+=['<h2>Separate validation-selected leaders</h2>',f"<p>Training ends {protocol['train_end']}; validation ends {protocol['validation_end']}. These rows maximize validation Sharpe with a penalty for disagreement with training Sharpe. Later returns and later missing-quote flags do not choose the winner. All quality flags remain visible.</p>"]
    for sizing,mode in groups:
        doc+=['<section><h3>'+e(SIZING[sizing]+' · '+MODE[mode])+'</h3>',table(selected_validation.loc[selected_validation.sizing.eq(sizing)&selected_validation.position_mode.eq(mode)]),'</section>']
    doc+=['<h2>Recent trades from the illustrated selector</h2><p>Spread strike targets are percentages of SPX at entry; the actual listed strikes and expiration are saved for every trade.</p>']
    recent=trades.tail(12)[['entry_date','expiry','actual_dte','short_target','width','short_strike','long_strike','entry_score','candidates_considered','exit_date','realized_pnl']].copy()
    recent.columns=['Entry','Expiration','Actual DTE','Short target %','Width, % points','Short strike','Long strike','Entry score','Candidates','Exit','Net P&L, dollars']
    doc+=['<section><div class="table">'+recent.to_html(index=False,border=0,float_format=lambda v:f'{v:,.3f}')+'</div></section>',
        '<h2>Best policy within each scoring family</h2>']
    for basis in ('Historical','Validation'):
        family=pd.read_csv(output/f'{basis} leaders by family.csv')
        doc+=['<section><details><summary>'+basis+' family rankings</summary>']
        for sizing,mode in groups:
            doc+=['<h3>'+e(SIZING[sizing]+' · '+MODE[mode])+'</h3>',table(family.loc[family.sizing.eq(sizing)&family.position_mode.eq(mode)])]
        doc+=['</details></section>']
    doc+=['<h2>Method and accounting</h2><section><p>'+e(accounting)+'</p>',
        '<p>Expected-payoff scores estimate terminal spread economics under stated volatility and drift assumptions; they do not directly forecast the 25%-profit exit. The history scores use each candidate’s own prior observations. Entry selection never sees subsequent P&amp;L or future quote availability.</p>',
        '<p>Entry schedules are weekly, every two weeks, every four weeks or monthly. There is no off-schedule immediate reentry. Expiry settles to actual cash intrinsic; early profit-taking requires observed executable quotes after costs. Estimated marks cannot authorize entry or a profit exit.</p>',
        f"<p>Independent checks passed for {validation['retained_strategies']} retained strategies, {validation['retained_trades']:,} trades and {validation['scheduled_entry_decisions_checked']:,} scheduled entry decisions. Full-grid summary files use Float32; shortlisted simulations and CSV metrics retain Float64.</p></section></body></html>"]
    (output/'index.html').write_text('\n'.join(doc),encoding='utf-8')
    write_json(output/'Report audit.json',dict(status='passed',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grid_rows=protocol['completed_combinations'],chart_count=3,retained_strategies=validation['retained_strategies']))
    print('\n\n'.join(findings),flush=True)
    return historical,comparisons


if __name__=='__main__':
    present()
