"""Readable research page and coverage summary for the completed grid."""
from __future__ import annotations

import html
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / 'results/maturity_profit_grid'
SIZINGS = {'risk_5pct_initial': '5% initial-capital risk budget',
           'notional_100pct_equity': '100% equity notional'}
CADENCES = {'weekly': 'Weekly', 'every_2_weeks': 'Every 2 weeks',
            'every_4_weeks': 'Every 4 weeks', 'monthly': 'Monthly'}


def percent(value):
    return f'{value:.2%}' if np.isfinite(value) else '—'


def number(value):
    return f'{value:.2f}' if np.isfinite(value) else '—'


def geometry(row):
    return f"{row.short_target:g}/{row.short_target-row.width:g}"


def research_table(frame, extra=False):
    rows = []
    for r in frame.itertuples():
        row = {'DTE': r.target_dte, 'Spread': geometry(r), 'Entries': CADENCES[r.cadence]}
        if extra:
            row['Profit target'] = 'Hold' if r.profit_target == 1 else f'{r.profit_target:.0%}'
        row.update({'CAGR': percent(r.cagr), 'Annual vol': percent(r.vol),
                    'Max drawdown': percent(r.max_drawdown), 'Sharpe': number(r.sharpe),
                    'Validation Sharpe': number(r.validation_sharpe),
                    'Test CAGR': percent(r.test_cagr), 'Test Sharpe': number(r.test_sharpe),
                    'Coverage': f'{r.entry_coverage:.1%}', 'Trades': int(r.trades)})
        rows.append(row)
    return '<div class="table">'+pd.DataFrame(rows).to_html(index=False, border=0)+'</div>'


def historical_plots(output, historical, original):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from run_maturity_profit_grid import strategy_id
    curves = pd.read_parquet(output/'Shortlisted daily curves.parquet')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for sizing, color in zip(SIZINGS, ['#356ad1', '#189187']):
        block = historical.loc[historical.sizing.eq(sizing)].sort_values('target_dte')
        for ax, metric, title in zip(axes.flat, ['cagr','vol','max_drawdown','sharpe'], ['CAGR','Annual volatility','Maximum drawdown','Annualized Sharpe']):
            ax.plot(block.target_dte, block[metric], marker='o', ms=3, color=color, label=SIZINGS[sizing])
            ax.set_title(title, loc='left', fontsize=11)
            ax.grid(alpha=.18)
            if metric != 'sharpe':
                ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0,0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('Target days to expiration')
    fig.suptitle('25% profit target · highest full-history Sharpe within each maturity', fontsize=13)
    fig.tight_layout()
    fig.savefig(output/'Historical 25 percent maturity comparison.png', dpi=150)
    plt.close(fig)
    risk = historical.loc[historical.sizing.eq('risk_5pct_initial')]
    focus = pd.concat([risk.nlargest(1, 'sharpe'), risk.loc[risk.target_dte.eq(56)],
        original.loc[original.sizing.eq('risk_5pct_initial')&original.cadence.eq('monthly')&original.target_dte.eq(60)]])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for r in focus.to_dict('records'):
        key = strategy_id(r)
        nav = curves[key]/1_000_000*100
        caption = f"{r['short_target']:g}/{r['short_target']-r['width']:g} | {r['target_dte']} DTE | CAGR {r['cagr']:.2%} | Vol {r['vol']:.2%} | Max DD {r['max_drawdown']:.2%} | Sharpe {r['sharpe']:.2f}"
        axes[0].plot(curves.index, nav, label=caption, lw=1.35)
        axes[1].plot(curves.index, nav/np.maximum.accumulate(np.r_[100, nav])[1:]-1, lw=1.15)
    axes[0].set_title('Monthly entries · 25% profit target · 5% initial-capital risk budget', loc='left')
    axes[0].set_ylabel('Growth of $100')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Drawdown')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(output/'Historical leaders and original spread.png', dpi=150)
    plt.close(fig)


def present(output=OUT):
    output = Path(output)
    protocol = json.loads((output/'Protocol.json').read_text())
    independent = json.loads((output/'Independent validation.json').read_text())
    assert independent['status'] == 'passed'
    primary = pd.read_csv(output/'25 percent winners by maturity.csv')
    original = pd.read_csv(output/'103 100 maturity comparison.csv')
    historical = pd.read_csv(output/'Historical 25 percent winners by maturity.csv')
    historical_all = pd.read_csv(output/'Historical all target leaders.csv')
    historical_plots(output, historical, original)
    coverage_parts = []
    best_parts = []
    expected = len(protocol['short_targets'])*len(protocol['widths'])*len(protocol['profit_targets'])*len(protocol['cadences'])*len(protocol['sizings'])
    for path in sorted((output/'grid').glob('*.parquet')):
        frame = pd.read_parquet(path)
        coverage_parts.append(dict(target_dte=int(frame.target_dte.iloc[0]), tested=len(frame),
            eligible=int(frame.eligible.sum()), maximum_entry_coverage=float(frame.entry_coverage.max()),
            unresolved_disqualified=int(frame.unresolved_marks.gt(0).sum())))
        for _, group in frame.loc[frame.eligible].groupby('sizing'):
            best_parts.append(group.nlargest(10, 'validation_score'))
    coverage = pd.DataFrame(coverage_parts).groupby('target_dte', as_index=False).agg(
        tested=('tested', 'sum'), eligible=('eligible', 'sum'),
        maximum_entry_coverage=('maximum_entry_coverage', 'max'),
        unresolved_disqualified=('unresolved_disqualified', 'sum'))
    assert coverage.tested.eq(expected).all()
    assert int(coverage.tested.sum()) == protocol['combinations']
    assert int(coverage.eligible.sum()) == protocol['eligible_combinations']
    coverage.to_csv(output/'Coverage by maturity.csv', index=False)
    # Separate modes prevent one sizing convention from filling the entire shortlist.
    all_targets = pd.concat(best_parts).sort_values('validation_score', ascending=False).groupby('sizing', sort=False).head(10)
    all_targets.to_csv(output/'All target leaders by sizing.csv', index=False)
    winners = primary.sort_values('validation_score', ascending=False).groupby('sizing', sort=False).head(1)
    findings = []
    risk_history = historical.loc[historical.sizing.eq('risk_5pct_initial')]
    historical_leader = risk_history.nlargest(1, 'sharpe').iloc[0]
    r = historical_leader
    historical_finding = f"Using the original 5% initial-capital risk budget, the highest full-history Sharpe at the 25% profit target is {geometry(r)}, {r.target_dte} DTE, {CADENCES[r.cadence].lower()} entries: CAGR {percent(r.cagr)}, annual volatility {percent(r.vol)}, maximum drawdown {percent(r.max_drawdown)}, Sharpe {number(r.sharpe)}. Its reused-test Sharpe falls to {number(r.test_sharpe)}. This is a hindsight ranking."
    r = risk_history.loc[risk_history.target_dte.eq(56)].iloc[0]
    consistency_finding = f"The 56-DTE comparison, {geometry(r)} with {CADENCES[r.cadence].lower()} entries and a 25% profit target, has CAGR {percent(r.cagr)}, annual volatility {percent(r.vol)}, maximum drawdown {percent(r.max_drawdown)}, Sharpe {number(r.sharpe)}. Training, validation and reused-test Sharpe are {number(r.train_sharpe)}, {number(r.validation_sharpe)} and {number(r.test_sharpe)}. This row is highlighted after inspecting all periods, not selected prospectively."
    r = original.loc[original.sizing.eq('risk_5pct_initial')&original.cadence.eq('monthly')&original.target_dte.eq(60)].iloc[0]
    baseline_finding = f"For a comparison under the same new data conventions, the original monthly 103/100 at 60 DTE has CAGR {percent(r.cagr)}, annual volatility {percent(r.vol)}, maximum drawdown {percent(r.max_drawdown)} and Sharpe {number(r.sharpe)}. The exact legacy-cache reconstruction remains 4.16% CAGR and 1.09 Sharpe under its earlier quote/calendar convention."
    for r in winners.itertuples():
        findings.append(f"{SIZINGS[r.sizing]}: {geometry(r)}, {r.target_dte} DTE, {CADENCES[r.cadence].lower()} entries, 25% profit target; full-period CAGR {percent(r.cagr)}, annual volatility {percent(r.vol)}, maximum drawdown {percent(r.max_drawdown)}, Sharpe {number(r.sharpe)}. Validation Sharpe {number(r.validation_sharpe)}; reused-test CAGR {percent(r.test_cagr)}, Sharpe {number(r.test_sharpe)}. Entry coverage {r.entry_coverage:.1%}; {int(r.trades)} trades. Mean holding period {r.mean_holding_days:.1f} calendar days; maximum entry-basis notional {r.max_notional_pct_initial:.2f} times initial capital.")
    low_coverage = coverage.loc[coverage.maximum_entry_coverage.lt(.8), 'target_dte'].astype(str).tolist()
    coverage_note = ('No configuration at '+', '.join(low_coverage)+' DTE achieved the required 80% entry coverage, so those maturities have no ranked winner.' if low_coverage else 'Maturities without a primary row did not produce an eligible 25%-profit configuration.')
    assumption = 'The original control is the monthly 103/100 vertical at about 60 DTE, closing when net profit reaches 25% of opening credit. Both legs use the same expiration; calendar spreads and maturity-switching signals are outside this grid.'
    sizing_note = 'Spread labels are strike percentages of entry SPX (103/100 means short 103%, long 100%). Results are options plus idle cash, using fractional contract quantities for research. SPX stock exposure is not included. The 5% risk budget is fixed to initial capital and can create substantial notional leverage; the 100% notional mode sizes from prior-close equity. These modes are not directly interchangeable.'
    split_note = f"Chronological training ends {protocol['chronological_splits']['train']}; validation ends {protocol['chronological_splits']['validation']}; the remaining data through {protocol['end']} is the reused test segment. The validation-selected tables use validation Sharpe with a penalty for disagreement with training Sharpe; full-period and test performance do not select those rows. The separate historical rankings use all dates, including the test segment. The archive has been studied before, and the large search remains exposed to selection bias."
    findings_text = '# Maturity-grid findings\n\n'+f"Completed {protocol['combinations']:,} combinations; {protocol['eligible_combinations']:,} qualified.\n\n"+assumption+'\n\n'+'\n\n'.join([historical_finding,consistency_finding,baseline_finding])+'\n\n## Separate validation-selected results\n\n'+'\n\n'.join(findings)+'\n\n'+coverage_note+'\n\n'+sizing_note+'\n\n'+split_note+'\n\nHistorical rankings explicitly use full-period Sharpe and are separate from validation-selected rankings. All-target validation tables come from the Float32 exhaustive-grid files; retained historical and 25%-profit winners are recomputed in Float64.\n'
    (output/'Findings.md').write_text(findings_text, encoding='utf-8')
    e = html.escape
    links = [('Report.md','Full method'), ('Historical 25 percent winners by maturity.csv','Historical 25% winners CSV'),
        ('25 percent winners by maturity.csv','Validation 25% winners CSV'),
        ('103 100 maturity comparison.csv','103/100 maturity CSV'),
        ('All target leaders by sizing.csv','All-target leaders CSV'),
        ('Coverage by maturity.csv','Coverage CSV'), ('Independent validation.json','Independent accounting check')]
    doc = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SPX maturity grid · 25% profit-taking</title>',
        '<style>body{font:15px Segoe UI,Arial,sans-serif;color:#18334c;background:#f4f6fa;margin:32px auto;max-width:1250px;padding:0 22px;line-height:1.6}h1{font-size:32px;line-height:1.2}h2{margin-top:32px}h3{margin-bottom:10px}p{max-width:1050px}a{color:#215bbb}.summary,section{background:white;border:1px solid #d8e2ed;border-radius:10px;padding:18px 22px;margin:20px 0}.summary p{margin:9px 0}.muted{color:#536b7d}.table{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px;width:100%}td,th{padding:9px 11px;border-bottom:1px solid #d8e2ed;text-align:right}th{background:#edf3f9;text-align:right}td:nth-child(2),td:nth-child(3){text-align:left}img{max-width:100%;background:white;border-radius:8px}.downloads{display:flex;gap:9px 20px;flex-wrap:wrap}.tag{font-size:13px;letter-spacing:.08em;color:#346c99;text-transform:uppercase}</style></head><body>',
        '<div class="tag">SPX research · Completed grid</div><h1>Maturity and profit-taking comparison</h1>',
        f"<p>{protocol['combinations']:,} combinations · {protocol['start']} to {protocol['end']} · {protocol['eligible_combinations']:,} passed coverage and quality screens</p>",
        '<p>'+e(assumption)+'</p><nav class="downloads">',
        ' '.join(f'<a href="{quote(filename)}">{e(title)}</a>' for filename,title in links), '</nav>',
        '<div class="summary"><h2 style="margin-top:0">The useful comparisons at 25% profit</h2>',
        *['<p>'+e(finding)+'</p>' for finding in [historical_finding,consistency_finding,baseline_finding]],
        '<p class="muted">'+e(sizing_note)+'</p></div>',
        '<p>'+e(coverage_note)+'</p>',
        '<img src="Historical%20leaders%20and%20original%20spread.png" alt="Daily growth and drawdowns for the 42-DTE and 56-DTE comparisons alongside the original spread at 60 DTE">',
        '<img src="Historical%2025%20percent%20maturity%20comparison.png" alt="Full-period CAGR, volatility, drawdown and Sharpe across historical maturity winners">',
        '<h2>25% profit target: highest historical Sharpe within each maturity</h2><p>These rows are selected using the full history and are retrospective comparisons. They use the same eligibility screens as the validation ranking. CAGR, annual volatility, maximum drawdown and the first Sharpe column describe the full period.</p>']
    for sizing, title in SIZINGS.items():
        doc += ['<section><h3>'+e(title)+'</h3>', research_table(historical.loc[historical.sizing.eq(sizing)].sort_values('target_dte')), '</section>']
    doc += ['<h2>Separate validation-selected 25% winners</h2>', *['<p>'+e(f)+'</p>' for f in findings],
        '<p>Selection in this table uses validation Sharpe with a training-consistency penalty. A strong validation score can coexist with weak full-history returns; these are not automatically improvements on the original strategy.</p>']
    for sizing, title in SIZINGS.items():
        doc += ['<section><h3>'+e(title)+'</h3>', research_table(primary.loc[primary.sizing.eq(sizing)].sort_values('target_dte')), '</section>']
    doc += ['<h2>Original 103/100 spread: monthly entries, 25% profit target</h2><p>This holds the spread and schedule fixed to isolate the maturity comparison. Ineligible rows remain visible as diagnostics; this table does not choose a winner.</p>']
    for sizing, title in SIZINGS.items():
        block = original.loc[original.sizing.eq(sizing)&original.cadence.eq('monthly')].sort_values('target_dte')
        doc += ['<section><h3>'+e(title)+'</h3>', research_table(block), '</section>']
    doc += ['<h2>All profit targets: highest historical Sharpe</h2><p>Five leading hindsight results per sizing mode. The grid also tests profit-taking levels from 5% to 95% and holding until expiration. Read CAGR alongside Sharpe: very small returns can produce high ratios.</p>']
    for sizing, title in SIZINGS.items():
        doc += ['<section><h3>'+e(title)+'</h3>', research_table(historical_all.loc[historical_all.sizing.eq(sizing)], extra=True), '</section>']
    doc += ['<h2>Separate search across all profit targets</h2><p>Top 10 validation scores within each sizing mode. These rows can have different profit targets from the original 25% rule; all displayed summary values here are from the compressed exhaustive grid.</p>']
    for sizing, title in SIZINGS.items():
        doc += ['<section><h3>'+e(title)+'</h3>', research_table(all_targets.loc[all_targets.sizing.eq(sizing)], extra=True), '</section>']
    coverage_display = coverage.rename(columns={'target_dte':'DTE','tested':'Combinations','eligible':'Passed screens', 'maximum_entry_coverage':'Best entry coverage','unresolved_disqualified':'Rows with unresolved held marks'}).copy()
    coverage_display['Best entry coverage'] = coverage_display['Best entry coverage'].map(lambda x:f'{x:.1%}')
    doc += ['<h2>Coverage and interpretation</h2><section>', '<div class="table">'+coverage_display.to_html(index=False,border=0)+'</div>',
        '<p>'+e(split_note)+'</p><p>Costs include 25% of each full bid/ask spread from midpoint and $1.50 per leg on entry and early exit. Profit triggers use executable P&amp;L after costs. Estimated marks cannot trigger entries or early exits; unresolved held marks disqualify a portfolio. Expiry settles at cash intrinsic. Cash interest, dividends, financing, tax and market impact are excluded.</p>',
        '<p>The legacy cached winner reproduced exactly at 4.16% CAGR and 1.09 Sharpe. Its archive underlying/calendar convention differs from this grid’s cash-SPX strike targets and weekday PM expirations. Nearest listed strikes and expirations can make different parameter combinations produce identical trades.</p>',
        '<p>Independent checks reconstruct every retained daily curve from its saved trade ledger and prepared leg prices, verify first supported profit exits and position sizes, and reconcile summary metrics. Full details are available in the accounting-check download.</p></section></body></html>']
    (output/'index.html').write_text('\n'.join(doc), encoding='utf-8')
    report = (output/'Report.md').read_text(encoding='utf-8')
    report = report.replace('Reproduce from Raw Data: `python -B spx_option_research/scripts/run_maturity_profit_grid.py`.',
        'Reproduce the full search, independent audit and readable report from Raw Data: `python -B spx_option_research/scripts/run_maturity_profit_study.py`.')
    if '## Retrospective comparisons' not in report:
        retrospective = '## Retrospective comparisons\n\n'+'\n\n'.join([historical_finding, consistency_finding, baseline_finding])+ '\n\nHistorical rankings explicitly use the entire period and are separate from the validation-selected table above. See `Historical 25 percent winners by maturity.csv`, `Historical 25 percent leaders.csv` and `Historical all target leaders.csv`. '+coverage_note+'\n\n'+sizing_note+'\n\n'
        report = report.replace('## Search and accounting', retrospective+'## Search and accounting')
    (output/'Report.md').write_text(report, encoding='utf-8')
    source_files = [Path(__file__), Path(__file__).with_name('validate_maturity_grid.py'), Path(__file__).with_name('run_maturity_profit_study.py'), Path(__file__).with_name('historical_maturity_grid.py')]
    audit = dict(status='passed', verified_grid_rows=int(coverage.tested.sum()),
        independently_verified_strategies=independent['validated_strategies'],
        independently_verified_trades=independent['validated_trades'],
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    (output/'Report audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(findings_text, flush=True)
    return winners, coverage


if __name__ == '__main__':
    present()
