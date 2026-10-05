"""Explicit hindsight rankings, separate from validation-selected portfolios."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from maturity_grid_data import CADENCES, SIZINGS, PROFITS
from maturity_grid_engine import simulate, statistics
from run_maturity_profit_grid import strategy_id, parameter_frame

OUT = Path(__file__).resolve().parents[1] / 'results/maturity_profit_grid'


def historical(prepared, output=OUT):
    output = Path(output)
    dates, _, masks, cycles, *_ = prepared
    fixed_parts = []
    all_parts = []
    for path in sorted((output/'grid').glob('*.parquet')):
        frame = pd.read_parquet(path)
        for _, group in frame.loc[frame.eligible].groupby('sizing'):
            fixed_parts.append(group.loc[np.isclose(group.profit_target, .25)].nlargest(5, 'sharpe'))
            all_parts.append(group.nlargest(5, 'sharpe'))
    fixed = pd.concat(fixed_parts).sort_values('sharpe', ascending=False)
    maturity = fixed.groupby(['target_dte', 'sizing'], sort=False).head(1)
    all_best = pd.concat(all_parts).sort_values('sharpe', ascending=False).groupby('sizing', sort=False).head(5)
    selected = pd.concat([maturity, fixed.groupby('sizing', sort=False).head(5), all_best])
    selected['strategy_id'] = [strategy_id(r) for r in selected.to_dict('records')]
    selected = selected.drop_duplicates('strategy_id')
    curve_frame = pd.read_parquet(output/'Shortlisted daily curves.parquet')
    ledger = pd.read_parquet(output/'Shortlisted trade ledger.parquet')
    precise = {}
    new_records = []
    for r in selected.to_dict('records'):
        profit = np.inf if r['profit_target'] == 1 else PROFITS[np.argmin(np.abs(PROFITS-r['profit_target']))]
        nav, meta, trades = simulate(cycles[int(r['target_dte'])], [r['short_target']], dates, masks,
            np.array([r['width']]), np.array([profit]), True)
        index = CADENCES.index(r['cadence'])*2 + SIZINGS.index(r['sizing'])
        metrics = statistics(nav, dates)
        row = parameter_frame(int(r['target_dte']), [r['short_target']], [r['width']], [profit]).iloc[index].to_dict()
        row.update({key:value[index] for key,value in {**meta, **metrics}.items()})
        row['eligible'] = True
        row['validation_score'] = row['validation_sharpe']-.1*abs(row['validation_sharpe']-row['train_sharpe'])
        key = strategy_id(row)
        row['strategy_id'] = key
        precise[key] = row
        if key in curve_frame:
            assert np.max(np.abs(curve_frame[key].to_numpy()-nav[index])) < 1e-5
            continue
        curve_frame[key] = nav[index]
        for trade in trades:
            if trade['index'] != index:
                continue
            trade = dict(trade)
            trade.pop('index')
            trade['entry_date'] = str(dates[trade.pop('entry')].date())
            trade['exit_date'] = str(dates[trade.pop('exit')].date())
            trade['strategy_id'] = key
            new_records.append(trade)
    def write(rows, name):
        keys = [strategy_id(r) for r in rows.to_dict('records')]
        result = pd.DataFrame([precise[k] for k in keys])
        result.to_csv(output/name, index=False)
        return result
    winners = write(maturity, 'Historical 25 percent winners by maturity.csv')
    leaders = write(fixed.groupby('sizing', sort=False).head(5), 'Historical 25 percent leaders.csv')
    write(all_best, 'Historical all target leaders.csv')
    curve_frame.to_parquet(output/'Shortlisted daily curves.parquet', compression='zstd')
    if new_records:
        pd.concat([ledger, pd.DataFrame(new_records)], ignore_index=True).to_parquet(output/'Shortlisted trade ledger.parquet', compression='zstd')
    daily_returns = curve_frame/curve_frame.shift(1).fillna(1_000_000)-1
    daily_returns.groupby(daily_returns.index.year).apply(lambda f:(1+f).prod()-1).to_csv(output/'Shortlisted calendar year returns.csv', index_label='year')
    (output/'Historical ranking protocol.json').write_text(json.dumps(dict(
        selection='Highest full-period daily Sharpe among rows passing the original eligibility screens. This explicitly uses the full historical period, including the reused test segment, and is not a validation or out-of-sample claim.',
        precision='Compressed grid locates candidates; each retained candidate is then simulated again in Float64.',
        retained_strategies=len(selected)), indent=2), encoding='utf-8')
    print(leaders[['target_dte','short_target','width','cadence','sizing','cagr','vol','max_drawdown','sharpe','validation_sharpe','test_sharpe']].to_string(index=False), flush=True)
    return winners, leaders
