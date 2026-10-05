"""Independently reconcile saved trade ledgers to every shortlisted daily NAV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parents[1] / 'results/maturity_profit_grid'
INITIAL = 1_000_000.
CADENCE_DAYS = {'weekly': 7, 'every_2_weeks': 14, 'every_4_weeks': 28, 'monthly': 30}


def validate(prepared, output=OUT):
    output = Path(output)
    dates, _, _, cycles, *_ = prepared
    protocol = json.loads((output / 'Protocol.json').read_text())
    parts = sorted((output / 'grid').glob('*.parquet'))
    row_count = sum(pq.read_metadata(p).num_rows for p in parts)
    assert row_count == protocol['combinations'], (row_count, protocol['combinations'])
    ledger = pd.read_parquet(output / 'Shortlisted trade ledger.parquet')
    curves = pd.read_parquet(output / 'Shortlisted daily curves.parquet')
    parameter_tables = [
        pd.read_csv(output / '25 percent winners by maturity.csv'),
        pd.read_csv(output / 'Validation shortlist all profit targets.csv'),
    ]
    for name in ['Historical 25 percent winners by maturity.csv', 'Historical 25 percent leaders.csv', 'Historical all target leaders.csv']:
        if (output/name).exists():
            parameter_tables.append(pd.read_csv(output/name))
    parameters = pd.concat(parameter_tables).drop_duplicates('strategy_id').set_index('strategy_id')
    by_cycle = {tenor: {c.entry: c for c in group} for tenor, group in cycles.items()}
    checks = []
    for strategy, trades in ledger.groupby('strategy_id', sort=False):
        tenor = int(strategy.split('_')[0][1:])
        # Reference strategies are also retained; their geometry is in the ledger.
        cadence = next(c for c in CADENCE_DAYS if '_' + c + '_' in strategy)
        risk_sizing = strategy.endswith('_risk_5pct_initial')
        target_token = next(t for t in strategy.split('_') if t.startswith('p') and t[1:].isdigit()) if '_hold_' not in strategy else None
        target = int(target_token[1:]) / 100 if target_token else np.inf
        rebuilt = np.full(len(dates), INITIAL)
        positions = []
        max_credit_error = max_exit_error = 0.
        for trade in trades.itertuples():
            entry = dates.get_loc(pd.Timestamp(trade.entry_date))
            exit_day = dates.get_loc(pd.Timestamp(trade.exit_date))
            cycle = by_cycle[tenor][entry]
            offset = exit_day - entry
            short = int(np.flatnonzero(cycle.strikes == trade.short_strike)[0])
            long = int(np.flatnonzero(cycle.strikes == trade.long_strike)[0])
            debit = cycle.mid[:offset+1, short] - cycle.mid[:offset+1, long]
            friction = cycle.cost[:offset+1, short] + cycle.cost[:offset+1, long]
            credit = (debit[0] - friction[0]) * 100
            max_credit_error = max(max_credit_error, abs(credit - trade.entry_credit))
            assert np.isfinite(debit).all(), strategy
            assert trade.unresolved == 0, strategy
            pnl = trade.quantity * (credit - debit * 100)
            pnl[-1] = trade.quantity * (credit - (debit[-1] + friction[-1]) * 100)
            max_exit_error = max(max_exit_error, abs(pnl[-1] - trade.realized_pnl))
            rebuilt[entry:exit_day+1] += pnl
            rebuilt[exit_day+1:] += pnl[-1]
            width = trade.short_strike - trade.long_strike
            observed = (cycle.quality[:offset+1, short] == 1) & (cycle.quality[:offset+1, long] == 1)
            bounded = (debit >= -.011) & (debit <= width + .011)
            liquidation = credit - (debit + friction) * 100
            reached = observed & bounded & (liquidation >= target * credit)
            reached[0] = False
            assert not reached[:-1].any(), (strategy, trade.entry_date, 'missed profit exit')
            assert exit_day == cycle.end or reached[-1], (strategy, 'unsupported early exit')
            assert observed[0] and observed[-1], (strategy, 'estimated execution')
            positions.append(dict(entry=entry, exit=exit_day, quantity=trade.quantity,
                risk=(width*100-credit)*trade.quantity, notional=cycle.spot*100*trade.quantity,
                unit_risk=width*100-credit, spot=cycle.spot, dte=cycle.dte,
                entry_equity=trade.entry_equity))
        nav_difference = float(np.max(np.abs(rebuilt-curves[strategy].to_numpy())))
        assert max(nav_difference, max_credit_error, max_exit_error) < 1e-5, strategy
        quantity_difference = equity_difference = 0.
        for pos in positions:
            entry = pos['entry']
            previous_equity = INITIAL if entry == 0 else rebuilt[entry-1]
            equity_difference = max(equity_difference, abs(previous_equity-pos['entry_equity']))
            held = [p for p in positions if p['entry'] < entry < p['exit']]
            active_risk = sum(p['risk'] for p in held)
            active_notional = sum(p['notional'] for p in held)
            tranches = max(1, int(np.ceil(pos['dte']/CADENCE_DAYS[cadence])))
            if risk_sizing:
                quantity = min(min(INITIAL*.05/tranches, max(INITIAL*.05-active_risk, 0))/pos['unit_risk'],
                    max(INITIAL*5-active_notional, 0)/(pos['spot']*100))
            else:
                quantity = min(max(previous_equity, 0)/tranches, max(previous_equity-active_notional, 0))/(pos['spot']*100)
            quantity_difference = max(quantity_difference, abs(quantity-pos['quantity']))
        assert quantity_difference < 1e-8 and equity_difference < 1e-5, strategy
        prior = np.r_[INITIAL, rebuilt[:-1]]
        returns = rebuilt/prior-1
        sd = returns.std(ddof=1)
        metrics = dict(cagr=(rebuilt[-1]/INITIAL)**(365.2425/(dates[-1]-dates[0]).days)-1,
            vol=sd*np.sqrt(252), sharpe=returns.mean()/sd*np.sqrt(252),
            max_drawdown=float(np.min(rebuilt/np.maximum.accumulate(np.r_[INITIAL, rebuilt])[1:]-1)))
        if strategy in parameters.index:
            for metric, value in metrics.items():
                assert abs(value-parameters.loc[strategy, metric]) < 1e-9, (strategy, metric)
        checks.append(dict(strategy_id=strategy, trades=len(trades),
            max_daily_nav_difference_dollars=nav_difference,
            max_entry_credit_difference_dollars=max_credit_error,
            max_trade_realized_pnl_difference_dollars=max_exit_error,
            max_entry_equity_difference_dollars=equity_difference,
            max_contract_quantity_difference=quantity_difference, **metrics))
    result = dict(status='passed',grid_rows=row_count,grid_parts=len(parts),
        validated_strategies=len(checks), validated_trades=len(ledger),
        method='Independent position-by-position daily NAV from prepared leg midpoints, execution costs, saved quantities and realized cashflows. No calls to the grid simulator or trade-path functions. Checks the first supported profit exit and independently recomputes entry sizing from prior-close NAV and the other live positions.',
        checks=checks)
    (output / 'Independent validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}), flush=True)
    return result


if __name__ == '__main__':
    from maturity_grid_data import prepare_paths
    validate(prepare_paths())
