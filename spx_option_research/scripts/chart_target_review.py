from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path(__file__).resolve().parents[1]
market = pd.read_parquet(root / 'data/cache/market_data.parquet').sort_index()
daily = pd.read_parquet(root / 'data/cache/core_finalist_daily_mtm.parquet')
cash = market.risk_free_daily.fillna(0)
returns = pd.DataFrame(index=market.index)
for rule, label in [('short_delta_75', '75-delta exit'), ('dte_0', 'Hold to expiration')]:
    g = daily[(daily.base_strategy_id == 'core_dte30_sd10_points500') & (daily.exit_rule == rule)]
    pnl = g.groupby('mark_date').daily_option_pnl.sum()
    pnl.index = pd.to_datetime(pnl.index)
    option = pnl.reindex(market.index, fill_value=0) / 1_000_000
    returns[label] = option + cash
    if rule == 'short_delta_75':
        returns['Options only'] = option
returns['Cash interest only'] = cash
equity = (1 + returns).cumprod() * 100
equity.to_csv(root / 'results/target_review_equity.csv')
stats = []
for name in returns:
    r = returns[name]
    wealth = equity[name]
    stats.append({'version': name, 'cagr': (wealth.iloc[-1]/100)**(252/len(r))-1,
                  'max_drawdown': (wealth/wealth.cummax().clip(lower=100)-1).min(),
                  'ending_value_per_100': wealth.iloc[-1]})
pd.DataFrame(stats).to_csv(root / 'results/target_review_metrics.csv', index=False)
print(pd.DataFrame(stats).to_string(index=False))
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax, dd) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
elapsed = (equity.index-equity.index[0]).days/365.25
ax.fill_between(equity.index, 100*1.03**elapsed, 100*1.05**elapsed, color='#cbd5e1', alpha=.4, label='3–5% annual growth reference')
colors = {'75-delta exit': '#006b76', 'Hold to expiration': '#d28b26', 'Options only': '#8b5ca6', 'Cash interest only': '#64748b'}
for name in ['75-delta exit', 'Hold to expiration', 'Cash interest only', 'Options only']:
    ax.plot(equity.index, equity[name], label=name, color=colors[name], lw=2 if name=='75-delta exit' else 1.3)
for name in ['75-delta exit', 'Hold to expiration']:
    drawdown = equity[name]/equity[name].cummax().clip(lower=100)-1
    dd.plot(equity.index, drawdown, color=colors[name], lw=1.4)
ax.set_title('SPX put spread: strongest completed fit for a 3–5% total-return target', loc='left', weight='bold', pad=22)
ax.text(0, 1.015, '30 DTE • short 10-delta put • long put 500 points lower • weekly entries', transform=ax.transAxes, fontsize=10)
ax.set_ylabel('Growth of $100')
ax.legend(loc='upper left', frameon=False, fontsize=9)
dd.set_ylabel('Drawdown')
dd.yaxis.set_major_formatter(PercentFormatter(1))
for a in (ax, dd):
    a.grid(alpha=.2)
fig.text(.08, .025, 'Saved daily MTM, Sep 2016–Sep 2026; realistic execution costs; fractional contracts; configured 5% risk budget.\nTotal returns include historical cash interest. Curves compound daily P&L / initial capital, matching the research convention.', fontsize=9, color='#475569')
fig.tight_layout(rect=[0, .07, 1, 1])
fig.savefig(root / 'results/target_review_equity.png', dpi=180)
