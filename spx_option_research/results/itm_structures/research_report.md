# ITM SPX spreads and butterfly research

## Scope

The test covers SPX option quotes from September 22, 2016 through September 22,
2026. It evaluates 105,945 trade candidates and 2,916 complete
strategy-and-exit combinations.

Vertical spreads include 100/97, 101/98, 102/99, 103/100, 104/101, and
105/102 strike-percent pairs. Delta-selected shorts from 50 through 80 delta
are also paired with a long put approximately 3% of spot lower. Target
expirations are 15, 21, 30, 45, 60, and 75 DTE, with both weekly and monthly
entry schedules.

Exit rules include expiration, 14 DTE, 7 DTE, half of the original life,
25%/50%/75% of maximum profit, and short-put delta falling to 40 or 25.

Butterfly variants include long and short symmetric 3x3 put flies, long and
short broken 3x6 put flies, and long and short four-strike 2x2x2 put condors.

All results use $1 million initial capital, a 5% total concurrent maximum-loss
budget, continuous contract equivalents, PM-settled options, realistic entry
and exit costs, and no return on cash. Longer-dated weekly positions divide
the risk budget across their expected overlapping tranches.

## Main result

The strongest risk-adjusted result was the **103/100 monthly 60-DTE vertical
with a 25% maximum-profit exit**.

| Metric | Result |
|---|---:|
| CAGR | 4.16% |
| Annualized volatility | 3.80% |
| Zero-cash Sharpe | 1.09 |
| Maximum drawdown | -4.85% |
| Win rate | 88.14% |
| Trades | 118 |
| Average short-put entry delta | 71.3 |
| Average exit DTE | 36.1 |
| Train CAGR | 4.19% |
| Validation CAGR | 3.75% |
| Test CAGR | 4.47% |

The neighboring 102/99 monthly 60-DTE vertical with a 50% profit exit also
held up well: 4.02% CAGR, 4.05% volatility, -5.18% maximum drawdown, and 5.12%
test-period CAGR. This makes the 102%-104% short-strike region more credible
than a single isolated optimum.

## Effect of moving farther ITM

At the same risk budget, the 105/102 monthly 60-DTE spread with a 25% profit
exit produced 5.77% CAGR, 7.19% volatility, and a -5.32% maximum drawdown. It
exceeded the 3%-5% objective and had weaker risk-adjusted performance than the
103/100 result. Holding the 105/102 spread to expiration increased CAGR to
7.66% but expanded maximum drawdown to -30.17%.

The result supports closing deep-ITM spreads early. The 25% profit rule removes
much of the prolonged downside exposure while retaining enough premium to meet
the return objective.

## Butterflies and four-leg structures

Only seven butterfly or condor variants met the 3%-5% full-period target while
remaining profitable in all three chronological samples.

The best observed four-leg result was a **short 101/99/97/95 put condor**,
entered monthly near 21 DTE and closed at 25% of maximum profit. Its legs are
short the 101% put, long the 99% put, long the 97% put, and short the 95% put.
It produced 3.59% CAGR, 6.19% volatility, a 0.60 Sharpe ratio, and a -9.18%
maximum drawdown.

A weekly 30-DTE version held to expiration produced 3.68% CAGR and a -6.25%
maximum drawdown, but its test-period CAGR fell to 0.99%. The butterfly and
condor results were less stable than the best vertical spreads.

## Interpretation

Monthly entries dominated the top 50 qualifying vertical results. The strongest
expiration region was 60 DTE, followed by 45 DTE. Seventy-five-DTE positions
generally earned too little after distributing the risk budget across concurrent
positions, while 15-DTE weekly positions had substantially larger drawdowns.

The preferred implementation from this grid is therefore:

1. Enter once per month near 60 DTE.
2. Sell the put nearest 103% of spot and buy the put nearest 100% of spot.
3. Close when the spread earns 25% of its maximum possible profit.
4. Divide a 5% total maximum-loss budget across overlapping positions.

This is an exploratory search across 2,916 variants. The chronological splits
show stability, but the same archive was used to compare and rank the grid, so
the reported leader is not an untouched out-of-sample result. A paper-trading
period or a locked forward test is appropriate before implementation.
