# 103/100 notional-capped optimization

## Definition and scope

This test defines notional as the short put strike multiplied by 100 and by the
number of contracts. Aggregate open short-strike notional is capped at 100% of
the original $1 million portfolio.

The underlying structure is the 103/100 SPX put spread selected near 60 DTE.
Entry intervals are one, two, three, four, six, and eight weeks, plus a third-
Friday monthly schedule. The 100% notional allowance is divided equally among
the maximum number of positions that could overlap if every position remains
open until expiration:

| Entry schedule | Maximum overlapping trades | Notional per trade |
|---|---:|---:|
| Every week | 9 | 11.11% |
| Every 2 weeks | 5 | 20.00% |
| Every 3 weeks | 3 | 33.33% |
| Every 4 weeks | 3 | 33.33% |
| Monthly | 3 | 33.33% |
| Every 6 weeks | 2 | 50.00% |
| Every 8 weeks | 2 | 50.00% |

Profit exits from 5% through 95% of maximum profit are tested in five-point
increments, along with holding to expiration. Results use realistic execution,
continuous contracts, and no cash interest.

## Best return configuration

The preferred return-maximizing setting is **one entry every three weeks with
an 85% profit target**.

| Metric | Result |
|---|---:|
| Option-only CAGR | 2.05% |
| Annualized volatility | 2.32% |
| Zero-cash Sharpe | 0.88 |
| Maximum drawdown | -4.45% |
| Win rate | 66.26% |
| Average exit DTE | 7.2 |
| Trades | 163 |
| Train CAGR | 2.46% |
| Validation CAGR | 1.69% |
| Test CAGR | 1.19% |
| Maximum actual notional | 100% |

The 95% target produced a nearly identical 2.05% CAGR, but its maximum drawdown
was larger at -5.02% and its Sharpe ratio was lower. The 85% exit is therefore
the cleaner choice.

## Best risk-adjusted configuration

The highest full-period Sharpe was the monthly schedule with a 15% profit
target. It produced 0.98% CAGR, 0.92% volatility, a 1.07 Sharpe ratio, and a
-1.06% maximum drawdown. Because positions closed quickly, actual aggregate
notional reached only 66.7% even though the scheduled cap was 100%.

The monthly 25% exit was more consistent across chronological samples and
produced 1.10% CAGR with a -1.51% maximum drawdown.

## Implication for the return target

None of the 140 notional-capped combinations reached the 3%-5% option-only
return objective. The best observed CAGR was 2.05%.

Because contract P&L scales linearly in this backtest, the every-three-week 85%
exit would require approximately 153% short-strike notional to reach a 3% CAGR
and approximately 280% notional to reach a 5% CAGR over this sample. Both exceed
the requested 100% cap.

The earlier 4.16% result for the monthly 25% exit used a 5% maximum-loss budget,
which translated into several times portfolio short-strike notional across the
overlapping 60-DTE positions. Applying the explicit 100% notional cap reduces
that result to roughly 1.10% CAGR.

The three-week result also weakened across the chronological samples, from
2.46% in training to 1.19% in the final test period. It should therefore be
treated as an upper-end historical estimate rather than a reliable 2% floor.
