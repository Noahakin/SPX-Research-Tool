# Quarterly SPX ratio-package backtest

The package is rolled on the January, April, July, and October monthly expiration dates into a PM-settled expiration near 90 DTE. One package unit is sized to 100% of current SPX notional using continuous package units. Positions are held to expiration; the final open quarter is liquidated at the last available realistic mark. Cash earns 0%.

Fixed legs are long 2 ATM puts, short 7 puts near 95% of entry SPX, and long 5 puts near 94%. Because the original description did not specify the deep-call strike or upside cap, the test covers deep calls from 50%-80% of spot and call caps from 105%-120%.

## Baseline: 60% deep call and 110% cap

| Metric | Package | SPX price |
|:---|---:|---:|
| CAGR | 6.12% | 11.92% |
| Annualized volatility | 12.40% | 16.85% |
| Zero-cash Sharpe | 0.54 | 0.76 |
| Maximum drawdown | -19.35% | -25.11% |
| Ending value from $1 million | $1,338,687 | $1,743,772 |

The baseline completed 20 quarterly entries, won 70.0%, and averaged 1.59% per quarter. Average selected tenor was 91.3 days.

The package cost an average 41.5% of SPX notional. Its average worst expiration payoff was -36.5% of notional and its average maximum expiration profit was 8.6%. From the 94% strike down to the 60% deep-call strike, the package retains approximately +1 terminal delta and continues losing as SPX falls. Below the deep-call strike, its expiration value flattens near 5% of initial notional. The structure therefore softens an ordinary crash but only reaches its ultimate floor after an extreme decline.

## Observed quarterly return zones

| SPX quarter | Observations | Package average | SPX average |
|:---|---:|---:|---:|
| Below -6% | 3 | -7.23% | -10.85% |
| -6% to -5% | 0 | N/A | N/A |
| -5% to 0% | 2 | 0.69% | -2.24% |
| 0% to +10% | 13 | 2.68% | 4.06% |
| Above +10% | 2 | 8.61% | 15.76% |

## Parameter sensitivity

The highest full-period Sharpe in the 16-combination sensitivity grid was **call50_cap120**, at 7.89% CAGR, 0.63 Sharpe, and -20.08% maximum drawdown. This is an in-sample sensitivity result rather than an independently validated selection.

Execution assumes fills 25% of the quoted bid/ask spread away from mid plus $1.50 per contract per leg. Daily equity uses midpoint marks, expiration uses intrinsic value, and the final incomplete trade uses realistic liquidation. Results exclude dividends, cash interest, taxes, and margin financing.