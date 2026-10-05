# Monthly 99/96 versus 103/100 with a credit-funded protective put

## Construction

Both strategies enter on the same third-Friday monthly roll date and expire on the following third-Friday PM settlement, averaging 30.4 DTE. Each strategy holds one position at a time and sizes the short-strike notional to current portfolio equity. Cash earns 0%.

- **Baseline:** sell the 99% strike put and buy the 96% strike put.
- **Hedged strategy:** sell the 103% strike put and buy the 100% strike put. Calculate the entry-credit surplus over the matched 99/96 spread, then buy the highest-strike put below the 100% long leg whose full one-for-one debit fits within that surplus.
- Execution uses 25% of the quoted bid/ask spread away from mid plus $1.50 per option contract.
- Both strategies hold every position to expiration.

## Results

| Metric | 99/96, no hedge | 103/100, surplus-funded put |
|---|---:|---:|
| Trades | 118 | 118 |
| CAGR | **2.16%** | **-0.63%** |
| Annualized volatility | 3.75% | 7.86% |
| Zero-cash Sharpe | 0.59 | -0.04 |
| Maximum drawdown | -8.25% | -26.03% |
| Ending equity from $1 million | $1,238,570 | $938,738 |
| Win rate | 82.2% | 52.5% |
| Average trade return | 0.187% | -0.026% |

The 103/100 spread collected an average **71.42 points**, versus **26.55 points** for the 99/96 spread, creating **44.87 points** of additional credit. The selected protective put cost **40.58 points** on average, using **90.7%** of the surplus. After the hedge, the package retained an average **30.83-point net credit**, about 4.29 points more than the baseline.

The selected put averaged **3.46% below spot** and **33.2 delta**. It finished in the money in 16.9% of trades and produced a positive standalone hedge profit in 14.4%. Its average expiration value was 26.23 points against a 40.58-point cost, for an average hedge loss of 14.35 points per trade.

## Downturn behavior

| Monthly SPX outcome | Observations | 99/96 average trade return | Hedged 103/100 average trade return |
|---|---:|---:|---:|
| SPX down at least 5% | 9 | -2.35% | **+4.34%** |
| SPX down at least 10% | 3 | -2.30% | **+9.31%** |

The construction achieved the intended downturn payoff. Its February–March 2020 trade gained roughly $222,375 on the contemporaneous portfolio sizing, producing the large upward step in the equity curve. The cost was persistent premium drag in normal and rising markets.

## Interpretation

Using the entire credit difference bought a relatively expensive near-the-money put. The 103/100 spread then had a much lower probability of expiring above its short strike than the 99/96 baseline, while approximately 91% of its extra credit was spent on the hedge. The package delivered strong crash convexity but did not preserve the baseline's long-run return.

The two alternatives therefore solve different objectives. The 99/96 spread was superior for long-run option return and drawdown in this sample. The 103/100 package was superior during large monthly declines, but the insurance cost overwhelmed that benefit over the full period.
