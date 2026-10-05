# Monthly 103/100 with small far-OTM protective-put budgets

## Construction

The test uses the same matched third-Friday monthly entries as the earlier comparison, averaging 30.4 DTE. Every position is held to expiration, only one position is open at a time, short-strike notional equals current equity, and cash earns 0%.

For each 103/100 spread, the put budget is 1%, 2.5%, 5%, 7.5%, or 10% of the spread's entry credit. The hedge is one full put per vertical, selected as the highest strike below the 100% long leg whose realistic entry debit fits within the budget. This moves the hedge farther OTM when the budget is smaller.

## Results

| Strategy | CAGR | Volatility | Zero-cash Sharpe | Max drawdown | Avg. hedge OTM | Avg. hedge delta |
|---|---:|---:|---:|---:|---:|---:|
| 99/96, unhedged | 2.16% | 3.75% | 0.59 | -8.25% | — | — |
| 103/100, unhedged | 2.81% | 4.33% | **0.66** | **-8.87%** | — | — |
| 103/100 + 1% budget | 2.61% | 4.59% | 0.59 | -9.05% | 33.55% | 0.35 delta |
| 103/100 + 2.5% budget | **2.93%** | 5.17% | 0.59 | -9.34% | 23.88% | 1.12 delta |
| 103/100 + 5% budget | **2.93%** | 5.85% | 0.52 | -9.83% | 17.34% | 2.62 delta |
| 103/100 + 7.5% budget | 2.67% | 6.15% | 0.46 | -10.38% | 14.05% | 4.23 delta |
| 103/100 + 10% budget | 2.43% | 6.45% | 0.41 | -10.92% | 12.02% | 5.85 delta |

The 2.5% and 5% budgets produced the highest full-period CAGR, narrowly below the 3% target. Neither improved the unhedged 103/100 spread's Sharpe ratio or maximum drawdown.

## Downturn behavior

| Strategy | Average return when SPX fell at least 5% | Average return when SPX fell at least 10% |
|---|---:|---:|
| 99/96, unhedged | -2.35% | -2.30% |
| 103/100, unhedged | -1.28% | -1.41% |
| 103/100 + 1% budget | -1.30% | -1.42% |
| 103/100 + 2.5% budget | -0.64% | **+0.61%** |
| 103/100 + 5% budget | **-0.10%** | **+2.32%** |
| 103/100 + 7.5% budget | **+0.15%** | **+3.15%** |
| 103/100 + 10% budget | **+0.42%** | **+3.70%** |

There were nine monthly SPX declines of at least 5% and three of at least 10%.

## Dependence on the 2020 crash

The 2.5%, 5%, and 7.5% hedge puts generated a positive standalone hedge profit in only **one of 118 trades**: February 21 to March 20, 2020. The 10% budget was profitable in two trades. The 2.5% hedge cost 1.75 points on average and had an average expiration value of 1.80 points; excluding its single 2020 payoff, it was a recurring premium expense.

| Strategy | Pre-2020 CAGR | 2020 return | Post-2020 CAGR |
|---|---:|---:|---:|
| 99/96, unhedged | 1.90% | 5.98% | **1.66%** |
| 103/100, unhedged | **4.74%** | 6.38% | 1.12% |
| 103/100 + 2.5% budget | 4.21% | 12.55% | 0.63% |
| 103/100 + 5% budget | 3.67% | 17.67% | 0.15% |
| 103/100 + 7.5% budget | 3.14% | 19.97% | -0.34% |
| 103/100 + 10% budget | 2.93% | 21.28% | -0.82% |

## Interpretation

The **5% budget** is the most balanced version if crash protection is required. It retained a 2.93% full-period CAGR, made the average loss approximately flat during 5% monthly declines, and gained 2.32% during declines exceeding 10%. Its put averaged 17.3% OTM and 2.6 delta.

The **unhedged 103/100** remains the stronger choice for repeatable risk-adjusted performance. The apparent CAGR improvement from the 2.5% and 5% hedges depends on one exceptional 2020 payoff, while both produced higher volatility and deeper drawdowns over the full equity path.
