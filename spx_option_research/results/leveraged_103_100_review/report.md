# Earlier leveraged 103/100 result

The standout strategy was the monthly 103/100 SPX put spread entered near 60 DTE and closed when 25% of the initial credit had been earned. Results use option P&L only and no cash interest.

| Version | CAGR | Volatility | Zero-cash Sharpe | Max drawdown | Ending equity |
|---|---:|---:|---:|---:|---:|
| Earlier 5% max-loss sizing | 4.16% | 3.80% | 1.09 | -4.85% | $1,501,370 |
| 100% short-strike notional cap | 1.10% | 1.08% | 1.02 | -1.51% | $1,115,227 |

The earlier version used a configured 5% maximum-loss budget divided across the number of positions expected to overlap. Its aggregate short-strike notional averaged 160.2% of original capital while positions were open and peaked at 377.7%. Relative to then-current equity, the peak was 357.7%. Irregular monthly spacing caused concurrent defined maximum loss to peak at 5.00% of original capital.

The earlier version completed 118 trades, won 88.1%, and exited with 36.1 DTE on average. Its train, validation, and test CAGRs were 4.19%, 3.75%, and 4.47%.
