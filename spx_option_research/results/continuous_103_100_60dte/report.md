# Continuous 103/100 60-DTE sequence

This version holds one 103/100 SPX put spread at a time, targets 60 DTE, closes when 25% of the initial credit has been earned or at the final available mark, and immediately opens a new spread using the same day's closing option surface. Every entry is sized to 100% of current portfolio short-strike notional. Results include realistic execution costs and no cash interest. The next-session version is included as a conservative execution sensitivity.

| Version | CAGR | Volatility | Sharpe | Max drawdown | Ending equity | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Continuous same-day re-entry | 1.98% | 2.89% | 0.69 | -5.88% | $1,215,756 | 126 |
| Continuous next-session re-entry | 1.54% | 2.88% | 0.54 | -6.73% | $1,164,309 | 117 |
| Calendar-month entries | 3.48% | 3.40% | 1.02 | -4.64% | $1,406,774 | 118 |

The same-day sequence was invested on 98.1% of trading days. Notional averaged 100.4% of equity while invested and peaked at 101.6%; no more than 1 position was open. Defined maximum loss peaked at 1.95% of equity. Average holding time was 28.5 calendar days, and the average exit occurred with 31.0 DTE remaining.

Train, validation, and test CAGRs for the same-day sequence were 2.69%, 1.14%, and 0.70%.
