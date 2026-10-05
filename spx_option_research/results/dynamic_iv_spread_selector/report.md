# Dynamic 1M SPX put-spread IV/RV selector

This test sells one of six three-percentage-point-wide put spreads each month: 98/95, 99/96, 100/97, 101/98, 102/99, or 103/100. Here “3-wide” means the target strikes differ by 3% of entry SPX spot, not three SPX index points. It uses the same matched third-Friday PM-settled SPXW rolls, cash-index strikes and settlement, expiration holding period, and realistic execution assumption as the requested premium study. Every entry targets SPX cash-spot notional equal to 100% of current portfolio equity; fractional contracts make the comparison exact.

The signal defines a spread's current IV as the option archive's implied volatility for its short put. Trailing realized volatility is the annualized sample standard deviation of the latest 21 SPX cash close-to-close log returns, including the entry-date close. Each candidate receives `short IV / 21-day realized vol - 1`, and the candidate with the largest value is sold. The long-leg IV is retained in the candidate file but does not enter the signal.

| Portfolio | CAGR | Sharpe | Annualized volatility | Maximum drawdown | Annual arithmetic P&L | Annual premium | Ending value of $1 | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dynamic highest IV/RV deviation | 1.41% | 0.47 | 3.10% | -6.38% | 1.46% | 5.75% | $1.149 | 83.9% |
| Fixed 99/96 every month | 1.65% | 0.50 | 3.45% | -8.48% | 1.71% | 7.30% | $1.176 | 80.5% |

The dynamic rule changed CAGR by -0.24% per year and Sharpe by -0.03 relative to fixed 99/96. It selected: 98/95: 118, 99/96: 0, 100/97: 0, 101/98: 0, 102/99: 0, 103/100: 0. Its selected short-leg IV exceeded trailing realized volatility in 79.7% of months.

## Fixed-candidate diagnostic

| Spread | Selected months | Mid CAGR | Mid Sharpe | Realistic CAGR | Realistic Sharpe |
|---|---:|---:|---:|---:|---:|
| 98/95 | 118 | 1.52% | 0.51 | 1.41% | 0.47 |
| 99/96 | 0 | 1.77% | 0.53 | 1.65% | 0.50 |
| 100/97 | 0 | 2.09% | 0.56 | 1.95% | 0.52 |
| 101/98 | 0 | 2.57% | 0.64 | 2.40% | 0.60 |
| 102/99 | 0 | 2.95% | 0.69 | 2.69% | 0.63 |
| 103/100 | 0 | 3.33% | 0.75 | 2.89% | 0.65 |

Because all six candidates share the same realized-volatility denominator on a given entry date, ranking by percentage deviation is exactly the same as ranking by short-leg IV. Realized volatility shows whether the selected option was rich relative to the recent SPX path, but it cannot change which candidate wins this cross-sectional rule. Put skew therefore has a large influence on the chosen spread.

Sharpe uses the non-overlapping monthly option returns, a zero cash rate, and a square-root-of-12 annualization. CAGR compounds each realized monthly return over the calendar time from the first entry through the last expiration. Results are option-only and exclude collateral interest, taxes, and settlement fees.
