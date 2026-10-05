# Fixed weekly 96/93, 97/94 and 98/95

January 5, 2024–September 18, 2026; 141 weekly trades, 678 daily marks. Short and long strikes are nearest listed puts at the specified ratios of cash SPX. Maturities are normally seven calendar days, holiday adjusted. Positions are resized weekly to 100% or 200% of current equity in SPX spot notional and held to PM cash settlement.

| Spread | Notional | CAGR | Daily Sharpe | Max drawdown | Worst day | Worst week | Ending $1M |
|---|---:|---:|---:|---:|---:|---:|---:|
| 96/93 | 100% | 1.69% | 0.8742 | -3.23% | -2.62% | -2.89% | $1,046,274 |
| 96/93 | 200% | 3.37% | 0.8727 | -6.44% | -5.25% | -5.78% | $1,093,592 |
| 97/94 | 100% | 3.51% | 1.5196 | -3.17% | -2.16% | -2.68% | $1,097,837 |
| 97/94 | 200% | 7.11% | 1.5167 | -6.31% | -4.34% | -5.36% | $1,203,838 |
| 98/95 | 100% | 5.67% | 1.7506 | -3.28% | -1.72% | -2.59% | $1,160,701 |
| 98/95 | 200% | 11.58% | 1.7483 | -6.52% | -3.48% | -5.17% | $1,344,584 |
| 99/96 | 100% | 4.77% | 1.0004 | -4.45% | -1.78% | -2.74% | $1,134,232 |

[Equity and drawdown chart](equity_drawdown.png)

Execution assumes 25% of each full bid/ask spread away from midpoint plus $1.50 per contract per leg. Costs scale with contracts. Cash earns zero; no additional financing or market-impact charge is modeled. The full bid/ask execution sensitivity is in summary.csv.

The 96/93 spread is a new requested comparison after the earlier research. Its quoted 96% put was already held as the hedge in 98/96, and its quoted 93% put as the hedge in 98/93. These exact entry contracts and daily quotes were reused to form and mark the new spread. No option prices were interpolated, and all 141 expirations reconcile to cash intrinsic. Provenance is retained by contract symbol and source candidate.

The 97/94 and 99/96 results reproduce the previous study. These are descriptive comparisons on the same previously examined period, not a new independent evaluation period.

[Summary including bid/ask sensitivity](summary.csv) · [Daily NAV](daily_portfolios.csv) · [Weekly trades](weekly_portfolios.csv) · [Calendar returns](calendar_returns.csv)
