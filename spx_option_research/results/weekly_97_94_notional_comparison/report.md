# Weekly 97/94: 200% notional

January 5, 2024–September 18, 2026: 141 weekly trades, 678 daily marks. Contracts are set to 100% or 200% of current equity in cash-SPX notional at each weekly entry, then held fixed until expiry. Strike targets remain 97%/94%. Friday-to-Friday maturities are normally seven calendar days, with holiday adjustments.

| Portfolio | CAGR | Daily Sharpe | Max drawdown | Worst day | Worst week | Ending $1M |
|---|---:|---:|---:|---:|---:|---:|
| 97/94 at 100% | 3.51% | 1.5196 | -3.17% | -2.16% | -2.68% | $1,097,837 |
| 97/94 at 200% | 7.11% | 1.5167 | -6.31% | -4.34% | -5.36% | $1,203,838 |
| 99/96 at 100% | 4.77% | 1.0004 | -4.45% | -1.78% | -2.74% | $1,134,232 |

[Daily equity and drawdown chart](equity_drawdown.png)

Execution assumes each leg fills 25% of its full bid/ask spread away from midpoint plus $1.50 per contract per leg. All costs scale with position size. Cash earns zero, and no additional borrowing or market-impact charge is modeled. This is SPX notional exposure; the theoretical maximum loss per 200%-notional 3%-wide spread is approximately 6% of entry equity less net premium.

At full bid/ask execution, 200% notional produces 6.92% CAGR, 1.4765 daily Sharpe and -6.34% maximum drawdown.

The 100% results reconcile to the previous study. Increasing notional does not double Sharpe: both expected P&L and fluctuations scale, while weekly compounding and intraperiod changes in equity produce small deviations from exact proportionality.

Data: [summary](summary.csv), [daily NAV](daily_portfolios.csv), [weekly trades](weekly_portfolios.csv), [calendar returns](calendar_returns.csv).
