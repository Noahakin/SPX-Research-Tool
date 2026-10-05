# Requested SPX premium study

The source archive runs from September 22, 2016 through September 22, 2026. The 1M tests use matched third-Friday-to-third-Friday PM-settled SPXW positions held to expiration. There are 118 complete matched 1M cycles, from October 21, 2016 through September 18, 2026, with an average tenor of 30.4 calendar days. Strike targets and expiration payoffs use the cash SPX close from Yahoo Finance; the option archive's expiration-specific `underlying_price` field is not used as cash spot.

All percentages and dollar equivalents use SPX spot notional at entry. For example, $1 million of notional means `contracts = $1,000,000 / (SPX × 100)`, with fractional contracts used solely to make results comparable. Annualized rates equal the sum of each normalized trade amount divided by total contract-years. The realistic execution case fills each leg one-quarter of the full quoted bid/ask spread away from mid and charges $1.50 per contract per leg. Expiration uses intrinsic value at that day's cached cash SPX close. Results are option-only and exclude interest on collateral, taxes, and settlement fees.

## Short put spreads

| Structure | Annual premium collected | Per $1M/year | Annual realized P&L | Per $1M/year | Natural-fill P&L | Compounded P&L CAGR | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1M 101/98 | 12.31% | $123,148 | 2.48% | $24,805 | 2.33% | 2.40% | 69.5% |
| 1M 102/99 | 16.08% | $160,753 | 2.78% | $27,770 | 2.54% | 2.69% | 62.7% |
| 1M 103/100 | 20.23% | $202,326 | 2.98% | $29,774 | 2.55% | 2.89% | 55.1% |

## Protection premium and P&L

| Structure | Annual premium paid | Per $1M/year | Annual realized P&L | Per $1M/year | P&L CAGR | Profitable trades |
|---|---:|---:|---:|---:|---:|---:|
| 1M 95/85 buffer | 6.00% | $60,022 | -2.83% | -$28,321 | -2.87% | 6.8% |
| 1M 97.5/92.5 buffer | 7.35% | $73,540 | -2.51% | -$25,139 | -2.54% | 13.6% |
| 3M 90% long put | 4.40% | $44,020 | -2.84% | -$28,412 | -2.90% average | 6.0% |

The 3M premium and arithmetic P&L estimates use 117 monthly entry observations with an average tenor of 91.3 days. Dividing the normalized results by total contract-years makes them phase-neutral annual estimates for continuously maintaining one 3M 90% put; it does not assume twelve overlapping puts. The three actual non-overlapping quarterly roll cohorts produced realistic P&L CAGRs from -3.63% to -1.61%, averaging -2.90%.

Premium-cost sensitivity from mid-market to natural bid/ask was 5.93%–6.07% for the 95/85 buffer, 7.25%–7.45% for the 97.5/92.5 buffer, and 4.38%–4.42% for the 3M put.

“Premium collected” is entry cash flow and includes intrinsic value in the 101%–103% short strikes. Realized P&L subtracts the cash-settled expiration payoff, which is why it is much smaller than gross premium.
