# Entry premium and protective-put cost

These averages use the same September 22, 2016 through September 22, 2026 PM-settled SPX archive and execution model as the sequential backtest: 25% of the bid/ask spread away from mid plus $1.50 per option contract. One SPX option point equals $100 per contract.

The 103/100 and 104/101 labels are percentages of spot. The average vertical was approximately 124 SPX points wide because its width was 3% of SPX, rather than three SPX points.

## Vertical entry credit

| Target DTE | 103/100 average credit | Credit / width | Credit / short notional | 104/101 average credit | Credit / width | Credit / short notional |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 81.62 points | 66.0% | 1.92% | 97.61 points | 78.8% | 2.27% |
| 21 | 76.77 points | 62.1% | 1.81% | 92.12 points | 74.5% | 2.15% |
| 30 | 71.60 points | 57.9% | 1.68% | 85.78 points | 69.4% | 2.00% |
| 45 | 66.83 points | 53.8% | 1.57% | 78.87 points | 63.7% | 1.84% |
| 60 | 62.45 points | 51.2% | 1.49% | 73.18 points | 60.1% | 1.73% |
| 75 | 60.90 points | 48.9% | 1.43% | 70.20 points | 56.6% | 1.63% |

For example, the average 21 DTE 103/100 spread collected about **$7,677** per one-lot against an average **$12,426** spread width. The average 21 DTE 104/101 collected about **$9,212** against an average **$12,417** width. Much of this credit is intrinsic value, so it is not equivalent to expected return.

## Same-expiration protective puts at 21 DTE

| Put strike below spot | Average delta | Average debit | Cost / 103/100 credit | Cost / 104/101 credit |
|---:|---:|---:|---:|---:|
| 1% | 37.7 | 45.07 points | 64.7% | 55.0% |
| 2% | 28.8 | 34.63 points | 50.7% | 43.2% |
| 3% | 22.1 | 26.96 points | 40.2% | 34.4% |
| 4% | 17.2 | 21.22 points | 32.3% | 27.6% |
| 5% | 13.5 | 16.89 points | 26.1% | 22.4% |
| 7.5% | 7.7 | 10.01 points | 16.1% | 13.9% |
| 10% | 4.7 | 6.41 points | 10.6% | 9.2% |
| 12.5% | 3.0 | 4.39 points | 7.4% | 6.4% |
| 15% | 2.0 | 3.18 points | 5.4% | 4.7% |

## Same-expiration protective puts at 45 DTE

| Put strike below spot | Average delta | Average debit | Cost / 103/100 credit | Cost / 104/101 credit |
|---:|---:|---:|---:|---:|
| 1% | 41.3 | 74.52 points | 119.4% | 103.7% |
| 2% | 34.6 | 62.63 points | 101.3% | 88.2% |
| 3% | 29.1 | 52.97 points | 86.5% | 75.4% |
| 4% | 24.5 | 45.03 points | 74.1% | 64.8% |
| 5% | 20.8 | 38.48 points | 63.9% | 55.9% |
| 7.5% | 13.9 | 26.37 points | 44.7% | 39.2% |
| 10% | 9.5 | 18.51 points | 31.9% | 28.1% |
| 12.5% | 6.6 | 13.33 points | 23.3% | 20.6% |
| 15% | 4.7 | 9.87 points | 17.4% | 15.4% |

The percentage-of-credit figures are averages of the cost ratio on each entry date, rather than ratios of the two full-sample averages. This preserves the actual hedge budget required through different volatility regimes.

At 21 DTE, spending approximately 5%–10% of the credit bought a put roughly 10%–15% below spot. At 15 DTE, the same budget generally bought a put around 7.5%–15% below spot. At 30 DTE, a 15% OTM put consumed about 8%–9% of credit. At 45 DTE and beyond, even a 15% OTM put averaged more than 10% of credit, so the earlier fixed-budget backtests usually purchased a fractional hedge ratio.
