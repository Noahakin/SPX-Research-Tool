# Ranking of short put-spread variants

The ranking uses the research score already applied in the strategy searches: 15% train Sharpe, 25% validation Sharpe, 50% test Sharpe, and 10% full-period zero-cash Sharpe. Eligibility requires a 3%-5% full-period CAGR and positive CAGR in the train, validation, and test samples. Duplicate reports of an identical rule and sizing method are removed; genuinely different sizing rules remain separate variants.

## Overall top ten

| Rank | Strategy | Sizing | CAGR | Sharpe | Max DD | Test CAGR | Score |
|---:|:---|:---|---:|---:|---:|---:|---:|
| 1 | 103/100, monthly 60 DTE, profit 25 | 5% concurrent max-loss budget; overlap allowed | 4.16% | 1.09 | -4.85% | 4.47% | 1.340 |
| 2 | 103/100, monthly 60 DTE, 25% target; Fixed $1 million per monthly entry | Fixed $1 million per monthly entry | 3.02% | 1.00 | -4.27% | 3.73% | 1.328 |
| 3 | 103/100, monthly 60 DTE, 25% target; 100% of current equity per monthly entry | 100% of current equity per monthly entry | 3.48% | 1.02 | -4.64% | 4.83% | 1.320 |
| 4 | delta55_3wide, monthly 60 DTE, profit 50 | 5% concurrent max-loss budget; overlap allowed | 3.21% | 0.89 | -7.83% | 4.64% | 1.245 |
| 5 | delta55_3wide, monthly 60 DTE, short delta 40 | 5% concurrent max-loss budget; overlap allowed | 3.04% | 1.06 | -5.87% | 4.03% | 1.234 |
| 6 | 101/98, monthly 60 DTE, profit 50 | 5% concurrent max-loss budget; overlap allowed | 3.22% | 0.79 | -8.28% | 4.83% | 1.220 |
| 7 | 102/99, monthly 60 DTE, profit 50 | 5% concurrent max-loss budget; overlap allowed | 4.02% | 0.99 | -5.18% | 5.12% | 1.211 |
| 8 | 103/100, monthly 60 DTE, 15% profit target | 5% concurrent max-loss budget; overlap allowed | 3.67% | 1.09 | -2.96% | 3.10% | 1.202 |
| 9 | delta75_3wide, monthly 60 DTE, profit 25 | 5% concurrent max-loss budget; overlap allowed | 4.22% | 0.92 | -5.11% | 6.35% | 1.189 |
| 10 | 103/100, monthly 60 DTE, 20% profit target | 5% concurrent max-loss budget; overlap allowed | 3.64% | 1.00 | -4.84% | 3.91% | 1.178 |

None of the protective-put variants reached the overall top ten. The strongest top-ten entries use monthly 60-DTE spreads and allow overlapping positions, so their return is not directly available without leverage.

## Unlevered references

The highest test-weighted unlevered rule was **101/98, sequential 21 DTE, profit 90** at 3.97% CAGR, 0.80 Sharpe, -9.23% maximum drawdown, and 5.58% test CAGR.

The strongest full-period risk-adjusted unlevered rule was **104/101, sequential 21 DTE, profit 90** at 4.36% CAGR, 1.00 Sharpe, and -7.53% maximum drawdown.

## Best hedged runner-up by total-return robustness

| Strategy | CAGR | Sharpe | Max DD | Test CAGR | Down 5% trade |
|:---|---:|---:|---:|---:|---:|
| 101/98, sequential 21 DTE, profit 90; + 20-delta put; 5% credit budget | 3.54% | 0.75 | -9.51% | 4.90% | -1.51% |

## Best hedge with positive average returns in 5% SPX declines

| Strategy | CAGR | Sharpe | Max DD | Test CAGR | Down 5% trade |
|:---|---:|---:|---:|---:|---:|
| 101/98, sequential 15 DTE, profit 80; + 20-delta put; 20% credit budget | 3.21% | 0.63 | -10.96% | 3.45% | 0.08% |

The full-credit-funded monthly hedge, small far-OTM monthly puts, and 5%-budget put-spread overlays were also reviewed. They did not make the ranking because their full-period Sharpe, drawdown, or chronological stability was weaker. Metrics use option P&L only and zero cash interest.