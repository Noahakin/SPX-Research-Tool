# Leveraged 103/100 profit-target comparison

This uses the earlier monthly 103/100 spread near 60 DTE with the original 5% maximum-loss sizing. Results are option-only with no cash interest.

The highest CAGR among 5%-95% targets was the 95% target at 5.56%. The highest zero-cash Sharpe was the 15% target at 1.09.

| Profit target | CAGR | Volatility | Sharpe | Max drawdown | Win rate | Average exit DTE | Train CAGR | Validation CAGR | Test CAGR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15% | 3.67% | 3.35% | 1.09 | -2.96% | 93.2% | 42.9 | 3.88% | 3.62% | 3.10% |
| 25% | 4.16% | 3.80% | 1.09 | -4.85% | 88.1% | 36.1 | 4.19% | 3.75% | 4.47% |
| 40% | 3.85% | 5.05% | 0.77 | -5.81% | 77.1% | 25.7 | 3.48% | 4.36% | 4.46% |
| 60% | 4.12% | 5.75% | 0.73 | -11.27% | 68.6% | 15.9 | 3.91% | 3.84% | 5.02% |
| 80% | 4.83% | 5.94% | 0.82 | -10.63% | 63.6% | 9.4 | 5.29% | 3.38% | 4.92% |

Reconstruction check against the previously stored 25% curve: maximum absolute difference $0.000000.
