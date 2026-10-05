# Full ten-year fixed weekly SPX spread comparison

2016-09-23–2026-09-18: 521 scheduled weekly cycles, 520 traded and one common cash week. This is the full available archive, rather than the previous 2024–2026 comparison. Maturities are normally seven calendar days, holiday adjusted. Each entry uses the nearest listed short/long strikes at the stated cash-SPX ratios, and contracts are resized to 100% or 200% of current equity in SPX notional. Positions remain fixed through PM cash expiry.

| Spread | Notional | CAGR | Daily Sharpe* | Max drawdown* | Worst week | Ending $1M |
|---|---:|---:|---:|---:|---:|---:|
| 96/93 | 100% | 1.22% | 0.5240 | -8.07% | -2.96% | $1,128,478 |
| 96/93 | 200% | 2.39% | 0.5213 | -15.78% | -5.91% | $1,265,713 |
| 97/94 | 100% | 1.80% | 0.5986 | -10.14% | -3.05% | $1,195,359 |
| 97/94 | 200% | 3.53% | 0.5951 | -19.65% | -6.10% | $1,414,197 |
| 98/95 | 100% | 2.40% | 0.6145 | -11.09% | -2.81% | $1,266,755 |
| 98/95 | 200% | 4.68% | 0.6118 | -21.30% | -5.62% | $1,578,397 |
| 99/96 | 100% | 1.93% | 0.3910 | -12.98% | -2.86% | $1,210,855 |

[200% equity/drawdown chart](equity_drawdown_2x.png) · [100% chart](equity_drawdown_1x.png)

Costs: each leg fills 25% of its full bid/ask spread away from midpoint, plus $1.50 per contract per leg. Costs scale with contracts. Cash earns zero; no additional financing or market-impact charge is modeled. Sharpe uses daily marked returns and sqrt(252), while CAGR uses exact elapsed calendar time.

*One February 27, 2020 quote for the February 28 3240 put is demonstrably corrupt. For the held 97/94 spread, it creates a negative spread liability even though both puts are deeply in the money. The primary daily series estimates that put from the adjacent same-expiry 3235/3245 put midpoints. This is a disclosed estimate, not a recovered observed quote. The raw marks are preserved, and same-expiry call-put parity with a one-day zero-discount assumption provides a separate sensitivity. Cash expiration P&L, total return and CAGR do not depend on this interim estimate.

| Portfolio | Primary daily Sharpe | Raw-quote Sharpe | Call-parity sensitivity Sharpe |
|---|---:|---:|---:|
| 96/93 at 100% | 0.5240 | 0.5240 | 0.5240 |
| 96/93 at 200% | 0.5213 | 0.5213 | 0.5213 |
| 97/94 at 100% | 0.5986 | 0.5252 | 0.5984 |
| 97/94 at 200% | 0.5951 | 0.5264 | 0.5948 |
| 98/95 at 100% | 0.6145 | 0.6145 | 0.6145 |
| 98/95 at 200% | 0.6118 | 0.6118 | 0.6118 |
| 99/96 at 100% | 0.3910 | 0.3910 | 0.3910 |

The December 9–16, 2016 weekly PM expiry is absent from the archive and is held as cash for all strategies; its sessions remain in every metric. All other quoted weeks are traded, including quiet weeks when execution costs exceed premium. The original exploratory candidate file had removed some such weeks because it required positive credit under every fill assumption; the fixed ten-year study restores their exact quotes and charges the resulting losses.

## Calendar returns at 200% notional

| Year | 96/93 | 97/94 | 98/95 |
|---|---:|---:|---:|
| 2016 | 1.00% | 1.89% | 3.41% |
| 2017 | 1.32% | 2.40% | 5.04% |
| 2018 | -6.17% | -14.54% | -18.53% |
| 2019 | 4.44% | 8.28% | 12.01% |
| 2020 | -1.24% | 2.59% | 6.77% |
| 2021 | 7.25% | 11.40% | 15.67% |
| 2022 | 5.22% | 0.35% | -10.36% |
| 2023 | 3.55% | 6.08% | 6.89% |
| 2024 | 2.37% | 3.02% | 5.09% |
| 2025 | 1.54% | 7.14% | 11.78% |
| 2026 | 5.22% | 9.10% | 14.56% |

2016 begins September 23; 2026 ends September 18. Other rows are complete calendar years.

[Summary and full bid/ask execution sensitivity](summary.csv) · [All marking sensitivities](marking_sensitivity.csv) · [Daily NAV](daily_portfolios.csv) · [Weekly trades](weekly_portfolios.csv) · [Calendar returns](calendar_returns.csv) · [Quote audit](data_quality_audit.md)
