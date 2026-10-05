# Sequential unlevered ITM put-spread rerun

## Conclusion

The best risk-adjusted result in the 3%–5% target range was the **104/101 spread entered near 21 DTE and closed at 90% of its initial credit**. It returned **4.36% annualized**, with **4.38% volatility**, a **1.00 zero-cash Sharpe ratio**, and a **-7.53% maximum drawdown**.

The strongest simple hold-to-expiration version was the **104/101 spread entered near 45 DTE**. It returned **3.26% annualized**, with **3.34% volatility**, a **0.98 zero-cash Sharpe ratio**, and a **-4.98% maximum drawdown**. This is the cleaner choice when lower drawdown and fewer management decisions matter more than maximizing return within the target band.

For the previously selected **103/100** structure, the best high-return result was **21 DTE with a 90% profit target** at **4.22% CAGR** and **-5.99% maximum drawdown**. The best expiration-only result was **45 DTE held to expiration** at **3.33% CAGR** and **-5.76% maximum drawdown**.

Protective puts created crash convexity, but they did not improve the full-period risk-adjusted result. The most practical tail-positive candidate was the **101/98, 15 DTE, 80% profit-target spread plus a 20-delta put funded with 20% of the spread credit**. It retained a **3.21% CAGR** and averaged **+0.08%** on trades in which SPX fell at least 5%, but only **6 of 15** such trades were profitable and the median was **-0.52%**. The unhedged version returned **4.20%**.

## Test design

- Data: PM-settled SPX option surfaces from **September 22, 2016 through September 22, 2026**.
- Core spreads: **100/97, 101/98, 102/99, 103/100, 104/101, and 105/102**, where strikes are selected near the stated percentages of spot and are always approximately three percentage points wide.
- Entry tenors: **15, 21, 30, 45, 60, and 75 DTE**, using the nearest available expiration within four days.
- Exits: hold to expiration; profit targets from **10% through 90%** in 10-point increments; half-life; 14 DTE; and 7 DTE.
- Protective puts: same expiration, below the vertical's long strike, selected near **5, 10, 15, or 20 delta**. Budgets were **5%, 10%, 15%, or 20%** of the core spread credit, capped at one hedge put per vertical.
- No overlap: only one position is open at a time. The next position enters on the first populated trading day after the prior position exits.
- No leverage: the short-strike notional of the open spread is limited to current equity. Contracts are recalculated from current equity at every entry.
- Execution: 25% of the quoted bid/ask spread away from mid plus **$1.50 per contract per leg**.
- Cash return: **0%**.
- Scale: **82,242** daily core candidates, **468** sequential core variants, and **7,956** core-plus-overlay strategies. The screen used realized exits; **61 finalists** were then repriced with daily marks for volatility, drawdown, and equity curves.

## Selected exact daily-marked results

| Strategy | CAGR | Volatility | Zero-cash Sharpe | Max drawdown | Trades |
|---|---:|---:|---:|---:|---:|
| 104/101, 21 DTE, 90% profit target | 4.36% | 4.38% | 1.00 | -7.53% | 171 |
| 104/101, 45 DTE, hold to expiration | 3.26% | 3.34% | 0.98 | -4.98% | 76 |
| 103/100, 21 DTE, 90% profit target | 4.22% | 4.89% | 0.87 | -5.99% | 173 |
| 103/100, 45 DTE, hold to expiration | 3.33% | 4.03% | 0.83 | -5.76% | 76 |
| 103/100, 30 DTE, 60% profit target | 3.31% | 3.87% | 0.86 | -7.46% | 140 |
| 103/100, 30 DTE, 60% target + 5-delta put using 10% of credit | 3.20% | 6.04% | 0.55 | -9.87% | 140 |
| 103/100, 21 DTE, hold to expiration | 3.61% | 4.94% | 0.74 | -6.42% | 167 |
| 103/100, 21 DTE, hold + 5-delta put using 5% of credit | 3.24% | 5.63% | 0.59 | -7.31% | 167 |
| 101/98, 15 DTE, 80% profit target | 4.20% | 5.57% | 0.77 | -12.06% | 257 |
| 101/98, 15 DTE, 80% target + 20-delta put using 20% of credit | 3.21% | 5.28% | 0.63 | -10.96% | 257 |

## What the put overlays did

The **103/100 30 DTE, 60% profit-target strategy with a 5-delta put and 10% budget** averaged **+0.13%** across 12 trades where SPX fell at least 5%. That average is driven by one February–March 2020 trade, which gained **16.75%**. The other 11 qualifying downturn trades all lost money. Its out-of-sample test CAGR was only **0.33%**, versus **2.29%** for the unhedged core.

The **101/98 15 DTE, 80% profit-target strategy with a 20-delta put and 20% budget** spread its protection more evenly. It averaged **+0.08%** in 15 SPX declines of at least 5% and **+1.56%** in three declines of at least 10%. It reduced maximum drawdown from **-12.06% to -10.96%**, while lowering CAGR from **4.20% to 3.21%**.

No tested overlay produced positive returns in most 5% downturn trades while also retaining a 3%–5% full-period CAGR. The strongest result was positive in 40% of those observations. The puts therefore behaved as intermittent crash insurance, rather than a reliable source of profit during ordinary corrections.

## Stability and interpretation

The 104/101 21 DTE, 90% profit-target result had train, validation, and test CAGRs of **4.35%, 6.52%, and 2.29%**. The recent test period fell below the target, despite the full-period 4.36% CAGR.

The 104/101 45 DTE hold-to-expiration result had train, validation, and test CAGRs of **3.59%, 2.39%, and 3.14%**. Its return was lower, but its performance was more even and its full-period drawdown was smaller.

These are exploratory backtests across 7,956 variants. The large search space raises selection risk, and the archive contains only a small number of 10% SPX declines. The results support the simple unhedged 104/101 configurations most strongly; the hedge variants require more independent data before treating their average downturn gains as repeatable.
