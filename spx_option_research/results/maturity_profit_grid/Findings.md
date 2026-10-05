# Maturity-grid findings

Completed 734,720 combinations; 306,903 qualified.

The original control is the monthly 103/100 vertical at about 60 DTE, closing when net profit reaches 25% of opening credit. Both legs use the same expiration; calendar spreads and maturity-switching signals are outside this grid.

Using the original 5% initial-capital risk budget, the highest full-history Sharpe at the 25% profit target is 103/98, 42 DTE, monthly entries: CAGR 3.77%, annual volatility 3.27%, maximum drawdown -4.24%, Sharpe 1.15. Its reused-test Sharpe falls to 0.35. This is a hindsight ranking.

The 56-DTE comparison, 104/99 with monthly entries and a 25% profit target, has CAGR 4.06%, annual volatility 3.62%, maximum drawdown -6.84%, Sharpe 1.12. Training, validation and reused-test Sharpe are 1.14, 1.20 and 1.03. This row is highlighted after inspecting all periods, not selected prospectively.

For a comparison under the same new data conventions, the original monthly 103/100 at 60 DTE has CAGR 3.25%, annual volatility 3.25%, maximum drawdown -5.02% and Sharpe 1.00. The exact legacy-cache reconstruction remains 4.16% CAGR and 1.09 Sharpe under its earlier quote/calendar convention.

## Separate validation-selected results

100% equity notional: 97/94, 3 DTE, weekly entries, 25% profit target; full-period CAGR 0.24%, annual volatility 1.26%, maximum drawdown -5.03%, Sharpe 0.20. Validation Sharpe 3.19; reused-test CAGR 0.86%, Sharpe 2.39. Entry coverage 84.1%; 438 trades. Mean holding period 3.1 calendar days; maximum entry-basis notional 1.02 times initial capital.

5% initial-capital risk budget: 99.5/98.5, 3 DTE, every 4 weeks entries, 25% profit target; full-period CAGR 2.17%, annual volatility 4.10%, maximum drawdown -8.11%, Sharpe 0.55. Validation Sharpe 2.71; reused-test CAGR 4.54%, Sharpe 1.32. Entry coverage 100.0%; 131 trades. Mean holding period 3.1 calendar days; maximum entry-basis notional 5.00 times initial capital.

No configuration at 150, 180 DTE achieved the required 80% entry coverage, so those maturities have no ranked winner.

Spread labels are strike percentages of entry SPX (103/100 means short 103%, long 100%). Results are options plus idle cash, using fractional contract quantities for research. SPX stock exposure is not included. The 5% risk budget is fixed to initial capital and can create substantial notional leverage; the 100% notional mode sizes from prior-close equity. These modes are not directly interchangeable.

Chronological training ends 2022-09-16; validation ends 2024-09-17; the remaining data through 2026-09-18 is the reused test segment. The validation-selected tables use validation Sharpe with a penalty for disagreement with training Sharpe; full-period and test performance do not select those rows. The separate historical rankings use all dates, including the test segment. The archive has been studied before, and the large search remains exposed to selection bias.

Historical rankings explicitly use full-period Sharpe and are separate from validation-selected rankings. All-target validation tables come from the Float32 exhaustive-grid files; retained historical and 25%-profit winners are recomputed in Float64.
