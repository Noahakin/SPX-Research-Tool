# SPX maturity and profit-taking grid

Completed **734,720 parameter combinations**, with **306,903** meeting the disclosed coverage, data-quality and train/validation screens.

The study covers September 23, 2016–September 18, 2026, 2,510 daily sessions. The original cached monthly 103/100, 60-DTE, 25%-profit winner was reproduced at 4.16% CAGR and 1.09 Sharpe, within $0.00000000 of its saved curve.

## The 25% profit target

Each row chooses a spread and entry schedule within its maturity and sizing group using validation data. The historical test segment has already been used in earlier research; it is a reused diagnostic, not a new untouched holdout.

| DTE | Sizing | Spread | Entry schedule | CAGR | Volatility | Max DD | Sharpe | Validation Sharpe | Test Sharpe | Entry coverage |
|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 100% notional | 97/94 | weekly | 0.24% | 1.26% | -5.03% | 0.20 | 3.19 | 2.39 | 84.1% |
| 7 | 100% notional | 90/80 | every_4_weeks | 0.14% | 2.49% | -5.40% | 0.07 | 2.07 | 2.13 | 93.1% |
| 10 | 100% notional | 97.5/90 | monthly | 0.87% | 2.38% | -6.76% | 0.38 | 2.55 | 1.22 | 98.3% |
| 14 | 100% notional | 97/87 | weekly | 2.65% | 3.64% | -10.75% | 0.74 | 2.11 | 1.78 | 99.8% |
| 21 | 100% notional | 93.5/83.5 | weekly | 1.11% | 2.05% | -7.61% | 0.55 | 2.33 | 1.44 | 98.3% |
| 28 | 100% notional | 92/87 | weekly | 0.52% | 1.01% | -4.06% | 0.52 | 1.73 | 1.57 | 99.6% |
| 35 | 100% notional | 90.5/80.5 | weekly | 0.53% | 1.49% | -5.98% | 0.36 | 1.52 | 1.44 | 96.4% |
| 42 | 100% notional | 107.5/97.5 | monthly | 4.06% | 4.43% | -6.27% | 0.92 | 1.44 | 0.71 | 95.8% |
| 45 | 100% notional | 90.5/80.5 | every_2_weeks | 0.69% | 1.42% | -4.81% | 0.50 | 1.31 | 1.11 | 93.9% |
| 56 | 100% notional | 101.5/98.5 | monthly | 1.13% | 1.24% | -2.18% | 0.91 | 1.62 | 0.49 | 100.0% |
| 60 | 100% notional | 101.5/98.5 | monthly | 0.98% | 1.02% | -1.51% | 0.97 | 1.57 | 0.44 | 100.0% |
| 75 | 100% notional | 99/95 | weekly | 0.57% | 0.97% | -1.85% | 0.59 | 1.42 | 0.42 | 84.5% |
| 90 | 100% notional | 109/99 | monthly | 1.88% | 2.50% | -5.32% | 0.76 | 1.37 | 0.19 | 90.8% |
| 120 | 100% notional | 108.5/98.5 | monthly | 1.46% | 2.33% | -4.96% | 0.64 | 1.23 | 0.54 | 83.2% |
| 3 | 5% risk budget | 99.5/98.5 | every_4_weeks | 2.17% | 4.10% | -8.11% | 0.55 | 2.71 | 1.32 | 100.0% |
| 7 | 5% risk budget | 90/80 | every_4_weeks | 0.07% | 1.25% | -2.77% | 0.06 | 2.06 | 2.12 | 93.1% |
| 10 | 5% risk budget | 97.5/90 | monthly | 0.67% | 1.68% | -4.49% | 0.40 | 2.49 | 1.20 | 98.3% |
| 14 | 5% risk budget | 97/87 | weekly | 1.35% | 1.99% | -6.47% | 0.69 | 2.11 | 1.75 | 99.8% |
| 21 | 5% risk budget | 93.5/83.5 | weekly | 0.57% | 1.11% | -4.22% | 0.52 | 2.33 | 1.44 | 98.3% |
| 28 | 5% risk budget | 92/87 | weekly | 0.55% | 1.10% | -4.44% | 0.51 | 1.68 | 1.58 | 99.6% |
| 35 | 5% risk budget | 90.5/80.5 | weekly | 0.29% | 0.80% | -3.22% | 0.36 | 1.52 | 1.44 | 96.4% |
| 42 | 5% risk budget | 107.5/97.5 | monthly | 4.22% | 4.68% | -6.27% | 0.91 | 1.46 | 0.70 | 95.8% |
| 45 | 5% risk budget | 107.5/97.5 | monthly | 3.85% | 4.55% | -6.34% | 0.85 | 1.26 | 0.62 | 87.4% |
| 56 | 5% risk budget | 102/98 | monthly | 2.80% | 2.84% | -5.16% | 0.99 | 1.58 | 0.49 | 100.0% |
| 60 | 5% risk budget | 100.5/99.5 | monthly | 2.01% | 2.35% | -3.90% | 0.86 | 1.57 | 0.35 | 100.0% |
| 75 | 5% risk budget | 99/95 | weekly | 0.91% | 1.54% | -3.03% | 0.60 | 1.38 | 0.45 | 84.5% |
| 90 | 5% risk budget | 109/99 | monthly | 2.21% | 2.98% | -6.26% | 0.75 | 1.39 | 0.16 | 90.8% |
| 120 | 5% risk budget | 108.5/98.5 | monthly | 1.46% | 2.40% | -4.60% | 0.62 | 1.21 | 0.54 | 83.2% |

## Retrospective comparisons

Using the original 5% initial-capital risk budget, the highest full-history Sharpe at the 25% profit target is 103/98, 42 DTE, monthly entries: CAGR 3.77%, annual volatility 3.27%, maximum drawdown -4.24%, Sharpe 1.15. Its reused-test Sharpe falls to 0.35. This is a hindsight ranking.

The 56-DTE comparison, 104/99 with monthly entries and a 25% profit target, has CAGR 4.06%, annual volatility 3.62%, maximum drawdown -6.84%, Sharpe 1.12. Training, validation and reused-test Sharpe are 1.14, 1.20 and 1.03. This row is highlighted after inspecting all periods, not selected prospectively.

For a comparison under the same new data conventions, the original monthly 103/100 at 60 DTE has CAGR 3.25%, annual volatility 3.25%, maximum drawdown -5.02% and Sharpe 1.00. The exact legacy-cache reconstruction remains 4.16% CAGR and 1.09 Sharpe under its earlier quote/calendar convention.

Historical rankings explicitly use the entire period and are separate from the validation-selected table above. See `Historical 25 percent winners by maturity.csv`, `Historical 25 percent leaders.csv` and `Historical all target leaders.csv`. No configuration at 150, 180 DTE achieved the required 80% entry coverage, so those maturities have no ranked winner.

Spread labels are strike percentages of entry SPX (103/100 means short 103%, long 100%). Results are options plus idle cash, using fractional contract quantities for research. SPX stock exposure is not included. The 5% risk budget is fixed to initial capital and can create substantial notional leverage; the 100% notional mode sizes from prior-close equity. These modes are not directly interchangeable.

## Search and accounting

- Maturities: 3, 7, 10, 14, 21, 28, 35, 42, 45, 56, 60, 75, 90, 120, 150, 180 calendar days.
- Short strikes: 90%–110% of SPX in 0.5-percentage-point steps. Widths: 1, 2, 3, 4, 5, 7.5 and 10 percentage points.
- Entries: weekly, every two weeks, every four weeks, or monthly. Profit targets: 5% through 95% in five-point steps, plus hold to expiration. Early exits do not cause off-schedule re-entry.
- The primary comparison fixes the profit target at 25%; the all-target shortlist is a separate search.
- The 5% risk-budget mode uses original capital, allows leverage and imposes a 500% initial-capital notional ceiling. The 100% notional mode sizes from prior-close equity. Different maturity/cadence combinations split capital across expected overlaps. Both leave unused capacity in cash.
- Entry and early-exit fills pay one quarter of each full bid/ask spread from midpoint plus $1.50 per leg. Profit triggers use net executable P&L after those costs. Cash earns zero; no dividends, financing, tax or market-impact returns are added.
- Holdings retain their fixed contract quantities; surviving positions are marked every session. Actual PM expiry settles at cash intrinsic; the common data endpoint liquidates remaining positions.

## Data and interpretation

The earlier cached winner includes 1 weekend-dated archive expirations and uses the archive's underlying field. This grid selects listed weekday PM expirations and cash-index strike targets. Its comparable 103/100 control is saved separately across every maturity and cadence; it need not reproduce the old quote/calendar convention.

The archive supplies EOD snapshots with generated 16:00 labels. Observed quotes can authorize execution; explicitly estimated same-day marks cannot trigger profit-taking. Portfolios holding unresolved marks are retained for diagnostics but excluded from rankings. Raw midpoint differences outside payoff bounds are explicitly counted and cannot trigger profit-taking. Where the bid/ask interval intersects the payoff range, those observed midpoint marks are preserved; the code does not clip them into manufactured prices. Nearest-listed strike or expiry choices can make different parameter rows produce identical trades.

This is a large retrospective search. A high historical score alone does not establish future performance; compare validation/test consistency, neighboring maturities and widths, costs, coverage, and actual leverage. The test segment never enters the new validation-selection score, but the archive has been studied before.

## Files

- `grid/`: every parameter row, as compressed Parquet. Scalar summary metrics use Float32; simulations and shortlisted curves/CSV results retain Float64 precision.
- `25 percent winners by maturity.csv`: primary validation-selected comparison.
- `103 100 maturity comparison.csv`: the original spread and 25% target, with each maturity and cadence.
- `Validation shortlist all profit targets.csv`: the separate all-profit-target shortlist.
- `Shortlisted daily curves.parquet`, `Shortlisted trade ledger.parquet`, and `Shortlisted calendar year returns.csv`: numerical paths and trade evidence.
- `Protocol.json`, `Data audit.json`, `Validation.json`, and the compressed omission/quote-estimate logs: assumptions, source fingerprint, coverage and checks.

Reproduce the full search, independent audit and readable report from Raw Data: `python -B spx_option_research/scripts/run_maturity_profit_study.py`.
