# Dynamic SPX spread selection across strikes, widths and expirations

Completed **202,752 dynamic policy configurations**, evaluating up to **4,592 candidate targets at each entry**, plus **3,058 fixed-candidate controls**. The archive supplied 1,877,523 valid candidate-entry combinations after the liquidity filter.

At every scheduled entry, each policy scores the then-available candidates and chooses the highest-scoring spread jointly across strike, width and expiration. The chosen contracts are held until the first supported EOD quote reaches 25% net profit, expiry, or the common data endpoint. The position is not switched merely because a different spread later scores better.

Common trading window: 2018-09-21 to 2026-09-18. The preceding 104 weekly opportunities warm up history-dependent rules. Training ends 2023-07-06; validation ends 2025-02-11.

## Historical leaders

5% initial-capital risk budget, overlapping positions, every 4 weeks entries: Expected edge / maximum loss; 63-session RV; volatility ×1.2; drift 0%; always choose the highest score. Allowed maturities 90–180 DTE, short strikes 90–110% and widths 1–10 percentage points. CAGR 3.83%, volatility 2.71%, maximum drawdown -2.48%, full-period Sharpe 1.41; reused-test Sharpe 1.47. Executed 101 trades across 4 maturity targets, 6 short-strike targets and 6 widths. Fixed comparison within the identical universe, with the same score gate, schedule and sizing: 91/81 at 120 DTE, CAGR 0.39%, Sharpe 0.69.

5% initial-capital risk budget, one position at a time, monthly entries: Whole-spread edge versus own history; 21-session RV; 104-week history; always choose the highest score. Allowed maturities 90–180 DTE, short strikes 90–110% and widths 5–10 percentage points. CAGR 5.77%, volatility 4.30%, maximum drawdown -6.17%, full-period Sharpe 1.33; reused-test Sharpe 0.91. Executed 63 trades across 3 maturity targets, 25 short-strike targets and 3 widths. Fixed comparison within the identical universe, with the same score gate, schedule and sizing: 104.5/94.5 at 120 DTE, CAGR 2.73%, Sharpe 0.85. No fixed candidate passed every primary eligibility screen. The displayed fixed comparison is a diagnostic with 51 trades and 79.2% score coverage; it uses a lower minimum trade count and can have lower coverage or nonpositive train/validation returns.

100% equity notional, overlapping positions, monthly entries: Whole-spread edge versus own history; 63-session RV; 104-week history; always choose the highest score. Allowed maturities 90–180 DTE, short strikes 90–110% and widths 1–10 percentage points. CAGR 0.72%, volatility 0.59%, maximum drawdown -0.88%, full-period Sharpe 1.24; reused-test Sharpe 1.11. Executed 92 trades across 3 maturity targets, 30 short-strike targets and 7 widths. Fixed comparison within the identical universe, with the same score gate, schedule and sizing: 104.5/94.5 at 120 DTE, CAGR 1.67%, Sharpe 0.95. No fixed candidate passed every primary eligibility screen. The displayed fixed comparison is a diagnostic with 76 trades and 79.2% score coverage; it uses a lower minimum trade count and can have lower coverage or nonpositive train/validation returns.

100% equity notional, one position at a time, every 2 weeks entries: IV/RV richness versus own history; 21-session RV; 104-week history; mean leg IV; positive score required. Allowed maturities 45–120 DTE, short strikes 98–105% and widths 5–10 percentage points. CAGR 3.60%, volatility 3.18%, maximum drawdown -3.85%, full-period Sharpe 1.13; reused-test Sharpe 0.61. Executed 68 trades across 6 maturity targets, 12 short-strike targets and 3 widths. Fixed comparison within the identical universe, with the same score gate, schedule and sizing: 105/100 at 45 DTE, CAGR 3.23%, Sharpe 1.01.

## Interpretation

Historical leaders were selected after looking across the entire period. Validation leaders are listed separately and are selected using training/validation data only. The archive has been studied before, so the later segment is reused historical evidence, not a fresh untouched out-of-sample test. Comparing hundreds of thousands of configurations creates substantial selection bias.

All returns are option P&L plus idle cash, without an SPX stock position. The 5% loss budget uses initial capital and can create up to 5 times initial-capital notional exposure. The 100% notional mode uses prior-close equity. Entries and early exits pay 25% of each leg’s full bid/ask spread from midpoint plus $1.50 per leg. Fractional contracts are used. Cash interest, financing, dividends, taxes and market impact are excluded.

Expected-payoff rules use a physical lognormal terminal-payoff model with trailing 21/63/126-session realized volatility, volatility multipliers 0.8/1/1.2, and annual drift assumptions of 0%/5%. They rank edge divided by maximum loss or payoff standard deviation, with or without annualization. These are entry proxies: terminal expected payoff is not a model of the expected 25%-profit exit. Credit-based scores do not estimate expected losses. IV/RV and whole-spread history z-scores measure relative richness against strictly prior observations.

Every policy checks six maturity universes, three short-strike universes and four width universes, with and without a positive-score gate. Four entry schedules, two sizing methods and two overlap modes produce the full grid. Maturity targets cover 3–180 days; the search selects the nearest listed weekday PM expiration to each of 16 targets. It does not enumerate every calendar-day target. Different parameters can produce identical decisions.

A candidate must have valid observed quotes on both legs, positive net credit and maximum loss, and estimated closing friction no greater than half the 25% profit objective. Realized-volatility inputs end on the prior session. Entry quotes and supplied IV use the current EOD snapshot; fills at that snapshot remain an assumption. No future P&L, exit date or missing-quote status enters selection.

The one-at-a-time policy waits until a position closes and then enters at its next scheduled opportunity. It does not immediately reopen off schedule. Overlapping policies divide allocations across anticipated overlaps and enforce aggregate capacity. Original strikes and quantities remain fixed during each trade.

The main charts compare each historical leader with fixed candidates inside its own strike, width and maturity universe, using identical score availability, positive-score gating, entry cadence, sizing and overlap mode. The highest-Sharpe fixed candidate passing the same primary screens is preferred. When none pass, the chart explicitly labels a diagnostic comparator requiring at least 20 trades and no unresolved marks; its coverage and trade count are disclosed. Both sides are chosen with hindsight over the same period. Earlier fixed-spread controls are also retained in the matched-controls file.

Estimated same-day marks cannot authorize entries or early profit exits. Unresolved marks are counted. Historical rankings exclude affected portfolios; validation selection checks quality only through validation end so future quote quality cannot influence the chosen policy. Any later quality flags remain visible in the validation table.

Independent reconciliation passed for 139 retained strategies, 20,829 trades and 13,190 scheduled entry decisions. Maximum difference in independently rebuilt daily NAV: $0.0000000035.

## Files

- `All policy results.csv.gz` and `grid/`: the complete search.
- `Overall leaders.csv`: historical and validation leaders by sizing and overlap mode.
- `Historical leaders by family.csv`, `Validation leaders by family.csv`: best policies within each scoring family.
- `Matched fixed controls.csv`, `Dynamic versus matched fixed controls.csv`: comparable fixed-spread results.
- `Within universe fixed grid.parquet`, `Within universe fixed winners.csv`: exhaustive fixed candidates within each illustrated dynamic leader’s allowed universe.
- `Retained daily curves.parquet`, `Retained trade ledger.parquet`, `Entry decision audit.parquet`: actual positions, daily curves, score and cash/entry decisions.
- `Selection distribution.csv`, `Calendar year returns.csv`: the changing chosen spreads and return history.
- `Protocol.json`, `Validation.json`, `Scoring rule definitions.csv`: assumptions, fingerprints and checks.

Reproduce: `python -B spx_option_research/scripts/run_dynamic_maturity_study.py` from Raw Data.
