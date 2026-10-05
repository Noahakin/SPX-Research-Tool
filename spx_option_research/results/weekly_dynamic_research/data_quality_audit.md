# Weekly SPX spread research: data quality and units

## Coverage and cash-index prices

The option archive contains 38,879,908 rows on 2,513 populated dates, September 22, 2016–September 22, 2026. The completed Friday/holiday-adjusted weekly schedule has 521 cycles ending by September 18, 2026. December 9–16, 2016 lacks the required PM-settled SPXW expiry; 520 cycles have PM quotes. The candidate builder retains 14,501 executable candidate spreads from the requested grid, with unavailable candidates recorded separately.

Strike selection, position sizing and cash settlement use actual Yahoo **SPX cash-index closes**. The archive's `underlying_price` is expiration-specific and must not be used as cash spot. `market_spx.csv` combines local Yahoo `^GSPC` prehistory with the corrected earlier cash-close cache, covering January 4, 2010–September 22, 2026. Local observations retrieved before their own New York close are excluded. The earlier corrected cache did not preserve its retrieval timestamp; its content hash is retained in `market_data_manifest.json`.

## Market sources and information timing

VIX, VVIX and VIX3M use official Cboe daily histories, with source URLs, retrieval timestamps and hashes saved:

- [VIX](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv)
- [VVIX](https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv)
- [VIX3M](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv)

Each covers every SPX trading session in the study. Official VIX matches all 2,513 cached FRED study observations exactly; VIX3M matches all 4,237 overlapping FRED observations. Official VVIX differs by more than 0.011 points from local Yahoo on 12 dates, with a maximum difference of 5 points. The official VVIX history is used throughout, including replacement of a locally cached August 20, 2026 observation retrieved before close.

All 25 cash-market features are aligned to the SPX session calendar and **lagged one session**. `market_source_date` identifies the latest input date. No forward-fill or backfill is used; there are no missing market features during the option study after prehistory warmup. Rolling level z-scores exclude the observation being scored.

Option bid/ask, IV, Greeks, recorded volume and open interest come from the entry EOD snapshot. Execution assumes fills against that same snapshot. This is an EOD research convention: the archive does not establish that final daily volume/open-interest values or final quotes were available early enough to compute and execute a live order at those exact prices. Market-index features have the additional one-session lag; option-snapshot features do not.

## Execution and daily marks

All 14,501 entry candidates have uncrossed quotes and midpoint/natural credits strictly between zero and spread width. There are no zero short bids and eight zero long bids. Combined leg bid/ask width has a median of 1.0 option point, 99th percentile of 17.2, and maximum of 37.9. It exceeds midpoint net credit in 75 candidates (0.52%).

Realistic execution assumes 25% of each full bid/ask spread away from midpoint plus $1.50 per contract per leg. Entry cost is a median **0.83 basis points of SPX notional**, 99th percentile **11.55 bp**, and maximum **22.28 bp**. ITM spreads cost more: median entry cost is about 0.49 bp for 99% short strikes and 4.23 bp for 103% short strikes across tested hedge widths. Natural bid/ask execution is an important sensitivity check.

Daily extraction validates 30,039 exact contract quotes across 2,504 required dates, producing 84,332 candidate/day unit marks. No required quotes are missing, duplicated or individually crossed. All 14,501 expiration P&Ls reconcile for all execution assumptions, with maximum absolute error below 7e-18 in notional-return units. Entry friction is charged once; expiry uses cash intrinsic value without requiring an expiration option quote.

Nevertheless, some individually valid leg quotes imply economically impossible spread midpoints. Values outside `[0, strike width]` by more than 0.01 points are flagged:

| Period | Candidate/day flags | Distinct dates | Largest bound violation |
|---|---:|---:|---:|
| Before 2021 | 261 | 144 | 134.125 points |
| Validation, 2021–2023 | 57 | 40 | 2.10 points |
| Final evaluation, 2024–2026 | 111 | 74 | 3.45 points |

These counts cover the entire candidate universe, rather than only final selected trades. The severe pre-validation example is February 27, 2020: the 3240 put has bid 0.05/ask 288.4 while neighboring strikes have ordinary positive bids, producing large invalid spread midpoints. The primary series preserves raw actual midpoints. Constraining spread marks to payoff bounds is a disclosed sensitivity for frozen strategies, without changing selection. Clipping restores basic bounds but is not proof of an accurate fair-value mark. These issues affect daily return volatility and drawdown; they do not change held-to-expiration P&L or CAGR.

## Greek units

Checks against one-week near-ATM BSM estimates on 2017, 2020, 2024 and 2026 snapshots establish the following units. Model assumptions and vendor forward/carry conventions cause small differences; theta is especially convention-sensitive.

| Field | Quoted unit | Portfolio interpretation |
|---|---|---|
| Delta | Option-price points per SPX point | Net delta is long-put delta minus short-put delta. |
| Gamma | Delta change per SPX point | `0.5 * net_gamma * SPX * 0.01^2` approximates gamma P&L as a fraction of SPX notional for a 1% move. |
| Vega | Option-price points per **one volatility percentage point** | Multiply by 100 for dollars per contract; divide by SPX for notional-return sensitivity. It is not per 1.0 decimal IV. |
| Theta | Option-price points per **calendar day** | `net_theta * DTE / SPX` is tenor-scaled local decay sensitivity, not an expected return or exact full-tenor decay forecast. |

For the January 19, 2024 4840 weekly put, vendor vega is 2.661095 versus BSM 2.663048 per volatility point, and gamma is 0.006180 versus 0.00617686. Raw net-vega features retain price-point units. Premium divided by delta/RV risk uses a denominator floor; it is a stabilized risk proxy, not a model-independent valuation measure.

Supporting artifacts: `market_data_manifest.json`, `audit_indicator_sources.json`, `candidate_manifest.json`, `weekly_daily_quotes_manifest.json`, `weekly_daily_quote_issues.csv`, `weekly_midpoint_bound_flags.csv`, `weekly_unit_settlement_reconciliation.csv` and `unit_marks_manifest.json`.

## Independent final portfolio verification

All six final portfolios were independently reconstructed under both realistic and natural fills, using original candidate bid/ask fills, contract counts, cached raw leg midpoints and cash-index intrinsic settlement. This calculation did not call the portfolio evaluation or unit-return functions. It reproduced all 12 summary rows and every saved realistic daily NAV/weekly return for 141 weeks and 678 sessions, January 5, 2024–September 18, 2026. Maximum discrepancies were $2.57e-9 in daily NAV, 5.60e-14 in Sharpe, 8.12e-16 in CAGR, and 9.16e-16 in maximum drawdown. Cash weeks, initial transaction costs, settlement before new sizing, and compounding all reconciled. Details are in `independent_final_daily_metrics_audit.csv`.
