# Independent data-quality audit: weekly fixed SPX spreads, 2016–2026

Coverage: September 23, 2016–September 18, 2026, 521 scheduled weeks and 2,510 cash-market sessions. Each fixed strategy has 520 executable quoted trades and one common cash week. Quote existence is distinct from quote accuracy.

## Trade coverage

| Cached source | Records | Missing entry dates |
|---|---:|---|
| Short 97%, width 3% | 515 | 2016-12-09; 2016-12-23; 2017-03-03; 2017-03-24; 2017-09-29; 2018-08-31 |
| Short 98%, width 3% | 520 | 2016-12-09 |
| Short 99%, width 3% | 520 | 2016-12-09 |
| Short 98%, width 2% | 519 | 2016-12-09; 2017-03-03 |
| Short 98%, width 5% | 519 | 2016-12-09; 2017-03-03 |

The common December 9–16, 2016 interval lacks the required PM expiry in the entry archive and stays in cash. Its sessions must remain in the return calendar. Other cache omissions result from the old positive-credit filter, including rejection when natural execution minus commissions is a debit; the raw entry option quotes exist. Independently selecting the nearest valid PM strikes on those omitted dates restores 520 trades for each fixed policy, allowing nonpositive net credits as required for an always-trade fixed policy.

All four strategies have 2,504 pre-expiry daily spread marks apiece. Every exact-symbol/date/strike/expiry quote is present and has finite nonnegative bids with ask at least bid. 5 additional leg/date pairs were read directly from the source archive because they were not in the cached quote subset. Expiry cash settlement needs no option mark.

## Selected midpoint flags

Only two selected spread/date marks are outside [0, width] by more than 0.01 point:

| Strategy / entry | Mark date | Raw liability | Width | Source details |
|---|---|---:|---:|---|
| 97/94 / 2020-02-21 | 2020-02-27 | -33.225 | 105 | Short 3240 put: 0.05 / 288.40; long 3135 put: 170.40 / 184.50 |
| 97/94 / 2020-03-20 | 2020-03-26 | -0.025 | 70 | Short 2240 put: 0 / 0.10; long 2170 put: 0 / 0.15 |

The second flag is a 2.5-cent spread-mark inversion among tiny quoted options. The February flag is materially different: an almost zero bid against a 288.40 ask on a deeply in-the-money short put. Among selected daily legs, it is the only quote with a bid/ask range above 30 index points and above 140% of midpoint. These screens do not prove the remaining quotes accurate.

## February 27, 2020 forensic evidence

The source has exactly one record for SPXW 200228P03240000, expiring February 28, 2020. Its bid is 0.05, ask 288.40, midpoint 144.225 and `last` 144.22499. The last field reproduces the defective midpoint and provides no independent replacement. Nearby put bids are also anomalous at strikes 3230 (0.50) and 3250 (0.25), while 3235 and 3245 have bid/ask 262.60/283.40 and 272.60/293.40.

All 15,036 source records show 16:00, but the downloader assigns `snapshot_timestamp = trade_date + 16 hours` (download_spx_option_surface.py, line 427). This is a synthetic label, not an observed exchange quote timestamp. The archive underlying is 2957.452 versus cash-index close 2978.760009765625, further limiting assertions about exact synchronous close valuation.

The workspace options-file search found one SPX archive file for February 27, 2020, under spx_options_6m. The other spx_options archive begins January 3, 2022. The downloader parses its raw CSV response in memory and writes normalized daily Parquet; no alternate saved February 2020 SPX source or intraday timestamp was found. No observed replacement quote was identified.

## Disclosed sensitivity estimates

Raw source quotes and the raw midpoint series must remain unchanged. Payoff clipping to zero removes the impossible negative liability but still values this deeply in-the-money vertical at zero. It does not repair the defect.

| Convention for the affected 97/94 liability | Index points | Basis |
|---|---:|---|
| Raw midpoint | -33.225 | Defective recorded short-put bid/ask |
| Payoff clipped | 0 | Mechanical bound only; no valuation evidence |
| Adjacent-put interpolation sensitivity | 100.550 | Mean of 3235 and 3245 put midpoints: (273 + 283)/2 = 278; subtract observed 3135 put midpoint 177.45 |
| Opposite-call parity sensitivity | 103.725 | 105 + 3240 call midpoint 0.025 - 3135 call midpoint 1.30; discount factor assumed 1 |

For European options of identical expiry, P(3240)-P(3135) = 105 × discount factor + C(3240)-C(3135); the underlying cancels. At discount factor 1, observed call bid/ask implies a 103.40–104.05 spread interval. Interpolating adjacent puts and applying call parity are explicit estimates, not recovered observed put quotes. They alter daily mark-dependent statistics and do not alter fixed-quantity expiration P&L, subsequent entry capital, terminal equity or CAGR.

The other selected February 27 spreads have raw liabilities inside their payoff bounds but disagree with same-expiry call parity: 96/93 (3205/3105) raw 93.25 versus call-parity midpoint 97.775; 98/95 (3270/3170) raw 97.40 versus 99.375; 99/96 (3305/3205) raw 99.40 versus 99.90. Wide bid/ask spreads can explain some midpoint disagreement. These observations show that passing payoff bounds does not certify synchronous, accurate daily marks; broad quote rewriting is not justified by this audit.

Evidence files: repair_evidence.csv preserves the same-day relevant puts and calls; mark_adjustments.csv specifies the single flagged-date sensitivity change. audit_fixed_trade_legs.csv, audit_candidate_cache_coverage.csv, audit_cash_weeks.csv, audit_selected_mark_flags.csv and audit_missing_daily_quotes.csv document independent coverage checks.
