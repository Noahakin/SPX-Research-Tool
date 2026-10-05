# Weekly study market data audit

The source SPX option archive covers September 22, 2016 through September 22, 2026. It contains 38,879,908 rows on 2,513 populated trading dates; 96 holiday files are empty. The source is `04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m`.

Using the last SPX trading session of each complete Friday-ending week produces 521 completed weekly cycles, September 23, 2016 through September 18, 2026: 489 with 7 calendar days, 16 with 6 days, and 16 with 8 days. One cycle lacks the requested PM-settled contract: December 9–16, 2016. Thus 520 cycles have PM put quotes. The final September 18–22 partial week is excluded. Exact entry/expiration dates and availability counts are in `weekly_archive_availability_audit.csv`.

For quoted puts between 94% and 104% of cash SPX, the inspected weekly entry snapshots contain no missing IV, gamma, option volume or open interest. All these nearby puts satisfy nonnegative bid and ask greater than or equal to bid. Available quote counts vary; this audit does not guarantee that every desired strike or delta has a listed contract, nor does it establish that every quote is executable.

Use actual cash SPX closes for strike selection, not the archive's expiration-specific `underlying_price`. `market_spx.csv` merges local Yahoo `^GSPC` prehistory with the corrected prior study cash cache, preferring the corrected cache on overlap. Locally retrieved records are rejected if they were retrieved before that observation's New York cash close. The original prior-study cache did not preserve its download timestamp; its content hash is included in `market_data_manifest.json`.

VIX, VVIX and VIX3M come from official Cboe daily histories:

- https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
- https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv
- https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv

All three official histories cover every SPX session during the option study. Official VIX exactly matches the previously cached FRED VIX on all 2,513 study rows. Official VIX3M matches all 4,237 observations overlapping the local FRED series. Official VVIX differs by more than 0.011 points from local Yahoo on 12 historical dates, including the local August 20, 2026 observation retrieved before close; the largest discrepancy is 5.00 points. The official VVIX history is used in full. Detailed comparison counts are in `audit_indicator_sources.json`.

`scripts/build_weekly_market_features.py` saves source files `market_spx.csv`, `market_vix.csv`, `market_vvix.csv` and `market_vix3m.csv`, with date, close, source and retrieval timestamp. These files stop at September 22, 2026. Raw official audit copies also contain later observations, but those observations are excluded from the study files and feature generation.

The primary `market_data.parquet` and CSV contain 4,205 SPX sessions, January 4, 2010 through September 22, 2026, with 25 numeric features. Features include SPX realized volatility over 5/21/63 sessions, momentum over 1/5/21/63/252 sessions, 252-session drawdown, VIX/VVIX/VIX3M levels and 5/21-session changes, prior-252-session level z-scores, and ratios between volatility measures. Realized volatility uses simple cash-index returns, sample standard deviation and annualization by sqrt(252).

All market features are aligned to the SPX calendar and then shifted by one SPX session. `market_source_date` records the latest available input date. Historical z-score references exclude the observation being scored. No source prices or features are forward-filled or back-filled. There are zero missing feature values during September 22, 2016 through September 22, 2026; early prehistory warmup rows retain their missing values.

Synthetic verification passed for invariance to future observations, one-session execution lag, the 21-session realized-volatility calculation, 5-session momentum, strictly prior 252-session z-score references, and preservation of missing source observations without filling.
