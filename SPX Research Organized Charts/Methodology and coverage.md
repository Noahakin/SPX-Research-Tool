# SPX Research Organized Charts

Open **index.html** in Edge or Chrome. **Put buying**, **Put selling**, and **SPX with overlay** keep their existing folders and images. **Both** has four top-level premium budgets: **05, 10, 15 and 20 percent premium**. Each contains 105 short variations, then **Best long put** and **Best put buffer**, with seven expiration charts and an expiration comparison. Budget and short-variation folders also have comparison charts.

The library has 8,820 strategy curves: 2,940 original standalone/overlay curves, 2,940 retained long-put combinations and 2,940 corrected buffer combinations. The completed export has 10,537 images at 1600 × 1000. Both uses WebP; the preserved categories keep their original PNGs. See **Chart inventory validation.json** for current export status and file verification.

Choose **05, 10, 15 or 20 percent premium**, then a short-put variation, then **Best long put** or **Best put buffer**. Each budget has 105 short variations and seven target expirations. All strike labels and spread widths use percentages of SPX at entry: 95/90 is a five-percentage-point buffer.


## Best put buffer: protection purchased at entry

At every roll, compute the allocated budget as 5%, 10%, 15% or 20% of the short strategy's **net credit after slippage and commissions**. Choose the highest affordable long-put target at or below 100% of SPX. At that same target, prefer a 5-point buffer, then 4, then 3. A narrower permitted buffer can win when it reaches a closer target. The actual listed strike width must cover at least 3% of entry SPX. Widths of 1 and 2 are excluded.

Affordability requires **at least one buffer per short contract**. Let C be net short credit, f the premium fraction, and D the all-in buffer debit. Buffer quantity is **q = f × C / D**, with q ≥ 1 and no upper cap. Fractional additional buffers use the whole allocation; **every traded buffer spends exactly the budget**. Retained credit is C − q × D. Quantities stay fixed until the next roll.

Selection uses only entry quotes. **Sharpe does not choose buffers**; it remains a reported performance metric. The search uses the audited one-percentage-point target grid, with both hedge strikes between 80% and 100% of SPX. Hedge strikes may overlap or sit above the base short strategy's lower strike. All legs share the expiration. Closest means closest among valid, affordable candidates on this grid, not every listed strike.

If the base is ineligible or no valid 3–5-point buffer can be bought at full size, the entire combination stays in cash for that roll. The model does not substitute a narrower buffer or a partial-notional hedge. Cash earns zero. Very small credits can still afford only distant protection or no buffer at all. Cash and no-affordable-buffer counts are reported explicitly; no performance or coverage filter removes these strategies.

Buffer strikes change from roll to roll. **Selected hedges.csv** reports target ranges, width counts, actual minimum width, spending and notional. Fixed hedge-strike columns are blank for dynamic buffers. The numerical entry ledger, including selected listed strikes and quantities, is in `spx_option_research/results/organized_spx_best_hedges/buffer_trade_ledger.parquet`.

## Best long put

The outright long-put choices retain the earlier requested highest full-period combined-strategy Sharpe selection, separately by short variation, budget and expiration. These are retrospective research winners. Their fixed target is at or below the base spread's lower target (or the single short's target), down to 80%. Candidates require at least 90% of eligible base rolls. Their hedge quantity remains q = min(1, f × C / D), so long puts may spend less than the allocation. The buffer correction does not change these selections.

## Example

For the 98/95 short spread, 3-day expiration, September 17, 2026: net short credit was 2.920 SPX points; 20% allocated 0.584 points. The selected 95/90 buffer cost 0.4925 points after costs. Buying 1.1857868 buffers per short spread spent the full 0.584 points. The five-point width refers to 5% of entry SPX, not five SPX index points.

## Common research period and exposure


The analysis runs from **September 23, 2016 through September 18, 2026**, approximately ten years (9.9851 elapsed years), with **2,510 cash-market observations**. Every strategy starts with **$1,000,000**. This uses the full common archive history, not a recent holdout period.

Standalone option notional and the short strategy in Both are **100% of current portfolio equity** at every roll, measured as contract quantity × SPX at entry × the 100 option multiplier. Fractional contracts are allowed. Quantities remain fixed between rolls; sizing is not reset daily. Both adds the variable hedge quantity defined above. The library's baseline is 100% short-strategy notional.

**SPX with overlay** also holds SPX price exposure equal to 100% of equity at each roll. It keeps that underlying exposure during cycles when the option is unavailable. SPX price returns exclude dividends. Cash earns zero; financing, borrowing costs, and taxes are not modeled. SPX plus a debit hedge can require financing, so these curves are not fully funded total-return indices. Both holds only its option combination and cash collateral.

## Expiration conventions

| Target | Actual entry DTE across available trades | Scheduled cycles |
|---|---:|---:|
| 3 days | 2–5 calendar days | 1043 |
| 1 week | 6–8 calendar days | 521 |
| 2 weeks | 13–15 calendar days | 261 |
| 3 weeks | 20–22 calendar days | 174 |
| 4 weeks | 27–29 calendar days | 131 |
| 6 weeks | 41–43 calendar days | 87 |
| 8 weeks | 49–63 calendar days | 65 |

One-, two-, three-, four-, and six-week strategies use fixed calendar-Friday anchors, shifted to the prior cash session for holidays. They hold to the exact PM expiration on the scheduled date. One unavailable expiration ending December 16, 2016 becomes an explicit cash cycle for each of these five schedules.

The three-day target renews at expiration into the nearest listed PM expiry within one to five calendar days, choosing the longer expiry on a tie. Early availability often produces four calendar days. The eight-week target renews at expiration into the nearest listed Friday or holiday-adjusted-Friday PM expiry, also choosing the longer expiry on a tie. Exact eight-week expirations were unavailable in much of the early archive; actual eight-week-target trades therefore span 49–63 days.

No new position opens on the final analysis day. A position expiring later remains marked at its observed final-date midpoint; no hypothetical closing commission is charged. Only PM SPXW contracts are used. Invalid non-cash-session expiration labels are excluded rather than silently reinterpreted.

## Strikes, fills, and daily valuation

Target strikes use the observed **cash SPX close**, not the archive's expiration-specific underlying field. The nearest listed strike is selected before quote screening, with the lower strike winning exact-distance ties. If any required selected leg differs from its target by more than **0.5 percentage point of SPX**, if spread legs collapse, or if an entry quote is invalid or estimated, that option cycle stays in cash. Entry decisions do not depend on future outcomes. The original fixed strategies retain negative net-credit trades after costs. Both requires positive net credit to fund its hedge and therefore skips those cycles.

There are 8,278 curves with at least one option cash cycle. Individual images disclose the number of option rolls in cash. Particularly far-in-the-money targets can have substantial missing-strike coverage; those are **trade-when-representable** strategies, not uninterrupted positions at the nominal target. The CSV includes cycle counts and maximum traded strike deviation.

Long entries pay midpoint plus **25% of the full bid/ask spread**. Short entries receive midpoint minus the same amount. Each entry leg also pays **$1.50 per contract**. Positions settle to cash intrinsic at PM expiry. Interim holdings use the same contracts' daily bid/ask midpoints. Returns include entry friction on the first date and on every subsequent roll.

EOD execution at these quotes and simultaneous cash-close sizing are research assumptions. The archive's 16:00 timestamp is generated metadata, not independently verified exchange timing. Quoted spreads and execution estimates do not establish obtainable live fills or capacity.

## Quote quality and estimates

The combined extractor found all **470,611 required daily date/contract records** and reconciled all entry records. Presence of a record does not guarantee a sound market quote. The audit identified defective zero/crossed quotes and large isolated cross-strike inconsistencies, including during the February 2020 selloff and June 29, 2020, plus zero/zero penny-put records at the deeper downside strikes.

**68 contract/date values use explicitly estimated marks.** Isolated defects use nearby puts with the same date and expiration. Where a bracket is unavailable, the fallback estimates a put through its same-strike call and put-call parity, fitting discount/forward terms from valid paired contracts in the same snapshot. This does not use the later outcome or cash close to force the fit. Evidence includes raw values, estimated values, and supporting observations/calibration. Estimates are research valuations, not recovered observed prices. They are never used to justify a new entry.

For far-downside penny puts, a same-expiry bracket can extend up to 50 strike points on each side when both observed donor asks are at most 0.25 option points. This accommodates coarser strike listings while keeping material-price options subject to the original tighter bracket. These marks are also explicitly estimated, and all raw records remain preserved.

Residual inconsistencies in observed leg midpoints remain; synthetic spread marks can occasionally lie outside theoretical payoff bounds because independently quoted legs are noisy. They are logged in the research artifacts. No general clipping to payoff bounds or forward filling is applied. Individual charts identify the estimated option marks they use. These data and execution limits matter when presenting Sharpe estimates as historical research.


## Metrics and validation

CAGR uses actual calendar years (365.2425 days). Annualized volatility is sample daily-return standard deviation × √252. Sharpe is mean daily return divided by sample standard deviation × √252, with zero cash/risk-free return. Maximum drawdown includes initial equity in the running peak. Sharpe is descriptive for buffers.

**Calculation validation.json** records the current independent quote audit, full-budget checks and daily accounting reconciliation. **Chart inventory validation.json** records export completeness and hashes of preserved category files. **All strategy metrics.csv** and **Both/Selected hedges.csv** contain the current results.

Research inputs remain in `spx_option_research/results/organized_spx_combinations`; original curves remain in `organized_spx_research`. Current buffer results are in `organized_spx_best_hedges`. Curves are reconstructed in memory from audited inputs and current rules. Prior Sharpe-buffer metadata is explicitly archived as superseded and is not validation of the corrected model.

Rebuild numerical results with `python -B spx_option_research/scripts/rebuild_premium_buffers.py --build`; audit with `verify_premium_buffers.py`; render with `render_premium_buffers.py`; publish these notes with `publish_premium_buffer_notes.py`.
