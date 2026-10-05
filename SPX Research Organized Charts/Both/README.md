# SPX premium-funded short puts

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
