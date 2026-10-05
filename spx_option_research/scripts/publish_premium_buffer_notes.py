"""Publish the corrected buffer method and current selection summaries."""
import json
import re
import pandas as pd
from optimize_organized_spx_hedges import OUTPUT
from render_organized_spx_charts import DEFAULT_OUTPUT


def run():
    client = DEFAULT_OUTPUT
    config = json.loads((OUTPUT/"run.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUTPUT/"strategy_metrics.csv", dtype={"folder": str}, low_memory=False)
    audit = json.loads((OUTPUT/"validation.json").read_text(encoding="utf-8"))
    export_report = client/"Chart inventory validation.json"
    export = json.loads(export_report.read_text(encoding="utf-8")) if export_report.exists() else {}
    if export.get("buffer_selection_rule_version") != "entry_affordability_v1":
        images = [p for p in client.rglob("*") if p.suffix.lower() in (".png", ".webp")]
        export_report.write_text(json.dumps(dict(status="incomplete; corrected buffer chart export in progress",
            buffer_selection_rule_version="entry_affordability_v1_pending",
            current_image_count=len(images), expected_image_count=10537,
            numerical_validation_status=audit["status"], storage_constraint="C: needs free space to finish export"), indent=2), encoding="utf-8")
    (client/"Research parameters.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (client/"Calculation validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metrics.to_csv(client/"All strategy metrics.csv", index=False)
    columns = ["premium_fraction", "short_variation", "tenor", "hedge_type", "selection_basis",
        "hedge_primary_pct", "hedge_secondary_pct", "hedge_width_pct", "hedge_target_min", "hedge_target_max",
        "buffer_width_min", "buffer_width_max", "actual_buffer_width_min", "mean_hedge_target",
        "width_3_cycles", "width_4_cycles", "width_5_cycles", "cagr", "annualized_volatility", "daily_sharpe",
        "max_drawdown", "traded_cycles", "cash_cycles", "no_affordable_buffer_cycles", "relative_base_coverage",
        "mean_premium_fraction_spent", "mean_hedge_notional_multiple", "max_hedge_notional_multiple", "strategy_id"]
    table = metrics.loc[metrics.category.eq("Both"), columns].sort_values(["premium_fraction", "short_variation", "tenor", "hedge_type"])
    table.to_csv(client/"Both"/"Selected hedges.csv", index=False)
    for fraction, rows in table.groupby("premium_fraction"):
        rows.to_csv(client/"Both"/f"{fraction*100:02.0f} percent premium"/"Selected hedges.csv", index=False)
    notes = """# SPX premium-funded short puts

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
"""
    if export.get("buffer_selection_rule_version") != "entry_affordability_v1":
        notes = notes.replace("# SPX premium-funded short puts\n", "# SPX premium-funded short puts\n\n**Export status:** Calculations are updated; the full buffer image export is waiting for disk space. The 20% / 98–95 buffer example is available for all seven expirations.\n", 1)
    (client/"Both"/"README.md").write_text(notes, encoding="utf-8")
    common = (OUTPUT/"Previous library documents"/"Methodology and coverage.md").read_text(encoding="utf-8")
    common = common.split("## Common research period and exposure", 1)[1].split("## Metrics and verification", 1)[0]
    common = common.replace("the short spread in Both", "the short strategy in Both")
    common = common.replace("100% short-spread notional", "100% short-strategy notional")
    common = re.sub(r"There are [\d,]+ curves with at least one option cash cycle",
                    f"There are {int(metrics.cash_cycles.gt(0).sum()):,} curves with at least one option cash cycle", common)
    text = """# SPX Research Organized Charts

Open **index.html** in Edge or Chrome. **Put buying**, **Put selling**, and **SPX with overlay** keep their existing folders and images. **Both** has four top-level premium budgets: **05, 10, 15 and 20 percent premium**. Each contains 105 short variations, then **Best long put** and **Best put buffer**, with seven expiration charts and an expiration comparison. Budget and short-variation folders also have comparison charts.

The library has 8,820 strategy curves: 2,940 original standalone/overlay curves, 2,940 retained long-put combinations and 2,940 corrected buffer combinations. The completed export has 10,537 images at 1600 × 1000. Both uses WebP; the preserved categories keep their original PNGs. See **Chart inventory validation.json** for current export status and file verification.

""" + notes.split("## Best put buffer", 1)[0].split("\n", 2)[-1] + "\n## Best put buffer" + notes.split("## Best put buffer", 1)[1]
    text += "\n## Common research period and exposure\n"+common
    text += """\n## Metrics and validation

CAGR uses actual calendar years (365.2425 days). Annualized volatility is sample daily-return standard deviation × √252. Sharpe is mean daily return divided by sample standard deviation × √252, with zero cash/risk-free return. Maximum drawdown includes initial equity in the running peak. Sharpe is descriptive for buffers.

**Calculation validation.json** records the current independent quote audit, full-budget checks and daily accounting reconciliation. **Chart inventory validation.json** records export completeness and hashes of preserved category files. **All strategy metrics.csv** and **Both/Selected hedges.csv** contain the current results.

Research inputs remain in `spx_option_research/results/organized_spx_combinations`; original curves remain in `organized_spx_research`. Current buffer results are in `organized_spx_best_hedges`. Curves are reconstructed in memory from audited inputs and current rules. Prior Sharpe-buffer metadata is explicitly archived as superseded and is not validation of the corrected model.

Rebuild numerical results with `python -B spx_option_research/scripts/rebuild_premium_buffers.py --build`; audit with `verify_premium_buffers.py`; render with `render_premium_buffers.py`; publish these notes with `publish_premium_buffer_notes.py`.
"""
    (client/"Methodology and coverage.md").write_text(text, encoding="utf-8")
    print("Published corrected buffer methodology and selection summaries", flush=True)


if __name__ == "__main__":
    run()
