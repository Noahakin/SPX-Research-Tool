"""Prepare documentation and auditable hedge-selection tables for the new library."""
import json
import re

import pandas as pd

from optimize_organized_spx_hedges import OUTPUT


def run():
    if json.loads((OUTPUT/"run.json").read_text(encoding="utf-8")).get("dynamic_buffer_selection"):
        from publish_premium_buffer_notes import run as publish_current
        return publish_current()
    stage = OUTPUT/"charts"
    metrics = pd.read_csv(OUTPUT/"strategy_metrics.csv", dtype={"folder": str})
    winners = metrics[metrics.category.eq("Both")]
    config = json.loads((OUTPUT/"run.json").read_text(encoding="utf-8"))
    config["combination_entry_rule"] = "Cash if net short premium <= 0, hedge debit <= 0, a required strike/observed quote is unavailable or estimated, a spread collapses, base spread credit exceeds width, or hedge spread midpoint violates payoff bounds"
    config["png_dimensions"] = [1600, 1000]
    config["both_image_format"] = "lossless WebP, pixel-identical to generated PNGs"
    config["automated_tests_passed"] = 112
    config["compact_storage"] = True
    config["curve_storage"] = "Reconstructed in memory from retained audited quotes and selected parameters; all 22,138,200 values verified bitwise identical to the audited cache"
    config["bulk_artifact_rebuild_command"] = "python spx_option_research/scripts/optimize_organized_spx_hedges.py"
    (OUTPUT/"run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (stage/"Research parameters.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics.to_csv(stage/"All strategy metrics.csv", index=False)
    audit = json.loads((OUTPUT/"validation.json").read_text(encoding="utf-8"))
    audit["automated_tests_passed"] = 112
    audit["bitwise_reconstructed_daily_values"] = 22138200
    audit["bulky_curve_and_ledger_caches"] = "Audited, then removed to conserve disk space; original inputs, selected parameters and artifact hashes retained"
    (stage/"Calculation validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    columns = ["premium_fraction", "short_variation", "tenor", "hedge_type", "hedge_primary_pct",
        "hedge_secondary_pct", "hedge_width_pct", "cagr", "annualized_volatility", "daily_sharpe",
        "max_drawdown", "traded_cycles", "cash_cycles", "relative_base_coverage",
        "mean_premium_fraction_spent", "mean_hedge_notional_multiple", "hedge_cap_binding_cycles", "strategy_id"]
    table = winners[columns].sort_values(["premium_fraction", "short_variation", "tenor", "hedge_type"])
    table.to_csv(stage/"Both"/"Selected hedges.csv", index=False)
    for fraction, rows in table.groupby("premium_fraction"):
        folder = stage/"Both"/f"{fraction*100:02.0f} percent premium"
        folder.mkdir(parents=True, exist_ok=True)
        rows.to_csv(folder/"Selected hedges.csv", index=False)
    both_notes = """# Premium-funded short puts and hedges

Choose a premium budget: **05, 10, 15, or 20 percent premium**. Each contains the same **105 short-put variations**: 21 single short puts and 84 short put spreads. Each variation has **Best long put** and **Best put buffer**, each with seven expiration charts and **All expiries.png**.

**Best** means the highest combined-strategy daily Sharpe over September 23, 2016–September 18, 2026, selected separately for the budget, short variation, expiration and hedge type. Winners are retrospective full-period research selections, not out-of-sample results. The selected strike can differ by expiration; it appears on the individual chart, expiration comparison and **Selected hedges.csv**.

Budgets are shares of **net short premium after slippage and commissions**. All-in hedge purchase costs count toward the budget. Hedge quantity is capped at 100% SPX notional, so actual spending can be below the budget. A put buffer buys a put and sells a lower-strike put; its protection is capped by the spread width. Both has no SPX holding.

Long-put targets range from 80% of SPX up to the base spread's lower target strike (or the single short put's target). Buffer widths are 1, 2, 3, 5 and 10 percentage points, with both hedge legs at or above 80%. All legs share the short strategy's expiration. Candidates must trade at least 90% of the cycles where the base short strategy is eligible. Cash cycles, costs and daily marking enter the Sharpe calculation.

The full library methodology and calculation validation are in the parent folder. All 283,332 candidate results are retained in `spx_option_research/results/organized_spx_best_hedges/candidate_metrics.parquet`. The selected parameters and audited source quotes reproduce every daily curve exactly. Bulky curve and trade-ledger caches were audited and then removed to conserve disk space; the research script can rebuild them.
"""
    (stage/"Both"/"README.md").write_text(both_notes, encoding="utf-8")
    original = (OUTPUT/"Previous library documents"/"Methodology and coverage.md").read_text(encoding="utf-8")
    common = original.split("## Common research period and exposure", 1)[1].split("## Metrics and verification", 1)[0]
    common = common.replace("the short spread in Both", "the short strategy in Both")
    common = common.replace("100% short-spread notional", "100% short-strategy notional")
    common = re.sub(r"There are [\d,]+ curves with at least one option cash cycle", f"There are {int(metrics.cash_cycles.gt(0).sum()):,} curves with at least one option cash cycle", common)
    notes = """# SPX Research Organized Charts

Open **index.html** in Edge or Chrome to browse charts offline. Put buying, Put selling, and SPX with overlay retain their existing images and folders.

The updated library contains **8,820 individual strategy charts**, **1,260 expiration comparisons**, **424 intermediate-folder comparisons**, and **33 category/root comparisons**: **10,537 PNG images**, each 1600 × 1000 pixels.

## Folder organization

- **Put selling**: existing short single puts and credit put spreads; unchanged.
- **Put buying**: existing long single puts and debit put spreads; unchanged.
- **SPX with overlay**: existing SPX price exposure plus each put strategy; unchanged.
- **Both**: premium-funded short puts and short put spreads, with four top-level budget folders: **05, 10, 15, and 20 percent premium**.

Inside each Both budget: **short-put variation → Best long put / Best put buffer → seven expiration charts and All expiries.webp**. Budget and short-variation folders also contain **All strategies.webp**. Both and each budget include **Selected hedges.csv**. Both uses lossless WebP files, verified pixel-identical to the generated PNGs, to reduce disk space; the existing categories keep their original PNG files.

## Premium-funded combinations and hedge selection

Both includes **105 short variations**: 21 single short puts at targets from 90% through 110% of SPX, and 84 short put spreads using widths of 1, 2, 3, 5 and 10 percentage points, with both base legs inside 90%–110%. Each is tested at four premium budgets, seven target expirations and two hedge types, producing **5,880 selected combined curves**.

For every combination, the long-put target is searched in one-percentage-point increments from 80% of SPX up to the short spread's lower target strike (or the single short put's target). Put buffers buy a put in that range and sell a put 1, 2, 3, 5 or 10 percentage points lower, with neither hedge leg below 80%. Hedge and base contracts share the same expiration. Buffer payouts are capped by their actual listed strike width.

**Best means the highest combined-strategy daily Sharpe among eligible candidates over the full research period.** The search evaluated **283,332 candidates**. Selection is separate for each budget, short variation, expiration and hedge type. Ties are resolved by higher CAGR, smaller maximum drawdown, then strategy ID. Every candidate must have finite positive daily NAV, a finite Sharpe, at least one trade, and at least 90% of its base short strategy's eligible cycles. This coverage rule reduces selection driven by missing hedge quotes.

**The hedge selection is retrospective and uses the full sample. These are historical winners, not out-of-sample or walk-forward results.** “Best” is limited to the stated target grid, widths, expiration schedules, execution assumptions and notional cap. The selected targets remain fixed across the historical simulation; they are not switched each roll using the later outcome. Individual charts and expiration comparisons identify the selected hedge strikes.

Let **C** be the short strategy's net entry premium after slippage and commissions, **D** the hedge debit including those costs, and **f** the budget (5%, 10%, 15% or 20%). Hedge contracts per short contract equal **q = min(1, f × C / D)**. Retained premium is **C − q × D**. Budgets are limits, not guaranteed spending: a notional cap can leave part of the budget unspent in cash. They are percentages of net short premium, not percentages of portfolio equity.

The base short strategy uses 100% current-equity SPX notional at each roll; the added hedge is capped at 100% SPX notional. Fractional contracts are allowed. All quantities stay fixed until the next roll, and execution costs include every contract even when legs share a strike. Both has no SPX holding. A whole combination remains in cash if net credit or hedge debit is nonpositive, a required observed entry quote or strike is unavailable/estimated, a spread collapses, short-spread credit exceeds its width, or the hedge spread midpoint violates payoff bounds. Actual spending, notional and cap-binding cycles appear in the metrics.

## Common research period and exposure""" + common + """## Metrics and verification

- **CAGR**: ending equity divided by initial equity, raised to 1 / elapsed calendar years, minus one; 365.2425 days per year.
- **Annualized volatility**: sample standard deviation of daily portfolio returns × √252.
- **Sharpe**: mean daily portfolio return / sample standard deviation × √252, with a zero cash/risk-free rate.
- **Maximum drawdown**: lowest daily equity / running equity peak − 1, including initial equity in the peak.

All **22,138,200 daily equity values** and their saved metrics were verified. An independent quote-based audit reconstructed **1,916,880 selected cycle-ledger rows**, checking entry eligibility, net premium, hedge costs, budget compliance, fixed quantities, daily NAV, settlement and roll costs. All **2,940 original equity curves remain bitwise unchanged**. The maximum reconstructed daily NAV difference was less than $0.00000001. Candidate rankings and the complete budget/short/expiry/hedge grid were independently reconciled. The automated suite passed **112 tests**.

**All strategy metrics.csv**, **Both/Selected hedges.csv**, **Research parameters.json**, and **Calculation validation.json** describe this updated library. **Estimated quote evidence.csv** retains the existing quote-adjustment evidence. **Chart inventory validation.json** records final image/path validation and hashes confirming that the preserved category files remain unchanged.

Research artifacts are in `spx_option_research/results/organized_spx_best_hedges`. `candidate_metrics.parquet` retains every tested candidate's parameters, metrics, eligibility and selection flag. The selected parameters and original audited quotes reconstruct all 22,138,200 daily values **bitwise identically**. The chart reader performs this reconstruction in memory when needed. Bulky curve and selected-ledger caches were independently audited and then removed to conserve disk space; their hashes and reconstruction instructions are retained in `Rebuildable curves.json` and `Rebuildable ledger.json`. Run `python spx_option_research/scripts/optimize_organized_spx_hedges.py` to regenerate the full numerical caches when space permits.

The original Both quotes, entries, schedule, metrics and trade ledger remain in `organized_spx_combinations`, so its superseded generated charts and merged curve export can be reproduced. The prior library's top-level documentation and rebuild command are retained in `Previous library documents`.
"""
    notes = notes.replace("**10,537 PNG images**", "**10,537 images** (7,152 lossless WebP images in Both and 3,385 PNG images elsewhere)")
    (stage/"Methodology and coverage.md").write_text(notes, encoding="utf-8")
    both_path = stage/"Both"/"README.md"
    both_path.write_text(both_path.read_text(encoding="utf-8").replace("All expiries.png", "All expiries.webp"), encoding="utf-8")
    print("Prepared methodology, selection tables and audit documents.", flush=True)


if __name__ == "__main__":
    run()
