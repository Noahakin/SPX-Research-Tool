"""Write portable client-library notes and verify the expected chart inventory."""
from __future__ import annotations

import json
import hashlib
import argparse
import re
from pathlib import Path
import shutil

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from organized_spx_data import DEFAULT_OUTPUT


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    preferred=DEFAULT_OUTPUT.parent/"organized_spx_combinations"
    parser.add_argument("--data-dir",type=Path,default=preferred if (preferred/"run.json").exists() else DEFAULT_OUTPUT)
    args=parser.parse_args()
    data = args.data_dir.resolve()
    client = data.parents[2] / "SPX Research Organized Charts"
    metrics = pd.read_csv(data / "strategy_metrics.csv", dtype={"folder":str})
    run = json.loads((data / "run.json").read_text())
    validation = json.loads((data / "validation.json").read_text())
    evidence = pd.read_csv(data / "quote_repair_evidence.csv")
    tenors = [(3,"3 days"),(7,"1 week"),(14,"2 weeks"),(21,"3 weeks"),(28,"4 weeks"),(42,"6 weeks"),(56,"8 weeks")]
    expected = [client / "All strategies.png"]
    intermediate=set()
    for category in metrics.category.unique():
        expected.append(client / category / "All strategies.png")
        for i,(_,name) in enumerate(tenors,1):
            expected.append(client / category / f"{i:02d} - {name} comparison.png")
    for (category,folder), group in metrics.groupby(["category","folder"]):
        expected.append(client / category / folder / "All expiries.png")
        parts=Path(folder).parts
        for count in range(1,len(parts)):
            intermediate.add(client / category / Path(*parts[:count]) / "All strategies.png")
        for i,(_,name) in enumerate(tenors,1):
            expected.append(client / category / folder / f"{i:02d} - {name}.png")
    expected.extend(sorted(intermediate))
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} expected charts: {missing[:5]}")
    wrong_size = []
    for path in expected:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (1600,1000):
                wrong_size.append(str(path))
    if wrong_size:
        raise ValueError(f"Unexpected chart resolution: {wrong_size[:5]}")
    leaf_count=metrics.groupby(["category","folder"]).ngroups
    overview_count=1+metrics.category.nunique()*8
    assert len(expected)==len(metrics)+leaf_count+len(intermediate)+overview_count and (client / "index.html").is_file()
    for source, destination in (("strategy_metrics.csv","All strategy metrics.csv"),
                                ("quote_repair_evidence.csv","Estimated quote evidence.csv"),
                                ("validation.json","Calculation validation.json"),
                                ("run.json","Research parameters.json")):
        atomic_copy(data/source,client/destination)
    rows = []
    for days,name in tenors:
        sample = metrics[metrics.tenor_days.eq(days)]
        rows.append(f"| {name} | {int(sample.actual_dte_min.min())}–{int(sample.actual_dte_max.max())} calendar days | {int(sample.cycles.iloc[0])} |")
    partial = int(metrics.cash_cycles.gt(0).sum())
    quote_count=pq.ParquetFile(data/"daily_quotes.parquet").metadata.num_rows
    test_path=data/"test_results.txt"
    test_match=re.search(r"Ran (\d+) tests",test_path.read_text(encoding="utf-8",errors="replace")) if test_path.exists() else None
    test_count=test_match.group(1) if test_match else "the recorded"
    combination_count=int(metrics.category.eq("Both").sum())
    note = f"""# SPX Research Organized Charts

Open **index.html** in Edge or Chrome to filter by strategy, strike, width, and expiration. Everything works offline. Keep the folder together when copying it to another computer.

The library contains **{len(metrics):,} individual strategy charts**, **{leaf_count:,} comparisons across expirations**, **{len(intermediate):,} intermediate-folder comparisons**, and **{overview_count} category/root comparisons**: **{len(expected):,} PNG images**, each 1600 × 1000 pixels. Every individual chart shows one daily equity curve, CAGR, annualized volatility, and daily Sharpe.

## Scope and folder names

- **Put selling**: short single puts and credit put spreads. `103-100` means sell the put near 103% of SPX and buy the put near 100%.
- **Put buying**: long single puts and debit put spreads. `103-100` means buy the put near 103% and sell the put near 100%.
- **SPX with overlay**: the former **Both** category, renamed. It holds 100% SPX price exposure plus each long or short put strategy, including protective puts, put-spread buffers, and short-put overlays.
- **Both**: a short put spread combined with a same-expiration downside hedge. The hedge is either an additional long put or a long downside put spread funded by part of the short spread's net premium. It has no SPX holding.

The original categories use requested strikes from **90% through 110% in one-percentage-point increments**. Spread widths are **1, 2, 3, 5, and 10 percentage points**, with both requested legs inside 90%–110%. These are 21 single puts and 84 vertical spreads in each direction. The new Both category uses all 84 short spreads and adds hedge legs down to **80%** where needed for downside buffers.

## Premium-funded combinations

For each short spread, Both tests spending **up to 10%, 25%, or 50%** of its net entry credit with three hedge choices. The hedge starts at the short spread's lower strike: a long put, a five-percentage-point downside buffer, or a ten-percentage-point downside buffer. A buffer buys that put and sells the lower-strike put. Its payout is capped by its actual strike width.

For example, selling **98/95** funds an additional **long 95 put**, **long 95/90 buffer**, or **long 95/85 buffer**. Each is tested at all three premium budgets and all seven target expirations. This produces **{combination_count:,} combined-strategy curves**.

**Net short credit C** is the short vertical's entry premium after its slippage and commissions. **Hedge debit D** includes the hedge's slippage and commissions. If the budget fraction is **f**, hedge contracts per short spread equal **q = min(1, f × C / D)**. The strategy retains **C − q × D**, at least **(1 − f) × C**, in net premium. The short spread has 100% current-equity SPX notional; the added hedge is capped at **100% SPX notional**. Quantities are fixed until the next roll. The shared middle put has total long quantity **1 + q**; a downside buffer adds short quantity **q** at its lower strike. Execution costs scale with every contract.

A 25% premium budget is a limit of **25% of net option credit**, not a 25% allocation of portfolio equity or a guaranteed 25% hedge of SPX. When the notional cap binds, the unspent premium stays in cash. Mean and maximum hedge notionals, cap-binding counts, and mean premium fraction spent are included in the metrics CSV. The cap prevents penny-priced tail options from creating extreme quantities merely to consume the full budget. The entire combination stays in cash if C or D is nonpositive, net base credit exceeds its strike width, the buffer's quoted midpoint is outside its payoff bounds, an entry quote is invalid or estimated, a requested strike is unavailable, or a spread collapses. No hedge is sized using a later price or outcome.

Both folders are organized as **short spread → hedge → premium budget → expiration images**. Base-spread and hedge folders contain **All strategies.png** comparisons; budget folders contain **All expiries.png**. Browser filters include premium budget and hedge type/strikes, with direct comparison views for these intermediate levels.

Every moneyness folder contains seven individual expiration images and **All expiries.png**. Each category contains **All strategies.png** and seven comparisons at the same expiration. The top-level **All strategies.png** contains every curve. All lines in a comparison represent separate portfolios.

## Common research period and exposure

The analysis runs from **September 23, 2016 through September 18, 2026**, approximately ten years ({run['elapsed_years']:.4f} elapsed years), with **2,510 cash-market observations**. Every strategy starts with **$1,000,000**. This uses the full common archive history, not a recent holdout period.

Standalone option notional and the short spread in Both are **100% of current portfolio equity** at every roll, measured as contract quantity × SPX at entry × the 100 option multiplier. Fractional contracts are allowed. Quantities remain fixed between rolls; sizing is not reset daily. Both adds the variable hedge quantity defined above. The library's baseline is 100% short-spread notional.

**SPX with overlay** also holds SPX price exposure equal to 100% of equity at each roll. It keeps that underlying exposure during cycles when the option is unavailable. SPX price returns exclude dividends. Cash earns zero; financing, borrowing costs, and taxes are not modeled. SPX plus a debit hedge can require financing, so these curves are not fully funded total-return indices. Both holds only its option combination and cash collateral.

## Expiration conventions

| Target | Actual entry DTE across available trades | Scheduled cycles |
|---|---:|---:|
{chr(10).join(rows)}

One-, two-, three-, four-, and six-week strategies use fixed calendar-Friday anchors, shifted to the prior cash session for holidays. They hold to the exact PM expiration on the scheduled date. One unavailable expiration ending December 16, 2016 becomes an explicit cash cycle for each of these five schedules.

The three-day target renews at expiration into the nearest listed PM expiry within one to five calendar days, choosing the longer expiry on a tie. Early availability often produces four calendar days. The eight-week target renews at expiration into the nearest listed Friday or holiday-adjusted-Friday PM expiry, also choosing the longer expiry on a tie. Exact eight-week expirations were unavailable in much of the early archive; actual eight-week-target trades therefore span 49–63 days.

No new position opens on the final analysis day. A position expiring later remains marked at its observed final-date midpoint; no hypothetical closing commission is charged. Only PM SPXW contracts are used. Invalid non-cash-session expiration labels are excluded rather than silently reinterpreted.

## Strikes, fills, and daily valuation

Target strikes use the observed **cash SPX close**, not the archive's expiration-specific underlying field. The nearest listed strike is selected before quote screening, with the lower strike winning exact-distance ties. If any required selected leg differs from its target by more than **0.5 percentage point of SPX**, if spread legs collapse, or if an entry quote is invalid or estimated, that option cycle stays in cash. Entry decisions do not depend on future outcomes. The original fixed strategies retain negative net-credit trades after costs. Both requires positive net credit to fund its hedge and therefore skips those cycles.

There are {partial:,} curves with at least one option cash cycle. Individual images disclose the number of option rolls in cash. Particularly far-in-the-money targets can have substantial missing-strike coverage; those are **trade-when-representable** strategies, not uninterrupted positions at the nominal target. The CSV includes cycle counts and maximum traded strike deviation.

Long entries pay midpoint plus **25% of the full bid/ask spread**. Short entries receive midpoint minus the same amount. Each entry leg also pays **$1.50 per contract**. Positions settle to cash intrinsic at PM expiry. Interim holdings use the same contracts' daily bid/ask midpoints. Returns include entry friction on the first date and on every subsequent roll.

EOD execution at these quotes and simultaneous cash-close sizing are research assumptions. The archive's 16:00 timestamp is generated metadata, not independently verified exchange timing. Quoted spreads and execution estimates do not establish obtainable live fills or capacity.

## Quote quality and estimates

The combined extractor found all **{quote_count:,} required daily date/contract records** and reconciled all entry records. Presence of a record does not guarantee a sound market quote. The audit identified defective zero/crossed quotes and large isolated cross-strike inconsistencies, including during the February 2020 selloff and June 29, 2020, plus zero/zero penny-put records at the deeper downside strikes.

**{int(evidence.status.eq('estimated').sum())} contract/date values use explicitly estimated marks.** Isolated defects use nearby puts with the same date and expiration. Where a bracket is unavailable, the fallback estimates a put through its same-strike call and put-call parity, fitting discount/forward terms from valid paired contracts in the same snapshot. This does not use the later outcome or cash close to force the fit. Evidence includes raw values, estimated values, and supporting observations/calibration. Estimates are research valuations, not recovered observed prices. They are never used to justify a new entry.

For far-downside penny puts, a same-expiry bracket can extend up to 50 strike points on each side when both observed donor asks are at most 0.25 option points. This accommodates coarser strike listings while keeping material-price options subject to the original tighter bracket. These marks are also explicitly estimated, and all raw records remain preserved.

Residual inconsistencies in observed leg midpoints remain; synthetic spread marks can occasionally lie outside theoretical payoff bounds because independently quoted legs are noisy. They are logged in the research artifacts. No general clipping to payoff bounds or forward filling is applied. Individual charts identify the estimated option marks they use. These data and execution limits matter when presenting Sharpe estimates as historical research.

## Metrics and verification

- **CAGR**: final equity divided by initial equity, raised to 1 / elapsed calendar years, minus one; a year is 365.2425 days.
- **Annualized volatility**: sample standard deviation of daily portfolio returns × √252.
- **Sharpe**: mean daily portfolio return divided by its sample standard deviation × √252, with a zero cash/risk-free rate. This uses daily marked equity, not expiration-only returns.

Saved daily values and metrics were independently reconciled for all {len(metrics):,} curves. The combination audit reprices {validation['independently_repriced_ledger_rows']:,} cycle-ledger rows from entry quotes and terminal settlement/mark and verifies premium funding, shared-leg quantities, roll-date costs, and compounding. All 2,940 original equity curves remain unchanged, including the previously audited weekly 96/93, 97/94, 98/95, and 99/96 portfolios. The output image inventory and PNG integrity were verified.

The complete automated test suite passed **{test_count} tests**, including premium-funded combination sizing and cash rules, portfolio accounting, initial and roll costs, final unexpired positions, quote-estimate safeguards, and parity recovery/rejection checks.

**All strategy metrics.csv** contains machine-readable results. **Estimated quote evidence.csv** preserves the adjustment evidence. **Calculation validation.json** records accounting checks. Reproducible scripts are in `spx_option_research/scripts`; original research remains in `results/organized_spx_research` and the combined library and new ledgers are in `results/organized_spx_combinations` under that project.
"""
    atomic_write(client / "Methodology and coverage.md", note)
    report = dict(expected_pngs=len(expected),verified_pngs=len(expected),resolution=[1600,1000],
                  strategy_folders=leaf_count,intermediate_overlays=len(intermediate),individual_curves=len(metrics),offline_browser=True,
                  source_sha256={name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ("run.json","strategy_metrics.csv","curves.npz")})
    atomic_write(client / "Chart inventory validation.json", json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
