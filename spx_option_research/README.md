# SPX Put-Spread and Downside-Hedge Research

The client chart library is in
[SPX Research Organized Charts](<../SPX Research Organized Charts/index.html>).
It covers September 23, 2016 through September 18, 2026 and contains Put selling,
Put buying, SPX with overlay, and Both. **SPX with overlay** is the former Both
category. **Both** has four premium-budget folders: 5%, 10%, 15% and 20%, then
short-put variations and long-put or buffer hedges. Buffers choose the closest
affordable 3–5-point protection at each entry and spend the entire allocated net
credit, with at least one buffer per short contract and no upper quantity cap.
Outright long puts retain the earlier retrospective Sharpe selections and
100% hedge-notional cap. This library uses 100% equity-based SPX
notional for the base option strategy, rather than the maximum-loss budgets in
the separate research runner described below.

The library's accounting, coverage, and execution assumptions are documented in
[Methodology and coverage](<../SPX Research Organized Charts/Methodology and coverage.md>).
Its current results and independent validation are in
`results/organized_spx_best_hedges`; audited quote inputs remain in
`results/organized_spx_combinations`; the original series remain in
`results/organized_spx_research`.

To resume chart exports and refresh the portable notes from the Raw Data folder:

```powershell
python -B spx_option_research/scripts/render_premium_buffers.py
python -B spx_option_research/scripts/publish_premium_buffer_notes.py
python -B spx_option_research/scripts/verify_best_spx_export.py
```

## Separate research runner

This repository evaluates defined-risk SPX option strategies using historical
IVolatility EOD quotes. The core income leg is always a short put spread. Hedge
families are layered onto stable core configurations only after chronological
train/validation/test screening.

## Run

```powershell
python spx_option_research/scripts/run_research.py
```

The runner performs a data audit, builds cached option selections and daily
marks, tests accounting invariants, screens core spreads, evaluates hedge
overlays, performs out-of-sample and robustness analysis, and writes tables,
charts, and a Markdown report under `results/`.

The full workflow is checkpointed in two commands:

```powershell
python spx_option_research/scripts/run_research.py
python spx_option_research/scripts/run_hedges.py
```

The first command screens the core put-spread grid and daily-marks stable
families. The second evaluates outright puts, long put spreads, put
butterflies, bounded 2x1 structures, same-expiration and calendar/diagonal
overlays, hedge budgets, robustness, charts, and the final report.

The source archive is expected at:

`04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m`

The archive contains actual EOD quotes through six calendar months. Therefore,
270- and 365-DTE hedge requests in the research brief cannot be tested from
this archive and are reported as unavailable rather than synthesized.

Broad comparisons use continuous contract equivalents so each strategy has the
same configured concurrent maximum-loss budget. Whole-contract feasibility and
one-contract weekly results are retained separately; this prevents wide spreads
from silently receiving more capital merely because one SPX contract is large.
