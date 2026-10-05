# SPX Research Tool

An offline interactive chart viewer and Python research library for SPX put spreads and downside hedges. The included chart data runs through September 18, 2026.

## Open the charts

Download or clone the complete repository, then open **[index.html](index.html)** in current Microsoft Edge or Chrome. No server, Python installation, API key, or internet connection is needed to view the included results. GitHub's file preview does not run the website; open the downloaded files locally.

The [interactive viewer](<SPX Research Interactive/index.html>) includes 7,350 strategies and 14,700 options-only/with-SPX lines. It opens with only SPX selected. Compare multiple strategies, change each line's SPX exposure, and choose a historical date window. The upper-left metrics show CAGR, annualized volatility, maximum drawdown, and Sharpe, including differences versus SPX for portfolios that include it.

Use **Sort by** to rank the filtered strategy list by CAGR or Sharpe, highest or lowest first. **Rank exposure** chooses options-only or with-SPX performance, independently of the exposure used when adding chart lines. Rankings follow the chart's current dates; unavailable metrics appear last. Standard date presets use compact precomputed rankings. Custom dates calculate the same metrics from the matching strategies' daily data, with progress shown while sorting.

The latest spread-selection experiments are separate from the interactive viewer's original strategy catalog:

- [Dynamic maturity selection](spx_option_research/results/dynamic_maturity_search/Report.md): 202,752 policies that choose strikes, width, and maturity at each scheduled entry, with 25% net-profit exits. Includes equity curves, trade ledgers, the complete grid, and selection/accounting audits.
- [Fixed maturity and profit-target grid](spx_option_research/results/maturity_profit_grid/Report.md): 734,720 fixed-rule combinations across maturity, strikes, widths, profit targets, cadence, and sizing.
- [Chart methodology and hedge rules](<SPX Research Organized Charts/Methodology and coverage.md>): buffers use the allocated premium to buy the closest affordable protection, preferring five-percentage-point width and allowing four or three. Sharpe does not select the buffers. Outright long-put targets retain the earlier retrospective selections.

## Repository contents

| Path | Contents |
| --- | --- |
| `SPX Research Interactive/` | Ready-to-open viewer and all 126 compressed daily-data shards |
| `spx_option_research/web/` | Viewer source assets |
| `spx_option_research/scripts/` | Data preparation, backtests, grid searches, audits, and chart generation |
| `spx_option_research/src/spxresearch/` | Shared research library |
| `spx_option_research/tests/` | Accounting, timing, selection, hedge, and overlay tests |
| `spx_option_research/results/` | Complete latest maturity studies and selected earlier research snapshots |
| `SPX Research Organized Charts/` | Original methodology and inventory metadata; static chart images are regenerated locally |
| `spx_weekly_put_spread/` | Earlier weekly spread comparison |

Raw IVolatility quote downloads, intermediate caches, installed dependencies, browser profiles, local credentials, and the duplicate static image library are excluded. The original image inventory describes the full local export, not images shipped in this repository. The interactive data includes those chart curves. Some older research folders contain only reports and summaries; their larger intermediate outputs must be rebuilt.

## Deploy to Vercel

Import **Noahakin/SPX-Research-Tool** using Vercel's GitHub integration, with the repository root (`./`) as **Root Directory** and **Other** as the **Framework Preset**. Deploy the `main` branch.

The committed `vercel.json` explicitly disables Python/framework detection, skips dependency installation, runs `node deployment/build.mjs`, and publishes `public/`. No Python entrypoint, environment variables, or API keys are needed for the website. If an existing Vercel project was detected as **Python**, change its Framework Preset to **Other** and redeploy the latest commit; clear old build/install/output overrides if present.

The build publishes the complete interactive chart dataset, methodology, and both completed maturity studies with their charts and downloads. Python research code and local caches remain outside the hosted output. Original `.html` links and paths containing spaces are preserved. The homepage links to the interactive viewer and both studies.

Use the GitHub import for this large dataset. Vercel's direct CLI source-upload limit on Hobby is 100 MB, smaller than this repository; that limit does not apply to the Git integration's repository checkout.

To check the same static build locally with Node.js 24:

```powershell
npm run build
python -B -m http.server 8080 --directory public
```

Open `http://localhost:8080/` in Edge. Python is used only as a convenient local file server in this example; Vercel serves the generated files directly. The root `requirements.txt` is solely for running the research locally.

## Python setup

Run commands from the repository root. The research environment used Python 3.14; dependencies are recorded in the requirements files.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For the earlier machine-learning selectors, also install:

```powershell
python -m pip install -r spx_option_research/requirements-weekly-research.txt
```

The included viewer does not require these packages. To refresh its interface from the source assets while keeping its existing dataset:

```powershell
python -B spx_option_research/scripts/build_interactive_spx.py --assets-only
```

After changing the chart dataset or statistics, regenerate the offline ranking tables with `npm run build:rankings`. The Vercel build also refreshes missing or outdated tables automatically. The tables retain full-precision CAGR and Sharpe for both exposures across all eight standard date presets.

## Rerun the research

Supply the historical IVolatility SPX archive at this path relative to the repository root:

```text
04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m
```

Keep the archive's existing partition layout and schemas. The full maturity studies use 2,510 daily snapshots from September 23, 2016 through September 18, 2026. The original data fingerprints and run assumptions are recorded in each study's `Protocol.json`. The cash SPX input used by these studies is included at `spx_option_research/results/weekly_dynamic_research/market_spx.csv`.

On Windows, an existing archive can be connected without copying it:

```powershell
New-Item -ItemType Directory -Force '04 - Options and Volatility/Raw Downloads/IVolatility/data/raw' | Out-Null
New-Item -ItemType Junction -Path '04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m' -Target 'D:\MarketData\spx_options_6m'
```

Replace the example target with the actual archive location. Then run:

```powershell
python -B spx_option_research/scripts/run_dynamic_maturity_study.py
python -B spx_option_research/scripts/run_maturity_profit_study.py
```

These are substantial computations, and they write their outputs into the corresponding results folders. Allow ample RAM and free disk space. Preserve the supplied snapshots if you want to compare a new run with the original findings.

The older core runner supports `--config` with `spx_option_research/config/base.json`. Its market-data step uses `FRED_API_KEY`, supplied as an environment variable or in a local, ignored `.env` file. Other older scripts may require additional cash-index or VIX datasets described by their source paths. Rebuilding the entire interactive dataset also requires the audited `organized_spx_*` intermediate results, which are excluded here; see [the research workflow](spx_option_research/README.md).

## Validation

```powershell
python -B -m unittest discover -s spx_option_research/tests -p test_dynamic_maturity.py -q
python -B -m unittest discover -s spx_option_research/tests -p test_maturity_grid.py -q
python -B -m unittest discover -s spx_option_research/tests -p test_interactive_spx.py -q
```

Historical winners and validation-selected winners are reported separately. The archive has been studied repeatedly; the later period is reused historical evidence, not an untouched out-of-sample test. See the reports for costs, fractional sizing, data-quality rules, and other assumptions.
