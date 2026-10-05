# Earlier weekly spread comparison

This folder preserves the comparison code and available result tables and charts.
The following older artifacts were unavailable in the original local snapshot and
are omitted: `moneyness_cadence_daily_paths.parquet`,
`moneyness_cadence_entries.parquet`, and `moneyness_cadence_equity_curves.png`.
Their corresponding CSV equity, P&L, summary, and trade outputs are included.

With the source archive configured as described in the root README, regenerate
the comparison from the repository root:

```powershell
python -B spx_weekly_put_spread/compare_moneyness_cadence.py
```
