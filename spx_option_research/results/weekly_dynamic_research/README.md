# Weekly SPX model research artifacts

Start with [the research report](research_report.md) and [daily equity/drawdown chart](holdout_equity_drawdown.png).

The archive spans September 2016–September 2026. There are 520 usable weekly cycles and 14,501 candidate spreads. Models train on earlier matured trades, are selected using 2021–2023 validation, and are evaluated chronologically on January 5, 2024–September 18, 2026. The protocol and finalists were fixed before the final comparison. This does not represent ten years of independent model test returns.

## Results and audit trail

- [Final performance, including execution sensitivity](holdout_summary.csv)
- [Daily marked portfolio paths](holdout_daily_portfolios.csv)
- [Weekly portfolio ledger](holdout_weekly_portfolios.csv) and [chosen trades](holdout_selections.csv)
- [All 199 validation policies](validation_policy_results.csv)
- [Consistent feature comparisons](validation_feature_ladder_positive_edge.csv)
- [Frozen finalists and input hashes](frozen_model_choices.json)
- [Paired uncertainty intervals](holdout_paired_bootstrap.csv)
- [Interim-mark sensitivity](holdout_mark_sensitivity.csv)
- [Same-trading-weeks benchmark](holdout_timing_attribution.csv)
- [Data quality and Greek units](data_quality_audit.md)
- [Methodology audit](method_audit.md)
- [Independent NAV reconstruction](independent_final_daily_metrics_audit.csv)

The adaptive model did not improve daily Sharpe versus fixed weekly 99/96. Its slightly higher CAGR came with wider spreads and a larger drawdown. Neither its validation nor final positive-edge gate skipped a week. Fixed 97/94, chosen from validation before final evaluation, had the strongest final Sharpe among the frozen finalists.

## Reproduction

Run commands from `spx_option_research`. The saved candidate, market, feature and unit-mark Parquet files permit model reproduction without rescanning the large archive. The model script loads supplementary scikit-learn dependencies from `.research_dependencies`; the package versions are in `requirements-weekly-research.txt`.

```powershell
python scripts/run_weekly_dynamic_research.py --stage validation
python scripts/run_weekly_dynamic_research.py --stage final
python scripts/check_weekly_research_robustness.py
python scripts/analyze_weekly_research_results.py
python -m unittest discover -s tests -v
```

To rebuild inputs, see `build_weekly_market_features.py`, `build_weekly_spread_candidates.py`, `build_weekly_daily_marks.py`, and the model script's `--stage features`. Preserve the saved source hashes when comparing runs; refreshed market data can differ from the original snapshot.

Returns use fractional contracts at 100% current-equity SPX notional. Cash earns zero. Sharpe uses actual daily marked returns with 252-day annualization. Option features and execution share an end-of-day snapshot, an execution assumption that requires an intraday test before treating the strategy as executable live.
