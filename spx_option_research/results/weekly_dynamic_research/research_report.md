# Weekly dynamic SPX put-spread research

## Main result

The adaptive dynamic policy did **not** improve the primary risk-adjusted result. It earned a 5.09% CAGR versus 4.77% for fixed 99/96, but its daily Sharpe was 0.90 versus 1.00 and its maximum drawdown was -8.72% versus -4.45%. The fixed-width dynamic policy also failed, with a 0.01% CAGR and 0.02 Sharpe. The validation-best fixed spread produced a 3.51% CAGR and 1.52 Sharpe.

The adaptive policy traded 141 of 141 weeks, averaged a 4.91% target width and 0.290 net delta. Fixed 99/96 averaged 0.227 net delta. Its higher return therefore came with more option exposure, not a better Sharpe ratio.

## Test definition

The ten-year September 2016–September 2026 archive supplies 520 usable weekly cycles and 14,501 candidate spreads. The grid uses short strikes at 97% through 103% of cash SPX and hedge widths of 1%, 2%, 3% and 5% of spot. Research compared 86 features, 28 regression/feature combinations, four economic or relative-IV rules, and 199 policy specifications. Earlier observations supply training history; there are no ten-year independent model test returns.

The final evaluation covers January 05, 2024 through September 18, 2026. All policies below were frozen from 2021–2023 validation before final-period results were evaluated. Entry decisions and fills use the EOD option snapshot. The comparison uses 100% of current equity in cash-SPX notional, fractional contracts, daily option marks, a zero cash return, and realistic execution at 25% of the full bid/ask spread from midpoint plus $1.50 per leg. The modeled quantity is joint vertical compensation: executable net credit less the expected two-leg terminal liability, scaled by risk.

The fixed-width dynamic winner uses the normal-RV economic rule and trades only when expected net edge divided by maximum loss is at least 0.10. The adaptive-width winner uses the frozen ridge core-plus-Greeks model and requires positive predicted net edge. These rules and thresholds come directly from `frozen_model_choices.json`.

## Final-period performance

| Frozen portfolio | Realistic CAGR | Daily Sharpe | Max drawdown | Cash rate | Natural CAGR |
|---|---:|---:|---:|---:|---:|
| Dynamic, fixed 3% width | 0.01% | 0.02 | -3.35% | 73.76% | -0.45% |
| Dynamic, adaptive width | 5.09% | 0.90 | -8.72% | 0.00% | 4.71% |
| Validation-best fixed 97%/94% | 3.51% | 1.52 | -3.17% | 0.00% | 3.42% |
| Fixed 99/96 | 4.77% | 1.00 | -4.45% | 0.00% | 4.64% |
| Economic rule, fixed width | 3.35% | 0.71 | -6.75% | 0.71% | 2.07% |
| Economic rule, adaptive width | 2.03% | 0.47 | -6.76% | 0.71% | 0.65% |

[Daily equity and drawdown](holdout_equity_drawdown.png)

## Calendar returns

| Portfolio | 2024 | 2025 | 2026 |
|---|---:|---:|---:|
| Dynamic, fixed 3% width | 0.64% | 0.24% | -0.84% |
| Dynamic, adaptive width | 7.81% | 0.78% | 5.24% |
| Validation-best fixed 97%/94% | 1.50% | 3.55% | 4.45% |
| Fixed 99/96 | 2.54% | 4.54% | 5.81% |
| Economic rule, fixed width | 1.88% | 0.99% | 6.24% |
| Economic rule, adaptive width | 0.75% | 0.49% | 4.27% |

Calendar returns compound actual daily marked returns within each year. 2024 covers January 05 through December 31; 2026 covers January 02 through September 18.

## What the frozen policies selected

| Portfolio | Trades | Cash rate | Mean short | Mean width | Mean net delta | Mean net credit | Historical-model edge | Historical short edge | Historical hedge edge | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dynamic, fixed 3% width | 37 | 73.76% | 101.9% | 3.0% | 0.416 | 1.94% | 0.33% | 0.21% | 0.13% | 40.5% |
| Dynamic, adaptive width | 141 | 0.00% | 98.7% | 4.9% | 0.290 | 0.64% | 0.02% | 0.04% | -0.02% | 84.4% |
| Validation-best fixed 97%/94% | 141 | 0.00% | 97.0% | 3.0% | 0.070 | 0.10% | 0.02% | 0.05% | -0.03% | 98.6% |
| Fixed 99/96 | 141 | 0.00% | 99.0% | 3.0% | 0.227 | 0.32% | -0.01% | 0.03% | -0.04% | 81.6% |
| Economic rule, fixed width | 140 | 0.71% | 100.4% | 3.0% | 0.311 | 1.29% | 0.16% | 0.11% | 0.05% | 66.4% |
| Economic rule, adaptive width | 140 | 0.71% | 100.3% | 2.5% | 0.226 | 1.06% | 0.15% | 0.11% | 0.04% | 65.0% |

Net credit and economic-edge components are fractions of entry cash-SPX notional. The three historical-model edge columns use the same filtered historical simulation diagnostic for every portfolio. They are not the ridge model's forecast or the normal-RV rule's forecast. “Historical hedge edge” is that estimator's expected long-put payoff less its executable cost. The selected policy's actual forecast net edge and its realized mean weekly P&L are separately saved as `mean_policy_forecast_edge` and `mean_weekly_return` in `holdout_selection_summary.csv`. Fixed benchmarks have no policy forecast. The policy does not require every selected hedge to be independently cheap. Cash weeks remain in every performance statistic with zero option return.

[Selection distributions and cash rates](holdout_selection_distribution.png)

## Validation feature ladder

Every primary feature comparison uses the same `positive_edge` gate. The full 56-row core/add-family/full/leave-one-out table is in `validation_feature_ladder_positive_edge.csv`; all 168 model-ladder rows across every gate are retained in `validation_feature_ladder_all_gates.csv`. `validation_policy_results_all_rows.csv` preserves all 199 validation policies, including rules and fixed spreads. No gate or feature set was chosen from final-period results.

All regression policies share a historical estimate of spread-liability risk in their ranking denominator. Adding or removing the economic feature group changes the prediction inputs, while retaining this common risk normalization. Both the forecast net premium advantage and this denominator vary by spread and week.

| Model family | Universe | Core Sharpe | Full Sharpe | Full minus core |
|---|---|---:|---:|---:|
| Ridge | Fixed 3% width | 0.15 | -0.88 | -1.03 |
| Ridge | Adaptive width | 0.83 | -0.77 | -1.60 |
| Boosting | Fixed 3% width | -0.03 | -0.21 | -0.19 |
| Boosting | Adaptive width | -0.24 | -0.44 | -0.20 |

For the ridge adaptive-width comparison, the exact core and single-family additions were:

| Ridge adaptive feature set | Validation Sharpe | Validation CAGR | Active weeks |
|---|---:|---:|---:|
| Core | 0.83 | 3.39% | 155 |
| Add economic | 0.34 | 1.73% | 151 |
| Add greeks | 1.20 | 5.96% | 155 |
| Add iv skew | 0.67 | 3.37% | 153 |
| Add vol regime | 0.81 | 4.05% | 155 |
| Add momentum | 0.38 | 2.00% | 131 |
| Add liquidity | -0.05 | -0.40% | 155 |
| Full | -0.77 | -6.51% | 138 |

Core plus Greeks was the strongest single-family addition in this fixed 2021–2023 validation comparison. The weaker results for other families in this sample do not show that those inputs are universally useless; they only describe these predefined models, this gate, and this validation period.

[Validation feature ladder](validation_feature_ladder.png)

## Paired uncertainty versus fixed 99/96

| Dynamic portfolio vs fixed 99/96 | Sharpe difference | 95% interval | Annual arithmetic-return difference | 95% interval |
|---|---:|---:|---:|---:|
| Dynamic, fixed 3% width | -0.98 | [-2.21, 0.02] | -4.75% | [-8.79%, -0.58%] |
| Dynamic, adaptive width | -0.10 | [-0.88, 0.80] | 0.35% | [-3.82%, 5.02%] |

The intervals are paired four-week block-bootstrap intervals from the frozen final-period daily paths. They describe sampling uncertainty within this history and do not convert the exploratory model comparison into an independent replication.

## Predefined robustness checks

### Daily-mark bound sensitivity

| Portfolio | Observed-mid Sharpe | Bounded-mid Sharpe | Difference | Flagged marks | Largest correction |
|---|---:|---:|---:|---:|---:|
| Dynamic, fixed 3% width | 0.02 | 0.02 | -0.00 | 2 | 0.50 points |
| Dynamic, adaptive width | 0.90 | 0.90 | 0.00 | 0 | 0.00 points |
| Validation-best fixed 97%/94% | 1.52 | 1.52 | 0.00 | 0 | 0.00 points |
| Fixed 99/96 | 1.00 | 1.00 | 0.00 | 0 | 0.00 points |

The bounded convention changes only interim quoted spread marks that fall outside the vertical’s payoff bounds; entry execution and expiration cash settlement stay fixed.

### Timing and strike attribution

| Dynamic portfolio | Chosen-spread Sharpe | 99/96 on same trade weeks | Difference | Chosen CAGR | Same-week 99/96 CAGR |
|---|---:|---:|---:|---:|---:|
| Dynamic, fixed 3% width | 0.02 | 0.33 | -0.31 | 0.01% | 0.81% |
| Dynamic, adaptive width | 0.90 | 1.00 | -0.10 | 5.09% | 4.77% |

The counterfactual preserves each dynamic policy’s cash/trade timing and substitutes 99/96 only on its traded weeks. It isolates spread choice from the value of staying in cash. For the fixed-width dynamic policy, same-week 99/96 also performed materially better, so the weak result was not explained by cash timing alone.

## Verification and execution limits

All 73 automated tests passed. An independent reconstruction from individual leg quotes, entry fills and cash settlement reproduced all 12 performance rows across six portfolios and two execution assumptions, with maximum daily NAV error below $0.000000003. The [data-quality audit](data_quality_audit.md) and [method audit](method_audit.md) document sources, units, timing and accounting.

Same-EOD option features and fills are a research assumption. The archive does not establish that final option volume, open interest and quotes were available early enough to calculate and execute orders at these exact prices. Cash earns zero, and Sharpe is annualized from daily returns using sqrt(252). These results do not establish a deployable improvement over weekly 99/96.
