# Independent weekly-model method audit

The economic feature helper, portfolio evaluator, model-selection driver, and robustness script were independently reviewed on October 2, 2026. The reviewer did not select models or change parameters using final-period returns.

The chronology and accounting checks passed:

- Market indicators use the prior SPX trading session. Option quotes, Greeks, volume, and open interest use the entry snapshot under the stated same-EOD execution convention.
- The empirical economic model uses at most 156 historical weeks, requires 104, and excludes every observation whose expiration is on or after the current entry. Cash-return volatility also ends before entry. One week contributes one return observation regardless of its number of candidate spreads.
- Annual regression fits use only labels with expiration strictly before January 1 of the forecast year. Candidate sample weights sum to one for each training week. Training imputation and scaling do not use prediction-year observations.
- The explicit feature list excludes terminal payoffs, future SPX prices, and realized future P&L. Independent tests confirmed strict cutoff handling, invariance to changed current/future labels, and correct prediction assignment after candidate rows were shuffled.
- Models forecast terminal spread liability divided by strike width. Forecasts are clipped to the valid zero-to-one range, multiplied by the same candidate's width as a fraction of spot, then subtracted from executable net credit. Both-leg execution costs and commissions are already included in that credit.
- Daily NAV includes the first entry cost. At a roll, the old position settles before the new position is sized and charged its entry cost. Daily returns retain cash sessions, and terminal equity reconciles to both source trade P&L and compounded weekly returns. Drawdown includes initial capital as the starting peak.

Three dedicated test modules cover these checks: `test_weekly_economic_features.py` (8 tests), `test_weekly_portfolio_evaluation.py` (9 tests), and `test_weekly_model_audit.py` (3 tests). Each passed when run independently; the root agent subsequently reported 72 passing tests in the full suite.

The frozen-choice audit independently reproduced the rankings and checked both stored SHA-256 hashes against the validation CSV and amended protocol. All 199 validation policies used the identical January 8, 2021–December 29, 2023 interval: 749 daily sessions and 155 weekly decisions. The final selection pool excluded always-trade controls, fixed benchmarks, and the pure IV z-score diagnostic, and required at least 30 active validation weeks. There were 62 eligible policies per width universe.

| Frozen choice | Validation daily Sharpe | Active validation weeks |
|---|---:|---:|
| Three-percent width: normal-RV expected-edge rule, threshold 0.10 | 0.741 | 46 |
| Adaptive width: ridge model with core and Greek features, positive expected edge | 1.202 | 155 |
| Validation-best fixed spread: 97/94 | 0.873 | 155 |

The three-percent-width rule divides expected net edge by maximum loss. Its 0.10 gate therefore requires forecast net profit of at least 10% of maximum loss. The regression models instead divide expected edge by the floored forecast liability standard deviation. These are different score scales. The IV z-score diagnostic was removed from primary eligibility because a positive relative-IV score does not establish positive expected economic edge; the correction was frozen before final-period predictions.

The objective is joint spread compensation. A positive net edge does not require both a positive short-put richness residual and a positive hedge-cheapness residual separately. A sufficiently rich short can compensate for an expensive hedge. Likewise, the adaptive winner traded every validation week, so validation does not demonstrate that its positive-edge gate successfully avoided weak weeks.

The robustness script retains the frozen selections. Payoff-bounded marks are a sensitivity check on interim quote noise and leave entry fills and terminal settlement unchanged. Its calendar Series needed conversion to a DatetimeIndex before simulation; the root agent applied that correction. The reviewer verified additional guards that reject adjustments at entry/expiry and reject missing fixed-99/96 IDs rather than silently changing active counterfactual weeks into cash. The same-active-weeks 99/96 comparison attributes the combined strike/hedge-width choice relative to timing; it does not isolate strike choice alone when width changes.

The paired uncertainty calculation resamples the same actual daily observations for both strategies in contiguous circular four-calendar-week blocks. Holidays and partial weeks retain their actual observation counts. Reported intervals concern differences in annualized daily Sharpe and annual arithmetic return, not CAGR, and do not convert exploratory research into proof of a persistent edge.

The final-period files cover January 5, 2024–September 18, 2026: 678 daily sessions and 141 weekly decisions. The reviewer independently recomputed daily returns directly from saved NAV for all six frozen portfolios, then recomputed Sharpe, exact-calendar CAGR, drawdown including initial capital, and compounded weekly terminal equity. All values matched the saved summaries. All 2024–2026 fit-audit rows satisfy the strict annual maturity cutoff.

| Final-period portfolio | CAGR | Daily Sharpe | Daily max drawdown | Active weeks |
|---|---:|---:|---:|---:|
| Frozen adaptive dynamic model | 5.09% | 0.896 | -8.72% | 141 |
| Fixed weekly 99/96 | 4.77% | 1.000 | -4.45% | 141 |
| Frozen three-percent-width dynamic rule | 0.014% | 0.019 | -3.35% | 37 |
| Validation-selected fixed 97/94 | 3.51% | 1.520 | -3.17% | 141 |

These results do not establish an improved dynamic strategy on a risk-adjusted basis. The adaptive model earned about 0.31 percentage points more CAGR than 99/96 while producing a lower Sharpe and almost twice its observed maximum drawdown. Its average width was 4.92% of spot versus 3.00% for the benchmark, and its average entry net delta was 0.290 versus 0.227. It traded every final-period week as well as every validation week, so the experiment has not demonstrated an effective weak-week avoidance filter for this winner.

The adaptive-minus-99/96 Sharpe difference was -0.105, with a paired 95% bootstrap interval of [-0.885, +0.796]. The annual arithmetic return difference was +0.35 percentage points, with an interval of [-3.82, +5.02] percentage points. Both intervals include zero. Using natural bid/ask execution preserves the same conclusion: adaptive CAGR 4.71% and Sharpe 0.833 versus fixed-99/96 CAGR 4.64% and Sharpe 0.975.

The fixed-width normal-RV dynamic rule's weak final result follows a positive validation result; selecting a replacement after observing that failure would invalidate this holdout comparison. The fixed 97/94 benchmark was selected in validation, so its higher final-period Sharpe is a legitimate reported comparator, with the associated lower CAGR shown alongside it. The final evaluation calculated performance for the six previously frozen choices only; prediction files for other feature bundles do not constitute an additional final-period model selection.

The robustness outputs were subsequently reviewed. The frozen adaptive model and both fixed benchmarks had no out-of-bound midpoint marks, so bounded marks leave their reported results exactly unchanged. The frozen three-percent-width rule had two flagged marks, with a maximum correction of 0.5 SPX points. Bounding them changes its Sharpe from 0.018856912 to 0.018856581, with identical CAGR and maximum drawdown. Interim quote-bound noise does not explain the findings.

For the three-percent-width rule, selling fixed 99/96 on exactly its 37 active weeks produces 0.805% CAGR and 0.330 Sharpe, compared with 0.014% CAGR and 0.019 Sharpe for its selected spreads. Fixed 99/96 in every week produced 4.771% CAGR and 1.000 Sharpe. On this realized test period, neither the restricted trading calendar nor the selected replacement spreads improved the benchmark's Sharpe. The adaptive winner traded all 141 weeks, so its same-trading-weeks counterfactual exactly reproduces the full fixed-99/96 baseline; its result difference comes from the spread choices, including width and direction, rather than from a no-trade filter.

The audit is complete. No model was reselected based on these final-period diagnostics.
