"""Predefined mark-quality and timing/strike attribution checks for frozen policies."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weekly_portfolio_evaluation import simulate_policy, summarize

OUT = Path(__file__).resolve().parents[1] / "results/weekly_dynamic_research"


def bounded_marks(marks, candidates):
    """Change interim liability marks only; entry fills and settlement stay fixed."""
    result = marks.copy()
    spot = result.candidate_id.map(candidates.set_index("candidate_id").spot_entry)
    adjustment = (result.spread_mid_points - result.bounded_spread_mid_points) / spot
    if (adjustment[result.is_expiration].abs() > 1e-12).any():
        raise ValueError("settlement cannot require clipping")
    entry_dates = result.candidate_id.map(candidates.set_index("candidate_id").entry_date)
    if (adjustment[result.date.eq(entry_dates)].abs() > 1e-12).any():
        raise ValueError("entry cannot require clipping")
    for fill in ("mid", "realistic", "natural"):
        result[f"cum_return_{fill}"] += adjustment
    return result


def main():
    candidates = pd.read_parquet(OUT / "research_features.parquet")
    marks = pd.read_parquet(OUT / "unit_marks.parquet")
    selections = pd.read_csv(OUT / "holdout_selections.csv", parse_dates=["entry_date", "expiration_date"])
    calendar = pd.DatetimeIndex(pd.read_csv(OUT / "market_spx.csv", parse_dates=["date"]).date)
    frozen = json.loads((OUT / "frozen_model_choices.json").read_text(encoding="utf-8"))
    bounded = bounded_marks(marks, candidates)
    mark_rows, attribution_rows = [], []
    fixed_ids = candidates[np.isclose(candidates.target_short_ratio, 0.99)
                           & np.isclose(candidates.target_width_pct, 0.03)].set_index("entry_date").candidate_id
    for name in frozen["choices"]:
        selected = selections[selections.portfolio.eq(name)].copy()
        ids = set(selected.candidate_id.dropna())
        selected_marks = marks[marks.candidate_id.isin(ids)]
        for convention, source in (("observed_mid", selected_marks),
                                   ("payoff_bounded_mid", bounded[bounded.candidate_id.isin(ids)])):
            daily, weekly = simulate_policy(selected, candidates, source, calendar)
            row = summarize(daily, weekly)
            row.update(portfolio=name, marking_convention=convention,
                       out_of_bounds_marks=int(selected_marks.mid_outside_payoff_bounds.sum()),
                       max_mark_correction_points=float((selected_marks.spread_mid_points
                            - selected_marks.bounded_spread_mid_points).abs().max()))
            mark_rows.append(row)
        if not name.startswith("Dynamic winner"):
            continue
        for counterfactual in ("chosen_spreads", "99_96_same_trade_weeks"):
            alternative = selected.copy()
            if counterfactual == "99_96_same_trade_weeks":
                active = alternative.candidate_id.notna()
                alternative.loc[active, "candidate_id"] = alternative.loc[active, "entry_date"].map(fixed_ids)
                if alternative.loc[active, "candidate_id"].isna().any():
                    raise ValueError("counterfactual baseline is unavailable for a traded week")
            subset = marks[marks.candidate_id.isin(alternative.candidate_id.dropna())]
            daily, weekly = simulate_policy(alternative, candidates, subset, calendar)
            row = summarize(daily, weekly)
            row.update(portfolio=name, counterfactual=counterfactual)
            attribution_rows.append(row)
    pd.DataFrame(mark_rows).to_csv(OUT / "holdout_mark_sensitivity.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(OUT / "holdout_timing_attribution.csv", index=False)
    print(pd.DataFrame(mark_rows)[["portfolio", "marking_convention", "daily_sharpe", "cagr", "max_drawdown_daily",
                                   "out_of_bounds_marks", "max_mark_correction_points"]].to_string(index=False))
    print(pd.DataFrame(attribution_rows)[["portfolio", "counterfactual", "daily_sharpe", "cagr", "trade_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
