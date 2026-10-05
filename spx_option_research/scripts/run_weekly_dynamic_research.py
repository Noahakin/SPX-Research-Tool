"""Chronological weekly spread research; validation choices precede final evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/weekly_dynamic_research"
sys.path.insert(0, str(PROJECT / ".research_dependencies"))
sys.path.insert(0, str(PROJECT / "scripts"))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from weekly_economic_features import economic_features

FILLS = ("realistic", "natural")
VALIDATION_START = pd.Timestamp("2021-01-01")
HOLDOUT_START = pd.Timestamp("2024-01-01")
GATES = {"always": -np.inf, "positive_edge": 0.0, "edge_risk_0.10": 0.10}


def feature_table(candidates: pd.DataFrame, market: pd.DataFrame, closes: pd.Series):
    frame = candidates.copy().sort_values(["entry_date", "candidate_id"]).reset_index(drop=True)
    frame = frame.merge(market, left_on="entry_date", right_index=True, how="left", validate="many_to_one")
    if frame.market_source_date.isna().any() or frame.market_source_date.ge(frame.entry_date).any():
        raise ValueError("market features must precede the option entry snapshot")
    frame = economic_features(frame, closes)
    s = frame.spot_entry
    credit = frame.premium_realistic_pct_spot_notional
    width = frame.width_pct
    rv = frame.spx_rv_21d.clip(lower=0.01)
    frame["credit_over_width"] = credit / width
    frame["credit_over_max_loss"] = credit / frame.max_loss_realistic_pct.clip(lower=0.0001)
    frame["short_price_pct"] = frame.short_mid / s
    frame["hedge_price_pct"] = frame.long_mid / s
    frame["hedge_over_short_price"] = frame.long_mid / frame.short_mid.clip(lower=0.1)
    short_intrinsic = (frame.upper_strike - s).clip(lower=0.0)
    long_intrinsic = (frame.lower_strike - s).clip(lower=0.0)
    frame["extrinsic_credit_pct"] = credit - (short_intrinsic - long_intrinsic) / s
    frame["short_extrinsic_pct"] = (frame.short_mid - short_intrinsic) / s
    frame["long_extrinsic_pct"] = (frame.long_mid - long_intrinsic) / s
    frame["short_rv_moneyness"] = (frame.actual_short_ratio - 1) / (rv * np.sqrt(frame.dte / 365.2425))
    frame["long_rv_moneyness"] = (frame.actual_long_ratio - 1) / (rv * np.sqrt(frame.dte / 365.2425))
    for greek in ("delta", "gamma", "theta", "vega"):
        frame[f"net_{greek}"] = frame[f"long_{greek}"] - frame[f"short_{greek}"]
    frame["gamma_one_percent_move"] = 0.5 * frame.net_gamma * s * 0.01**2
    frame["theta_tenor_pct"] = frame.net_theta * frame.dte / s
    frame["credit_per_delta_rv_risk"] = credit / (frame.net_delta.abs() * rv * np.sqrt(frame.dte / 365.2425)).clip(lower=0.0005)
    frame["short_iv_rv21"] = frame.short_implied_volatility / rv
    frame["long_iv_rv21"] = frame.long_implied_volatility / rv
    frame["short_iv_rv5"] = frame.short_implied_volatility / frame.spx_rv_5d.clip(lower=0.01)
    frame["long_minus_short_iv"] = frame.long_implied_volatility - frame.short_implied_volatility
    frame["short_iv_minus_atm"] = frame.short_implied_volatility - frame.same_expiry_atm_put_iv
    frame["long_iv_minus_atm"] = frame.long_implied_volatility - frame.same_expiry_atm_put_iv
    frame["weekly_monthly_iv_ratio"] = frame.same_expiry_atm_put_iv / frame.one_month_atm_put_iv
    keys = ["target_short_ratio", "target_width_pct"]
    for leg in ("short", "long"):
        values = frame.groupby(keys, sort=False)[f"{leg}_iv_rv21"]
        mean = values.transform(lambda x: x.shift(1).rolling(52, min_periods=26).mean())
        sd = values.transform(lambda x: x.shift(1).rolling(52, min_periods=26).std(ddof=1))
        frame[f"{leg}_iv_rv_z52"] = (frame[f"{leg}_iv_rv21"] - mean) / sd.where(sd.gt(1e-8))
        frame[f"log_{leg}_volume"] = np.log1p(frame[f"{leg}_volume"].clip(lower=0))
        frame[f"log_{leg}_oi"] = np.log1p(frame[f"{leg}_open_interest"].clip(lower=0))
        frame[f"{leg}_volume_oi"] = frame[f"{leg}_volume"] / frame[f"{leg}_open_interest"].clip(lower=1)
        frame[f"{leg}_ba_pct_mid"] = (frame[f"{leg}_ask"] - frame[f"{leg}_bid"]) / frame[f"{leg}_mid"].clip(lower=0.1)
        frame[f"lagged_log_{leg}_volume"] = frame.groupby(keys, sort=False)[f"log_{leg}_volume"].shift(1)
    frame["two_leg_iv_richness"] = frame.short_iv_rv_z52 - frame.long_iv_rv_z52
    frame["execution_cost_pct"] = frame.premium_mid_pct_spot_notional - credit
    frame["friction_over_credit"] = frame.execution_cost_pct / credit.clip(lower=0.00005)
    groups = {
        "core": ["dte", "actual_short_ratio", "actual_long_ratio", "width_pct", "premium_realistic_pct_spot_notional",
                 "max_loss_realistic_pct", "credit_over_width", "credit_over_max_loss", "short_price_pct",
                 "hedge_price_pct", "hedge_over_short_price", "extrinsic_credit_pct", "short_extrinsic_pct", "long_extrinsic_pct"],
        "economic": ["econ_expected_edge", "econ_short_fair_edge", "econ_long_fair_edge", "econ_std_liability",
                     "econ_tail_loss_95", "econ_probability_loss", "econ_probability_max_loss", "econ_normal_expected_edge"],
        "greeks": ["short_delta", "long_delta", "net_delta", "net_gamma", "net_theta", "net_vega",
                   "short_vega", "long_vega", "gamma_one_percent_move", "theta_tenor_pct", "credit_per_delta_rv_risk"],
        "iv_skew": ["short_implied_volatility", "long_implied_volatility", "short_iv_rv21", "long_iv_rv21", "short_iv_rv5",
                    "short_iv_rv_z52", "long_iv_rv_z52", "long_minus_short_iv", "short_iv_minus_atm", "long_iv_minus_atm",
                    "same_expiry_put_skew_95_minus_atm", "same_expiry_put_skew_25d_minus_atm", "weekly_monthly_iv_ratio",
                    "short_rv_moneyness", "long_rv_moneyness", "spx_rv_5d", "spx_rv_21d", "spx_rv_63d"],
        "vol_regime": ["vix", "vvix", "vix3m", "vix_to_vix3m", "vvix_to_vix", "vix_to_spx_rv_21d",
                       "vix_zscore_252d", "vvix_zscore_252d", "vix3m_zscore_252d"],
        "momentum": ["spx_momentum_1d", "spx_momentum_5d", "spx_momentum_21d", "spx_momentum_63d", "spx_momentum_252d",
                     "spx_drawdown_252d", "vix_change_5d", "vix_change_21d", "vvix_change_5d", "vvix_change_21d"],
        "liquidity": ["log_short_volume", "log_long_volume", "log_short_oi", "log_long_oi", "short_volume_oi", "long_volume_oi",
                      "lagged_log_short_volume", "lagged_log_long_volume", "short_ba_pct_mid", "long_ba_pct_mid",
                      "execution_cost_pct", "friction_over_credit", "chain_put_call_volume_ratio", "chain_put_call_open_interest_ratio",
                      "same_expiry_put_call_volume_ratio", "same_expiry_put_call_open_interest_ratio"],
    }
    all_columns = list(dict.fromkeys(column for columns in groups.values() for column in columns))
    missing = set(all_columns) - set(frame)
    if missing:
        raise ValueError(f"missing input features: {sorted(missing)}")
    frame[all_columns] = frame[all_columns].replace([np.inf, -np.inf], np.nan)
    forbidden = ("pnl", "terminal", "spot_expiration", "forward_return")
    if any(any(token in column for token in forbidden) for column in all_columns):
        raise ValueError("outcome entered feature list")
    frame["target_liability_fraction"] = -frame.expiration_value_cash / (frame.width_points * 100.0)
    if not frame.target_liability_fraction.between(-1e-9, 1 + 1e-9).all():
        raise ValueError("spread liability is outside its bounded range")
    return frame, groups


def feature_sets(groups):
    bundles = {"core": groups["core"]}
    for name in groups:
        if name != "core":
            bundles[f"core_plus_{name}"] = groups["core"] + groups[name]
    bundles["full"] = sum(groups.values(), [])
    for name in groups:
        if name != "core":
            bundles[f"full_without_{name}"] = sum((v for k, v in groups.items() if k != name), [])
    return {name: list(dict.fromkeys(columns)) for name, columns in bundles.items()}


def fit_model(family, x, y, weights):
    if family == "ridge":
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0))
        model.fit(x, y, ridge__sample_weight=weights)
    elif family == "boosting":
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7, max_depth=3,
            min_samples_leaf=100, learning_rate=0.05, l2_regularization=10.0, random_state=1729, early_stopping=False)
        model.fit(x, y, sample_weight=weights)
    else:
        raise ValueError(family)
    return model


def predict_years(frame, groups, years, *, save_prefix):
    bundles = feature_sets(groups)
    predictions = frame[["candidate_id", "entry_date", "expiration_date"]].copy().set_index("candidate_id")
    audits = []
    with threadpool_limits(limits=4):
        for year in years:
            cutoff = pd.Timestamp(year=year, month=1, day=1)
            train = frame[frame.expiration_date.lt(cutoff)]
            test = frame[frame.entry_date.ge(cutoff) & frame.entry_date.lt(cutoff + pd.DateOffset(years=1))]
            if test.empty:
                continue
            if train.entry_date.nunique() < 156:
                raise ValueError("fewer than 156 independent training weeks")
            weights = 1.0 / train.groupby("entry_date").candidate_id.transform("size")
            for family in ("ridge", "boosting"):
                for bundle, columns in bundles.items():
                    name = f"{family}__{bundle}"
                    model = fit_model(family, train[columns], train.target_liability_fraction, weights)
                    predicted_liability = np.clip(model.predict(test[columns]), 0.0, 1.0) * test.width_pct.to_numpy()
                    edge = test.premium_realistic_pct_spot_notional.to_numpy() - predicted_liability
                    denominator = test.econ_risk_denominator.fillna(np.maximum(0.1 * test.width_pct, 0.0005)).to_numpy()
                    predictions.loc[test.candidate_id, name] = edge / denominator
                    audits.append({"year": year, "model": name, "features": len(columns), "training_weeks": train.entry_date.nunique(),
                                   "training_rows": len(train), "latest_training_expiration": train.expiration_date.max(),
                                   "prediction_first_entry": test.entry_date.min(), "prediction_last_entry": test.entry_date.max(),
                                   "refit_cutoff": cutoff})
            print(f"Predicted {year}: {len(test):,} candidates using {train.entry_date.nunique()} prior weeks; {len(bundles)*2} model/feature combinations.", flush=True)
    predicted = predictions.reset_index()
    predicted.to_parquet(OUT / f"{save_prefix}_predictions.parquet", index=False)
    pd.DataFrame(audits).to_csv(OUT / f"{save_prefix}_fit_audit.csv", index=False)
    return predicted


def add_rule_scores(frame, predictions):
    predictions = predictions.set_index("candidate_id").copy()
    source = frame.set_index("candidate_id")
    predictions["rule__economic_risk"] = source.econ_edge_per_risk
    predictions["rule__economic_max_loss"] = source.econ_edge_per_max_loss
    predictions["rule__normal_rv"] = source.econ_normal_edge_per_max_loss
    predictions["rule__two_leg_iv_z"] = source.two_leg_iv_richness
    return predictions.reset_index()


def choose_policy(frame, predictions, model, universe, gate, *, start, end=None):
    eligible = frame[frame.entry_date.ge(start)].copy()
    if end is not None:
        eligible = eligible[eligible.expiration_date.lt(end)]
    schedule = eligible[["entry_date", "expiration_date"]].drop_duplicates().sort_values("entry_date")
    if schedule.entry_date.duplicated().any():
        raise ValueError("multiple maturities per weekly decision")
    if universe == "width3":
        eligible = eligible[np.isclose(eligible.target_width_pct, 0.03)]
    elif universe != "adaptive_width":
        raise ValueError(universe)
    if model.startswith("fixed__"):
        short = float(model.split("__")[1])
        choices = eligible[np.isclose(eligible.target_short_ratio, short) & np.isclose(eligible.target_width_pct, 0.03)].copy()
        choices["score"] = 0.0
    else:
        score = predictions.set_index("candidate_id")[model]
        eligible["score"] = eligible.candidate_id.map(score)
        eligible = eligible[np.isfinite(eligible.score)]
        choices = eligible.sort_values(["entry_date", "score", "target_width_pct", "target_short_ratio"],
                                       ascending=[True, False, True, True], kind="stable").groupby("entry_date").head(1)
        threshold = GATES[gate]
        if gate == "positive_edge":
            choices = choices[choices.score.gt(threshold)]
        else:
            choices = choices[choices.score.ge(threshold)]
    result = schedule.merge(choices[["entry_date", "candidate_id", "score"]], on="entry_date", how="left", validate="one_to_one")
    result["model"] = model
    result["universe"] = universe
    result["gate"] = gate
    return result


def policy_grid(predictions):
    columns = [c for c in predictions if c not in ("candidate_id", "entry_date", "expiration_date")]
    for universe in ("width3", "adaptive_width"):
        for model in columns:
            for gate in GATES:
                yield model, universe, gate
    for ratio in (0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03):
        yield f"fixed__{ratio:.2f}", "width3", "always"


def run_validation(frame, groups, unit_marks, cash_dates):
    from weekly_portfolio_evaluation import simulate_policy, summarize
    raw = predict_years(frame, groups, (2021, 2022, 2023), save_prefix="validation")
    predictions = add_rule_scores(frame, raw)
    stats, policy_info = [], {}
    for number, (model, universe, gate) in enumerate(policy_grid(predictions), start=1):
        policy_id = f"{model}__{universe}__{gate}"
        selection = choose_policy(frame, predictions, model, universe, gate, start=VALIDATION_START, end=HOLDOUT_START)
        selected_marks = unit_marks[unit_marks.candidate_id.isin(selection.candidate_id.dropna())]
        daily, weekly = simulate_policy(selection, frame, selected_marks, cash_dates)
        result = summarize(daily, weekly)
        result["sharpe"] = result["daily_sharpe"]
        result.update({"policy_id": policy_id, "model": model, "universe": universe, "gate": gate,
                       "active_weeks": int(selection.candidate_id.notna().sum())})
        stats.append(result)
        policy_info[policy_id] = {"model": model, "universe": universe, "gate": gate}
        if number % 30 == 0:
            print(f"Validated {number} policy specifications on 2021–2023 daily marks.", flush=True)
    table = pd.DataFrame(stats)
    table.to_csv(OUT / "validation_policy_results.csv", index=False)
    return table, policy_info


def freeze_choices(table, policy_info):
    eligible = table[table.active_weeks.ge(30) & np.isfinite(table.sharpe)]
    choices = {}
    for universe in ("width3", "adaptive_width"):
        pool = eligible[eligible.universe.eq(universe) & ~eligible.model.str.startswith("fixed__")
                        & eligible.model.ne("rule__two_leg_iv_z") & eligible.gate.ne("always")]
        winner = pool.sort_values(["sharpe", "cagr", "policy_id"], ascending=[False, False, True]).iloc[0]
        choices[f"Dynamic winner: {universe}"] = {**policy_info[winner.policy_id], "validation_policy_id": winner.policy_id,
                                                 "validation_sharpe": float(winner.sharpe), "validation_cagr": float(winner.cagr)}
    fixed = eligible[eligible.model.str.startswith("fixed__")].sort_values(["sharpe", "cagr"], ascending=False).iloc[0]
    choices["Validation-best fixed spread"] = policy_info[fixed.policy_id]
    choices["Fixed weekly 99/96"] = {"model": "fixed__0.99", "universe": "width3", "gate": "always"}
    choices["Economic model: fixed width"] = {"model": "rule__economic_risk", "universe": "width3", "gate": "positive_edge"}
    choices["Economic model: adaptive width"] = {"model": "rule__economic_risk", "universe": "adaptive_width", "gate": "positive_edge"}
    frozen = {"selection_period": "2021–2023; expiration before 2024-01-01", "holdout_start": "2024-01-01",
              "criterion": "highest validation daily Sharpe among gated policies, minimum 30 active weeks", "choices": choices,
              "validation_results_sha256": hashlib.sha256((OUT / "validation_policy_results.csv").read_bytes()).hexdigest(),
              "research_protocol_sha256": hashlib.sha256((OUT / "research_protocol.json").read_bytes()).hexdigest()}
    (OUT / "frozen_model_choices.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("Model and gate choices frozen using validation only:", json.dumps(choices, indent=2), flush=True)
    return frozen


def final_evaluation(frame, groups, unit_marks, cash_dates, frozen):
    from weekly_portfolio_evaluation import simulate_policy, summarize, paired_block_bootstrap
    raw = predict_years(frame, groups, (2024, 2025, 2026), save_prefix="holdout")
    predictions = add_rule_scores(frame, raw)
    rows, daily_frames, weekly_frames, selections = [], [], [], []
    for name, policy in frozen["choices"].items():
        selected = choose_policy(frame, predictions, policy["model"], policy["universe"], policy["gate"], start=HOLDOUT_START)
        selected["portfolio"] = name
        selections.append(selected)
        selected_marks = unit_marks[unit_marks.candidate_id.isin(selected.candidate_id.dropna())]
        for fill in FILLS:
            daily, weekly = simulate_policy(selected, frame, selected_marks, cash_dates, fill=fill)
            stats = summarize(daily, weekly)
            stats.update({"portfolio": name, "fill": fill, **policy})
            rows.append(stats)
            if fill == "realistic":
                daily["portfolio"] = name
                weekly["portfolio"] = name
                daily_frames.append(daily)
                weekly_frames.append(weekly)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "holdout_summary.csv", index=False)
    daily_all = pd.concat(daily_frames, ignore_index=True)
    weekly_all = pd.concat(weekly_frames, ignore_index=True)
    daily_all.to_parquet(OUT / "holdout_daily_portfolios.parquet", index=False)
    daily_all.to_csv(OUT / "holdout_daily_portfolios.csv", index=False)
    weekly_all.to_csv(OUT / "holdout_weekly_portfolios.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(OUT / "holdout_selections.csv", index=False)
    baseline = daily_all[daily_all.portfolio.eq("Fixed weekly 99/96")]
    intervals = []
    for name in frozen["choices"]:
        if name.startswith("Dynamic winner"):
            interval = paired_block_bootstrap(daily_all[daily_all.portfolio.eq(name)], baseline)
            interval["portfolio"] = name
            intervals.append(interval)
    pd.DataFrame(intervals).to_csv(OUT / "holdout_paired_bootstrap.csv", index=False)
    return result, daily_all, weekly_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("features", "validation", "final", "all"), default="all")
    args = parser.parse_args()
    candidates = pd.read_parquet(OUT / "candidate_trades.parquet")
    market = pd.read_parquet(OUT / "market_data.parquet")
    closes = pd.read_csv(OUT / "market_spx.csv", parse_dates=["date"]).set_index("date").close
    if args.stage in ("features", "all") or not (OUT / "research_features.parquet").exists():
        frame, groups = feature_table(candidates, market, closes)
        frame.to_parquet(OUT / "research_features.parquet", index=False)
        frame.to_csv(OUT / "research_features.csv", index=False)
        (OUT / "feature_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
        frame[list(dict.fromkeys(sum(groups.values(), [])))].isna().mean().rename("missing_fraction").to_csv(OUT / "feature_missingness.csv")
        print(f"Prepared {len(frame):,} candidates, {frame.entry_date.nunique()} weeks, {len(set(sum(groups.values(), [])))} explicit entry-known features.", flush=True)
    else:
        frame = pd.read_parquet(OUT / "research_features.parquet")
        groups = json.loads((OUT / "feature_groups.json").read_text(encoding="utf-8"))
    if args.stage == "features":
        return
    unit_marks = pd.read_parquet(OUT / "unit_marks.parquet")
    if args.stage in ("validation", "all"):
        table, info = run_validation(frame, groups, unit_marks, closes.index)
        frozen = freeze_choices(table, info)
    else:
        frozen = json.loads((OUT / "frozen_model_choices.json").read_text(encoding="utf-8"))
    if args.stage in ("final", "all"):
        results, _, _ = final_evaluation(frame, groups, unit_marks, closes.index, frozen)
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
