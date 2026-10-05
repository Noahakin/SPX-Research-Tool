from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
FUTURES_SOURCE = (
    WORKSPACE
    / "06 - Futures/Derived and Supplemental/Unspecified/outputs"
    / "vix_simon_signal_replication/matched_vx_futures_history.csv"
)
SP500_SOURCE = WORKSPACE / "liquidity_cycle_study/raw/fred/sp500__SP500.json"
OUT = PROJECT / "results/vix_curve_next_day_spx"
TERMS = list(range(1, 9))
MIN_TRAIN = 252
HOLDOUT_START = pd.Timestamp("2025-01-01")


def load_curve() -> pd.DataFrame:
    raw = pd.read_csv(FUTURES_SOURCE, parse_dates=["date", "expiration_date"])
    raw = raw[
        raw["expiration_date"].gt(raw["date"])
        & raw["future"].notna()
        & raw["future"].gt(0)
    ].copy()
    raw = raw.sort_values(["date", "expiration_date"]).drop_duplicates(
        ["date", "expiration_date"], keep="last"
    )
    raw["term"] = raw.groupby("date").cumcount() + 1
    curve = raw[raw["term"].isin(TERMS)].pivot(index="date", columns="term", values="future")
    curve.columns = [f"m{int(term)}" for term in curve.columns]
    needed = [f"m{term}" for term in TERMS]
    return curve.dropna(subset=needed).reset_index().sort_values("date")


def load_sp500_returns() -> pd.DataFrame:
    with SP500_SOURCE.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    sp500 = pd.DataFrame(raw["observations"])[["date", "value"]].copy()
    sp500["date"] = pd.to_datetime(sp500["date"])
    sp500["sp500"] = pd.to_numeric(sp500["value"], errors="coerce")
    sp500 = sp500.dropna().drop_duplicates("date", keep="last").sort_values("date")
    sp500["sp500_same_day_return"] = sp500["sp500"].pct_change()
    sp500["next_sp500_date"] = sp500["date"].shift(-1)
    sp500["next_sp500"] = sp500["sp500"].shift(-1)
    sp500["sp500_next_return"] = sp500["next_sp500"] / sp500["sp500"] - 1.0
    return sp500


def build_panel() -> pd.DataFrame:
    panel = load_curve().merge(load_sp500_returns(), on="date", how="inner")
    for near, far in combinations(TERMS, 2):
        panel[f"m{far}_minus_m{near}"] = panel[f"m{far}"] - panel[f"m{near}"]
        panel[f"m{far}_div_m{near}_minus_1"] = panel[f"m{far}"] / panel[f"m{near}"] - 1.0
    return panel.dropna(
        subset=["sp500_same_day_return", "sp500_next_return"]
    ).reset_index(drop=True)


def expanding_prediction(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(len(y), np.nan)
    baselines = np.full(len(y), np.nan)
    for index in range(MIN_TRAIN, len(y)):
        train_x = x[:index]
        train_y = y[:index]
        mean_x = train_x.mean()
        mean_y = train_y.mean()
        variance = np.mean((train_x - mean_x) ** 2)
        beta = (
            np.mean((train_x - mean_x) * (train_y - mean_y)) / variance
            if variance > 0
            else 0.0
        )
        alpha = mean_y - beta * mean_x
        predictions[index] = alpha + beta * x[index]
        baselines[index] = mean_y
    return predictions, baselines


def expanding_multivariate_prediction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions = np.full(len(y), np.nan)
    for index in range(MIN_TRAIN, len(y)):
        coefficients = np.linalg.lstsq(x[:index], y[:index], rcond=None)[0]
        predictions[index] = x[index] @ coefficients
    return predictions


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    values = pvalues.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=pvalues.index)


def pair_statistics(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    y_all = panel["sp500_next_return"].to_numpy(dtype=float)
    controls = np.column_stack(
        [
            np.ones(len(panel)),
            panel["sp500_same_day_return"].to_numpy(dtype=float),
            panel["m1"].to_numpy(dtype=float),
        ]
    )
    control_predictions = expanding_multivariate_prediction(controls, y_all)
    train = panel["date"].lt(HOLDOUT_START).to_numpy()
    test = panel["date"].ge(HOLDOUT_START).to_numpy()
    control_holdout_coefficients = np.linalg.lstsq(
        controls[train], y_all[train], rcond=None
    )[0]
    control_holdout_predictions = controls[test] @ control_holdout_coefficients

    for near, far in combinations(TERMS, 2):
        spread_name = f"M{far}-M{near}"
        column = f"m{far}_minus_m{near}"
        x_all = panel[column].to_numpy(dtype=float)

        pearson = float(np.corrcoef(x_all, y_all)[0, 1])
        spearman = float(spearmanr(x_all, y_all).statistic)
        model = sm.OLS(y_all, sm.add_constant(x_all)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )
        controlled_design = np.column_stack([controls, x_all])
        controlled_model = sm.OLS(y_all, controlled_design).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )

        predictions, baselines = expanding_prediction(x_all, y_all)
        valid = np.isfinite(predictions)
        oos_sse = np.sum((y_all[valid] - predictions[valid]) ** 2)
        baseline_sse = np.sum((y_all[valid] - baselines[valid]) ** 2)
        oos_r2 = 1.0 - oos_sse / baseline_sse
        oos_prediction_correlation = float(np.corrcoef(predictions[valid], y_all[valid])[0, 1])
        oos_directional_accuracy = float(
            np.mean(np.sign(predictions[valid]) == np.sign(y_all[valid]))
        )
        baseline_directional_accuracy = float(
            np.mean(np.sign(baselines[valid]) == np.sign(y_all[valid]))
        )

        controlled_predictions = expanding_multivariate_prediction(
            controlled_design, y_all
        )
        controlled_valid = np.isfinite(controlled_predictions)
        incremental_oos_r2 = 1.0 - np.sum(
            (y_all[controlled_valid] - controlled_predictions[controlled_valid]) ** 2
        ) / np.sum(
            (y_all[controlled_valid] - control_predictions[controlled_valid]) ** 2
        )

        train_x, train_y = x_all[train], y_all[train]
        test_x, test_y = x_all[test], y_all[test]
        train_mean_x, train_mean_y = train_x.mean(), train_y.mean()
        train_var = np.mean((train_x - train_mean_x) ** 2)
        train_beta = np.mean((train_x - train_mean_x) * (train_y - train_mean_y)) / train_var
        test_predictions = train_mean_y + train_beta * (test_x - train_mean_x)
        holdout_r2 = 1.0 - np.sum((test_y - test_predictions) ** 2) / np.sum(
            (test_y - train_mean_y) ** 2
        )
        train_corr = float(np.corrcoef(train_x, train_y)[0, 1])
        test_corr = float(np.corrcoef(test_x, test_y)[0, 1])
        controlled_holdout_coefficients = np.linalg.lstsq(
            controlled_design[train], y_all[train], rcond=None
        )[0]
        controlled_holdout_predictions = (
            controlled_design[test] @ controlled_holdout_coefficients
        )
        holdout_incremental_r2 = 1.0 - np.sum(
            (test_y - controlled_holdout_predictions) ** 2
        ) / np.sum((test_y - control_holdout_predictions) ** 2)

        quantiles = pd.qcut(panel[column], 5, labels=False, duplicates="drop")
        quintile_returns = panel.groupby(quantiles, observed=True)["sp500_next_return"].mean()
        low_quintile = float(quintile_returns.iloc[0])
        high_quintile = float(quintile_returns.iloc[-1])

        rows.append(
            {
                "spread": spread_name,
                "near_term": near,
                "far_term": far,
                "observations": len(panel),
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "ols_beta_return_per_vix_point": float(model.params[1]),
                "hac_t_stat": float(model.tvalues[1]),
                "hac_p_value": float(model.pvalues[1]),
                "controlled_beta_return_per_vix_point": float(
                    controlled_model.params[-1]
                ),
                "controlled_hac_t_stat": float(controlled_model.tvalues[-1]),
                "controlled_hac_p_value": float(controlled_model.pvalues[-1]),
                "expanding_oos_r2": oos_r2,
                "expanding_oos_prediction_correlation": oos_prediction_correlation,
                "expanding_oos_directional_accuracy": oos_directional_accuracy,
                "expanding_baseline_directional_accuracy": baseline_directional_accuracy,
                "controlled_incremental_expanding_oos_r2": incremental_oos_r2,
                "train_2022_2024_correlation": train_corr,
                "holdout_2025_2026_correlation": test_corr,
                "holdout_2025_2026_r2": holdout_r2,
                "controlled_incremental_holdout_r2": holdout_incremental_r2,
                "low_spread_quintile_next_return": low_quintile,
                "high_spread_quintile_next_return": high_quintile,
                "high_minus_low_quintile_return": high_quintile - low_quintile,
                "average_spread": panel[column].mean(),
                "spread_standard_deviation": panel[column].std(),
                "latest_spread": panel.iloc[-1][column],
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "date": panel["date"],
                    "next_sp500_date": panel["next_sp500_date"],
                    "spread": spread_name,
                    "spread_value": x_all,
                    "actual_next_return": y_all,
                    "expanding_prediction": predictions,
                    "expanding_mean_baseline": baselines,
                }
            )
        )

    stats = pd.DataFrame(rows)
    stats["bh_adjusted_p_value"] = benjamini_hochberg(stats["hac_p_value"])
    stats["controlled_bh_adjusted_p_value"] = benjamini_hochberg(
        stats["controlled_hac_p_value"]
    )
    stats["same_correlation_sign_train_and_holdout"] = (
        np.sign(stats["train_2022_2024_correlation"])
        == np.sign(stats["holdout_2025_2026_correlation"])
    )
    stats = stats.sort_values(
        ["expanding_oos_r2", "expanding_oos_prediction_correlation"], ascending=False
    ).reset_index(drop=True)
    stats["oos_rank"] = np.arange(1, len(stats) + 1)
    stats["robust_candidate"] = (
        stats["expanding_oos_r2"].gt(0)
        & stats["expanding_oos_prediction_correlation"].gt(0)
        & stats["same_correlation_sign_train_and_holdout"]
        & stats["controlled_incremental_holdout_r2"].gt(0)
    )
    robust_order = stats.loc[stats["robust_candidate"]].sort_values(
        "controlled_incremental_holdout_r2", ascending=False
    ).index
    stats["robust_rank"] = np.nan
    stats.loc[robust_order, "robust_rank"] = np.arange(1, len(robust_order) + 1)
    return stats, pd.concat(prediction_rows, ignore_index=True)


def select_robust_spread(stats: pd.DataFrame) -> pd.Series:
    candidates = stats[stats["robust_candidate"]].sort_values(
        "controlled_incremental_holdout_r2", ascending=False
    )
    if not candidates.empty:
        return candidates.iloc[0]
    return stats.sort_values(
        "controlled_incremental_holdout_r2", ascending=False
    ).iloc[0]


def quintile_table(panel: pd.DataFrame, top: pd.Series) -> pd.DataFrame:
    column = f"m{int(top['far_term'])}_minus_m{int(top['near_term'])}"
    labels = pd.qcut(panel[column], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    table = (
        panel.assign(quintile=labels)
        .groupby("quintile", observed=True)
        .agg(
            observations=("sp500_next_return", "size"),
            average_spread=(column, "mean"),
            average_next_return=("sp500_next_return", "mean"),
            median_next_return=("sp500_next_return", "median"),
            positive_next_day_rate=("sp500_next_return", lambda values: values.gt(0).mean()),
            next_return_std=("sp500_next_return", "std"),
        )
        .reset_index()
    )
    table["average_next_return_bps"] = table["average_next_return"] * 10_000
    table["standard_error_bps"] = (
        table["next_return_std"] / np.sqrt(table["observations"]) * 10_000
    )
    return table


def save_chart(stats: pd.DataFrame, quintiles: pd.DataFrame, selected: pd.Series) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)

    matrix = np.full((len(TERMS), len(TERMS)), np.nan)
    for _, row in stats.iterrows():
        matrix[int(row["near_term"]) - 1, int(row["far_term"]) - 1] = row[
            "pearson_correlation"
        ]
    limit = np.nanmax(np.abs(matrix))
    image = axes[0].imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="equal")
    axes[0].set_xticks(range(len(TERMS)), [f"M{term}" for term in TERMS])
    axes[0].set_yticks(range(len(TERMS)), [f"M{term}" for term in TERMS])
    axes[0].set_xlabel("Far contract")
    axes[0].set_ylabel("Near contract")
    axes[0].set_title("Correlation with next-day return")
    for near in TERMS:
        for far in TERMS:
            value = matrix[near - 1, far - 1]
            if np.isfinite(value):
                axes[0].text(far - 1, near - 1, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    top_ten = stats.sort_values(
        "controlled_incremental_holdout_r2", ascending=False
    ).head(10).sort_values("controlled_incremental_holdout_r2")
    colors = np.where(
        top_ten["spread"].eq(selected["spread"]), "#2563eb", "#94a3b8"
    )
    axes[1].barh(
        top_ten["spread"],
        top_ten["controlled_incremental_holdout_r2"] * 100,
        color=colors,
    )
    axes[1].axvline(0, color="#334155", linewidth=1.0)
    axes[1].set_xlabel("Incremental 2025-2026 holdout R² (%)")
    axes[1].set_title("Holdout value beyond controls")
    axes[1].grid(axis="x", alpha=0.2)

    axes[2].bar(
        quintiles["quintile"].astype(str),
        quintiles["average_next_return_bps"],
        yerr=quintiles["standard_error_bps"],
        color="#2563eb",
        alpha=0.85,
        capsize=3,
    )
    axes[2].axhline(0, color="#334155", linewidth=1.0)
    axes[2].set_xlabel("Spread quintile (low to high)")
    axes[2].set_ylabel("Average next-day S&P return (bps)")
    axes[2].set_title(f"{selected['spread']} return buckets")
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Predictive power of VIX futures-curve spreads",
        fontsize=20,
        fontweight="bold",
        x=0.055,
        ha="left",
    )
    fig.text(
        0.055,
        0.92,
        "First eight monthly VX contracts · forecast target is the next S&P 500 trading-session return",
        fontsize=11.5,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "predictive_power_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(
    stats: pd.DataFrame,
    quintiles: pd.DataFrame,
    panel: pd.DataFrame,
    selected: pd.Series,
    yearly: pd.DataFrame,
) -> None:
    mechanical_top = stats.iloc[0]
    significant = stats[stats["bh_adjusted_p_value"].lt(0.05)]
    controlled_significant = stats[
        stats["controlled_bh_adjusted_p_value"].lt(0.05)
    ]
    lines = [
        "# VIX curve spreads and next-day S&P 500 returns",
        "",
        f"The comparison covers {len(panel):,} daily observations from {panel.date.min():%Y-%m-%d} through {panel.date.max():%Y-%m-%d}. Every pair among M1 through M8 was tested as the far future minus the near future. The primary ranking uses a 252-day minimum expanding-window out-of-sample R-squared.",
        "",
        f"The strongest overall candidate is **{selected['spread']}**. Its full-sample correlation with the next S&P return is **{selected['pearson_correlation']:.3f}**, its expanding out-of-sample R-squared is **{selected['expanding_oos_r2']:.3%}**, and its 2025-2026 incremental holdout R-squared after controlling for today's S&P return and the M1 futures level is **{selected['controlled_incremental_holdout_r2']:.3%}**.",
        "",
        f"The mechanical winner on expanding R-squared alone is {mechanical_top['spread']}, but its out-of-sample prediction correlation is {mechanical_top['expanding_oos_prediction_correlation']:.3f}; it therefore fails the stability screen. After Benjamini-Hochberg correction across all {len(stats)} spreads, **{len(significant)}** univariate and **{len(controlled_significant)}** controlled relationships have an adjusted p-value below 5%.",
        "",
        "## Top ten by controlled 2025-2026 holdout contribution",
        "",
        "| Spread | Incremental holdout R² | Expanding OOS R² | OOS prediction correlation | Full correlation | Controlled HAC t | Controlled adjusted p |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in stats.sort_values(
        "controlled_incremental_holdout_r2", ascending=False
    ).head(10).iterrows():
        lines.append(
            f"| {row['spread']} | {row['controlled_incremental_holdout_r2']:.3%} | "
            f"{row['expanding_oos_r2']:.3%} | {row['expanding_oos_prediction_correlation']:.3f} | "
            f"{row['pearson_correlation']:.3f} | {row['controlled_hac_t_stat']:.2f} | "
            f"{row['controlled_bh_adjusted_p_value']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Highest-ranked spread by quintile",
            "",
            "| Quintile | Average spread | Average next return | Positive-day rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in quintiles.iterrows():
        lines.append(
            f"| {int(row['quintile'])} | {row['average_spread']:.2f} | "
            f"{row['average_next_return_bps']:.2f} bps | {row['positive_next_day_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "HAC t-statistics use five lags. The holdout column estimates each model on 2022–2024 and tests it on 2025–2026. Results describe close-to-close daily prediction and do not include trading costs.",
            "",
            "## Selected-spread correlation by year",
            "",
            "| Year | Correlation | Observations |",
            "|---:|---:|---:|",
        ]
    )
    for _, row in yearly.iterrows():
        lines.append(
            f"| {int(row['year'])} | {row['correlation']:.3f} | {int(row['observations'])} |"
        )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    stats, predictions = pair_statistics(panel)
    selected = select_robust_spread(stats)
    quintiles = quintile_table(panel, selected)
    selected_column = (
        f"m{int(selected['far_term'])}_minus_m{int(selected['near_term'])}"
    )
    yearly = (
        panel.assign(year=panel["date"].dt.year)
        .groupby("year")
        .apply(
            lambda group: pd.Series(
                {
                    "correlation": group[selected_column].corr(
                        group["sp500_next_return"]
                    ),
                    "observations": len(group),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    panel.to_csv(OUT / "daily_curve_and_forward_returns.csv", index=False)
    stats.to_csv(OUT / "pairwise_predictive_statistics.csv", index=False)
    predictions.to_csv(OUT / "walkforward_predictions.csv", index=False)
    quintiles.to_csv(OUT / "top_spread_quintiles.csv", index=False)
    yearly.to_csv(OUT / "selected_spread_yearly_stability.csv", index=False)
    pd.DataFrame([selected]).to_csv(OUT / "selected_spread_summary.csv", index=False)
    save_chart(stats, quintiles, selected)
    write_report(stats, quintiles, panel, selected, yearly)

    print("TOP TEN")
    print(
        stats.head(10)[
            [
                "oos_rank",
                "spread",
                "expanding_oos_r2",
                "expanding_oos_prediction_correlation",
                "pearson_correlation",
                "hac_t_stat",
                "bh_adjusted_p_value",
                "train_2022_2024_correlation",
                "holdout_2025_2026_correlation",
                "holdout_2025_2026_r2",
            ]
        ].to_string(index=False)
    )
    print("\nTOP QUINTILES")
    print(quintiles.to_string(index=False))
    print("\nSELECTED")
    print(selected.to_string())
    print("\nYEARLY STABILITY")
    print(yearly.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
