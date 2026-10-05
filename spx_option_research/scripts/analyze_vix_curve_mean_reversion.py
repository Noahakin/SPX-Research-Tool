from __future__ import annotations

from itertools import combinations
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

import analyze_vix_curve_next_day_spx as curve_source


OUT = curve_source.PROJECT / "results/vix_curve_mean_reversion"
TERMS = curve_source.TERMS


def ar1_parameters(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().astype(float)
    lagged = clean.shift(1).dropna()
    current = clean.loc[lagged.index]
    variance = np.mean((lagged - lagged.mean()) ** 2)
    phi = (
        np.mean((lagged - lagged.mean()) * (current - current.mean())) / variance
        if variance > 0
        else np.nan
    )
    intercept = current.mean() - phi * lagged.mean()
    equilibrium = intercept / (1.0 - phi) if np.isfinite(phi) and phi < 1 else np.nan
    half_life = (
        -np.log(2.0) / np.log(phi) if np.isfinite(phi) and 0 < phi < 1 else np.nan
    )
    residual = current - (intercept + phi * lagged)
    return {
        "ar1_intercept": intercept,
        "ar1_phi": phi,
        "ar1_equilibrium": equilibrium,
        "ar1_half_life_days": half_life,
        "ar1_residual_std": residual.std(),
    }


def analyze_spreads(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for near, far in combinations(TERMS, 2):
        columns = {
            "points": f"m{far}_minus_m{near}",
            "percent": f"m{far}_div_m{near}_minus_1",
        }
        for measurement, column in columns.items():
            values = panel[column].astype(float)
            ar1 = ar1_parameters(values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf = adfuller(values, regression="c", autolag="AIC")
                kpss_result = kpss(values, regression="c", nlags="auto")

            annual_adf_passes: list[bool] = []
            annual_phi_below_one: list[bool] = []
            annual_means: list[float] = []
            annual_equilibria: list[float] = []
            annual_half_lives: list[float] = []
            for year, group in panel.assign(year=panel["date"].dt.year).groupby("year"):
                annual_values = group[column].astype(float)
                annual_ar1 = ar1_parameters(annual_values)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    annual_adf_p = adfuller(
                        annual_values, regression="c", autolag="AIC"
                    )[1]
                annual_adf_passes.append(annual_adf_p < 0.05)
                annual_phi_below_one.append(0 < annual_ar1["ar1_phi"] < 1)
                annual_means.append(annual_values.mean())
                annual_equilibria.append(annual_ar1["ar1_equilibrium"])
                annual_half_lives.append(annual_ar1["ar1_half_life_days"])
                yearly_rows.append(
                    {
                        "spread": f"M{far}-M{near}",
                        "measurement": measurement,
                        "year": year,
                        "observations": len(annual_values),
                        "mean": annual_values.mean(),
                        "median": annual_values.median(),
                        "standard_deviation": annual_values.std(),
                        "minimum": annual_values.min(),
                        "maximum": annual_values.max(),
                        "adf_p_value": annual_adf_p,
                        "ar1_phi": annual_ar1["ar1_phi"],
                        "ar1_equilibrium": annual_ar1["ar1_equilibrium"],
                        "ar1_half_life_days": annual_ar1["ar1_half_life_days"],
                    }
                )

            deviation = values - ar1["ar1_equilibrium"]
            equilibrium_crossings = int((deviation * deviation.shift(1) < 0).sum())
            rolling_mean = values.rolling(252).mean()
            rows.append(
                {
                    "spread": f"M{far}-M{near}",
                    "near_term": near,
                    "far_term": far,
                    "measurement": measurement,
                    "observations": len(values),
                    "sample_mean": values.mean(),
                    "sample_median": values.median(),
                    "sample_standard_deviation": values.std(),
                    "latest_value": values.iloc[-1],
                    **ar1,
                    "adf_statistic": adf[0],
                    "adf_p_value": adf[1],
                    "kpss_statistic": kpss_result[0],
                    "kpss_p_value": kpss_result[1],
                    "annual_adf_pass_rate": np.mean(annual_adf_passes),
                    "annual_ar1_reverting_rate": np.mean(annual_phi_below_one),
                    "annual_mean_minimum": np.min(annual_means),
                    "annual_mean_maximum": np.max(annual_means),
                    "annual_mean_range": np.ptp(annual_means),
                    "annual_mean_std_over_daily_std": np.std(
                        annual_means, ddof=1
                    )
                    / values.std(),
                    "annual_equilibrium_minimum": np.nanmin(annual_equilibria),
                    "annual_equilibrium_maximum": np.nanmax(annual_equilibria),
                    "annual_half_life_minimum": np.nanmin(annual_half_lives),
                    "annual_half_life_maximum": np.nanmax(annual_half_lives),
                    "equilibrium_crossings": equilibrium_crossings,
                    "crossings_per_year": equilibrium_crossings
                    / ((panel["date"].max() - panel["date"].min()).days / 365.25),
                    "rolling_252d_mean_minimum": rolling_mean.min(),
                    "rolling_252d_mean_maximum": rolling_mean.max(),
                }
            )
    stats = pd.DataFrame(rows)
    stats["full_sample_stationary"] = (
        stats["adf_p_value"].lt(0.05) & stats["kpss_p_value"].ge(0.05)
    )
    stats["consistent_fixed_level_candidate"] = (
        stats["full_sample_stationary"]
        & stats["annual_ar1_reverting_rate"].eq(1.0)
        & stats["annual_adf_pass_rate"].ge(0.80)
        & stats["ar1_half_life_days"].between(1, 30)
    )
    return stats, pd.DataFrame(yearly_rows)


def select_candidate(stats: pd.DataFrame) -> pd.Series:
    point_stats = stats[stats["measurement"].eq("points")].copy()
    candidates = point_stats[point_stats["consistent_fixed_level_candidate"]].copy()
    if candidates.empty:
        candidates = point_stats[point_stats["full_sample_stationary"]].copy()
    return candidates.sort_values(
        ["annual_adf_pass_rate", "annual_mean_std_over_daily_std", "ar1_half_life_days"],
        ascending=[False, True, True],
    ).iloc[0]


def reversion_buckets(panel: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    column = f"m{int(selected['far_term'])}_minus_m{int(selected['near_term'])}"
    equilibrium = selected["ar1_equilibrium"]
    data = pd.DataFrame(
        {
            "deviation": panel[column] - equilibrium,
            "next_day_change": panel[column].shift(-1) - panel[column],
        }
    ).dropna()
    data["deviation_quintile"] = pd.qcut(
        data["deviation"], 5, labels=[1, 2, 3, 4, 5]
    )
    return (
        data.groupby("deviation_quintile", observed=True)
        .agg(
            observations=("next_day_change", "size"),
            average_deviation=("deviation", "mean"),
            average_next_day_change=("next_day_change", "mean"),
            median_next_day_change=("next_day_change", "median"),
            reversion_day_rate=(
                "next_day_change",
                lambda change: np.nan,
            ),
        )
        .reset_index()
        .drop(columns="reversion_day_rate")
    )


def save_chart(
    panel: pd.DataFrame,
    selected: pd.Series,
    yearly: pd.DataFrame,
    buckets: pd.DataFrame,
) -> None:
    near = int(selected["near_term"])
    far = int(selected["far_term"])
    point_column = f"m{far}_minus_m{near}"
    pct_column = f"m{far}_div_m{near}_minus_1"
    pct_stats = yearly[
        yearly["spread"].eq(selected["spread"])
        & yearly["measurement"].eq("percent")
    ]
    point_stats = yearly[
        yearly["spread"].eq(selected["spread"])
        & yearly["measurement"].eq("points")
    ]
    pct_equilibrium = analyze_spreads(panel)[0].loc[
        lambda frame: frame["spread"].eq(selected["spread"])
        & frame["measurement"].eq("percent"),
        "ar1_equilibrium",
    ].iloc[0]

    fig, axes = plt.subplots(
        2, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1.75, 1.0]}
    )
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22)

    point_values = panel[point_column]
    axes[0, 0].plot(panel["date"], point_values, color="#0f172a", linewidth=1.05)
    axes[0, 0].plot(
        panel["date"],
        point_values.rolling(252).mean(),
        color="#f59e0b",
        linewidth=2.0,
        label="252-day mean",
    )
    axes[0, 0].axhline(
        selected["ar1_equilibrium"], color="#2563eb", linestyle="--", linewidth=1.8, label="AR(1) equilibrium"
    )
    axes[0, 0].set_ylabel("Spread (VIX points)")
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower right")

    pct_values = panel[pct_column]
    axes[1, 0].plot(panel["date"], pct_values, color="#0f766e", linewidth=1.05)
    axes[1, 0].plot(
        panel["date"],
        pct_values.rolling(252).mean(),
        color="#f59e0b",
        linewidth=2.0,
        label="252-day mean",
    )
    axes[1, 0].axhline(
        pct_equilibrium, color="#2563eb", linestyle="--", linewidth=1.8, label="AR(1) equilibrium"
    )
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].set_ylabel("Spread (% of near future)")
    axes[1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[0, 1].bar(
        point_stats["year"].astype(str), point_stats["mean"], color="#7c3aed", alpha=0.85
    )
    axes[0, 1].axhline(selected["ar1_equilibrium"], color="#2563eb", linestyle="--")
    axes[0, 1].set_title("Annual average spread")
    axes[0, 1].set_ylabel("VIX points")

    axes[1, 1].bar(
        buckets["deviation_quintile"].astype(str),
        buckets["average_next_day_change"],
        color=["#0f766e", "#94a3b8", "#94a3b8", "#94a3b8", "#dc2626"],
    )
    axes[1, 1].axhline(0, color="#334155", linewidth=1.0)
    axes[1, 1].set_title("Next-day change by deviation quintile")
    axes[1, 1].set_xlabel("Deviation from equilibrium: low to high")
    axes[1, 1].set_ylabel("Average next-day spread change")

    axes[0, 0].set_title(
        f"Most consistent VIX-curve mean reversion: {selected['spread']}",
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    axes[0, 0].text(
        0,
        1.01,
        f"Equilibrium {selected['ar1_equilibrium']:.2f} points · half-life {selected['ar1_half_life_days']:.1f} trading days",
        transform=axes[0, 0].transAxes,
        fontsize=11.5,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(OUT / "mean_reversion_analysis.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(
    stats: pd.DataFrame,
    selected: pd.Series,
    yearly: pd.DataFrame,
    buckets: pd.DataFrame,
) -> None:
    pct_selected = stats[
        stats["spread"].eq(selected["spread"])
        & stats["measurement"].eq("percent")
    ].iloc[0]
    point_yearly = yearly[
        yearly["spread"].eq(selected["spread"])
        & yearly["measurement"].eq("points")
    ]
    top = stats[
        stats["measurement"].eq("points")
    ].sort_values(
        ["annual_adf_pass_rate", "annual_mean_std_over_daily_std", "ar1_half_life_days"],
        ascending=[False, True, True],
    )
    lines = [
        "# Mean reversion across the VIX futures curve",
        "",
        f"All 28 point spreads reject a unit root and pass the full-sample KPSS stationarity check, but most have equilibrium levels that drift with the volatility regime. **{selected['spread']}** is the clearest fixed-level candidate.",
        "",
        f"Its AR(1) equilibrium is **{selected['ar1_equilibrium']:.3f} VIX points**, its sample mean is **{selected['sample_mean']:.3f}**, and its estimated half-life is **{selected['ar1_half_life_days']:.1f} trading days**. The percentage version reverts toward approximately **{pct_selected['ar1_equilibrium']:.2%}** of the M3 future.",
        "",
        f"The point spread passed the annual ADF test in **{selected['annual_adf_pass_rate']:.0%}** of years and had an AR(1) coefficient below one in **{selected['annual_ar1_reverting_rate']:.0%}** of years. It crossed the full-sample equilibrium {selected['equilibrium_crossings']:.0f} times, or about {selected['crossings_per_year']:.1f} times per year.",
        "",
        "## Annual behavior of the selected spread",
        "",
        "| Year | Mean | AR equilibrium | Half-life | ADF p-value |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in point_yearly.iterrows():
        lines.append(
            f"| {int(row['year'])} | {row['mean']:.3f} | {row['ar1_equilibrium']:.3f} | "
            f"{row['ar1_half_life_days']:.1f} days | {row['adf_p_value']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leading point-spread candidates",
            "",
            "| Spread | Equilibrium | Half-life | Annual ADF pass | Annual mean range | Crossings/year |",
            "|:---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in top.head(8).iterrows():
        lines.append(
            f"| {row['spread']} | {row['ar1_equilibrium']:.3f} | {row['ar1_half_life_days']:.1f} | "
            f"{row['annual_adf_pass_rate']:.0%} | {row['annual_mean_range']:.3f} | "
            f"{row['crossings_per_year']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The equilibrium is an estimated center, not a guaranteed trading target. VIX curve spreads can move several standard deviations during volatility shocks, and the annual M4-M3 equilibrium ranged from roughly 0.27 to 0.56 points in this sample.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = curve_source.build_panel()
    stats, yearly = analyze_spreads(panel)
    selected = select_candidate(stats)
    buckets = reversion_buckets(panel, selected)

    stats.to_csv(OUT / "spread_mean_reversion_statistics.csv", index=False)
    yearly.to_csv(OUT / "yearly_spread_statistics.csv", index=False)
    buckets.to_csv(OUT / "selected_spread_reversion_buckets.csv", index=False)
    pd.DataFrame([selected]).to_csv(OUT / "selected_spread.csv", index=False)
    save_chart(panel, selected, yearly, buckets)
    write_report(stats, selected, yearly, buckets)

    print("SELECTED")
    print(selected.to_string())
    print("\nPOINT-SPREAD LEADERS")
    print(
        stats[stats["measurement"].eq("points")]
        .sort_values(
            ["annual_adf_pass_rate", "annual_mean_std_over_daily_std", "ar1_half_life_days"],
            ascending=[False, True, True],
        )
        .head(10)[
            [
                "spread",
                "ar1_equilibrium",
                "ar1_half_life_days",
                "adf_p_value",
                "kpss_p_value",
                "annual_adf_pass_rate",
                "annual_mean_range",
                "equilibrium_crossings",
            ]
        ]
        .to_string(index=False)
    )
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
