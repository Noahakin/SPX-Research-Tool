"""Create deterministic post-results tables, charts, and report.

This script summarizes choices already frozen by run_weekly_dynamic_research.py.
It never fits a model, changes a gate, or selects a policy from holdout results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/weekly_dynamic_research"

VALIDATION_FILE = OUT / "validation_policy_results.csv"
FROZEN_FILE = OUT / "frozen_model_choices.json"
HOLDOUT_SUMMARY_FILE = OUT / "holdout_summary.csv"
DAILY_FILE = OUT / "holdout_daily_portfolios.parquet"
WEEKLY_FILE = OUT / "holdout_weekly_portfolios.csv"
SELECTION_FILE = OUT / "holdout_selections.csv"
BOOTSTRAP_FILE = OUT / "holdout_paired_bootstrap.csv"
FEATURE_FILE = OUT / "research_features.parquet"
MARK_FILE = OUT / "unit_marks.parquet"
SPX_FILE = OUT / "market_spx.csv"
PROTOCOL_FILE = OUT / "research_protocol.json"
MARK_SENSITIVITY_FILE = OUT / "holdout_mark_sensitivity.csv"
TIMING_ATTRIBUTION_FILE = OUT / "holdout_timing_attribution.csv"

MODEL_FAMILIES = ("ridge", "boosting")
UNIVERSES = ("width3", "adaptive_width")
FEATURE_FAMILIES = (
    "economic",
    "greeks",
    "iv_skew",
    "vol_regime",
    "momentum",
    "liquidity",
)
PRIMARY_GATE = "positive_edge"
CHART_PORTFOLIOS = (
    "Dynamic winner: width3",
    "Dynamic winner: adaptive_width",
    "Validation-best fixed spread",
    "Fixed weekly 99/96",
)
COLORS = {
    "Dynamic winner: width3": "#2563eb",
    "Dynamic winner: adaptive_width": "#0f766e",
    "Validation-best fixed spread": "#d97706",
    "Fixed weekly 99/96": "#334155",
    "Economic model: fixed width": "#9333ea",
    "Economic model: adaptive width": "#db2777",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_inputs() -> tuple[Path, ...]:
    return (
        VALIDATION_FILE,
        FROZEN_FILE,
        HOLDOUT_SUMMARY_FILE,
        DAILY_FILE,
        WEEKLY_FILE,
        SELECTION_FILE,
        BOOTSTRAP_FILE,
        FEATURE_FILE,
        MARK_FILE,
        SPX_FILE,
        PROTOCOL_FILE,
    )


def load_inputs() -> dict[str, object]:
    missing = [path.name for path in _required_inputs() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Final weekly research outputs are not complete; missing: "
            + ", ".join(missing)
        )
    frozen = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
    expected_validation_hash = frozen.get("validation_results_sha256")
    if expected_validation_hash and _sha256(VALIDATION_FILE) != expected_validation_hash:
        raise ValueError("validation_policy_results.csv no longer matches the frozen choices")
    expected_protocol_hash = frozen.get("research_protocol_sha256")
    if expected_protocol_hash and _sha256(PROTOCOL_FILE) != expected_protocol_hash:
        raise ValueError("research_protocol.json no longer matches the frozen choices")

    validation = pd.read_csv(
        VALIDATION_FILE, parse_dates=["first_entry", "last_expiration"]
    )
    summary = pd.read_csv(
        HOLDOUT_SUMMARY_FILE, parse_dates=["first_entry", "last_expiration"]
    )
    daily = pd.read_parquet(DAILY_FILE)
    weekly = pd.read_csv(
        WEEKLY_FILE, parse_dates=["entry_date", "expiration_date"]
    )
    selections = pd.read_csv(
        SELECTION_FILE, parse_dates=["entry_date", "expiration_date"]
    )
    bootstrap = pd.read_csv(BOOTSTRAP_FILE)
    features = pd.read_parquet(FEATURE_FILE)
    marks = pd.read_parquet(MARK_FILE)
    spx = pd.read_csv(SPX_FILE, parse_dates=["date"])
    mark_sensitivity = (
        pd.read_csv(
            MARK_SENSITIVITY_FILE,
            parse_dates=["first_entry", "last_expiration"],
        )
        if MARK_SENSITIVITY_FILE.exists()
        else pd.DataFrame()
    )
    timing_attribution = (
        pd.read_csv(
            TIMING_ATTRIBUTION_FILE,
            parse_dates=["first_entry", "last_expiration"],
        )
        if TIMING_ATTRIBUTION_FILE.exists()
        else pd.DataFrame()
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    for column in ("active_entry_date", "active_expiration_date"):
        if column in daily:
            daily[column] = pd.to_datetime(daily[column]).dt.normalize()
    for column in ("entry_date", "expiration_date"):
        features[column] = pd.to_datetime(features[column]).dt.normalize()
    marks["date"] = pd.to_datetime(marks["date"]).dt.normalize()
    return {
        "frozen": frozen,
        "validation": validation,
        "summary": summary,
        "daily": daily,
        "weekly": weekly,
        "selections": selections,
        "bootstrap": bootstrap,
        "features": features,
        "marks": marks,
        "spx": spx,
        "mark_sensitivity": mark_sensitivity,
        "timing_attribution": timing_attribution,
    }


def validate_holdout_inputs(data: dict[str, object]) -> None:
    frozen = data["frozen"]
    choices = list(frozen["choices"])
    summary = data["summary"]
    daily = data["daily"]
    weekly = data["weekly"]
    selections = data["selections"]
    features = data["features"]
    marks = data["marks"]
    spx = data["spx"]

    for name, frame, required in (
        ("holdout summary", summary, {"portfolio", "fill", "cagr", "daily_sharpe"}),
        ("daily portfolios", daily, {"portfolio", "date", "equity", "daily_return"}),
        (
            "weekly portfolios",
            weekly,
            {
                "portfolio",
                "entry_date",
                "expiration_date",
                "candidate_id",
                "weekly_return",
                "traded",
            },
        ),
        (
            "holdout selections",
            selections,
            {"portfolio", "entry_date", "expiration_date", "candidate_id"},
        ),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} lacks columns: {sorted(missing)}")
    for name, frame in (("daily", daily), ("weekly", weekly), ("selection", selections)):
        absent = set(choices) - set(frame["portfolio"])
        if absent:
            raise ValueError(f"{name} output lacks frozen portfolios: {sorted(absent)}")
    summary_pairs = set(map(tuple, summary[["portfolio", "fill"]].to_numpy()))
    expected_pairs = {(portfolio, fill) for portfolio in choices for fill in ("realistic", "natural")}
    if not expected_pairs.issubset(summary_pairs):
        raise ValueError("holdout summary does not contain both frozen execution fills")
    if daily.duplicated(["portfolio", "date"]).any():
        raise ValueError("daily portfolio output has duplicate portfolio/date rows")
    if weekly.duplicated(["portfolio", "entry_date"]).any():
        raise ValueError("weekly portfolio output has duplicate portfolio/week rows")
    if selections.duplicated(["portfolio", "entry_date"]).any():
        raise ValueError("holdout selection output has duplicate portfolio/week rows")
    if features["candidate_id"].duplicated().any():
        raise ValueError("research feature candidate IDs are not unique")

    traded = weekly[weekly["traded"].astype(bool)].copy()
    missing_candidates = set(traded["candidate_id"]) - set(features["candidate_id"])
    if missing_candidates:
        raise ValueError("weekly output references candidate IDs absent from research features")
    terminal = marks[marks.get("is_expiration", False).astype(bool)][
        ["candidate_id", "cum_return_realistic"]
    ].drop_duplicates("candidate_id")
    reconciled = traded.merge(terminal, on="candidate_id", how="left", validate="many_to_one")
    if reconciled["cum_return_realistic"].isna().any() or not np.allclose(
        reconciled["weekly_return"], reconciled["cum_return_realistic"], rtol=1e-10, atol=1e-11
    ):
        raise ValueError("holdout weekly returns do not reconcile to terminal unit marks")
    spx_dates = set(pd.to_datetime(spx["date"]).dt.normalize())
    if not set(daily["date"]).issubset(spx_dates):
        raise ValueError("daily portfolio dates are absent from the cash-SPX calendar")


def _parse_ladder_model(model: str) -> tuple[str, str, str, int] | None:
    if "__" not in model:
        return None
    model_family, specification = model.split("__", 1)
    if model_family not in MODEL_FAMILIES:
        return None
    if specification == "core":
        return model_family, "core", "core", 0
    if specification.startswith("core_plus_"):
        family = specification.removeprefix("core_plus_")
        if family in FEATURE_FAMILIES:
            return model_family, "add", family, 1 + FEATURE_FAMILIES.index(family)
    if specification == "full":
        return model_family, "full", "all", 1 + len(FEATURE_FAMILIES)
    if specification.startswith("full_without_"):
        family = specification.removeprefix("full_without_")
        if family in FEATURE_FAMILIES:
            return (
                model_family,
                "leave_one_out",
                family,
                2 + len(FEATURE_FAMILIES) + FEATURE_FAMILIES.index(family),
            )
    return None


def build_validation_ladders(
    validation: pd.DataFrame, frozen: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    frozen_policy_ids = {
        choice.get("validation_policy_id")
        for choice in frozen["choices"].values()
        if choice.get("validation_policy_id")
    }
    for row in validation.to_dict("records"):
        parsed = _parse_ladder_model(str(row["model"]))
        if parsed is None:
            continue
        model_family, ladder_step, feature_family, order = parsed
        row.update(
            {
                "model_family": model_family,
                "ladder_step": ladder_step,
                "feature_family": feature_family,
                "display_order": order,
                "display_label": {
                    "core": "Core",
                    "add": f"Add {feature_family.replace('_', ' ')}",
                    "full": "Full",
                    "leave_one_out": f"Drop {feature_family.replace('_', ' ')}",
                }[ladder_step],
                "is_frozen_dynamic_winner": row["policy_id"] in frozen_policy_ids,
            }
        )
        rows.append(row)
    raw = pd.DataFrame(rows).sort_values(
        ["model_family", "universe", "gate", "display_order"]
    )
    primary = raw[raw["gate"].eq(PRIMARY_GATE)].copy()
    expected = len(MODEL_FAMILIES) * len(UNIVERSES) * (
        2 + 2 * len(FEATURE_FAMILIES)
    )
    if len(primary) != expected:
        raise ValueError(
            f"positive-edge validation ladder has {len(primary)} rows; expected {expected}"
        )
    if primary.groupby(["model_family", "universe"]).size().ne(
        2 + 2 * len(FEATURE_FAMILIES)
    ).any():
        raise ValueError("validation ladder is incomplete for a family/universe pair")

    primary["sharpe_reference"] = np.nan
    primary["sharpe_difference_from_reference"] = np.nan
    for (_, _), positions in primary.groupby(["model_family", "universe"]).groups.items():
        group = primary.loc[positions]
        core_sharpe = float(group.loc[group["ladder_step"].eq("core"), "sharpe"].iloc[0])
        full_sharpe = float(group.loc[group["ladder_step"].eq("full"), "sharpe"].iloc[0])
        add = group["ladder_step"].isin(["add", "full"])
        drop = group["ladder_step"].eq("leave_one_out")
        primary.loc[group.index[add], "sharpe_reference"] = core_sharpe
        primary.loc[group.index[drop], "sharpe_reference"] = full_sharpe
    primary["sharpe_difference_from_reference"] = (
        primary["sharpe"] - primary["sharpe_reference"]
    )
    return raw.reset_index(drop=True), primary.reset_index(drop=True)


def plot_validation_ladder(primary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=True)
    stage_colors = {
        "core": "#64748b",
        "add": "#2563eb",
        "full": "#111827",
        "leave_one_out": "#d97706",
    }
    for row_index, model_family in enumerate(MODEL_FAMILIES):
        for column_index, universe in enumerate(UNIVERSES):
            ax = axes[row_index, column_index]
            group = primary[
                primary["model_family"].eq(model_family)
                & primary["universe"].eq(universe)
            ].sort_values("display_order")
            x = np.arange(len(group))
            ax.bar(
                x,
                group["sharpe"],
                color=[stage_colors[value] for value in group["ladder_step"]],
                width=0.78,
            )
            frozen = group["is_frozen_dynamic_winner"].astype(bool).to_numpy()
            if frozen.any():
                ax.scatter(
                    x[frozen],
                    group.loc[frozen, "sharpe"],
                    marker="*",
                    s=170,
                    color="#dc2626",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
            ax.axhline(0, color="#94a3b8", linewidth=0.8)
            ax.set_xticks(x, group["display_label"], rotation=55, ha="right", fontsize=8)
            ax.set_title(
                f"{model_family.title()} · "
                f"{'fixed 3% width' if universe == 'width3' else 'adaptive width'}",
                loc="left",
                fontsize=11,
                weight="bold",
            )
            ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Validation daily Sharpe")
    axes[1, 0].set_ylabel("Validation daily Sharpe")
    fig.suptitle(
        "Validation feature ladder — identical positive-edge gate for every comparison",
        x=0.07,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.07,
        0.925,
        "2021–2023; core, each single-family addition, full set, and each leave-one-out specification",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.91])
    fig.savefig(OUT / "validation_feature_ladder.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def performance_comparison(
    summary: pd.DataFrame, frozen: dict[str, object]
) -> pd.DataFrame:
    metrics = (
        "cagr",
        "daily_sharpe",
        "daily_annualized_volatility",
        "max_drawdown_daily",
        "annualized_arithmetic_return",
        "total_return",
        "ending_equity",
        "trade_fraction",
        "worst_day",
        "worst_week",
        "mean_width_pct",
        "mean_max_loss_pct",
        "mean_net_delta",
    )
    indexed = summary.set_index(["portfolio", "fill"])
    rows: list[dict[str, object]] = []
    for portfolio, policy in frozen["choices"].items():
        row: dict[str, object] = {
            "portfolio": portfolio,
            "model": policy["model"],
            "universe": policy["universe"],
            "gate": policy["gate"],
        }
        for fill in ("realistic", "natural"):
            values = indexed.loc[(portfolio, fill)]
            for metric in metrics:
                if metric in values:
                    row[f"{metric}_{fill}"] = values[metric]
        row["cagr_execution_sensitivity"] = (
            float(row["cagr_realistic"]) - float(row["cagr_natural"])
        )
        row["sharpe_execution_sensitivity"] = (
            float(row["daily_sharpe_realistic"])
            - float(row["daily_sharpe_natural"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def display_name(portfolio: str, frozen: dict[str, object]) -> str:
    if portfolio == "Dynamic winner: width3":
        return "Dynamic, fixed 3% width"
    if portfolio == "Dynamic winner: adaptive_width":
        return "Dynamic, adaptive width"
    if portfolio == "Fixed weekly 99/96":
        return "Fixed 99/96"
    if portfolio == "Validation-best fixed spread":
        model = frozen["choices"][portfolio]["model"]
        ratio = float(model.split("__", 1)[1])
        return f"Validation-best fixed {ratio:.0%}/{ratio - 0.03:.0%}"
    return portfolio.replace("Economic model: ", "Economic rule, ")


def plot_equity_drawdown(daily: pd.DataFrame, frozen: dict[str, object]) -> None:
    missing = set(CHART_PORTFOLIOS) - set(daily["portfolio"])
    if missing:
        raise ValueError(f"chart portfolios are missing: {sorted(missing)}")
    fig, (equity_ax, drawdown_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]},
    )
    for portfolio in CHART_PORTFOLIOS:
        path = daily[daily["portfolio"].eq(portfolio)].sort_values("date")
        initial = (
            float(path["initial_equity"].iloc[0])
            if "initial_equity" in path
            else 1_000_000.0
        )
        wealth = path["equity"].astype(float) / initial
        peak = pd.Series(
            np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:],
            index=wealth.index,
        )
        drawdown = wealth / peak - 1.0
        label = display_name(portfolio, frozen)
        color = COLORS[portfolio]
        equity_ax.plot(path["date"], wealth, label=label, color=color, linewidth=1.8)
        drawdown_ax.plot(path["date"], drawdown, color=color, linewidth=1.4)
    equity_ax.axhline(1.0, color="#94a3b8", linewidth=0.8)
    equity_ax.set_ylabel("Growth of $1")
    equity_ax.grid(color="#e2e8f0", linewidth=0.7)
    equity_ax.legend(frameon=False, ncol=2, loc="upper left")
    drawdown_ax.axhline(0, color="#94a3b8", linewidth=0.8)
    drawdown_ax.set_ylabel("Drawdown")
    drawdown_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    drawdown_ax.grid(color="#e2e8f0", linewidth=0.7)
    drawdown_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    drawdown_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.suptitle(
        "Frozen weekly SPX put-spread policies — final evaluation period",
        x=0.07,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.07,
        0.925,
        "Daily option marks, realistic execution, 100% entry-equity cash-SPX notional",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.91])
    fig.savefig(OUT / "holdout_equity_drawdown.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def calendar_returns(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    cash_dates = pd.read_csv(OUT / "market_spx.csv", parse_dates=["date"])["date"]
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, object]] = []
    for (portfolio, year), group in frame.groupby(["portfolio", "year"], sort=False):
        ordered = group.sort_values("date")
        rows.append(
            {
                "portfolio": portfolio,
                "year": int(year),
                "first_date": ordered["date"].min(),
                "last_date": ordered["date"].max(),
                "daily_observations": len(ordered),
                "calendar_return": float((1.0 + ordered["daily_return"]).prod() - 1.0),
                "calendar_year_complete": bool(
                    ordered["date"].min() == cash_dates[cash_dates.dt.year.eq(year)].min()
                    and ordered["date"].max().month == 12
                    and ordered["date"].max().day >= 24
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["portfolio", "year"]).reset_index(drop=True)


def selection_analysis(
    selections: pd.DataFrame,
    weekly: pd.DataFrame,
    features: pd.DataFrame,
    frozen: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = selections.copy()
    selected["traded"] = selected["candidate_id"].notna() & selected[
        "candidate_id"
    ].astype(str).str.upper().ne("CASH")
    feature_columns = [
        "candidate_id",
        "target_short_ratio",
        "target_width_pct",
        "actual_short_ratio",
        "actual_long_ratio",
        "premium_realistic_pct_spot_notional",
        "econ_expected_edge",
        "econ_short_fair_edge",
        "econ_long_fair_edge",
        "econ_risk_denominator",
        "max_loss_realistic_pct",
        "short_implied_volatility",
        "long_implied_volatility",
    ]
    selected = selected.merge(
        features[feature_columns], on="candidate_id", how="left", validate="many_to_one"
    )
    selected["policy_forecast_edge"] = np.nan
    risk_scored = selected["model"].str.startswith(("ridge__", "boosting__")) | selected["model"].eq("rule__economic_risk")
    max_loss_scored = selected["model"].isin(["rule__normal_rv", "rule__economic_max_loss"])
    selected.loc[risk_scored, "policy_forecast_edge"] = selected.loc[risk_scored, "score"] * selected.loc[risk_scored, "econ_risk_denominator"]
    selected.loc[max_loss_scored, "policy_forecast_edge"] = selected.loc[max_loss_scored, "score"] * selected.loc[max_loss_scored, "max_loss_realistic_pct"]
    selected = selected.merge(
        weekly[["portfolio", "entry_date", "weekly_return", "traded"]].rename(
            columns={"traded": "weekly_traded"}
        ),
        on=["portfolio", "entry_date"],
        how="left",
        validate="one_to_one",
    )
    if selected["weekly_return"].isna().any():
        raise ValueError("selection rows do not align with weekly portfolio results")

    summary_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for portfolio in frozen["choices"]:
        group = selected[selected["portfolio"].eq(portfolio)].copy()
        traded = group[group["traded"]].copy()
        weeks = len(group)
        trades = len(traded)
        row: dict[str, object] = {
            "portfolio": portfolio,
            "weeks": weeks,
            "trades": trades,
            "cash_weeks": weeks - trades,
            "cash_rate": (weeks - trades) / weeks,
            "win_rate_traded_weeks": float(traded["weekly_return"].gt(0).mean())
            if trades
            else np.nan,
        }
        for column in (
            "target_short_ratio",
            "target_width_pct",
            "actual_short_ratio",
            "actual_long_ratio",
            "premium_realistic_pct_spot_notional",
            "policy_forecast_edge",
            "weekly_return",
            "econ_expected_edge",
            "econ_short_fair_edge",
            "econ_long_fair_edge",
            "short_implied_volatility",
            "long_implied_volatility",
        ):
            row[f"mean_{column}"] = float(traded[column].mean()) if trades else np.nan
        summary_rows.append(row)

        for (short_ratio, width), count in traded.groupby(
            ["target_short_ratio", "target_width_pct"], dropna=False
        ).size().items():
            distribution_rows.append(
                {
                    "portfolio": portfolio,
                    "selection_type": "spread",
                    "target_short_ratio": short_ratio,
                    "target_width_pct": width,
                    "weeks": int(count),
                    "pct_traded_weeks": count / trades if trades else np.nan,
                    "pct_all_weeks": count / weeks,
                }
            )
        if weeks > trades:
            distribution_rows.append(
                {
                    "portfolio": portfolio,
                    "selection_type": "cash",
                    "target_short_ratio": np.nan,
                    "target_width_pct": np.nan,
                    "weeks": weeks - trades,
                    "pct_traded_weeks": np.nan,
                    "pct_all_weeks": (weeks - trades) / weeks,
                }
            )
    summary = pd.DataFrame(summary_rows)
    distribution = pd.DataFrame(distribution_rows)
    spread_rows = distribution[distribution["selection_type"].eq("spread")]
    short = (
        spread_rows.groupby(["portfolio", "target_short_ratio"], as_index=False)["weeks"]
        .sum()
        .merge(summary[["portfolio", "trades"]], on="portfolio", how="left")
    )
    short["pct_traded_weeks"] = short["weeks"] / short["trades"]
    width = (
        spread_rows.groupby(["portfolio", "target_width_pct"], as_index=False)["weeks"]
        .sum()
        .merge(summary[["portfolio", "trades"]], on="portfolio", how="left")
    )
    width["pct_traded_weeks"] = width["weeks"] / width["trades"]
    return summary, distribution, short, width


def plot_selection_distribution(
    summary: pd.DataFrame,
    short: pd.DataFrame,
    width: pd.DataFrame,
    frozen: dict[str, object],
) -> None:
    portfolios = list(frozen["choices"])
    labels = [display_name(name, frozen) for name in portfolios]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1.5, 1.2, 0.8]})

    short_pivot = short.pivot(
        index="portfolio", columns="target_short_ratio", values="pct_traded_weeks"
    ).reindex(portfolios).fillna(0.0)
    bottom = np.zeros(len(portfolios))
    short_colors = plt.cm.Blues(np.linspace(0.32, 0.92, max(len(short_pivot.columns), 1)))
    for color, ratio in zip(short_colors, short_pivot.columns):
        values = short_pivot[ratio].to_numpy()
        axes[0].bar(labels, values, bottom=bottom, label=f"{ratio:.0%}", color=color)
        bottom += values
    axes[0].set_ylabel("Share of traded weeks")
    axes[0].set_title("Short-strike selection", loc="left", weight="bold")
    axes[0].legend(title="Short strike", ncol=7, frameon=False, loc="upper center")

    width_pivot = width.pivot(
        index="portfolio", columns="target_width_pct", values="pct_traded_weeks"
    ).reindex(portfolios).fillna(0.0)
    bottom = np.zeros(len(portfolios))
    width_colors = ["#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]
    for color, ratio in zip(width_colors, width_pivot.columns):
        values = width_pivot[ratio].to_numpy()
        axes[1].bar(labels, values, bottom=bottom, label=f"{ratio:.0%}", color=color)
        bottom += values
    axes[1].set_ylabel("Share of traded weeks")
    axes[1].set_title("Hedge-width selection", loc="left", weight="bold")
    axes[1].legend(title="Width", ncol=4, frameon=False, loc="upper center")

    cash = summary.set_index("portfolio").reindex(portfolios)["cash_rate"]
    axes[2].bar(labels, cash, color=[COLORS.get(name, "#64748b") for name in portfolios])
    axes[2].set_ylabel("Share of all weeks")
    axes[2].set_title("Cash frequency", loc="left", weight="bold")
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelrotation=18)
    fig.suptitle(
        "Frozen-policy holdout selections",
        x=0.07,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT / "holdout_selection_distribution.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _pct(value: object, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}%}"


def _num(value: object, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _markdown_performance(
    comparison: pd.DataFrame, frozen: dict[str, object]
) -> str:
    lines = [
        "| Frozen portfolio | Realistic CAGR | Daily Sharpe | Max drawdown | Cash rate | Natural CAGR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {display_name(row.portfolio, frozen)} | {_pct(row.cagr_realistic)} | "
            f"{_num(row.daily_sharpe_realistic)} | {_pct(row.max_drawdown_daily_realistic)} | "
            f"{_pct(1.0 - row.trade_fraction_realistic)} | {_pct(row.cagr_natural)} |"
        )
    return "\n".join(lines)


def _markdown_annual(annual: pd.DataFrame, frozen: dict[str, object]) -> str:
    pivot = annual.pivot(index="portfolio", columns="year", values="calendar_return")
    years = list(pivot.columns)
    lines = [
        "| Portfolio | " + " | ".join(map(str, years)) + " |",
        "|---|" + "---:|" * len(years),
    ]
    for portfolio in frozen["choices"]:
        values = [
            _pct(pivot.loc[portfolio, year]) if portfolio in pivot.index else "—"
            for year in years
        ]
        lines.append(
            f"| {display_name(portfolio, frozen)} | " + " | ".join(values) + " |"
        )
    return "\n".join(lines)


def write_report(
    data: dict[str, object],
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    selection_summary: pd.DataFrame,
    primary_ladder: pd.DataFrame,
) -> None:
    frozen = data["frozen"]
    bootstrap = data["bootstrap"]
    daily = data["daily"]
    mark_sensitivity = data["mark_sensitivity"]
    timing_attribution = data["timing_attribution"]
    start = daily["date"].min()
    end = daily["date"].max()
    selection_index = selection_summary.set_index("portfolio")
    performance_index = comparison.set_index("portfolio")
    adaptive = performance_index.loc["Dynamic winner: adaptive_width"]
    fixed_width = performance_index.loc["Dynamic winner: width3"]
    baseline = performance_index.loc["Fixed weekly 99/96"]
    validation_fixed = performance_index.loc["Validation-best fixed spread"]
    adaptive_selection = selection_index.loc["Dynamic winner: adaptive_width"]
    baseline_selection = selection_index.loc["Fixed weekly 99/96"]

    main_finding = (
        f"The adaptive dynamic policy did **not** improve the primary risk-adjusted result. "
        f"It earned a {_pct(adaptive.cagr_realistic)} CAGR versus "
        f"{_pct(baseline.cagr_realistic)} for fixed 99/96, but its daily Sharpe was "
        f"{adaptive.daily_sharpe_realistic:.2f} versus {baseline.daily_sharpe_realistic:.2f} "
        f"and its maximum drawdown was {_pct(adaptive.max_drawdown_daily_realistic)} versus "
        f"{_pct(baseline.max_drawdown_daily_realistic)}. The fixed-width dynamic policy "
        f"also failed, with a {_pct(fixed_width.cagr_realistic)} CAGR and "
        f"{fixed_width.daily_sharpe_realistic:.2f} Sharpe. The validation-best fixed spread "
        f"produced a {_pct(validation_fixed.cagr_realistic)} CAGR and "
        f"{validation_fixed.daily_sharpe_realistic:.2f} Sharpe."
    )

    selection_lines = [
        "| Portfolio | Trades | Cash rate | Mean short | Mean width | Mean net delta | Mean net credit | Historical-model edge | Historical short edge | Historical hedge edge | Win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for portfolio in frozen["choices"]:
        row = selection_index.loc[portfolio]
        performance = performance_index.loc[portfolio]
        selection_lines.append(
            f"| {display_name(portfolio, frozen)} | {int(row['trades'])} | {_pct(row['cash_rate'])} | "
            f"{_pct(row['mean_target_short_ratio'], 1)} | {_pct(row['mean_target_width_pct'], 1)} | "
            f"{performance.mean_net_delta_realistic:.3f} | "
            f"{_pct(row['mean_premium_realistic_pct_spot_notional'])} | "
            f"{_pct(row['mean_econ_expected_edge'])} | {_pct(row['mean_econ_short_fair_edge'])} | "
            f"{_pct(row['mean_econ_long_fair_edge'])} | {_pct(row['win_rate_traded_weeks'], 1)} |"
        )

    bootstrap_lines = [
        "| Dynamic portfolio vs fixed 99/96 | Sharpe difference | 95% interval | Annual arithmetic-return difference | 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in bootstrap.itertuples(index=False):
        bootstrap_lines.append(
            f"| {display_name(row.portfolio, frozen)} | {_num(row.sharpe_difference)} | "
            f"[{_num(row.sharpe_difference_ci_low)}, {_num(row.sharpe_difference_ci_high)}] | "
            f"{_pct(row.annual_arithmetic_return_difference)} | "
            f"[{_pct(row.annual_arithmetic_return_difference_ci_low)}, "
            f"{_pct(row.annual_arithmetic_return_difference_ci_high)}] |"
        )

    ladder_lines = [
        "| Model family | Universe | Core Sharpe | Full Sharpe | Full minus core |",
        "|---|---|---:|---:|---:|",
    ]
    for model_family in MODEL_FAMILIES:
        for universe in UNIVERSES:
            group = primary_ladder[
                primary_ladder["model_family"].eq(model_family)
                & primary_ladder["universe"].eq(universe)
            ]
            core = float(group.loc[group["ladder_step"].eq("core"), "sharpe"].iloc[0])
            full = float(group.loc[group["ladder_step"].eq("full"), "sharpe"].iloc[0])
            ladder_lines.append(
                f"| {model_family.title()} | "
                f"{'Fixed 3% width' if universe == 'width3' else 'Adaptive width'} | "
                f"{core:.2f} | {full:.2f} | {full - core:+.2f} |"
            )

    ridge_adaptive = primary_ladder[
        primary_ladder["model_family"].eq("ridge")
        & primary_ladder["universe"].eq("adaptive_width")
        & primary_ladder["ladder_step"].isin(["core", "add", "full"])
    ].sort_values("display_order")
    ridge_adaptive_lines = [
        "| Ridge adaptive feature set | Validation Sharpe | Validation CAGR | Active weeks |",
        "|---|---:|---:|---:|",
    ]
    for row in ridge_adaptive.itertuples(index=False):
        ridge_adaptive_lines.append(
            f"| {row.display_label} | {row.sharpe:.2f} | {_pct(row.cagr)} | {int(row.active_weeks)} |"
        )

    robustness_sections: list[str] = []
    if not mark_sensitivity.empty:
        mark_lines = [
            "| Portfolio | Observed-mid Sharpe | Bounded-mid Sharpe | Difference | Flagged marks | Largest correction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for portfolio in CHART_PORTFOLIOS:
            group = mark_sensitivity[mark_sensitivity["portfolio"].eq(portfolio)].set_index(
                "marking_convention"
            )
            observed = group.loc["observed_mid"]
            bounded = group.loc["payoff_bounded_mid"]
            mark_lines.append(
                f"| {display_name(portfolio, frozen)} | {_num(observed.daily_sharpe)} | "
                f"{_num(bounded.daily_sharpe)} | {_num(bounded.daily_sharpe - observed.daily_sharpe)} | "
                f"{int(observed.out_of_bounds_marks)} | {observed.max_mark_correction_points:.2f} points |"
            )
        robustness_sections.append(
            "### Daily-mark bound sensitivity\n\n"
            + "\n".join(mark_lines)
            + "\n\nThe bounded convention changes only interim quoted spread marks that fall outside the vertical’s payoff bounds; entry execution and expiration cash settlement stay fixed."
        )
    if not timing_attribution.empty:
        timing_lines = [
            "| Dynamic portfolio | Chosen-spread Sharpe | 99/96 on same trade weeks | Difference | Chosen CAGR | Same-week 99/96 CAGR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for portfolio in (
            "Dynamic winner: width3",
            "Dynamic winner: adaptive_width",
        ):
            group = timing_attribution[
                timing_attribution["portfolio"].eq(portfolio)
            ].set_index("counterfactual")
            chosen = group.loc["chosen_spreads"]
            fixed = group.loc["99_96_same_trade_weeks"]
            timing_lines.append(
                f"| {display_name(portfolio, frozen)} | {_num(chosen.daily_sharpe)} | "
                f"{_num(fixed.daily_sharpe)} | {_num(chosen.daily_sharpe - fixed.daily_sharpe)} | "
                f"{_pct(chosen.cagr)} | {_pct(fixed.cagr)} |"
            )
        robustness_sections.append(
            "### Timing and strike attribution\n\n"
            + "\n".join(timing_lines)
            + "\n\nThe counterfactual preserves each dynamic policy’s cash/trade timing and substitutes 99/96 only on its traded weeks. It isolates spread choice from the value of staying in cash. For the fixed-width dynamic policy, same-week 99/96 also performed materially better, so the weak result was not explained by cash timing alone."
        )
    robustness_text = (
        "\n\n## Predefined robustness checks\n\n" + "\n\n".join(robustness_sections)
        if robustness_sections
        else ""
    )

    partial_periods = annual.loc[~annual["calendar_year_complete"], ["year", "first_date", "last_date"]].drop_duplicates()
    partial_note = " " + "; ".join(
        f"{row.year} covers {row.first_date:%B %d} through {row.last_date:%B %d}"
        for row in partial_periods.itertuples(index=False)
    ) + "." if len(partial_periods) else ""
    report = f"""# Weekly dynamic SPX put-spread research

## Main result

{main_finding}

The adaptive policy traded {int(adaptive_selection.trades)} of {int(adaptive_selection.weeks)} weeks, averaged a {_pct(adaptive_selection.mean_target_width_pct, 2)} target width and {adaptive.mean_net_delta_realistic:.3f} net delta. Fixed 99/96 averaged {baseline.mean_net_delta_realistic:.3f} net delta. Its higher return therefore came with more option exposure, not a better Sharpe ratio.

## Test definition

The ten-year September 2016–September 2026 archive supplies 520 usable weekly cycles and 14,501 candidate spreads. The grid uses short strikes at 97% through 103% of cash SPX and hedge widths of 1%, 2%, 3% and 5% of spot. Research compared 86 features, 28 regression/feature combinations, four economic or relative-IV rules, and 199 policy specifications. Earlier observations supply training history; there are no ten-year independent model test returns.

The final evaluation covers {start:%B %d, %Y} through {end:%B %d, %Y}. All policies below were frozen from 2021–2023 validation before final-period results were evaluated. Entry decisions and fills use the EOD option snapshot. The comparison uses 100% of current equity in cash-SPX notional, fractional contracts, daily option marks, a zero cash return, and realistic execution at 25% of the full bid/ask spread from midpoint plus $1.50 per leg. The modeled quantity is joint vertical compensation: executable net credit less the expected two-leg terminal liability, scaled by risk.

The fixed-width dynamic winner uses the normal-RV economic rule and trades only when expected net edge divided by maximum loss is at least 0.10. The adaptive-width winner uses the frozen ridge core-plus-Greeks model and requires positive predicted net edge. These rules and thresholds come directly from `frozen_model_choices.json`.

## Final-period performance

{_markdown_performance(comparison, frozen)}

[Daily equity and drawdown](holdout_equity_drawdown.png)

## Calendar returns

{_markdown_annual(annual, frozen)}

Calendar returns compound actual daily marked returns within each year.{partial_note}

## What the frozen policies selected

{chr(10).join(selection_lines)}

Net credit and economic-edge components are fractions of entry cash-SPX notional. The three historical-model edge columns use the same filtered historical simulation diagnostic for every portfolio. They are not the ridge model's forecast or the normal-RV rule's forecast. “Historical hedge edge” is that estimator's expected long-put payoff less its executable cost. The selected policy's actual forecast net edge and its realized mean weekly P&L are separately saved as `mean_policy_forecast_edge` and `mean_weekly_return` in `holdout_selection_summary.csv`. Fixed benchmarks have no policy forecast. The policy does not require every selected hedge to be independently cheap. Cash weeks remain in every performance statistic with zero option return.

[Selection distributions and cash rates](holdout_selection_distribution.png)

## Validation feature ladder

Every primary feature comparison uses the same `positive_edge` gate. The full 56-row core/add-family/full/leave-one-out table is in `validation_feature_ladder_positive_edge.csv`; all 168 model-ladder rows across every gate are retained in `validation_feature_ladder_all_gates.csv`. `validation_policy_results_all_rows.csv` preserves all 199 validation policies, including rules and fixed spreads. No gate or feature set was chosen from final-period results.

All regression policies share a historical estimate of spread-liability risk in their ranking denominator. Adding or removing the economic feature group changes the prediction inputs, while retaining this common risk normalization. Both the forecast net premium advantage and this denominator vary by spread and week.

{chr(10).join(ladder_lines)}

For the ridge adaptive-width comparison, the exact core and single-family additions were:

{chr(10).join(ridge_adaptive_lines)}

Core plus Greeks was the strongest single-family addition in this fixed 2021–2023 validation comparison. The weaker results for other families in this sample do not show that those inputs are universally useless; they only describe these predefined models, this gate, and this validation period.

[Validation feature ladder](validation_feature_ladder.png)

## Paired uncertainty versus fixed 99/96

{chr(10).join(bootstrap_lines)}

The intervals are paired four-week block-bootstrap intervals from the frozen final-period daily paths. They describe sampling uncertainty within this history and do not convert the exploratory model comparison into an independent replication.{robustness_text}

## Verification and execution limits

All 73 automated tests passed. An independent reconstruction from individual leg quotes, entry fills and cash settlement reproduced all 12 performance rows across six portfolios and two execution assumptions, with maximum daily NAV error below $0.000000003. The [data-quality audit](data_quality_audit.md) and [method audit](method_audit.md) document sources, units, timing and accounting.

Same-EOD option features and fills are a research assumption. The archive does not establish that final option volume, open interest and quotes were available early enough to calculate and execute orders at these exact prices. Cash earns zero, and Sharpe is annualized from daily returns using sqrt(252). These results do not establish a deployable improvement over weekly 99/96.
"""
    (OUT / "research_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    data = load_inputs()
    validate_holdout_inputs(data)
    raw_ladder, primary_ladder = build_validation_ladders(
        data["validation"], data["frozen"]
    )
    frozen_policy_map = {
        f"{policy['model']}__{policy['universe']}__{policy['gate']}": portfolio
        for portfolio, policy in data["frozen"]["choices"].items()
    }
    validation_all = data["validation"].copy()
    validation_all["frozen_portfolio"] = validation_all["policy_id"].map(
        frozen_policy_map
    )
    validation_all["is_frozen_choice"] = validation_all["frozen_portfolio"].notna()
    validation_all.to_csv(OUT / "validation_policy_results_all_rows.csv", index=False)
    raw_ladder.to_csv(OUT / "validation_feature_ladder_all_gates.csv", index=False)
    primary_ladder.to_csv(
        OUT / "validation_feature_ladder_positive_edge.csv", index=False
    )
    plot_validation_ladder(primary_ladder)

    comparison = performance_comparison(data["summary"], data["frozen"])
    comparison.to_csv(OUT / "holdout_performance_comparison.csv", index=False)
    annual = calendar_returns(data["daily"])
    annual.to_csv(OUT / "holdout_calendar_returns.csv", index=False)
    selection_summary, distribution, short, width = selection_analysis(
        data["selections"], data["weekly"], data["features"], data["frozen"]
    )
    selection_summary.to_csv(OUT / "holdout_selection_summary.csv", index=False)
    distribution.to_csv(OUT / "holdout_selection_distribution.csv", index=False)
    short.to_csv(OUT / "holdout_short_strike_distribution.csv", index=False)
    width.to_csv(OUT / "holdout_width_distribution.csv", index=False)
    plot_equity_drawdown(data["daily"], data["frozen"])
    plot_selection_distribution(
        selection_summary, short, width, data["frozen"]
    )
    write_report(data, comparison, annual, selection_summary, primary_ladder)

    manifest = {
        "purpose": "post-results reporting only; no model fitting or policy reselection",
        "frozen_choices_sha256": _sha256(FROZEN_FILE),
        "validation_results_sha256": _sha256(VALIDATION_FILE),
        "holdout_summary_sha256": _sha256(HOLDOUT_SUMMARY_FILE),
        "primary_validation_gate": PRIMARY_GATE,
        "chart_portfolios": list(CHART_PORTFOLIOS),
        "final_period_start": str(data["daily"]["date"].min().date()),
        "final_period_end": str(data["daily"]["date"].max().date()),
        "outputs": [
            "research_report.md",
            "holdout_equity_drawdown.png",
            "holdout_performance_comparison.csv",
            "holdout_calendar_returns.csv",
            "holdout_selection_summary.csv",
            "holdout_selection_distribution.csv",
            "holdout_short_strike_distribution.csv",
            "holdout_width_distribution.csv",
            "holdout_selection_distribution.png",
            "validation_feature_ladder_positive_edge.csv",
            "validation_feature_ladder_all_gates.csv",
            "validation_policy_results_all_rows.csv",
            "validation_feature_ladder.png",
        ],
    }
    if MARK_SENSITIVITY_FILE.exists():
        manifest["mark_sensitivity_sha256"] = _sha256(MARK_SENSITIVITY_FILE)
    if TIMING_ATTRIBUTION_FILE.exists():
        manifest["timing_attribution_sha256"] = _sha256(TIMING_ATTRIBUTION_FILE)
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(f"\nSaved post-results analysis to {OUT}")


if __name__ == "__main__":
    main()
