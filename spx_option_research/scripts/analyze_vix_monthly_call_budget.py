from __future__ import annotations

import calendar
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
VIX_ROOT = WORKSPACE / "_inventory/vix_options"
OUT = PROJECT / "results/vix_monthly_call_budget"

PORTFOLIO = 1_000_000.0
ANNUAL_BUDGET_PCT = 0.05
MONTHLY_BUDGET = PORTFOLIO * ANNUAL_BUDGET_PCT / 12.0
TARGET_DTE = 60
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2026-07-27")
MULTIPLIER = 100.0
COMMISSION = 1.50
SPREAD_FRACTION = 0.25
DELTA_TARGETS = [5, 10, 15, 20, 25]
OTM_TARGETS = [110, 125, 150, 175, 200]

COLUMNS = [
    "snapshot_date",
    "expiration_date",
    "dte",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "delta",
    "underlying_price",
    "option_symbol",
    "volume",
    "open_interest",
]


def build_paths() -> dict[pd.Timestamp, Path]:
    paths: dict[pd.Timestamp, Path] = {}
    for path in VIX_ROOT.rglob("eod.parquet"):
        match = re.search(r"date=(\d{4}-\d{2}-\d{2})", str(path))
        if match and path.stat().st_size > 10_000:
            date = pd.Timestamp(match.group(1))
            if START <= date <= END:
                paths[date] = path
    return paths


def third_friday(year: int, month: int) -> pd.Timestamp:
    weeks = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in weeks if week[calendar.FRIDAY]]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def monthly_dates(paths: dict[pd.Timestamp, Path]) -> list[pd.Timestamp]:
    available = set(paths)
    dates: list[pd.Timestamp] = []
    for period in pd.period_range(START, END, freq="M"):
        scheduled = third_friday(period.year, period.month)
        choices = [
            scheduled - pd.Timedelta(days=offset)
            for offset in range(4)
            if scheduled - pd.Timedelta(days=offset) in available
        ]
        if choices:
            dates.append(max(choices))
    return dates


def read_calls(path: Path) -> pd.DataFrame:
    chain = pd.read_parquet(path, columns=COLUMNS)
    chain["expiration_date"] = pd.to_datetime(chain["expiration_date"])
    symbols = chain["option_symbol"].fillna("").astype(str)
    calls = chain.loc[
        chain["option_type"].astype(str).str.lower().eq("call")
        & symbols.str.match(r"^VIX")
        & chain["bid"].ge(0)
        & chain["ask"].gt(0)
        & chain["ask"].ge(chain["bid"])
        & chain["delta"].between(0, 1)
    ].copy()
    return calls


def nearest_expiration(calls: pd.DataFrame) -> tuple[pd.Timestamp, int]:
    expirations = calls[["expiration_date", "dte"]].drop_duplicates()
    expirations = expirations[expirations["dte"].between(45, 75)].copy()
    if expirations.empty:
        raise LookupError("no expiration between 45 and 75 DTE")
    expirations["distance"] = (expirations["dte"] - TARGET_DTE).abs()
    selected = expirations.sort_values(["distance", "dte", "expiration_date"]).iloc[0]
    return pd.Timestamp(selected["expiration_date"]), int(selected["dte"])


def quote_costs(quote: pd.Series) -> dict[str, float | int]:
    modeled_price = float(
        quote["mid"] + SPREAD_FRACTION * (quote["ask"] - quote["bid"])
    )
    modeled_cost = modeled_price * MULTIPLIER + COMMISSION
    ask_cost = float(quote["ask"]) * MULTIPLIER + COMMISSION
    modeled_contracts = int(np.floor(MONTHLY_BUDGET / modeled_cost))
    ask_contracts = int(np.floor(MONTHLY_BUDGET / ask_cost))
    return {
        "modeled_option_price": modeled_price,
        "modeled_cost_per_contract": modeled_cost,
        "ask_cost_per_contract": ask_cost,
        "whole_contracts_modeled": modeled_contracts,
        "whole_contracts_at_ask": ask_contracts,
        "modeled_monthly_spend": modeled_contracts * modeled_cost,
        "ask_monthly_spend": ask_contracts * ask_cost,
    }


def selection_row(
    entry_date: pd.Timestamp,
    expiration: pd.Timestamp,
    actual_dte: int,
    spot: float,
    quote: pd.Series,
    selection_type: str,
    target: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "entry_date": entry_date,
        "expiration_date": expiration,
        "actual_dte": actual_dte,
        "selection_type": selection_type,
        "target": target,
        "spot": spot,
        "strike": float(quote["strike"]),
        "strike_pct_spot": float(quote["strike"]) / spot,
        "delta": float(quote["delta"]),
        "bid": float(quote["bid"]),
        "ask": float(quote["ask"]),
        "mid": float(quote["mid"]),
        "zero_bid": bool(float(quote["bid"]) == 0),
        "volume": float(quote["volume"]),
        "open_interest": float(quote["open_interest"]),
        "option_symbol": str(quote["option_symbol"]),
    }
    row.update(quote_costs(quote))
    return row


def summarize(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (selection_type, target), group in selections.groupby(
        ["selection_type", "target"], sort=True
    ):
        modeled_cost = group["modeled_cost_per_contract"]
        ask_cost = group["ask_cost_per_contract"]
        modeled_spend = group["modeled_monthly_spend"]
        ask_spend = group["ask_monthly_spend"]
        rows.append(
            {
                "selection_type": selection_type,
                "target": target,
                "months": len(group),
                "average_actual_dte": group["actual_dte"].mean(),
                "median_actual_delta": group["delta"].median(),
                "median_strike_pct_spot": group["strike_pct_spot"].median(),
                "median_modeled_cost_per_contract": modeled_cost.median(),
                "average_modeled_cost_per_contract": modeled_cost.mean(),
                "p10_modeled_cost_per_contract": modeled_cost.quantile(0.10),
                "p90_modeled_cost_per_contract": modeled_cost.quantile(0.90),
                "median_ask_cost_per_contract": ask_cost.median(),
                "median_whole_contracts_modeled": group[
                    "whole_contracts_modeled"
                ].median(),
                "minimum_whole_contracts_modeled": group[
                    "whole_contracts_modeled"
                ].min(),
                "maximum_whole_contracts_modeled": group[
                    "whole_contracts_modeled"
                ].max(),
                "median_whole_contracts_at_ask": group[
                    "whole_contracts_at_ask"
                ].median(),
                "annual_cost_one_contract_modeled": modeled_cost.mean() * 12.0,
                "annual_cost_one_contract_pct_portfolio": (
                    modeled_cost.mean() * 12.0 / PORTFOLIO
                ),
                "annualized_spend_modeled": modeled_spend.mean() * 12.0,
                "annualized_spend_at_ask": ask_spend.mean() * 12.0,
                "budget_utilization_modeled": (
                    modeled_spend.mean() * 12.0 / (PORTFOLIO * ANNUAL_BUDGET_PCT)
                ),
                "budget_utilization_at_ask": (
                    ask_spend.mean() * 12.0 / (PORTFOLIO * ANNUAL_BUDGET_PCT)
                ),
                "zero_bid_rate": group["zero_bid"].mean(),
                "median_volume": group["volume"].median(),
                "median_open_interest": group["open_interest"].median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = build_paths()
    entries = monthly_dates(paths)
    rows: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []

    for entry_date in entries:
        try:
            calls = read_calls(paths[entry_date])
            expiration, actual_dte = nearest_expiration(calls)
        except LookupError as error:
            skips.append({"entry_date": entry_date, "reason": str(error)})
            continue
        expiry_calls = calls[calls["expiration_date"].eq(expiration)].copy()
        spot = float(expiry_calls["underlying_price"].dropna().median())
        otm = expiry_calls[expiry_calls["strike"].gt(spot)].copy()
        if otm.empty:
            skips.append({"entry_date": entry_date, "reason": "no OTM calls"})
            continue

        for target_delta in DELTA_TARGETS:
            quote = otm.loc[
                (otm["delta"] - target_delta / 100.0).abs().idxmin()
            ]
            rows.append(
                selection_row(
                    entry_date,
                    expiration,
                    actual_dte,
                    spot,
                    quote,
                    "delta",
                    float(target_delta),
                )
            )
        for target_otm in OTM_TARGETS:
            quote = otm.loc[
                (otm["strike"] - spot * target_otm / 100.0).abs().idxmin()
            ]
            rows.append(
                selection_row(
                    entry_date,
                    expiration,
                    actual_dte,
                    spot,
                    quote,
                    "strike_pct_spot",
                    float(target_otm),
                )
            )

    selections = pd.DataFrame(rows)
    summary = summarize(selections)
    selections.to_csv(OUT / "monthly_selections.csv", index=False)
    summary.to_csv(OUT / "affordability_summary.csv", index=False)
    pd.DataFrame(skips).to_csv(OUT / "skips.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.patch.set_facecolor("#f8fafc")
    configurations = [
        ("delta", "Target call delta", "Delta-selected 60-DTE calls"),
        ("strike_pct_spot", "Target strike (% of VIX spot)", "Spot-OTM 60-DTE calls"),
    ]
    for ax, (selection_type, xlabel, title) in zip(axes, configurations):
        subset = summary[summary["selection_type"].eq(selection_type)].sort_values(
            "target"
        )
        bars = ax.bar(
            subset["target"],
            subset["median_whole_contracts_modeled"],
            color="#087f8c",
            alpha=0.88,
            width=3.2 if selection_type == "delta" else 10,
            label="Median contracts",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Whole contracts for $4,166.67")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, value in zip(bars, subset["median_whole_contracts_modeled"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(
        "VIX call capacity with a 5% annual premium budget",
        x=0.06,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.925,
        "$1 million portfolio · $50,000/year · $4,166.67/month · ~60 DTE · modeled limit execution",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.99, 0.89])
    fig.savefig(OUT / "monthly_contract_capacity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    delta_summary = summary[summary["selection_type"].eq("delta")].sort_values(
        "target"
    )
    otm_summary = summary[
        summary["selection_type"].eq("strike_pct_spot")
    ].sort_values("target")
    report_lines = [
        "# VIX monthly call affordability",
        "",
        f"A 5% annual premium budget on a ${PORTFOLIO:,.0f} portfolio is ${PORTFOLIO * ANNUAL_BUDGET_PCT:,.0f} per year, or ${MONTHLY_BUDGET:,.2f} for each monthly purchase. Calls are selected near 60 DTE on the third-Friday monthly schedule from January 2022 through July 2026. Modeled execution pays midpoint plus 25% of the bid/ask spread and $1.50 commission per contract; ask-price capacity is also reported.",
        "",
        "## Delta-selected calls",
        "",
        "| Target delta | Median strike/spot | Median cost/contract | Median contracts | Median contracts at ask | Annual cost of one/month | Zero-bid rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in delta_summary.itertuples():
        report_lines.append(
            f"| {row.target:.0f} | {row.median_strike_pct_spot:.0%} | "
            f"${row.median_modeled_cost_per_contract:,.0f} | "
            f"{row.median_whole_contracts_modeled:.0f} | "
            f"{row.median_whole_contracts_at_ask:.0f} | "
            f"${row.annual_cost_one_contract_modeled:,.0f} | {row.zero_bid_rate:.1%} |"
        )
    report_lines.extend(
        [
            "",
            "## Strike-percentage calls",
            "",
            "| Target strike/spot | Median actual delta | Median cost/contract | Median contracts | Median contracts at ask | Annual cost of one/month | Zero-bid rate |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in otm_summary.itertuples():
        report_lines.append(
            f"| {row.target:.0f}% | {row.median_actual_delta:.0%} | "
            f"${row.median_modeled_cost_per_contract:,.0f} | "
            f"{row.median_whole_contracts_modeled:.0f} | "
            f"{row.median_whole_contracts_at_ask:.0f} | "
            f"${row.annual_cost_one_contract_modeled:,.0f} | {row.zero_bid_rate:.1%} |"
        )
    report_lines.extend(
        [
            "",
            "Contract counts are purchasing capacity, not a recommended position size. With roughly 60 DTE and monthly purchases, approximately two cohorts will usually overlap. Premium paid is the maximum loss on each long-call purchase.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("DELTA TARGETS")
    print(delta_summary.to_string(index=False))
    print("\nOTM TARGETS")
    print(otm_summary.to_string(index=False))
    print(f"\nMonthly observations: {selections['entry_date'].nunique()}")
    print(f"Saved results to {OUT}")


if __name__ == "__main__":
    main()
