from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
SOURCE = (
    WORKSPACE
    / "06 - Futures/Derived and Supplemental/Unspecified/outputs"
    / "vix_simon_signal_replication/matched_vx_futures_history.csv"
)
OUT = PROJECT / "results/vix_m7_m4_spread"


def build_curve() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE, parse_dates=["date", "expiration_date"])
    raw = raw[
        raw["expiration_date"].gt(raw["date"])
        & raw["future"].notna()
        & raw["future"].gt(0)
    ].copy()
    raw = raw.sort_values(["date", "expiration_date"]).drop_duplicates(
        ["date", "expiration_date"], keep="last"
    )
    raw["contract_number"] = raw.groupby("date").cumcount() + 1
    selected = raw[raw["contract_number"].isin([4, 7])].copy()

    prices = selected.pivot(index="date", columns="contract_number", values="future")
    expiries = selected.pivot(
        index="date", columns="contract_number", values="expiration_date"
    )
    panel = pd.DataFrame(index=prices.index)
    panel["m4_future"] = prices.get(4)
    panel["m7_future"] = prices.get(7)
    panel["m4_expiration"] = expiries.get(4)
    panel["m7_expiration"] = expiries.get(7)
    panel = panel.dropna().reset_index().sort_values("date")
    panel["m7_minus_m4"] = panel["m7_future"] - panel["m4_future"]
    panel["m7_minus_m4_pct_m4"] = panel["m7_future"] / panel["m4_future"] - 1.0
    panel["spread_21d_average"] = panel["m7_minus_m4"].rolling(21).mean()
    return panel


def save_chart(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.25]},
    )
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].plot(
        panel["date"], panel["m4_future"], color="#2563eb", linewidth=1.5, label="M4 future"
    )
    axes[0].plot(
        panel["date"], panel["m7_future"], color="#7c3aed", linewidth=1.5, label="M7 future"
    )
    axes[0].set_ylabel("VIX futures level")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    spread = panel["m7_minus_m4"]
    axes[1].plot(panel["date"], spread, color="#0f172a", linewidth=1.25, label="M7 − M4")
    axes[1].plot(
        panel["date"],
        panel["spread_21d_average"],
        color="#f59e0b",
        linewidth=2.0,
        label="21-day average",
    )
    axes[1].fill_between(
        panel["date"], 0, spread, where=spread.ge(0), color="#10b981", alpha=0.26, interpolate=True
    )
    axes[1].fill_between(
        panel["date"], 0, spread, where=spread.lt(0), color="#ef4444", alpha=0.30, interpolate=True
    )
    axes[1].axhline(0, color="#334155", linewidth=1.0)
    axes[1].set_ylabel("Spread (VIX points)")
    axes[1].legend(frameon=False, ncol=2, loc="lower right")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    latest = panel.iloc[-1]
    axes[1].annotate(
        f"Latest: {latest['m7_minus_m4']:+.2f}",
        xy=(latest["date"], latest["m7_minus_m4"]),
        xytext=(-12, 18),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color="#0f172a",
        arrowprops={"arrowstyle": "-", "color": "#64748b"},
    )

    axes[0].set_title(
        "VIX futures term spread: seventh month minus fourth month",
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    axes[0].text(
        0,
        1.01,
        "Positive = M7 above M4 · Negative = M4–M7 inversion",
        transform=axes[0].transAxes,
        fontsize=11.5,
        color="#475569",
    )
    axes[1].text(
        0.995,
        0.015,
        "Source: local matched VX futures history",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout()
    fig.savefig(OUT / "vix_m7_minus_m4_spread.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_summary(panel: pd.DataFrame) -> pd.DataFrame:
    spread = panel["m7_minus_m4"]
    latest = panel.iloc[-1]
    minimum = panel.loc[spread.idxmin()]
    maximum = panel.loc[spread.idxmax()]
    summary = pd.DataFrame(
        [
            {
                "start_date": panel["date"].min(),
                "end_date": panel["date"].max(),
                "observations": len(panel),
                "latest_m4": latest["m4_future"],
                "latest_m7": latest["m7_future"],
                "latest_spread_m7_minus_m4": latest["m7_minus_m4"],
                "latest_spread_pct_m4": latest["m7_minus_m4_pct_m4"],
                "mean_spread": spread.mean(),
                "median_spread": spread.median(),
                "positive_spread_pct_days": spread.gt(0).mean(),
                "negative_spread_pct_days": spread.lt(0).mean(),
                "minimum_spread": minimum["m7_minus_m4"],
                "minimum_spread_date": minimum["date"],
                "maximum_spread": maximum["m7_minus_m4"],
                "maximum_spread_date": maximum["date"],
            }
        ]
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_curve()
    panel.to_csv(OUT / "vix_m7_m4_daily.csv", index=False)
    save_chart(panel)
    summary = save_summary(panel)
    print(summary.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
