from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import chart_vix_m7_m4_spread as spread_source


OUT = spread_source.OUT
SP500_SOURCE = spread_source.WORKSPACE / "liquidity_cycle_study/raw/fred/sp500__SP500.json"


def load_sp500() -> pd.DataFrame:
    with SP500_SOURCE.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    sp500 = pd.DataFrame(raw["observations"])[["date", "value"]].copy()
    sp500["date"] = pd.to_datetime(sp500["date"])
    sp500["sp500"] = pd.to_numeric(sp500["value"], errors="coerce")
    return sp500[["date", "sp500"]].dropna().drop_duplicates("date", keep="last")


def build_panel() -> pd.DataFrame:
    spread = spread_source.build_curve()
    panel = spread.merge(load_sp500(), on="date", how="inner").sort_values("date")
    panel["sp500_return"] = panel["sp500"].pct_change()
    panel["spread_change"] = panel["m7_minus_m4"].diff()
    panel["sp500_drawdown"] = panel["sp500"] / panel["sp500"].cummax() - 1.0
    return panel


def save_chart(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].plot(
        panel["date"], panel["sp500"], color="#2563eb", linewidth=2.0, label="S&P 500"
    )
    axes[0].set_ylabel("S&P 500 index")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[0].legend(frameon=False, loc="upper left")

    spread = panel["m7_minus_m4"]
    axes[1].plot(
        panel["date"], spread, color="#0f172a", linewidth=1.25, label="M7 - M4 spread"
    )
    axes[1].plot(
        panel["date"],
        panel["spread_21d_average"],
        color="#f59e0b",
        linewidth=2.0,
        label="21-day average",
    )
    axes[1].fill_between(
        panel["date"],
        0,
        spread,
        where=spread.ge(0),
        color="#10b981",
        alpha=0.26,
        interpolate=True,
    )
    axes[1].fill_between(
        panel["date"],
        0,
        spread,
        where=spread.lt(0),
        color="#ef4444",
        alpha=0.30,
        interpolate=True,
    )
    axes[1].axhline(0, color="#334155", linewidth=1.0)
    axes[1].set_ylabel("M7 - M4 (VIX points)")
    axes[1].legend(frameon=False, ncol=2, loc="lower right")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    latest = panel.iloc[-1]
    axes[0].annotate(
        f"{latest['sp500']:,.0f}",
        xy=(latest["date"], latest["sp500"]),
        xytext=(-10, 12),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color="#2563eb",
    )
    axes[1].annotate(
        f"{latest['m7_minus_m4']:+.2f}",
        xy=(latest["date"], latest["m7_minus_m4"]),
        xytext=(-10, 14),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color="#0f172a",
    )

    axes[0].set_title(
        "S&P 500 versus the VIX M7-M4 futures spread",
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    axes[0].text(
        0,
        1.01,
        "January 2022-July 2026 · Negative spread means M4 is above M7",
        transform=axes[0].transAxes,
        fontsize=11.5,
        color="#475569",
    )
    axes[1].text(
        0.995,
        0.015,
        "Sources: local matched VX futures history and FRED S&P 500 close",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout()
    fig.savefig(OUT / "vix_m7_m4_vs_sp500.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    daily = panel[["spread_change", "sp500_return"]].dropna()
    negative = panel["m7_minus_m4"].lt(0)
    stats = pd.DataFrame(
        [
            {
                "start_date": panel["date"].min(),
                "end_date": panel["date"].max(),
                "observations": len(panel),
                "latest_sp500": panel.iloc[-1]["sp500"],
                "latest_m7_minus_m4": panel.iloc[-1]["m7_minus_m4"],
                "correlation_spread_change_vs_sp500_daily_return": daily[
                    "spread_change"
                ].corr(daily["sp500_return"]),
                "average_sp500_daily_return_when_spread_negative": panel.loc[
                    negative, "sp500_return"
                ].mean(),
                "average_sp500_daily_return_when_spread_positive": panel.loc[
                    ~negative, "sp500_return"
                ].mean(),
                "worst_sp500_drawdown": panel["sp500_drawdown"].min(),
                "worst_sp500_drawdown_date": panel.loc[
                    panel["sp500_drawdown"].idxmin(), "date"
                ],
            }
        ]
    )
    stats.to_csv(OUT / "sp500_comparison_summary.csv", index=False)
    return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    panel.to_csv(OUT / "vix_m7_m4_vs_sp500_daily.csv", index=False)
    save_chart(panel)
    stats = save_statistics(panel)
    print(stats.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
