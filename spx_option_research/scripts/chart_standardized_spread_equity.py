from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[1] / "results/standardized_iv_rv_spread_selector"
INITIAL = 1_000_000.0


def main() -> None:
    trades = pd.read_csv(OUT / "portfolio_trades.csv", parse_dates=["entry_date", "expiration_date"])
    summary = pd.read_csv(OUT / "summary.csv").set_index("portfolio")
    styles = {
        "Dynamic: 24-month IV/RV z-score": ("Dynamic IV/RV z-score", "#087f8c"),
        "Fixed 99/96": ("Fixed 99/96", "#d68028"),
    }
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("white")
    curves = []
    for name, (label, color) in styles.items():
        group = trades[trades.portfolio.eq(name)].sort_values("entry_date")
        recomputed = (1.0 + group.pnl_realistic_pct_spot_notional).cumprod().to_numpy()
        np.testing.assert_allclose(recomputed, group.ending_equity_realistic, rtol=1e-12)
        np.testing.assert_allclose(recomputed[-1], summary.loc[name, "ending_wealth_realistic"], rtol=1e-12)
        dates = pd.DatetimeIndex([group.entry_date.min(), *group.expiration_date])
        values = INITIAL * np.r_[1.0, recomputed]
        curve = pd.Series(values, index=dates, name=label)
        if curves and not curve.index.equals(curves[0].index):
            raise ValueError("equity curves must use identical valuation dates")
        curves.append(curve)
        metrics = summary.loc[name]
        ax.plot(dates, values, color=color, linewidth=2.5,
                label=f"{label}   |   CAGR {metrics.cagr_realistic:.2%}   |   Sharpe {metrics.sharpe_realistic:.2f}")
        ax.scatter(dates[-1], values[-1], color=color, s=28, zorder=4)
        ax.annotate(f"${values[-1]:,.0f}", (dates[-1], values[-1]), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=11, color=color, fontweight="bold")
    combined = pd.concat(curves, axis=1)
    combined.to_csv(OUT / "equity_curves_1m.csv", index_label="date")
    ax.set_title("SPX put spreads: growth of $1 million", loc="left", fontsize=18, fontweight="bold", pad=30)
    ax.text(0, 1.025, "Dynamic selection versus fixed 99/96  |  November 2018–September 2026", transform=ax.transAxes,
            fontsize=11, color="#475569")
    ax.set_ylabel("Portfolio equity", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1_000_000:.2f}M"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(combined.index.min() - pd.Timedelta(days=55), combined.index.max() + pd.Timedelta(days=340))
    ax.axhline(INITIAL, color="#94a3b8", linewidth=0.8, linestyle="--", zorder=1)
    ax.grid(alpha=0.25, color="#94a3b8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.text(0.08, 0.035, "100% of current equity as SPX spot notional at each entry · After assumed execution costs\n"
             "Option P&L only; no collateral interest · Equity measured at monthly expirations", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0.01, 0.09, 0.99, 0.98))
    fig.savefig(OUT / "equity_curve_1m.png", dpi=180)
    fig.savefig(OUT / "equity_curve_1m.svg")
    plt.close(fig)
    print(combined.iloc[[0, -1]].to_string())


if __name__ == "__main__":
    main()
