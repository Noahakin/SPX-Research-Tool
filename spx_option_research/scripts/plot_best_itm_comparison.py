from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/sequential_itm_hedges"
INITIAL_CAPITAL = 1_000_000.0

STRATEGIES = {
    "104/101_dte21_profit_90_none": {
        "label": "104/101 • 21 DTE • exit at 90% profit",
        "color": "#006b76",
    },
    "104/101_dte45_hold_none": {
        "label": "104/101 • 45 DTE • hold to expiration",
        "color": "#c56a1a",
    },
}


def main() -> None:
    curves = pd.read_parquet(OUT / "finalist_equity_curves.parquet")
    metrics = pd.read_csv(OUT / "finalist_metrics.csv").set_index("strategy_id")

    fig, ax = plt.subplots(figsize=(13, 7.5))
    for strategy_id, style in STRATEGIES.items():
        row = metrics.loc[strategy_id]
        label = (
            f"{style['label']}  |  CAGR {row['cagr']:.2%}, "
            f"max DD {row['max_drawdown']:.2%}"
        )
        ax.plot(
            curves.index,
            curves[strategy_id] / INITIAL_CAPITAL - 1.0,
            color=style["color"],
            linewidth=2.2,
            label=label,
        )

    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.set_title("Best-return versus conservative ITM SPX put spread", loc="left", weight="bold", pad=20)
    ax.set_ylabel("Cumulative option return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(
        0.075,
        0.025,
        "One position at a time; short-strike notional equals current equity; realistic execution; no cash interest.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "best_itm_two_curve_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
