from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
strategy = "core_dte45_sd45_delta15_capture_50"
curves = pd.read_csv(root / "results/option_target_equity_curves.csv", index_col=0, parse_dates=True)
metrics = pd.read_csv(root / "results/option_target_daily_metrics.csv")
equity = curves[strategy]
returns = equity.pct_change().fillna(equity.iloc[0] / 100.0 - 1.0)
peak = pd.Series(np.maximum.accumulate(np.r_[100.0, equity.to_numpy()])[1:], index=equity.index)
drawdown = equity / peak - 1.0
elapsed = (equity.index - equity.index[0]).days / 365.25

annual = (1.0 + returns).resample("YE").prod() - 1.0
annual.index = annual.index.year
annual.rename("option_return").to_csv(root / "results/option_only_best_annual_returns.csv")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, dd) = plt.subplots(
    2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.fill_between(
    equity.index,
    100.0 * 1.03 ** elapsed,
    100.0 * 1.05 ** elapsed,
    color="#cbd5e1",
    alpha=0.45,
    label="3–5% annual growth reference",
)
ax.plot(equity.index, equity, color="#006b76", lw=2.2, label="Recommended option strategy")
ax.axhline(100, color="#64748b", lw=0.8)
dd.fill_between(drawdown.index, drawdown, 0, color="#0f8090", alpha=0.35)
dd.plot(drawdown.index, drawdown, color="#006b76", lw=1.1)
ax.set_title("SPX option-only equity curve", loc="left", weight="bold", pad=22)
ax.text(
    0,
    1.015,
    "45 DTE • sell 45-delta put • buy ~30-delta put • exit at 50% profit • weekly entries",
    transform=ax.transAxes,
    fontsize=10,
)
ax.set_ylabel("Growth of $100")
ax.legend(loc="upper left", frameon=False)
dd.set_ylabel("Drawdown")
dd.yaxis.set_major_formatter(PercentFormatter(1))
for axis in (ax, dd):
    axis.grid(alpha=0.2)
fig.text(
    0.08,
    0.025,
    "Option P&L only; no cash interest. Sep 2016–Sep 2026; realistic execution costs; fractional contracts; configured 5% portfolio risk budget.\n"
    "CAGR 3.05% • annualized volatility 4.34% • maximum drawdown −10.20% • zero-cash Sharpe 0.71",
    fontsize=9,
    color="#475569",
)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(root / "results/option_only_target_equity.png", dpi=180)
print(annual.to_string())
