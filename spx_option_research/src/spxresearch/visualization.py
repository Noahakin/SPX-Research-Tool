from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown_series


STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f7f9fb",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_strategy_diagnostics(
    returns: pd.Series,
    benchmark: pd.Series,
    output_dir: str | Path,
    *,
    label: str,
) -> None:
    output = Path(output_dir)
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark.rename("SPX")], axis=1
    ).dropna()
    strategy = aligned["strategy"]
    spx = aligned["SPX"]

    figure, axis = plt.subplots(figsize=(10, 5))
    ((1 + aligned).cumprod()).plot(ax=axis, linewidth=1.5)
    axis.set_title(f"Cumulative NAV — {label}")
    axis.set_ylabel("Growth of $1")
    _save(figure, output / "cumulative_nav.png")

    figure, axis = plt.subplots(figsize=(10, 4))
    drawdown_series(strategy).plot(ax=axis, color="#b22222")
    axis.fill_between(strategy.index, drawdown_series(strategy), 0, color="#b22222", alpha=0.2)
    axis.set_title(f"Drawdown — {label}")
    axis.set_ylabel("Drawdown")
    _save(figure, output / "drawdown.png")

    rolling_return = (1 + strategy).rolling(252).apply(np.prod, raw=True) - 1
    rolling_vol = strategy.rolling(252).std() * np.sqrt(252)
    rolling_sharpe = strategy.rolling(252).mean() / strategy.rolling(252).std() * np.sqrt(252)
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    rolling_return.plot(ax=axes[0], color="#0b5a8f", title="Rolling 12-month return")
    rolling_vol.plot(ax=axes[1], color="#d47b00", title="Rolling 12-month volatility")
    rolling_sharpe.plot(ax=axes[2], color="#277a3d", title="Rolling 12-month Sharpe")
    _save(figure, output / "rolling_metrics.png")

    monthly = (1 + strategy).resample("ME").prod() - 1
    heat = monthly.to_frame("return")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.month
    pivot = heat.pivot(index="year", columns="month", values="return")
    figure, axis = plt.subplots(figsize=(11, max(4, len(pivot) * 0.45)))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.1)
    axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
    axis.set_xticks(range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    axis.set_title(f"Monthly returns — {label}")
    figure.colorbar(image, ax=axis, label="Return")
    _save(figure, output / "monthly_heatmap.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(strategy, bins=60, color="#0b5a8f", alpha=0.8)
    axes[0].set_title("Daily-return distribution")
    tail = strategy[strategy <= strategy.quantile(0.10)].sort_values()
    axes[1].plot(np.arange(1, len(tail) + 1), tail.to_numpy(), color="#b22222")
    axes[1].set_title("Worst-decile tail")
    _save(figure, output / "return_and_tail_distributions.png")

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(spx, strategy, s=9, alpha=0.35, color="#594a9b")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set_xlabel("SPX daily return")
    axis.set_ylabel("Strategy daily return")
    axis.set_title(f"SPX vs strategy — {label}")
    _save(figure, output / "spx_scatter.png")

    bins = [-np.inf, -0.10, -0.05, 0.0, 0.05, np.inf]
    names = ["< -10%", "-10% to -5%", "-5% to 0%", "0% to 5%", "> 5%"]
    buckets = pd.cut(spx, bins=bins, labels=names)
    conditional = strategy.groupby(buckets, observed=True).mean()
    figure, axis = plt.subplots(figsize=(8, 4))
    conditional.plot.bar(ax=axis, color=np.where(conditional >= 0, "#277a3d", "#b22222"))
    axis.set_title("Average strategy return conditional on SPX")
    axis.set_xlabel("SPX return bucket")
    _save(figure, output / "conditional_spx_buckets.png")


def plot_core_heatmaps(screen: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    for width_method, frame in screen.groupby("width_method"):
        finite = frame.dropna(subset=["robust_score", "oos_sharpe"])
        if finite.empty:
            continue
        selected = finite.loc[
            finite.groupby("base_strategy_id")["robust_score"].idxmax()
        ]
        pivot = selected.pivot_table(
            index="short_put_delta", columns="width_value", values="oos_sharpe", aggfunc="median"
        ).sort_index()
        if pivot.empty:
            continue
        figure, axis = plt.subplots(figsize=(10, 6))
        values = pivot.to_numpy()
        finite_values = values[np.isfinite(values)]
        if not len(finite_values):
            plt.close(figure)
            continue
        image = axis.imshow(
            values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=np.percentile(finite_values, 5),
            vmax=np.percentile(finite_values, 95),
        )
        axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
        axis.set_xticks(range(len(pivot.columns)), labels=[f"{value:g}" for value in pivot.columns], rotation=45)
        axis.set_ylabel("Short-put delta")
        axis.set_xlabel(f"Spread width ({width_method})")
        axis.set_title("Out-of-sample Sharpe parameter plateau")
        figure.colorbar(image, ax=axis, label="OOS Sharpe")
        _save(figure, output / f"heatmap_delta_width_{width_method}.png")


def plot_frontier(
    table: pd.DataFrame,
    x: str,
    y: str,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    clean = table[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(clean[x], clean[y], s=18, alpha=0.55, color="#0b5a8f")
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(y.replace("_", " ").title())
    axis.set_title(title)
    _save(figure, Path(output_path))


def plot_metric_heatmap(
    table: pd.DataFrame,
    row: str,
    column: str,
    value: str,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    clean = table[[row, column, value]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return
    pivot = clean.pivot_table(index=row, columns=column, values=value, aggfunc="median").sort_index()
    values = pivot.to_numpy(float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=np.percentile(finite, 5),
        vmax=np.percentile(finite, 95),
    )
    axis.set_yticks(range(len(pivot.index)), labels=[f"{item:g}" for item in pivot.index])
    axis.set_xticks(
        range(len(pivot.columns)),
        labels=[f"{item:g}" for item in pivot.columns],
        rotation=45,
    )
    axis.set_ylabel(row.replace("_", " ").title())
    axis.set_xlabel(column.replace("_", " ").title())
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=value.replace("_", " ").title())
    _save(figure, Path(output_path))
