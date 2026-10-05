"""Export the organized SPX chart library and its self-contained HTML browser.

Inputs live outside the client folder: run.json, strategy_metrics.csv, and
curves.npz (dates, strategy_ids, equity with shape dates x strategies).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
BASE_DATA = PROJECT / "results" / "organized_spx_research"
COMBINED_DATA = PROJECT / "results" / "organized_spx_combinations"
BEST_HEDGE_DATA = PROJECT / "results" / "organized_spx_best_hedges"
DEFAULT_DATA = BEST_HEDGE_DATA if (BEST_HEDGE_DATA / "run.json").is_file() else COMBINED_DATA if (COMBINED_DATA / "run.json").is_file() else BASE_DATA
DEFAULT_OUTPUT = PROJECT.parent / "SPX Research Organized Charts"
CATEGORY_ORDER = ("Put selling", "Put buying", "Both", "SPX with overlay")
TENORS = ("3 days", "1 week", "2 weeks", "3 weeks", "4 weeks", "6 weeks", "8 weeks")
TENOR_DAYS = (3, 7, 14, 21, 28, 42, 56)
COLORS = ("#24658B", "#C18436", "#398476", "#B8616E", "#7762A1", "#75954F", "#626F88")
CATEGORY_COLORS = {"Put selling": "#24658B", "Put buying": "#B8616E", "Both": "#A0742E", "SPX with overlay": "#398476"}
INK = "#163247"
MUTED = "#637787"
GRID = "#E3E9EE"
PAPER = "#FFFFFF"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "text.parse_math": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "path.simplify": True,
    "path.simplify_threshold": 0.3,
    "agg.path.chunksize": 10000,
    "savefig.facecolor": PAPER,
})


@dataclass
class Library:
    config: dict[str, Any]
    metrics: pd.DataFrame
    dates: np.ndarray
    equity: np.ndarray
    multiple_notionals: bool


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _record(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _clean_value(v) for k, v in row.items()}


def _safe_component(value: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', text):
        raise ValueError(f"Unsafe chart folder or filename: {value!r}")
    if text.endswith((".", " ")):
        raise ValueError(f"Unsafe chart folder or filename: {value!r}")
    return text


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _percent(value: Any) -> str:
    value = _number(value)
    return f"{value:.1%}" if math.isfinite(value) else "n/a"


def _ratio(value: Any) -> str:
    value = _number(value)
    return f"{value:.2f}" if math.isfinite(value) else "n/a"


def _notional_name(value: Any) -> str:
    return f"{_number(value, 1.0) * 100:g}% notional"


def _tenor_index(row: dict[str, Any]) -> int:
    label = str(row.get("tenor", ""))
    if label in TENORS:
        return TENORS.index(label)
    days = _number(row.get("tenor_days"))
    if days in TENOR_DAYS:
        return TENOR_DAYS.index(int(days))
    raise ValueError(f"Unknown tenor: {label!r}, {days!r}")


def _relative_folder(row: dict[str, Any], multiple_notionals: bool) -> Path:
    category = _safe_component(row["category"])
    raw_parts = str(row["folder"]).replace("\\", "/").split("/")
    parts = [_safe_component(part) for part in raw_parts]
    if category != "Both" and len(parts) != 1:
        raise ValueError(f"Nested folders are only supported for combined put strategies: {row['folder']}")
    folder = Path(*parts)
    if multiple_notionals:
        return Path(category) / _notional_name(row.get("notional_multiple", 1.0)) / folder
    return Path(category) / folder


def _individual_name(row: dict[str, Any]) -> str:
    order = _tenor_index(row)
    return f"{order + 1:02d} - {TENORS[order]}.png"


def load_library(data_dir: Path) -> Library:
    with (data_dir / "run.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    metrics = pd.read_csv(data_dir / "strategy_metrics.csv", dtype={"strategy_id": str, "folder": str}, low_memory=False)
    required = {
        "strategy_id", "category", "folder", "tenor", "tenor_days", "structure",
        "primary_pct", "secondary_pct", "width_pct", "cagr", "annualized_volatility", "daily_sharpe",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    if metrics.empty or metrics["strategy_id"].duplicated().any():
        raise ValueError("Metrics must contain unique strategy IDs and at least one strategy")
    if not set(metrics["category"]).issubset(CATEGORY_ORDER):
        raise ValueError("Unexpected strategy category")
    if "notional_multiple" not in metrics:
        metrics["notional_multiple"] = 1.0
    # Read legacy datasets during the folder/category migration without changing
    # accounting inputs; genuine combined-put portfolios have no SPX exposure.
    if "underlying" in metrics:
        legacy_overlay = metrics["category"].eq("Both") & pd.to_numeric(metrics["underlying"], errors="coerce").eq(1)
        metrics.loc[legacy_overlay, "category"] = "SPX with overlay"
    if not (data_dir / "curves.npz").exists() and config.get("best_hedge_selection"):
        from optimize_organized_spx_hedges import rebuild_selected_library
        dates, ids, equity = rebuild_selected_library(data_dir)
    else:
        with np.load(data_dir / "curves.npz", allow_pickle=False) as archive:
            dates = pd.to_datetime(archive["dates"]).to_numpy(dtype="datetime64[ns]")
            ids = archive["strategy_ids"].astype(str)
            equity = np.asarray(archive["equity"], dtype=float)
    if equity.shape != (len(dates), len(ids)):
        raise ValueError(f"Equity shape {equity.shape} does not match dates and strategy IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate strategy IDs in curves.npz")
    lookup = {key: pos for pos, key in enumerate(ids)}
    absent = set(metrics["strategy_id"]) - lookup.keys()
    if absent:
        raise ValueError(f"Metrics are missing equity curves for {len(absent)} strategies")
    order = [lookup[key] for key in metrics["strategy_id"]]
    equity = equity[:, order]
    if len(dates) < 2 or not np.all(dates[1:] > dates[:-1]):
        raise ValueError("Equity dates must be strictly increasing")
    if np.isinf(equity).any():
        raise ValueError("Equity curves may contain explicit missing values, but cannot contain infinity")
    if pd.Timestamp(dates[0]).date() != pd.Timestamp(config["start_date"]).date():
        raise ValueError("Curve start date differs from run.json")
    if pd.Timestamp(dates[-1]).date() != pd.Timestamp(config["end_date"]).date():
        raise ValueError("Curve end date differs from run.json")
    metrics["_curve_index"] = np.arange(len(metrics))
    metrics["_hedge_notional_cap"] = _number(config.get("hedge_notional_cap"))
    metrics["_missing_daily_values"] = np.isnan(equity).sum(axis=0)
    incomplete = metrics["_missing_daily_values"] > 0
    metrics.loc[incomplete, ["annualized_volatility", "daily_sharpe"]] = np.nan
    metrics.loc[~np.isfinite(equity[0]) | ~np.isfinite(equity[-1]), "cagr"] = np.nan
    metrics["_tenor_index"] = [_tenor_index(row) for row in metrics.to_dict("records")]
    metrics["_category_order"] = metrics["category"].map({name: i for i, name in enumerate(CATEGORY_ORDER)})
    multi = metrics["notional_multiple"].nunique() > 1
    return Library(config, metrics, dates, equity, multi)


def _period(config: dict[str, Any]) -> str:
    start = pd.Timestamp(config["start_date"]).strftime("%b %d, %Y")
    end = pd.Timestamp(config["end_date"]).strftime("%b %d, %Y")
    return f"{start} – {end}"


def _dollars(value: float, _position: int | None = None) -> str:
    if not math.isfinite(value):
        return "n/a"
    sign = "−" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}${value / 1e9:.3g}bn"
    if value >= 1e6:
        return f"{sign}${value / 1e6:.3g}m"
    if value >= 1e3:
        return f"{sign}${value / 1e3:.3g}k"
    return f"{sign}${value:g}"


def _limits(values: np.ndarray, initial: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    low = min(float(np.min(finite)), initial) if finite.size else initial
    high = max(float(np.max(finite)), initial) if finite.size else initial
    span = max(high - low, abs(initial) * 0.05, 1.0)
    lower = low - span * 0.08
    if low >= 0 and lower < 0:
        lower = 0.0
    return lower, high + span * 0.1


def _axis(fig: Figure, dates: np.ndarray, rect: tuple[float, float, float, float]) -> Any:
    ax = fig.add_axes(rect)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", length=0, pad=9, labelsize=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_dollars))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(mdates.date2num(dates[0]), mdates.date2num(dates[-1]))
    ax.set_ylabel("Portfolio value", fontsize=10, labelpad=15)
    return ax


def _rule(fig: Figure, y: float, left: float = 0.06, right: float = 0.95) -> None:
    fig.add_artist(Line2D([left, right], [y, y], transform=fig.transFigure, color=GRID, linewidth=0.8))


def _title_for(row: dict[str, Any]) -> str:
    title = row.get("title")
    if isinstance(title, str) and title.strip():
        title = title.strip()
        cap = _number(row.get("hedge_notional_cap"))
        if not math.isfinite(cap):
            cap = _number(row.get("_hedge_notional_cap"))
        if row.get("category") == "Both" and math.isfinite(cap) and "up to" not in title.lower() and "closest affordable buffer" not in title.lower():
            title = re.sub(r"(\d+(?:\.\d+)?)%\s*premium", r"up to \1% premium", title, flags=re.IGNORECASE)
        return title
    primary = _number(row.get("primary_pct"))
    secondary = _number(row.get("secondary_pct"))
    is_spread = math.isfinite(secondary) and _number(row.get("width_pct"), 0) > 0
    category = row["category"]
    if category == "SPX with overlay":
        return row["folder"].replace("SPX plus", "SPX +")
    side = "Sell" if category == "Put selling" else "Buy"
    if is_spread:
        other = "buy" if side == "Sell" else "sell"
        return f"{side} {primary:g}% put · {other} {secondary:g}% put"
    return f"{side} {primary:g}% put"


def _notional_text(config: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    value = _number(row.get("notional_multiple"), 1.0) if row else 1.0
    if row and row.get("category") == "SPX with overlay":
        return f"SPX: 100% of equity + options: {value * 100:g}% SPX notional at each roll; cash earns zero"
    if row and row.get("category") == "Both":
        fraction = _number(row.get("premium_fraction"))
        budget = f"{fraction * 100:g}%" if math.isfinite(fraction) else "the selected share"
        if config.get("dynamic_buffer_selection") and row.get("hedge_type") == "Downside buffer":
            return f"Short: {value * 100:g}% SPX notional; buffer spends exactly {budget} of net credit after costs; at least 1 buffer per short; cash earns zero"
        if config.get("dynamic_buffer_selection") and row.get("hedge_type") == "Mixed":
            return f"Short: {value * 100:g}% SPX notional; buffer spends {budget} of net credit; long put spends up to {budget}; cash earns zero"
        cap = _number(config.get("hedge_notional_cap"))
        if math.isfinite(cap):
            return f"Short spread: {value * 100:g}% SPX notional; hedge: up to {budget} of net credit after costs, capped at {cap * 100:g}% SPX notional; cash earns zero"
        return f"Short spread: {value * 100:g}% SPX notional; hedge spends {budget} of net spread credit after costs; cash earns zero"
    label = str(config.get("notional_label", "100% SPX notional at each roll; cash earns zero"))
    if row and value != 1.0:
        label = re.sub(r"100%", f"{value * 100:g}%", label, count=1)
    return label


def _footer(fig: Figure, config: dict[str, Any], row: dict[str, Any] | None = None, y: float = 0.105, notional_label: str | None = None) -> None:
    fig.text(0.06, y, f"Costs: {config.get('cost_label', 'Included in portfolio returns')}", fontsize=8.6, color=MUTED)
    sizing = notional_label or _notional_text(config, row)
    if config.get("best_hedge_selection"):
        sizing = sizing.replace("Short spread:", "Short strategy:").replace("net spread credit", "net short credit")
    fig.text(0.06, y - 0.027, sizing, fontsize=8.6, color=MUTED)
    fig.text(0.06, y - 0.054, "Historical simulation · Strikes are percentages of SPX at entry · Sharpe uses daily returns, annualized at 252 sessions", fontsize=8.1, color=MUTED)
    if config.get("best_hedge_selection") and (row is None or row.get("category") == "Both"):
        if config.get("dynamic_buffer_selection") and row and row.get("hedge_type") == "Downside buffer":
            selection = "At each roll: closest affordable buffer; prefer 5 points, allow 4 or 3 · Full budget spent · No Sharpe selection"
        elif config.get("dynamic_buffer_selection") and (row is None or row.get("hedge_type") == "Mixed"):
            selection = "Buffers: closest affordable 3–5 point protection at entry · Long puts: retrospective highest combined Sharpe"
        else:
            selection = "Best = highest full-period combined Sharpe among eligible candidates · Retrospective selection; not out-of-sample"
        fig.text(0.06, y - 0.077, selection, fontsize=8.1, color=MUTED)


MINIMUM_FREE_BYTES = 30 * 1024 * 1024


def _require_free_space(path: Path, reserve: int | None = None) -> None:
    reserve = MINIMUM_FREE_BYTES if reserve is None else reserve
    existing = path if path.is_dir() else path.parent
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < reserve:
        raise RuntimeError(f"Export stopped to preserve {reserve / 1024**2:.1f} MiB of free space; available {free / 1024**2:.1f} MiB")


def _save(fig: Figure, path: Path, dpi: int, palette_colors: int = 0, canvas_ready: bool = False, lossless_webp: bool = False) -> None:
    if lossless_webp:
        path = path.with_suffix(".webp")
    reserve = path.stat().st_size + 5 * 1024 * 1024 if path.is_file() else MINIMUM_FREE_BYTES
    _require_free_space(path, reserve)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial")
    try:
        if palette_colors or canvas_ready or lossless_webp:
            fig.set_dpi(dpi)
            if not canvas_ready:
                fig.canvas.draw()
            raster = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
            if palette_colors:
                raster = _chart_palette(raster, palette_colors)
            if lossless_webp:
                raster = raster.convert("RGB")
                raster.save(temporary, format="WEBP", lossless=True, method=1)
                with Image.open(temporary) as encoded:
                    if not np.array_equal(np.asarray(raster), np.asarray(encoded.convert("RGB"))):
                        raise ValueError("Lossless image encoding changed pixels")
            else:
                raster.save(temporary, format="PNG", optimize=True, dpi=(dpi, dpi))
        else:
            fig.savefig(temporary, format="png", dpi=dpi, pil_kwargs={"compress_level": 4})
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _chart_palette(raster: Image.Image, colors: int) -> Image.Image:
    # Rare legend swatches must retain their colors even when translucent curves
    # dominate the adaptive palette. Reserve chart colors before quantization.
    fixed = list(dict.fromkeys((*COLORS, *CATEGORY_COLORS.values(), INK, MUTED, GRID, PAPER)))
    rgb = raster.convert("RGB")
    adaptive = rgb.quantize(colors=colors - len(fixed), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    fixed_rgb = [tuple(int(value[i:i + 2], 16) for i in (1, 3, 5)) for value in fixed]
    entries = [channel for color in fixed_rgb for channel in color] + adaptive.getpalette()[:(colors - len(fixed)) * 3]
    palette = Image.new("P", (1, 1))
    palette.putpalette(entries + entries[-3:] * (256 - colors))
    quantized = rgb.quantize(palette=palette, dither=Image.Dither.NONE)
    # Pillow's color-cube approximation can otherwise shift even an exact
    # reserved source color by a few RGB units. Preserve solid swatch pixels.
    pixels = np.asarray(rgb).astype(np.uint32)
    packed = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]
    indices = np.asarray(quantized).copy()
    for index, value in enumerate(fixed):
        indices[packed == int(value[1:], 16)] = index
    result = Image.fromarray(indices)
    result.putpalette(palette.getpalette())
    return result


def _compact_one(task: tuple[str, int]) -> tuple[int, int, bool]:
    filename, colors = task
    path = Path(filename)
    before = path.stat().st_size
    _require_free_space(path, before + 5 * 1024 * 1024)
    with Image.open(path) as original:
        if original.mode == "P" and len(original.getcolors(maxcolors=256) or []) <= colors:
            return before, before, False
        dimensions = original.size
        resolution = original.info.get("dpi")
        raster = _chart_palette(original, colors)
    temporary = path.with_name("." + path.name + ".partial")
    try:
        raster.save(temporary, format="PNG", optimize=True, **({"dpi": resolution} if resolution else {}))
        with Image.open(temporary) as check:
            check.verify()
        if raster.size != dimensions:
            raise AssertionError("PNG compaction changed image dimensions")
        after = temporary.stat().st_size
        if after < before:
            os.replace(temporary, path)
            return before, after, True
        temporary.unlink()
        return before, before, False
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def compact_existing(output: Path, colors: int, workers: int) -> None:
    paths = sorted(output.rglob("*.png"))
    before_total = after_total = rewritten = 0
    started = time.monotonic()
    print(f"Compacting {len(paths):,} existing PNGs to {colors} colors; original-size + 5 MiB reserve", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_compact_one, (str(path), colors)) for path in paths]
        for i, future in enumerate(as_completed(futures), 1):
            before, after, changed = future.result()
            before_total += before
            after_total += after
            rewritten += changed
            if i % 200 == 0 or i == len(paths):
                print(f"Compacted {i:,}/{len(paths):,}; saved {(before_total - after_total) / 1024**2:.1f} MiB; elapsed {time.monotonic() - started:.0f}s", flush=True)
    print(f"Compaction complete: {rewritten:,} PNGs rewritten; saved {(before_total - after_total) / 1024**2:.1f} MiB; free {shutil.disk_usage(output).free / 1024**2:.1f} MiB", flush=True)


def _new_figure(dpi: int = 125) -> Figure:
    fig = Figure(figsize=(12.8, 8), dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def _render_group(task: tuple[Any, ...]) -> dict[str, Any]:
    rows, dates, equity, config, output_text, multi, dpi, overwrite = task
    output = Path(output_text)
    rows = sorted(rows, key=_tenor_index)
    folder = output / _relative_folder(rows[0], multi)
    rendered = 0
    initial = float(config.get("initial_equity", 1_000_000))
    xs = mdates.date2num(dates)
    fig = _new_figure(dpi)
    fig.text(0.06, 0.943, "SPX OPTIONS  /  " + rows[0]["category"].upper(), fontsize=9.5, color=MUTED, weight="bold")
    title = fig.text(0.06, 0.89, "", fontsize=21, color=INK, weight="bold")
    subtitle = fig.text(0.06, 0.85, "", fontsize=10.5, color=MUTED)
    _rule(fig, 0.824)
    metric_values = []
    for x, label in zip((0.06, 0.365, 0.67), ("CAGR", "ANNUALIZED VOLATILITY", "DAILY SHARPE · ANNUALIZED")):
        fig.text(x, 0.78, label, fontsize=9, color=MUTED, weight="bold")
        metric_values.append(fig.text(x, 0.729, "", fontsize=26, color=INK, weight="bold"))
    ax = _axis(fig, dates, (0.095, 0.215, 0.84, 0.452))
    line, = ax.plot(xs, equity[:, 0], color=COLORS[0], linewidth=1.6, zorder=3)
    ax.axhline(initial, color="#A4B2BE", linewidth=0.8, linestyle=(0, (3, 4)), zorder=1)
    endpoint, = ax.plot([xs[-1]], [equity[-1, 0]], marker="o", markersize=4, color=COLORS[0], zorder=4)
    status = fig.text(0.06, 0.16, "", fontsize=8.5, color=MUTED)
    _rule(fig, 0.14)
    _footer(fig, config, rows[0])
    for position, row in enumerate(rows):
        path = folder / _individual_name(row)
        existing_image = path.exists() or (config.get("_allow_webp_resume") and path.with_suffix(".webp").exists())
        if not config.get("_render_individuals", True) or (existing_image and not overwrite):
            continue
        curve = equity[:, position]
        color = COLORS[_tenor_index(row)]
        heading = _title_for(row)
        title.set_text(heading)
        title.set_fontsize(21 if len(heading) < 68 else 17)
        dte_low = _number(row.get("actual_dte_min"))
        dte_high = _number(row.get("actual_dte_max"))
        dte = f" · Actual entry: {dte_low:g}–{dte_high:g} days" if math.isfinite(dte_low) and math.isfinite(dte_high) else ""
        subtitle.set_text(f"{row['tenor']} target expiry{dte}  |  {_period(config)}")
        for artist, value in zip(metric_values, (_percent(row["cagr"]), _percent(row["annualized_volatility"]), _ratio(row["daily_sharpe"]))):
            artist.set_text(value)
        line.set_ydata(curve)
        line.set_color(color)
        endpoint.set_ydata([curve[-1]])
        endpoint.set_color(color)
        ax.set_ylim(*_limits(curve, initial))
        estimates = _number(row.get("estimated_quote_count", row.get("quote_estimates")), 0)
        note = f"Initial portfolio: {_dollars(initial)}  ·  Final portfolio: {_dollars(float(curve[-1]))}"
        if estimates > 0:
            note += f"  ·  {int(estimates):,} estimated option marks"
        skipped = _number(row.get("skipped_cycles", row.get("cash_cycles")), 0)
        if skipped > 0:
            note += f"  ·  {int(skipped):,} option rolls in cash"
        missing_values = int(_number(row.get("_missing_daily_values"), 0))
        if missing_values:
            note += f"  ·  {missing_values:,} daily values unavailable (gaps shown)"
        if config.get("dynamic_buffer_selection") and row.get("hedge_type") == "Downside buffer":
            low, high = _number(row.get("hedge_target_min")), _number(row.get("hedge_target_max"))
            if math.isfinite(low):
                note += f"  ·  Buffer entry targets: {low:g}–{high:g}%"
        coverage = str(row.get("coverage_status", ""))
        if coverage and coverage.lower() not in {"complete", "full", "ok", "nan", "none"} and not (skipped > 0 and "cash" in coverage.lower()):
            note += f"  ·  {coverage.replace('_', ' ')}"
        status.set_text(note)
        status.set_fontsize(8.5 if len(note) < 145 else 7.2)
        _save(fig, path, dpi, int(config.get("_png_palette_colors", 0)), lossless_webp=bool(config.get("_both_webp") and rows[0]["category"] == "Both"))
        rendered += 1
    fig.clear()
    overlay = folder / "All expiries.png"
    if overwrite or not (overlay.exists() or (config.get("_allow_webp_resume") and overlay.with_suffix(".webp").exists())):
        fig = _new_figure(dpi)
        fig.text(0.06, 0.943, "SPX OPTIONS  /  " + rows[0]["category"].upper(), fontsize=9.5, color=MUTED, weight="bold")
        heading = rows[0].get("comparison_title") or _title_for(rows[0])
        fig.text(0.06, 0.89, heading, fontsize=21 if len(heading) < 68 else 17, weight="bold")
        fig.text(0.06, 0.85, f"Expiry comparison  |  {_period(config)}  |  Initial portfolio: {_dollars(initial)}", fontsize=10.5, color=MUTED)
        incomplete = sum(_number(row.get("_missing_daily_values"), 0) > 0 for row in rows)
        if incomplete:
            fig.text(0.06, 0.809, f"{incomplete} of {len(rows)} expiry tests have unavailable daily marks; gaps are shown and daily metrics are n/a.", fontsize=8.6, color=MUTED)
        ax = _axis(fig, dates, (0.095, 0.42, 0.84, 0.366))
        ax.axhline(initial, color="#A4B2BE", linewidth=0.8, linestyle=(0, (3, 4)), zorder=1)
        for position, row in enumerate(rows):
            ax.plot(xs, equity[:, position], color=COLORS[_tenor_index(row)], linewidth=1.35, label=row["tenor"])
        ax.set_ylim(*_limits(equity, initial))
        _rule(fig, 0.367)
        best = config.get("best_hedge_selection") and rows[0]["category"] == "Both"
        xs_table = (0.08, 0.28, 0.52, 0.66, 0.82) if best else (0.08, 0.34, 0.55, 0.78)
        labels = ("TARGET EXPIRY", "SELECTED HEDGE", "CAGR", "ANN. VOL.", "SHARPE") if best else ("TARGET EXPIRY", "CAGR", "ANN. VOLATILITY", "DAILY SHARPE · ANN.")
        for x, label in zip(xs_table, labels):
            fig.text(x, 0.34, label, fontsize=8.8, color=MUTED, weight="bold")
        for i, row in enumerate(rows):
            y = 0.313 - i * 0.025
            color = COLORS[_tenor_index(row)]
            fig.add_artist(Line2D([0.062, 0.073], [y + 0.003, y + 0.003], transform=fig.transFigure, color=color, linewidth=2))
            values = (row["tenor"], _percent(row["cagr"]), _percent(row["annualized_volatility"]), _ratio(row["daily_sharpe"]))
            if best:
                if config.get("dynamic_buffer_selection") and row.get("hedge_type") == "Downside buffer":
                    low, high = _number(row.get("hedge_target_min")), _number(row.get("hedge_target_max"))
                    hedge = f"{low:g}–{high:g}% / 3–5 pt" if math.isfinite(low) else "No affordable buffer"
                else:
                    hedge = f"{row['hedge_primary_pct']:g} put" if not row["hedge_width_pct"] else f"{row['hedge_primary_pct']:g}/{row['hedge_secondary_pct']:g} buffer"
                values = (row["tenor"], hedge, *values[1:])
            for x, value in zip(xs_table, values):
                fig.text(x, y, value, fontsize=9.2, color=color if x == xs_table[0] else INK)
        _rule(fig, 0.14)
        _footer(fig, config, rows[0])
        _save(fig, overlay, dpi, int(config.get("_png_palette_colors", 0)), lossless_webp=bool(config.get("_both_webp") and rows[0]["category"] == "Both"))
        rendered += 1
        fig.clear()
    return {"folder": str(folder), "rendered": rendered, "expected": len(rows) + 1}


def _draw_overview(library: Library, rows: pd.DataFrame, path: Path, title: str, subtitle: str, color_by: str, dpi: int, overwrite: bool) -> int:
    if (path.exists() or (library.config.get("_allow_webp_resume") and path.with_suffix(".webp").exists())) and not overwrite:
        return 0
    config = library.config
    initial = float(config.get("initial_equity", 1_000_000))
    fig = _new_figure(dpi)
    fig.text(0.06, 0.943, "SPX OPTION RESEARCH", fontsize=9.5, color=MUTED, weight="bold")
    fig.text(0.06, 0.89, title, fontsize=23, weight="bold")
    fig.text(0.06, 0.85, f"{subtitle}  |  {_period(config)}", fontsize=10.5, color=MUTED)
    fig.text(0.06, 0.79, f"{len(rows):,} independent strategy curves", fontsize=13, weight="bold")
    fig.text(0.06, 0.759, "Each line shows a separately funded portfolio; all start with " + _dollars(initial) + ".", fontsize=9.5, color=MUTED)
    incomplete = int(rows["_missing_daily_values"].gt(0).sum())
    if incomplete:
        fig.text(0.06, 0.726, f"{incomplete:,} curves contain unavailable daily marks; gaps are shown.", fontsize=8.6, color=MUTED)
    ax = _axis(fig, library.dates, (0.095, 0.26, 0.84, 0.44))
    xs = mdates.date2num(library.dates)
    positions = rows["_curve_index"].to_numpy(int)
    low, high = initial, initial
    for start in range(0, len(positions), 128):
        values = library.equity[:, positions[start:start+128]]
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = min(low, float(finite.min())), max(high, float(finite.max()))
    labels = CATEGORY_ORDER if color_by == "category" else TENORS
    handles = []
    paint_groups = []
    for i, label in enumerate(labels):
        field = "category" if color_by == "category" else "tenor"
        match = rows[field].to_numpy() == label
        if not np.any(match):
            continue
        color = CATEGORY_COLORS[label] if color_by == "category" else COLORS[i]
        paint_groups.append((positions[match], color))
        handles.append(Line2D([], [], color=color, linewidth=2.2, label=f"{label} ({int(match.sum()):,})"))
    ax.axhline(initial, color="#8799A8", linewidth=0.75, linestyle=(0, (3, 4)), zorder=1)
    ax.set_ylim(*_limits(np.array([low, high]), initial))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.505, 0.197), ncol=min(4, len(handles)), frameon=False, fontsize=9.5, handlelength=2.0, columnspacing=2)
    _rule(fig, 0.14)
    row = _record(rows.iloc[0].to_dict()) if rows["category"].nunique() == 1 else None
    if row and row["category"] == "Both" and rows.hedge_type.nunique() > 1:
        row["hedge_type"] = "Mixed"
    notionals = sorted(rows["notional_multiple"].unique())
    mixed_label = None
    if len(notionals) > 1:
        sizes = " / ".join(f"{value * 100:g}%" for value in notionals)
        mixed_label = f"Options: {sizes} SPX notional at each roll; cash earns zero"
        if row and row["category"] == "SPX with overlay":
            mixed_label = "SPX: 100% of equity + " + mixed_label[0].lower() + mixed_label[1:]
    if row and row["category"] == "Both":
        budgets = sorted(pd.to_numeric(rows.get("premium_fraction", pd.Series(dtype=float)), errors="coerce").dropna().unique())
        if len(budgets) > 1:
            sizes = " / ".join(f"{value * 100:g}%" for value in budgets)
            cap = _number(config.get("hedge_notional_cap"))
            if math.isfinite(cap):
                mixed_label = f"Short spread: 100% SPX notional; hedge budgets: up to {sizes} of net credit after costs; hedge cap: {cap * 100:g}% SPX notional; cash earns zero"
            else:
                mixed_label = f"Short spread: 100% SPX notional; hedge budgets: {sizes} of net credit after costs; cash earns zero"
            if config.get("dynamic_buffer_selection"):
                mixed_label = f"Short: 100% SPX notional; budgets: {sizes} of net credit; buffer spends full allocation, long put up to allocation; cash earns zero"
    _footer(fig, config, row, notional_label=mixed_label)
    if row is None:
        fig.text(0.95, 0.131, "SPX with overlay = SPX + put strategy", ha="right", fontsize=8.2, color=MUTED)
    # Paint complete daily paths in bounded batches. Retaining thousands of
    # vector paths together can exhaust commit/disk space through the pagefile.
    fig.canvas.draw()
    for selected_positions, color in paint_groups:
        for start in range(0, len(selected_positions), 128):
            curves = library.equity[:, selected_positions[start:start+128]]
            segments = np.empty((curves.shape[1], len(xs), 2), dtype=float)
            segments[:, :, 0] = xs[None, :]
            segments[:, :, 1] = curves.T
            collection = LineCollection(segments, colors=[color], linewidths=0.65 if len(rows) > 300 else 0.9,
                alpha=0.28 if len(rows) > 300 else 0.48, zorder=2)
            ax.add_collection(collection, autolim=False)
            ax.draw_artist(collection)
            collection.remove()
    _save(fig, path, dpi, int(config.get("_png_palette_colors", 0)), canvas_ready=True,
          lossless_webp=bool(config.get("_both_webp") and rows.category.nunique() == 1 and rows.iloc[0].category == "Both"))
    fig.clear()
    return 1


def render_overviews(library: Library, output: Path, dpi: int, overwrite: bool, overwrite_categories: set[str] | None = None) -> int:
    force = overwrite_categories or set()
    # The root universe changes whenever a category is added or renamed.
    rendered = _draw_overview(library, library.metrics, output / "All strategies.png", "All strategy categories", "Put selling · Put buying · Combined puts · SPX with overlay", "category", dpi, True)
    for category in CATEGORY_ORDER:
        frame = library.metrics.loc[library.metrics["category"] == category]
        if frame.empty:
            continue
        subtitle = "All strikes, structures and target expiries"
        category_overwrite = overwrite or category in force
        rendered += _draw_overview(library, frame, output / category / "All strategies.png", category, subtitle, "tenor", dpi, category_overwrite)
        for i, tenor in enumerate(TENORS):
            selected = frame.loc[frame["_tenor_index"] == i]
            if selected.empty:
                continue
            rendered += _draw_overview(library, selected, output / category / f"{i + 1:02d} - {tenor} comparison.png", f"{category} · {tenor}", "All strikes and structures at this target expiry", "tenor", dpi, category_overwrite)
    for relative, frame, depth in _intermediate_groups(library):
        parts = str(frame.iloc[0]["folder"]).replace("\\", "/").split("/")
        title = "Short spread " + parts[0]
        if depth > 1:
            title += " · " + parts[1]
        subtitle = "Both · All hedge structures, premium budgets and expiries" if depth == 1 else "Both · All premium budgets and expiries"
        if library.config.get("best_hedge_selection"):
            title = parts[0] if depth == 1 else parts[0] + " · Sell " + parts[1].replace("-", "/")
            subtitle = "Both · Best long puts and buffers across short variations and expiries" if depth == 1 else "Both · Best long put and best buffer at each expiry"
        rendered += _draw_overview(library, frame, output / relative / "All strategies.png", title, subtitle, "tenor", dpi, overwrite or "Both" in force)
    return rendered


def _intermediate_groups(library: Library) -> list[tuple[Path, pd.DataFrame, int]]:
    groups: dict[tuple[str, int], list[int]] = {}
    for idx, row in library.metrics.loc[library.metrics["category"].eq("Both")].iterrows():
        relative = _relative_folder(row.to_dict(), library.multiple_notionals)
        folder_depth = len(str(row["folder"]).replace("\\", "/").split("/"))
        if folder_depth != 3:
            raise ValueError(f"Combined strategy folders need exactly three levels: {row['folder']}")
        for depth in (1, 2):
            ancestor = relative.parents[folder_depth - depth - 1]
            groups.setdefault((ancestor.as_posix(), depth), []).append(idx)
    return [(Path(path), library.metrics.loc[indices], depth) for (path, depth), indices in sorted(groups.items())]


def _browser_rows(library: Library) -> list[dict[str, Any]]:
    records = []
    frame = library.metrics.sort_values(["_category_order", "notional_multiple", "primary_pct", "width_pct", "folder", "_tenor_index"], ascending=[True, True, False, True, True, True])
    for raw in frame.to_dict("records"):
        row = _record(raw)
        relative = _relative_folder(row, library.multiple_notionals)
        order = _tenor_index(row)
        best = bool(library.config.get("best_hedge_selection")) and row["category"] == "Both"
        records.append({
            "id": row["strategy_id"], "category": row["category"], "folder": row["folder"],
            "title": _title_for(row), "tenor": TENORS[order], "tenorOrder": order,
            "structure": row["structure"], "primary": row["primary_pct"], "secondary": row["secondary_pct"],
            "width": row["width_pct"], "notional": row["notional_multiple"],
            "premium": row.get("premium_fraction"), "hedgeType": row.get("hedge_type"),
            "hedgeStrike": row.get("hedge_primary_pct"), "hedgeWidth": row.get("hedge_width_pct"),
            "cagr": row["cagr"], "vol": row["annualized_volatility"], "sharpe": row["daily_sharpe"],
            "path": (relative / _individual_name(row)).as_posix(),
            "expiryPath": (relative / "All expiries.png").as_posix(),
            "tenorPath": (Path(row["category"]) / f"{order + 1:02d} - {TENORS[order]} comparison.png").as_posix(),
            "categoryPath": (Path(row["category"]) / "All strategies.png").as_posix(),
            "basePath": ((relative.parent if best else relative.parents[1]) / "All strategies.png").as_posix() if row["category"] == "Both" else None,
            "hedgePath": ((relative.parents[1] if best else relative.parent) / "All strategies.png").as_posix() if row["category"] == "Both" else None,
            "bestHedge": best, "comparisonTitle": row.get("comparison_title") if best else None,
            "baseTitle": f"Sell {row.get('short_variation')} · {row.get('premium_fraction', 0) * 100:g}% premium" if best else None,
        })
    return records


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPX option strategy library</title>
<style>
:root{--ink:#163247;--muted:#637787;--line:#dfe7ed;--blue:#24658b;--bg:#f3f6f8;--white:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;font-size:14px}
button,input,select{font:inherit}button,a,select{touch-action:manipulation}button,a{cursor:pointer}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #83bfdc;outline-offset:2px}
header{padding:25px 34px 20px;background:#fff;border-bottom:1px solid var(--line)}.eyebrow{font-size:11px;letter-spacing:.13em;font-weight:700;color:var(--muted)}h1{font-size:27px;letter-spacing:-.6px;margin:8px 0 9px;font-weight:650}.intro{color:var(--muted);font-size:13px;line-height:1.65;margin:0}.counts{color:var(--ink);font-weight:600}.document-links{display:flex;gap:18px;margin-top:10px;font-size:12px}.document-links a{color:var(--blue);text-decoration:none}.document-links a:hover{text-decoration:underline}
.workspace{display:grid;grid-template-columns:310px minmax(0,1fr);max-width:1900px;margin:0 auto}.sidebar{background:#fff;border-right:1px solid var(--line);padding:24px 20px;align-self:start;position:sticky;top:0;height:calc(100vh - 157px);min-height:610px;display:flex;flex-direction:column}.filter-grid{display:grid;grid-template-columns:1fr 1fr;gap:11px}.wide{grid-column:1/-1}label{display:block;font-size:11px;font-weight:650;color:var(--muted);margin-bottom:5px;text-transform:uppercase;letter-spacing:.035em}select,input{width:100%;height:36px;border:1px solid #cdd9e1;border-radius:5px;color:var(--ink);background:#fff;padding:0 8px}input::placeholder{color:#8898a4}.search{margin-top:13px}.result-heading{display:flex;align-items:center;justify-content:space-between;margin:21px 0 8px;font-size:12px}.text-button{border:0;background:none;color:var(--blue);padding:0;font-size:12px}.results{overflow-y:auto;min-height:110px;border-top:1px solid var(--line);flex:1}.result{display:block;width:100%;border:0;border-bottom:1px solid #edf1f4;background:#fff;text-align:left;padding:11px 9px;line-height:1.45;color:var(--ink);border-left:3px solid transparent}.result:hover{background:#f2f7fa}.result.active{border-left-color:var(--blue);background:#eaf3f8}.result strong{font-size:12px;font-weight:650;display:block}.result span{font-size:11px;color:var(--muted)}.empty-list{color:var(--muted);padding:18px 8px;font-size:12px;line-height:1.5}.main{padding:24px 27px 30px;min-width:0}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:15px;margin-bottom:15px;flex-wrap:wrap}.tabs{display:flex;gap:4px;background:#e7eef3;padding:4px;border-radius:7px;flex-wrap:wrap}.tab{border:0;background:transparent;padding:8px 11px;border-radius:4px;color:var(--muted);font-size:12px;white-space:nowrap}.tab[aria-selected=true]{background:#fff;box-shadow:0 1px 4px #18394f17;color:var(--ink);font-weight:650}.actions{display:flex;align-items:center;gap:12px;font-size:12px}.actions a{color:var(--blue);text-decoration:none}.nav-button{background:#fff;border:1px solid #cdd9e1;border-radius:4px;padding:5px 9px;color:var(--ink)}.nav-button:disabled{opacity:.35;cursor:default}.chart-panel{background:#fff;border:1px solid var(--line);border-radius:8px;box-shadow:0 2px 8px #18394f05;overflow:hidden;min-height:260px}.chart-panel img{display:block;width:100%;height:auto}.caption{display:flex;gap:12px;justify-content:space-between;align-items:center;padding:13px 17px;border-top:1px solid var(--line);font-size:12px;line-height:1.5}.caption-title{font-weight:600}.caption-note{color:var(--muted)}.metric-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:15px}.metric{background:#fff;border:1px solid var(--line);border-radius:7px;padding:15px 18px}.metric-label{font-size:10px;letter-spacing:.07em;font-weight:650;color:var(--muted)}.metric-value{font-size:24px;font-weight:650;margin-top:5px}.context{margin:20px 3px 0;color:var(--muted);font-size:12px;line-height:1.8;max-width:1100px}.context strong{font-weight:600;color:var(--ink)}.error{padding:50px;color:var(--muted);text-align:center}.hidden{display:none!important}footer{padding:18px 34px;color:var(--muted);font-size:11px;border-top:1px solid var(--line);background:#fff}
@media(min-width:1550px){.chart-panel{max-width:1350px}.sidebar{height:calc(100vh - 149px)}}@media(max-width:1020px){.workspace{grid-template-columns:260px minmax(0,1fr)}.sidebar{padding:18px 14px}.main{padding:18px 16px}.tab{padding:7px 8px}.metric{padding:12px}.metric-value{font-size:21px}}@media(max-width:760px){header{padding:20px}.workspace{display:block}.sidebar{position:static;height:auto;min-height:0;border-right:0;border-bottom:1px solid var(--line)}.filter-grid{grid-template-columns:repeat(3,1fr)}.wide{grid-column:auto}.results{max-height:160px}.result-heading{margin-top:15px}.main{padding:16px 10px}.caption{align-items:start;flex-direction:column;gap:3px}.metric-strip{gap:7px}.metric{padding:12px 10px}.metric-label{font-size:9px}.metric-value{font-size:20px}.context{padding:0 7px}footer{padding:18px 20px}}
</style></head><body>
<header><div class="eyebrow">SPX OPTION RESEARCH</div><h1>A library built for comparison</h1><p class="intro"><span class="counts">__COUNT__ strategies · 7 target expiries</span> &nbsp;|&nbsp; __PERIOD__<br>Explore a single strategy, compare its expiries, or see the complete strategy universe.</p><div class="document-links"><a href="../SPX%20Research%20Interactive/index.html">Interactive comparison ↗</a><a href="Methodology%20and%20coverage.md">Methodology and coverage</a><a href="All%20strategy%20metrics.csv" download>All strategy metrics</a></div></header>
<div class="workspace"><aside class="sidebar" aria-label="Strategy filters"><div class="filter-grid">
<div class="wide"><label for="category">Strategy category</label><select id="category"></select></div>
<div><label for="primary">Primary strike</label><select id="primary"></select></div><div><label for="width">Spread width</label><select id="width"></select></div>
<div><label for="tenor">Target expiry</label><select id="tenor"></select></div><div><label for="structure">Position</label><select id="structure"></select></div>
<div class="combination-filter hidden"><label for="premium">Premium budget</label><select id="premium"></select></div><div class="combination-filter hidden"><label for="hedgeType">Hedge type</label><select id="hedgeType"></select></div>
<div class="combination-filter hidden"><label for="hedgeStrike">Hedge strike</label><select id="hedgeStrike"></select></div><div class="combination-filter hidden"><label for="hedgeWidth">Hedge width</label><select id="hedgeWidth"></select></div>
<div id="notional-wrap" class="wide hidden"><label for="notional">Option notional</label><select id="notional"></select></div>
</div><div class="search"><label for="search">Find a strategy</label><input id="search" type="search" placeholder="e.g. 103-100, long, short" autocomplete="off"></div><div class="result-heading"><span id="result-count" aria-live="polite"></span><button class="text-button" id="reset">Reset filters</button></div><div id="results" class="results" aria-label="Matching strategies"></div></aside>
<main class="main"><div class="toolbar"><div class="tabs" role="tablist" aria-label="Comparison view"><button class="tab" role="tab" data-view="single" aria-selected="true">Individual</button><button class="tab" role="tab" data-view="expiries" aria-selected="false">All expiries</button><button class="tab combination-view hidden" role="tab" data-view="hedge" aria-selected="false">Hedge budgets</button><button class="tab combination-view hidden" role="tab" data-view="base" aria-selected="false">Short spread</button><button class="tab" role="tab" data-view="tenor" aria-selected="false">Same expiry</button><button class="tab" role="tab" data-view="category" aria-selected="false">Category</button><button class="tab" role="tab" data-view="all" aria-selected="false">All strategies</button></div><div class="actions"><button id="previous" class="nav-button" aria-label="Previous strategy">←</button><button id="next" class="nav-button" aria-label="Next strategy">→</button><a id="open-image" target="_blank" rel="noopener">Open PNG</a><a id="save-image" download>Save PNG</a></div></div>
<div class="chart-panel"><img id="chart" alt="Strategy equity curve"><div id="image-error" class="error hidden">This chart image is unavailable. Keep the category folders next to this index file.</div><div class="caption"><span id="caption-title" class="caption-title"></span><span id="caption-note" class="caption-note"></span></div></div>
<div class="metric-strip" id="metrics"><div class="metric"><div class="metric-label">CAGR</div><div id="cagr" class="metric-value"></div></div><div class="metric"><div class="metric-label">ANNUALIZED VOLATILITY</div><div id="vol" class="metric-value"></div></div><div class="metric"><div class="metric-label">DAILY SHARPE · ANNUALIZED</div><div id="sharpe" class="metric-value"></div></div></div>
<div class="context"><strong>Reading the charts.</strong> Put selling and Put buying show standalone option strategies. Both combines a short put spread with a long put or long put spread funded from the short spread's net premium. SPX with overlay adds a 100% SPX position to a put strategy. Strike labels are percentages of SPX at entry; a 103–100 spread uses strikes near 103% and 100% of SPX. Spread widths are percentage points.<br><strong>Portfolio assumptions.</strong> __NOTIONAL__. __HEDGE_SIZING__ SPX with overlay holds 100% SPX exposure plus the stated option notional, rebalanced at each roll. Positions remain fixed between rolls. SPX exposure uses the price index with no dividends or financing return; cash earns zero. __COSTS__. Sharpe uses daily portfolio returns and is annualized at 252 sessions. Initial portfolio: __INITIAL__.<br><strong>Historical research.</strong> Target expiries may differ from actual available expiries; individual images state the observed range. Unavailable or unrepresentable positions stay in cash for that roll. Both also skips a roll when the short spread has no positive net credit or the hedge has no positive cost. Estimated entry quotes are excluded. Estimated option marks and cash rolls, where used, are identified on individual charts. Unavailable daily values remain visible as gaps; daily volatility and Sharpe are unavailable for incomplete daily curves. All comparison lines represent separate portfolios.</div>
</main></div><footer>Static PNG charts · This browser works offline · __PERIOD__</footer>
<script id="library-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('library-data').textContent),byId=new Map(data.map(r=>[r.id,r]));
const $=id=>document.getElementById(id),filters=['category','primary','width','tenor','structure','notional','premium','hedgeType','hedgeStrike','hedgeWidth'];
const preferred=data.find(r=>r.bestHedge&&r.primary===98&&r.width===3&&r.premium===0.1&&r.tenorOrder===1&&r.hedgeType==='Long put');
const state={selected:preferred?.id||(byId.has('combo_98-95_h95-85_p25_d07')?'combo_98-95_h95-85_p25_d07':byId.has('short_99-96_d07')?'short_99-96_d07':data[0]?.id),view:'single',filtered:data};
const unique=(field)=>[...new Set(data.map(r=>r[field]))].filter(v=>v!==null&&v!==undefined);
const multiNotional=unique('notional').length>1;
function options(id,values,label,format=v=>v){const el=$(id);el.replaceChildren(new Option(label,''));values.forEach(v=>el.add(new Option(format(v),String(v))));}
options('category',['Put selling','Put buying','Both','SPX with overlay'].filter(v=>unique('category').includes(v)),'All categories');
options('primary',unique('primary').sort((a,b)=>b-a),'All strikes',v=>v+'% of SPX');
options('width',unique('width').sort((a,b)=>a-b),'All widths',v=>Number(v)===0?'Single put':v+' points');
options('tenor',[...new Map(data.map(r=>[r.tenorOrder,r.tenor])).entries()].sort((a,b)=>a[0]-b[0]).map(v=>v[1]),'All expiries');
options('structure',unique('structure').sort(),'All positions',v=>String(v).replaceAll('_',' '));
options('notional',unique('notional').sort((a,b)=>a-b),'All notionals',v=>(v*100)+'% of SPX');
options('premium',unique('premium').sort((a,b)=>a-b),'All budgets',v=>(v*100)+'%');
options('hedgeType',unique('hedgeType').sort(),'All hedges',v=>String(v).replaceAll('_',' '));
options('hedgeStrike',unique('hedgeStrike').sort((a,b)=>b-a),'All strikes',v=>v+'% of SPX');
options('hedgeWidth',unique('hedgeWidth').sort((a,b)=>a-b),'All widths',v=>Number(v)===0?'Long put':v+' points');
const hasCombinations=data.some(r=>r.category==='Both');
const hasBestHedges=data.some(r=>r.bestHedge);
if(hasBestHedges){document.querySelector('[data-view="hedge"]').textContent='Premium budget';document.querySelector('[data-view="base"]').textContent='Short variation';}
if(hasCombinations)document.querySelectorAll('.combination-filter').forEach(el=>el.classList.remove('hidden'));
if(multiNotional)$('notional-wrap').classList.remove('hidden');
function pct(v){return typeof v==='number'&&Number.isFinite(v)?(v*100).toFixed(1)+'%':'n/a';}
function ratio(v){return typeof v==='number'&&Number.isFinite(v)?v.toFixed(2):'n/a';}
function fileUrl(path){return path.split('/').map(encodeURIComponent).join('/');}
function syncCombinationFilters(){const active=hasCombinations&&(!$('category').value||$('category').value==='Both');document.querySelectorAll('.combination-filter').forEach(el=>el.classList.toggle('hidden',!active));if(!active)['premium','hedgeType','hedgeStrike','hedgeWidth'].forEach(id=>$(id).value='');}
function filterRows(){syncCombinationFilters();const q=$('search').value.toLowerCase().trim().split(/\s+/).filter(Boolean);state.filtered=data.filter(r=>filters.every(k=>!$(k).value||String(r[k])===$(k).value)&&q.every(term=>(r.title+' '+r.folder+' '+r.category+' '+r.tenor+' '+r.structure).toLowerCase().includes(term)));if(!state.filtered.some(r=>r.id===state.selected))state.selected=state.filtered[0]?.id;renderList();renderChart();}
function renderList(){const fragment=document.createDocumentFragment();state.filtered.forEach(r=>{const button=document.createElement('button');button.className='result'+(r.id===state.selected?' active':'');button.dataset.id=r.id;button.setAttribute('aria-pressed',String(r.id===state.selected));const title=document.createElement('strong');title.textContent=r.title;const detail=document.createElement('span');detail.textContent=r.category+' · '+r.tenor+(multiNotional?' · '+(r.notional*100)+'%':'');button.append(title,detail);button.addEventListener('click',()=>select(r.id));fragment.append(button);});$('results').replaceChildren(fragment);$('result-count').textContent=state.filtered.length.toLocaleString()+' matching strategies';if(!state.filtered.length){const p=document.createElement('p');p.className='empty-list';p.textContent='No strategies match these filters. Try another strike, width, or search.';$('results').append(p);}}
function select(id){state.selected=id;document.querySelectorAll('.result').forEach(el=>{const active=el.dataset.id===id;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active));});renderChart();}
function renderChart(){
const r=byId.get(state.selected),has=Boolean(r),combined=has&&r.category==='Both';
document.querySelectorAll('.combination-view').forEach(el=>el.classList.toggle('hidden',!combined));
if(!combined&&['base','hedge'].includes(state.view))state.view='expiries';
$('metrics').classList.toggle('hidden',state.view!=='single'||!has);
document.querySelectorAll('.tab').forEach(el=>el.setAttribute('aria-selected',String(el.dataset.view===state.view)));
if(!has&&state.view!=='all'){$('chart').classList.add('hidden');$('image-error').textContent='No matching strategies. Clear one or more filters to view a chart.';$('image-error').classList.remove('hidden');$('caption-title').textContent='No matching strategies';$('caption-note').textContent='';$('open-image').removeAttribute('href');$('save-image').removeAttribute('href');$('previous').disabled=true;$('next').disabled=true;return;}
let path='All strategies.png',title='All strategy categories',note=data.length.toLocaleString()+' independent strategies';
if(r){
if(state.view==='single'){path=r.path;title=r.title;note=r.category+' · '+r.tenor;}
else if(state.view==='expiries'){path=r.expiryPath;title=r.comparisonTitle||r.title;note=r.bestHedge?'Best hedge selected separately for each expiry':'All target expiries';}
else if(state.view==='base'){path=r.basePath;title=r.bestHedge?r.baseTitle:'Short spread '+r.folder.split('/')[0];note=r.bestHedge?'Best long put and buffer at each expiry':'All hedges, premium budgets and expiries';}
else if(state.view==='hedge'){path=r.hedgePath;title=r.bestHedge?r.folder.split('/')[0]:'Short spread '+r.folder.split('/')[0]+' · '+r.folder.split('/')[1];note=r.bestHedge?'All short variations and best hedges at this budget':'All premium budgets and expiries';}
else if(state.view==='tenor'){path=r.tenorPath;title=r.category+' · '+r.tenor;note='All strikes and structures';}
else if(state.view==='category'){path=r.categoryPath;title=r.category;note='All strikes, structures and expiries';}}
const source=fileUrl(path);$('chart').classList.remove('hidden');$('image-error').classList.add('hidden');$('image-error').textContent='This chart image is unavailable. Keep the category folders next to this index file.';
if($('chart').getAttribute('src')!==source)$('chart').src=source;
$('chart').alt=title+' — historical portfolio equity';$('open-image').href=source;$('save-image').href=source;$('save-image').download=path.replaceAll('/',' - ');$('caption-title').textContent=title;$('caption-note').textContent=note;
if(r){$('cagr').textContent=pct(r.cagr);$('vol').textContent=pct(r.vol);$('sharpe').textContent=ratio(r.sharpe);}
const i=state.filtered.findIndex(v=>v.id===state.selected);$('previous').disabled=i<=0;$('next').disabled=i<0||i>=state.filtered.length-1;if(r)history.replaceState(null,'','#'+encodeURIComponent(r.id)+'/'+state.view);
}
function move(delta){const i=state.filtered.findIndex(r=>r.id===state.selected),r=state.filtered[i+delta];if(r){select(r.id);document.querySelector('.result.active')?.scrollIntoView({block:'nearest'});}}
filters.forEach(id=>$(id).addEventListener('change',filterRows));$('search').addEventListener('input',filterRows);$('reset').addEventListener('click',()=>{filters.forEach(id=>$(id).value='');$('search').value='';filterRows();});$('previous').addEventListener('click',()=>move(-1));$('next').addEventListener('click',()=>move(1));document.querySelectorAll('.tab').forEach(el=>el.addEventListener('click',()=>{state.view=el.dataset.view;renderChart();}));$('chart').addEventListener('error',()=>{$('chart').classList.add('hidden');$('image-error').classList.remove('hidden');});document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight'){move(1);e.preventDefault();}else if(e.key==='ArrowLeft'){move(-1);e.preventDefault();}});
if(location.hash){const [id,view]=location.hash.slice(1).split('/');try{const decoded=decodeURIComponent(id);if(byId.has(decoded))state.selected=decoded;if(['single','expiries','hedge','base','tenor','category','all'].includes(view))state.view=view;}catch{}}
renderList();renderChart();
const currentButton=$('results').querySelector('.active');if(currentButton)$('results').scrollTop=currentButton.offsetTop-$('results').offsetTop;
</script></body></html>'''


def write_browser(library: Library, output: Path) -> None:
    rows = _browser_rows(library)
    payload = json.dumps(rows, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    page = HTML_TEMPLATE
    if library.config.get("best_hedge_selection"):
        page = page.replace('Both combines a short put spread', 'Both combines a short put or short put spread')
        page = page.replace("the short spread's net premium", "the short strategy's net premium")
        if library.config.get("dynamic_buffer_selection"):
            selection_text = ('<strong>Buffer selection.</strong> Each roll buys the closest affordable buffer to the money using entry quotes only. At the same long-put target, prefer 5 points wide, then 4, then 3. Actual listed width must cover at least 3% of entry SPX. At least one buffer per short contract is required; fractional additional buffers spend the full allocation. Sharpe is reported but never used to select buffers. A roll stays in cash if no valid buffer is affordable. Entry strikes vary over time.<br><strong>Long-put selection.</strong> Outright long puts retain the previously selected highest full-period combined Sharpe; these are retrospective winners, not out-of-sample results.<br>')
        else:
            selection_text = '<strong>Historical hedge selection.</strong> Best means the highest combined-strategy daily Sharpe over the full research period. These are retrospective winners, not out-of-sample results.<br>'
        page = page.replace('<strong>Historical research.</strong>', selection_text+'<strong>Historical research.</strong>')
        page = page.replace('<a href="All%20strategy%20metrics.csv" download>All strategy metrics</a>', '<a href="All%20strategy%20metrics.csv" download>All strategy metrics</a><a href="Both/Selected%20hedges.csv" download>Selected hedges</a>')
    notional_label = str(library.config.get("notional_label", "100% SPX notional at each roll; cash earns zero")).rstrip(".")
    if library.multiple_notionals:
        sizes = " or ".join(f"{value * 100:g}%" for value in sorted(library.metrics["notional_multiple"].unique()))
        notional_label = f"Option notional is {sizes} of equity at each roll, as selected; cash earns zero"
    cap = _number(library.config.get("hedge_notional_cap"))
    if math.isfinite(cap):
        hedge_sizing = f"In Both, the short spread has 100% SPX notional. The hedge spends up to the selected share of net short-spread credit after execution costs, subject to a {cap * 100:g}% SPX-notional hedge cap. Hedge purchase costs count toward the premium budget, and any unspent budget remains cash."
    else:
        hedge_sizing = "In Both, the short spread has 100% SPX notional and the hedge quantity varies to spend the selected share of net short-spread credit after execution costs; the hedge purchase also includes costs."
    if library.config.get("best_hedge_selection"):
        hedge_sizing = hedge_sizing.replace("short spread", "short strategy").replace("short-spread", "short-strategy")
    if library.config.get("dynamic_buffer_selection"):
        hedge_sizing = "In Both, the short strategy has 100% SPX notional. Buffers spend exactly 5%, 10%, 15% or 20% of net short credit after all entry costs on each traded roll, with no upper quantity cap. Long puts retain the 100% hedge-notional cap and can spend less than their allocation."
    replacements = {
        "__COUNT__": f"{len(rows):,}", "__PERIOD__": html.escape(_period(library.config)),
        "__NOTIONAL__": html.escape(notional_label),
        "__HEDGE_SIZING__": html.escape(hedge_sizing),
        "__COSTS__": html.escape("Trading costs: " + str(library.config.get("cost_label", "Included in returns")).rstrip(".")),
        "__INITIAL__": html.escape(_dollars(float(library.config.get("initial_equity", 1_000_000)))),
        "__DATA__": payload,
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    if library.config.get("dynamic_buffer_selection"):
        missing_buffers = any(row["category"] == "Both" and row["hedgeType"] == "Downside buffer"
            and not (output/row["path"]).with_suffix(".webp").exists() for row in rows)
        if missing_buffers:
            page = page.replace('</header>', '</header><div style="padding:12px 34px;background:#fff3d8;color:#664b1c;font-size:13px">Buffer calculations are updated. The image export is incomplete while disk space is unavailable. The 20% premium / 98–95 buffer example is available for all seven expirations.</div>', 1)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "index.html"
    _require_free_space(path)
    temporary = output / ".index.html.partial"
    try:
        temporary.write_text(page, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _tasks(library: Library, output: Path, dpi: int, overwrite: bool, limit_groups: int, overwrite_categories: set[str] | None = None) -> list[tuple[Any, ...]]:
    tasks = []
    group_fields = ["_category_order", "category", "notional_multiple", "folder"]
    for _, frame in library.metrics.groupby(group_fields, sort=True, dropna=False):
        frame = frame.sort_values("_tenor_index")
        if frame["_tenor_index"].duplicated().any():
            raise ValueError("A moneyness/notional folder contains duplicate expiries")
        if set(frame["_tenor_index"]) != set(range(len(TENORS))):
            raise ValueError(f"Each strategy folder must contain all seven expiries: {frame.iloc[0]['folder']}")
        rows = [_record(row) for row in frame.to_dict("records")]
        positions = frame["_curve_index"].to_numpy(int)
        force = overwrite or rows[0]["category"] in (overwrite_categories or set())
        tasks.append((rows, library.dates, library.equity[:, positions], library.config, str(output), library.multiple_notionals, dpi, force))
        if limit_groups and len(tasks) >= limit_groups:
            break
    if overwrite_categories:
        tasks.sort(key=lambda task: task[0][0]["category"] not in overwrite_categories)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=125)
    parser.add_argument("--png-colors", type=int, choices=(0, 128, 256), default=0, help="Compact PNG palette size; zero preserves RGB output")
    parser.add_argument("--compact-existing", action="store_true", help="Compact existing PNG encodings without rerendering their content")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing PNGs; by default resume missing files")
    parser.add_argument("--overwrite-category", action="append", choices=CATEGORY_ORDER, default=[], help="Regenerate a renamed or changed category while retaining other existing category PNGs")
    parser.add_argument("--overwrite-overview-category", action="append", choices=CATEGORY_ORDER, default=[], help="Regenerate only category and expiry comparisons for the selected category")
    parser.add_argument("--only-index", action="store_true", help="Refresh the offline browser without rendering images")
    parser.add_argument("--skip-overviews", action="store_true")
    parser.add_argument("--limit-groups", type=int, default=0, help="Render only the first N folders for a preview")
    args = parser.parse_args()
    if args.workers < 1 or args.dpi < 50 or args.limit_groups < 0:
        parser.error("Workers must be positive, DPI at least 50, and group limit nonnegative")
    started = time.monotonic()
    if args.compact_existing:
        if not args.png_colors:
            parser.error("--compact-existing requires --png-colors 128 or 256")
        compact_existing(args.output_dir, args.png_colors, args.workers)
        return
    library = load_library(args.data_dir)
    library.config["_png_palette_colors"] = args.png_colors
    force_categories = set(args.overwrite_category)
    tasks = _tasks(library, args.output_dir, args.dpi, args.overwrite, args.limit_groups, force_categories)
    if args.limit_groups:
        chosen_ids = {row["strategy_id"] for task in tasks for row in task[0]}
        library.metrics = library.metrics.loc[library.metrics["strategy_id"].isin(chosen_ids)].copy()
    if args.only_index:
        write_browser(library, args.output_dir)
        print(f"Updated {args.output_dir / 'index.html'}", flush=True)
        return
    print(f"Rendering {len(tasks):,} strategy folders, {len(library.metrics):,} individual curves, {args.workers} workers", flush=True)
    rendered = 0
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            rendered += _render_group(task)["rendered"]
            if i % 50 == 0 or i == len(tasks):
                print(f"Folders {i:,}/{len(tasks):,}; PNGs written {rendered:,}; elapsed {time.monotonic() - started:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_render_group, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                rendered += future.result()["rendered"]
                if i % 50 == 0 or i == len(tasks):
                    print(f"Folders {i:,}/{len(tasks):,}; PNGs written {rendered:,}; elapsed {time.monotonic() - started:.0f}s", flush=True)
    if not args.skip_overviews:
        overview_categories = force_categories | set(args.overwrite_overview_category)
        rendered += render_overviews(library, args.output_dir, args.dpi, args.overwrite, overview_categories)
    write_browser(library, args.output_dir)
    expected = len(library.metrics) + len(tasks) + (0 if args.skip_overviews else 1 + library.metrics["category"].nunique() + library.metrics.groupby(["category", "tenor"]).ngroups + len(_intermediate_groups(library)))
    print(f"Complete: {rendered:,} PNGs written; {expected:,} expected PNGs in this export; {time.monotonic() - started:.0f}s. Browser: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
