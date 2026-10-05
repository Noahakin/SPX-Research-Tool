from __future__ import annotations

import calendar
import math
import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
VIX_ROOT = WORKSPACE / "_inventory/vix_options"
OUT = PROJECT / "results/vix_30d_accrual_calls"

INITIAL_PRINCIPAL = 1_000_000.0
ANNUAL_ACCRUAL_RATE = 0.05
MONTHLY_ACCRUAL = INITIAL_PRINCIPAL * ANNUAL_ACCRUAL_RATE / 12.0
START = pd.Timestamp("2022-01-01")
DATA_END = pd.Timestamp("2026-07-27")
TARGET_DTE = 30
MIN_DTE = 25
MAX_DTE = 35
DELTA_TARGETS = [5, 10, 15, 20, 25]
MULTIPLIER = 100.0
COMMISSION = 1.50
SPREAD_FRACTION = 0.25

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
            if START <= date <= DATA_END:
                paths[date] = path
    return paths


def first_session_each_month(paths: dict[pd.Timestamp, Path]) -> list[pd.Timestamp]:
    dates = pd.Series(sorted(paths), dtype="datetime64[ns]")
    frame = pd.DataFrame({"date": dates})
    frame["month"] = frame["date"].dt.to_period("M")
    return frame.groupby("month", sort=True)["date"].min().tolist()


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


def buy_price(quote: pd.Series) -> float:
    return float(quote["mid"] + SPREAD_FRACTION * (quote["ask"] - quote["bid"]))


def select_entries(paths: dict[pd.Timestamp, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for entry_date in first_session_each_month(paths):
        calls = read_calls(paths[entry_date])
        expirations = calls[["expiration_date", "dte"]].drop_duplicates()
        expirations = expirations[
            expirations["dte"].between(MIN_DTE, MAX_DTE)
            & expirations["expiration_date"].le(DATA_END)
        ].copy()
        if expirations.empty:
            skips.append({"entry_date": entry_date, "reason": "no completed 25-35 DTE expiry"})
            continue
        expirations["distance"] = (expirations["dte"] - TARGET_DTE).abs()
        chosen = expirations.sort_values(["distance", "dte", "expiration_date"]).iloc[0]
        expiration = pd.Timestamp(chosen["expiration_date"])
        actual_dte = int(chosen["dte"])
        expiry_calls = calls[calls["expiration_date"].eq(expiration)].copy()
        spot = float(expiry_calls["underlying_price"].dropna().median())
        otm = expiry_calls[expiry_calls["strike"].gt(spot)].copy()
        if otm.empty:
            skips.append({"entry_date": entry_date, "reason": "no OTM call"})
            continue
        for target_delta in DELTA_TARGETS:
            quote = otm.loc[(otm["delta"] - target_delta / 100.0).abs().idxmin()]
            option_price = buy_price(quote)
            cost_per_contract = option_price * MULTIPLIER + COMMISSION
            contracts = int(math.floor(MONTHLY_ACCRUAL / cost_per_contract))
            if contracts < 1:
                skips.append(
                    {
                        "entry_date": entry_date,
                        "target_delta": target_delta,
                        "reason": "monthly accrual cannot buy one contract",
                    }
                )
                continue
            premium_spent = contracts * cost_per_contract
            rows.append(
                {
                    "trade_id": f"vix_{target_delta:02d}d_{entry_date:%Y%m%d}",
                    "target_delta": target_delta,
                    "entry_date": entry_date,
                    "expiration_date": expiration,
                    "actual_dte": actual_dte,
                    "spot_entry": spot,
                    "strike": float(quote["strike"]),
                    "strike_pct_spot": float(quote["strike"]) / spot,
                    "actual_delta": float(quote["delta"]),
                    "option_symbol": str(quote["option_symbol"]),
                    "entry_bid": float(quote["bid"]),
                    "entry_ask": float(quote["ask"]),
                    "entry_mid": float(quote["mid"]),
                    "entry_modeled_price": option_price,
                    "cost_per_contract": cost_per_contract,
                    "contracts": contracts,
                    "premium_spent": premium_spent,
                    "monthly_accrual": MONTHLY_ACCRUAL,
                    "unused_accrual": MONTHLY_ACCRUAL - premium_spent,
                    "entry_volume": float(quote["volume"]),
                    "entry_open_interest": float(quote["open_interest"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(skips)


def load_marks(entries: pd.DataFrame) -> pd.DataFrame:
    selected = entries[
        ["trade_id", "option_symbol", "entry_date", "expiration_date"]
    ].copy()
    glob = (VIX_ROOT / "year=*/month=*/date=*/eod.parquet").resolve().as_posix()
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.register("selected", selected)
    marks = con.execute(
        f"""
        SELECT
            s.trade_id,
            CAST(q.snapshot_date AS DATE) AS mark_date,
            q.dte,
            q.bid,
            q.ask,
            q.mid,
            q.delta,
            q.underlying_price
        FROM read_parquet('{glob}', hive_partitioning=true, union_by_name=true) q
        JOIN selected s
          ON CAST(q.option_symbol AS VARCHAR) = s.option_symbol
         AND CAST(q.snapshot_date AS DATE)
             BETWEEN CAST(s.entry_date AS DATE) AND CAST(s.expiration_date AS DATE)
        WHERE lower(CAST(q.option_type AS VARCHAR)) = 'call'
        ORDER BY s.trade_id, mark_date
        """
    ).fetchdf()
    con.close()
    marks["mark_date"] = pd.to_datetime(marks["mark_date"])
    marks = marks[
        marks["bid"].ge(0)
        & marks["ask"].gt(0)
        & marks["ask"].ge(marks["bid"])
    ].copy()
    return marks


def attach_exits(entries: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    metadata = entries[
        ["trade_id", "entry_date", "expiration_date", "contracts", "premium_spent"]
    ]
    candidates = marks.merge(metadata, on="trade_id", how="inner")
    # VIX option data record unusable zero marks on settlement day. Close on
    # the last valid quote strictly before expiration, which is executable and
    # avoids substituting spot VIX for the VRO settlement value.
    candidates = candidates[candidates["mark_date"].lt(candidates["expiration_date"])]
    last_idx = candidates.groupby("trade_id")["mark_date"].idxmax()
    exits = candidates.loc[last_idx].copy()
    exits["exit_modeled_price"] = np.maximum(
        0.0, exits["mid"] - SPREAD_FRACTION * (exits["ask"] - exits["bid"])
    )
    gross = exits["exit_modeled_price"] * MULTIPLIER
    net_per_contract = np.where(gross > 0, np.maximum(0.0, gross - COMMISSION), 0.0)
    exits["exit_proceeds"] = net_per_contract * exits["contracts"]
    exits["option_pnl"] = exits["exit_proceeds"] - exits["premium_spent"]
    exits = exits.rename(
        columns={
            "mark_date": "exit_date",
            "bid": "exit_bid",
            "ask": "exit_ask",
            "mid": "exit_mid",
            "dte": "exit_quote_dte",
            "underlying_price": "spot_exit",
        }
    )
    cols = [
        "trade_id",
        "exit_date",
        "exit_quote_dte",
        "spot_exit",
        "exit_bid",
        "exit_ask",
        "exit_mid",
        "exit_modeled_price",
        "exit_proceeds",
        "option_pnl",
    ]
    return entries.merge(exits[cols], on="trade_id", how="inner")


def build_equity_curves(
    trades: pd.DataFrame, marks: pd.DataFrame, paths: dict[pd.Timestamp, Path]
) -> pd.DataFrame:
    all_dates = pd.DatetimeIndex(sorted(paths))
    end = pd.Timestamp(trades["exit_date"].max())
    all_dates = all_dates[(all_dates >= trades["entry_date"].min()) & (all_dates <= end)]
    curves: list[pd.DataFrame] = []

    # Baseline retains the 5% accrual instead of spending it.
    entry_calendar = sorted(pd.to_datetime(trades["entry_date"].unique()))
    baseline = pd.DataFrame({"date": all_dates})
    baseline["accrual_events"] = baseline["date"].isin(entry_calendar).astype(int)
    baseline["equity"] = INITIAL_PRINCIPAL + baseline["accrual_events"].cumsum() * MONTHLY_ACCRUAL
    baseline["strategy"] = "Retain 5% accrual"
    curves.append(baseline[["date", "strategy", "equity"]])

    for target_delta, group in trades.groupby("target_delta", sort=True):
        group = group.sort_values("entry_date").copy()
        target_marks = marks.merge(
            group[["trade_id", "target_delta", "entry_date", "exit_date", "contracts"]],
            on="trade_id",
            how="inner",
        )
        target_marks = target_marks[
            target_marks["mark_date"].between(
                target_marks["entry_date"], target_marks["exit_date"], inclusive="both"
            )
        ].copy()
        target_marks["liquidation_price"] = np.maximum(
            0.0,
            target_marks["mid"]
            - SPREAD_FRACTION * (target_marks["ask"] - target_marks["bid"]),
        )
        gross = target_marks["liquidation_price"] * MULTIPLIER
        target_marks["liquidation_value"] = (
            np.where(gross > 0, np.maximum(0.0, gross - COMMISSION), 0.0)
            * target_marks["contracts"]
        )
        mark_grid = (
            target_marks.pivot_table(
                index="mark_date", columns="trade_id", values="liquidation_value", aggfunc="last"
            )
            .reindex(all_dates)
            .ffill()
        )
        for _, trade in group.iterrows():
            trade_id = trade["trade_id"]
            if trade_id not in mark_grid.columns:
                continue
            # On the exit date the liquidation proceeds are already in cash,
            # so the option itself must no longer be included in open value.
            inactive = (mark_grid.index < trade["entry_date"]) | (mark_grid.index >= trade["exit_date"])
            mark_grid.loc[inactive, trade_id] = 0.0
        open_value = mark_grid.fillna(0.0).sum(axis=1)

        daily = pd.DataFrame({"date": all_dates})
        daily["accrual"] = daily["date"].isin(group["entry_date"]).astype(float) * MONTHLY_ACCRUAL
        premium_by_date = group.groupby("entry_date")["premium_spent"].sum()
        proceeds_by_date = group.groupby("exit_date")["exit_proceeds"].sum()
        daily["premium_spent"] = daily["date"].map(premium_by_date).fillna(0.0)
        daily["exit_proceeds"] = daily["date"].map(proceeds_by_date).fillna(0.0)
        daily["cash_flow"] = daily["accrual"] - daily["premium_spent"] + daily["exit_proceeds"]
        daily["cash"] = INITIAL_PRINCIPAL + daily["cash_flow"].cumsum()
        daily["open_option_value"] = open_value.to_numpy()
        daily["equity"] = daily["cash"] + daily["open_option_value"]
        daily["strategy"] = f"{int(target_delta)}-delta calls"
        curves.append(daily[["date", "strategy", "equity"]])
    return pd.concat(curves, ignore_index=True)


def summarize(trades: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start_date = curves["date"].min()
    end_date = curves["date"].max()
    years = (end_date - start_date).days / 365.25
    for target_delta, group in trades.groupby("target_delta", sort=True):
        curve = curves[curves["strategy"].eq(f"{int(target_delta)}-delta calls")].copy()
        start_equity = INITIAL_PRINCIPAL
        final_equity = float(curve.iloc[-1]["equity"])
        rolling_peak = curve["equity"].cummax()
        drawdown = curve["equity"] / rolling_peak - 1.0
        rows.append(
            {
                "target_delta": target_delta,
                "trades": len(group),
                "average_actual_dte": group["actual_dte"].mean(),
                "median_strike_pct_spot": group["strike_pct_spot"].median(),
                "median_cost_per_contract": group["cost_per_contract"].median(),
                "median_contracts": group["contracts"].median(),
                "total_accrual": len(group) * MONTHLY_ACCRUAL,
                "total_premium_spent": group["premium_spent"].sum(),
                "total_exit_proceeds": group["exit_proceeds"].sum(),
                "option_pnl": group["option_pnl"].sum(),
                "proceeds_divided_by_premium": group["exit_proceeds"].sum()
                / group["premium_spent"].sum(),
                "winning_trades": int(group["option_pnl"].gt(0).sum()),
                "win_rate": group["option_pnl"].gt(0).mean(),
                "final_equity": final_equity,
                "total_return": final_equity / start_equity - 1.0,
                "cagr": (final_equity / start_equity) ** (1.0 / years) - 1.0,
                "max_drawdown": drawdown.min(),
                "best_trade_pnl": group["option_pnl"].max(),
                "best_trade_entry": group.loc[group["option_pnl"].idxmax(), "entry_date"],
                "worst_trade_pnl": group["option_pnl"].min(),
            }
        )
    baseline = curves[curves["strategy"].eq("Retain 5% accrual")]
    baseline_final = float(baseline.iloc[-1]["equity"])
    for row in rows:
        row["final_vs_retain_accrual"] = row["final_equity"] - baseline_final
    return pd.DataFrame(rows)


def save_chart(curves: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    fig.patch.set_facecolor("#f8fafc")
    colors = {
        "Retain 5% accrual": "#64748b",
        "5-delta calls": "#94a3b8",
        "10-delta calls": "#0f766e",
        "15-delta calls": "#2563eb",
        "20-delta calls": "#7c3aed",
        "25-delta calls": "#dc2626",
    }
    for ax in axes:
        ax.set_facecolor("#ffffff")
        for strategy, group in curves.groupby("strategy", sort=False):
            if ax is axes[1] and strategy == "Retain 5% accrual":
                continue
            style = "--" if strategy == "Retain 5% accrual" else "-"
            width = 2.8 if strategy in {"Retain 5% accrual", "10-delta calls", "15-delta calls"} else 1.8
            ax.plot(
                group["date"],
                group["equity"],
                label=strategy,
                color=colors[strategy],
                linestyle=style,
                linewidth=width,
                alpha=0.95,
            )
        ax.axhline(INITIAL_PRINCIPAL, color="#0f172a", linewidth=1.0, alpha=0.45)
        ax.set_ylabel("Portfolio equity")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value/1_000_000:.2f}M"))
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title(
        "Monthly 30-DTE VIX calls funded by a 5% annual accrual",
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )
    axes[0].text(
        0,
        1.01,
        r"\$1 million principal · \$4,166.67/month · first trading day · last tradable quote exit",
        transform=axes[0].transAxes,
        fontsize=11.5,
        color="#475569",
    )
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].set_title("Call-funded portfolios (expanded scale)", fontsize=13, loc="left")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].text(
        0.995,
        0.015,
        "Source: local VIX option archive, Jan 2022–Jul 2026",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#64748b",
    )
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, trades: pd.DataFrame) -> None:
    lines = [
        "# Monthly 30-DTE VIX calls funded by a 5% accrual",
        "",
        (
            f"A constant $1,000,000 principal produces $50,000 per year, credited as "
            f"$4,166.67 at the start of each month. The full available monthly accrual is "
            f"used to buy whole VIX call contracts on the first trading day. Calls target "
            f"30 DTE and are closed at their last valid quote before expiration."
        ),
        "",
        "| Delta | Median strike / spot | Median contracts | Premium spent | Exit proceeds | Payoff / premium | Final equity | CAGR | Max drawdown | Wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['target_delta']:.0f} | {row['median_strike_pct_spot']:.0%} | "
            f"{row['median_contracts']:.0f} | ${row['total_premium_spent']:,.0f} | "
            f"${row['total_exit_proceeds']:,.0f} | {row['proceeds_divided_by_premium']:.2f}x | "
            f"${row['final_equity']:,.0f} | {row['cagr']:.2%} | {row['max_drawdown']:.2%} | "
            f"{row['winning_trades']:.0f}/{row['trades']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Entry fills pay midpoint plus 25% of the bid/ask spread and $1.50 per contract. Exit fills receive midpoint minus 25% of the spread and pay $1.50 per contract. Unused monthly cash is retained. The benchmark retains every monthly accrual and buys no calls.",
            "",
            "VIX options settle to VRO rather than spot VIX. Because the local archive's expiration-day marks are not usable, this test exits at the last valid pre-expiration quote instead of estimating VRO.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = build_paths()
    entries, skips = select_entries(paths)
    marks = load_marks(entries)
    trades = attach_exits(entries, marks)
    complete_ids = set(trades["trade_id"])
    incomplete = entries[~entries["trade_id"].isin(complete_ids)].copy()
    if not incomplete.empty:
        incomplete["reason"] = "no valid pre-expiration exit quote"
        skips = pd.concat(
            [skips, incomplete[["entry_date", "target_delta", "reason"]]],
            ignore_index=True,
        )
    marks = marks[marks["trade_id"].isin(complete_ids)].copy()
    curves = build_equity_curves(trades, marks, paths)
    summary = summarize(trades, curves)

    trades.to_csv(OUT / "trades.csv", index=False)
    curves.to_csv(OUT / "equity_curves.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    skips.to_csv(OUT / "skips.csv", index=False)
    save_chart(curves, summary)
    write_report(summary, trades)

    print(summary.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
