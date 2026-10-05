from __future__ import annotations

import calendar
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent.parent
PROJECT = WORKSPACE / "spx_option_research"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.execution import ExecutionModel
from spxresearch.option_selector import clean_puts
from spxresearch.warehouse import ResearchWarehouse


INITIAL_CAPITAL = 1_000_000.0
MAX_LOSS_BUDGET = 0.05 * INITIAL_CAPITAL
MULTIPLIER = 100.0
MODEL = ExecutionModel(spread_fraction=0.25, commission_per_contract=1.50)
STRUCTURES = {"101/98": (1.01, 0.98), "99/96": (0.99, 0.96)}


def third_friday(year: int, month: int) -> pd.Timestamp:
    weeks = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in weeks if week[calendar.FRIDAY]]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def monthly_roll_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    available = set(dates)
    result: list[pd.Timestamp] = []
    for period in pd.period_range(dates.min(), dates.max(), freq="M"):
        scheduled = third_friday(period.year, period.month)
        choices = [scheduled - pd.Timedelta(days=n) for n in range(4) if scheduled - pd.Timedelta(days=n) in available]
        if choices:
            result.append(max(choices))
    return sorted(set(result))


def nearest_put(puts: pd.DataFrame, target_strike: float) -> pd.Series:
    return puts.loc[(puts["strike"].astype(float) - target_strike).abs().idxmin()]


def build_entries(archive: SPXSurfaceArchive) -> tuple[pd.DataFrame, pd.DataFrame]:
    populated = archive.populated_dates
    available = set(populated)
    friday_dates = [date for date in populated if date.weekday() == calendar.FRIDAY]
    weekly_pairs = [(date, date + pd.Timedelta(days=7)) for date in friday_dates if date + pd.Timedelta(days=7) in available]
    monthly_dates = monthly_roll_dates(populated)
    monthly_pairs = list(zip(monthly_dates[:-1], monthly_dates[1:]))
    calendars = {"Weekly": weekly_pairs, "Monthly": monthly_pairs}

    records: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for cadence, pairs in calendars.items():
        for entry_date, expiration_date in pairs:
            chain = archive.read(entry_date)
            puts = clean_puts(chain, pm_only=True)
            puts = puts[puts["expiration_date"].eq(expiration_date)].copy()
            if puts.empty:
                skips.append({"cadence": cadence, "entry_date": entry_date, "reason": "no_matching_pm_expiration"})
                continue
            spot = float(puts["underlying_price"].dropna().median())
            for structure, (short_ratio, long_ratio) in STRUCTURES.items():
                short = nearest_put(puts, spot * short_ratio)
                eligible_longs = puts[puts["strike"].astype(float) < float(short["strike"])]
                if eligible_longs.empty:
                    skips.append({"cadence": cadence, "entry_date": entry_date, "structure": structure, "reason": "no_lower_long"})
                    continue
                long = nearest_put(eligible_longs, spot * long_ratio)
                width = float(short["strike"] - long["strike"])
                entry_credit = (
                    MODEL.cash_flow(short, -1) + MODEL.cash_flow(long, 1)
                ) / MULTIPLIER
                max_loss = (width - entry_credit) * MULTIPLIER
                if entry_credit <= 0 or max_loss <= 0:
                    skips.append({"cadence": cadence, "entry_date": entry_date, "structure": structure, "reason": "nonpositive_credit_or_risk"})
                    continue
                strategy = f"{structure} {cadence}"
                records.append(
                    {
                        "trade_id": f"{strategy}_{entry_date:%Y%m%d}",
                        "strategy": strategy,
                        "structure": structure,
                        "cadence": cadence,
                        "entry_date": entry_date,
                        "expiration_date": expiration_date,
                        "entry_dte": int((expiration_date - entry_date).days),
                        "spot_entry": spot,
                        "short_target_ratio": short_ratio,
                        "long_target_ratio": long_ratio,
                        "short_strike": float(short["strike"]),
                        "long_strike": float(long["strike"]),
                        "short_actual_ratio": float(short["strike"]) / spot,
                        "long_actual_ratio": float(long["strike"]) / spot,
                        "short_symbol": str(short["option_symbol"]).strip(),
                        "long_symbol": str(long["option_symbol"]).strip(),
                        "entry_credit_points": entry_credit,
                        "width_points": width,
                        "max_loss_one_contract": max_loss,
                        "contracts": MAX_LOSS_BUDGET / max_loss,
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(skips)


def performance(equity: pd.Series, daily_pnl: pd.Series, trade_pnl: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().fillna((equity.iloc[0] - INITIAL_CAPITAL) / INITIAL_CAPITAL)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    peak = pd.Series(np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity.to_numpy()])[1:], index=equity.index)
    drawdown = equity / peak - 1.0
    volatility = returns.std(ddof=1) * math.sqrt(252.0)
    return {
        "trades": int(len(trade_pnl)),
        "cagr": (equity.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0,
        "annualized_volatility": volatility,
        "sharpe_zero_cash": returns.mean() / returns.std(ddof=1) * math.sqrt(252.0),
        "max_drawdown": float(drawdown.min()),
        "ending_equity": float(equity.iloc[-1]),
        "total_option_pnl": float(daily_pnl.sum()),
        "win_rate": float((trade_pnl > 0).mean()),
        "worst_trade": float(trade_pnl.min()),
    }


def main() -> None:
    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    archive = SPXSurfaceArchive(data_root)
    entries, skips = build_entries(archive)
    if entries.empty:
        raise RuntimeError("No trades selected")
    entries.to_csv(OUT / "moneyness_cadence_trades.csv", index=False)
    entries.to_parquet(OUT / "moneyness_cadence_entries.parquet", index=False)
    skips.to_csv(OUT / "moneyness_cadence_skips.csv", index=False)

    with ResearchWarehouse(
        PROJECT / "data/cache/research.duckdb", data_root, threads=8, memory_limit="12GB"
    ) as warehouse:
        warehouse.connection.register("chosen", entries)
        paths = warehouse.connection.execute(
            """
            WITH marks AS (
                SELECT
                    c.trade_id, c.strategy, c.entry_date, c.expiration_date,
                    s.trade_date AS mark_date, c.contracts,
                    CASE WHEN s.trade_date = c.expiration_date OR s.dte = 0
                         THEN greatest(c.short_strike - s.spot, 0.0)
                              - greatest(c.long_strike - s.spot, 0.0)
                         ELSE s.mid - l.mid END AS marked_debit,
                    c.entry_credit_points
                FROM chosen c
                JOIN surface s ON s.option_symbol = c.short_symbol
                  AND s.trade_date BETWEEN c.entry_date AND c.expiration_date
                JOIN surface l ON l.option_symbol = c.long_symbol
                  AND l.trade_date = s.trade_date
            ), cumulative AS (
                SELECT *,
                    (entry_credit_points - marked_debit) * 100.0 * contracts AS cumulative_pnl
                FROM marks
            ), changes AS (
                SELECT *, cumulative_pnl - lag(cumulative_pnl, 1, 0.0)
                    OVER (PARTITION BY trade_id ORDER BY mark_date) AS daily_pnl
                FROM cumulative
            )
            SELECT * FROM changes ORDER BY strategy, trade_id, mark_date
            """
        ).fetchdf()

    paths["mark_date"] = pd.to_datetime(paths["mark_date"])
    paths.to_parquet(OUT / "moneyness_cadence_daily_paths.parquet", index=False)
    all_dates = pd.DatetimeIndex(archive.populated_dates)
    equity_curves: dict[str, pd.Series] = {}
    daily_table = pd.DataFrame(index=all_dates)
    summary_rows: list[dict[str, object]] = []
    for strategy in ["101/98 Weekly", "101/98 Monthly", "99/96 Weekly", "99/96 Monthly"]:
        group = paths[paths["strategy"].eq(strategy)]
        daily = group.groupby("mark_date")["daily_pnl"].sum().reindex(all_dates, fill_value=0.0)
        equity = INITIAL_CAPITAL + daily.cumsum()
        trade_pnl = group.groupby("trade_id")["daily_pnl"].sum()
        row: dict[str, object] = {"strategy": strategy}
        row.update(performance(equity, daily, trade_pnl))
        selected = entries[entries["strategy"].eq(strategy)]
        row.update(
            {
                "first_entry": selected["entry_date"].min(),
                "last_expiration": selected["expiration_date"].max(),
                "average_entry_dte": selected["entry_dte"].mean(),
                "average_short_ratio": selected["short_actual_ratio"].mean(),
                "average_long_ratio": selected["long_actual_ratio"].mean(),
            }
        )
        summary_rows.append(row)
        equity_curves[strategy] = equity
        daily_table[strategy] = daily

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "moneyness_cadence_summary.csv", index=False)
    pd.DataFrame(equity_curves, index=all_dates).to_csv(OUT / "moneyness_cadence_equity.csv")
    daily_table.to_csv(OUT / "moneyness_cadence_daily_pnl.csv")

    colors = {
        "101/98 Weekly": "#006b76",
        "101/98 Monthly": "#5ab1bb",
        "99/96 Weekly": "#c56a1a",
        "99/96 Monthly": "#e8ae65",
    }
    styles = {"101/98 Weekly": "-", "101/98 Monthly": "--", "99/96 Weekly": "-", "99/96 Monthly": "--"}
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for strategy, equity in equity_curves.items():
        ax.plot(equity.index, equity, label=strategy, color=colors[strategy], linestyle=styles[strategy], linewidth=2.0)
    ax.axhline(INITIAL_CAPITAL, color="#64748b", linewidth=0.9)
    ax.set_title("SPX put-spread equity: strike structure and selling cadence", loc="left", weight="bold", pad=22)
    ax.text(
        0, 1.015,
        "101/98 and 99/96 denote strike percentages of entry spot • option P&L only",
        transform=ax.transAxes, fontsize=10,
    )
    ax.set_ylabel("Portfolio equity")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.text(
        0.075, 0.025,
        "Initial capital $1,000,000; each position sized to 5% maximum loss; continuous contract equivalents; realistic entry execution.\n"
        "Weekly: Friday-to-Friday PM-settled spread. Monthly: third-Friday roll to the next third-Friday PM-settled spread. Held to expiration; no cash interest.",
        fontsize=9, color="#475569",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / "moneyness_cadence_equity_curves.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
