from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUT = PROJECT / "results/compare_45d_profit25"
CACHE = PROJECT / "data/cache/compare_45d_profit25"
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.execution import ExecutionModel
from spxresearch.option_selector import clean_puts, select_expiration
from spxresearch.warehouse import ResearchWarehouse


INITIAL_CAPITAL = 1_000_000.0
TARGET_DTE = 45
PROFIT_TARGET = 0.25
MODEL = ExecutionModel(spread_fraction=0.25, commission_per_contract=1.50)
STRUCTURES = {"99/96": (0.99, 0.96), "103/100": (1.03, 1.00)}


def qpath(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def nearest_put(puts: pd.DataFrame, target: float) -> pd.Series:
    return puts.loc[(puts["strike"].astype(float) - target).abs().idxmin()]


def build_candidates(archive: SPXSurfaceArchive) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates: list[dict[str, object]] = []
    legs: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for entry_date in archive.populated_dates:
        chain = archive.read(entry_date)
        puts_all = clean_puts(chain, pm_only=True)
        try:
            expiration, actual_dte = select_expiration(
                puts_all, TARGET_DTE, tolerance=4, pm_only=True
            )
        except LookupError as error:
            skips.append({"entry_date": entry_date, "reason": str(error)})
            continue
        puts = puts_all[puts_all["expiration_date"].eq(expiration)].copy()
        spot = float(puts["underlying_price"].dropna().median())
        for structure, (short_ratio, long_ratio) in STRUCTURES.items():
            short = nearest_put(puts, spot * short_ratio)
            lower = puts[puts["strike"].astype(float) < float(short["strike"])]
            if lower.empty:
                skips.append({"entry_date": entry_date, "structure": structure, "reason": "no lower long"})
                continue
            long = nearest_put(lower, spot * long_ratio)
            entry_cash = MODEL.cash_flow(short, -1) + MODEL.cash_flow(long, 1)
            width_cash = (float(short["strike"]) - float(long["strike"])) * 100.0
            if entry_cash <= 0 or width_cash <= entry_cash:
                skips.append({"entry_date": entry_date, "structure": structure, "reason": "invalid credit or max loss"})
                continue
            trade_id = f"{structure.replace('/', '_')}_{entry_date:%Y%m%d}"
            candidates.append(
                {
                    "trade_id": trade_id,
                    "structure": structure,
                    "entry_date": entry_date,
                    "expiration_date": expiration,
                    "actual_dte": int(actual_dte),
                    "spot_entry": spot,
                    "short_strike": float(short["strike"]),
                    "long_strike": float(long["strike"]),
                    "entry_credit_cash": float(entry_cash),
                    "max_loss_cash": float(width_cash - entry_cash),
                }
            )
            for role, quantity, option in [("short", -1, short), ("long", 1, long)]:
                legs.append(
                    {
                        "trade_id": trade_id,
                        "role": role,
                        "quantity": quantity,
                        "option_symbol": str(option["option_symbol"]).strip(),
                        "strike": float(option["strike"]),
                    }
                )
    return pd.DataFrame(candidates), pd.DataFrame(legs), pd.DataFrame(skips)


def build_marks(candidates: pd.DataFrame, legs: pd.DataFrame, data_root: Path) -> Path:
    marks_path = CACHE / "marks_v2.parquet"
    with ResearchWarehouse(
        PROJECT / "data/cache/compare_45d_profit25.duckdb",
        data_root,
        threads=8,
        memory_limit="12GB",
    ) as warehouse:
        con = warehouse.connection
        con.register("candidates", candidates)
        con.register("legs", legs)
        con.execute(
            f"""
            COPY (
                WITH leg_marks AS (
                    SELECT
                        c.*, l.role, l.quantity, l.strike,
                        s.trade_date AS mark_date, s.dte AS current_dte,
                        s.spot, s.bid, s.ask, s.mid
                    FROM candidates c
                    JOIN legs l USING (trade_id)
                    JOIN surface s
                      ON s.option_symbol = l.option_symbol
                     AND s.trade_date BETWEEN c.entry_date AND c.expiration_date
                ), aggregated AS (
                    SELECT
                        trade_id,
                        min(structure) AS structure,
                        min(entry_date) AS entry_date,
                        min(expiration_date) AS expiration_date,
                        min(actual_dte) AS actual_dte,
                        min(spot_entry) AS spot_entry,
                        min(short_strike) AS short_strike,
                        min(long_strike) AS long_strike,
                        min(entry_credit_cash) AS entry_credit_cash,
                        min(max_loss_cash) AS max_loss_cash,
                        mark_date,
                        min(current_dte) AS current_dte,
                        min(spot) AS spot_mark,
                        sum(quantity * mid * 100.0) AS position_value_mid,
                        sum(
                            CASE
                                WHEN mark_date = expiration_date OR current_dte = 0 THEN
                                    quantity * greatest(strike - spot, 0.0) * 100.0
                                ELSE quantity * mid * 100.0
                                     - abs(quantity) * 0.25 * (ask - bid) * 100.0
                                     - abs(quantity) * 1.50
                            END
                        ) AS close_cash_realistic,
                        count(*) AS legs_marked
                    FROM leg_marks
                    GROUP BY trade_id, mark_date
                    HAVING count(*) = 2
                )
                SELECT
                    *,
                    entry_credit_cash + position_value_mid AS pnl_mid,
                    entry_credit_cash + close_cash_realistic AS pnl_realistic,
                    (mark_date = expiration_date OR current_dte = 0) AS is_final
                FROM aggregated
            ) TO '{qpath(marks_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    return marks_path


def choose_exits(marks: pd.DataFrame) -> pd.DataFrame:
    eligible = marks[
        marks["is_final"]
        | (
            marks["mark_date"].gt(marks["entry_date"])
            & marks["pnl_realistic"].ge(PROFIT_TARGET * marks["entry_credit_cash"])
        )
    ].copy()
    exits = eligible.sort_values("mark_date").groupby("trade_id", as_index=False).first()
    exits = exits.rename(
        columns={
            "mark_date": "exit_date",
            "spot_mark": "spot_exit",
            "pnl_realistic": "realized_pnl_per_spread",
        }
    )
    selected: list[pd.DataFrame] = []
    for structure, group in exits.groupby("structure", sort=False):
        available_after = pd.Timestamp.min
        positions: list[int] = []
        for index, row in group.sort_values("entry_date").iterrows():
            entry = pd.Timestamp(row["entry_date"])
            if entry > available_after:
                positions.append(index)
                available_after = pd.Timestamp(row["exit_date"])
        chosen = group.loc[positions].sort_values("entry_date").copy()
        chosen["sequence_number"] = np.arange(len(chosen))
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


def simulate(selected: pd.DataFrame, marks: pd.DataFrame, market_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    mark_groups = {key: group.sort_values("mark_date") for key, group in marks.groupby("trade_id")}
    for structure, group in selected.groupby("structure", sort=False):
        equity_realized = INITIAL_CAPITAL
        daily_pnl = pd.Series(0.0, index=market_index)
        for row in group.sort_values("entry_date").itertuples():
            pre_equity = equity_realized
            contracts = pre_equity / (float(row.short_strike) * 100.0)
            dollar_pnl = contracts * float(row.realized_pnl_per_spread)
            equity_realized += dollar_pnl
            path = mark_groups[row.trade_id]
            path = path[path["mark_date"].le(pd.Timestamp(row.exit_date))]
            cumulative = path.set_index(pd.to_datetime(path["mark_date"]))["pnl_mid"].astype(float)
            cumulative.iloc[-1] = float(row.realized_pnl_per_spread)
            scaled = cumulative * contracts
            changes = scaled.diff().fillna(scaled.iloc[0])
            daily_pnl.loc[changes.index] += changes
            trade_rows.append(
                {
                    "strategy": f"{structure} 45D profit25",
                    "structure": structure,
                    "trade_id": row.trade_id,
                    "entry_date": row.entry_date,
                    "exit_date": row.exit_date,
                    "expiration_date": row.expiration_date,
                    "holding_days": (pd.Timestamp(row.exit_date) - pd.Timestamp(row.entry_date)).days,
                    "exited_early": pd.Timestamp(row.exit_date) < pd.Timestamp(row.expiration_date),
                    "contracts": contracts,
                    "pre_trade_equity": pre_equity,
                    "dollar_pnl": dollar_pnl,
                    "trade_return": dollar_pnl / pre_equity,
                    "spot_return": float(row.spot_exit) / float(row.spot_entry) - 1.0,
                }
            )
        equity = INITIAL_CAPITAL + daily_pnl.cumsum()
        returns = daily_pnl / equity.shift(1, fill_value=INITIAL_CAPITAL)
        years = len(returns) / 252.0
        peak = pd.Series(np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity.to_numpy()])[1:], index=market_index)
        std = float(returns.std(ddof=1))
        structure_trades = pd.DataFrame([row for row in trade_rows if row["structure"] == structure])
        metric_rows.append(
            {
                "strategy": f"{structure} 45D profit25",
                "structure": structure,
                "trades": len(structure_trades),
                "cagr": float((equity.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0),
                "annualized_volatility": std * math.sqrt(252.0),
                "sharpe_zero_cash": float(returns.mean() / std * math.sqrt(252.0)),
                "max_drawdown": float((equity / peak - 1.0).min()),
                "ending_equity": float(equity.iloc[-1]),
                "win_rate": float((structure_trades["dollar_pnl"] > 0).mean()),
                "average_holding_days": float(structure_trades["holding_days"].mean()),
                "early_exit_rate": float(structure_trades["exited_early"].mean()),
                "mean_return_spx_down_5": float(structure_trades.loc[structure_trades["spot_return"].le(-0.05), "trade_return"].mean()),
                "observations_spx_down_5": int(structure_trades["spot_return"].le(-0.05).sum()),
            }
        )
        curves[structure] = equity
    return pd.DataFrame(metric_rows), pd.DataFrame(curves, index=market_index), pd.DataFrame(trade_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    archive = SPXSurfaceArchive(data_root)
    candidates_path = CACHE / "candidates.parquet"
    legs_path = CACHE / "legs.parquet"
    marks_path = CACHE / "marks_v2.parquet"
    if candidates_path.exists() and legs_path.exists():
        candidates = pd.read_parquet(candidates_path)
        legs = pd.read_parquet(legs_path)
        skips = pd.DataFrame()
    else:
        candidates, legs, skips = build_candidates(archive)
        candidates.to_parquet(candidates_path, index=False)
        legs.to_parquet(legs_path, index=False)
        skips.to_csv(OUT / "skips.csv", index=False)
    if not marks_path.exists():
        marks_path = build_marks(candidates, legs, data_root)
    marks = pd.read_parquet(marks_path)
    for column in ["entry_date", "expiration_date", "mark_date"]:
        marks[column] = pd.to_datetime(marks[column])
    selected = choose_exits(marks)
    selected.to_csv(OUT / "selected_trades.csv", index=False)
    market_index = pd.DatetimeIndex(archive.populated_dates)
    summary, curves, trade_detail = simulate(selected, marks, market_index)
    summary.to_csv(OUT / "summary.csv", index=False)
    curves.to_parquet(OUT / "equity_curves.parquet")
    curves.to_csv(OUT / "equity_curves.csv")
    trade_detail.to_csv(OUT / "trade_detail.csv", index=False)

    colors = {"99/96": "#006b76", "103/100": "#c56a1a"}
    indexed = summary.set_index("structure")
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for structure in ["99/96", "103/100"]:
        row = indexed.loc[structure]
        label = f"{structure} • 45 DTE • 25% profit target ({row['cagr']:.2%} CAGR)"
        ax.plot(curves.index, curves[structure] / INITIAL_CAPITAL - 1.0, lw=2.1, color=colors[structure], label=label)
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title("45-DTE SPX put spreads with a 25% profit target", loc="left", weight="bold", pad=20)
    ax.set_ylabel("Cumulative option return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.075,
        0.025,
        "One position at a time; re-enter next trading day after exit; short-strike notional equals current equity; realistic execution; no cash interest.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "equity_curve_comparison.png", dpi=180)
    plt.close(fig)

    manifest = {
        "archive_start": str(market_index.min().date()),
        "archive_end": str(market_index.max().date()),
        "target_dte": TARGET_DTE,
        "profit_target": PROFIT_TARGET,
        "structures": list(STRUCTURES),
        "candidates": len(candidates),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
