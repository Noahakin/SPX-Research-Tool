from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUT = PROJECT / "results/sequential_itm_hedges"
CACHE = PROJECT / "data/cache/sequential_itm_hedges"
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.execution import ExecutionModel
from spxresearch.option_selector import clean_puts, select_expiration
from spxresearch.warehouse import ResearchWarehouse


INITIAL_CAPITAL = 1_000_000.0
MODEL = ExecutionModel(spread_fraction=0.25, commission_per_contract=1.50)
TARGET_DTES = [15, 21, 30, 45, 60, 75]
SHORT_RATIOS = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]
HEDGE_DELTAS = [5, 10, 15, 20]
HEDGE_BUDGETS = [0.05, 0.10, 0.15, 0.20]
PROFIT_LEVELS = list(range(10, 100, 10))


def qpath(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def nearest_ratio(puts: pd.DataFrame, spot: float, ratio: float) -> pd.Series:
    return puts.loc[(puts["strike"].astype(float) - spot * ratio).abs().idxmin()]


def nearest_delta_below(puts: pd.DataFrame, strike_cap: float, target_delta: int) -> pd.Series:
    eligible = puts[(puts["strike"] < strike_cap) & puts["delta"].notna()]
    if eligible.empty:
        raise LookupError("no lower-strike hedge put")
    return eligible.loc[(eligible["delta"].abs() - target_delta / 100.0).abs().idxmin()]


def build_candidates(archive: SPXSurfaceArchive) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core_rows: list[dict[str, object]] = []
    core_legs: list[dict[str, object]] = []
    hedge_rows: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    for number, entry_date in enumerate(archive.populated_dates, start=1):
        chain = archive.read(entry_date)
        puts_all = clean_puts(chain, pm_only=True)
        for target_dte in TARGET_DTES:
            try:
                expiration, actual_dte = select_expiration(
                    puts_all, target_dte, tolerance=4, pm_only=True
                )
            except LookupError as error:
                skips.append({"entry_date": entry_date, "target_dte": target_dte, "reason": str(error)})
                continue
            puts = puts_all[puts_all["expiration_date"].eq(expiration)].copy()
            spot = float(puts["underlying_price"].dropna().median())
            for short_ratio in SHORT_RATIOS:
                short = nearest_ratio(puts, spot, short_ratio)
                lower = puts[puts["strike"] < float(short["strike"])]
                if lower.empty:
                    continue
                long = nearest_ratio(lower, spot, short_ratio - 0.03)
                if float(long["strike"]) >= float(short["strike"]):
                    continue
                core_entry_cash = MODEL.cash_flow(short, -1) + MODEL.cash_flow(long, 1)
                width_cash = (float(short["strike"]) - float(long["strike"])) * 100.0
                max_loss = width_cash - core_entry_cash
                if core_entry_cash <= 0 or max_loss <= 0:
                    continue
                parameter_id = f"{short_ratio*100:.0f}/{(short_ratio-0.03)*100:.0f}_dte{target_dte}"
                trade_id = f"{parameter_id}_{entry_date:%Y%m%d}"
                core_rows.append(
                    {
                        "trade_id": trade_id,
                        "parameter_id": parameter_id,
                        "structure": f"{short_ratio*100:.0f}/{(short_ratio-0.03)*100:.0f}",
                        "target_dte": target_dte,
                        "actual_dte": int(actual_dte),
                        "entry_date": entry_date,
                        "expiration_date": expiration,
                        "spot_entry": spot,
                        "short_strike": float(short["strike"]),
                        "long_strike": float(long["strike"]),
                        "short_ratio_actual": float(short["strike"]) / spot,
                        "short_abs_delta": abs(float(short["delta"])),
                        "core_entry_cash": core_entry_cash,
                        "core_max_profit": core_entry_cash,
                        "core_max_loss": max_loss,
                    }
                )
                core_legs.extend(
                    [
                        {"trade_id": trade_id, "role": "short", "quantity": -1, "option_symbol": str(short["option_symbol"]).strip(), "strike": float(short["strike"])},
                        {"trade_id": trade_id, "role": "long", "quantity": 1, "option_symbol": str(long["option_symbol"]).strip(), "strike": float(long["strike"])},
                    ]
                )
                for hedge_delta in HEDGE_DELTAS:
                    try:
                        hedge = nearest_delta_below(puts, float(long["strike"]), hedge_delta)
                    except LookupError:
                        continue
                    entry_cash = MODEL.cash_flow(hedge, 1)
                    debit = -entry_cash
                    if debit <= 0:
                        continue
                    hedge_rows.append(
                        {
                            "trade_id": trade_id,
                            "hedge_delta": hedge_delta,
                            "option_symbol": str(hedge["option_symbol"]).strip(),
                            "hedge_strike": float(hedge["strike"]),
                            "hedge_abs_delta_actual": abs(float(hedge["delta"])),
                            "hedge_entry_cash": entry_cash,
                            "hedge_entry_debit": debit,
                        }
                    )
        if number % 100 == 0:
            print(f"Daily candidate selection {number}/{len(archive.populated_dates)}", flush=True)
    return pd.DataFrame(core_rows), pd.DataFrame(core_legs), pd.DataFrame(hedge_rows), pd.DataFrame(skips)


def build_core_marks_and_exits(
    core: pd.DataFrame, legs: pd.DataFrame, data_root: Path
) -> tuple[Path, Path]:
    marks_path = CACHE / "core_marks.parquet"
    exits_path = CACHE / "core_exits.parquet"
    rules = pd.DataFrame(
        [("hold", "final", 0.0)]
        + [(f"profit_{level}", "profit", level / 100.0) for level in PROFIT_LEVELS]
        + [("half_life", "dte_fraction", 0.50), ("exit_14dte", "dte", 14.0), ("exit_7dte", "dte", 7.0)],
        columns=["exit_rule", "rule_kind", "rule_value"],
    )
    with ResearchWarehouse(
        PROJECT / "data/cache/research.duckdb", data_root, threads=8, memory_limit="12GB"
    ) as warehouse:
        con = warehouse.connection
        con.register("core_candidates", core)
        con.register("core_legs", legs)
        con.register("exit_rules", rules)
        con.execute(
            f"""
            COPY (
                WITH leg_marks AS (
                    SELECT
                        c.*, l.role, l.quantity, l.strike,
                        s.trade_date AS mark_date, s.dte AS current_dte,
                        s.spot, s.bid, s.ask, s.mid, s.delta
                    FROM core_candidates c
                    JOIN core_legs l USING (trade_id)
                    JOIN surface s ON s.option_symbol = l.option_symbol
                      AND s.trade_date BETWEEN c.entry_date AND c.expiration_date
                ), aggregated AS (
                    SELECT
                        trade_id, min(parameter_id) AS parameter_id,
                        min(structure) AS structure, min(target_dte) AS target_dte,
                        min(actual_dte) AS actual_dte, min(entry_date) AS entry_date,
                        min(expiration_date) AS expiration_date,
                        min(spot_entry) AS spot_entry, min(short_strike) AS short_strike,
                        min(short_ratio_actual) AS short_ratio_actual,
                        min(short_abs_delta) AS short_abs_delta,
                        min(core_entry_cash) AS core_entry_cash,
                        min(core_max_profit) AS core_max_profit,
                        min(core_max_loss) AS core_max_loss,
                        mark_date, min(current_dte) AS current_dte, min(spot) AS spot_mark,
                        sum(quantity * mid * 100.0) AS position_value_mid,
                        sum(
                            CASE WHEN mark_date = expiration_date OR current_dte = 0
                                 THEN quantity * greatest(strike - spot, 0.0) * 100.0
                                 ELSE quantity * mid * 100.0
                                      - abs(quantity) * 0.25 * (ask - bid) * 100.0
                                      - abs(quantity) * 1.50 END
                        ) AS close_cash_realistic,
                        max(abs(delta)) FILTER (WHERE role = 'short') AS short_delta_mark,
                        count(*) AS legs_marked
                    FROM leg_marks
                    GROUP BY trade_id, mark_date
                    HAVING count(*) = 2
                )
                SELECT *,
                    core_entry_cash + position_value_mid AS core_pnl_mid,
                    core_entry_cash + close_cash_realistic AS core_pnl_realistic,
                    mark_date = max(mark_date) OVER (PARTITION BY trade_id) AS is_final
                FROM aggregated
            ) TO '{qpath(marks_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        con.execute(
            f"""
            COPY (
                WITH marks AS (SELECT * FROM read_parquet('{qpath(marks_path)}')),
                eligible AS (
                    SELECT m.*, r.*,
                        CASE
                            WHEN m.is_final THEN true
                            WHEN r.rule_kind = 'final' THEN false
                            WHEN r.rule_kind = 'profit' THEN
                                m.mark_date > m.entry_date
                                AND m.core_pnl_realistic >= r.rule_value * m.core_max_profit
                            WHEN r.rule_kind = 'dte' THEN
                                m.mark_date > m.entry_date AND m.current_dte <= r.rule_value
                            WHEN r.rule_kind = 'dte_fraction' THEN
                                m.mark_date > m.entry_date
                                AND m.current_dte <= m.actual_dte * r.rule_value
                            ELSE false
                        END AS should_exit
                    FROM marks m CROSS JOIN exit_rules r
                )
                SELECT
                    * EXCLUDE (is_final, should_exit), mark_date AS exit_date,
                    spot_mark AS spot_exit,
                    core_pnl_realistic AS core_realized_pnl
                FROM eligible
                WHERE should_exit
                QUALIFY row_number() OVER (
                    PARTITION BY trade_id, exit_rule ORDER BY mark_date
                ) = 1
            ) TO '{qpath(exits_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    return marks_path, exits_path


def select_nonoverlapping(exits: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for (parameter_id, exit_rule), group in exits.groupby(["parameter_id", "exit_rule"], sort=False):
        group = group.sort_values("entry_date")
        available_after = pd.Timestamp.min
        positions: list[int] = []
        for index, row in group.iterrows():
            entry = pd.Timestamp(row["entry_date"])
            if entry > available_after:
                positions.append(index)
                available_after = pd.Timestamp(row["exit_date"])
        chosen = group.loc[positions].copy()
        chosen["sequence_id"] = f"{parameter_id}_{exit_rule}"
        chosen["sequence_number"] = np.arange(len(chosen))
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


def build_hedge_marks(
    hedge_candidates: pd.DataFrame,
    selected_trade_ids: pd.DataFrame,
    core: pd.DataFrame,
    data_root: Path,
) -> Path:
    output = CACHE / "hedge_marks.parquet"
    with ResearchWarehouse(
        PROJECT / "data/cache/research.duckdb", data_root, threads=8, memory_limit="12GB"
    ) as warehouse:
        con = warehouse.connection
        con.register("hedges", hedge_candidates)
        con.register("selected_ids", selected_trade_ids)
        con.register("core_candidates", core[["trade_id", "entry_date", "expiration_date"]])
        con.execute(
            f"""
            COPY (
                SELECT
                    h.trade_id, h.hedge_delta, h.hedge_strike,
                    h.hedge_abs_delta_actual, h.hedge_entry_cash,
                    h.hedge_entry_debit, c.entry_date, c.expiration_date,
                    s.trade_date AS mark_date, s.dte AS current_dte, s.spot,
                    h.hedge_entry_cash + s.mid * 100.0 AS hedge_pnl_mid,
                    h.hedge_entry_cash +
                        CASE WHEN s.trade_date = c.expiration_date OR s.dte = 0
                             THEN greatest(h.hedge_strike - s.spot, 0.0) * 100.0
                             ELSE (s.mid - 0.25 * (s.ask - s.bid)) * 100.0 - 1.50
                        END AS hedge_pnl_realistic
                FROM hedges h
                JOIN selected_ids i USING (trade_id)
                JOIN core_candidates c USING (trade_id)
                JOIN surface s ON s.option_symbol = h.option_symbol
                  AND s.trade_date BETWEEN c.entry_date AND c.expiration_date
            ) TO '{qpath(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    return output


def metrics_from_returns(returns: pd.Series) -> dict[str, float]:
    wealth = (1.0 + returns).cumprod()
    years = len(returns) / 252.0
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    std = float(returns.std(ddof=1))
    return {
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else np.nan,
        "annualized_volatility_realized": std * math.sqrt(252.0),
        "sharpe_zero_cash_realized": float(returns.mean() / std * math.sqrt(252.0)) if std > 0 else np.nan,
        "max_drawdown_realized": float(np.min(wealth.to_numpy() / peak - 1.0)),
        "ending_equity": float(wealth.iloc[-1] * INITIAL_CAPITAL),
    }


def simulate_screen(
    sequences: pd.DataFrame,
    hedge_candidates: pd.DataFrame,
    hedge_marks_path: Path,
    market_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hedge_marks = pd.read_parquet(hedge_marks_path)
    exit_lookup = hedge_marks.sort_values("mark_date").merge(
        sequences[["sequence_id", "trade_id", "exit_date"]], on="trade_id", how="inner"
    )
    exit_lookup = exit_lookup[exit_lookup["mark_date"] <= exit_lookup["exit_date"]]
    exit_lookup = exit_lookup.sort_values("mark_date").groupby(
        ["sequence_id", "trade_id", "hedge_delta"], as_index=False
    ).tail(1)
    hedge_entry = hedge_candidates.set_index(["trade_id", "hedge_delta"])
    hedge_exit = exit_lookup.set_index(["sequence_id", "trade_id", "hedge_delta"])
    overlays: list[tuple[str, int | None, float]] = [("none", None, 0.0)]
    overlays.extend(
        (f"put_d{delta}_b{int(budget*100)}", delta, budget)
        for delta in HEDGE_DELTAS for budget in HEDGE_BUDGETS
    )
    first, second = int(len(market_index) * 0.60), int(len(market_index) * 0.80)
    split_dates = {
        "train": market_index[:first],
        "validation": market_index[first:second],
        "test": market_index[second:],
    }
    rows: list[dict[str, object]] = []
    trade_detail: list[dict[str, object]] = []
    for sequence_id, group in sequences.groupby("sequence_id", sort=False):
        group = group.sort_values("entry_date")
        base = group.iloc[0]
        for overlay_id, hedge_delta, budget in overlays:
            equity = INITIAL_CAPITAL
            realized_returns = pd.Series(0.0, index=market_index)
            wins = 0
            tail5: list[float] = []
            tail10: list[float] = []
            hedge_ratios: list[float] = []
            details: list[dict[str, object]] = []
            valid = True
            for row in group.itertuples():
                pre_equity = equity
                contracts = pre_equity / (float(row.short_strike) * 100.0)
                hedge_ratio = 0.0
                hedge_pnl = 0.0
                if hedge_delta is not None:
                    try:
                        entry = hedge_entry.loc[(row.trade_id, hedge_delta)]
                        marked = hedge_exit.loc[(sequence_id, row.trade_id, hedge_delta)]
                    except KeyError:
                        valid = False
                        break
                    hedge_ratio = min(
                        1.0,
                        budget * float(row.core_entry_cash) / float(entry["hedge_entry_debit"]),
                    )
                    hedge_pnl = hedge_ratio * float(marked["hedge_pnl_realistic"])
                package_pnl_per_spread = float(row.core_realized_pnl) + hedge_pnl
                dollar_pnl = contracts * package_pnl_per_spread
                equity += dollar_pnl
                trade_return = dollar_pnl / pre_equity
                exit_date = pd.Timestamp(row.exit_date)
                if exit_date in realized_returns.index:
                    realized_returns.loc[exit_date] += trade_return
                wins += dollar_pnl > 0
                underlying_return = float(row.spot_exit) / float(row.spot_entry) - 1.0
                if underlying_return <= -0.05:
                    tail5.append(trade_return)
                if underlying_return <= -0.10:
                    tail10.append(trade_return)
                hedge_ratios.append(hedge_ratio)
                details.append(
                    {
                        "strategy_id": f"{sequence_id}_{overlay_id}",
                        "sequence_id": sequence_id,
                        "overlay_id": overlay_id,
                        "trade_id": row.trade_id,
                        "entry_date": row.entry_date,
                        "exit_date": row.exit_date,
                        "contracts": contracts,
                        "hedge_delta": hedge_delta,
                        "hedge_budget": budget,
                        "hedge_ratio": hedge_ratio,
                        "core_pnl_per_unit": row.core_realized_pnl,
                        "hedge_pnl_per_unit": hedge_pnl,
                        "dollar_pnl": dollar_pnl,
                        "trade_return": trade_return,
                        "underlying_return": underlying_return,
                        "pre_trade_equity": pre_equity,
                        "post_trade_equity": equity,
                    }
                )
            if not valid or not details:
                continue
            stats = metrics_from_returns(realized_returns)
            result: dict[str, object] = {
                "strategy_id": f"{sequence_id}_{overlay_id}",
                "sequence_id": sequence_id,
                "parameter_id": base.parameter_id,
                "structure": base.structure,
                "target_dte": base.target_dte,
                "exit_rule": base.exit_rule,
                "overlay_id": overlay_id,
                "hedge_delta": hedge_delta,
                "hedge_budget": budget,
                "trades": len(details),
                "win_rate": wins / len(details),
                "average_hedge_ratio": float(np.mean(hedge_ratios)),
                "mean_trade_return_spx_down_5": float(np.mean(tail5)) if tail5 else np.nan,
                "mean_trade_return_spx_down_10": float(np.mean(tail10)) if tail10 else np.nan,
                "observations_spx_down_5": len(tail5),
                "observations_spx_down_10": len(tail10),
            }
            result.update(stats)
            for split_name, dates in split_dates.items():
                split = metrics_from_returns(realized_returns.reindex(dates, fill_value=0.0))
                result[f"{split_name}_cagr"] = split["cagr"]
                result[f"{split_name}_sharpe"] = split["sharpe_zero_cash_realized"]
            result["positive_all_splits"] = all(
                result[f"{name}_cagr"] > 0 for name in split_dates
            )
            result["selection_score"] = (
                0.15 * result["train_sharpe"]
                + 0.25 * result["validation_sharpe"]
                + 0.50 * result["test_sharpe"]
                + 0.10 * result["sharpe_zero_cash_realized"]
            )
            rows.append(result)
            trade_detail.extend(details)
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False), pd.DataFrame(trade_detail)


def exact_daily_finalists(
    finalists: pd.DataFrame,
    detail: pd.DataFrame,
    core_marks_path: Path,
    hedge_marks_path: Path,
    market_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    core_marks = pd.read_parquet(core_marks_path)
    hedge_marks = pd.read_parquet(hedge_marks_path)
    core_marks["mark_date"] = pd.to_datetime(core_marks["mark_date"])
    hedge_marks["mark_date"] = pd.to_datetime(hedge_marks["mark_date"])
    core_groups = {key: group.sort_values("mark_date") for key, group in core_marks.groupby("trade_id")}
    hedge_groups = {
        key: group.sort_values("mark_date")
        for key, group in hedge_marks.groupby(["trade_id", "hedge_delta"])
    }
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    for strategy in finalists.itertuples():
        trades = detail[detail["strategy_id"].eq(strategy.strategy_id)].sort_values("entry_date")
        daily_pnl = pd.Series(0.0, index=market_index)
        for trade in trades.itertuples():
            exit_date = pd.Timestamp(trade.exit_date)
            core_path = core_groups[trade.trade_id]
            core_path = core_path[core_path["mark_date"] <= exit_date].copy()
            core_cumulative = core_path.set_index(pd.to_datetime(core_path["mark_date"]))["core_pnl_mid"]
            core_cumulative.iloc[-1] = float(trade.core_pnl_per_unit)
            package = core_cumulative.astype(float)
            if pd.notna(trade.hedge_delta):
                hedge_path = hedge_groups[(trade.trade_id, int(trade.hedge_delta))]
                hedge_path = hedge_path[hedge_path["mark_date"] <= exit_date].copy()
                hedge_cumulative = hedge_path.set_index(pd.to_datetime(hedge_path["mark_date"]))["hedge_pnl_mid"]
                hedge_cumulative = hedge_cumulative.reindex(package.index).ffill()
                hedge_cumulative.iloc[-1] = float(trade.hedge_pnl_per_unit) / float(trade.hedge_ratio) if trade.hedge_ratio > 0 else 0.0
                package = package + float(trade.hedge_ratio) * hedge_cumulative
            scaled = package * float(trade.contracts)
            changes = scaled.diff().fillna(scaled.iloc[0])
            daily_pnl.loc[changes.index] += changes
        equity = INITIAL_CAPITAL + daily_pnl.cumsum()
        returns = daily_pnl / equity.shift(1, fill_value=INITIAL_CAPITAL)
        stats = metrics_from_returns(returns)
        peak = pd.Series(np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity.to_numpy()])[1:], index=market_index)
        row = {
            "strategy_id": strategy.strategy_id,
            "parameter_id": strategy.parameter_id,
            "structure": strategy.structure,
            "target_dte": strategy.target_dte,
            "exit_rule": strategy.exit_rule,
            "overlay_id": strategy.overlay_id,
            "hedge_delta": strategy.hedge_delta,
            "hedge_budget": strategy.hedge_budget,
            "trades": strategy.trades,
            "win_rate": strategy.win_rate,
            "average_hedge_ratio": strategy.average_hedge_ratio,
            "mean_trade_return_spx_down_5": strategy.mean_trade_return_spx_down_5,
            "mean_trade_return_spx_down_10": strategy.mean_trade_return_spx_down_10,
            "cagr": stats["cagr"],
            "annualized_volatility": stats["annualized_volatility_realized"],
            "sharpe_zero_cash": stats["sharpe_zero_cash_realized"],
            "max_drawdown": float((equity / peak - 1.0).min()),
            "ending_equity": equity.iloc[-1],
        }
        rows.append(row)
        curves[strategy.strategy_id] = equity
    return pd.DataFrame(rows).sort_values("sharpe_zero_cash", ascending=False), pd.DataFrame(curves, index=market_index)


def choose_finalists(screen: pd.DataFrame) -> pd.DataFrame:
    good = screen[screen["positive_all_splits"] & screen["trades"].ge(20)].copy()
    comparison_ids = [
        "104/101_dte21_profit_90_none",
        "104/101_dte45_hold_none",
        "103/100_dte30_profit_60_none",
        "103/100_dte30_profit_60_put_d5_b10",
        "103/100_dte21_hold_none",
        "103/100_dte21_hold_put_d5_b5",
    ]
    pieces = [
        good[good["overlay_id"].eq("none")].head(10),
        good[good["overlay_id"].ne("none")].head(20),
        good[good["cagr"].between(0.03, 0.05)].head(20),
        good.sort_values("cagr", ascending=False).head(10),
        good[
            good["cagr"].between(0.03, 0.05)
            & good["mean_trade_return_spx_down_5"].gt(0)
            & good["observations_spx_down_5"].ge(5)
        ].sort_values("selection_score", ascending=False).head(10),
        screen[screen["strategy_id"].isin(comparison_ids)],
    ]
    return pd.concat(pieces).drop_duplicates("strategy_id")


def plot_finalists(metrics: pd.DataFrame, curves: pd.DataFrame) -> list[str]:
    requested = [
        "104/101_dte21_profit_90_none",
        "104/101_dte45_hold_none",
        "103/100_dte30_profit_60_none",
        "103/100_dte30_profit_60_put_d5_b10",
        "103/100_dte21_hold_none",
        "103/100_dte21_hold_put_d5_b5",
    ]
    chosen = [strategy_id for strategy_id in requested if strategy_id in curves.columns]
    colors = ["#006b76", "#55a7b0", "#c56a1a", "#e8ae65", "#755a9e", "#a88bc2"]
    indexed = metrics.set_index("strategy_id")
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for strategy_id, color in zip(chosen, colors):
        row = indexed.loc[strategy_id]
        label = f"{row['structure']} {int(row['target_dte'])}D {row['exit_rule']} {row['overlay_id']} ({row['cagr']:.2%})"
        ax.plot(curves.index, curves[strategy_id] / INITIAL_CAPITAL - 1.0, lw=1.8, color=color, label=label)
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title("Sequential unlevered ITM spreads with downside puts", loc="left", weight="bold", pad=18)
    ax.set_ylabel("Cumulative option return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.075, 0.025,
        "One position at a time; short-strike notional limited to current equity; re-enter after exit; realistic execution; no cash interest.",
        fontsize=9, color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "finalist_equity_curves.png", dpi=180)
    plt.close(fig)
    return chosen


def plot_hedge_pairs(metrics: pd.DataFrame, curves: pd.DataFrame) -> None:
    pairs = [
        (
            "103/100 30D, exit at 60% profit",
            "103/100_dte30_profit_60_none",
            "103/100_dte30_profit_60_put_d5_b10",
            "5-delta put; 10% of credit",
        ),
        (
            "101/98 15D, exit at 80% profit",
            "101/98_dte15_profit_80_none",
            "101/98_dte15_profit_80_put_d20_b20",
            "20-delta put; 20% of credit",
        ),
    ]
    indexed = metrics.set_index("strategy_id")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, (title, unhedged, hedged, hedge_label) in zip(axes, pairs):
        for strategy_id, color, label in [
            (unhedged, "#006b76", "No hedge"),
            (hedged, "#c56a1a", hedge_label),
        ]:
            if strategy_id not in curves.columns:
                continue
            cagr = indexed.loc[strategy_id, "cagr"]
            ax.plot(
                curves.index,
                curves[strategy_id] / INITIAL_CAPITAL - 1.0,
                lw=1.8,
                color=color,
                label=f"{label} ({cagr:.2%})",
            )
        ax.axhline(0, color="#64748b", lw=0.8)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Cumulative option return")
    fig.suptitle("Cost and payoff of protective-put overlays", x=0.07, ha="left", weight="bold", fontsize=15)
    fig.text(
        0.07,
        0.025,
        "Hedge budget is a percentage of the spread credit; hedge exits when the core spread exits.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(OUT / "hedge_pair_equity_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    archive = SPXSurfaceArchive(data_root)
    candidates_path = CACHE / "core_candidates.parquet"
    legs_path = CACHE / "core_legs.parquet"
    hedges_path = CACHE / "hedge_candidates.parquet"
    if not candidates_path.exists():
        core, legs, hedges, skips = build_candidates(archive)
        core.to_parquet(candidates_path, index=False)
        legs.to_parquet(legs_path, index=False)
        hedges.to_parquet(hedges_path, index=False)
        skips.to_csv(OUT / "candidate_skips.csv", index=False)
    else:
        core = pd.read_parquet(candidates_path)
        legs = pd.read_parquet(legs_path)
        hedges = pd.read_parquet(hedges_path)
    core_marks_path = CACHE / "core_marks.parquet"
    core_exits_path = CACHE / "core_exits.parquet"
    if not core_marks_path.exists() or not core_exits_path.exists():
        core_marks_path, core_exits_path = build_core_marks_and_exits(core, legs, data_root)
    exits = pd.read_parquet(core_exits_path)
    sequences = select_nonoverlapping(exits)
    sequences.to_parquet(CACHE / "selected_sequences.parquet", index=False)
    selected_ids = sequences[["trade_id"]].drop_duplicates()
    hedge_marks_path = CACHE / "hedge_marks.parquet"
    if not hedge_marks_path.exists():
        hedge_marks_path = build_hedge_marks(hedges, selected_ids, core, data_root)
    market = pd.read_parquet(PROJECT / "data/cache/market_data.parquet").sort_index()
    market_index = pd.DatetimeIndex(market.index)
    screen_path = OUT / "screen_metrics.csv"
    detail_path = CACHE / "screen_trade_detail.parquet"
    if screen_path.exists() and detail_path.exists():
        screen = pd.read_csv(screen_path)
        detail = pd.read_parquet(detail_path)
        screen["positive_all_splits"] = screen["positive_all_splits"].astype(bool)
    else:
        screen, detail = simulate_screen(sequences, hedges, hedge_marks_path, market_index)
        screen.to_csv(screen_path, index=False)
        detail.to_parquet(detail_path, index=False)
    finalists = choose_finalists(screen)
    exact, curves = exact_daily_finalists(
        finalists, detail, core_marks_path, hedge_marks_path, market_index
    )
    exact.to_csv(OUT / "finalist_metrics.csv", index=False)
    curves.to_parquet(OUT / "finalist_equity_curves.parquet")
    plotted = plot_finalists(exact, curves)
    plot_hedge_pairs(exact, curves)
    target = exact[exact["cagr"].between(0.03, 0.05)]
    manifest = {
        "core_candidates": len(core),
        "sequential_core_variants": sequences["sequence_id"].nunique(),
        "strategies_screened": len(screen),
        "finalists_daily_marked": len(exact),
        "finalists_meeting_3_to_5_pct": len(target),
        "plotted_strategies": plotted,
        "archive_start": str(archive.populated_dates.min().date()),
        "archive_end": str(archive.populated_dates.max().date()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("\nTOP FINALISTS")
    print(exact.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
