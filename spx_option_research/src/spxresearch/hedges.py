from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from .data_loader import SPXSurfaceArchive
from .execution import ExecutionModel
from .option_selector import clean_puts, select_expiration
from .optimizer import weekly_entry_dates


def hedge_parameter_grid(config: dict[str, Any], maximum_dte: int = 184) -> list[dict[str, Any]]:
    dtes = [int(value) for value in config["hedge_dtes"] if int(value) <= maximum_dte]
    parameters: list[dict[str, Any]] = []
    for dte in dtes:
        for delta in config["hedge_deltas"]:
            parameters.append({"family": "outright_put", "target_dte": dte, "delta_1": float(delta)})

    spread_dtes = [dte for dte in dtes if dte >= 14]
    for dte in spread_dtes:
        for long_delta in [5, 10, 20, 30, 40, 50]:
            for width in [2, 3, 5, 7.5, 10, 15, 20, 30]:
                parameters.append(
                    {
                        "family": "long_put_spread",
                        "target_dte": dte,
                        "delta_1": float(long_delta),
                        "width_pct": float(width),
                    }
                )
        for long_delta in [5, 10, 15, 20, 25, 30, 35, 40, 50]:
            for short_delta in [1, 5, 10, 15, 20, 25, 30, 35, 40]:
                if short_delta >= long_delta:
                    continue
                parameters.append(
                    {
                        "family": "long_put_spread",
                        "target_dte": dte,
                        "delta_1": float(long_delta),
                        "delta_2": float(short_delta),
                    }
                )

    fly_dtes = [dte for dte in dtes if dte >= 14]
    for dte in fly_dtes:
        for drawdown in [3, 5, 7.5, 10, 12.5, 15, 20, 25]:
            for wing in [2, 3, 5, 7.5]:
                parameters.append(
                    {
                        "family": "put_butterfly",
                        "target_dte": dte,
                        "center_drawdown": float(drawdown),
                        "width_pct": float(wing),
                    }
                )
                for lower_ratio in [0.5, 0.75]:
                    parameters.append(
                        {
                            "family": "broken_wing_butterfly",
                            "target_dte": dte,
                            "center_drawdown": float(drawdown),
                            "width_pct": float(wing),
                            "lower_wing_ratio": float(lower_ratio),
                        }
                    )

    ratio_dtes = [dte for dte in dtes if dte >= 14]
    for dte in ratio_dtes:
        for long_delta in [10, 20, 30, 40]:
            for width in [3, 5, 7.5, 10, 15]:
                parameters.append(
                    {
                        "family": "bounded_2x1",
                        "target_dte": dte,
                        "delta_1": float(long_delta),
                        "width_pct": float(width),
                    }
                )
    return parameters


def hedge_strategy_id(parameter: dict[str, Any]) -> str:
    parts = [parameter["family"], f"dte{int(parameter['target_dte'])}"]
    for key in ["delta_1", "delta_2", "center_drawdown", "width_pct", "lower_wing_ratio"]:
        if key in parameter:
            parts.append(f"{key}{float(parameter[key]):g}")
    return "_".join(parts).replace(".", "p")


def _nearest_delta(puts: pd.DataFrame, target: float) -> pd.Series:
    return puts.loc[(puts["delta"].abs() - target / 100.0).abs().idxmin()]


def _nearest_strike(puts: pd.DataFrame, target: float) -> pd.Series:
    return puts.loc[(puts["strike"] - target).abs().idxmin()]


def _terminal_bounds(legs: list[tuple[pd.Series, int]]) -> tuple[float, float, float]:
    strikes = [float(quote["strike"]) for quote, _ in legs]
    spots = np.unique(np.array([0.0, *strikes, max(strikes) * 1.25]))
    payoff = np.zeros(len(spots))
    for quote, quantity in legs:
        payoff += quantity * np.maximum(float(quote["strike"]) - spots, 0.0)
    return float(payoff.min()), float(payoff.max()), float(payoff[0])


def build_hedge_entry_candidates(
    archive: SPXSurfaceArchive,
    config: dict[str, Any],
    output_path: str | Path,
    *,
    dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Select bounded long-downside structures using entry-day information only."""
    entries = (
        pd.DatetimeIndex(dates)
        if dates is not None
        else weekly_entry_dates(archive.populated_dates, int(config["entry_weekday"]))
    )
    parameters = hedge_parameter_grid(config)
    by_dte: dict[int, list[dict[str, Any]]] = {}
    for parameter in parameters:
        by_dte.setdefault(int(parameter["target_dte"]), []).append(parameter)
    models = {
        name: ExecutionModel(float(values["spread_fraction"]), float(values["commission_per_contract"]))
        for name, values in config["execution_scenarios"].items()
    }
    tolerance = int(config.get("dte_tolerance", 4))
    records: list[dict[str, Any]] = []

    for number, entry_date in enumerate(entries, start=1):
        puts = clean_puts(archive.read(entry_date), pm_only=bool(config.get("pm_settlement_only", True)))
        if puts.empty:
            continue
        spot = float(puts["underlying_price"].dropna().median())
        for target_dte, dte_parameters in by_dte.items():
            try:
                expiration, actual_dte = select_expiration(
                    puts,
                    target_dte,
                    tolerance=tolerance,
                    pm_only=bool(config.get("pm_settlement_only", True)),
                )
            except LookupError:
                continue
            chain = puts[puts["expiration_date"].eq(expiration)].copy()
            if chain.empty:
                continue
            for parameter in dte_parameters:
                family = str(parameter["family"])
                try:
                    if family == "outright_put":
                        legs = [(_nearest_delta(chain, parameter["delta_1"]), 1)]
                    elif family == "long_put_spread":
                        upper = _nearest_delta(chain, parameter["delta_1"])
                        eligible_lower = chain[chain["strike"] < float(upper["strike"])]
                        if "delta_2" in parameter:
                            lower = _nearest_delta(eligible_lower, parameter["delta_2"])
                        else:
                            lower = _nearest_strike(
                                eligible_lower,
                                float(upper["strike"])
                                * (1.0 - parameter["width_pct"] / 100.0),
                            )
                        legs = [(upper, 1), (lower, -1)]
                    elif family in {"put_butterfly", "broken_wing_butterfly"}:
                        middle_target = spot * (1.0 - parameter["center_drawdown"] / 100.0)
                        width = spot * parameter["width_pct"] / 100.0
                        lower_ratio = float(parameter.get("lower_wing_ratio", 1.0))
                        middle = _nearest_strike(chain, middle_target)
                        upper = _nearest_strike(chain[chain["strike"] > float(middle["strike"])], float(middle["strike"]) + width)
                        lower = _nearest_strike(
                            chain[chain["strike"] < float(middle["strike"])],
                            float(middle["strike"]) - width * lower_ratio,
                        )
                        legs = [(upper, 1), (middle, -2), (lower, 1)]
                    elif family == "bounded_2x1":
                        upper = _nearest_delta(chain, parameter["delta_1"])
                        lower = _nearest_strike(
                            chain[chain["strike"] < float(upper["strike"])],
                            float(upper["strike"]) * (1.0 - parameter["width_pct"] / 100.0),
                        )
                        legs = [(upper, 2), (lower, -1)]
                    else:
                        continue
                except (KeyError, ValueError):
                    continue

                symbols = [str(quote["option_symbol"]) for quote, _ in legs]
                if len(symbols) != len(set(symbols)):
                    continue
                minimum_payoff, maximum_payoff, zero_payoff = _terminal_bounds(legs)
                if minimum_payoff < -1e-8 or zero_payoff < -1e-8:
                    continue
                row: dict[str, Any] = {
                    "hedge_trade_id": _hedge_trade_id(hedge_strategy_id(parameter), entry_date),
                    "hedge_strategy_id": hedge_strategy_id(parameter),
                    "family": family,
                    "entry_date": pd.Timestamp(entry_date),
                    "expiration_date": expiration,
                    "target_dte": target_dte,
                    "entry_dte": actual_dte,
                    "spot_entry": spot,
                    "delta_1": parameter.get("delta_1", math.nan),
                    "delta_2": parameter.get("delta_2", math.nan),
                    "center_drawdown": parameter.get("center_drawdown", math.nan),
                    "width_pct": parameter.get("width_pct", math.nan),
                    "lower_wing_ratio": parameter.get("lower_wing_ratio", math.nan),
                    "number_of_legs": len(legs),
                    "terminal_payoff_min_points": minimum_payoff,
                    "terminal_payoff_max_points": maximum_payoff,
                    "terminal_payoff_at_zero_points": zero_payoff,
                }
                for leg_number in range(1, 4):
                    if leg_number <= len(legs):
                        quote, quantity = legs[leg_number - 1]
                        row[f"leg{leg_number}_symbol"] = str(quote["option_symbol"])
                        row[f"leg{leg_number}_quantity"] = int(quantity)
                        row[f"leg{leg_number}_strike"] = float(quote["strike"])
                        row[f"leg{leg_number}_bid"] = float(quote["bid"])
                        row[f"leg{leg_number}_ask"] = float(quote["ask"])
                    else:
                        row[f"leg{leg_number}_symbol"] = None
                        row[f"leg{leg_number}_quantity"] = 0
                        row[f"leg{leg_number}_strike"] = math.nan
                        row[f"leg{leg_number}_bid"] = math.nan
                        row[f"leg{leg_number}_ask"] = math.nan
                valid = True
                for scenario, model in models.items():
                    entry_cash = sum(model.cash_flow(quote, quantity) for quote, quantity in legs)
                    debit = -entry_cash / model.multiplier
                    row[f"entry_debit_{scenario}"] = debit
                    if debit <= 0:
                        valid = False
                if valid:
                    records.append(row)
        if number % 25 == 0:
            print(f"Hedge selections: {number}/{len(entries)} dates, {len(records):,} trades", flush=True)

    result = pd.DataFrame(records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    return result


def build_hedge_trade_outcomes(
    connection: duckdb.DuckDBPyConnection,
    candidates_path: str | Path,
    output_path: str | Path,
) -> None:
    candidates = Path(candidates_path).resolve().as_posix().replace("'", "''")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target = output.as_posix().replace("'", "''")
    connection.execute(f"CREATE OR REPLACE TEMP VIEW hedge_candidates AS SELECT * FROM read_parquet('{candidates}')")
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE hedge_legs AS
        SELECT hedge_trade_id, leg1_symbol AS option_symbol, leg1_quantity AS quantity, leg1_strike AS strike FROM hedge_candidates
        UNION ALL
        SELECT hedge_trade_id, leg2_symbol, leg2_quantity, leg2_strike FROM hedge_candidates WHERE leg2_quantity <> 0
        UNION ALL
        SELECT hedge_trade_id, leg3_symbol, leg3_quantity, leg3_strike FROM hedge_candidates WHERE leg3_quantity <> 0
        """
    )
    connection.execute(
        f"""
        COPY (
            WITH surface_calendar AS (
                SELECT expiration_date, trade_date, min(dte) AS current_dte
                FROM surface
                GROUP BY expiration_date, trade_date
            ),
            rules(exit_rule) AS (VALUES ('hold'), ('roll_30d'), ('roll_half_life')),
            cohorts AS (
                SELECT DISTINCT entry_date, expiration_date, entry_dte
                FROM hedge_candidates
            ),
            exit_dates AS (
                SELECT
                    c.entry_date, c.expiration_date, c.entry_dte, r.exit_rule,
                    coalesce(
                        min(s.trade_date) FILTER (
                            WHERE s.current_dte <= CASE r.exit_rule
                                WHEN 'hold' THEN 0
                                WHEN 'roll_30d' THEN 30
                                WHEN 'roll_half_life' THEN c.entry_dte * 0.50
                            END
                        ),
                        max(s.trade_date)
                    ) AS exit_date
                FROM cohorts c
                CROSS JOIN rules r
                JOIN surface_calendar s
                  ON s.expiration_date = c.expiration_date
                 AND s.trade_date BETWEEN c.entry_date AND c.expiration_date
                WHERE r.exit_rule = 'hold' OR entry_dte > 30
                GROUP BY c.entry_date, c.expiration_date, c.entry_dte, r.exit_rule
            ),
            leg_marks AS (
                SELECT
                    c.*, e.exit_rule, e.exit_date,
                    s.dte AS current_dte, s.spot,
                    l.quantity, l.strike, s.mid, s.bid, s.ask,
                    CASE WHEN l.quantity > 0
                        THEN s.mid - 0.25 * (s.ask - s.bid)
                        ELSE s.mid + 0.25 * (s.ask - s.bid) END AS realistic_fill,
                    CASE WHEN l.quantity > 0 THEN s.bid ELSE s.ask END AS conservative_fill
                FROM hedge_candidates c
                JOIN exit_dates e
                  ON e.entry_date = c.entry_date
                 AND e.expiration_date = c.expiration_date
                 AND e.entry_dte = c.entry_dte
                JOIN hedge_legs l USING (hedge_trade_id)
                JOIN surface s
                  ON s.option_symbol = l.option_symbol
                 AND s.trade_date = e.exit_date
                WHERE s.bid >= 0 AND s.ask >= s.bid
            ),
            chosen AS (
                SELECT
                    * EXCLUDE (quantity, strike, mid, bid, ask, realistic_fill, conservative_fill),
                    count(*) AS legs_marked,
                    sum(quantity * mid) AS value_ideal,
                    sum(quantity * realistic_fill - abs(quantity) * 1.5 / 100.0)
                      AS value_realistic,
                    sum(quantity * conservative_fill - abs(quantity) * 2.5 / 100.0)
                      AS value_conservative,
                    sum(quantity * greatest(strike - spot, 0.0)) AS settlement_value
                FROM leg_marks
                GROUP BY ALL
            )
            SELECT
                * EXCLUDE (legs_marked),
                CASE WHEN current_dte = 0 THEN settlement_value ELSE value_ideal END AS exit_value_ideal,
                CASE WHEN current_dte = 0 THEN settlement_value ELSE value_realistic END AS exit_value_realistic,
                CASE WHEN current_dte = 0 THEN settlement_value ELSE value_conservative END AS exit_value_conservative,
                (CASE WHEN current_dte = 0 THEN settlement_value ELSE value_ideal END - entry_debit_ideal) * 100.0 AS pnl_ideal,
                (CASE WHEN current_dte = 0 THEN settlement_value ELSE value_realistic END - entry_debit_realistic) * 100.0 AS pnl_realistic,
                (CASE WHEN current_dte = 0 THEN settlement_value ELSE value_conservative END - entry_debit_conservative) * 100.0 AS pnl_conservative
            FROM chosen
            WHERE legs_marked = number_of_legs
        ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def summarize_hedge_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Coarse realized screen used before expensive daily marking."""
    frame = outcomes.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame["underlying_return"] = frame["spot"] / frame["spot_entry"] - 1.0
    years = max((frame["entry_date"].max() - frame["entry_date"].min()).days / 365.25, 1.0)
    records: list[dict[str, Any]] = []
    for (strategy_id, exit_rule), group in frame.groupby(
        ["hedge_strategy_id", "exit_rule"], sort=False
    ):
        debit = group["entry_debit_realistic"] * 100.0
        exit_value = group["exit_value_realistic"] * 100.0
        pnl = group["pnl_realistic"]
        down_5 = group["underlying_return"] <= -0.05
        down_10 = group["underlying_return"] <= -0.10
        total_cost = float(debit.sum())
        record: dict[str, Any] = {
            "hedge_strategy_id": strategy_id,
            "exit_rule": exit_rule,
            "family": group["family"].iloc[0],
            "target_dte": int(group["target_dte"].iloc[0]),
            "number_of_legs": int(group["number_of_legs"].iloc[0]),
            "delta_1": float(group["delta_1"].iloc[0]),
            "delta_2": float(group["delta_2"].iloc[0])
            if "delta_2" in group
            else math.nan,
            "center_drawdown": float(group["center_drawdown"].iloc[0]),
            "width_pct": float(group["width_pct"].iloc[0]),
            "trades": len(group),
            "annual_hedge_cost_one_lot": total_cost / years,
            "annual_hedge_cost_ideal_one_lot": float(
                (group["entry_debit_ideal"] * 100.0).sum()
            )
            / years,
            "annual_hedge_cost_conservative_one_lot": float(
                (group["entry_debit_conservative"] * 100.0).sum()
            )
            / years,
            "annual_pnl_ideal_one_lot": float(group["pnl_ideal"].sum()) / years,
            "annual_pnl_one_lot": float(pnl.sum()) / years,
            "annual_pnl_conservative_one_lot": float(group["pnl_conservative"].sum()) / years,
            "payoff_to_cost": float(exit_value.sum()) / total_cost if total_cost > 0 else math.nan,
            "mean_trade_pnl": float(pnl.mean()),
            "worst_trade_pnl": float(pnl.min()),
            "down_5_observations": int(down_5.sum()),
            "down_10_observations": int(down_10.sum()),
            "down_5_payoff_to_cost": float(exit_value[down_5].sum()) / total_cost
            if total_cost > 0
            else math.nan,
            "down_10_payoff_to_cost": float(exit_value[down_10].sum()) / total_cost
            if total_cost > 0
            else math.nan,
            "mean_payoff_when_down_5": float(exit_value[down_5].mean()) if down_5.any() else math.nan,
            "mean_payoff_when_down_10": float(exit_value[down_10].mean()) if down_10.any() else math.nan,
        }
        records.append(record)
    return pd.DataFrame(records)


def build_hedge_daily_mtm(
    connection: duckdb.DuckDBPyConnection,
    outcomes_path: str | Path,
    output_path: str | Path,
) -> None:
    """Mark selected multi-leg hedge cohorts daily using actual quotes."""
    outcomes = Path(outcomes_path).resolve().as_posix().replace("'", "''")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target = output.as_posix().replace("'", "''")
    connection.execute(f"CREATE OR REPLACE TEMP VIEW selected_hedges AS SELECT * FROM read_parquet('{outcomes}')")
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE selected_hedge_legs AS
        SELECT hedge_trade_id, exit_rule, leg1_symbol AS option_symbol, leg1_quantity AS quantity FROM selected_hedges
        UNION ALL
        SELECT hedge_trade_id, exit_rule, leg2_symbol, leg2_quantity FROM selected_hedges WHERE leg2_quantity <> 0
        UNION ALL
        SELECT hedge_trade_id, exit_rule, leg3_symbol, leg3_quantity FROM selected_hedges WHERE leg3_quantity <> 0
        """
    )
    connection.execute(
        f"""
        COPY (
            WITH leg_marks AS (
                SELECT
                    o.hedge_trade_id, o.hedge_strategy_id, o.exit_rule,
                    o.entry_date, o.exit_date, o.entry_debit_realistic,
                    o.exit_value_realistic, o.number_of_legs,
                    s.trade_date AS mark_date, l.quantity, s.mid
                FROM selected_hedges o
                JOIN selected_hedge_legs l USING (hedge_trade_id, exit_rule)
                JOIN surface s ON s.option_symbol = l.option_symbol
                  AND s.trade_date BETWEEN o.entry_date AND o.exit_date
            ),
            marks AS (
                SELECT
                    hedge_trade_id, hedge_strategy_id, exit_rule,
                    entry_date, exit_date, entry_debit_realistic,
                    exit_value_realistic, number_of_legs, mark_date,
                    count(*) AS legs_marked,
                    sum(quantity * mid) AS mid_value
                FROM leg_marks
                GROUP BY ALL
            ),
            paths AS (
                SELECT *,
                    ((CASE WHEN mark_date = exit_date THEN exit_value_realistic ELSE mid_value END)
                      - entry_debit_realistic) * 100.0 AS cumulative_trade_pnl
                FROM marks
                WHERE legs_marked = number_of_legs
            ),
            changes AS (
                SELECT *,
                    cumulative_trade_pnl - lag(cumulative_trade_pnl, 1, 0.0)
                      OVER (PARTITION BY hedge_trade_id, exit_rule ORDER BY mark_date)
                      AS daily_trade_pnl
                FROM paths
            )
            SELECT
                hedge_strategy_id, exit_rule, mark_date,
                sum(daily_trade_pnl) AS daily_hedge_pnl_one_lot,
                count(DISTINCT hedge_trade_id) AS positions_marked
            FROM changes
            GROUP BY hedge_strategy_id, exit_rule, mark_date
            ORDER BY hedge_strategy_id, exit_rule, mark_date
        ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def _hedge_trade_id(strategy_id: str, entry_date: pd.Timestamp) -> str:
    value = f"{strategy_id}|{pd.Timestamp(entry_date).date()}"
    return hashlib.blake2b(value.encode("utf-8"), digest_size=10).hexdigest()
