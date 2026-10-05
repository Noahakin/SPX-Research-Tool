from __future__ import annotations

import calendar
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
OUT = PROJECT / "results/itm_structures"
CACHE = PROJECT / "data/cache/itm_structures"
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.execution import ExecutionModel
from spxresearch.option_selector import clean_puts, select_expiration
from spxresearch.warehouse import ResearchWarehouse


INITIAL_CAPITAL = 1_000_000.0
TOTAL_RISK_BUDGET = 0.05
MODEL = ExecutionModel(spread_fraction=0.25, commission_per_contract=1.50)
TARGET_DTES = [15, 21, 30, 45, 60, 75]
MONEYNESS_SHORTS = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]
DELTA_SHORTS = [50, 55, 60, 65, 70, 75, 80]
BUTTERFLY_ANCHORS = [1.01, 1.03, 1.05]


def third_friday(year: int, month: int) -> pd.Timestamp:
    weeks = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in weeks if week[calendar.FRIDAY]]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def entry_schedules(dates: pd.DatetimeIndex) -> dict[str, list[pd.Timestamp]]:
    available = set(dates)
    weekly = [date for date in dates if date.weekday() == calendar.FRIDAY]
    monthly: list[pd.Timestamp] = []
    for period in pd.period_range(dates.min(), dates.max(), freq="M"):
        scheduled = third_friday(period.year, period.month)
        choices = [
            scheduled - pd.Timedelta(days=offset)
            for offset in range(4)
            if scheduled - pd.Timedelta(days=offset) in available
        ]
        if choices:
            monthly.append(max(choices))
    return {"weekly": weekly, "monthly": sorted(set(monthly))}


def nearest_ratio(puts: pd.DataFrame, spot: float, ratio: float) -> pd.Series:
    return puts.loc[(puts["strike"].astype(float) - spot * ratio).abs().idxmin()]


def nearest_delta(puts: pd.DataFrame, spot: float, target: float) -> pd.Series:
    eligible = puts[
        puts["strike"].between(spot * 0.995, spot * 1.055)
        & puts["delta"].notna()
    ]
    if eligible.empty:
        raise LookupError("no ITM/ATM put within the 100%-105% strike region")
    return eligible.loc[(eligible["delta"].abs() - target / 100.0).abs().idxmin()]


def payoff_limits(legs: list[tuple[pd.Series, int]], entry_cash: float) -> tuple[float, float]:
    strikes = sorted({float(quote["strike"]) for quote, _ in legs})
    test_spots = [0.0, *strikes, max(strikes) * 2.0]
    pnls = []
    for spot in test_spots:
        payoff = sum(qty * max(float(quote["strike"]) - spot, 0.0) * 100.0 for quote, qty in legs)
        pnls.append(entry_cash + payoff)
    return float(max(pnls)), float(-min(pnls))


def add_trade(
    trade_rows: list[dict[str, object]],
    leg_rows: list[dict[str, object]],
    *,
    entry_date: pd.Timestamp,
    expiration: pd.Timestamp,
    cadence: str,
    target_dte: int,
    family: str,
    structure: str,
    selector: str,
    anchor: float,
    spot: float,
    legs: list[tuple[pd.Series, int, str]],
) -> None:
    strikes = [float(quote["strike"]) for quote, _, _ in legs]
    if len(set(strikes)) != len(strikes):
        return
    quoted_legs = [(quote, qty) for quote, qty, _ in legs]
    entry_cash = sum(MODEL.cash_flow(quote, qty) for quote, qty in quoted_legs)
    max_profit, max_loss = payoff_limits(quoted_legs, entry_cash)
    if max_profit <= 0 or max_loss <= 0 or not np.isfinite(max_loss):
        return
    actual_dte = int((expiration - entry_date).days)
    concurrent = max(1, math.ceil(actual_dte / (7 if cadence == "weekly" else 30)))
    risk_per_trade = INITIAL_CAPITAL * TOTAL_RISK_BUDGET / concurrent
    contracts = risk_per_trade / max_loss
    parameter_id = f"{family}_{structure}_{cadence}_dte{target_dte}_{selector}{anchor:g}"
    trade_id = f"{parameter_id}_{entry_date:%Y%m%d}"
    short_anchor = next((quote for quote, qty, role in legs if role == "short_anchor"), None)
    trade_rows.append(
        {
            "trade_id": trade_id,
            "parameter_id": parameter_id,
            "family": family,
            "structure": structure,
            "cadence": cadence,
            "target_dte": target_dte,
            "actual_dte": actual_dte,
            "selector": selector,
            "anchor": anchor,
            "entry_date": entry_date,
            "expiration_date": expiration,
            "spot_entry": spot,
            "entry_cash_per_unit": entry_cash,
            "max_profit_per_unit": max_profit,
            "max_loss_per_unit": max_loss,
            "contracts": contracts,
            "leg_count": len(legs),
            "short_strike_ratio": float(short_anchor["strike"]) / spot if short_anchor is not None else np.nan,
            "short_abs_delta": abs(float(short_anchor["delta"])) if short_anchor is not None else np.nan,
        }
    )
    for leg_id, (quote, qty, role) in enumerate(legs):
        leg_rows.append(
            {
                "trade_id": trade_id,
                "leg_id": leg_id,
                "role": role,
                "quantity": qty,
                "option_symbol": str(quote["option_symbol"]).strip(),
                "strike": float(quote["strike"]),
            }
        )


def construct_universe(archive: SPXSurfaceArchive) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades: list[dict[str, object]] = []
    legs: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    schedules = entry_schedules(archive.populated_dates)
    for cadence, entry_dates in schedules.items():
        for number, entry_date in enumerate(entry_dates, start=1):
            chain = archive.read(entry_date)
            clean = clean_puts(chain, pm_only=True)
            for target_dte in TARGET_DTES:
                try:
                    expiration, _ = select_expiration(
                        clean, target_dte, tolerance=4, pm_only=True
                    )
                except LookupError as error:
                    skips.append({"cadence": cadence, "entry_date": entry_date, "target_dte": target_dte, "reason": str(error)})
                    continue
                puts = clean[clean["expiration_date"].eq(expiration)].copy()
                spot = float(puts["underlying_price"].dropna().median())

                # Fixed 3%-of-spot verticals from 100/97 through 105/102.
                for short_ratio in MONEYNESS_SHORTS:
                    short = nearest_ratio(puts, spot, short_ratio)
                    lower = puts[puts["strike"] < float(short["strike"])]
                    if lower.empty:
                        continue
                    long = nearest_ratio(lower, spot, short_ratio - 0.03)
                    add_trade(
                        trades, legs, entry_date=entry_date, expiration=expiration,
                        cadence=cadence, target_dte=target_dte, family="vertical",
                        structure=f"{short_ratio*100:.0f}/{(short_ratio-0.03)*100:.0f}",
                        selector="m", anchor=short_ratio * 100, spot=spot,
                        legs=[(short, -1, "short_anchor"), (long, 1, "long")],
                    )

                # Delta-selected ITM/ATM shorts, still paired with a put 3% of spot lower.
                for short_delta in DELTA_SHORTS:
                    try:
                        short = nearest_delta(puts, spot, short_delta)
                    except LookupError:
                        continue
                    lower = puts[puts["strike"] < float(short["strike"])]
                    if lower.empty:
                        continue
                    long = nearest_ratio(lower, spot, float(short["strike"]) / spot - 0.03)
                    add_trade(
                        trades, legs, entry_date=entry_date, expiration=expiration,
                        cadence=cadence, target_dte=target_dte, family="vertical",
                        structure=f"delta{short_delta}_3wide", selector="d", anchor=short_delta,
                        spot=spot, legs=[(short, -1, "short_anchor"), (long, 1, "long")],
                    )

                # Butterfly and condor variants anchored in the same ITM region.
                fly_specs = {
                    "long_fly_3x3": ([0, -3, -6], [1, -2, 1]),
                    "short_fly_3x3": ([0, -3, -6], [-1, 2, -1]),
                    "long_broken_fly_3x6": ([0, -3, -9], [1, -2, 1]),
                    "short_broken_fly_3x6": ([0, -3, -9], [-1, 2, -1]),
                    "long_condor_2x2x2": ([0, -2, -4, -6], [1, -1, -1, 1]),
                    "short_condor_2x2x2": ([0, -2, -4, -6], [-1, 1, 1, -1]),
                }
                for upper_ratio in BUTTERFLY_ANCHORS:
                    for name, (offsets, quantities) in fly_specs.items():
                        selected: list[tuple[pd.Series, int, str]] = []
                        for index, (offset, qty) in enumerate(zip(offsets, quantities)):
                            quote = nearest_ratio(puts, spot, upper_ratio + offset / 100.0)
                            selected.append((quote, qty, "leg"))
                        add_trade(
                            trades, legs, entry_date=entry_date, expiration=expiration,
                            cadence=cadence, target_dte=target_dte, family="butterfly",
                            structure=f"{name}_upper{upper_ratio*100:.0f}", selector="m",
                            anchor=upper_ratio * 100, spot=spot, legs=selected,
                        )
            if number % 50 == 0:
                print(f"Selected {cadence} entries {number}/{len(entry_dates)}", flush=True)
    return pd.DataFrame(trades), pd.DataFrame(legs), pd.DataFrame(skips)


def build_paths(trades: pd.DataFrame, legs: pd.DataFrame, data_root: Path) -> tuple[Path, Path]:
    marks_path = CACHE / "trade_marks.parquet"
    exits_path = CACHE / "trade_exits.parquet"
    daily_path = CACHE / "strategy_daily.parquet"
    qmarks = marks_path.resolve().as_posix().replace("'", "''")
    qexits = exits_path.resolve().as_posix().replace("'", "''")
    qdaily = daily_path.resolve().as_posix().replace("'", "''")
    rules = pd.DataFrame(
        [
            ("hold", "final", 0.0, "all"),
            ("profit_25", "profit", 0.25, "all"),
            ("profit_50", "profit", 0.50, "all"),
            ("profit_75", "profit", 0.75, "all"),
            ("half_life", "dte_fraction", 0.50, "all"),
            ("exit_14dte", "dte", 14.0, "all"),
            ("exit_7dte", "dte", 7.0, "all"),
            ("short_delta_40", "delta", 0.40, "vertical"),
            ("short_delta_25", "delta", 0.25, "vertical"),
        ],
        columns=["exit_rule", "rule_kind", "rule_value", "applies_to"],
    )
    with ResearchWarehouse(
        PROJECT / "data/cache/research.duckdb", data_root, threads=8, memory_limit="12GB"
    ) as warehouse:
        con = warehouse.connection
        con.register("selected_trades", trades)
        con.register("selected_legs", legs)
        con.register("exit_rules", rules)
        con.execute(
            f"""
            COPY (
                WITH leg_marks AS (
                    SELECT
                        t.trade_id, t.parameter_id, t.family, t.structure, t.cadence,
                        t.target_dte, t.actual_dte, t.selector, t.anchor,
                        t.entry_date, t.expiration_date, t.entry_cash_per_unit,
                        t.max_profit_per_unit, t.max_loss_per_unit, t.contracts,
                        t.leg_count, l.leg_id, l.role, l.quantity, l.strike,
                        s.trade_date AS mark_date, s.dte AS current_dte,
                        s.spot, s.bid, s.ask, s.mid, s.delta
                    FROM selected_trades t
                    JOIN selected_legs l USING (trade_id)
                    JOIN surface s ON s.option_symbol = l.option_symbol
                      AND s.trade_date BETWEEN t.entry_date AND t.expiration_date
                ), aggregated AS (
                    SELECT
                        trade_id, min(parameter_id) AS parameter_id,
                        min(family) AS family, min(structure) AS structure,
                        min(cadence) AS cadence, min(target_dte) AS target_dte,
                        min(actual_dte) AS actual_dte, min(selector) AS selector,
                        min(anchor) AS anchor, min(entry_date) AS entry_date,
                        min(expiration_date) AS expiration_date,
                        min(entry_cash_per_unit) AS entry_cash_per_unit,
                        min(max_profit_per_unit) AS max_profit_per_unit,
                        min(max_loss_per_unit) AS max_loss_per_unit,
                        min(contracts) AS contracts, mark_date,
                        min(current_dte) AS current_dte,
                        sum(quantity * mid * 100.0) AS position_value_mid,
                        sum(
                            CASE WHEN mark_date = expiration_date OR current_dte = 0
                                 THEN quantity * greatest(strike - spot, 0.0) * 100.0
                                 ELSE quantity * mid * 100.0
                                      - abs(quantity) * 0.25 * (ask - bid) * 100.0
                                      - abs(quantity) * 1.50 END
                        ) AS close_cash_realistic,
                        max(abs(delta)) FILTER (WHERE role = 'short_anchor') AS short_abs_delta_mark,
                        count(*) AS legs_marked, max(leg_count) AS required_legs
                    FROM leg_marks
                    GROUP BY trade_id, mark_date
                    HAVING count(*) = max(leg_count)
                )
                SELECT *,
                    entry_cash_per_unit + position_value_mid AS pnl_mid,
                    entry_cash_per_unit + close_cash_realistic AS pnl_realistic,
                    mark_date = max(mark_date) OVER (PARTITION BY trade_id) AS is_final
                FROM aggregated
            ) TO '{qmarks}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        con.execute(
            f"""
            COPY (
                WITH marks AS (SELECT * FROM read_parquet('{qmarks}')),
                eligible AS (
                    SELECT m.*, r.exit_rule, r.rule_kind, r.rule_value,
                        CASE
                            WHEN m.is_final THEN true
                            WHEN r.rule_kind = 'final' THEN false
                            WHEN r.rule_kind = 'profit' THEN
                                m.mark_date > m.entry_date
                                AND m.pnl_realistic >= r.rule_value * m.max_profit_per_unit
                            WHEN r.rule_kind = 'dte' THEN
                                m.mark_date > m.entry_date AND m.current_dte <= r.rule_value
                            WHEN r.rule_kind = 'dte_fraction' THEN
                                m.mark_date > m.entry_date
                                AND m.current_dte <= m.actual_dte * r.rule_value
                            WHEN r.rule_kind = 'delta' THEN
                                m.mark_date > m.entry_date
                                AND m.short_abs_delta_mark <= r.rule_value
                            ELSE false
                        END AS should_exit
                    FROM marks m
                    CROSS JOIN exit_rules r
                    WHERE r.applies_to = 'all' OR r.applies_to = m.family
                )
                SELECT
                    * EXCLUDE (is_final, should_exit), mark_date AS exit_date,
                    pnl_realistic AS realized_pnl_per_unit
                FROM eligible
                WHERE should_exit
                QUALIFY row_number() OVER (
                    PARTITION BY trade_id, exit_rule ORDER BY mark_date
                ) = 1
            ) TO '{qexits}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        con.execute(
            f"""
            COPY (
                WITH marks AS (SELECT * FROM read_parquet('{qmarks}')),
                exits AS (SELECT * FROM read_parquet('{qexits}')),
                paths AS (
                    SELECT
                        e.parameter_id, e.family, e.structure, e.cadence,
                        e.target_dte, e.selector, e.anchor, e.trade_id,
                        e.exit_rule, m.mark_date,
                        CASE WHEN m.mark_date = e.exit_date
                             THEN e.realized_pnl_per_unit * e.contracts
                             ELSE m.pnl_mid * e.contracts END AS cumulative_trade_pnl
                    FROM exits e
                    JOIN marks m USING (trade_id)
                    WHERE m.mark_date <= e.exit_date
                ), changes AS (
                    SELECT *, cumulative_trade_pnl
                        - lag(cumulative_trade_pnl, 1, 0.0)
                          OVER (PARTITION BY trade_id, exit_rule ORDER BY mark_date)
                          AS daily_pnl
                    FROM paths
                )
                SELECT
                    parameter_id, family, structure, cadence, target_dte,
                    selector, anchor, exit_rule, mark_date,
                    sum(daily_pnl) AS daily_pnl
                FROM changes
                GROUP BY ALL
                ORDER BY parameter_id, exit_rule, mark_date
            ) TO '{qdaily}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    return exits_path, daily_path


def slice_stats(returns: pd.Series) -> tuple[float, float]:
    wealth = (1.0 + returns).prod()
    years = len(returns) / 252.0
    cagr = wealth ** (1.0 / years) - 1.0 if wealth > 0 and years > 0 else np.nan
    std = returns.std(ddof=1)
    sharpe = returns.mean() / std * math.sqrt(252.0) if std > 0 else np.nan
    return float(cagr), float(sharpe)


def summarize(
    trades: pd.DataFrame, exits_path: Path, daily_path: Path, market_index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(daily_path)
    daily["mark_date"] = pd.to_datetime(daily["mark_date"])
    daily["strategy_id"] = daily["parameter_id"] + "_" + daily["exit_rule"]
    exits = pd.read_parquet(exits_path)
    exits["strategy_id"] = exits["parameter_id"] + "_" + exits["exit_rule"]
    trade_stats = exits.groupby("strategy_id").agg(
        trades=("trade_id", "nunique"),
        win_rate=("realized_pnl_per_unit", lambda values: float((values > 0).mean())),
        average_exit_dte=("current_dte", "mean"),
    )
    entry_stats = trades.groupby("parameter_id").agg(
        average_short_delta=("short_abs_delta", "mean"),
        average_short_ratio=("short_strike_ratio", "mean"),
    )
    first, second = int(len(market_index) * 0.60), int(len(market_index) * 0.80)
    splits = {
        "train": market_index[:first],
        "validation": market_index[first:second],
        "test": market_index[second:],
    }
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    for strategy_id, group in daily.groupby("strategy_id", sort=False):
        pnl = group.groupby("mark_date")["daily_pnl"].sum().reindex(market_index, fill_value=0.0)
        equity = INITIAL_CAPITAL + pnl.cumsum()
        prior = equity.shift(1, fill_value=INITIAL_CAPITAL)
        returns = pnl / prior
        years = len(returns) / 252.0
        peak = pd.Series(
            np.maximum.accumulate(np.r_[INITIAL_CAPITAL, equity.to_numpy()])[1:],
            index=market_index,
        )
        row = group.iloc[0][
            ["parameter_id", "family", "structure", "cadence", "target_dte", "selector", "anchor", "exit_rule"]
        ].to_dict()
        ending_ratio = equity.iloc[-1] / INITIAL_CAPITAL
        row.update(
            {
                "strategy_id": strategy_id,
                "cagr": ending_ratio ** (1.0 / years) - 1.0 if ending_ratio > 0 else np.nan,
                "annualized_volatility": returns.std(ddof=1) * math.sqrt(252.0),
                "sharpe_zero_cash": returns.mean() / returns.std(ddof=1) * math.sqrt(252.0),
                "max_drawdown": float((equity / peak - 1.0).min()),
                "ending_equity": float(equity.iloc[-1]),
            }
        )
        for split_name, split_index in splits.items():
            cagr, sharpe = slice_stats(returns.reindex(split_index, fill_value=0.0))
            row[f"{split_name}_cagr"] = cagr
            row[f"{split_name}_sharpe"] = sharpe
        rows.append(row)
        curves[strategy_id] = equity
    summary = pd.DataFrame(rows).set_index("strategy_id")
    summary = summary.join(trade_stats).join(entry_stats, on="parameter_id").reset_index()
    summary["positive_all_splits"] = (
        summary["train_cagr"].gt(0)
        & summary["validation_cagr"].gt(0)
        & summary["test_cagr"].gt(0)
    )
    summary["selection_score"] = (
        0.15 * summary["train_sharpe"]
        + 0.25 * summary["validation_sharpe"]
        + 0.50 * summary["test_sharpe"]
        + 0.10 * summary["sharpe_zero_cash"]
    )
    summary.sort_values("selection_score", ascending=False, inplace=True)
    curve_frame = pd.DataFrame(curves, index=market_index)
    return summary, curve_frame


def plot_top(summary: pd.DataFrame, curves: pd.DataFrame) -> list[str]:
    qualifying = summary[
        summary["cagr"].between(0.03, 0.05)
        & summary["positive_all_splits"]
        & summary["trades"].ge(50)
    ]
    selected: list[str] = []
    for family in ["vertical", "butterfly"]:
        family_rows = qualifying[qualifying["family"].eq(family)].head(2)
        selected.extend(family_rows["strategy_id"].tolist())
    if len(selected) < 4:
        selected.extend(
            strategy for strategy in qualifying["strategy_id"] if strategy not in selected
        )
    selected = selected[:4]
    colors = ["#006b76", "#55a7b0", "#c56a1a", "#e8ae65"]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for strategy_id, color in zip(selected, colors):
        row = summary.set_index("strategy_id").loc[strategy_id]
        label = (
            f"{row['structure']} | {row['cadence']} | {int(row['target_dte'])} DTE | "
            f"{row['exit_rule']} ({row['cagr']:.1%})"
        )
        ax.plot(curves.index, curves[strategy_id] / INITIAL_CAPITAL - 1.0, label=label, color=color, lw=1.8)
    ax.axhline(0, color="#64748b", lw=0.8)
    ax.set_title("ITM SPX structures: leading option-only equity curves", loc="left", weight="bold", pad=18)
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(
        0.075, 0.025,
        "Weekly or monthly entries; 5% total concurrent maximum-loss budget; continuous contracts; realistic execution; no cash interest.\n"
        "Exploratory selection requires 3–5% CAGR, positive train/validation/test returns, and at least 50 trades.",
        fontsize=9, color="#475569",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / "top_equity_curves.png", dpi=180)
    plt.close(fig)
    return selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    data_root = WORKSPACE / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
    archive = SPXSurfaceArchive(data_root)
    trades, legs, skips = construct_universe(archive)
    trades.to_parquet(CACHE / "trades.parquet", index=False)
    legs.to_parquet(CACHE / "legs.parquet", index=False)
    skips.to_csv(OUT / "selection_skips.csv", index=False)
    exits_path, daily_path = build_paths(trades, legs, data_root)
    market = pd.read_parquet(PROJECT / "data/cache/market_data.parquet").sort_index()
    summary, curves = summarize(trades, exits_path, daily_path, pd.DatetimeIndex(market.index))
    summary.to_csv(OUT / "strategy_metrics.csv", index=False)
    curves.to_parquet(CACHE / "equity_curves.parquet")
    selected = plot_top(summary, curves)
    qualifying = summary[
        summary["cagr"].between(0.03, 0.05)
        & summary["positive_all_splits"]
        & summary["trades"].ge(50)
    ]
    top_verticals = qualifying[qualifying["family"].eq("vertical")].head(25)
    top_flies = qualifying[qualifying["family"].eq("butterfly")].head(25)
    top_verticals.to_csv(OUT / "top_verticals.csv", index=False)
    top_flies.to_csv(OUT / "top_butterflies.csv", index=False)
    manifest = {
        "trade_candidates": len(trades),
        "leg_rows": len(legs),
        "strategies_tested": len(summary),
        "qualifying_strategies": len(qualifying),
        "qualifying_verticals": int(qualifying["family"].eq("vertical").sum()),
        "qualifying_butterflies": int(qualifying["family"].eq("butterfly").sum()),
        "plotted_strategies": selected,
        "archive_start": str(archive.populated_dates.min().date()),
        "archive_end": str(archive.populated_dates.max().date()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("\nTOP VERTICALS")
    print(top_verticals.head(10).to_string(index=False))
    print("\nTOP BUTTERFLIES")
    print(top_flies.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
