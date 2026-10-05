from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_loader import SPXSurfaceArchive
from .execution import ExecutionModel
from .option_selector import clean_puts, select_expiration


@dataclass(frozen=True)
class CoreParameter:
    target_dte: int
    short_delta: int
    width_method: str
    width_value: float

    @property
    def strategy_id(self) -> str:
        value = f"core_dte{self.target_dte}_sd{self.short_delta}_{self.width_method}{self.width_value:g}"
        return value.replace(".", "p")


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def core_parameters(config: dict[str, Any]) -> list[CoreParameter]:
    widths = [
        *(('delta', float(value)) for value in config["delta_widths"]),
        *(('percent', float(value)) for value in config["percent_widths"]),
        *(('points', float(value)) for value in config["point_widths"]),
    ]
    return [
        CoreParameter(int(dte), int(delta), method, value)
        for dte in config["short_dtes"]
        for delta in config["short_deltas"]
        for method, value in widths
    ]


def weekly_entry_dates(dates: pd.DatetimeIndex, preferred_weekday: int = 0) -> pd.DatetimeIndex:
    """One entry per ISO week, preferring the configured weekday or the next session."""
    frame = pd.DataFrame({"date": pd.DatetimeIndex(dates).sort_values()})
    iso = frame["date"].dt.isocalendar()
    frame["year"] = iso.year
    frame["week"] = iso.week
    selected: list[pd.Timestamp] = []
    for _, group in frame.groupby(["year", "week"], sort=True):
        preferred = group[group["date"].dt.weekday >= preferred_weekday]
        selected.append(pd.Timestamp((preferred if not preferred.empty else group).iloc[0]["date"]))
    return pd.DatetimeIndex(selected)


def build_core_entry_candidates(
    archive: SPXSurfaceArchive,
    config: dict[str, Any],
    output_path: str | Path,
    *,
    dates: Iterable[pd.Timestamp] | None = None,
    parameters: Iterable[CoreParameter] | None = None,
) -> pd.DataFrame:
    """Select every core spread at each weekly entry using only that day's quotes."""
    parameter_values = list(parameters) if parameters is not None else core_parameters(config)
    entry_dates = (
        pd.DatetimeIndex(dates)
        if dates is not None
        else weekly_entry_dates(archive.populated_dates, int(config["entry_weekday"]))
    )
    by_dte: dict[int, list[CoreParameter]] = {}
    for parameter in parameter_values:
        by_dte.setdefault(parameter.target_dte, []).append(parameter)

    models = {
        name: ExecutionModel(float(values["spread_fraction"]), float(values["commission_per_contract"]))
        for name, values in config["execution_scenarios"].items()
    }
    records: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    pm_only = bool(config.get("pm_settlement_only", True))
    tolerance = int(config.get("dte_tolerance", 4))
    capital = float(config["initial_capital"])
    portfolio_risk = float(config["risk_budget_fraction"])

    for date_number, entry_date in enumerate(entry_dates, start=1):
        chain = archive.read(entry_date)
        puts = clean_puts(chain, pm_only=pm_only)
        if puts.empty:
            skips.append({"entry_date": entry_date, "reason": "no_valid_puts"})
            continue

        for target_dte, dte_parameters in by_dte.items():
            try:
                expiration, actual_dte = select_expiration(
                    puts,
                    target_dte,
                    tolerance=tolerance,
                    pm_only=pm_only,
                )
            except LookupError:
                skips.append(
                    {
                        "entry_date": entry_date,
                        "target_dte": target_dte,
                        "reason": "no_expiration_in_tolerance",
                    }
                )
                continue

            expiration_puts = puts[puts["expiration_date"].eq(expiration)].copy()
            expiration_puts.sort_values("strike", inplace=True)
            strikes = expiration_puts["strike"].to_numpy(float)
            abs_deltas = expiration_puts["delta"].abs().to_numpy(float)
            spot = float(expiration_puts["underlying_price"].dropna().median())

            for short_delta in sorted({p.short_delta for p in dte_parameters}):
                short_index = int(np.nanargmin(np.abs(abs_deltas - short_delta / 100.0)))
                short = expiration_puts.iloc[short_index]
                short_strike = float(short["strike"])
                eligible_index = np.flatnonzero(strikes < short_strike)
                if not len(eligible_index):
                    continue

                for parameter in (p for p in dte_parameters if p.short_delta == short_delta):
                    if parameter.width_method == "delta":
                        target = max((short_delta - parameter.width_value) / 100.0, 0.005)
                        distance = np.abs(abs_deltas[eligible_index] - target)
                    elif parameter.width_method == "percent":
                        target = short_strike * (1.0 - parameter.width_value / 100.0)
                        distance = np.abs(strikes[eligible_index] - target)
                    else:
                        target = short_strike - parameter.width_value
                        distance = np.abs(strikes[eligible_index] - target)
                    long_index = int(eligible_index[int(np.nanargmin(distance))])
                    long = expiration_puts.iloc[long_index]
                    long_strike = float(long["strike"])
                    width = short_strike - long_strike
                    if width <= 0:
                        continue

                    row: dict[str, Any] = {
                        "trade_id": _trade_id(parameter.strategy_id, entry_date),
                        "pair_id": _pair_id(entry_date, str(short["option_symbol"]), str(long["option_symbol"])),
                        "base_strategy_id": parameter.strategy_id,
                        "entry_date": entry_date,
                        "expiration_date": expiration,
                        "target_dte": target_dte,
                        "entry_dte": actual_dte,
                        "short_delta_target": short_delta,
                        "short_delta_actual": abs(float(short["delta"])),
                        "long_delta_actual": abs(float(long["delta"])),
                        "width_method": parameter.width_method,
                        "width_value": parameter.width_value,
                        "spot_entry": spot,
                        "short_strike": short_strike,
                        "long_strike": long_strike,
                        "width_points": width,
                        "short_symbol": str(short["option_symbol"]),
                        "long_symbol": str(long["option_symbol"]),
                        "settlement": str(short["settlement"]),
                        "short_bid": float(short["bid"]),
                        "short_ask": float(short["ask"]),
                        "long_bid": float(long["bid"]),
                        "long_ask": float(long["ask"]),
                    }
                    for scenario, model in models.items():
                        entry_cash = model.cash_flow(short, -1) + model.cash_flow(long, 1)
                        credit = entry_cash / model.multiplier
                        max_loss = (width - credit) * model.multiplier
                        concurrent_tranches = max(1, math.ceil(target_dte / 7))
                        risk_per_trade = capital * portfolio_risk / concurrent_tranches
                        # Continuous contract equivalents make the broad screen
                        # comparable across widths. Whole-contract feasibility is
                        # retained separately for implementation review.
                        contracts = risk_per_trade / max_loss if max_loss > 0 else 0.0
                        row[f"entry_credit_{scenario}"] = credit
                        row[f"max_loss_{scenario}"] = max_loss
                        row[f"contracts_{scenario}"] = contracts
                        row[f"whole_contracts_{scenario}"] = (
                            math.floor(risk_per_trade / max_loss) if max_loss > 0 else 0
                        )
                        row[f"minimum_capital_one_contract_{scenario}"] = (
                            max_loss * concurrent_tranches / portfolio_risk
                            if max_loss > 0 and portfolio_risk > 0
                            else math.inf
                        )
                    if row["entry_credit_realistic"] > 0 and row["contracts_realistic"] > 0:
                        records.append(row)

        if date_number % 25 == 0:
            print(
                f"Candidate selections: {date_number}/{len(entry_dates)} entry dates, "
                f"{len(records):,} spreads",
                flush=True,
            )

    result = pd.DataFrame(records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    if skips:
        pd.DataFrame(skips).to_csv(output.with_name("core_entry_skips.csv"), index=False)
    return result


def _trade_id(strategy_id: str, entry_date: pd.Timestamp) -> str:
    text = f"{strategy_id}|{pd.Timestamp(entry_date).date()}"
    return hashlib.blake2b(text.encode("utf-8"), digest_size=10).hexdigest()


def _pair_id(entry_date: pd.Timestamp, short_symbol: str, long_symbol: str) -> str:
    text = f"{pd.Timestamp(entry_date).date()}|{short_symbol}|{long_symbol}"
    return hashlib.blake2b(text.encode("utf-8"), digest_size=10).hexdigest()
