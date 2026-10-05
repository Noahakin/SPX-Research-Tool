"""Independently audit premium-funded put combinations and the merged library.

No portfolio-engine pricing, sizing, or statistic functions are imported. The
validator reconstructs entry eligibility, contract quantities, terminal P&L,
daily NAV, and statistics directly from audited quotes and the cash index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from organized_spx_data import DEFAULT_OUTPUT as BASE_OUTPUT, load_cash


DEFAULT_OUTPUT = BASE_OUTPUT.parent / "organized_spx_combinations"
TENORS = (3, 7, 14, 21, 28, 42, 56)
QUOTE_KEYS = ["snapshot_date", "option_symbol"]


def require(condition, label: str) -> None:
    if not np.all(condition):
        raise AssertionError(label)


def close(actual, expected, label: str, atol: float = 1e-10) -> float:
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    finite = np.isfinite(actual) & np.isfinite(expected)
    error = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0
    require(np.allclose(actual, expected, rtol=2e-11, atol=atol, equal_nan=True),
            f"{label}: maximum finite absolute difference {error:.17g}")
    return error


def compare_metrics(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    require(len(actual) == len(expected), f"{label}: row count")
    for name in expected:
        a, b = actual[name].reset_index(drop=True), expected[name].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(b):
            close(a, b, f"{label}: {name}")
        else:
            if name == "category":
                b = b.replace({"Both": "SPX with overlay"}) if label == "Original metrics" else b
            require(a.fillna("").eq(b.fillna("")), f"{label}: {name}")


def verify_statistics(nav: np.ndarray, metrics: pd.DataFrame,
                      dates: pd.DatetimeIndex, initial: float) -> float:
    maximum = 0.0
    elapsed = (dates[-1] - dates[0]).days / 365.2425
    for first in range(0, nav.shape[1], 192):
        values = nav[:, first:first + 192]
        daily = values / np.vstack([np.full((1, values.shape[1]), initial), values[:-1]]) - 1
        mean, sd = daily.mean(axis=0), daily.std(axis=0, ddof=1)
        sharpe = np.divide(mean * np.sqrt(252), sd, out=np.full(len(sd), np.nan), where=sd != 0)
        peaks = np.maximum.accumulate(np.vstack([np.full((1, values.shape[1]), initial), values]), axis=0)[1:]
        expected = dict(cagr=(values[-1] / initial) ** (1 / elapsed) - 1,
                        annualized_volatility=sd * np.sqrt(252), daily_sharpe=sharpe,
                        annualized_arithmetic_return=mean * 252, ending_equity=values[-1],
                        total_return=values[-1] / initial - 1,
                        max_drawdown=(values / peaks - 1).min(axis=0),
                        missing_daily_marks=(~np.isfinite(values)).sum(axis=0),
                        daily_observations=np.isfinite(values).sum(axis=0))
        selected = metrics.iloc[first:first + values.shape[1]]
        for name, target in expected.items():
            maximum = max(maximum, close(selected[name], target, f"Daily statistic {name}", atol=1e-9))
        close(np.prod(1 + daily, axis=0), values[-1] / initial, "Daily return compounding")
    return maximum


def verify(output: Path, base: Path) -> dict:
    config = json.loads((output / "run.json").read_text(encoding="utf-8"))
    require(config["hedge_notional_cap"] == 1.0, "Confirmed 100% hedge notional cap")
    require(config["premium_budget_is_limit"] is True, "Premium fraction is a budget ceiling")
    initial = float(config["initial_equity"])
    require(initial == 1_000_000, "Initial equity")
    combo = pd.read_csv(output / "combination_metrics.csv", dtype={"folder": str})
    metrics = pd.read_csv(output / "strategy_metrics.csv", dtype={"folder": str})
    old_metrics = pd.read_csv(base / "strategy_metrics.csv", dtype={"folder": str})
    with np.load(output / "curves.npz") as saved:
        dates = pd.DatetimeIndex(saved["dates"])
        ids, all_nav = saved["strategy_ids"], saved["equity"]
    with np.load(base / "curves.npz") as saved:
        old_dates, old_ids, old_nav = saved["dates"], saved["strategy_ids"], saved["equity"]
    require((len(old_metrics), len(combo), len(metrics)) == (2940, 5292, 8232), "Complete library counts")
    require(all_nav.shape == (2510, 8232), "Complete daily library shape")
    require(np.array_equal(dates.to_numpy(), old_dates), "Original date order")
    require(np.array_equal(ids[:2940], old_ids), "Original ID order")
    require(np.array_equal(all_nav[:, :2940], old_nav), "Original curves bitwise unchanged")
    require(np.array_equal(ids, metrics.strategy_id.to_numpy()), "Merged metric/curve ID order")
    require(np.array_equal(ids[2940:], combo.strategy_id.to_numpy()), "Combination metric/curve ID order")
    nav = all_nav[:, 2940:]
    # The standalone archive is redundant after a verified merge. Low-storage
    # installations may retain only the merged archive; no disk copy is needed.
    if (output / "combination_curves.npz").exists():
        with np.load(output / "combination_curves.npz") as saved:
            require(np.array_equal(saved["dates"], dates.to_numpy()), "Standalone combination date order")
            require(np.array_equal(saved["strategy_ids"], ids[2940:]), "Standalone combination ID order")
            require(np.array_equal(saved["equity"], nav), "Standalone combination NAV")
    compare_metrics(metrics.iloc[:2940], old_metrics, "Original metrics")
    compare_metrics(metrics.iloc[2940:], combo, "Combination metrics")
    cash = load_cash()
    require(dates.equals(cash.loc[config["start_date"]:config["end_date"]].index), "Every research cash session")
    require(np.isfinite(all_nav).all() and (all_nav > 0).all(), "Finite positive daily equity")
    require(not metrics.strategy_id.duplicated().any(), "Unique strategy IDs")
    require(combo.category.eq("Both") & combo.underlying.eq(0) & combo.side.eq(-1)
            & combo.notional_multiple.eq(1) & combo.hedge_notional_cap.eq(1), "Combination identity and no SPX holding")
    expected_grid = {(high, high - width, hedge, fraction, tenor)
                     for high in range(90, 111) for width in (1, 2, 3, 5, 10)
                     if high - width >= 90 for hedge in (0, 5, 10)
                     for fraction in (.1, .25, .5) for tenor in TENORS}
    actual_grid = set(combo[["primary_pct", "secondary_pct", "hedge_width_pct", "premium_fraction", "tenor_days"]]
                      .itertuples(index=False, name=None))
    require(actual_grid == expected_grid and len(actual_grid) == len(combo), "Independent full parameter grid")
    require(combo.width_pct.eq(combo.primary_pct - combo.secondary_pct), "Base spread target widths")
    require(combo.hedge_primary_pct.eq(combo.secondary_pct), "Shared middle put target")
    buffer_strategy = combo.hedge_width_pct.gt(0)
    require(combo.loc[~buffer_strategy, "hedge_secondary_pct"].isna(), "Single-put hedge has no low leg")
    require(combo.loc[buffer_strategy, "hedge_secondary_pct"].eq(
        combo.loc[buffer_strategy, "secondary_pct"] - combo.loc[buffer_strategy, "hedge_width_pct"]), "Buffer low targets")
    stat_error = verify_statistics(all_nav, metrics, dates, initial)
    schedule = pd.read_parquet(output / "schedule.parquet")
    entries = pd.read_parquet(output / "entries.parquet")
    quotes = pd.read_parquet(output / "audited_quotes.parquet")
    require(len(schedule) == 2282 and not schedule.roll_id.duplicated().any(), "Unique complete roll schedule")
    require(not entries.duplicated(["roll_id", "ratio"]).any(), "Unique entry legs")
    require(not quotes.duplicated(QUOTE_KEYS).any(), "Unique exact-contract daily quotes")
    require(not quotes.unresolved.any(), "No unresolved quote marks")
    quote_index = quotes.set_index(QUOTE_KEYS)
    roll_index = pd.Index(schedule.roll_id)
    entry_roll = roll_index.get_indexer(entries.roll_id)
    entry_leg = np.rint(entries.ratio.to_numpy() * 100).astype(int) - 80
    require((entry_roll >= 0) & (entry_leg >= 0) & (entry_leg < 31), "Entry grid indices")
    require(len(entries) == int(schedule.has_expiry.sum()) * 31, "Complete 31-leg quoted entry grid")
    source_keys = pd.MultiIndex.from_arrays([entries.entry_date, entries.option_symbol], names=QUOTE_KEYS)
    source = quote_index.reindex(source_keys)
    for name in ("bid", "ask", "strike"):
        close(entries[name], source[name], f"Raw entry {name} equals audited source")
    require(np.array_equal(entries.expiration_date.to_numpy(), source.expiration_date.to_numpy()), "Exact entry expiry identity")
    shape = (len(schedule), 31)
    strike, bid, ask = (np.full(shape, np.nan) for _ in range(3))
    symbols = np.full(shape, "", dtype=object)
    entry_estimated = np.ones(shape, dtype=bool)
    strike[entry_roll, entry_leg] = entries.strike.to_numpy()
    bid[entry_roll, entry_leg] = source.used_bid.to_numpy()
    ask[entry_roll, entry_leg] = source.used_ask.to_numpy()
    symbols[entry_roll, entry_leg] = entries.option_symbol.to_numpy()
    entry_estimated[entry_roll, entry_leg] = source.estimated.to_numpy(bool)
    spot = cash.reindex(schedule.entry_date).to_numpy()
    terminal_spot = cash.reindex(schedule.valuation_end).to_numpy()
    close(entries.spot_entry, spot[entry_roll], "Cash-index entry sizing")
    mid = (bid + ask) / 2
    friction = .25 * (ask - bid) + .015
    target_error = np.abs(strike / spot[:, None] - np.arange(80, 111)[None, :] / 100)
    valid_leg = (np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & (ask > 0)
                 & (ask >= bid) & ~entry_estimated & (target_error <= .005 + 1e-12))
    terminal_marks = np.maximum(strike - terminal_spot[:, None], 0)
    open_rolls = np.flatnonzero(schedule.has_expiry & schedule.expiration_date.gt(dates[-1]))
    for ri in open_rolls:
        keys = pd.MultiIndex.from_product([[dates[-1]], symbols[ri]], names=QUOTE_KEYS)
        terminal_marks[ri] = quote_index.reindex(keys).used_mid.to_numpy()
    strategy_index = pd.Index(combo.strategy_id)
    specs = combo.drop_duplicates("structure_id").reset_index(drop=True)
    require(len(specs) == 756, "Independent combination structure count")
    spec_index = pd.Index(specs.structure_id)
    slot_for_strategy = spec_index.get_indexer(combo.structure_id)
    columns_for_tenor = {}
    for tenor in TENORS:
        selected = combo.loc[combo.tenor_days.eq(tenor)]
        require(len(selected) == 756 and not selected.structure_id.duplicated().any(), "Every structure per tenor")
        columns_for_tenor[tenor] = strategy_index.get_indexer(selected.set_index("structure_id").loc[spec_index].strategy_id)
    hi = specs.primary_pct.to_numpy(int) - 80
    middle = specs.secondary_pct.to_numpy(int) - 80
    is_buffer = specs.hedge_width_pct.to_numpy(int) > 0
    low = np.where(is_buffer, middle - specs.hedge_width_pct.to_numpy(int), middle)
    fractions = specs.premium_fraction.to_numpy(float)
    nroll, nspec, nstrategy = len(schedule), len(specs), len(combo)
    seen = np.zeros((nroll, nspec), bool)
    before_map = np.full((nroll, nspec), np.nan)
    quantity_map = np.zeros((nroll, nspec))
    traded_map = np.zeros((nroll, nspec), bool)
    estimate_map = np.zeros((nroll, nspec), np.int32)
    state = np.full(nstrategy, initial)
    previous_entry = np.full(nstrategy, -1, int)
    sums = {key: np.zeros(nstrategy) for key in
            ("cycles", "traded", "quantity", "credit", "spend", "spend_fraction", "estimates", "capped")}
    maxima = {key: np.zeros(nstrategy) for key in ("quantity", "strike_error")}
    dte_min, dte_max = np.full(nstrategy, np.inf), np.full(nstrategy, -np.inf)
    errors = {key: 0.0 for key in ("terminal", "quantity", "retained", "entry", "compounding", "daily")}
    ledger_rows = 0
    for batch in pq.ParquetFile(output / "trade_ledger.parquet").iter_batches(batch_size=75600):
        ledger = batch.to_pandas()
        ri, ci = roll_index.get_indexer(ledger.roll_id), strategy_index.get_indexer(ledger.strategy_id)
        require((ri >= 0) & (ci >= 0), "Recognized ledger roll and strategy")
        si = slot_for_strategy[ci]
        flat = ri * nspec + si
        require(len(np.unique(flat)) == len(flat) and not seen[ri, si].any(), "Every roll/spec ledger cell once")
        seen[ri, si] = True
        scheduled = schedule.iloc[ri].reset_index(drop=True)
        for name in ("target_dte", "actual_dte", "entry_date", "expiration_date", "valuation_end"):
            require(ledger[name].eq(scheduled[name]) | (ledger[name].isna() & scheduled[name].isna()), f"Ledger schedule {name}")
        require(ledger.target_dte.to_numpy() == combo.tenor_days.to_numpy()[ci], "Strategy tenor matches schedule")
        h, m, l, buffer = hi[si], middle[si], low[si], is_buffer[si]
        credit = (mid[ri, h] - friction[ri, h]) - (mid[ri, m] + friction[ri, m])
        debit = mid[ri, m] + friction[ri, m] - np.where(buffer, mid[ri, l] - friction[ri, l], 0)
        hedge_mid = mid[ri, m] - np.where(buffer, mid[ri, l], 0)
        eligible = (scheduled.has_expiry.to_numpy() & (credit > 0) & (debit > 0)
                    & (credit <= strike[ri, h] - strike[ri, m] + 1e-12)
                    & (hedge_mid >= 0) & (~buffer | (hedge_mid <= strike[ri, m] - strike[ri, l] + 1e-12))
                    & valid_leg[ri, h] & valid_leg[ri, m] & (~buffer | valid_leg[ri, l])
                    & (strike[ri, h] > strike[ri, m]) & (~buffer | (strike[ri, m] > strike[ri, l])))
        quantity = np.zeros(len(ledger))
        np.divide(fractions[si] * credit, debit, out=quantity, where=eligible)
        quantity = np.minimum(quantity, 1)
        require(ledger.traded.to_numpy(bool) == eligible, "Independent cycle eligibility")
        errors["quantity"] = max(errors["quantity"], close(ledger.hedge_quantity_ratio, quantity, "Entry hedge sizing"))
        close(ledger.premium_fraction, fractions[si], "Ledger premium budget")
        close(ledger.hedge_notional_cap, np.ones(len(ledger)), "Ledger hedge cap")
        close(ledger.net_short_credit_points, credit, "All-in short spread credit")
        close(ledger.hedge_debit_points, debit, "All-in downside hedge debit")
        retained = np.where(eligible, credit - quantity * debit, 0)
        errors["retained"] = max(errors["retained"], close(ledger.net_credit_retained_points, retained, "Net retained credit"))
        require((quantity >= 0) & (quantity <= 1) & (eligible | (quantity == 0)), "Quantity bounds and cash cycles")
        require(quantity[eligible] * debit[eligible] <= fractions[si[eligible]] * credit[eligible] + 1e-10, "Premium budget ceiling")
        require(retained[eligible] >= (1 - fractions[si[eligible]]) * credit[eligible] - 1e-10, "Minimum retained credit")
        uncapped = eligible & (quantity < 1)
        close(quantity[uncapped] * debit[uncapped], fractions[si[uncapped]] * credit[uncapped], "Uncapped exact budget spend")
        for prefix, leg, present in (("primary", h, np.ones(len(ledger), bool)),
                                      ("secondary", m, np.ones(len(ledger), bool)),
                                      ("hedge_secondary", l, buffer)):
            for suffix, source_array in (("strike", strike), ("bid", bid), ("ask", ask)):
                close(ledger[f"{prefix}_{suffix}"], np.where(present, source_array[ri, leg], np.nan), f"Ledger {prefix} {suffix}")
        close(ledger.spot_entry, spot[ri], "Ledger cash index")
        require(ledger.final_position_open.to_numpy() == scheduled.expiration_date.gt(dates[-1]).to_numpy(), "Open position flag")
        terminal = np.where(eligible, (retained - terminal_marks[ri, h]
                            + (1 + quantity) * terminal_marks[ri, m]
                            - np.where(buffer, quantity * terminal_marks[ri, l], 0)) / spot[ri], 0)
        errors["terminal"] = max(errors["terminal"], close(ledger.terminal_return, terminal, "Independent terminal P&L"))
        entry_pos = dates.get_indexer(ledger.entry_date)
        require(entry_pos >= 0, "Ledger entries are cash sessions")
        grouped_dates = pd.Series(entry_pos).groupby(ci, sort=False)
        require(grouped_dates.diff().dropna().gt(0), "Chronological ledger cycles within batch")
        first = ~pd.Series(ci).duplicated().to_numpy()
        require(entry_pos[first] > previous_entry[ci[first]], "Chronological ledger cycles across batches")
        local = pd.Series(1 + terminal).groupby(ci, sort=False).cumprod().to_numpy()
        after = state[ci] * local
        before = after / (1 + terminal)
        errors["compounding"] = max(errors["compounding"], close(ledger.entry_equity, before, "Independent entry compounding", 1e-7),
                                     close(ledger.ending_equity, after, "Independent terminal compounding", 1e-7))
        _, reverse_last = np.unique(ci[::-1], return_index=True)
        last = len(ci) - 1 - reverse_last
        state[ci[last]], previous_entry[ci[last]] = after[last], entry_pos[last]
        entry_cost = np.where(eligible, (friction[ri, h] + (1 + quantity) * friction[ri, m]
                                          + np.where(buffer, quantity * friction[ri, l], 0)) / spot[ri], 0)
        errors["entry"] = max(errors["entry"], close(nav[entry_pos, ci], before * (1 - entry_cost), "Day-one entry friction", 1e-7))
        before_map[ri, si], quantity_map[ri, si], traded_map[ri, si] = before, quantity, eligible
        estimate_map[ri, si] = ledger.estimated_quote_count.to_numpy()
        spend_fraction = np.zeros(len(ledger))
        np.divide(quantity * debit, credit, out=spend_fraction, where=eligible)
        additions = dict(cycles=np.ones(len(ledger)), traded=eligible, quantity=quantity,
                         credit=np.where(eligible, credit / spot[ri], 0),
                         spend=np.where(eligible, quantity * debit / spot[ri], 0), spend_fraction=spend_fraction,
                         estimates=ledger.estimated_quote_count.to_numpy(),
                         capped=eligible & np.isclose(quantity, 1, rtol=0, atol=1e-12))
        for name, values in additions.items():
            np.add.at(sums[name], ci, values)
        np.maximum.at(maxima["quantity"], ci, quantity)
        error = np.maximum(target_error[ri, h], np.maximum(target_error[ri, m], np.where(buffer, target_error[ri, l], 0)))
        np.maximum.at(maxima["strike_error"], ci, np.where(eligible, error, 0))
        np.minimum.at(dte_min, ci, np.where(eligible, ledger.actual_dte, np.inf))
        np.maximum.at(dte_max, ci, np.where(eligible, ledger.actual_dte, -np.inf))
        ledger_rows += len(ledger)
    require(seen.all() and ledger_rows == 1_725_192, "Complete ledger coverage")
    errors["compounding"] = max(errors["compounding"], close(state, nav[-1], "Final compounded equity", 1e-7))
    count = sums["traded"]
    def mean_of(name):
        return np.divide(sums[name], count, out=np.zeros(nstrategy), where=count > 0)
    aggregate = dict(cycles=sums["cycles"], traded_cycles=count, cash_cycles=sums["cycles"] - count,
                     trade_fraction=count / sums["cycles"], estimated_quote_count=sums["estimates"],
                     actual_dte_min=np.where(count > 0, dte_min, np.nan),
                     actual_dte_max=np.where(count > 0, dte_max, np.nan),
                     max_strike_error_pct_points=maxima["strike_error"] * 100,
                     mean_hedge_notional_multiple=mean_of("quantity"), max_hedge_notional_multiple=maxima["quantity"],
                     hedge_cap_binding_cycles=sums["capped"], mean_premium_fraction_spent=mean_of("spend_fraction"),
                     mean_short_credit_pct_notional=mean_of("credit"), mean_hedge_spend_pct_notional=mean_of("spend"))
    for name, values in aggregate.items():
        close(combo[name], values, f"Independent aggregate {name}")
    print(f"CORE passed: {ledger_rows:,} ledger rows, original curves preserved, all statistics reconciled", flush=True)
    daily_values = 0
    daily_seen = {tenor: np.zeros(len(dates), bool) for tenor in TENORS}
    for ri, cycle in enumerate(schedule.itertuples(index=False)):
        held = np.flatnonzero((dates >= cycle.entry_date) & (dates <= cycle.valuation_end))
        written = held if cycle.valuation_end == dates[-1] else held[dates[held] < cycle.valuation_end]
        columns = columns_for_tenor[int(cycle.target_dte)]
        require(not daily_seen[int(cycle.target_dte)][written].any(), "Daily NAV reconstructed exactly once")
        daily_seen[int(cycle.target_dte)][written] = True
        quantity, traded, before = quantity_map[ri], traded_map[ri], before_map[ri]
        cumulative = np.zeros((len(held), nspec))
        estimated_counts = np.zeros(nspec, int)
        if cycle.has_expiry:
            keys = pd.MultiIndex.from_product([dates[held], symbols[ri]], names=QUOTE_KEYS)
            observed = quote_index.reindex(keys)
            marks = observed.used_mid.to_numpy(copy=True).reshape(len(held), 31)
            estimated = observed.estimated.fillna(False).to_numpy(dtype=bool, copy=True).reshape(len(held), 31)
            if dates[held[-1]] == cycle.expiration_date:
                marks[-1] = np.maximum(strike[ri] - cash.loc[cycle.expiration_date], 0)
                estimated[-1] = False
            credit = (mid[ri, hi] - friction[ri, hi]) - (mid[ri, middle] + friction[ri, middle])
            debit = mid[ri, middle] + friction[ri, middle] - np.where(is_buffer, mid[ri, low] - friction[ri, low], 0)
            cumulative = (credit - quantity * debit - marks[:, hi]
                          + (1 + quantity) * marks[:, middle]
                          - np.where(is_buffer[None, :], quantity[None, :] * marks[:, low], 0)) / spot[ri]
            cumulative[:, ~traded] = 0
            estimated_leg_counts = estimated.sum(axis=0)
            estimated_counts = (estimated_leg_counts[hi] + estimated_leg_counts[middle]
                                + np.where(is_buffer, estimated_leg_counts[low], 0)) * traded
        require(np.array_equal(estimate_map[ri], estimated_counts), f"Exact held estimate counts: {cycle.roll_id}")
        expected_nav = before[None, :] * (1 + cumulative[:len(written)])
        errors["daily"] = max(errors["daily"], close(nav[np.ix_(written, columns)], expected_nav,
                                                       f"Daily quote reconstruction: {cycle.roll_id}", 1e-7))
        daily_values += expected_nav.size
        if (ri + 1) % 400 == 0:
            print(f"Daily reconstruction: {ri + 1:,}/{nroll:,} rolls", flush=True)
    require(all(mask.all() for mask in daily_seen.values()) and daily_values == nav.size, "Every combination NAV independently repriced")
    for name, expected in config["input_artifact_sha256"].items():
        with (output / name).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        require(actual == expected, f"Input artifact SHA256: {name}")
    return dict(status="passed", strategies=len(metrics), combination_strategies=len(combo),
                preserved_original_strategies=len(old_metrics), original_curves_bitwise_unchanged=True,
                daily_sessions=len(dates), daily_nav_values=int(all_nav.size),
                independently_repriced_daily_nav_values=int(daily_values), independently_repriced_ledger_rows=ledger_rows,
                maximum_terminal_return_error=errors["terminal"], maximum_daily_nav_error=errors["daily"],
                maximum_hedge_quantity_error=errors["quantity"], maximum_retained_credit_error=errors["retained"],
                maximum_entry_nav_error=errors["entry"], maximum_compounding_equity_error=errors["compounding"],
                maximum_recomputed_statistic_error=stat_error, quote_estimates=int(quotes.estimated.sum()),
                unresolved_quotes=int(quotes.unresolved.sum()), missing_daily_nav_values=int((~np.isfinite(all_nav)).sum()),
                hedge_notional_cap=1.0, protocol_checks=["entry-only premium sizing", "net-credit budget ceiling",
                "100% added hedge notional cap", "entry spread price bounds", "all-in hedge debit",
                "shared middle quantity 1+q", "zero quantity and zero P&L for ineligible cycles",
                "fixed hedge quantity through roll", "no SPX holding", "cash intrinsic settlement",
                "final audited marks", "entry friction", "roll compounding", "all daily statistics"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base", type=Path, default=BASE_OUTPUT)
    args = parser.parse_args()
    try:
        result = verify(args.output, args.base)
    except Exception as exc:
        print(json.dumps(dict(status="failed", error=f"{type(exc).__name__}: {exc}")), flush=True)
        raise
    # Preserve the last complete validation report if writing ever fails.
    target = args.output / "validation.json"
    partial = target.with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    partial.replace(target)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
