"""Independently reconcile the saved chart portfolios to their quote/roll ledger."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from organized_spx_data import DEFAULT_OUTPUT, load_cash


def main() -> None:
    output = DEFAULT_OUTPUT
    metrics = pd.read_csv(output / "strategy_metrics.csv")
    trades = pd.read_parquet(output / "trade_ledger.parquet")
    quotes = pd.read_parquet(output / "audited_quotes.parquet")
    config = json.loads((output / "run.json").read_text())
    cash = load_cash()
    archive = np.load(output / "curves.npz")
    dates, ids, curves = pd.DatetimeIndex(archive["dates"]), archive["strategy_ids"], archive["equity"]
    initial = config["initial_equity"]
    assert len(metrics) == 2940 and curves.shape == (2510, 2940)
    assert np.array_equal(ids, metrics.strategy_id.to_numpy())
    assert metrics.groupby(["category", "folder"]).size().eq(7).all()
    assert metrics.groupby(["category", "folder"]).ngroups == 420
    assert dates.equals(cash.loc[config["start_date"]:config["end_date"]].index)
    assert np.isfinite(curves).all(), "Incomplete daily quotes require explicit unavailable metrics"

    # Independently recompute every metric from saved NAV, including day-one costs.
    returns = curves / np.vstack([np.full((1, curves.shape[1]), initial), curves[:-1]]) - 1
    elapsed = (dates[-1]-dates[0]).days / 365.2425
    sd = returns.std(axis=0, ddof=1)
    expected_cagr = (curves[-1]/initial)**(1/elapsed)-1
    expected_vol = sd*np.sqrt(252)
    expected_sharpe = returns.mean(axis=0)/sd*np.sqrt(252)
    np.testing.assert_allclose(metrics.cagr, expected_cagr, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(metrics.annualized_volatility, expected_vol, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(metrics.daily_sharpe, expected_sharpe, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.prod(1+returns, axis=0), curves[-1]/initial, rtol=1e-11)

    # Reprice every terminal ledger P&L from entry fills, cash settlement, or
    # end-date observed option mark. This avoids trusting saved terminal returns.
    source = trades.merge(metrics[["strategy_id", "side", "underlying", "width_pct"]], on="strategy_id", validate="many_to_one")
    spot_out = source.valuation_end.map(cash).to_numpy(float)
    spot_in = source.spot_entry.to_numpy(float)
    traded = source.traded.to_numpy(bool)
    is_spread = source.width_pct.to_numpy() > 0
    side = source.side.to_numpy()
    primary = np.maximum(source.primary_strike.to_numpy() - spot_out, 0)
    secondary = np.maximum(source.secondary_strike.to_numpy() - spot_out, 0)
    final = quotes[quotes.snapshot_date.eq(dates[-1])].set_index(["expiration_date", "strike"]).used_mid
    if final.index.duplicated().any():
        raise AssertionError("End quote expiry/strike lookup is ambiguous")
    opened = source.final_position_open.to_numpy(bool) & traded
    for name, target in (("primary_strike", primary), ("secondary_strike", secondary)):
        mask = opened & (is_spread if name == "secondary_strike" else True)
        keys = pd.MultiIndex.from_arrays([source.loc[mask, "expiration_date"], source.loc[mask, name]])
        target[mask] = final.reindex(keys).to_numpy()
    mid_a = (source.primary_bid.to_numpy()+source.primary_ask.to_numpy())/2
    mid_b = (source.secondary_bid.to_numpy()+source.secondary_ask.to_numpy())/2
    cost_a = .25*(source.primary_ask.to_numpy()-source.primary_bid.to_numpy())+.015
    cost_b = .25*(source.secondary_ask.to_numpy()-source.secondary_bid.to_numpy())+.015
    option = (side*(primary-mid_a)-cost_a + np.where(is_spread, -side*(secondary-mid_b)-cost_b, 0))/spot_in
    expected_terminal = np.where(traded, option, 0) + source.underlying.to_numpy()*(spot_out/spot_in-1)
    errors = np.abs(source.terminal_return.to_numpy()-expected_terminal)
    np.testing.assert_allclose(source.terminal_return, expected_terminal, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(source.ending_equity, source.entry_equity*(1+expected_terminal), rtol=1e-11, atol=1e-8)

    lookup = {key: pos for pos, key in enumerate(ids)}
    day_lookup = {day: pos for pos, day in enumerate(dates)}
    row_ix = source.entry_date.map(day_lookup).to_numpy()
    col_ix = source.strategy_id.map(lookup).to_numpy()
    entry_cost = np.where(traded, (cost_a+np.where(is_spread,cost_b,0))/spot_in,0)
    expected_entry = source.entry_equity.to_numpy()*(1-entry_cost)
    np.testing.assert_allclose(curves[row_ix,col_ix],expected_entry,rtol=1e-11,atol=1e-8)
    ledger_compounding = (source.assign(multiplier=1+expected_terminal).groupby("strategy_id").multiplier.prod().reindex(ids))
    np.testing.assert_allclose(ledger_compounding.to_numpy()*initial,curves[-1],rtol=1e-11,atol=1e-7)
    cash_counts = source.assign(cash_cycle=~traded).groupby("strategy_id").cash_cycle.sum().reindex(ids)
    np.testing.assert_array_equal(metrics.cash_cycles.to_numpy(),cash_counts.to_numpy())

    # Same convention and period must recover all four independently audited
    # weekly comparison curves from the previous client research.
    previous = pd.read_csv(output.parent / "weekly_fixed_strikes_10y/daily_portfolios.csv")
    regression = []
    for label in ("96-93","97-94","98-95","99-96"):
        key = f"short_{label}_d07"
        old = previous[(previous.portfolio == label.replace("-","/")+" at 100%") & previous["fill"].eq("realistic")].equity.to_numpy()
        new = curves[:,lookup[key]]
        np.testing.assert_allclose(old,new,rtol=1e-11,atol=1e-7)
        regression.append(dict(strategy=key,max_equity_difference=float(np.max(abs(old-new)))))
    result = dict(status="passed",strategies=len(metrics),daily_sessions=len(dates),daily_nav_values=int(curves.size),
                  independently_repriced_ledger_rows=len(source),maximum_terminal_return_error=float(errors.max()),
                  missing_daily_nav_values=int((~np.isfinite(curves)).sum()),
                  quote_estimates=int(quotes.estimated.sum()),unresolved_quotes=int(quotes.unresolved.sum()),
                  regression=regression)
    (output / "validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
