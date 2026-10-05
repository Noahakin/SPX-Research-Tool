"""Independent raw-quote audit of fixed weekly SPX spreads.

Uses no portfolio simulation or mark-building helpers. The 96/93 legs come
from the long legs of existing 98/96 and 98/93 candidate records.
"""
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results" / "weekly_dynamic_research"
OUT = PROJECT / "results" / "weekly_fixed_strike_comparison"


def load_raw_inputs():
    candidates = pd.read_parquet(SOURCE / "candidate_trades.parquet")
    selections = pd.read_csv(
        SOURCE / "holdout_selections.csv",
        parse_dates=["entry_date", "expiration_date"],
    )
    schedule = selections.loc[
        selections.portfolio.eq("Fixed weekly 99/96"),
        ["entry_date", "expiration_date"],
    ].sort_values("entry_date").reset_index(drop=True)
    assert len(schedule) == 141
    assert schedule.entry_date.min() == pd.Timestamp("2024-01-05")
    assert schedule.expiration_date.max() == pd.Timestamp("2026-09-18")
    assert schedule.entry_date.is_unique
    market = pd.read_csv(SOURCE / "market_spx.csv", parse_dates=["date"])
    spot = market.set_index("date").close.to_dict()
    calendar = pd.DatetimeIndex(market.date)
    calendar = calendar[
        (calendar >= schedule.entry_date.min())
        & (calendar <= schedule.expiration_date.max())
    ]
    assert len(calendar) == 678
    quotes = pd.read_parquet(SOURCE / "weekly_daily_option_quotes.parquet")
    assert not quotes.duplicated(["snapshot_date", "option_symbol"]).any()
    assert (quotes.ask >= quotes.bid).all()
    assert np.isclose(quotes.mid, (quotes.bid + quotes.ask) / 2).all()
    quote_mid = {
        (r.snapshot_date, r.option_symbol): (r.bid + r.ask) / 2
        for r in quotes.itertuples(index=False)
    }
    return candidates, schedule, spot, calendar, quote_mid


def raw_legs(candidates, schedule, label):
    def subset(short_ratio, width):
        selected = candidates[
            np.isclose(candidates.target_short_ratio, short_ratio)
            & np.isclose(candidates.target_width_pct, width)
        ]
        selected = schedule.merge(
            selected, on=["entry_date", "expiration_date"],
            how="left", validate="one_to_one",
        )
        assert selected.candidate_id.notna().all()
        return selected

    if label == "96/93":
        first, second = subset(.98, .02), subset(.98, .05)
        short_role = "long"
    else:
        first = subset(float(label[:2]) / 100, .03)
        second = first
        short_role = "short"

    trades = schedule.copy()
    trades["spot_entry"] = first.spot_entry
    trades["spot_expiration"] = first.spot_expiration
    trades["short_source_candidate_id"] = first.candidate_id
    trades["long_source_candidate_id"] = second.candidate_id
    for field in ("symbol", "strike", "bid", "ask"):
        trades[f"short_{field}"] = first[f"{short_role}_{field}"]
        trades[f"long_{field}"] = second[f"long_{field}"]
    assert (trades.short_strike > trades.long_strike).all()
    assert np.isclose(first.spot_entry, second.spot_entry).all()
    return trades


def independent_cash_ledger(trades, spot, calendar, quote_mid, multiple, fill):
    # Each date first settles a maturing position, then opens a new position,
    # then values the portfolio. This puts both Friday actions in one return.
    entries = {r.entry_date: r for r in trades.itertuples(index=False)}
    position = None
    cash = 1_000_000.0
    prior_nav = cash
    daily, weekly = [], []
    for date in calendar:
        if position is not None and date == position["trade"].expiration_date:
            trade = position["trade"]
            assert spot[date] == trade.spot_expiration
            intrinsic = max(trade.short_strike - spot[date], 0.0) - max(
                trade.long_strike - spot[date], 0.0
            )
            cash -= 100 * position["contracts"] * intrinsic
            weekly.append({
                "entry_date": trade.entry_date,
                "expiration_date": trade.expiration_date,
                "entry_equity": position["entry_equity"],
                "contracts": position["contracts"],
                "net_credit_per_contract": position["net_credit"],
                "ending_equity": cash,
                "weekly_return": cash / position["entry_equity"] - 1,
            })
            position = None

        if date in entries:
            assert position is None
            trade = entries[date]
            assert spot[date] == trade.spot_entry
            contracts = multiple * cash / (100 * spot[date])
            short_mid = (trade.short_bid + trade.short_ask) / 2
            long_mid = (trade.long_bid + trade.long_ask) / 2
            assert quote_mid[(date, trade.short_symbol)] == short_mid
            assert quote_mid[(date, trade.long_symbol)] == long_mid
            if fill == "realistic":
                sold = short_mid - .25 * (trade.short_ask - trade.short_bid)
                bought = long_mid + .25 * (trade.long_ask - trade.long_bid)
            else:
                assert fill == "natural"
                sold, bought = trade.short_bid, trade.long_ask
            net_credit = 100 * (sold - bought) - 2 * 1.50
            position = dict(
                trade=trade, contracts=contracts, entry_equity=cash,
                net_credit=net_credit,
            )
            cash += contracts * net_credit

        liability = 0.0
        if position is not None:
            trade = position["trade"]
            short_mark = quote_mid[(date, trade.short_symbol)]
            long_mark = quote_mid[(date, trade.long_symbol)]
            liability = 100 * position["contracts"] * (short_mark - long_mark)
        nav = cash - liability
        daily.append(dict(
            date=date, equity=nav, cash=cash,
            option_liability=liability, daily_return=nav / prior_nav - 1,
        ))
        prior_nav = nav
    assert position is None
    daily, weekly = pd.DataFrame(daily), pd.DataFrame(weekly)
    assert len(weekly) == len(trades)
    # Gregorian mean year, matching the existing reported annualization basis.
    years = (calendar[-1] - calendar[0]).days / 365.2425
    returns = daily.daily_return
    peaks = np.maximum.accumulate(np.r_[1_000_000.0, daily.equity])[1:]
    summary = dict(
        first_entry=calendar[0], last_expiration=calendar[-1],
        trades=len(weekly), daily_observations=len(daily),
        elapsed_years=years, initial_equity=1_000_000.0,
        ending_equity=cash, total_return=cash / 1_000_000 - 1,
        cagr=(cash / 1_000_000) ** (1 / years) - 1,
        daily_sharpe=returns.mean() / returns.std(ddof=1) * np.sqrt(252),
        daily_annualized_volatility=returns.std(ddof=1) * np.sqrt(252),
        max_drawdown_daily=float((daily.equity.to_numpy() / peaks - 1).min()),
        worst_day=returns.min(), worst_week=weekly.weekly_return.min(),
        best_day=returns.max(), best_week=weekly.weekly_return.max(),
    )
    return summary, daily, weekly


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, schedule, spot, calendar, quote_mid = load_raw_inputs()
    summaries, daily_frames, weekly_frames, leg_frames = [], [], [], []
    missing = []
    for label in ("98/95", "96/93", "97/94"):
        trades = raw_legs(candidates, schedule, label)
        trades["spread"] = label
        leg_frames.append(trades)
        for row in trades.itertuples(index=False):
            for date in calendar[
                (calendar >= row.entry_date) & (calendar < row.expiration_date)
            ]:
                for role in ("short", "long"):
                    symbol = getattr(row, f"{role}_symbol")
                    if (date, symbol) not in quote_mid:
                        missing.append(dict(spread=label, date=date, symbol=symbol))
        assert not missing, missing[:10]
        for multiple in (1.0, 2.0):
            for fill in ("realistic", "natural"):
                summary, daily, weekly = independent_cash_ledger(
                    trades, spot, calendar, quote_mid, multiple, fill,
                )
                metadata = dict(
                    spread=label, notional_multiple=multiple, fill=fill,
                    portfolio=f"{label} at {multiple:.0%}",
                )
                if label == "97/94":
                    prior = pd.read_csv(
                        PROJECT / "results" / "weekly_97_94_notional_comparison"
                        / "independent_notional_summary.csv"
                    )
                    expected = prior[
                        prior.notional_multiplier.eq(multiple) & prior.fill.eq(fill)
                    ].iloc[0]
                    for key in (
                        "ending_equity", "cagr", "daily_sharpe",
                        "max_drawdown_daily", "worst_day", "worst_week",
                    ):
                        assert np.isclose(summary[key], expected[key], atol=1e-10, rtol=1e-10), (
                            label, multiple, fill, key, summary[key], expected[key]
                        )
                summaries.append({**metadata, **summary})
                daily_frames.append(daily.assign(**metadata))
                weekly_frames.append(weekly.assign(**metadata))
    pd.DataFrame(summaries).to_csv(OUT / "independent_summary.csv", index=False)
    pd.concat(daily_frames).to_csv(OUT / "independent_daily_nav.csv", index=False)
    pd.concat(weekly_frames).to_csv(OUT / "independent_weekly_trades.csv", index=False)
    pd.concat(leg_frames).to_csv(OUT / "independent_raw_legs.csv", index=False)
    pd.DataFrame(missing, columns=["spread", "date", "symbol"]).to_csv(
        OUT / "independent_missing_quotes.csv", index=False,
    )
    print(pd.DataFrame(summaries)[[
        "spread", "notional_multiple", "fill", "cagr", "daily_sharpe",
        "max_drawdown_daily", "worst_day", "worst_week", "ending_equity",
    ]].to_string(index=False, float_format=lambda x: f"{x:.12f}"))
    print("All 97/94 reference values reconcile; zero missing exact-symbol daily quotes.")


if __name__ == "__main__":
    main()
