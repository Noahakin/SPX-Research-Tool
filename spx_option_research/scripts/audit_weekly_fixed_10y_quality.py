"""Independently inspect fixed-spread coverage and selected daily quote quality."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "results/weekly_dynamic_research"
OUT = PROJECT / "results/weekly_fixed_strikes_10y"
ARCHIVE = PROJECT.parent / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"


def archive_day(date):
    path = ARCHIVE / f"year={date.year}/month={date.month:02}/date={date:%Y-%m-%d}/eod.parquet"
    frame = pq.ParquetFile(path).read().to_pandas()
    frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date)
    frame["expiration_date"] = pd.to_datetime(frame.expiration_date)
    frame["option_symbol"] = frame.option_symbol.str.strip()
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_parquet(SOURCE / "candidate_trades.parquet")
    q = pd.read_parquet(SOURCE / "weekly_daily_option_quotes.parquet")
    availability = pd.read_csv(SOURCE / "weekly_availability.csv", parse_dates=["entry_date", "expiration_date"])
    market = pd.read_csv(SOURCE / "market_spx.csv", parse_dates=["date"]).set_index("date").close
    calendar = market.index[(market.index >= availability.entry_date.min()) & (market.index <= availability.expiration_date.max())]
    cache = {(r.snapshot_date, r.option_symbol): r for r in q.itertuples(index=False)}
    raw_cache = {}
    source_coverage, reconstructed, missing_trade = [], [], []
    for sr, width in ((.97, .03), (.98, .03), (.99, .03), (.98, .02), (.98, .05)):
        subset = c[np.isclose(c.target_short_ratio, sr) & np.isclose(c.target_width_pct, width)]
        source_coverage.append(dict(
            source_short_ratio=sr, source_width=width, candidates=len(subset),
            missing_dates="; ".join(availability.loc[~availability.entry_date.isin(subset.entry_date), "entry_date"].dt.strftime("%Y-%m-%d")),
        ))
    for label, ratio in (("96/93", .96), ("97/94", .97), ("98/95", .98), ("99/96", .99)):
        for schedule in availability.itertuples(index=False):
            same = c[c.entry_date.eq(schedule.entry_date)]
            if ratio == .96:
                short_source = same[np.isclose(same.target_short_ratio, .98) & np.isclose(same.target_width_pct, .02)]
                long_source = same[np.isclose(same.target_short_ratio, .98) & np.isclose(same.target_width_pct, .05)]
                short_role = "long"
            else:
                short_source = same[np.isclose(same.target_short_ratio, ratio) & np.isclose(same.target_width_pct, .03)]
                long_source = short_source
                short_role = "short"
            row = dict(spread=label, entry_date=schedule.entry_date, expiration_date=schedule.expiration_date)
            if len(short_source) == 1 and len(long_source) == 1:
                for field in ("symbol", "strike", "bid", "ask"):
                    row["short_" + field] = short_source.iloc[0][short_role + "_" + field]
                    row["long_" + field] = long_source.iloc[0]["long_" + field]
                row["entry_source"] = "stored candidate leg quotes"
            else:
                day = schedule.entry_date
                if day not in raw_cache:
                    raw_cache[day] = archive_day(day)
                raw = raw_cache[day]
                puts = raw[
                    raw.option_type.eq("put") & raw.option_symbol.str.startswith("SPXW")
                    & raw.expiration_date.eq(schedule.expiration_date)
                    & raw.strike.gt(0) & raw.bid.ge(0) & raw.ask.ge(raw.bid)
                ].copy()
                if puts.empty:
                    missing_trade.append({**row, "reason": "No valid PM put quotes for scheduled expiry"})
                    continue
                for role, target_ratio in (("short", ratio), ("long", ratio - .03)):
                    leg = puts.assign(distance=(puts.strike - target_ratio * market.loc[day]).abs()).sort_values(["distance", "strike", "option_symbol"]).iloc[0]
                    for field in ("symbol", "strike", "bid", "ask"):
                        row[role + "_" + field] = leg["option_symbol" if field == "symbol" else field]
                row["entry_source"] = "raw archive; economics filter bypassed"
            reconstructed.append(row)
    trades = pd.DataFrame(reconstructed)
    rows, absent, archive_reads = [], [], set()
    for trade in trades.itertuples(index=False):
        for day in calendar[(calendar >= trade.entry_date) & (calendar < trade.expiration_date)]:
            quotes = []
            for role in ("short", "long"):
                symbol = getattr(trade, role + "_symbol")
                if (day, symbol) not in cache:
                    if day not in raw_cache:
                        raw_cache[day] = archive_day(day)
                    leg = raw_cache[day][raw_cache[day].option_symbol.eq(symbol)]
                    if len(leg) != 1:
                        absent.append(dict(spread=trade.spread, date=day, symbol=symbol))
                        continue
                    cache[(day, symbol)] = next(leg.itertuples(index=False))
                    archive_reads.add((day, symbol))
                leg = cache[(day, symbol)]
                assert leg.expiration_date == trade.expiration_date
                assert leg.strike == getattr(trade, role + "_strike")
                assert np.isfinite([leg.bid, leg.ask]).all() and 0 <= leg.bid <= leg.ask
                quotes.append(leg)
            if len(quotes) != 2:
                continue
            short, long = quotes
            value = (short.bid + short.ask - long.bid - long.ask) / 2
            width = trade.short_strike - trade.long_strike
            rows.append(dict(
                spread=trade.spread, entry_date=trade.entry_date, date=day,
                short_symbol=trade.short_symbol, long_symbol=trade.long_symbol,
                short_bid=short.bid, short_ask=short.ask, long_bid=long.bid, long_ask=long.ask,
                spread_mid_points=value, width_points=width,
                payoff_bound_flag=value < -.01 or value > width + .01,
                wide_dollar_quote=((short.ask - short.bid > 30) and (short.ask - short.bid > .7 * (short.ask + short.bid)))
                or ((long.ask - long.bid > 30) and (long.ask - long.bid > .7 * (long.ask + long.bid))),
            ))
    marks = pd.DataFrame(rows)
    flags = marks[marks.payoff_bound_flag]
    big = marks[marks.wide_dollar_quote]
    assert not absent
    assert trades.groupby("spread").size().eq(520).all()
    assert marks.groupby("spread").size().eq(2504).all()
    assert len(flags) == 2 and flags.spread.eq("97/94").all()
    assert len(big) == 1 and big.iloc[0].date == pd.Timestamp("2020-02-27")
    pd.DataFrame(source_coverage).to_csv(OUT / "audit_candidate_cache_coverage.csv", index=False)
    pd.DataFrame(missing_trade).to_csv(OUT / "audit_cash_weeks.csv", index=False)
    pd.DataFrame(absent, columns=["spread", "date", "symbol"]).to_csv(OUT / "audit_missing_daily_quotes.csv", index=False)
    trades.to_csv(OUT / "audit_fixed_trade_legs.csv", index=False)
    flags.to_csv(OUT / "audit_selected_mark_flags.csv", index=False)

    suspect_date = pd.Timestamp("2020-02-27")
    raw = archive_day(suspect_date)
    expiry = raw[raw.expiration_date.eq("2020-02-28") & raw.option_symbol.str.startswith("SPXW")]
    evidence = expiry[
        (expiry.option_type.eq("put") & expiry.strike.isin([3105, 3135, 3170, 3205, 3225, 3230, 3235, 3240, 3245, 3250, 3255, 3270, 3305]))
        | (expiry.option_type.eq("call") & expiry.strike.isin([3105, 3135, 3170, 3205, 3240, 3270, 3305]))
    ].copy()
    evidence["timestamp_provenance"] = "Synthetic trade date plus 16 hours; downloader line 427"
    evidence["source_file"] = str(ARCHIVE / "year=2020/month=02/date=2020-02-27/eod.parquet")
    evidence.to_csv(OUT / "repair_evidence.csv", index=False)
    puts = expiry[expiry.option_type.eq("put")].set_index("strike")
    calls = expiry[expiry.option_type.eq("call")].set_index("strike")
    bad_mid = (puts.loc[3240, "bid"] + puts.loc[3240, "ask"]) / 2
    long_mid = (puts.loc[3135, "bid"] + puts.loc[3135, "ask"]) / 2
    adjacent_mid = ((puts.loc[3235, "bid"] + puts.loc[3235, "ask"]) / 2 + (puts.loc[3245, "bid"] + puts.loc[3245, "ask"]) / 2) / 2
    parity_spread = 105 + (calls.loc[3240, "bid"] + calls.loc[3240, "ask"]) / 2 - (calls.loc[3135, "bid"] + calls.loc[3135, "ask"]) / 2
    adjustment = dict(
        date=suspect_date, candidate_id="20200221_s097_w03",
        raw_spread=bad_mid - long_mid,
        adjacent_put_spread=adjacent_mid - long_mid,
        call_parity_spread=parity_spread,
        reason="Sensitivity estimates for clearly defective 3240 put midpoint; adjacent observed put midpoints interpolated, or opposite observed call midpoints with DF=1. Raw source preserved; neither is a recovered observed put quote.",
    )
    assert np.isclose(adjustment["raw_spread"], -33.225)
    assert np.isclose(adjustment["adjacent_put_spread"], 100.55)
    assert np.isclose(adjustment["call_parity_spread"], 103.725)
    pd.DataFrame([adjustment]).to_csv(OUT / "mark_adjustments.csv", index=False)

    lines = [
        "# Independent data-quality audit: weekly fixed SPX spreads, 2016–2026", "",
        f"Coverage: September 23, 2016–September 18, 2026, {len(availability)} scheduled weeks and {len(calendar):,} cash-market sessions. Each fixed strategy has 520 executable quoted trades and one common cash week. Quote existence is distinct from quote accuracy.", "",
        "## Trade coverage", "",
        "| Cached source | Records | Missing entry dates |", "|---|---:|---|",
    ]
    for row in source_coverage:
        lines.append(f"| Short {row['source_short_ratio']:.0%}, width {row['source_width']:.0%} | {row['candidates']} | {row['missing_dates']} |")
    lines += [
        "", "The common December 9–16, 2016 interval lacks the required PM expiry in the entry archive and stays in cash. Its sessions must remain in the return calendar. Other cache omissions result from the old positive-credit filter, including rejection when natural execution minus commissions is a debit; the raw entry option quotes exist. Independently selecting the nearest valid PM strikes on those omitted dates restores 520 trades for each fixed policy, allowing nonpositive net credits as required for an always-trade fixed policy.", "",
        f"All four strategies have 2,504 pre-expiry daily spread marks apiece. Every exact-symbol/date/strike/expiry quote is present and has finite nonnegative bids with ask at least bid. {len(archive_reads)} additional leg/date pairs were read directly from the source archive because they were not in the cached quote subset. Expiry cash settlement needs no option mark.", "",
        "## Selected midpoint flags", "",
        "Only two selected spread/date marks are outside [0, width] by more than 0.01 point:", "",
        "| Strategy / entry | Mark date | Raw liability | Width | Source details |", "|---|---|---:|---:|---|",
        "| 97/94 / 2020-02-21 | 2020-02-27 | -33.225 | 105 | Short 3240 put: 0.05 / 288.40; long 3135 put: 170.40 / 184.50 |",
        "| 97/94 / 2020-03-20 | 2020-03-26 | -0.025 | 70 | Short 2240 put: 0 / 0.10; long 2170 put: 0 / 0.15 |", "",
        "The second flag is a 2.5-cent spread-mark inversion among tiny quoted options. The February flag is materially different: an almost zero bid against a 288.40 ask on a deeply in-the-money short put. Among selected daily legs, it is the only quote with a bid/ask range above 30 index points and above 140% of midpoint. These screens do not prove the remaining quotes accurate.", "",
        "## February 27, 2020 forensic evidence", "",
        "The source has exactly one record for SPXW 200228P03240000, expiring February 28, 2020. Its bid is 0.05, ask 288.40, midpoint 144.225 and `last` 144.22499. The last field reproduces the defective midpoint and provides no independent replacement. Nearby put bids are also anomalous at strikes 3230 (0.50) and 3250 (0.25), while 3235 and 3245 have bid/ask 262.60/283.40 and 272.60/293.40.", "",
        "All 15,036 source records show 16:00, but the downloader assigns `snapshot_timestamp = trade_date + 16 hours` (download_spx_option_surface.py, line 427). This is a synthetic label, not an observed exchange quote timestamp. The archive underlying is 2957.452 versus cash-index close 2978.760009765625, further limiting assertions about exact synchronous close valuation.", "",
        "The workspace options-file search found one SPX archive file for February 27, 2020, under spx_options_6m. The other spx_options archive begins January 3, 2022. The downloader parses its raw CSV response in memory and writes normalized daily Parquet; no alternate saved February 2020 SPX source or intraday timestamp was found. No observed replacement quote was identified.", "",
        "## Disclosed sensitivity estimates", "",
        "Raw source quotes and the raw midpoint series must remain unchanged. Payoff clipping to zero removes the impossible negative liability but still values this deeply in-the-money vertical at zero. It does not repair the defect.", "",
        "| Convention for the affected 97/94 liability | Index points | Basis |", "|---|---:|---|",
        "| Raw midpoint | -33.225 | Defective recorded short-put bid/ask |",
        "| Payoff clipped | 0 | Mechanical bound only; no valuation evidence |",
        "| Adjacent-put interpolation sensitivity | 100.550 | Mean of 3235 and 3245 put midpoints: (273 + 283)/2 = 278; subtract observed 3135 put midpoint 177.45 |",
        "| Opposite-call parity sensitivity | 103.725 | 105 + 3240 call midpoint 0.025 - 3135 call midpoint 1.30; discount factor assumed 1 |", "",
        "For European options of identical expiry, P(3240)-P(3135) = 105 × discount factor + C(3240)-C(3135); the underlying cancels. At discount factor 1, observed call bid/ask implies a 103.40–104.05 spread interval. Interpolating adjacent puts and applying call parity are explicit estimates, not recovered observed put quotes. They alter daily mark-dependent statistics and do not alter fixed-quantity expiration P&L, subsequent entry capital, terminal equity or CAGR.", "",
        "The other selected February 27 spreads have raw liabilities inside their payoff bounds but disagree with same-expiry call parity: 96/93 (3205/3105) raw 93.25 versus call-parity midpoint 97.775; 98/95 (3270/3170) raw 97.40 versus 99.375; 99/96 (3305/3205) raw 99.40 versus 99.90. Wide bid/ask spreads can explain some midpoint disagreement. These observations show that passing payoff bounds does not certify synchronous, accurate daily marks; broad quote rewriting is not justified by this audit.", "",
        "Evidence files: repair_evidence.csv preserves the same-day relevant puts and calls; mark_adjustments.csv specifies the single flagged-date sensitivity change. audit_fixed_trade_legs.csv, audit_candidate_cache_coverage.csv, audit_cash_weeks.csv, audit_selected_mark_flags.csv and audit_missing_daily_quotes.csv document independent coverage checks.",
    ]
    (OUT / "data_quality_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Sessions={len(calendar)}, scheduled weeks={len(availability)}, trades per strategy=520; missing daily quotes=0; extra raw quote reads={len(archive_reads)}")
    print(flags.to_string(index=False))
    print(pd.DataFrame([adjustment]).to_string(index=False))


if __name__ == "__main__":
    main()
