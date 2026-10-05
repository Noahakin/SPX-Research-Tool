"""Daily portfolio accounting and paired evaluation of weekly spread policies."""
from __future__ import annotations

import numpy as np
import pandas as pd


FILLS = ("mid", "realistic", "natural")
ANNUAL_SESSIONS = 252.0
CALENDAR_DAYS_PER_YEAR = 365.2425


def _dated_frame(data: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = data.copy()
    if column not in frame and frame.index.name == column:
        frame = frame.reset_index()
    if column not in frame:
        raise ValueError(f"table requires {column} as a column or named index")
    frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    if frame[column].isna().any():
        raise ValueError(f"{column} must be present")
    return frame


def _close(actual: float, expected: float, description: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-11):
        raise ValueError(f"{description} does not reconcile: {actual} versus {expected}")


def _cash_id(candidate_id: object) -> bool:
    return bool(pd.isna(candidate_id)) or str(candidate_id).strip().upper() == "CASH"


def _number(row: pd.Series, *columns: str) -> float:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return float(row[column])
    return np.nan


def simulate_policy(
    selection: pd.DataFrame,
    candidates: pd.DataFrame,
    unit_marks: pd.DataFrame,
    cash_dates,
    fill: str = "realistic",
    *,
    initial_equity: float = 1_000_000.0,
    notional_multiple: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark a weekly policy at a fixed multiple of entry-equity SPX notional.

    selection has entry_date (column or named index), expiration_date and
    candidate_id.  None/NaN/'CASH' means no trade.  Include every scheduled
    week, including skipped weeks.  Consecutive entries must equal the prior
    expiry, and no source-candidate week within the selected span may be
    silently omitted.  Extra explicit cash weeks can fill unavailable-data
    gaps in the candidate table.

    candidates has unique candidate_id, entry_date, expiration_date and
    pnl_{fill}_pct_spot_notional.  Optional width_pct, max_loss_{fill}_pct,
    short_delta and long_delta provide exposure diagnostics.  Vendor deltas
    are per-unit index deltas; signed position delta is long minus short.

    unit_marks has unique (candidate_id,date) and cum_return_{fill}, expressed
    as cumulative P&L per unit entry SPX notional, including entry costs and
    terminal settlement.  Every selected candidate must have every cash
    session from entry through expiry.  Source P&L must match its terminal
    mark.  cash_dates is a DatetimeIndex, date sequence, or a Series indexed
    by actual cash-SPX sessions; dates outside the policy span are ignored.

    During a position, NAV = pre-entry equity *
    (1 + notional_multiple * cumulative unit return).  Contract quantity is
    fixed until the weekly roll; leverage is not reset daily.  All execution
    costs scale with quantity.  width_pct describes contract strike width;
    max_loss_pct and net_delta describe the scaled portfolio exposure.
    At a roll, old settlement determines the next sizing equity before the
    new position's entry cost is recognized.  Daily NAV on a roll includes
    both; weekly ending_equity is the old settlement value before new costs.
    Cash weeks have zero weekly P&L and retain all their daily observations.
    The first entry cost is measured against initial_equity, never discarded.
    """
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}")
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError("initial equity must be finite and positive")
    if not np.isfinite(notional_multiple) or notional_multiple <= 0:
        raise ValueError("notional multiple must be finite and positive")
    policy = _dated_frame(selection, "entry_date")
    if not {"expiration_date", "candidate_id"}.issubset(policy.columns) or policy.empty:
        raise ValueError("selection must contain weekly expiration_date and candidate_id rows")
    policy["expiration_date"] = pd.to_datetime(policy.expiration_date).dt.normalize()
    policy = policy.sort_values("entry_date").reset_index(drop=True)
    if policy.entry_date.duplicated().any() or policy.expiration_date.isna().any():
        raise ValueError("selection dates must be unique and nonmissing")
    if policy.expiration_date.le(policy.entry_date).any():
        raise ValueError("each expiration must follow its entry")
    if len(policy) > 1 and not np.array_equal(policy.expiration_date.iloc[:-1].to_numpy(),
                                              policy.entry_date.iloc[1:].to_numpy()):
        raise ValueError("selection weeks must be consecutive: each expiry equals the next entry")

    source = _dated_frame(candidates, "entry_date")
    required = {"candidate_id", "expiration_date", f"pnl_{fill}_pct_spot_notional"}
    if missing := required.difference(source.columns):
        raise ValueError(f"candidate table lacks {sorted(missing)}")
    source["expiration_date"] = pd.to_datetime(source.expiration_date).dt.normalize()
    if source.candidate_id.isna().any() or source.candidate_id.duplicated().any():
        raise ValueError("source candidate IDs must be unique and nonmissing")
    schedule = source.loc[source.entry_date.between(policy.entry_date.min(), policy.entry_date.max())]
    if schedule.groupby("entry_date").expiration_date.nunique().gt(1).any():
        raise ValueError("source candidates disagree on the weekly expiry schedule")
    expected_dates = set(schedule.entry_date)
    if not expected_dates.issubset(set(policy.entry_date)):
        raise ValueError("selection omits a scheduled candidate week; use an explicit CASH row")
    scheduled_expiry = schedule.drop_duplicates("entry_date").set_index("entry_date").expiration_date
    for week in policy.itertuples(index=False):
        if week.entry_date in scheduled_expiry.index and scheduled_expiry.loc[week.entry_date] != week.expiration_date:
            raise ValueError("selection expiry disagrees with the source weekly schedule")
    source = source.set_index("candidate_id", drop=False)

    calendar_input = cash_dates.index if isinstance(cash_dates, pd.Series) else cash_dates
    if isinstance(calendar_input, pd.DataFrame):
        calendar_input = calendar_input["date"] if "date" in calendar_input else calendar_input.index
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_input)).normalize().sort_values()
    if calendar.has_duplicates or calendar.hasnans:
        raise ValueError("cash-session dates must be unique and nonmissing")
    start, end = policy.entry_date.iloc[0], policy.expiration_date.iloc[-1]
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    if not set(policy.entry_date).union(policy.expiration_date).issubset(set(calendar)):
        raise ValueError("cash calendar omits an entry or expiration session")

    marks = _dated_frame(unit_marks, "date")
    mark_column = f"cum_return_{fill}"
    if not {"candidate_id", mark_column}.issubset(marks.columns):
        raise ValueError(f"unit marks require candidate_id and {mark_column}")
    if marks.duplicated(["candidate_id", "date"]).any():
        raise ValueError("unit marks have duplicate candidate/date keys")
    mark_lookup = marks.set_index(["candidate_id", "date"])[mark_column]
    equity = float(initial_equity)
    daily_rows, weekly_rows = [], []
    for week in policy.itertuples(index=False):
        entry, expiration = week.entry_date, week.expiration_date
        days = calendar[(calendar >= entry) & (calendar <= expiration)]
        traded = not _cash_id(week.candidate_id)
        candidate_id = week.candidate_id if traded else "CASH"
        if traded:
            if candidate_id not in source.index:
                raise ValueError(f"selected candidate {candidate_id!r} is absent")
            quote = source.loc[candidate_id]
            if quote.entry_date != entry or quote.expiration_date != expiration:
                raise ValueError(f"selected candidate {candidate_id!r} has different trade dates")
            keys = pd.MultiIndex.from_product([[candidate_id], days], names=["candidate_id", "date"])
            missing_keys = keys.difference(mark_lookup.index)
            if len(missing_keys):
                raise ValueError(f"missing unit marks for {candidate_id!r}: {list(missing_keys[:3])}")
            returns = mark_lookup.reindex(keys).to_numpy(float)
            if not np.isfinite(returns).all() or (returns <= -1).any():
                raise ValueError(f"invalid/nonpositive-equity unit return for {candidate_id!r}")
            terminal = float(returns[-1])
            source_pnl = float(quote[f"pnl_{fill}_pct_spot_notional"])
            _close(terminal, source_pnl, f"{candidate_id!r} terminal source P&L")
            width = _number(quote, "width_pct")
            if not np.isfinite(width):
                upper = _number(quote, "upper_strike", "short_strike")
                lower = _number(quote, "lower_strike", "long_strike")
                width = (upper - lower) / _number(quote, "spot_entry")
            max_loss = _number(quote, f"max_loss_{fill}_pct", f"max_loss_{fill}_pct_spot_notional")
            if not np.isfinite(max_loss):
                max_loss = width - _number(quote, f"premium_{fill}_pct_spot_notional")
            short_delta, long_delta = _number(quote, "short_delta"), _number(quote, "long_delta")
            delta = long_delta - short_delta
            if not np.isfinite(delta):
                delta = _number(quote, "net_delta")
        else:
            returns = np.zeros(len(days))
            terminal = source_pnl = 0.0
            width = max_loss = delta = 0.0
        returns = returns * notional_multiple
        terminal *= notional_multiple
        source_pnl *= notional_multiple
        max_loss *= notional_multiple
        delta *= notional_multiple
        if not np.isfinite(returns).all() or (returns <= -1).any():
            raise ValueError("notional multiple produces nonpositive marked equity")
        entry_equity = equity
        for day, cumulative in zip(days[:-1], returns[:-1]):
            daily_rows.append({
                "date": day, "equity": entry_equity * (1 + float(cumulative)),
                "active_candidate_id": candidate_id, "active_entry_date": entry,
                "active_expiration_date": expiration, "traded": traded,
                "unit_cum_return": float(cumulative) / notional_multiple,
                "position_cum_return": float(cumulative), "fill": fill,
                "notional_multiple": notional_multiple,
            })
        equity = entry_equity * (1 + terminal)
        weekly_rows.append({
            "entry_date": entry, "expiration_date": expiration,
            "candidate_id": candidate_id, "traded": traded, "fill": fill,
            "weekly_return": terminal, "entry_equity": entry_equity,
            "ending_equity": equity, "pnl_dollars": equity - entry_equity,
            "width_pct": width, "max_loss_pct": max_loss, "net_delta": delta,
            "notional_multiple": notional_multiple,
            "source_pnl_reconciliation_error": terminal - source_pnl,
        })
    daily_rows.append({
        "date": end, "equity": equity, "active_candidate_id": "CASH",
        "active_entry_date": pd.NaT, "active_expiration_date": pd.NaT,
        "traded": False, "unit_cum_return": 0.0, "position_cum_return": 0.0, "fill": fill,
        "notional_multiple": notional_multiple,
    })
    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    weekly = pd.DataFrame(weekly_rows)
    if daily.date.duplicated().any() or not np.array_equal(daily.date.to_numpy(), calendar.to_numpy()):
        raise ValueError("daily simulation did not account for every cash session exactly once")
    previous = daily.equity.shift(1).fillna(initial_equity)
    daily["daily_return"] = daily.equity / previous - 1
    daily["daily_pnl"] = daily.equity - previous
    daily["initial_equity"] = float(initial_equity)
    _close(float((1 + daily.daily_return).prod()), equity / initial_equity, "daily compounding")
    _close(float((1 + weekly.weekly_return).prod()), equity / initial_equity, "weekly compounding")
    return daily, weekly


def summarize(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict[str, object]:
    """Daily-marked zero-cash-rate metrics; active exposure means exclude cash weeks."""
    path = _dated_frame(daily, "date").sort_values("date")
    trades = _dated_frame(weekly, "entry_date").sort_values("entry_date")
    if path.empty or trades.empty or path.date.duplicated().any():
        raise ValueError("nonempty unique daily observations and weekly rows are required")
    required_daily = {"equity", "daily_return", "daily_pnl"}
    required_weekly = {"expiration_date", "entry_equity", "ending_equity", "weekly_return", "traded"}
    if not required_daily.issubset(path) or not required_weekly.issubset(trades):
        raise ValueError("summarize requires simulate_policy daily and weekly fields")
    initial, final = float(trades.entry_equity.iloc[0]), float(path.equity.iloc[-1])
    returns = path.daily_return.to_numpy(float)
    if not np.isfinite(returns).all() or initial <= 0 or final <= 0:
        raise ValueError("performance requires finite returns and positive equity")
    _close(float((1 + returns).prod()), final / initial, "summary daily compounding")
    _close(float(trades.ending_equity.iloc[-1]), final, "summary terminal weekly equity")
    years = (path.date.max() - path.date.min()).days / CALENDAR_DAYS_PER_YEAR
    if years <= 0:
        raise ValueError("performance requires positive elapsed calendar time")
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
    peak = np.maximum.accumulate(np.r_[initial, path.equity.to_numpy(float)])[1:]
    active = trades.loc[trades.traded.astype(bool)]
    stats: dict[str, object] = {
        "first_entry": path.date.min(), "last_expiration": path.date.max(),
        "initial_equity": initial, "ending_equity": final,
        "daily_observations": len(path), "weeks": len(trades), "trades": len(active),
        "trade_fraction": len(active) / len(trades), "elapsed_years": years,
        "total_return": final / initial - 1,
        "cagr": (final / initial) ** (1 / years) - 1,
        "daily_sharpe": float(np.mean(returns) / sd * np.sqrt(ANNUAL_SESSIONS)) if sd > 0 else np.nan,
        "daily_annualized_volatility": sd * np.sqrt(ANNUAL_SESSIONS),
        "annualized_arithmetic_return": float(np.mean(returns) * ANNUAL_SESSIONS),
        "max_drawdown_daily": float(np.min(path.equity.to_numpy(float) / peak - 1)),
        "worst_day": float(np.min(returns)),
        "worst_week": float(trades.weekly_return.min()),
    }
    for column in ("width_pct", "max_loss_pct", "net_delta"):
        stats[f"mean_{column}"] = float(active[column].mean()) if column in active and len(active) else np.nan
        stats[f"mean_{column}_all_weeks"] = float(trades[column].mean()) if column in trades else np.nan
    return stats


def paired_block_bootstrap(
    daily_a: pd.DataFrame,
    daily_b: pd.DataFrame,
    *,
    draws: int = 1000,
    block_weeks: int = 4,
    seed: int = 1729,
) -> dict[str, object]:
    """Paired circular calendar-week block bootstrap, differences A minus B.

    Both tables need identical dates and finite daily_return observations.
    Group actual daily observations into Saturday-Friday calendar weeks;
    draw contiguous circular four-week blocks and retain the original number
    of weeks.  Holidays and partial boundary weeks retain their actual daily
    counts.  Both policies always receive the same sampled dates.  Return
    percentile 95% intervals for annualized daily Sharpe and annual arithmetic
    return (252 * mean daily return) differences.  The model is never refit or
    selected here.  Undefined zero-volatility Sharpes remain NaN.
    """
    if draws < 1 or block_weeks < 1:
        raise ValueError("draws and block_weeks must be positive")
    left = _dated_frame(daily_a, "date").sort_values("date").reset_index(drop=True)
    right = _dated_frame(daily_b, "date").sort_values("date").reset_index(drop=True)
    if left.empty or left.date.duplicated().any() or right.date.duplicated().any():
        raise ValueError("bootstrap needs nonempty unique dates")
    if not left.date.equals(right.date):
        raise ValueError("paired bootstrap requires exactly matching daily dates")
    if "daily_return" not in left or "daily_return" not in right:
        raise ValueError("bootstrap inputs require daily_return")
    a, b = left.daily_return.to_numpy(float), right.daily_return.to_numpy(float)
    if not np.isfinite(np.r_[a, b]).all():
        raise ValueError("bootstrap returns must be finite")
    groups = pd.DataFrame({
        "week": left.date.dt.to_period("W-FRI"), "a": a, "b": b,
        "a2": a * a, "b2": b * b,
    }).groupby("week", sort=True).agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"),
                                     a2=("a2", "sum"), b2=("b2", "sum"))
    weeks = len(groups)
    if weeks < 2 or len(a) < 3:
        raise ValueError("bootstrap requires at least two calendar weeks and three sessions")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, weeks, size=(draws, int(np.ceil(weeks / block_weeks))))
    sampled = (starts[:, :, None] + np.arange(block_weeks)[None, None, :]) % weeks
    sampled = sampled.reshape(draws, -1)[:, :weeks]
    n = groups.n.to_numpy()[sampled].sum(axis=1)
    sa, sb = groups.a.to_numpy()[sampled].sum(axis=1), groups.b.to_numpy()[sampled].sum(axis=1)
    sa2, sb2 = groups.a2.to_numpy()[sampled].sum(axis=1), groups.b2.to_numpy()[sampled].sum(axis=1)
    ma, mb = sa / n, sb / n
    sda = np.sqrt(np.maximum((sa2 - sa * sa / n) / (n - 1), 0.0))
    sdb = np.sqrt(np.maximum((sb2 - sb * sb / n) / (n - 1), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_draws = np.sqrt(ANNUAL_SESSIONS) * (ma / sda - mb / sdb)
    valid_sharpes = sharpe_draws[np.isfinite(sharpe_draws)]
    return_draws = ANNUAL_SESSIONS * (ma - mb)
    sda_original, sdb_original = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    observed_sharpe = (np.sqrt(ANNUAL_SESSIONS) * (float(np.mean(a)) / sda_original - float(np.mean(b)) / sdb_original)
                      if sda_original > 0 and sdb_original > 0 else np.nan)
    sharpe_ci = np.quantile(valid_sharpes, [0.025, 0.975]) if len(valid_sharpes) else [np.nan, np.nan]
    return_ci = np.quantile(return_draws, [0.025, 0.975])
    return {
        "draws": draws, "block_weeks": block_weeks, "seed": seed,
        "calendar_weeks": weeks, "daily_observations": len(a),
        "finite_sharpe_draws": len(valid_sharpes),
        "sharpe_difference": float(observed_sharpe),
        "sharpe_difference_ci_low": float(sharpe_ci[0]),
        "sharpe_difference_ci_high": float(sharpe_ci[1]),
        "annual_arithmetic_return_difference": float(ANNUAL_SESSIONS * (np.mean(a) - np.mean(b))),
        "annual_arithmetic_return_difference_ci_low": float(return_ci[0]),
        "annual_arithmetic_return_difference_ci_high": float(return_ci[1]),
        "fraction_draws_sharpe_difference_positive": float(np.mean(valid_sharpes > 0)) if len(valid_sharpes) else np.nan,
        "fraction_draws_return_difference_positive": float(np.mean(return_draws > 0)),
    }
