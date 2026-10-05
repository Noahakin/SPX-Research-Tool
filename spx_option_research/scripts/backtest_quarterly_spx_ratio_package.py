from __future__ import annotations

import calendar
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
OUT = PROJECT / "results/quarterly_spx_ratio_package"
DATA_ROOT = (
    WORKSPACE
    / "04 - Options and Volatility/Raw Downloads/IVolatility/data/raw/spx_options_6m"
)
sys.path.insert(0, str(PROJECT / "src"))

from spxresearch.data_loader import SPXSurfaceArchive
from spxresearch.execution import ExecutionModel
from spxresearch.option_selector import clean_puts, select_expiration
from spxresearch.warehouse import ResearchWarehouse


INITIAL_CAPITAL = 1_000_000.0
TARGET_DTE = 90
DTE_TOLERANCE = 10
DEEP_CALL_RATIOS = [0.50, 0.60, 0.70, 0.80]
CAP_RATIOS = [1.05, 1.10, 1.15, 1.20]
BASELINE_DEEP_CALL = 0.60
BASELINE_CAP = 1.10
MODEL = ExecutionModel(spread_fraction=0.25, commission_per_contract=1.50)


def third_friday(year: int, month: int) -> pd.Timestamp:
    weeks = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in weeks if week[calendar.FRIDAY]]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def quarterly_entry_dates(
    dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    available = set(dates)
    entries: list[pd.Timestamp] = []
    for year in range(start.year, end.year + 1):
        for month in [1, 4, 7, 10]:
            scheduled = third_friday(year, month)
            if scheduled < start or scheduled > end:
                continue
            choices = [
                scheduled - pd.Timedelta(days=offset)
                for offset in range(4)
                if scheduled - pd.Timedelta(days=offset) in available
            ]
            if choices:
                entries.append(max(choices))
    return sorted(set(entries))


def clean_calls(chain: pd.DataFrame) -> pd.DataFrame:
    valid = (
        chain["option_type"].eq("call")
        & chain["settlement"].eq("PM")
        & chain["strike"].gt(0)
        & chain["bid"].ge(0)
        & chain["ask"].ge(chain["bid"])
        & chain["delta"].between(0.0, 1.0, inclusive="both")
    )
    return chain.loc[valid].copy()


def nearest_ratio(options: pd.DataFrame, spot: float, ratio: float) -> pd.Series:
    return options.loc[(options["strike"].astype(float) - spot * ratio).abs().idxmin()]


def expiry_pnl_limits(
    legs: list[tuple[pd.Series, int, str]], entry_cash: float
) -> tuple[float, float]:
    strikes = sorted({float(quote["strike"]) for quote, _, _ in legs})
    test_spots = [0.0, *strikes, max(strikes) * 2.0]
    pnls: list[float] = []
    for spot in test_spots:
        payoff = 0.0
        for quote, quantity, _ in legs:
            strike = float(quote["strike"])
            if str(quote["option_type"]).lower() == "call":
                intrinsic = max(spot - strike, 0.0)
            else:
                intrinsic = max(strike - spot, 0.0)
            payoff += quantity * intrinsic * 100.0
        pnls.append(entry_cash + payoff)
    return float(max(pnls)), float(min(pnls))


def build_candidates(
    archive: SPXSurfaceArchive,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    skips: list[dict[str, object]] = []
    entries = quarterly_entry_dates(archive.populated_dates, start, end)
    for entry_date in entries:
        chain = archive.read(entry_date)
        puts = clean_puts(chain, pm_only=True)
        try:
            expiration, actual_dte = select_expiration(
                puts,
                TARGET_DTE,
                tolerance=DTE_TOLERANCE,
                pm_only=True,
            )
        except LookupError as error:
            skips.append({"entry_date": entry_date, "reason": str(error)})
            continue
        expiry_puts = puts[puts["expiration_date"].eq(expiration)].copy()
        expiry_calls = clean_calls(chain)
        expiry_calls = expiry_calls[expiry_calls["expiration_date"].eq(expiration)].copy()
        if expiry_puts.empty or expiry_calls.empty:
            skips.append({"entry_date": entry_date, "reason": "missing PM puts or calls"})
            continue
        spot = float(expiry_puts["underlying_price"].dropna().median())
        atm_put = nearest_ratio(expiry_puts, spot, 1.00)
        short_95 = nearest_ratio(expiry_puts, spot, 0.95)
        long_94 = nearest_ratio(expiry_puts, spot, 0.94)
        if len({float(atm_put["strike"]), float(short_95["strike"]), float(long_94["strike"])}) < 3:
            skips.append({"entry_date": entry_date, "reason": "put strikes collided"})
            continue
        for deep_ratio in DEEP_CALL_RATIOS:
            deep_call = nearest_ratio(expiry_calls, spot, deep_ratio)
            for cap_ratio in CAP_RATIOS:
                short_call = nearest_ratio(expiry_calls, spot, cap_ratio)
                if float(short_call["strike"]) <= float(deep_call["strike"]):
                    continue
                variant_id = (
                    f"call{int(round(deep_ratio*100))}_cap{int(round(cap_ratio*100))}"
                )
                trade_id = f"{variant_id}_{entry_date:%Y%m%d}"
                legs = [
                    (deep_call, 1, "long_deep_itm_call"),
                    (atm_put, 2, "long_atm_put"),
                    (short_95, -7, "short_95_put"),
                    (long_94, 5, "long_94_put"),
                    (short_call, -1, "short_cap_call"),
                ]
                entry_cash = sum(MODEL.cash_flow(quote, quantity) for quote, quantity, _ in legs)
                max_profit, min_pnl = expiry_pnl_limits(legs, entry_cash)
                trades.append(
                    {
                        "trade_id": trade_id,
                        "variant_id": variant_id,
                        "deep_call_ratio": deep_ratio,
                        "cap_ratio": cap_ratio,
                        "entry_date": entry_date,
                        "expiration_date": expiration,
                        "actual_dte": actual_dte,
                        "spot_entry": spot,
                        "entry_cash_per_package": entry_cash,
                        "max_profit_per_package": max_profit,
                        "min_expiry_pnl_per_package": min_pnl,
                        "leg_count": len(legs),
                        "deep_call_strike": float(deep_call["strike"]),
                        "deep_call_delta": float(deep_call["delta"]),
                        "atm_put_strike": float(atm_put["strike"]),
                        "short_95_strike": float(short_95["strike"]),
                        "long_94_strike": float(long_94["strike"]),
                        "cap_call_strike": float(short_call["strike"]),
                        "cap_call_delta": float(short_call["delta"]),
                    }
                )
                for leg_id, (quote, quantity, role) in enumerate(legs):
                    leg_rows.append(
                        {
                            "trade_id": trade_id,
                            "leg_id": leg_id,
                            "role": role,
                            "quantity": quantity,
                            "option_symbol": str(quote["option_symbol"]).strip(),
                            "strike": float(quote["strike"]),
                            "option_type": str(quote["option_type"]).lower(),
                        }
                    )
    return pd.DataFrame(trades), pd.DataFrame(leg_rows), pd.DataFrame(skips)


def build_marks(
    trades: pd.DataFrame,
    legs: pd.DataFrame,
    end: pd.Timestamp,
) -> pd.DataFrame:
    with ResearchWarehouse(
        PROJECT / "data/cache/research.duckdb",
        DATA_ROOT,
        threads=8,
        memory_limit="12GB",
    ) as warehouse:
        con = warehouse.connection
        con.register("ratio_trades", trades)
        con.register("ratio_legs", legs)
        marks = con.execute(
            """
            WITH leg_marks AS (
                SELECT
                    t.trade_id, t.variant_id, t.entry_date, t.expiration_date,
                    t.entry_cash_per_package, t.leg_count,
                    l.leg_id, l.quantity, l.strike, l.option_type,
                    s.trade_date AS mark_date, s.spot, s.bid, s.ask, s.mid
                FROM ratio_trades t
                JOIN ratio_legs l USING (trade_id)
                JOIN surface s ON s.option_symbol = l.option_symbol
                    AND s.trade_date BETWEEN t.entry_date
                        AND least(t.expiration_date, CAST(? AS DATE))
            ), aggregated AS (
                SELECT
                    trade_id, min(variant_id) AS variant_id,
                    min(entry_date) AS entry_date,
                    min(expiration_date) AS expiration_date,
                    min(entry_cash_per_package) AS entry_cash_per_package,
                    mark_date, median(spot) AS spot,
                    sum(quantity * mid * 100.0) AS position_value_mid,
                    sum(
                        quantity *
                        CASE WHEN option_type = 'call'
                             THEN greatest(spot - strike, 0.0)
                             ELSE greatest(strike - spot, 0.0) END * 100.0
                    ) AS position_value_intrinsic,
                    sum(
                        quantity * mid * 100.0
                        - abs(quantity) * 0.25 * (ask - bid) * 100.0
                        - abs(quantity) * 1.50
                    ) AS close_cash_realistic,
                    count(DISTINCT leg_id) AS legs_marked,
                    max(leg_count) AS required_legs
                FROM leg_marks
                GROUP BY trade_id, mark_date
                HAVING count(DISTINCT leg_id) = max(leg_count)
            )
            SELECT *
            FROM aggregated
            ORDER BY variant_id, entry_date, mark_date
            """,
            [end.date()],
        ).fetchdf()
    for column in ["entry_date", "expiration_date", "mark_date"]:
        marks[column] = pd.to_datetime(marks[column])
    return marks


def strategy_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().fillna(0.0)
    years = (equity.index.max() - equity.index.min()).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    volatility = returns.std() * np.sqrt(252)
    annual_return = returns.mean() * 252
    drawdown = equity / equity.cummax() - 1.0
    return {
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_cash": annual_return / volatility if volatility > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "ending_equity": equity.iloc[-1],
    }


def simulate_variant(
    variant_trades: pd.DataFrame,
    marks: pd.DataFrame,
    market_index: pd.DatetimeIndex,
    spx_spot: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    increments: defaultdict[pd.Timestamp, float] = defaultdict(float)
    trade_rows: list[dict[str, object]] = []
    current_equity = INITIAL_CAPITAL
    for trade in variant_trades.sort_values("entry_date").itertuples(index=False):
        path = marks[marks["trade_id"].eq(trade.trade_id)].sort_values("mark_date").copy()
        if path.empty or path.iloc[0]["mark_date"] != trade.entry_date:
            continue
        is_expired = trade.expiration_date <= market_index.max()
        if is_expired and not path["mark_date"].eq(trade.expiration_date).any():
            continue
        final_date = trade.expiration_date if is_expired else path["mark_date"].max()
        path = path[path["mark_date"].le(final_date)].copy()
        path["pnl_per_package"] = (
            trade.entry_cash_per_package + path["position_value_mid"]
        )
        if is_expired:
            final_value = float(
                path.loc[path["mark_date"].eq(final_date), "position_value_intrinsic"].iloc[-1]
            )
            exit_reason = "expiration"
        else:
            final_value = float(path.iloc[-1]["close_cash_realistic"])
            exit_reason = "end_of_sample_liquidation"
        path.loc[path["mark_date"].eq(final_date), "pnl_per_package"] = (
            trade.entry_cash_per_package + final_value
        )
        entry_equity = current_equity
        package_units = entry_equity / (trade.spot_entry * 100.0)
        scaled = path.set_index("mark_date")["pnl_per_package"] * package_units
        daily_change = scaled.diff().fillna(scaled.iloc[0])
        for date, pnl in daily_change.items():
            increments[pd.Timestamp(date)] += float(pnl)
        realized_pnl = float(scaled.iloc[-1])
        current_equity += realized_pnl
        exit_spot = float(path.iloc[-1]["spot"])
        trade_rows.append(
            {
                **trade._asdict(),
                "exit_date": final_date,
                "exit_reason": exit_reason,
                "exit_spot": exit_spot,
                "package_units": package_units,
                "entry_equity": entry_equity,
                "realized_pnl": realized_pnl,
                "trade_return": realized_pnl / entry_equity,
                "spx_return": exit_spot / trade.spot_entry - 1.0,
            }
        )
    daily = pd.DataFrame(index=market_index)
    daily["daily_pnl"] = pd.Series(increments).reindex(market_index, fill_value=0.0)
    daily["equity"] = INITIAL_CAPITAL + daily["daily_pnl"].cumsum()
    first_entry = pd.Timestamp(variant_trades["entry_date"].min())
    daily = daily[daily.index >= first_entry].copy()
    daily["spx"] = spx_spot.reindex(daily.index).ffill()
    daily["spx_equity"] = INITIAL_CAPITAL * daily["spx"] / daily["spx"].iloc[0]
    metrics = strategy_metrics(daily["equity"])
    spx_metrics = strategy_metrics(daily["spx_equity"])
    completed = pd.DataFrame(trade_rows)
    metrics.update(
        {
            "variant_id": variant_trades["variant_id"].iloc[0],
            "deep_call_ratio": variant_trades["deep_call_ratio"].iloc[0],
            "cap_ratio": variant_trades["cap_ratio"].iloc[0],
            "trades": len(completed),
            "win_rate": completed["trade_return"].gt(0).mean(),
            "average_trade_return": completed["trade_return"].mean(),
            "median_trade_return": completed["trade_return"].median(),
            "average_spx_return": completed["spx_return"].mean(),
            "average_actual_dte": completed["actual_dte"].mean(),
            "average_entry_debit_pct_notional": (
                -completed["entry_cash_per_package"]
                / (completed["spot_entry"] * 100.0)
            ).mean(),
            "average_min_expiry_pnl_pct_notional": (
                completed["min_expiry_pnl_per_package"]
                / (completed["spot_entry"] * 100.0)
            ).mean(),
            "average_max_expiry_pnl_pct_notional": (
                completed["max_profit_per_package"]
                / (completed["spot_entry"] * 100.0)
            ).mean(),
            "spx_cagr": spx_metrics["cagr"],
            "spx_annualized_volatility": spx_metrics["annualized_volatility"],
            "spx_sharpe_zero_cash": spx_metrics["sharpe_zero_cash"],
            "spx_max_drawdown": spx_metrics["max_drawdown"],
            "spx_ending_equity": spx_metrics["ending_equity"],
        }
    )
    return metrics, daily.reset_index(names="date"), completed


def plot_results(daily: pd.DataFrame, trades: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1.35, 1.0]})
    fig.patch.set_facecolor("#f8fafc")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
    axes[0].plot(daily["date"], daily["equity"] / INITIAL_CAPITAL, label="Quarterly option package", color="#2563eb", linewidth=2.2)
    axes[0].plot(daily["date"], daily["spx_equity"] / INITIAL_CAPITAL, label="SPX price return", color="#0f172a", linewidth=1.8)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("Quarterly SPX ratio-package backtest", fontsize=20, fontweight="bold", loc="left", pad=18)
    axes[0].text(0, 1.01, "1x deep-ITM call · 2x ATM puts · -7x 95% puts · +5x 94% puts · -1x capped call", transform=axes[0].transAxes, fontsize=11, color="#475569")

    labels = pd.to_datetime(trades["entry_date"]).dt.to_period("Q").astype(str)
    x = np.arange(len(trades))
    width = 0.40
    axes[1].bar(x - width / 2, trades["trade_return"], width, label="Package", color="#2563eb")
    axes[1].bar(x + width / 2, trades["spx_return"], width, label="SPX", color="#94a3b8")
    axes[1].axhline(0, color="#334155", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=55, ha="right")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylabel("Quarter return")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_equity_and_quarterly_returns.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    pivot = summary.pivot(index="deep_call_ratio", columns="cap_ratio", values="cagr")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    image = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{value:.0%}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"{value:.0%}" for value in pivot.index])
    ax.set_xlabel("Upside cap strike / entry SPX")
    ax.set_ylabel("Deep-ITM call strike / entry SPX")
    ax.set_title("CAGR sensitivity", fontsize=18, fontweight="bold", loc="left", pad=14)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            ax.text(column, row, f"{pivot.iloc[row, column]:.2%}", ha="center", va="center", color="#0f172a")
    fig.colorbar(image, ax=ax, format=PercentFormatter(1.0), shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "parameter_sensitivity.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, baseline_trades: pd.DataFrame) -> None:
    baseline = summary[summary["variant_id"].eq("call60_cap110")].iloc[0]
    best = summary.sort_values(["sharpe_zero_cash", "cagr"], ascending=False).iloc[0]
    lines = [
        "# Quarterly SPX ratio-package backtest",
        "",
        "The package is rolled on the January, April, July, and October monthly expiration dates into a PM-settled expiration near 90 DTE. One package unit is sized to 100% of current SPX notional using continuous package units. Positions are held to expiration; the final open quarter is liquidated at the last available realistic mark. Cash earns 0%.",
        "",
        "Fixed legs are long 2 ATM puts, short 7 puts near 95% of entry SPX, and long 5 puts near 94%. Because the original description did not specify the deep-call strike or upside cap, the test covers deep calls from 50%-80% of spot and call caps from 105%-120%.",
        "",
        "## Baseline: 60% deep call and 110% cap",
        "",
        "| Metric | Package | SPX price |",
        "|:---|---:|---:|",
        f"| CAGR | {baseline['cagr']:.2%} | {baseline['spx_cagr']:.2%} |",
        f"| Annualized volatility | {baseline['annualized_volatility']:.2%} | {baseline['spx_annualized_volatility']:.2%} |",
        f"| Zero-cash Sharpe | {baseline['sharpe_zero_cash']:.2f} | {baseline['spx_sharpe_zero_cash']:.2f} |",
        f"| Maximum drawdown | {baseline['max_drawdown']:.2%} | {baseline['spx_max_drawdown']:.2%} |",
        f"| Ending value from $1 million | ${baseline['ending_equity']:,.0f} | ${baseline['spx_ending_equity']:,.0f} |",
        "",
        f"The baseline completed {int(baseline['trades'])} quarterly entries, won {baseline['win_rate']:.1%}, and averaged {baseline['average_trade_return']:.2%} per quarter. Average selected tenor was {baseline['average_actual_dte']:.1f} days.",
        "",
        f"The package cost an average {baseline['average_entry_debit_pct_notional']:.1%} of SPX notional. Its average worst expiration payoff was {baseline['average_min_expiry_pnl_pct_notional']:.1%} of notional and its average maximum expiration profit was {baseline['average_max_expiry_pnl_pct_notional']:.1%}. From the 94% strike down to the 60% deep-call strike, the package retains approximately +1 terminal delta and continues losing as SPX falls. Below the deep-call strike, its expiration value flattens near 5% of initial notional. The structure therefore softens an ordinary crash but only reaches its ultimate floor after an extreme decline.",
        "",
        "## Observed quarterly return zones",
        "",
        "| SPX quarter | Observations | Package average | SPX average |",
        "|:---|---:|---:|---:|",
    ]
    buckets = pd.read_csv(OUT / "baseline_return_buckets.csv")
    for _, row in buckets.iterrows():
        package_text = "N/A" if pd.isna(row["package_average"]) else f"{row['package_average']:.2%}"
        spx_text = "N/A" if pd.isna(row["spx_average"]) else f"{row['spx_average']:.2%}"
        lines.append(
            f"| {row['spx_bucket']} | {int(row['observations'])} | {package_text} | {spx_text} |"
        )
    lines.extend([
        "",
        "## Parameter sensitivity",
        "",
        f"The highest full-period Sharpe in the 16-combination sensitivity grid was **{best['variant_id']}**, at {best['cagr']:.2%} CAGR, {best['sharpe_zero_cash']:.2f} Sharpe, and {best['max_drawdown']:.2%} maximum drawdown. This is an in-sample sensitivity result rather than an independently validated selection.",
        "",
        "Execution assumes fills 25% of the quoted bid/ask spread away from mid plus $1.50 per contract per leg. Daily equity uses midpoint marks, expiration uses intrinsic value, and the final incomplete trade uses realistic liquidation. Results exclude dividends, cash interest, taxes, and margin financing.",
    ])
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = SPXSurfaceArchive(DATA_ROOT)
    end = archive.populated_dates.max()
    start = end - pd.DateOffset(years=5)
    trades, legs, skips = build_candidates(archive, start, end)
    marks = build_marks(trades, legs, end)
    market = pd.read_parquet(PROJECT / "data/cache/market_data.parquet").sort_index()
    market.index = pd.to_datetime(market.index)
    market_index = pd.DatetimeIndex(market.loc[start:end].index)
    rows: list[dict[str, object]] = []
    trade_outputs: list[pd.DataFrame] = []
    baseline_daily: pd.DataFrame | None = None
    baseline_trades: pd.DataFrame | None = None
    for variant_id, group in trades.groupby("variant_id"):
        metrics, daily, completed = simulate_variant(
            group, marks, market_index, market["spx_spot"]
        )
        rows.append(metrics)
        trade_outputs.append(completed)
        if variant_id == "call60_cap110":
            baseline_daily = daily
            baseline_trades = completed
    summary = pd.DataFrame(rows).sort_values(
        ["sharpe_zero_cash", "cagr"], ascending=False
    )
    all_trades = pd.concat(trade_outputs, ignore_index=True)
    if baseline_daily is None or baseline_trades is None:
        raise RuntimeError("Baseline variant was not produced")
    trades.to_csv(OUT / "selected_contracts.csv", index=False)
    legs.to_csv(OUT / "selected_legs.csv", index=False)
    marks.to_parquet(OUT / "daily_option_marks.parquet", index=False)
    skips.to_csv(OUT / "skips.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    all_trades.to_csv(OUT / "quarterly_trades_all_variants.csv", index=False)
    baseline_trades.to_csv(OUT / "baseline_quarterly_trades.csv", index=False)
    baseline_daily.to_csv(OUT / "baseline_daily_equity.csv", index=False)
    bucket_edges = [-np.inf, -0.06, -0.05, 0.0, 0.10, np.inf]
    bucket_labels = [
        "Below -6%",
        "-6% to -5%",
        "-5% to 0%",
        "0% to +10%",
        "Above +10%",
    ]
    bucketed = baseline_trades.assign(
        spx_bucket=pd.cut(
            baseline_trades["spx_return"],
            bins=bucket_edges,
            labels=bucket_labels,
            right=False,
        )
    )
    buckets = bucketed.groupby("spx_bucket", observed=False).agg(
        observations=("trade_return", "size"),
        package_average=("trade_return", "mean"),
        spx_average=("spx_return", "mean"),
    ).reset_index()
    buckets.to_csv(OUT / "baseline_return_buckets.csv", index=False)
    plot_results(baseline_daily, baseline_trades, summary)
    write_report(summary, baseline_trades)
    manifest = {
        "archive_start": str(start.date()),
        "archive_end": str(end.date()),
        "quarterly_entries": int(baseline_trades.shape[0]),
        "variants": int(summary.shape[0]),
        "target_dte": TARGET_DTE,
        "dte_tolerance": DTE_TOLERANCE,
        "baseline": "call60_cap110",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
