from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2%}" if np.isfinite(number) else "n/a"


def _number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2f}" if np.isfinite(number) else "n/a"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                text = f"{float(value):.4f}" if np.isfinite(value) else "n/a"
            else:
                text = str(value)
            rendered.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def generate_report(
    output_path: str | Path,
    *,
    audit: dict[str, Any],
    core_screen: pd.DataFrame,
    overlay_results: pd.DataFrame,
    finalists: pd.DataFrame,
    robustness: pd.DataFrame,
    stress: pd.DataFrame,
) -> None:
    tested = len(core_screen) + len(overlay_results)
    family = (
        overlay_results.groupby("hedge_family")
        .agg(
            combinations=("strategy_id", "count"),
            median_oos_sharpe=("oos_sharpe", "median"),
            best_oos_sharpe=("oos_sharpe", "max"),
            median_net_option_return=("annualized_option_return", "median"),
            median_max_drawdown=("max_drawdown", "median"),
        )
        .sort_values("median_oos_sharpe", ascending=False)
    )
    target = overlay_results[overlay_results["annualized_option_return"] >= 0.04]
    target = target.sort_values("robust_score", ascending=False).head(10)
    lines = [
        "# SPX Put-Spread and Downside-Hedge Research",
        "",
        "## Executive conclusion",
        "",
    ]
    if finalists.empty:
        lines.append("No strategy passed the data, OOS, and implementation screens.")
    else:
        winner = finalists.iloc[0]
        lines.extend(
            [
                f"The highest-ranked robust specification is `{winner['strategy_id']}`. ",
                f"It produced a {_percent(winner['annualized_option_return'])} annualized net option return, "
                f"a {_number(winner['oos_sharpe'])} out-of-sample Sharpe, and a "
                f"{_percent(winner['max_drawdown'])} maximum drawdown under the realistic execution case.",
                "",
                "This is a historical research result, not a forecast. The preferred conclusion is a parameter family/plateau, not a single optimized point.",
            ]
        )
    lines.extend(
        [
            "",
            "## Data and limitations",
            "",
            f"- Actual IVolatility EOD SPX quotes span {audit.get('first_date')} through {audit.get('last_date')}.",
            f"- {int(audit.get('populated_dates', 0)):,} sessions and {int(audit.get('total_rows', 0)):,} option rows passed the archive query.",
            f"- Maximum available maturity is {audit.get('maximum_dte')} calendar days. Requested 270- and 365-DTE tests are unavailable and were not synthesized.",
            "- The baseline uses SPXW PM-settled contracts. AM-settled SPX contracts are excluded to avoid using the EOD index level as an incorrect settlement value.",
            "- VIX is used only for analysis. Collateral uses historical 3-month Treasury yields and is reported separately.",
            "",
            "## Search and validation",
            "",
            f"The run evaluated {tested:,} core or core-plus-hedge configurations before finalist robustness resampling. "
            "Core entries were selected from contemporaneous quotes by delta and DTE. Results were split chronologically 60%/20%/20%; ranking emphasizes validation and final test Sharpe, nearby-parameter stability, transaction costs, and simplicity.",
            "",
            "Three execution cases were evaluated: midpoint, partial bid/ask plus commissions, and full bid/ask plus higher fees. Final rankings use the realistic case.",
            "",
            "## Strategy-family evidence",
            "",
            _markdown_table(family.reset_index()),
            "",
            "## Finalists",
            "",
            _markdown_table(
                finalists.head(20)[
                    [
                        "strategy_id",
                        "annualized_option_return",
                        "annualized_collateral_return",
                        "annualized_return",
                        "oos_sharpe",
                        "max_drawdown",
                        "es_95",
                        "annual_hedge_cost",
                    ]
                ]
            ),
            "",
            "## 4% net-option-return constraint",
            "",
        ]
    )
    if target.empty:
        lines.append("No tested realistic configuration reached 4% annualized net option return while passing the retained screens.")
    else:
        lines.append(
            _markdown_table(target[
                [
                    "strategy_id",
                    "annualized_option_return",
                    "annualized_return",
                    "oos_sharpe",
                    "max_drawdown",
                    "es_95",
                    "annual_hedge_cost",
                ]
            ])
        )
    lines.extend(
        [
            "",
            "## Stress periods",
            "",
            _markdown_table(stress) if not stress.empty else "No covered stress periods.",
            "",
            "## Multiple-testing and robustness",
            "",
            _markdown_table(robustness) if not robustness.empty else "Robustness output unavailable.",
            "",
            "## Interpretation",
            "",
            "Hedge efficiency is evaluated as the change in Sharpe, expected shortfall, and drawdown per unit of annual hedge cost. "
            "A result is not treated as credible when its validation/test performance collapses, when neighboring parameters fail, or when a single stress episode explains most of the result.",
            "",
            "The master Parquet/CSV files contain the full parameter and metric records. Charts in `results/charts` show NAV, drawdowns, rolling behavior, conditional returns, tails, and parameter plateaus.",
        ]
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
