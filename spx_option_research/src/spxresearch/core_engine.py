from __future__ import annotations

from pathlib import Path

import duckdb


def build_core_trade_outcomes(
    connection: duckdb.DuckDBPyConnection,
    candidates_path: str | Path,
    output_path: str | Path,
) -> None:
    """Create fixed-exit outcomes from actual entry and exit quotes.

    Contract pairs are deduplicated before daily quote joins. This preserves all
    parameter labels while avoiding repeated marking work when nearby width
    targets select the same listed strikes.
    """
    candidates = Path(candidates_path).resolve().as_posix().replace("'", "''")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_sql = output.as_posix().replace("'", "''")

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW core_candidates AS
        SELECT * FROM read_parquet('{candidates}')
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE core_pairs AS
        SELECT DISTINCT
            pair_id, entry_date, expiration_date, short_symbol, long_symbol,
            short_strike, long_strike
        FROM core_candidates
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE pair_marks AS
        SELECT
            p.pair_id,
            s.trade_date AS mark_date,
            s.dte AS current_dte,
            s.spot,
            s.delta AS short_delta,
            s.bid AS short_bid,
            s.ask AS short_ask,
            s.mid AS short_mid,
            l.bid AS long_bid,
            l.ask AS long_ask,
            l.mid AS long_mid
        FROM core_pairs p
        JOIN surface s
          ON s.option_symbol = p.short_symbol
         AND s.trade_date BETWEEN p.entry_date AND p.expiration_date
        JOIN surface l
          ON l.option_symbol = p.long_symbol
         AND l.trade_date = s.trade_date
        WHERE s.bid >= 0 AND s.ask >= s.bid
          AND l.bid >= 0 AND l.ask >= l.bid
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE fixed_exit_dates AS
        SELECT
            pair_id,
            max(mark_date) AS fallback_date,
            min(mark_date) FILTER (WHERE current_dte <= 0) AS exit_0,
            min(mark_date) FILTER (WHERE current_dte <= 1) AS exit_1,
            min(mark_date) FILTER (WHERE current_dte <= 3) AS exit_3,
            min(mark_date) FILTER (WHERE current_dte <= 5) AS exit_5,
            min(mark_date) FILTER (WHERE current_dte <= 7) AS exit_7
        FROM pair_marks
        GROUP BY pair_id
        """
    )
    connection.execute(
        f"""
        COPY (
            WITH exit_rules(exit_dte) AS (VALUES (0), (1), (3), (5), (7)),
            chosen AS (
                SELECT
                    c.*,
                    r.exit_dte,
                    CASE r.exit_dte
                        WHEN 0 THEN coalesce(e.exit_0, e.fallback_date)
                        WHEN 1 THEN coalesce(e.exit_1, e.fallback_date)
                        WHEN 3 THEN coalesce(e.exit_3, e.fallback_date)
                        WHEN 5 THEN coalesce(e.exit_5, e.fallback_date)
                        WHEN 7 THEN coalesce(e.exit_7, e.fallback_date)
                    END AS exit_date
                FROM core_candidates c
                JOIN fixed_exit_dates e USING (pair_id)
                CROSS JOIN exit_rules r
                WHERE r.exit_dte < c.entry_dte
            ),
            valued AS (
                SELECT
                    c.*,
                    m.current_dte AS actual_exit_dte,
                    m.spot AS spot_exit,
                    m.short_delta AS short_delta_exit,
                    CASE WHEN c.exit_dte = 0 AND m.current_dte = 0
                         THEN greatest(c.short_strike - m.spot, 0.0)
                              - greatest(c.long_strike - m.spot, 0.0)
                         ELSE m.short_mid - m.long_mid END AS exit_debit_ideal,
                    CASE WHEN c.exit_dte = 0 AND m.current_dte = 0
                         THEN greatest(c.short_strike - m.spot, 0.0)
                              - greatest(c.long_strike - m.spot, 0.0)
                         ELSE (m.short_mid + 0.25 * (m.short_ask - m.short_bid))
                              - (m.long_mid - 0.25 * (m.long_ask - m.long_bid))
                              + 0.03 END AS exit_debit_realistic,
                    CASE WHEN c.exit_dte = 0 AND m.current_dte = 0
                         THEN greatest(c.short_strike - m.spot, 0.0)
                              - greatest(c.long_strike - m.spot, 0.0)
                         ELSE m.short_ask - m.long_bid + 0.05 END AS exit_debit_conservative
                FROM chosen c
                JOIN pair_marks m
                  ON m.pair_id = c.pair_id AND m.mark_date = c.exit_date
            )
            SELECT
                *,
                (entry_credit_ideal - exit_debit_ideal) * 100.0 AS pnl_per_spread_ideal,
                (entry_credit_realistic - exit_debit_realistic) * 100.0 AS pnl_per_spread_realistic,
                (entry_credit_conservative - exit_debit_conservative) * 100.0 AS pnl_per_spread_conservative,
                (entry_credit_ideal - exit_debit_ideal) * 100.0 / max_loss_ideal AS return_on_risk_ideal,
                (entry_credit_realistic - exit_debit_realistic) * 100.0 / max_loss_realistic AS return_on_risk_realistic,
                (entry_credit_conservative - exit_debit_conservative) * 100.0 / max_loss_conservative AS return_on_risk_conservative
            FROM valued
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def build_dynamic_exit_outcomes(
    connection: duckdb.DuckDBPyConnection,
    shortlisted_candidates_path: str | Path,
    output_path: str | Path,
) -> None:
    """Evaluate 50%/75% capture and 75-delta exits for shortlisted cores."""
    candidates = Path(shortlisted_candidates_path).resolve().as_posix().replace("'", "''")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_sql = output.as_posix().replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW shortlisted AS SELECT * FROM read_parquet('{candidates}')"
    )
    connection.execute(
        f"""
        COPY (
            WITH marks AS (
                SELECT
                    c.*,
                    s.trade_date AS mark_date,
                    s.dte AS current_dte,
                    s.spot,
                    s.delta AS short_delta_exit,
                    CASE
                      WHEN s.dte = 0 THEN
                        greatest(c.short_strike - s.spot, 0.0)
                        - greatest(c.long_strike - s.spot, 0.0)
                      ELSE
                        (s.mid + 0.25 * (s.ask - s.bid))
                        - (l.mid - 0.25 * (l.ask - l.bid)) + 0.03
                    END AS close_debit
                FROM shortlisted c
                JOIN surface s ON s.option_symbol = c.short_symbol
                  AND s.trade_date BETWEEN c.entry_date AND c.expiration_date
                JOIN surface l ON l.option_symbol = c.long_symbol
                  AND l.trade_date = s.trade_date
                WHERE s.bid >= 0 AND s.ask >= s.bid
                  AND l.bid >= 0 AND l.ask >= l.bid
            ),
            enriched AS (
                SELECT *,
                    (entry_credit_realistic - close_debit)
                      / nullif(entry_credit_realistic, 0) AS captured,
                    mark_date = max(mark_date) OVER (PARTITION BY trade_id)
                      AS is_final_mark
                FROM marks
            ),
            rules(exit_rule) AS (
                VALUES ('capture_50'), ('capture_75'), ('short_delta_75')
            ),
            eligible AS (
                SELECT m.*, r.exit_rule,
                    CASE r.exit_rule
                      WHEN 'capture_50' THEN captured >= 0.50 OR is_final_mark
                      WHEN 'capture_75' THEN captured >= 0.75 OR is_final_mark
                      WHEN 'short_delta_75' THEN abs(short_delta_exit) >= 0.75 OR is_final_mark
                    END AS should_exit
                FROM enriched m CROSS JOIN rules r
            )
            SELECT
                * EXCLUDE (should_exit, mark_date, close_debit, captured, is_final_mark),
                mark_date AS exit_date,
                close_debit AS exit_debit_realistic,
                (entry_credit_realistic - close_debit) * 100.0
                  AS pnl_per_spread_realistic,
                (entry_credit_realistic - close_debit) * 100.0
                  / max_loss_realistic AS return_on_risk_realistic
            FROM eligible
            WHERE should_exit
            QUALIFY row_number() OVER (
                PARTITION BY trade_id, exit_rule ORDER BY mark_date
            ) = 1
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def build_daily_mtm(
    connection: duckdb.DuckDBPyConnection,
    outcomes_path: str | Path,
    output_path: str | Path,
) -> None:
    """Build genuine daily option P&L for selected trade outcomes."""
    outcomes = Path(outcomes_path).resolve().as_posix().replace("'", "''")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output_sql = output.as_posix().replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW selected_outcomes AS SELECT * FROM read_parquet('{outcomes}')"
    )
    columns = {
        row[0]
        for row in connection.execute("DESCRIBE selected_outcomes").fetchall()
    }
    if "exit_rule" in columns:
        rule_expression = "CAST(o.exit_rule AS VARCHAR)"
    elif "exit_dte" in columns:
        rule_expression = "'dte_' || CAST(o.exit_dte AS VARCHAR)"
    else:
        raise ValueError("outcomes must contain exit_rule or exit_dte")
    connection.execute(
        f"""
        COPY (
            WITH marks AS (
                SELECT
                    o.trade_id,
                    o.base_strategy_id,
                    {rule_expression} AS exit_rule,
                    o.entry_date,
                    o.exit_date,
                    s.trade_date AS mark_date,
                    o.contracts_realistic,
                    CASE
                      WHEN s.trade_date = o.exit_date THEN o.exit_debit_realistic
                      ELSE s.mid - l.mid
                    END AS marked_debit,
                    o.entry_credit_realistic
                FROM selected_outcomes o
                JOIN surface s ON s.option_symbol = o.short_symbol
                  AND s.trade_date BETWEEN o.entry_date AND o.exit_date
                JOIN surface l ON l.option_symbol = o.long_symbol
                  AND l.trade_date = s.trade_date
            ),
            trade_paths AS (
                SELECT
                    *,
                    (entry_credit_realistic - marked_debit) * 100.0
                      AS cumulative_trade_pnl_one_lot,
                    (entry_credit_realistic - marked_debit) * 100.0
                      * contracts_realistic AS cumulative_trade_pnl
                FROM marks
            ),
            changes AS (
                SELECT
                    *,
                    cumulative_trade_pnl_one_lot
                      - lag(cumulative_trade_pnl_one_lot, 1, 0.0)
                      OVER (PARTITION BY trade_id, exit_rule ORDER BY mark_date)
                      AS daily_trade_pnl_one_lot,
                    cumulative_trade_pnl - lag(cumulative_trade_pnl, 1, 0.0)
                      OVER (PARTITION BY trade_id, exit_rule ORDER BY mark_date) AS daily_trade_pnl
                FROM trade_paths
            )
            SELECT
                base_strategy_id,
                exit_rule,
                mark_date,
                sum(daily_trade_pnl) AS daily_option_pnl,
                sum(daily_trade_pnl_one_lot) AS daily_option_pnl_one_lot,
                count(DISTINCT trade_id) AS positions_marked
            FROM changes
            GROUP BY base_strategy_id, exit_rule, mark_date
            ORDER BY base_strategy_id, exit_rule, mark_date
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
