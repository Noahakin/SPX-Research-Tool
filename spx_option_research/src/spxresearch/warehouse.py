from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class ArchiveAudit:
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    file_dates: int
    populated_dates: int
    empty_dates: int
    total_rows: int
    minimum_dte: int
    maximum_dte: int
    duplicate_rows: int
    unbalanced_dates: int


class ResearchWarehouse:
    """DuckDB-backed, out-of-core query layer over daily surface parquets."""

    def __init__(
        self,
        database_path: str | Path,
        surface_root: str | Path,
        *,
        threads: int = 4,
        memory_limit: str = "6GB",
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.surface_root = Path(surface_root).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.database_path))
        self.connection.execute(f"SET threads={int(threads)}")
        self.connection.execute(f"SET memory_limit='{memory_limit}'")
        temp_directory = (self.database_path.parent / "duckdb_tmp").as_posix().replace("'", "''")
        self.connection.execute(f"SET temp_directory='{temp_directory}'")
        self._create_surface_view()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ResearchWarehouse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_surface_view(self) -> None:
        glob = (self.surface_root / "**" / "eod.parquet").as_posix().replace("'", "''")
        self.connection.execute(
            f"""
            CREATE OR REPLACE VIEW surface AS
            SELECT
                CAST(snapshot_date AS DATE) AS trade_date,
                CAST(expiration_date AS DATE) AS expiration_date,
                CAST(dte AS INTEGER) AS dte,
                CAST(underlying_price AS DOUBLE) AS spot,
                CAST(strike AS DOUBLE) AS strike,
                lower(CAST(option_type AS VARCHAR)) AS option_type,
                CAST(bid AS DOUBLE) AS bid,
                CAST(ask AS DOUBLE) AS ask,
                CAST(mid AS DOUBLE) AS mid,
                CAST(last AS DOUBLE) AS last,
                CAST(volume AS BIGINT) AS volume,
                CAST(open_interest AS BIGINT) AS open_interest,
                CAST(implied_volatility AS DOUBLE) AS iv,
                CAST(delta AS DOUBLE) AS delta,
                CAST(gamma AS DOUBLE) AS gamma,
                CAST(theta AS DOUBLE) AS theta,
                CAST(vega AS DOUBLE) AS vega,
                CAST(rho AS DOUBLE) AS rho,
                trim(CAST(option_symbol AS VARCHAR)) AS option_symbol,
                CASE
                    WHEN upper(trim(CAST(option_symbol AS VARCHAR))) LIKE 'SPXW%' THEN 'PM'
                    WHEN upper(trim(CAST(option_symbol AS VARCHAR))) LIKE 'SPX%' THEN 'AM'
                    ELSE 'UNKNOWN'
                END AS settlement
            FROM read_parquet('{glob}', union_by_name=true, hive_partitioning=true)
            WHERE snapshot_date IS NOT NULL
            """
        )

    def audit(self) -> ArchiveAudit:
        row = self.connection.execute(
            """
            WITH daily AS (
                SELECT
                    trade_date,
                    count(*) AS rows,
                    count_if(option_type = 'call') AS calls,
                    count_if(option_type = 'put') AS puts,
                    min(dte) AS min_dte,
                    max(dte) AS max_dte,
                    count(*) - count(DISTINCT option_symbol) AS duplicates
                FROM surface
                GROUP BY trade_date
            )
            SELECT
                min(trade_date), max(trade_date), count(*), sum(rows),
                min(min_dte), max(max_dte), sum(duplicates),
                count_if(calls <> puts)
            FROM daily
            """
        ).fetchone()
        file_dates = sum(1 for _ in self.surface_root.rglob("eod.parquet"))
        populated = int(row[2])
        return ArchiveAudit(
            first_date=pd.Timestamp(row[0]),
            last_date=pd.Timestamp(row[1]),
            file_dates=file_dates,
            populated_dates=populated,
            empty_dates=file_dates - populated,
            total_rows=int(row[3]),
            minimum_dte=int(row[4]),
            maximum_dte=int(row[5]),
            duplicate_rows=int(row[6]),
            unbalanced_dates=int(row[7]),
        )

    def daily_market(self) -> pd.DataFrame:
        return self.connection.execute(
            """
            SELECT
                trade_date,
                median(spot) AS spx_spot,
                count(*) AS option_rows,
                count(DISTINCT expiration_date) AS expirations,
                min(dte) AS min_dte,
                max(dte) AS max_dte,
                count_if(settlement = 'PM')::DOUBLE / count(*) AS pm_fraction
            FROM surface
            GROUP BY trade_date
            ORDER BY trade_date
            """
        ).fetchdf().set_index("trade_date")

    def execute(self, sql: str, parameters: list[object] | None = None) -> duckdb.DuckDBPyConnection:
        return self.connection.execute(sql, parameters or [])

