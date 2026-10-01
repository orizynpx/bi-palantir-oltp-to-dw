import logging
import os
import time
import pandas as pd
from sqlalchemy import Date, create_engine, text

logger = logging.getLogger(__name__)

DW_HOST = os.getenv("DW_HOST", "dw")
DW_PORT = os.getenv("DW_PORT", "5432")
DW_NAME = os.getenv("DW_NAME", "palantir_dw_pg")
DW_USER = os.getenv("DW_USER", "admin")
DW_PASS = os.getenv("DW_PASSWORD", "password")

DATABASE_URL = f"postgresql://{DW_USER}:{DW_PASS}@{DW_HOST}:{DW_PORT}/{DW_NAME}"
engine = create_engine(DATABASE_URL)

LOAD_ORDER = [
    "dim_sensor",
    "dim_mission",
    "dim_target_category",
    "dim_effector",
    "dim_operator",
    "dim_date",
    "dim_time",
    "fact_threat_detections",
    "fact_killchain_decisions",
    "fact_sensor_telemetry_snapshot",
]
TRUNCATE_TABLES = LOAD_ORDER
TABLE_COLUMN_TYPES = {
    "dim_sensor": {"valid_to": Date},
}


def load_all_dw_data(dw_tables: dict[str, pd.DataFrame]) -> list[dict]:
    missing_tables = set(LOAD_ORDER) - set(dw_tables)
    if missing_tables:
        raise ValueError(f"Transformed output is missing tables: {sorted(missing_tables)}")

    all_metrics = []
    logger.info("Loading full DW snapshot into target PostgreSQL instance")
    start_time = time.perf_counter()

    with engine.begin() as connection:
        tables_sql = ", ".join(TRUNCATE_TABLES)
        connection.execute(text(f"TRUNCATE TABLE {tables_sql} RESTART IDENTITY"))

        for table_name in LOAD_ORDER:
            frame = dw_tables[table_name]
            table_start = time.perf_counter()
            if not frame.empty:
                frame.to_sql(
                    name=table_name,
                    con=connection,
                    if_exists="append",
                    index=False,
                    dtype=TABLE_COLUMN_TYPES.get(table_name),
                    method="multi",
                    chunksize=1000,
                )
            duration = time.perf_counter() - table_start
            row_count = len(frame)
            throughput = row_count / duration if duration > 0 else row_count
            metrics = {
                "table": table_name,
                "rows_loaded": row_count,
                "duration_seconds": round(duration, 4),
                "throughput_rows_per_sec": round(throughput, 2),
            }
            logger.info(
                "[LOAD] %s: %s rows loaded in %ss (%s rows/s)",
                table_name,
                row_count,
                metrics["duration_seconds"],
                metrics["throughput_rows_per_sec"],
            )
            all_metrics.append(metrics)

    logger.info("DW snapshot loaded in %.2f seconds", time.perf_counter() - start_time)
    return all_metrics


def load(dw_tables: dict[str, pd.DataFrame] | None = None) -> list[dict]:
    logger.info("Loading started")
    
    if dw_tables is None:
        raise ValueError("No transformed data passed to load().")

    metrics = load_all_dw_data(dw_tables)
    logger.info("Loading finished")
    return metrics