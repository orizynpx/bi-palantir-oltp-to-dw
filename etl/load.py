import logging
import os
import time
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

DW_HOST = os.getenv("DW_HOST", "dw")
DW_PORT = os.getenv("DW_PORT", "5432")
DW_NAME = os.getenv("DW_NAME", "palantir_dw_pg")
DW_USER = os.getenv("DW_USER", "admin")
DW_PASS = os.getenv("DW_PASSWORD", "password")

DATABASE_URL = f"postgresql://{DW_USER}:{DW_PASS}@{DW_HOST}:{DW_PORT}/{DW_NAME}"
engine = create_engine(DATABASE_URL)

def load_table(df: pd.DataFrame, table_name: str, if_exists: str = "append") -> dict:
    if df.empty:
        logger.warning(f"[LOAD] Skipping {table_name}: DataFrame is empty.")
        return {
            "table": table_name,
            "rows_loaded": 0,
            "duration_seconds": 0.0,
            "throughput_rows_per_sec": 0.0,
        }

    start_time = time.time()

    with engine.begin() as connection:
        df.to_sql(
            name=table_name,
            con=connection,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=1000,
        )

    end_time = time.time()
    duration = end_time - start_time
    row_count = len(df)
    throughput = row_count / duration if duration > 0 else row_count

    metrics = {
        "table": table_name,
        "rows_loaded": row_count,
        "duration_seconds": round(duration, 4),
        "throughput_rows_per_sec": round(throughput, 2),
    }

    logger.info(
        f"[LOAD] {table_name}: {row_count} rows loaded in "
        f"{metrics['duration_seconds']}s ({metrics['throughput_rows_per_sec']} rows/s)"
    )
    return metrics


def load_all_dw_data(dw_tables: dict[str, pd.DataFrame]) -> list[dict]:
    all_metrics = []

    load_order = [
        "dim_sensor",
        "dim_mission",
        "dim_target_category",
        "dim_effector",
        "dim_operator",
        "dim_geography",
        "fact_threat_detections",
        "fact_killchain_decisions",
        "fact_sensor_telemetry_snapshot",
    ]

    logger.info("Loading DW data into target PostgreSQL instance...")

    for table_name in load_order:
        if table_name in dw_tables:
            metrics = load_table(dw_tables[table_name], table_name)
            all_metrics.append(metrics)
        else:
            logger.warning(f"[LOAD] Table '{table_name}' not provided in transformed data.")

    return all_metrics


def load(dw_tables: dict[str, pd.DataFrame] = None) -> list[dict]:
    logger.info("Loading started")
    
    if dw_tables is None:
        logger.warning("No data passed to load(). Exiting load phase.")
        return []

    metrics = load_all_dw_data(dw_tables)
    logger.info("Loading finished")
    return metrics