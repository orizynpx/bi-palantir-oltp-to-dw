import logging
import os
import time
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

OLTP_HOST = os.getenv("OLTP_HOST", "oltp")
OLTP_PORT = os.getenv("OLTP_PORT", "5432")
OLTP_NAME = os.getenv("OLTP_NAME", "palantir_oltp_pg")
OLTP_USER = os.getenv("OLTP_USER", "admin")
OLTP_PASS = os.getenv("OLTP_PASSWORD", "password")

DATABASE_URL = f"postgresql://{OLTP_USER}:{OLTP_PASS}@{OLTP_HOST}:{OLTP_PORT}/{OLTP_NAME}"

engine = create_engine(DATABASE_URL)

def extract_table(table_name: str) -> tuple[pd.DataFrame, dict]:
    start_time = time.time()

    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, con=engine)

    end_time = time.time()
    duration = end_time - start_time
    row_count = len(df)
    throughput = row_count / duration if duration > 0 else row_count

    metrics = {
        "table": table_name,
        "rows_extracted": row_count,
        "duration_seconds": round(duration, 4),
        "throughput_rows_per_sec": round(throughput, 2),
    }

    logger.info(
        f"[EXTRACT] {table_name}: {row_count} rows extracted in "
        f"{metrics['duration_seconds']}s ({metrics['throughput_rows_per_sec']} rows/s)"
    )
    return df, metrics

def extract_all_oltp_data() -> tuple[dict[str, pd.DataFrame], list[dict]]:
    tables_to_extract = [
        "edge_sensors",
        "mission_deployments",
        "target_categories",
        "effectors",
        "mission_sensors",
        "sensor_telemetry_logs",
        "threat_detections",
        "targeting_effector_pairings",
        "operator_decision_logs",
    ]

    raw_data = {}
    all_metrics = []

    logger.info("Extracting OLTP data")
    for table in tables_to_extract:
        df, metrics = extract_table(table)
        raw_data[table] = df
        all_metrics.append(metrics)

    return raw_data, all_metrics

def extract() -> tuple[dict[str, pd.DataFrame], list[dict]]:
    logger.info("Extracting started")
    raw_data, metrics = extract_all_oltp_data()
    logger.info("Extracting finished")
    return raw_data, metrics