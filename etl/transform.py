import pandas as pd
import logging

def transform(raw_data: dict[str, pd.DataFrame], dimensions: list[dict]) -> dict[str, pd.DataFrame]:
    """
    Pure transformation orchestrator.

    Accepts extracted raw datasets and dimension lookup dictionaries, transforms them
    according to DW target schemas, and returns a dict of ready-to-load DataFrames.    
    """

    logger = logging.getLogger(__name__)
    logger.info('Starting transformation process')

    dim_df = pd.DataFrame(dimensions)
    dim_lookups = {}
    if not dim_df.empty and "dimension_type" in dim_df.columns:
        for dim_type, group in dim_df.groupby("dimension_type"):
            dim_lookups[dim_type] = group.drop(
                columns=["dimension_type"]
            ).dropna(how="all")

    fact_tables = {}

    if "killchain" in raw_data and not raw_data["killchain"].empty:
        fact_tables["fact_killchain_decisions"] = transform_to_fact_killchain(
            raw_data["killchain"], dim_lookups
        )

    if (
        "threat_detections" in raw_data
        and not raw_data["threat_detections"].empty
    ):
        fact_tables["fact_threat_detections"] = (
            transform_to_fact_threat_detections(
                raw_data["threat_detections"], dim_lookups
            )
        )

    if "telemetry" in raw_data and not raw_data["telemetry"].empty:
        raw_mission_sensors = raw_data.get(
            "mission_sensors", pd.DataFrame()
        )
        fact_tables["fact_sensor_telemetry_snapshot"] = (
            transform_to_fact_sensor_telemetry_snapshot(
                raw_data["telemetry"], raw_mission_sensors, dim_lookups
            )
        )

    logger.info('Transformation process completed')
    return fact_tables

def transform_to_fact_killchain(raw_df: pd.DataFrame, dim_lookups: dict) -> pd.DataFrame:
    """Transforms raw killchain workflow data into fact_killchain_decisions schema"""

    logger = logging.getLogger(__name__)
    logger.info('Starting transformation to fact_killchain')

    df = raw_df.copy()

    df["detected_at"] = pd.to_datetime(df["detected_at"])
    df["paired_at"] = pd.to_datetime(df["paired_at"])
    df["decided_at"] = pd.to_datetime(df["decided_at"])

    df["kill_chain_latency_ms"] = (
        (df["paired_at"] - df["detected_at"]).dt.total_seconds() * 1000
    ).astype(int)
    df["operator_response_latency_ms"] = (
        (df["decided_at"] - df["paired_at"]).dt.total_seconds() * 1000
    ).astype(int)
    df["total_end_to_end_latency_ms"] = (
        (df["decided_at"] - df["detected_at"]).dt.total_seconds() * 1000
    ).astype(int)

    df["decided_date_key"] = (
        df["decided_at"].dt.strftime("%Y%m%d").astype(int)
    )
    df["is_authorized"] = (df["decision_type"] == "AUTHORIZE").astype(int)
    df["is_overridden"] = (df["decision_type"] == "OVERRIDE").astype(int)

    df = df.merge(
        dim_lookups.get("dim_sensor", pd.DataFrame()),
        on="sensor_id",
        how="left",
    )
    df = df.merge(
        dim_lookups.get("dim_mission", pd.DataFrame()),
        on="mission_id",
        how="left",
    )
    df = df.merge(
        dim_lookups.get("dim_target_category", pd.DataFrame()),
        on="category_id",
        how="left",
    )
    df = df.merge(
        dim_lookups.get("dim_effector", pd.DataFrame()),
        on="effector_id",
        how="left",
    )
    df = df.merge(
        dim_lookups.get("dim_operator", pd.DataFrame()),
        on="operator_id",
        how="left",
    )

    for col in [
        "sensor_key",
        "mission_key",
        "category_key",
        "effector_key",
        "operator_key",
    ]:
        df[col] = df[col].fillna(-1).astype(int) if col in df.columns else -1

    logger.info('Transformation to fact_killchain completed')

    return df[[
        "sensor_key",
        "mission_key",
        "category_key",
        "effector_key",
        "operator_key",
        "decided_date_key",
        "pairing_status",
        "decision_type",
        "operational_context",
        "kill_chain_latency_ms",
        "operator_response_latency_ms",
        "total_end_to_end_latency_ms",
        "is_authorized",
        "is_overridden",
    ]].copy()

def transform_to_fact_threat_detections(raw_df: pd.DataFrame, dim_lookups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Transforms raw threat detections into fact_threat_detections schema."""

    logger = logging.getLogger(__name__)
    logger.info('Starting transformation to fact_threat_detections')

    df = raw_df.copy()
    df["detected_at"] = pd.to_datetime(df["detected_at"])
    df["detected_date_key"] = (
        df["detected_at"].dt.strftime("%Y%m%d").astype(int)
    )
    df["detection_count"] = 1

    df = df.merge(
        dim_lookups.get("dim_sensor", pd.DataFrame()),
        on="sensor_id",
        how="left",
    )
    df = df.merge(
        dim_lookups.get("dim_mission", pd.DataFrame()),
        on="mission_id",
        how="left",
    )
    df = df.merge(
        dim_lookups.get("dim_target_category", pd.DataFrame()),
        on="category_id",
        how="left",
    )

    for col in ["sensor_key", "mission_key", "category_key"]:
        df[col] = df[col].fillna(-1).astype(int) if col in df.columns else -1

    logger.info('Transformation to fact_threat_detections completed')

    return df[[
        "sensor_key",
        "mission_key",
        "category_key",
        "detected_date_key",
        "edge_model_version",
        "detection_id",
        "confidence_score",
        "detection_count",
    ]].copy()

def transform_to_fact_sensor_telemetry_snapshot(
        raw_telemetry_df: pd.DataFrame,
        raw_mission_sensors_df: pd.DataFrame,
        dim_lookups: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
    """Aggregates sensor telemetry logs into 1-minute snapshot buckets."""

    logger = logging.getLogger(__name__)
    logger.info('Starting transformation to fact_sensor_telemetry_snapshot')

    df = raw_telemetry_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if not raw_mission_sensors_df.empty:
        ms = raw_mission_sensors_df.copy()
        ms["assigned_at"] = pd.to_datetime(ms["assigned_at"])
        ms["unassigned_at"] = pd.to_datetime(
            ms["unassigned_at"]
        ).fillna(pd.Timestamp.max)

        df = df.merge(
            ms[["mission_id", "sensor_id", "assigned_at", "unassigned_at"]],
            on="sensor_id",
            how="left",
        )
        mask = (df["timestamp"] >= df["assigned_at"]) & (
            df["timestamp"] <= df["unassigned_at"]
        )
        df = df[mask].drop(columns=["assigned_at", "unassigned_at"])
    else:
        if "mission_id" not in df.columns:
            df["mission_id"] = -1

    df["snapshot_bucket"] = df["timestamp"].dt.floor("1min")
    aggregated = (
        df.groupby(["sensor_id", "mission_id", "snapshot_bucket"])
        .agg(
            avg_altitude_meters=("altitude_meters", "mean"),
            min_battery_bandwidth_pct=("battery_bandwidth_pct", "min"),
            telemetry_event_count=("timestamp", "count"),
            network_uptime_seconds=(
                "network_connected",
                lambda x: int(x.sum() * 5),
            ),
        )
        .reset_index()
    )

    aggregated["snapshot_date_key"] = (
        aggregated["snapshot_bucket"].dt.strftime("%Y%m%d").astype(int)
    )
    aggregated["snapshot_time_key"] = (
        aggregated["snapshot_bucket"].dt.strftime("%H%M%S").astype(int)
    )

    aggregated = aggregated.merge(
        dim_lookups.get("dim_sensor", pd.DataFrame()),
        on="sensor_id",
        how="left",
    )
    aggregated = aggregated.merge(
        dim_lookups.get("dim_mission", pd.DataFrame()),
        on="mission_id",
        how="left",
    )

    for col in ["sensor_key", "mission_key"]:
        aggregated[col] = (
            aggregated[col].fillna(-1).astype(int)
            if col in aggregated.columns
            else -1
        )

    logger.info('Transformation to fact_sensor_telemetry_snapshot completed')

    return aggregated[[
        "sensor_key",
        "mission_key",
        "snapshot_date_key",
        "snapshot_time_key",
        "avg_altitude_meters",
        "min_battery_bandwidth_pct",
        "network_uptime_seconds",
        "telemetry_event_count",
    ]].copy()
