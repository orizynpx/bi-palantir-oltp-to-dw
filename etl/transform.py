import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


def _dimension_keys(frame: pd.DataFrame, natural_key: str) -> dict:
    if natural_key not in frame:
        return {}
    values = frame[natural_key].dropna().drop_duplicates().sort_values().tolist()
    return {value: index for index, value in enumerate(values, start=1)}


def _date_key(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True)
    return timestamps.dt.strftime("%Y%m%d").astype("int64")


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _build_dimensions(raw_data: dict[str, pd.DataFrame]) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
    sensors = raw_data.get("edge_sensors", pd.DataFrame()).copy()
    missions = raw_data.get("mission_deployments", pd.DataFrame()).copy()
    categories = raw_data.get("target_categories", pd.DataFrame()).copy()
    effectors = raw_data.get("effectors", pd.DataFrame()).copy()
    decisions = raw_data.get("operator_decision_logs", pd.DataFrame()).copy()

    sensor_keys = _dimension_keys(sensors, "sensor_id")
    mission_keys = _dimension_keys(missions, "mission_id")
    category_keys = _dimension_keys(categories, "category_id")
    effector_keys = _dimension_keys(effectors, "effector_id")

    sensors["sensor_key"] = sensors["sensor_id"].map(sensor_keys) if "sensor_id" in sensors else pd.Series(dtype="int64")
    if "created_at" in sensors:
        sensors["valid_from"] = pd.to_datetime(sensors["created_at"], utc=True).dt.date
    else:
        sensors["valid_from"] = None
    sensors["valid_to"] = None
    sensors["is_current"] = True
    missions["mission_key"] = missions["mission_id"].map(mission_keys) if "mission_id" in missions else pd.Series(dtype="int64")
    categories["category_key"] = categories["category_id"].map(category_keys) if "category_id" in categories else pd.Series(dtype="int64")
    effectors["effector_key"] = effectors["effector_id"].map(effector_keys) if "effector_id" in effectors else pd.Series(dtype="int64")

    operator_ids = (
        decisions["operator_id"].dropna().drop_duplicates().sort_values().tolist()
        if "operator_id" in decisions else []
    )
    operator_keys = {value: index for index, value in enumerate(operator_ids, start=1)}
    operators = pd.DataFrame({
        "operator_key": list(operator_keys.values()),
        "operator_id": list(operator_keys.keys()),
        "command_post_unit": [None] * len(operator_keys),
        "clearance_level": [None] * len(operator_keys),
    })

    dimensions = {
        "dim_sensor": sensors.reindex(columns=[
            "sensor_key", "sensor_id", "sensor_name", "sensor_type", "status",
            "is_current", "valid_from", "valid_to",
        ]),
        "dim_mission": missions.reindex(columns=[
            "mission_key", "mission_id", "mission_code", "mission_status", "priority",
        ]),
        "dim_target_category": categories.reindex(columns=[
            "category_key", "category_id", "category_name", "threat_level",
        ]),
        "dim_effector": effectors.reindex(columns=[
            "effector_key", "effector_id", "effector_name", "effector_type",
            "max_range_km", "is_current",
        ]),
        "dim_operator": operators,
    }
    key_maps = {
        "sensor": sensor_keys,
        "mission": mission_keys,
        "category": category_keys,
        "effector": effector_keys,
        "operator": operator_keys,
    }
    return key_maps, dimensions


def _transform_threat_detections(
    raw_df: pd.DataFrame, key_maps: dict[str, dict]
) -> pd.DataFrame:
    columns = [
        "detection_fact_id", "sensor_key", "mission_key", "category_key",
        "detected_date_key", "edge_model_version", "detection_id",
        "confidence_score", "detection_count",
    ]
    if raw_df.empty:
        return _empty_frame(columns)

    frame = raw_df.copy()
    frame["detection_fact_id"] = range(1, len(frame) + 1)
    frame["sensor_key"] = frame["sensor_id"].map(key_maps["sensor"])
    frame["mission_key"] = frame["mission_id"].map(key_maps["mission"])
    frame["category_key"] = frame["category_id"].map(key_maps["category"])
    frame["detected_date_key"] = _date_key(frame["detected_at"])
    frame["detection_count"] = 1
    return frame.reindex(columns=columns)


def _transform_killchain(
    raw_data: dict[str, pd.DataFrame], key_maps: dict[str, dict]
) -> pd.DataFrame:
    columns = [
        "decision_fact_id", "sensor_key", "mission_key", "category_key",
        "effector_key", "operator_key", "decided_date_key", "pairing_status",
        "decision_type", "operational_context", "kill_chain_latency_ms",
        "operator_response_latency_ms", "total_end_to_end_latency_ms",
        "is_authorized", "is_overridden",
    ]
    detections = raw_data.get("threat_detections", pd.DataFrame())
    pairings = raw_data.get("targeting_effector_pairings", pd.DataFrame())
    decisions = raw_data.get("operator_decision_logs", pd.DataFrame())
    if detections.empty or pairings.empty or decisions.empty:
        return _empty_frame(columns)

    frame = detections.merge(pairings, on="detection_id", how="inner")
    frame = frame.merge(decisions, on="pairing_id", how="inner")
    for column in ("detected_at", "paired_at", "decided_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)

    frame["decision_fact_id"] = range(1, len(frame) + 1)
    for source, target in (
        ("sensor_id", "sensor_key"), ("mission_id", "mission_key"),
        ("category_id", "category_key"), ("effector_id", "effector_key"),
        ("operator_id", "operator_key"),
    ):
        map_name = target.removesuffix("_key")
        frame[target] = frame[source].map(key_maps[map_name])
    frame["decided_date_key"] = _date_key(frame["decided_at"])
    frame["kill_chain_latency_ms"] = (
        (frame["paired_at"] - frame["detected_at"]).dt.total_seconds() * 1000
    ).round().astype("int64")
    frame["operator_response_latency_ms"] = (
        (frame["decided_at"] - frame["paired_at"]).dt.total_seconds() * 1000
    ).round().astype("int64")
    frame["total_end_to_end_latency_ms"] = (
        (frame["decided_at"] - frame["detected_at"]).dt.total_seconds() * 1000
    ).round().astype("int64")
    frame["is_authorized"] = frame["decision_type"].eq("AUTHORIZE").astype(int)
    frame["is_overridden"] = frame["decision_type"].eq("OVERRIDE").astype(int)
    return frame.reindex(columns=columns)


def _transform_telemetry(
    telemetry: pd.DataFrame,
    mission_sensors: pd.DataFrame,
    key_maps: dict[str, dict],
) -> pd.DataFrame:
    columns = [
        "telemetry_snapshot_id", "sensor_key", "mission_key", "snapshot_date_key",
        "snapshot_time_key", "avg_altitude_meters", "min_battery_bandwidth_pct",
        "network_uptime_seconds", "telemetry_event_count",
    ]
    if telemetry.empty:
        return _empty_frame(columns)

    frame = telemetry.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if not mission_sensors.empty:
        assignments = mission_sensors[[
            "mission_id", "sensor_id", "assigned_at", "unassigned_at",
        ]].copy()
        assignments["assigned_at"] = pd.to_datetime(assignments["assigned_at"], utc=True)
        assignments["unassigned_at"] = pd.to_datetime(assignments["unassigned_at"], utc=True)
        expanded = frame.merge(assignments, on="sensor_id", how="left")
        active = expanded["assigned_at"].isna() | (
            (expanded["timestamp"] >= expanded["assigned_at"])
            & (expanded["unassigned_at"].isna() | (expanded["timestamp"] <= expanded["unassigned_at"]))
        )
        frame = expanded.loc[active].drop(columns=["assigned_at", "unassigned_at"])
    else:
        frame["mission_id"] = pd.NA

    frame["snapshot_bucket"] = frame["timestamp"].dt.floor("min")
    grouped = frame.groupby(
        ["sensor_id", "mission_id", "snapshot_bucket"], dropna=False, as_index=False
    ).agg(
        avg_altitude_meters=("altitude_meters", "mean"),
        min_battery_bandwidth_pct=("battery_bandwidth_pct", "min"),
        telemetry_event_count=("timestamp", "count"),
        network_uptime_seconds=(
            "network_connected",
            lambda values: int(values.fillna(False).sum() * 5),
        ),
    )
    grouped["telemetry_snapshot_id"] = range(1, len(grouped) + 1)
    grouped["sensor_key"] = grouped["sensor_id"].map(key_maps["sensor"])
    grouped["mission_key"] = grouped["mission_id"].map(key_maps["mission"])
    grouped["snapshot_date_key"] = _date_key(grouped["snapshot_bucket"])
    grouped["snapshot_time_key"] = grouped["snapshot_bucket"].dt.strftime("%H%M%S").astype("int64")
    return grouped.reindex(columns=columns)


def _build_date_and_time_dimensions(
    dw_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_keys = []
    for table_name, column in (
        ("fact_threat_detections", "detected_date_key"),
        ("fact_killchain_decisions", "decided_date_key"),
        ("fact_sensor_telemetry_snapshot", "snapshot_date_key"),
    ):
        frame = dw_tables[table_name]
        if not frame.empty:
            date_keys.extend(frame[column].dropna().astype(int).tolist())

    unique_date_keys = sorted(set(date_keys))
    dates = [pd.to_datetime(str(key), format="%Y%m%d").date() for key in unique_date_keys]
    dim_date = pd.DataFrame({
        "date_key": unique_date_keys,
        "full_date": dates,
        "year": [value.year for value in dates],
        "quarter": [((value.month - 1) // 3) + 1 for value in dates],
        "month": [value.month for value in dates],
        "day_of_week": [value.weekday() for value in dates],
    })

    snapshots = dw_tables["fact_sensor_telemetry_snapshot"]
    time_keys = (
        sorted(set(snapshots["snapshot_time_key"].dropna().astype(int).tolist()))
        if not snapshots.empty else []
    )
    dim_time = pd.DataFrame({
        "time_key": time_keys,
        "hour": [value // 10000 for value in time_keys],
        "minute": [(value // 100) % 100 for value in time_keys],
        "second": [value % 100 for value in time_keys],
    })
    return dim_date, dim_time


def transform(raw_data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Build DW dimensions and facts from extracted OLTP tables."""
    started_at = time.perf_counter()
    logger.info("Starting transformation process")

    key_maps, dw_tables = _build_dimensions(raw_data)
    dw_tables["fact_threat_detections"] = _transform_threat_detections(
        raw_data.get("threat_detections", pd.DataFrame()), key_maps
    )
    dw_tables["fact_killchain_decisions"] = _transform_killchain(raw_data, key_maps)
    dw_tables["fact_sensor_telemetry_snapshot"] = _transform_telemetry(
        raw_data.get("sensor_telemetry_logs", pd.DataFrame()),
        raw_data.get("mission_sensors", pd.DataFrame()),
        key_maps,
    )
    dw_tables["dim_date"], dw_tables["dim_time"] = _build_date_and_time_dimensions(dw_tables)

    metrics = [
        {"table": table_name, "rows_transformed": len(frame)}
        for table_name, frame in dw_tables.items()
    ]
    duration = round(time.perf_counter() - started_at, 4)
    metrics.append({"duration_seconds": duration})
    logger.info(
        "Transformation completed: %s rows across %s tables",
        sum(item["rows_transformed"] for item in metrics[:-1]),
        len(dw_tables),
    )
    return dw_tables, metrics
