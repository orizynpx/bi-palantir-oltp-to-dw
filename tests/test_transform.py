import unittest

import pandas as pd

from etl.transform import transform


class TransformTests(unittest.TestCase):
    def test_sensor_dimension_assigns_sorted_surrogate_keys(self):
        raw_data = {
            "edge_sensors": pd.DataFrame(
                [
                    {"sensor_id": "S-2", "sensor_name": "Beta"},
                    {"sensor_id": "S-1", "sensor_name": "Alpha"},
                ]
            )
        }

        dw_tables, _ = transform(raw_data)
        actual = dw_tables["dim_sensor"].set_index("sensor_id")["sensor_key"].to_dict()

        self.assertEqual(actual, {"S-1": 1, "S-2": 2})

    def test_detection_fact_maps_keys_and_date(self):
        raw_data = {
            "edge_sensors": pd.DataFrame([{"sensor_id": "S-1"}]),
            "mission_deployments": pd.DataFrame([{"mission_id": 10}]),
            "target_categories": pd.DataFrame([{"category_id": 20}]),
            "threat_detections": pd.DataFrame(
                [
                    {
                        "detection_id": 100,
                        "sensor_id": "S-1",
                        "mission_id": 10,
                        "category_id": 20,
                        "detected_at": "2026-08-01T10:15:30Z",
                        "edge_model_version": "v1",
                        "confidence_score": 0.95,
                    }
                ]
            ),
        }

        dw_tables, _ = transform(raw_data)
        actual = dw_tables["fact_threat_detections"].iloc[0]

        self.assertEqual(
            (
                actual["sensor_key"],
                actual["mission_key"],
                actual["category_key"],
                actual["detected_date_key"],
                actual["detection_count"],
            ),
            (1, 1, 1, 20260801, 1),
        )

    def test_killchain_fact_calculates_latencies_and_decision_flags(self):
        raw_data = {
            "edge_sensors": pd.DataFrame([{"sensor_id": "S-1"}]),
            "mission_deployments": pd.DataFrame([{"mission_id": 10}]),
            "target_categories": pd.DataFrame([{"category_id": 20}]),
            "effectors": pd.DataFrame([{"effector_id": "E-1"}]),
            "threat_detections": pd.DataFrame(
                [
                    {
                        "detection_id": 100,
                        "sensor_id": "S-1",
                        "mission_id": 10,
                        "category_id": 20,
                        "detected_at": "2026-08-01T10:00:00Z",
                    }
                ]
            ),
            "targeting_effector_pairings": pd.DataFrame(
                [
                    {
                        "pairing_id": 200,
                        "detection_id": 100,
                        "effector_id": "E-1",
                        "paired_at": "2026-08-01T10:00:02.500Z",
                        "pairing_status": "EXECUTED",
                    }
                ]
            ),
            "operator_decision_logs": pd.DataFrame(
                [
                    {
                        "pairing_id": 200,
                        "operator_id": "OP-1",
                        "decided_at": "2026-08-01T10:00:05Z",
                        "decision_type": "AUTHORIZE",
                        "operational_context": "Test",
                    }
                ]
            ),
        }

        dw_tables, _ = transform(raw_data)
        actual = dw_tables["fact_killchain_decisions"].iloc[0]

        self.assertEqual(
            (
                actual["kill_chain_latency_ms"],
                actual["operator_response_latency_ms"],
                actual["total_end_to_end_latency_ms"],
                actual["is_authorized"],
                actual["is_overridden"],
            ),
            (2500, 2500, 5000, 1, 0),
        )

    def test_telemetry_is_aggregated_per_sensor_mission_and_minute(self):
        raw_data = {
            "edge_sensors": pd.DataFrame([{"sensor_id": "S-1"}]),
            "mission_deployments": pd.DataFrame([{"mission_id": 10}]),
            "mission_sensors": pd.DataFrame(
                [
                    {
                        "mission_id": 10,
                        "sensor_id": "S-1",
                        "assigned_at": "2026-08-01T10:00:00Z",
                        "unassigned_at": None,
                    }
                ]
            ),
            "sensor_telemetry_logs": pd.DataFrame(
                [
                    {
                        "sensor_id": "S-1",
                        "timestamp": "2026-08-01T10:01:05Z",
                        "altitude_meters": 100,
                        "battery_bandwidth_pct": 80,
                        "network_connected": True,
                    },
                    {
                        "sensor_id": "S-1",
                        "timestamp": "2026-08-01T10:01:45Z",
                        "altitude_meters": 120,
                        "battery_bandwidth_pct": 70,
                        "network_connected": False,
                    },
                ]
            ),
        }

        dw_tables, _ = transform(raw_data)
        actual = dw_tables["fact_sensor_telemetry_snapshot"].iloc[0]

        self.assertEqual(
            (
                actual["avg_altitude_meters"],
                actual["min_battery_bandwidth_pct"],
                actual["network_uptime_seconds"],
                actual["telemetry_event_count"],
            ),
            (110.0, 70, 5, 2),
        )


if __name__ == "__main__":
    unittest.main()