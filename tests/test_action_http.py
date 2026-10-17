"""Two synthetic local videos drive temporal candidates into Central SQLite."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import pytest
import requests

from cameras.video_source import VideoFileSource
from clients.central_client import CentralClient, RequestsTransport
from edge.action_pipeline import ActionProcessor
from edge.action_runtime import ActionEngine, ActionSettings
from edge.run_gateway import main, run_gateway
from edge.shared_gateway import CameraRuntime, SharedGateway
from tests.test_week1b_http import local_server


class SyntheticDetection:
    def process(self, camera_id, frame, state):
        state.clear()
        state[7] = {"bbox": [8, 4, 40, 44], "score": 0.9}
        return ()


class ControlledModel:
    def predict_batch(self, windows):
        assert all(len(window) == 32 for window in windows)
        return [[0.01, 0.01, 0.98] for _ in windows]


def synthetic_video(path, frame_count):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[4:44, 8:40, :] = (index % 10) * 20
            writer.write(frame)
    finally:
        writer.release()


def test_action_candidates_from_two_video_timelines_reach_sqlite(tmp_path):
    first, second = tmp_path / "first.avi", tmp_path / "second.avi"
    synthetic_video(first, 47)
    synthetic_video(second, 47)
    engine = ActionEngine(ControlledModel(), ("a", "b"), ActionSettings())
    processor = ActionProcessor(SyntheticDetection(), engine)
    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        transport = RequestsTransport(origin)
        try:
            ids = iter([UUID(int=1), UUID(int=2)])
            gateway = SharedGateway(
                [CameraRuntime("a", VideoFileSource(first)), CameraRuntime("b", VideoFileSource(second))],
                processor, CentralClient(transport), gateway_id="g", gateway_boot_id="boot",
                configuration_version="action-v1", model_versions={"action": "controlled-test"},
                clock=lambda: datetime(2026, 9, 28, tzinfo=timezone.utc),
                event_id_factory=lambda: next(ids),
            )
            summary = run_gateway(gateway, fps=30, sleep=lambda _: None)
            assert summary.failures == 0
            assert summary.events_sent == 2
            response = reader.get(origin + "/api/v1/events", timeout=5)
            assert response.status_code == 200
            events = response.json()["events"]
            assert {event["camera_id"] for event in events} == {"a", "b"}
            assert {event["candidate_type"] for event in events} == {"possible_person_violence"}
            assert {event["local_track_id"] for event in events} == {7}
            assert all(event["attributes"]["media_time_seconds"] == 4.6 for event in events)
            assert all(event["reason_codes"] == [
                "action_temporal_2_of_3", "threshold_baseline_unvalidated"
            ] for event in events)
        finally:
            transport.close()


def test_action_cli_loads_existing_models_and_waits_for_full_window(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    weights = root / "yolov8n.pt"
    if not weights.is_file():
        pytest.skip("Local detector baseline unavailable; no download attempted")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics"))
    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    synthetic_video(first, 1)
    synthetic_video(second, 1)
    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        result = main([
            "--camera", f"a={first}", "--camera", f"b={second}",
            "--weights", str(weights), "--device", "cpu", "--action",
            "--action-device", "cpu", "--central-url", origin, "--fps", "30",
        ])
        assert result == 0
        assert reader.get(origin + "/api/v1/events", timeout=5).json()["events"] == []
        cameras = reader.get(origin + "/api/v1/cameras", timeout=5).json()["cameras"]
        assert len(cameras) == 2
        assert all("action" in camera["model_versions"] for camera in cameras)
