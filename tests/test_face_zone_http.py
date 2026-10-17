"""Two synthetic videos exercise local face matching and zone decisions over localhost."""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import requests

from cameras.video_source import VideoFileSource
from clients.central_client import CentralClient, RequestsTransport
from edge.detection import DetectionProcessor
from edge.face_pipeline import FacePipeline, FaceRuntimeSettings
from edge.run_gateway import run_gateway
from edge.shared_gateway import CameraRuntime, SharedGateway
from edge.spatial import SpatialEngine
from tests.test_action_http import synthetic_video
from tests.test_face_matching import registered, unit_vector
from tests.test_week1b_http import local_server


class Detector:
    def detect(self, frame):
        return None


class Tracker:
    def update(self, boxes, frame):
        return np.array([[8, 4, 40, 44, 7, 0.9, 0, 0]])


class Extractor:
    def extract_person(self, frame, bbox):
        return unit_vector(), 0.9


def test_two_cameras_only_unauthorized_zone_emits_experimental_high_candidate(tmp_path):
    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    synthetic_video(first, 3)
    synthetic_video(second, 3)
    rules = {"schema_version": 1, "cameras": {
        camera: {"restricted_zones": [{"id": "storage", "polygon": [
            [0, 0], [1, 0], [1, 1], [0, 1],
        ]}]} for camera in ("camera_01", "camera_02")
    }}
    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        response = reader.put(origin + "/api/v1/face-registry/employee_01", json=registered(
            "ตัวอย่าง", [{"camera_id": "camera_01", "zone_id": "storage"}]), timeout=5)
        assert response.status_code == 201
        transport = RequestsTransport(origin)
        try:
            client = CentralClient(transport)
            face = FacePipeline(Extractor(), client, FaceRuntimeSettings(
                model_version="synthetic-v1", sample_every_frames=1,
                max_samples_per_track=3, max_tracks_per_frame=1))
            processor = DetectionProcessor(
                Detector(), ["camera_01", "camera_02"],
                tracker_factory=lambda _: Tracker(), spatial=SpatialEngine(rules), face=face,
            )
            gateway = SharedGateway(
                [CameraRuntime("camera_01", VideoFileSource(first)),
                 CameraRuntime("camera_02", VideoFileSource(second))],
                processor, client, gateway_id="synthetic-gateway", gateway_boot_id="boot",
                configuration_version="face-zone-test", model_versions={"face": "synthetic-v1"},
                clock=lambda: datetime(2026, 9, 28, tzinfo=timezone.utc),
                event_id_factory=uuid4,
            )
            summary = run_gateway(gateway, fps=30, sleep=lambda _: None)
            assert summary.failures == 0
            assert summary.events_sent == 1
            events = reader.get(origin + "/api/v1/events", timeout=5).json()["events"]
            assert [(event["camera_id"], event["candidate_type"]) for event in events] == [
                ("camera_02", "unauthorized_zone_entry")]
            assert events[0]["attributes"] == {
                "restricted_zone_id": "storage", "person_id": "employee_01",
                "severity": "HIGH", "experimental": True,
            }
            assert "vector" not in str(events)
            dashboard = reader.get(origin + "/", timeout=5)
            assert dashboard.status_code == 200
            assert all(heading in dashboard.text for heading in (
                "<th>Severity</th>", "<th>Identity</th>", "<th>Zone</th>",
                "<th>Validation</th>",
            ))
        finally:
            transport.close()


def test_runner_face_flag_connects_local_videos_to_zone_candidates(tmp_path, monkeypatch):
    from edge.run_gateway import main

    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    synthetic_video(first, 3)
    synthetic_video(second, 3)
    spatial = tmp_path / "spatial.yaml"
    spatial.write_text(
        "schema_version: 1\ncameras:\n"
        "  a:\n    restricted_zones:\n      - id: storage\n"
        "        polygon: [[0, 0], [1, 0], [1, 1], [0, 1]]\n"
        "  b:\n    restricted_zones:\n      - id: storage\n"
        "        polygon: [[0, 0], [1, 0], [1, 1], [0, 1]]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("edge.detection.SharedYOLODetector", lambda *a, **k: Detector())
    monkeypatch.setattr("edge.detection.create_bytetracker", lambda rate: Tracker())
    monkeypatch.setattr("edge.face_pipeline.FaceExtractor", lambda *a, **k: Extractor())
    monkeypatch.setattr("edge.face_pipeline.load_face_settings", lambda path: FaceRuntimeSettings(
        model_version="synthetic-v1", sample_every_frames=1,
        max_samples_per_track=3, max_tracks_per_frame=1))
    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        response = reader.put(origin + "/api/v1/face-registry/employee_01", json=registered(
            "Example", [{"camera_id": "a", "zone_id": "storage"}]), timeout=5)
        assert response.status_code == 201
        assert main([
            "--camera", f"a={first}", "--camera", f"b={second}",
            "--central-url", origin, "--spatial-config", str(spatial),
            "--face", "--fps", "10", "--max-cycles", "3",
        ]) == 0
        events = reader.get(origin + "/api/v1/events", timeout=5).json()["events"]
        assert [(event["camera_id"], event["candidate_type"]) for event in events] == [
            ("b", "unauthorized_zone_entry")]
