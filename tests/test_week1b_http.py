"""Actual localhost HTTP round trip with two generated, unidentifiable videos."""

import socket
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import pytest
import requests
import uvicorn

from cameras.video_source import VideoFileSource
from clients.central_client import CentralClient, RequestsTransport
from edge.run_gateway import DemoProcessor, main, run_gateway
from edge.shared_gateway import CameraRuntime, SharedGateway
from server.app import create_app


@contextmanager
def local_server(database_path):
    """Use an ephemeral loopback port and stop the server even when assertions fail."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app = create_app(database_path)
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("Test localhost server did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), "Test server failed to stop"


def make_video(path, count, color):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened(), "MJPG fixture encoder unavailable"
    try:
        for frame_index in range(count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, color] = 40 + frame_index * 20
            writer.write(frame)
    finally:
        writer.release()


def test_two_generated_videos_reach_sqlite_and_dashboard_through_http(tmp_path):
    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    make_video(first, 2, 0)
    make_video(second, 1, 1)
    database = tmp_path / "central.db"
    with local_server(database) as origin, requests.Session() as reader:
        reader.trust_env = False
        transport = RequestsTransport(origin)
        try:
            processor = DemoProcessor()
            ids = iter([UUID(int=1), UUID(int=2)])
            gateway = SharedGateway(
                [CameraRuntime("camera_01", VideoFileSource(first)),
                 CameraRuntime("camera_02", VideoFileSource(second))],
                processor, CentralClient(transport), gateway_id="synthetic-gateway",
                gateway_boot_id="test-boot", configuration_version="fixture-v1",
                model_versions={"processor": "synthetic-v1"},
                clock=lambda: datetime(2026, 9, 28, tzinfo=timezone.utc),
                event_id_factory=lambda: next(ids),
            )
            summary = run_gateway(gateway, fps=10, sleep=lambda _: None)
            assert (summary.cycles, summary.events_sent, summary.heartbeats_sent, summary.failures) == (3, 2, 6, 0)
            response = reader.get(origin + "/api/v1/events", timeout=5)
            assert response.status_code == 200
            events = response.json()["events"]
            assert [(event["camera_id"], event["event_id"]) for event in events] == [
                ("camera_02", "00000000-0000-0000-0000-000000000002"),
                ("camera_01", "00000000-0000-0000-0000-000000000001"),
            ]
            assert all(event["attributes"] == {"synthetic": True, "purpose": "pipeline_smoke_test"}
                       for event in events)
            assert all(event["model_versions"] == {"processor": "synthetic-v1"} for event in events)
            assert all(event["reason_codes"] == ["synthetic_test_only"] for event in events)
            replay = reader.post(origin + "/api/v1/events", json=events[0], timeout=5)
            assert replay.status_code == 200 and replay.json()["duplicate"] is True
            cameras = reader.get(origin + "/api/v1/cameras", timeout=5).json()["cameras"]
            assert [(camera["camera_id"], camera["status"]) for camera in cameras] == [
                ("camera_01", "OFFLINE"), ("camera_02", "OFFLINE"),
            ]
            assert all(camera["dropped_frames"] == 0 for camera in cameras)
            page = reader.get(origin + "/", timeout=5)
            assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        finally:
            transport.close()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 2


def test_real_cli_synthetic_demo_reaches_central_and_reports_clean_shutdown(tmp_path, capsys):
    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        result = main(["--synthetic", "--demo", "--central-url", origin,
                       "--max-cycles", "2", "--fps", "30"])
        output = capsys.readouterr().out
        assert result == 0 and "synthetic" in output and "failures=0" in output
        assert len(reader.get(origin + "/api/v1/events", timeout=5).json()["events"]) == 2
        cameras = reader.get(origin + "/api/v1/cameras", timeout=5).json()["cameras"]
        assert [camera["status"] for camera in cameras] == ["OFFLINE", "OFFLINE"]


def test_video_file_cli_accepts_paths_with_spaces(tmp_path, capsys):
    first, second = tmp_path / "camera one.avi", tmp_path / "camera two.avi"
    make_video(first, 1, 0)
    make_video(second, 1, 1)
    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        code = main(["--demo", "--camera", f"camera_01={first}", "--camera", f"camera_02={second}",
                     "--central-url", origin, "--fps", "30"])
        assert code == 0
        assert "failures=0" in capsys.readouterr().out
        cameras = reader.get(origin + "/api/v1/cameras", timeout=5).json()["cameras"]
        assert [camera["status"] for camera in cameras] == ["OFFLINE", "OFFLINE"]
        events = reader.get(origin + "/api/v1/events", timeout=5).json()["events"]
        assert {event["camera_id"] for event in events} == {"camera_01", "camera_02"}


def test_detection_cli_uses_local_baseline_and_keeps_observations_out_of_events(tmp_path, capsys, monkeypatch):
    weights = Path(__file__).resolve().parents[1] / "yolov8n.pt"
    if not weights.is_file():
        pytest.skip("Local baseline is unavailable; no download attempted")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics"))
    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    make_video(first, 1, 0)
    make_video(second, 1, 1)
    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        result = main(["--camera", f"camera_01={first}", "--camera", f"camera_02={second}",
                       "--weights", str(weights), "--device", "cpu", "--central-url", origin,
                       "--fps", "30"])
        assert result == 0 and "failures=0" in capsys.readouterr().out
        assert reader.get(origin + "/api/v1/events", timeout=5).json()["events"] == []
        cameras = reader.get(origin + "/api/v1/cameras", timeout=5).json()["cameras"]
        assert len(cameras) == 2
        assert all(camera["status"] == "OFFLINE" for camera in cameras)
        assert all(camera["sequence_number"] == 2 for camera in cameras)
        assert all(camera["model_versions"] == {"detector": "yolov8n.pt", "tracker": "bytetrack"}
                   for camera in cameras)
