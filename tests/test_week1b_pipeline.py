"""File playback, shared inference ownership, and runner lifecycle without model weights."""

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest


def test_file_source_reads_fresh_frames_then_finishes(tmp_path):
    from cameras.video_source import VideoFileSource

    path = tmp_path / "clip.avi"
    path.touch()
    capture = FakeCapture([np.zeros((8, 8, 3)), np.ones((8, 8, 3))])
    source = VideoFileSource(path, capture_factory=lambda _: capture)
    assert not source.finished
    assert source.read()[1].sum() == 0
    assert source.read()[1].sum() == 192
    assert source.read() == (False, None)
    assert source.finished
    assert source.read() == (False, None)
    assert capture.reads == 3
    source.close()
    source.close()
    assert capture.releases == 1


def test_file_source_rejects_nonfiles_before_opening(tmp_path):
    from cameras.video_source import VideoFileSource

    calls = []
    for value in ["rtsp://private:secret@camera/live", "0", tmp_path]:
        with pytest.raises(ValueError) as error:
            VideoFileSource(value, capture_factory=lambda path: calls.append(path))
        assert "secret" not in str(error.value)
    assert calls == []


def test_unopened_capture_is_released_and_error_sanitized(tmp_path):
    from cameras.video_source import VideoFileSource

    path = tmp_path / "clip.avi"
    path.touch()
    capture = FakeCapture([], opened=False)
    with pytest.raises(RuntimeError, match="Cannot open video file"):
        VideoFileSource(path, capture_factory=lambda _: capture)
    assert capture.releases == 1


def test_decoder_exception_finishes_source_and_preserves_safe_error(tmp_path):
    from cameras.video_source import VideoFileSource

    path = tmp_path / "clip.avi"
    path.touch()

    class BrokenCapture(FakeCapture):
        def read(self):
            raise RuntimeError("private-file-location")

    source = VideoFileSource(path, capture_factory=lambda _: BrokenCapture([]))
    with pytest.raises(RuntimeError) as error:
        source.read()
    assert "private-file-location" not in str(error.value)
    assert source.finished
    source.close()


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = iter(frames)
        self.opened = opened
        self.reads = self.releases = 0

    def isOpened(self):  # noqa: N802 - mirrors OpenCV's capture API
        return self.opened

    def read(self):
        self.reads += 1
        frame = next(self.frames, None)
        return (frame is not None, frame)

    def get(self, _):
        return 25.0

    def release(self):
        self.releases += 1


def test_local_weights_guard_runs_before_model_factory(tmp_path):
    from edge.detection import SharedYOLODetector

    calls = []
    with pytest.raises(FileNotFoundError):
        SharedYOLODetector(tmp_path / "yolo26n.pt", model_factory=lambda p: calls.append(p))
    assert calls == []


def test_detector_loads_once_and_predicts_person_only(tmp_path):
    from edge.detection import SharedYOLODetector

    weight = tmp_path / "yolo26n.pt"
    weight.touch()
    calls = []
    boxes = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: "cpu-boxes"))

    class Model:
        def predict(self, frame, **kwargs):
            calls.append((frame, kwargs))
            return [SimpleNamespace(boxes=boxes)]

    loads = []

    def factory(path):
        loads.append(path)
        return Model()

    detector = SharedYOLODetector(weight, device="cpu", half=True, model_factory=factory)
    assert detector.detect("a") == "cpu-boxes"
    assert detector.detect("b") == "cpu-boxes"
    assert len(loads) == 1
    assert [frame for frame, _ in calls] == ["a", "b"]
    assert all(options["classes"] == [0] and options["half"] is False for _, options in calls)
    assert all(options["conf"] == 0.1 and options["imgsz"] == 640 for _, options in calls)


def test_shared_processor_keeps_trackers_and_track_state_camera_scoped():
    from edge.detection import DetectionProcessor

    detections = []
    created = []

    class Detector:
        def detect(self, frame):
            detections.append(frame)
            return frame

    class Tracker:
        def __init__(self):
            self.updates = 0

        def update(self, boxes, frame):
            self.updates += 1
            return np.array([[1, 2, 10, 20, 7, 0.9, 0, 0]]) if frame else np.empty((0, 8))

    def factory(_):
        tracker = Tracker()
        created.append(tracker)
        return tracker

    processor = DetectionProcessor(Detector(), ["a", "b"], tracker_factory=factory)
    a, b = {}, {}
    assert len(created) == 2  # all trackers constructed before the first inference
    assert list(processor.process("a", True, a)) == []
    assert list(processor.process("b", True, b)) == []
    assert a is not b and list(a) == [7] and list(b) == [7]
    assert a[7]["bbox"] == [1.0, 2.0, 10.0, 20.0]
    processor.process("a", False, a)
    assert a == {} and list(b) == [7]
    assert [tracker.updates for tracker in created] == [2, 1]
    assert detections == [True, True, False]


def test_demo_candidates_are_explicit_and_not_generated_every_frame():
    from edge.run_gateway import DemoProcessor

    processor = DemoProcessor()
    state = {}
    drafts = list(processor.process("a", np.zeros((8, 8, 3)), state))
    assert len(drafts) == 1
    assert drafts[0].attributes == {"synthetic": True, "purpose": "pipeline_smoke_test"}
    assert drafts[0].reason_codes == ["synthetic_test_only"]
    assert list(processor.process("a", np.zeros((8, 8, 3)), state)) == []
    assert state == {}  # demo state must not inflate active_tracks telemetry


def test_runner_stops_at_eof_sends_offline_and_closes_sources():
    from edge.run_gateway import run_gateway
    from edge.shared_gateway import CameraRuntime, SharedGateway

    sources = [FiniteSource(2), FiniteSource(1)]
    sink = RecordingSink()
    gateway = SharedGateway(
        [CameraRuntime("a", sources[0]), CameraRuntime("b", sources[1])],
        EmptyProcessor(), sink, gateway_id="g", gateway_boot_id="boot",
        configuration_version="v1", model_versions={},
        clock=lambda: datetime(2026, 9, 28, tzinfo=timezone.utc),
        event_id_factory=lambda: UUID(int=1),
    )
    summary = run_gateway(gateway, fps=10, sleep=lambda _: None)
    assert summary.cycles == 3
    assert [source.reads for source in sources] == [3, 2]
    assert [source.closes for source in sources] == [1, 1]
    assert [(hb.camera_id, hb.status.value) for hb in sink.heartbeats] == [
        ("a", "ONLINE"), ("b", "ONLINE"), ("a", "ONLINE"), ("b", "OFFLINE"),
        ("a", "OFFLINE"), ("b", "OFFLINE"),
    ]
    assert all(hb.dropped_frames == 0 for hb in sink.heartbeats)


def test_runner_limit_clears_active_tracks_and_reports_offline():
    from edge.run_gateway import run_gateway
    from edge.shared_gateway import CameraRuntime, SharedGateway

    source = FiniteSource(5)
    sink = RecordingSink()

    class Processor:
        def process(self, camera_id, frame, state):
            state[7] = {"bbox": [1, 2, 3, 4]}
            return []

    gateway = SharedGateway(
        [CameraRuntime("a", source)], Processor(), sink, gateway_id="g", gateway_boot_id="b",
        configuration_version="v1", model_versions={},
        clock=lambda: datetime(2026, 9, 28, tzinfo=timezone.utc), event_id_factory=lambda: UUID(int=1),
    )
    result = run_gateway(gateway, fps=10, max_cycles=1)
    assert result.cycles == 1
    assert [(hb.status.value, hb.active_tracks) for hb in sink.heartbeats] == [("ONLINE", 1), ("OFFLINE", 0)]
    assert result.heartbeats_sent == 2


def test_runner_closes_both_sources_on_unexpected_exception():
    from edge.run_gateway import run_gateway
    from edge.shared_gateway import CameraRuntime

    sources = [FiniteSource(1), FiniteSource(1)]

    class BrokenGateway:
        cameras = [CameraRuntime("a", sources[0]), CameraRuntime("b", sources[1])]

        def run_once(self):
            raise RuntimeError("broken")

    with pytest.raises(RuntimeError):
        run_gateway(BrokenGateway(), fps=10, sleep=lambda _: None)
    assert [source.closes for source in sources] == [1, 1]


class FiniteSource:
    def __init__(self, count):
        self.count = count
        self.finished = False
        self.reads = self.closes = 0

    def read(self):
        if self.finished:
            return False, None
        self.reads += 1
        if self.count == 0:
            self.finished = True
            return False, None
        self.count -= 1
        return True, object()

    def close(self):
        self.closes += 1


class EmptyProcessor:
    def process(self, camera_id, frame, state):
        return []


class RecordingSink:
    def __init__(self):
        self.heartbeats = []

    def send_candidate(self, event):
        pass

    def send_heartbeat(self, heartbeat):
        self.heartbeats.append(heartbeat)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_runner_rejects_invalid_fps_before_reads(value):
    from edge.run_gateway import run_gateway

    with pytest.raises(ValueError):
        run_gateway(SimpleNamespace(cameras=[]), fps=value)


def test_cli_requires_explicit_demo_for_synthetic_sources(capsys):
    from edge.run_gateway import main

    assert main(["--synthetic"]) == 2
    assert "--demo" in capsys.readouterr().err


def test_cli_default_model_is_yolo26n():
    from edge.run_gateway import build_parser

    assert build_parser().parse_args(["--synthetic", "--demo"]).weights.name == "yolo26n.pt"


def test_transport_uses_session_without_proxy_redirect_or_auth(monkeypatch):
    import requests

    from clients.central_client import RequestsTransport

    calls = []

    class Session:
        trust_env = True

        def post(self, url, **kwargs):
            calls.append((url, kwargs, self.trust_env))
            return SimpleNamespace(status_code=200)

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(requests, "Session", Session)
    transport = RequestsTransport("http://127.0.0.1:8000")
    assert calls == []
    transport.post("/api/v1/events", {"synthetic": True})
    assert calls == [("http://127.0.0.1:8000/api/v1/events", {
        "json": {"synthetic": True}, "timeout": 5.0, "allow_redirects": False,
    }, False)]
    transport.close()
    assert calls[-1] == "closed"


def test_cli_resource_configuration_is_applied(tmp_path):
    from edge.run_gateway import load_resources

    path = tmp_path / "resources.yaml"
    path.write_text("resource_limits:\n  max_fps_per_camera: 12\n  yolo_device: cpu\n  yolo_half: true\n")
    settings = load_resources(path)
    assert settings.max_fps == 12
    assert settings.device == "cpu"
    assert settings.half is False


def test_real_bytetrack_adapter_isolates_camera_lifetimes(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics"))
    from ultralytics.engine.results import Boxes

    from edge.detection import DetectionProcessor

    class Detector:
        def detect(self, frame):
            return Boxes(np.array([[8, 8, 25, 35, 0.9, 0]], dtype=np.float32),
                         orig_shape=(48, 64)).numpy()

    processor = DetectionProcessor(Detector(), ["a", "b"], frame_rate=10)
    a, b = {}, {}
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    processor.process("a", frame, a)
    processor.process("b", frame, b)
    original_a, original_b = set(a), set(b)
    assert len(a) == len(b) == 1
    processor.process("a", frame, a)
    assert set(a) == original_a and set(b) == original_b
    assert processor.trackers["a"] is not processor.trackers["b"]
    assert processor.trackers["a"].frame_id == 2
    assert processor.trackers["b"].frame_id == 1


def test_local_yolov8_baseline_predicts_synthetic_frame_offline(tmp_path, monkeypatch):
    from pathlib import Path

    from edge.detection import SharedYOLODetector

    root = Path(__file__).resolve().parents[1]
    weights = root / "yolov8n.pt"
    if not weights.is_file():
        pytest.skip("Local baseline checkpoint is unavailable; downloading is disabled")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics"))
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    detector = SharedYOLODetector(weights, device="cpu", half=False)
    boxes = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
    assert boxes.data.shape[1] == 6
    assert np.isfinite(boxes.data).all()
    assert all(int(class_id) == 0 for class_id in boxes.cls)
