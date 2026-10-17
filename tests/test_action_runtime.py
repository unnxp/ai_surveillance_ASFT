"""Action scheduling uses synthetic frames and deterministic model outputs."""

from collections import deque
from datetime import datetime, timezone
from uuid import UUID

import numpy as np
import pytest

from common.schemas import CameraStatus

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)
BOX = [8, 4, 40, 44]


class FakeActionModel:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.windows = []

    def predict_batch(self, windows):
        self.windows.extend(windows)
        return [self.outputs.popleft() for _ in windows]


def settings(**changes):
    from edge.action_runtime import ActionSettings

    return ActionSettings(**changes)


def test_requires_32_distinct_sample_times_and_separates_same_track_id_across_cameras():
    from edge.action_runtime import ActionEngine

    model = FakeActionModel([[0.01, 0.01, 0.98], [0.01, 0.01, 0.98]])
    engine = ActionEngine(model, ("a", "b"), settings(sample_hz=10, window_frames=32))
    for index in range(31):
        for camera in ("a", "b"):
            engine.observe(camera, 7, FRAME, BOX, index / 10)
        assert engine.finish_cycle() == []
    assert model.windows == []
    for camera in ("a", "b"):
        engine.observe(camera, 7, FRAME, BOX, 3.1)
        engine.observe(camera, 7, FRAME, BOX, 3.1)  # same source frame is not evidence twice
    assert engine.finish_cycle() == []  # first positive window does not make an event
    assert len(model.windows) == 2
    assert all(len(window) == 32 for window in model.windows)
    assert engine.active_tracks == 2


def test_temporal_two_of_three_cooldown_and_low_confidence_abstention():
    from edge.action_runtime import ActionEngine

    model = FakeActionModel([
        [0.05, 0.20, 0.75], [0.10, 0.25, 0.65], [0.05, 0.15, 0.80],
        [0.05, 0.10, 0.85], [0.05, 0.10, 0.85],
    ])
    engine = ActionEngine(model, ("a",), settings(
        sample_hz=10, window_frames=32, inference_interval_seconds=0.1,
        cooldown_seconds=10,
    ))
    emitted = []
    for index in range(36):
        engine.observe("a", 7, FRAME, BOX, index / 10)
        emitted.extend(engine.finish_cycle())
    assert len(model.windows) == 5
    assert [(camera, draft.candidate_type.value) for camera, draft in emitted] == [
        ("a", "possible_person_violence")
    ]
    assert emitted[0][1].local_track_id == 7
    assert emitted[0][1].reason_codes == ["action_temporal_2_of_3", "threshold_baseline_unvalidated"]
    assert emitted[0][1].attributes["class_probabilities"]["Personal_Violence"] == 0.8


def test_capacity_is_bounded_and_each_camera_gets_a_slot():
    from edge.action_runtime import ActionEngine

    model = FakeActionModel([[1.0, 0.0, 0.0]] * 2)
    engine = ActionEngine(model, ("a", "b"), settings(max_active_tracks=2))
    for track in (1, 2):
        engine.observe("a", track, FRAME, BOX, 0)
    assert engine.active_tracks == 2
    engine.observe("b", 1, FRAME, BOX, 0)
    assert engine.active_tracks == 2
    assert engine.rejected_tracks("a") == 1
    assert engine.rejected_tracks("b") == 0
    assert engine.last_decision("b", 1) == "Insufficient"
    with pytest.raises(KeyError):
        engine.last_decision("a", 1)  # oldest from the dominant camera yields


def test_single_camera_can_borrow_all_eight_action_track_slots():
    from edge.action_runtime import ActionEngine

    engine = ActionEngine(FakeActionModel([]), ("a", "b"), settings(max_active_tracks=8))
    for track_id in range(1, 9):
        engine.observe("a", track_id, FRAME, BOX, 0)
    assert engine.active_tracks == 8
    assert engine.rejected_tracks("a") == 0
    engine.observe("b", 1, FRAME, BOX, 0)
    assert engine.active_tracks == 8
    assert engine.rejected_tracks("a") == 1
    assert engine.last_decision("b", 1) == "Insufficient"


def test_fair_scheduler_limits_inference_and_does_not_reinfer_same_window():
    from edge.action_runtime import ActionEngine

    model = FakeActionModel([[1.0, 0.0, 0.0]] * 3)
    engine = ActionEngine(model, ("a", "b"), settings(
        window_frames=2, max_active_tracks=4, max_inferences_per_cycle=1,
        inference_interval_seconds=0.1,
    ))
    for time in (0.0, 0.1):
        for camera in ("a", "b"):
            engine.observe(camera, 1, FRAME, BOX, time)
    engine.finish_cycle()
    assert len(model.windows) == 1
    assert engine.backlog("a") + engine.backlog("b") == 1
    engine.finish_cycle()
    assert len(model.windows) == 2
    engine.finish_cycle()
    assert len(model.windows) == 2


def test_gap_resets_window_instead_of_padding_stale_frames():
    from edge.action_runtime import ActionEngine

    model = FakeActionModel([])
    engine = ActionEngine(model, ("a",), settings(window_frames=3, max_sample_gap_seconds=0.3))
    for time in (0.0, 0.1, 1.0, 1.1):
        engine.observe("a", 1, FRAME, BOX, time)
        assert engine.finish_cycle() == []
    assert model.windows == []


def test_local_checkpoint_missing_fails_closed_before_model_creation(tmp_path):
    from edge.action_runtime import ActionModel

    with pytest.raises(FileNotFoundError, match="local action checkpoint"):
        ActionModel(tmp_path / "missing.pth", tmp_path / "missing.yaml")


def test_video_file_source_exposes_original_media_time(tmp_path):
    from cameras.video_source import VideoFileSource

    class Capture:
        def __init__(self):
            self.count = 0

        def isOpened(self):  # noqa: N802
            return True

        def get(self, _):
            return 25.0

        def read(self):
            self.count += 1
            return True, FRAME

        def release(self):
            pass

    path = tmp_path / "two.avi"
    path.touch()
    source = VideoFileSource(path, capture_factory=lambda _: Capture())
    assert source.timestamp_seconds is None
    source.read()
    assert source.timestamp_seconds == 0.0
    source.read()
    assert source.timestamp_seconds == pytest.approx(0.04)


def test_action_processor_uses_detection_track_state_and_returns_temporal_candidates():
    from edge.action_pipeline import ActionProcessor
    from edge.action_runtime import ActionEngine

    class Detection:
        def process(self, camera_id, frame, state):
            state.clear()
            state[7] = {"bbox": BOX, "score": 0.95}
            return ()

    engine = ActionEngine(FakeActionModel([[0.01, 0.01, 0.98]] * 4),
                          ("a", "b"), settings(window_frames=2,
                                               inference_interval_seconds=0.1))
    processor = ActionProcessor(Detection(), engine)
    first, second = {}, {}
    results = []
    for time in (0.0, 0.1, 0.2):
        processor.process_timed("a", FRAME, first, time)
        processor.process_timed("b", FRAME, second, time)
        results.extend(processor.finish_cycle())
    assert set((name, event.local_track_id) for name, event in results) == {("a", 7), ("b", 7)}
    assert list(first) == list(second) == [7]


def test_action_cli_refuses_missing_checkpoint_before_opening_videos(tmp_path):
    from edge.run_gateway import main

    paths = [tmp_path / "a.avi", tmp_path / "b.avi"]
    for path in paths:
        path.touch()
    assert main([
        "--camera", f"a={paths[0]}", "--camera", f"b={paths[1]}",
        "--action", "--action-weights", str(tmp_path / "missing.pth"),
    ]) == 2


def test_existing_action_checkpoint_runs_offline_on_synthetic_window():
    from pathlib import Path

    from edge.action_runtime import ActionModel

    root = Path(__file__).resolve().parents[1]
    model = ActionModel(root / "models" / "action_classifier_3cls.pth",
                        root / "models" / "action_classifier_3cls_meta.yaml")
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    probabilities = model.predict_batch([tuple(rgb.copy() for _ in range(32))])
    assert len(probabilities) == 1
    assert len(probabilities[0]) == 3
    assert all(np.isfinite(probabilities[0]))
    assert sum(probabilities[0]) == pytest.approx(1.0)


def test_numeric_action_device_uses_cuda_when_present_and_cpu_fallback(monkeypatch):
    import torch

    from edge.action_runtime import resolve_action_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(resolve_action_device(0)) == "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_action_device(0)) == "cpu"
    assert str(resolve_action_device("cpu")) == "cpu"


def test_action_preprocessing_matches_training_validation_contract():
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms

    from edge.action_runtime import _crop_rgb

    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    crop_rgb = _crop_rgb(frame, [8, 4, 40, 44], 0)
    assert crop_rgb.shape == (224, 224, 3)
    deployed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(Image.fromarray(crop_rgb))
    validation = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(cv2.cvtColor(frame[4:44, 8:40], cv2.COLOR_BGR2RGB))
    assert torch.equal(deployed, validation)


def test_low_confidence_normal_is_insufficient_not_confirmed_normal():
    from edge.action_runtime import ActionEngine

    engine = ActionEngine(FakeActionModel([[0.5, 0.3, 0.2]]), ("a",),
                          settings(window_frames=2))
    for time in (0.0, 0.1):
        engine.observe("a", 1, FRAME, BOX, time)
    assert engine.finish_cycle() == []
    assert engine.last_decision("a", 1) == "Insufficient"


def test_bad_second_probability_row_does_not_commit_first_track():
    from edge.action_runtime import ActionEngine

    model = FakeActionModel([
        [0.01, 0.01, 0.98], [float("nan"), 0.0, 1.0],
        [0.01, 0.01, 0.98], [0.01, 0.01, 0.98],
    ])
    engine = ActionEngine(model, ("a", "b"), settings(window_frames=2))
    for time in (0.0, 0.1):
        for camera in ("a", "b"):
            engine.observe(camera, 1, FRAME, BOX, time)
    with pytest.raises(RuntimeError, match="invalid probabilities"):
        engine.finish_cycle()
    assert engine.backlog("a") == engine.backlog("b") == 1
    assert engine.finish_cycle() == []
    assert len(model.windows) == 4


def test_gateway_emits_action_after_both_cameras_observed():
    from edge.shared_gateway import CameraRuntime, SharedGateway

    class Source:
        def __init__(self):
            self.timestamp_seconds = 0.0

        def read(self):
            return True, FRAME

    class Processor:
        def __init__(self):
            self.seen = []

        def process_timed(self, camera_id, frame, state, timestamp_seconds):
            self.seen.append((camera_id, timestamp_seconds))
            state[1] = {"bbox": BOX}
            return []

        def finish_cycle(self):
            from common.schemas import CandidateDraft

            assert self.seen == [("a", 0.0), ("b", 0.0)]
            return [("b", CandidateDraft(candidate_type="possible_person_violence", local_track_id=1))]

        def backlog(self, camera_id):
            return 1 if camera_id == "a" else 0

    class Sink:
        def __init__(self):
            self.events, self.heartbeats = [], []

        def send_candidate(self, event):
            self.events.append(event)

        def send_heartbeat(self, heartbeat):
            self.heartbeats.append(heartbeat)

    sink = Sink()
    gateway = SharedGateway(
        [CameraRuntime("a", Source()), CameraRuntime("b", Source())], Processor(), sink,
        gateway_id="g", gateway_boot_id="b", configuration_version="v2",
        model_versions={"action": "local"},
        clock=lambda: datetime(2026, 9, 28, tzinfo=timezone.utc),
        event_id_factory=lambda: UUID(int=1),
    )
    result = gateway.run_once()
    assert result.events_sent == 1
    assert [(e.camera_id, e.candidate_type.value) for e in sink.events] == [
        ("b", "possible_person_violence")
    ]
    assert [(h.camera_id, h.status, h.action_backlog) for h in sink.heartbeats] == [
        ("a", CameraStatus.DEGRADED, 1), ("b", CameraStatus.ONLINE, 0)
    ]


def test_gateway_records_each_capture_timestamp_once_before_action_inference():
    from edge.shared_gateway import CameraRuntime, SharedGateway

    class Source:
        def read(self):
            return True, FRAME

    class Processor:
        def process(self, camera_id, frame, state):
            return ()

    class Sink:
        def __init__(self):
            self.heartbeats = []

        def send_heartbeat(self, heartbeat):
            self.heartbeats.append(heartbeat)

    moments = iter([
        datetime(2026, 9, 28, 0, 0, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 28, 0, 0, 2, tzinfo=timezone.utc),
    ])
    sink = Sink()
    gateway = SharedGateway(
        [CameraRuntime("a", Source()), CameraRuntime("b", Source())], Processor(), sink,
        gateway_id="g", gateway_boot_id="b", configuration_version="v1",
        model_versions={}, clock=lambda: next(moments),
        event_id_factory=lambda: UUID(int=1),
    )
    assert gateway.run_once().failures == ()
    assert [heartbeat.observed_at_utc.second for heartbeat in sink.heartbeats] == [1, 2]


def test_action_cycle_never_emits_stale_candidate_from_failed_camera():
    from common.schemas import CandidateDraft
    from edge.shared_gateway import CameraRuntime, SharedGateway

    class Source:
        def read(self):
            return True, FRAME

    class Processor:
        def __init__(self):
            self.cleared = []

        def process_timed(self, camera_id, frame, state, timestamp_seconds):
            if camera_id == "a":
                raise RuntimeError("private-source")
            return ()

        def finish_cycle(self):
            return [("a", CandidateDraft(candidate_type="possible_person_violence"))]

        def clear_camera(self, camera_id):
            self.cleared.append(camera_id)

    class Sink:
        def __init__(self):
            self.events = []

        def send_candidate(self, event):
            self.events.append(event)

        def send_heartbeat(self, heartbeat):
            pass

    sink = Sink()
    processor = Processor()
    gateway = SharedGateway(
        [CameraRuntime("a", Source()), CameraRuntime("b", Source())], processor, sink,
        gateway_id="g", gateway_boot_id="b", configuration_version="v1",
        model_versions={}, clock=lambda: datetime(2026, 9, 28, tzinfo=timezone.utc),
        event_id_factory=lambda: UUID(int=1),
    )
    result = gateway.run_once()
    assert result.events_sent == 0
    assert sink.events == []
    assert processor.cleared == ["a"]
    assert [(failure.camera_id, failure.stage) for failure in result.failures] == [("a", "processor")]
