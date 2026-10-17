"""Camera-scoped spatial rules use synthetic coordinates and explicit identity state."""

import numpy as np
import pytest

FRAME = np.zeros((100, 100, 3), dtype=np.uint8)
LEFT = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
CENTER = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]


def config(*, roi=None, zones=None, wires=None):
    return {"schema_version": 1, "cameras": {"a": {
        "inference_rois": roi or [], "restricted_zones": zones or [],
        "tripwires": wires or [],
    }}}


def person(x, foot_y, **extra):
    return {"bbox": [x - 5, foot_y - 30, x + 5, foot_y], "score": 0.9, **extra}


def test_no_geometry_keeps_all_tracks_eligible_and_never_emits_spatial_candidate():
    from edge.spatial import SpatialEngine

    engine = SpatialEngine(config())
    tracks = {1: person(20, 60), 2: person(90, 70)}
    assert engine.process("a", tracks, FRAME.shape) == []
    assert all(track["in_inference_roi"] for track in tracks.values())
    assert engine.process("b", {1: person(20, 60)}, FRAME.shape) == []


def test_roi_filters_downstream_but_does_not_remove_tracks_or_create_candidates():
    from edge.spatial import SpatialEngine

    engine = SpatialEngine(config(roi=[{"id": "left", "polygon": LEFT}]))
    tracks = {1: person(20, 50), 2: person(80, 50)}
    assert engine.process("a", tracks, FRAME.shape) == []
    assert set(tracks) == {1, 2}
    assert tracks[1]["in_inference_roi"] is True
    assert tracks[2]["in_inference_roi"] is False


def test_restricted_zone_requires_explicit_identity_result_and_stable_entry():
    from edge.spatial import SpatialEngine

    zone = {"id": "store", "polygon": CENTER, "min_inside_frames": 2}
    engine = SpatialEngine(config(zones=[zone]))
    for _ in range(3):
        assert engine.process("a", {1: person(50, 50)}, FRAME.shape) == []
    assert engine.process("a", {1: person(50, 50, identity_evaluated=True,
                                           identity_status="unknown", identity_sample_id=1)}, FRAME.shape) == []
    events = engine.process("a", {1: person(50, 50, identity_evaluated=True,
                                           identity_status="unknown", identity_sample_id=2)}, FRAME.shape)
    assert len(events) == 1
    assert events[0].candidate_type.value == "unknown_person_in_restricted_zone"
    assert events[0].attributes["restricted_zone_id"] == "store"
    assert engine.process("a", {1: person(50, 50, identity_evaluated=True,
                                          identity_status="unknown", identity_sample_id=2)}, FRAME.shape) == []


def test_registered_denied_person_is_unauthorized_not_unknown():
    from edge.spatial import SpatialEngine

    zone = {"id": "store", "polygon": CENTER, "min_inside_frames": 2}
    engine = SpatialEngine(config(zones=[zone]))
    values = {1: person(50, 50, identity_evaluated=True, identity_status="registered",
                        zone_access={"store": "unauthorized"}, person_id="employee_01",
                        identity_sample_id=1)}
    assert engine.process("a", values, FRAME.shape) == []
    values[1]["identity_sample_id"] = 2
    events = engine.process("a", values, FRAME.shape)
    assert [event.candidate_type.value for event in events] == ["unauthorized_zone_entry"]


def test_zone_rejects_repeated_sample_and_unavailable_face():
    from edge.spatial import SpatialEngine

    engine = SpatialEngine(config(zones=[{"id": "store", "polygon": CENTER}]))
    result = person(50, 50, identity_evaluated=True, identity_status="unknown",
                    identity_sample_id=1)
    for _ in range(5):
        assert engine.process("a", {1: result}, FRAME.shape) == []
    unavailable = person(50, 50, identity_evaluated=False,
                         identity_status="unavailable")
    assert engine.process("a", {1: unavailable}, FRAME.shape) == []
    result["identity_sample_id"] = 2
    assert engine.process("a", {1: result}, FRAME.shape) == []


def test_registered_access_is_checked_independently_for_each_zone():
    from edge.spatial import SpatialEngine

    engine = SpatialEngine(config(zones=[
        {"id": "office", "polygon": CENTER}, {"id": "store", "polygon": CENTER},
    ]))
    values = {1: person(50, 50, identity_evaluated=True, identity_status="registered",
                        person_id="employee_01", identity_sample_id=1,
                        zone_access={"office": "authorized", "store": "unauthorized"})}
    assert engine.process("a", values, FRAME.shape) == []
    values[1]["identity_sample_id"] = 2
    events = engine.process("a", values, FRAME.shape)
    assert [(event.candidate_type.value, event.attributes["restricted_zone_id"])
            for event in events] == [("unauthorized_zone_entry", "store")]
    assert events[0].attributes["person_id"] == "employee_01"
    assert events[0].attributes["severity"] == "HIGH"
    assert events[0].attributes["experimental"] is True


def test_two_prior_matching_samples_remain_valid_when_track_enters_zone_later():
    from edge.spatial import SpatialEngine

    engine = SpatialEngine(config(zones=[{"id": "store", "polygon": CENTER}]))
    outside = person(90, 50, identity_evaluated=True, identity_status="unknown",
                     identity_sample_id=2, identity_confirmed=True)
    for _ in range(3):
        assert engine.process("a", {1: outside}, FRAME.shape) == []
    inside = person(50, 50, identity_evaluated=True, identity_status="unknown",
                    identity_sample_id=2, identity_confirmed=True)
    assert engine.process("a", {1: inside}, FRAME.shape) == []
    events = engine.process("a", {1: inside}, FRAME.shape)
    assert [event.candidate_type.value for event in events] == [
        "unknown_person_in_restricted_zone"]


def test_detection_only_exposes_roi_eligible_tracks_to_face_pipeline():
    from edge.detection import DetectionProcessor
    from edge.spatial import SpatialEngine

    class Detector:
        def detect(self, frame):
            return None

    class Tracker:
        def update(self, boxes, frame):
            return np.array([[15, 20, 25, 50, 1, 0.9, 0, 0],
                             [75, 20, 85, 50, 2, 0.9, 0, 0]])

    class Face:
        def __init__(self):
            self.eligible = []

        def begin_frame(self, camera_id):
            pass

        def prune(self, camera_id, active):
            pass

        def observe(self, camera_id, track_id, frame, bbox, eligible, zone_ids):
            self.eligible.append((track_id, eligible))
            return {"identity_evaluated": False, "identity_status": "unavailable"}

    face = Face()
    processor = DetectionProcessor(Detector(), ["a"], tracker_factory=lambda _: Tracker(),
                                   spatial=SpatialEngine(config(roi=[{"id": "left", "polygon": LEFT}])),
                                   face=face)
    state = {}
    assert list(processor.process("a", FRAME, state)) == []
    assert face.eligible == [(1, True), (2, False)]


def test_tripwire_requires_stable_finite_segment_crossing_in_configured_direction():
    from edge.spatial import SpatialEngine

    wire = {"id": "gate", "start": [0.3, 0.5], "end": [0.7, 0.5],
            "direction": "negative_to_positive", "min_displacement": 0.05}
    engine = SpatialEngine(config(wires=[wire]))
    emitted = []
    for foot in (20, 35, 75, 85):
        emitted.extend(engine.process("a", {1: person(50, foot)}, FRAME.shape))
    assert [(item.candidate_type.value, item.attributes["tripwire_id"])
            for item in emitted] == [("tripwire_crossing", "gate")]
    assert engine.process("a", {2: person(90, 20)}, FRAME.shape) == []
    assert engine.process("a", {2: person(90, 35)}, FRAME.shape) == []
    assert engine.process("a", {2: person(90, 75)}, FRAME.shape) == []
    for foot in (80, 65, 20):
        assert engine.process("a", {3: person(50, foot)}, FRAME.shape) == []


@pytest.mark.parametrize("bad", [
    {"schema_version": 1, "cameras": {"a": {"inference_rois": [
        {"id": "bad", "polygon": [[0, 0], [1.1, 0], [0, 1]]}]}}},
    config(roi=[{"id": "line", "polygon": [[0, 0], [0.5, 0], [1, 0]]}]),
    config(wires=[{"id": "same", "start": [0, 0], "end": [0, 0],
                   "direction": "both"}]),
    config(zones=[{"id": "x", "polygon": CENTER}, {"id": "x", "polygon": CENTER}]),
    config(roi=[{"id": "x", "polygon": CENTER, "enabled": 0}]),
])
def test_malformed_geometry_fails_before_processing(bad):
    from edge.spatial import SpatialEngine

    with pytest.raises(ValueError):
        SpatialEngine(bad)


def test_detection_keeps_both_tracks_but_action_only_observes_roi_member():
    from edge.action_pipeline import ActionProcessor
    from edge.action_runtime import ActionEngine, ActionSettings
    from edge.detection import DetectionProcessor
    from edge.spatial import SpatialEngine

    class Detector:
        def detect(self, frame):
            return None

    class Tracker:
        def update(self, boxes, frame):
            return np.array([
                [15, 20, 25, 50, 1, 0.9, 0, 0],
                [75, 20, 85, 50, 2, 0.9, 0, 0],
            ])

    class Model:
        def predict_batch(self, windows):
            assert len(windows) == 1
            return [[1.0, 0.0, 0.0]]

    detection = DetectionProcessor(Detector(), ["a"], tracker_factory=lambda _: Tracker(),
                                   spatial=SpatialEngine(config(roi=[{"id": "left", "polygon": LEFT}])))
    engine = ActionEngine(Model(), ("a",), ActionSettings(window_frames=2))
    processor = ActionProcessor(detection, engine)
    state = {}
    for time in (0.0, 0.1):
        assert processor.process_timed("a", FRAME, state, time) == ()
        processor.finish_cycle()
    assert set(state) == {1, 2}
    assert engine.active_tracks == 1
    assert engine.last_decision("a", 1) == "Normal"
    with pytest.raises(KeyError):
        engine.last_decision("a", 2)


def test_detection_without_action_emits_tripwire_candidate_from_stable_track():
    from edge.detection import DetectionProcessor
    from edge.spatial import SpatialEngine

    wire = {"id": "gate", "start": [0.3, 0.5], "end": [0.7, 0.5],
            "direction": "negative_to_positive"}

    class Detector:
        def detect(self, frame):
            return None

    class Tracker:
        def __init__(self):
            self.feet = iter((20, 35, 75))

        def update(self, boxes, frame):
            foot = next(self.feet)
            return np.array([[45, foot - 30, 55, foot, 1, 0.9, 0, 0]])

    processor = DetectionProcessor(Detector(), ["a"], tracker_factory=lambda _: Tracker(),
                                   spatial=SpatialEngine(config(wires=[wire])))
    state = {}
    assert list(processor.process("a", FRAME, state)) == []
    assert list(processor.process("a", FRAME, state)) == []
    events = list(processor.process("a", FRAME, state))
    assert [(event.candidate_type.value, event.local_track_id) for event in events] == [
        ("tripwire_crossing", 1)
    ]


def test_spatial_cli_validates_local_config_before_loading_detector(tmp_path, monkeypatch):
    from edge.run_gateway import main

    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    first.touch()
    second.touch()

    def forbidden(*args, **kwargs):
        raise AssertionError("Detector must not load before spatial configuration")

    monkeypatch.setattr("edge.detection.SharedYOLODetector", forbidden)
    result = main([
        "--camera", f"a={first}", "--camera", f"b={second}",
        "--spatial-config", str(tmp_path / "missing.yaml"),
    ])
    assert result == 2


def test_existing_empty_spatial_config_keeps_full_frame_eligible():
    from pathlib import Path

    from edge.spatial import load_spatial_config

    root = Path(__file__).resolve().parents[1]
    engine = load_spatial_config(root / "configs" / "spatial_rules.yaml")
    tracks = {1: person(95, 80)}
    assert engine.process("camera_01", tracks, FRAME.shape) == []
    assert tracks[1]["in_inference_roi"] is True


def test_drawer_saves_independent_normalized_rules_without_erasing_existing(tmp_path):
    import yaml

    from tools.spatial_drawer import build_geometry, normalize_points, save_geometry

    assert normalize_points([(0, 0), (50, 25)], 100, 50) == [[0.0, 0.0], [0.5, 0.5]]
    path = tmp_path / "spatial.yaml"
    roi = build_geometry("roi", "front", [(0, 0), (50, 0), (50, 50), (0, 50)], 100, 100)
    zone = build_geometry("zone", "vault", [(20, 20), (80, 20), (80, 80), (20, 80)],
                          100, 100)
    wire = build_geometry("tripwire", "gate", [(0, 50), (100, 50)], 100, 100,
                          direction="negative_to_positive")
    for kind, entry in (("roi", roi), ("zone", zone), ("tripwire", wire)):
        save_geometry(path, "a", kind, entry)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["cameras"]["a"]["inference_rois"] == [roi]
    assert document["cameras"]["a"]["restricted_zones"] == [zone]
    assert document["cameras"]["a"]["tripwires"] == [wire]


def test_drawer_rejects_duplicate_name_without_changing_existing_config(tmp_path):
    from tools.spatial_drawer import build_geometry, save_geometry

    path = tmp_path / "spatial.yaml"
    entry = build_geometry("roi", "front", [(0, 0), (50, 0), (0, 50)], 100, 100)
    save_geometry(path, "a", "roi", entry)
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        save_geometry(path, "a", "roi", entry)
    assert path.read_bytes() == previous


def test_drawer_clicks_local_frame_and_persists_polygon_without_camera_hardware(tmp_path, monkeypatch):
    import cv2
    import yaml

    from tools import spatial_drawer

    class Source:
        def read(self):
            return True, FRAME

        def close(self):
            pass

    holder = {}
    monkeypatch.setattr(spatial_drawer, "VideoFileSource", lambda _: Source())
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda window, callback: holder.update(callback=callback))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: None)

    def click_and_save(_):
        for x, y in ((10, 10), (50, 10), (50, 50), (10, 50)):
            holder["callback"](cv2.EVENT_LBUTTONDOWN, x, y, None, None)
        return 13

    monkeypatch.setattr(cv2, "waitKey", click_and_save)
    path = tmp_path / "drawn.yaml"
    assert spatial_drawer.main([
        "--camera", "a=local.avi", "--kind", "roi", "--name", "front",
        "--config", str(path),
    ]) == 0
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"]["a"]["inference_rois"][0]
    assert entry["polygon"] == [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
