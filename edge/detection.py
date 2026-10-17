"""One YOLO detector and independent ByteTrack instances; no model loaded on import."""

import importlib.util
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from common.schemas import CandidateDraft

ROOT = Path(__file__).resolve().parents[1]


class SharedYOLODetector:
    def __init__(
        self, weights: str | Path, *, device: str | int = "cpu", half: bool = False,
        confidence: float = 0.1, image_size: int = 640, model_factory: Callable | None = None,
    ):
        weight_path = Path(weights).resolve()
        if not weight_path.is_file() or weight_path.suffix.lower() != ".pt":
            raise FileNotFoundError("A local detector .pt checkpoint is required; no download attempted")
        if not 0 < confidence <= 1 or image_size < 32:
            raise ValueError("Invalid detector settings")
        if model_factory is None:
            # Check the checkpoint before importing Ultralytics or constructing YOLO.
            os.environ["YOLO_AUTOINSTALL"] = "false"
            os.environ["YOLO_OFFLINE"] = "true"
            from ultralytics import YOLO

            model_factory = YOLO
        self.model = model_factory(str(weight_path))
        self.device = device
        self.half = half and str(device).lower() != "cpu"
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame: Any) -> Any:
        result = self.model.predict(
            frame, classes=[0], conf=self.confidence, imgsz=self.image_size,
            device=self.device, half=self.half, verbose=False,
        )[0]
        return result.boxes.cpu().numpy()


def create_bytetracker(frame_rate: float, config_path: Path | None = None) -> Any:
    """Use the checked-in tracker settings and fail before any auto-install path."""
    if importlib.util.find_spec("lap") is None:
        raise RuntimeError("ByteTrack dependency lap is unavailable; no install attempted")
    import yaml

    with (config_path or ROOT / "configs" / "bytetrack_custom.yaml").open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    required = {
        "track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer",
        "match_thresh", "fuse_score",
    }
    if not isinstance(values, dict) or not required <= values.keys():
        raise ValueError("ByteTrack configuration is incomplete")
    if values.get("tracker_type") != "bytetrack":
        raise ValueError("Shared gateway requires a ByteTrack configuration")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics.trackers.byte_tracker import BYTETracker

    return BYTETracker(SimpleNamespace(**values), frame_rate=frame_rate)


class DetectionProcessor:
    """Detection is an observation, so this stage does not manufacture crime candidates."""

    def __init__(
        self, detector: Any, camera_ids: Iterable[str], *, frame_rate: float = 10,
        tracker_factory: Callable | None = None, spatial: Any | None = None,
        face: Any | None = None,
    ):
        ids = tuple(camera_ids)
        if not ids or len(set(ids)) != len(ids) or any(not name.strip() for name in ids):
            raise ValueError("Camera IDs must be nonblank and unique")
        self.detector = detector
        self.spatial = spatial
        self.face = face
        factory = tracker_factory or (lambda _: create_bytetracker(frame_rate))
        # BYTETracker constructors reset a global ID counter. Create every camera's
        # tracker before the first detection; never lazily add one during playback.
        self.trackers = {camera_id: factory(camera_id) for camera_id in ids}

    def process(self, camera_id: str, frame: Any, state: dict) -> Iterable[CandidateDraft]:
        boxes = self.detector.detect(frame)
        rows = self.trackers[camera_id].update(boxes, frame)
        observations = {
            int(row[4]): {"bbox": [float(value) for value in row[:4]], "score": float(row[5])}
            for row in rows if int(row[6]) == 0
        }
        state.clear()
        state.update(observations)
        if self.face is not None:
            if self.spatial is not None:
                self.spatial.annotate_roi(camera_id, state, frame.shape)
                zone_ids = self.spatial.zone_ids(camera_id)
            else:
                zone_ids = []
            self.face.begin_frame(camera_id)
            self.face.prune(camera_id, set(state))
            for track_id, observation in state.items():
                observation.update(self.face.observe(
                    camera_id, track_id, frame, observation["bbox"],
                    observation.get("in_inference_roi", True), zone_ids))
        if self.spatial is not None:
            return self.spatial.process(camera_id, state, frame.shape)
        return ()

    def clear_camera(self, camera_id: str) -> None:
        if self.face is not None:
            self.face.clear_camera(camera_id)
        if self.spatial is not None:
            self.spatial.clear_camera(camera_id)
