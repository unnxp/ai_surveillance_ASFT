"""CPU-only local face extraction and bounded, camera-isolated match sampling."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from ai.face_recognition import REFERENCE_FACIAL_POINTS, ArcFaceEmbedder, SCRFDDetector
from common.identity_schemas import FaceMatchRequest


@dataclass(frozen=True)
class FaceRuntimeSettings:
    model_version: str = "scrfd-det10g_arcface-w600k-r50_v1"
    min_detection_score: float = 0.7
    min_face_size: int = 40
    min_sharpness: float = 30.0
    sample_every_frames: int = 5
    max_samples_per_track: int = 3
    max_tracks_per_frame: int = 2

    def __post_init__(self):
        if (not self.model_version.strip() or not 0 < self.min_detection_score <= 1
                or self.min_face_size < 1 or self.min_sharpness < 0
                or not isfinite(self.min_sharpness) or self.sample_every_frames < 1
                or self.max_samples_per_track < 2 or self.max_samples_per_track > 3
                or self.max_tracks_per_frame < 1):
            raise ValueError("Invalid face runtime settings")


def load_face_settings(path: Path) -> FaceRuntimeSettings:
    import yaml

    with Path(path).open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict) or set(values) != {"face_runtime"}:
        raise ValueError("Invalid face runtime configuration")
    return FaceRuntimeSettings(**values["face_runtime"])


class FaceExtractor:
    """Use existing SCRFD/ArcFace inference code without its gallery side effects."""

    def __init__(self, detection_path: Path, embedding_path: Path,
                 settings: FaceRuntimeSettings, *, detector: Any = None,
                 embedder: Any = None):
        self.detection_path = Path(detection_path)
        self.embedding_path = Path(embedding_path)
        self.settings = settings
        self.detector = detector
        self.embedder = embedder

    def _load_models(self) -> None:
        if self.detector is not None and self.embedder is not None:
            return
        if not self.detection_path.is_file() or not self.embedding_path.is_file():
            raise FileNotFoundError("Local face ONNX weights are required; no download attempted")
        if self.detector is None:
            detector = SCRFDDetector.__new__(SCRFDDetector)
            detector.session = ort.InferenceSession(str(self.detection_path),
                                                    providers=["CPUExecutionProvider"])
            detector.input_name = detector.session.get_inputs()[0].name
            detector.conf_thresh = self.settings.min_detection_score
            detector.nms_thresh = 0.4
            detector.strides = [8, 16, 32]
            detector.anchor_num = 2
            self.detector = detector
        if self.embedder is None:
            embedder = ArcFaceEmbedder.__new__(ArcFaceEmbedder)
            embedder.session = ort.InferenceSession(str(self.embedding_path),
                                                    providers=["CPUExecutionProvider"])
            embedder.input_name = embedder.session.get_inputs()[0].name
            self.embedder = embedder

    def extract_person(self, frame: np.ndarray, bbox: list[float] | tuple[float, ...],
                       *, require_single_face: bool = False) -> tuple[list[float], float] | None:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or len(bbox) != 4:
            return None
        if not all(isfinite(float(value)) for value in bbox):
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        self._load_models()
        faces = self.detector.detect(crop)
        if not faces or (require_single_face and len(faces) != 1):
            return None
        face = max(faces, key=lambda item: item["score"])
        fx1, fy1, fx2, fy2 = [int(value) for value in face["bbox"][:4]]
        if (face["score"] < self.settings.min_detection_score
                or min(fx2 - fx1, fy2 - fy1) < self.settings.min_face_size):
            return None
        face_crop = crop[max(0, fy1):min(crop.shape[0], fy2),
                         max(0, fx1):min(crop.shape[1], fx2)]
        if face_crop.size == 0:
            return None
        sharpness = float(cv2.Laplacian(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY),
                                       cv2.CV_64F).var())
        if sharpness < self.settings.min_sharpness:
            return None
        landmarks = np.asarray(face["kps"], dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.all(np.isfinite(landmarks)):
            return None
        transform, _ = cv2.estimateAffinePartial2D(landmarks, REFERENCE_FACIAL_POINTS)
        if transform is None or not np.all(np.isfinite(transform)):
            return None
        aligned = cv2.warpAffine(crop, transform, (112, 112))
        vector = np.asarray(self.embedder.get_embedding(aligned), dtype=np.float64)
        if vector.shape != (512,) or not np.all(np.isfinite(vector)):
            return None
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return None
        return (vector / norm).tolist(), float(face["score"])


@dataclass
class _TrackFace:
    last_attempt_frame: int = -1_000_000
    samples: int = 0
    votes: list[tuple[str, str | None]] = field(default_factory=list)
    result: dict = field(default_factory=dict)


class FacePipeline:
    def __init__(self, extractor: FaceExtractor, client: Any,
                 settings: FaceRuntimeSettings):
        self.extractor, self.client, self.settings = extractor, client, settings
        self._frames: dict[str, int] = {}
        self._budgets: dict[str, int] = {}
        self._tracks: dict[tuple[str, int], _TrackFace] = {}

    def begin_frame(self, camera_id: str) -> None:
        self._frames[camera_id] = self._frames.get(camera_id, 0) + 1
        self._budgets[camera_id] = 0

    def observe(self, camera_id: str, track_id: int, frame: np.ndarray,
                bbox: list[float], eligible: bool, zone_ids: list[str]) -> dict:
        unavailable = {"identity_evaluated": False, "identity_status": "unavailable"}
        key = (camera_id, track_id)
        if not eligible:
            self._tracks.pop(key, None)
            return unavailable
        track = self._tracks.setdefault(key, _TrackFace())
        current = self._frames.get(camera_id, 0)
        if (track.samples >= self.settings.max_samples_per_track
                or current - track.last_attempt_frame < self.settings.sample_every_frames
                or self._budgets.get(camera_id, 0) >= self.settings.max_tracks_per_frame):
            return dict(track.result or unavailable)
        self._budgets[camera_id] = self._budgets.get(camera_id, 0) + 1
        track.last_attempt_frame = current
        try:
            sample = self.extractor.extract_person(frame, bbox)
            if sample is None:
                track.result = unavailable
                return dict(unavailable)
            vector, quality = sample
            result = self.client.match_face(FaceMatchRequest(
                camera_id=camera_id, zone_ids=zone_ids,
                model_version=self.settings.model_version, vector=vector))
        except Exception:
            track.result = unavailable
            return dict(unavailable)
        if result.status not in {"registered", "unknown"}:
            track.result = unavailable
            return dict(unavailable)
        track.samples += 1
        track.votes.append((result.status, result.person_id))
        track.votes = track.votes[-2:]
        confirmed = len(track.votes) == 2 and track.votes[0] == track.votes[1]
        track.result = {
            "identity_evaluated": True, "identity_status": result.status,
            "identity_sample_id": track.samples, "zone_access": result.zone_access,
            "identity_confirmed": confirmed,
            "person_id": result.person_id, "display_name": result.display_name,
            "face_similarity": result.similarity, "face_quality": quality,
        }
        return dict(track.result)

    def prune(self, camera_id: str, active: set[int]) -> None:
        for key in tuple(self._tracks):
            if key[0] == camera_id and key[1] not in active:
                del self._tracks[key]

    def clear_camera(self, camera_id: str) -> None:
        self.prune(camera_id, set())
        self._frames.pop(camera_id, None)
        self._budgets.pop(camera_id, None)
