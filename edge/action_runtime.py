"""Bounded, camera-scoped 32-frame action inference for local playback."""

from collections import deque
from dataclasses import dataclass, field
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common.schemas import CandidateDraft

CLASS_NAMES = ("Normal", "Property_Crime", "Personal_Violence")
EVENT_TYPES = {1: "possible_property_crime", 2: "possible_person_violence"}


@dataclass(frozen=True)
class ActionSettings:
    sample_hz: float = 10.0
    window_frames: int = 32
    inference_interval_seconds: float = 1.5
    max_sample_gap_seconds: float = 0.5
    max_active_tracks: int = 8
    max_inferences_per_cycle: int = 2
    normal_threshold: float = 0.70
    property_threshold: float = 0.70
    violence_threshold: float = 0.70
    cooldown_seconds: float = 10.0
    context_margin: float = 0.2

    def __post_init__(self) -> None:
        positive = (self.sample_hz, self.inference_interval_seconds,
                    self.max_sample_gap_seconds)
        if (any(not isfinite(value) or value <= 0 for value in positive)
                or self.window_frames < 2 or self.max_active_tracks < 1
                or self.max_inferences_per_cycle < 1
                or not 0 < self.normal_threshold <= 1
                or not 0 < self.property_threshold <= 1
                or not 0 < self.violence_threshold <= 1
                or not isfinite(self.cooldown_seconds) or self.cooldown_seconds < 0
                or not isfinite(self.context_margin) or self.context_margin < 0):
            raise ValueError("Invalid action settings")


def load_action_settings(path: Path) -> ActionSettings:
    import yaml

    with path.open(encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict) or not isinstance(values.get("action"), dict):
        raise ValueError("Invalid action configuration")
    return ActionSettings(**values["action"])


def _crop_rgb(frame: np.ndarray, bbox: list[float], margin: float) -> np.ndarray | None:
    if frame.ndim != 3 or frame.shape[2] != 3 or len(bbox) != 4:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if not all(isfinite(value) for value in bbox) or x2 <= x1 or y2 <= y1:
        return None
    extra_x, extra_y = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, int(x1 - extra_x)), max(0, int(y1 - extra_y))
    right, bottom = min(width, ceil(x2 + extra_x)), min(height, ceil(y2 + extra_y))
    if right - left < 2 or bottom - top < 2:
        return None
    crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    return np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR))


@dataclass
class _TrackWindow:
    frames: deque[np.ndarray]
    history: deque[int | None] = field(default_factory=lambda: deque(maxlen=3))
    last_sample_time: float | None = None
    last_inferred_time: float | None = None
    last_emitted: dict[int, float] = field(default_factory=dict)
    last_decision: str = "Insufficient"


class ActionEngine:
    """Admit a bounded number of tracks; infer ready windows fairly each cycle."""

    def __init__(self, model: Any, camera_ids: tuple[str, ...], settings: ActionSettings):
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Action cameras must be unique")
        self.model, self.camera_ids, self.settings = model, camera_ids, settings
        self._tracks: dict[tuple[str, int], _TrackWindow] = {}
        self._rejected: dict[str, int] = dict.fromkeys(camera_ids, 0)
        self._next_camera = 0

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def rejected_tracks(self, camera_id: str) -> int:
        return self._rejected[camera_id]

    def last_decision(self, camera_id: str, track_id: int) -> str:
        return self._tracks[(camera_id, track_id)].last_decision

    def clear_missing(self, camera_id: str, active_ids: set[int]) -> None:
        for key in tuple(self._tracks):
            if key[0] == camera_id and key[1] not in active_ids:
                del self._tracks[key]

    def observe(self, camera_id: str, track_id: int, frame: np.ndarray,
                bbox: list[float], timestamp_seconds: float) -> None:
        if camera_id not in self.camera_ids or not isfinite(timestamp_seconds):
            raise ValueError("Invalid action observation")
        key = (camera_id, track_id)
        track = self._tracks.get(key)
        if track is None:
            if len(self._tracks) >= self.settings.max_active_tracks:
                counts = {name: sum(key[0] == name for key in self._tracks)
                          for name in self.camera_ids}
                dominant = max(self.camera_ids, key=counts.get)
                if counts[dominant] <= counts[camera_id]:
                    self._rejected[camera_id] += 1
                    return
                # The oldest resident from an overrepresented camera yields
                # one slot; no unbounded pending-track queue is created.
                victim = next(key for key in self._tracks if key[0] == dominant)
                del self._tracks[victim]
                self._rejected[dominant] += 1
            track = _TrackWindow(deque(maxlen=self.settings.window_frames))
            self._tracks[key] = track
        previous = track.last_sample_time
        if previous is not None:
            if timestamp_seconds == previous:
                return
            if (timestamp_seconds < previous
                    or timestamp_seconds - previous > self.settings.max_sample_gap_seconds):
                track.frames.clear()
                track.history.clear()
                track.last_inferred_time = None
                track.last_decision = "Insufficient"
            elif timestamp_seconds - previous + 1e-8 < 1 / self.settings.sample_hz:
                return
        crop = _crop_rgb(frame, bbox, self.settings.context_margin)
        if crop is None:
            track.frames.clear()
            track.history.clear()
            track.last_sample_time = None
            track.last_decision = "Insufficient"
            return
        track.frames.append(crop)
        track.last_sample_time = timestamp_seconds

    def _ready(self, key: tuple[str, int]) -> bool:
        track = self._tracks[key]
        if len(track.frames) < self.settings.window_frames:
            return False
        return (track.last_inferred_time is None
                or track.last_sample_time - track.last_inferred_time + 1e-8
                >= self.settings.inference_interval_seconds)

    def backlog(self, camera_id: str) -> int:
        return sum(key[0] == camera_id and self._ready(key) for key in self._tracks)

    def finish_cycle(self) -> list[tuple[str, CandidateDraft]]:
        selected: list[tuple[str, int]] = []
        for offset in range(len(self.camera_ids)):
            camera = self.camera_ids[(self._next_camera + offset) % len(self.camera_ids)]
            ready = [key for key in self._tracks if key[0] == camera and self._ready(key)]
            ready.sort(key=lambda key: (self._tracks[key].last_inferred_time or -1,
                                        self._tracks[key].last_sample_time))
            for key in ready:
                if len(selected) >= self.settings.max_inferences_per_cycle:
                    break
                selected.append(key)
            if len(selected) >= self.settings.max_inferences_per_cycle:
                break
        self._next_camera = (self._next_camera + 1) % len(self.camera_ids)
        if not selected:
            return []
        probabilities = self.model.predict_batch([tuple(self._tracks[key].frames) for key in selected])
        if len(probabilities) != len(selected):
            raise RuntimeError("Action model returned an invalid batch")
        validated = []
        for values in probabilities:
            numbers = [float(value) for value in values]
            if (len(numbers) != 3 or not all(isfinite(value) and 0 <= value <= 1 for value in numbers)
                    or abs(sum(numbers) - 1) > 1e-3):
                raise RuntimeError("Action model returned invalid probabilities")
            validated.append(numbers)
        events = []
        for key, numbers in zip(selected, validated, strict=True):
            track = self._tracks[key]
            track.last_inferred_time = track.last_sample_time
            predicted = int(np.argmax(numbers))
            threshold = (self.settings.normal_threshold,
                         self.settings.property_threshold,
                         self.settings.violence_threshold)[predicted]
            label = predicted if numbers[predicted] >= threshold else None
            track.last_decision = CLASS_NAMES[label] if label is not None else "Insufficient"
            track.history.append(label)
            if label not in EVENT_TYPES or track.history.count(label) < 2:
                continue
            previous = track.last_emitted.get(label)
            if previous is not None and track.last_sample_time - previous < self.settings.cooldown_seconds:
                continue
            track.last_emitted[label] = track.last_sample_time
            events.append((key[0], CandidateDraft(
                candidate_type=EVENT_TYPES[label], local_track_id=key[1],
                confidence=numbers[label],
                reason_codes=["action_temporal_2_of_3", "threshold_baseline_unvalidated"],
                attributes={"class_probabilities": dict(zip(CLASS_NAMES, numbers, strict=True)),
                            "threshold_status": "baseline_unvalidated",
                            "media_time_seconds": track.last_sample_time},
            )))
        return events


class ActionModel:
    """Load only a local trained state_dict; never infer from random weights."""

    def __init__(self, weights: Path, metadata: Path, *, device: str | int = "cpu"):
        weights, metadata = Path(weights), Path(metadata)
        if not weights.is_file():
            raise FileNotFoundError("A local action checkpoint is required")
        if not metadata.is_file():
            raise FileNotFoundError("Local action metadata is required")
        import torch
        import yaml

        with metadata.open(encoding="utf-8") as file:
            meta = yaml.safe_load(file)
        if (not isinstance(meta, dict) or meta.get("input_frames") != 32
                or meta.get("image_size") != [224, 224]
                or meta.get("num_classes") != 3
                or meta.get("class_names") != dict(enumerate(CLASS_NAMES))):
            raise ValueError("Action checkpoint metadata does not match runtime")
        from ai.action_classifier import ActionClassifier

        target = resolve_action_device(device)
        model = ActionClassifier(num_classes=3, pretrained=False)
        try:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
        except Exception:
            raise RuntimeError("Local action checkpoint is incompatible") from None
        self.model = model.to(target).eval()
        self.device = target
        from torchvision import transforms

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict_batch(self, windows: list[tuple[np.ndarray, ...]]) -> list[list[float]]:
        import torch

        if any(len(window) != 32 for window in windows):
            raise ValueError("Action inference requires 32 fresh frames")
        batch = torch.stack([
            torch.stack([self.transform(Image.fromarray(frame)) for frame in window])
            for window in windows
        ]).to(self.device)
        with torch.inference_mode():
            return torch.softmax(self.model(batch), dim=1).cpu().tolist()


def resolve_action_device(device: str | int):
    """Treat numeric resource config as a CUDA index with a CPU-only fallback."""
    import torch

    if str(device).lower() == "cpu":
        return torch.device("cpu")
    if isinstance(device, int) or str(device).isdigit():
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
