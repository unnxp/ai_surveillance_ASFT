"""Appearance tracklets for cross-camera association (baseline person re-identification).

Each local track is sampled a few times per second; person crops are embedded with an
ImageNet EfficientNet-B0 (global-average-pooled 1280-d features) and averaged into one
L2-normalized tracklet vector. Central compares tracklets across cameras, gated by
camera topology and travel time. Vectors stay in memory and are never written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common.identity_schemas import TrackletSummary

MODEL_VERSION = "efficientnet_b0-imagenet-gap1280-v1"
WEIGHTS_FILE = "efficientnet_b0_rwightman-7f5810bc.pth"
CROP_SIZE = (128, 256)  # width, height: person aspect ratio


class AppearanceEncoder:
    """Loads ImageNet EfficientNet-B0 from the local torch cache only (never downloads)."""

    def __init__(self, device: str | int = "cpu", weights: Path | None = None):
        import torch
        from torchvision import models

        path = weights or Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILE
        if not Path(path).is_file():
            raise FileNotFoundError("EfficientNet-B0 ImageNet weights are not in the local cache")
        model = models.efficientnet_b0(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) or str(device).isdigit()
                                   else device)
        self.features = model.features.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self._torch = torch

    def encode(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2

        torch = self._torch
        batch = np.stack([cv2.resize(c, CROP_SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1]
                          for c in crops_bgr]).astype(np.float32) / 255.0
        with torch.inference_mode():
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
            x = (x - self.mean) / self.std
            pooled = self.features(x).mean(dim=(2, 3))
            pooled = torch.nn.functional.normalize(pooled, dim=1)
        return pooled.cpu().numpy()


@dataclass
class _Track:
    first_seen: datetime
    last_seen: datetime
    last_sample: float = -1e9
    vectors: list[np.ndarray] = field(default_factory=list)
    ready_sent: bool = False
    global_person_id: str | None = None


class TrackletBuilder:
    """Samples crops per track, emits ``ready`` once and ``end`` when the track is lost."""

    def __init__(self, encoder: Any, send: Callable[[TrackletSummary], Any], gateway_boot_id: str, *,
                 sample_interval: float = 0.4, ready_samples: int = 4, max_samples: int = 16,
                 lost_after: float = 1.5, min_crop_height: int = 64,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.encoder, self.send, self.boot_id = encoder, send, gateway_boot_id
        self.sample_interval, self.ready_samples = sample_interval, ready_samples
        self.max_samples, self.lost_after = max_samples, lost_after
        self.min_crop_height, self.clock = min_crop_height, clock
        self.tracks: dict[tuple[str, int], _Track] = {}
        self.failures = 0

    def _emit(self, camera_id: str, track_id: int, track: _Track, phase: str) -> None:
        vector = np.mean(track.vectors, axis=0)
        vector = vector / (np.linalg.norm(vector) or 1.0)
        try:
            ack = self.send(TrackletSummary(
                camera_id=camera_id, gateway_boot_id=self.boot_id, local_track_id=track_id,
                phase=phase, first_seen_utc=track.first_seen, last_seen_utc=track.last_seen,
                sample_count=min(len(track.vectors), 64), model_version=MODEL_VERSION,
                appearance=[float(v) for v in vector]))
            track.global_person_id = getattr(ack, "global_person_id", None) or track.global_person_id
        except Exception:
            self.failures += 1  # association is best-effort; detection keeps running

    def observe(self, frames_by_camera: dict[str, Any], states_by_camera: dict[str, dict]) -> None:
        now = self.clock()
        stamp = now.timestamp()
        pending: list[tuple[tuple[str, int], np.ndarray]] = []
        for camera_id, state in states_by_camera.items():
            frame = frames_by_camera.get(camera_id)
            if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                continue
            height, width = frame.shape[:2]
            for raw_id, info in state.items():
                bbox = info.get("bbox") if isinstance(info, dict) else None
                if not bbox or len(bbox) != 4 or not info.get("in_inference_roi", True):
                    continue
                key = (camera_id, int(raw_id))
                track = self.tracks.setdefault(key, _Track(first_seen=now, last_seen=now))
                track.last_seen = now
                if track.global_person_id:
                    info["global_person_id"] = track.global_person_id
                if stamp - track.last_sample < self.sample_interval or len(track.vectors) >= self.max_samples:
                    continue
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(width, int(bbox[2])), min(height, int(bbox[3]))
                if y2 - y1 < self.min_crop_height or x2 - x1 < self.min_crop_height // 4:
                    continue
                track.last_sample = stamp
                pending.append((key, frame[y1:y2, x1:x2]))
        if pending:
            try:
                vectors = self.encoder.encode([crop for _, crop in pending])
            except Exception:
                self.failures += 1
                vectors = []
            for (key, _), vector in zip(pending, vectors):
                track = self.tracks[key]
                track.vectors.append(np.asarray(vector, dtype=np.float32))
                if not track.ready_sent and len(track.vectors) >= self.ready_samples:
                    track.ready_sent = True
                    self._emit(*key, track, "ready")
                    camera_state = states_by_camera.get(key[0], {})
                    if track.global_person_id and key[1] in camera_state:
                        camera_state[key[1]]["global_person_id"] = track.global_person_id
        for key, track in list(self.tracks.items()):
            if (now - track.last_seen).total_seconds() > self.lost_after:
                if track.ready_sent:
                    self._emit(*key, track, "end")
                del self.tracks[key]
