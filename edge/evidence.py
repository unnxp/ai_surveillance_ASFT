"""Build bounded in-memory JPEG evidence only when a candidate is emitted."""

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import cv2
import numpy as np

from common.schemas import EvidenceKind, EvidenceReference

MAX_JPEG_BYTES = 2_000_000


@dataclass(frozen=True)
class EncodedEvidence:
    reference: EvidenceReference
    content: bytes


class EvidenceEncoder:
    def __init__(self, id_factory: Callable[[], UUID] = uuid4):
        self.id_factory = id_factory

    def _encode(self, image: np.ndarray, kind: EvidenceKind, maximum_side: int,
                captured_at_utc: datetime) -> EncodedEvidence | None:
        height, width = image.shape[:2]
        if max(height, width) > maximum_side:
            scale = maximum_side / max(height, width)
            image = cv2.resize(image, (max(1, round(width * scale)),
                                       max(1, round(height * scale))))
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return None
        content = encoded.tobytes()
        if len(content) > MAX_JPEG_BYTES:
            return None
        reference = EvidenceReference(
            evidence_id=self.id_factory(), kind=kind,
            sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content),
            captured_at_utc=captured_at_utc,
        )
        return EncodedEvidence(reference, content)

    def prepare(
        self, frame: Any, bbox: list[float] | None, captured_at_utc: datetime
    ) -> tuple[tuple[EncodedEvidence, ...], bool]:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3
                or frame.shape[0] == 0 or frame.shape[1] == 0):
            return (), True
        full = self._encode(frame, EvidenceKind.KEYFRAME, 1280, captured_at_utc)
        if full is None:
            return (), True
        if bbox is None or len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
            return (full,), True
        height, width = frame.shape[:2]
        left, top, right, bottom = [int(value) for value in bbox]
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        if left >= right or top >= bottom:
            return (full,), True
        crop = self._encode(frame[top:bottom, left:right], EvidenceKind.PERSON_CROP,
                            512, captured_at_utc)
        return ((full, crop), False) if crop is not None else ((full,), True)
