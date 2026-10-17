"""Bounded local JPEG storage for evidence declared by event metadata."""

import hashlib
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from common.schemas import EvidenceReference

MAX_JPEG_BYTES = 2_000_000


class InvalidEvidenceError(ValueError):
    pass


class EvidenceTooLargeError(InvalidEvidenceError):
    pass


class EvidenceStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for(self, event_id: UUID, evidence_id: UUID) -> Path:
        return self.root / str(event_id) / f"{evidence_id}.jpg"

    def validate(self, declared: EvidenceReference, jpeg: bytes) -> None:
        if len(jpeg) > MAX_JPEG_BYTES:
            raise EvidenceTooLargeError("Evidence image exceeds local size limit")
        if (not jpeg or declared.sha256 is None or declared.size_bytes is None
                or len(jpeg) != declared.size_bytes
                or hashlib.sha256(jpeg).hexdigest() != declared.sha256
                or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
            raise InvalidEvidenceError("Invalid evidence image")
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise InvalidEvidenceError("Invalid evidence image")

    def save(
        self, event_id: UUID, declared: EvidenceReference, jpeg: bytes
    ) -> EvidenceReference:
        self.validate(declared, jpeg)
        target = self.path_for(event_id, declared.evidence_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as file:
                file.write(jpeg)
        except FileExistsError:
            if target.read_bytes() != jpeg:
                raise InvalidEvidenceError("Conflicting evidence image") from None
        return declared.model_copy(update={
            "relative_path": f"evidence/{event_id}/{declared.evidence_id}.jpg"
        })
