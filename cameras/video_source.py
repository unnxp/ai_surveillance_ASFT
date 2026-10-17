"""Finite local video playback: each successful read consumes a fresh frame."""

from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Any


class VideoFileSource:
    """The runner paces reads; EOF never loops or repeats the last frame."""

    def __init__(self, path: str | Path, *, capture_factory: Callable | None = None):
        try:
            resolved = Path(path).resolve()
            valid = resolved.is_file()
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError("Video source must be an existing local file")
        if capture_factory is None:
            import cv2

            capture_factory = cv2.VideoCapture
        self._capture = capture_factory(str(resolved))
        self.finished = False
        self._closed = False
        self.timestamp_seconds: float | None = None
        self._frame_index = 0
        if not self._capture.isOpened():
            self.close()
            raise RuntimeError("Cannot open video file")
        import cv2

        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._media_fps = fps if isfinite(fps) and fps > 0 else None

    def read(self) -> tuple[bool, Any]:
        if self.finished or self._closed:
            return False, None
        try:
            available, frame = self._capture.read()
        except Exception:
            self.finished = True
            raise RuntimeError("Video decoder failed") from None
        if not available or frame is None:
            self.finished = True
            return False, None
        if self._media_fps is not None:
            self.timestamp_seconds = self._frame_index / self._media_fps
        self._frame_index += 1
        return True, frame

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._capture.release()


class SyntheticSource:
    """Finite unidentifiable frames for the explicitly selected demo mode."""

    def __init__(self, frame_count: int = 30):
        if frame_count < 1:
            raise ValueError("Synthetic source needs at least one frame")
        self.remaining = frame_count
        self.finished = False

    def read(self) -> tuple[bool, Any]:
        if self.finished or not self.remaining:
            self.finished = True
            return False, None
        import numpy as np

        self.remaining -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        self.finished = True
