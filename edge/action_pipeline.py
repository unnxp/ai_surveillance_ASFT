"""Bridge camera-scoped detector observations to shared action scheduling."""

from typing import Any

import numpy as np

from common.schemas import CandidateDraft
from edge.action_runtime import ActionEngine


class ActionProcessor:
    def __init__(self, detection: Any, engine: ActionEngine):
        self.detection = detection
        self.engine = engine

    def process_timed(self, camera_id: str, frame: np.ndarray, state: dict,
                      timestamp_seconds: float) -> tuple[CandidateDraft, ...]:
        drafts = tuple(self.detection.process(camera_id, frame, state))
        eligible = {track_id for track_id, observation in state.items()
                    if observation.get("in_inference_roi", True)}
        self.engine.clear_missing(camera_id, eligible)
        for track_id, observation in state.items():
            if track_id not in eligible:
                continue
            self.engine.observe(camera_id, int(track_id), frame,
                                observation["bbox"], timestamp_seconds)
        return drafts

    def finish_cycle(self) -> list[tuple[str, CandidateDraft]]:
        return self.engine.finish_cycle()

    def backlog(self, camera_id: str) -> int:
        return self.engine.backlog(camera_id)

    def clear_camera(self, camera_id: str) -> None:
        self.engine.clear_missing(camera_id, set())
        clear = getattr(self.detection, "clear_camera", None)
        if clear is not None:
            clear(camera_id)
