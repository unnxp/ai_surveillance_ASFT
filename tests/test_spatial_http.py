"""A synthetic directional crossing reaches Central through the existing gateway."""

from datetime import datetime, timezone
from uuid import UUID

import numpy as np
import requests

from cameras.video_source import VideoFileSource
from clients.central_client import CentralClient, RequestsTransport
from edge.detection import DetectionProcessor
from edge.run_gateway import run_gateway
from edge.shared_gateway import CameraRuntime, SharedGateway
from edge.spatial import SpatialEngine
from tests.test_action_http import synthetic_video
from tests.test_week1b_http import local_server


class Detector:
    def detect(self, frame):
        return None


class Tracker:
    def __init__(self):
        self.feet = iter((10, 17, 36))

    def update(self, boxes, frame):
        foot = next(self.feet)
        return np.array([[27, foot - 8, 37, foot, 7, 0.9, 0, 0]])


def test_two_local_videos_only_configured_camera_sends_tripwire_candidate(tmp_path):
    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    synthetic_video(first, 3)
    synthetic_video(second, 3)
    # Source frames are 64x48; x=32 crosses a horizontal line on camera a.
    spatial = SpatialEngine({"schema_version": 1, "cameras": {"a": {
        "tripwires": [{"id": "gate", "start": [0.3, 0.5], "end": [0.7, 0.5],
                       "direction": "negative_to_positive"}],
    }}})
    processor = DetectionProcessor(Detector(), ["a", "b"],
                                   tracker_factory=lambda _: Tracker(), spatial=spatial)
    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        transport = RequestsTransport(origin)
        try:
            gateway = SharedGateway(
                [CameraRuntime("a", VideoFileSource(first)), CameraRuntime("b", VideoFileSource(second))],
                processor, CentralClient(transport), gateway_id="g", gateway_boot_id="b",
                configuration_version="spatial-v1", model_versions={"detector": "controlled-test"},
                clock=lambda: datetime(2026, 9, 28, tzinfo=timezone.utc),
                event_id_factory=lambda: UUID(int=1),
            )
            summary = run_gateway(gateway, fps=30, sleep=lambda _: None)
            assert summary.failures == 0
            assert summary.events_sent == 1
            events = reader.get(origin + "/api/v1/events", timeout=5).json()["events"]
            assert [(event["camera_id"], event["candidate_type"], event["attributes"]) for event in events] == [
                ("a", "tripwire_crossing", {"tripwire_id": "gate", "direction": "negative_to_positive"})
            ]
        finally:
            transport.close()
