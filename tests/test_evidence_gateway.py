"""Synthetic frame evidence stays scoped to its camera and event."""

import hashlib
from datetime import datetime, timezone
from uuid import UUID

import cv2
import numpy as np
import pytest
import requests

from cameras.video_source import VideoFileSource
from clients.central_client import CentralClient, RequestsTransport
from common.schemas import CandidateDraft
from edge.run_gateway import run_gateway
from edge.shared_gateway import CameraRuntime, SharedGateway
from tests.test_week1b_http import local_server, make_video

NOW = datetime(2026, 9, 28, 10, tzinfo=timezone.utc)


def frame(color):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[10:70, 20:80] = color
    return image


def test_encoder_makes_bounded_keyframe_and_clipped_crop():
    from edge.evidence import EvidenceEncoder

    ids = iter([UUID(int=11), UUID(int=12)])
    items, incomplete = EvidenceEncoder(id_factory=lambda: next(ids)).prepare(
        frame((20, 80, 160)), [10, 5, 120, 75], NOW
    )
    assert incomplete is False
    assert [item.reference.kind.value for item in items] == ["keyframe", "person_crop"]
    assert [item.reference.evidence_id for item in items] == [UUID(int=11), UUID(int=12)]
    assert all(item.reference.sha256 == hashlib.sha256(item.content).hexdigest()
               and item.reference.size_bytes == len(item.content) for item in items)
    assert [cv2.imdecode(np.frombuffer(item.content, dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2]
            for item in items] == [(80, 100), (70, 90)]


def test_encoder_omits_invalid_crop_and_abstains_on_invalid_frame():
    from edge.evidence import EvidenceEncoder

    encoder = EvidenceEncoder(id_factory=lambda: UUID(int=31))
    items, incomplete = encoder.prepare(frame((1, 2, 3)), [90, 70, 20, 10], NOW)
    assert [item.reference.kind.value for item in items] == ["keyframe"]
    assert incomplete is True
    assert encoder.prepare(None, [1, 2, 3, 4], NOW) == ((), True)


def test_client_uploads_raw_jpeg_and_validates_ack():
    from clients.central_client import CentralClient, CentralProtocolError
    from edge.evidence import EvidenceEncoder

    item = EvidenceEncoder(id_factory=lambda: UUID(int=41)).prepare(
        frame((1, 2, 3)), None, NOW
    )[0][0]

    class Transport:
        def put_bytes(self, path, body, content_type):
            self.sent = (path, body, content_type)
            return type("Response", (), {
                "status_code": 201,
                "json": lambda self: {"evidence_id": str(item.reference.evidence_id),
                                      "accepted": True, "duplicate": False},
            })()

    transport = Transport()
    client = CentralClient(transport)
    assert client.upload_evidence(UUID(int=1), item).accepted is True
    assert transport.sent == (
        f"/api/v1/events/{UUID(int=1)}/evidence/{UUID(int=41)}",
        item.content, "image/jpeg",
    )

    class FailingTransport:
        def put_bytes(self, path, body, content_type):
            raise RuntimeError("private-camera-url")

    with pytest.raises(CentralProtocolError) as error:
        CentralClient(FailingTransport()).upload_evidence(UUID(int=1), item)
    assert "private-camera-url" not in str(error.value)


class Source:
    def __init__(self, image):
        self.image = image

    def read(self):
        return True, self.image


class Processor:
    def process(self, camera_id, image, state):
        state.clear()
        state[7] = {"bbox": [20, 10, 80, 70], "score": 0.9}
        return (CandidateDraft(candidate_type="tripwire_crossing", local_track_id=7),)


class Sink:
    def __init__(self, fail_camera=None):
        self.calls = []
        self.events = []
        self.images = []
        self.fail_camera = fail_camera

    def send_candidate(self, event):
        self.calls.append(("event", event.camera_id))
        self.events.append(event)

    def upload_evidence(self, event_id, item):
        camera_id = next(event.camera_id for event in self.events if event.event_id == event_id)
        self.calls.append(("image", camera_id))
        if camera_id == self.fail_camera:
            raise RuntimeError("private-camera-url")
        self.images.append((camera_id, item))

    def send_heartbeat(self, heartbeat):
        self.calls.append(("heartbeat", heartbeat.camera_id))


def gateway_with_evidence(sink):
    from edge.evidence import EvidenceEncoder

    event_ids = iter([UUID(int=101), UUID(int=102)])
    evidence_ids = iter(UUID(int=value) for value in range(201, 205))
    return SharedGateway(
        [CameraRuntime("camera_01", Source(frame((0, 0, 180)))),
         CameraRuntime("camera_02", Source(frame((0, 180, 0))))],
        Processor(), sink, gateway_id="g", gateway_boot_id="boot",
        configuration_version="cfg", model_versions={"test": "v1"},
        clock=lambda: NOW, event_id_factory=lambda: next(event_ids),
        evidence_encoder=EvidenceEncoder(id_factory=lambda: next(evidence_ids)),
    )


def test_gateway_uploads_two_camera_images_after_metadata():
    sink = Sink()
    result = gateway_with_evidence(sink).run_once()
    assert result.failures == () and result.events_sent == 2
    assert [call for call in sink.calls if call[0] != "heartbeat"] == [
        ("event", "camera_01"), ("image", "camera_01"), ("image", "camera_01"),
        ("event", "camera_02"), ("image", "camera_02"), ("image", "camera_02"),
    ]
    assert all(len(event.evidence) == 2 for event in sink.events)
    crops = {camera: cv2.imdecode(np.frombuffer(item.content, dtype=np.uint8),
                                  cv2.IMREAD_COLOR)
             for camera, item in sink.images if item.reference.kind.value == "person_crop"}
    assert int(crops["camera_01"][:, :, 2].mean()) > 100
    assert int(crops["camera_02"][:, :, 1].mean()) > 100


def test_failed_upload_keeps_both_candidates_and_other_camera():
    sink = Sink(fail_camera="camera_01")
    result = gateway_with_evidence(sink).run_once()
    assert result.events_sent == 2
    assert [failure.stage for failure in result.failures] == ["evidence", "evidence"]
    assert {camera for camera, _ in sink.images} == {"camera_02"}
    assert all("private-camera-url" not in repr(failure) for failure in result.failures)


def test_generated_two_video_http_path_records_complete_and_incomplete_evidence(tmp_path):
    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    make_video(first, 1, 0)
    make_video(second, 1, 1)

    class FailFirstCameraEvidence:
        def __init__(self, client):
            self.client = client
            self.camera_by_event = {}

        def send_candidate(self, event):
            self.camera_by_event[event.event_id] = event.camera_id
            return self.client.send_candidate(event)

        def upload_evidence(self, event_id, item):
            if self.camera_by_event[event_id] == "camera_01":
                raise RuntimeError("fixture upload failure")
            return self.client.upload_evidence(event_id, item)

        def send_heartbeat(self, heartbeat):
            return self.client.send_heartbeat(heartbeat)

    from edge.evidence import EvidenceEncoder

    with local_server(tmp_path / "central.db") as origin, requests.Session() as reader:
        reader.trust_env = False
        transport = RequestsTransport(origin)
        try:
            event_ids = iter([UUID(int=301), UUID(int=302)])
            evidence_ids = iter(UUID(int=value) for value in range(401, 405))
            gateway = SharedGateway(
                [CameraRuntime("camera_01", VideoFileSource(first)),
                 CameraRuntime("camera_02", VideoFileSource(second))],
                Processor(), FailFirstCameraEvidence(CentralClient(transport)),
                gateway_id="g", gateway_boot_id="boot", configuration_version="cfg",
                model_versions={"fixture": "v1"}, clock=lambda: NOW,
                event_id_factory=lambda: next(event_ids),
                evidence_encoder=EvidenceEncoder(id_factory=lambda: next(evidence_ids)),
            )
            summary = run_gateway(gateway, fps=30, sleep=lambda _: None)
            assert summary.events_sent == 2 and summary.failures == 2
            events = reader.get(origin + "/api/v1/events", timeout=5).json()["events"]
            assert {event["camera_id"] for event in events} == {"camera_01", "camera_02"}
            assert all(len(event["evidence"]) == 2 for event in events)
            assert len({ref["evidence_id"] for event in events for ref in event["evidence"]}) == 4
            by_camera = {event["camera_id"]: event for event in events}
            first_status = reader.get(
                origin + f"/api/v1/events/{by_camera['camera_01']['event_id']}/evidence",
                timeout=5,
            ).json()
            second_status = reader.get(
                origin + f"/api/v1/events/{by_camera['camera_02']['event_id']}/evidence",
                timeout=5,
            ).json()
            assert first_status["evidence_incomplete"] is True
            assert all(not item["available"] for item in first_status["evidence"])
            assert second_status["evidence_incomplete"] is False
            assert all(item["available"] for item in second_status["evidence"])
            for item in second_status["evidence"]:
                response = reader.get(origin + item["url"], timeout=5)
                assert response.status_code == 200
                assert cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8),
                                    cv2.IMREAD_COLOR) is not None
        finally:
            transport.close()


def test_finish_cycle_action_candidate_uses_same_camera_frame():
    from edge.evidence import EvidenceEncoder

    class ActionLikeProcessor:
        def process(self, camera_id, image, state):
            state[7] = {"bbox": [20, 10, 80, 70]}
            return ()

        def finish_cycle(self):
            return (("camera_02", CandidateDraft(
                candidate_type="possible_person_violence", local_track_id=7,
            )),)

    sink = Sink()
    gateway = SharedGateway(
        [CameraRuntime("camera_01", Source(frame((180, 0, 0)))),
         CameraRuntime("camera_02", Source(frame((0, 180, 0))))],
        ActionLikeProcessor(), sink, gateway_id="g", gateway_boot_id="boot",
        configuration_version="cfg", model_versions={"fixture": "v1"},
        clock=lambda: NOW, event_id_factory=lambda: UUID(int=501),
        evidence_encoder=EvidenceEncoder(id_factory=iter(
            [UUID(int=601), UUID(int=602)]).__next__),
    )
    assert gateway.run_once().failures == ()
    assert [event.camera_id for event in sink.events] == ["camera_02"]
    crop = next(item for camera, item in sink.images if item.reference.kind.value == "person_crop")
    decoded = cv2.imdecode(np.frombuffer(crop.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert int(decoded[:, :, 1].mean()) > 100
