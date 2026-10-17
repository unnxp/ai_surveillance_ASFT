"""Tests for Camera Topology, Anonymous Global Person ID, and Cross-Camera Association (Week 3A)."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from common.identity_schemas import TopologyConfig, TopologyEdgeInput
from server.app import create_app
from server.association import CrossCameraAssociationEngine
from server.database import Database

BASE_TIME = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)


async def request(app, method, url, payload=None):
    """Invoke the real ASGI app without HTTPX, sockets, or external services."""
    target = urlsplit(url)
    body = b"" if payload is None else json.dumps(payload).encode()
    complete = asyncio.Event()
    delivered = False
    messages = []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await complete.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            complete.set()

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method, "scheme": "http",
        "path": target.path, "raw_path": target.path.encode(), "root_path": "",
        "query_string": target.query.encode(),
        "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1), "server": ("test", 80), "extensions": {},
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    start = next(item for item in messages if item["type"] == "http.response.start")
    response_body = b"".join(item.get("body", b"") for item in messages
                             if item["type"] == "http.response.body")
    return start["status"], json.loads(response_body) if response_body else None


@pytest.fixture
def database(tmp_path):
    db = Database(tmp_path / "test_surveillance.db")
    db.initialize()
    return db


@pytest.fixture
def association_engine(database):
    return CrossCameraAssociationEngine(database, session_timeout_seconds=120.0)


def test_topology_validation_rejects_invalid_edges():
    # 1. Source equals destination
    with pytest.raises(ValueError, match="cannot be identical"):
        TopologyEdgeInput(source_camera="cam1", destination_camera="cam1",
                          min_travel_seconds=5.0, max_travel_seconds=20.0)

    # 2. Max time <= Min time
    with pytest.raises(ValueError, match="must exceed minimum"):
        TopologyEdgeInput(source_camera="cam1", destination_camera="cam2",
                          min_travel_seconds=20.0, max_travel_seconds=10.0)

    # 3. Duplicate pairs in TopologyConfig
    with pytest.raises(ValueError, match="Duplicate topology edge"):
        TopologyConfig(edges=[
            TopologyEdgeInput(source_camera="cam1", destination_camera="cam2",
                              min_travel_seconds=2.0, max_travel_seconds=15.0),
            TopologyEdgeInput(source_camera="cam1", destination_camera="cam2",
                              min_travel_seconds=5.0, max_travel_seconds=20.0),
        ])


def test_topology_crud_in_database(database):
    edges = [
        TopologyEdgeInput(source_camera="camera_01", destination_camera="camera_02",
                          min_travel_seconds=3.0, max_travel_seconds=25.0),
        TopologyEdgeInput(source_camera="camera_02", destination_camera="camera_01",
                          min_travel_seconds=4.0, max_travel_seconds=30.0),
    ]
    saved = database.set_topology_edges(edges)
    assert len(saved) == 2
    assert saved[0].source_camera == "camera_01"
    assert saved[0].destination_camera == "camera_02"
    assert saved[0].min_travel_seconds == 3.0

    edge = database.get_topology_edge("camera_01", "camera_02")
    assert edge is not None
    assert edge.max_travel_seconds == 25.0

    # Non-existent edge returns None
    assert database.get_topology_edge("camera_01", "camera_03") is None


def test_global_person_session_creation_and_extension(association_engine, database):
    # First observation in camera_01 -> creates GP-0001
    gid1, status1, is_handoff1 = association_engine.correlate(
        camera_id="camera_01",
        observed_at_utc=BASE_TIME,
        local_track_id=10,
        person_id="emp_01",
    )
    assert gid1 == "GP-0001"
    assert status1 == "matched"
    assert not is_handoff1

    session = database.get_session("GP-0001")
    assert session is not None
    assert session.status == "active"
    assert session.observation_count == 1
    assert session.associated_cameras == ["camera_01"]

    # Same person continuing in camera_01 after 5 seconds -> extends GP-0001
    t2 = BASE_TIME + timedelta(seconds=5)
    gid2, status2, is_handoff2 = association_engine.correlate(
        camera_id="camera_01",
        observed_at_utc=t2,
        local_track_id=10,
        person_id="emp_01",
    )
    assert gid2 == "GP-0001"
    assert not is_handoff2

    session = database.get_session("GP-0001")
    assert session.observation_count == 2
    assert session.last_seen_at_utc == t2


def test_session_expires_after_120_seconds(association_engine, database):
    # Observation at T=0s
    gid1, _, _ = association_engine.correlate(
        camera_id="camera_01",
        observed_at_utc=BASE_TIME,
        local_track_id=1,
    )
    assert gid1 == "GP-0001"

    # Observation at T=150s (past the 120s timeout)
    t_later = BASE_TIME + timedelta(seconds=150)
    gid2, _, _ = association_engine.correlate(
        camera_id="camera_01",
        observed_at_utc=t_later,
        local_track_id=1,
    )
    # The old session should be expired and a new one created
    assert gid2 == "GP-0002"
    old_session = database.get_session("GP-0001")
    assert old_session.status == "expired"
    new_session = database.get_session("GP-0002")
    assert new_session.status == "active"


def test_cross_camera_association_success(association_engine, database):
    # Set up topology: camera_01 -> camera_02 (travel time: 5s - 30s)
    database.set_topology_edges([
        TopologyEdgeInput(source_camera="camera_01", destination_camera="camera_02",
                          min_travel_seconds=5.0, max_travel_seconds=30.0)
    ])

    # 1. Person seen at camera_01 at T=0s
    t0 = BASE_TIME
    gid_cam1, _, is_handoff_1 = association_engine.correlate(
        camera_id="camera_01",
        observed_at_utc=t0,
        local_track_id=101,
        person_id="alice",
    )
    assert gid_cam1 == "GP-0001"
    assert not is_handoff_1

    # 2. Person seen at camera_02 at T=12s (within 5s - 30s) with matching identity
    t1 = t0 + timedelta(seconds=12)
    gid_cam2, status_cam2, is_handoff_2 = association_engine.correlate(
        camera_id="camera_02",
        observed_at_utc=t1,
        local_track_id=202,
        person_id="alice",
    )
    # MUST associate to the same Global Person ID!
    assert gid_cam2 == "GP-0001"
    assert status_cam2 == "matched"
    assert is_handoff_2 is True

    session = database.get_session("GP-0001")
    assert session.associated_cameras == ["camera_01", "camera_02"]


def test_cross_camera_rejected_when_too_fast_or_too_slow(association_engine, database):
    # Topology: camera_01 -> camera_02 (travel time: 10s - 40s)
    database.set_topology_edges([
        TopologyEdgeInput(source_camera="camera_01", destination_camera="camera_02",
                          min_travel_seconds=10.0, max_travel_seconds=40.0)
    ])

    # Person at camera_01 at T=0s
    t0 = BASE_TIME
    association_engine.correlate(
        camera_id="camera_01", observed_at_utc=t0, local_track_id=1, face_similarity=0.9
    )

    # A) Appears at camera_02 at T=3s (too fast, min=10s) -> should NOT match
    t_fast = t0 + timedelta(seconds=3)
    gid_fast, _, is_handoff_fast = association_engine.correlate(
        camera_id="camera_02", observed_at_utc=t_fast, local_track_id=2, face_similarity=0.9
    )
    assert gid_fast == "GP-0002"
    assert not is_handoff_fast

    # B) Appears at camera_02 at T=60s (too slow, max=40s) -> should NOT match
    t_slow = t0 + timedelta(seconds=60)
    gid_slow, _, is_handoff_slow = association_engine.correlate(
        camera_id="camera_02", observed_at_utc=t_slow, local_track_id=3, face_similarity=0.9
    )
    assert gid_slow == "GP-0003"
    assert not is_handoff_slow


@pytest.mark.asyncio
async def test_api_topology_and_sessions(tmp_path):
    app = create_app(tmp_path / "api_test.db")
    app_db = Database(tmp_path / "api_test.db")
    app_db.initialize()

    # 1. Put topology
    payload = {
        "edges": [
            {
                "source_camera": "camera_01",
                "destination_camera": "camera_02",
                "min_travel_seconds": 2.5,
                "max_travel_seconds": 20.0,
                "enabled": True,
            }
        ]
    }
    status, data = await request(app, "PUT", "/api/v1/topology", payload)
    assert status == 200
    assert len(data["edges"]) == 1
    assert data["edges"][0]["source_camera"] == "camera_01"

    # 2. Get topology
    status, data = await request(app, "GET", "/api/v1/topology")
    assert status == 200
    assert len(data["edges"]) == 1

    # 3. Ingest an event and verify global_person_id assignment
    event_payload = {
        "schema_version": "1.0",
        "camera_id": "camera_01",
        "gateway_id": "gateway_test",
        "gateway_boot_id": "boot_123",
        "sequence_number": 1,
        "observed_at_utc": BASE_TIME.isoformat(),
        "model_versions": {"yolo": "yolov8n"},
        "configuration_version": "v1",
        "event_id": str(uuid4()),
        "candidate_type": "tripwire_crossing",
        "local_track_id": 5,
        "reason_codes": ["crossed_doorway"],
        "attributes": {"tripwire_id": "doorway"},
    }
    status_ingest, _ = await request(app, "POST", "/api/v1/events", event_payload)
    assert status_ingest == 201

    # 4. List global person sessions
    status_sess, data_sess = await request(app, "GET", "/api/v1/global-person-sessions")
    assert status_sess == 200
    sessions = data_sess["sessions"]
    assert len(sessions) >= 1
    assert sessions[0]["global_person_id"] == "GP-0001"
    assert sessions[0]["associated_cameras"] == ["camera_01"]
