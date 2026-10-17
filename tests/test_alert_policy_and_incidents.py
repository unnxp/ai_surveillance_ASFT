"""Tests for Alert Policy, Deduplication, and Incident Timeline (Week 3B)."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from common.schemas import (
    AlertRouting,
    AlertSeverity,
    CandidateType,
    EventCandidate,
)
from server.alert_policy import AlertPolicyEngine
from server.app import create_app
from server.database import Database

BASE_TIME = datetime(2026, 10, 10, 14, 0, 0, tzinfo=timezone.utc)


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
    db = Database(tmp_path / "policy_test.db")
    db.initialize()
    return db


@pytest.fixture
def policy_engine(database):
    return AlertPolicyEngine(database)


def make_candidate(ctype: CandidateType, camera_id="camera_01", time_offset_sec=0) -> EventCandidate:
    return EventCandidate(
        schema_version="1.0",
        camera_id=camera_id,
        gateway_id="gw1",
        gateway_boot_id="boot1",
        sequence_number=1,
        observed_at_utc=BASE_TIME + timedelta(seconds=time_offset_sec),
        model_versions={"model": "v1"},
        configuration_version="v1",
        event_id=uuid4(),
        candidate_type=ctype,
        attributes={"test": True},
    )


def test_default_alert_policy_rules(database):
    policy = database.get_alert_policy()
    assert len(policy.rules) >= 7

    violence_rule = policy.rules[CandidateType.POSSIBLE_PERSON_VIOLENCE]
    assert violence_rule.severity == AlertSeverity.HIGH
    assert violence_rule.routing == AlertRouting.IMMEDIATE_ALERT

    property_rule = policy.rules[CandidateType.POSSIBLE_PROPERTY_CRIME]
    assert property_rule.severity == AlertSeverity.MEDIUM
    assert property_rule.routing == AlertRouting.REVIEW_QUEUE

    handoff_rule = policy.rules[CandidateType.CROSS_CAMERA_HANDOFF]
    assert handoff_rule.severity == AlertSeverity.LOW
    assert handoff_rule.routing == AlertRouting.CONTEXT_ONLY


def test_incident_timeline_correlates_events_for_same_person(policy_engine, database):
    # 1. First event: Tripwire crossing at camera_01
    ev1 = make_candidate(CandidateType.TRIPWIRE_CROSSING, camera_id="camera_01", time_offset_sec=0)
    decision1 = policy_engine.process_candidate(ev1, global_person_id="GP-0001")
    assert decision1.is_alert is True
    assert decision1.severity == AlertSeverity.MEDIUM
    assert not decision1.is_deduped
    assert decision1.incident.incident_id == "INC-0001"
    assert len(decision1.incident.timeline) == 1

    # 2. Second event: Restricted Zone breach at camera_02 after 10s
    ev2 = make_candidate(CandidateType.UNKNOWN_PERSON_IN_RESTRICTED_ZONE, camera_id="camera_02", time_offset_sec=10)
    decision2 = policy_engine.process_candidate(ev2, global_person_id="GP-0001")
    assert decision2.is_alert is True
    assert decision2.severity == AlertSeverity.HIGH
    assert not decision2.is_deduped
    # Must associate to the same incident!
    assert decision2.incident.incident_id == "INC-0001"
    assert decision2.incident.highest_severity == AlertSeverity.HIGH
    assert len(decision2.incident.timeline) == 2
    assert decision2.incident.timeline[0].candidate_type == CandidateType.TRIPWIRE_CROSSING
    assert decision2.incident.timeline[1].candidate_type == CandidateType.UNKNOWN_PERSON_IN_RESTRICTED_ZONE


def test_cooldown_deduplication(policy_engine):
    # First Tripwire event at T=0s (cooldown = 10s)
    ev1 = make_candidate(CandidateType.TRIPWIRE_CROSSING, camera_id="camera_01", time_offset_sec=0)
    decision1 = policy_engine.process_candidate(ev1, global_person_id="GP-0001")
    assert decision1.is_alert is True
    assert not decision1.is_deduped

    # Rapid second Tripwire event from same person & camera at T=3s -> must be deduped!
    ev2 = make_candidate(CandidateType.TRIPWIRE_CROSSING, camera_id="camera_01", time_offset_sec=3)
    decision2 = policy_engine.process_candidate(ev2, global_person_id="GP-0001")
    assert decision2.is_deduped is True
    assert decision2.is_alert is False  # Suppressed by cooldown!
    # But still attached to the incident timeline for complete evidence
    assert len(decision2.incident.timeline) == 2


@pytest.mark.asyncio
async def test_alert_policy_api_crud_and_incidents(tmp_path):
    app = create_app(tmp_path / "api_policy.db")
    db = Database(tmp_path / "api_policy.db")
    db.initialize()

    # 1. Get alert policy
    status, data = await request(app, "GET", "/api/v1/alert-policy")
    assert status == 200
    assert "policy" in data
    assert "rules" in data["policy"]

    # 2. Update alert policy: set property crime to CRITICAL
    policy_dict = data["policy"]
    policy_dict["rules"]["possible_property_crime"]["severity"] = "CRITICAL"
    status_put, data_put = await request(app, "PUT", "/api/v1/alert-policy", policy_dict)
    assert status_put == 200
    assert data_put["policy"]["rules"]["possible_property_crime"]["severity"] == "CRITICAL"

    # 3. Ingest an event and verify incident creation
    event = {
        "schema_version": "1.0",
        "camera_id": "camera_01",
        "gateway_id": "gw1",
        "gateway_boot_id": "boot1",
        "sequence_number": 1,
        "observed_at_utc": BASE_TIME.isoformat(),
        "model_versions": {"model": "v1"},
        "configuration_version": "v1",
        "event_id": str(uuid4()),
        "candidate_type": "possible_property_crime",
        "attributes": {},
    }
    status_ingest, ack = await request(app, "POST", "/api/v1/events", event)
    assert status_ingest == 201

    # 4. List incidents
    status_inc, inc_data = await request(app, "GET", "/api/v1/incidents")
    assert status_inc == 200
    incidents = inc_data["incidents"]
    assert len(incidents) == 1
    assert incidents[0]["highest_severity"] == "CRITICAL"

    # 5. Reset alert policy to default
    status_reset, data_reset = await request(app, "POST", "/api/v1/alert-policy/reset")
    assert status_reset == 200
    assert data_reset["policy"]["rules"]["possible_property_crime"]["severity"] == "MEDIUM"
