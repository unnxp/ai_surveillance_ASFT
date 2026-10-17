"""Tests for Human Review Workflow and Audit Records (Week 3C)."""

import asyncio
import json
from datetime import datetime, timezone
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from common.schemas import (
    AlertSeverity,
    CandidateType,
    EventCandidate,
    IncidentReviewRequest,
    ReviewAction,
)
from server.app import create_app
from server.database import Database

BASE_TIME = datetime(2026, 10, 10, 14, 0, 0, tzinfo=timezone.utc)


async def request(app, method, url, payload=None):
    """Invoke ASGI app directly without HTTPX or external network calls."""
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
    parsed = json.loads(response_body.decode()) if response_body else None
    return start["status"], parsed


def make_candidate(camera_id: str, candidate_type: CandidateType, observed_at: datetime) -> dict:
    return {
        "schema_version": "1.0",
        "camera_id": camera_id,
        "gateway_id": "gw_01",
        "gateway_boot_id": "boot_01",
        "sequence_number": 1,
        "model_versions": {"action": "3class_v1"},
        "configuration_version": "config_v1",
        "event_id": str(uuid4()),
        "candidate_type": candidate_type.value,
        "observed_at_utc": observed_at.isoformat(),
        "confidence": 0.85,
        "local_track_id": 1,
        "evidence": [],
        "attributes": {},
    }


def test_review_acknowledge(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_path)

    async def scenario():
        async with app.router.lifespan_context(app):
            # Ingest an event to generate INC-0001
            evt = make_candidate("cam1", CandidateType.POSSIBLE_PERSON_VIOLENCE, BASE_TIME)
            status, _ = await request(app, "POST", "/api/v1/events", evt)
            assert status == 201

            status, inc = await request(app, "GET", "/api/v1/incidents/INC-0001")
            assert status == 200
            assert inc["status"] == "open"
            assert len(inc["reviews"]) == 0

            # Operator acknowledges incident
            review_payload = {
                "action": "acknowledge",
                "reviewed_by": "operator_alpha",
            }
            status, reviewed = await request(app, "POST", "/api/v1/incidents/INC-0001/review", review_payload)
            assert status == 200
            assert reviewed["status"] == "acknowledged"
            assert len(reviewed["reviews"]) == 1
            assert reviewed["reviews"][0]["action"] == "acknowledge"
            assert reviewed["reviews"][0]["reviewed_by"] == "operator_alpha"

    asyncio.run(scenario())


def test_review_confirm_with_final_severity_and_notes(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_path)

    async def scenario():
        async with app.router.lifespan_context(app):
            evt = make_candidate("cam1", CandidateType.POSSIBLE_PROPERTY_CRIME, BASE_TIME)
            await request(app, "POST", "/api/v1/events", evt)

            # Confirm with severity upgrade to CRITICAL and operator notes
            review_payload = {
                "action": "confirm",
                "reviewed_by": "supervisor_01",
                "final_severity": "CRITICAL",
                "notes": "Confirmed burglary attempt at loading dock.",
            }
            status, reviewed = await request(app, "POST", "/api/v1/incidents/INC-0001/review", review_payload)
            assert status == 200
            assert reviewed["status"] == "confirmed"
            assert reviewed["highest_severity"] == "CRITICAL"
            assert len(reviewed["reviews"]) == 1
            rec = reviewed["reviews"][0]
            assert rec["action"] == "confirm"
            assert rec["final_severity"] == "CRITICAL"
            assert rec["notes"] == "Confirmed burglary attempt at loading dock."

    asyncio.run(scenario())


def test_review_reject_with_false_positive_reason(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_path)

    async def scenario():
        async with app.router.lifespan_context(app):
            evt = make_candidate("cam1", CandidateType.POSSIBLE_PERSON_VIOLENCE, BASE_TIME)
            await request(app, "POST", "/api/v1/events", evt)

            # Reject as false positive
            review_payload = {
                "action": "reject",
                "reviewed_by": "operator_beta",
                "false_positive_reason": "benign_activity",
                "notes": "Subjects were hugging, not fighting.",
            }
            status, reviewed = await request(app, "POST", "/api/v1/incidents/INC-0001/review", review_payload)
            assert status == 200
            assert reviewed["status"] == "rejected"
            rec = reviewed["reviews"][0]
            assert rec["action"] == "reject"
            assert rec["false_positive_reason"] == "benign_activity"
            assert rec["notes"] == "Subjects were hugging, not fighting."

    asyncio.run(scenario())


def test_review_reclassify(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_path)

    async def scenario():
        async with app.router.lifespan_context(app):
            evt = make_candidate("cam1", CandidateType.TRIPWIRE_CROSSING, BASE_TIME)
            await request(app, "POST", "/api/v1/events", evt)

            # Reclassify to UNAUTHORIZED_ZONE_ENTRY with HIGH severity
            review_payload = {
                "action": "reclassify",
                "reviewed_by": "security_lead",
                "reclassified_type": "unauthorized_zone_entry",
                "final_severity": "HIGH",
                "notes": "Subject actually crossed into restricted zone perimeter.",
            }
            status, reviewed = await request(app, "POST", "/api/v1/incidents/INC-0001/review", review_payload)
            assert status == 200
            assert reviewed["status"] == "reclassified"
            assert reviewed["highest_severity"] == "HIGH"
            rec = reviewed["reviews"][0]
            assert rec["action"] == "reclassify"
            assert rec["reclassified_type"] == "unauthorized_zone_entry"
            assert rec["final_severity"] == "HIGH"

    asyncio.run(scenario())


def test_review_nonexistent_incident(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_path)

    async def scenario():
        async with app.router.lifespan_context(app):
            review_payload = {
                "action": "acknowledge",
                "reviewed_by": "operator",
            }
            status, err = await request(app, "POST", "/api/v1/incidents/INC-9999/review", review_payload)
            assert status == 404
            assert err["detail"] == "Incident not found"

    asyncio.run(scenario())


def test_multiple_sequential_reviews_audit_trail(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_path)

    async def scenario():
        async with app.router.lifespan_context(app):
            evt = make_candidate("cam1", CandidateType.POSSIBLE_PERSON_VIOLENCE, BASE_TIME)
            await request(app, "POST", "/api/v1/events", evt)

            # 1. Acknowledge
            await request(app, "POST", "/api/v1/incidents/INC-0001/review", {
                "action": "acknowledge", "reviewed_by": "junior_operator",
            })
            # 2. Confirm
            await request(app, "POST", "/api/v1/incidents/INC-0001/review", {
                "action": "confirm", "reviewed_by": "senior_operator",
                "final_severity": "HIGH", "notes": "Confirmed assault.",
            })

            status, inc = await request(app, "GET", "/api/v1/incidents/INC-0001")
            assert status == 200
            assert inc["status"] == "confirmed"
            assert len(inc["reviews"]) == 2
            assert inc["reviews"][0]["action"] == "acknowledge"
            assert inc["reviews"][0]["reviewed_by"] == "junior_operator"
            assert inc["reviews"][1]["action"] == "confirm"
            assert inc["reviews"][1]["reviewed_by"] == "senior_operator"

    asyncio.run(scenario())
