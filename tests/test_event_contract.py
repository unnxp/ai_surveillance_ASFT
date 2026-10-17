"""Synthetic contract tests for gateway-to-central metadata."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError


@pytest.fixture
def schemas():
    from common import schemas

    return schemas


@pytest.fixture
def envelope():
    return {
        "schema_version": "1.0",
        "camera_id": " cam1 ",
        "gateway_id": " gateway1 ",
        "gateway_boot_id": " boot1 ",
        "sequence_number": 0,
        "observed_at_utc": "2026-09-27T10:00:00+07:00",
        "model_versions": {" detector ": " v1 "},
        "configuration_version": " config1 ",
    }


@pytest.fixture
def candidate(envelope):
    return dict(envelope, event_id="11111111-1111-4111-8111-111111111111",
                candidate_type="tripwire_crossing", confidence=0.75,
                reason_codes=[" crossing "], evidence=[], attributes={"zone": "a"})


def test_candidate_serializes_normalized_json(schemas, candidate):
    event = schemas.EventCandidate(**candidate)
    data = json.loads(event.model_dump_json())
    assert data == {
        "schema_version": "1.0", "camera_id": "cam1", "gateway_id": "gateway1",
        "gateway_boot_id": "boot1", "sequence_number": 0,
        "observed_at_utc": "2026-09-27T03:00:00Z", "model_versions": {"detector": "v1"},
        "configuration_version": "config1", "event_id": "11111111-1111-4111-8111-111111111111",
        "candidate_type": "tripwire_crossing", "local_track_id": None,
        "global_person_id": None, "confidence": 0.75, "reason_codes": ["crossing"],
        "evidence": [], "attributes": {"zone": "a"},
    }
    assert event.observed_at_utc == datetime(2026, 9, 27, 3, tzinfo=timezone.utc)
    assert event.observed_at_utc.tzinfo == timezone.utc


@pytest.mark.parametrize("field,value", [
    ("schema_version", "2.0"), ("sequence_number", -1),
    ("confidence", -0.01), ("confidence", 1.01), ("confidence", float("nan")),
    ("camera_id", " "), ("gateway_id", ""), ("gateway_boot_id", "\t"),
    ("configuration_version", " "), ("model_versions", {" ": "v1"}),
    ("model_versions", {"detector": " "}), ("reason_codes", [" "]),
    ("local_track_id", " "), ("global_person_id", " "),
    ("observed_at_utc", "2026-09-27T10:00:00"),
    ("event_id", "bad-uuid"), ("candidate_type", "confirmed_crime"),
    ("unexpected", True),
])
def test_candidate_rejects_invalid_contract_fields(schemas, candidate, field, value):
    candidate[field] = value
    with pytest.raises(ValidationError):
        schemas.EventCandidate(**candidate)


@pytest.mark.parametrize("kind", [
    "possible_person_violence", "possible_property_crime", "unauthorized_zone_entry",
    "unknown_person_in_restricted_zone", "tripwire_crossing", "possible_loitering",
    "cross_camera_handoff",
])
def test_candidate_accepts_documented_types(schemas, candidate, kind):
    candidate["candidate_type"] = kind
    assert schemas.EventCandidate(**candidate).model_dump(mode="json")["candidate_type"] == kind


@pytest.mark.parametrize("confidence", [0.0, 1.0, None])
def test_confidence_accepts_boundaries_and_absence(schemas, candidate, confidence):
    candidate["confidence"] = confidence
    assert schemas.EventCandidate(**candidate).confidence == confidence


def test_draft_has_no_envelope_and_can_be_enriched(schemas, envelope):
    draft = schemas.CandidateDraft(candidate_type="possible_loitering", local_track_id=7)
    assert draft.model_dump(mode="json") == {
        "candidate_type": "possible_loitering",
        "local_track_id": 7, "global_person_id": None, "confidence": None,
        "reason_codes": [], "evidence": [], "attributes": {},
    }
    enriched = schemas.EventCandidate(**envelope, **draft.model_dump(),
                                      event_id="11111111-1111-4111-8111-111111111111")
    assert enriched.local_track_id == 7
    assert str(enriched.event_id) == "11111111-1111-4111-8111-111111111111"
    with pytest.raises(ValidationError):
        schemas.CandidateDraft(**draft.model_dump(), event_id="11111111-1111-4111-8111-111111111111")
    with pytest.raises(ValidationError):
        schemas.CandidateDraft(**draft.model_dump(), camera_id="cam1")


@pytest.fixture
def evidence():
    return {"evidence_id": "22222222-2222-4222-8222-222222222222", "kind": "keyframe",
            "relative_path": " evidence/frame.jpg ", "sha256": "a" * 64,
            "size_bytes": 0, "captured_at_utc": "2026-09-27T10:00:00+07:00"}


@pytest.mark.parametrize("kind", ["keyframe", "person_crop", "clip"])
def test_evidence_serializes_utc_metadata(schemas, evidence, kind):
    evidence["kind"] = kind
    reference = schemas.EvidenceReference(**evidence)
    data = json.loads(reference.model_dump_json())
    assert data["kind"] == kind
    assert data["captured_at_utc"] == "2026-09-27T03:00:00Z"
    assert data["relative_path"] == "evidence/frame.jpg"
    assert data["size_bytes"] == 0
    assert data["sha256"] == "a" * 64


@pytest.mark.parametrize("field,value", [
    ("sha256", "A" * 64), ("sha256", "g" * 64), ("sha256", "a" * 63),
    ("sha256", "a" * 65), ("sha256", "junk" + "a" * 64),
    ("sha256", "a" * 64 + "junk"),
    ("size_bytes", -1), ("kind", "audio"), ("relative_path", " "),
    ("captured_at_utc", "2026-09-27T10:00:00"), ("unexpected", True),
])
def test_evidence_rejects_invalid_metadata(schemas, evidence, field, value):
    evidence[field] = value
    with pytest.raises(ValidationError):
        schemas.EvidenceReference(**evidence)


def test_candidate_validates_nested_evidence(schemas, candidate, evidence):
    candidate["evidence"] = [evidence]
    assert json.loads(schemas.EventCandidate(**candidate).model_dump_json())["evidence"][0]["kind"] == "keyframe"
    evidence["sha256"] = "invalid"
    with pytest.raises(ValidationError):
        schemas.EventCandidate(**candidate)


@pytest.mark.parametrize("status", ["ONLINE", "DEGRADED", "OFFLINE", "ERROR"])
def test_heartbeat_serializes_status_and_metrics(schemas, envelope, status):
    heartbeat = schemas.CameraHeartbeat(**envelope, status=status, capture_fps=0,
                                       ai_fps=2.5, p95_latency_ms=10, dropped_frames=0,
                                       active_tracks=2, action_backlog=0)
    data = json.loads(heartbeat.model_dump_json())
    assert data["status"] == status
    assert data["observed_at_utc"] == "2026-09-27T03:00:00Z"
    assert data["capture_fps"] == 0
    assert data["ai_fps"] == 2.5
    assert data["p95_latency_ms"] == 10
    assert data["dropped_frames"] == 0
    assert data["active_tracks"] == 2
    assert data["action_backlog"] == 0


@pytest.mark.parametrize("field,value", [
    ("capture_fps", -1), ("ai_fps", -1), ("p95_latency_ms", -1),
    ("dropped_frames", -1), ("active_tracks", -1), ("action_backlog", -1),
    ("status", "online"), ("unexpected", True),
])
def test_heartbeat_rejects_invalid_status_or_metrics(schemas, envelope, field, value):
    payload = dict(envelope, status="ONLINE", dropped_frames=0, active_tracks=0, action_backlog=0)
    payload[field] = value
    with pytest.raises(ValidationError):
        schemas.CameraHeartbeat(**payload)


def test_acknowledgements_serialize_and_reject_extra_fields(schemas):
    ack = schemas.EventIngestAck(event_id="11111111-1111-4111-8111-111111111111", accepted=True, duplicate=False)
    assert json.loads(ack.model_dump_json()) == {
        "event_id": "11111111-1111-4111-8111-111111111111", "accepted": True, "duplicate": False}
    assert json.loads(schemas.HeartbeatAck(camera_id=" cam1 ", accepted=True).model_dump_json()) == {
        "camera_id": "cam1", "accepted": True}
    with pytest.raises(ValidationError):
        schemas.EventIngestAck(**ack.model_dump(), unexpected=True)
    with pytest.raises(ValidationError):
        schemas.HeartbeatAck(camera_id=" ", accepted=True)
    with pytest.raises(ValidationError):
        schemas.HeartbeatAck(camera_id="cam1", accepted=True, unexpected=True)


@pytest.mark.parametrize("field", ["capture_fps", "ai_fps", "p95_latency_ms"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_heartbeat_rejects_nonfinite_metrics(schemas, envelope, field, value):
    payload = dict(envelope, status="ONLINE", dropped_frames=0, active_tracks=0, action_backlog=0)
    payload[field] = value
    with pytest.raises(ValidationError):
        schemas.CameraHeartbeat(**payload)


@pytest.mark.parametrize("model_name,field", [
    ("EventCandidate", "schema_version"), ("EventCandidate", "camera_id"),
    ("EventCandidate", "gateway_id"), ("EventCandidate", "gateway_boot_id"),
    ("EventCandidate", "sequence_number"), ("EventCandidate", "observed_at_utc"),
    ("EventCandidate", "model_versions"), ("EventCandidate", "configuration_version"),
    ("EventCandidate", "event_id"), ("EventCandidate", "candidate_type"),
    ("CameraHeartbeat", "schema_version"), ("CameraHeartbeat", "camera_id"),
    ("CameraHeartbeat", "gateway_id"), ("CameraHeartbeat", "gateway_boot_id"),
    ("CameraHeartbeat", "sequence_number"), ("CameraHeartbeat", "observed_at_utc"),
    ("CameraHeartbeat", "model_versions"), ("CameraHeartbeat", "configuration_version"),
    ("CameraHeartbeat", "status"), ("CameraHeartbeat", "dropped_frames"),
    ("CameraHeartbeat", "active_tracks"), ("CameraHeartbeat", "action_backlog"),
    ("EvidenceReference", "evidence_id"), ("EvidenceReference", "kind"),
    ("EvidenceReference", "captured_at_utc"),
    ("EventIngestAck", "event_id"), ("EventIngestAck", "accepted"),
    ("EventIngestAck", "duplicate"),
    ("HeartbeatAck", "camera_id"), ("HeartbeatAck", "accepted"),
])
def test_contract_rejects_omitted_required_field(
    schemas, envelope, candidate, evidence, model_name, field
):
    payloads = {
        "EventCandidate": candidate,
        "CameraHeartbeat": dict(envelope, status="ONLINE", dropped_frames=0,
                                active_tracks=0, action_backlog=0),
        "EvidenceReference": evidence,
        "EventIngestAck": {"event_id": "11111111-1111-4111-8111-111111111111",
                           "accepted": True, "duplicate": False},
        "HeartbeatAck": {"camera_id": "cam1", "accepted": True},
    }
    payload = payloads[model_name].copy()
    del payload[field]
    with pytest.raises(ValidationError) as error:
        getattr(schemas, model_name)(**payload)
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize("model_name", ["CandidateDraft", "EventCandidate"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object()])
def test_attributes_reject_nested_non_json_values(schemas, candidate, model_name, value):
    payload = candidate if model_name == "EventCandidate" else {"candidate_type": "tripwire_crossing"}
    payload = dict(payload, attributes={"outer": [{"inner": [None, {"value": value}]}]})
    with pytest.raises(ValidationError):
        getattr(schemas, model_name)(**payload)


@pytest.mark.parametrize("model_name", ["CandidateDraft", "EventCandidate"])
def test_attributes_accept_nested_json_scalars(schemas, candidate, model_name):
    attributes = {"outer": [None, True, False, 7, -2, 0.5, "text", {"inner": []}]}
    payload = candidate if model_name == "EventCandidate" else {"candidate_type": "tripwire_crossing"}
    result = getattr(schemas, model_name)(**dict(payload, attributes=attributes))
    assert json.loads(result.model_dump_json())["attributes"] == {
        "outer": [None, True, False, 7, -2, 0.5, "text", {"inner": []}]}
    assert attributes == {"outer": [None, True, False, 7, -2, 0.5, "text", {"inner": []}]}
