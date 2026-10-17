"""Image evidence API tests use synthetic pixels and temporary local storage."""

import hashlib
import sqlite3

import cv2
import numpy as np
import pytest
import requests

from tests.test_week1b_http import local_server

EVENT_ID = "00000000-0000-4000-8000-000000000101"
EVIDENCE_ID = "00000000-0000-4000-8000-000000000201"
CAPTURED_AT = "2026-09-28T10:00:00Z"


def jpeg_bytes() -> bytes:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[5:30, 10:35] = (20, 80, 160)
    success, encoded = cv2.imencode(".jpg", frame)
    assert success
    return encoded.tobytes()


def event_payload(image: bytes) -> dict:
    return {
        "schema_version": "1.0", "camera_id": "camera_01", "gateway_id": "test-gateway",
        "gateway_boot_id": "boot", "sequence_number": 1,
        "observed_at_utc": CAPTURED_AT, "model_versions": {"detector": "fixture"},
        "configuration_version": "test", "event_id": EVENT_ID,
        "candidate_type": "tripwire_crossing", "local_track_id": 7,
        "evidence": [{"evidence_id": EVIDENCE_ID, "kind": "keyframe",
                      "sha256": hashlib.sha256(image).hexdigest(),
                      "size_bytes": len(image), "captured_at_utc": CAPTURED_AT}],
    }


def test_declared_image_is_pending_then_uploaded_and_retrievable(tmp_path):
    image = jpeg_bytes()
    event = event_payload(image)
    database_path = tmp_path / "central.db"
    with local_server(database_path) as origin, requests.Session() as client:
        client.trust_env = False
        assert client.post(origin + "/api/v1/events", json=event, timeout=5).status_code == 201
        status_url = origin + f"/api/v1/events/{EVENT_ID}/evidence"
        pending = client.get(status_url, timeout=5)
        assert pending.status_code == 200
        assert pending.json()["evidence_incomplete"] is True
        assert pending.json()["evidence"] == [{
            "evidence_id": EVIDENCE_ID, "kind": "keyframe",
            "available": False, "url": None,
        }]
        assert client.get(origin + f"/api/v1/evidence/{EVIDENCE_ID}", timeout=5).status_code == 404

        upload_url = origin + f"/api/v1/events/{EVENT_ID}/evidence/{EVIDENCE_ID}"
        uploaded = client.put(upload_url, data=image,
                              headers={"Content-Type": "image/jpeg"}, timeout=5)
        assert uploaded.status_code == 201
        assert uploaded.json() == {"evidence_id": EVIDENCE_ID,
                                   "accepted": True, "duplicate": False}
        complete = client.get(status_url, timeout=5).json()
        assert complete["evidence_incomplete"] is False
        assert complete["evidence"] == [{
            "evidence_id": EVIDENCE_ID, "kind": "keyframe", "available": True,
            "url": f"/api/v1/evidence/{EVIDENCE_ID}",
        }]
        served = client.get(origin + complete["evidence"][0]["url"], timeout=5)
        assert served.status_code == 200 and served.content == image
        duplicate = client.put(upload_url, data=image,
                               headers={"Content-Type": "image/jpeg"}, timeout=5)
        assert duplicate.status_code == 200
        assert duplicate.json()["duplicate"] is True
        replay = client.post(origin + "/api/v1/events", json=event, timeout=5)
        assert replay.status_code == 200 and replay.json()["duplicate"] is True

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT count(*) FROM evidence").fetchone()[0] == 1
        assert image not in database_path.read_bytes()
    assert len(list((tmp_path / "evidence").rglob("*.jpg"))) == 1


def test_retry_restores_a_missing_file_without_changing_event(tmp_path):
    image = jpeg_bytes()
    database_path = tmp_path / "central.db"
    with local_server(database_path) as origin, requests.Session() as client:
        client.trust_env = False
        event = event_payload(image)
        assert client.post(origin + "/api/v1/events", json=event, timeout=5).status_code == 201
        url = origin + f"/api/v1/events/{EVENT_ID}/evidence/{EVIDENCE_ID}"
        assert client.put(url, data=image, headers={"Content-Type": "image/jpeg"},
                          timeout=5).status_code == 201
        path = next((tmp_path / "evidence").rglob("*.jpg"))
        path.unlink()
        assert client.get(origin + f"/api/v1/events/{EVENT_ID}/evidence",
                          timeout=5).json()["evidence_incomplete"] is True
        repaired = client.put(url, data=image, headers={"Content-Type": "image/jpeg"},
                              timeout=5)
        assert repaired.status_code == 200
        assert path.read_bytes() == image
        assert client.get(origin + f"/api/v1/events/{EVENT_ID}/evidence",
                          timeout=5).json()["evidence_incomplete"] is False


@pytest.mark.parametrize("body,content_type,expected", [
    (b"bad", "image/jpeg", 422),
    (b"x" * 2_000_001, "image/jpeg", 413),
    (b"bad", "application/octet-stream", 422),
], ids=["malformed", "oversize", "wrong-content-type"])
def test_invalid_upload_never_creates_evidence_file(tmp_path, body, content_type, expected):
    image = jpeg_bytes()
    with local_server(tmp_path / "central.db") as origin, requests.Session() as client:
        client.trust_env = False
        assert client.post(origin + "/api/v1/events", json=event_payload(image),
                           timeout=5).status_code == 201
        response = client.put(origin + f"/api/v1/events/{EVENT_ID}/evidence/{EVIDENCE_ID}",
                              data=body, headers={"Content-Type": content_type}, timeout=5)
        assert response.status_code == expected
        assert not (tmp_path / "evidence").exists() or not list(
            (tmp_path / "evidence").rglob("*.jpg")
        )


def test_unknown_reference_and_conflicting_retry_do_not_overwrite(tmp_path):
    image = jpeg_bytes()
    with local_server(tmp_path / "central.db") as origin, requests.Session() as client:
        client.trust_env = False
        event = event_payload(image)
        event["evidence"][0]["relative_path"] = "../../outside.jpg"
        assert client.post(origin + "/api/v1/events", json=event, timeout=5).status_code == 201
        missing = client.put(origin + f"/api/v1/events/{EVENT_ID}/evidence/"
                             "00000000-0000-4000-8000-000000000999", data=image,
                             headers={"Content-Type": "image/jpeg"}, timeout=5)
        assert missing.status_code == 404
        url = origin + f"/api/v1/events/{EVENT_ID}/evidence/{EVIDENCE_ID}"
        assert client.put(url, data=image, headers={"Content-Type": "image/jpeg"},
                          timeout=5).status_code == 201
        conflict = client.put(url, data=image + b"x", headers={"Content-Type": "image/jpeg"},
                              timeout=5)
        assert conflict.status_code == 409
        assert client.get(origin + f"/api/v1/evidence/{EVIDENCE_ID}",
                          timeout=5).content == image
        assert not (tmp_path / "outside.jpg").exists()


def test_dashboard_offers_evidence_view_without_external_alert(tmp_path):
    with local_server(tmp_path / "central.db") as origin, requests.Session() as client:
        client.trust_env = False
        page = client.get(origin + "/", timeout=5)
        assert page.status_code == 200
        assert 'Evidence' in page.text
        assert 'id="evidence-viewer"' in page.text
        assert '/api/v1/events/' in page.text
        assert '/api/v1/evidence/' in page.text
        assert 'external alert' in page.text


def test_duplicate_reference_ids_are_rejected_before_event_ingest(tmp_path):
    image = jpeg_bytes()
    event = event_payload(image)
    crop = dict(event["evidence"][0], kind="person_crop")
    event["evidence"].append(crop)
    with local_server(tmp_path / "central.db") as origin, requests.Session() as client:
        client.trust_env = False
        assert client.post(origin + "/api/v1/events", json=event,
                           timeout=5).status_code == 422
        assert client.get(origin + "/api/v1/events", timeout=5).json()["events"] == []
