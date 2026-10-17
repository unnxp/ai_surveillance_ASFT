"""Synthetic face vectors only; no model, photo, or biometric fixture is used."""

import asyncio

import pytest
from pydantic import ValidationError


def unit_vector(index=0):
    values = [0.0] * 512
    values[index] = 1.0
    return values


def query(**changes):
    values = {
        "camera_id": "camera_01", "zone_ids": ["storage", "office"],
        "model_version": "synthetic-v1", "vector": unit_vector(),
    }
    values.update(changes)
    return values


def test_match_request_rejects_duplicate_zones_and_non_unit_vector():
    from common.identity_schemas import FaceMatchRequest

    assert FaceMatchRequest.model_validate(query()).zone_ids == ["storage", "office"]
    with pytest.raises(ValidationError):
        FaceMatchRequest.model_validate(query(zone_ids=["storage", "storage"]))
    with pytest.raises(ValidationError):
        FaceMatchRequest.model_validate(query(vector=[0.1] * 512))


def test_match_result_contains_identity_context_but_no_vector():
    from common.identity_schemas import FaceMatchResult

    result = FaceMatchResult.model_validate({
        "status": "registered", "person_id": "employee_01",
        "display_name": "ตัวอย่าง", "similarity": 0.8,
        "zone_access": {"storage": "authorized", "office": "unauthorized"},
        "model_version": "synthetic-v1",
    })
    assert result.zone_access["office"] == "unauthorized"
    assert "vector" not in result.model_dump_json()


def test_match_result_rejects_identity_fields_for_unknown():
    from common.identity_schemas import FaceMatchResult

    with pytest.raises(ValidationError):
        FaceMatchResult.model_validate({
            "status": "unknown", "person_id": "employee_01", "display_name": "ตัวอย่าง",
            "similarity": 0.2, "zone_access": {}, "model_version": "synthetic-v1",
        })


def test_client_sends_match_request_and_validates_response():
    from clients.central_client import CentralClient
    from common.identity_schemas import FaceMatchRequest

    class Response:
        status_code = 200

        def json(self):
            return {"status": "unknown", "person_id": None, "display_name": None,
                    "similarity": 0.1, "zone_access": {}, "model_version": "synthetic-v1"}

    class Transport:
        def post(self, path, payload):
            assert path == "/api/v1/face-registry/match"
            assert payload == query()
            return Response()

    result = CentralClient(Transport()).match_face(FaceMatchRequest.model_validate(query()))
    assert result.status == "unknown"


def test_client_sanitizes_transport_failure_without_echoing_vector():
    from clients.central_client import CentralClient, CentralProtocolError
    from common.identity_schemas import FaceMatchRequest

    class Transport:
        def post(self, path, payload):
            raise RuntimeError("private-vector-value")

    with pytest.raises(CentralProtocolError) as error:
        CentralClient(Transport()).match_face(FaceMatchRequest.model_validate(query()))
    assert "private-vector-value" not in str(error.value)


def registered(name, grants, vector=None):
    return {
        "display_name": name, "role": "staff", "consent_confirmed": True,
        "allowed_zones": grants,
        "templates": [{"model_version": "synthetic-v1", "vector": vector or unit_vector()}],
    }


def test_database_matches_only_same_model_version_and_scopes_zone_grants(tmp_path):
    from common.identity_schemas import FaceMatchRequest, FaceRegistryUpsert
    from server.database import Database

    database = Database(tmp_path / "face.db")
    database.initialize()
    database.upsert_identity("employee_01", FaceRegistryUpsert.model_validate(registered(
        "พนักงาน", [{"camera_id": "camera_01", "zone_id": "storage"}])))
    result = database.match_face(FaceMatchRequest.model_validate(query()))
    assert result.status == "registered"
    assert result.person_id == "employee_01"
    assert result.display_name == "พนักงาน"
    assert result.zone_access == {"storage": "authorized", "office": "unauthorized"}
    assert database.match_face(FaceMatchRequest.model_validate(query(
        camera_id="camera_02", zone_ids=["storage"]))).zone_access == {
            "storage": "unauthorized"
        }
    assert database.match_face(FaceMatchRequest.model_validate(query(
        vector=unit_vector(1)))).status == "unknown"
    assert database.match_face(FaceMatchRequest.model_validate(query(
        model_version="different-v1"))).status == "unavailable"


def test_database_empty_gallery_abstains_and_ties_are_deterministic(tmp_path):
    from common.identity_schemas import FaceMatchRequest, FaceRegistryUpsert
    from server.database import Database

    database = Database(tmp_path / "face.db")
    database.initialize()
    request = FaceMatchRequest.model_validate(query())
    assert database.match_face(request).status == "unavailable"
    for person_id in ("employee_b", "employee_a"):
        database.upsert_identity(person_id, FaceRegistryUpsert.model_validate(registered(
            person_id, [], unit_vector())))
    assert database.match_face(request).person_id == "employee_a"


def test_database_cosine_normalizes_near_unit_vectors_before_scoring(tmp_path):
    from common.identity_schemas import FaceMatchRequest, FaceRegistryUpsert
    from server.database import Database

    database = Database(tmp_path / "face.db")
    database.initialize()
    near_unit = [1.005] + [0.0] * 511
    database.upsert_identity("employee_01", FaceRegistryUpsert.model_validate(registered(
        "Example", [], near_unit)))
    result = database.match_face(FaceMatchRequest.model_validate(query(vector=near_unit)))
    assert result.status == "registered"
    assert result.similarity == pytest.approx(1.0)


def test_match_api_returns_identity_without_query_vector_and_sanitizes_errors(tmp_path):
    from server.app import create_app
    from tests.test_central_api import json_request, request

    async def scenario():
        app = create_app(tmp_path / "face.db")
        async with app.router.lifespan_context(app):
            status, _ = await json_request(
                app, "PUT", "/api/v1/face-registry/employee_01",
                registered("พนักงาน", [{"camera_id": "camera_01", "zone_id": "storage"}]),
            )
            assert status == 201
            status, result = await json_request(
                app, "POST", "/api/v1/face-registry/match", query()
            )
            assert status == 200
            assert result["person_id"] == "employee_01"
            assert result["zone_access"] == {
                "storage": "authorized", "office": "unauthorized"
            }
            assert "vector" not in result
            status, _, body = await request(
                app, "POST", "/api/v1/face-registry/match",
                query(vector=[1.0] + [0.0] * 510),
            )
            assert status == 422
            assert b"vector" not in body
            assert b"1.0" not in body

    asyncio.run(scenario())
