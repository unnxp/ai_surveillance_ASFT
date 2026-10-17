"""Consent-backed registry tests use invented identities and synthetic unit vectors only."""

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError


@pytest.fixture
def schemas():
    from common import identity_schemas

    return identity_schemas


def synthetic_vector(index=0):
    vector = [0.0] * 512
    vector[index] = 1.0
    return vector


def enrollment(**overrides):
    values = {
        "display_name": "พนักงานตัวอย่าง",
        "role": "staff",
        "consent_confirmed": True,
        "allowed_zones": [{"camera_id": "camera_01", "zone_id": "storage"}],
        "templates": [{"model_version": "synthetic-v1", "vector": synthetic_vector()}],
    }
    values.update(overrides)
    return values


def test_registry_contract_accepts_synthetic_template_and_thai_name(schemas):
    profile = schemas.FaceRegistryUpsert.model_validate(enrollment())
    assert profile.display_name == "พนักงานตัวอย่าง"
    assert profile.allowed_zones[0].camera_id == "camera_01"
    assert profile.templates[0].vector[0] == 1.0


@pytest.mark.parametrize("change", [
    {"consent_confirmed": False}, {"consent_confirmed": None},
    {"consent_confirmed": 1}, {"consent_confirmed": "true"},
])
def test_registry_contract_requires_explicit_consent(schemas, change):
    values = enrollment()
    values.update(change)
    if values["consent_confirmed"] is None:
        del values["consent_confirmed"]
    with pytest.raises(ValidationError):
        schemas.FaceRegistryUpsert.model_validate(values)


@pytest.mark.parametrize("vector", [
    [1.0] + [0.0] * 510,
    [1.0] + [0.0] * 512,
    [float("nan")] + [0.0] * 511,
    [float("inf")] + [0.0] * 511,
    [0.5] + [0.0] * 511,
])
def test_registry_contract_rejects_invalid_vectors(schemas, vector):
    with pytest.raises(ValidationError):
        schemas.FaceTemplateInput.model_validate(
            {"model_version": "synthetic-v1", "vector": vector}
        )


def test_registry_contract_rejects_duplicate_camera_zone_pair(schemas):
    values = enrollment(allowed_zones=[
        {"camera_id": "camera_01", "zone_id": "storage"},
        {"camera_id": "camera_01", "zone_id": "storage"},
    ])
    with pytest.raises(ValidationError):
        schemas.FaceRegistryUpsert.model_validate(values)


def test_registry_contract_rejects_unknown_field(schemas):
    with pytest.raises(ValidationError):
        schemas.FaceRegistryUpsert.model_validate(enrollment(unexpected=True))


@pytest.mark.parametrize("person_id", ["", "a/b", "space here", "x" * 65])
def test_registry_summary_rejects_invalid_person_id(schemas, person_id):
    with pytest.raises(ValidationError):
        schemas.FaceRegistrySummary.model_validate({
            "person_id": person_id, "display_name": "ตัวอย่าง", "role": "staff",
            "allowed_zones": [], "template_count": 1, "model_versions": ["synthetic-v1"],
            "consent_status": "active", "consent_recorded_at_utc": datetime.now(timezone.utc),
            "enrolled_at_utc": datetime.now(timezone.utc),
        })


def test_registry_summary_contains_metadata_but_never_vector(schemas):
    summary = schemas.FaceRegistrySummary.model_validate({
        "person_id": "employee_01", "display_name": "พนักงานตัวอย่าง", "role": "staff",
        "allowed_zones": [{"camera_id": "camera_01", "zone_id": "storage"}],
        "template_count": 1, "model_versions": ["synthetic-v1"],
        "consent_status": "active", "consent_recorded_at_utc": datetime.now(timezone.utc),
        "enrolled_at_utc": datetime.now(timezone.utc),
    })
    body = summary.model_dump_json()
    assert "พนักงานตัวอย่าง" in body
    assert "vector" not in body
    assert "synthetic-v1" in body


def test_registry_database_is_lazy_and_persists_metadata(tmp_path, schemas):
    from server.database import Database

    path = tmp_path / "nested" / "registry.db"
    database = Database(path)
    assert not path.parent.exists()
    database.initialize()
    with closing(sqlite3.connect(path)) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert {"events", "camera_status", "registered_identities", "face_templates",
            "zone_grants"} <= tables

    created, profile = database.upsert_identity(
        "employee_02", schemas.FaceRegistryUpsert.model_validate(enrollment())
    )
    assert created is True
    assert profile.person_id == "employee_02"
    assert profile.consent_status == "active"
    assert profile.template_count == 1
    assert "vector" not in profile.model_dump_json()

    other = schemas.FaceRegistryUpsert.model_validate(enrollment(display_name="อีกคน"))
    database.upsert_identity("employee_01", other)
    reopened = Database(path)
    assert [item.person_id for item in reopened.list_identities()] == [
        "employee_01", "employee_02"
    ]
    assert reopened.list_identities()[1].model_versions == ["synthetic-v1"]


def test_registry_access_grant_is_scoped_to_camera_and_zone(tmp_path, schemas):
    from server.database import Database

    database = Database(tmp_path / "registry.db")
    database.initialize()
    profile = schemas.FaceRegistryUpsert.model_validate(enrollment())
    database.upsert_identity("employee_01", profile)
    assert database.check_access("employee_01", "camera_01", "storage") == "authorized"
    assert database.check_access("employee_01", "camera_02", "storage") == "unauthorized"
    assert database.check_access("employee_01", "camera_01", "office") == "unauthorized"
    assert database.check_access("missing", "camera_01", "storage") is None


def test_registry_replacement_removes_old_grants_and_templates(tmp_path, schemas):
    from server.database import Database

    path = tmp_path / "registry.db"
    database = Database(path)
    database.initialize()
    original = schemas.FaceRegistryUpsert.model_validate(enrollment())
    _, before = database.upsert_identity("employee_01", original)
    updated = schemas.FaceRegistryUpsert.model_validate(enrollment(
        display_name="ชื่อใหม่",
        allowed_zones=[{"camera_id": "camera_02", "zone_id": "storage"}],
        templates=[{"model_version": "synthetic-v2", "vector": synthetic_vector(1)}],
    ))
    created, after = database.upsert_identity("employee_01", updated)
    assert created is False
    assert after.enrolled_at_utc == before.enrolled_at_utc
    assert after.display_name == "ชื่อใหม่"
    assert after.model_versions == ["synthetic-v2"]
    assert database.check_access("employee_01", "camera_01", "storage") == "unauthorized"
    assert database.check_access("employee_01", "camera_02", "storage") == "authorized"
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT count(*) FROM face_templates").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM zone_grants").fetchone()[0] == 1


def test_registry_withdrawal_deletes_profile_templates_and_grants(tmp_path, schemas):
    from server.database import Database

    path = tmp_path / "registry.db"
    database = Database(path)
    database.initialize()
    database.upsert_identity("employee_01", schemas.FaceRegistryUpsert.model_validate(enrollment()))
    assert database.delete_identity("employee_01") is True
    assert database.delete_identity("employee_01") is False
    assert database.list_identities() == []
    assert database.check_access("employee_01", "camera_01", "storage") is None
    with closing(sqlite3.connect(path)) as connection:
        for table in ("registered_identities", "face_templates", "zone_grants"):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_registry_api_creates_replaces_lists_and_revokes(tmp_path):
    from server.app import create_app
    from tests.test_central_api import json_request, request

    path = tmp_path / "registry.db"

    async def scenario():
        first = create_app(path)
        async with first.router.lifespan_context(first):
            status, created = await json_request(
                first, "PUT", "/api/v1/face-registry/employee_01", enrollment()
            )
            assert status == 201
            assert created["person_id"] == "employee_01"
            assert created["display_name"] == "พนักงานตัวอย่าง"
            assert created["template_count"] == 1
            assert created["consent_status"] == "active"
            assert "vector" not in str(created)

            replacement = enrollment(
                role="supervisor", allowed_zones=[],
                templates=[{"model_version": "synthetic-v2", "vector": synthetic_vector(1)}],
            )
            status, updated = await json_request(
                first, "PUT", "/api/v1/face-registry/employee_01", replacement
            )
            assert status == 200
            assert updated["role"] == "supervisor"
            assert updated["model_versions"] == ["synthetic-v2"]
            assert updated["allowed_zones"] == []
            assert updated["enrolled_at_utc"] == created["enrolled_at_utc"]
            _, listing = await json_request(first, "GET", "/api/v1/face-registry")
            assert [item["person_id"] for item in listing["identities"]] == ["employee_01"]
            assert "vector" not in str(listing)

        reopened = create_app(path)
        async with reopened.router.lifespan_context(reopened):
            _, listing = await json_request(reopened, "GET", "/api/v1/face-registry")
            assert listing["identities"][0]["role"] == "supervisor"
            status, _, body = await request(
                reopened, "DELETE", "/api/v1/face-registry/employee_01"
            )
            assert status == 204 and body == b""
            assert (await request(reopened, "DELETE", "/api/v1/face-registry/employee_01"))[0] == 404
            assert (await json_request(reopened, "GET", "/api/v1/face-registry"))[1] == {
                "identities": []
            }

    asyncio.run(scenario())


def test_registry_api_rejects_bad_input_without_echo_or_write(tmp_path):
    from server.app import create_app
    from tests.test_central_api import json_request, request

    path = tmp_path / "registry.db"
    app = create_app(path)
    private_name = "Private Example Name"
    marker = 0.123456789

    async def scenario():
        async with app.router.lifespan_context(app):
            payloads = [
                enrollment(display_name=private_name, consent_confirmed=False),
                enrollment(display_name=private_name, templates=[{
                    "model_version": "synthetic-v1",
                    "vector": [marker] + [0.0] * 511,
                }]),
            ]
            for payload in payloads:
                status, _, body = await request(
                    app, "PUT", "/api/v1/face-registry/employee_01", payload
                )
                assert status == 422
                assert private_name.encode() not in body
                assert str(marker).encode() not in body
            status, _, body = await request(
                app, "PUT", "/api/v1/face-registry/bad!id", enrollment()
            )
            assert status == 422 and b"vector" not in body
            status, _, body = await request(
                app, "PUT", "/api/v1/face-registry/employee_01", raw_body=b"{invalid"
            )
            assert status == 422 and b"invalid" not in body
            assert (await json_request(app, "GET", "/api/v1/face-registry"))[1] == {
                "identities": []
            }

    asyncio.run(scenario())


def test_registry_routes_preserve_event_and_camera_api(tmp_path):
    from server.app import create_app
    from tests.test_central_api import candidate, heartbeat, json_request

    app = create_app(tmp_path / "registry.db")

    async def scenario():
        async with app.router.lifespan_context(app):
            assert (await json_request(app, "POST", "/api/v1/events", candidate()))[0] == 201
            assert (await json_request(app, "POST", "/api/v1/cameras/heartbeat",
                                       heartbeat()))[0] == 200
            assert len((await json_request(app, "GET", "/api/v1/events"))[1]["events"]) == 1
            assert len((await json_request(app, "GET", "/api/v1/cameras"))[1]["cameras"]) == 1

    asyncio.run(scenario())
