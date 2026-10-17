"""Central contracts exercised through ASGI and synthetic, temporary SQLite data."""

import asyncio
import json
import sqlite3
from contextlib import closing
from urllib.parse import urlsplit

import pytest


def envelope(camera_id="cam1"):
    return {
        "schema_version": "1.0", "camera_id": camera_id, "gateway_id": "synthetic",
        "gateway_boot_id": "boot1", "sequence_number": 0,
        "observed_at_utc": "2026-09-27T10:00:00+07:00",
        "model_versions": {"detector": "synthetic-v1"}, "configuration_version": "v1",
    }


def candidate(number=1, camera_id="cam1"):
    return dict(envelope(camera_id), event_id=f"00000000-0000-4000-8000-{number:012d}",
                candidate_type="tripwire_crossing", confidence=0.75)


def heartbeat(camera_id="cam1", status="ONLINE"):
    return dict(envelope(camera_id), status=status, dropped_frames=0,
                active_tracks=1, action_backlog=0)


async def request(app, method, url, payload=None, *, raw_body=None):
    """Invoke the real ASGI app without HTTPX, sockets, or external services."""
    target = urlsplit(url)
    body = b"" if payload is None else json.dumps(payload).encode()
    if raw_body is not None:
        body = raw_body
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
    return start["status"], dict(start["headers"]), response_body


async def json_request(app, method, url, payload=None):
    status, _, body = await request(app, method, url, payload)
    return status, json.loads(body)


@pytest.fixture
def create_app():
    def factory(path):
        try:
            from server.app import create_app
        except ModuleNotFoundError as error:
            if error.name != "server.app":
                raise
            pytest.fail("Central ASGI application is not implemented yet")
        return create_app(path)

    return factory


def test_database_constructor_and_app_factory_do_not_create_files(tmp_path, create_app):
    path = tmp_path / "nested" / "events.db"
    create_app(path)
    from server.database import Database

    Database(path)
    assert not path.parent.exists()


def test_lifespan_initializes_tables_and_wal(tmp_path, create_app):
    path = tmp_path / "nested" / "events.db"
    app = create_app(path)

    async def scenario():
        async with app.router.lifespan_context(app):
            assert path.is_file()
            with closing(sqlite3.connect(path)) as connection:
                tables = {row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                assert {"events", "camera_status"} <= tables
                assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    asyncio.run(scenario())


def test_event_duplicate_and_conflict_preserve_original(tmp_path, create_app):
    path = tmp_path / "events.db"
    app = create_app(path)
    event = candidate()

    async def scenario():
        async with app.router.lifespan_context(app):
            status, ack = await json_request(app, "POST", "/api/v1/events", event)
            assert status == 201
            assert ack == {"event_id": event["event_id"], "accepted": True, "duplicate": False}
            # UTC normalization and explicit defaults must preserve idempotency.
            replay = dict(event, observed_at_utc="2026-09-27T03:00:00Z", reason_codes=[])
            status, ack = await json_request(app, "POST", "/api/v1/events", replay)
            assert status == 200
            assert ack == {"event_id": event["event_id"], "accepted": True, "duplicate": True}
            status, error = await json_request(app, "POST", "/api/v1/events",
                                               dict(event, confidence=0.5))
            assert status == 409
            assert isinstance(error["detail"], str)
            _, listing = await json_request(app, "GET", "/api/v1/events")
            assert len(listing["events"]) == 1
            assert listing["events"][0]["confidence"] == 0.75
            assert listing["events"][0]["observed_at_utc"] == "2026-09-27T03:00:00Z"
            with closing(sqlite3.connect(path)) as connection:
                assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("field,value", [
    ("event_id", "invalid"), ("camera_id", " "), ("schema_version", "2.0"),
    ("observed_at_utc", "2026-09-27T03:00:00"), ("unexpected", True),
])
def test_invalid_event_does_not_write(tmp_path, create_app, field, value):
    app = create_app(tmp_path / "events.db")
    event = candidate()
    event[field] = value

    async def scenario():
        async with app.router.lifespan_context(app):
            status, _ = await json_request(app, "POST", "/api/v1/events", event)
            assert status == 422
            _, listing = await json_request(app, "GET", "/api/v1/events")
            assert listing == {"events": [], "limit": 100, "offset": 0}

    asyncio.run(scenario())


def test_events_persist_across_app_instances(tmp_path, create_app):
    path = tmp_path / "events.db"

    async def scenario():
        first = create_app(path)
        async with first.router.lifespan_context(first):
            assert (await json_request(first, "POST", "/api/v1/events", candidate()))[0] == 201
        second = create_app(path)
        async with second.router.lifespan_context(second):
            status, listing = await json_request(second, "GET", "/api/v1/events")
            assert status == 200
            assert [item["event_id"] for item in listing["events"]] == [candidate()["event_id"]]

    asyncio.run(scenario())


def test_event_filter_pagination_and_insertion_order(tmp_path, create_app):
    app = create_app(tmp_path / "events.db")

    async def scenario():
        async with app.router.lifespan_context(app):
            for event in [candidate(1), candidate(2, "cam2"), candidate(3)]:
                assert (await json_request(app, "POST", "/api/v1/events", event))[0] == 201
            status, listing = await json_request(app, "GET", "/api/v1/events")
            assert status == 200
            assert [item["event_id"] for item in listing["events"]] == [
                candidate(3)["event_id"], candidate(2)["event_id"], candidate(1)["event_id"]]
            _, page = await json_request(app, "GET", "/api/v1/events?camera_id=cam1&limit=1&offset=1")
            assert page["limit"] == 1 and page["offset"] == 1
            assert [item["event_id"] for item in page["events"]] == [candidate(1)["event_id"]]
            _, empty = await json_request(app, "GET", "/api/v1/events?camera_id=missing")
            assert empty["events"] == []
            _, injection = await json_request(app, "GET", "/api/v1/events?camera_id=%27%20OR%201%3D1--")
            assert injection["events"] == []
            for query in ["limit=0", "limit=201", "offset=-1", "limit=oops"]:
                assert (await json_request(app, "GET", f"/api/v1/events?{query}"))[0] == 422

    asyncio.run(scenario())


def test_heartbeat_upsert_and_stable_camera_list(tmp_path, create_app):
    path = tmp_path / "events.db"
    app = create_app(path)

    async def scenario():
        async with app.router.lifespan_context(app):
            assert await json_request(app, "GET", "/api/v1/cameras") == (200, {"cameras": []})
            for payload in [heartbeat("cam2"), heartbeat("cam1"), heartbeat("cam2", "DEGRADED")]:
                status, ack = await json_request(app, "POST", "/api/v1/cameras/heartbeat", payload)
                assert status == 200
                assert ack == {"camera_id": payload["camera_id"], "accepted": True}
            _, listing = await json_request(app, "GET", "/api/v1/cameras")
            assert [item["camera_id"] for item in listing["cameras"]] == ["cam1", "cam2"]
            assert [item["status"] for item in listing["cameras"]] == ["ONLINE", "DEGRADED"]
            assert all(item["observed_at_utc"] == "2026-09-27T03:00:00Z"
                       for item in listing["cameras"])
            assert all("received_at_utc" not in item for item in listing["cameras"])
            assert (await json_request(app, "POST", "/api/v1/cameras/heartbeat",
                                       heartbeat("cam2", "invalid")))[0] == 422
            with closing(sqlite3.connect(path)) as connection:
                assert connection.execute("SELECT count(*) FROM camera_status").fetchone()[0] == 2

    asyncio.run(scenario())


def test_root_serves_dashboard(tmp_path, create_app):
    app = create_app(tmp_path / "events.db")

    async def scenario():
        async with app.router.lifespan_context(app):
            status, headers, body = await request(app, "GET", "/")
            assert status == 200
            assert headers[b"content-type"].startswith(b"text/html")
            assert b"<!doctype html>" in body.lower()

    asyncio.run(scenario())


def test_nonfinite_nested_attributes_do_not_poison_event_listing(tmp_path, create_app):
    path = tmp_path / "events.db"
    app = create_app(path)
    invalid_body = (
        b'{"schema_version":"1.0","camera_id":"cam1","gateway_id":"synthetic",'
        b'"gateway_boot_id":"boot1","sequence_number":1,'
        b'"observed_at_utc":"2026-09-27T03:00:00Z",'
        b'"model_versions":{"detector":"synthetic-v1"},"configuration_version":"v1",'
        b'"event_id":"00000000-0000-4000-8000-000000000002",'
        b'"candidate_type":"tripwire_crossing","attributes":{"outer":[{"value":1e400}]}}'
    )

    async def scenario():
        async with app.router.lifespan_context(app):
            assert (await json_request(app, "POST", "/api/v1/events", candidate()))[0] == 201
            status, _, _ = await request(app, "POST", "/api/v1/events", raw_body=invalid_body)
            assert status == 422
            status, listing = await json_request(app, "GET", "/api/v1/events")
            assert status == 200
            assert [event["event_id"] for event in listing["events"]] == [
                "00000000-0000-4000-8000-000000000001"]
            with closing(sqlite3.connect(path)) as connection:
                assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 1

    asyncio.run(scenario())
