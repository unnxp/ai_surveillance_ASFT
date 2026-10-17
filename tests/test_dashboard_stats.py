"""Dashboard aggregates and static model-card assets, using synthetic events only."""

import asyncio
import json

from tests.test_central_api import (  # noqa: F401  (create_app is a fixture)
    candidate, create_app, heartbeat, json_request, request,
)


def test_stats_aggregate_incidents_reviews_events_and_cameras(tmp_path, create_app):
    app = create_app(tmp_path / "events.db")

    async def scenario():
        async with app.router.lifespan_context(app):
            status, empty = await json_request(app, "GET", "/api/v1/stats")
            assert status == 200
            assert empty["incidents"] == {"total": 0, "open": 0, "by_status": {}, "by_severity": {},
                                          "reviewed": 0, "false_positive_rate": None}
            assert len(empty["events"]["per_hour"]) == 12

            for number, camera in ((1, "cam1"), (2, "cam2")):
                assert (await json_request(app, "POST", "/api/v1/events",
                                           candidate(number, camera)))[0] == 201
            await json_request(app, "POST", "/api/v1/cameras/heartbeat", heartbeat("cam1"))
            await json_request(app, "POST", "/api/v1/cameras/heartbeat", heartbeat("cam2", "OFFLINE"))
            _, incidents = await json_request(app, "GET", "/api/v1/incidents")
            first = incidents["incidents"][0]["incident_id"]
            await json_request(app, "POST", f"/api/v1/incidents/{first}/review",
                               {"action": "reject", "reviewed_by": "operator",
                                "false_positive_reason": "benign_activity"})

            _, stats = await json_request(app, "GET", "/api/v1/stats")
            assert stats["events"]["sampled"] == 2
            assert stats["events"]["by_type"] == {"tripwire_crossing": 2}
            assert stats["cameras"] == {"total": 2, "online": 1}
            assert stats["incidents"]["total"] == len(incidents["incidents"])
            assert stats["incidents"]["by_status"]["rejected"] == 1
            assert stats["incidents"]["false_positive_rate"] == 1.0

    asyncio.run(scenario())


def test_model_card_assets_are_served(tmp_path, create_app):
    app = create_app(tmp_path / "events.db")

    async def scenario():
        async with app.router.lifespan_context(app):
            status, _, body = await request(app, "GET", "/static/model_card/model_card.json")
            assert status == 200
            card = json.loads(body)
            assert card["face"]["operating_point"]["threshold"] == 0.275
            status, headers, _ = await request(app, "GET", "/static/model_card/face_far_frr.png")
            assert status == 200 and headers[b"content-type"] == b"image/png"
            status, _, page = await request(app, "GET", "/")
            assert status == 200 and b"/api/v1/stats" in page

    asyncio.run(scenario())
