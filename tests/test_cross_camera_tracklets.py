"""Appearance tracklets link people across cameras only with topology, timing and similarity."""

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tests.test_central_api import (  # noqa: F401  (create_app is a fixture)
    candidate, create_app, json_request,
)

T0 = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)


def vec(seed, noise=0.0, other=None):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=128)
    if other is not None:
        base = other + noise * rng.normal(size=128)
    return base / np.linalg.norm(base)


def tracklet(camera, track, appearance, start, end, phase="ready", boot="boot1"):
    from common.identity_schemas import TrackletSummary

    return TrackletSummary(camera_id=camera, gateway_boot_id=boot, local_track_id=track, phase=phase,
                           first_seen_utc=T0 + timedelta(seconds=start),
                           last_seen_utc=T0 + timedelta(seconds=end), sample_count=4,
                           model_version="test", appearance=[float(v) for v in appearance])


@pytest.fixture
def engine(tmp_path):
    from common.identity_schemas import TopologyEdgeInput
    from server.association import CrossCameraAssociationEngine
    from server.database import Database

    database = Database(tmp_path / "central.db")
    database.initialize()
    database.set_topology_edges([TopologyEdgeInput(source_camera="cam1", destination_camera="cam2",
                                                   min_travel_seconds=0, max_travel_seconds=30)])
    return CrossCameraAssociationEngine(database, appearance_match=0.8, appearance_possible=0.7)


def test_same_appearance_after_valid_travel_time_is_one_person(engine):
    person = vec(1)
    first = engine.correlate_tracklet(tracklet("cam1", 3, person, 0, 4))
    engine.correlate_tracklet(tracklet("cam1", 3, person, 0, 9, phase="end"))
    second = engine.correlate_tracklet(tracklet("cam2", 7, vec(2, 0.05, person), 14, 16))
    assert second.global_person_id == first.global_person_id
    assert second.association_status == "matched" and second.cross_camera_handoff
    session = engine.database.get_session(first.global_person_id)
    assert session.associated_cameras == ["cam1", "cam2"]


def test_different_appearance_or_missing_topology_or_late_arrival_stays_separate(engine):
    person = vec(1)
    first = engine.correlate_tracklet(tracklet("cam1", 3, person, 0, 4))
    stranger = engine.correlate_tracklet(tracklet("cam2", 8, vec(99), 6, 8))
    assert stranger.global_person_id != first.global_person_id
    assert stranger.association_status == "unknown"
    reverse = engine.correlate_tracklet(tracklet("cam1", 9, vec(3, 0.05, person), 20, 22))
    assert reverse.global_person_id != first.global_person_id  # no cam2 -> cam1 topology edge
    late = engine.correlate_tracklet(tracklet("cam2", 10, vec(4, 0.05, person), 60, 62))
    assert late.global_person_id != first.global_person_id  # outside the 30 s travel window


def test_two_similar_candidates_are_ambiguous_not_forced(engine):
    person = vec(1)
    a = engine.correlate_tracklet(tracklet("cam1", 1, person, 0, 4))
    b = engine.correlate_tracklet(tracklet("cam1", 2, vec(5, 0.01, person), 0, 4))
    result = engine.correlate_tracklet(tracklet("cam2", 3, person, 10, 12))
    assert result.association_status == "ambiguous"
    assert result.global_person_id not in {a.global_person_id, b.global_person_id}


def test_events_from_a_linked_track_reuse_its_global_id(tmp_path, create_app):
    app = create_app(tmp_path / "events.db")

    async def scenario():
        async with app.router.lifespan_context(app):
            ready = tracklet("cam1", 4, vec(1), 0, 2).model_dump(mode="json")
            ready["first_seen_utc"] = ready["last_seen_utc"] = "2026-09-27T03:00:00Z"
            status, ack = await json_request(app, "POST", "/api/v1/tracklets", ready)
            assert status == 200 and ack["global_person_id"].startswith("GP-")
            event = candidate(1) | {"local_track_id": 4, "gateway_boot_id": "boot1"}
            assert (await json_request(app, "POST", "/api/v1/events", event))[0] == 201
            _, events = await json_request(app, "GET", "/api/v1/events")
            assert events["events"][0]["global_person_id"] == ack["global_person_id"]
            bad = dict(ready, appearance=[1.0] * 128)  # not L2-normalized
            assert (await json_request(app, "POST", "/api/v1/tracklets", bad))[0] == 422

    asyncio.run(scenario())


class FakeEncoder:
    def encode(self, crops):
        return [vec(int(c.mean()) + 1) for c in crops]


def test_builder_emits_ready_then_end_and_tags_preview_state():
    from common.identity_schemas import TrackletAck
    from edge.appearance import TrackletBuilder

    sent, now = [], [T0]
    send = lambda t: sent.append(t) or TrackletAck(global_person_id="GP-0042",
                                                    association_status="unknown",
                                                    cross_camera_handoff=False)
    builder = TrackletBuilder(FakeEncoder(), send, "boot1", sample_interval=0.0, ready_samples=2,
                              lost_after=1.0, clock=lambda: now[0])
    frame = np.full((300, 300, 3), 10, np.uint8)
    for step in range(3):
        now[0] = T0 + timedelta(seconds=step * 0.5)
        state = {5: {"bbox": [10, 10, 60, 200], "score": 0.9}, 6: {"bbox": [0, 0, 5, 5]}}
        builder.observe({"cam1": frame}, {"cam1": state})
    assert [t.phase for t in sent] == ["ready"]
    assert state[5]["global_person_id"] == "GP-0042" and "global_person_id" not in state[6]
    now[0] = T0 + timedelta(seconds=5)
    builder.observe({"cam1": frame}, {"cam1": {}})
    assert [t.phase for t in sent] == ["ready", "end"] and not builder.tracks
    assert abs(np.linalg.norm(sent[0].appearance) - 1) < 1e-4


def test_builder_survives_central_failure():
    from edge.appearance import TrackletBuilder

    def boom(_):
        raise RuntimeError("down")

    builder = TrackletBuilder(FakeEncoder(), boom, "boot1", sample_interval=0.0, ready_samples=1)
    builder.observe({"cam1": np.zeros((300, 300, 3), np.uint8)},
                    {"cam1": {1: {"bbox": [10, 10, 60, 200]}}})
    assert builder.failures == 1


def test_overlapping_views_within_tolerance_still_link(engine):
    person = vec(1)
    first = engine.correlate_tracklet(tracklet("cam1", 3, person, 0, 10))
    overlap = engine.correlate_tracklet(tracklet("cam2", 4, vec(6, 0.05, person), 8, 12))
    assert overlap.global_person_id == first.global_person_id
    far = engine.correlate_tracklet(tracklet("cam2", 5, vec(7, 0.05, person), 2, 4, boot="boot2"))
    assert far.global_person_id != first.global_person_id  # 6 s before last sighting: too early


def test_unlinked_track_is_merged_when_its_end_summary_matches(engine):
    person = vec(1)
    first = engine.correlate_tracklet(tracklet("cam1", 3, person, 0, 6))
    early = engine.correlate_tracklet(tracklet("cam2", 4, vec(8), 7, 8))  # partial entry crops
    assert early.global_person_id != first.global_person_id
    final = engine.correlate_tracklet(tracklet("cam2", 4, vec(9, 0.05, person), 7, 15, phase="end"))
    assert final.global_person_id == first.global_person_id and final.cross_camera_handoff
    assert engine.database.get_session(early.global_person_id).status == "expired"


def test_long_overlap_links_only_on_zero_minimum_edges(tmp_path):
    from common.identity_schemas import TopologyEdgeInput
    from server.association import CrossCameraAssociationEngine
    from server.database import Database

    database = Database(tmp_path / "c.db")
    database.initialize()
    database.set_topology_edges([
        TopologyEdgeInput(source_camera="cam1", destination_camera="cam2", min_travel_seconds=0, max_travel_seconds=30),
        TopologyEdgeInput(source_camera="cam1", destination_camera="cam3", min_travel_seconds=5, max_travel_seconds=30),
    ])
    engine = CrossCameraAssociationEngine(database, appearance_match=0.8, appearance_possible=0.7)
    person = vec(1)
    first = engine.correlate_tracklet(tracklet("cam1", 1, person, 0, 4))
    engine.correlate_tracklet(tracklet("cam1", 1, person, 0, 40, phase="end"))
    adjacent = engine.correlate_tracklet(tracklet("cam2", 2, vec(2, 0.05, person), 30, 32))  # 10 s overlap
    assert adjacent.global_person_id == first.global_person_id
    other = vec(3)
    second = engine.correlate_tracklet(tracklet("cam1", 5, other, 50, 54))
    engine.correlate_tracklet(tracklet("cam1", 5, other, 50, 90, phase="end"))
    remote = engine.correlate_tracklet(tracklet("cam3", 6, vec(4, 0.05, other), 80, 82))
    assert remote.global_person_id != second.global_person_id  # positive minimum: no overlap allowed
