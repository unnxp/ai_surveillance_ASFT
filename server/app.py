"""Local FastAPI entry point; importing the module creates no database."""

from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from re import fullmatch
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from common.identity_schemas import (
    FaceMatchRequest,
    FaceMatchResult,
    FaceRegistrySummary,
    FaceRegistryUpsert,
    GlobalPersonSessionSummary,
    TopologyConfig,
    TopologyEdge,
    TrackletAck,
    TrackletSummary,
)
from common.schemas import (
    AlertPolicyConfig,
    CameraHeartbeat,
    EventCandidate,
    EventIngestAck,
    EvidenceKind,
    EvidenceUploadAck,
    HeartbeatAck,
    IncidentReviewRequest,
    IncidentSummary,
)
from server.alert_policy import AlertPolicyEngine
from server.association import CrossCameraAssociationEngine
from server.database import Database
from server.evidence_store import (
    MAX_JPEG_BYTES,
    EvidenceStore,
    EvidenceTooLargeError,
    InvalidEvidenceError,
)


def create_app(database_path: str | Path | None = None) -> FastAPI:
    server_directory = Path(__file__).resolve().parent
    default_path = server_directory.parent / "data" / "surveillance.db"
    database = Database(default_path if database_path is None else database_path)
    evidence_store = EvidenceStore(database.path.parent / "evidence")
    association_engine = CrossCameraAssociationEngine(database)
    alert_policy_engine = AlertPolicyEngine(database)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        database.initialize()
        yield

    application = FastAPI(title="Local Surveillance Metadata", lifespan=lifespan)

    @application.post("/api/v1/events", response_model=EventIngestAck, status_code=201)
    def ingest_event(event: EventCandidate, response: Response) -> EventIngestAck:
        existing = database.get_event(event.event_id)
        if existing is not None:
            if existing.global_person_id and not event.global_person_id:
                event = event.model_copy(update={"global_person_id": existing.global_person_id})
        else:
            # Cross-camera identity correlation
            attrs = dict(event.attributes)
            person_id = attrs.get("person_id")
            display_name = attrs.get("display_name")
            # The edge face pipeline reports "face_similarity"; "similarity" is the legacy key.
            face_sim = attrs.get("face_similarity", attrs.get("similarity"))
            if face_sim is not None:
                try:
                    face_sim = float(face_sim)
                except (ValueError, TypeError):
                    face_sim = None

            gid, status, is_handoff = association_engine.correlate(
                camera_id=event.camera_id,
                observed_at_utc=event.observed_at_utc,
                local_track_id=event.local_track_id,
                person_id=person_id,
                display_name=display_name,
                face_similarity=face_sim,
                event_id=event.event_id,
                gateway_boot_id=event.gateway_boot_id,
            )
            if not event.global_person_id:
                event = event.model_copy(update={"global_person_id": gid})

            # Process alert policy and associate with incident timeline
            alert_policy_engine.process_candidate(event, gid)

        outcome = database.ingest_event(event)
        if outcome == "conflict":
            raise HTTPException(status_code=409, detail="Event ID conflicts with existing metadata")
        duplicate = outcome == "duplicate"
        if duplicate:
            response.status_code = 200
        return EventIngestAck(event_id=event.event_id, accepted=True, duplicate=duplicate)

    @application.get("/api/v1/events")
    def list_events(
        camera_id: str | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict:
        return {"events": database.list_events(camera_id, limit, offset),
                "limit": limit, "offset": offset}

    @application.put(
        "/api/v1/events/{event_id}/evidence/{evidence_id}",
        response_model=EvidenceUploadAck, status_code=201,
    )
    async def upload_evidence(
        event_id: UUID, evidence_id: UUID, request: Request, response: Response
    ) -> EvidenceUploadAck:
        event = database.get_event(event_id)
        declared = (next((item for item in event.evidence if item.evidence_id == evidence_id), None)
                    if event is not None else None)
        if declared is None:
            raise HTTPException(status_code=404, detail="Evidence reference not found")
        if declared.kind not in {EvidenceKind.KEYFRAME, EvidenceKind.PERSON_CROP}:
            raise HTTPException(status_code=422, detail="JPEG evidence kind required")
        if request.headers.get("content-type") != "image/jpeg":
            raise HTTPException(status_code=422, detail="JPEG content type required")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_JPEG_BYTES:
                raise HTTPException(status_code=413, detail="Evidence image too large")
        jpeg = bytes(body)
        existing = database.get_evidence_record(evidence_id)
        if existing is not None:
            previous_event, stored = existing
            if previous_event != event_id or stored.sha256 != declared.sha256:
                raise HTTPException(status_code=409, detail="Evidence ID conflicts")
            path = evidence_store.path_for(previous_event, evidence_id)
            if path.is_file():
                if jpeg != path.read_bytes():
                    raise HTTPException(status_code=409, detail="Evidence ID conflicts")
            else:
                try:
                    evidence_store.save(event_id, declared, jpeg)
                except EvidenceTooLargeError as error:
                    raise HTTPException(status_code=413, detail="Evidence image too large") from error
                except InvalidEvidenceError as error:
                    raise HTTPException(status_code=422, detail="Invalid evidence image") from error
            response.status_code = 200
            return EvidenceUploadAck(evidence_id=evidence_id, accepted=True, duplicate=True)
        try:
            stored = evidence_store.save(event_id, declared, jpeg)
        except EvidenceTooLargeError as error:
            raise HTTPException(status_code=413, detail="Evidence image too large") from error
        except InvalidEvidenceError as error:
            raise HTTPException(status_code=422, detail="Invalid evidence image") from error
        outcome = database.record_evidence(event_id, stored)
        if outcome == "conflict":
            raise HTTPException(status_code=409, detail="Evidence ID conflicts")
        if outcome == "duplicate":
            response.status_code = 200
        return EvidenceUploadAck(evidence_id=evidence_id, accepted=True,
                                 duplicate=outcome == "duplicate")

    @application.get("/api/v1/events/{event_id}/evidence")
    def event_evidence(event_id: UUID) -> dict:
        event = database.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        stored = {item.evidence_id for item in database.list_evidence(event_id)}
        rows = []
        for item in event.evidence:
            available = (item.evidence_id in stored
                         and evidence_store.path_for(event_id, item.evidence_id).is_file())
            rows.append({"evidence_id": item.evidence_id, "kind": item.kind,
                         "available": available,
                         "url": f"/api/v1/evidence/{item.evidence_id}" if available else None})
        incomplete = bool(event.attributes.get("evidence_incomplete")) or any(
            not row["available"] for row in rows
        )
        return {"event_id": event_id, "evidence_incomplete": incomplete, "evidence": rows}

    @application.get("/api/v1/evidence/{evidence_id}", response_class=FileResponse)
    def read_evidence(evidence_id: UUID) -> FileResponse:
        record = database.get_evidence_record(evidence_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Evidence not found")
        event_id, _ = record
        path = evidence_store.path_for(event_id, evidence_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Evidence not found")
        return FileResponse(path, media_type="image/jpeg")

    @application.post("/api/v1/cameras/heartbeat", response_model=HeartbeatAck)
    def heartbeat(payload: CameraHeartbeat) -> HeartbeatAck:
        database.upsert_heartbeat(payload)
        return HeartbeatAck(camera_id=payload.camera_id, accepted=True)

    @application.get("/api/v1/cameras")
    def list_cameras() -> dict:
        return {"cameras": database.list_cameras()}

    @application.post("/api/v1/face-registry/match", response_model=FaceMatchResult)
    async def match_face(request: Request) -> FaceMatchResult:
        # Parsing manually keeps FastAPI's 422 response from echoing query embeddings.
        try:
            payload = FaceMatchRequest.model_validate(await request.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="Invalid face match request") from error
        return database.match_face(payload)

    @application.put("/api/v1/face-registry/{person_id}", response_model=FaceRegistrySummary)
    async def upsert_face_registry(
        person_id: str, request: Request, response: Response
    ) -> FaceRegistrySummary:
        if fullmatch(r"[A-Za-z0-9_-]{1,64}", person_id) is None:
            raise HTTPException(status_code=422, detail="Invalid face registry request")
        try:
            payload = FaceRegistryUpsert.model_validate(await request.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="Invalid face registry request") from error
        created, summary = database.upsert_identity(person_id, payload)
        response.status_code = 201 if created else 200
        return summary

    @application.get("/api/v1/face-registry")
    def list_face_registry() -> dict:
        return {"identities": database.list_identities()}

    @application.delete("/api/v1/face-registry/{person_id}", status_code=204)
    def delete_face_registry(person_id: str) -> Response:
        if fullmatch(r"[A-Za-z0-9_-]{1,64}", person_id) is None:
            raise HTTPException(status_code=422, detail="Invalid face registry request")
        if not database.delete_identity(person_id):
            raise HTTPException(status_code=404, detail="Registered identity not found")
        return Response(status_code=204)

    @application.get("/api/v1/topology")
    def get_topology() -> dict:
        return {"edges": database.get_topology_edges()}

    @application.put("/api/v1/topology")
    async def update_topology(request: Request) -> dict:
        try:
            config = TopologyConfig.model_validate(await request.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="Invalid topology configuration") from error
        saved = database.set_topology_edges(config.edges)
        return {"edges": saved}

    @application.post("/api/v1/tracklets", response_model=TrackletAck)
    def ingest_tracklet(tracklet: TrackletSummary) -> TrackletAck:
        """Appearance-only association input; never creates alerts and is never stored."""
        return association_engine.correlate_tracklet(tracklet)

    @application.get("/api/v1/global-person-sessions")
    def list_global_person_sessions(status: str | None = None) -> dict:
        return {"sessions": database.list_sessions(status=status)}

    @application.get("/api/v1/alert-policy")
    def get_alert_policy() -> dict:
        return {"policy": database.get_alert_policy()}

    @application.put("/api/v1/alert-policy")
    async def update_alert_policy(request: Request) -> dict:
        try:
            config = AlertPolicyConfig.model_validate(await request.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="Invalid alert policy configuration") from error
        saved = database.set_alert_policy(config)
        return {"policy": saved}

    @application.post("/api/v1/alert-policy/reset")
    def reset_alert_policy() -> dict:
        return {"policy": database.reset_default_alert_policy()}

    @application.get("/api/v1/incidents")
    def list_incidents(status: str | None = None) -> dict:
        return {"incidents": database.list_incidents(status=status)}

    @application.get("/api/v1/incidents/{incident_id}")
    def get_incident(incident_id: str) -> IncidentSummary:
        incident = database.get_incident(incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="Incident not found")
        return incident

    @application.post("/api/v1/incidents/{incident_id}/review", response_model=IncidentSummary)
    def review_incident(incident_id: str, review: IncidentReviewRequest) -> IncidentSummary:
        incident = database.record_incident_review(incident_id, review)
        if incident is None:
            raise HTTPException(status_code=404, detail="Incident not found")
        return incident

    @application.get("/api/v1/stats")
    def dashboard_stats() -> dict:
        """Aggregates for the operator dashboard (recent 200 events, all incidents)."""
        now = datetime.now(timezone.utc)
        incidents = database.list_incidents()
        events = database.list_events(limit=200)
        cameras = database.list_cameras()
        outcomes = Counter(incident.status for incident in incidents)
        decided = outcomes["confirmed"] + outcomes["rejected"] + outcomes["reclassified"]
        hours = [(now - timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0)
                 for offset in range(11, -1, -1)]
        per_hour = Counter(event.observed_at_utc.replace(minute=0, second=0, microsecond=0)
                           for event in events)
        return {
            "generated_at_utc": now,
            "incidents": {
                "total": len(incidents),
                "open": outcomes["open"],
                "by_status": dict(outcomes),
                "by_severity": dict(Counter(i.highest_severity.value for i in incidents
                                            if i.status in ("open", "acknowledged"))),
                "reviewed": len(incidents) - outcomes["open"],
                "false_positive_rate": (outcomes["rejected"] / decided) if decided else None,
            },
            "events": {
                "sampled": len(events),
                "last_hour": sum(1 for e in events if e.observed_at_utc >= now - timedelta(hours=1)),
                "by_type": dict(Counter(e.candidate_type.value for e in events)),
                "per_hour": [{"hour_utc": h, "count": per_hour.get(h, 0)} for h in hours],
            },
            "cameras": {
                "total": len(cameras),
                "online": sum(1 for c in cameras if c.status.value in ("ONLINE", "DEGRADED")),
            },
        }

    application.mount("/static", StaticFiles(directory=server_directory / "static"), name="static")

    @application.get("/", response_class=FileResponse)
    def dashboard() -> FileResponse:
        return FileResponse(server_directory / "static" / "index.html", media_type="text/html")

    return application


app = create_app()
