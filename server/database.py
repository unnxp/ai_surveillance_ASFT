"""SQLite ownership for central event and camera metadata."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Literal
from uuid import UUID

import yaml

FACE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "face_runtime.yaml"


def load_face_match_threshold(path: Path = FACE_CONFIG_PATH, default: float = 0.275) -> float:
    """Read the open-set operating point (FAR <= 1%) chosen from the threshold sweep."""
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError:
        return default
    return float(values.get("face_matching", {}).get("match_threshold", default))


from common.identity_schemas import (
    FaceMatchRequest,
    FaceMatchResult,
    FaceRegistrySummary,
    FaceRegistryUpsert,
    GlobalPersonSessionSummary,
    TopologyEdge,
    TopologyEdgeInput,
)
from common.schemas import (
    AlertPolicyConfig,
    AlertPolicyRule,
    AlertRouting,
    AlertSeverity,
    CameraHeartbeat,
    CandidateType,
    EventCandidate,
    EvidenceReference,
    IncidentReviewRecord,
    IncidentReviewRequest,
    IncidentSummary,
    IncidentTimelineEntry,
    ReviewAction,
)

DEFAULT_ALERT_RULES: dict[CandidateType, AlertPolicyRule] = {
    CandidateType.POSSIBLE_PERSON_VIOLENCE: AlertPolicyRule(
        candidate_type=CandidateType.POSSIBLE_PERSON_VIOLENCE,
        enabled=True, severity=AlertSeverity.HIGH, routing=AlertRouting.IMMEDIATE_ALERT, cooldown_seconds=15.0,
    ),
    CandidateType.POSSIBLE_PROPERTY_CRIME: AlertPolicyRule(
        candidate_type=CandidateType.POSSIBLE_PROPERTY_CRIME,
        enabled=True, severity=AlertSeverity.MEDIUM, routing=AlertRouting.REVIEW_QUEUE, cooldown_seconds=15.0,
    ),
    CandidateType.UNAUTHORIZED_ZONE_ENTRY: AlertPolicyRule(
        candidate_type=CandidateType.UNAUTHORIZED_ZONE_ENTRY,
        enabled=True, severity=AlertSeverity.HIGH, routing=AlertRouting.IMMEDIATE_ALERT, cooldown_seconds=10.0,
    ),
    CandidateType.UNKNOWN_PERSON_IN_RESTRICTED_ZONE: AlertPolicyRule(
        candidate_type=CandidateType.UNKNOWN_PERSON_IN_RESTRICTED_ZONE,
        enabled=True, severity=AlertSeverity.HIGH, routing=AlertRouting.IMMEDIATE_ALERT, cooldown_seconds=10.0,
    ),
    CandidateType.TRIPWIRE_CROSSING: AlertPolicyRule(
        candidate_type=CandidateType.TRIPWIRE_CROSSING,
        enabled=True, severity=AlertSeverity.MEDIUM, routing=AlertRouting.REVIEW_QUEUE, cooldown_seconds=10.0,
    ),
    CandidateType.POSSIBLE_LOITERING: AlertPolicyRule(
        candidate_type=CandidateType.POSSIBLE_LOITERING,
        enabled=True, severity=AlertSeverity.MEDIUM, routing=AlertRouting.REVIEW_QUEUE, cooldown_seconds=30.0,
    ),
    CandidateType.CROSS_CAMERA_HANDOFF: AlertPolicyRule(
        candidate_type=CandidateType.CROSS_CAMERA_HANDOFF,
        enabled=True, severity=AlertSeverity.LOW, routing=AlertRouting.CONTEXT_ONLY, cooldown_seconds=5.0,
    ),
}
SEVERITY_ORDER = {AlertSeverity.LOW: 1, AlertSeverity.MEDIUM: 2, AlertSeverity.HIGH: 3, AlertSeverity.CRITICAL: 4}


def _canonical_json(payload: EventCandidate | CameraHeartbeat) -> str:
    return json.dumps(payload.model_dump(mode="json"), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _received_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Open short-lived connections; constructing this object never touches disk."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS events ("
                    "insertion_id INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE, "
                    "camera_id TEXT NOT NULL, payload_json TEXT NOT NULL, "
                    "received_at_utc TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS camera_status ("
                    "camera_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, "
                    "received_at_utc TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS evidence ("
                    "evidence_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, kind TEXT NOT NULL, "
                    "relative_path TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
                    "captured_at_utc TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS registered_identities ("
                    "person_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, role TEXT NOT NULL, "
                    "consent_status TEXT NOT NULL CHECK (consent_status = 'active'), "
                    "consent_recorded_at_utc TEXT NOT NULL, enrolled_at_utc TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS face_templates ("
                    "template_id INTEGER PRIMARY KEY, person_id TEXT NOT NULL, "
                    "model_version TEXT NOT NULL, vector_json TEXT NOT NULL, "
                    "FOREIGN KEY(person_id) REFERENCES registered_identities(person_id))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS zone_grants ("
                    "person_id TEXT NOT NULL, camera_id TEXT NOT NULL, zone_id TEXT NOT NULL, "
                    "PRIMARY KEY(person_id, camera_id, zone_id), "
                    "FOREIGN KEY(person_id) REFERENCES registered_identities(person_id))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS camera_topology ("
                    "edge_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "source_camera TEXT NOT NULL, destination_camera TEXT NOT NULL, "
                    "min_travel_seconds REAL NOT NULL, max_travel_seconds REAL NOT NULL, "
                    "enabled INTEGER NOT NULL DEFAULT 1, "
                    "UNIQUE(source_camera, destination_camera))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS global_person_sessions ("
                    "global_person_id TEXT PRIMARY KEY, "
                    "status TEXT NOT NULL CHECK(status IN ('active', 'expired')), "
                    "created_at_utc TEXT NOT NULL, last_seen_at_utc TEXT NOT NULL, "
                    "person_id TEXT, display_name TEXT, association_status TEXT NOT NULL, "
                    "last_similarity REAL, observation_count INTEGER NOT NULL DEFAULT 1, "
                    "associated_cameras_json TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS session_observations ("
                    "observation_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "global_person_id TEXT NOT NULL, camera_id TEXT NOT NULL, "
                    "local_track_id TEXT, observed_at_utc TEXT NOT NULL, event_id TEXT, "
                    "FOREIGN KEY(global_person_id) REFERENCES global_person_sessions(global_person_id))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS alert_policy_rules ("
                    "candidate_type TEXT PRIMARY KEY, enabled INTEGER NOT NULL, "
                    "severity TEXT NOT NULL, routing TEXT NOT NULL, cooldown_seconds REAL NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS incidents ("
                    "incident_id TEXT PRIMARY KEY, global_person_id TEXT NOT NULL, "
                    "status TEXT NOT NULL CHECK(status IN ('open', 'acknowledged', 'confirmed', 'rejected', 'reclassified')), "
                    "highest_severity TEXT NOT NULL, created_at_utc TEXT NOT NULL, "
                    "updated_at_utc TEXT NOT NULL, primary_camera TEXT NOT NULL, "
                    "FOREIGN KEY(global_person_id) REFERENCES global_person_sessions(global_person_id))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS incident_events ("
                    "entry_id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id TEXT NOT NULL, "
                    "event_id TEXT NOT NULL UNIQUE, camera_id TEXT NOT NULL, candidate_type TEXT NOT NULL, "
                    "observed_at_utc TEXT NOT NULL, severity TEXT NOT NULL, evidence_count INTEGER NOT NULL, "
                    "FOREIGN KEY(incident_id) REFERENCES incidents(incident_id))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS incident_reviews ("
                    "review_id TEXT PRIMARY KEY, incident_id TEXT NOT NULL, "
                    "action TEXT NOT NULL, reviewed_by TEXT NOT NULL, "
                    "reviewed_at_utc TEXT NOT NULL, final_severity TEXT, "
                    "reclassified_type TEXT, false_positive_reason TEXT, notes TEXT, "
                    "FOREIGN KEY(incident_id) REFERENCES incidents(incident_id))"
                )
                # Seed default alert policy rules if empty
                count = connection.execute("SELECT count(*) FROM alert_policy_rules").fetchone()[0]
                if count == 0:
                    for rule in DEFAULT_ALERT_RULES.values():
                        connection.execute(
                            "INSERT INTO alert_policy_rules (candidate_type, enabled, severity, routing, cooldown_seconds) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (rule.candidate_type.value, 1 if rule.enabled else 0, rule.severity.value,
                             rule.routing.value, rule.cooldown_seconds),
                        )

    @staticmethod
    def _identity_summary(connection: sqlite3.Connection, person_id: str) -> FaceRegistrySummary:
        row = connection.execute(
            "SELECT person_id, display_name, role, consent_status, consent_recorded_at_utc, "
            "enrolled_at_utc FROM registered_identities WHERE person_id = ?", (person_id,)
        ).fetchone()
        grants = connection.execute(
            "SELECT camera_id, zone_id FROM zone_grants WHERE person_id = ? "
            "ORDER BY camera_id, zone_id", (person_id,)
        ).fetchall()
        count = connection.execute(
            "SELECT count(*) FROM face_templates WHERE person_id = ?", (person_id,)
        ).fetchone()[0]
        versions = connection.execute(
            "SELECT DISTINCT model_version FROM face_templates WHERE person_id = ? "
            "ORDER BY model_version", (person_id,)
        ).fetchall()
        return FaceRegistrySummary.model_validate({
            "person_id": row[0], "display_name": row[1], "role": row[2],
            "consent_status": row[3], "consent_recorded_at_utc": row[4],
            "enrolled_at_utc": row[5],
            "allowed_zones": [dict(camera_id=camera_id, zone_id=zone_id)
                              for camera_id, zone_id in grants],
            "template_count": count, "model_versions": [version[0] for version in versions],
        })

    def upsert_identity(
        self, person_id: str, payload: FaceRegistryUpsert
    ) -> tuple[bool, FaceRegistrySummary]:
        now = _received_at_utc()
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                previous = connection.execute(
                    "SELECT enrolled_at_utc FROM registered_identities WHERE person_id = ?",
                    (person_id,),
                ).fetchone()
                created = previous is None
                if created:
                    connection.execute(
                        "INSERT INTO registered_identities "
                        "(person_id, display_name, role, consent_status, "
                        "consent_recorded_at_utc, enrolled_at_utc) VALUES (?, ?, ?, 'active', ?, ?)",
                        (person_id, payload.display_name, payload.role, now, now),
                    )
                else:
                    connection.execute(
                        "UPDATE registered_identities SET display_name = ?, role = ?, "
                        "consent_status = 'active', consent_recorded_at_utc = ? "
                        "WHERE person_id = ?",
                        (payload.display_name, payload.role, now, person_id),
                    )
                    connection.execute(
                        "DELETE FROM face_templates WHERE person_id = ?", (person_id,)
                    )
                    connection.execute(
                        "DELETE FROM zone_grants WHERE person_id = ?", (person_id,)
                    )
                connection.executemany(
                    "INSERT INTO face_templates (person_id, model_version, vector_json) "
                    "VALUES (?, ?, ?)",
                    [(person_id, template.model_version,
                      json.dumps(template.vector, separators=(",", ":"), allow_nan=False))
                     for template in payload.templates],
                )
                connection.executemany(
                    "INSERT INTO zone_grants (person_id, camera_id, zone_id) VALUES (?, ?, ?)",
                    [(person_id, grant.camera_id, grant.zone_id)
                     for grant in payload.allowed_zones],
                )
                summary = self._identity_summary(connection, person_id)
        return created, summary

    def list_identities(self) -> list[FaceRegistrySummary]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            ids = connection.execute(
                "SELECT person_id FROM registered_identities ORDER BY person_id"
            ).fetchall()
            return [self._identity_summary(connection, person_id) for (person_id,) in ids]

    def delete_identity(self, person_id: str) -> bool:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                exists = connection.execute(
                    "SELECT 1 FROM registered_identities WHERE person_id = ?", (person_id,)
                ).fetchone()
                if exists is None:
                    return False
                connection.execute("DELETE FROM face_templates WHERE person_id = ?", (person_id,))
                connection.execute("DELETE FROM zone_grants WHERE person_id = ?", (person_id,))
                connection.execute(
                    "DELETE FROM registered_identities WHERE person_id = ?", (person_id,)
                )
                return True

    def check_access(
        self, person_id: str, camera_id: str, zone_id: str
    ) -> Literal["authorized", "unauthorized"] | None:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            exists = connection.execute(
                "SELECT 1 FROM registered_identities WHERE person_id = ?", (person_id,)
            ).fetchone()
            if exists is None:
                return None
            grant = connection.execute(
                "SELECT 1 FROM zone_grants WHERE person_id = ? AND camera_id = ? AND zone_id = ?",
                (person_id, camera_id, zone_id),
            ).fetchone()
        return "authorized" if grant else "unauthorized"

    def match_face(
        self, request: FaceMatchRequest, threshold: float | None = None
    ) -> FaceMatchResult:
        """Search consented same-version templates without retaining the query vector."""
        if threshold is None:
            threshold = load_face_match_threshold()
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            rows = connection.execute(
                "SELECT t.person_id, t.vector_json FROM face_templates AS t "
                "JOIN registered_identities AS i ON i.person_id = t.person_id "
                "WHERE t.model_version = ? AND i.consent_status = 'active' "
                "ORDER BY t.person_id, t.template_id",
                (request.model_version,),
            ).fetchall()
            if not rows:
                return FaceMatchResult(status="unavailable", model_version=request.model_version)
            scores: dict[str, float] = {}
            query_norm = sqrt(sum(value * value for value in request.vector))
            for person_id, encoded in rows:
                vector = json.loads(encoded)
                template_norm = sqrt(sum(value * value for value in vector))
                score = sum(left * right for left, right in zip(request.vector, vector, strict=True))
                score = max(-1.0, min(1.0, score / (query_norm * template_norm)))
                scores[person_id] = max(score, scores.get(person_id, -1.0))
            person_id, score = min(scores.items(), key=lambda item: (-item[1], item[0]))
            if score < threshold:
                return FaceMatchResult(status="unknown", similarity=score,
                                       model_version=request.model_version)
            display_name = connection.execute(
                "SELECT display_name FROM registered_identities WHERE person_id = ?",
                (person_id,),
            ).fetchone()[0]
            grants = {
                row[0] for row in connection.execute(
                    "SELECT zone_id FROM zone_grants WHERE person_id = ? AND camera_id = ?",
                    (person_id, request.camera_id),
                ).fetchall()
            }
            access = {zone: "authorized" if zone in grants else "unauthorized"
                      for zone in request.zone_ids}
        return FaceMatchResult(status="registered", person_id=person_id,
                               display_name=display_name, similarity=score,
                               zone_access=access, model_version=request.model_version)

    def ingest_event(self, event: EventCandidate) -> Literal["accepted", "duplicate", "conflict"]:
        payload = _canonical_json(event)
        event_id = str(event.event_id)
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO events (event_id, camera_id, payload_json, received_at_utc) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(event_id) DO NOTHING",
                    (event_id, event.camera_id, payload, _received_at_utc()),
                )
                if cursor.rowcount == 1:
                    return "accepted"
                existing = connection.execute(
                    "SELECT payload_json FROM events WHERE event_id = ?", (event_id,)
                ).fetchone()
                return "duplicate" if existing[0] == payload else "conflict"

    def list_events(
        self, camera_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[EventCandidate]:
        query = "SELECT payload_json FROM events"
        parameters: tuple = ()
        if camera_id is not None:
            query += " WHERE camera_id = ?"
            parameters = (camera_id,)
        query += " ORDER BY insertion_id DESC LIMIT ? OFFSET ?"
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            rows = connection.execute(query, (*parameters, limit, offset)).fetchall()
        return [EventCandidate.model_validate_json(row[0]) for row in rows]

    def get_event(self, event_id: UUID) -> EventCandidate | None:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            row = connection.execute(
                "SELECT payload_json FROM events WHERE event_id = ?", (str(event_id),)
            ).fetchone()
        return EventCandidate.model_validate_json(row[0]) if row else None

    def record_evidence(
        self, event_id: UUID, reference: EvidenceReference
    ) -> Literal["accepted", "duplicate", "conflict"]:
        values = (
            str(reference.evidence_id), str(event_id), reference.kind.value,
            reference.relative_path, reference.sha256, reference.size_bytes,
            reference.captured_at_utc.isoformat(),
        )
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                inserted = connection.execute(
                    "INSERT INTO evidence (evidence_id, event_id, kind, relative_path, sha256, "
                    "size_bytes, captured_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(evidence_id) DO NOTHING", values,
                )
                if inserted.rowcount == 1:
                    return "accepted"
                existing = connection.execute(
                    "SELECT evidence_id, event_id, kind, relative_path, sha256, size_bytes, "
                    "captured_at_utc FROM evidence WHERE evidence_id = ?",
                    (str(reference.evidence_id),),
                ).fetchone()
                return "duplicate" if existing == values else "conflict"

    def list_evidence(self, event_id: UUID) -> list[EvidenceReference]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            rows = connection.execute(
                "SELECT evidence_id, kind, relative_path, sha256, size_bytes, captured_at_utc "
                "FROM evidence WHERE event_id = ? ORDER BY evidence_id", (str(event_id),)
            ).fetchall()
        return [EvidenceReference(evidence_id=row[0], kind=row[1], relative_path=row[2],
                                  sha256=row[3], size_bytes=row[4], captured_at_utc=row[5])
                for row in rows]

    def get_evidence_record(self, evidence_id: UUID) -> tuple[UUID, EvidenceReference] | None:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            row = connection.execute(
                "SELECT event_id, kind, relative_path, sha256, size_bytes, captured_at_utc "
                "FROM evidence WHERE evidence_id = ?", (str(evidence_id),)
            ).fetchone()
        if row is None:
            return None
        return UUID(row[0]), EvidenceReference(
            evidence_id=evidence_id, kind=row[1], relative_path=row[2], sha256=row[3],
            size_bytes=row[4], captured_at_utc=row[5],
        )

    def upsert_heartbeat(self, heartbeat: CameraHeartbeat) -> None:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO camera_status (camera_id, payload_json, received_at_utc) "
                    "VALUES (?, ?, ?) ON CONFLICT(camera_id) DO UPDATE SET "
                    "payload_json=excluded.payload_json, received_at_utc=excluded.received_at_utc",
                    (heartbeat.camera_id, _canonical_json(heartbeat), _received_at_utc()),
                )

    def list_cameras(self) -> list[CameraHeartbeat]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            rows = connection.execute(
                "SELECT payload_json FROM camera_status ORDER BY camera_id"
            ).fetchall()
        return [CameraHeartbeat.model_validate_json(row[0]) for row in rows]

    def set_topology_edges(self, edges: list[TopologyEdgeInput]) -> list[TopologyEdge]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                connection.execute("DELETE FROM camera_topology")
                for edge in edges:
                    connection.execute(
                        "INSERT INTO camera_topology (source_camera, destination_camera, "
                        "min_travel_seconds, max_travel_seconds, enabled) VALUES (?, ?, ?, ?, ?)",
                        (edge.source_camera, edge.destination_camera,
                         edge.min_travel_seconds, edge.max_travel_seconds, 1 if edge.enabled else 0),
                    )
        return self.get_topology_edges()

    def get_topology_edges(self) -> list[TopologyEdge]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            rows = connection.execute(
                "SELECT edge_id, source_camera, destination_camera, min_travel_seconds, "
                "max_travel_seconds, enabled FROM camera_topology ORDER BY edge_id"
            ).fetchall()
        return [
            TopologyEdge(
                edge_id=row[0], source_camera=row[1], destination_camera=row[2],
                min_travel_seconds=row[3], max_travel_seconds=row[4], enabled=bool(row[5]),
            )
            for row in rows
        ]

    def get_topology_edge(self, source_camera: str, destination_camera: str) -> TopologyEdge | None:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            row = connection.execute(
                "SELECT edge_id, source_camera, destination_camera, min_travel_seconds, "
                "max_travel_seconds, enabled FROM camera_topology WHERE source_camera = ? "
                "AND destination_camera = ? AND enabled = 1",
                (source_camera, destination_camera),
            ).fetchone()
        if row is None:
            return None
        return TopologyEdge(
            edge_id=row[0], source_camera=row[1], destination_camera=row[2],
            min_travel_seconds=row[3], max_travel_seconds=row[4], enabled=bool(row[5]),
        )

    def upsert_session(self, session: GlobalPersonSessionSummary) -> None:
        cameras_json = json.dumps(session.associated_cameras)
        created_str = session.created_at_utc.isoformat()
        last_seen_str = session.last_seen_at_utc.isoformat()
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO global_person_sessions (global_person_id, status, created_at_utc, "
                    "last_seen_at_utc, person_id, display_name, association_status, last_similarity, "
                    "observation_count, associated_cameras_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(global_person_id) DO UPDATE SET status=excluded.status, "
                    "last_seen_at_utc=excluded.last_seen_at_utc, person_id=excluded.person_id, "
                    "display_name=excluded.display_name, association_status=excluded.association_status, "
                    "last_similarity=excluded.last_similarity, observation_count=excluded.observation_count, "
                    "associated_cameras_json=excluded.associated_cameras_json",
                    (session.global_person_id, session.status, created_str, last_seen_str,
                     session.person_id, session.display_name, session.association_status,
                     session.last_similarity, session.observation_count, cameras_json),
                )

    def record_session_observation(
        self, global_person_id: str, camera_id: str, local_track_id: str | None,
        observed_at_utc: str, event_id: str | None = None,
    ) -> None:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO session_observations (global_person_id, camera_id, local_track_id, "
                    "observed_at_utc, event_id) VALUES (?, ?, ?, ?, ?)",
                    (global_person_id, camera_id, local_track_id, observed_at_utc, event_id),
                )

    def get_session(self, global_person_id: str) -> GlobalPersonSessionSummary | None:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            row = connection.execute(
                "SELECT global_person_id, status, created_at_utc, last_seen_at_utc, person_id, "
                "display_name, association_status, last_similarity, observation_count, "
                "associated_cameras_json FROM global_person_sessions WHERE global_person_id = ?",
                (global_person_id,),
            ).fetchone()
        if row is None:
            return None
        return GlobalPersonSessionSummary(
            global_person_id=row[0], status=row[1], created_at_utc=row[2],
            last_seen_at_utc=row[3], person_id=row[4], display_name=row[5],
            association_status=row[6], last_similarity=row[7], observation_count=row[8],
            associated_cameras=json.loads(row[9]),
        )

    def list_sessions(self, status: str | None = None) -> list[GlobalPersonSessionSummary]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            query = "SELECT global_person_id, status, created_at_utc, last_seen_at_utc, person_id, display_name, association_status, last_similarity, observation_count, associated_cameras_json FROM global_person_sessions "
            params = ()
            if status is not None:
                query += "WHERE status = ? "
                params = (status,)
            query += "ORDER BY last_seen_at_utc DESC"
            rows = connection.execute(query, params).fetchall()
        return [
            GlobalPersonSessionSummary(
                global_person_id=row[0], status=row[1], created_at_utc=row[2],
                last_seen_at_utc=row[3], person_id=row[4], display_name=row[5],
                association_status=row[6], last_similarity=row[7], observation_count=row[8],
                associated_cameras=json.loads(row[9]),
            )
            for row in rows
        ]

    def expire_stale_sessions(self, now_utc: datetime, timeout_seconds: float = 120.0) -> int:
        cutoff = now_utc.timestamp() - timeout_seconds
        expired_count = 0
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                rows = connection.execute(
                    "SELECT global_person_id, last_seen_at_utc FROM global_person_sessions "
                    "WHERE status = 'active'"
                ).fetchall()
                for row in rows:
                    last_seen = datetime.fromisoformat(row[1])
                    if last_seen.timestamp() < cutoff:
                        connection.execute(
                            "UPDATE global_person_sessions SET status = 'expired' WHERE global_person_id = ?",
                            (row[0],),
                        )
                        expired_count += 1
        return expired_count

    def get_alert_policy(self) -> AlertPolicyConfig:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            rows = connection.execute(
                "SELECT candidate_type, enabled, severity, routing, cooldown_seconds FROM alert_policy_rules"
            ).fetchall()
        rules = {}
        for row in rows:
            try:
                ctype = CandidateType(row[0])
                rules[ctype] = AlertPolicyRule(
                    candidate_type=ctype, enabled=bool(row[1]), severity=AlertSeverity(row[2]),
                    routing=AlertRouting(row[3]), cooldown_seconds=float(row[4]),
                )
            except (ValueError, KeyError):
                pass
        for ctype, default_rule in DEFAULT_ALERT_RULES.items():
            if ctype not in rules:
                rules[ctype] = default_rule
        return AlertPolicyConfig(version="1.0", rules=rules)

    def set_alert_policy(self, config: AlertPolicyConfig) -> AlertPolicyConfig:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                for rule in config.rules.values():
                    connection.execute(
                        "INSERT INTO alert_policy_rules (candidate_type, enabled, severity, routing, cooldown_seconds) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(candidate_type) DO UPDATE SET "
                        "enabled=excluded.enabled, severity=excluded.severity, routing=excluded.routing, "
                        "cooldown_seconds=excluded.cooldown_seconds",
                        (rule.candidate_type.value, 1 if rule.enabled else 0, rule.severity.value,
                         rule.routing.value, rule.cooldown_seconds),
                    )
        return self.get_alert_policy()

    def reset_default_alert_policy(self) -> AlertPolicyConfig:
        return self.set_alert_policy(AlertPolicyConfig(version="1.0", rules=DEFAULT_ALERT_RULES))

    def _next_incident_id(self, connection: sqlite3.Connection) -> str:
        row = connection.execute("SELECT count(*) FROM incidents").fetchone()
        count = (row[0] if row else 0) + 1
        return f"INC-{count:04d}"

    def create_or_update_incident(
        self, global_person_id: str, event: EventCandidate, severity: AlertSeverity,
    ) -> tuple[IncidentSummary, bool]:
        """Correlate candidate to an active/open incident for this person or create a new incident."""
        observed_str = event.observed_at_utc.isoformat()
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                # Find an open incident for this person
                row = connection.execute(
                    "SELECT incident_id, highest_severity, created_at_utc, primary_camera "
                    "FROM incidents WHERE global_person_id = ? AND status = 'open' "
                    "ORDER BY updated_at_utc DESC LIMIT 1",
                    (global_person_id,),
                ).fetchone()

                if row is not None:
                    incident_id, current_highest, created_at, primary_cam = (
                        row[0], AlertSeverity(row[1]), row[2], row[3]
                    )
                    # Update highest severity if new event is higher
                    highest = (
                        severity if SEVERITY_ORDER.get(severity, 0) > SEVERITY_ORDER.get(current_highest, 0)
                        else current_highest
                    )
                    connection.execute(
                        "UPDATE incidents SET highest_severity = ?, updated_at_utc = ? WHERE incident_id = ?",
                        (highest.value, observed_str, incident_id),
                    )
                    is_new = False
                else:
                    incident_id = self._next_incident_id(connection)
                    connection.execute(
                        "INSERT INTO incidents (incident_id, global_person_id, status, highest_severity, "
                        "created_at_utc, updated_at_utc, primary_camera) VALUES (?, ?, 'open', ?, ?, ?, ?)",
                        (incident_id, global_person_id, severity.value, observed_str, observed_str, event.camera_id),
                    )
                    is_new = True

                # Record event entry in incident_events (idempotent ON CONFLICT IGNORE)
                connection.execute(
                    "INSERT INTO incident_events (incident_id, event_id, camera_id, candidate_type, "
                    "observed_at_utc, severity, evidence_count) VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(event_id) DO NOTHING",
                    (incident_id, str(event.event_id), event.camera_id, event.candidate_type.value,
                     observed_str, severity.value, len(event.evidence)),
                )

        return self.get_incident(incident_id), is_new

    def get_incident(self, incident_id: str) -> IncidentSummary | None:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            row = connection.execute(
                "SELECT incident_id, global_person_id, status, highest_severity, created_at_utc, "
                "updated_at_utc, primary_camera FROM incidents WHERE incident_id = ?",
                (incident_id,),
            ).fetchone()
            if row is None:
                return None
            events_rows = connection.execute(
                "SELECT event_id, camera_id, candidate_type, observed_at_utc, severity, evidence_count "
                "FROM incident_events WHERE incident_id = ? ORDER BY observed_at_utc",
                (incident_id,),
            ).fetchall()
            reviews_rows = connection.execute(
                "SELECT review_id, incident_id, action, reviewed_by, reviewed_at_utc, "
                "final_severity, reclassified_type, false_positive_reason, notes "
                "FROM incident_reviews WHERE incident_id = ? ORDER BY reviewed_at_utc",
                (incident_id,),
            ).fetchall()

        timeline = [
            IncidentTimelineEntry(
                event_id=UUID(r[0]), camera_id=r[1], candidate_type=CandidateType(r[2]),
                observed_at_utc=r[3], severity=AlertSeverity(r[4]), evidence_count=r[5],
            )
            for r in events_rows
        ]
        reviews = [
            IncidentReviewRecord(
                review_id=r[0], incident_id=r[1], action=ReviewAction(r[2]),
                reviewed_by=r[3], reviewed_at_utc=r[4],
                final_severity=AlertSeverity(r[5]) if r[5] else None,
                reclassified_type=CandidateType(r[6]) if r[6] else None,
                false_positive_reason=r[7], notes=r[8],
            )
            for r in reviews_rows
        ]
        return IncidentSummary(
            incident_id=row[0], global_person_id=row[1], status=row[2],
            highest_severity=AlertSeverity(row[3]), created_at_utc=row[4],
            updated_at_utc=row[5], primary_camera=row[6], timeline=timeline,
            reviews=reviews,
        )

    def _next_review_id(self, connection: sqlite3.Connection) -> str:
        row = connection.execute("SELECT count(*) FROM incident_reviews").fetchone()
        count = (row[0] if row else 0) + 1
        return f"REV-{count:04d}"

    def record_incident_review(
        self, incident_id: str, review: IncidentReviewRequest,
    ) -> IncidentSummary | None:
        now_str = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            with connection:
                incident_row = connection.execute(
                    "SELECT incident_id, highest_severity FROM incidents WHERE incident_id = ?",
                    (incident_id,),
                ).fetchone()
                if incident_row is None:
                    return None

                review_id = self._next_review_id(connection)
                final_sev = review.final_severity.value if review.final_severity else None
                reclass_type = review.reclassified_type.value if review.reclassified_type else None

                connection.execute(
                    "INSERT INTO incident_reviews (review_id, incident_id, action, reviewed_by, "
                    "reviewed_at_utc, final_severity, reclassified_type, false_positive_reason, notes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (review_id, incident_id, review.action.value, review.reviewed_by,
                     now_str, final_sev, reclass_type, review.false_positive_reason, review.notes),
                )

                action_status_map = {
                    ReviewAction.ACKNOWLEDGE: "acknowledged",
                    ReviewAction.CONFIRM: "confirmed",
                    ReviewAction.REJECT: "rejected",
                    ReviewAction.RECLASSIFY: "reclassified",
                }
                new_status = action_status_map.get(review.action, review.action.value)
                if final_sev is not None:
                    connection.execute(
                        "UPDATE incidents SET status = ?, highest_severity = ?, updated_at_utc = ? "
                        "WHERE incident_id = ?",
                        (new_status, final_sev, now_str, incident_id),
                    )
                else:
                    connection.execute(
                        "UPDATE incidents SET status = ?, updated_at_utc = ? WHERE incident_id = ?",
                        (new_status, now_str, incident_id),
                    )

        return self.get_incident(incident_id)

    def list_incidents(self, status: str | None = None) -> list[IncidentSummary]:
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            query = "SELECT incident_id FROM incidents "
            params = ()
            if status is not None:
                query += "WHERE status = ? "
                params = (status,)
            query += "ORDER BY updated_at_utc DESC"
            rows = connection.execute(query, params).fetchall()

        results = []
        for r in rows:
            inc = self.get_incident(r[0])
            if inc is not None:
                results.append(inc)
        return results
