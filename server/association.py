"""Cross-camera identity fusion and session-scoped anonymous person tracking."""

from datetime import datetime, timezone
from math import isfinite
from typing import Literal
from uuid import UUID

from common.identity_schemas import GlobalPersonSessionSummary
from server.database import Database


class CrossCameraAssociationEngine:
    """Manages anonymous global person sessions (120-second timeout) and topology-aware handoffs."""

    def __init__(self, database: Database, session_timeout_seconds: float = 120.0):
        self.database = database
        self.session_timeout_seconds = session_timeout_seconds
        self._counter: int | None = None

    def _next_global_id(self) -> str:
        if self._counter is None:
            sessions = self.database.list_sessions()
            existing_indices = []
            for s in sessions:
                if s.global_person_id.startswith("GP-"):
                    try:
                        existing_indices.append(int(s.global_person_id.split("-")[1]))
                    except (IndexError, ValueError):
                        pass
            self._counter = max(existing_indices, default=0)
        self._counter += 1
        return f"GP-{self._counter:04d}"

    def correlate(
        self,
        camera_id: str,
        observed_at_utc: datetime,
        local_track_id: str | int | None = None,
        person_id: str | None = None,
        display_name: str | None = None,
        face_similarity: float | None = None,
        event_id: UUID | str | None = None,
    ) -> tuple[str, Literal["matched", "possible_match", "ambiguous", "unknown"], bool]:
        """Correlate an observation to an active global person session or create a new one.

        Returns:
            (global_person_id, association_status, is_cross_camera_handoff)
        """
        # Ensure UTC timezone
        if observed_at_utc.tzinfo is None:
            observed_at_utc = observed_at_utc.replace(tzinfo=timezone.utc)
        else:
            observed_at_utc = observed_at_utc.astimezone(timezone.utc)

        # 1. Expire stale sessions older than 2 minutes (120s)
        self.database.expire_stale_sessions(observed_at_utc, self.session_timeout_seconds)

        active_sessions = self.database.list_sessions(status="active")
        str_track_id = str(local_track_id) if local_track_id is not None else None
        event_id_str = str(event_id) if event_id is not None else None

        # 2. Check if this is the same track continuing within the same camera
        for session in active_sessions:
            if (
                session.associated_cameras
                and session.associated_cameras[-1] == camera_id
                and person_id is not None
                and session.person_id == person_id
            ):
                updated_session = session.model_copy(
                    update={
                        "last_seen_at_utc": observed_at_utc,
                        "observation_count": session.observation_count + 1,
                        "last_similarity": face_similarity if face_similarity is not None else session.last_similarity,
                    }
                )
                self.database.upsert_session(updated_session)
                self.database.record_session_observation(
                    session.global_person_id, camera_id, str_track_id,
                    observed_at_utc.isoformat(), event_id_str,
                )
                return session.global_person_id, session.association_status, False

        # 3. Check for cross-camera handoff based on topology and travel times
        best_candidate: GlobalPersonSessionSummary | None = None
        best_status: Literal["matched", "possible_match", "ambiguous"] = "ambiguous"

        for session in active_sessions:
            if not session.associated_cameras:
                continue
            last_cam = session.associated_cameras[-1]
            if last_cam == camera_id:
                continue

            edge = self.database.get_topology_edge(last_cam, camera_id)
            if edge is None or not edge.enabled:
                continue

            elapsed_seconds = (observed_at_utc - session.last_seen_at_utc).total_seconds()
            if not (edge.min_travel_seconds <= elapsed_seconds <= edge.max_travel_seconds):
                continue

            # Within travel time window: evaluate identity evidence
            if person_id is not None and session.person_id == person_id:
                best_candidate = session
                best_status = "matched"
                break
            elif face_similarity is not None and isfinite(face_similarity):
                if face_similarity >= 0.30:
                    best_candidate = session
                    best_status = "matched"
                    break
                elif 0.225 <= face_similarity < 0.30:
                    if best_candidate is None:
                        best_candidate = session
                        best_status = "possible_match"
                    else:
                        best_status = "ambiguous"
            else:
                # No facial evidence, but topology and travel time match
                if best_candidate is None:
                    best_candidate = session
                    best_status = "possible_match"

        if best_candidate is not None and best_status in {"matched", "possible_match"}:
            cams = list(best_candidate.associated_cameras)
            cams.append(camera_id)
            updated_session = best_candidate.model_copy(
                update={
                    "last_seen_at_utc": observed_at_utc,
                    "associated_cameras": cams,
                    "observation_count": best_candidate.observation_count + 1,
                    "association_status": best_status,
                    "person_id": person_id or best_candidate.person_id,
                    "display_name": display_name or best_candidate.display_name,
                    "last_similarity": face_similarity if face_similarity is not None else best_candidate.last_similarity,
                }
            )
            self.database.upsert_session(updated_session)
            self.database.record_session_observation(
                best_candidate.global_person_id, camera_id, str_track_id,
                observed_at_utc.isoformat(), event_id_str,
            )
            return best_candidate.global_person_id, best_status, True

        # 4. No matching active session -> start a new anonymous session
        new_gid = self._next_global_id()
        assoc_status: Literal["matched", "possible_match", "ambiguous", "unknown"] = (
            "matched" if person_id else "unknown"
        )
        new_session = GlobalPersonSessionSummary(
            global_person_id=new_gid,
            status="active",
            created_at_utc=observed_at_utc,
            last_seen_at_utc=observed_at_utc,
            associated_cameras=[camera_id],
            person_id=person_id,
            display_name=display_name,
            association_status=assoc_status,
            last_similarity=face_similarity,
            observation_count=1,
        )
        self.database.upsert_session(new_session)
        self.database.record_session_observation(
            new_gid, camera_id, str_track_id, observed_at_utc.isoformat(), event_id_str
        )
        return new_gid, assoc_status, False
