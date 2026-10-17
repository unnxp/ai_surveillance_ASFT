"""Cross-camera identity fusion and session-scoped anonymous person tracking."""

from datetime import datetime, timezone
from math import isfinite, sqrt
from typing import Literal
from uuid import UUID

from common.identity_schemas import GlobalPersonSessionSummary, TrackletAck, TrackletSummary
from server.database import Database

# Cosine thresholds for ImageNet EfficientNet-B0 person-crop embeddings (tracklet means).
# Calibrated on the controlled walk videos/test1.mp4 -> test2.mp4: same person 0.65-0.68,
# other person 0.28-0.29. In dense crowds strangers average ~0.65, so appearance alone is
# not discriminative there; topology + travel time is the primary gate and a close
# runner-up yields "ambiguous". A dedicated ReID model (e.g. OSNet) is future work.
APPEARANCE_MATCH = 0.60
APPEARANCE_POSSIBLE = 0.50
AMBIGUITY_MARGIN = 0.03
# Adjacent cameras can see the same person at once (overlapping views, clock skew).
OVERLAP_TOLERANCE_SECONDS = 3.0


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right))
    return max(-1.0, min(1.0, dot / norm)) if norm > 0 else 0.0


def _blend(previous: list[float] | None, new: list[float], weight: int) -> list[float]:
    if previous is None or len(previous) != len(new):
        return list(new)
    mixed = [(p * weight + n) / (weight + 1) for p, n in zip(previous, new, strict=True)]
    norm = sqrt(sum(v * v for v in mixed)) or 1.0
    return [v / norm for v in mixed]


class CrossCameraAssociationEngine:
    """Manages anonymous global person sessions (120-second timeout) and topology-aware handoffs."""

    def __init__(self, database: Database, session_timeout_seconds: float = 120.0,
                 appearance_match: float = APPEARANCE_MATCH,
                 appearance_possible: float = APPEARANCE_POSSIBLE):
        self.database = database
        self.session_timeout_seconds = session_timeout_seconds
        self.appearance_match = appearance_match
        self.appearance_possible = appearance_possible
        self._counter: int | None = None
        # Session-scoped, in-memory only: appearance vectors are never persisted.
        self._appearance: dict[str, list[float]] = {}
        self._tracks: dict[tuple[str, str, int], str] = {}

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
        gateway_boot_id: str | None = None,
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

        # 1b. A track already linked by its appearance tracklet keeps that session.
        if gateway_boot_id is not None and isinstance(local_track_id, int):
            linked = self._tracks.get((camera_id, gateway_boot_id, local_track_id))
            session = next((s for s in active_sessions if s.global_person_id == linked), None)
            if session is not None:
                self.database.upsert_session(session.model_copy(update={
                    "last_seen_at_utc": max(observed_at_utc, session.last_seen_at_utc),
                    "observation_count": session.observation_count + 1,
                    "person_id": person_id or session.person_id,
                    "display_name": display_name or session.display_name,
                }))
                self.database.record_session_observation(
                    session.global_person_id, camera_id, str_track_id,
                    observed_at_utc.isoformat(), event_id_str,
                )
                return session.global_person_id, session.association_status, False

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

    def _rank(self, tracklet: TrackletSummary, exclude: str | None = None):
        """Return (status, similarity, best session or None) for an appearance tracklet."""
        scored: list[tuple[float, GlobalPersonSessionSummary]] = []
        for candidate in self.database.list_sessions(status="active"):
            vector = self._appearance.get(candidate.global_person_id)
            if (vector is None or not candidate.associated_cameras
                    or candidate.global_person_id == exclude):
                continue
            last_camera = candidate.associated_cameras[-1]
            if last_camera == tracklet.camera_id:
                continue
            edge = self.database.get_topology_edge(last_camera, tracklet.camera_id)
            if edge is None or not edge.enabled:
                continue
            elapsed = (tracklet.first_seen_utc - candidate.last_seen_at_utc).total_seconds()
            # min_travel_seconds == 0 declares adjacent/overlapping views: the person may still be
            # visible upstream, so any overlap is allowed if they appeared there first.
            overlapping = (edge.min_travel_seconds == 0 and elapsed < 0
                           and tracklet.first_seen_utc >= candidate.created_at_utc)
            if not (overlapping or edge.min_travel_seconds - OVERLAP_TOLERANCE_SECONDS
                    <= elapsed <= edge.max_travel_seconds):
                continue
            scored.append((_cosine(tracklet.appearance, vector), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < self.appearance_possible:
            return "unknown", (scored[0][0] if scored else None), None
        if len(scored) > 1 and scored[0][0] - scored[1][0] < AMBIGUITY_MARGIN                 and scored[1][0] >= self.appearance_possible:
            return "ambiguous", scored[0][0], None
        status = "matched" if scored[0][0] >= self.appearance_match else "possible_match"
        return status, scored[0][0], scored[0][1]

    def _link(self, tracklet, candidate, status, similarity) -> str:
        gid = candidate.global_person_id
        self.database.upsert_session(candidate.model_copy(update={
            "last_seen_at_utc": max(tracklet.last_seen_utc, candidate.last_seen_at_utc),
            "associated_cameras": [*candidate.associated_cameras, tracklet.camera_id],
            "observation_count": candidate.observation_count + 1,
            "association_status": status,
            "last_similarity": similarity,
        }))
        self._appearance[gid] = _blend(self._appearance[gid], tracklet.appearance, 1)
        self.database.record_session_observation(
            gid, tracklet.camera_id, str(tracklet.local_track_id), tracklet.last_seen_utc.isoformat(), None)
        return gid

    def correlate_tracklet(self, tracklet: TrackletSummary) -> TrackletAck:
        """Link a local track to a session using appearance, gated by topology and travel time.

        Topology alone never links an anonymous tracklet: the appearance cosine must reach
        ``appearance_possible``; a close runner-up makes the result ``ambiguous`` instead.
        A track that is still unlinked is re-scored on later updates with its fuller
        appearance (early "ready" crops are often partial while a person enters the view).
        """
        key = (tracklet.camera_id, tracklet.gateway_boot_id, tracklet.local_track_id)
        self.database.expire_stale_sessions(tracklet.last_seen_utc, self.session_timeout_seconds)
        for gid in list(self._appearance):
            session = self.database.get_session(gid)
            if session is None or session.status != "active":
                del self._appearance[gid]

        known = self._tracks.get(key)
        session = self.database.get_session(known) if known else None
        if session is not None and session.status == "active":
            if (session.associated_cameras == [tracklet.camera_id]
                    and session.association_status in ("unknown", "ambiguous")):
                status, similarity, candidate = self._rank(tracklet, exclude=known)
                if candidate is not None:
                    # Merge: the provisional single-camera session is closed in favour of the match.
                    self.database.upsert_session(session.model_copy(update={"status": "expired"}))
                    self._appearance.pop(known, None)
                    gid = self._link(tracklet, candidate, status, similarity)
                    if tracklet.phase == "ready":
                        self._tracks[key] = gid
                    else:
                        self._tracks.pop(key, None)
                    return TrackletAck(global_person_id=gid, association_status=status,
                                       cross_camera_handoff=True, appearance_similarity=similarity)
            self._appearance[known] = _blend(self._appearance.get(known), tracklet.appearance,
                                             min(session.observation_count, 8))
            self.database.upsert_session(session.model_copy(update={
                "last_seen_at_utc": max(tracklet.last_seen_utc, session.last_seen_at_utc),
                "observation_count": session.observation_count + 1,
            }))
            if tracklet.phase == "end":
                self._tracks.pop(key, None)
            return TrackletAck(global_person_id=known, association_status=session.association_status,
                               cross_camera_handoff=False)

        status, similarity, candidate = self._rank(tracklet)
        if candidate is not None:
            gid, handoff = self._link(tracklet, candidate, status, similarity), True
        else:
            gid, handoff = self._next_global_id(), False
            self.database.upsert_session(GlobalPersonSessionSummary(
                global_person_id=gid, status="active",
                created_at_utc=tracklet.first_seen_utc, last_seen_at_utc=tracklet.last_seen_utc,
                associated_cameras=[tracklet.camera_id], association_status=status,
                last_similarity=similarity, observation_count=1,
            ))
            self._appearance[gid] = list(tracklet.appearance)
            self.database.record_session_observation(
                gid, tracklet.camera_id, str(tracklet.local_track_id),
                tracklet.last_seen_utc.isoformat(), None)
        if tracklet.phase == "ready":
            self._tracks[key] = gid
        return TrackletAck(global_person_id=gid, association_status=status,
                           cross_camera_handoff=handoff, appearance_similarity=similarity)
