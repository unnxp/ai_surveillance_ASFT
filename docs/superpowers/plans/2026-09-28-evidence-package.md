# Week 2E Image Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository `AGENTS.md` takes precedence: this plan is not permission to edit code, run tests, use real footage, or change Git state. Obtain a separate written approval for implementation commands and files. No subagents, commits, branches, or worktrees without their own authorization.

**Goal:** Attach viewable full-frame and person-crop JPEG evidence to real Gateway Event Candidates from two video streams, while preserving metadata-only synthetic demo behavior.

**Architecture:** Gateway encodes bounded JPEGs only when it emits a Candidate, places content hashes and stable evidence IDs in the existing `EventCandidate.evidence` references, sends event metadata first, then uploads bytes separately to Central. Central validates each upload against the event's declared reference, stores files below its local `data/evidence` directory and metadata in SQLite, and serves an event-specific availability view plus image endpoint for the Dashboard. A failed image preparation or upload never suppresses the Event Candidate or the other camera.

**Tech Stack:** Python 3.10+, OpenCV already in the project, Pydantic, FastAPI, Requests, SQLite/WAL, plain JavaScript, pytest, Ruff. No new dependency or model.

**Spec:** [ARCHITECTURE.md](../../../ARCHITECTURE.md) §§7.9, 8.1, 8.3, 13–14 and the approved Week 2E scope in chat. This is an incremental **image-evidence slice**, not the complete §7.9 package: the 3-second pre-trigger/5-second post-trigger clip, retention automation, external notification, Cross-camera ReID, and alert review remain separate work.

## Global Constraints

- Preserve current `schema_version="1.0"`, existing event-ID idempotency, two-camera processing, Action/Tripwire/Face-Zone Candidates, Face Registry, and metadata-only `--demo` behavior.
- Use only synthetic arrays, generated temporary video, mock processors, temporary SQLite/files, and localhost test HTTP. Do not read the repository's real videos, cameras, RTSP, or images of real people; do not download models or dependencies.
- Do not put JPEG bytes, arbitrary client paths, or biometric vectors into event JSON or SQLite. Image files live below Central's `data/evidence`; SQLite keeps event association, kind, hash, size, timestamp, and relative path.
- Keep the canonical stored Event Candidate unchanged after ingestion. Availability is a separate evidence view, so replaying the original metadata remains idempotent even after uploads.
- No external alerts, deployment, network access beyond localhost test HTTP, Git state changes, destructive cleanup, or unrelated formatting.
- The implementation must be separately approved under `AGENTS.md`, including the exact file scope and verification commands below. This document only records the proposed work.

## File Structure

| Path | Responsibility |
|---|---|
| `edge/evidence.py` (new) | Encode bounded JPEG full-frame and valid person crop; return bytes plus `EvidenceReference` with stable IDs, SHA-256, size, and capture time. |
| `edge/shared_gateway.py`, `edge/run_gateway.py` | Preserve the current camera frame through candidate emission; opt in image evidence for real-video mode, not `--demo`; send metadata before bytes and isolate failures by camera/event. |
| `clients/central_client.py` | Raw-byte localhost upload with sanitized errors and acknowledgement validation. |
| `common/schemas.py` | Small `EvidenceUploadAck` contract; retain the existing candidate/reference schemas. |
| `server/evidence_store.py` (new) | Validate declared JPEG content and bounded size; generate server-owned paths; write/read only under the configured evidence root. |
| `server/database.py`, `server/app.py` | Evidence metadata table, declared-reference lookup, idempotent upload endpoint, availability query, and JPEG serving endpoint. |
| `server/static/index.html`, `README.md` | Show image links and incomplete/pending evidence state; document the image-only scope and local-data limitation. |
| `tests/test_evidence_http.py`, `tests/test_evidence_gateway.py` (new) | Synthetic API, storage, client, two-camera, and failure-path tests; existing regression tests remain in scope for verification. |

## Review Focus

1. **Declared but missing bytes:** event remains listed, availability says incomplete, image endpoint returns 404; covered by Task 1 tests.
2. **Retry or conflicting upload ID:** identical bytes are a no-op; changed bytes/reference cannot overwrite stored evidence; covered by Task 1 tests.
3. **Wrong camera/track crop:** two cameras with the same local track ID must produce distinct, correctly clipped crops; covered by Task 2 tests.
4. **Malformed/oversized JPEG or path injection:** reject before persisting; no client-provided filesystem path is trusted; covered by Task 1 and Task 2 tests.
5. **Upload failure or no usable frame:** Candidate and other camera continue, with incomplete evidence visible; no raw path/URL or image bytes enter error text; covered by Task 2 and Task 3 tests.

---

### Task 1: Central Binary Upload and Availability

**Files:** Create `server/evidence_store.py`, `tests/test_evidence_http.py`; modify `server/database.py`, `server/app.py`, `common/schemas.py`.

**Interfaces:** `EvidenceUploadAck(evidence_id: UUID, accepted: bool, duplicate: bool)`; `Database.get_event(event_id: UUID) -> EventCandidate | None`; `Database.record_evidence(event_id: UUID, reference: EvidenceReference) -> Literal["accepted", "duplicate", "conflict"]`; `Database.list_evidence(event_id: UUID) -> list[EvidenceReference]`. `EvidenceStore(root: Path).save(event_id: UUID, declared: EvidenceReference, jpeg: bytes) -> EvidenceReference` owns a deterministic relative path and rejects body size over 2,000,000 bytes, a non-JPEG body, or SHA/size disagreement. It never overwrites an existing file with different bytes; the API checks existing metadata before writing a retry. `PUT /api/v1/events/{event_id}/evidence/{evidence_id}` consumes raw `image/jpeg` bytes and only an ID already declared in that event's `evidence` list; `GET /api/v1/events/{event_id}/evidence` returns each declared reference with availability and sets `evidence_incomplete` when any declared file is missing or the original event marked preparation incomplete; `GET /api/v1/evidence/{evidence_id}` serves only a recorded local image. Successful first upload returns 201, exact retry 200, conflict 409, unknown event/reference 404, invalid body 422, oversized body 413.

- [ ] **Step 1:** Write `tests/test_evidence_http.py` with a synthetic JPEG and event declaring its hash/size; assert pending → uploaded → retrievable, SQLite contains metadata but no JPEG bytes, and canonical event replay still returns duplicate.
- [ ] **Step 2:** Run `./env/Scripts/python.exe -m pytest -o addopts= tests/test_evidence_http.py -q`; expect missing endpoint/module failures (RED).
- [ ] **Step 3:** Add the acknowledgement model, SQLite evidence table/methods, bounded store, and three API routes matching the interfaces above. Derive the default store from `database_path.parent / "evidence"`; perform no filesystem write on import or app construction.
- [ ] **Step 4:** Add API tests for unknown IDs, duplicate versus conflict, incomplete after failed/missing upload, invalid type/content/hash, >2 MB body, and server-owned paths; rerun the focused test and `tests/test_central_api.py` to GREEN.

### Task 2: Gateway Image Preparation and Upload

**Files:** Create `edge/evidence.py`, `tests/test_evidence_gateway.py`; modify `edge/shared_gateway.py`, `edge/run_gateway.py`, `clients/central_client.py`.

**Interfaces:** `EvidenceEncoder(id_factory: Callable[[], UUID] = uuid4).prepare(frame: Any, bbox: list[float] | None, captured_at_utc: datetime) -> tuple[tuple[EncodedEvidence, ...], bool]`, where each `EncodedEvidence` holds an `EvidenceReference` and JPEG bytes, and the boolean marks incomplete preparation. Resize the full frame to a maximum 1,280-pixel side and a crop to a maximum 512-pixel side; JPEG quality 80 and each output no larger than 2,000,000 bytes. Clip the track bbox to frame bounds; invalid/missing bbox omits only the crop. `Transport.put_bytes(path: str, body: bytes, content_type: str) -> TransportResponse`; `CentralClient.upload_evidence(event_id: UUID, item: EncodedEvidence) -> EvidenceUploadAck` validates the returned ID/status and sanitizes transport errors. `SharedGateway(..., evidence_encoder: EvidenceEncoder | None = None)` leaves injected-test and demo behavior unchanged when absent. Real-video CLI mode supplies one encoder; per candidate, use that camera's current frame and the candidate's local track bbox, attach references, send metadata, then upload bytes. A `finish_cycle()` Action Candidate uses that camera's latest frame/state from the same cycle.

- [ ] **Step 1:** Write encoder/client tests for image dimensions, hashes, two distinct camera frames sharing track ID, clipped or absent bbox, invalid frame, raw-byte transport, and sanitized failed upload. Add a Gateway test asserting `send_candidate` precedes `upload_evidence` and that one upload failure leaves metadata and the other camera intact.
- [ ] **Step 2:** Run `./env/Scripts/python.exe -m pytest -o addopts= tests/test_evidence_gateway.py -q`; expect missing encoder/upload interfaces (RED).
- [ ] **Step 3:** Implement the encoder, Requests `PUT` transport, client method, and optional Gateway wiring. For an encoding failure, set only `attributes.evidence_incomplete=true` on that candidate; for an upload failure, record a sanitized `GatewayFailure(stage="evidence", code="upload_failed")` while continuing. Do not attach evidence to `--demo` candidates.
- [ ] **Step 4:** Run focused Gateway/client tests plus `tests/test_shared_gateway.py tests/test_action_http.py tests/test_face_zone_http.py`; confirm GREEN and no regression in event counts or camera isolation.

### Task 3: Two-Camera HTTP Path and Dashboard

**Files:** Extend `tests/test_evidence_gateway.py`, `tests/test_evidence_http.py`; modify `server/static/index.html`, `README.md`.

**Interfaces:** Dashboard uses the existing event list; for an event with declared references, an Evidence control requests `GET /api/v1/events/{event_id}/evidence`, displays `GET /api/v1/evidence/{evidence_id}` images only when available, and shows an incomplete/pending label otherwise. Build DOM nodes and URLs from UUIDs; do not inject event fields as HTML. No automatic external alert or real-image fixture.

- [ ] **Step 1:** Add a generated two-camera-video localhost integration test with a controlled Candidate processor and synthetic frames; assert distinct event/evidence IDs, per-camera image retrieval, and that a failed upload is reported incomplete without losing the event. Add a Dashboard assertion for evidence links/state and no external notification call.
- [ ] **Step 2:** Run `./env/Scripts/python.exe -m pytest -o addopts= tests/test_evidence_gateway.py tests/test_evidence_http.py -q`; expect the UI/integration assertions to fail (RED).
- [ ] **Step 3:** Add the minimal Dashboard Evidence control and document real-video versus synthetic-demo behavior, image-only limitation, local file location, and absence of retention/clip/external alert in `README.md`.
- [ ] **Step 4:** Rerun the focused tests; confirm GREEN.

### Task 4: Final Regression and Review

**Files:** No new file scope; inspect only the files listed above and existing regression tests.

- [ ] **Step 1:** Run `./env/Scripts/python.exe -m pytest -o addopts= tests -q`; record exit status and failures, if any.
- [ ] **Step 2:** Run `./env/Scripts/ruff.exe check common/schemas.py clients/central_client.py edge/evidence.py edge/shared_gateway.py edge/run_gateway.py server/evidence_store.py server/database.py server/app.py tests/test_evidence_http.py tests/test_evidence_gateway.py`; record exit status.
- [ ] **Step 3:** Inspect `git diff --` for scoped tracked files and `git status --short` for new files and unrelated work. Do not stage, commit, run real footage, or delete test artifacts without separate approval.
- [ ] **Step 4:** Report changed files, exact command exit statuses, what remained untested, and the remaining clip/retention/ReID/policy work. Do not describe this image-only slice as the complete Evidence Package.

## Execution Boundary

Plan review and a separate written implementation approval are required before any task checkbox is executed. Approval must name the intended source/test/documentation files, the pytest/Ruff commands, and the expected synthetic temporary SQLite/JPEG artifacts. The current approval covers creation of this plan file only.
