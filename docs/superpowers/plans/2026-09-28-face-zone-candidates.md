# Face-Zone Event Candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Workspace `AGENTS.md` takes precedence: do not create commits, branches, worktrees, or use real biometric data without separate approval.

**Goal:** Connect local SCRFD/ArcFace extraction to Central Face Registry matching and per-zone access checks, then show experimental HIGH Restricted Zone event candidates in the local Dashboard.

**Architecture:** The shared Gateway extracts a bounded number of quality-qualified face embeddings from person tracks. Central performs model-version-scoped 1:N cosine matching against consent-backed SQLite templates and returns an explicit identity result plus access decisions for the requested camera zones. The Gateway emits a zone candidate only after stable footpoint membership and two distinct usable face samples agree. `unavailable` never means `unknown`.

**Tech Stack:** Python, ONNX Runtime, OpenCV, FastAPI, Pydantic, SQLite, pytest, Ruff. Use only local existing weights `models/det_10g.onnx` and `models/w600k_r50.onnx`.

**Spec:** [ARCHITECTURE.md](../../../ARCHITECTURE.md) §§7.8, 9.1–9.3, 10.2–10.4, 16.4; [CONTEXT.md](../../../CONTEXT.md). The user specifically approved turning on local face-based zone candidates with a provisional legacy cosine threshold of `0.55`. This is an experimental exception to the spec's validated-threshold requirement, not a claim of open-set accuracy or permission to send external alerts.

## Global Constraints

- Preserve the existing two-camera Gateway, Action, Tripwire, Face Registry CRUD, and SQLite event contracts.
- A missing face, poor-quality face, model mismatch, Central error, or empty gallery returns `unavailable`, not an Unknown alert.
- `unknown` requires a usable face embedding, a nonempty same-version gallery, and a best similarity below `0.55`; a registered match requires similarity at least `0.55`.
- Central alone applies the fixed provisional `0.55` threshold; `configs/face_runtime.yaml` controls only extraction quality and sampling limits. The Dashboard and README must label face-derived events **experimental**; this value is not calibrated on open-set data.
- Query embeddings travel only to localhost Central and are never persisted, logged, echoed in errors, or included in event payloads. Enrolled templates persist only for explicitly consented profiles.
- No model or dependency download, real photo/video test, live camera, production connection, outgoing alert, or deployment. Tests use synthetic vectors/frames, mocks, temporary SQLite, and localhost test HTTP only.
- No Git state changes. Preserve unrelated dirty worktree content and do not format unrelated files.

## File Structure

| Path | Responsibility |
|---|---|
| `common/identity_schemas.py` | Strict match request/response contracts; no query vector in response |
| `server/database.py`, `server/app.py` | Same-version template search and zone-scoped authorization API |
| `clients/central_client.py` | Sanitized localhost match request and response validation |
| `edge/face_pipeline.py`, `configs/face_runtime.yaml` | Local face extraction, quality gate, bounded per-track sampling and result state |
| `edge/detection.py`, `edge/spatial.py`, `edge/run_gateway.py` | Join tracked person observations with identity and zone rules; opt-in CLI |
| `tools/enroll_face.py` | Consent-attested local-photo enrollment CLI; never run on real people during this task |
| `server/static/index.html`, `README.md` | Show experimental candidate/identity context and document use/limits |
| `tests/test_face_matching.py`, `tests/test_face_pipeline.py`, `tests/test_face_zone_http.py`, existing related tests | Synthetic, deterministic contract/integration/regression coverage |

## Review Focus

1. **No face or low quality:** produce `unavailable` and no Unknown event; test in Task 3 and Task 4.
2. **Cross-zone permission bleed:** a grant for `(camera_01, zone_A)` never authorizes zone B or camera 02; test in Task 2 and Task 4.
3. **Repeated cached identity:** one face sample repeated on subsequent frames never counts as two independent confirmations; test in Task 4.
4. **Registry/model mismatch or empty gallery:** abstain rather than mark every person Unknown; test in Task 2.
5. **Biometric leakage on malformed input/error:** responses, event payloads, and logs never echo query vectors; test in Task 1, Task 2, and Task 5.

---

### Task 1: Match Contracts and Local Transport

**Files:** Modify `common/identity_schemas.py`, `clients/central_client.py`; create `tests/test_face_matching.py`.

**Interfaces:** `FaceMatchRequest(camera_id, zone_ids, model_version, vector)` requires one normalized 512-dimensional finite vector and unique zone IDs. `FaceMatchResult(status: registered|unknown|unavailable, person_id?, display_name?, similarity?, zone_access: dict[str, authorized|unauthorized], model_version)` contains no vector. `CentralClient.match_face(request) -> FaceMatchResult` uses `POST /api/v1/face-registry/match` and wraps errors with sanitized `CentralProtocolError`.

- [ ] Write contract/transport tests for valid unit vectors, duplicate zone IDs, malformed vectors, response without vector, and sanitized transport failure.
- [ ] Run `./env/Scripts/python.exe -m pytest -o addopts= tests/test_face_matching.py -q`; verify expected missing-interface failures (RED).
- [ ] Implement only the schema and client method needed by the tests.
- [ ] Rerun the focused tests and Ruff on touched Python files; confirm GREEN.

### Task 2: Central Open-Set Match and Per-Zone Access

**Files:** Modify `server/database.py`, `server/app.py`; extend `tests/test_face_matching.py`.

**Interfaces:** `Database.match_face(request: FaceMatchRequest, threshold: float = 0.55) -> FaceMatchResult` compares only templates with the exact `model_version`, selects each identity's maximum cosine, then the best identity with deterministic `person_id` tie-break. Only a successful registered match returns `person_id`, `display_name`, and per-requested-zone access. Empty same-version gallery is `unavailable`; below-threshold valid match is `unknown`. `POST /api/v1/face-registry/match` validates without echoing malformed vectors.

- [ ] Add synthetic SQLite/API tests for registered authorized, registered unauthorized on another zone/camera, true below-threshold Unknown, empty/mismatched gallery abstention, deterministic tie, and sanitized invalid request.
- [ ] Run focused tests and verify expected failures (RED).
- [ ] Implement the database search and FastAPI endpoint; do not persist query embeddings.
- [ ] Rerun `tests/test_face_matching.py tests/test_face_registry.py tests/test_central_api.py` and Ruff; confirm GREEN.

### Task 3: Local Extraction, Enrollment, and Bounded Scheduling

**Files:** Create `edge/face_pipeline.py`, `configs/face_runtime.yaml`, `tools/enroll_face.py`, `tests/test_face_pipeline.py`; modify `edge/run_gateway.py`, `README.md`.

**Interfaces:** `FaceExtractor` lazily loads the existing local SCRFD and ArcFace ONNX files once, with CPU execution providers explicitly selected in the new adapter, and returns `(normalized_vector, quality)` or no sample from a clipped person crop. `FacePipeline.observe(camera_id, track_id, frame, bbox, eligible)` samples no more often than configured cadence, at most three usable samples per tracklet, and no more than a configured number of tracks per cycle; its result carries a fresh sample ID only after a usable embedding and successful Central match. `clear_camera(camera_id)` purges local track state. `--face` is opt-in and rejected in synthetic demo mode. The enrollment CLI accepts explicit consent, local image path(s), person metadata, and zone grants, then uses the existing Registry PUT; it is not executed with real images in this task.

- [ ] Add mock/synthetic tests for no face, small/blurred face, clipped crop, normalized vector, CPU-only existing-weight loading path, per-camera isolation, rate limit, transport failure abstention, CLI argument/consent validation, and no implicit download.
- [ ] Run focused tests and verify expected failures (RED).
- [ ] Implement the extractor/pipeline/config/CLI and runner wiring without changing default non-face behavior.
- [ ] Rerun focused tests and Ruff; confirm GREEN.

### Task 4: Stable Zone Candidates Across Two Cameras

**Files:** Modify `edge/detection.py`, `edge/spatial.py`; extend `tests/test_spatial.py`; create `tests/test_face_zone_http.py`.

**Interfaces:** Detection gives FacePipeline person crops for `InferenceROI`-eligible tracks before the identity-sensitive zone decision. The observation carries `identity_status`, `identity_sample_id`, matched person context, and zone access keyed by zone ID. Spatial emits `unknown_person_in_restricted_zone` or `unauthorized_zone_entry` only after the configured inside-frame stability and at least two distinct usable face samples agreeing on the kind. A registered authorized person emits no event. Candidate attributes include zone ID, experimental marker, and registered `person_id` only when present; never an embedding.

- [ ] Add tests for authorized/unauthorized/unknown, two zones with different permissions, no zone, no face/unavailable, ROI filtering, repeated sample ID, and two isolated local-video camera streams reaching Central through the Gateway.
- [ ] Run the focused tests and verify expected failures (RED).
- [ ] Implement the minimum detection/spatial integration, preserving Action and Tripwire behavior.
- [ ] Rerun `tests/test_spatial.py tests/test_spatial_http.py tests/test_face_zone_http.py tests/test_action_http.py` and Ruff; confirm GREEN.

### Task 5: Local Display, Documentation, and Final Verification

**Files:** Modify `server/static/index.html`, `README.md`; extend `tests/test_face_zone_http.py` or another scoped existing test.

- [ ] Add a failing UI/API assertion that a face-derived local event displays HIGH and an **experimental** marker, with zone/person context but no vector; no external notification is sent.
- [ ] Verify RED, then implement the minimal Dashboard display and usage/limitation documentation.
- [ ] Verify GREEN with the focused tests.
- [ ] Run `./env/Scripts/python.exe -m pytest -o addopts= tests -q` and `./env/Scripts/ruff.exe check common/identity_schemas.py clients/central_client.py server/database.py server/app.py edge/face_pipeline.py edge/detection.py edge/spatial.py edge/run_gateway.py tools/enroll_face.py tests/test_face_matching.py tests/test_face_pipeline.py tests/test_face_zone_http.py tests/test_spatial.py`.
- [ ] Inspect `git diff --` for tracked scoped paths and `git status --short` for new scoped files and unrelated user changes; report exact commands, exit codes, results, and checks not run. Do not claim an open-set threshold is validated or face alerts are production-ready.

## Completion Boundary

This slice is complete only when the synthetic two-camera path reaches Central and the Dashboard for eligible face-derived zone candidates, while authorized/unavailable cases abstain and existing tests still pass. Real-photo enrollment, threshold calibration with consented identities/unknown impostors, performance benchmark, external notification, review confirmation, cross-camera ReID, and configurable alert policy are separate steps requiring their own scope and approval.

## Execution Ledger

- Ruling: Implement inline in the approved existing workspace without worktree/commit or the skill's ignored SDD scratch directory — `AGENTS.md` explicitly forbids unapproved Git and file changes; cost if wrong: no isolated rollback or independent commit history.
- Pre-flight: Tasks 1→2 share `FaceMatchRequest`/`FaceMatchResult`; Task 2→3 shares `CentralClient.match_face`; Task 3→4 shares fresh sample IDs and per-zone access; Task 4→5 shares event attributes. Interfaces align with the plan; no conflicting producer/consumer names found.
- Task 1: complete — `tests/test_face_matching.py` RED (5 missing-interface failures) → GREEN (5 passed); Ruff exit 0.
- Task 2: complete — Central search/API tests RED (3 expected missing-method/route failures) → GREEN (45 focused/regression tests passed); Ruff exit 0 after import-format correction.
- Task 3: complete — extraction/sampling/enrollment/CLI tests RED (missing modules/flag) → GREEN (7 focused tests passed); local ONNX SCRFD and ArcFace smoke on synthetic blank arrays passed. Runner-to-face wiring was completed with Task 4 because DetectionProcessor did not accept the face dependency until that task.
- Task 4: complete — distinct-sample, per-zone, ROI, two-camera HTTP and `--face` runner tests RED → GREEN (32 related tests passed); Ruff exit 0.
- Task 4 Ruling: allow two agreeing face samples captured earlier in the same tracklet to authorize a later zone decision after stable footpoint entry — otherwise the three-sample cap can be exhausted before a person reaches a zone; cost if wrong: an old track association may carry stale identity into the zone until tracking resets.
- Task 4 Ruling: enforce the face extraction cap per camera frame rather than globally per Gateway cycle, because the approved file scope does not include the Gateway scheduler and the two streams are processed separately — cost if wrong: up to four face attempts may be scheduled in a two-camera cycle with the default cap of two per camera, so measured latency may miss the intended budget.
- Task 5: local Dashboard displays severity, person ID, zone and experimental marker; README documents opt-in operation and limitations. UI/API test RED on missing columns → GREEN in focused suite.
- Final self-review fix: near-unit vectors accepted by the schema made raw dot product exceed 1. Added a failing regression test, changed matching to true normalized cosine, and verified GREEN; cost if wrong without fix: valid match queries could raise a 500 response.
- Final self-review fix: a hard 224-pixel pre-resize on full enrollment photos shrank otherwise usable faces below the quality gate. Added a failing synthetic large-photo test and let SCRFD perform its own 640-pixel preprocessing; test RED→GREEN.
- Final self-review fix: legacy alignment falls back to resizing the entire person crop for degenerate landmarks. Added a failing test and changed the adapter to abstain when affine alignment is invalid; test RED→GREEN.
- Final review: self-review only because agent delegation was not authorized for this task; no independent reviewer or Git commit was used.
