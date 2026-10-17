# Face Registry and Access Permission Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repository approval rules take precedence: this document does not authorize implementation, tests, or Git changes.

**Goal:** Persist consent-backed Face Registry profiles and synthetic face templates on Central, manage camera-scoped zone permissions, and expose minimal local CRUD without enabling face matching or zone alerts yet.

**Architecture:** Central FastAPI owns the registry and SQLite remains its source of truth. A profile has an opaque person ID, display name, role, explicit consent, one or more versioned 512-dimensional normalized templates, and per-person grants keyed by `(camera_id, zone_id)`. The API returns metadata only; the later face-matching stage will produce identity results for the existing spatial pipeline after an open-set threshold is validated.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, SQLite/WAL, pytest, Ruff; existing local environment only.

**Spec:** [ARCHITECTURE.md](../../../ARCHITECTURE.md), especially §§7.8, 8.3, 9.1–9.3, 10.2–10.3, 11.4, 13, 17–18; [CONTEXT.md](../../../CONTEXT.md) for Registered Identity and Access Permission.

## Global Constraints

- Follow [AGENTS.md](../../../AGENTS.md): obtain separate written approval before code edits, project tests, or Git mutations; preserve unrelated changes.
- This slice is registry/access foundation only. Do not load SCRFD/ArcFace, use real faces or biometric templates, evaluate thresholds, emit zone candidates, add UI, or perform cross-camera association.
- Tests use deterministic synthetic unit vectors and temporary SQLite only. No network, downloads, new dependencies, live devices, real alerts, or external deployment.
- A registered match is not a zone permission. A role is metadata in this slice; a person's explicit grants determine access. Role-wide grants can be designed later if needed.
- `camera_id` and `zone_id` are a pair: equal zone names on different cameras are distinct permissions.
- Do not return or log template vectors, including in validation errors. Revoking consent deletes profile, templates, and grants transactionally.
- Existing events, camera status, Action, Tripwire, and current RestrictedZone abstention behavior must remain unchanged.
- No Git staging or commit without explicit approval beyond the current plan-document approval.

## File Map

| Path | Responsibility |
|---|---|
| `common/identity_schemas.py` (create) | Strict enrollment, template, grant, and metadata-only response contracts. |
| `server/database.py` (modify) | SQLite registry tables and atomic profile/grant/template operations. |
| `server/app.py` (modify) | Local registry CRUD routes and redacted validation responses. |
| `tests/test_face_registry.py` (create) | Synthetic contract, persistence, permissions, API, and privacy regression tests. |
| `README.md` (modify) | State what Week 2C can and cannot do; local API usage without real data examples. |

## Review Focus

1. Missing or false consent must fail contract validation, return HTTP 422, and leave no row (Tasks 1 and 3 tests).
2. Wrong-length, nonfinite, or non-unit template vectors must be rejected without reflecting vector contents in HTTP errors (Tasks 1 and 3 tests).
3. A grant for `(camera_01, storage)` must not authorize `(camera_02, storage)` (Task 2 test).
4. Replacing a profile must replace templates and grants atomically, not accumulate stale grants (Task 2 test).
5. Consent withdrawal must remove profile, templates, and grants; repeat deletion must be 404 (Tasks 2 and 3 tests).

---

### Task 1: Define registry contracts

**Files:** Create `common/identity_schemas.py`; create `tests/test_face_registry.py`.

**Interfaces:**
- `ZoneGrant(camera_id: NonBlankString, zone_id: NonBlankString)`; duplicate pairs in one request are rejected.
- `FaceTemplateInput(model_version: NonBlankString, vector: list[float])`; vector has exactly 512 finite values and L2 norm within `0.99–1.01`.
- `FaceRegistryUpsert(display_name: NonBlankString, role: NonBlankString, consent_confirmed: Literal[True], allowed_zones: list[ZoneGrant], templates: list[FaceTemplateInput])`; at least one template, unknown fields rejected.
- `FaceRegistrySummary(person_id, display_name, role, allowed_zones, template_count, model_versions, consent_status, consent_recorded_at_utc, enrolled_at_utc)` contains no vector or image data; retained profiles have `consent_status="active"`, and withdrawal deletes the record.
- `person_id` is a stable opaque 1–64 character identifier restricted to ASCII letters, digits, `_`, and `-`; display names may contain Thai text.

- [ ] **Step 1: Write failing contract tests** in `tests/test_face_registry.py`: valid synthetic unit vector; missing/false consent; `NaN`, infinity, 511/513 values and non-unit vector; duplicate camera-zone pair; unknown field; Thai display name; summary serialization never contains `vector`.
- [ ] **Step 2: Verify red** with `.\env\Scripts\python.exe -m pytest -o addopts= tests/test_face_registry.py -q` (expected import/validation failure).
- [ ] **Step 3: Implement the four models** and validators in `common/identity_schemas.py`. Reuse `ContractModel`, `NonBlankString`, and `UtcDatetime` from `common/schemas.py` without changing event contracts.
- [ ] **Step 4: Verify green** with the same focused pytest command, then `.\env\Scripts\ruff.exe check common/identity_schemas.py tests/test_face_registry.py`.
- [ ] **Step 5: Commit only if separately authorized**; stage only the two task files.

### Task 2: Persist profiles and decide access

**Files:** Modify `server/database.py`; extend `tests/test_face_registry.py`.

**Interfaces:**
- `Database.upsert_identity(person_id: str, payload: FaceRegistryUpsert) -> tuple[bool, FaceRegistrySummary]`, where the boolean means newly created.
- `Database.list_identities() -> list[FaceRegistrySummary]`, ordered by `person_id`.
- `Database.delete_identity(person_id: str) -> bool`, `False` when absent.
- `Database.check_access(person_id: str, camera_id: str, zone_id: str) -> Literal["authorized", "unauthorized"] | None`; `None` means no registered identity, not an automatic `unknown_person_in_restricted_zone` candidate.

- [ ] **Step 1: Write failing database tests** using `tmp_path`: no files from constructor; tables appear only on `initialize`; insert/list/reopen; grants isolated by camera; no-grant registered person is unauthorized; unknown ID is `None`; replacement removes old templates/grants; deletion removes all three row types and repeat deletion is false. Verify `events` and `camera_status` tables still work.
- [ ] **Step 2: Verify red** with `.\env\Scripts\python.exe -m pytest -o addopts= tests/test_face_registry.py -q`.
- [ ] **Step 3: Add** `registered_identities`, `face_templates`, and `zone_grants` tables to `Database.initialize`; implement the four methods with parameterized SQL, short-lived connections, and one transaction per mutation. Record active consent and its timestamp. Keep vectors in SQLite only, never in list results. Preserve original `enrolled_at_utc` across replacement; delete children and parent in one transaction.
- [ ] **Step 4: Verify green** with the focused pytest command, then `.\env\Scripts\ruff.exe check server/database.py tests/test_face_registry.py`.
- [ ] **Step 5: Commit only if separately authorized**; stage only this task's changed files.

### Task 3: Expose metadata-only Central API

**Files:** Modify `server/app.py`; extend `tests/test_face_registry.py`.

**Interfaces:**
- `PUT /api/v1/face-registry/{person_id}` creates (201) or replaces (200) a profile and returns `FaceRegistrySummary`.
- `GET /api/v1/face-registry` returns `{ "identities": [FaceRegistrySummary, ...] }`.
- `DELETE /api/v1/face-registry/{person_id}` returns 204 after full deletion or 404 when absent.
- Bad path ID, malformed JSON, invalid consent or vector returns a generic 422 response with no submitted vector or display name echoed.

- [ ] **Step 1: Write failing ASGI tests** using the in-process request helper pattern from `tests/test_central_api.py`: create, replace, list, delete, restart persistence, 404 on repeated delete, invalid ID, and invalid consent/vector. Assert successful and list responses contain display names but no vectors; invalid-input responses contain neither the submitted vector nor display name. Confirm existing event and heartbeat endpoints still respond.
- [ ] **Step 2: Verify red** with `.\env\Scripts\python.exe -m pytest -o addopts= tests/test_face_registry.py -q`.
- [ ] **Step 3: Add the three routes** to `create_app` in `server/app.py`. Validate the path ID and parse/validate registry request bodies in the route so FastAPI's default 422 detail cannot echo a biometric vector; return a fixed generic error for invalid registry input. Never include templates in response models or logs.
- [ ] **Step 4: Verify green** with `.\env\Scripts\python.exe -m pytest -o addopts= tests/test_face_registry.py tests/test_central_api.py -q`, then `.\env\Scripts\ruff.exe check common/identity_schemas.py server/database.py server/app.py tests/test_face_registry.py`.
- [ ] **Step 5: Commit only if separately authorized**; stage only this task's changed files.

### Task 4: Document boundary and run regression checks

**Files:** Modify `README.md`; no new application component.

- [ ] **Step 1: Update README status** to Week 2C foundation and describe metadata-only Registry API, synthetic-only verification, per-camera zone grants, and explicit non-capabilities: no face extraction/matching, no validated threshold, no RestrictedZone identity candidate, no real-data enrollment UI.
- [ ] **Step 2: Verify** `.\env\Scripts\python.exe -m pytest -o addopts= tests -q` and `.\env\Scripts\ruff.exe check common/identity_schemas.py server/database.py server/app.py tests/test_face_registry.py`; inspect `git diff --` for the five scoped code/doc paths and `git status --short` to identify unrelated user changes without altering them.
- [ ] **Step 3: Commit only if separately authorized**; stage only approved Week 2C paths.

## Acceptance and Handoff

Week 2C is done when local CRUD and camera-scoped access decisions work with synthetic templates in temporary SQLite, deletion purges all registry data, and the existing Central API regression tests remain green. This does **not** claim face-recognition accuracy or trigger `unauthorized_zone_entry`/`unknown_person_in_restricted_zone`.

The next separately approved slice will add Gateway SCRFD/ArcFace keyframe extraction, quality gating, Central 1:N matching, open-set threshold validation, and explicit identity results for RestrictedZone. The previous `0.3275` estimate came from a closed-set score-average heuristic while legacy runtime uses `0.55`; neither is an approved operating threshold for the new alert path.

## Execution Handoff

The user requested incremental progress and has not approved implementation or Git mutations under this plan. Present this document for review, then request a written, scoped approval for the five file paths and verification commands before using `superpowers:executing-plans` natively. Do not dispatch subagents unless the user explicitly requests delegation.
