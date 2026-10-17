"""Bounded offline-friendly scheduling with isolated camera state."""
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic
from typing import Any, Protocol
from uuid import UUID

from common.schemas import CameraHeartbeat, CameraStatus, CandidateDraft, EventCandidate
from edge.evidence import EvidenceEncoder


class FrameSource(Protocol):
    def read(self) -> tuple[bool, Any]: ...


class CandidateProcessor(Protocol):
    def process(self, camera_id: str, frame: Any, state: dict) -> Iterable[CandidateDraft]: ...


class CentralSink(Protocol):
    def send_candidate(self, event: EventCandidate) -> Any: ...
    def upload_evidence(self, event_id: UUID, item: Any) -> Any: ...
    def send_heartbeat(self, heartbeat: CameraHeartbeat) -> Any: ...


@dataclass
class CameraRuntime:
    camera_id: str
    source: FrameSource
    state: dict = field(default_factory=dict)
    event_sequence: int = 0
    heartbeat_sequence: int = 0
    frames_read: int = 0
    dropped_frames: int = 0
    status: CameraStatus = CameraStatus.OFFLINE


@dataclass(frozen=True)
class GatewayFailure:
    camera_id: str
    stage: str
    code: str


@dataclass(frozen=True)
class GatewayRunResult:
    events_sent: int
    heartbeats_sent: int
    failures: tuple[GatewayFailure, ...]


class SharedGateway:
    def __init__(self, cameras: Iterable[CameraRuntime], processor: CandidateProcessor,
                 sink: CentralSink, *, gateway_id: str, gateway_boot_id: str,
                 configuration_version: str, model_versions: dict[str, str],
                 clock: Callable[[], datetime], event_id_factory: Callable[[], UUID],
                 evidence_encoder: EvidenceEncoder | None = None,
                 preview_callback: Callable[[dict[str, Any], dict[str, dict]], None] | None = None):
        self.cameras = tuple(cameras)
        names = [camera.camera_id.strip() for camera in self.cameras]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError('Camera IDs must be nonblank and unique')
        for camera, name in zip(self.cameras, names, strict=True):
            camera.camera_id = name
        self.processor, self.sink = processor, sink
        self.gateway_id, self.gateway_boot_id = gateway_id, gateway_boot_id
        self.configuration_version = configuration_version
        self.model_versions = dict(model_versions)
        self.clock, self.event_id_factory = clock, event_id_factory
        self.evidence_encoder = evidence_encoder
        self.preview_callback = preview_callback

    def _envelope(self, camera, sequence, observed):
        return dict(schema_version='1.0', camera_id=camera.camera_id,
                    gateway_id=self.gateway_id, gateway_boot_id=self.gateway_boot_id,
                    sequence_number=sequence, observed_at_utc=observed,
                    configuration_version=self.configuration_version,
                    model_versions=self.model_versions)

    def _clear_processor_camera(self, camera_id: str, failures: list[GatewayFailure]) -> None:
        clear = getattr(self.processor, 'clear_camera', None)
        if clear is not None:
            try:
                clear(camera_id)
            except Exception:
                failures.append(GatewayFailure(camera_id, 'processor', 'cleanup_failed'))

    def run_once(self) -> GatewayRunResult:
        events_sent = heartbeats_sent = 0
        failures = []
        drafts_by_camera: dict[str, list[CandidateDraft]] = {camera.camera_id: [] for camera in self.cameras}
        cameras_by_id = {camera.camera_id: camera for camera in self.cameras}
        observed_by_camera: dict[str, datetime] = {}
        frames_by_camera: dict[str, Any] = {}
        for camera in self.cameras:
            try:
                available, frame = (False, None) if getattr(camera.source, 'finished', False) else camera.source.read()
            except Exception:
                camera.status = CameraStatus.ERROR
                failures.append(GatewayFailure(camera.camera_id, 'source', 'read_failed'))
                self._clear_processor_camera(camera.camera_id, failures)
            else:
                if not available:
                    camera.status = CameraStatus.OFFLINE
                    if getattr(camera.source, 'finished', False):
                        camera.state.clear()
                        self._clear_processor_camera(camera.camera_id, failures)
                    if not getattr(camera.source, 'finished', False):
                        camera.dropped_frames += 1
                else:
                    camera.frames_read += 1
                    observed_by_camera[camera.camera_id] = self.clock()
                    frames_by_camera[camera.camera_id] = frame
                    try:
                        timed = getattr(self.processor, 'process_timed', None)
                        if timed is not None:
                            timestamp = getattr(camera.source, 'timestamp_seconds', None)
                            if timestamp is None:
                                timestamp = monotonic()
                            result = timed(camera.camera_id, frame, camera.state, timestamp)
                        else:
                            result = self.processor.process(camera.camera_id, frame, camera.state)
                        drafts_by_camera[camera.camera_id].extend(
                            CandidateDraft.model_validate(draft) for draft in result)
                        camera.status = CameraStatus.ONLINE
                    except Exception:
                        camera.status = CameraStatus.ERROR
                        failures.append(GatewayFailure(camera.camera_id, 'processor', 'processing_failed'))
                        self._clear_processor_camera(camera.camera_id, failures)

        finish = getattr(self.processor, 'finish_cycle', None)
        if finish is not None:
            try:
                camera_ids = drafts_by_camera.keys()
                for camera_id, draft in finish():
                    if camera_id not in camera_ids:
                        raise ValueError('Unknown action camera')
                    if cameras_by_id[camera_id].status == CameraStatus.ONLINE:
                        drafts_by_camera[camera_id].append(CandidateDraft.model_validate(draft))
            except Exception:
                failures.append(GatewayFailure('gateway', 'action', 'inference_failed'))
                for camera in self.cameras:
                    if camera.status == CameraStatus.ONLINE:
                        camera.status = CameraStatus.DEGRADED

        if self.preview_callback is not None:
            try:
                self.preview_callback(frames_by_camera, {c.camera_id: c.state for c in self.cameras})
            except Exception:
                pass

        for camera in self.cameras:
            observed = observed_by_camera.get(camera.camera_id)
            if observed is None:
                observed = self.clock()
            for draft in drafts_by_camera[camera.camera_id]:
                camera.event_sequence += 1
                evidence_items = ()
                if self.evidence_encoder is not None:
                    try:
                        observation = camera.state.get(draft.local_track_id)
                        bbox = observation.get('bbox') if isinstance(observation, dict) else None
                        evidence_items, incomplete = self.evidence_encoder.prepare(
                            frames_by_camera.get(camera.camera_id), bbox, observed,
                        )
                    except Exception:
                        evidence_items, incomplete = (), True
                    attributes = dict(draft.attributes)
                    if incomplete:
                        attributes['evidence_incomplete'] = True
                    draft = draft.model_copy(update={
                        'evidence': [item.reference for item in evidence_items],
                        'attributes': attributes,
                    })
                try:
                    event = EventCandidate(**draft.model_dump(),
                                           **self._envelope(camera, camera.event_sequence, observed),
                                           event_id=self.event_id_factory())
                    self.sink.send_candidate(event)
                    events_sent += 1
                except Exception:
                    failures.append(GatewayFailure(camera.camera_id, 'candidate', 'send_failed'))
                    continue
                for item in evidence_items:
                    try:
                        self.sink.upload_evidence(event.event_id, item)
                    except Exception:
                        failures.append(GatewayFailure(camera.camera_id, 'evidence', 'upload_failed'))
            try:
                backlog = getattr(self.processor, 'backlog', None)
                pending = backlog(camera.camera_id) if backlog is not None else 0
                if pending and camera.status == CameraStatus.ONLINE:
                    camera.status = CameraStatus.DEGRADED
                camera.heartbeat_sequence += 1
                heartbeat = CameraHeartbeat(**self._envelope(camera, camera.heartbeat_sequence, observed),
                                            status=camera.status, dropped_frames=camera.dropped_frames,
                                            active_tracks=len(camera.state), action_backlog=pending)
                self.sink.send_heartbeat(heartbeat)
                heartbeats_sent += 1
            except Exception:
                failures.append(GatewayFailure(camera.camera_id, 'heartbeat', 'send_failed'))
        return GatewayRunResult(events_sent, heartbeats_sent, tuple(failures))

    def report_stopped(self) -> GatewayRunResult:
        """Publish final status without reading sources or invoking the processor."""
        sent = 0
        failures = []
        for camera in self.cameras:
            camera.state.clear()
            self._clear_processor_camera(camera.camera_id, failures)
            if camera.status == CameraStatus.OFFLINE:
                continue
            camera.status = CameraStatus.OFFLINE
            camera.heartbeat_sequence += 1
            try:
                heartbeat = CameraHeartbeat(
                    **self._envelope(camera, camera.heartbeat_sequence, self.clock()),
                    status=CameraStatus.OFFLINE, dropped_frames=camera.dropped_frames,
                    active_tracks=0, action_backlog=0,
                )
                self.sink.send_heartbeat(heartbeat)
                sent += 1
            except Exception:
                failures.append(GatewayFailure(camera.camera_id, 'heartbeat', 'send_failed'))
        return GatewayRunResult(0, sent, tuple(failures))
