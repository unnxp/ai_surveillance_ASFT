"""Injected central metadata transport; construction performs no I/O."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from common.identity_schemas import FaceMatchRequest, FaceMatchResult
from common.schemas import (
    CameraHeartbeat,
    EventCandidate,
    EventIngestAck,
    EvidenceUploadAck,
    HeartbeatAck,
)

if TYPE_CHECKING:
    from edge.evidence import EncodedEvidence


class TransportResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...


class Transport(Protocol):
    def post(self, path: str, payload: dict) -> TransportResponse: ...
    def put_bytes(self, path: str, body: bytes, content_type: str) -> TransportResponse: ...


class CentralProtocolError(RuntimeError):
    """Sanitized transport or acknowledgement failure."""


class RequestsTransport:
    def __init__(self, base_url: str, timeout: float = 5.0):
        if not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError('Timeout must be finite and within (0, 60] seconds')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._session = None

    def post(self, path: str, payload: dict) -> TransportResponse:
        import requests

        try:
            if self._session is None:
                self._session = requests.Session()
                self._session.trust_env = False
            return self._session.post(
                self.base_url + path, json=payload, timeout=self.timeout, allow_redirects=False
            )
        except Exception:
            raise CentralProtocolError('Central transport failed') from None

    def put_bytes(self, path: str, body: bytes, content_type: str) -> TransportResponse:
        import requests

        try:
            if self._session is None:
                self._session = requests.Session()
                self._session.trust_env = False
            return self._session.put(
                self.base_url + path, data=body, headers={"Content-Type": content_type},
                timeout=self.timeout, allow_redirects=False,
            )
        except Exception:
            raise CentralProtocolError('Central transport failed') from None

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


class CentralClient:
    def __init__(self, transport: Transport):
        self.transport = transport

    def _send(self, path, message, statuses, ack_type):
        try:
            response = self.transport.post(path, message.model_dump(mode='json'))
            if response.status_code not in statuses:
                raise CentralProtocolError('Central status rejected')
            ack = ack_type.model_validate(response.json())
            if not ack.accepted:
                raise CentralProtocolError('Central acknowledgement rejected')
            return ack
        except Exception:
            raise CentralProtocolError('Central request or acknowledgement failed') from None

    def send_candidate(self, event: EventCandidate) -> EventIngestAck:
        ack = self._send('/api/v1/events', event, (200, 201), EventIngestAck)
        if ack.event_id != event.event_id:
            raise CentralProtocolError('Central event acknowledgement mismatch')
        return ack

    def send_heartbeat(self, heartbeat: CameraHeartbeat) -> HeartbeatAck:
        ack = self._send('/api/v1/cameras/heartbeat', heartbeat, (200,), HeartbeatAck)
        if ack.camera_id != heartbeat.camera_id:
            raise CentralProtocolError('Central camera acknowledgement mismatch')
        return ack

    def upload_evidence(self, event_id: UUID, item: EncodedEvidence) -> EvidenceUploadAck:
        try:
            response = self.transport.put_bytes(
                f'/api/v1/events/{event_id}/evidence/{item.reference.evidence_id}',
                item.content, 'image/jpeg',
            )
            if response.status_code not in (200, 201):
                raise CentralProtocolError('Central evidence status rejected')
            ack = EvidenceUploadAck.model_validate(response.json())
            if not ack.accepted or ack.evidence_id != item.reference.evidence_id:
                raise CentralProtocolError('Central evidence acknowledgement mismatch')
            return ack
        except Exception:
            raise CentralProtocolError('Central evidence upload failed') from None

    def match_face(self, request: FaceMatchRequest) -> FaceMatchResult:
        try:
            response = self.transport.post('/api/v1/face-registry/match',
                                           request.model_dump(mode='json'))
            if response.status_code != 200:
                raise CentralProtocolError('Central face match status rejected')
            result = FaceMatchResult.model_validate(response.json())
            if result.model_version != request.model_version:
                raise CentralProtocolError('Central face model version mismatch')
            return result
        except Exception:
            raise CentralProtocolError('Central face match failed') from None
