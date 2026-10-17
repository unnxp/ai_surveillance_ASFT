"""Offline gateway scheduling and central protocol boundary tests."""
from datetime import datetime, timezone
from uuid import UUID
import json

import pytest

from common.schemas import CandidateDraft, EventCandidate, CameraHeartbeat

NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)


@pytest.fixture
def api():
    class Interfaces:
        def __getitem__(self, key):
            from clients.central_client import CentralClient, CentralProtocolError, RequestsTransport
            from edge.shared_gateway import CameraRuntime, SharedGateway
            return (CentralClient, CentralProtocolError, RequestsTransport, CameraRuntime, SharedGateway)[key]
    return Interfaces()


class Source:
    def __init__(self, outcome=(True, object())):
        self.outcome = outcome
        self.reads = 0

    def read(self):
        self.reads += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class Processor:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def process(self, camera_id, frame, state):
        self.calls.append((camera_id, state))
        if camera_id == self.fail:
            raise RuntimeError('secret-source-url')
        state[7] = state.get(7, 0) + 1
        return [CandidateDraft(candidate_type='tripwire_crossing', local_track_id=7,
                               confidence=0.8, reason_codes=['crossed'], attributes={'count': state[7]})]


class Sink:
    def __init__(self, fail=None):
        self.events = []
        self.heartbeats = []
        self.fail = fail

    def send_candidate(self, event):
        if self.fail == ('a', 'candidate') and event.camera_id == 'a':
            raise RuntimeError('secret-source-url')
        self.events.append(event)

    def send_heartbeat(self, heartbeat):
        if self.fail == ('a', 'heartbeat') and heartbeat.camera_id == 'a':
            raise RuntimeError('secret-source-url')
        self.heartbeats.append(heartbeat)


def gateway(api, sources=None, processor=None, sink=None, ids=('a', 'b')):
    runtime, shared = api[3:]
    cameras = [runtime(name, source) for name, source in zip(ids, sources or [Source(), Source()])]
    counter = iter(range(1, 100))
    result = shared(cameras, processor or Processor(), sink or Sink(), gateway_id='gateway',
                    gateway_boot_id='boot', configuration_version='cfg1', model_versions={'fake': 'v1'},
                    clock=lambda: NOW, event_id_factory=lambda: UUID(int=next(counter)))
    return result, cameras


def test_shared_processing_is_camera_scoped_and_bounded(api):
    sources, processor, sink = [Source(), Source()], Processor(), Sink()
    runner, cameras = gateway(api, sources, processor, sink)
    assert [source.reads for source in sources] == [0, 0]
    first = runner.run_once()
    sources[1].outcome = (False, None)
    second = runner.run_once()
    assert [source.reads for source in sources] == [2, 2]
    assert [camera for camera, _ in processor.calls] == ['a', 'b', 'a']
    assert cameras[0].state is not cameras[1].state
    assert [camera.state[7] for camera in cameras] == [2, 1]
    assert [(e.camera_id, e.local_track_id, e.sequence_number) for e in sink.events] == [('a', 7, 1), ('b', 7, 1), ('a', 7, 2)]
    assert [(h.camera_id, h.sequence_number) for h in sink.heartbeats] == [('a', 1), ('b', 1), ('a', 2), ('b', 2)]
    event = sink.events[0]
    assert event.event_id == UUID(int=1)
    assert (event.gateway_id, event.gateway_boot_id, event.configuration_version) == ('gateway', 'boot', 'cfg1')
    assert event.observed_at_utc == NOW and event.model_versions == {'fake': 'v1'}
    assert event.confidence == 0.8 and event.reason_codes == ['crossed']
    assert event.attributes == {'count': 1}
    assert (first.events_sent, first.heartbeats_sent) == (2, 2)
    assert (second.events_sent, second.heartbeats_sent) == (1, 2)
    with pytest.raises((AttributeError, TypeError)):
        first.events_sent = 5


@pytest.mark.parametrize('mode,status,stage', [('empty', 'OFFLINE', None), ('source', 'ERROR', 'source'), ('processor', 'ERROR', 'processor'), ('candidate', 'ONLINE', 'candidate'), ('heartbeat', 'ONLINE', 'heartbeat')])
def test_camera_failure_does_not_block_other_camera(api, mode, status, stage):
    source = Source((False, None) if mode == 'empty' else RuntimeError('secret-source-url') if mode == 'source' else (True, object()))
    sink = Sink(('a', mode))
    runner, cameras = gateway(api, [source, Source()], Processor('a' if mode == 'processor' else None), sink)
    result = runner.run_once()
    assert any(e.camera_id == 'b' for e in sink.events)
    assert any(h.camera_id == 'b' for h in sink.heartbeats)
    assert cameras[0].status == status
    assert cameras[0].dropped_frames == (1 if mode == 'empty' else 0)
    expected_heartbeats = [('b', 'ONLINE')] if mode == 'heartbeat' else [('a', status), ('b', 'ONLINE')]
    assert [(h.camera_id, h.status) for h in sink.heartbeats] == expected_heartbeats
    assert result.heartbeats_sent == (1 if mode == 'heartbeat' else 2)
    assert result.events_sent == (2 if mode == 'heartbeat' else 1)
    assert 'secret-source-url' not in repr(result)
    if stage:
        assert [(f.camera_id, f.stage) for f in result.failures] == [('a', stage)]
    else:
        assert result.failures == ()


@pytest.mark.parametrize('ids', [('a', 'a'), ('a', ' '), ('a', ' a ')])
def test_invalid_camera_ids_rejected_before_reads(api, ids):
    sources = [Source(), Source()]
    with pytest.raises(ValueError):
        gateway(api, sources=sources, ids=ids)
    assert [s.reads for s in sources] == [0, 0]


class Response:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def json(self):
        return self.body


class Transport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, path, payload):
        self.calls.append((path, payload))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def messages(api):
    sink = Sink()
    runner, _ = gateway(api, sink=sink)
    runner.run_once()
    return sink.events[0], sink.heartbeats[0]


@pytest.mark.parametrize('status', [200, 201])
def test_client_sends_json_candidate_and_validates_ack(api, status):
    event, _ = messages(api)
    transport = Transport(Response(status, {'event_id': str(event.event_id), 'accepted': True, 'duplicate': status == 200}))
    client = api[0](transport)
    assert transport.calls == []
    ack = client.send_candidate(event)
    assert ack.event_id == event.event_id
    path, payload = transport.calls[0]
    assert path == '/api/v1/events'
    assert payload['event_id'] == str(UUID(int=1))
    assert payload['observed_at_utc'] == '2026-09-27T00:00:00Z'
    json.dumps(payload)


@pytest.mark.parametrize('kind,status,body', [
    ('candidate', 202, {}), ('candidate', 201, {}),
    ('candidate', 201, {'event_id': str(UUID(int=99)), 'accepted': True, 'duplicate': False}),
    ('candidate', 201, {'event_id': str(UUID(int=1)), 'accepted': False, 'duplicate': False}),
    ('heartbeat', 201, {'camera_id': 'a', 'accepted': True}),
    ('heartbeat', 200, {'camera_id': 'b', 'accepted': True}),
    ('heartbeat', 200, {'camera_id': 'a', 'accepted': False}),
    ('heartbeat', 200, {}),
])
def test_client_rejects_invalid_ack(api, kind, status, body):
    event, heartbeat = messages(api)
    client = api[0](Transport(Response(status, body)))
    with pytest.raises(api[1]):
        getattr(client, 'send_' + kind)(event if kind == 'candidate' else heartbeat)


def test_heartbeat_serialization_and_sanitized_transport_error(api):
    _, heartbeat = messages(api)
    transport = Transport(Response(200, {'camera_id': 'a', 'accepted': True}))
    assert api[0](transport).send_heartbeat(heartbeat).accepted
    assert transport.calls[0][0] == '/api/v1/cameras/heartbeat'
    assert transport.calls[0][1]['status'] == 'ONLINE'
    with pytest.raises(api[1]) as error:
        api[0](Transport(RuntimeError('secret-source-url'))).send_heartbeat(heartbeat)
    assert 'secret-source-url' not in str(error.value)


def test_requests_transport_construction_is_inert_and_timeout_bounded(api):
    transport = api[2]('http://localhost:8000///', timeout=3)
    assert transport.base_url == 'http://localhost:8000'
    with pytest.raises(ValueError):
        api[2]('http://localhost:8000', timeout=0)
    with pytest.raises(ValueError):
        api[2]('http://localhost:8000', timeout=float('inf'))
