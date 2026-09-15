"""Fan-out bounds and existing ordered transport at one, four and eight workers."""
import queue
import sys
from types import SimpleNamespace

import pytest

from erc_phase1_solution import geometry_process_pool as transport
from erc_phase1_solution import installed_geometry_identity as installed
from erc_phase1_solution import place_geometry_backend as place
from erc_phase1_solution import geometry_process_parent as parent
from erc_phase1_solution import pickup_geometry_backend as pickup
from test_geometry_process_protocol import SchedulingPool, capture
from test_place_geometry_backend import node as place_node
from test_pickup_geometry_process import node as pickup_node


@pytest.mark.parametrize('count', range(1, 9))
def test_valid_explicit_counts(count):
    assert transport.checked_geometry_process_workers(count) == count


@pytest.mark.parametrize('count', [None, True, False, 0, -1, 9, 100, 1., 4., '8', [], {}])
def test_invalid_count_is_not_coerced(count):
    with pytest.raises(ValueError, match='integer from 1 to 8'):
        transport.checked_geometry_process_workers(count)


def scheduling(count):
    pool = SchedulingPool()
    pool.workers = [SimpleNamespace(send=lambda value, i=i: pool.sent.append((i, value)))
                    for i in range(count)]
    pool.last_worker_requests = [-1] * count
    return pool


@pytest.mark.parametrize('count', [1, 4, 8])
def test_reverse_arrivals_preserve_input_order_and_bounded_prefix(count):
    pool = scheduling(count); consumed = []; capture_window = []
    def captured(index, item):
        capture_window.append(index - len(consumed) + 1)
        return capture(index, item)
    assert pool.evaluate_sequence(range(19), capture=captured,
        consume=lambda q, d: consumed.append(q.request_id) is None)
    assert consumed == list(range(19))
    assert pool.replies[:count] == list(reversed(range(count)))
    assert max(capture_window) == count
    assert {i for i, value in pool.sent} == set(range(count))


@pytest.mark.parametrize('count', [1, 4, 8])
def test_first_rejection_commits_no_speculative_suffix_and_next_epoch_sequence_is_clean(count):
    pool = scheduling(count); consumed = []
    def consume(query, delta):
        consumed.append(query.request_id)
        return False
    assert not pool.evaluate_sequence(range(20), capture=capture, consume=consume)
    assert consumed == [0] and len(pool.sent) == count
    assert pool.statistics['discarded'] == count - 1
    assert pool.evaluate_sequence([99], capture=capture, consume=lambda q, d: True)
    assert pool.sent[-1][1]['query']['request_id'] == count


@pytest.mark.parametrize('count', [1, 4, 8])
def test_cancellation_after_reply_prevents_all_commits_and_closes(count):
    pool = scheduling(count); consumed = []; reply = pool._reply
    def cancel_after_reply(pending):
        result = reply(pending); pool.cancelled = lambda: True
        return result
    pool._reply = cancel_after_reply
    with pytest.raises(transport.GeometryProcessError):
        pool.evaluate_sequence(range(20), capture=capture,
            consume=lambda q, d: consumed.append(q.request_id) is None)
    assert pool.closed and not pool.active and consumed == []


@pytest.mark.parametrize('count', [1, 4, 8])
def test_cached_samples_still_capture_individually_without_worker_requests(count):
    pool = scheduling(count); captured = []
    def fresh(index, item):
        captured.append(index); return capture(index, item)
    assert pool.evaluate_sequence(range(19), capture=fresh,
        is_cached=lambda q: True, consume=lambda q, d: d is None)
    assert captured == list(range(19)) and pool.sent == []


def fake_transport(monkeypatch, *, fail_start=None, bad_ready=None, bad_finish=None):
    """Run production lifecycle/closure with owned process interfaces, no fork."""
    created = []; killed = []
    class Identity:
        source_id = '1'*64; model_id = '2'*64
        def verify(self): pass
        def to_descriptor(self): return {}
    class Process:
        def __init__(self, pid):
            self.pid = pid; self.returncode = None
            self.stdin = self.stdout = self.stderr = SimpleNamespace(close=lambda: None)
        def poll(self): return self.returncode
        def wait(self, timeout): self.returncode = -15; return self.returncode
    class Worker:
        def __init__(self, number, events, environment):
            self.number = number; self.events = events; self.process = Process(2000+number)
            self.fault = None; self.identity = None; self.identity_capture_error = None
            self.identity_observation = None; self.threads = []; self.outgoing = queue.Queue(2)
            self.stderr_bytes = 0; self.stderr_prefix = b''; created.append(self)
        def start(self):
            if self.number == fail_start: raise RuntimeError('last worker start failed')
        def capture_identity(self):
            self.identity = dict(pid=self.process.pid, parent_pid=9999,
                process_group=self.process.pid, session=self.process.pid,
                start_ticks=123+self.number, boot_id='test-boot', argv=['test-worker'])
        def send(self, value, maximum=None):
            if value['kind'] == 'begin':
                self.binding = dict(token=value['token'], worker=self.number,
                    pid=self.process.pid, source_id=Identity.source_id, model_id=Identity.model_id,
                    epoch=value['epoch'], scene_id=value['scene_id'], geometry_id=value['geometry_id'])
                reply = dict(kind='ready', **self.binding)
                if self.number == bad_ready: reply['model_id'] = '9'*64
            elif value['kind'] == 'finish':
                reply = dict(kind='finished', last_request_id=value['last_request_id'], **self.binding)
                if self.number == bad_finish: reply['last_request_id'] += 1
            else: raise AssertionError('unexpected lifecycle test query')
            self.events.put_nowait((self.number, reply))
    monkeypatch.setattr(installed, 'InstalledGeometryIdentity', Identity)
    monkeypatch.setattr(transport, '_Worker', Worker)
    monkeypatch.setattr(transport, 'os', SimpleNamespace(name='posix', environ={},
        getpid=lambda: 9999, killpg=lambda pid, sig: killed.append(pid)))
    monkeypatch.setattr(transport, 'sys', SimpleNamespace(platform='linux'))
    return Identity(), created, killed


@pytest.mark.parametrize('count', [1, 4, 8, None])
def test_actual_constructor_completion_and_reaping_cover_every_selected_worker(count, monkeypatch):
    identity, created, killed = fake_transport(monkeypatch)
    options = {} if count is None else {'worker_count': count}
    expected_count = 4 if count is None else count
    with transport.GeometryProcessPool(identity, epoch=1, scene={}, geometry_id='4'*64,
            cancelled=lambda: False, **options) as pool:
        assert len(created) == expected_count and not pool.closed
    status = pool.worker_status()
    assert pool.finished and status['children_closed']
    assert status['workers_created'] == status['workers_reaped'] == expected_count
    assert [row['worker'] for row in status['workers']] == list(range(expected_count))
    assert killed == [worker.process.pid for worker in created]
    assert all(row['identity'] is not None and row['reaped'] for row in status['workers'])


@pytest.mark.parametrize('count', [True, 0, 9, 8., '8'])
def test_invalid_pool_count_fails_before_constructing_any_worker(count, monkeypatch):
    identity, created, killed = fake_transport(monkeypatch)
    with pytest.raises(ValueError, match='integer from 1 to 8'):
        transport.GeometryProcessPool(identity, epoch=1, scene={}, geometry_id='4'*64,
            cancelled=lambda: False, worker_count=count)
    assert created == [] and killed == []


@pytest.mark.parametrize('count', [1, 4, 8])
@pytest.mark.parametrize('failure', ['start', 'ready', 'finish'])
def test_failure_at_last_worker_never_leaves_an_owned_child(count, failure, monkeypatch):
    option = {'start': 'fail_start', 'ready': 'bad_ready', 'finish': 'bad_finish'}[failure]
    identity, created, killed = fake_transport(monkeypatch, **{option: count-1})
    with pytest.raises((RuntimeError, transport.GeometryProcessError)):
        with transport.GeometryProcessPool(identity, epoch=1, scene={}, geometry_id='4'*64,
                cancelled=lambda: False, worker_count=count):
            pass
    assert len(created) == count
    assert killed == [worker.process.pid for worker in created]
    assert all(worker.process.returncode is not None for worker in created)


@pytest.mark.parametrize('count, gib, admitted', [(1, 4, True), (4, 4, True),
    (8, 2, True), (8, 3, False), (5, 4, False)])
def test_aggregate_address_space_budget_is_bounded_before_any_spawn(count, gib, admitted, monkeypatch):
    identity, created, killed = fake_transport(monkeypatch)
    def build():
        return transport.GeometryProcessPool(identity, epoch=1, scene={}, geometry_id='4'*64,
            cancelled=lambda: False, worker_count=count, address_space_bytes=gib*1024**3)
    if admitted:
        with build(): pass
        assert len(created) == count
    else:
        with pytest.raises(ValueError, match='aggregate geometry address space'):
            build()
        assert created == [] and killed == []


@pytest.mark.parametrize('count', [1, 4, 8, None])
@pytest.mark.parametrize('operation', ['place', 'pickup'])
def test_node_backends_propagate_selection_and_keep_default_four_call_contract(count, operation, monkeypatch):
    owner = place_node() if operation == 'place' else pickup_node()
    if count is not None: owner.geometry_process_workers = count
    received = []; finished = []
    class Backend:
        pool = None; fallback_reason = None
        def __init__(self, *args, **kwargs): received.append(kwargs)
        def __enter__(self): return self
        def __exit__(self, *args): finished.append(True)
    result = object()
    if operation == 'place':
        monkeypatch.setattr(parent, 'ProcessPlanningBackend', Backend)
        actual = place.plan_node_scene_checked_place(owner, lambda *a, **k: result)
        options = received[0]
    else:
        monkeypatch.setattr(pickup, 'PickupGeometryBackend', Backend)
        actual = pickup.run_pickup_geometry(owner, lambda *a, **k: result,
            [], [], [], scene_reference={}, attached_corners=[], aperture=.04)
        options = received[0].get('pool_limits', {})
    assert actual is result and finished == [True]
    if count in (4, None): assert 'worker_count' not in options
    else: assert options['worker_count'] == count


@pytest.mark.parametrize('operation', ['place', 'pickup'])
def test_invalid_node_count_starts_no_backend_and_reserves_no_epoch(operation, monkeypatch):
    owner = place_node() if operation == 'place' else pickup_node()
    owner.geometry_process_workers = True
    if operation == 'place':
        monkeypatch.setattr(parent, 'ProcessPlanningBackend', lambda *a, **k: pytest.fail('started'))
        with pytest.raises(ValueError): place.plan_node_scene_checked_place(owner, lambda *a, **k: None)
        assert owner._place_parallel_geometry_epoch == 0 and not owner._place_parallel_geometry_active
    else:
        monkeypatch.setattr(pickup, 'PickupGeometryBackend', lambda *a, **k: pytest.fail('started'))
        with pytest.raises(ValueError):
            pickup.run_pickup_geometry(owner, lambda *a, **k: None, [], [], [], scene_reference={})
        assert owner._pickup_parallel_geometry_epoch == 0 and not owner._pickup_parallel_geometry_active


@pytest.mark.parametrize('number, admitted', [(0, True), (3, True), (7, True),
    (-1, False), (8, False), (True, False)])
def test_worker_bootstrap_accepts_only_number_zero_through_seven(number, admitted, monkeypatch):
    from erc_phase1_solution import geometry_process_worker as worker
    class ReachedModels(Exception): pass
    def reached(descriptor): raise ReachedModels()
    monkeypatch.setattr(installed.InstalledGeometryIdentity, 'from_descriptor', staticmethod(reached))
    monkeypatch.setitem(sys.modules, 'ctypes', SimpleNamespace(CDLL=lambda *a, **k:
        SimpleNamespace(prctl=lambda *a: 0)))
    monkeypatch.setitem(sys.modules, 'resource', SimpleNamespace(RLIMIT_AS=1, RLIMIT_CPU=2,
        RLIMIT_CORE=3, setrlimit=lambda *a: None))
    monkeypatch.setattr(worker, 'os', SimpleNamespace(name='posix', getppid=lambda: 100))
    monkeypatch.setattr(worker, 'sys', SimpleNamespace(platform='linux'))
    message = dict(kind='begin', token='a'*64, worker=number, identity={}, epoch=1,
        scene={}, scene_id='3'*64, address_space_bytes=2*1024**3, cpu_seconds=120,
        geometry_id='4'*64, parent_pid=100)
    monkeypatch.setattr(worker, '_receive', lambda maximum: message)
    with pytest.raises(ReachedModels if admitted else ValueError): worker.main()
