"""Owned geometry protocol regressions; physical-model differential is a separate gate."""
from dataclasses import replace
from types import SimpleNamespace as NS
import math
import numpy as np
import pytest
from erc_phase1_solution import pickup_geometry_backend as backend
from erc_phase1_solution import shelf_cradle_geometry as cradle
from erc_phase1_solution.geometry_process_protocol import SampleDelta, query_to_wire, query_from_wire
from erc_phase1_solution.manipulation_node import ManipulationNode
from test_pickup_geometry_process import node, query, corners, measurements
from test_pickup_post_retreat_geometry import draft, pure_owner


def test_bound_tool_operation_has_distinct_wire_identity():
    value = query('pickup_post_retreat_tool')
    assert query_from_wire(query_to_wire(value)) == value
    changed = replace(value, operation='pickup_tool')
    assert changed.input_sha256 != value.input_sha256
    result = SampleDelta(value.request_id, value.epoch, value.source_id, value.model_id,
        value.scene_id, value.input_sha256, True, None, None, False, None)
    assert not result.matches(changed)


@pytest.mark.parametrize('plane', [.6, None])
def test_worker_uses_exact_bound_plane_and_original_sweep(monkeypatch, plane):
    value = pure_owner(); calls = []
    def sweep(owner, front, grasp, start, end, actual_plane, **kwargs):
        calls.append((start.tobytes(), end.tobytes(), actual_plane, kwargs))
        assert owner is value and front == value._pickup_scene['front']
        return 'cradle_robot_shelf_clearance:arm_left_6_link:sample0'
    monkeypatch.setattr(cradle, 'check_cradle_tool_sweep', sweep)
    result = value.evaluate_independent(query(
        'pickup_post_retreat_tool' if plane is not None else 'pickup_tool'))
    assert not result.verdict
    assert result.rejection_update['reason'] == 'cradle_robot_shelf_clearance:arm_left_6_link:sample0'
    assert len(calls) == 1 and calls[0][0] == calls[0][1]
    assert calls[0][2] == plane
    assert set(calls[0][3]) == {'aperture', 'finger_positions', 'right_positions', 'head_positions'}


def test_worker_rejects_unbound_scene_before_advancing_query():
    value = pure_owner(False)
    with pytest.raises(RuntimeError, match='bound shelf scene'):
        value.evaluate_independent(query('pickup_post_retreat_tool'))
    assert value._queries_seen == 0 and value._last_request_id == -1


@pytest.mark.parametrize('plane', [.6 + 1e-12, .6 - 1e-12, True, False, '0.6', float('nan'), float('inf')])
def test_parent_rejects_different_plane_before_queries(plane):
    value = draft()
    value._sequence = lambda *args: pytest.fail('invalid plane enqueued')
    with pytest.raises(backend.GeometryProcessError, match='shelf plane'):
        value.tool_sweep([0., 0., .5], np.zeros(8), np.zeros(8), np.zeros(8), plane, aperture=.017)


@pytest.mark.parametrize('distance,count', [(0., 1), (.04, 61), (1.4, 71)])
def test_post_retreat_tool_preserves_dense_adaptive_exact_grid(distance, count):
    value = draft(); samples = []
    def sequence(values, operation, context):
        assert operation == 'pickup_post_retreat_tool' and not context
        samples.extend(values); return True
    value._sequence = sequence
    start = np.zeros(8); end = start.copy(); end[1] = distance
    assert value.tool_sweep([0., 0., .5], start, start, end, .6, aperture=.017) is None
    expected = [start+(end-start)*f for f in np.linspace(0., 1., count)]
    assert [q.tobytes() for q in samples] == [q.tobytes() for q in expected]


@pytest.mark.parametrize('kind', ['straight', 'corner', 'reverse', 'stationary', 'near'])
def test_parallel_and_serial_route_share_exact_sections(monkeypatch, kind):
    start = np.zeros(8); one = start.copy(); one[1] = .2
    two = start.copy(); two[1] = .4
    route = [start.copy(), one, two]
    if kind == 'corner': two[2] = .1
    if kind == 'reverse': two[1] = .1
    if kind == 'stationary': route = [start.copy(), start.copy()]
    if kind == 'near': two[2] = 2e-10
    value = draft(); serial = []; pooled = []
    def record(target):
        def sweep(*args, **kwargs):
            # Serial receives node; bound backend does not.
            offset = len(args)-5
            front, grasp, first, last, plane = args[offset:]
            target.append((first.tobytes(), last.tobytes(), plane))
            return None
        return sweep
    monkeypatch.setattr(cradle, 'check_cradle_tool_sweep', record(serial))
    value.tool_sweep = record(pooled)
    assert cradle.check_cradle_tool_route(value.node, [0., 0., .5], start, start, route, .6, aperture=.017) is None
    assert value.tool_route([0., 0., .5], start, start, route, .6, aperture=.017) is None
    assert pooled == serial
    assert len(pooled) == (2 if kind in ('corner', 'reverse', 'near') else 1)


def test_tool_sequence_uses_fresh_context_per_sample_and_first_failure_index():
    value = draft(); calls = []
    value.identity = NS(source_id='a'*64, model_id='b'*64)
    def fresh(reference):
        sample = measurements(); sample['stamp_ns'] += len(calls)
        sample['joints']['head_1_joint'] = len(calls)*.0001
        calls.append(sample)
        return sample
    value.node._lift_first_measurements = fresh
    queries = []
    def evaluate(samples, capture, consume):
        for i, sample in enumerate(samples):
            q = capture(i, sample); queries.append(q)
            passed = i < 2
            result = SampleDelta(q.request_id, q.epoch, q.source_id, q.model_id,
                q.scene_id, q.input_sha256, passed, None,
                None if passed else {'reason': 'cradle_robot_shelf_clearance:arm_left_6_link:sample0'}, False, None)
            if not consume(q, result): return False
        return True
    value.pool = NS(epoch=1, scene_id='c'*64, evaluate_sequence=evaluate)
    first = np.zeros(8); last = first.copy(); last[1] = .04
    reason = value.tool_route([0., 0., .5], first, first, [last], .6, aperture=.017)
    assert reason == 'leg0:cradle_robot_shelf_clearance:arm_left_6_link:sample2'
    assert len(queries) == len(calls) == 3
    assert [np.frombuffer(q.head, dtype=np.float64)[0] for q in queries] == [0., .0001, .0002]
    assert value.terminal_failure is None


@pytest.mark.parametrize('fault', ['freshness', 'transport', 'mismatch'])
def test_backend_fault_is_latched_with_original_identity_and_cannot_retry(fault):
    value = draft(); failure = RuntimeError('fresh joint stamp expired')
    if fault == 'transport': failure = backend.GeometryProcessError('worker disconnected')
    value.identity = NS(source_id='a'*64, model_id='b'*64)
    value.node._lift_first_measurements = lambda reference: measurements()
    calls = []
    def evaluate(samples, capture, consume):
        calls.append('evaluate')
        if fault == 'freshness':
            def stale(reference): raise failure
            value.node._lift_first_measurements = stale
        if fault == 'transport': raise failure
        q = capture(0, np.zeros(8))
        return consume(q, None)
    value.pool = NS(epoch=1, scene_id='c'*64, evaluate_sequence=evaluate)
    with pytest.raises(RuntimeError) as caught:
        value._sequence([np.zeros(8)], 'pickup_post_retreat_tool', {})
    if fault != 'mismatch': assert caught.value is failure
    assert value.terminal_failure is caught.value
    assert value.node._pickup_parallel_geometry_terminal_failure is caught.value
    with pytest.raises(RuntimeError) as retry:
        value._sequence([np.zeros(8)], 'pickup_post_retreat_tool', {})
    assert retry.value is caught.value and calls == ['evaluate']


@pytest.mark.parametrize('stage', ['lift', 'carry'])
def test_swallowed_backend_fault_still_closes_and_discards_plan(stage, monkeypatch):
    value = node(); trace = []; failure = RuntimeError('parked context expired')
    class Managed:
        def __init__(self, *args, **kwargs): self.pool = None
        def __enter__(self): trace.append('enter'); return self
        def __exit__(self, *args): trace.append('close')
    monkeypatch.setattr(backend, 'PickupGeometryBackend', Managed)
    def local(*args, **kwargs):
        trace.append('lift')
        if stage == 'lift': value._pickup_parallel_geometry_terminal_failure = failure
        return object()
    def carry(result, owner):
        trace.append('carry'); value._cached_post_retreat_plan = object()
        if stage == 'carry': value._pickup_parallel_geometry_terminal_failure = failure
        return object()
    with pytest.raises(RuntimeError) as caught:
        backend.run_pickup_geometry(value, local, np.zeros(3), np.zeros(8), [],
            scene_reference={}, aperture=.017, post_retreat_plan=carry)
    assert caught.value is failure
    assert trace == (['enter', 'lift', 'close'] if stage == 'lift' else ['enter', 'lift', 'carry', 'close'])
    assert value._cached_post_retreat_plan is None and not value._pickup_parallel_geometry_active
    # A distinct managed invocation does not inherit the previous failure.
    assert backend.run_pickup_geometry(value, lambda *a, **k: 3, np.zeros(3), np.zeros(8), [],
        scene_reference={}, aperture=.017) == 3
    assert value._pickup_parallel_geometry_terminal_failure is None


def test_actual_cartesian_solver_does_not_swallow_validator_fault():
    value = node(); failure = RuntimeError('worker fresh context failed')
    value.position_tolerance = .001; value.orientation_tolerance = .01
    value._current_seed = lambda: np.zeros(8)
    solves = []
    value.chain.solve = lambda *a, **k: solves.append(a) or (np.zeros(8), 0.)
    value._plan_retracted_transition = lambda *a, **k: []
    value._arm_route_cost = lambda *a: 0.
    def validate(*args): raise failure
    with pytest.raises(RuntimeError) as caught:
        ManipulationNode._solve_cartesian_path(value, [np.zeros(3)], [np.eye(3)], .1,
            candidate_validator=validate, first_valid=True)
    assert caught.value is failure and len(solves) == 1


@pytest.mark.parametrize('stage', ['lift', 'carry'])
def test_translated_backend_fault_preserves_original_exception_and_discards_cache(stage, monkeypatch):
    value = node(); trace = []
    failure = ValueError('malformed worker result frame')
    replacement = RuntimeError('No payload-safe supported compact transport route')
    class Managed:
        def __init__(self, *args, **kwargs): self.pool = None
        def __enter__(self): trace.append('enter'); return self
        def __exit__(self, *args): trace.append('close')
    monkeypatch.setattr(backend, 'PickupGeometryBackend', Managed)
    def translate():
        try:
            value._pickup_parallel_geometry_terminal_failure = failure
            raise failure
        except ValueError as error:
            raise replacement from error
    def local(*args, **kwargs):
        trace.append('lift')
        if stage == 'lift': translate()
        return object()
    def carry(result, owner):
        trace.append('carry')
        value._cached_post_retreat_plan = object()
        translate()
    with pytest.raises(ValueError) as caught:
        backend.run_pickup_geometry(value, local, np.zeros(3), np.zeros(8), [],
            scene_reference={}, aperture=.017, post_retreat_plan=carry)
    assert caught.value is failure and caught.value is not replacement
    assert caught.value.__cause__ is None
    assert value._pickup_parallel_geometry_terminal_failure is failure
    assert value._cached_post_retreat_plan is None
    assert not value._pickup_parallel_geometry_active
    assert trace == (['enter', 'lift', 'close'] if stage == 'lift' else
                     ['enter', 'lift', 'carry', 'close'])
    assert value.events[-1][1]['failure_type'] == 'ValueError'
