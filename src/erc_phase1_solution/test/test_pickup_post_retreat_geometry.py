"""Bound shelf constraints, exact ordered grids and continuation ownership."""
from dataclasses import replace
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from erc_phase1_solution import pickup_geometry_backend as backend
from erc_phase1_solution.pickup_geometry_owner import (
    PickupGeometryOwner, POST_RETREAT_SCENE_KIND, validate_pickup_scene,
)
from erc_phase1_solution.manipulation_node import ManipulationNode
from test_pickup_geometry_process import corners, node, scene, query, Chain


def post_scene():
    value = scene()
    value.update(kind=POST_RETREAT_SCENE_KIND, retreat_clearance=.6,
                 post_retreat_shelf_front_x=.6)
    return value


def draft():
    value = node(); value.carried_shelf_retreat_clearance = .6
    return backend.PickupGeometryBackend(value, object(), epoch=1,
        front=[0., 0., .5], grasp=np.zeros(8), attached=corners(), aperture=.017,
        finger_positions=None, reference={}, include_post_retreat=True)


@pytest.mark.parametrize('change', [dict(retreat_clearance=-.01),
    dict(retreat_clearance=True), dict(retreat_clearance=float('nan')),
    dict(retreat_clearance=2.01), dict(post_retreat_shelf_front_x=.6000000001),
    dict(post_retreat_shelf_front_x=True), dict(post_retreat_shelf_front_x=float('inf')),
    dict(front=[.01, 0., .5])])
def test_bound_shelf_requires_finite_exact_perceived_front_plus_retreat(change):
    value = post_scene(); value.update(change)
    with pytest.raises(ValueError): validate_pickup_scene(value)


def test_new_scene_is_distinct_and_old_schema_still_exact():
    assert validate_pickup_scene(scene()) == scene()
    assert validate_pickup_scene(post_scene()) == post_scene()
    wrong = post_scene(); wrong['kind'] = 'pickup_lift_v1'
    with pytest.raises(ValueError): validate_pickup_scene(wrong)


@pytest.mark.parametrize('value', [None, 0, 1, 'false', [], {}])
def test_new_selection_requires_boolean(value):
    with pytest.raises(ValueError):
        backend.checked_pickup_post_retreat_parallel_geometry_enabled(value)


def test_retreat_rebinding_is_detected_before_capture():
    current = draft(); current._verify_bindings()
    current.node.carried_shelf_retreat_clearance += .001
    with pytest.raises(backend.GeometryProcessError, match='retreat binding'):
        current._verify_bindings()


@pytest.mark.parametrize('values', [(None, .575), (.575, None), (.574, .575),
    (.575, .574), (float('nan'), .575), (True, .575)])
def test_constraints_cannot_be_missing_or_different_from_bound_scene(values):
    current = draft()
    current._sequence = lambda *a: pytest.fail('sample dispatched')
    with pytest.raises(backend.GeometryProcessError):
        current.volume(np.zeros(8), np.zeros(8), corners(),
            maximum_payload_x=values[0], maximum_robot_x=values[1])


@pytest.mark.parametrize('distance,count', [(0., 61), (1e-13, 61), (.05, 61), (.24, 17)])
def test_constrained_payload_then_adaptive_body_grid_matches_original(distance, count):
    current = draft(); value = current.node
    value.carried_transition_samples = count
    current.scene['transition_samples'] = count
    local = []
    def payload(q, world, **kwargs):
        assert kwargs == {'maximum_robot_x': .575}
        local.append(('pickup_post_retreat_payload', q.copy()))
    value._carried_robot_collision = payload
    value._robot_self_collision = lambda q, **kwargs: local.append(('pickup_body', q.copy())) or None
    first, last = np.zeros(8), np.zeros(8); last[1] = distance
    assert ManipulationNode._carried_volume_transition_is_safe(value, first, last, corners(),
        maximum_payload_x=.575, maximum_robot_x=.575)
    remote = []
    def sequence(samples, operation, context):
        remote.extend((operation, q.copy()) for q in samples); return True
    current._sequence = sequence
    assert current.volume(first, last, corners(), maximum_payload_x=.575, maximum_robot_x=.575)
    assert [x[0] for x in local] == [x[0] for x in remote]
    assert len(local) == len(remote)
    assert all(a.tobytes() == b.tobytes() for (_, a), (_, b) in zip(local, remote))


def test_constrained_payload_failure_never_dispatches_body():
    current = draft(); calls = []
    current._sequence = lambda samples, operation, context: calls.append(operation) or False
    assert not current.volume(np.zeros(8), np.full(8, .02), corners(),
        maximum_payload_x=.575, maximum_robot_x=.575)
    assert calls == ['pickup_post_retreat_payload']


def pure_owner(new_scene=True):
    value = object.__new__(PickupGeometryOwner)
    value.chain = Chain(); value._epoch = 1; value._scene_id = 'c'*64
    value.assets = SimpleNamespace(source_id='a'*64, model_id='b'*64)
    value._last_request_id = -1; value._queries_seen = 0; value.maximum_queries = 100
    value._cancel = threading.Event(); value._pickup_scene = post_scene() if new_scene else scene()
    value._held_book_corners = corners(); value.carried_maximum_tilt = .6
    value.carried_supported_jaw_vertical_component = .7; value.carried_shelf_margin = .025
    return value


def test_constrained_operation_requires_new_scene_even_when_cancelled():
    value = pure_owner(False); value._cancel.set()
    with pytest.raises(RuntimeError, match='bound shelf scene'):
        value.evaluate_independent(query('pickup_post_retreat_payload'))
    assert value._queries_seen == 0


@pytest.mark.parametrize('axis,expected', [('payload', 'pickup_post_retreat_payload_shelf'),
    ('robot', 'robot_shelf:head_1_link'), ('none', None)])
def test_worker_preserves_payload_shelf_then_original_robot_predicate(axis, expected):
    value = pure_owner(); calls = []
    def robot(q, world, **kwargs):
        assert kwargs['maximum_robot_x'] == .575
        calls.append(kwargs)
        return 'robot_shelf:head_1_link' if axis == 'robot' else None
    value._carried_robot_collision = robot
    q = query('pickup_post_retreat_payload')
    if axis == 'payload':
        position = np.zeros(8); position[1] = 1.
        q = replace(q, sample=replace(q.sample, q=position.tobytes()))
    result = value.evaluate_independent(q)
    assert result.verdict == (expected is None)
    assert result.rejection_update == (None if expected is None else {'reason': expected})
    assert len(calls) == (0 if axis == 'payload' else 1)


@pytest.mark.parametrize('failure', [None, 'lift', 'carry', 'close'])
def test_continuation_runs_before_close_and_failed_lifetime_discards_cached_plan(failure, monkeypatch):
    value = node(); value._cached_post_retreat_plan = 'old'; trace = []
    class Managed:
        def __init__(self, *args, **kwargs):
            assert kwargs['include_post_retreat'] is True
            assert value._pickup_parallel_geometry_active
            self.pool = None
        def __enter__(self): trace.append('enter'); return self
        def __exit__(self, *args):
            trace.append('close')
            if failure == 'close': raise RuntimeError('readmission failed')
    monkeypatch.setattr(backend, 'PickupGeometryBackend', Managed)
    lift_result = object(); carry_result = object()
    def lift(*args, **kwargs):
        trace.append('lift')
        if failure == 'lift': raise RuntimeError('lift failure')
        return lift_result
    def carry(result, pool):
        trace.append('carry'); assert result is lift_result and isinstance(pool, Managed)
        value._cached_post_retreat_plan = carry_result
        if failure == 'carry': raise RuntimeError('carry failure')
        return carry_result
    def call():
        return backend.run_pickup_geometry(value, lift, np.zeros(3), np.zeros(8), [],
            scene_reference={}, aperture=.017, post_retreat_plan=carry)
    if failure:
        with pytest.raises(RuntimeError): call()
        assert value._cached_post_retreat_plan is None
    else:
        assert call() == (lift_result, carry_result)
        assert value._cached_post_retreat_plan is carry_result
    assert trace == (['enter', 'lift', 'close'] if failure == 'lift' else ['enter', 'lift', 'carry', 'close'])
    assert not value._pickup_parallel_geometry_active
    assert value._pickup_parallel_geometry_epoch == 1
    assert value.events[-1][1]['passed'] == (failure is None)


def test_disabled_continuation_uses_ordinary_planner_without_backend_keyword(monkeypatch):
    value = node(); value.pickup_parallel_geometry_enabled = False
    monkeypatch.setattr(backend, 'PickupGeometryBackend', lambda *a, **k: pytest.fail('pool started'))
    marker = object(); observed = []
    def lift(*args, **kwargs): observed.append(kwargs); return marker
    def carry(result, pool): assert result is marker and pool is None; return 7
    assert backend.run_pickup_geometry(value, lift, [], [], [], scene_reference={},
        option=3, post_retreat_plan=carry) == (marker, 7)
    assert observed == [{'option': 3}]


@pytest.mark.parametrize('rejection', [None, 'volume', 'gravity'])
def test_route_delegates_volume_in_order_and_retains_serial_gravity_gate(rejection):
    value = node(); calls = []
    first = np.zeros(8); last = first.copy(); last[1] = .1
    def volume(start, end, attached, **kwargs):
        calls.append('volume'); assert kwargs == {'maximum_payload_x': .575, 'maximum_robot_x': .575}
        return rejection != 'volume'
    value._gravity_supported_transition_is_safe = lambda *a: calls.append('gravity') or rejection != 'gravity'
    result = ManipulationNode._plan_carried_joint_route(value, first, [last], corners(),
        post_retreat_shelf_front_x=.6, require_gravity_support=True,
        geometry_backend=SimpleNamespace(volume=volume))
    assert calls == (['volume'] if rejection == 'volume' else ['volume', 'gravity'])
    assert (result is None) == (rejection is not None)
    if result is not None: assert np.array_equal(result[-1], last)


def test_unsupported_shelf_branch_rejected_even_for_empty_goals():
    with pytest.raises(ValueError, match='post-retreat'):
        ManipulationNode._plan_carried_joint_route(node(), np.zeros(8), [], corners(),
            shelf_front_x=.6, geometry_backend=object())


@pytest.mark.parametrize('pooled', [False, True])
def test_high_row_planning_preserves_section_order_and_compact_fallback(pooled):
    value = node(); value.carried_shelf_retreat_clearance = .6
    value.carried_navigation_radius_limit = .5
    vector = lambda marker: np.r_[0., marker, np.zeros(6)]
    value._solve_post_retreat_clearance_extension = lambda q: [vector(1.)]
    value._solve_carried_cradle = lambda q: vector(2.)
    value._solve_supported_post_retreat_staging = lambda q: ([], [vector(3.)], [vector(4.)])
    powers = []
    def compact(q, *, shoulder_progress_power=1.):
        powers.append(shoulder_progress_power); return [vector(5.+shoulder_progress_power)]
    value._supported_compact_goals = compact
    value._carried_navigation_radius = lambda *a: .4
    calls = []; handle = object()
    def route(start, goals, attached, **kwargs):
        calls.append((float(goals[0][1]), kwargs))
        assert kwargs['post_retreat_shelf_front_x'] == .6
        assert ('geometry_backend' in kwargs) == pooled
        if pooled: assert kwargs['geometry_backend'] is handle
        return None if goals[0][1] == 6. else goals
    value._plan_carried_joint_route = route
    options = {'geometry_backend': handle} if pooled else {}
    result = ManipulationNode._plan_carried_return(value, [0., 0., 1.5], np.zeros(8),
        np.zeros(8), np.eye(3), 0., **options)
    assert [marker for marker, _ in calls] == [1., 2., 3., 4., 6., 5.8]
    assert [kwargs.get('require_gravity_support', False) for _, kwargs in calls] == [False, False, True, True, True, True]
    assert powers == [1., .8]
    assert result[:4] == ([], [], None, [])
    assert [phase for q, phase in value._cached_post_retreat_plan['legs']] == [
        'post_retreat_clearance_extension', 'post_retreat_cradle_roll',
        'supported_cradle_lowering', 'supported_cradle_retraction', 'compact_transport']
    assert value._cached_post_retreat_plan['compact_shoulder_progress_power'] == .8
