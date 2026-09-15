"""Pickup backend behavior and exact grid parity with the current live method."""
from dataclasses import replace
import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution import pickup_geometry_backend as backend
from erc_phase1_solution import pickup_geometry_owner as owner_module
from erc_phase1_solution.geometry_process_protocol import (
    SampleDelta, query_from_wire, query_to_wire, encode_frame, decode_frame,
)
from erc_phase1_solution.pure_geometry_owner import GeometryQuery, HEAD_JOINTS
from erc_phase1_solution.pickup_geometry_owner import (
    PickupGeometryQuery, PickupGeometryOwner, validate_pickup_scene,
)
from erc_phase1_solution.motion_profiles import RIGHT_ARM_JOINTS


def corners():
    return np.asarray([(x, y, z) for x in (-.08, .08) for y in (-.01, .01)
                       for z in (-.125, .125)])


def scene():
    return dict(kind='pickup_lift_v1', front=[0., 0., .5], grasp=[0.]*8,
        attached_corners=corners().tolist(), book_dimensions=[.16, .02, .25],
        aperture=.017, finger_positions=None, transition_samples=61,
        maximum_tilt=.6, supported_jaw_vertical_component=.7, shelf_margin=.025)


def query(operation='pickup_payload', index=0):
    sample = GeometryQuery.capture(request_id=index, epoch=1, source_id='a'*64,
        model_id='b'*64, scene_id='c'*64, q=np.zeros(8), right=np.zeros(7),
        head=np.zeros(2), aperture=.017, loaded=True)
    return PickupGeometryQuery(sample, operation)


@pytest.mark.parametrize('operation', sorted(owner_module.KINDS))
def test_pickup_wire_roundtrip_preserves_explicit_operation_and_exact_identity(operation):
    value = query(operation)
    actual = query_from_wire(decode_frame(encode_frame(query_to_wire(value))))
    assert actual == value and actual.input_sha256 == value.input_sha256
    assert value.input_sha256 != value.sample.input_sha256


@pytest.mark.parametrize('operation', [None, False, [], {}, 'place', 'payload'])
def test_unknown_operation_is_never_admitted(operation):
    with pytest.raises(ValueError):
        query(operation).validated()


def test_operation_change_cannot_reuse_an_existing_result():
    first = query()
    changed = replace(first, operation='pickup_body')
    result = SampleDelta(first.request_id, first.epoch, first.source_id,
        first.model_id, first.scene_id, first.input_sha256, True, None, None, False, None)
    assert not result.matches(changed)
    wire = query_to_wire(first)
    wire['pickup_operation'] = 'pickup_body'
    with pytest.raises(ValueError, match='hash mismatch'):
        query_from_wire(wire)


def test_pickup_cannot_be_unloaded_or_nested():
    value = query()
    with pytest.raises(ValueError, match='carried book'):
        replace(value, sample=replace(value.sample, loaded=False)).validated()
    wire = query_to_wire(value)
    wire['sample'] = query_to_wire(value)
    with pytest.raises(ValueError, match='nested'):
        query_from_wire(wire)


@pytest.mark.parametrize('change', [dict(bin_scene={}), dict(table_scene={}),
    dict(kind='place'), dict(front=[0., 1.]), dict(grasp=[0.]*7),
    dict(attached_corners=[[0., 0., 0.]]*8), dict(book_dimensions=[.16, 0., .25]),
    dict(aperture=.07), dict(aperture=True), dict(transition_samples=1),
    dict(transition_samples=61.), dict(maximum_tilt=float('nan')),
    dict(finger_positions={'wrong_joint': 0.}), dict(finger_positions={'gripper_left_finger_joint': float('inf')})])
def test_pickup_scene_rejects_wrong_or_fabricated_registrations(change):
    value = scene(); value.update(change)
    with pytest.raises(ValueError):
        validate_pickup_scene(value)


def test_scene_owns_its_inputs_and_needs_no_bin_or_table():
    original = scene(); admitted = validate_pickup_scene(original)
    original['front'][0] = 999.
    assert admitted['front'][0] == 0.
    assert not {'bin_scene', 'table_scene'} & set(admitted)


class Chain:
    lower = np.full(8, -3.)
    upper = np.full(8, 3.)
    active_names = tuple(str(i) for i in range(8))
    def forward(self, q):
        pose = np.eye(4); pose[:3, 3] = [q[1], q[2], 1.+q[0]]
        return pose
    def pose_error(self, first, last):
        return np.r_[last[:3, 3]-first[:3, 3], np.zeros(3)]


def node():
    result = SimpleNamespace(chain=Chain(), _cancel=threading.Event(),
        _lock=threading.Lock(), carried_transition_samples=61,
        carried_book_dimensions=np.array([.16, .02, .25]), carried_maximum_tilt=.6,
        carried_supported_jaw_vertical_component=.7, carried_shelf_margin=.025,
        cartesian_joint_step=.25, carried_orientation_step_limit=.3,
        right_chain=object(), head_chain=object(), carried_collision_meshes=object(),
        _shelf_cradle_geometry=object(), pickup_parallel_geometry_enabled=True,
        _pickup_parallel_geometry_identity=object(), _pickup_parallel_geometry_epoch=0,
        _pickup_parallel_geometry_active=False, events=[])
    result._publish_status = lambda name, **values: result.events.append((name, values))
    result._attached_book_corners = lambda *args: corners()
    result._resolved_right_positions = lambda values: np.asarray(values)
    result._resolved_head_positions = lambda values: np.asarray(values)
    return result


def draft_backend(value=None):
    value = node() if value is None else value
    return backend.PickupGeometryBackend(value, object(), epoch=1,
        front=[0., 0., .5], grasp=np.zeros(8), attached=corners(), aperture=.017,
        finger_positions=None, reference={'base_pose': np.zeros(3)})


@pytest.mark.parametrize('distance,count', [(0., 61), (1e-13, 61), (.05, 61), (.24, 61), (.24, 17)])
def test_parallel_payload_and_body_grids_match_actual_live_method(distance, count):
    # Import the real method; no AST rewriting or saved copy of that method.
    from erc_phase1_solution.manipulation_node import ManipulationNode
    value = node(); value.carried_transition_samples = count
    local = []
    value._carried_robot_collision = lambda q, *a, **k: local.append(('pickup_payload', q.copy())) or None
    value._robot_self_collision = lambda q, **k: local.append(('pickup_body', q.copy())) or None
    first, last = np.zeros(8), np.zeros(8); last[1] = distance
    assert ManipulationNode._carried_volume_transition_is_safe(value, first, last, corners())
    current = draft_backend(value); remote = []
    def sequence(samples, operation, context):
        remote.extend((operation, q.copy()) for q in samples)
        return True
    current._sequence = sequence
    assert current.volume(first, last, corners())
    assert [item[0] for item in local] == [item[0] for item in remote]
    assert all(a.tobytes() == b.tobytes() for (_, a), (_, b) in zip(local, remote))


def test_payload_rejection_never_starts_body_pass():
    current = draft_backend(); calls = []
    def reject(samples, operation, context):
        calls.append(operation); return False
    current._sequence = reject
    assert not current.volume(np.zeros(8), np.full(8, .02), corners())
    assert calls == ['pickup_payload']


def test_joint_step_rejection_dispatches_no_queries():
    current = draft_backend()
    current._sequence = lambda *a: pytest.fail('queries started')
    assert not current.volume(np.zeros(8), np.ones(8), corners())


@pytest.mark.parametrize('distance,expected', [(0., 1), (.04, 61), (1.4, 71)])
def test_tool_sequence_preserves_original_minimum_adaptive_and_stationary_counts(distance, expected):
    current = draft_backend(); observed = []
    def sequence(samples, operation, context):
        observed.extend(samples); assert operation == 'pickup_tool'; return True
    current._sequence = sequence
    first, last = np.zeros(8), np.zeros(8); last[1] = distance
    assert current.tool_sweep([0., 0., .5], first, first, last, None, aperture=.017) is None
    assert len(observed) == expected
    assert observed[0].tobytes() == first.tobytes()
    assert observed[-1].tobytes() == last.tobytes()


@pytest.mark.parametrize('change', [dict(aperture=.018), dict(finger_positions={}),
    dict(shelf_plane=.8), dict(front=[.01, 0., .5]), dict(grasp=np.ones(8))])
def test_tool_changed_scene_cannot_use_current_pool(change):
    current = draft_backend()
    current._sequence = lambda *a: pytest.fail('queries started')
    kwargs = dict(front=[0., 0., .5], grasp=np.zeros(8), start=np.zeros(8),
                  end=np.zeros(8), shelf_plane=None, aperture=.017)
    kwargs.update(change)
    with pytest.raises(backend.GeometryProcessError):
        current.tool_sweep(**kwargs)


def test_tool_rejection_reports_first_committed_index():
    current = draft_backend()
    current._capture = lambda index, item, **kwargs: query('pickup_tool', index)
    def evaluate(samples, capture, consume):
        for i, sample in enumerate(samples):
            q = capture(i, sample)
            okay = i < 2
            delta = SampleDelta(q.request_id, q.epoch, q.source_id, q.model_id, q.scene_id,
                q.input_sha256, okay, None,
                None if okay else {'reason': 'cradle_tool_floor_clearance:palm:sample0'}, False, None)
            if not consume(q, delta): return False
        return True
    current.pool = SimpleNamespace(evaluate_sequence=evaluate)
    assert not current._sequence([np.zeros(8)]*8, 'pickup_tool', {})
    assert current.last_rejection == 'cradle_tool_floor_clearance:palm:sample2'


def measurements():
    return dict(joints={name: 0. for name in (*RIGHT_ARM_JOINTS, *HEAD_JOINTS)}, stamp_ns=100)


def captured_backend():
    current = draft_backend()
    current.identity = SimpleNamespace(source_id='a'*64, model_id='b'*64)
    current.pool = SimpleNamespace(epoch=1, scene_id='c'*64)
    current.node._lift_first_measurements = lambda reference: measurements()
    return current


def test_every_query_rechecks_fresh_measurements_and_keeps_explicit_measured_context():
    current = captured_backend(); calls = []
    current.node._lift_first_measurements = lambda reference: calls.append(reference) or measurements()
    for i in range(3):
        q = current._capture(i, np.zeros(8), operation='pickup_payload',
            context={'right_positions': np.full(7, .0005)})
        assert np.array_equal(np.frombuffer(q.right, dtype=np.float64), np.full(7, .0005))
    assert len(calls) == 3


def test_stale_measurement_never_creates_a_query():
    current = captured_backend()
    def stale(reference): raise RuntimeError('joint stale')
    current.node._lift_first_measurements = stale
    with pytest.raises(RuntimeError, match='stale'):
        current._capture(0, np.zeros(8), operation='pickup_body', context={})


def test_measured_context_drift_and_model_replacement_fail():
    current = captured_backend()
    with pytest.raises(RuntimeError, match='context moved'):
        current._capture(0, np.zeros(8), operation='pickup_body', context={'head_positions': [.00101, 0.]})
    current.node.chain = Chain()
    with pytest.raises(RuntimeError, match='generation changed'):
        current._capture(1, np.zeros(8), operation='pickup_body', context={})


@pytest.mark.parametrize('absent', [False, True])
def test_disabled_call_preserves_exact_local_arguments_and_result(absent, monkeypatch):
    value = node(); value.pickup_parallel_geometry_enabled = False
    if absent: del value.pickup_parallel_geometry_enabled
    monkeypatch.setattr(backend, 'PickupGeometryBackend', lambda *a, **k: pytest.fail('pool started'))
    first, grasp, route, result, option = (object() for _ in range(5)); calls = []
    def local(*args, **kwargs): calls.append((args, kwargs)); return result
    assert backend.run_pickup_geometry(value, local, first, grasp, route,
                                      scene_reference={}, option=option) is result
    assert calls == [((value, first, grasp, route), {'option': option})]


@pytest.mark.parametrize('invalid', [None, 0, 1, 'true', []])
def test_nonboolean_selection_rejected_before_planner(invalid):
    value = node(); value.pickup_parallel_geometry_enabled = invalid
    with pytest.raises(ValueError, match='Boolean'):
        backend.run_pickup_geometry(value, lambda *a, **k: pytest.fail('planner ran'),
                                   [], [], [], scene_reference={})


def test_disabled_initialization_resolves_nothing():
    value = node(); value.pickup_parallel_geometry_enabled = False
    backend.initialize_pickup_geometry_backend(value, lambda *a: pytest.fail('resolved shares'))
    assert value._pickup_parallel_geometry_identity is None
    assert value._pickup_parallel_geometry_epoch == 0
    assert not value._pickup_parallel_geometry_active


def test_enabled_initialization_reuses_already_verified_place_identity():
    value = node(); identity = object(); value._place_parallel_geometry_identity = identity
    backend.initialize_pickup_geometry_backend(value, lambda *a: pytest.fail('duplicate resolution'))
    assert value._pickup_parallel_geometry_identity is identity


@pytest.mark.parametrize('failure', [None, 'planner', 'exit'])
def test_return_requires_managed_cleanup_and_unique_epoch(failure, monkeypatch):
    value = node(); trace = []
    class Managed:
        def __init__(self, *args, **kwargs):
            assert value._pickup_parallel_geometry_active
            trace.append(('construct', kwargs['epoch'])); self.pool = None
        def __enter__(self): trace.append('enter'); return self
        def __exit__(self, *args):
            trace.append('close')
            if failure == 'exit': raise RuntimeError('end admission failed')
    monkeypatch.setattr(backend, 'PickupGeometryBackend', Managed)
    result = object()
    def local(*args, **kwargs):
        trace.append('plan'); assert isinstance(kwargs['geometry_backend'], Managed)
        if failure == 'planner': raise RuntimeError('planner failed')
        return result
    call = lambda: backend.run_pickup_geometry(value, local, np.zeros(3), np.zeros(8),
                                               [], scene_reference={}, aperture=.017)
    if failure:
        with pytest.raises(RuntimeError): call()
    else:
        assert call() is result
    assert trace == [('construct', 1), 'enter', 'plan', 'close']
    assert not value._pickup_parallel_geometry_active and value._pickup_parallel_geometry_epoch == 1
    assert value.events[-1][1]['passed'] == (failure is None)


def test_concurrent_planning_cannot_start_another_pool(monkeypatch):
    value = node(); value._pickup_parallel_geometry_active = True
    monkeypatch.setattr(backend, 'PickupGeometryBackend', lambda *a, **k: pytest.fail('pool started'))
    with pytest.raises(RuntimeError, match='active'):
        backend.run_pickup_geometry(value, None, [], [], [], scene_reference={}, aperture=.017)


@pytest.mark.parametrize('centre,expected', [([0., 0., 1.], 'torso_base_link'),
    ([1., 0., 1.], 'arm_left_3_link'), ([3., .4, 1.], 'head_1_link'),
    ([4., .2, 1.], 'arm_right_4_link'), ([10., 0., 1.], None)])
def test_shared_payload_predicate_matches_live_body_and_parked_geometry(centre, expected):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    from test_pure_geometry_owner import robot
    value = robot()
    book = corners()+np.asarray(centre)
    context = dict(right_positions=np.zeros(7), head_positions=np.zeros(2))
    actual = value._carried_robot_collision(np.zeros(8), book, **context)
    reference = ManipulationNode._carried_robot_collision(value, np.zeros(8), book, **context)
    assert actual == reference == expected
