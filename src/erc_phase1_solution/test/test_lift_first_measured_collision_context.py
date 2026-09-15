"""Measured-context propagation/admission; no simulator or robot publishers."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import JointState
from erc_phase1_solution import manipulation_node as manipulation
from erc_phase1_solution import lift_first_extraction as lift
from test_lift_first_integration import _enabled, _official_actuated_node
from test_shutdown import _mocked_pick_trace


NAMES = (*manipulation.RIGHT_ARM_JOINTS, *manipulation.HEAD_JOINTS)


def _feed(node, changes, stamp=1_000_000_000):
    """Exercise the real JointState callback, preserving all other measured joints."""
    values = dict(node.joints)
    values.update(changes)
    message = JointState()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    message.name = list(values)
    message.position = [float(values[n]) for n in message.name]
    message.velocity = [0.] * len(values)
    message.effort = [0.] * len(values)
    node._on_joint_state(message)


def _context(sample):
    return dict(right_positions=tuple(sample['joints'][n] for n in manipulation.RIGHT_ARM_JOINTS),
                head_positions=tuple(sample['joints'][n] for n in manipulation.HEAD_JOINTS),
                stamp_ns=sample['stamp_ns'])


def _recheck_node():
    node = _official_actuated_node()
    node._cancel = threading.Event()
    node.chain = SimpleNamespace(forward=lambda q: np.eye(4), pose_error=lambda a,b: np.zeros(6))
    node.pick_position_tolerance = .0005
    node.pick_orientation_tolerance = .01
    node.lift_first_extraction_lift_m = .005
    node._cached_post_retreat_plan = {'unchanged': object()}
    node.statuses = []
    node._publish_status = lambda event, **fields: node.statuses.append((event, fields))
    node._move_arm_solution = lambda *args: pytest.fail('geometry checking issued an arm command')
    node._command_gripper = lambda *args: pytest.fail('geometry checking issued a hand command')
    _feed(node, dict(zip(NAMES, [0.1 * i for i in range(7)] + [.2, -.28])))
    return node


def _check(node, reference):
    plan = lift.LiftFirstPlan((np.zeros(8),), np.zeros(8), np.zeros((8,3)), {})
    return node._recheck_lift_first_geometry(np.zeros(3), np.zeros(8), plan, object(), reference)


def test_recheck_freezes_one_message_context_despite_new_callbacks_and_preserves_carry_cache(monkeypatch):
    node = _recheck_node()
    original = node._lift_first_measurements()
    expected = _context(original)
    cache = node._cached_post_retreat_plan
    seen = {}
    def validate(*args, **kwargs):
        seen.update(kwargs)
        assert kwargs['right_positions'] == expected['right_positions']
        assert kwargs['head_positions'] == expected['head_positions']
        # New callbacks while geometry runs must not change its fixed snapshot.
        _feed(node, {n: node.joints[n] + .0004 for n in NAMES}, 1_004_000_000)
        with pytest.raises(TypeError):
            kwargs['right_positions'][0] = 9.
        return {'full_shelf_collision_certificate': False}
    monkeypatch.setattr(lift, 'validate_lift_first_route', validate)
    result = _check(node, original)
    assert result['geometry_context'] == expected
    assert result['joints'][NAMES[0]] == pytest.approx(original['joints'][NAMES[0]] + .0004)
    assert node._cached_post_retreat_plan is cache
    assert [e for e, _ in node.statuses] == ['lift_first_geometry_started',
                                            'lift_first_measured_geometry_verified']
    assert node.statuses[-1][1]['geometry_context'] == expected


@pytest.mark.parametrize('name', NAMES)
def test_end_of_geometry_checks_original_and_actual_checked_context(monkeypatch, name):
    node = _recheck_node()
    original = node._lift_first_measurements()
    baseline = original['joints'][name]
    # Both later states are individually within1mrad of original, but differ
    # by1.5mrad from each other. Checking only original would incorrectly pass.
    _feed(node, {name: baseline + .00075}, 1_002_000_000)
    def validate(*args, **kwargs):
        _feed(node, {name: baseline - .00075}, 1_006_000_000)
        return {}
    monkeypatch.setattr(lift, 'validate_lift_first_route', validate)
    with pytest.raises(RuntimeError, match='collision snapshot moved: ' + name):
        _check(node, original)
    assert not any(e == 'lift_first_measured_geometry_verified' for e, _ in node.statuses)


@pytest.mark.parametrize('name', NAMES)
def test_context_rejects_nonfinite_new_measurements(name):
    node = _recheck_node()
    original = node._lift_first_measurements()
    changed = {**original, 'joints': {**original['joints'], name: float('nan')}}
    with pytest.raises(RuntimeError, match='context is invalid'):
        manipulation.ManipulationNode._check_lift_collision_context(changed, _context(original))


@pytest.mark.parametrize('context', [None, {}, {'right_positions': [0.] * 6, 'head_positions': [0.,0.]},
                                    'nonfinite_head'])
def test_missing_or_malformed_context_cannot_be_a_certificate(context):
    node = _recheck_node()
    sample = node._lift_first_measurements()
    if context == 'nonfinite_head':
        context = {**_context(sample), 'head_positions': [float('inf'), 0.]}
    with pytest.raises(RuntimeError, match='collision context'):
        manipulation.ManipulationNode._check_lift_collision_context(sample, context)


def test_context_accepts_bounded_drift_without_rewriting_the_checked_reference():
    node = _recheck_node()
    original = node._lift_first_measurements()
    context = _context(original)
    _feed(node, {n: node.joints[n] + .0009 for n in NAMES})
    manipulation.ManipulationNode._check_lift_collision_context(node._lift_first_measurements(), context)
    assert context == _context(original)


def _volume_node():
    node = object.__new__(manipulation.ManipulationNode)
    node._cancel = threading.Event()
    node.joints = dict(zip(NAMES, [9.] * 9))  # Deliberately unrelated live state.
    node.chain = SimpleNamespace(forward=lambda q: np.eye(4), pose_error=lambda a,b: np.zeros(6))
    node.cartesian_joint_step = .4
    node.carried_orientation_step_limit = .45
    node.carried_transition_samples = 3
    node.carried_maximum_tilt = 1.
    node.carried_supported_jaw_vertical_component = .75
    node.carried_collision_meshes = []
    node._self_collision_cache = {}
    node._static_self_collision_cache = {}
    return node


def test_robot_volume_payload_and_self_meshes_all_receive_fixed_context_despite_live_changes():
    node = _volume_node()
    calls = []
    right, head = tuple(np.arange(7) * .1), (.2, -.28)
    def record(stage, **kwargs):
        calls.append((stage, {k: np.asarray(v).copy() for k,v in kwargs.items()}))
        node.joints.update({n: node.joints[n] + 1. for n in NAMES})
        return {}
    node._collision_link_transforms = lambda q, **kw: record('payload', **kw)
    node._world_collision_surfaces = lambda q, **kw: record('self', **kw)
    book = np.asarray([(x,y,z) for x in (-.08,.08) for y in (-.01,.01) for z in (-.125,.125)])
    first, last = np.zeros(8), np.zeros(8)
    last[1] = .1
    assert node._carried_robot_transition_is_safe(first,last,book,
                                                 right_positions=right,head_positions=head)
    assert sum(stage == 'payload' for stage,_ in calls) == 3
    assert sum(stage == 'self' for stage,_ in calls) >= 3
    for stage, context in calls:
        assert context['right_positions'] == pytest.approx(right)
        assert context['head_positions'] == pytest.approx(head)


def test_explicit_context_cache_reuse_is_independent_of_live_state_but_tracks_both_arrays():
    node = _volume_node()
    calls = []
    node._world_collision_surfaces = lambda q, **kw: calls.append(kw) or {}
    q = np.zeros(8)
    right, head = np.zeros(7), np.zeros(2)
    assert node._robot_self_collision(q,right_positions=right,head_positions=head) is None
    node.joints.update({n: .7 for n in NAMES})
    assert node._robot_self_collision(q,right_positions=right,head_positions=head) is None
    assert len(calls) == 1
    right = right.copy(); right[3] += .01
    assert node._robot_self_collision(q,right_positions=right,head_positions=head) is None
    head = head.copy(); head[1] += .01
    assert node._robot_self_collision(q,right_positions=right,head_positions=head) is None
    assert len(calls) == 3
    # Omitted arguments retain the previous live-feedback behavior.
    assert node._robot_self_collision(q) is None
    assert len(calls) == 4
    assert calls[-1]['right_positions'] == pytest.approx([.7] * 7)
    assert calls[-1]['head_positions'] == pytest.approx([.7] * 2)
    assert len(node._self_collision_cache) == len(node._static_self_collision_cache) == 4


def test_cancellation_during_payload_sampling_stops_before_next_mesh_pass():
    node = _volume_node()
    calls = []
    def payload(*args, **kwargs):
        calls.append('payload'); node._cancel.set(); return None
    node._carried_robot_collision = payload
    node._robot_self_collision = lambda *a, **kw: pytest.fail('self pass after cancellation')
    book = np.asarray([(x,y,z) for x in (-.08,.08) for y in (-.01,.01) for z in (-.125,.125)])
    first, last = np.zeros(8), np.zeros(8); last[1] = .1
    assert not node._carried_robot_transition_is_safe(first,last,book,
                                                     right_positions=(0.,)*7,head_positions=(0.,)*2)
    assert calls == ['payload']


@pytest.mark.parametrize('name', [manipulation.RIGHT_ARM_JOINTS[3], manipulation.HEAD_JOINTS[1]])
def test_new_context_drift_during_contact_probe_blocks_first_lift_and_recovery(monkeypatch, name):
    def configure(node, seen):
        original = node._lift_first_measurements
        baseline = original()['joints'][name]
        checked = original()
        checked = {**checked, 'joints': {**checked['joints'], name: baseline + .00075}}
        checked['geometry_context'] = _context(checked)
        node._recheck_lift_first_geometry = lambda *args: checked
        seen['probe_finished'] = False
        def probe(command, phase, **kwargs):
            if phase == 'before_initial_shelf_lift': seen['probe_finished'] = True
            return True
        node._fresh_retention_probe = probe
        def measure(reference=None):
            data = original(reference)
            if seen['probe_finished']:
                return {**data, 'joints': {**data['joints'], name: baseline - .00075}}
            return data
        node._lift_first_measurements = measure
        seen['moves'] = []
        move = node._move_arm_solution
        node._move_arm_solution = lambda q,t: seen['moves'].append(q[1]) or move(q,t)
        node._recover_closed_pick = lambda **kw: pytest.fail('unmeasured recovery after context drift')
    seen, setup = _enabled(monkeypatch, configure=configure)
    with pytest.raises(RuntimeError, match='pick_recovery_failed'):
        _mocked_pick_trace([.70,0.,1.58], configure_node=setup)
    assert seen['probe_finished']
    assert seen['moves'] == [10,11,12]
    assert seen['node']._held_book_corners is not None
