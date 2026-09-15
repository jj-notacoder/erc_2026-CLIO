"""Current production methods with real timing gates; no physical-speed claim."""
import ast
import copy
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import bin_clearance_timing as clearance
from erc_phase1_solution import additional_arm_timing as extra
from erc_phase1_solution import arm_velocity_admission as velocity
from erc_phase1_solution import place_contact_guard
from erc_phase1_solution.motion_profiles import IK_JOINTS
from erc_phase1_solution.placement_scene_context import measured_scene_context
import test_optional_arm_timing as timing
import test_arm_velocity_admission as sender_fixtures


def load(name, **kwargs):
    return timing.load(name,
        validate_bin_clearance_execution=clearance.validate_execution,
        bin_clearance_timing_input=clearance.timing_input,
        require_bin_clearance_endpoint=clearance.require_endpoint,
        require_bin_clearance_publication_locked=clearance.require_publication_locked,
        retime_admitted_arm_goal=extra.retime_admitted_arm_goal,
        require_retimed_arm_headroom=extra.require_retimed_arm_headroom,
        **kwargs)


IDENTITY = dict(trial_id='trial', placement_attempt_id='place-1',
                target_model='book_col_1_row_2_red')
NOW = 10_000_000_000


def fixture():
    q = np.array([.35, .02, .03, .04, .05, .06, .07, .08])
    legs = [(q.copy(), .8, 'bin_transition'), (q.copy(), 2.8, 'bin_clearance'),
            (q.copy(), .65, 'bin_approach')]
    n, calls, probes = timing.retained_node()
    n.bin_clearance_timing_enabled = True
    n._lock = threading.Lock()
    n._adaptive_command_guard = threading.RLock
    n._place_contact_guard = place_contact_guard.PlaceContactGuard(1, dict(IDENTITY))
    n._target_book_model = IDENTITY['target_model']
    n._held_book_corners = np.zeros((8, 3))
    n._gripper_open_confirmed = False
    n.delivery_evidence_enabled = n.table_scene_required = n.bin_scene_required = True
    right = tuple(f'arm_right_{i}_joint' for i in range(1, 8))
    head = ('head_1_joint', 'head_2_joint')
    parked = dict.fromkeys((*right, *head), 0.)
    n.right_chain = NS(active_names=('torso_lift_joint', *right))
    n.head_chain = NS(active_names=('torso_lift_joint', *head))
    n._active_place_scene_reference = dict(base_pose=[0., 0., 0.], parked_joints=parked)
    n.joints = dict(parked, **dict(zip(IK_JOINTS, q)))
    n._joint_stamps_ns = dict.fromkeys(n.joints, NOW)
    n._staging_odom = dict(stamp_ns=NOW, pose=[0., 0., 0.], linear_speed=0., angular_speed=0.)
    n.get_clock = lambda: NS(now=lambda: NS(nanoseconds=NOW))
    n.grasp_contact_max_age = .2
    n._target_contact_recent = lambda **kw: True
    hazard = load('_payload_hazard_reason', _place_contact_fault=place_contact_guard.fault_reason,
                  _stock_diagnostic=lambda n: False, measured_scene_context=measured_scene_context)
    n._payload_hazard_reason = lambda **kw: hazard(n, **kw)
    endpoint_calls = []
    n._wait_for_retained_endpoint = lambda *a, **kw: (endpoint_calls.append((a, kw)) or q.copy())
    events = []
    n._publish_status = lambda event, **kw: events.append((event, kw))
    options = clearance.normal_options(n, IDENTITY, [], legs)
    return n, legs, options, calls, probes, endpoint_calls, events


@pytest.mark.parametrize('value', [0, 1, None, 'false', 'true', [], {}, np.bool_(True)])
def test_strict_boolean(value):
    with pytest.raises(ValueError):
        clearance.checked_enabled(value)


def test_absent_default_does_not_read_any_motion_context():
    assert clearance.normal_options(NS(), None, None, None) == {}
    assert clearance.normal_options(NS(bin_clearance_timing_enabled=False), None, None, None) == {}


@pytest.mark.parametrize('field', ['delivery_evidence_enabled', 'table_scene_required', 'bin_scene_required'])
def test_normal_selection_requires_all_existing_registered_evidence_modes(field):
    n, legs, *_ = fixture()
    setattr(n, field, False)
    with pytest.raises(RuntimeError):
        clearance.normal_options(n, IDENTITY, [], legs)


@pytest.mark.parametrize('fault', ['direct', 'scene', 'identity', 'guard', 'released', 'target'])
def test_normal_selection_refuses_uncorrelated_or_unloaded_scope(fault):
    n, legs, *_ = fixture()
    identity = dict(IDENTITY)
    direct = []
    if fault == 'direct': direct = None
    elif fault == 'scene': n._active_place_scene_reference = None
    elif fault == 'identity': identity['placement_attempt_id'] = 'different'
    elif fault == 'guard': n._place_contact_guard.fault = {'reason': 'contact'}
    elif fault == 'released': n._held_book_corners = None
    else: n._target_book_model = 'book_col_2_row_2_red'
    with pytest.raises(RuntimeError):
        clearance.normal_options(n, identity, direct, legs)


@pytest.mark.parametrize('change', ['duplicate', 'duration', 'phase', 'nonfinite'])
def test_normal_route_order_and_original_durations_are_bound(change):
    n, legs, *_ = fixture()
    if change == 'duplicate': legs.append(legs[1])
    elif change == 'duration': legs[1] = (legs[1][0], .8, 'bin_clearance')
    elif change == 'phase': legs[0] = (legs[0][0], .8, 'withdrawal')
    else: legs[1][0][2] = np.nan
    with pytest.raises(RuntimeError):
        clearance.normal_options(n, IDENTITY, [], legs)


@pytest.mark.parametrize('scale', [1., 1.25, 3.])
def test_actual_executor_changes_only_clearance_time_and_preserves_watchdog_and_order(scale):
    n, legs, options, calls, probes, waits, events = fixture()
    saved = copy.deepcopy(legs)
    sequence = []
    send = n._send_retained_arm_trajectory
    n._send_retained_arm_trajectory = lambda *a, **kw: (sequence.append('send:' + a[2][0][2]) or send(*a, **kw))
    wait = n._wait_for_retained_endpoint
    n._wait_for_retained_endpoint = lambda *a, **kw: (sequence.append('endpoint') or wait(*a, **kw))
    retain = n._retention_after_leg
    n._retention_after_leg = lambda *a: (sequence.append('retain:' + a[1]) or retain(*a))
    assert load('_execute_retained_arm_legs')(n, legs, 'place', arm_speed_scale=scale, **options) == (True, 3, False)
    assert [call[1] for call in calls] == [.8, 2.8, .65]
    assert [timing.seconds(call[0].trajectory.points[0].time_from_start) for call in calls] == pytest.approx([.8/scale, .8/scale, .65/scale])
    assert [bool(call[4].get('velocity_headroom')) for call in calls] == [False, True, False]
    assert calls[1][4]['velocity_admission'] is True
    assert all(call[2][0][0] is leg[0] for call, leg in zip(calls, legs))
    assert all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(legs, saved))
    assert sequence == ['send:bin_transition', 'retain:bin_transition', 'send:bin_clearance',
                        'endpoint', 'retain:bin_clearance', 'send:bin_approach', 'retain:bin_approach']
    assert waits[0][1] == dict(command='place', phase='bin_clearance', leg=1)
    assert any(e == 'bin_clearance_endpoint_verified' and d['stationary_verified'] is False for e, d in events)


@pytest.mark.parametrize('fault', ['motion', 'contact', 'unknown', 'endpoint'])
def test_failed_clearance_never_sends_next_approach_or_advances_endpoint(fault):
    n, legs, options, calls, probes, waits, _ = fixture()
    send = n._send_retained_arm_trajectory
    def send_fault(*a, **kw):
        if a[2][0][2] != 'bin_clearance': return send(*a, **kw)
        calls.append((a[0], a[1], a[2], a[3], kw))
        if fault == 'unknown': raise timing.RetainedMotionNotStopped('unknown terminal')
        return (True, False) if fault == 'endpoint' else (False, fault == 'contact')
    n._send_retained_arm_trajectory = send_fault
    if fault == 'endpoint': n._wait_for_retained_endpoint = lambda *a, **kw: None
    if fault in ('unknown', 'endpoint'):
        with pytest.raises(RuntimeError):
            load('_execute_retained_arm_legs')(n, legs, 'place', arm_speed_scale=1.25, **options)
    else:
        assert load('_execute_retained_arm_legs')(n, legs, 'place', arm_speed_scale=1.25, **options) == (False, 1, fault == 'contact')
    assert len(calls) == 2 and len(probes) == 1


@pytest.mark.parametrize('fault', ['cancel', 'guard_replaced', 'scene_replaced', 'payload', 'base', 'right', 'head'])
def test_context_failure_before_connector_keeps_it_unsent(fault):
    n, legs, options, calls, probes, *_ = fixture()
    retain = n._retention_after_leg
    def change(*a):
        if fault == 'cancel': n._cancel.set()
        elif fault == 'guard_replaced': n._place_contact_guard = copy.copy(n._place_contact_guard)
        elif fault == 'scene_replaced': n._active_place_scene_reference = copy.deepcopy(n._active_place_scene_reference)
        elif fault == 'payload': n._payload_hazard_latched = 'lost'
        elif fault == 'base': n._staging_odom['pose'][0] = .003
        elif fault == 'right': n.joints['arm_right_1_joint'] = .002
        else: n.joints['head_1_joint'] = .002
        return retain(*a)
    n._retention_after_leg = change
    assert load('_execute_retained_arm_legs')(n, legs, 'place', arm_speed_scale=1.25, **options) == (False, 1, False)
    assert len(calls) == 1


@pytest.mark.parametrize('fault', ['stale', 'future', 'residual', 'cancel', 'hazard', 'scene'])
def test_added_endpoint_rechecks_current_feedback_and_context_after_wait(fault):
    n, legs, options, *_ = fixture(); token = options['clearance_timing']
    def wait(*a, **kw):
        if fault == 'stale': n._joint_stamps_ns[IK_JOINTS[1]] = NOW-150_000_001
        elif fault == 'future': n._joint_stamps_ns[IK_JOINTS[1]] = NOW+50_000_001
        elif fault == 'residual': n.joints[IK_JOINTS[1]] += .00501
        elif fault == 'cancel': n._cancel.set()
        elif fault == 'hazard': n._payload_hazard_latched = 'lost'
        else: n._staging_odom['pose'][0] = .003
        return np.asarray(token.target)
    n._wait_for_retained_endpoint = wait
    with pytest.raises(RuntimeError):
        clearance.require_endpoint(n, token, 1)


def sender(*, factor=2., positions=None, gate_override=None):
    n, events, sent, records, acceptance, clock = sender_fixtures.sender_node(retained=True)
    n.additional_arm_time_scale = factor
    calls = []
    def checked(n, goal):
        assert n.command.depth == n._lock.depth == 1
        calls.append(copy.deepcopy(goal))
        return (gate_override or velocity.require_arm_velocity_locked)(n, goal)
    fn = load('_send_retained_arm_trajectory', time=clock, require_arm_velocity_locked=checked)
    positions = [.02]*7 if positions is None else positions
    goal = sender_fixtures.goal([(positions, 640_000_000)])
    legs = [([.35, *positions], .64, 'bin_clearance')]
    n.bin_clearance_timing_enabled = True
    n.delivery_evidence_enabled = n.table_scene_required = n.bin_scene_required = True
    n._place_contact_guard = place_contact_guard.PlaceContactGuard(1, dict(IDENTITY))
    n._active_place_scene_reference = {}
    n._held_book_corners = np.zeros((8, 3))
    n._target_book_model = IDENTITY['target_model']
    n._gripper_open_confirmed = False
    token = clearance.normal_options(n, IDENTITY, [], [(legs[0][0], 2.8, 'bin_clearance')])['clearance_timing']
    def run():
        return fn(n, goal, 2.8, legs, 'place', velocity_admission=True,
                  velocity_headroom=True, clearance_admission=token)
    return n, sent, records, calls, goal, acceptance, clock, run


def test_actual_sender_selected_point_is_032_and_watchdog_stays_28():
    n, sent, records, gates, original, acceptance, clock, run = sender()
    done = iter([False, True])
    acceptance.result().get_result_async = lambda: NS(done=lambda: next(done), result=lambda: NS(status=4))
    ticks = iter([0., 0., 11.])
    clock.monotonic = lambda: next(ticks)
    assert run() == (True, False)
    assert len(gates) == 2 and len(sent) == 1
    assert timing.seconds(sent[0].trajectory.points[0].time_from_start) == .32
    assert timing.seconds(original.trajectory.points[0].time_from_start) == .64
    assert records[0]['additional_arm_timing']['nominal_duration_seconds'] == 2.8
    assert not n._goal_handles and not n._pending_retained_acceptances


def test_velocity_floor_still_extends_only_serialized_time():
    n, sent, records, _, _, _, _, run = sender(positions=[.6]*7)
    assert run() == (True, False)
    record = records[0]['additional_arm_timing']
    assert record['new_total_ns'] > 320_000_000
    assert record['nominal_duration_seconds'] == 2.8 and record['unchanged_nominal_watchdog']


@pytest.mark.parametrize('when', [1, 2])
def test_either_actual_locked_admission_refuses_stale_feedback(when):
    calls = []
    def checked(n, goal):
        calls.append(1)
        if len(calls) == when: n._joint_stamps_ns[sender_fixtures.NAMES[0]] = sender_fixtures.NOW-150_000_001
        return velocity.require_arm_velocity_locked(n, goal)
    n, sent, _, _, _, _, _, run = sender(gate_override=checked)
    with pytest.raises(velocity.ArmVelocityAdmissionRejected): run()
    assert not sent and not n._pending_retained_acceptances


def test_headroom_is_required_even_when_additional_scale_is_one():
    n, sent, _, _, _, _, _, run = sender(factor=1., positions=[.6]*7)
    n._faster_arm_velocity_limits = (1.,)*7
    with pytest.raises(velocity.ArmVelocityAdmissionRejected, match='80-percent'): run()
    assert not sent and not n._pending_retained_acceptances


@pytest.mark.parametrize('when', [1, 2])
@pytest.mark.parametrize('fault', ['guard', 'scene', 'target', 'cancel'])
def test_exact_clearance_context_is_rechecked_at_locked_publication(when, fault):
    calls = []
    def checked(n, goal):
        calls.append(1)
        result = velocity.require_arm_velocity_locked(n, goal)
        if len(calls) == when:
            if fault == 'guard': n._place_contact_guard = copy.copy(n._place_contact_guard)
            elif fault == 'scene': n._active_place_scene_reference = {}
            elif fault == 'target': n._target_book_model = 'book_col_2_row_2_red'
            else: n._cancel.set()
        return result
    n, sent, records, _, _, _, _, run = sender(gate_override=checked)
    with pytest.raises(velocity.ArmVelocityAdmissionRejected): run()
    assert not sent and not n._pending_retained_acceptances and not n._goal_handles
    assert len(records) == 1 and records[0]['admitted'] is False


@pytest.mark.parametrize('outcome', ['converged', 'ros_timeout', 'wall_timeout', 'cancel', 'hazard'])
def test_actual_retained_endpoint_wait_keeps_original_bounds_and_faults(outcome):
    n, _, options, *_ = fixture(); token = options['clearance_timing']
    n.timeout = 120.
    state = dict(ros=NOW, wall=0., sleeps=0)
    n.get_clock = lambda: NS(now=lambda: NS(nanoseconds=state['ros']))
    n.joints[IK_JOINTS[1]] += .02
    n._measured_left_solution = lambda: np.asarray([n.joints[name] for name in IK_JOINTS])
    def sleep(seconds):
        assert seconds == .02 and not n._lock.locked()
        state['sleeps'] += 1
        state['ros'] += 2_100_000_000 if outcome == 'ros_timeout' else 20_000_000
        state['wall'] += 129. if outcome == 'wall_timeout' else .02
        n._joint_stamps_ns = dict.fromkeys(n.joints, state['ros'])
        n._staging_odom['stamp_ns'] = state['ros']
        if outcome == 'converged': n.joints.update(zip(IK_JOINTS, token.target))
        elif outcome == 'cancel': n._cancel.set()
        elif outcome == 'hazard': n._payload_hazard_latched = 'lost'
    clock = NS(monotonic=lambda: state['wall'], sleep=sleep)
    actual_wait = load('_wait_for_retained_endpoint', time=clock)
    n._wait_for_retained_endpoint = lambda *a, **kw: actual_wait(n, *a, **kw)
    if outcome == 'converged': clearance.require_endpoint(n, token, 1)
    else:
        with pytest.raises(RuntimeError, match='endpoint_unverified'):
            clearance.require_endpoint(n, token, 1)
    assert state['sleeps'] == 1


@pytest.mark.parametrize('command,kwargs', [
    ('pick', {}), ('place', {'leg_offset': 1}),
    ('place', {'initial_pressure_gate': object()}),
    ('place', {'withdrawal_speed_scale': 1.25}),
    ('place', {'fresh_retention_phases': ('bin_clearance',)})])
def test_explicit_token_cannot_be_reused_for_recovery_or_other_phase_scope(command, kwargs):
    n, legs, options, calls, *_ = fixture()
    with pytest.raises(RuntimeError):
        load('_execute_retained_arm_legs')(n, legs, command, **options, **kwargs)
    assert not calls


def test_default_executor_matches_exact_parent_goal_check_and_index_sequence():
    root = Path(__file__).resolve().parents[1]
    text = (root/'erc_phase1_solution/manipulation_node.py').read_text()
    from candidate_composition_support import restore_all_books_source
    text = restore_all_books_source(text)
    inverse = json.loads((root/'test/fixtures/bin_clearance_timing_inverse.json').read_text())
    head = json.loads((root/'test/fixtures/empty_head_timing_inverse.json').read_text())
    for fragment in reversed(head['node_fragments']):
        assert text.count(fragment['new']) == 1
        text = text.replace(fragment['new'], fragment['old'])
    assert hashlib.sha256(text.encode()).hexdigest() == head['parent_node_sha256']
    for f in reversed(inverse['fragments']): text = text.replace(f['new'], f['old'])
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'ManipulationNode')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_retained_arm_legs')
    current = load('_execute_retained_arm_legs')
    scope = dict(current.__globals__)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<exact parent executor>', 'exec'), scope)
    results = []
    for fn in (scope['_execute_retained_arm_legs'], current):
        n, legs, _, calls, probes, waits, events = fixture()
        result = fn(n, legs, 'place', arm_speed_scale=1.25)
        results.append((result, [(c[0].trajectory.points[0].positions,
            timing.seconds(c[0].trajectory.points[0].time_from_start), c[1], c[3], c[4]) for c in calls], probes, waits, events))
    assert results[0] == results[1]


def test_default_node_inverse_restores_exact_parent_and_recovery_methods():
    root = Path(__file__).resolve().parents[1]
    text = (root/'erc_phase1_solution/manipulation_node.py').read_text()
    from candidate_composition_support import restore_all_books_source
    text = restore_all_books_source(text)
    data = json.loads((root/'test/fixtures/bin_clearance_timing_inverse.json').read_text())
    head = json.loads((root/'test/fixtures/empty_head_timing_inverse.json').read_text())
    for fragment in reversed(head['node_fragments']):
        assert text.count(fragment['new']) == 1
        text = text.replace(fragment['new'], fragment['old'])
    assert hashlib.sha256(text.encode()).hexdigest() == head['parent_node_sha256']
    before = ast.parse(text)
    for fragment in reversed(data['fragments']):
        assert text.count(fragment['new']) == 1
        text = text.replace(fragment['new'], fragment['old'])
    assert hashlib.sha256(text.encode()).hexdigest() == data['parent_node_lf_sha256']
    def methods(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManipulationNode')
        return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
    current, original = methods(before), methods(ast.parse(text))
    changed = {'__init__', '_declare_parameters', '_send_retained_arm_trajectory', '_execute_retained_arm_legs', '_place'}
    assert current.keys() == original.keys()
    assert all(current[n] == original[n] for n in current.keys()-changed)
    assert "'bin_clearance_timing_enabled': False" in (root/'erc_phase1_solution/manipulation_node.py').read_text()


def test_new_flag_alone_loads_actual_urdf_velocity_limits_once(tmp_path, monkeypatch):
    from erc_phase1_solution import empty_pickup_collision as module
    urdf = tmp_path/'official-shape.urdf'
    urdf.write_text('<robot>'+''.join(f'<joint name="{name}"><limit velocity="{i+1}"/></joint>'
        for i, name in enumerate(IK_JOINTS))+'</robot>')
    actual = module.joint_velocity_limits; calls = []
    monkeypatch.setattr(module, 'joint_velocity_limits', lambda p: (calls.append(p) or actual(p)))
    init = timing.method_ast('__init__')
    blocks = [n for n in init.body if isinstance(n, ast.If) and any(isinstance(x, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == '_faster_arm_velocity_limits'
                for t in x.targets) for x in n.body)]
    assert len(blocks) == 1
    node = NS(placement_transport_speed_scale=1., withdrawal_speed_scale=1.,
              empty_pickup_setup_retiming_enabled=False, bin_clearance_timing_enabled=True)
    scope = dict(self=node, urdf=urdf, __name__='erc_phase1_solution.fixture', __package__='erc_phase1_solution')
    exec(compile(ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[])), '<actual optional limits>', 'exec'), scope)
    assert calls == [urdf] and node._faster_arm_velocity_limits == tuple(float(i) for i in range(2, 9))
    assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == 'bin_clearance_timing_enabled' for t in n.targets))
    assert assignment.value.func.id == 'checked_bin_clearance_timing_enabled'


def normal_place_continuation():
    """Execute the unchanged actual normal PLACE suffix after checked planning.

    Geometry and perception are fixture inputs here; this is call-site, action
    sequencing, and abort-path coverage, not another planning admission test.
    """
    from erc_phase1_solution.motion_profiles import HOME
    from erc_phase1_solution.raised_place_finish import normal_finish
    original = timing.method_ast('_place')
    start = next(i for i, statement in enumerate(original.body)
                 if isinstance(statement, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'arm_timing_options'
                         for target in statement.targets))
    wrapper = ast.parse('def continuation(self, correlation, direct_empty_home, '
                        'carried_transition_waypoints, solutions, torso_ready, '
                        'unloaded_home_waypoints, place_scene_diagnostics=None):\n    pass\n').body[0]
    initialization = original.body[0]
    assert isinstance(initialization, ast.Assign)
    assert [target.id for target in initialization.targets] == ['release_only_request', 'release_only_endpoint']
    assert isinstance(initialization.value, ast.Constant) and initialization.value.value is None
    wrapper.body = copy.deepcopy([initialization, *original.body[start:]])
    scope = dict(bin_clearance_normal_options=clearance.normal_options,
                 checked_arm_speed_scale=timing.timing.checked_arm_speed_scale,
                 raised_place_normal_finish=normal_finish, HOME=HOME,
                 observe_measured_open_pose=lambda *a: {'verified': True})
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), wrapper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<actual normal PLACE suffix>', 'exec'), scope)
    return scope['continuation']


@pytest.mark.parametrize('fault', ['none', 'motion', 'contact', 'endpoint', 'unknown'])
def test_actual_normal_place_call_site_keeps_endpoint_and_abort_only_recovery(fault):
    n, legs, _, calls, probes, waits, events = fixture()
    n.placement_transport_speed_scale = 3.
    n.loaded_place_speed_scale_cap = 1.25
    lifecycle = []
    send = n._send_retained_arm_trajectory
    def sender_at_clearance(*args, **kwargs):
        if args[2][0][2] == 'bin_clearance' and fault in ('motion', 'contact', 'unknown'):
            calls.append((*args, kwargs))
            if fault == 'unknown':
                raise timing.RetainedMotionNotStopped('acceptance unknown')
            return False, fault == 'contact'
        return send(*args, **kwargs)
    n._send_retained_arm_trajectory = sender_at_clearance
    n._execute_retained_arm_legs = lambda *a, **kw: load('_execute_retained_arm_legs')(n, *a, **kw)
    if fault == 'endpoint':
        n._wait_for_retained_endpoint = lambda *a, **kw: None
    def opened(**kwargs):
        lifecycle.append(('open', kwargs))
        return kwargs['verify_measurement']()
    n._open_gripper = opened
    n._return_from_bin = lambda *a, **kw: (lifecycle.append(('return', kw)) or True)
    n._recover_closed_place = lambda **kw: (lifecycle.append(('recover', kw)) or False)
    invoke = lambda: normal_place_continuation()(n, IDENTITY, [], [legs[0][0]],
        [legs[1][0], legs[2][0]], legs[0][0], [])
    if fault in ('endpoint', 'unknown'):
        with pytest.raises(RuntimeError, match='endpoint_unverified|acceptance unknown'):
            invoke()
        assert lifecycle == []
    else:
        assert invoke() is (fault == 'none')
    connector = calls[1]
    assert connector[4]['clearance_admission'].identity.placement_attempt_id == IDENTITY['placement_attempt_id']
    assert connector[4]['velocity_headroom'] is True and connector[1] == 2.8
    # This fixture stops at the actual sender boundary. The real sender's
    # additional factor2 and0.32-second message are covered separately below.
    assert timing.seconds(connector[0].trajectory.points[0].time_from_start) == .64
    if fault == 'none':
        assert len(calls) == len(probes) == 3 and len(waits) == 1
        assert [entry[0] for entry in lifecycle] == ['open', 'return']
        assert 'clearance_timing' not in lifecycle[-1][1]
    else:
        assert len(calls) == 2 and len(probes) == 1
        if fault in ('motion', 'contact'):
            assert [entry[0] for entry in lifecycle] == ['recover']
            recovery = lifecycle[0][1]
            assert recovery['release_allowed'] is False
            assert recovery['cause'] == ('contact_lost' if fault == 'contact' else 'motion_failed')
            assert [leg[2] for leg in recovery['remaining_approach_legs']] == ['bin_clearance', 'bin_approach']


@pytest.mark.parametrize('fault', ['mid_hazard', 'pending_hazard', 'terminal_hazard',
                                  'terminal_cancel', 'unknown_acceptance'])
def test_enabled_actual_sender_watchdog_and_late_acceptance_precede_new_endpoint(monkeypatch, fault):
    import test_retained_leg_watchdog as watchdog
    module = watchdog.module
    def terminal(owner):
        if fault == 'terminal_hazard':
            owner._payload_hazard_reason = lambda **kw: 'contact_lost'
        elif fault == 'terminal_cancel':
            owner._cancel.set()
    n, clock = watchdog.fixture(monkeypatch,
        fault_at=.06 if fault in ('mid_hazard', 'pending_hazard') else None,
        acceptance_delay=.5 if fault == 'unknown_acceptance' else (.12 if fault == 'pending_hazard' else 0.),
        terminal_hook=terminal)
    command_lock = threading.RLock()
    n._adaptive_command_guard = lambda: command_lock
    n.additional_arm_time_scale = 2.
    n._faster_arm_velocity_limits = sender_fixtures.LIMITS
    n._faster_arm_velocity_urdf = 'fixture/official.urdf'
    n.joints = dict.fromkeys(IK_JOINTS, 0.)
    n.joints[IK_JOINTS[0]] = .35
    n._joint_stamps_ns = dict.fromkeys(IK_JOINTS, NOW)
    n.bin_clearance_timing_enabled = True
    n.delivery_evidence_enabled = n.table_scene_required = n.bin_scene_required = True
    n._place_contact_guard = place_contact_guard.PlaceContactGuard(1, dict(IDENTITY))
    n._active_place_scene_reference = {}
    n._held_book_corners = np.zeros((8, 3))
    n._target_book_model = IDENTITY['target_model']
    n._gripper_open_confirmed = False
    endpoint_calls = []
    n._wait_for_retained_endpoint = lambda *a, **kw: (endpoint_calls.append((a, kw)) or np.zeros(8))
    route = [(np.asarray([.35, *([.02]*7)]), 2.8, 'bin_clearance'),
             (np.asarray([.35, *([.03]*7)]), .65, 'bin_approach')]
    options = clearance.normal_options(n, IDENTITY, [], route)
    if fault == 'unknown_acceptance':
        with pytest.raises(module.RetainedMotionNotStopped, match='acceptance'):
            n._execute_retained_arm_legs(route, 'place', arm_speed_scale=1.25, **options)
        assert n._cancel.is_set() and n._pending_retained_acceptances
        assert n.cancelled == []
        clock.sleep(.30)
        assert n.acceptance_futures[0].done()
        assert len(n.cancelled) == len(n._goal_handles) == 1
        clock.sleep(.05)
        assert n.result_futures[0].done()
        assert not n._goal_handles and not n._pending_retained_acceptances
    else:
        assert n._execute_retained_arm_legs(route, 'place', arm_speed_scale=1.25, **options) == (
            False, 0, fault != 'terminal_cancel')
        assert not n._goal_handles and not n._pending_retained_acceptances
        if fault in ('mid_hazard', 'pending_hazard'):
            assert len(n.cancelled) == 1 and n.cancelled[0][0] < 10.2
    assert len(n.sent) == 1 and not n.probed and not endpoint_calls
    assert timing.seconds(n.sent[0][1].trajectory.points[0].time_from_start) == .32
    velocity_records = [data for event, data in n.events if event == 'arm_velocity_admission']
    assert len(velocity_records) == 1 and velocity_records[0]['admitted']
    assert velocity_records[0]['additional_arm_timing']['nominal_duration_seconds'] == 2.8
