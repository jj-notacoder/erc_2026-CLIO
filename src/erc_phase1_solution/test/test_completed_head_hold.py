"""Actual extracted head/follow methods, ordered locks and immediate action futures.

No ROS, executor, simulator, mesh traversal or source mutation. NumPy supplies
the same array arithmetic used by the candidate; collision predicates are spies.
"""
from __future__ import annotations

import ast
from pathlib import Path
from erc_phase1_solution.completed_torso_hold import track_completed_torso_hold
import math
import threading
import time
from types import SimpleNamespace as NS
from typing import Optional, Sequence

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'erc_phase1_solution/manipulation_node.py'
BASELINE = ROOT / 'test/fixtures/head_move_original.py'
HEAD = ('head_1_joint', 'head_2_joint')
IK = ('torso_lift_joint', *(f'arm_left_{i}_joint' for i in range(1, 8)))
RIGHT = tuple(f'arm_right_{i}_joint' for i in range(1, 8))
METHODS = ('_follow', '_move_head', '_head_hold_snapshot', '_head_hold_admission_clear',
           '_head_hold_geometry_is_safe', '_try_completed_head_hold', '_head_hold_snapshot_locked')


def _place(node):
    with node._lock:
        if node.place_fault:
            raise RuntimeError('placement_contact:' + node.place_fault)


def _scene(node, reference):
    with node._lock:
        node.context_calls += 1
        if node.scene_fault:
            raise RuntimeError('placement_scene:' + node.scene_fault)


class Duration:
    def __init__(self, *, seconds):
        self.seconds = seconds

    def to_msg(self):
        return self.seconds


class Goal:
    def __init__(self):
        self.trajectory = NS(joint_names=[], points=[])


def load_class():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManipulationNode')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert {n.name for n in methods} == set(METHODS)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
        ast.ClassDef(name='Extracted', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    scope = dict(track_completed_torso_hold=track_completed_torso_hold, np=np, math=math, time=time, HEAD_JOINTS=HEAD, IK_JOINTS=IK, RIGHT_ARM_JOINTS=RIGHT,
                 _require_place_contact_clear=_place, measured_scene_context=_scene,
                 FollowJointTrajectory=NS(Goal=Goal), JointTrajectoryPoint=NS,
                 Duration=Duration, GoalStatus=NS(STATUS_SUCCEEDED=4))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), scope)
    return scope['Extracted']


class Lock:
    def __init__(self, node, label):
        self.node, self.label, self.depth = node, label, 0
        self.raw = threading.Lock() if label == 'sensor' else threading.RLock()

    def __enter__(self):
        if self.label == 'command':
            assert self.node._lock.depth == 0, 'sensor->command lock inversion'
        assert self.raw.acquire(timeout=.1), 'non-reentrant sensor-lock deadlock'
        self.depth += 1
        return self

    def __exit__(self, *args):
        self.depth -= 1
        self.raw.release()


class Future:
    def __init__(self, result):
        self.value = result

    def done(self):
        return True

    def result(self):
        return self.value


class Client:
    def __init__(self, node):
        self.node = node
        self.waits, self.sends = 0, 0
        self.available, self.accepted, self.status = True, True, 4
        self.send_error = None

    def wait_for_server(self, **kwargs):
        assert self.node._lock.depth == self.node.command.depth == 0
        self.waits += 1
        return self.available

    def send_goal_async(self, goal):
        self.sends += 1
        if self.send_error:
            raise self.send_error
        self.goal = goal
        handle = NS(accepted=self.accepted, get_result_async=lambda: Future(NS(status=self.status)))
        return Future(handle)


class Node(load_class()):
    def __init__(self):
        self._lock = Lock(self, 'sensor')
        self.command = Lock(self, 'command')
        self._cancel = threading.Event()
        self.now = 1_000_000_000
        self.joints = {n: 0. for n in (*IK, *RIGHT, *HEAD)}
        self.joints['torso_lift_joint'] = .35
        self.joints[HEAD[1]] = -.6
        self._joint_velocities = {n: 0. for n in self.joints}
        self._joint_stamps_ns = {n: 950_000_000 for n in self.joints}
        self._last_completed_head_target = ((0., -.6), 800_000_000)
        self._held_book_corners = np.array([(x, y, z) for x in (-.1, .1) for y in (-.02, .02) for z in (-.15, .15)])
        self._goal_handles, self._pending_retained_acceptances = [], set()
        self._payload_robot_watchdog_enabled = True
        self._target_robot_contact_latched = False
        self._active_place_scene_reference = None
        self._place_contact_guard = None
        self._payload_hazard_latched = None
        self.place_fault = self.scene_fault = self.retention_fault = None
        self.head_client = Client(self)
        self.timeout = 1.
        self.events, self.geometry_calls, self.full_preflights = [], [], []
        self.context_calls = 0
        self.hazard_calls = 0
        self.hazard_hook = lambda: None
        self._contact_generation = self._contact_epoch = 0
        self._gripper_feedback_samples = []
        self.full_safe = True
        self.body_collision = self.book_collision = self.head_collision = None
        self.after_geometry = lambda: None
        self.head_chain = NS(lower=np.array([0., -2., -2.]), upper=np.array([1., 2., 2.]))
        self.chain = NS(forward=lambda q: np.eye(4))

    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=self.now))

    def _adaptive_command_guard(self):
        return self.command

    def _payload_hazard_reason(self, *, max_age):
        assert max_age == .15
        with self._lock:
            self.hazard_calls += 1
            result = self.retention_fault
            self.hazard_hook()
            return result

    def _publish_status(self, event, **fields):
        self.events.append((event, fields))

    def _wait_future(self, future, timeout):
        assert self._lock.depth == self.command.depth == 0
        return future.result()

    def _carried_head_transition_is_safe(self, *target):
        self.full_preflights.append(target)
        return self.full_safe

    def _robot_self_collision(self, q, **context):
        assert self._lock.depth == self.command.depth == 0
        self.geometry_calls.append(('body', q.copy(), context))
        return self.body_collision

    def _carried_robot_collision(self, q, corners, **context):
        self.geometry_calls.append(('book', q.copy(), context))
        return self.book_collision

    def _head_motion_collision(self, q, head, corners, **context):
        self.geometry_calls.append(('head', q.copy(), context))
        self.after_geometry()
        return self.head_collision


def normal_motion(node, pan=0., tilt=-.6):
    assert node._move_head(pan, tilt)
    assert node.head_client.sends == node.head_client.waits == 1
    assert node.full_preflights == [(pan, tilt)]
    assert node._goal_handles == []
    assert not any(event == 'head_motion_skipped' for event, _ in node.events)


def test_successful_hold_has_current_checks_no_server_goal_or_full_sweep():
    node = Node()
    original = node._last_completed_head_target
    assert node._move_head(0., -.6)
    assert node.head_client.sends == node.head_client.waits == 0
    assert node.full_preflights == []
    assert [row[0] for row in node.geometry_calls] == ['body', 'book', 'head']
    assert node._last_completed_head_target == original
    assert node.events[-1][0] == 'head_motion_skipped'
    assert node.events[-1][1]['completed_action_ros_ns'] == original[1]


@pytest.mark.parametrize('case', ['missing_position', 'nan_position', 'missing_velocity', 'nan_velocity',
    'fast_head', 'fast_torso', 'stale_head', 'future_head', 'stale_body', 'precompletion', 'clock_reversed',
    'position_moved', 'no_prior_success'])
def test_ineligible_feedback_requires_fresh_start_before_normal_action(case):
    node = Node()
    node.timeout = .025
    if case == 'missing_position': del node.joints[HEAD[0]]
    elif case == 'nan_position': node.joints[HEAD[0]] = math.nan
    elif case == 'missing_velocity': del node._joint_velocities[HEAD[0]]
    elif case == 'nan_velocity': node._joint_velocities[HEAD[0]] = math.nan
    elif case == 'fast_head': node._joint_velocities[HEAD[0]] = 1.01e-4
    elif case == 'fast_torso': node._joint_velocities[IK[0]] = 1.01e-5
    elif case == 'stale_head': node._joint_stamps_ns[HEAD[0]] = 849_999_999
    elif case == 'future_head': node._joint_stamps_ns[HEAD[0]] = 1_050_000_001
    elif case == 'stale_body': node._joint_stamps_ns[RIGHT[0]] = 849_999_999
    elif case == 'precompletion': node._last_completed_head_target = ((0., -.6), 950_000_000)
    elif case == 'clock_reversed': node.now = 799_999_999
    elif case == 'position_moved': node.joints[HEAD[0]] = 1.01e-6
    elif case == 'no_prior_success': node._last_completed_head_target = None
    if case in ('position_moved', 'no_prior_success'):
        normal_motion(node)
    else:
        assert not node._move_head(0., -.6)
        assert node.full_preflights == [] and node.head_client.sends == 0
        assert any(event == 'head_hold_rejected' for event, _ in node.events)


def test_changed_goal_even_inside_position_tolerance_is_new_motion():
    normal_motion(Node(), pan=.5e-6)


@pytest.mark.parametrize('kind', ['active', 'pending'])
def test_unresolved_goal_blocks_hold_and_new_send(kind):
    node = Node()
    if kind == 'active':
        node._goal_handles.append(object())
    else:
        node._pending_retained_acceptances.add(object())
    assert not node._move_head(0., -.6)
    assert node.geometry_calls == [] and node.head_client.sends == 0


@pytest.mark.parametrize('part', ['body', 'book', 'head'])
def test_current_geometry_rejection_is_not_bypassed(part):
    node = Node()
    setattr(node, part + '_collision', 'collision')
    with pytest.raises(RuntimeError, match='Current head hold geometry'):
        node._move_head(0., -.6)
    assert node.head_client.sends == 0 and node._last_completed_head_target is None
    assert [r[0] for r in node.geometry_calls] == ['body', 'book', 'head'][:['body', 'book', 'head'].index(part)+1]


def test_normal_preflight_failure_prevents_action():
    node = Node()
    node._last_completed_head_target = None
    node.full_safe = False
    with pytest.raises(RuntimeError, match='blocks requested head motion'):
        node._move_head(0., -.6)
    assert node.head_client.sends == 0


@pytest.mark.parametrize('fault', ['cancel', 'retention', 'payload_robot', 'active', 'pending',
    'stale', 'moving', 'body_moved', 'attachment_changed'])
def test_after_geometry_interlocks_prevent_success_and_send(fault):
    node = Node()
    node.timeout = .025
    def changed():
        if fault == 'cancel': node._cancel.set()
        elif fault == 'retention': node.retention_fault = 'contact_lost'
        elif fault == 'payload_robot': node._target_robot_contact_latched = True
        elif fault == 'active': node._goal_handles.append(object())
        elif fault == 'pending': node._pending_retained_acceptances.add(object())
        elif fault == 'stale': node.now += 200_000_000
        elif fault == 'moving': node._joint_velocities[HEAD[0]] = .01
        elif fault == 'body_moved': node.joints[RIGHT[0]] += 2e-6
        elif fault == 'attachment_changed': node._held_book_corners = node._held_book_corners + .001
    node.after_geometry = changed
    if fault == 'body_moved':
        normal_motion(node)
        return
    assert not node._move_head(0., -.6)
    assert node.head_client.sends == 0 and node._last_completed_head_target is None
    assert not any(event == 'head_motion_skipped' for event, _ in node.events)


@pytest.mark.parametrize('fault', ['contact', 'scene'])
def test_after_geometry_scene_or_contact_fault_raises_without_send(fault):
    node = Node()
    node._active_place_scene_reference = {'fixed': True}
    def changed():
        if fault == 'contact': node.place_fault = 'arm_bin'
        else: node.scene_fault = 'base_moved'
    node.after_geometry = changed
    with pytest.raises(RuntimeError, match='placement_'):
        node._move_head(0., -.6)
    assert node.head_client.sends == 0


def test_snapshot_copies_geometry_inputs_and_preserves_measured_head():
    node = Node()
    node.joints[HEAD[0]] = .4e-6
    captured = node._head_hold_snapshot((0., -.6), node._last_completed_head_target)
    assert captured['joints'][HEAD[0]] == .4e-6
    node.joints[RIGHT[0]] = .2
    node._held_book_corners += .01
    assert captured['joints'][RIGHT[0]] == 0.
    assert not np.array_equal(captured['attached'], node._held_book_corners)


def test_one_head_sample_uses_same_right_and_head_context_as_body():
    node = Node()
    node.joints[HEAD[0]] = .4e-6
    node.joints[RIGHT[0]] = .2
    assert node._move_head(0., -.6)
    body, book, head = node.geometry_calls
    np.testing.assert_array_equal(body[2]['right_positions'], head[2]['right_positions'])
    np.testing.assert_array_equal(body[2]['head_positions'], book[2]['head_positions'])
    assert body[2]['head_positions'][0] == .4e-6


@pytest.mark.parametrize('failure', ['server', 'rejected', 'aborted', 'exception'])
def test_failed_new_action_cannot_reuse_previous_success(failure):
    node = Node()
    node.joints[HEAD[0]] = .01
    if failure == 'server': node.head_client.available = False
    elif failure == 'rejected': node.head_client.accepted = False
    elif failure == 'aborted': node.head_client.status = 6
    else: node.head_client.send_error = RuntimeError('uncertain send')
    if failure in ('server', 'exception'):
        with pytest.raises(RuntimeError): node._move_head(0., -.6)
    else:
        assert not node._move_head(0., -.6)
    assert node._last_completed_head_target is None


def test_successful_normal_action_records_only_completed_target_then_fresh_hold():
    node = Node()
    node._last_completed_head_target = None
    normal_motion(node)
    assert node._last_completed_head_target == ((0., -.6), node.now)
    node.now += 20_000_000
    node._joint_stamps_ns = {n: node.now for n in node.joints}
    node.full_preflights.clear()
    assert node._move_head(0., -.6)
    assert node.head_client.sends == 1
    assert node.full_preflights == []


def test_torso_follow_retains_ordinary_action_and_does_not_use_head_shortcut():
    node = Node()
    client = Client(node)
    prior = node._last_completed_head_target
    assert node._follow(client, [IK[0]], [.35], 2.2)
    assert client.sends == 1 and node.geometry_calls == []
    assert node._last_completed_head_target == prior


def test_head_preflight_cannot_be_applied_to_another_client_or_group():
    node = Node()
    with pytest.raises(ValueError, match='head preflight requires'):
        node._follow(Client(node), [IK[0]], [.35], 2.2, head_preflight=True)


def test_original_head_positive_control_repeats_sweep_at_same_target():
    original = next(n for n in ast.parse(BASELINE.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_move_head')
    scope = {'HEAD_JOINTS': HEAD}
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(BASELINE), 'exec'), scope)
    node = Node()
    original_calls = []
    node._carried_head_transition_is_safe = lambda *target: original_calls.append(('preflight', target)) or True
    node._follow = lambda *args: original_calls.append(('follow', tuple(args[2]))) or True
    assert scope['_move_head'](node, 0., -.6)
    assert scope['_move_head'](node, 0., -.6)
    assert original_calls == [('preflight', (0., -.6)), ('follow', (0., -.6))] * 2



def test_empty_head_keeps_original_action_without_new_geometry_guard():
    node = Node()
    node._held_book_corners = None
    node.body_collision = 'would reject if incorrectly checked'
    assert node._move_head(0., -.6)
    assert node.head_client.sends == node.head_client.waits == 1
    assert node.geometry_calls == node.full_preflights == []


@pytest.mark.parametrize('race', ['contact', 'feedback', 'epoch'])
def test_safety_epoch_change_during_final_helpers_prevents_hold(race):
    node = Node()
    def changed():
        if node.hazard_calls != 2:
            return
        if race == 'contact': node._contact_generation += 1
        elif race == 'feedback': node._gripper_feedback_samples.append(
            NS(position=.017, velocity=0., effort=.1, stamp_ns=node.now))
        else: node._contact_epoch += 1
    node.hazard_hook = changed
    if race == 'epoch':
        assert not node._move_head(0., -.6)
        assert node.head_client.sends == 0
    else:
        normal_motion(node)
    assert not any(event == 'head_motion_skipped' for event, _ in node.events)


def test_real_non_reentrant_lock_completes_with_relocking_sensor_helpers():
    node = Node()
    node._active_place_scene_reference = {'fixed': True}
    outcomes = []
    thread = threading.Thread(target=lambda: outcomes.append(node._move_head(0., -.6)), daemon=True)
    thread.start()
    thread.join(1.)
    assert not thread.is_alive(), 'bounded no-op completion exceeded one second'
    assert outcomes == [True]
    assert node._lock.depth == node.command.depth == 0
