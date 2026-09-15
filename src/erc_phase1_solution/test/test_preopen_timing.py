"""Current stock retention parity and actual callback races; no simulation.

The real ROS lifecycle fixture executes the unchanged open/stock-close/PICK and
PLACE suffixes. Its expensive planning/actions are stubs, as in the parent test.
"""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import pytest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from erc_phase1_solution import preopen_stationary as gate
from test_preopen_stationary import node, run, raw, GOAL, MASTER
from test_stock_close_lifecycle import lifecycle
from preopen_timing_test_support import restore_preopen_timing_source

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('supported', [False, True])
@pytest.mark.parametrize('max_age', [.05, .15])
@pytest.mark.parametrize('case', [
    'healthy', 'left_old', 'right_old', 'left_future', 'feedback_absent',
    'feedback_old', 'feedback_future', 'effort_nan', 'feedback_position_inf',
    'width_invalid', 'place_fault', 'held_fault', 'payload_fault', 'robot_fault'])
def test_locked_stock_predicate_matches_actual_existing_helpers(lifecycle, supported, max_age, case):
    f = lifecycle; n = f.n; now = f.now.nanoseconds
    n._gravity_supported_payload = supported
    n._left_target_contact_ns = n._right_target_contact_ns = now
    n.joints['gripper_left_finger_joint'] = MASTER
    if case == 'left_old': n._left_target_contact_ns = now-round(max_age*1e9)-1
    elif case == 'right_old': n._right_target_contact_ns = now-round(max_age*1e9)-1
    elif case == 'left_future': n._left_target_contact_ns = now+100_000_001
    elif case == 'feedback_absent': n._gripper_feedback_samples.clear()
    elif case in ('feedback_old', 'feedback_future', 'effort_nan', 'feedback_position_inf'):
        changes = {'feedback_old': dict(stamp_ns=now-150_000_001),
                   'feedback_future': dict(stamp_ns=now+100_000_001),
                   'effort_nan': dict(effort=float('nan')),
                   'feedback_position_inf': dict(position=float('inf'))}
        n._gripper_feedback_samples.append(replace(n._gripper_feedback_samples[-1], **changes[case]))
    elif case == 'width_invalid': n.joints['gripper_left_finger_joint'] = .070
    elif case == 'place_fault': n._place_contact_guard = NS(fault={'reason': 'contact'})
    elif case == 'held_fault': n._held_grip_sensor_fault = 'invalid_stock_contact_force'
    elif case == 'payload_fault': n._payload_hazard_latched = 'contact_lost'
    elif case == 'robot_fault': n._target_robot_contact_latched = True
    expected = n._payload_hazard_reason(max_age=max_age)
    if expected is None and not n._pinch_sample(max_age=max_age)[0]:
        expected = 'closed_grip_not_retained'
    # The real plain Lock is held; a helper that reacquires it would deadlock.
    with n._lock:
        actual = gate._stock_retention_locked(n, now, max_age)
    assert actual == expected


def _joint(n, stamp, arm_velocity=0.):
    message = JointState()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 10**9)
    message.name = list(n.joints)
    message.position = list(n.joints.values())
    message.velocity = [0.] * len(message.name)
    message.velocity[message.name.index('arm_left_1_joint')] = arm_velocity
    message.effort = [-.1] * len(message.name)
    return message


@pytest.mark.parametrize('case', ['healthy', 'raw_velocity', 'odom_velocity', 'cancel', 'epoch', 'scene'])
def test_real_stock_flow_with_concurrent_callback_delivery(lifecycle, monkeypatch, case):
    f = lifecycle; n = f.n
    assert n._open_gripper() and f.pick()
    f.prepare_place()
    original_context = gate.measured_scene_context
    request, done, stop = threading.Event(), threading.Event(), threading.Event()
    failures = []; deliveries = []

    def worker():
        while not stop.is_set():
            if not request.wait(.5): continue
            request.clear()
            if stop.is_set(): return
            try:
                command_acquired = n._adaptive_command_guard().acquire(blocking=False)
                if command_acquired: n._adaptive_command_guard().release()
                assert not command_acquired
                # Real odometry, JointState and two contact callbacks in a
                # separate thread while final admission holds only command lock.
                before = n._contact_generation
                f.sensors()
                assert n._contact_generation > before
                if case == 'raw_velocity':
                    n._on_joint_state(_joint(n, f.now.nanoseconds, .02))
                elif case == 'odom_velocity':
                    message = Odometry()
                    message.header.stamp.sec, message.header.stamp.nanosec = divmod(f.now.nanoseconds, 10**9)
                    message.pose.pose.orientation.w = 1.
                    message.twist.twist.linear.x = .02
                    n._on_staging_odom(message)
                elif case == 'cancel': n._cancel.set()
                elif case == 'epoch':
                    with n._lock: n._contact_epoch += 1
                elif case == 'scene':
                    with n._lock: n._active_place_scene_reference = {'changed': True}
                deliveries.append(n._contact_generation)
            except BaseException as exc:
                failures.append(exc)
            finally:
                done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    def final_context(owner, reference):
        original_context(owner, reference)
        done.clear(); request.set()
        assert done.wait(.5), 'actual callback worker failed to finish; possible lock reentry'
        assert not failures, failures
    monkeypatch.setattr(gate, 'measured_scene_context', final_context)
    before_open = len(f.sent)
    try:
        if case == 'healthy':
            assert f.place()
            event = next(fields for kind, fields in f.events if kind == 'placement_preopen_stationary')
            assert event['verified'] and event['current_stock_retention_revalidated']
            assert event['contact_generation'] > event['earlier_contact_generation']
            assert event['timing_diagnostics']['generation_advances_revalidated'] == 1
            assert len(f.sent) > before_open and n._test_returns
        else:
            with pytest.raises(RuntimeError): f.place()
            event = next(fields for kind, fields in f.events if kind == 'placement_preopen_stationary')
            assert not event['verified'] and len(f.sent) == before_open
            assert not n._test_returns and n._held_book_corners is not None
            if case in ('raw_velocity', 'odom_velocity'):
                assert event['timing_diagnostics']['final_raw_retries'] > 0
                assert event['timing_diagnostics']['last_final_veto'].startswith('raw:')
    finally:
        stop.set(); request.set(); thread.join(timeout=1.)
        assert not thread.is_alive() and not failures
    assert deliveries and not n._delivery_measurement_active


@pytest.mark.parametrize('settles_after', [0., 5.25, 9.5])
def test_slow_ros_clock_requires_full_span_and_respects_ten_second_cap(node, settles_after):
    # Deterministic .16 ROS/wall pacing; no claim about actual DDS throughput.
    def sleep(seconds):
        assert not node.command_lock._is_owned()
        node.require_sensor_available()
        node.wall += seconds; node.now += round(seconds*.16*1e9); node.ticks += 1
        sample = raw(node.now)
        if node.wall < settles_after: sample['velocities']['arm_left_1_joint'] = .01
        node.feed(sample)
    gate.time.sleep = sleep
    if settles_after == 9.5:
        with pytest.raises(gate.PreopenStationaryRejected, match='measurement_timeout'): run(node)
        assert 10. <= node.wall <= 10.003
        assert node.events[-1][1]['timing_diagnostics']['maximum_stationary_span_ns'] < 100_000_000
    else:
        result = run(node)
        assert result['producer_stamp_ns']-result['stationary_start_ns'] >= 100_000_000
        assert settles_after+.625 <= node.wall <= settles_after+.63
        assert node.events[-1][1]['timing_diagnostics']['stationary_samples'] >= 2


@pytest.mark.parametrize('case', ['no_callbacks', 'clock_paused', 'continuous_motion'])
def test_ten_second_wall_cap_cannot_substitute_for_producer_span(node, case):
    def sleep(seconds):
        node.wall += seconds; node.ticks += 1
        if case != 'clock_paused': node.now += round(seconds*.16*1e9)
        if case != 'no_callbacks':
            sample = raw(node.now)
            if case == 'continuous_motion': sample['velocities']['arm_left_1_joint'] = .01
            node.feed(sample)
    gate.time.sleep = sleep
    with pytest.raises(gate.PreopenStationaryRejected, match='measurement_timeout'): run(node)
    assert 10. <= node.wall <= 10.003
    info = node.events[-1][1]['timing_diagnostics']
    assert info['maximum_stationary_span_ns'] < 100_000_000 and info['final_attempts'] == 0
    assert not node._delivery_measurement_active


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'unrelated'])
def test_exact_helper_inverse_rejects_undeclared_changes(damage):
    text = (ROOT/'erc_phase1_solution/preopen_stationary.py').read_text()
    fragment = 'deadline = started_wall+10.'
    if damage == 'missing': text = text.replace(fragment, fragment.replace('10.', '11.'), 1)
    elif damage == 'duplicate': text += '\n# '+fragment+'\n'
    else: text += '\n# unrelated source\n'
    with pytest.raises(AssertionError): restore_preopen_timing_source(text)


def test_original_stationary_predicate_and_all_numeric_bounds_restore_exactly():
    text = (ROOT/'erc_phase1_solution/preopen_stationary.py').read_text()
    parent = restore_preopen_timing_source(text)
    def function(source, name):
        return ast.dump(next(n for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
    assert function(text, 'stationary_closed_sample') == function(parent, 'stationary_closed_sample')
    assert hashlib.sha256(parent.encode()).hexdigest() == 'd7fa3bdbf1b9bd2bccc7b5f65cba26de52a512bdaf7ae270db093f433b1f9450'
