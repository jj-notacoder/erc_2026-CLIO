"""Real message/worker races: an overload hold must win over closing commands."""
from collections import deque
from types import SimpleNamespace
import threading
import time

import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import JointState

from erc_phase1_solution import manipulation_node as m
from test_contact_force_timing import contact_node, force_message, BOOK, LEFT


def motion_node():
    node, now = contact_node()
    node._lock = threading.Lock()
    node._adaptive_command_lock = threading.RLock()
    node._adaptive_motion_started = False
    node._adaptive_motion_halt_reason = None
    node._adaptive_hold_sent = False
    node._cancel = threading.Event()
    node._gripper_feedback_samples = deque([
        m.GripperFeedback(now.nanoseconds, .0188, 0., .1)], maxlen=128)
    node.adaptive_step_motion, node.adaptive_step_settle = .32, .10
    node.adaptive_effort_maximum, node.adaptive_effort_delta_maximum = 4., 3.
    node._adaptive_effort_baseline_value = .1
    node.timeout, node.gripper_settle = .2, 1.
    messages, statuses = [], []

    def publish(message):
        # A callback must release its sensor mutex before publishing a hold.
        assert node._lock.acquire(blocking=False)
        node._lock.release()
        messages.append(message)
    node.gripper_pub = SimpleNamespace(publish=publish)
    node._publish_status = lambda event, **fields: statuses.append((event, fields))
    return node, now, messages, statuses


def fake_progress(monkeypatch, node, now, hook=None):
    invoked = False
    def sleep(seconds):
        nonlocal invoked
        now.nanoseconds += int(seconds * 1e9)
        node._gripper_feedback_samples.append(
            m.GripperFeedback(now.nanoseconds, .0188, 0., .1))
        if hook is not None and not invoked:
            invoked = True
            hook()
    monkeypatch.setattr(m.time, 'sleep', sleep)


def test_ordinary_adaptive_step_is_one_publication_not_delayed_duplicate(monkeypatch):
    node, now, messages, _ = motion_node()
    fake_progress(monkeypatch, node, now)
    assert node._command_adaptive_gripper_step(.018)
    assert len(messages) == 1
    assert list(messages[0].points[0].positions) == [.018]


def test_force_callback_interrupts_before_step_end_and_no_later_close_can_replace_hold(monkeypatch):
    node, now, messages, statuses = motion_node()
    begin = now.nanoseconds
    fake_progress(monkeypatch, node, now,
                  lambda: node._on_contacts(force_message(now.nanoseconds, (LEFT, BOOK, 9.))))
    assert not node._command_adaptive_gripper_step(.018)
    assert now.nanoseconds - begin < 10_000_000
    assert [msg.points[0].positions[0] for msg in messages] == [.018, .0188]
    assert list(messages[-1].points[0].velocities) == [0.]
    assert statuses[-1][1]['reason'] == 'force_overload'
    assert statuses[-1][1]['hold_published'] is True
    assert not node._command_adaptive_gripper_step(.017)
    assert len(messages) == 2
    assert not node._cancel.is_set()


@pytest.mark.parametrize('effort,expected', [(5., 'effort_overload'), (3.5, 'effort_delta_overload')])
def test_effort_callback_holds_only_after_releasing_sensor_lock(effort, expected):
    node, now, messages, statuses = motion_node()
    node._adaptive_motion_started = True
    message = JointState()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(now.nanoseconds, 10**9)
    message.name, message.position, message.velocity, message.effort = (
        ['gripper_left_finger_joint'], [.0187], [0.], [effort])
    node._on_joint_state(message)
    assert len(messages) == 1
    assert list(messages[0].points[0].positions) == [.0187]
    assert statuses[-1][1]['reason'] == expected
    assert not node._cancel.is_set()


def test_queued_inward_command_cannot_overwrite_callback_hold():
    node, now, messages, _ = motion_node()
    first_publish_started, release_first = threading.Event(), threading.Event()
    original_publish = node.gripper_pub.publish
    def blocked_publish(message):
        original_publish(message)
        if list(message.points[0].positions) == [.018]:
            first_publish_started.set()
            assert release_first.wait(2.)
    node.gripper_pub.publish = blocked_publish
    outcomes, errors = [], []
    def step(target):
        try:
            outcomes.append(node._command_adaptive_gripper_step(target))
        except Exception as exc:
            errors.append(exc)
    first = threading.Thread(target=step, args=(.018,))
    first.start()
    assert first_publish_started.wait(1.)
    callback = threading.Thread(target=lambda: node._on_contacts(
        force_message(now.nanoseconds, (LEFT, BOOK, 9.))))
    callback.start()
    deadline = time.monotonic() + 1.
    while node._adaptive_overload_latched is None and time.monotonic() < deadline:
        time.sleep(.001)
    assert node._adaptive_overload_latched == 'force_overload'
    queued = threading.Thread(target=step, args=(.017,))
    queued.start()
    release_first.set()
    for thread in (first, callback, queued):
        thread.join(timeout=2.)
        assert not thread.is_alive()
    assert not errors
    assert outcomes == [False, False]
    assert [msg.points[0].positions[0] for msg in messages] == [.018, .0188]


@pytest.mark.parametrize('feedback', [
    None,
    m.GripperFeedback(1, .018, 0., .1),
    m.GripperFeedback(10_000_000_000, float('nan'), 0., .1),
    m.GripperFeedback(10_000_000_000, .018, float('nan'), .1),
])
def test_invalid_hold_feedback_never_publishes_an_invented_position(feedback):
    node, _, messages, statuses = motion_node()
    node._gripper_feedback_samples.clear()
    if feedback is not None:
        node._gripper_feedback_samples.append(feedback)
    assert not node._command_adaptive_gripper_step(.018)
    assert not messages
    assert statuses[-1][1]['hold_published'] is False
    assert statuses[-1][1]['feedback_reason']
    assert not node._cancel.is_set()


def test_hold_preserves_checked_reopening_and_blocks_stale_callback_after_close(monkeypatch):
    node, now, messages, _ = motion_node()
    node._adaptive_motion_started = True
    node._adaptive_overload_latched = 'force_overload'
    node._interrupt_adaptive_gripper_if_fault()
    assert list(messages[-1].points[0].positions) == [.0188]
    # Normal adaptive-scope teardown precedes the existing checked recovery.
    node._adaptive_close_active = False
    node._adaptive_motion_started = False
    node._wait_sim_duration = lambda seconds: True
    monkeypatch.setattr(m.time, 'sleep', lambda seconds: None)
    assert node._command_gripper(.069)
    node._interrupt_adaptive_gripper_if_fault()
    assert [msg.points[0].positions[0] for msg in messages] == [.0188, .069, .069]
    assert not node._cancel.is_set()


def test_fault_at_close_completion_cannot_return_verified_acquisition():
    node, _, messages, _ = motion_node()
    def attempt():
        node._adaptive_motion_started = True
        node._adaptive_overload_latched = 'force_overload'
        node._transport_lock_engaged = True
        return True, .0188, True, True, True
    node._adaptive_close_attempt = attempt
    result = node._adaptive_close_for_grasp()
    assert result == (False, .0188, True, True, True)
    assert len(messages) == 1
    assert node._adaptive_close_active is False
    assert node._adaptive_motion_started is False
    assert node._adaptive_motion_halt_reason == 'force_overload'
    assert node._transport_lock_engaged is False


def test_monitor_disarm_has_no_gap_for_a_fault_after_its_final_snapshot():
    node, _, _, _ = motion_node()
    attempt_finished = False
    injected_faults = []
    original_lock = node._lock

    class InterleavedSensorLock:
        def acquire(self, *args, **kwargs):
            return original_lock.acquire(*args, **kwargs)

        def release(self):
            # Simulate a callback winning the lock immediately after the
            # final snapshot, if that snapshot left monitoring active.
            if attempt_finished and node._adaptive_close_active:
                injected_faults.append('force_overload')
                node._adaptive_overload_latched = 'force_overload'
            original_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *args):
            self.release()

    node._lock = InterleavedSensorLock()

    def attempt():
        nonlocal attempt_finished
        node._adaptive_motion_started = True
        node._transport_lock_engaged = True
        attempt_finished = True
        return True, .0188, True, True, True

    node._adaptive_close_attempt = attempt
    result = node._adaptive_close_for_grasp()
    if injected_faults:
        assert result[0] is False
        assert node._transport_lock_engaged is False
    assert node._adaptive_close_active is False
