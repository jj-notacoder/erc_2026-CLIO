"""Real retained-action methods with fake action futures; no ROS publications."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution import manipulation_node as module


class Clock:
    def __init__(self):
        self.seconds = 10.0

    def sleep(self, seconds):
        self.seconds += seconds

    def monotonic(self):
        return self.seconds

    def now(self):
        return SimpleNamespace(nanoseconds=round(self.seconds * 1e9))


class Future:
    def __init__(self, ready, value):
        self.ready, self.value = ready, value
        self.callbacks = []
        self.notified = False

    def done(self):
        ready = self.ready()
        if ready and not self.notified:
            self.notified = True
            for callback in tuple(self.callbacks):
                callback(self)
        return ready

    def result(self):
        return self.value()

    def add_done_callback(self, callback):
        if self.done():
            callback(self)
        else:
            self.callbacks.append(callback)


def fixture(monkeypatch, *, fault_at=None, fault='contact_lost',
            acceptance_delay=0., accepted=True, cancel_terminal=True,
            terminal_hook=None):
    clock = Clock()
    monkeypatch.setattr(module.time, 'sleep', clock.sleep)
    monkeypatch.setattr(module.time, 'monotonic', clock.monotonic)
    node = object.__new__(module.ManipulationNode)
    node._cancel = threading.Event()
    node._lock = threading.RLock()
    node._goal_handles = []
    node._busy = False
    node.timeout = .25
    node.grasp_contact_max_age = .75
    node.get_clock = lambda: clock
    node._payload_hazard_reason = lambda **kwargs: (
        fault if fault_at is not None and clock.seconds >= 10. + fault_at else None)
    node.events = []
    node._publish_status = lambda event, **fields: node.events.append((event, fields))
    node.sent, node.cancelled, node.probed = [], [], []
    node.acceptance_futures, node.result_futures = [], []

    def send(goal):
        sent_at = clock.seconds
        point = goal.trajectory.points[-1]
        duration = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
        handle = SimpleNamespace(accepted=accepted, cancelled_at=None)

        def terminal():
            if handle.cancelled_at is not None:
                return cancel_terminal and clock.seconds >= handle.cancelled_at + .04
            return clock.seconds >= sent_at + duration

        def result():
            if terminal_hook is not None:
                terminal_hook(node)
            return SimpleNamespace(status=(
                module.GoalStatus.STATUS_CANCELED if handle.cancelled_at is not None
                else module.GoalStatus.STATUS_SUCCEEDED))

        result_future = Future(terminal, result)
        node.result_futures.append(result_future)
        handle.get_result_async = lambda: result_future

        def cancel():
            node.cancelled.append((clock.seconds, handle))
            handle.cancelled_at = clock.seconds
            return Future(lambda: True, lambda: SimpleNamespace())

        handle.cancel_goal_async = cancel
        node.sent.append((clock.seconds, goal, handle))
        future = Future(lambda: clock.seconds >= sent_at + acceptance_delay, lambda: handle)
        node.acceptance_futures.append(future)
        return future

    node.arm_client = SimpleNamespace(send_goal_async=send)
    node._retention_after_leg = lambda command, phase, leg: (
        node.probed.append((command, phase, leg)) or True)
    node._fresh_retention_probe = lambda command, phase, **kw: (
        node.probed.append((command, phase, kw['leg'], 'fresh')) or True)
    node._move_arm_solution = lambda *a, **kw: pytest.fail('unmonitored action used')
    return node, clock


def legs():
    return [(np.zeros(8), 5.8, 'extraction'), (np.ones(8), .65, 'must_not_run')]


@pytest.mark.parametrize('reason', ['contact_lost', 'payload_robot_contact'])
def test_mid_leg_hazard_cancels_before_endpoint_and_preserves_partial_index(monkeypatch, reason):
    node, clock = fixture(monkeypatch, fault_at=.30, fault=reason)
    result = node._execute_retained_arm_legs(legs(), 'pick', leg_offset=4)
    assert result == (False, 0, True)
    assert len(node.sent) == len(node.cancelled) == 1
    assert node.cancelled[0][0] - 10. < .34
    assert clock.seconds < 10.5  # Not the former5.8-second end-of-leg check.
    assert node._goal_handles == []
    assert not node._cancel.is_set()  # Checked normal recovery is still permitted.
    assert node.probed == []
    assert node.events == [('grasp_lost', dict(
        command='pick', phase='extraction', leg=4, reason=reason))]


def test_hazard_before_goal_publication_sends_nothing(monkeypatch):
    node, _ = fixture(monkeypatch, fault_at=0.)
    assert node._execute_retained_arm_legs(legs(), 'pick') == (False, 0, True)
    assert node.sent == node.cancelled == node.probed == []


def test_hazard_while_goal_acceptance_is_pending_cancels_accepted_goal(monkeypatch):
    node, clock = fixture(monkeypatch, fault_at=.06, acceptance_delay=.12)
    assert node._execute_retained_arm_legs(legs(), 'pick') == (False, 0, True)
    assert len(node.cancelled) == 1
    assert clock.seconds < 10.25
    assert node._goal_handles == []
    assert node.probed == []


def test_transient_hazard_while_acceptance_pending_cannot_be_erased(monkeypatch):
    node, clock = fixture(monkeypatch, acceptance_delay=.16)
    node._payload_hazard_reason = lambda **kwargs: (
        'contact_lost' if 10.04 <= clock.seconds < 10.1 else None)
    assert node._execute_retained_arm_legs(legs(), 'pick') == (False, 0, True)
    assert len(node.cancelled) == 1
    assert node.events[0][1]['reason'] == 'contact_lost'
    assert not node._cancel.is_set()
    assert node.probed == []


def test_unknown_acceptance_is_terminal_and_late_goal_is_cancelled(monkeypatch):
    node, clock = fixture(monkeypatch, acceptance_delay=.5)
    with pytest.raises(module.RetainedMotionNotStopped, match='acceptance'):
        node._execute_retained_arm_legs(legs(), 'pick')
    assert node._cancel.is_set()
    assert node._pending_retained_acceptances
    assert len(node.sent) == 1
    assert node.probed == node.cancelled == []
    clock.sleep(.30)
    assert node.acceptance_futures[0].done()  # Deliver the late acceptance callback.
    assert len(node.cancelled) == len(node._goal_handles) == 1
    clock.sleep(.05)
    assert node.result_futures[0].done()
    assert node._goal_handles == []
    assert node._pending_retained_acceptances == set()


def test_late_rejected_goal_after_timeout_never_cancels_or_moves_again(monkeypatch):
    node, clock = fixture(monkeypatch, acceptance_delay=.5, accepted=False)
    with pytest.raises(module.RetainedMotionNotStopped):
        node._execute_retained_arm_legs(legs(), 'pick')
    clock.sleep(.30)
    assert node.acceptance_futures[0].done()
    assert node._goal_handles == node.cancelled == node.probed == []
    assert node._pending_retained_acceptances == set()
    assert len(node.sent) == 1


def test_send_exception_is_unknown_acceptance_not_recoverable_failure(monkeypatch):
    node, _ = fixture(monkeypatch)
    node.arm_client.send_goal_async = lambda goal: (_ for _ in ()).throw(
        RuntimeError('request transport failed'))
    with pytest.raises(module.RetainedMotionNotStopped, match='acceptance'):
        node._execute_retained_arm_legs(legs(), 'pick')
    assert node._cancel.is_set()
    assert node.probed == []
    assert node._pending_retained_acceptances


def test_new_command_is_refused_while_goal_acceptance_remains_unknown(monkeypatch):
    node, _ = fixture(monkeypatch, acceptance_delay=.5)
    with pytest.raises(module.RetainedMotionNotStopped):
        node._execute_retained_arm_legs(legs(), 'pick')
    monkeypatch.setattr(module.threading, 'Thread', lambda **kw: pytest.fail('new worker started'))
    node._on_command(SimpleNamespace(data=module.encode_event('open_gripper')))
    assert node.events[-1] == ('rejected', dict(
        command='open_gripper', reason='retained_goal_acceptance_unresolved'))
    assert node._cancel.is_set()


@pytest.mark.parametrize('kind', ['exception', 'malformed', 'nonterminal', 'boolean'])
def test_late_invalid_result_preserves_admission_interlock(monkeypatch, kind):
    node, clock = fixture(monkeypatch, acceptance_delay=.5)
    with pytest.raises(module.RetainedMotionNotStopped):
        node._execute_retained_arm_legs(legs(), 'pick')
    values = {
        'malformed': lambda: SimpleNamespace(),
        'nonterminal': lambda: SimpleNamespace(status=module.GoalStatus.STATUS_EXECUTING),
        'boolean': lambda: SimpleNamespace(status=True),
        'exception': lambda: (_ for _ in ()).throw(RuntimeError('result unavailable')),
    }
    node.result_futures[0].value = values[kind]
    clock.sleep(.30)
    assert node.acceptance_futures[0].done()
    clock.sleep(.05)
    assert node.result_futures[0].done()
    assert len(node.cancelled) == len(node._goal_handles) == 1
    assert node._pending_retained_acceptances
    monkeypatch.setattr(module.threading, 'Thread', lambda **kw: pytest.fail('new worker started'))
    node._on_command(SimpleNamespace(data=module.encode_event('stow')))
    assert node.events[-1][1]['reason'] == 'retained_goal_acceptance_unresolved'
    assert node._cancel.is_set()


def test_exception_obtaining_accepted_result_requests_cancel_before_terminal_error(monkeypatch):
    node, _ = fixture(monkeypatch)
    original_send = node.arm_client.send_goal_async
    def send(goal):
        future = original_send(goal)
        node.sent[-1][2].get_result_async = lambda: (_ for _ in ()).throw(
            RuntimeError('result transport failed'))
        return future
    node.arm_client.send_goal_async = send
    with pytest.raises(module.RetainedMotionNotStopped, match='accepted retained arm'):
        node._execute_retained_arm_legs(legs(), 'pick')
    assert len(node.cancelled) == len(node._goal_handles) == 1
    assert node._cancel.is_set()
    assert node.probed == []


def test_done_but_malformed_cancel_result_is_not_a_confirmed_stop(monkeypatch):
    node, _ = fixture(monkeypatch, fault_at=.1)
    original_send = node.arm_client.send_goal_async
    def send(goal):
        future = original_send(goal)
        node.result_futures[-1].value = lambda: SimpleNamespace()
        return future
    node.arm_client.send_goal_async = send
    with pytest.raises(module.RetainedMotionNotStopped, match='could not be cancelled'):
        node._execute_retained_arm_legs(legs(), 'pick')
    assert len(node.cancelled) == len(node._goal_handles) == 1
    assert node._cancel.is_set()


def test_hazard_racing_terminal_success_never_advances(monkeypatch):
    def latch(node):
        node._payload_hazard_reason = lambda **kwargs: 'contact_lost'
    node, _ = fixture(monkeypatch, terminal_hook=latch)
    assert node._execute_retained_arm_legs(legs(), 'pick') == (False, 0, True)
    assert len(node.sent) == 1
    assert node.probed == []
    assert node._goal_handles == []


def test_external_cancel_racing_terminal_success_never_advances(monkeypatch):
    node, _ = fixture(monkeypatch, terminal_hook=lambda node: node._cancel.set())
    assert node._execute_retained_arm_legs(legs(), 'pick') == (False, 0, False)
    assert len(node.sent) == 1
    assert node.probed == []
    assert node._goal_handles == []


def test_unconfirmed_cancellation_propagates_without_recovery(monkeypatch):
    node, clock = fixture(monkeypatch, fault_at=.30, cancel_terminal=False)
    with pytest.raises(module.RetainedMotionNotStopped, match='could not be cancelled'):
        node._execute_retained_arm_legs(legs(), 'pick')
    assert len(node.sent) == len(node.cancelled) == len(node._goal_handles) == 1
    assert node._cancel.is_set()
    assert node.probed == []
    assert clock.seconds < 11.


def test_exception_after_acceptance_is_not_recoverable_motion_failure(monkeypatch):
    node, _ = fixture(monkeypatch)
    def fail(goal, timeout, active, command, **kwargs):
        node._goal_handles.append(object())
        raise RuntimeError('result transport failed')
    node._send_retained_arm_trajectory = fail
    with pytest.raises(module.RetainedMotionNotStopped, match='confirmed stop'):
        node._execute_retained_arm_legs(legs(), 'pick')
    assert node._cancel.is_set()
    assert node.probed == []


def test_rejected_goal_does_not_run_endpoint_probe_or_next_leg(monkeypatch):
    node, _ = fixture(monkeypatch, accepted=False)
    assert node._execute_retained_arm_legs(legs(), 'pick') == (False, 0, False)
    assert len(node.sent) == 1
    assert node.cancelled == node.probed == node._goal_handles == []


def test_success_preserves_every_goal_duration_and_endpoint_probe(monkeypatch):
    node, _ = fixture(monkeypatch)
    route = legs()
    assert node._execute_retained_arm_legs(
        route, 'pick', leg_offset=3, fresh_retention_phases=('must_not_run',)
    ) == (True, 2, False)
    assert len(node.sent) == 2
    assert node.cancelled == node._goal_handles == []
    assert node.probed == [('pick', 'extraction', 3), ('pick', 'must_not_run', 4, 'fresh')]
    for (_, goal, _), (q, duration, _) in zip(node.sent, route):
        point = goal.trajectory.points[0]
        assert point.positions == pytest.approx(q[1:])
        assert point.time_from_start.sec + point.time_from_start.nanosec / 1e9 == pytest.approx(duration)


def test_endpoint_only_contact_failure_uses_next_index(monkeypatch):
    node, _ = fixture(monkeypatch)
    node._retention_after_leg = lambda *args: False
    assert node._execute_retained_arm_legs(legs(), 'pick') == (False, 1, True)
    assert len(node.sent) == 1
    assert node.cancelled == []


@pytest.mark.parametrize('supported,robot_contact', [(False, False), (True, False), (False, True)])
def test_confirmed_partial_contact_loss_recovery_never_replays_route(monkeypatch, supported, robot_contact):
    node, _ = fixture(monkeypatch, fault_at=.1)
    ok, index, lost = node._execute_retained_arm_legs(legs(), 'pick')
    assert (ok, index, lost) == (False, 0, True)
    node._gravity_supported_payload = supported
    node._target_robot_contact_latched = robot_contact
    opened = []
    node._open_gripper = lambda: opened.append(True) or True
    with pytest.raises(RuntimeError, match='pick_payload_lost'):
        node._recover_closed_pick(
            cause='contact_lost', remaining_carried_legs=legs()[index:],
            unloaded_recovery_route=[np.ones(8)])
    assert opened == ([] if supported or robot_contact else [True])
    assert len(node.sent) == 1
