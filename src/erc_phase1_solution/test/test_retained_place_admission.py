"""Actual installed-source method versus certified original; no ROS or asset imports."""
from __future__ import annotations

import ast
from candidate_composition_support import original_send_method
import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

PACKAGE = Path(__file__).resolve().parents[1] / 'erc_phase1_solution'
METHOD = '_send_retained_arm_trajectory'
CANDIDATE = PACKAGE / 'manipulation_node.py'
FIXTURE = Path(__file__).resolve().parent / 'fixtures/retained_place_send_original.py'
EXPECTED_FIXTURE_SHA256 = '5fbc333e5142dff28da565f408b0cac92d0595e40463799323baf67236d8bf30'


class LiftPressureRejected(RuntimeError):
    pass


class RetainedMotionNotStopped(RuntimeError):
    pass


def method_node(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(
        isinstance(m, ast.FunctionDef) and m.name == METHOD for m in n.body))
    return tree, owner, next(m for m in owner.body if isinstance(m, ast.FunctionDef) and m.name == METHOD)


def compile_method(path):
    method = deepcopy(method_node(path)[2])
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    scope = dict(Optional=object, LiftPressureRejected=LiftPressureRejected,
                 RetainedMotionNotStopped=RetainedMotionNotStopped,
                 GoalStatus=NS(STATUS_SUCCEEDED=4),
                 time=NS(monotonic=lambda: 100., sleep=lambda _: (_ for _ in ()).throw(AssertionError('unexpected wait'))))
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope[METHOD]


OLD = compile_method(FIXTURE)
NEW = compile_method(CANDIDATE)


class Event:
    value = False

    def is_set(self):
        return self.value

    def set(self):
        self.value = True


class RecordingLock:
    def __init__(self, node, name):
        self.node, self.name = node, name

    def __enter__(self):
        n = self.node
        # Deterministic interleaving after the first successful watchdog read,
        # immediately before a lock acquisition. No callback runs under a lock.
        if n.interrupt_armed and not n.interrupted and self.name in n.interrupt_locks:
            n.interrupted = True
            n.fault = n.interrupt_fault
            if n.interrupt_cancel:
                n._cancel.set()
            n.trace.append(('interrupt', self.name, n.fault, n._cancel.is_set()))
        if self.name == 'command':
            assert 'sensor' not in n.stack, 'sensor -> command lock inversion'
        n.stack.append(self.name)
        n.trace.append(('enter', self.name))
        return self

    def __exit__(self, *_):
        n = self.node
        assert n.stack.pop() == self.name
        n.trace.append(('exit', self.name))


class ImmediateFuture:
    def __init__(self, node, value):
        self.node, self.value = node, value

    def done(self):
        assert not self.node.stack, 'acceptance/result polling while locked'
        return True

    def result(self):
        assert not self.node.stack, 'acceptance/result inspection while locked'
        return self.value


class FakeNode:
    def __init__(self, *, place=True, accepted=True, fault=None, initial_cancel=False,
                 interrupt=False, interrupt_locks=('command', 'sensor'),
                 interrupt_fault='place_scene_contact', interrupt_cancel=True, send_raises=False):
        self.stack, self.trace, self.sent, self.status = [], [], [], []
        self._lock = RecordingLock(self, 'sensor')
        self.command_lock = RecordingLock(self, 'command')
        self._cancel = Event()
        if initial_cancel:
            self._cancel.set()
        self._place_contact_guard = object() if place else None
        self._goal_handles = []
        self._pending_retained_acceptances = set()
        self.timeout, self.grasp_contact_max_age = 20., .75
        self.fault, self.watchdog_reads = fault, 0
        self.interrupt_enabled = interrupt
        self.interrupt_armed = self.interrupted = False
        self.interrupt_locks = interrupt_locks
        self.interrupt_fault, self.interrupt_cancel = interrupt_fault, interrupt_cancel
        self.accepted, self.send_raises = accepted, send_raises
        self.arm_client = NS(send_goal_async=self.send)

    def _adaptive_command_guard(self):
        return self.command_lock

    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=100_000_000_000))

    def _payload_hazard_reason(self, **_):
        assert 'sensor' not in self.stack, 'watchdog would re-acquire sensor mutex'
        self.watchdog_reads += 1
        self.trace.append(('watchdog', self.fault))
        value = self.fault
        if self.watchdog_reads == 1 and self.interrupt_enabled:
            self.interrupt_armed = True
        return value

    def _publish_status(self, event, **fields):
        assert 'sensor' not in self.stack
        self.status.append((event, fields))

    def _trajectory_leg_at_elapsed_time(self, legs, _):
        return 0, legs[0][2]

    def _valid_retained_terminal_result(self, value):
        return value.status == 4

    def _cancel_retained_goal_and_confirm(self, *_):
        self.trace.append(('confirmed_cancel',))
        return True

    def send(self, goal):
        assert 'sensor' in self.stack
        assert len(self._pending_retained_acceptances) == 1
        self.sent.append((goal, tuple(self.stack)))
        if self.send_raises:
            raise RuntimeError('send failed')
        handle = NS(accepted=self.accepted)
        handle.get_result_async = lambda: ImmediateFuture(self, NS(status=4))
        return ImmediateFuture(self, handle)


class PressureGate:
    def __init__(self, node, rejected=False):
        self.node, self.rejected = node, rejected

    def send(self, request):
        if self.rejected:
            raise LiftPressureRejected('pressure rejected')
        with self.node._adaptive_command_guard():
            with self.node._lock:
                return request()


def invoke(function, node, gate=None):
    return function(node, 'goal', .8, [(None, .8, 'bin_transition')], 'place',
                    leg_offset=3, initial_pressure_gate=gate)


class ActualMethodTests(unittest.TestCase):
    def test_baseline_fixture_is_exact_original_method(self):
        # The fixture hash binds its original certified-source provenance and
        # method text. Preparation independently compared it to the full node.
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), EXPECTED_FIXTURE_SHA256)

    def test_only_actual_send_method_changes(self):
        # Removing only the scoped PLACE admission branch must restore the
        # certified original method, including all acceptance/error handling.
        original = method_node(FIXTURE)[2]
        current = original_send_method(CANDIDATE.read_text())
        expression = lambda text: ast.dump(ast.parse(text, mode='eval').body)
        pressure = [n for n in ast.walk(current) if isinstance(n, ast.If)
                    and ast.dump(n.test) == expression('initial_pressure_gate is not None')]
        self.assertEqual(len(pressure), 1)
        self.assertEqual(len(pressure[0].orelse), 1)
        branch = pressure[0].orelse[0]
        self.assertIsInstance(branch, ast.If)
        self.assertEqual(ast.dump(branch.test),
            expression("getattr(self, '_place_contact_guard', None) is not None"))
        pressure[0].orelse = branch.orelse
        self.assertEqual(ast.dump(current), ast.dump(original))

    def test_existing_clear_nonplace_behavior_exact(self):
        old, new = FakeNode(place=False), FakeNode(place=False)
        self.assertEqual(invoke(OLD, old), invoke(NEW, new))
        self.assertEqual(old.trace, new.trace)
        self.assertEqual(old.status, new.status)
        self.assertEqual(old.sent, new.sent)
        self.assertEqual(new._pending_retained_acceptances, set())

    def test_safe_place_send_is_under_command_then_sensor(self):
        node = FakeNode()
        self.assertEqual(invoke(NEW, node), (True, False))
        self.assertEqual(node.sent, [('goal', ('command', 'sensor'))])
        self.assertEqual(node.status, [])
        self.assertEqual(node._goal_handles, [])
        self.assertEqual(node._pending_retained_acceptances, set())
        self.assertEqual(node.stack, [])

    def test_recorded_race_window_baseline_publishes_after_latch_candidate_does_not(self):
        old, new = FakeNode(interrupt=True), FakeNode(interrupt=True)
        self.assertEqual(invoke(OLD, old), (False, False))
        self.assertEqual(len(old.sent), 1, 'baseline positive control must expose publication')
        self.assertEqual(invoke(NEW, new), (False, False))
        self.assertEqual(new.sent, [])
        self.assertEqual(new._pending_retained_acceptances, set())

    def test_cancel_after_final_watchdog_before_sensor_lock_prevents_send(self):
        node = FakeNode(interrupt=True, interrupt_locks=('sensor',), interrupt_fault=None)
        self.assertEqual(invoke(NEW, node), (False, False))
        self.assertEqual(node.watchdog_reads, 2)
        self.assertEqual(node.sent, [])
        self.assertEqual(node._pending_retained_acceptances, set())

    def test_fresh_hazard_preserves_grasp_lost_reason_leg_phase(self):
        node = FakeNode(interrupt=True, interrupt_cancel=False,
                        interrupt_fault='placement_scene_stale_odometry')
        self.assertEqual(invoke(NEW, node), (False, True))
        self.assertEqual(node.sent, [])
        self.assertEqual(node.status, [('grasp_lost', dict(command='place', phase='bin_transition',
            leg=3, reason='placement_scene_stale_odometry'))])
        self.assertEqual(node._pending_retained_acceptances, set())

    def test_initial_cancel_behavior_unchanged(self):
        for function in (OLD, NEW):
            node = FakeNode(initial_cancel=True)
            self.assertEqual(invoke(function, node), (False, False))
            self.assertEqual((node.sent, node.status, node.trace), ([], [], []))

    def test_initial_hazard_reporting_unchanged(self):
        old, new = FakeNode(fault='contact_lost'), FakeNode(fault='contact_lost')
        self.assertEqual(invoke(OLD, old), invoke(NEW, new))
        self.assertEqual(old.status, new.status)
        self.assertEqual(old.trace, new.trace)
        self.assertEqual(new.sent, [])

    def test_pressure_gated_admission_unchanged_even_with_place_guard(self):
        old, new = FakeNode(), FakeNode()
        self.assertEqual(invoke(OLD, old, PressureGate(old)), invoke(NEW, new, PressureGate(new)))
        self.assertEqual(old.trace, new.trace)
        self.assertEqual(old.sent, new.sent)
        self.assertEqual(old.watchdog_reads, new.watchdog_reads)

    def test_pressure_rejection_keeps_original_exception(self):
        for function in (OLD, NEW):
            node = FakeNode()
            with self.assertRaisesRegex(LiftPressureRejected, '^pressure rejected$'):
                invoke(function, node, PressureGate(node, rejected=True))
            self.assertEqual(node.sent, [])
            self.assertFalse(node._cancel.is_set())

    def test_server_rejection_cleans_pending_token(self):
        node = FakeNode(accepted=False)
        self.assertEqual(invoke(NEW, node), (False, False))
        self.assertEqual(node._pending_retained_acceptances, set())
        self.assertEqual(node._goal_handles, [])

    def test_uncertain_send_exception_preserves_original_failure_policy(self):
        for function in (OLD, NEW):
            node = FakeNode(send_raises=True)
            with self.assertRaisesRegex(RetainedMotionNotStopped,
                    '^retained arm goal acceptance has no confirmed terminal state$'):
                invoke(function, node)
            self.assertTrue(node._cancel.is_set())
            self.assertEqual(len(node._pending_retained_acceptances), 1)
            self.assertEqual(node.stack, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
