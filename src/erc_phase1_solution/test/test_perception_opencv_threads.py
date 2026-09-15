"""Actual perception main with stdlib doubles; no ROS or OpenCV import."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).parents[1] / 'erc_phase1_solution/perception_node.py'


class ExternalShutdown(Exception):
    pass


def invoke(*, spin_error=None, cap_error=None, okay=True, args=None):
    events = []

    def cap(count):
        events.append(('opencv_threads', count))
        if cap_error is not None:
            raise cap_error

    class Node:
        def __init__(self):
            events.append(('node',))

        def destroy_node(self):
            events.append(('destroy',))

    def spin(node):
        events.append(('spin', isinstance(node, Node)))
        if spin_error is not None:
            raise spin_error

    scope = dict(
        cv2=SimpleNamespace(setNumThreads=cap),
        rclpy=SimpleNamespace(
            init=lambda **kw: events.append(('init', kw)), spin=spin,
            ok=lambda: okay, shutdown=lambda: events.append(('shutdown',))),
        PerceptionNode=Node, ExternalShutdownException=ExternalShutdown,
    )
    main = next(node for node in ast.parse(SOURCE.read_text(encoding='utf-8')).body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(SOURCE), 'exec'), scope)
    error = None
    try:
        scope['main'](args=args)
    except BaseException as exc:
        error = exc
    return events, error


class PerceptionOpenCVThreads(unittest.TestCase):
    def test_caps_once_before_ros_and_node_with_original_arguments(self):
        args = ['--ros-args', '-p', 'use_sim_time:=true']
        events, error = invoke(args=args)
        self.assertIsNone(error)
        self.assertEqual(events, [('opencv_threads', 1), ('init', {'args': args}),
                                  ('node',), ('spin', True), ('destroy',), ('shutdown',)])

    def test_cap_failure_never_initializes_ros_or_constructs_node(self):
        fault = RuntimeError('thread backend failed')
        events, error = invoke(cap_error=fault)
        self.assertIs(error, fault)
        self.assertEqual(events, [('opencv_threads', 1)])

    def test_external_shutdown_still_destroys_without_double_shutdown(self):
        events, error = invoke(spin_error=ExternalShutdown(), okay=False)
        self.assertIsNone(error)
        self.assertEqual(events[-1], ('destroy',))
        self.assertNotIn(('shutdown',), events)

    def test_invalid_context_runtime_error_retains_original_shutdown_policy(self):
        events, error = invoke(spin_error=RuntimeError('invalid context'), okay=False)
        self.assertIsNone(error)
        self.assertEqual(events[-1], ('destroy',))

    def test_genuine_runtime_error_is_not_hidden(self):
        fault = RuntimeError('processing failed')
        events, error = invoke(spin_error=fault, okay=True)
        self.assertIs(error, fault)
        self.assertEqual(events[-2:], [('destroy',), ('shutdown',)])

    def test_keyboard_interrupt_keeps_cleanup(self):
        events, error = invoke(spin_error=KeyboardInterrupt())
        self.assertIsNone(error)
        self.assertEqual(events[-2:], [('destroy',), ('shutdown',)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
