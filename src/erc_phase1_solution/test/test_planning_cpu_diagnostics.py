"""Portable stdlib tests; no ROS, model, NumPy or geometry imports."""
import ast
from candidate_composition_support import restore_to_r51_source
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'erc_phase1_solution/planning_cpu_diagnostics.py'
NODE = ROOT / 'erc_phase1_solution/manipulation_node.py'
BASE = ROOT / 'test/fixtures/planning_cpu_original_node.py'
BASE_SHA = 'ad241f8d27b4879b85f0b9432f0311b75129da66d52638e93417e63386e65b38'
spec = importlib.util.spec_from_file_location('planning_cpu_test_module', MODULE)
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


class Clock:
    def __init__(self):
        self.wall, self.cpu, self.tid = 100, 10, 111

    def recorder(self):
        return cpu.PlanningCpuDiagnostics(wall=lambda: self.wall,
            cpu=lambda: self.cpu, native_tid=lambda: self.tid)

    def advance(self, wall=20, thread_cpu=5):
        self.wall += wall
        self.cpu += thread_cpu


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.recorder = self.clock.recorder()
        self.node = types.SimpleNamespace(_planning_cpu_diagnostics=self.recorder)

    def test_callback_preserves_arguments_result_and_inclusive_time(self):
        seen = []
        result = object()
        def original(*args, **kwargs):
            seen.append((args, kwargs))
            self.clock.advance()
            return result
        token = self.recorder.command_begin('place')
        wrapped = cpu.callback(self.node, '_on_contacts', original)
        self.assertIs(wrapped(4, force=9), result)
        self.assertEqual(seen, [((4,), {'force': 9})])
        row = self.recorder.snapshot('started', 'place')['callbacks']['_on_contacts']
        self.assertEqual((row['completed'], row['exceptional'],
                          row['inclusive_wall_ns'], row['inclusive_thread_cpu_ns']),
                         (1, 0, 20, 5))
        self.assertEqual(wrapped.__name__, original.__name__)
        self.recorder.command_end(token)

    def test_original_callback_exception_survives_diagnostic_finish_failure(self):
        error = RuntimeError('original contact error')
        def original():
            raise error
        self.recorder.callback_end = lambda *args: (_ for _ in ()).throw(ValueError('diag'))
        with self.assertRaises(RuntimeError) as caught:
            cpu.callback(self.node, '_on_contacts', original)()
        self.assertIs(caught.exception, error)

    def test_begin_and_attribute_failures_do_not_skip_callback(self):
        class BrokenNode:
            @property
            def _planning_cpu_diagnostics(self):
                raise RuntimeError('diagnostic property')
        self.recorder.callback_begin = lambda *args: (_ for _ in ()).throw(ValueError('diag'))
        for node in (self.node, BrokenNode(), types.SimpleNamespace()):
            seen = []
            self.assertEqual(cpu.callback(node, '_on_contacts', lambda x: seen.append(x) or 7)(3), 7)
            self.assertEqual(seen, [3])

    def test_exceptional_and_non_exception_baseexceptions_are_preserved(self):
        for error in (ValueError('failure'), KeyboardInterrupt(), SystemExit(3)):
            token = self.recorder.command_begin('place')
            def original():
                self.clock.advance()
                raise error
            try:
                cpu.callback(self.node, '_on_contacts', original)()
            except BaseException as caught:
                self.assertIs(caught, error)
            else:
                self.fail('original exception swallowed')
            self.assertEqual(self.recorder.snapshot('failed', 'place')['callbacks']['_on_contacts']['exceptional'], 1)
            self.recorder.command_end(token)

    def test_command_role_is_actual_entry_thread_and_exact_arguments(self):
        self.recorder.mark_executor()
        self.clock.tid = 222
        seen = []
        def original(command, payload, *, extra):
            self.clock.advance(55, 17)
            fields = {'command': command}
            cpu.add_status_fields(self.node, 'place_planning_stage', fields)
            seen.append((payload, extra, fields['planning_cpu']))
            return False
        payload = {'trial_id': 'actual'}
        self.assertIs(cpu.command_target(self.node, original)('place', payload, extra=7), False)
        self.assertIs(seen[0][0], payload)
        row = seen[0][2]
        self.assertEqual((row['executor_native_tid'], row['worker_native_tid']), (111, 222))
        self.assertEqual((row['command_monotonic_ns'], row['command_thread_cpu_ns']), (55, 17))
        self.assertFalse(row['gil_wait_measured'])
        self.assertIsNone(self.recorder._active)

    def test_command_original_exception_and_finish_failure(self):
        error = RuntimeError('original command error')
        def original(command):
            raise error
        self.recorder.command_end = lambda *args: (_ for _ in ()).throw(ValueError('diag'))
        with self.assertRaises(RuntimeError) as caught:
            cpu.command_target(self.node, original)('place')
        self.assertIs(caught.exception, error)

    def test_command_start_failure_preserves_return_and_one_call(self):
        seen = []
        self.recorder.command_begin = lambda *args: (_ for _ in ()).throw(ValueError('diag'))
        self.assertEqual(cpu.command_target(self.node, lambda *a: seen.append(a) or 'done')('pick', 4), 'done')
        self.assertEqual(seen, [('pick', 4)])

    def test_foreign_thread_and_unknown_events_are_not_misattributed(self):
        self.recorder.command_begin('place')
        self.assertIsNone(self.recorder.snapshot('ik_ready', 'place'))
        self.assertIsNone(self.recorder.snapshot('started', 'pick'))
        self.clock.tid = 999
        self.assertIsNone(self.recorder.snapshot('place_planning_stage', 'place'))

    def test_stage_cap_terminal_record_and_next_command_reset(self):
        token = self.recorder.command_begin('place')
        for _ in range(cpu.MAX_STAGE_RECORDS):
            self.assertIsNotNone(self.recorder.snapshot('place_planning_stage', 'place'))
        self.assertIsNone(self.recorder.snapshot('place_planning_stage', 'place'))
        self.assertIsNotNone(self.recorder.snapshot('succeeded', 'place'))
        self.recorder.command_end(token)
        self.recorder.command_begin('place')
        self.assertEqual(self.recorder.snapshot('started', 'place')['stage_records'], 0)

    def test_fixed_counter_names_saturation_and_no_mutable_view(self):
        self.assertIsNone(self.recorder.callback_begin('unknown-callback'))
        self.assertEqual(tuple(self.recorder._totals), cpu.CALLBACK_NAMES)
        self.recorder._totals['_on_contacts'][:4] = [cpu.MAX_COUNTER-1] * 4
        self.recorder.command_begin('place')
        token = self.recorder.callback_begin('_on_contacts')
        self.clock.advance()
        self.recorder.callback_end(token, True)
        first = self.recorder.snapshot('started', 'place')
        row = first['callbacks']['_on_contacts']
        self.assertTrue(row['saturated'])
        self.assertEqual([row[k] for k in ('completed', 'exceptional', 'inclusive_wall_ns', 'inclusive_thread_cpu_ns')], [1]*4)
        row['completed'] = -100
        self.assertEqual(self.recorder.snapshot('started', 'place')['callbacks']['_on_contacts']['completed'], 1)

    def test_backward_or_bad_clock_drops_only_diagnostic(self):
        token = self.recorder.callback_begin('_on_contacts')
        self.clock.wall -= 1
        self.recorder.callback_end(token, False)
        self.assertEqual(self.recorder._totals['_on_contacts'][0], 0)
        self.clock.wall = float('nan')
        self.assertEqual(cpu.callback(self.node, '_on_contacts', lambda: 42)(), 42)

    def test_status_transaction_preserves_fields_on_bad_diagnostic(self):
        originals = {'command': 'place', 'stage': 'exact_stage'}
        for value in ({'bad': float('nan')}, {'bad': object()}, {'large': 'x'*17000}):
            self.recorder.snapshot = lambda *args, value=value: value
            fields = dict(originals)
            cpu.add_status_fields(self.node, 'place_planning_stage', fields)
            self.assertEqual(fields, originals)
        fields = dict(originals, planning_cpu='caller-value')
        cpu.add_status_fields(self.node, 'place_planning_stage', fields)
        self.assertEqual(fields['planning_cpu'], 'caller-value')

    def test_overlapping_command_does_not_replace_or_clear_owner(self):
        first = self.recorder.command_begin('place')
        self.assertIsNone(self.recorder.command_begin('pick'))
        self.recorder.command_end(None)
        self.assertIs(self.recorder._active[0], first)
        self.recorder.command_end(first)
        self.assertIsNone(self.recorder._active)

    def test_real_thread_identity_and_callback_lock_are_independent(self):
        recorder = cpu.PlanningCpuDiagnostics()
        node = types.SimpleNamespace(_planning_cpu_diagnostics=recorder)
        cpu.mark_executor_thread(node)
        main_tid = threading.get_native_id()
        seen = []
        def worker(command):
            fields = {'command': command}
            cpu.add_status_fields(node, 'started', fields)
            seen.append(fields['planning_cpu'])
        thread = threading.Thread(target=cpu.command_target(node, worker), args=('place',))
        thread.start(); thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(seen[0]['executor_native_tid'], main_tid)
        self.assertNotEqual(seen[0]['worker_native_tid'], main_tid)
        # Diagnostic locking must not surround a callback's original work.
        def original():
            acquired = recorder._lock.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired:
                recorder._lock.release()
            return 'unchanged'
        self.assertEqual(cpu.callback(node, '_on_contacts', original)(), 'unchanged')

    def test_publish_status_original_serialization_and_publish_errors_survive(self):
        methods = {}
        for label, path in (('base', BASE), ('candidate', NODE)):
            cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name=='ManipulationNode')
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='_publish_status')
            def failing_diagnostic(*args):
                raise RuntimeError('diagnostic')
            env = {'String': types.SimpleNamespace, 'encode_event': lambda event, **fields: json.dumps({'event': event, **fields}),
                   '_add_planning_cpu_fields': failing_diagnostic}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), env)
            methods[label] = env['_publish_status']
        outputs = []
        for method in methods.values():
            node = types.SimpleNamespace(status_pub=types.SimpleNamespace(publish=lambda msg: outputs.append(msg.data)))
            method(node, 'started', command='place')
            with self.assertRaises(TypeError):
                method(node, 'started', invalid=object())
            error = RuntimeError('original publish error')
            node.status_pub.publish = lambda msg: (_ for _ in ()).throw(error)
            with self.assertRaises(RuntimeError) as caught:
                method(node, 'started')
            self.assertIs(caught.exception, error)
        self.assertEqual(outputs[0], outputs[1])

    def test_whole_node_restoration_and_exact_callback_wiring(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(), BASE_SHA)
        original, current = ast.parse(BASE.read_text()), ast.parse(NODE.read_text())
        current = ast.parse(restore_to_r51_source(NODE.read_text()))
        wrappers = [n for n in ast.walk(current) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id=='_planning_cpu_callback']
        self.assertEqual(sorted(n.args[1].value for n in wrappers), sorted(cpu.CALLBACK_NAMES))
        for call in wrappers:
            name = call.args[1].value
            if name != 'bin_contacts':
                self.assertIsInstance(call.args[2], ast.Attribute)
                self.assertEqual(call.args[2].attr, name)
        class Restore(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module=='planning_cpu_diagnostics' else node
            def visit_Expr(self, node):
                if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id in ('_initialize_planning_cpu', '_mark_planning_cpu_executor')):
                    return None
                return self.generic_visit(node)
            def visit_Try(self, node):
                if (len(node.body)==1 and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Call)
                        and isinstance(node.body[0].value.func, ast.Name)
                        and node.body[0].value.func.id in ('_add_planning_cpu_fields',
                            '_initialize_planning_cpu', '_mark_planning_cpu_executor')):
                    return None
                if (len(node.body)==1 and isinstance(node.body[0], ast.Assign)
                        and isinstance(node.body[0].value, ast.Call)
                        and isinstance(node.body[0].value.func, ast.Name)
                        and node.body[0].value.func.id=='_planning_cpu_command_target'):
                    return None
                return self.generic_visit(node)
            def visit_Assign(self, node):
                if (len(node.targets)==1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id=='_cpu_command_target'):
                    return None
                return self.generic_visit(node)
            def visit_Name(self, node):
                if node.id=='_cpu_command_target':
                    return ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                         attr='_run_command', ctx=ast.Load())
                return node
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id=='_planning_cpu_callback':
                        return node.args[2]
                    if node.func.id=='_planning_cpu_command_target':
                        return node.args[1]
                return self.generic_visit(node)
        restored = Restore().visit(current)
        self.assertEqual(ast.dump(original, include_attributes=False),
                         ast.dump(restored, include_attributes=False))


if __name__ == '__main__':
    unittest.main()
