"""Runner evidence and process isolation tests; no ROS or Gazebo required."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('run_book_matrix')
recorder = load('record_perception_status')


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'src').mkdir()
        (self.root / 'src/core.py').write_text('original\n')
        self.root_patch = patch.object(runner, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def fake_trial(self, *, success=True, inside=True, ready=True, mutate=False,
                   solution_mode='terminal'):
        """Run the actual supervisor against tiny subprocess stand-ins."""
        original_start = runner.OwnedProcesses.start
        summary = {
            'trial_id': 'test123', 'success': success, 'dry_run': False,
            'book_colour': 'blue', 'shelf_column_number': 4,
            'target_book_model': 'book_col_5_row_4_blue',
            'expected_target_book_model': 'book_col_5_row_4_blue',
            'target_identity_confirmed': True, 'collision_episodes': 0,
            'failure_reason': '' if success else 'book_reacquisition_failed',
        }
        scripts = {
            'simulation': 'import time; time.sleep(60)',
            'readiness': f'raise SystemExit({0 if ready else 1})',
            'perception_diagnostics': (
                "import os,pathlib,time; "
                "pathlib.Path(os.environ['ERC_RESULTS_DIR'],'perception_recorder_ready.json').write_text('{}'); "
                'time.sleep(60)'),
            'solution': (
                'import json,os,pathlib,time; '
                f'd={summary!r}; '
                "d.update(timing={'mission_initialization': {'monotonic_seconds': time.monotonic()}, "
                "'launch_origin': {'monotonic_seconds': time.monotonic()}}, elapsed_seconds=.01); "
                + ("pathlib.Path('src/core.py').write_text('changed'); " if mutate else '') +
                "time.sleep(.02); pathlib.Path(os.environ['ERC_RESULTS_DIR'], "
                "'trial_test123_summary.json').write_text(json.dumps(d)); time.sleep(60)"),
            'delivery_check': (
                'import json,os,pathlib; '
                f"d={{'trial_id': 'test123', 'target_model': 'book_col_5_row_4_blue', "
                f"'whole_book_inside_conservative_core': {inside!r}, 'minimum_clearance_m': .01}}; "
                "pathlib.Path(os.environ['ERC_RESULTS_DIR'],'containment.json').write_text(json.dumps(d)); "
                f'raise SystemExit({0 if inside else 2})'),
        }
        if solution_mode == 'hang':
            scripts['solution'] = 'import time; time.sleep(60)'
        elif solution_mode == 'crash':
            scripts['solution'] = 'raise SystemExit(9)'

        def fake_start(processes, name, argv):
            return original_start(processes, name, [sys.executable, '-c', scripts[name]])

        args = argparse.Namespace(viewer=False, timeout=1.2 if solution_mode == 'hang' else 5.)
        with patch.object(runner.OwnedProcesses, 'start', fake_start):
            return runner.run_trial(args, self.root / 'trial', 101, 4, 'blue',
                                    runner.source_manifest(self.root), lambda: False)

    def test_success_uses_requested_target_and_independent_containment(self):
        result = self.fake_trial()
        self.assertEqual(result['mission_status'], 'DONE')
        self.assertTrue(result['validated_delivery'])
        self.assertEqual(result['target_book_model'], 'book_col_5_row_4_blue')
        self.assertTrue(result['cleanup']['all_owned_groups_stopped'])
        self.assertGreater(result['simulator_launch_to_terminal_wall_seconds'],
                           result['solution_process_launch_to_terminal_wall_seconds'])
        persisted = json.loads((self.root / 'trial/run_review.json').read_text())
        self.assertEqual(persisted, result)

    def test_done_does_not_override_failed_physical_containment(self):
        result = self.fake_trial(inside=False)
        self.assertEqual(result['mission_status'], 'DONE')
        self.assertFalse(result['validated_delivery'])
        self.assertEqual(result['delivery_check_returncode'], 2)

    def test_abort_is_saved_and_skips_delivery_sampler(self):
        result = self.fake_trial(success=False)
        self.assertEqual(result['mission_status'], 'ABORTED')
        self.assertEqual(result['failure_reason'], 'book_reacquisition_failed')
        self.assertFalse(result['validated_delivery'])
        self.assertFalse((self.root / 'trial/delivery_check.log').exists())
        self.assertTrue(result['cleanup']['all_owned_groups_stopped'])

    def test_readiness_failure_never_starts_solution(self):
        result = self.fake_trial(ready=False)
        self.assertEqual(result['mission_status'], 'READINESS_FAILED')
        self.assertFalse((self.root / 'trial/solution.log').exists())
        self.assertTrue(result['cleanup']['all_owned_groups_stopped'])

    def test_missing_terminal_summary_times_out_and_stops_owned_groups(self):
        result = self.fake_trial(solution_mode='hang')
        self.assertEqual(result['mission_status'], 'TIMEOUT')
        self.assertFalse(result['validated_delivery'])
        self.assertTrue(result['cleanup']['all_owned_groups_stopped'])

    def test_crashed_solution_is_reported_without_waiting_for_trial_timeout(self):
        result = self.fake_trial(solution_mode='crash')
        self.assertEqual(result['mission_status'], 'RUNNER_ERROR')
        self.assertIn('solution exited with status 9', result['runner_error'])
        self.assertTrue(result['cleanup']['all_owned_groups_stopped'])

    def test_changed_core_invalidates_apparent_success(self):
        result = self.fake_trial(mutate=True)
        self.assertEqual(result['mission_status'], 'DONE')
        self.assertFalse(result['source_unchanged'])
        self.assertFalse(result['validated_delivery'])
        self.assertTrue((self.root / 'trial/source_manifest_after.json').exists())

    def test_manifest_ignores_tooling_and_python_caches(self):
        before = runner.source_manifest(self.root)
        (self.root / 'src/__pycache__').mkdir()
        (self.root / 'src/__pycache__/core.pyc').write_bytes(b'cache')
        (self.root / 'tools').mkdir()
        (self.root / 'tools/new_tool.py').write_text('tool')
        self.assertEqual(before, runner.source_manifest(self.root))
        (self.root / 'src/core.py').write_text('changed')
        self.assertNotEqual(before, runner.source_manifest(self.root))

    def test_partial_summary_write_is_not_mistaken_for_terminal(self):
        (self.root / 'trial_x_summary.json').write_text('{"success":')
        self.assertIsNone(runner.read_summary(self.root))

    def test_cleanup_finds_orphan_and_leaves_unowned_process_alone(self):
        processes = runner.OwnedProcesses(self.root, dict(os.environ))
        unowned = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                   start_new_session=True)
        self.addCleanup(lambda: unowned.poll() is None and unowned.kill())
        self.addCleanup(lambda: unowned.poll())
        child_file = self.root / 'child_ready'
        code = (
            'import os,pathlib,signal,time\n'
            'if os.fork() == 0:\n'
            ' signal.signal(signal.SIGINT, signal.SIG_IGN)\n'
            ' signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
            f' pathlib.Path({str(child_file)!r}).write_text(str(os.getpid()))\n'
            ' time.sleep(60)\n'
            'else:\n'
            ' os._exit(0)\n')
        leader = processes.start('orphan_parent', [sys.executable, '-c', code])
        self.addCleanup(lambda: processes.cleanup(waits=(.05, .05, .3)))
        deadline = time.monotonic() + 2.
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(child_file.exists())
        leader.wait(timeout=1)
        cleanup = processes.cleanup(waits=(.05, .05, .3))
        self.assertTrue(cleanup['all_owned_groups_stopped'])
        self.assertIn('SIGKILL', [step['signal'] for step in cleanup['steps']])
        self.assertIsNone(unowned.poll())
        unowned.terminate()
        unowned.wait(timeout=2)

    def test_early_terminal_timing_is_rejected(self):
        summary = {'timing': {'mission_initialization': {'monotonic_seconds': 5.},
                             'launch_origin': {'monotonic_seconds': 4.}},
                   'elapsed_seconds': 1.}
        with self.assertRaisesRegex(ValueError, 'precedes'):
            runner.collect_result(summary, {'monotonic_seconds': 7.}, {'monotonic_seconds': 8.})

    def test_diagnostics_preserve_producer_stamp_and_reason(self):
        record = recorder.make_record('/status', '{"reason":"invalid_plane","stamp_ns":123}', 456)
        self.assertEqual(record['reason'], 'invalid_plane')
        self.assertEqual(record['producer_stamp_ns'], 123)
        self.assertEqual(record['receipt_ros_time_ns'], 456)
        malformed = recorder.make_record('/status', 'bad JSON', 456)
        self.assertEqual(malformed['raw'], 'bad JSON')


if __name__ == '__main__':
    unittest.main()
