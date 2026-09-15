#!/usr/bin/env python3
"""Sequential Gazebo checks of requested printed columns and book colours.

Use run_book_matrix.sh through the prepared container's /entrypoint.sh.
All 20 targets at seed 101 are selected by default. Robot sensors and physics
are unchanged; only the spectator viewer is optional. Each completed mission
gets a separate, read-only whole-book containment sample before world cleanup.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


COLOURS = ('red', 'blue', 'green', 'yellow')
ROOT = Path(__file__).resolve().parent.parent
TARGET_MODEL = re.compile(r'book_col_[1-5]_row_[1-5]_(red|blue|green|yellow)')


def stamp(basis):
    return {'monotonic_seconds': time.monotonic(),
            'utc': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
            'basis': basis}


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def source_manifest(root):
    paths = list((root / 'src').rglob('*'))
    paths += [root / 'install_release/gz_ros2_control/lib' / name for name in (
        'libgz_ros2_control-system.so', 'libgz_hardware_plugins.so')]
    hashes = {}
    for path in sorted(set(paths)):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': hashes}


def active_group_members(pgid):
    """Ignore zombies while detecting descendants whose launch parent exited."""
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                result.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def existing_simulators():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        # Match actual executable argv, not shell text containing this command.
        is_launch = ('launch' in argv and 'erc_bringup' in argv and
                     'simulation.launch.py' in argv and
                     any(Path(arg).name == 'ros2' for arg in argv[:2]))
        is_server = (len(argv) > 2 and Path(argv[0]).name == 'gz' and
                     argv[1] == 'sim' and '-g' not in argv)
        if is_launch or is_server:
            found.append({'pid': int(entry.name), 'argv': argv[:-1]})
    return found


class OwnedProcesses:
    def __init__(self, directory, env):
        self.directory, self.env, self.entries = directory, env, []

    def start(self, name, argv):
        started = stamp('immediately_before_Popen_start_new_session')
        handle = (self.directory / f'{name}.log').open('w')
        try:
            proc = subprocess.Popen(argv, cwd=ROOT, env=self.env, stdout=handle,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            handle.close()
            raise
        self.entries.append({'name': name, 'process': proc, 'handle': handle,
                             'started': started, 'argv': list(argv)})
        self.capture()
        return proc

    def capture(self):
        save_json(self.directory / 'processes.json', [
            {'name': row['name'], 'pid': row['process'].pid,
             'pgid': row['process'].pid, 'argv': row['argv'],
             'started': row['started'], 'returncode': row['process'].poll()}
            for row in self.entries])

    def cleanup(self, waits=(8., 3., 3.)):
        """Stop only process groups created here, including surviving children."""
        steps = []
        try:
            for sig, duration in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), waits):
                groups = [row['process'].pid for row in self.entries
                          if active_group_members(row['process'].pid)]
                if not groups:
                    break
                steps.append({'signal': sig.name, 'groups': groups})
                for pgid in groups:
                    try:
                        os.killpg(pgid, sig)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline:
                    for row in self.entries:
                        row['process'].poll()
                    if not any(active_group_members(pgid) for pgid in groups):
                        break
                    time.sleep(.1)
            remaining = {str(row['process'].pid): active_group_members(row['process'].pid)
                         for row in self.entries}
            remaining = {pgid: members for pgid, members in remaining.items() if members}
            self.capture()
            return {'all_owned_groups_stopped': not remaining,
                    'remaining_members': remaining, 'steps': steps}
        finally:
            for row in self.entries:
                row['handle'].close()


def read_summary(directory):
    paths = list(directory.glob('trial_*_summary.json'))
    if len(paths) > 1:
        raise ValueError('multiple trial summaries in a fresh trial directory')
    if not paths:
        return None
    try:
        data = json.loads(paths[0].read_text())
    except json.JSONDecodeError:
        return None  # The mission writes this file directly; wait for a complete write.
    if type(data.get('success')) is not bool or not data.get('trial_id'):
        raise ValueError('terminal summary lacks success or trial identity')
    return paths[0], data


def collect_result(summary, simulator_start, solution_start):
    initialization = summary['timing']['mission_initialization']['monotonic_seconds']
    terminal = initialization + summary['elapsed_seconds']
    origin = summary['timing']['launch_origin'].get('monotonic_seconds')
    values = (terminal, simulator_start['monotonic_seconds'],
              solution_start['monotonic_seconds'])
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError('non-finite terminal timing')
    if not simulator_start['monotonic_seconds'] <= solution_start['monotonic_seconds'] <= terminal:
        raise ValueError('terminal timing precedes one of the runner launch origins')
    return {
        'mission_status': 'DONE' if summary['success'] else 'ABORTED',
        'trial_id': summary['trial_id'], 'failure_reason': summary.get('failure_reason'),
        'target_book_model': summary.get('target_book_model'),
        'expected_target_book_model': summary.get('expected_target_book_model'),
        'detected_row': summary.get('detected_row'),
        'collision_episodes': summary.get('collision_episodes'),
        'target_identity_confirmed': summary.get('target_identity_confirmed'),
        'delivery_outcome': summary.get('delivery_outcome'),
        'terminal_monotonic_seconds': terminal,
        'simulator_launch_to_terminal_wall_seconds': terminal - values[1],
        'solution_process_launch_to_terminal_wall_seconds': terminal - values[2],
        'solution_reported_launch_to_terminal_wall_seconds': (
            terminal - origin if type(origin) in (int, float) and math.isfinite(origin) else None),
        'timing_scope': 'To the mission summary; independent post-mission evaluation and cleanup excluded.',
    }


class Progress:
    def __init__(self):
        self.offset = 0

    def report(self, directory, label):
        paths = list(directory.glob('trial_*.jsonl'))
        if not paths:
            return
        with paths[0].open() as stream:
            stream.seek(self.offset)
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line or not line.endswith('\n'):
                    self.offset = start
                    return
                self.offset = stream.tell()
                event = json.loads(line)
                if event.get('event') == 'state_transition':
                    print(f"[{label}] {event['current']}", flush=True)


def run_trial(args, directory, seed, column, colour, manifest, interrupted):
    directory.mkdir()
    result = {'seed': seed, 'shelf_column_number': column, 'book_colour': colour,
              'viewer': args.viewer, 'directory': str(directory), 'mission_status': 'STARTING',
              'validated_delivery': False, 'source_manifest_sha256': manifest['sha256']}
    save_json(directory / 'source_manifest.json', manifest)
    save_json(directory / 'run_review.json', result)
    env = dict(os.environ, ERC_SEED=str(seed), ERC_RESULTS_DIR=str(directory),
               ERC_ERC_IMAGES_DIR=str(directory / 'erc_images'))
    processes = OwnedProcesses(directory, env)
    progress = Progress()
    label = f'seed {seed}, column {column}, {colour}'
    print(f'[{label}] START {directory}', flush=True)
    origin = stamp('before_simulator_Popen_and_optional_viewer')
    save_json(directory / 'simulator_start.json', origin)
    deadline = origin['monotonic_seconds'] + args.timeout

    def check_running(essential):
        if interrupted():
            raise InterruptedError('runner interrupted')
        if time.monotonic() >= deadline:
            raise TimeoutError('simulator-start wall-time budget exhausted')
        for name, process in essential:
            if process.poll() is not None:
                raise RuntimeError(f'{name} exited with status {process.returncode}')

    try:
        simulator = processes.start('simulation', ['ros2', 'launch', 'erc_bringup',
            'simulation.launch.py', 'headless:=true', 'depth_cloud:=false'])
        essential = [('simulator', simulator)]
        if args.viewer:
            viewer = processes.start('viewer', ['gz', 'sim', '-g', '--render-engine-gui',
                'ogre', '--gui-config', str(ROOT / 'tools/gazebo_viewer.config')])
            essential.append(('viewer', viewer))
        probe = processes.start('readiness', [sys.executable,
            str(ROOT / 'tools/check_sensor_readiness.py'), '--timeout', '60'])
        ready_deadline = min(deadline, time.monotonic() + 65.)
        while probe.poll() is None:
            check_running(essential)
            if time.monotonic() >= ready_deadline:
                raise TimeoutError('sensor readiness process exceeded its wall-time budget')
            time.sleep(.25)
        if probe.returncode:
            result['mission_status'] = 'READINESS_FAILED'
            raise RuntimeError(f'sensor readiness failed with status {probe.returncode}')
        recorder = processes.start('perception_diagnostics', [sys.executable,
            str(ROOT / 'tools/record_perception_status.py'), '--output',
            str(directory / 'perception_status.jsonl'), '--ready-file',
            str(directory / 'perception_recorder_ready.json')])
        essential.append(('perception diagnostics', recorder))
        recorder_deadline = min(deadline, time.monotonic() + 10.)
        while not (directory / 'perception_recorder_ready.json').is_file():
            check_running(essential)
            if time.monotonic() >= recorder_deadline:
                raise TimeoutError('perception diagnostic subscriber did not initialize')
            time.sleep(.1)
        solution_start = stamp('before_solution_launch_Popen_including_ros2_CLI_startup')
        save_json(directory / 'solution_start.json', solution_start)
        solution = processes.start('solution', ['ros2', 'launch', 'erc_phase1_solution',
            'solution.launch.py', f'shelf_column_number:={column}', f'book_colour:={colour}',
            f'trial_timeout_seconds:={args.timeout}'])
        essential.append(('solution', solution))
        result['mission_status'] = 'RUNNING'
        save_json(directory / 'run_review.json', result)
        while True:
            progress.report(directory, label)
            terminal = read_summary(directory)
            if terminal is not None:
                path, summary = terminal
                result.update(collect_result(summary, origin, solution_start))
                result['summary_path'] = str(path)
                # Preserve terminal result before post-mission checks or cleanup.
                save_json(directory / 'run_review.json', result)
                break
            check_running(essential)
            time.sleep(.25)
        if summary['success']:
            target = summary.get('target_book_model')
            if not isinstance(target, str) or not TARGET_MODEL.fullmatch(target):
                raise ValueError('DONE summary has no valid exact target_book_model')
            if summary.get('dry_run') is not False or summary.get('target_identity_confirmed') is not True:
                raise ValueError('DONE lacks physical-run target identity confirmation')
            if (summary.get('book_colour') != colour or summary.get('shelf_column_number') != column
                    or not target.endswith('_' + colour)
                    or target != summary.get('expected_target_book_model')):
                raise ValueError('DONE target identity does not match the requested task')
            evaluator = processes.start('delivery_check', [sys.executable,
                str(ROOT / 'tools/check_delivery.py'), '--trial-id', summary['trial_id'],
                '--target-model', target, '--timeout', '20', '--output',
                str(directory / 'containment.json')])
            evaluation_deadline = time.monotonic() + 30.
            while evaluator.poll() is None:
                if interrupted():
                    raise InterruptedError('runner interrupted during independent delivery check')
                if time.monotonic() >= evaluation_deadline:
                    raise TimeoutError('independent delivery check exceeded wall-time budget')
                time.sleep(.25)
            result['delivery_check_returncode'] = evaluator.returncode
            if not (directory / 'containment.json').is_file():
                raise RuntimeError('independent delivery check produced no containment result')
            containment = json.loads((directory / 'containment.json').read_text())
            result['whole_book_inside_conservative_core'] = containment.get('whole_book_inside_conservative_core')
            result['minimum_clearance_m'] = containment.get('minimum_clearance_m')
            result['validated_delivery'] = (
                evaluator.returncode == 0 and
                containment.get('whole_book_inside_conservative_core') is True and
                containment.get('target_model') == target and
                containment.get('trial_id') == summary['trial_id'] and
                summary.get('collision_episodes') == 0)
    except InterruptedError as exc:
        result['runner_error'] = str(exc)
        if result['mission_status'] not in ('DONE', 'ABORTED'):
            result['mission_status'] = 'INTERRUPTED'
    except TimeoutError as exc:
        result['runner_error'] = str(exc)
        if result['mission_status'] not in ('DONE', 'ABORTED'):
            result['mission_status'] = 'TIMEOUT'
    except Exception as exc:
        result['runner_error'] = f'{type(exc).__name__}: {exc}'
        if result['mission_status'] in ('STARTING', 'RUNNING'):
            result['mission_status'] = 'RUNNER_ERROR'
    finally:
        save_json(directory / 'run_review.json', result)
        result['cleanup'] = processes.cleanup()
        after = source_manifest(ROOT)
        result['source_unchanged'] = after == manifest
        if not result['source_unchanged']:
            save_json(directory / 'source_manifest_after.json', after)
        result['validated_delivery'] = (result['validated_delivery'] and
            result['cleanup']['all_owned_groups_stopped'] and result['source_unchanged'])
        result['finished'] = stamp('after_owned_process_cleanup_and_source_verification')
        save_json(directory / 'run_review.json', result)
    print(f"[{label}] {result['mission_status']}; verified delivery={result['validated_delivery']}; "
          f"elapsed={result.get('simulator_launch_to_terminal_wall_seconds', 'unavailable')} s", flush=True)
    return result


def target_argument(value):
    match = re.fullmatch(r'([1-5]):(red|blue|green|yellow)', value)
    if not match:
        raise argparse.ArgumentTypeError('target must be COLUMN:COLOUR, e.g. 4:blue')
    return int(match[1]), match[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=target_argument, nargs='+',
                        help='Requested printed column:colour pairs; default all 20.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101])
    parser.add_argument('--timeout', type=float, default=900.,
                        help='Maximum wall seconds per world, starting before simulator launch (default 900).')
    parser.add_argument('--viewer', action='store_true', help='Show the normal Gazebo viewer during each trial.')
    parser.add_argument('--stop-on-failure', action='store_true')
    parser.add_argument('--output', type=Path, help='New matrix output directory, never an existing directory.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 60. <= args.timeout <= 1800.:
        parser.error('--timeout must be between 60 and 1800 seconds')
    targets = args.targets or [(column, colour) for column in range(1, 6) for colour in COLOURS]
    if len(set(targets)) != len(targets) or len(set(args.seeds)) != len(args.seeds):
        parser.error('duplicate targets or seeds are not allowed')
    if args.viewer and not os.environ.get('DISPLAY'):
        parser.error('--viewer requires DISPLAY')
    root_results = ROOT / 'results'
    root_results.mkdir(exist_ok=True)
    directory = (args.output or root_results /
        datetime.now(timezone.utc).strftime('matrix_%Y%m%dT%H%M%S_%fZ')).resolve()
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    with (root_results / '.book_matrix.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another matrix runner owns this workspace; no worlds started')
        active = existing_simulators()
        if active:
            parser.error('a simulator already runs in this container: ' + json.dumps(active))
        directory.mkdir(parents=True, exist_ok=False)
        manifest = source_manifest(ROOT)
        save_json(directory / 'source_manifest.json', manifest)
        report = {'started': stamp('before_matrix_first_trial'), 'viewer': args.viewer,
                  'seeds': args.seeds, 'targets': targets,
                  'source_manifest_sha256': manifest['sha256'], 'trials': [],
                  'requested_trials': len(targets) * len(args.seeds), 'complete': False,
                  'all_deliveries_verified': False,
                  'scope': 'Selected seeded trials only; one post-mission containment sample per DONE. '
                           'Headless timings do not establish visible-viewer performance.'}
        save_json(directory / 'matrix_summary.json', report)
        for seed in args.seeds:
            for column, colour in targets:
                if stop[0]:
                    break
                if source_manifest(ROOT) != manifest:
                    report['runner_error'] = 'source changed during matrix; remaining worlds not started'
                    stop[0] = True
                    break
                trial_dir = directory / f'{len(report["trials"]) + 1:02d}_seed{seed}_column{column}_{colour}'
                result = run_trial(args, trial_dir, seed, column, colour, manifest, lambda: stop[0])
                report['trials'].append(result)
                report['completed_trials'] = len(report['trials'])
                save_json(directory / 'matrix_summary.json', report)
                if (not result['cleanup']['all_owned_groups_stopped'] or
                        not result['source_unchanged'] or
                        (args.stop_on_failure and not result['validated_delivery'])):
                    stop[0] = True
        report['complete'] = len(report['trials']) == report['requested_trials']
        report['all_deliveries_verified'] = (report['complete'] and
            all(row['validated_delivery'] for row in report['trials']))
        report['finished'] = stamp('after_matrix_last_trial_cleanup')
        save_json(directory / 'matrix_summary.json', report)
        print(f'Matrix results: {directory / "matrix_summary.json"}', flush=True)
        return 0 if report['all_deliveries_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
