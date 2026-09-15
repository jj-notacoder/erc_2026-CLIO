"""Best-effort bounded CPU attribution; timers do not measure GIL waiting.

Only registered callbacks are measured. Durations are inclusive and totals count
completed callbacks, including a callback that began before a command boundary.
No sample hooks, ROS timers, background threads, control calls or retained logs.
"""
from functools import wraps
import json
import threading
import time


CALLBACK_NAMES = (
    '_on_joint_state', '_on_staging_odom', '_on_contacts', 'bin_contacts',
    '_on_book', '_on_target_tracking', '_on_target_tracking_status', '_on_bin',
    '_on_bin_status', '_on_command', '_announce_ready', '_monitor_held_payload',
)
MAX_COUNTER = (1 << 63) - 1
MAX_STAGE_RECORDS = 128
STATUS_EVENTS = frozenset(('started', 'succeeded', 'failed', 'cancelled',
                           'place_planning_stage'))


class PlanningCpuDiagnostics:
    def __init__(self, *, wall=time.monotonic_ns, cpu=time.thread_time_ns,
                 native_tid=threading.get_native_id):
        self._wall, self._cpu, self._tid = wall, cpu, native_tid
        self._lock = threading.Lock()  # Never the node's sensor/control lock.
        # completed, exceptional, wall_ns, cpu_ns, first_tid, last_tid, mixed
        self._totals = {name: [0, 0, 0, 0, None, None, False]
                        for name in CALLBACK_NAMES}
        self._active = None
        self._executor_tid = None

    def _reading(self):
        value = (self._wall(), self._cpu(), self._tid())
        if (any(type(v) is not int or v < 0 or v > MAX_COUNTER for v in value)
                or value[2] == 0):
            raise ValueError('invalid diagnostic clock or thread identity')
        return value

    def mark_executor(self):
        _, _, tid = self._reading()
        with self._lock:
            self._executor_tid = tid

    def callback_begin(self, name):
        if name not in self._totals:
            return None
        return (name, *self._reading())

    def callback_end(self, token, exceptional):
        if token is None:
            return
        name, start_wall, start_cpu, start_tid = token
        wall, cpu, tid = self._reading()
        if tid != start_tid or wall < start_wall or cpu < start_cpu:
            return
        with self._lock:
            row = self._totals[name]
            increments = (1, int(exceptional), wall-start_wall, cpu-start_cpu)
            for index, increment in enumerate(increments):
                row[index] = min(MAX_COUNTER, row[index]+increment)
            if row[4] is None:
                row[4] = tid
            row[5], row[6] = tid, row[6] or row[4] != tid

    def command_begin(self, command):
        if type(command) is not str or not 0 < len(command) <= 80:
            return None
        reading = self._reading()
        with self._lock:
            if self._active is not None:
                return None
            token = object()
            self._active = (token, command, reading,
                            {name: tuple(row[:4]) for name, row in self._totals.items()}, 0)
            return token

    def command_end(self, token):
        with self._lock:
            if self._active is not None and self._active[0] is token:
                self._active = None

    def snapshot(self, event, command):
        if event not in STATUS_EVENTS:
            return None
        wall, cpu, tid = self._reading()
        with self._lock:
            active = self._active
            if active is None or command != active[1] or tid != active[2][2]:
                return None
            token, command, (start_wall, start_cpu, worker_tid), baseline, stages = active
            if wall < start_wall or cpu < start_cpu:
                return None
            if event == 'place_planning_stage':
                if stages >= MAX_STAGE_RECORDS:
                    return None
                stages += 1
                self._active = (token, command, active[2], baseline, stages)
            rows = {name: tuple(row) for name, row in self._totals.items()}
            executor_tid = self._executor_tid
        callbacks = {}
        for name, row in rows.items():
            delta = [row[i]-baseline[name][i] for i in range(4)]
            callbacks[name] = dict(completed=delta[0], exceptional=delta[1],
                inclusive_wall_ns=delta[2], inclusive_thread_cpu_ns=delta[3],
                first_observed_native_tid=row[4], last_observed_native_tid=row[5],
                multiple_native_tids=row[6], saturated=MAX_COUNTER in row[:4])
        return dict(schema='planning_cpu_v1', worker_native_tid=worker_tid,
            executor_native_tid=executor_tid, command_entry_monotonic_ns=start_wall,
            command_monotonic_ns=wall-start_wall,
            command_thread_cpu_ns=cpu-start_cpu, callbacks=callbacks,
            callback_scope='registered callbacks completed since command entry; inclusive',
            callback_native_tid_scope='node lifetime, not reset per command',
            command_coverage='best effort; overlapping worker lifetimes may omit diagnostics',
            unmeasured='clock, TF, action callbacks, executor dispatch and other threads',
            stage_records=stages, stage_record_limit=MAX_STAGE_RECORDS,
            gil_wait_measured=False)


def initialize(node):
    try:
        node._planning_cpu_diagnostics = PlanningCpuDiagnostics()
    except Exception:
        pass


def mark_executor_thread(node):
    try:
        node._planning_cpu_diagnostics.mark_executor()
    except Exception:
        pass


def callback(node, name, function):
    @wraps(function)
    def measured(*args, **kwargs):
        recorder = token = None
        try:
            recorder = node._planning_cpu_diagnostics
            token = recorder.callback_begin(name)
        except Exception:
            pass
        exceptional = True
        try:
            value = function(*args, **kwargs)
            exceptional = False
            return value
        finally:
            try:
                if recorder is not None and token is not None:
                    recorder.callback_end(token, exceptional)
            except Exception:
                pass
    return measured


def command_target(node, function):
    @wraps(function)
    def measured(command, *args, **kwargs):
        recorder = token = None
        try:
            recorder = node._planning_cpu_diagnostics
            token = recorder.command_begin(command)
        except Exception:
            pass
        try:
            return function(command, *args, **kwargs)
        finally:
            try:
                if recorder is not None and token is not None:
                    recorder.command_end(token)
            except Exception:
                pass
    return measured


def add_status_fields(node, event, fields):
    """Commit only a bounded JSON value; original status errors remain original."""
    try:
        if event not in STATUS_EVENTS or 'planning_cpu' in fields:
            return
        value = node._planning_cpu_diagnostics.snapshot(event, fields.get('command'))
        if value is None:
            return
        encoded = json.dumps(value, allow_nan=False, separators=(',', ':'))
        if len(encoded) > 16384:
            return
        fields['planning_cpu'] = json.loads(encoded)
    except Exception:
        pass
