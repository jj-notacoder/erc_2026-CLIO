"""Bounded process geometry transport with ordered parent consumption.

This module has no ROS dependency and never authorizes an action. The caller
captures a newly admitted paired context for every enqueue and commits only
the requested ordered prefix. Each pool belongs to one immutable scene epoch.
"""
import hashlib
import io
import json
import math
import os
import queue
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .geometry_process_protocol import (
    MAX_FRAME_BYTES, MAX_QUERY_BYTES, SampleDelta, decode_frame, encode_frame,
    query_to_wire,
)


class GeometryProcessError(RuntimeError):
    pass


def _process_stat_identity(raw, pid):
    """Parse owned Linux child identity, including commands containing ')' ."""
    prefix, fields = raw.rsplit(')', 1)
    values = fields.split()
    if int(prefix.split('(', 1)[0]) != pid or len(values) < 20:
        raise GeometryProcessError('invalid owned geometry process stat')
    result = dict(pid=pid, parent_pid=int(values[1]), process_group=int(values[2]),
                  session=int(values[3]), start_ticks=int(values[19]))
    if result['start_ticks'] <= 0:
        raise GeometryProcessError('invalid owned geometry process start time')
    return result


def checked_geometry_process_workers(value):
    """Keep process fan-out explicit, bounded and free of Boolean coercion."""
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError('geometry_process_workers must be an integer from 1 to 8')
    return value


class _Worker:
    """Own one process and bounded pipe threads, without blocking the caller."""
    def __init__(self, number, events, environment):
        self.number, self.events = number, events
        self.fault = None
        self.identity = None
        self.identity_observation = None
        self.identity_capture_error = None
        self.stderr_bytes = 0
        self.stderr_prefix = bytearray()
        self.outgoing = queue.Queue(maxsize=2)
        self.threads = [threading.Thread(target=target, daemon=True,
            name=f'geometry-{number}-{name}') for name, target in (
                ('writer', self._write), ('reader', self._read), ('stderr', self._stderr))]
        self.process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'erc_phase1_solution.geometry_process_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, start_new_session=True, bufsize=0)

    def capture_identity(self):
        # Called only after registration in the pool, so any failure is cleaned
        # up by the same owner. Never accept a caller-provided process identity.
        pid = self.process.pid
        proc = Path('/proc') / str(pid)
        first = _process_stat_identity((proc/'stat').read_text(), pid)
        argv = (proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        argv = [part.decode() for part in argv]
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        last = _process_stat_identity((proc/'stat').read_text(), pid)
        expected = [sys.executable, '-u', '-m', 'erc_phase1_solution.geometry_process_worker']
        self.identity_observation = dict(first=first,last=last,argv=argv,
            expected_argv=expected,expected_parent_pid=os.getpid(),boot_id=boot_id)
        if (first != last or first['parent_pid'] != os.getpid()
                or first['process_group'] != pid or first['session'] != pid
                or argv != expected or not boot_id):
            raise GeometryProcessError('owned geometry worker identity changed')
        self.identity = dict(first, argv=argv, boot_id=boot_id)

    def start(self):
        # The pool registers ownership before any thread can fail to start.
        # Buffer reads only: raw FileIO.readline otherwise reads byte by byte.
        # The existing bounded newline decoder and immediate input writes stay
        # unchanged. Setup failure is covered by the pool's owned cleanup.
        self.process.stdout = io.BufferedReader(self.process.stdout, buffer_size=65536)
        for thread in self.threads:
            thread.start()

    def _fail(self, error):
        self.fault = str(error)[:2048]
        try:
            self.events.put_nowait((self.number, None))
        except queue.Full:
            pass

    def _write(self):
        try:
            while True:
                data = self.outgoing.get()
                if data is None:
                    return
                view = memoryview(data)
                while view:
                    count = self.process.stdin.write(view)
                    if count is None or count <= 0:
                        raise GeometryProcessError('geometry input pipe closed')
                    view = view[count:]
        except BaseException as error:
            self._fail(error)

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_FRAME_BYTES+1)
                if not line:
                    raise GeometryProcessError('geometry output pipe closed')
                if not line.endswith(b'\n'):
                    raise GeometryProcessError('truncated geometry output')
                self.events.put_nowait((self.number, decode_frame(line)))
        except BaseException as error:
            self._fail(error)

    def _stderr(self):
        try:
            while True:
                part = self.process.stderr.read(4096)
                if not part:
                    return
                self.stderr_bytes += len(part)
                room = max(0, 65536-len(self.stderr_prefix))
                self.stderr_prefix.extend(part[:room])
        except (OSError, ValueError):
            return

    def send(self, value, maximum=MAX_QUERY_BYTES):
        try:
            self.outgoing.put_nowait(encode_frame(value, maximum))
        except queue.Full as error:
            raise GeometryProcessError('geometry input queue exceeded bound') from error


class GeometryProcessPool:
    """One to eight owned workers, at most one in-flight query per worker.

    No worker result is a final motion admission. Constructor/bootstrap failure,
    protocol mismatch, cancellation, timeout or death closes the whole epoch.
    Ordinary geometry rejection drains/discards the speculative suffix, so a
    later candidate can use the same immutable scene without stale diagnostics.
    """
    maximum_queries = 20_000

    def __init__(self, identity, *, epoch, scene, geometry_id, cancelled,
                 startup_seconds=15., query_seconds=15., total_seconds=180.,
                 address_space_bytes=2*1024**3, cpu_seconds=120, worker_count=4):
        self.startup_wall_seconds=None
        self.sequence_wall_seconds=0.
        startup_began=time.monotonic()
        if os.name != 'posix' or not sys.platform.startswith('linux'):
            raise GeometryProcessError('geometry process backend requires Linux')
        from .installed_geometry_identity import InstalledGeometryIdentity
        if type(identity) is not InstalledGeometryIdentity:
            raise ValueError('installed geometry identity required')
        if type(epoch) is not int or epoch < 0 or not callable(cancelled):
            raise ValueError('invalid epoch or cancellation predicate')
        if type(geometry_id) is not str or len(geometry_id)!=64 or any(c not in '0123456789abcdef' for c in geometry_id):
            raise ValueError('actual parent geometry digest required')
        for value, limit in ((startup_seconds,30.), (query_seconds,30.), (total_seconds,300.)):
            if type(value) not in (int,float) or not math.isfinite(value) or not 0 < value <= limit:
                raise ValueError('invalid geometry wall time bound')
        if type(address_space_bytes) is not int or not 1024**3 <= address_space_bytes <= 4*1024**3:
            raise ValueError('invalid geometry address space bound')
        if type(cpu_seconds) is not int or not 1 <= cpu_seconds <= 180:
            raise ValueError('invalid geometry CPU bound')
        self.worker_count = checked_geometry_process_workers(worker_count)
        if self.worker_count * address_space_bytes > 16*1024**3:
            raise ValueError('aggregate geometry address space bound exceeds 16 GiB')
        identity.verify()
        self.identity, self.epoch, self.cancelled = identity, epoch, cancelled
        self.geometry_id=geometry_id
        self.scene = json.loads(json.dumps(scene, sort_keys=True, allow_nan=False))
        self.scene_id = hashlib.sha256(json.dumps(self.scene, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.token = secrets.token_hex(32)
        self.query_seconds = float(query_seconds)
        self.deadline = time.monotonic()+total_seconds
        self.events = queue.Queue(maxsize=32)
        self.workers = []
        self.closed = False
        self.active = False
        self.next_request_id = 0
        self.last_worker_requests=[-1]*self.worker_count
        self.finished=False
        self.statistics = dict(submitted=0, consumed=0, discarded=0, cached_at_enqueue=0)
        environment = dict(os.environ)
        environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                           PYTHONDONTWRITEBYTECODE='1')
        initial = dict(kind='begin', token=self.token, identity=identity.to_descriptor(),
            epoch=epoch, scene=self.scene, scene_id=self.scene_id,
            address_space_bytes=address_space_bytes, cpu_seconds=cpu_seconds, geometry_id=geometry_id,
            parent_pid=os.getpid())
        encode_frame(dict(initial, worker=0))
        try:
            for number in range(self.worker_count):
                self._check()
                worker = _Worker(number, self.events, environment)
                self.workers.append(worker)
                worker.start()
                worker.send(dict(initial, worker=number), MAX_FRAME_BYTES)
            ready = set()
            startup_deadline = min(self.deadline, time.monotonic()+startup_seconds)
            while len(ready) != self.worker_count:
                number, message = self._event(startup_deadline)
                expected = dict(kind='ready', token=self.token, worker=number,
                    pid=self.workers[number].process.pid, source_id=identity.source_id,
                    model_id=identity.model_id, epoch=epoch, scene_id=self.scene_id, geometry_id=geometry_id)
                if number in ready or message != expected:
                    raise GeometryProcessError('geometry bootstrap identity mismatch')
                # The verified ready barrier establishes completed Python/module
                # startup before reading the exact OS argv and start identity.
                self.workers[number].capture_identity()
                ready.add(number)
            identity.verify()
            self.startup_wall_seconds=time.monotonic()-startup_began
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                cleanup_error.geometry_worker_status = self.worker_status()
                raise cleanup_error from error
            error.geometry_worker_status = self.worker_status()
            raise

    def worker_status(self):
        """Small post-close diagnostics; absent identity is explicitly unknown."""
        closure = list(getattr(self, 'closure', []))
        return dict(workers_created=len(self.workers), workers=closure,
            workers_reaped=sum(row['reaped'] for row in closure),
            children_closed=(len(closure) == len(self.workers)
                             and all(row['reaped'] for row in closure)),
            pool_startup_wall_seconds=self.startup_wall_seconds,
            sequence_wall_seconds=self.sequence_wall_seconds,
            query_statistics=dict(self.statistics))

    def _check(self):
        if self.closed:
            raise GeometryProcessError('geometry pool is closed')
        if self.cancelled():
            raise GeometryProcessError('geometry evaluation cancelled')
        if time.monotonic() >= self.deadline:
            raise GeometryProcessError('geometry scene wall deadline exceeded')
        for worker in self.workers:
            if worker.fault is not None or worker.process.poll() is not None:
                raise GeometryProcessError(f'geometry worker {worker.number} failed: {worker.fault}')

    def _event(self, deadline):
        while True:
            self._check()
            remaining = min(deadline, self.deadline)-time.monotonic()
            if remaining <= 0:
                raise GeometryProcessError('geometry response deadline exceeded')
            try:
                number, message = self.events.get(timeout=min(.02,remaining))
            except queue.Empty:
                continue
            if message is None:
                self._check()
                raise GeometryProcessError('geometry worker failed without reply')
            return number, message

    def _reply(self, pending):
        number, message = self._event(min(value[1] for value in pending.values()))
        if number not in pending:
            raise GeometryProcessError('unsolicited or duplicate geometry result')
        query, deadline = pending[number]
        if (type(message) is not dict or set(message) != {'kind','token','worker','pid','delta'}
                or message['kind'] != 'result' or message['token'] != self.token
                or message['worker'] != number or message['pid'] != self.workers[number].process.pid
                or time.monotonic() > deadline):
            raise GeometryProcessError('geometry reply envelope mismatch or late reply')
        delta = SampleDelta.from_wire(message['delta'])
        if not delta.matches(query):
            raise GeometryProcessError('geometry reply query identity mismatch')
        del pending[number]
        return number, query, delta

    def evaluate_sequence(self, items, *, capture, consume, is_cached=lambda query: False):
        """Capture at enqueue; call consume(query, delta) only in input order.

        capture receives (request_id, item) and must return a new paired,
        admitted GeometryQuery. A true is_cached result allows delta=None;
        consume must still perform the cancellation-before-cache-hit rule.
        No sample index, spacing or success decision is generated here.
        """
        if self.active:
            raise GeometryProcessError('geometry sequences cannot overlap')
        if getattr(self,'finished',False):
            raise GeometryProcessError('geometry scene is already finished')
        self.active = True
        sequence_began=time.monotonic()
        pending, ready, order = {}, {}, []
        free = list(range(len(self.workers)))
        exhausted = False
        try:
            iterator = iter(items)
            self._check()
            while True:
                while not exhausted and free and len(order) < len(self.workers):
                    self._check()
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    request_id = self.next_request_id
                    if request_id >= self.maximum_queries:
                        raise GeometryProcessError('geometry query count exceeded bound')
                    query = capture(request_id, item).validated()
                    if (query.request_id != request_id or query.epoch != self.epoch
                            or query.source_id != self.identity.source_id
                            or query.model_id != self.identity.model_id or query.scene_id != self.scene_id):
                        raise GeometryProcessError('captured geometry query identity mismatch')
                    self._check()
                    self.next_request_id += 1
                    order.append(query)
                    cached = is_cached(query)
                    if type(cached) is not bool:
                        raise GeometryProcessError('geometry cache admission must be boolean')
                    if cached:
                        ready[request_id] = None
                        self.statistics['cached_at_enqueue'] += 1
                    else:
                        number = free.pop(0)
                        self.workers[number].send(dict(kind='query', token=self.token,
                                                      query=query_to_wire(query)))
                        pending[number] = query, min(self.deadline,time.monotonic()+self.query_seconds)
                        self.last_worker_requests[number]=request_id
                        self.statistics['submitted'] += 1
                while order and order[0].request_id in ready:
                    self._check()
                    query = order.pop(0)
                    delta = ready.pop(query.request_id)
                    accepted = consume(query, delta)
                    if type(accepted) is not bool:
                        raise GeometryProcessError('geometry consumer must return a boolean')
                    self.statistics['consumed'] += 1
                    if not accepted:
                        # The suffix has no committed cache, minimum, table or
                        # rejection side effects. Validate/drain its exact replies.
                        self.statistics['discarded'] += len(order)
                        while pending:
                            self._reply(pending)
                        return False
                if not order and exhausted:
                    self._check()
                    return True
                if pending:
                    number, query, delta = self._reply(pending)
                    ready[query.request_id] = delta
                    free.append(number)
                elif order:
                    raise GeometryProcessError('geometry sequence has an unresolved result')
        except BaseException:
            self.close()
            raise
        finally:
            self.active = False
            self.sequence_wall_seconds=getattr(self,'sequence_wall_seconds',0.)+time.monotonic()-sequence_began

    def finish(self):
        """Confirm all prior worker output and unchanged files before admission.

        Per-pipe completion barriers expose duplicate/unsolicited late replies.
        This closes the request stream; it does not authorize robot commands.
        """
        if self.finished:
            return
        if self.active:
            raise GeometryProcessError('cannot finish an active geometry sequence')
        try:
            self._check()
            for worker,last in zip(self.workers,self.last_worker_requests):
                worker.send(dict(kind='finish',token=self.token,last_request_id=last))
            completed=set();deadline=min(self.deadline,time.monotonic()+self.query_seconds)
            while len(completed)!=len(self.workers):
                number,message=self._event(deadline)
                expected=dict(kind='finished',token=self.token,worker=number,
                    pid=self.workers[number].process.pid,source_id=self.identity.source_id,
                    model_id=self.identity.model_id,epoch=self.epoch,scene_id=self.scene_id,
                    geometry_id=self.geometry_id,last_request_id=self.last_worker_requests[number])
                if number in completed or message!=expected:
                    raise GeometryProcessError('geometry completion barrier mismatch')
                completed.add(number)
            self.identity.verify()
            self.finished=True
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for worker in self.workers:
            if getattr(worker,'identity',None) is None and worker.process.poll() is None:
                try:
                    worker.capture_identity()
                except BaseException as error:
                    worker.identity_capture_error=str(error)[:2048]
        # Every process was created in its own session and never receives a
        # caller PID. Only these owned groups can be signalled.
        for worker in self.workers:
            if worker.process.poll() is None:
                try:
                    os.killpg(worker.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+2.
        for worker in self.workers:
            try:
                worker.process.wait(timeout=max(0.,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(worker.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        failures = []
        for worker in self.workers:
            try:
                worker.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                failures.append(f'worker {worker.number} was not reaped')
            finally:
                try:
                    worker.outgoing.put_nowait(None)
                except queue.Full:
                    pass
                for pipe in (worker.process.stdin, worker.process.stdout, worker.process.stderr):
                    try:
                        pipe.close()
                    except OSError as error:
                        failures.append(str(error))
        thread_deadline = time.monotonic()+2.
        for worker in self.workers:
            for thread in worker.threads:
                if thread.ident is not None:
                    thread.join(timeout=max(0.,thread_deadline-time.monotonic()))
        if any(thread.is_alive() for worker in self.workers for thread in worker.threads):
            failures.append('geometry pipe thread did not stop')
        self.closure = [dict(worker=w.number, pid=w.process.pid,
            returncode=w.process.poll(), stderr_bytes=w.stderr_bytes,
            identity=getattr(w, 'identity', None),
            identity_observation=getattr(w, 'identity_observation', None),
            identity_capture_error=getattr(w, 'identity_capture_error', None),
            reaped=w.process.returncode is not None,
            remaining_owned_child=False if w.process.returncode is not None else None,
            stderr_prefix=bytes(w.stderr_prefix).decode(errors='replace')) for w in self.workers]
        if failures:
            raise GeometryProcessError('; '.join(failures))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, *_):
        try:
            if exception_type is None:
                self.finish()
        finally:
            self.close()
