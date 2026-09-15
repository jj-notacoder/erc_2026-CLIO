"""Default-off empty geometry batches in one closed-before-motion pool."""
from contextlib import contextmanager
import os
import sys
import time

import numpy as np

from .empty_pickup_collision import EmptyPickupCollision, MASTER
from .empty_pickup_geometry_owner import (
    SCENE_KIND, EmptyPickupGeometryQuery, validate_empty_pickup_scene,
    empty_model_signature,
)
from .geometry_process_pool import (
    GeometryProcessPool, GeometryProcessError, checked_geometry_process_workers,
)
from .geometry_process_protocol import SampleDelta
from .pure_geometry_owner import GeometryQuery


def checked_empty_pickup_parallel_geometry_enabled(value):
    if type(value) is not bool:
        raise ValueError('empty_pickup_parallel_geometry_enabled must be a Boolean')
    return value


def _key(query):
    return query.q, np.float64(query.aperture).tobytes()


def commit_empty_sample(checker, query, delta):
    """Commit only the requested prefix, with the original sample semantics."""
    if checker.node._cancel.is_set():
        checker._reject('empty_pickup_cancelled')
        return False
    query.validated()
    q = np.frombuffer(query.q, dtype=np.float64)
    key = _key(query)
    if key in checker.cache:
        # Preserve cancellation, exact cached-rejection diagnostics and counts.
        return checker.sample(q, query.aperture) is None
    if type(delta) is not SampleDelta or not delta.matches(query):
        raise GeometryProcessError('empty geometry result does not match sample')
    if (delta.minimum_update is not None or delta.table_touched
            or delta.table_intersection is not None
            or (delta.rejection_update is not None
                and set(delta.rejection_update) != {'reason'})):
        raise GeometryProcessError('empty geometry result has unrelated updates')
    reason = None if delta.verdict else delta.rejection_update['reason']
    if reason == 'empty_pickup_joint_limit':
        # The original limit rejection does not populate cache/sample counts.
        actual = checker.sample(q, query.aperture)
        if actual != reason:
            raise GeometryProcessError('empty limit result differs from local predicate')
        return False
    checker.checked_samples += 1
    checker.cache[key] = reason
    if reason is not None:
        checker.last_rejected_q = q.tolist()
        checker.last_rejected_aperture = query.aperture
        checker._reject(reason)
        return False
    return True


class EmptyPickupGeometryBackend:
    """A separate empty epoch; the existing later loaded pool stays unchanged."""
    def __init__(self, node, checker, identity, epoch, scene, geometry_id):
        self.node, self.checker, self.identity = node, checker, identity
        self.epoch, self.scene, self.geometry_id = epoch, scene, geometry_id
        self.pool = None
        self.last_admission = None
        self.models = tuple(getattr(node, n) for n in (
            'chain', 'right_chain', 'head_chain', 'carried_collision_meshes'))
        self.components = checker.geometry, checker.screen
        self.parameters = self._parameters()

    def _parameters(self):
        return (self.node.gripper_open, self.node.carried_transition_samples,
                self.node.pick_torso_height, self.node.cartesian_clearance,
                self.node.cartesian_joint_step)

    def _admit(self):
        if (self.checker.node is not self.node
                or any(getattr(self.node, n) is not obj for n, obj in zip(
                    ('chain', 'right_chain', 'head_chain', 'carried_collision_meshes'), self.models))
                or self.checker.geometry is not self.components[0]
                or self.checker.screen is not self.components[1]
                or self._parameters() != self.parameters):
            raise GeometryProcessError('empty geometry model or parameters changed')
        for name in ('start', 'right', 'head'):
            actual = np.asarray(getattr(self.checker, name), dtype=np.float64)
            expected = np.asarray(self.scene[name], dtype=np.float64)
            if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                raise GeometryProcessError('empty geometry frozen context changed')
        if (self.checker.initial_aperture != self.scene['initial_aperture']
                or self.checker.open_aperture != self.scene['open_aperture']):
            raise GeometryProcessError('empty geometry aperture binding changed')
        return self.checker.require_fresh(self.checker.start,
            self.checker.initial_aperture, return_context=True)

    def __enter__(self):
        self._admit()
        self.identity.verify()
        self.pool = GeometryProcessPool(self.identity, epoch=self.epoch,
            scene=self.scene, geometry_id=self.geometry_id,
            cancelled=self.node._cancel.is_set,
            worker_count=checked_geometry_process_workers(self.node.geometry_process_workers))
        return self

    def __exit__(self, kind, error, traceback):
        if self.pool is not None:
            try:
                if kind is None:
                    self.pool.finish()
            finally:
                self.pool.close()
        if kind is None:
            self._admit()
            self.identity.verify()
            if empty_model_signature(self.checker, self.scene) != self.geometry_id:
                raise GeometryProcessError('empty geometry model changed during planning')

    def _capture(self, index, item):
        actual, stamps = self._admit()
        q, aperture = item
        sample = GeometryQuery.capture(request_id=index, epoch=self.pool.epoch,
            source_id=self.identity.source_id, model_id=self.identity.model_id,
            scene_id=self.pool.scene_id, q=q, right=self.checker.right,
            head=self.checker.head, aperture=aperture, loaded=False)
        query = EmptyPickupGeometryQuery(sample).validated()
        self.last_admission = dict(request_id=index, input_sha256=query.input_sha256,
            actual_joints={n: float(actual[n]) for n in stamps}, joint_stamps_ns=dict(stamps),
            measured_master_position=float(actual[MASTER]))
        return query

    def sequence(self, items):
        if self.pool is None:
            raise GeometryProcessError('empty geometry backend is not active')
        return self.pool.evaluate_sequence(items, capture=self._capture,
            consume=lambda q, d: commit_empty_sample(self.checker, q, d),
            is_cached=lambda q: _key(q) in self.checker.cache)


@contextmanager
def empty_pickup_geometry_scope(node, checker):
    """Default/unsupported paths stay serial; active pool faults abort safely."""
    enabled = checked_empty_pickup_parallel_geometry_enabled(
        getattr(node, 'empty_pickup_parallel_geometry_enabled', False))
    if not enabled or checker is None:
        yield
        return
    if (os.name != 'posix' or not sys.platform.startswith('linux')
            or type(checker) is not EmptyPickupCollision):
        yield
        return
    if getattr(checker, '_parallel_sequence', None) is not None:
        raise GeometryProcessError('empty geometry batch ownership is already active')
    scene = validate_empty_pickup_scene(dict(kind=SCENE_KIND,
        start=checker.start.tolist(), right=checker.right.tolist(), head=checker.head.tolist(),
        initial_aperture=checker.initial_aperture, open_aperture=checker.open_aperture,
        transition_samples=int(node.carried_transition_samples)))
    try:
        geometry_id = empty_model_signature(checker, scene)
    except ValueError:
        # No pool was constructed. Unsupported custom models use their ordinary
        # local checker; model/transport failure after bootstrap is never hidden.
        yield
        return
    with node._lock:
        identity = node._pickup_parallel_geometry_identity
        epoch = node._pickup_parallel_geometry_epoch
        if (identity is None or type(epoch) is not int or not 0 <= epoch < 2**63-1
                or node._pickup_parallel_geometry_active
                or getattr(node, '_place_parallel_geometry_active', False)):
            raise GeometryProcessError('empty geometry identity unavailable or pool active')
        epoch += 1
        node._pickup_parallel_geometry_epoch = epoch
        node._pickup_parallel_geometry_active = True
    started = time.monotonic()
    backend = failure = None
    passed = False
    try:
        with EmptyPickupGeometryBackend(node, checker, identity, epoch, scene, geometry_id) as backend:
            checker._parallel_sequence = backend.sequence
            try:
                yield
            finally:
                del checker._parallel_sequence
        passed = True
    except BaseException as error:
        failure = error
        raise
    finally:
        with node._lock:
            node._pickup_parallel_geometry_active = False
        try:
            pool = None if backend is None else backend.pool
            status = pool.worker_status() if pool is not None else getattr(failure, 'geometry_worker_status', None)
            if status is None:
                status = dict(workers_created=0, workers=[], workers_reaped=0, children_closed=True)
            node._publish_status('empty_pickup_geometry_process_completed', command='pick',
                epoch=epoch, passed=passed, wall_seconds=time.monotonic()-started,
                failure_type=None if failure is None else type(failure).__name__, **status)
        except Exception:
            pass
