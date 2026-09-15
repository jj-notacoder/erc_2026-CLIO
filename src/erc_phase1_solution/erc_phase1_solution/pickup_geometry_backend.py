"""Opt-in parallel lift mesh checking; planning and live actions stay serial."""
import copy
import math
import time

import numpy as np

from .geometry_process_pool import (
    GeometryProcessPool, GeometryProcessError, checked_geometry_process_workers,
)
from .motion_profiles import RIGHT_ARM_JOINTS
from .kinematics import interpolate_joint_waypoints
from .pure_geometry_owner import GeometryQuery, HEAD_JOINTS
from .pickup_geometry_owner import (
    PickupGeometryQuery, SCENE_KIND, POST_RETREAT_SCENE_KIND, validate_pickup_scene, pickup_model_signature,
)


def checked_pickup_parallel_geometry_enabled(value):
    if type(value) is not bool:
        raise ValueError('pickup_parallel_geometry_enabled must be a Boolean')
    return value


def checked_pickup_post_retreat_parallel_geometry_enabled(value):
    if type(value) is not bool:
        raise ValueError('pickup_post_retreat_parallel_geometry_enabled must be a Boolean')
    return value


def initialize_pickup_geometry_backend(node, resolve_package):
    node._pickup_parallel_geometry_identity = None
    node._pickup_parallel_geometry_epoch = 0
    node._pickup_parallel_geometry_active = False
    if checked_pickup_parallel_geometry_enabled(getattr(node, 'pickup_parallel_geometry_enabled', False)):
        identity = getattr(node, '_place_parallel_geometry_identity', None)
        if identity is None:
            from .installed_geometry_identity import InstalledGeometryIdentity, required_package_names
            shares = {name: resolve_package(name) for name in required_package_names(resolve_package('erc_description'))}
            identity = InstalledGeometryIdentity.capture(package_shares=shares)
        node._pickup_parallel_geometry_identity = identity


def _same(first, second):
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


class PickupGeometryBackend:
    """One closed-before-motion lifetime with exact parent-ordered samples."""
    def __init__(self, node, identity, *, epoch, front, grasp, attached,
                 aperture, finger_positions, reference, pool_limits=None, include_post_retreat=False):
        self.node, self.identity, self.epoch = node, identity, epoch
        self.reference = copy.deepcopy(reference)
        if type(include_post_retreat) is not bool:
            raise ValueError('include_post_retreat must be a Boolean')
        shelf_fields = {}
        if include_post_retreat:
            retreat = float(node.carried_shelf_retreat_clearance)
            shelf_fields = dict(retreat_clearance=retreat,
                post_retreat_shelf_front_x=float(np.asarray(front)[0]) + retreat)
        self.scene = validate_pickup_scene(dict(kind=(POST_RETREAT_SCENE_KIND if include_post_retreat else SCENE_KIND),
            front=np.asarray(front, dtype=float).tolist(), grasp=np.asarray(grasp, dtype=float).tolist(),
            attached_corners=np.asarray(attached, dtype=float).tolist(),
            book_dimensions=np.asarray(node.carried_book_dimensions, dtype=float).tolist(),
            aperture=float(aperture), finger_positions=copy.deepcopy(finger_positions),
            transition_samples=int(node.carried_transition_samples),
            maximum_tilt=float(node.carried_maximum_tilt),
            supported_jaw_vertical_component=float(node.carried_supported_jaw_vertical_component),
            shelf_margin=float(node.carried_shelf_margin), **shelf_fields))
        self.bindings = tuple(getattr(node, name) for name in (
            'chain', 'right_chain', 'head_chain', 'carried_collision_meshes', '_shelf_cradle_geometry'))
        self.grid_parameters = (float(node.cartesian_joint_step),
                                float(node.carried_orientation_step_limit))
        self.pool_limits = dict(pool_limits or {})
        self.pool = None
        self.last_rejection = None
        self.last_admission = None

    def _verify_bindings(self):
        node, scene = self.node, self.scene
        if (scene['kind'] == POST_RETREAT_SCENE_KIND
                and node.carried_shelf_retreat_clearance != scene['retreat_clearance']):
            raise GeometryProcessError('pickup retreat binding changed')
        if any(getattr(node, name) is not original for name, original in zip((
                'chain', 'right_chain', 'head_chain', 'carried_collision_meshes', '_shelf_cradle_geometry'), self.bindings)):
            raise GeometryProcessError('pickup model generation changed')
        if (not _same(node.carried_book_dimensions, scene['book_dimensions'])
                or node.carried_transition_samples != scene['transition_samples']
                or node.carried_maximum_tilt != scene['maximum_tilt']
                or node.carried_supported_jaw_vertical_component != scene['supported_jaw_vertical_component']
                or node.carried_shelf_margin != scene['shelf_margin']
                or (node.cartesian_joint_step, node.carried_orientation_step_limit) != self.grid_parameters):
            raise GeometryProcessError('pickup planning parameters changed')

    def __enter__(self):
        if self.pool is not None:
            raise GeometryProcessError('pickup backend cannot be reused')
        self._verify_bindings()
        self.node._lift_first_measurements(self.reference)
        self.identity.verify()
        self.geometry_id = pickup_model_signature(self.node, self.scene)
        self.pool = GeometryProcessPool(self.identity, epoch=self.epoch, scene=self.scene,
            geometry_id=self.geometry_id, cancelled=self.node._cancel.is_set, **self.pool_limits)
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if self.pool is not None:
                try:
                    if kind is None:
                        self.pool.finish()
                finally:
                    self.pool.close()
            if kind is None:
                self._verify_bindings()
                self.node._lift_first_measurements(self.reference)
                self.identity.verify()
                if pickup_model_signature(self.node, self.scene) != self.geometry_id:
                    raise GeometryProcessError('pickup geometry changed during planning')
        finally:
            pass

    def _capture(self, index, item, *, operation, context):
        self._verify_bindings()
        if self.node._cancel.is_set():
            raise GeometryProcessError('pickup geometry cancelled')
        # Existing admission captures all joint stamps and odometry under one
        # sensor lock. Enqueued nominal samples use those fresh parked values;
        # measured validation retains its original explicit parked snapshot,
        # comparing every fresh enqueue against the existing .001 rad gate.
        measured = self.node._lift_first_measurements(self.reference)
        actual = {key: np.asarray([measured['joints'][name] for name in names], dtype=float)
                  for key, names in (('right_positions', RIGHT_ARM_JOINTS), ('head_positions', HEAD_JOINTS))}
        for key, supplied in context.items():
            if not _same_shape_finite(supplied, actual[key].shape):
                raise GeometryProcessError('pickup parked query context invalid')
            if np.max(np.abs(actual[key]-np.asarray(supplied))) > .001:
                raise GeometryProcessError('pickup parked collision context moved')
            actual[key] = np.asarray(supplied, dtype=float)
        sample = GeometryQuery.capture(request_id=index, epoch=self.pool.epoch,
            source_id=self.identity.source_id, model_id=self.identity.model_id,
            scene_id=self.pool.scene_id, q=item,
            right=actual['right_positions'], head=actual['head_positions'],
            aperture=self.scene['aperture'], loaded=True)
        query = PickupGeometryQuery(sample, operation).validated()
        self.last_admission = dict(request_id=index, operation=operation,
            input_sha256=query.input_sha256, measured_stamp_ns=measured['stamp_ns'])
        return query

    def _sequence(self, samples, operation, context):
        if self.pool is None:
            raise GeometryProcessError('pickup backend is not active')
        self.last_rejection = None
        consumed = [0]
        def consume(query, delta):
            if self.node._cancel.is_set():
                raise GeometryProcessError('pickup geometry cancelled before commit')
            if delta is None or not delta.matches(query):
                raise GeometryProcessError('pickup geometry result mismatch')
            if not delta.verdict:
                reason = delta.rejection_update['reason']
                # Original cradle calls number samples within each sweep. The
                # worker evaluates the same state as a one-sample sweep.
                if operation == 'pickup_tool' and reason.endswith(':sample0'):
                    reason = reason[:-len(':sample0')]+':sample'+str(consumed[0])
                self.last_rejection = reason
            consumed[0] += 1
            return delta.verdict
        return self.pool.evaluate_sequence(samples,
            capture=lambda index, item: self._capture(index, item, operation=operation, context=context),
            consume=consume)

    def volume(self, start, end, attached_corners, *, maximum_payload_x=None,
               maximum_robot_x=None, **context):
        """Original dense payload pass, then original adaptive body pass."""
        if set(context)-{'right_positions', 'head_positions'}:
            raise GeometryProcessError('unsupported pickup volume context')
        if not _same(attached_corners, self.scene['attached_corners']):
            raise GeometryProcessError('pickup attachment changed')
        operation = 'pickup_payload'
        if maximum_payload_x is not None or maximum_robot_x is not None:
            if self.scene['kind'] != POST_RETREAT_SCENE_KIND:
                raise GeometryProcessError('pickup shelf constraints need a post-retreat scene')
            bound = self.scene['post_retreat_shelf_front_x'] - self.scene['shelf_margin']
            if (type(maximum_payload_x) not in (int, float) or type(maximum_robot_x) not in (int, float)
                    or maximum_payload_x != bound or maximum_robot_x != bound):
                raise GeometryProcessError('pickup shelf constraints differ from bound scene')
            operation = 'pickup_post_retreat_payload'
        node = self.node
        first, last = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if not _same_shape_finite(first, (8,)) or not _same_shape_finite(last, (8,)):
            raise ValueError('pickup volume joint shape invalid')
        if float(np.max(np.abs(last[1:]-first[1:]))) > node.cartesian_joint_step:
            return False
        delta = node.chain.pose_error(node.chain.forward(first), node.chain.forward(last))[3:]
        if float(np.linalg.norm(delta)) > node.carried_orientation_step_limit:
            return False
        unchanged = np.allclose(first, last, rtol=0., atol=1e-12)
        fractions = (0.,) if unchanged else np.linspace(0., 1., node.carried_transition_samples)
        if not self._sequence((first+(last-first)*fraction for fraction in fractions), operation, context):
            return False
        waypoints = list(interpolate_joint_waypoints(first, last, first_arm_index=1,
            maximum_joint_step=.5*node.cartesian_joint_step,
            orientation_distance=lambda a, b: np.linalg.norm(
                node.chain.pose_error(node.chain.forward(a), node.chain.forward(b))[3:]),
            maximum_orientation_step=.5*node.carried_orientation_step_limit))
        if len(waypoints) == 1 and not unchanged:
            waypoints.insert(0, .5*(first+last))
        return self._sequence((first, *waypoints), 'pickup_body', context)

    def tool_sweep(self, front, grasp, start, end, shelf_plane, *, aperture=None,
                   finger_positions=None, **context):
        """All original61/adaptive tool samples, with first failure in order."""
        if (shelf_plane is not None or set(context)-{'right_positions', 'head_positions'}
                or not _same(front, self.scene['front']) or not _same(grasp, self.scene['grasp'])
                or aperture != self.scene['aperture'] or finger_positions != self.scene['finger_positions']):
            raise GeometryProcessError('pickup tool request differs from bound scene')
        first, last = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if not _same_shape_finite(first, (8,)) or not _same_shape_finite(last, (8,)):
            return 'cradle_tool_joint_shape_invalid'
        if (np.any(first < self.node.chain.lower) or np.any(first > self.node.chain.upper)
                or np.any(last < self.node.chain.lower) or np.any(last > self.node.chain.upper)):
            return 'cradle_tool_joint_limit'
        count = max(3, int(self.node.carried_transition_samples),
                    int(math.ceil(float(np.max(np.abs(last-first)))/.02))+1)
        if np.array_equal(first, last):
            count = 1
        if self._sequence((first+(last-first)*f for f in np.linspace(0., 1., count)), 'pickup_tool', context):
            return None
        return self.last_rejection


def _same_shape_finite(value, shape):
    current = np.asarray(value, dtype=float)
    return current.shape == shape and bool(np.isfinite(current).all())


def run_pickup_geometry(node, local_planner, front, grasp, route, *,
                        scene_reference, post_retreat_plan=None, **kwargs):
    """Close all workers and re-admit models before returning a usable plan."""
    if post_retreat_plan is not None and not callable(post_retreat_plan):
        raise ValueError('post-retreat continuation must be callable')
    if not checked_pickup_parallel_geometry_enabled(getattr(node, 'pickup_parallel_geometry_enabled', False)):
        try:
            result = local_planner(node, front, grasp, route, **kwargs)
            return result if post_retreat_plan is None else (result, post_retreat_plan(result, None))
        except BaseException:
            if post_retreat_plan is not None:
                node._cached_post_retreat_plan = None
            raise
    count = checked_geometry_process_workers(getattr(node, 'geometry_process_workers', 4))
    pool_options = {} if count == 4 else {'pool_limits': {'worker_count': count}}
    if post_retreat_plan is not None:
        pool_options['include_post_retreat'] = True
    if 'geometry_backend' in kwargs:
        raise ValueError('pickup backend lifetime is owned by this call')
    with node._lock:
        identity = node._pickup_parallel_geometry_identity
        epoch = node._pickup_parallel_geometry_epoch
        if (identity is None or type(epoch) is not int or not 0 <= epoch < 2**63-1
                or node._pickup_parallel_geometry_active):
            raise GeometryProcessError('parallel pickup geometry unavailable or active')
        epoch += 1
        node._pickup_parallel_geometry_epoch = epoch
        node._pickup_parallel_geometry_active = True
    started = time.monotonic()
    backend = failure = None
    passed = False
    try:
        attached = kwargs.get('attached_corners')
        if attached is None:
            attached = node._attached_book_corners(front, grasp)
        with PickupGeometryBackend(node, identity, epoch=epoch, front=front, grasp=grasp,
                attached=attached, aperture=kwargs['aperture'],
                finger_positions=kwargs.get('finger_positions'), reference=scene_reference, **pool_options) as backend:
            result = local_planner(node, front, grasp, route, geometry_backend=backend, **kwargs)
            if post_retreat_plan is not None:
                result = (result, post_retreat_plan(result, backend))
        passed = True
        return result
    except BaseException as error:
        failure = error
        if post_retreat_plan is not None:
            node._cached_post_retreat_plan = None
        raise
    finally:
        with node._lock:
            node._pickup_parallel_geometry_active = False
        try:
            pool = None if backend is None else backend.pool
            status = (pool.worker_status() if pool is not None else
                      getattr(failure, 'geometry_worker_status', None))
            if status is None:
                status = dict(workers_created=0, workers=[], workers_reaped=0, children_closed=True)
            node._publish_status('pickup_geometry_process_completed', command='pick',
                epoch=epoch, passed=passed, wall_seconds=time.monotonic()-started,
                failure_type=None if failure is None else type(failure).__name__, **status)
        except Exception:
            pass
