"""Optional carried-volume samples inside an already registered PLACE pool.

No new worker, scene, motion primitive or scene-verdict cache is introduced.
The dense payload pass and later adaptive body pass keep their original order.
"""
from dataclasses import dataclass
import json
import math

import numpy as np

from .geometry_process_protocol import SampleDelta
from .geometry_process_pool import GeometryProcessError
from .kinematics import interpolate_joint_waypoints
from .pure_geometry_owner import GeometryQuery, _canonical_digest


@dataclass(frozen=True)
class PlaceVolumeQuery:
    sample: GeometryQuery
    operation: str

    def __getattr__(self, name):
        return getattr(self.sample, name)

    @property
    def input_sha256(self):
        return _canonical_digest(dict(sample=self.sample.input_sha256,
                                      place_volume_operation=self.operation))

    def validated(self):
        if (type(self) is not PlaceVolumeQuery or type(self.operation) is not str
                or self.operation not in ('place_payload', 'place_body')):
            raise ValueError('invalid PLACE volume operation')
        if type(self.sample) is not GeometryQuery:
            raise ValueError('nested PLACE volume queries are invalid')
        self.sample.validated()
        if self.sample.loaded is not True:
            raise ValueError('PLACE volume requires a carried book')
        return self


CONTEXT_FIELDS = dict(maximum_tilt='carried_maximum_tilt',
    supported_jaw_vertical_component='carried_supported_jaw_vertical_component',
    joint_step='cartesian_joint_step', orientation_step='carried_orientation_step_limit')


def validate_volume_context(value):
    if type(value) is not dict or set(value) != {*CONTEXT_FIELDS, 'aperture'}:
        raise ValueError('invalid PLACE volume context fields')
    for key, low, high in (('maximum_tilt', 0., math.pi),
            ('supported_jaw_vertical_component', 0., 1.), ('joint_step', 0., math.inf),
            ('orientation_step', 0., math.inf), ('aperture', 0., .069)):
        current=value[key]
        if (type(current) not in (int, float) or not math.isfinite(current)
                or current < low or current > high
                or (key in ('joint_step', 'orientation_step', 'supported_jaw_vertical_component')
                    and current == 0.)):
            raise ValueError('invalid PLACE volume '+key)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def capture_volume_context(node, aperture):
    return validate_volume_context(dict(aperture=aperture,
        **{key: getattr(node, name) for key, name in CONTEXT_FIELDS.items()}))


def ordinary_volume_parent(node):
    # The existing scene admission already verifies model/body producers.
    # Also reject overridden original volume/route/payload methods before
    # substituting this narrowly equivalent sample implementation.
    from .pure_geometry_owner import RobotGeometryOwner, RobotCollisionGeometry
    if type(node) is RobotGeometryOwner:
        return all(getattr(getattr(node,name,None),'__func__',None) is getattr(RobotCollisionGeometry,name)
            for name in ('_carried_robot_collision','_robot_self_collision',
                         '_world_collision_surfaces','_collision_link_transforms'))
    body=getattr(getattr(node, '_robot_self_collision', None), '__func__', None)
    kind=getattr(body, '__globals__', {}).get('ManipulationNode')
    names=('_carried_robot_collision', '_carried_volume_transition_is_safe',
           '_carried_robot_transition_is_safe', '_plan_carried_joint_route')
    return kind is not None and type(node) is kind and all(
        getattr(getattr(node, name, None), '__func__', None) is getattr(kind, name)
        for name in names)


def evaluate_owned_volume(owner, query):
    """One pure predicate; never touches PlaceSceneChecker diagnostic state."""
    if type(query) is not PlaceVolumeQuery:
        raise ValueError('exact PLACE volume query required')
    query.validated()
    scene=owner._scene
    context=None if scene is None else getattr(scene, '_place_volume_context', None)
    if context is None:
        raise RuntimeError('PLACE volume requires a bound registered scene')
    if (query.epoch != owner._epoch or query.source_id != owner.assets.source_id
            or query.model_id != owner.assets.model_id or query.scene_id != owner._scene_id):
        raise RuntimeError('PLACE volume query identity mismatch')
    if (query.request_id <= owner._last_request_id or owner._queries_seen >= owner.maximum_queries
            or (query.aperture not in owner._apertures
                and len(owner._apertures) >= owner.maximum_distinct_apertures)):
        raise RuntimeError('PLACE volume query order/resource bound exceeded')
    if query.aperture != context['aperture']:
        raise RuntimeError('PLACE volume aperture changed')
    owner._last_request_id=query.request_id
    owner._queries_seen+=1
    owner._apertures.add(query.aperture)
    q=np.frombuffer(query.q, dtype=np.float64)
    parked=dict(right_positions=np.frombuffer(query.right, dtype=np.float64),
                head_positions=np.frombuffer(query.head, dtype=np.float64))
    reason=None
    if owner._cancel.is_set():
        reason='cancelled'
    elif query.operation == 'place_body':
        reason=owner._robot_self_collision(q, **parked)
    else:
        transform=owner.chain.forward(q)
        corners=scene.attached
        vertical=corners[1]-corners[0]
        norm=float(np.linalg.norm(vertical))
        if norm <= 1e-9:
            raise ValueError('attached book envelope has no vertical axis')
        vertical/=norm
        if (float((transform[:3, :3] @ vertical)[2]) < float(np.cos(context['maximum_tilt']))
                and abs(float(transform[2, 1])) < context['supported_jaw_vertical_component']):
            reason='place_payload_tilt'
        else:
            world=corners @ transform[:3, :3].T + transform[:3, 3]
            reason=owner._carried_robot_collision(q, world, maximum_robot_x=None, **parked)
    if owner._cancel.is_set():
        reason='cancelled'
    return SampleDelta(query.request_id, query.epoch, query.source_id,
        query.model_id, query.scene_id, query.input_sha256, reason is None,
        None, None if reason is None else {'reason': str(reason)}, False, None).validate()


def evaluate_volume(backend, start, end, attached_corners):
    """Original controller-sized dense payload pass then adaptive body pass."""
    if not backend.volume_active or backend.pool is None or not backend.entered:
        raise GeometryProcessError('PLACE volume backend is not active')
    backend.capture.verify_bindings()
    node=backend.node
    first,last=np.asarray(start,dtype=float),np.asarray(end,dtype=float)
    if first.shape != (8,) or last.shape != (8,) or not np.isfinite((first,last)).all():
        raise ValueError('PLACE volume joint shape invalid')
    if float(np.max(np.abs(last[1:]-first[1:]))) > node.cartesian_joint_step:
        return False
    delta=node.chain.pose_error(node.chain.forward(first),node.chain.forward(last))[3:]
    if float(np.linalg.norm(delta)) > node.carried_orientation_step_limit:
        return False
    corners=np.asarray(attached_corners,dtype=float)
    if corners.shape != (8,3):
        raise ValueError('attached book envelope must contain eight 3-D corners')
    if not np.isfinite(corners).all() or corners.tobytes() != backend.state.attached.tobytes():
        raise GeometryProcessError('PLACE volume attachment changed')
    if float(np.linalg.norm(corners[1]-corners[0])) <= 1e-9:
        raise ValueError('attached book envelope has no vertical axis')
    aperture=backend.capture.volume_context['aperture']
    backend.last_volume_rejection=None

    def sequence(samples,operation):
        committed=[0]
        def capture(index,q):
            sample=backend.capture.capture(index,(q,aperture,True),pool=backend.pool)
            return PlaceVolumeQuery(sample,operation).validated()
        def consume(query,reply):
            if node._cancel.is_set():
                raise GeometryProcessError('PLACE volume cancelled before commit')
            if reply is None or not reply.matches(query):
                raise GeometryProcessError('PLACE volume result mismatch')
            if (reply.minimum_update is not None or reply.table_touched
                    or reply.table_intersection is not None):
                raise GeometryProcessError('PLACE volume cannot update scene diagnostics')
            if not reply.verdict:
                backend.last_volume_rejection=dict(operation=operation,
                    sample=committed[0],reason=reply.rejection_update['reason'])
            committed[0]+=1
            return reply.verdict
        return backend.pool.evaluate_sequence(samples,capture=capture,consume=consume)

    unchanged=np.allclose(first,last,rtol=0.,atol=1e-12)
    fractions=(0.,) if unchanged else np.linspace(0.,1.,node.carried_transition_samples)
    if not sequence((first+(last-first)*f for f in fractions),'place_payload'):
        return False
    waypoints=list(interpolate_joint_waypoints(first,last,first_arm_index=1,
        maximum_joint_step=.5*node.cartesian_joint_step,
        orientation_distance=lambda a,b:np.linalg.norm(
            node.chain.pose_error(node.chain.forward(a),node.chain.forward(b))[3:]),
        maximum_orientation_step=.5*node.carried_orientation_step_limit))
    if len(waypoints)==1 and not unchanged:
        waypoints.insert(0,.5*(first+last))
    return sequence((first,*waypoints),'place_body')
