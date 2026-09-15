"""Pure pickup samples for the bounded geometry workers.

The scene contains a held-book transform and robot/tool models. It has no bin
or table registration and cannot run PLACE queries. The live parent owns all
measurement admissions, sample ordering, planning and controller actions.
"""
from dataclasses import dataclass
import json
import math

import numpy as np

from .pure_geometry_owner import GeometryQuery, RobotGeometryOwner, _canonical_digest
from .geometry_process_protocol import SampleDelta


KINDS = frozenset(('pickup_payload', 'pickup_body', 'pickup_tool',
                   'pickup_post_retreat_payload', 'pickup_post_retreat_tool'))
SCENE_KIND = 'pickup_lift_v1'
POST_RETREAT_SCENE_KIND = 'pickup_post_retreat_v1'


@dataclass(frozen=True)
class PickupGeometryQuery:
    """An ordinary exact input plus one explicitly bound pickup operation."""
    sample: GeometryQuery
    operation: str

    def __getattr__(self, name):
        return getattr(self.sample, name)

    @property
    def input_sha256(self):
        return _canonical_digest(dict(sample=self.sample.input_sha256,
                                      operation=self.operation))

    def validated(self):
        if (type(self) is not PickupGeometryQuery or type(self.operation) is not str
                or self.operation not in KINDS):
            raise ValueError('invalid pickup operation')
        self.sample.validated()
        if self.sample.loaded is not True:
            raise ValueError('pickup geometry requires a carried book')
        return self


def _array(value, shape, label):
    current = np.asarray(value, dtype=np.float64)
    if current.shape != shape or not np.isfinite(current).all():
        raise ValueError('invalid pickup '+label)
    return np.frombuffer(current.tobytes(), dtype=np.float64).reshape(shape)


def validate_pickup_scene(scene):
    fields = {'kind', 'front', 'grasp', 'attached_corners', 'book_dimensions',
              'aperture', 'finger_positions', 'transition_samples',
              'maximum_tilt', 'supported_jaw_vertical_component', 'shelf_margin'}
    if type(scene) is not dict or scene.get('kind') not in (SCENE_KIND, POST_RETREAT_SCENE_KIND):
        raise ValueError('invalid pickup scene fields')
    if scene['kind'] == POST_RETREAT_SCENE_KIND:
        fields = fields | {'retreat_clearance', 'post_retreat_shelf_front_x'}
    if set(scene) != fields:
        raise ValueError('invalid pickup scene fields')
    current = json.loads(json.dumps(scene, sort_keys=True, allow_nan=False))
    for key, shape in (('front', (3,)), ('grasp', (8,)),
                       ('attached_corners', (8, 3)), ('book_dimensions', (3,))):
        _array(current[key], shape, key)
    if np.any(np.asarray(current['book_dimensions']) <= 0):
        raise ValueError('invalid pickup book dimensions')
    corners = np.asarray(current['attached_corners'])
    if np.linalg.norm(corners[1]-corners[0]) <= 1e-9:
        raise ValueError('pickup attachment has no vertical axis')
    if (type(current['transition_samples']) is not int
            or not 3 <= current['transition_samples'] <= 1000):
        raise ValueError('invalid pickup transition count')
    for key, low, high in (('aperture', 0., .069), ('maximum_tilt', 0., math.pi),
                          ('supported_jaw_vertical_component', 0., 1.),
                          ('shelf_margin', 0., 1.)):
        value = current[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('invalid pickup '+key)
    fingers = current['finger_positions']
    if fingers is not None and (type(fingers) is not dict or len(fingers) > 32
            or any(type(name) is not str or not name.startswith('gripper_left_')
                   or type(value) not in (int, float) or not math.isfinite(value)
                   for name, value in fingers.items())):
        raise ValueError('invalid pickup finger positions')
    if current['kind'] == POST_RETREAT_SCENE_KIND:
        retreat = current['retreat_clearance']
        plane = current['post_retreat_shelf_front_x']
        if (type(retreat) not in (int, float) or not math.isfinite(retreat) or not 0. <= retreat <= 2.
                or type(plane) not in (int, float) or not math.isfinite(plane)
                or plane != float(current['front'][0]) + float(retreat)):
            raise ValueError('invalid pickup post-retreat shelf binding')
    return current


def pickup_model_signature(node, scene):
    """Bind actual robot and nine-link tool geometry without a fake obstacle."""
    from .geometry_process_models import _array as model_array, _chain, _meshes
    from .shelf_cradle_geometry import ShelfCradleGeometry
    from .exact_world_geometry import ordinary_model_producer
    from .pure_geometry_owner import RobotCollisionGeometry
    current = validate_pickup_scene(scene)
    tool = node._shelf_cradle_geometry
    owner_methods = ('_robot_self_collision', '_carried_robot_collision',
                     '_world_collision_surfaces', '_collision_link_transforms')
    ordinary_owner = (type(node) is PickupGeometryOwner and all(
        getattr(getattr(node, name), '__func__', None) is getattr(RobotCollisionGeometry, name)
        for name in owner_methods))
    live_body = getattr(getattr(node, '_robot_self_collision', None), '__func__', None)
    live_type = getattr(live_body, '__globals__', {}).get('ManipulationNode')
    expected_payload = (getattr(live_type, '_carried_robot_collision', None)
                        if live_type is not None else RobotCollisionGeometry._carried_robot_collision)
    ordinary_payload = (getattr(getattr(node, '_carried_robot_collision', None), '__func__', None)
                        is expected_payload)
    if ((not ordinary_model_producer(node) and not ordinary_owner) or not ordinary_payload
            or type(tool) is not ShelfCradleGeometry
            or getattr(tool.local_surfaces, '__func__', None) is not ShelfCradleGeometry.local_surfaces
            or getattr(tool.model_local_surface, '__func__', None) is not ShelfCradleGeometry.model_local_surface):
        raise ValueError('ordinary pickup geometry required')
    local = tool.local_surfaces(current['aperture'], current['finger_positions'])
    values = dict(scene=current,
        chains=[_chain(node.chain), _chain(node.right_chain), _chain(node.head_chain)],
        meshes=_meshes(node.carried_collision_meshes),
        tool=dict(meshes=_meshes(tool.model.meshes), grasp=_chain(tool.grasp_chain),
            chains=[[name, _chain(chain)] for name, chain in tool.chains.items()],
            mimics=[[name, dict(element.attrib)] for name, element in tool.mimics.items()],
            watertight=list(tool.watertight.items()),
            local=[[name, model_array(surface)] for name, surface in local.items()]))
    return _canonical_digest(values)


class PickupGeometryOwner(RobotGeometryOwner):
    """Reuses the normal pure robot models and exact private caches."""
    def begin_scene(self, *, epoch, scene):
        if type(epoch) is not int or epoch <= self._epoch:
            raise ValueError('pickup scene epoch must advance')
        current = validate_pickup_scene(scene)
        grasp = _array(current['grasp'], (8,), 'grasp')
        if np.any(grasp < self.chain.lower) or np.any(grasp > self.chain.upper):
            raise ValueError('pickup grasp outside joint limits')
        self.carried_transition_samples = current['transition_samples']
        self.carried_book_dimensions = _array(current['book_dimensions'], (3,), 'book')
        self.carried_maximum_tilt = current['maximum_tilt']
        self.carried_supported_jaw_vertical_component = current['supported_jaw_vertical_component']
        self.carried_shelf_margin = current['shelf_margin']
        self._held_book_corners = _array(current['attached_corners'], (8, 3), 'attachment')
        self._pickup_scene = current
        self._scene = None  # No PLACE checker or fabricated registered obstacle.
        self._scene_id = _canonical_digest(current)
        self._epoch = epoch
        self._last_request_id = -1
        self._queries_seen = 0
        self._cancel.clear()
        return self._scene_id

    def model_signature(self):
        return pickup_model_signature(self, self._pickup_scene)

    def evaluate_independent(self, query):
        if type(query) is not PickupGeometryQuery:
            raise ValueError('pickup owner requires a pickup query')
        query.validated()
        if (query.epoch != self._epoch or query.source_id != self.assets.source_id
                or query.model_id != self.assets.model_id or query.scene_id != self._scene_id):
            raise RuntimeError('pickup query identity mismatch')
        if query.request_id <= self._last_request_id or self._queries_seen >= self.maximum_queries:
            raise RuntimeError('pickup query order/resource bound exceeded')
        if query.aperture != self._pickup_scene['aperture']:
            raise RuntimeError('pickup query aperture differs from scene')
        if (query.operation in ('pickup_post_retreat_payload', 'pickup_post_retreat_tool')
                and self._pickup_scene['kind'] != POST_RETREAT_SCENE_KIND):
            raise RuntimeError('post-retreat operation requires bound shelf scene')
        self._last_request_id = query.request_id
        self._queries_seen += 1
        q = np.frombuffer(query.q, dtype=np.float64)
        context = dict(right_positions=np.frombuffer(query.right, dtype=np.float64),
                       head_positions=np.frombuffer(query.head, dtype=np.float64))
        reason = None
        if self._cancel.is_set():
            reason = 'cancelled'
        elif np.any(q < self.chain.lower) or np.any(q > self.chain.upper):
            reason = 'pickup_joint_limit'
        elif query.operation == 'pickup_body':
            reason = self._robot_self_collision(q, **context)
        elif query.operation in ('pickup_payload', 'pickup_post_retreat_payload'):
            shelf_options = {}
            maximum_x = None
            if query.operation == 'pickup_post_retreat_payload':
                maximum_x = (self._pickup_scene['post_retreat_shelf_front_x']
                             - self.carried_shelf_margin)
                shelf_options['maximum_robot_x'] = maximum_x
            transform = self.chain.forward(q)
            corners = self._held_book_corners
            vertical = corners[1]-corners[0]
            vertical = vertical/np.linalg.norm(vertical)
            if (float((transform[:3, :3] @ vertical)[2]) < float(np.cos(self.carried_maximum_tilt))
                    and abs(float(transform[2, 1])) < self.carried_supported_jaw_vertical_component):
                reason = 'pickup_payload_tilt'
            else:
                world = corners @ transform[:3, :3].T + transform[:3, 3]
                if maximum_x is not None and float(np.max(world[:, 0])) > maximum_x:
                    reason = 'pickup_post_retreat_payload_shelf'
                else:
                    reason = self._carried_robot_collision(q, world, **context, **shelf_options)
        elif query.operation in ('pickup_tool', 'pickup_post_retreat_tool'):
            from .shelf_cradle_geometry import check_cradle_tool_sweep
            scene = self._pickup_scene
            # A stationary call has exactly one unique sampled state. All
            # original preconditions, palm and complete tool/body checks run.
            reason = check_cradle_tool_sweep(self, scene['front'], scene['grasp'],
                q, q, (scene['post_retreat_shelf_front_x']
                       if query.operation == 'pickup_post_retreat_tool' else None),
                aperture=query.aperture,
                finger_positions=scene['finger_positions'], **context)
        if self._cancel.is_set():
            reason = 'cancelled'
        return SampleDelta(query.request_id, query.epoch, query.source_id,
            query.model_id, query.scene_id, query.input_sha256, reason is None,
            None, None if reason is None else {'reason': str(reason)}, False, None).validate()
