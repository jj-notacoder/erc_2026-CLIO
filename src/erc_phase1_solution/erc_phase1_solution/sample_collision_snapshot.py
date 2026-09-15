"""Private per-sample robot geometry; no verdict or cross-sample cache.

Only fresh ordinary transform/concatenate outputs may be offered. Cradle
capture can defer that producer until an original body-cache miss. Capture binds exact current q/right/head bytes and all immutable model
owners. A self-collision cache miss freezes those outputs into bytes and computes
the original min/max once. Both callers then receive fresh metadata over those
same immutable bytes. Cache hits need not copy or reduce anything here.

Custom meshes/arrays and a changed context/model fall back to original work.
The factory is private provenance, not a public validator of externally supplied
world geometry. Concurrent native mutation of its private inputs is unsupported.
"""
import numpy as np

from .exact_local_bounds import ModelLocalMesh
from .kinematics import CollisionMesh
from .exact_world_geometry import _WorldSurface, ordinary_model_producer


def _state_bytes(value, size):
    if (type(value) is not np.ndarray or value.dtype != np.dtype(np.float64)
            or value.shape != (size,) or not value.flags.c_contiguous
            or not np.all(np.isfinite(value))):
        return None
    return value.tobytes()


def _model_bindings(node):
    meshes = getattr(node, 'carried_collision_meshes', None)
    if type(meshes) is not tuple or not meshes:
        return None
    bindings = []
    for mesh in meshes:
        if (type(mesh) is not CollisionMesh or type(mesh.link) is not str
                or type(mesh.watertight) is not bool
                or type(mesh.local_mesh) is not ModelLocalMesh
                or not mesh.local_mesh.matches(mesh.triangles)):
            return None
        bindings.append((mesh, mesh.link, mesh.watertight, mesh.local_mesh,
                         mesh.triangles.shape))
    return tuple(bindings)


class _SceneRobotSnapshot:
    __slots__ = ('_owner', '_state', '_models', '_world', '_frozen', '_geometry')

    def __init__(self, owner, state, models, world, geometry=None):
        self._owner, self._state, self._models = owner, state, models
        self._world = world
        self._frozen = None
        self._geometry = {} if geometry is None else dict(geometry)

    def _matches(self, node, q, right, head):
        if node is not self._owner:
            return False
        state = (_state_bytes(q, 8), _state_bytes(right, 7), _state_bytes(head, 2))
        if None in state or state != self._state:
            return False
        current = _model_bindings(node)
        if current is None or len(current) != len(self._models):
            return False
        return all(a[0] is b[0] and a[1:3] == b[1:3]
                   and a[3] is b[3] and a[4] == b[4]
                   for a, b in zip(current, self._models))

    def _views(self):
        surfaces, bounds = {}, {}
        for link, shape, world_bytes, bounds_bytes in self._frozen:
            surfaces[link] = np.frombuffer(world_bytes, dtype=np.float64).reshape(shape)
            bounds[link] = np.frombuffer(bounds_bytes, dtype=np.float64).reshape(2, 3)
        return surfaces, bounds

    def resolve(self, node, q, right, head):
        if not self._matches(node, q, right, head):
            return None
        if self._frozen is None:
            if self._world is None:
                # The original body predicate resolves parked context and
                # checks its full-state cache before it reaches this method.
                # Preserve that ordering, including cached rejection errors.
                if not ordinary_model_producer(node):
                    return None
                robot = node._world_collision_surfaces(
                    q, right_positions=right, head_positions=head)
                captured = _capture_scene_robot_snapshot(node, q, right, head, robot)
                if (captured is None or not ordinary_model_producer(node)
                        or not self._matches(node, q, right, head)):
                    return None
                self._world = captured._world
            frozen = []
            for link, surface in self._world:
                geometry = self._geometry.get(link)
                if type(geometry) is _WorldSurface and geometry.matches(surface):
                    frozen.append((link, *geometry.frozen()))
                    continue
                content = surface.tobytes(order='C')
                current = np.frombuffer(content, dtype=np.float64).reshape(surface.shape)
                # The old self-collision expression and min-before-max order.
                bounds = np.asarray([np.min(current, axis=(0, 1)),
                                     np.max(current, axis=(0, 1))])
                if not np.all(np.isfinite(bounds)):
                    return None
                frozen.append((link, current.shape, content, bounds.tobytes()))
            if not self._matches(node, q, right, head):
                return None
            self._frozen = tuple(frozen)
            self._world = ()
            self._geometry = {}
        return self._views()

    def resolved(self, node, q, right, head):
        if self._frozen is None or not self._matches(node, q, right, head):
            return None
        return self._views()


def _capture_cradle_robot_snapshot(node, q, right, head):
    """Bind a single exact sample without generating facets before body admission."""
    if not ordinary_model_producer(node):
        return None
    state = (_state_bytes(q, 8), _state_bytes(right, 7), _state_bytes(head, 2))
    if None in state:
        return None
    models = _model_bindings(node)
    if models is None:
        return None
    return _SceneRobotSnapshot(node, state, models, None)


def _capture_scene_robot_snapshot(node, q, right, head, robot, *, _world_geometry=None):
    """Private caller already built robot from these exact current inputs."""
    consumer = getattr(getattr(node, '_robot_self_collision', None), '__func__', None)
    original = getattr(getattr(node, '_scene_snapshot_body', None), '__func__', None)
    if consumer is None or consumer is not original:
        return None
    state = (_state_bytes(q, 8), _state_bytes(right, 7), _state_bytes(head, 2))
    if None in state or type(robot) is not dict:
        return None
    models = _model_bindings(node)
    if models is None:
        return None
    counts = {}
    for _, link, _, _, shape in models:
        counts[link] = counts.get(link, 0) + shape[0]
    if tuple(robot) != tuple(counts):
        return None
    world = []
    geometry = _world_geometry if type(_world_geometry) is dict else {}
    for link, surface in robot.items():
        cached = geometry.get(link)
        immutable = type(cached) is _WorldSurface and cached.matches(surface)
        if (type(surface) is not np.ndarray or surface.dtype != np.dtype(np.float64)
                or surface.shape != (counts[link], 3, 3)
                or not surface.flags.c_contiguous
                or not (immutable or (surface.flags.owndata and surface.base is None))):
            return None
        world.append((link, surface))
    return _SceneRobotSnapshot(node, state, models, tuple(world), geometry)
