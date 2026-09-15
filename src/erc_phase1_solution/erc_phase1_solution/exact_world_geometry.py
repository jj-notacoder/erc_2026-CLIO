"""Bounded exact transformed geometry for one ordinary PLACE planner.

Only explicit immutable model meshes qualify. Keys contain their immutable
owner and all current transform bytes, never a joint tolerance or a verdict.
Every caller receives fresh metadata over immutable facet/bounds bytes. The
original dense expression and min-before-max reductions run on each miss.
Raw/custom arrays and unsupported transforms retain their original path.

The scene owns this cache on its planning thread. It is not a sensor snapshot,
model validator, cross-command cache or concurrent mutable-input contract.
"""
from collections import OrderedDict

import numpy as np

from .exact_local_bounds import ModelLocalMesh
from .kinematics import CollisionMesh


def _surface_extrema(surface):
    """Exact bounds with strided reductions for ordinary immutable triangles.

    Finite nonzero extrema are input values, independent of reduction order.
    Zero extrema can have order-dependent signs, so those and nonfinite
    results retain the original reductions. Unsupported arrays retain NumPy
    dispatch. No transformed facet, collision margin or verdict is changed.
    """
    if (type(surface) is np.ndarray and surface.dtype == np.dtype(np.float64)
            and surface.ndim == 3 and surface.shape[1:] == (3, 3)
            and surface.shape[0] >= 64 and surface.flags.c_contiguous
            and not surface.flags.writeable):
        low = [surface[:, :, axis].min() for axis in range(3)]
        high = [surface[:, :, axis].max() for axis in range(3)]
        bounds = np.asarray([low, high])
        if np.isfinite(bounds).all() and not (bounds == 0.).any():
            return bounds
    return np.asarray([np.min(surface, axis=(0, 1)),
                       np.max(surface, axis=(0, 1))])


class _WorldSurface:
    __slots__ = ('_shape', '_content', '_bounds')

    def __init__(self, surface):
        content = surface.tobytes(order='C')
        current = np.frombuffer(content, dtype=np.float64).reshape(surface.shape)
        bounds = _surface_extrema(current)
        if not np.all(np.isfinite(bounds)):
            raise ValueError('world geometry is nonfinite')
        object.__setattr__(self, '_shape', tuple(surface.shape))
        object.__setattr__(self, '_content', content)
        object.__setattr__(self, '_bounds', bounds.tobytes())

    def __setattr__(self, name, value):
        raise AttributeError('world geometry is immutable')

    def __delattr__(self, name):
        raise AttributeError('world geometry is immutable')

    def views(self):
        return (np.frombuffer(self._content, dtype=np.float64).reshape(self._shape),
                np.frombuffer(self._bounds, dtype=np.float64).reshape(2, 3))

    def matches(self, surface):
        if (type(surface) is not np.ndarray or surface.dtype != np.dtype(np.float64)
                or surface.shape != self._shape or surface.flags.writeable
                or not surface.flags.c_contiguous or surface.nbytes != len(self._content)):
            return False
        owner = surface
        while type(owner) is np.ndarray:
            owner = owner.base
        return owner is self._content

    def frozen(self):
        return self._shape, self._content, self._bounds


def ordinary_model_producer(node):
    """Match the same original producer/consumer family as scene snapshots."""
    consumer = getattr(getattr(node, '_robot_self_collision', None), '__func__', None)
    original = getattr(getattr(node, '_scene_snapshot_body', None), '__func__', None)
    if consumer is None or consumer is not original:
        return False
    owner = getattr(consumer, '__globals__', {}).get('ManipulationNode')
    if owner is None:
        from .pure_geometry_owner import RobotCollisionGeometry, RobotGeometryOwner
        if type(node) is RobotGeometryOwner:
            owner = RobotCollisionGeometry
    if (owner is None or consumer is not getattr(owner, '_robot_self_collision', None)
            or getattr(getattr(node, '_world_collision_surfaces', None), '__func__', None)
                is not getattr(owner, '_world_collision_surfaces', None)
            or getattr(getattr(node, '_collision_link_transforms', None), '__func__', None)
                is not getattr(owner, '_collision_link_transforms', None)):
        return False
    meshes = getattr(node, 'carried_collision_meshes', None)
    return (type(meshes) is tuple and bool(meshes) and all(
        type(mesh) is CollisionMesh and type(mesh.link) is str
        and type(mesh.watertight) is bool and type(mesh.local_mesh) is ModelLocalMesh
        and mesh.local_mesh.matches(mesh.triangles) for mesh in meshes))


class ExactWorldGeometry:
    """LRU of exact facets/bounds; retained local owners count toward the cap."""

    def __init__(self, *, maximum_entries=64, maximum_bytes=64*1024*1024):
        if (type(maximum_entries) is not int or maximum_entries <= 0
                or type(maximum_bytes) is not int or maximum_bytes <= 0):
            raise ValueError('world geometry capacities must be positive integers')
        self.maximum_entries, self.maximum_bytes = maximum_entries, maximum_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self.hits = self.misses = self.fallbacks = 0

    def capture(self, mesh, surface, transform):
        """Return immutable geometry or None before the original fallback."""
        if (type(mesh) is not ModelLocalMesh or not mesh.matches(surface)
                or type(transform) is not np.ndarray
                or transform.dtype != np.dtype(np.float64) or transform.shape != (4, 4)
                or not transform.flags.c_contiguous or not transform.flags.owndata
                or transform.base is not None):
            self.fallbacks += 1
            return None
        data = transform.tobytes(order='C')
        current = np.frombuffer(data, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(current)):
            self.fallbacks += 1
            return None
        weight = mesh.nbytes + surface.nbytes + 48 + len(data)
        if weight > self.maximum_bytes:
            self.fallbacks += 1
            return None
        key = (mesh, data)
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return existing[0]
        local = mesh.snapshot()[0]
        # Exactly the expression used by PlaceSceneChecker before this cache.
        with np.errstate(over='ignore', invalid='ignore'):
            world = local @ current[:3, :3].T + current[:3, 3]
        try:
            value = _WorldSurface(world)
        except ValueError:
            # Nonfinite transformed extrema cannot enter the immutable cache.
            self.fallbacks += 1
            return None
        while self._entries and (len(self._entries) >= self.maximum_entries
                                 or self._bytes + weight > self.maximum_bytes):
            _, (_, old_weight) = self._entries.popitem(last=False)
            self._bytes -= old_weight
        self._entries[key] = (value, weight)
        self._bytes += weight
        self.misses += 1
        return value
