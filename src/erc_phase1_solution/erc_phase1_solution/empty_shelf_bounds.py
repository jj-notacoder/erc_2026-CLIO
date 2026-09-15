"""DRAFT registered local empty shelf bounds; no dispatch or source edits.

Shelf lip is anchored to the observed number-marker front surface, using only
unchanged official relative geometry. Vertical floor/roof retain the explicit
upright supported-book assumption because digit centroid Z is not plate center.
The 10 mm normal uncertainty is a disclosed engineering allowance, not measured
covariance. Existing 15 mm margin, 200 mm lateral uncertainty, 10 mm roof
uncertainty, and 5 mm tool allowance all remain. Unknown adjacent objects and
full shelf-mesh collision certification remain outside this local model.
"""
import math
from collections import OrderedDict
import threading
import numpy as np
from erc_phase1_solution.lift_first_extraction import _bay_values
from erc_phase1_solution.rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS
from erc_phase1_solution.kinematics import URDFChain
from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.shelf_cradle_geometry import (
    ShelfCradleGeometry, _ordinary_cradle_state_access,
)

_CHAIN_FORWARD = URDFChain.forward
_CHAIN_TRANSFORMS = URDFChain.link_transforms
_TOOL_MODEL = ShelfCradleGeometry.model_local_surface


class ExactShelfSampleCache:
    """Bounded exact shelf predicates for one node/guard on one planning thread.

    Model-owned immutable facets and freshly computed transform bytes bind a
    query. Mutable/custom producers retain the original evaluator. No sensor
    admission, cancellation, sample grid or edge admission is cached.
    """
    def __init__(self, node, guard, *, maximum_entries=4096):
        if type(maximum_entries) is not int or not 0 < maximum_entries <= 8192:
            raise ValueError('empty_shelf_sample_cache_capacity')
        self.node, self.guard = node, guard
        self.thread = threading.get_ident()
        self.maximum_entries = maximum_entries
        self.entries = OrderedDict()
        self.context = None
        self.generation = 0
        self.hits = self.misses = self.fallbacks = 0

    def _identity(self, bounds, q, allow_entry):
        node, guard = bounds.node, bounds.guard
        if (type(bounds) is not EmptyShelfBounds
                or node is not self.node or guard is not self.guard
                or type(allow_entry) is not bool
                or type(q) is not np.ndarray or q.dtype != np.dtype(np.float64)
                or q.shape != (8,) or not np.isfinite(q).all()
                or not _ordinary_cradle_state_access(node)
                or type(guard.geometry) is not ShelfCradleGeometry
                or getattr(guard.geometry.model_local_surface, '__func__', None) is not _TOOL_MODEL
                or getattr(bounds._sample_uncached, '__func__', None) is not _SHELF_UNCACHED
                or getattr(bounds._record_metrics, '__func__', None) is not _SHELF_RECORD
                or _forward_vertices is not _SHELF_CLIP):
            return None
        chains = tuple(getattr(node, name, None) for name in ('chain', 'right_chain', 'head_chain'))
        if any(type(chain) is not URDFChain
               or getattr(chain.forward, '__func__', None) is not _CHAIN_FORWARD
               or getattr(chain.link_transforms, '__func__', None) is not _CHAIN_TRANSFORMS
               for chain in chains):
            return None
        if set(guard.context) != {'right_positions', 'head_positions'}:
            return None
        context = []
        for name, count in (('right_positions', 7), ('head_positions', 2)):
            value = np.asarray(guard.context[name], dtype=float)
            if value.shape != (count,) or not np.isfinite(value).all():
                return None
            context.append(value.tobytes())
        tool = []
        for name, surface in bounds.local.items():
            model = guard.geometry.model_local_surface(surface)
            if type(model) is not ModelLocalMesh or not model.matches(surface):
                return None
            tool.append((name, model))
        vectors = []
        for name in ('marker', 'left', 'inward', 'plane_point', 'registered_lip'):
            value = getattr(bounds, name)
            if (type(value) is not np.ndarray or value.dtype != np.dtype(np.float64)
                    or value.shape != (3,) or not np.isfinite(value).all()):
                return None
            vectors.append(value.tobytes())
        scalar_values = [bounds.side_min, bounds.side_max, bounds.margin,
            bounds.normal_uncertainty_m, bounds.floor, bounds.roof,
            guard.open_aperture]
        if any(type(value) not in (float, np.float64) or not math.isfinite(value)
               for value in scalar_values):
            return None
        scalars = np.asarray(scalar_values, dtype=float).tobytes()
        meshes = tuple((mesh.link, mesh.local_mesh) for mesh in node.carried_collision_meshes)
        identity = (chains, guard.geometry, meshes, tuple(tool), tuple(context), tuple(vectors), scalars,
            getattr(node._world_collision_surfaces, '__func__', None),
            getattr(node._collision_link_transforms, '__func__', None))
        # Current FK remains live even on a hit: in-place joint-axis/origin or
        # chain-index changes cannot inherit a result for an old transform.
        hand = node.chain.forward(q)
        transforms = node._collision_link_transforms(q, **guard.context)
        matrices = (hand, *(transforms[name] for name, _ in meshes))
        if any(type(value) is not np.ndarray or value.dtype != np.dtype(np.float64)
               or value.shape != (4, 4) or not np.isfinite(value).all() for value in matrices):
            return None
        query = (q.tobytes(), allow_entry, tuple(value.tobytes() for value in matrices))
        return identity, query

    def key(self, bounds, q, allow_entry):
        if threading.get_ident() != self.thread:
            return None
        identity = self._identity(bounds, q, allow_entry)
        if identity is None:
            self.entries.clear()
            self.context = None
            self.generation += 1
            self.fallbacks += 1
            return None
        context, key = identity
        if context != self.context:
            self.entries.clear()
            self.context = context
            self.generation += 1
        return self.generation, key

    def lookup(self, key):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
            self.hits += 1
        else:
            self.misses += 1
        return value

    def remember(self, key, result):
        if key not in self.entries and len(self.entries) >= self.maximum_entries:
            self.entries.popitem(last=False)
        self.entries[key] = result

def _cancel(node):
    if node._cancel.is_set():
        raise RuntimeError('empty_shelf_cancelled')

def _forward_vertices(triangles, plane_point, inward, margin):
    """Vertices of triangles clipped to the occupied side of a shelf plane.

    Include edge/plane intersections, since checking only original vertices
    would miss a side or roof violation at a triangle's crossing point.
    """
    facets = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    distances = (facets-plane_point) @ inward + margin
    active = np.any(distances >= 0, axis=1)
    facets, distances = facets[active], distances[active]
    if not len(facets):
        return np.empty((0, 3))
    result = [facets[distances >= 0]]
    for index in range(3):
        other = (index+1) % 3
        first, last = distances[:, index], distances[:, other]
        crosses = first*last < 0
        fractions = first[crosses]/(first[crosses]-last[crosses])
        result.append(facets[crosses, index] + fractions[:, None]*(
            facets[crosses, other]-facets[crosses, index]))
    return np.concatenate(result, axis=0)



class EmptyShelfBounds:
    """Supplement robot/book checks with the measured local shelf opening.

    Shelf floor follows the same explicit upright supported-book assumption
    as lift-first planning.  This is a bounded local model, not a certificate
    for all unobserved shelf geometry or neighboring books.
    """
    def __init__(self, node, front, grasp, guard, bay, *, normal_uncertainty_m=.010,
                 sample_cache=None):
        self.node, self.guard = node, guard
        # grasp is accepted only for compatibility with the bottom draft;
        # None is valid because shelf geometry never depends on a grasp pose.
        dimensions = np.asarray(node.carried_book_dimensions, dtype=float)
        if dimensions.shape != (3,) or not np.allclose(dimensions, [.16,.02,.25], rtol=0., atol=1e-12):
            raise ValueError('empty_shelf_requires_official_book_dimensions')
        self.marker, self.left, self.side_min, self.side_max = _bay_values(bay)
        self.inward = np.asarray(bay.inward_axis_base, dtype=float)
        # Official shelf STL front is .244984 m outward from its origin;
        # marker origin is .245 m outward, with a .001 m visible face.
        # Register only their relative geometry, never the simulator world pose.
        self.plane_point = self.marker + .001016*self.inward
        self.normal_uncertainty_m = float(normal_uncertainty_m)
        if not np.isfinite(self.normal_uncertainty_m) or not .010 <= self.normal_uncertainty_m <= .050:
            raise ValueError('empty_shelf_normal_uncertainty_outside_bound')
        front = np.asarray(front, dtype=float)
        if front.shape != (3,) or not np.all(np.isfinite(front)):
            raise ValueError('empty_shelf_front_shape')
        # Marker glyph vertical position is not the card center. Preserve the
        # explicitly supported-book floor reference, not a fabricated row Z.
        if bay.assume_upright_supported is not True:
            raise ValueError('empty_shelf_floor_requires_supported_book')
        self.registered_lip = self.plane_point.copy()
        book_depth = float((front-self.registered_lip) @ self.inward)
        if not .015 <= book_depth <= .150:
            raise ValueError('empty_shelf_marker_book_depth_inconsistent')
        # Preserve the earlier nominal book-derived exclusion as an additional
        # conservative bound; it cannot authorize motion past the marker plane.
        old_plane = front - .065*self.inward
        earlier_shift = min(0., float((old_plane-self.plane_point) @ self.inward))
        self.plane_point += earlier_shift*self.inward
        self.book_depth_m = book_depth
        self.margin = float(bay.margin_m)
        self.floor = float(front[2])-.5*float(node.carried_book_dimensions[2])+self.margin
        self.roof = float(front[2])-.5*float(node.carried_book_dimensions[2])+.300-bay.roof_uncertainty_m-self.margin
        geometry = guard.geometry
        self.local = geometry.local_surfaces(guard.open_aperture)
        if set(self.local) != set(LEFT_GRIPPER_COLLISION_LINKS):
            raise ValueError('empty_tool_inventory')
        self.samples = 0
        self.minimum_floor = self.minimum_roof = self.minimum_side = self.minimum_back = math.inf
        if sample_cache is not None and type(sample_cache) is not ExactShelfSampleCache:
            raise ValueError('empty_shelf_sample_cache_type')
        self.sample_cache = sample_cache

    def sample(self, q, allow_entry):
        _cancel(self.node)
        cache = self.sample_cache
        key = None if cache is None else cache.key(self, q, allow_entry)
        result = None if key is None else cache.lookup(key)
        # FK identity capture is additional work, never a cancellation bypass.
        _cancel(self.node)
        if result is not None:
            reason, metrics = result
            self.samples += 1
            for values in metrics:
                self._record_metrics(values)
            return reason
        reason, metrics = self._sample_uncached(q, allow_entry)
        # Never retain a result whose producer changed during its evaluation.
        # Ordinary immutable producers are pure; a mutation makes this a miss.
        if (key is not None and not self.node._cancel.is_set()
                and cache.key(self, q, allow_entry) == key):
            cache.remember(key, (reason, tuple(metrics)))
        return reason

    def _record_metrics(self, values):
        back, floor, roof, side = values
        self.minimum_back = min(self.minimum_back, back)
        self.minimum_floor = min(self.minimum_floor, floor)
        self.minimum_roof = min(self.minimum_roof, roof)
        self.minimum_side = min(self.minimum_side, side)

    def _sample_uncached(self, q, allow_entry):
        metrics = []
        transform = self.node.chain.forward(q)
        surfaces = [(name, vertices, 0.) for name, vertices in
            self.node._world_collision_surfaces(q, **self.guard.context).items()]
        surfaces.extend((name, triangles @ transform[:3, :3].T+transform[:3, 3], .005)
                        for name, triangles in self.local.items())
        self.samples += 1
        for name, triangles, allowance in surfaces:
            triangles = np.asarray(triangles, dtype=float)
            if not np.all(np.isfinite(triangles)):
                return 'empty_shelf_nonfinite_surface:'+name, metrics
            occupied = _forward_vertices(triangles, self.plane_point,
                                          self.inward, self.margin+self.normal_uncertainty_m+allowance)
            if not len(occupied):
                continue
            if not allow_entry:
                return 'empty_setup_enters_shelf:'+name, metrics
            lateral = (occupied-self.marker) @ self.left
            floor = float(occupied[:, 2].min()-self.floor-allowance)
            roof = float(self.roof-occupied[:, 2].max()-allowance)
            side = float(min(lateral.min()-self.side_min-allowance,
                             self.side_max-lateral.max()-allowance))
            # Official clear depth: front STL z=.244984 to inner back z=-.055016.
            depth = (occupied-self.registered_lip) @ self.inward
            back = float(.300-self.margin-self.normal_uncertainty_m-allowance-depth.max())
            values = (back, floor, roof, side)
            metrics.append(values)
            self._record_metrics(values)
            if min(floor, roof, side, back) < 0:
                return ('empty_bay_clearance:'+name
                        +f':floor={floor}:roof={roof}:side={side}:back={back}'), metrics
        return None, metrics

    def edge(self, start, end, *, allow_entry):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if start.shape != (8,) or end.shape != (8,) or not np.all(np.isfinite([start, end])):
            raise ValueError('empty_shelf_edge_shape')
        count = max(61, int(self.node.carried_transition_samples),
            int(math.ceil(float(np.max(np.abs(end-start)))/.02))+1)
        for fraction in np.linspace(0., 1., count):
            reason = self.sample(start+(end-start)*fraction, allow_entry)
            if reason:
                return reason
        return None


_ShelfEntryBounds = EmptyShelfBounds
_SHELF_UNCACHED = EmptyShelfBounds._sample_uncached
_SHELF_RECORD = EmptyShelfBounds._record_metrics
_SHELF_CLIP = _forward_vertices
