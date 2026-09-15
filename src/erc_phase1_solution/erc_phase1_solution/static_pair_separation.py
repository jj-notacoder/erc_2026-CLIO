"""Sufficient separation of complete current meshes, never cached verdicts.

Hull normals and a previously successful direction only suggest axes. Each use
projects every supplied world vertex again. This can refine conservative legacy
SAT false positives, including degenerate facets; it is not Boolean equivalence
with that numerical fallback. Anything uncertain retains the original predicate.
"""
from collections import OrderedDict

import numpy as np
from scipy.spatial import ConvexHull, QhullError


class StaticPairSeparation:
    """Bounded, node-local direction hints for parked-link mesh pairs."""

    def __init__(self, *, maximum_links=32, maximum_pairs=128, maximum_axes=256):
        if any(type(v) is not int or v <= 0 for v in
               (maximum_links, maximum_pairs, maximum_axes)):
            raise ValueError('separation hint capacities must be positive integers')
        self.maximum_links = maximum_links
        self.maximum_pairs = maximum_pairs
        self.maximum_axes = maximum_axes
        self._link_hints = OrderedDict()
        self._pair_hints = OrderedDict()

    @staticmethod
    def _vertices(surface):
        try:
            vertices = np.asarray(surface, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return None
        if (vertices.ndim != 3 or vertices.shape[1:] != (3, 3)
                or not len(vertices) or not np.all(np.isfinite(vertices))):
            return None
        return vertices.reshape(-1, 3)

    @staticmethod
    def _unit(axis):
        axis = np.asarray(axis, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            return None
        scale = float(np.max(np.abs(axis)))
        if scale == 0.0:
            return None
        scaled = axis / scale
        norm = float(np.sqrt(np.dot(scaled, scaled)))
        if not np.isfinite(norm) or norm == 0.0:
            return None
        direction = scaled / norm
        return direction if np.all(np.isfinite(direction)) else None

    @staticmethod
    def _separates(first, second, direction, operand_scale):
        # The operand scale accounts for cancellation in the dot products;
        # scaling only the small resulting projections would be insufficient.
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            projected_first = first @ direction
            projected_second = second @ direction
            if not all(np.all(np.isfinite(value)) for value in
                       (projected_first, projected_second)):
                return False
            epsilon = np.finfo(np.float64).eps
            # Upper allowance for the rounded normalization of the axis.
            norm_upper = float(np.linalg.norm(direction)) + 8.0 * epsilon
            margin = norm_upper * (100e-6 + 128.0 * epsilon * operand_scale)
            gap = max(float(np.min(projected_second) - np.max(projected_first)),
                      float(np.min(projected_first) - np.max(projected_second)))
            return bool(np.isfinite(gap) and np.isfinite(margin) and gap > margin)

    @staticmethod
    def _remember(cache, key, value, maximum):
        if key in cache:
            del cache[key]
        elif len(cache) >= maximum:
            cache.popitem(last=False)
        cache[key] = value

    def _directions(self, link, vertices):
        cached = self._link_hints.get(link)
        if cached is not None:
            return cached
        try:
            # Exact unique vertices only reduce hull construction input size;
            # the projection certificate still uses every original vertex.
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                hull = ConvexHull(np.unique(vertices, axis=0))
            candidates = np.unique(hull.equations[:, :3], axis=0)
            directions = []
            for candidate in candidates:
                direction = self._unit(candidate)
                if direction is not None:
                    directions.append(direction)
                if len(directions) == self.maximum_axes:
                    break
            owned = np.asarray(directions, dtype=np.float64).reshape(-1, 3).copy()
        except (QhullError, ValueError, OverflowError, FloatingPointError):
            owned = np.empty((0, 3), dtype=np.float64)
        # Empty hints retain the old predicate too; no repeated failing hull.
        owned.setflags(write=False)
        self._remember(self._link_hints, link, owned, self.maximum_links)
        return owned

    def separated(self, first_link, second_link, first_surface, second_surface):
        first = self._vertices(first_surface)
        second = self._vertices(second_surface)
        if first is None or second is None:
            return False
        with np.errstate(over='ignore', invalid='ignore'):
            # L1 norms upper-bound sum(abs(p_i * d_i)) for every unit
            # direction. Compute this stronger operand bound once per pair.
            operand_scale = max(1.0, float(np.max(np.sum(np.abs(first), axis=1))),
                                float(np.max(np.sum(np.abs(second), axis=1))))
        if not np.isfinite(operand_scale):
            return False
        pair = (first_link, second_link)
        previous = self._pair_hints.get(pair)
        if previous is not None and self._separates(first, second, previous, operand_scale):
            return True
        for link, vertices in ((first_link, first), (second_link, second)):
            for direction in self._directions(link, vertices):
                if self._separates(first, second, direction, operand_scale):
                    owned = direction.copy()
                    owned.setflags(write=False)
                    self._remember(self._pair_hints, pair, owned, self.maximum_pairs)
                    return True
        return False


def separated_on_axes(first_surface, second_surface, axes):
    """Sufficient separation of all current vertices along supplied hints.

    Axes may come from the current hand frame, but neither link identity nor
    presumed rigid attachment is a clearance certificate. Every invocation
    projects every supplied world vertex again with the same strict 100 um
    plus floating-point allowance as parked-link separation. Any uncertainty
    falls through to the original detailed mesh intersection predicate.
    """
    first = StaticPairSeparation._vertices(first_surface)
    second = StaticPairSeparation._vertices(second_surface)
    if first is None or second is None:
        return False
    try:
        directions = np.asarray(axes, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return False
    if directions.ndim != 2 or directions.shape[1:] != (3,) or len(directions) > 6:
        return False
    with np.errstate(over='ignore', invalid='ignore'):
        operand_scale = max(1.0, float(np.max(np.sum(np.abs(first), axis=1))),
                            float(np.max(np.sum(np.abs(second), axis=1))))
    if not np.isfinite(operand_scale):
        return False
    for axis in directions:
        direction = StaticPairSeparation._unit(axis)
        if direction is not None and StaticPairSeparation._separates(
                first, second, direction, operand_scale):
            return True
    return False
