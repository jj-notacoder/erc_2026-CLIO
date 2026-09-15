"""Exact R43 predicate AST fixture; raw original behavior for differential tests."""
from __future__ import annotations
import numpy as np

def oriented_box_from_corners(
    corners: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, column axes, and half extents for ordered box corners."""
    points = np.asarray(corners, dtype=float)
    if points.shape != (8, 3):
        raise ValueError('oriented box must contain eight 3-D corners')
    edge_indices = (4, 2, 1)
    edges = np.asarray([points[index] - points[0] for index in edge_indices])
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('oriented box edges must have positive length')
    axes = (edges / lengths[:, None]).T
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-7):
        raise ValueError('oriented box edges must be orthogonal')
    return np.mean(points, axis=0), axes, 0.5 * lengths

def oriented_box_intersects_triangles(
    corners: Sequence[Sequence[float]],
    triangles: Sequence[Sequence[Sequence[float]]],
    tolerance: float = 1e-9,
    *,
    closed_surface: bool = False,
) -> bool:
    """Return whether an oriented box intersects any triangle via SAT tests."""
    center, axes, half_extents = oriented_box_from_corners(corners)
    surface = np.asarray(triangles, dtype=float)
    if surface.size == 0:
        return False
    if surface.ndim != 3 or surface.shape[1:] != (3, 3):
        raise ValueError('collision surface must have shape (N, 3, 3)')

    # Transform each triangle into the box frame, making the tested box AABB.
    vertices = (surface - center) @ axes
    separated = np.any(
        (np.max(vertices, axis=1) < -half_extents - tolerance)
        | (np.min(vertices, axis=1) > half_extents + tolerance),
        axis=1,
    )
    # A facet rejected on a box axis cannot intersect the box. Keep only the
    # remaining facets for the more expensive SAT axes, preserving their order
    # and the exact comparisons above. Retain the complete ``surface`` for the
    # closed-mesh containment test below: a contained box may touch no facets.
    vertices = vertices[~separated]
    separated = np.zeros(len(vertices), dtype=bool)

    edge_a = vertices[:, 1] - vertices[:, 0]
    edge_b = vertices[:, 2] - vertices[:, 1]
    edge_c = vertices[:, 0] - vertices[:, 2]
    normals = np.cross(edge_a, -edge_c)
    plane_distance = np.abs(np.einsum('ij,ij->i', normals, vertices[:, 0]))
    plane_radius = np.abs(normals) @ half_extents
    separated |= plane_distance > plane_radius + tolerance

    basis = np.eye(3, dtype=float)
    for edge in (edge_a, edge_b, edge_c):
        for box_axis in basis:
            test_axis = np.cross(edge, box_axis)
            projections = np.einsum('nij,nj->ni', vertices, test_axis)
            radius = np.abs(test_axis) @ half_extents
            separated |= (
                (np.max(projections, axis=1) < -radius - tolerance)
                | (np.min(projections, axis=1) > radius + tolerance)
            )
    if np.any(~separated):
        return True
    if not closed_surface:
        return False

    # Triangle SAT detects surface crossings and meshes contained by the box,
    # but not a box wholly contained by a larger closed mesh.  A ray parity
    # test from the box centre covers that remaining containment case.
    direction = np.asarray([1.0, 0.3713906764, 0.6947465906], dtype=float)
    direction /= np.linalg.norm(direction)
    edge_a = surface[:, 1] - surface[:, 0]
    edge_b = surface[:, 2] - surface[:, 0]
    cross_direction = np.cross(direction, edge_b)
    determinant = np.einsum('ij,ij->i', edge_a, cross_direction)
    usable = np.abs(determinant) > tolerance
    inverse = np.zeros_like(determinant)
    inverse[usable] = 1.0 / determinant[usable]
    offset = center - surface[:, 0]
    barycentric_u = np.einsum('ij,ij->i', offset, cross_direction) * inverse
    cross_offset = np.cross(offset, edge_a)
    barycentric_v = np.einsum('j,ij->i', direction, cross_offset) * inverse
    distance = np.einsum('ij,ij->i', edge_b, cross_offset) * inverse
    hit_distances = np.sort(
        distance[
            usable
            & (barycentric_u >= -tolerance)
            & (barycentric_v >= -tolerance)
            & (barycentric_u + barycentric_v <= 1.0 + tolerance)
            & (distance > tolerance)
        ]
    )
    if not len(hit_distances):
        return False
    distinct_hits = 1 + int(
        np.count_nonzero(np.diff(hit_distances) > max(tolerance, 1e-7))
    )
    return bool(distinct_hits % 2)
