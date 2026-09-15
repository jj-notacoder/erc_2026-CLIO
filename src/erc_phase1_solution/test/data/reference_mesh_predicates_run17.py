"""Frozen Run17 mesh predicates for exact differential coverage.

Only public function names were prefixed; component and ray helpers are
unchanged production implementations. Source SHA256: cf4007094c517a4db241d2b48759cffe037d74eab88b06d50c15ccab73f01854
"""
from typing import Sequence
import numpy as np
from erc_phase1_solution.kinematics import _mesh_components, _watertight_components_contain_component

def reference_triangles_intersect(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    tolerance: float = 1e-9,
) -> bool:
    """Return whether two closed 3-D triangles touch or intersect.

    The separating-axis set contains both face normals, every edge cross
    product, and in-plane edge normals.  The latter are required when the
    triangles are coplanar, where the ordinary 3-D edge-cross axes collapse
    onto the shared face normal.
    """
    triangle_a = np.asarray(first, dtype=float)
    triangle_b = np.asarray(second, dtype=float)
    if triangle_a.shape != (3, 3) or triangle_b.shape != (3, 3):
        raise ValueError('triangles must each have shape (3, 3)')
    if tolerance < 0.0:
        raise ValueError('tolerance cannot be negative')
    if not np.all(np.isfinite(triangle_a)) or not np.all(np.isfinite(triangle_b)):
        raise ValueError('triangle coordinates must be finite')

    edges_a = np.roll(triangle_a, -1, axis=0) - triangle_a
    edges_b = np.roll(triangle_b, -1, axis=0) - triangle_b
    normal_a = np.cross(edges_a[0], edges_a[1])
    normal_b = np.cross(edges_b[0], edges_b[1])
    # Coplanar triangles also need the three 2-D edge-normal axes.  Include
    # both triangles' axes so the test remains symmetric for degenerate input.
    # Batch only the independent cross products. Keep axis order, float64
    # arithmetic, scalar normalization and projection comparisons unchanged,
    # including the early separating axis and degenerate-axis threshold.
    axes = np.concatenate((
        np.asarray([normal_a, normal_b]),
        np.cross(edges_a[:, None, :], edges_b[None, :, :]).reshape(9, 3),
        np.cross(normal_a, edges_a),
        np.cross(normal_b, edges_b),
    ))

    usable_axis = False
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            continue
        usable_axis = True
        direction = axis / norm
        projection_a = triangle_a @ direction
        projection_b = triangle_b @ direction
        if (
            float(np.max(projection_a)) < float(np.min(projection_b)) - tolerance
            or float(np.max(projection_b))
            < float(np.min(projection_a)) - tolerance
        ):
            return False

    # Collision meshes should not contain zero-area facets.  Treat two fully
    # degenerate triangles as intersecting only when their bounds overlap;
    # this is conservative for a safety guard.
    if not usable_axis:
        return bool(
            np.all(
                np.max(triangle_a, axis=0) + tolerance
                >= np.min(triangle_b, axis=0)
            )
            and np.all(
                np.max(triangle_b, axis=0) + tolerance
                >= np.min(triangle_a, axis=0)
            )
        )
    return True

def reference_triangle_meshes_intersect(
    first: Sequence[Sequence[Sequence[float]]],
    second: Sequence[Sequence[Sequence[float]]],
    tolerance: float = 1e-9,
    *,
    first_watertight: bool = False,
    second_watertight: bool = False,
) -> bool:
    """Return whether two triangle meshes intersect or one contains the other.

    Mesh and per-facet axis-aligned bounds provide a cheap broad phase before
    the exact triangle SAT test.  Set the corresponding ``*_watertight`` flag
    only for a verified closed surface to enable containment detection.  This
    function deliberately treats touching surfaces as a collision.
    """
    mesh_a = np.asarray(first, dtype=float)
    mesh_b = np.asarray(second, dtype=float)
    for mesh in (mesh_a, mesh_b):
        if mesh.size == 0:
            continue
        if mesh.ndim != 3 or mesh.shape[1:] != (3, 3):
            raise ValueError('collision surface must have shape (N, 3, 3)')
        if not np.all(np.isfinite(mesh)):
            raise ValueError('collision surface coordinates must be finite')
    if tolerance < 0.0:
        raise ValueError('tolerance cannot be negative')
    if mesh_a.size == 0 or mesh_b.size == 0:
        return False

    bounds_a = np.asarray(
        [np.min(mesh_a, axis=(0, 1)), np.max(mesh_a, axis=(0, 1))]
    )
    bounds_b = np.asarray(
        [np.min(mesh_b, axis=(0, 1)), np.max(mesh_b, axis=(0, 1))]
    )
    if np.any(bounds_a[1] < bounds_b[0] - tolerance) or np.any(
        bounds_b[1] < bounds_a[0] - tolerance
    ):
        return False

    # Remove facets whose AABB is disjoint from the other *whole* mesh before
    # choosing the smaller loop side.  Such a facet cannot participate in a
    # triangle intersection.  Keep the complete meshes below for watertight
    # containment: slicing them there could change connected components and
    # their representative points, which would change collision semantics.
    minimum_a = np.min(mesh_a, axis=1)
    maximum_a = np.max(mesh_a, axis=1)
    minimum_b = np.min(mesh_b, axis=1)
    maximum_b = np.max(mesh_b, axis=1)
    candidates_a = (
        np.all(maximum_a + tolerance >= bounds_b[0], axis=1)
        & np.all(bounds_b[1] + tolerance >= minimum_a, axis=1)
    )
    candidates_b = (
        np.all(maximum_b + tolerance >= bounds_a[0], axis=1)
        & np.all(bounds_a[1] + tolerance >= minimum_b, axis=1)
    )
    narrow_a = mesh_a[candidates_a]
    narrow_minimum_a = minimum_a[candidates_a]
    narrow_maximum_a = maximum_a[candidates_a]
    narrow_b = mesh_b[candidates_b]
    narrow_minimum_b = minimum_b[candidates_b]
    narrow_maximum_b = maximum_b[candidates_b]

    # Preserve the original full-mesh length decision, including triangle
    # argument order, then loop only its filtered facets.  The unchanged
    # triangle SAT remains the narrow phase and still treats touching as
    # collision.
    if len(mesh_a) > len(mesh_b):
        narrow_a, narrow_b = narrow_b, narrow_a
        narrow_minimum_a, narrow_minimum_b = (
            narrow_minimum_b,
            narrow_minimum_a,
        )
        narrow_maximum_a, narrow_maximum_b = (
            narrow_maximum_b,
            narrow_maximum_a,
        )
    for triangle_a, triangle_minimum_a, triangle_maximum_a in zip(
        narrow_a,
        narrow_minimum_a,
        narrow_maximum_a,
        strict=True,
    ):
        candidates = np.flatnonzero(
            np.all(
                triangle_maximum_a + tolerance >= narrow_minimum_b,
                axis=1,
            )
            & np.all(
                narrow_maximum_b + tolerance >= triangle_minimum_a,
                axis=1,
            )
        )
        for index in candidates:
            if reference_triangles_intersect(triangle_a, narrow_b[index], tolerance):
                return True
    if not first_watertight and not second_watertight:
        return False
    components_a = _mesh_components(mesh_a)
    components_b = _mesh_components(mesh_b)
    return bool(
        first_watertight
        and _watertight_components_contain_component(
            components_a,
            components_b,
            tolerance,
        )
        or second_watertight
        and _watertight_components_contain_component(
            components_b,
            components_a,
            tolerance,
        )
    )
