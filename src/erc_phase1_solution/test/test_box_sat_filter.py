"""Differential coverage for pruning already-separated box/triangle facets."""

from __future__ import annotations

import numpy as np
import pytest

from erc_phase1_solution.kinematics import (
    oriented_box_from_corners,
    oriented_box_intersects_triangles,
)


def unfiltered_reference(corners, triangles, tolerance=1e-9, *, closed_surface=False):
    """Frozen pre-filter algorithm, including complete-mesh ray containment."""
    center, axes, half_extents = oriented_box_from_corners(corners)
    surface = np.asarray(triangles, dtype=float)
    if surface.size == 0:
        return False
    if surface.ndim != 3 or surface.shape[1:] != (3, 3):
        raise ValueError('collision surface must have shape (N, 3, 3)')
    vertices = (surface - center) @ axes
    separated = np.any(
        (np.max(vertices, axis=1) < -half_extents - tolerance)
        | (np.min(vertices, axis=1) > half_extents + tolerance), axis=1,
    )
    edge_a = vertices[:, 1] - vertices[:, 0]
    edge_b = vertices[:, 2] - vertices[:, 1]
    edge_c = vertices[:, 0] - vertices[:, 2]
    normals = np.cross(edge_a, -edge_c)
    plane_distance = np.abs(np.einsum('ij,ij->i', normals, vertices[:, 0]))
    plane_radius = np.abs(normals) @ half_extents
    separated |= plane_distance > plane_radius + tolerance
    for edge in (edge_a, edge_b, edge_c):
        for box_axis in np.eye(3, dtype=float):
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
    hit_distances = np.sort(distance[
        usable & (barycentric_u >= -tolerance) & (barycentric_v >= -tolerance)
        & (barycentric_u + barycentric_v <= 1.0 + tolerance)
        & (distance > tolerance)
    ])
    if not len(hit_distances):
        return False
    distinct_hits = 1 + int(np.count_nonzero(
        np.diff(hit_distances) > max(tolerance, 1e-7)
    ))
    return bool(distinct_hits % 2)


SIGNS = np.asarray([(x, y, z) for x in (-1., 1.)
                    for y in (-1., 1.) for z in (-1., 1.)])
BOX_FACES = np.asarray([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
])


def box_corners(center=(0., 0., 0.), half=(1., 1., 1.), rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    return SIGNS * np.asarray(half) @ rotation.T + center


def rotation_from_rng(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q[:, 0] *= np.linalg.det(q)
    return q


@pytest.mark.parametrize('closed_surface,expected', [(False, False), (True, True)])
def test_contained_box_keeps_all_facets_for_ray_parity(closed_surface, expected):
    # Every triangle is separated on a box axis, but the closed shell contains
    # the entire box. Filtering the containment input would incorrectly pass.
    corners = box_corners(half=(.1, .2, .3))
    shell = box_corners(half=(2., 2., 2.))[BOX_FACES]
    assert unfiltered_reference(corners, shell, closed_surface=closed_surface) == expected
    assert oriented_box_intersects_triangles(
        corners, shell, closed_surface=closed_surface
    ) == expected


@pytest.mark.parametrize('offset,expected', [(0., True), (5e-10, True), (2e-9, False)])
def test_contact_on_box_face_preserves_tolerance(offset, expected):
    triangle = np.asarray([[[1. + offset, -.5, -.5],
                            [1. + offset, .5, -.5], [1. + offset, 0., .5]]])
    corners = box_corners()
    assert unfiltered_reference(corners, triangle) == expected
    assert oriented_box_intersects_triangles(corners, triangle) == expected


def test_mixed_disjoint_and_crossing_facets_preserves_intersection():
    disjoint = box_corners(center=(10., 0., 0.))[BOX_FACES]
    crossing = np.asarray([[[-2., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
    for surface in (np.concatenate((disjoint, crossing)),
                    np.concatenate((crossing, disjoint))):
        assert oriented_box_intersects_triangles(box_corners(), surface)


@pytest.mark.parametrize('seed', range(8))
def test_random_rotations_surfaces_and_degenerate_facets_match_reference(seed):
    rng = np.random.default_rng(81931 + seed)
    for index in range(80):
        corners = box_corners(rng.uniform(-2., 2., 3),
                              rng.uniform(.01, .7, 3), rotation_from_rng(rng))
        surface = rng.uniform(-3., 3., (1 + index % 37, 3, 3))
        if index % 3 == 0:
            # Include point/line degeneracies, not just generic triangles.
            surface[0, 1] = surface[0, 0]
        if index % 5 == 0:
            surface[0, 2] = surface[0, 0]
        for tolerance in (0., 1e-9, 1e-6):
            expected = unfiltered_reference(corners, surface, tolerance)
            assert oriented_box_intersects_triangles(corners, surface, tolerance) == expected


@pytest.mark.parametrize('seed', range(4))
def test_random_closed_shells_match_reference(seed):
    rng = np.random.default_rng(50182 + seed)
    for _ in range(100):
        rotation = rotation_from_rng(rng)
        center = rng.uniform(-1., 1., 3)
        half = rng.uniform(.1, 1.5, 3)
        shell = box_corners(center, half, rotation)[BOX_FACES]
        corners = box_corners(center + rng.uniform(-1.5, 1.5, 3),
                              rng.uniform(.01, .8, 3), rotation_from_rng(rng))
        expected = unfiltered_reference(corners, shell, closed_surface=True)
        assert oriented_box_intersects_triangles(corners, shell, closed_surface=True) == expected


def test_empty_surface_and_invalid_shape_behavior_unchanged():
    assert not oriented_box_intersects_triangles(box_corners(), [])
    with pytest.raises(ValueError, match='collision surface must have shape'):
        oriented_box_intersects_triangles(box_corners(), np.zeros((2, 3)))
