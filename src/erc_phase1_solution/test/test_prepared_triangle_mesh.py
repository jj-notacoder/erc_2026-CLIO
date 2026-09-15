"""Exact world-coordinate mesh reuse and lazy SAT, compared with frozen Run17."""

import numpy as np
import pytest

from erc_phase1_solution import kinematics as k
from data.reference_mesh_predicates_run17 import (
    reference_triangle_meshes_intersect, reference_triangles_intersect,
)


SIGNS = np.asarray([(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
FACES = np.asarray([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
])


def box(center=(0., 0., 0.), half=(1., 1., 1.)):
    return (SIGNS*np.asarray(half)+center)[FACES]


def assert_mesh_equivalent(a, b, tolerance=1e-9, *, closed=(False, False)):
    prepared_a, prepared_b = k.PreparedTriangleMesh(a), k.PreparedTriangleMesh(b)
    for first, second, flags in ((a, b, closed), (b, a, closed[::-1])):
        kwargs = dict(first_watertight=flags[0], second_watertight=flags[1])
        expected = reference_triangle_meshes_intersect(first, second, tolerance, **kwargs)
        p, q = (prepared_a, prepared_b) if first is a else (prepared_b, prepared_a)
        for left, right in ((first, second), (p, second), (first, q), (p, q)):
            assert k.triangle_meshes_intersect(left, right, tolerance, **kwargs) == expected


def test_prepared_surface_owns_immutable_exact_coordinates_and_lazy_data():
    source = box()
    source[0, 0, 0] = -0.
    prepared = k.PreparedTriangleMesh(source)
    original_bytes = source.tobytes()
    source[:] = 50.
    assert prepared.surface.tobytes() == original_bytes
    assert prepared._bounds is prepared._facet_bounds is prepared._components is None
    arrays = (prepared.surface, prepared.bounds, *prepared.facet_bounds, *prepared.components)
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert prepared.bounds is prepared.bounds
    assert prepared.facet_bounds is prepared.facet_bounds
    assert prepared.components is prepared.components


def test_preparation_does_not_round_nearby_world_coordinates():
    first = box()
    second = first.copy()
    second[0, 0, 0] = np.nextafter(first[0, 0, 0], np.inf)
    left, right = k.PreparedTriangleMesh(first), k.PreparedTriangleMesh(second)
    assert left.surface.tobytes() != right.surface.tobytes()
    np.testing.assert_array_equal(left.surface, first)
    np.testing.assert_array_equal(right.surface, second)


def test_whole_bounds_early_exit_does_not_prepare_facets_or_components():
    left, right = k.PreparedTriangleMesh(box()), k.PreparedTriangleMesh(box(center=(10., 0., 0.)))
    assert not k.triangle_meshes_intersect(left, right, first_watertight=True, second_watertight=True)
    assert left._facet_bounds is right._facet_bounds is None
    assert left._components is right._components is None


def test_same_exact_components_are_prepared_once_across_pairs(monkeypatch):
    outer, inner = k.PreparedTriangleMesh(box()), k.PreparedTriangleMesh(box(half=(.1, .1, .1)))
    original, calls = k._mesh_components, []
    def counted(mesh):
        calls.append(mesh)
        return original(mesh)
    monkeypatch.setattr(k, '_mesh_components', counted)
    for _ in range(4):
        assert k.triangle_meshes_intersect(outer, inner, first_watertight=True)
        assert k.triangle_meshes_intersect(inner, outer, second_watertight=True)
    assert len(calls) == 2


def test_component_order_and_facet_bytes_follow_first_source_triangle():
    first, second = box(center=(8., 0., 0.)), box()
    mesh = np.stack([facet for pair in zip(first, second) for facet in pair])
    groups = k.PreparedTriangleMesh(mesh).components
    assert len(groups) == 2
    assert groups[0].tobytes() == first.tobytes()
    assert groups[1].tobytes() == second.tobytes()


def test_vertex_connectivity_is_exact_even_at_one_ulp_separation():
    mesh = np.asarray([
        [[1., 0., 0.], [2., 0., 0.], [1., 1., 0.]],
        [[np.nextafter(1., 2.), 0., 0.], [4., 0., 0.], [4., 1., 0.]],
    ])
    assert len(k.PreparedTriangleMesh(mesh).components) == 2
    mesh[1, 0, 0] = 1.
    groups = k.PreparedTriangleMesh(mesh).components
    assert len(groups) == 1 and groups[0].tobytes() == mesh.tobytes()


def test_transitive_vertex_connectivity_preserves_degenerate_facet_order():
    mesh = np.asarray([
        [[0., -0., 0.], [1., 0., 0.], [1., 0., 0.]],
        [[2., 0., 0.], [2., 0., 0.], [3., 0., 0.]],
        [[1., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
        [[9., 9., 9.], [9., 9., 9.], [9., 9., 9.]],
    ])
    groups = k.PreparedTriangleMesh(mesh).components
    assert len(groups) == 2
    assert groups[0].tobytes() == mesh[:3].tobytes()
    assert groups[1].tobytes() == mesh[3:].tobytes()


@pytest.mark.parametrize('tolerance', [0., 1e-9, 1e-6])
@pytest.mark.parametrize('closed', [(False, False), (True, False), (False, True), (True, True)])
def test_prepared_containment_disconnected_and_overlapping_shells(tolerance, closed):
    outer = box()
    inside = box(half=(.1, .2, .3))
    disconnected = np.concatenate((box(center=(4., 0., 0.), half=(.1, .1, .1)), inside))
    overlapping = np.concatenate((box(center=(-.25, 0., 0.)), box(center=(.25, 0., 0.))))
    for a, b in ((outer, inside), (outer, disconnected), (overlapping, inside)):
        assert_mesh_equivalent(a, b, tolerance, closed=closed)


@pytest.mark.parametrize('seed', range(5))
def test_random_world_surfaces_and_rotations_match_frozen_predicates(seed):
    rng = np.random.default_rng(170318+seed)
    for index in range(24):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        translation = rng.uniform(-2., 2., 3)
        first = rng.normal(size=(1+index%5, 3, 3)) @ rotation.T + translation
        second = rng.normal(size=(1+index%7, 3, 3)) @ rotation.T + translation
        if index%3 == 0:
            first[0, 1:] = first[0, 0]  # Preserve degenerate fallback too.
        if index%4 == 0:
            second[:, :, 2] = first[0, 0, 2]
        assert_mesh_equivalent(first, second, (0., 1e-9, 1e-6)[index%3])


@pytest.mark.parametrize('degenerate', [False, True])
def test_exact_touch_tolerance_and_adjacent_float_values(degenerate):
    triangle = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    if degenerate:
        triangle[:] = 0.
    tolerance = 1e-9
    for distance in (0., np.nextafter(tolerance, 0.), tolerance, np.nextafter(tolerance, np.inf)):
        shifted = triangle+[0., 0., distance]
        expected = reference_triangles_intersect(triangle, shifted, tolerance)
        assert k.triangles_intersect(triangle, shifted, tolerance) == expected
        assert_mesh_equivalent(triangle[None], shifted[None], tolerance)


@pytest.mark.parametrize('seed', range(5))
def test_lazy_sat_matches_eager_axis_order_with_random_coplanar_and_degenerate_pairs(seed):
    rng = np.random.default_rng(91900+seed)
    for index in range(160):
        first, second = rng.normal(size=(2, 3, 3))
        if index%4 == 0:
            first[:, 2] = second[:, 2] = 0.
        if index%7 == 0:
            first[1] = first[0]
        if index%11 == 0:
            second[:] = second[0]
        tolerance = (0., 1e-9, 1e-6)[index%3]
        for a, b in ((first, second), (second, first)):
            assert k.triangles_intersect(a, b, tolerance) == reference_triangles_intersect(a, b, tolerance)


def test_lazy_sat_avoids_later_cross_products_only_after_a_valid_separating_axis(monkeypatch):
    first = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    original, calls = k.np.cross, []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(k.np, 'cross', counted)
    assert not k.triangles_intersect(first, first+[0., 0., .1])
    assert len(calls) == 2  # Both face normals, no unused edge/coplanar groups.
    calls.clear()
    assert k.triangles_intersect(first, first)
    assert len(calls) == 5  # Every group is still checked for touching facets.


@pytest.mark.parametrize('bad', [np.zeros((2, 3)), np.full((1, 3, 3), np.nan), np.full((1, 3, 3), np.inf)])
def test_prepared_input_validation_matches_raw_mesh_errors(bad):
    with pytest.raises(ValueError) as raw:
        k.triangle_meshes_intersect(bad, box())
    with pytest.raises(ValueError) as prepared:
        k.PreparedTriangleMesh(bad)
    assert str(prepared.value) == str(raw.value)


@pytest.mark.parametrize('empty', [[], np.empty((0, 3, 3)), np.empty((1, 0))])
def test_empty_input_and_invalid_tolerance_preserve_existing_results(empty):
    assert_mesh_equivalent(empty, box())
    with pytest.raises(ValueError, match='tolerance cannot be negative'):
        k.triangle_meshes_intersect(k.PreparedTriangleMesh(empty), box(), -1.)
