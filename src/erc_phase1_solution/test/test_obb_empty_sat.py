"""Original/candidate differentials for skipping only zero-facet SAT work."""
from pathlib import Path
import ast
import importlib.util
import inspect
import itertools
import warnings

import numpy as np
import pytest

from erc_phase1_solution import kinematics as candidate


_fixture = Path(__file__).parent/'fixtures/obb_empty_sat_original.py'
_spec = importlib.util.spec_from_file_location('obb_empty_sat_original', _fixture)
original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(original)

FACES = np.asarray([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])


def corners(center=(0., 0., 0.), dimensions=(1., 1., 1.)):
    return (np.asarray(list(itertools.product((-1., 1.), repeat=3)))
            * np.asarray(dimensions)/2 + np.asarray(center))


def shell(center=(0., 0., 0.), dimensions=(4., 4., 4.)):
    return corners(center, dimensions)[FACES]


def outcome(function, box, surface, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            result = function(box, surface, **kwargs)
            value = ('return', type(result).__name__, bool(result))
        except Exception as exc:
            value = ('error', type(exc).__name__, str(exc))
    return value, [(w.category.__name__, str(w.message)) for w in caught]


def compare(box, surface, **kwargs):
    before_box, before_surface = np.asarray(box).copy(), np.asarray(surface).copy()
    expected = outcome(original.oriented_box_intersects_triangles, box, surface, **kwargs)
    actual = outcome(candidate.oriented_box_intersects_triangles, box, surface, **kwargs)
    assert actual == expected
    assert np.array_equal(np.asarray(box), before_box, equal_nan=True)
    assert np.array_equal(np.asarray(surface), before_surface, equal_nan=True)
    return actual[0]


def filtered_count(box, surface, tolerance=1e-9):
    center, axes, half = original.oriented_box_from_corners(box)
    vertices = (surface-center) @ axes
    separated = np.any((np.max(vertices, axis=1) < -half-tolerance)
                       | (np.min(vertices, axis=1) > half+tolerance), axis=1)
    return int(np.count_nonzero(~separated))


@pytest.mark.parametrize('closed', [False, True])
def test_outside_box_has_empty_filter_and_stays_clear(closed):
    box, mesh = corners((6., 0., 0.)), shell()
    assert filtered_count(box, mesh) == 0
    assert compare(box, mesh, closed_surface=closed) == ('return', 'bool', False)


@pytest.mark.parametrize('closed,expected', [(False, False), (True, True)])
def test_empty_filter_inside_shell_still_uses_closed_containment(closed, expected):
    box, mesh = corners(), shell()
    assert filtered_count(box, mesh) == 0
    assert compare(box, mesh, closed_surface=closed) == ('return', 'bool', expected)


@pytest.mark.parametrize('center', [(2.5, 0., 0.), (2., 0., 0.), (0., 0., 0.)])
def test_nonempty_touching_intersection_and_mesh_inside_box(center):
    box = corners(center, (5., 5., 5.)) if center == (0., 0., 0.) else corners(center)
    mesh = shell()
    assert filtered_count(box, mesh) > 0
    assert compare(box, mesh, closed_surface=True) == ('return', 'bool', True)


@pytest.mark.parametrize('angle', [0.31, -0.73])
def test_rotated_box_inside_rotated_shell_retains_ray_result(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
    translation = np.asarray([.7, -.2, 1.1])
    box = corners() @ rotation.T + translation
    mesh = shell() @ rotation.T + translation
    assert filtered_count(box, mesh) == 0
    assert compare(box, mesh, closed_surface=True) == ('return', 'bool', True)


@pytest.mark.parametrize('mesh', [
    np.asarray([[[0., 0., 0.]]*3]),
    np.asarray([[[8., 0., 0.]]*3]),
    np.asarray([[[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]]]),
    np.asarray([[[-.2, -.2, 0.], [.2, -.2, 0.], [0., .2, 0.]]]),
])
@pytest.mark.parametrize('closed', [False, True])
def test_degenerate_and_coplanar_surfaces_match(mesh, closed):
    compare(corners(), mesh, closed_surface=closed)


@pytest.mark.parametrize('box,mesh,tolerance', [
    (np.zeros((7, 3)), shell(), 1e-9),
    (np.zeros((8, 3)), shell(), 1e-9),
    (corners(), np.ones((2, 3)), 1e-9),
    (corners(), np.empty((0, 3, 3)), 1e-9),
    (corners(), np.empty((0, 5)), 1e-9),
    (corners(), np.full((1, 3, 3), np.nan), 1e-9),
    (corners(), np.full((1, 3, 3), np.inf), 1e-9),
    (corners(), shell(), -1e-9),
    (corners(), shell(), 0.),
])
def test_validation_warnings_and_existing_tolerance_behavior_match(box, mesh, tolerance):
    compare(box, mesh, tolerance=tolerance, closed_surface=True)


def test_mutated_and_aliased_current_geometry_is_not_cached():
    mesh = shell()
    alias = mesh.view()
    assert compare(corners(), alias, closed_surface=True)[-1] is True
    mesh += [6., 0., 0.]
    assert compare(corners(), alias, closed_surface=True)[-1] is False
    mesh -= [6., 0., 0.]
    assert compare(corners(), alias, closed_surface=True)[-1] is True


def cross_trace(monkeypatch, function, box, mesh, closed):
    calls = []
    real = np.cross
    def traced(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append((tuple((np.asarray(a).shape, np.asarray(a).tobytes()) for a in args),
                      tuple(sorted(kwargs.items())), result.shape, result.tobytes()))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(np, 'cross', traced)
        result = function(box, mesh, closed_surface=closed)
    return result, calls


@pytest.mark.parametrize('closed', [False, True])
def test_only_empty_facet_cross_work_removed_and_ray_arguments_unchanged(monkeypatch, closed):
    box, mesh = corners(), shell()
    expected, old_calls = cross_trace(monkeypatch, original.oriented_box_intersects_triangles, box, mesh, closed)
    actual, new_calls = cross_trace(monkeypatch, candidate.oriented_box_intersects_triangles, box, mesh, closed)
    assert actual == expected
    assert len(old_calls) == (12 if closed else 10)
    assert len(new_calls) == (2 if closed else 0)
    assert all(call[0][0][0][0] == 0 for call in old_calls[:10])
    assert new_calls == old_calls[10:]


def test_nonempty_facet_cross_arguments_and_order_are_identical(monkeypatch):
    box, mesh = corners((2., 0., 0.)), shell()
    assert filtered_count(box, mesh) > 0
    expected, old_calls = cross_trace(monkeypatch, original.oriented_box_intersects_triangles, box, mesh, True)
    actual, new_calls = cross_trace(monkeypatch, candidate.oriented_box_intersects_triangles, box, mesh, True)
    assert actual == expected
    assert new_calls == old_calls


def test_stripping_the_single_guard_restores_exact_original_function_ast():
    function = ast.parse(inspect.getsource(candidate.oriented_box_intersects_triangles)).body[0]
    expected = ast.parse(inspect.getsource(original.oriented_box_intersects_triangles)).body[0]
    guards = [(i,x) for i,x in enumerate(function.body) if isinstance(x, ast.If)
              and ast.dump(x.test) == ast.dump(ast.parse('len(vertices)', mode='eval').body)]
    assert len(guards) == 1 and not guards[0][1].orelse
    index, guard = guards[0]
    function.body[index:index+1] = guard.body
    assert ast.dump(function) == ast.dump(expected)
