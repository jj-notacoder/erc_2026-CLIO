import numpy as np
import pytest

from erc_phase1_solution import static_pair_separation as module
from erc_phase1_solution.kinematics import _box_triangles, _rpy_matrix, triangle_meshes_intersect
from erc_phase1_solution.static_pair_separation import StaticPairSeparation


def box(center=(0., 0., 0.), size=(1., 1., 1.), angle=0.):
    return _box_triangles(size) @ _rpy_matrix((0., 0., angle)).T + center


@pytest.mark.parametrize('gap,expected', [(0., False), (-.1, False),
    (99e-6, False), (100e-6, False), (101e-6, True), (.2, True)])
def test_strict_margin_and_touch(gap, expected):
    assert StaticPairSeparation().separated('a', 'b', box(), box((1.+gap, 0., 0.))) == expected


def test_containment_and_compound_component_collision_preserved():
    helper = StaticPairSeparation()
    outer = box(size=(3., 3., 3.)); inner = box(size=(.2, .2, .2))
    assert not helper.separated('a', 'b', outer, inner)
    assert triangle_meshes_intersect(outer, inner, first_watertight=True, second_watertight=True)
    compound = np.concatenate((box((10., 0., 0.)), box((.25, 0., 0.))))
    assert not helper.separated('a', 'b', box(), compound)
    assert triangle_meshes_intersect(box(), compound)


def test_stale_hint_reprojects_changed_geometry_and_context():
    helper = StaticPairSeparation()
    first, second = box(), box((2., 0., 0.))
    assert helper.separated('a', 'b', first, second)
    saved = helper._pair_hints[('a', 'b')].copy()
    assert not helper.separated('a', 'b', first, box((.25, 0., 0.)))
    for angle, translation in ((.8, (2., -4., .5)), (-1.2, (-3., 1., 2.))):
        rotation = _rpy_matrix((0., 0., angle))
        a = first @ rotation.T + translation
        b = (first + (.25, 0., 0.)) @ rotation.T + translation
        assert not helper.separated('a', 'b', a, b)
    np.testing.assert_array_equal(helper._pair_hints[('a', 'b')], saved)


def test_degenerate_legacy_false_positive_refined_but_touch_preserved():
    helper = StaticPairSeparation()
    # Learn a valid arbitrary direction from two full-dimensional thin boxes.
    a = box(size=(2., .01, .01), angle=np.pi/4)
    assert helper.separated('a', 'b', a, a+(0., .2, 0.))
    first = np.asarray([[(0., 0., 0.), (1., 1., 0.), (2., 2., 0.)]])
    second = first+(0., .2, 0.)
    assert triangle_meshes_intersect(first, second)  # Conservative legacy fallback.
    assert helper.separated('a', 'b', first, second)
    assert not helper.separated('a', 'b', first, first)
    assert not helper.separated('a', 'b', first, first+(2., 2., 0.))


@pytest.mark.parametrize('surface', [np.empty((0, 3, 3)), np.zeros((3, 3)),
    np.full((1, 3, 3), np.nan), np.full((1, 3, 3), np.inf)])
def test_invalid_current_surface_falls_through(surface):
    assert not StaticPairSeparation().separated('a', 'b', box(), surface)


@pytest.mark.parametrize('axis', [(0., 0., 0.), (np.nan, 0., 0.), (np.inf, 1., 0.), (1., 2.)])
def test_invalid_axis_is_not_used(axis):
    assert StaticPairSeparation._unit(axis) is None


def test_safe_axis_normalization_handles_extreme_finite_scale():
    for scale in (1e-300, 1e300):
        direction = StaticPairSeparation._unit(np.asarray((1., -1., 1.))*scale)
        assert np.all(np.isfinite(direction))
        assert np.linalg.norm(direction) == pytest.approx(1.)


def test_large_cancellation_requires_operand_scaled_margin():
    helper = StaticPairSeparation()
    a = box(size=(2., .01, .01), angle=np.pi/4)
    assert helper.separated('a', 'b', a, a+(0., .2, 0.))
    shifted = a+np.asarray((1e12, 1e12, 0.))
    # Dot products along the diagonal nearly cancel, but their operands are huge.
    assert not helper.separated('a', 'b', shifted, shifted+(0., .02, 0.))
    assert not helper.separated('a', 'b', shifted, shifted)


def test_overflow_falls_through_without_hull(monkeypatch):
    monkeypatch.setattr(module, 'ConvexHull', lambda *a: pytest.fail('uncertain overflow reached hull'))
    huge = np.full((1, 3, 3), 1e308)
    assert not StaticPairSeparation().separated('a', 'b', huge, -huge)


def test_hull_failure_is_bounded_and_retains_original_fallback(monkeypatch):
    calls = []
    def fail(*args):
        calls.append(1)
        raise module.QhullError('synthetic hull failure')
    monkeypatch.setattr(module, 'ConvexHull', fail)
    helper = StaticPairSeparation()
    for _ in range(3):
        assert not helper.separated('a', 'b', box(), box((2., 0., 0.)))
    assert len(calls) == 2


def test_bounded_owned_immutable_hints_and_eviction():
    helper = StaticPairSeparation(maximum_links=2, maximum_pairs=1, maximum_axes=3)
    for i in range(4):
        assert helper.separated(f'a{i}', f'b{i}', box(), box((2., 0., 0.)))
    assert len(helper._link_hints) <= 2
    assert len(helper._pair_hints) == 1
    assert ('a0', 'b0') not in helper._pair_hints
    for value in (*helper._link_hints.values(), *helper._pair_hints.values()):
        assert not value.flags.writeable
        assert value.flags.owndata
    assert all(len(v) <= 3 for v in helper._link_hints.values())
    assert not helper.separated('a0', 'b0', box(), box())


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_invalid_hint_capacity(count):
    with pytest.raises(ValueError):
        StaticPairSeparation(maximum_axes=count)
