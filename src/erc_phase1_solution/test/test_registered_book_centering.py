"""Geometric target guarantees for a fitted bin, independent of its view ray."""
import itertools
import numpy as np
import pytest

from erc_phase1_solution.book_centered_place import book_centered_place_target


def corners():
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    # A known rigid attachment with neither its center nor axes at the TCP.
    rotation = np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    return signs * [.095, .025, .14] @ rotation.T + [.07, -.012, .023]


def bin_frame(yaw, tilt=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    z = np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    c, s = np.cos(tilt), np.sin(tilt)
    x = np.asarray([[1., 0., 0.], [0., c, -s], [0., s, c]])
    return z @ x @ np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


@pytest.mark.parametrize('yaw,tilt', [(0.2, 0.), (-0.7, .025), (1.9, -.04)])
def test_entire_attached_box_is_centered_and_aligned_in_the_measured_bin(yaw, tilt):
    frame, point = bin_frame(yaw, tilt), np.asarray([.91, -.12, .75])
    held = corners()
    target = book_centered_place_target(held, point, .27, bin_rotation=frame)
    placed = held @ target.tool_rotation.T + target.tool_position
    local = (placed - point) @ frame
    np.testing.assert_allclose(local.mean(axis=0), [0., .27, 0.], atol=1e-12)
    # Book axes retain the same orientation used by existing supported carry.
    np.testing.assert_allclose(local[4]-local[0], [-.19, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(local[2]-local[0], [0., -.05, 0.], atol=1e-12)
    np.testing.assert_allclose(local[1]-local[0], [0., 0., .28], atol=1e-12)
    assert np.linalg.det(target.tool_rotation) == pytest.approx(1.)


def test_fitted_orientation_is_independent_of_the_camera_approach_ray():
    frame = bin_frame(.3)
    first = book_centered_place_target(corners(), [.9, -.2, .75], .27, bin_rotation=frame)
    second = book_centered_place_target(corners(), [.7, .4, .75], .27, bin_rotation=frame)
    np.testing.assert_array_equal(first.tool_rotation, second.tool_rotation)
    np.testing.assert_allclose(second.book_center-first.book_center, [-.2, .6, 0.], atol=1e-12)


@pytest.mark.parametrize('bad', [np.eye(3), -bin_frame(0.), bin_frame(0.)*1.01,
                                bin_frame(0., .10), np.full((3, 3), np.nan), np.eye(4)])
def test_invalid_or_excessively_tilted_fitted_frame_cannot_produce_a_target(bad):
    with pytest.raises(ValueError, match='rigid upright'):
        book_centered_place_target(corners(), [.9, 0., .75], .27, bin_rotation=bad)
