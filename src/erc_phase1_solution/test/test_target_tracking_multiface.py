"""Regression coverage for one uniformly coloured book exposing several faces."""

from pathlib import Path

import numpy as np
import pytest

from erc_phase1_solution.target_tracking import (
    estimate_target_book_observation,
)
from erc_phase1_solution.vision import (
    BookDetection,
    CameraIntrinsics,
    detect_shelf_books,
)


COLOURS = {
    'red': (0, 0, 255), 'blue': (255, 0, 0),
    'green': (0, 255, 0), 'yellow': (0, 255, 255),
}


def _render_book(*, lateral=0.09, pitch=0.30, colour='blue', row=3):
    """Raycast a 2 cm spine, 25 cm tall cuboid; no production fitter involved."""
    camera = CameraIntrinsics(fx=500., fy=500., cx=200., cy=150.)
    rows, columns = np.indices((300, 400))
    rays = np.stack(((columns - camera.cx) / camera.fx,
                     (rows - camera.cy) / camera.fy,
                     np.ones_like(rows)), axis=-1)
    rotation = np.array(((1., 0., 0.), (0., np.cos(pitch), -np.sin(pitch)),
                         (0., np.sin(pitch), np.cos(pitch))))
    center = np.array((lateral, 0.0, 0.80))
    half = np.array((0.01, 0.125, 0.08))
    local_rays = rays @ rotation
    local_origin = -center @ rotation
    with np.errstate(divide='ignore', invalid='ignore'):
        first = (-half - local_origin) / local_rays
        last = (half - local_origin) / local_rays
    enter = np.minimum(first, last)
    leave = np.maximum(first, last)
    near = np.max(enter, axis=-1)
    far = np.min(leave, axis=-1)
    visible = (near > 0.) & (far >= near)
    depth = np.where(visible, near, 0.).astype(np.float32)
    rgb = np.zeros((*depth.shape, 3), np.uint8)
    rgb[visible] = COLOURS[colour]
    v, u = np.nonzero(visible)
    box = (int(u.min()), int(v.min()), int(np.ptp(u)) + 1, int(np.ptp(v)) + 1)
    detection = BookDetection(colour, box, float(len(u)), 0.98, row=row)
    front = visible & (np.argmax(enter, axis=-1) == 2)
    points = np.stack((rays[..., 0] * depth, rays[..., 1] * depth, depth), axis=-1)
    return rgb, depth, detection, camera, front, points


def _estimate(scene, **kwargs):
    rgb, depth, detection, camera, _, _ = scene
    return estimate_target_book_observation(
        rgb, depth, detection, camera,
        stamp_ns=456_000_000, frame_id='depth', row=detection.row, **kwargs,
    )


@pytest.mark.parametrize('colour', COLOURS)
@pytest.mark.parametrize('row', (1, 2, 3, 4))
@pytest.mark.parametrize('lateral', (-0.16, 0.09, 0.16))
def test_rgbd_coherent_front_with_side_and_top_for_all_labels(colour, row, lateral):
    scene = _render_book(lateral=lateral, colour=colour, row=row)
    rgb, depth, detection, camera, front, points = scene
    fraction = np.count_nonzero(front) / np.count_nonzero(depth)
    assert 0.25 < fraction < 0.60  # Exercise the new certificate, not majority.
    result = _estimate(scene)
    assert result.ok, (fraction, result)
    observation = result.observation
    assert observation.target_colour == colour
    assert observation.row == row
    assert observation.long_extent_m == pytest.approx(0.25, abs=0.004)
    assert observation.short_extent_m == pytest.approx(0.02, abs=0.003)
    assert observation.face_normal == pytest.approx(
        (0., np.sin(0.30), -np.cos(0.30)), abs=0.002
    )
    assert observation.plane_residual_m < 0.0001
    assert observation.center_uncertainty_m < 0.0005
    # All returned corners/centre belong to the physical front plane, even
    # though the side and top can together occupy most coloured pixels.
    expected_normal = np.array((0., np.sin(0.30), -np.cos(0.30)))
    expected_front = np.array((lateral, 0., 0.80)) + 0.08 * expected_normal
    for point in (observation.center, *observation.corners):
        assert abs((np.asarray(point) - expected_front) @ expected_normal) < 0.0001


def test_column4_blue_live_capture_selects_spine_under_original_uncertainty_limits():
    root = Path(__file__).resolve().parents[3]
    path = (root / 'docs/evidence/local_gazebo_20260915/column4_blue'
            / 'camera_after_abort.npz')
    with np.load(path) as captured:
        rgb, depth, k = captured['rgb'], captured['depth'], captured['intrinsics']
    detection, = [book for book in detect_shelf_books(rgb) if book.color == 'blue']
    result = estimate_target_book_observation(
        rgb, depth, detection,
        CameraIntrinsics(k[0, 0], k[1, 1], k[0, 2], k[1, 2]),
        stamp_ns=1, frame_id='depth', row=3,
    )
    assert result.ok, result
    observation = result.observation
    assert observation.long_extent_m == pytest.approx(0.250, abs=0.002)
    assert observation.short_extent_m == pytest.approx(0.020, abs=0.002)
    assert observation.center == pytest.approx((0.06637, 0.07801, 0.57876), abs=0.0001)
    assert observation.plane_residual_m < 0.0001
    assert observation.center_uncertainty_m < 0.0005


@pytest.mark.parametrize('corruption', ('parallel_layers', 'fragmented_front',
                                       'disconnected_outlier', 'metric_gap',
                                       'nonorthogonal_side', 'unexplained_depth'))
def test_fallback_rejects_ambiguous_or_incoherent_faces(corruption):
    rgb, depth, detection, camera, front, points = _render_book(lateral=0.16)
    visible = depth > 0
    side = visible & ~front
    v, u = np.nonzero(front)
    if corruption == 'parallel_layers':
        # The back layer is adjacent in the image but is a second parallel
        # plane, not the receding orthogonal side of a single book.
        depth[side] = np.median(depth[front]) + 0.12
    elif corruption == 'fragmented_front':
        band = front & (np.indices(depth.shape)[0] > int(np.median(v)) - 3)
        band &= np.indices(depth.shape)[0] < int(np.median(v)) + 3
        depth[band] += 0.05
    elif corruption == 'disconnected_outlier':
        # A coplanar patch separated from the spine must not enlarge its box.
        cut = front & (np.indices(depth.shape)[0] > int(np.quantile(v, 0.75)))
        rgb[cut] = 0
        depth[cut] = 0
        depth[5:15, 5:15] = 0.7
        rgb[5:15, 5:15] = COLOURS['blue']
        detection = BookDetection('blue', (0, 0, 400, 300), 3000., .98, row=3)
    elif corruption == 'metric_gap':
        # Keep an orthogonal side plane and its touching image pixels, but
        # move that plane far away. Two-dimensional contact is insufficient.
        depth[side] *= 1.50
    elif corruption == 'nonorthogonal_side':
        vv, uu = np.indices(depth.shape)
        depth[side] = 0.75 + 0.0002 * uu[side] + 0.0001 * vv[side]
    else:
        indices = np.flatnonzero(side)[::5]
        depth.flat[indices] += 0.07
    result = _estimate((rgb, depth, detection, camera, front, points))
    assert not result.ok, (corruption, result)
    assert result.reason == 'inconsistent_depth_geometry'


def test_fallback_keeps_missing_depth_and_uncertainty_gates():
    scene = _render_book(lateral=0.16)
    assert _estimate(scene, maximum_center_uncertainty_m=0.00001).reason == (
        'center_uncertainty_too_large'
    )
    rgb, depth, detection, camera, front, points = scene
    depth.flat[np.flatnonzero(depth)[::2]] = 0.
    result = _estimate(scene)
    assert not result.ok
    assert result.reason == 'insufficient_depth_coverage'
