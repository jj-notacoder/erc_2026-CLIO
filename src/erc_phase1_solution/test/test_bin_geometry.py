"""Captured/synthetic bin geometry regressions; CLIO attribution in notices."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution.bin_geometry import (
    BinObservation, BinTracker, bin_message_is_current, bin_message_is_verified, bin_stamp_is_fresh,
    bin_surface_camera_point, depth_in_metres, red_bin_candidates,
    verify_bin_candidate,
)
from erc_phase1_solution.vision import BinDetection, detect_colored_books, detect_red_bin


def render_scene(distance=2.0, side=0.0, object_width=0.50, object_height=0.21, object_depth=0.31,
                 table=True, wall_print=False, floor_print=False):
    """Pinhole ray-box/plane fixture, not a runtime target-location shortcut."""
    k = np.array([420., 0., 320., 0., 420., 180., 0., 0., 1.])
    transform = np.eye(4)
    origin = np.array([0., side, 1.4])
    look = np.array([distance, 0., 0.835])-origin
    look /= np.linalg.norm(look)
    right = np.cross(look, [0., 0., 1.])
    right /= np.linalg.norm(right)
    transform[:3, :3] = np.column_stack([right, np.cross(look, right), look])
    transform[:3, 3] = origin
    v, u = np.mgrid[:360, :640]
    rays = np.stack([(u-320)/420., (v-180)/420., np.ones_like(u)], axis=-1)
    rays = np.einsum('ij,hwj->hwi', transform[:3, :3], rays)
    depth = np.full(u.shape, np.inf)
    bgr = np.full((*u.shape, 3), 200, dtype=np.uint8)

    def paint(t, mask, colour):
        visible = mask & np.isfinite(t) & (t > 0) & (t < depth)
        depth[visible] = t[visible]
        bgr[visible] = colour

    with np.errstate(divide='ignore', invalid='ignore'):
        t = -origin[2]/rays[..., 2]
        paint(t, np.ones(u.shape, bool), [130, 130, 130])
        if floor_print:
            p = origin + t[..., None]*rays
            paint(t-1e-5, (abs(p[..., 0]-distance) < .4) & (abs(p[..., 1]) < .3), [0, 0, 210])
        elif wall_print:
            t = (distance-origin[0])/rays[..., 0]
            p = origin + t[..., None]*rays
            paint(t, np.ones(u.shape, bool), [180, 180, 180])
            paint(t-1e-5, (abs(p[..., 1]) < .25) & (abs(p[..., 2]-.835) < .105), [0, 0, 210])
        else:
            if table:
                t = (.73-origin[2])/rays[..., 2]
                p = origin + t[..., None]*rays
                paint(t, (abs(p[..., 0]-distance) < .7) & (abs(p[..., 1]) < .4), [230, 230, 230])
            lower = np.array([distance-object_depth/2, -object_width/2, .73])
            upper = np.array([distance+object_depth/2, object_width/2, .73+object_height])
            t1, t2 = (lower-origin)/rays, (upper-origin)/rays
            near, far = np.min(np.stack([t1, t2]), axis=0), np.max(np.stack([t1, t2]), axis=0)
            enter, leave = np.max(near, axis=-1), np.min(far, axis=-1)
            paint(enter, enter <= leave, [0, 0, 210])
    return bgr, depth.astype(np.float32), k, transform


def observations(scene):
    bgr, depth, k, transform = scene
    return [verify_bin_candidate(bgr, depth, k, transform, d)
            for d in red_bin_candidates(bgr)]


@pytest.mark.parametrize('distance, side', [(1.0, 0), (2.0, 0), (3.5, 0),
                                          (2.0, .7), (2.0, -.7)])
def test_table_supported_bin_at_different_views(distance, side):
    assert any(o is not None for o, _ in observations(render_scene(distance, side)))


@pytest.mark.parametrize('kwargs', [dict(floor_print=True), dict(wall_print=True),
                                  dict(table=False), dict(object_width=.16, object_height=.25,
                                                         object_depth=.02)])
def test_red_distractors_do_not_become_bins(kwargs):
    assert not any(o is not None for o, _ in observations(render_scene(**kwargs)))


def test_missing_depth_fails_closed():
    bgr, _, k, transform = render_scene()
    d = red_bin_candidates(bgr)[0]
    assert verify_bin_candidate(bgr, None, k, transform, d)[0] is None


def test_captured_arena_logo_has_wrong_3d_height():
    fixture = np.load(Path(__file__).parent/'data'/'bin_geometry'/'arena_logo_rgbd.npz')
    results = observations(tuple(fixture[key] for key in ['bgr', 'depth', 'k', 'base_from_camera']))
    assert results and all(o is None for o, _ in results)
    assert any(reason == 'wrong_height_above_floor' for _, reason in results)


def test_captured_real_bin_is_accepted():
    fixture = np.load(Path(__file__).parent/'data'/'bin_geometry'/'real_bin_rgbd.npz')
    results = observations(tuple(fixture[key] for key in ['bgr', 'depth', 'k', 'base_from_camera']))
    accepted = [o for o, _ in results if o is not None]
    assert len(accepted) == 1
    assert .10 <= accepted[0].metrics['height_m'] <= .27
    assert accepted[0].metrics['support_pixels'] >= 30


def observation(x=1.0):
    return BinObservation(BinDetection((100, 100, 60, 30), 1800., .9),
                          np.array([0., 0., 1.]), np.array([x, 0., .84]))


def test_confirmation_needs_three_distinct_frames():
    tracker = BinTracker()
    assert not tracker.observe(observation(), 0)
    assert not tracker.observe(observation(), 0)
    assert not tracker.observe(observation(), 125_000_000)
    assert tracker.observe(observation(), 250_000_000)


@pytest.mark.parametrize('gap, shift', [(600_000_000, 0), (125_000_000, .5)])
def test_stale_or_different_object_restarts_confirmation(gap, shift):
    tracker = BinTracker()
    tracker.observe(observation(), 0)
    tracker.observe(observation(), 125_000_000)
    assert not tracker.observe(observation(1+shift), 125_000_000+gap)
    assert tracker.count == 1


def test_lost_detection_clears_confirmation():
    tracker = BinTracker()
    for stamp in (0, 125_000_000, 250_000_000):
        tracker.observe(observation(), stamp)
    assert not tracker.observe(None, 375_000_000)
    assert tracker.count == 0


def test_old_colour_detector_accepts_logo_but_metric_gate_rejects_it():
    """Reproduce the actual false target that motivated the integration."""
    fixture = np.load(Path(__file__).parent/'data'/'bin_geometry'/'arena_logo_rgbd.npz')
    bgr = fixture['bgr']
    books = detect_colored_books(bgr, min_area=70, max_area_ratio=.08, assign_rows=False)
    candidate = detect_red_bin(bgr, book_detections=books, min_area=500, min_area_ratio=.006)
    assert candidate is not None
    result, reason = verify_bin_candidate(
        bgr, fixture['depth'], fixture['k'], fixture['base_from_camera'], candidate,
    )
    assert result is None
    assert reason == 'wrong_height_above_floor'


def test_uint16_millimetres_match_float_metres_and_keep_front_surface_contract():
    fixture = np.load(Path(__file__).parent/'data'/'bin_geometry'/'real_bin_rgbd.npz')
    bgr, depth, k, transform = (fixture[key] for key in ('bgr', 'depth', 'k', 'base_from_camera'))
    depth_mm = np.rint(np.clip(np.nan_to_num(depth), 0, 65.535)*1000).astype(np.uint16)
    candidate = red_bin_candidates(bgr)[0]
    assert verify_bin_candidate(bgr, depth_mm, k, transform, candidate)[0] is not None
    floating = bin_surface_camera_point(bgr, depth, k, candidate)
    integer = bin_surface_camera_point(bgr, depth_mm, k, candidate)
    np.testing.assert_allclose(integer, floating, atol=.001)
    x, y, width, height = candidate.bbox
    assert floating[0]/floating[2] == pytest.approx((x+width*.5-k[2])/k[0])
    assert floating[1]/floating[2] == pytest.approx((y+height*.70-k[5])/k[4])


def test_invalid_calibration_and_missing_surface_depth_are_rejected():
    bgr, depth, k, transform = render_scene()
    candidate = red_bin_candidates(bgr)[0]
    broken = k.copy()
    broken[0] = 0
    assert verify_bin_candidate(bgr, depth, broken, transform, candidate)[1] == 'invalid_calibration'
    assert bin_surface_camera_point(bgr, np.zeros_like(depth), k, candidate) is None
    np.testing.assert_allclose(depth_in_metres(np.array([1000], np.uint16)), [1.0])


@pytest.mark.parametrize('stamp, now, fresh', [
    (1_000_000_000, 1_000_000_000, True),
    (1_000_000_000, 1_500_000_000, True),
    (1_000_000_000, 1_500_000_001, False),
    (1_000_000_001, 1_000_000_000, False),
])
def test_bin_expiry_and_future_frames(stamp, now, fresh):
    assert bin_stamp_is_fresh(stamp, now) is fresh


def test_invalidation_rejects_latched_replay_without_rejecting_newer_points():
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=2, nanosec=0), frame_id='camera'),
        point=SimpleNamespace(x=0., y=0., z=2.),
    )
    assert bin_message_is_current(message, 2_100_000_000)
    assert not bin_message_is_verified(message, 2_100_000_000, -1)
    assert not bin_message_is_verified(message, 2_100_000_000, 1_900_000_000)
    assert bin_message_is_verified(message, 2_100_000_000, 2_000_000_000)
    assert not bin_message_is_current(message, 2_100_000_000, 2_000_000_000)
    assert bin_message_is_current(message, 2_100_000_000, 1_900_000_000)
    assert not bin_message_is_current(message, 2_600_000_000)
    message.point.z = float('nan')
    assert not bin_message_is_current(message, 2_100_000_000)
