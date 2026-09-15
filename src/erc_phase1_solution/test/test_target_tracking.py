"""Focused tests for production RGB-D target-book tracking guards."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from erc_phase1_solution.target_tracking import (
    TargetBookObservation,
    TargetBookTracker,
    estimate_target_book_observation,
    guard_target_motion,
)
from erc_phase1_solution.vision import BookDetection, CameraIntrinsics


def _observation(
    stamp_ns: int,
    *,
    center=(0.0, 0.0, 0.70),
    colour='red',
    row=2,
    bbox=(50, 25, 20, 80),
    short_extent=0.030,
    long_extent=0.250,
) -> TargetBookObservation:
    x, y, z = center
    half_width = short_extent / 2.0
    half_height = long_extent / 2.0
    corners = (
        (x - half_width, y - half_height, z),
        (x + half_width, y - half_height, z),
        (x + half_width, y + half_height, z),
        (x - half_width, y + half_height, z),
    )
    return TargetBookObservation(
        stamp_ns=stamp_ns,
        frame_id='camera_depth_optical_frame',
        target_colour=colour,
        row=row,
        center=center,
        corners=corners,
        face_normal=(0.0, 0.0, -1.0),
        long_axis=(0.0, 1.0, 0.0),
        short_extent_m=short_extent,
        long_extent_m=long_extent,
        bbox=bbox,
        image_center=(bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0),
        detection_confidence=0.96,
        depth_coverage=0.98,
        plane_residual_m=0.00005,
        center_uncertainty_m=0.00005,
        extent_uncertainty_m=0.00005,
        orientation_uncertainty_rad=0.0001,
        quality_confidence=0.90,
    )


def _tracked_samples(tracker: TargetBookTracker):
    accepted = []
    for stamp_ns in (0, 50_000_000, 100_000_000):
        result = tracker.observe(_observation(stamp_ns))
        assert result.ok and result.observation is not None
        accepted.append(result.observation)
    return accepted


def test_rgbd_face_estimate_contains_metric_center_extent_and_orientation():
    rgb = np.zeros((130, 180, 3), dtype=np.uint8)
    cv2.rectangle(rgb, (65, 25), (85, 105), (0, 0, 255), cv2.FILLED)
    depth = np.ones(rgb.shape[:2], dtype=np.float32)
    detection = BookDetection('red', (65, 25, 21, 81), 1600.0, 0.98, row=2)

    result = estimate_target_book_observation(
        rgb,
        depth,
        detection,
        CameraIntrinsics(fx=400.0, fy=400.0, cx=90.0, cy=65.0),
        stamp_ns=123_000_000,
        frame_id='camera_depth_optical_frame',
        row=2,
    )

    assert result.ok, result
    observation = result.observation
    assert observation is not None
    assert observation.center == pytest.approx((-0.0375, 0.0, 1.0), abs=0.002)
    assert observation.short_extent_m == pytest.approx(0.05, abs=0.004)
    assert observation.long_extent_m == pytest.approx(0.20, abs=0.004)
    assert observation.face_normal == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert abs(observation.long_axis[1]) > 0.99
    assert observation.depth_coverage == pytest.approx(1.0)
    assert observation.center_uncertainty_m <= 0.0005
    assert observation.orientation_uncertainty_rad <= 0.05
    assert len(observation.corners) == 4


def test_rgbd_face_estimate_selects_dominant_front_of_same_colour_prism():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(rgb, (60, 25), (72, 89), (0, 0, 255), cv2.FILLED)
    depth = np.zeros(rgb.shape[:2], dtype=np.float32)

    # The right eight columns are the camera-facing spine.  The left five are
    # a receding side of the same uniformly red box, so colour segmentation
    # cannot split the two planes.  A whole-mask PCA centre is consequently
    # more than 6 cm behind the actual front face.
    depth[25:90, 65:73] = 1.0
    side_normal_z = 7.0 / 60.0
    side_plane_offset = side_normal_z - 25.0 / 400.0
    for u in range(60, 65):
        ray_x = (u - 90.0) / 400.0
        depth[25:90, u] = side_plane_offset / (ray_x + side_normal_z)

    detection = BookDetection('red', (60, 25, 13, 65), 845.0, 0.98, row=1)
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=90.0, cy=57.0)
    mask_depths = depth[25:90, 60:73]
    assert float(np.mean(mask_depths)) > 1.06

    result = estimate_target_book_observation(
        rgb,
        depth,
        detection,
        intrinsics,
        stamp_ns=456_000_000,
        frame_id='depth',
        row=1,
    )

    assert result.ok, result
    observation = result.observation
    assert observation is not None
    assert observation.center == pytest.approx((-0.05375, 0.0, 1.0), abs=0.001)
    assert observation.image_center == pytest.approx((68.5, 57.0), abs=0.1)
    assert observation.short_extent_m == pytest.approx(0.0175, abs=0.001)
    assert observation.long_extent_m == pytest.approx(0.1600, abs=0.002)
    assert observation.plane_residual_m < 1e-6
    assert abs(observation.face_normal[2]) > 0.99


def test_rgbd_face_estimate_fails_closed_on_depth_or_colour_occlusion():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(rgb, (60, 20), (80, 100), (255, 0, 0), cv2.FILLED)
    detection = BookDetection('blue', (60, 20, 21, 81), 1600.0, 0.95, row=3)
    intrinsics = CameraIntrinsics(fx=420.0, fy=420.0, cx=80.0, cy=60.0)

    mostly_missing_depth = np.zeros(rgb.shape[:2], dtype=np.float32)
    mostly_missing_depth[20:35, 60:81] = 0.8
    missing = estimate_target_book_observation(
        rgb,
        mostly_missing_depth,
        detection,
        intrinsics,
        stamp_ns=1,
        frame_id='depth',
        row=3,
    )
    assert not missing.ok
    assert missing.reason == 'insufficient_depth_coverage'

    rgb[20:100, 60:81] = 0
    rgb[58:62, 68:72] = (255, 0, 0)
    colour_hidden = estimate_target_book_observation(
        rgb,
        np.full(rgb.shape[:2], 0.8, dtype=np.float32),
        detection,
        intrinsics,
        stamp_ns=2,
        frame_id='depth',
        row=3,
    )
    assert not colour_hidden.ok
    assert colour_hidden.reason == 'target_occluded'


def test_rgbd_face_estimate_rejects_inconsistent_depth_geometry():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(rgb, (60, 20), (80, 100), (0, 255, 0), cv2.FILLED)
    depth = np.full(rgb.shape[:2], 0.7, dtype=np.float32)
    depth[20:101:2, 60:81] = 1.0
    detection = BookDetection('green', (60, 20, 21, 81), 1600.0, 0.95, row=1)

    result = estimate_target_book_observation(
        rgb,
        depth,
        detection,
        CameraIntrinsics(fx=420.0, fy=420.0, cx=80.0, cy=60.0),
        stamp_ns=1,
        frame_id='depth',
        row=1,
    )

    assert not result.ok
    assert result.reason == 'inconsistent_depth_geometry'


def test_latest_requires_high_rate_warmup_and_rejects_stale_observation():
    tracker = TargetBookTracker(
        minimum_rate_hz=10.0,
        minimum_consecutive_observations=3,
        maximum_age_seconds=0.15,
    )
    first = tracker.observe(_observation(0))
    assert first.ok
    assert tracker.latest(10_000_000).reason == 'target_track_warming_up'
    tracker.observe(_observation(50_000_000))
    tracker.observe(_observation(100_000_000))

    fresh = tracker.latest(120_000_000)
    assert fresh.ok
    assert fresh.observation is not None
    assert fresh.observation.observed_rate_hz == pytest.approx(20.0)
    stale = tracker.latest(300_000_001)
    assert not stale.ok
    assert stale.reason == 'stale_target_observation'


def test_identity_continuity_rejects_label_jump_and_spatial_jump():
    tracker = TargetBookTracker(minimum_consecutive_observations=1)
    original = tracker.observe(_observation(0))
    assert original.ok

    wrong_label = tracker.observe(
        _observation(50_000_000, colour='blue', row=2)
    )
    assert not wrong_label.ok
    assert wrong_label.reason == 'target_identity_changed'

    tracker.reset()
    original = tracker.observe(_observation(0))
    jumped = tracker.observe(
        _observation(
            50_000_000,
            center=(0.10, 0.0, 0.70),
            bbox=(120, 25, 20, 80),
        )
    )
    assert original.ok
    assert not jumped.ok
    assert jumped.reason == 'target_identity_discontinuity'


def test_extent_loss_invalidates_track_as_occluded():
    tracker = TargetBookTracker(minimum_consecutive_observations=1)
    tracker.observe(_observation(0))
    partial = tracker.observe(
        _observation(
            50_000_000,
            bbox=(50, 45, 20, 40),
            long_extent=0.125,
        )
    )
    assert not partial.ok
    assert partial.reason == 'target_occluded_or_extent_changed'
    assert tracker.latest(60_000_000).reason == partial.reason


def test_occlusion_invalidates_pose_until_three_new_frames():
    tracker = TargetBookTracker(minimum_consecutive_observations=3)
    before_occlusion = _tracked_samples(tracker)[-1]
    tracker.mark_unavailable(110_000_000, 'target_occluded')
    blocked = tracker.latest(115_000_000)
    assert not blocked.ok and blocked.reason == 'target_occluded'

    tracker.observe(_observation(150_000_000))
    tracker.observe(_observation(200_000_000))
    assert tracker.latest(210_000_000).reason == 'target_track_warming_up'
    tracker.observe(_observation(250_000_000))
    reacquired = tracker.latest(260_000_000)
    assert reacquired.ok and reacquired.observation is not None
    assert reacquired.observation.track_id != before_occlusion.track_id
    assert guard_target_motion(
        before_occlusion,
        reacquired.observation,
    ).reason == 'target_identity_not_continuous'


def test_long_gap_starts_new_identity_generation():
    tracker = TargetBookTracker(minimum_consecutive_observations=1)
    reference = tracker.observe(_observation(0)).observation
    current = tracker.observe(_observation(300_000_000)).observation
    assert reference is not None and current is not None
    assert reference.track_id != current.track_id
    assert current.sequence == 1
    assert guard_target_motion(reference, current).reason == (
        'target_identity_not_continuous'
    )


def test_motion_guard_includes_uncertainty_in_submillimetre_bound():
    tracker = TargetBookTracker(minimum_consecutive_observations=1)
    reference = tracker.observe(_observation(0)).observation
    assert reference is not None
    small = tracker.observe(
        _observation(50_000_000, center=(0.0002, 0.0, 0.70))
    ).observation
    assert small is not None

    guarded = guard_target_motion(
        reference,
        small,
        maximum_center_motion_m=0.0005,
        maximum_corner_motion_m=0.0005,
    )
    assert guarded.ok
    assert guarded.metrics[
        'center_motion_upper_bound_m'
    ] == pytest.approx(0.0003)

    moved_center = replace(
        small,
        stamp_ns=100_000_000,
        center=(0.0010, 0.0, 0.70),
    )
    rejected = guard_target_motion(reference, moved_center)
    assert not rejected.ok
    assert rejected.reason == 'target_center_motion_exceeded'


def test_default_config_processes_faster_than_guard_minimum():
    config_path = (
        Path(__file__).resolve().parents[1] / 'config' / 'solution.yaml'
    )
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    parameters = config['erc_perception']['ros__parameters']
    assert parameters['publish_rate_hz'] >= 10.0
    assert (
        parameters['publish_rate_hz']
        > parameters['target_tracking_minimum_rate_hz']
    )
    assert parameters['target_tracking_maximum_age_seconds'] <= 0.15
