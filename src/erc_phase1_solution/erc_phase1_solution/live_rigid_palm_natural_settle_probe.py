#!/usr/bin/env python3
"""Diagnostic-only shelf-edge natural-settle proof for the rigid palm.

This probe is intentionally *not* competition logic.  It uses Gazebo entity
poses, collision names, and the temporary palm contact sensor to answer one
engineering question: can the already proven three-point cage take the target
from the supported extraction checkpoint into free space without losing it?

The only accepted initial state is the exact seed-101 step-18 cage.  The probe
then commands one-millimetre outward arm legs.  A live watchdog records the
first valid rear-clearance or non-rigid settle trigger while allowing that
already bounded leg to finish; every actual hazard still cancels immediately.
Exactly one uncommanded settling episode may then be observed; the arm, base,
and aperture must remain fixed and every drop, tilt, contact, neighbour, and
stability bound below must pass.  The settled, tilted OBB is re-anchored from a
fresh Gazebo frame.  Finally, only the shortest of a 5 mm or 10 mm outward leg
that produces 0.75 mm padded shelf clearance is preflighted and executed.  The
probe stops there: it never moves the base or opens the gripper.

Importing this module is side-effect free.  ROS/controller access is reachable
only from :func:`main`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from erc_phase1_solution.kinematics import (
    oriented_box_from_corners,
    oriented_box_intersects_triangles,
    pose_matrix,
)
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    EXPECTED_CAGED_APERTURE_M,
    GuardResult,
    SHELF_FRONT_X_M,
    STEP_BASE_POSITION_LIMIT_M,
    STEP_BASE_YAW_LIMIT_RAD,
    TARGET_BOOK_MODEL,
    _angle_error,
    _attached_corners,
    _emit,
    _fresh_cage_gate,
    _preflight_leg,
    _probe_types,
    _quaternion_from_rotation,
    _unexpected_pairs,
)
from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (
    EDGE_RESUME_BASE_POSITION_TOLERANCE_M,
    EDGE_RESUME_BASE_YAW_TOLERANCE_RAD,
    EDGE_RESUME_BOOK_POSITION_TOLERANCE_M,
    EDGE_RESUME_BOOK_ROTATION_TOLERANCE_RAD,
    EDGE_RESUME_BOOK_VERTICAL_TOLERANCE_M,
    EDGE_RESUME_JOINT_TOLERANCE_RAD,
    _inflated_ordered_corners,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    BOOK_HALF_EXTENTS_M,
    CAGE_MEASURED_TARGET_TOLERANCE_M,
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
    quaternion_matrix,
    rotation_matrix_distance,
    world_hand_pose,
)


EDGE_TRANSFER_EVENT = 'rigid_palm_natural_settle_edge_transfer'
POST_SETTLE_EVENT = 'rigid_palm_natural_settle_clearance'

# Same-world checkpoint recorded at the end of the newer guarded extraction
# route.  Keep it local: the older shelf-edge cradle route has a distinct
# checkpoint and must retain its independent 8 mm support invariant.
NATURAL_SETTLE_STEP18_JOINTS = np.asarray(
    [
        0.34999999234790324,
        0.2964100674087549,
        0.5485206268410916,
        0.5454247692205901,
        -1.823269763176584,
        1.2991260825214985,
        0.8462354148630724,
        -1.3982444214463716,
    ],
    dtype=float,
)
NATURAL_SETTLE_STEP18_BOOK_POSITION = np.asarray(
    [2.679920184261594, -0.14716010088042255, 1.5770272549887803],
    dtype=float,
)
NATURAL_SETTLE_STEP18_BOOK_MAXIMUM_X_M = 2.7601924666549964
NATURAL_SETTLE_MINIMUM_SHELF_OVERLAP_M = 0.005
INITIAL_SHELF_CONTACT_TOLERANCE_M = 0.00025
INITIAL_BOOK_LOWER_FACE_ALIGNMENT_MINIMUM = math.cos(math.radians(3.0))
INITIAL_SHELF_UPWARD_NORMAL_MINIMUM = math.cos(math.radians(1.0))

EDGE_LEG_DISTANCE_M = 0.001
EDGE_LEG_DURATION_S = 0.30
EDGE_MAXIMUM_LEGS = 20
EDGE_PROGRESS_MINIMUM_M = 0.0005
EDGE_PROGRESS_MAXIMUM_M = 0.0015
EDGE_HAND_BOOK_MISMATCH_LIMIT_M = 0.00075
EDGE_BOOK_ROTATION_LIMIT_RAD = math.radians(1.0)
EDGE_REAR_CLEARANCE_X_M = SHELF_FRONT_X_M - 0.00075
PREMATURE_SETTLE_OVERLAP_M = 0.002
SOFT_SETTLE_TRIGGER_REASONS = frozenset(
    {'nonrigid_motion_started', 'rear_clearance_reached'}
)

SETTLE_TIMEOUT_S = 2.0
SETTLE_STABILITY_WINDOW_S = 0.50
SETTLE_MINIMUM_STABILITY_SAMPLES = 3
SETTLE_CENTER_SPAN_LIMIT_M = 0.00050
SETTLE_CORNER_SPAN_LIMIT_M = 0.00075
SETTLE_ATTITUDE_SPAN_LIMIT_RAD = math.radians(0.50)
SETTLE_ARM_SPAN_LIMIT_RAD = 0.004
SETTLE_BASE_SPAN_LIMIT_M = 0.00035
SETTLE_BASE_YAW_SPAN_LIMIT_RAD = 0.00050
SETTLE_APERTURE_SPAN_LIMIT_M = 0.001
SETTLE_DROP_LIMIT_M = 0.060
SETTLE_FINAL_TILT_MINIMUM_RAD = math.radians(20.0)
SETTLE_FINAL_TILT_MAXIMUM_RAD = math.radians(30.0)
SETTLE_PEAK_TILT_LIMIT_RAD = math.radians(30.0)
SETTLE_CENTER_Y_DRIFT_LIMIT_M = 0.001
SETTLE_OFF_AXIS_TILT_LIMIT_RAD = math.radians(3.0)
SETTLE_ARM_HOLD_LIMIT_RAD = 0.004
SETTLE_BASE_HOLD_LIMIT_M = 0.00035
SETTLE_BASE_YAW_HOLD_LIMIT_RAD = 0.00050
SETTLE_APERTURE_HOLD_LIMIT_M = 0.001
NON_TARGET_BOOK_MOTION_LIMIT_M = 0.00050

POST_SETTLE_CANDIDATE_DISTANCES_M = (0.005, 0.010)
POST_SETTLE_PADDING_M = 0.00075
POST_SETTLE_PROGRESS_TOLERANCE_M = 0.0010
POST_SETTLE_DURATION_PER_5MM_S = 0.60
POST_SETTLE_ENDPOINT_ARM_LIMIT_RAD = 0.004

SCENE_WAIT_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class WorldSample:
    """One coherent diagnostic snapshot from a Gazebo Pose_V frame."""

    book: BookSnapshot
    base: np.ndarray
    arm: np.ndarray
    hand_world: np.ndarray
    aperture_m: float
    observed_at: float
    corners: np.ndarray
    all_book_corners: Mapping[str, np.ndarray]
    shelf_triangles: np.ndarray


@dataclass(frozen=True)
class SettleSample:
    """The values needed to prove one trailing stable interval."""

    observed_at: float
    book: BookSnapshot
    corners: np.ndarray
    arm: np.ndarray
    base: np.ndarray
    aperture_m: float


@dataclass(frozen=True)
class LegWatchResult:
    """Last coherent observation and any watchdog cancellation reason."""

    reason: str | None
    sample: WorldSample | None
    metrics: Mapping[str, float]


def natural_settle_step18_resume_guard(
    book: BookSnapshot,
    base: Sequence[float],
    arm: Sequence[float],
    *,
    aperture_m: float,
) -> GuardResult:
    """Recognize only the newer zero-angle step-18 extraction checkpoint."""

    try:
        current_base = np.asarray(base, dtype=float)
        current_arm = np.asarray(arm, dtype=float)
        aperture = float(aperture_m)
        if (
            current_base.shape != (3,)
            or current_arm.shape != NATURAL_SETTLE_STEP18_JOINTS.shape
            or not np.all(np.isfinite(current_base))
            or not np.all(np.isfinite(current_arm))
            or not math.isfinite(aperture)
        ):
            raise ValueError('natural-settle resume observation is malformed')
        book_xy_error = float(
            np.linalg.norm(
                book.position[:2]
                - NATURAL_SETTLE_STEP18_BOOK_POSITION[:2]
            )
        )
        book_vertical_error = abs(
            float(
                book.position[2]
                - NATURAL_SETTLE_STEP18_BOOK_POSITION[2]
            )
        )
        book_rotation_error = quaternion_distance(
            book.quaternion,
            EXPECTED_RELEASED_BOOK_QUATERNION,
        )
        shelf_overlap = float(book.maximum[0] - SHELF_FRONT_X_M)
        book_maximum_error = abs(
            float(
                book.maximum[0]
                - NATURAL_SETTLE_STEP18_BOOK_MAXIMUM_X_M
            )
        )
        base_error = float(
            np.linalg.norm(current_base[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
        )
        base_yaw_error = _angle_error(
            current_base[2], EXPECTED_RELEASED_BASE_POSE[2]
        )
        arm_error = float(
            np.max(np.abs(current_arm - NATURAL_SETTLE_STEP18_JOINTS))
        )
        aperture_error = abs(aperture - EXPECTED_CAGED_APERTURE_M)
        metrics = {
            'book_xy_error_m': book_xy_error,
            'book_vertical_error_m': book_vertical_error,
            'book_rotation_error_rad': book_rotation_error,
            'shelf_overlap_m': shelf_overlap,
            'book_maximum_x_error_m': book_maximum_error,
            'base_error_m': base_error,
            'base_yaw_error_rad': base_yaw_error,
            'arm_error_rad': arm_error,
            'aperture_error_m': aperture_error,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})

    checks = (
        (
            aperture_error <= CAGE_MEASURED_TARGET_TOLERANCE_M,
            'gripper_not_at_proven_cage',
        ),
        (
            book_xy_error <= EDGE_RESUME_BOOK_POSITION_TOLERANCE_M,
            'wrong_step18_book_position',
        ),
        (
            book_vertical_error <= EDGE_RESUME_BOOK_VERTICAL_TOLERANCE_M,
            'wrong_step18_book_height',
        ),
        (
            book_rotation_error <= EDGE_RESUME_BOOK_ROTATION_TOLERANCE_RAD,
            'wrong_step18_book_rotation',
        ),
        (
            shelf_overlap + 1e-12
            >= NATURAL_SETTLE_MINIMUM_SHELF_OVERLAP_M,
            'shelf_overlap_lost',
        ),
        (
            book_maximum_error <= EDGE_RESUME_BOOK_POSITION_TOLERANCE_M,
            'wrong_step18_book_extent',
        ),
        (
            base_error <= EDGE_RESUME_BASE_POSITION_TOLERANCE_M,
            'wrong_step18_base',
        ),
        (
            base_yaw_error <= EDGE_RESUME_BASE_YAW_TOLERANCE_RAD,
            'wrong_step18_base_yaw',
        ),
        (
            arm_error <= EDGE_RESUME_JOINT_TOLERANCE_RAD,
            'wrong_step18_arm',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        sum(
            _cross_2d(point, polygon[(index + 1) % len(polygon)])
            for index, point in enumerate(polygon)
        )
    )


def _clip_convex_polygon_to_triangle(
    polygon: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    """Clip one CCW convex XY polygon by one CCW XY triangle."""

    output = [np.asarray(point, dtype=float) for point in polygon]
    for index, clip_start in enumerate(triangle):
        clip_end = triangle[(index + 1) % 3]
        clip_edge = clip_end - clip_start
        incoming = output
        output = []
        if not incoming:
            break
        previous = incoming[-1]
        previous_inside = (
            _cross_2d(clip_edge, previous - clip_start) >= -1e-12
        )
        for current in incoming:
            current_inside = (
                _cross_2d(clip_edge, current - clip_start) >= -1e-12
            )
            if current_inside != previous_inside:
                segment = current - previous
                denominator = _cross_2d(clip_edge, segment)
                if abs(denominator) > 1e-15:
                    fraction = _cross_2d(
                        clip_edge, clip_start - previous
                    ) / denominator
                    output.append(
                        previous + min(1.0, max(0.0, fraction)) * segment
                    )
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def _plane_height_at_xy(
    point: np.ndarray, normal: np.ndarray, xy: np.ndarray
) -> float:
    return float(
        point[2]
        - (
            normal[0] * (xy[0] - point[0])
            + normal[1] * (xy[1] - point[1])
        )
        / normal[2]
    )


def initial_shelf_support_geometry_guard(
    corners: Sequence[Sequence[float]],
    shelf_triangles: Sequence[Sequence[Sequence[float]]],
    *,
    minimum_overlap_m: float = NATURAL_SETTLE_MINIMUM_SHELF_OVERLAP_M,
    contact_tolerance_m: float = INITIAL_SHELF_CONTACT_TOLERANCE_M,
    upward_normal_minimum: float = INITIAL_SHELF_UPWARD_NORMAL_MINIMUM,
    lower_face_alignment_minimum: float = (
        INITIAL_BOOK_LOWER_FACE_ALIGNMENT_MINIMUM
    ),
) -> GuardResult:
    """Prove that the physical lower book face rests on a shelf top panel."""

    try:
        book = np.asarray(corners, dtype=float)
        shelf = np.asarray(shelf_triangles, dtype=float)
        minimum_overlap = float(minimum_overlap_m)
        contact_tolerance = float(contact_tolerance_m)
        upward_minimum = float(upward_normal_minimum)
        face_alignment_minimum = float(lower_face_alignment_minimum)
        if (
            book.shape != (8, 3)
            or shelf.ndim != 3
            or shelf.shape[1:] != (3, 3)
            or not len(shelf)
            or not np.all(np.isfinite(book))
            or not np.all(np.isfinite(shelf))
            or not math.isfinite(minimum_overlap)
            or minimum_overlap <= 0.0
            or not math.isfinite(contact_tolerance)
            or contact_tolerance <= 0.0
            or not math.isfinite(upward_minimum)
            or not 0.0 < upward_minimum <= 1.0
            or not math.isfinite(face_alignment_minimum)
            or not 0.0 < face_alignment_minimum <= 1.0
        ):
            raise ValueError('initial shelf-support geometry is malformed')

        center, axes, half_extents = oriented_box_from_corners(book)
        vertical_axis = int(np.argmax(np.abs(axes[2])))
        upward_face_normal = axes[:, vertical_axis].copy()
        if upward_face_normal[2] < 0.0:
            upward_face_normal *= -1.0
        lower_face_alignment = float(upward_face_normal[2])

        signs = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=float,
        )
        lower_sign = -1.0 if axes[2, vertical_axis] > 0.0 else 1.0
        lower_face = book[signs[:, vertical_axis] == lower_sign]
        lower_face_center = center - half_extents[vertical_axis] * (
            upward_face_normal
        )
        face_xy = lower_face[:, :2]
        face_angles = np.arctan2(
            face_xy[:, 1] - np.mean(face_xy[:, 1]),
            face_xy[:, 0] - np.mean(face_xy[:, 0]),
        )
        face_xy = face_xy[np.argsort(face_angles)]

        shelf_edges_a = shelf[:, 1] - shelf[:, 0]
        shelf_edges_b = shelf[:, 2] - shelf[:, 0]
        shelf_normals = np.cross(shelf_edges_a, shelf_edges_b)
        normal_lengths = np.linalg.norm(shelf_normals, axis=1)
        usable = normal_lengths > 1e-12
        unit_normals = np.zeros_like(shelf_normals)
        unit_normals[usable] = (
            shelf_normals[usable] / normal_lengths[usable, np.newaxis]
        )
        upward_indices = np.flatnonzero(
            usable & (unit_normals[:, 2] >= upward_minimum)
        )

        projected: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for triangle_index in upward_indices:
            triangle = shelf[triangle_index]
            clipped = _clip_convex_polygon_to_triangle(
                face_xy, triangle[:, :2]
            )
            if _polygon_area(clipped) > 1e-12:
                projected.append(
                    (clipped, triangle, unit_normals[triangle_index])
                )

        metrics = {
            'book_lower_face_alignment': lower_face_alignment,
            'upward_shelf_triangle_count': float(len(upward_indices)),
            'projected_support_triangle_count': float(len(projected)),
            'contact_tolerance_m': contact_tolerance,
        }
        if projected:
            records = []
            for polygon, triangle, shelf_normal in projected:
                gaps = np.asarray(
                    [
                        _plane_height_at_xy(
                            lower_face_center, upward_face_normal, xy
                        )
                        - _plane_height_at_xy(
                            triangle[0], shelf_normal, xy
                        )
                        for xy in polygon
                    ],
                    dtype=float,
                )
                center_gap = _plane_height_at_xy(
                    lower_face_center,
                    upward_face_normal,
                    lower_face_center[:2],
                ) - _plane_height_at_xy(
                    triangle[0], shelf_normal, lower_face_center[:2]
                )
                records.append((polygon, gaps, float(center_gap)))

            closest_center_gap = min(records, key=lambda item: abs(item[2]))[2]
            same_panel_tolerance = max(1e-6, 0.1 * contact_tolerance)
            support_records = [
                record
                for record in records
                if abs(record[2] - closest_center_gap)
                <= same_panel_tolerance
            ]
            support_points = np.concatenate(
                [record[0] for record in support_records], axis=0
            )
            support_gaps = np.concatenate(
                [record[1] for record in support_records], axis=0
            )
            minimum_gap = float(np.min(support_gaps))
            maximum_gap = float(np.max(support_gaps))
            projected_overlap = float(
                np.max(support_points[:, 0]) - np.min(support_points[:, 0])
            )
            projected_area = float(
                sum(_polygon_area(record[0]) for record in support_records)
            )
            metrics.update(
                {
                    'near_contact_triangle_count': float(
                        len(support_records)
                    ),
                    'projected_shelf_overlap_m': projected_overlap,
                    'projected_support_area_m2': projected_area,
                    'minimum_support_gap_m': minimum_gap,
                    'maximum_support_gap_m': maximum_gap,
                    'book_lower_face_minimum_z_m': float(
                        np.min(lower_face[:, 2])
                    ),
                    'book_lower_face_maximum_z_m': float(
                        np.max(lower_face[:, 2])
                    ),
                }
            )
        else:
            minimum_gap = math.inf
            projected_overlap = 0.0
            projected_area = 0.0
            metrics.update(
                {
                    'near_contact_triangle_count': 0.0,
                    'projected_shelf_overlap_m': 0.0,
                    'projected_support_area_m2': 0.0,
                }
            )
    except (TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_geometry:{exc}', {})

    checks = (
        (
            lower_face_alignment >= face_alignment_minimum,
            'book_lower_face_not_upward_aligned',
        ),
        (bool(len(upward_indices)), 'upward_shelf_surface_unavailable'),
        (bool(len(projected)), 'book_lower_face_has_no_projected_support'),
        (
            minimum_gap >= -contact_tolerance,
            'book_shelf_penetration_exceeds_tolerance',
        ),
        (
            minimum_gap <= contact_tolerance,
            'book_lower_face_not_near_shelf',
        ),
        (projected_area > 1e-12, 'book_lower_face_has_no_projected_support'),
        (
            projected_overlap + 1e-12 >= minimum_overlap,
            'projected_shelf_overlap_below_minimum',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _validate_transform(
    value: Sequence[Sequence[float]], label: str
) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f'{label} must be a finite 4x4 transform')
    return transform


def _scene_sample(
    node: Any,
    environment: Any,
    *,
    newer_than: float | None = None,
) -> WorldSample:
    """Read target, neighbours, base, and measured arm from one scene frame."""

    deadline = time.monotonic() + SCENE_WAIT_TIMEOUT_S
    while True:
        scene = environment._read_scene()
        stamp = float(scene.observed_at)
        if newer_than is None or stamp > float(newer_than) + 1e-12:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Gazebo scene timestamp did not advance')
        time.sleep(0.02)

    base_transform = _validate_transform(
        scene.base_transform, 'base transform'
    )
    base = np.asarray(
        [
            base_transform[0, 3],
            base_transform[1, 3],
            math.atan2(base_transform[1, 0], base_transform[0, 0]),
        ],
        dtype=float,
    )
    all_corners = {
        str(name): np.asarray(book.corners, dtype=float).copy()
        for name, book in scene.books.items()
    }
    if TARGET_BOOK_MODEL not in all_corners:
        raise RuntimeError('Gazebo scene omitted the concrete target book')
    corners = all_corners[TARGET_BOOK_MODEL]
    if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
        raise RuntimeError('Gazebo target OBB is malformed')
    rotation = np.column_stack(
        (
            (corners[4] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[0]),
            (corners[2] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[1]),
            (corners[1] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[2]),
        )
    )
    book = BookSnapshot(
        np.mean(corners, axis=0),
        _quaternion_from_rotation(rotation),
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )
    arm = np.asarray(node._measured_left_solution(), dtype=float)
    if arm.shape != (8,) or not np.all(np.isfinite(arm)):
        raise RuntimeError('measured torso/left-arm state is malformed')
    aperture = float(node.joints.get('gripper_left_finger_joint', math.nan))
    if not math.isfinite(aperture):
        raise RuntimeError('measured gripper aperture is unavailable')
    return WorldSample(
        book=book,
        base=base,
        arm=arm,
        hand_world=world_hand_pose(base, node.chain.forward(arm)),
        aperture_m=aperture,
        observed_at=stamp,
        corners=corners,
        all_book_corners=all_corners,
        shelf_triangles=np.asarray(scene.shelf_triangles, dtype=float).copy(),
    )


def _outward_world(base: Sequence[float]) -> np.ndarray:
    pose = np.asarray(base, dtype=float)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError('base pose must contain finite x, y, and yaw')
    return np.asarray([-math.cos(pose[2]), -math.sin(pose[2]), 0.0])


def _outward_in_base(base: Sequence[float]) -> np.ndarray:
    pose = np.asarray(base, dtype=float)
    outward = _outward_world(pose)
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    local = np.zeros(3, dtype=float)
    local[:2] = rotation.T @ outward[:2]
    return local


def _book_vertical_axis(
    reference: BookSnapshot,
    current: BookSnapshot,
) -> tuple[np.ndarray, np.ndarray]:
    first = quaternion_matrix(reference.quaternion)
    second = quaternion_matrix(current.quaternion)
    vertical_index = int(np.argmax(np.abs(first[2, :])))
    sign = 1.0 if first[2, vertical_index] >= 0.0 else -1.0
    initial_up = sign * first[:, vertical_index]
    current_up = sign * second[:, vertical_index]
    return initial_up, current_up


def settle_attitude_metrics(
    reference: BookSnapshot,
    current: BookSnapshot,
    reference_base: Sequence[float],
) -> dict[str, float]:
    """Return total tilt and the component outside the radial x-z plane."""

    initial_up, current_up = _book_vertical_axis(reference, current)
    tilt = math.acos(float(np.clip(np.dot(initial_up, current_up), -1.0, 1.0)))
    outward = _outward_world(reference_base)
    lateral = np.asarray([-outward[1], outward[0], 0.0], dtype=float)
    off_axis = math.asin(
        float(np.clip(abs(np.dot(current_up, lateral)), 0.0, 1.0))
    )
    return {
        'drop_m': max(0.0, float(reference.position[2] - current.position[2])),
        'tilt_rad': tilt,
        'off_axis_tilt_rad': off_axis,
        'center_y_drift_m': abs(
            float(current.position[1] - reference.position[1])
        ),
    }


def edge_leg_metrics(
    before: WorldSample,
    after: WorldSample,
) -> dict[str, float]:
    """Measure rigid following for a one-millimetre commanded edge leg."""

    outward = _outward_world(before.base)
    book_delta = after.book.position - before.book.position
    hand_delta = after.hand_world[:3, 3] - before.hand_world[:3, 3]
    return {
        'book_outward_progress_m': float(np.dot(book_delta, outward)),
        'hand_outward_progress_m': float(np.dot(hand_delta, outward)),
        'book_hand_translation_mismatch_m': float(
            np.linalg.norm(book_delta - hand_delta)
        ),
        'book_rotation_rad': quaternion_distance(
            before.book.quaternion, after.book.quaternion
        ),
        'hand_rotation_rad': rotation_matrix_distance(
            before.hand_world[:3, :3], after.hand_world[:3, :3]
        ),
        'rear_x_m': float(after.book.maximum[0]),
        'shelf_overlap_m': float(after.book.maximum[0] - SHELF_FRONT_X_M),
        'base_drift_m': float(
            np.linalg.norm(after.base[:2] - before.base[:2])
        ),
        'base_yaw_drift_rad': _angle_error(after.base[2], before.base[2]),
        'aperture_error_m': abs(
            float(after.aperture_m - EXPECTED_CAGED_APERTURE_M)
        ),
    }


def commanded_edge_leg_guard(
    before: WorldSample,
    after: WorldSample,
    *,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Require one completed pre-settle leg to remain a rigid 1 mm carry."""

    try:
        metrics = edge_leg_metrics(before, after)
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    checks = (
        (left_contact, 'left_contact_missing'),
        (right_contact, 'right_contact_missing'),
        (palm_contact, 'palm_contact_missing'),
        (not unexpected_contacts, 'unexpected_contact'),
        (
            metrics['aperture_error_m']
            <= CAGE_MEASURED_TARGET_TOLERANCE_M,
            'cage_aperture_changed',
        ),
        (
            EDGE_PROGRESS_MINIMUM_M
            <= metrics['hand_outward_progress_m']
            <= EDGE_PROGRESS_MAXIMUM_M,
            'hand_progress_out_of_bounds',
        ),
        (
            EDGE_PROGRESS_MINIMUM_M
            <= metrics['book_outward_progress_m']
            <= EDGE_PROGRESS_MAXIMUM_M,
            'book_progress_out_of_bounds',
        ),
        (
            metrics['book_hand_translation_mismatch_m']
            <= EDGE_HAND_BOOK_MISMATCH_LIMIT_M,
            'nonrigid_translation_started',
        ),
        (
            metrics['book_rotation_rad'] <= EDGE_BOOK_ROTATION_LIMIT_RAD,
            'nonrigid_rotation_started',
        ),
        (
            metrics['hand_rotation_rad'] <= 0.004,
            'hand_rotated_during_translation',
        ),
        (
            metrics['base_drift_m'] <= STEP_BASE_POSITION_LIMIT_M,
            'base_moved_during_step',
        ),
        (
            metrics['base_yaw_drift_rad'] <= STEP_BASE_YAW_LIMIT_RAD,
            'base_rotated_during_step',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def settle_trigger_guard(
    step18_book: BookSnapshot,
    before: WorldSample,
    current: WorldSample,
    *,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Classify the one allowed transition from commanded carry to settling."""

    try:
        metrics = edge_leg_metrics(before, current)
        metrics.update(
            settle_attitude_metrics(step18_book, current.book, before.base)
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    basic = (
        (left_contact, 'left_contact_missing'),
        (right_contact, 'right_contact_missing'),
        (palm_contact, 'palm_contact_missing'),
        (not unexpected_contacts, 'unexpected_contact'),
        (
            metrics['aperture_error_m']
            <= CAGE_MEASURED_TARGET_TOLERANCE_M,
            'cage_aperture_changed',
        ),
        (
            -0.00010
            <= metrics['hand_outward_progress_m']
            <= EDGE_PROGRESS_MAXIMUM_M + 0.00010,
            'hand_progress_out_of_bounds',
        ),
        (
            metrics['base_drift_m'] <= STEP_BASE_POSITION_LIMIT_M,
            'base_moved_during_step',
        ),
        (
            metrics['base_yaw_drift_rad'] <= STEP_BASE_YAW_LIMIT_RAD,
            'base_rotated_during_step',
        ),
        (metrics['drop_m'] <= SETTLE_DROP_LIMIT_M, 'settle_drop_limit'),
        (
            metrics['tilt_rad'] <= SETTLE_PEAK_TILT_LIMIT_RAD,
            'settle_peak_tilt_limit',
        ),
        (
            metrics['center_y_drift_m'] <= SETTLE_CENTER_Y_DRIFT_LIMIT_M,
            'settle_center_y_drift',
        ),
        (
            metrics['off_axis_tilt_rad'] <= SETTLE_OFF_AXIS_TILT_LIMIT_RAD,
            'settle_off_axis_tilt',
        ),
    )
    for accepted, reason in basic:
        if not accepted:
            return GuardResult(False, reason, metrics)

    nonrigid = bool(
        metrics['book_hand_translation_mismatch_m']
        > EDGE_HAND_BOOK_MISMATCH_LIMIT_M
        or metrics['book_rotation_rad'] > EDGE_BOOK_ROTATION_LIMIT_RAD
    )
    if nonrigid:
        if metrics['shelf_overlap_m'] > PREMATURE_SETTLE_OVERLAP_M:
            return GuardResult(False, 'premature_settle', metrics)
        return GuardResult(True, 'nonrigid_motion_started', metrics)
    if metrics['rear_x_m'] <= EDGE_REAR_CLEARANCE_X_M:
        return GuardResult(True, 'rear_clearance_reached', metrics)
    return GuardResult(False, 'no_settle_trigger', metrics)


def non_target_book_motion_guard(
    reference: Mapping[str, Sequence[Sequence[float]]],
    current: Mapping[str, Sequence[Sequence[float]]],
    *,
    target_book: str = TARGET_BOOK_MODEL,
) -> GuardResult:
    """Reject any missing, added, or displaced non-target shelf book."""

    try:
        expected_names = set(reference) - {target_book}
        current_names = set(current) - {target_book}
        if expected_names != current_names:
            return GuardResult(False, 'non_target_book_set_changed', {})
        maximum_center = 0.0
        maximum_corner = 0.0
        for name in sorted(expected_names):
            first = np.asarray(reference[name], dtype=float)
            second = np.asarray(current[name], dtype=float)
            if (
                first.shape != (8, 3)
                or second.shape != (8, 3)
                or not np.all(np.isfinite(first))
                or not np.all(np.isfinite(second))
            ):
                raise ValueError(f'book {name!r} has malformed corners')
            maximum_center = max(
                maximum_center,
                float(
                    np.linalg.norm(
                        np.mean(second, axis=0) - np.mean(first, axis=0)
                    )
                ),
            )
            maximum_corner = max(
                maximum_corner,
                float(np.max(np.linalg.norm(second - first, axis=1))),
            )
        metrics = {
            'non_target_maximum_center_motion_m': maximum_center,
            'non_target_maximum_corner_motion_m': maximum_corner,
        }
    except (TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    if max(maximum_center, maximum_corner) > NON_TARGET_BOOK_MOTION_LIMIT_M:
        return GuardResult(False, 'non_target_book_moved', metrics)
    return GuardResult(True, 'ok', metrics)


def settle_envelope_guard(
    reference: WorldSample,
    hold_arm: Sequence[float],
    current: WorldSample,
    *,
    unexpected_contacts: bool,
) -> GuardResult:
    """Bound the dynamic settle while all robot commands remain fixed."""

    try:
        fixed_arm = np.asarray(hold_arm, dtype=float)
        if fixed_arm.shape != (8,) or not np.all(np.isfinite(fixed_arm)):
            raise ValueError('held arm state must be a finite eight-vector')
        metrics = settle_attitude_metrics(
            reference.book, current.book, reference.base
        )
        metrics.update(
            {
                'arm_hold_error_rad': float(
                    np.max(np.abs(current.arm - fixed_arm))
                ),
                'base_hold_error_m': float(
                    np.linalg.norm(current.base[:2] - reference.base[:2])
                ),
                'base_yaw_hold_error_rad': _angle_error(
                    current.base[2], reference.base[2]
                ),
                'aperture_hold_error_m': abs(
                    float(current.aperture_m - EXPECTED_CAGED_APERTURE_M)
                ),
            }
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    checks = (
        (not unexpected_contacts, 'unexpected_contact'),
        (metrics['drop_m'] <= SETTLE_DROP_LIMIT_M, 'settle_drop_limit'),
        (
            metrics['tilt_rad'] <= SETTLE_PEAK_TILT_LIMIT_RAD,
            'settle_peak_tilt_limit',
        ),
        (
            metrics['center_y_drift_m'] <= SETTLE_CENTER_Y_DRIFT_LIMIT_M,
            'settle_center_y_drift',
        ),
        (
            metrics['off_axis_tilt_rad'] <= SETTLE_OFF_AXIS_TILT_LIMIT_RAD,
            'settle_off_axis_tilt',
        ),
        (
            metrics['arm_hold_error_rad'] <= SETTLE_ARM_HOLD_LIMIT_RAD,
            'arm_moved_during_settle',
        ),
        (
            metrics['base_hold_error_m'] <= SETTLE_BASE_HOLD_LIMIT_M,
            'base_moved_during_settle',
        ),
        (
            metrics['base_yaw_hold_error_rad']
            <= SETTLE_BASE_YAW_HOLD_LIMIT_RAD,
            'base_rotated_during_settle',
        ),
        (
            metrics['aperture_hold_error_m']
            <= SETTLE_APERTURE_HOLD_LIMIT_M,
            'aperture_moved_during_settle',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def commanded_motion_envelope_guard(
    reference: WorldSample,
    current: WorldSample,
    *,
    unexpected_contacts: bool,
) -> GuardResult:
    """Apply settle hard caps during a commanded leg without freezing it."""

    try:
        metrics = settle_attitude_metrics(
            reference.book, current.book, reference.base
        )
        metrics.update(
            {
                'base_hold_error_m': float(
                    np.linalg.norm(current.base[:2] - reference.base[:2])
                ),
                'base_yaw_hold_error_rad': _angle_error(
                    current.base[2], reference.base[2]
                ),
                'aperture_hold_error_m': abs(
                    float(current.aperture_m - EXPECTED_CAGED_APERTURE_M)
                ),
            }
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    checks = (
        (not unexpected_contacts, 'unexpected_contact'),
        (metrics['drop_m'] <= SETTLE_DROP_LIMIT_M, 'settle_drop_limit'),
        (
            metrics['tilt_rad'] <= SETTLE_PEAK_TILT_LIMIT_RAD,
            'settle_peak_tilt_limit',
        ),
        (
            metrics['center_y_drift_m'] <= SETTLE_CENTER_Y_DRIFT_LIMIT_M,
            'settle_center_y_drift',
        ),
        (
            metrics['off_axis_tilt_rad'] <= SETTLE_OFF_AXIS_TILT_LIMIT_RAD,
            'settle_off_axis_tilt',
        ),
        (
            metrics['base_hold_error_m'] <= SETTLE_BASE_HOLD_LIMIT_M,
            'base_moved_during_transfer',
        ),
        (
            metrics['base_yaw_hold_error_rad']
            <= SETTLE_BASE_YAW_HOLD_LIMIT_RAD,
            'base_rotated_during_transfer',
        ),
        (
            metrics['aperture_hold_error_m']
            <= SETTLE_APERTURE_HOLD_LIMIT_M,
            'aperture_moved_during_transfer',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _maximum_pairwise_vector_span(values: Sequence[np.ndarray]) -> float:
    maximum = 0.0
    for first_index, first in enumerate(values):
        for second in values[first_index + 1:]:
            maximum = max(maximum, float(np.linalg.norm(second - first)))
    return maximum


def settle_stability_guard(samples: Sequence[SettleSample]) -> GuardResult:
    """Prove a 0.5 s trailing window is genuinely stationary."""

    try:
        observations = tuple(samples)
        if len(observations) < SETTLE_MINIMUM_STABILITY_SAMPLES:
            return GuardResult(False, 'insufficient_stability_samples', {})
        stamps = np.asarray(
            [sample.observed_at for sample in observations], dtype=float
        )
        if not np.all(np.isfinite(stamps)) or np.any(np.diff(stamps) <= 0.0):
            raise ValueError('settle timestamps must be finite and increasing')
        duration = float(stamps[-1] - stamps[0])
        center_span = _maximum_pairwise_vector_span(
            [sample.book.position for sample in observations]
        )
        corner_span = 0.0
        for first_index, first in enumerate(observations):
            for second in observations[first_index + 1:]:
                corner_span = max(
                    corner_span,
                    float(
                        np.max(
                            np.linalg.norm(
                                second.corners - first.corners, axis=1
                            )
                        )
                    ),
                )
        attitude_span = 0.0
        for first_index, first in enumerate(observations):
            for second in observations[first_index + 1:]:
                attitude_span = max(
                    attitude_span,
                    quaternion_distance(
                        first.book.quaternion, second.book.quaternion
                    ),
                )
        arm_span = max(
            float(np.max(np.abs(second.arm - first.arm)))
            for first_index, first in enumerate(observations)
            for second in observations[first_index + 1:]
        )
        base_span = _maximum_pairwise_vector_span(
            [sample.base[:2] for sample in observations]
        )
        base_yaw_span = max(
            _angle_error(second.base[2], first.base[2])
            for first_index, first in enumerate(observations)
            for second in observations[first_index + 1:]
        )
        aperture_span = float(
            max(sample.aperture_m for sample in observations)
            - min(sample.aperture_m for sample in observations)
        )
        metrics = {
            'stability_duration_s': duration,
            'stability_center_span_m': center_span,
            'stability_corner_span_m': corner_span,
            'stability_attitude_span_rad': attitude_span,
            'stability_arm_span_rad': arm_span,
            'stability_base_span_m': base_span,
            'stability_base_yaw_span_rad': base_yaw_span,
            'stability_aperture_span_m': aperture_span,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    checks = (
        (
            duration >= SETTLE_STABILITY_WINDOW_S - 1e-9,
            'stability_window_too_short',
        ),
        (center_span <= SETTLE_CENTER_SPAN_LIMIT_M, 'center_not_stable'),
        (corner_span <= SETTLE_CORNER_SPAN_LIMIT_M, 'corners_not_stable'),
        (
            attitude_span <= SETTLE_ATTITUDE_SPAN_LIMIT_RAD,
            'attitude_not_stable',
        ),
        (arm_span <= SETTLE_ARM_SPAN_LIMIT_RAD, 'arm_not_stable'),
        (base_span <= SETTLE_BASE_SPAN_LIMIT_M, 'base_not_stable'),
        (
            base_yaw_span <= SETTLE_BASE_YAW_SPAN_LIMIT_RAD,
            'base_yaw_not_stable',
        ),
        (
            aperture_span <= SETTLE_APERTURE_SPAN_LIMIT_M,
            'aperture_not_stable',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _trailing_stability_window(
    history: Sequence[SettleSample],
) -> list[SettleSample]:
    """Return the shortest suffix spanning at least the required 0.5 s."""

    observations = list(history)
    if not observations:
        return []
    cutoff = observations[-1].observed_at - SETTLE_STABILITY_WINDOW_S
    first_index = 0
    for index, sample in enumerate(observations):
        if sample.observed_at <= cutoff + 1e-12:
            first_index = index
        else:
            break
    return observations[first_index:]


def padded_book_shelf_clearance_guard(
    corners: Sequence[Sequence[float]],
    shelf_triangles: Sequence[Sequence[Sequence[float]]],
    *,
    padding_m: float = POST_SETTLE_PADDING_M,
    shelf_front_x_m: float = SHELF_FRONT_X_M,
) -> GuardResult:
    """Require the physical OBB plus padding to be wholly shelf-disjoint."""

    try:
        physical = np.asarray(corners, dtype=float)
        shelf = np.asarray(shelf_triangles, dtype=float)
        padding = float(padding_m)
        front = float(shelf_front_x_m)
        if (
            physical.shape != (8, 3)
            or shelf.ndim != 3
            or shelf.shape[1:] != (3, 3)
            or not np.all(np.isfinite(physical))
            or not np.all(np.isfinite(shelf))
            or not math.isfinite(padding)
            or padding < 0.0
            or not math.isfinite(front)
        ):
            raise ValueError('clearance geometry is malformed')
        inflated = _inflated_ordered_corners(physical, padding)
        rear_clearance = float(front - np.max(physical[:, 0]))
        padded_rear_clearance = float(front - np.max(inflated[:, 0]))
        intersects = oriented_box_intersects_triangles(
            # The shelf STL contains separately meshed panels, so a global
            # watertight-containment assumption is invalid.  Surface crossing
            # plus the independently required front-plane margin proves the
            # book is outside in the only permitted (outward) direction.
            inflated,
            shelf,
            closed_surface=False,
        )
        metrics = {
            'rear_clearance_m': rear_clearance,
            'padded_rear_clearance_m': padded_rear_clearance,
            'padded_shelf_intersection': float(bool(intersects)),
        }
    except (TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_geometry:{exc}', {})
    if rear_clearance + 1e-12 < padding:
        return GuardResult(False, 'rear_clearance_below_padding', metrics)
    if intersects:
        return GuardResult(False, 'padded_book_intersects_shelf', metrics)
    return GuardResult(True, 'ok', metrics)


def choose_shortest_clearance_candidate(
    candidate_corners: Mapping[float, Sequence[Sequence[float]]],
    shelf_triangles: Sequence[Sequence[Sequence[float]]],
    *,
    padding_m: float = POST_SETTLE_PADDING_M,
    shelf_front_x_m: float = SHELF_FRONT_X_M,
) -> tuple[float, GuardResult]:
    """Select only the shortest audited 5/10 mm shelf-clear candidate."""

    keys = tuple(float(value) for value in candidate_corners)
    if set(keys) != set(POST_SETTLE_CANDIDATE_DISTANCES_M):
        raise ValueError('candidate set must contain exactly 5 mm and 10 mm')
    last = GuardResult(False, 'no_candidate_checked', {})
    for distance in POST_SETTLE_CANDIDATE_DISTANCES_M:
        last = padded_book_shelf_clearance_guard(
            candidate_corners[distance],
            shelf_triangles,
            padding_m=padding_m,
            shelf_front_x_m=shelf_front_x_m,
        )
        if last.safe:
            return distance, last
    raise RuntimeError(
        'neither post-settle leg yields padded shelf clearance; '
        f'last rejection={last.reason}'
    )


def _exact_contacts(
    node: Any, *, max_age: float = 0.22
) -> tuple[bool, bool, bool]:
    left, right = node.probe_exact_finger_sides(max_age=max_age)
    palm = node.probe_exact_palm_contact(max_age=max_age)
    return bool(left), bool(right), bool(palm)


def _hazard_is_latched(node: Any) -> bool:
    return bool(
        _unexpected_pairs(node)
        or getattr(node, '_target_robot_contact_latched', False)
        or getattr(node, '_payload_hazard_latched', None) is not None
        or getattr(node, 'probe_transfer_watchdog_reason', None) is not None
    )


def _solve_translation(
    node: Any,
    start_arm: Sequence[float],
    outward_in_base: Sequence[float],
    distance_m: float,
) -> np.ndarray:
    start = np.asarray(start_arm, dtype=float)
    outward = np.asarray(outward_in_base, dtype=float)
    distance = float(distance_m)
    if (
        start.shape != (8,)
        or outward.shape != (3,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(outward))
        or not math.isfinite(distance)
        or distance <= 0.0
    ):
        raise ValueError('translation IK input is invalid')
    reference = node.chain.forward(start)
    target = pose_matrix(
        reference[:3, 3] + outward * distance,
        reference[:3, :3],
    )
    solution, _ = node.chain.solve(
        target,
        [start],
        position_tolerance=0.00025,
        orientation_tolerance=0.0010,
        max_iterations=400,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solution is None:
        raise RuntimeError(f'outward IK failed for {distance:.3f} m')
    result = np.asarray(solution, dtype=float)
    achieved = node.chain.forward(result)
    error = node.chain.pose_error(achieved, target)
    if (
        result.shape != (8,)
        or not np.all(np.isfinite(result))
        or float(np.max(np.abs(result[1:] - start[1:]))) > 0.16
        or float(np.linalg.norm(error[:3])) > 0.00025
        or float(np.linalg.norm(error[3:])) > 0.0010
    ):
        raise RuntimeError(
            'outward IK endpoint is not an exact continuous solution'
        )
    return result


def _predicted_corners(
    reference: WorldSample,
    target_arm: Sequence[float],
    node: Any,
) -> np.ndarray:
    hand = reference.hand_world
    relative = (reference.corners - hand[:3, 3]) @ hand[:3, :3]
    target_hand = world_hand_pose(
        reference.base, node.chain.forward(target_arm)
    )
    return relative @ target_hand[:3, :3].T + target_hand[:3, 3]


def _probe_node_types(runtime: SimpleNamespace) -> tuple[type, type]:
    BaseProbe, ProbeNavigation = _probe_types(runtime)

    class NaturalSettleProbeNode(BaseProbe):
        """Add the natural-settle transfer cancel latch."""

        def __init__(self) -> None:
            self.probe_transfer_watchdog_active = False
            self.probe_transfer_watchdog_reason: str | None = None
            super().__init__()

        def reset_probe_audit(self) -> None:
            super().reset_probe_audit()

            def reset_extra() -> None:
                self.probe_transfer_watchdog_active = False
                self.probe_transfer_watchdog_reason = None

            self._with_probe_lock(reset_extra)

        def begin_transfer_watchdog(self) -> None:
            def begin() -> None:
                self.probe_transfer_watchdog_reason = None
                self.probe_transfer_watchdog_active = True

            self._with_probe_lock(begin)

        def latch_transfer_watchdog(self, reason: str) -> None:
            value = str(reason)

            def latch() -> None:
                if self.probe_transfer_watchdog_reason is None:
                    self.probe_transfer_watchdog_reason = value

            self._with_probe_lock(latch)

        def end_transfer_watchdog(self) -> str | None:
            def end() -> str | None:
                self.probe_transfer_watchdog_active = False
                return self.probe_transfer_watchdog_reason

            return self._with_probe_lock(end)

        def _payload_hazard_reason(
            self, *, max_age: float = 0.20
        ) -> str | None:
            reason = super()._payload_hazard_reason(max_age=max_age)
            active, latched = self._with_probe_lock(
                lambda: (
                    bool(self.probe_transfer_watchdog_active),
                    self.probe_transfer_watchdog_reason,
                )
            )
            if reason is not None:
                if active:
                    self.latch_transfer_watchdog(reason)
                return reason
            if active:
                left, right, palm = _exact_contacts(self, max_age=max_age)
                if not left or not right or not palm:
                    reason = 'exact_three_point_contact_lost'
                    self.latch_transfer_watchdog(reason)
                    return reason
            latched = self._with_probe_lock(
                lambda: self.probe_transfer_watchdog_reason
            )
            return None if latched is None else str(latched)

    return NaturalSettleProbeNode, ProbeNavigation


def _watch_commanded_leg(
    node: Any,
    runtime: SimpleNamespace,
    step18: WorldSample,
    before: WorldSample,
    stop: threading.Event,
    result: dict[str, Any],
    *,
    allow_settle_trigger: bool,
) -> None:
    stamp = before.observed_at
    try:
        while not stop.is_set():
            sample = _scene_sample(
                node, runtime.environment_preflight, newer_than=stamp
            )
            stamp = sample.observed_at
            result['latest_sample'] = sample
            left, right, palm = _exact_contacts(node)
            neighbour = non_target_book_motion_guard(
                step18.all_book_corners, sample.all_book_corners
            )
            envelope = commanded_motion_envelope_guard(
                step18,
                sample,
                unexpected_contacts=_hazard_is_latched(node),
            )
            if not neighbour.safe:
                reason = neighbour.reason
                metrics = neighbour.metrics
            elif not envelope.safe:
                reason = envelope.reason
                metrics = envelope.metrics
            elif not allow_settle_trigger:
                metrics = edge_leg_metrics(before, sample)
                if (
                    metrics['book_hand_translation_mismatch_m']
                    > EDGE_HAND_BOOK_MISMATCH_LIMIT_M
                    or metrics['book_rotation_rad']
                    > EDGE_BOOK_ROTATION_LIMIT_RAD
                ):
                    reason = 'post_settle_nonrigid_motion'
                else:
                    continue
            else:
                trigger = settle_trigger_guard(
                    step18.book,
                    before,
                    sample,
                    left_contact=left,
                    right_contact=right,
                    palm_contact=palm,
                    unexpected_contacts=_hazard_is_latched(node),
                )
                metrics = trigger.metrics
                if trigger.safe:
                    reason = trigger.reason
                    if reason in SOFT_SETTLE_TRIGGER_REASONS:
                        if result.get('reason') is None:
                            result['reason'] = reason
                            result['sample'] = sample
                            result['metrics'] = dict(metrics)
                        # This is the expected transition, not a controller
                        # hazard.  Keep watching the remainder of the bounded
                        # 1 mm leg so any later hard fault still cancels it.
                        continue
                    reason = f'unexpected_safe_settle_trigger:{reason}'
                elif (
                    not trigger.safe
                    and trigger.reason != 'no_settle_trigger'
                ):
                    reason = trigger.reason
                else:
                    continue
            result['reason'] = reason
            result['sample'] = sample
            result['metrics'] = dict(metrics)
            node.latch_transfer_watchdog(reason)
            return
    except BaseException as exc:
        reason = f'watchdog_error:{type(exc).__name__}:{exc}'
        result['reason'] = reason
        result['sample'] = result.get('latest_sample')
        result['metrics'] = {}
        node.latch_transfer_watchdog(reason)


def _run_arm_leg_with_watchdog(
    node: Any,
    runtime: SimpleNamespace,
    step18: WorldSample,
    before: WorldSample,
    target: np.ndarray,
    duration: float,
    event: str,
    *,
    allow_settle_trigger: bool,
) -> tuple[bool, bool, LegWatchResult]:
    stop = threading.Event()
    result: dict[str, Any] = {
        'reason': None,
        'sample': None,
        'latest_sample': None,
        'metrics': {},
    }
    node.begin_transfer_watchdog()
    monitor = threading.Thread(
        target=_watch_commanded_leg,
        args=(node, runtime, step18, before, stop, result),
        kwargs={'allow_settle_trigger': allow_settle_trigger},
        name='natural-settle-live-watchdog',
        daemon=True,
    )
    monitor.start()
    try:
        legs = ((target, duration, event),)
        goal, total = node._make_retained_arm_trajectory_goal(legs)
        moved, cancelled = node._send_retained_arm_trajectory(
            goal, total, legs, event
        )
    finally:
        stop.set()
        monitor.join(timeout=6.0)
        reason = node.end_transfer_watchdog()
    if monitor.is_alive():
        raise RuntimeError('live Gazebo watchdog did not stop')
    return moved, cancelled, LegWatchResult(
        reason=reason or result.get('reason'),
        sample=result.get('sample') or result.get('latest_sample'),
        metrics=dict(result.get('metrics', {})),
    )


def _preflight_translation(
    node: Any,
    runtime: SimpleNamespace,
    before: WorldSample,
    target: np.ndarray,
) -> None:
    attached = _attached_corners(node, before.book, before.base, before.arm)
    if not node._carried_robot_transition_is_safe(
        before.arm, target, attached
    ):
        raise RuntimeError(
            'actual tilted OBB/robot sweep rejected outward leg'
        )
    node._held_book_corners = attached.copy()
    _preflight_leg(node, runtime.environment_preflight, target)
    _fresh_cage_gate(node)
    if _hazard_is_latched(node):
        raise RuntimeError('contact hazard appeared during outward preflight')


def _edge_to_settle_trigger(
    node: Any,
    runtime: SimpleNamespace,
    step18: WorldSample,
) -> tuple[WorldSample, str, int]:
    current = step18
    outward = _outward_in_base(step18.base)
    expected_cancel_reasons = SOFT_SETTLE_TRIGGER_REASONS
    for leg_index in range(1, EDGE_MAXIMUM_LEGS + 1):
        target = _solve_translation(
            node, current.arm, outward, EDGE_LEG_DISTANCE_M
        )
        _preflight_translation(node, runtime, current, target)
        moved, cancelled, watch = _run_arm_leg_with_watchdog(
            node,
            runtime,
            step18,
            current,
            target,
            EDGE_LEG_DURATION_S,
            EDGE_TRANSFER_EVENT,
            allow_settle_trigger=True,
        )
        if not moved or cancelled:
            raise RuntimeError(
                f'edge leg {leg_index} stopped unsafely: {watch.reason}'
            )

        if watch.reason is not None:
            if watch.reason not in expected_cancel_reasons:
                raise RuntimeError(
                    f'edge leg {leg_index} watchdog failed: {watch.reason}'
                )
            sample = watch.sample
            if sample is None:
                sample = _scene_sample(
                    node,
                    runtime.environment_preflight,
                    newer_than=current.observed_at,
                )
            return sample, str(watch.reason), leg_index

        endpoint = node._wait_for_retained_endpoint(
            target,
            command=EDGE_TRANSFER_EVENT,
            phase=EDGE_TRANSFER_EVENT,
            leg=leg_index,
        )
        if endpoint is None:
            raise RuntimeError(
                f'edge endpoint {leg_index} was not reached safely'
            )
        _fresh_cage_gate(node)
        after = _scene_sample(
            node,
            runtime.environment_preflight,
            newer_than=current.observed_at,
        )
        left, right, palm = _exact_contacts(node)
        guard = commanded_edge_leg_guard(
            current,
            after,
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_hazard_is_latched(node),
        )
        neighbour = non_target_book_motion_guard(
            step18.all_book_corners, after.all_book_corners
        )
        _emit(
            'edge_leg',
            leg=leg_index,
            passed=guard.safe and neighbour.safe,
            reason=guard.reason if not guard.safe else neighbour.reason,
            book_position=after.book.position.tolist(),
            book_maximum=after.book.maximum.tolist(),
            **guard.metrics,
            **neighbour.metrics,
        )
        if not neighbour.safe:
            raise RuntimeError(neighbour.reason)
        if guard.safe and after.book.maximum[0] > EDGE_REAR_CLEARANCE_X_M:
            current = after
            continue
        trigger = settle_trigger_guard(
            step18.book,
            current,
            after,
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_hazard_is_latched(node),
        )
        if not trigger.safe:
            raise RuntimeError(
                f'edge transfer did not enter a valid settle: {trigger.reason}'
            )
        _emit(
            'settle_trigger',
            leg=leg_index,
            reason=trigger.reason,
            goal_cancelled=False,
            **trigger.metrics,
        )
        return after, trigger.reason, leg_index
    raise RuntimeError('edge transfer exhausted its 20 mm hard limit')


def _settle_sample(sample: WorldSample) -> SettleSample:
    return SettleSample(
        observed_at=sample.observed_at,
        book=sample.book,
        corners=sample.corners,
        arm=sample.arm,
        base=sample.base,
        aperture_m=sample.aperture_m,
    )


def _wait_for_single_settle(
    node: Any,
    runtime: SimpleNamespace,
    step18: WorldSample,
    trigger: WorldSample,
) -> tuple[WorldSample, GuardResult, GuardResult]:
    hold_arm = np.asarray(node._measured_left_solution(), dtype=float)
    start_stamp = float(trigger.observed_at)
    stamp = start_stamp
    history: list[SettleSample] = [_settle_sample(trigger)]
    contact_dwell_completed = False
    peak_tilt = settle_attitude_metrics(
        step18.book, trigger.book, step18.base
    )['tilt_rad']
    while stamp - start_stamp <= SETTLE_TIMEOUT_S + 1e-12:
        current = _scene_sample(
            node, runtime.environment_preflight, newer_than=stamp
        )
        stamp = current.observed_at
        if stamp - start_stamp > SETTLE_TIMEOUT_S + 1e-12:
            raise RuntimeError(
                'one natural settle exceeded 2.0 simulated seconds'
            )
        envelope = settle_envelope_guard(
            step18,
            hold_arm,
            current,
            unexpected_contacts=_hazard_is_latched(node),
        )
        neighbour = non_target_book_motion_guard(
            step18.all_book_corners, current.all_book_corners
        )
        if not envelope.safe:
            raise RuntimeError(f'settle envelope failed: {envelope.reason}')
        if not neighbour.safe:
            raise RuntimeError(
                f'settle neighbour gate failed: {neighbour.reason}'
            )
        peak_tilt = max(peak_tilt, envelope.metrics['tilt_rad'])
        if peak_tilt > SETTLE_PEAK_TILT_LIMIT_RAD:
            raise RuntimeError('settle exceeded the 30 degree peak-tilt cap')
        history.append(_settle_sample(current))
        trailing = _trailing_stability_window(history)
        stability = settle_stability_guard(trailing)
        final_tilt = envelope.metrics['tilt_rad']
        if (
            stability.safe
            and SETTLE_FINAL_TILT_MINIMUM_RAD
            <= final_tilt
            <= SETTLE_FINAL_TILT_MAXIMUM_RAD
        ):
            if not contact_dwell_completed:
                _fresh_cage_gate(node)
                contact_dwell_completed = True
                continue
            left, right, palm = _exact_contacts(node, max_age=0.18)
            if not left or not right or not palm:
                contact_dwell_completed = False
                continue
            _emit(
                'natural_settle_stable',
                left_exact_target_contact=left,
                right_exact_target_contact=right,
                exact_target_palm_contact=palm,
                peak_tilt_rad=peak_tilt,
                **envelope.metrics,
                **stability.metrics,
                **neighbour.metrics,
            )
            return current, envelope, stability
        if stamp - start_stamp >= SETTLE_TIMEOUT_S:
            break
    raise RuntimeError(
        'one natural settle did not become stable within 2.0 simulated seconds'
    )


def _post_settle_clearance_leg(
    node: Any,
    runtime: SimpleNamespace,
    step18: WorldSample,
    settled: WorldSample,
) -> tuple[WorldSample, float, GuardResult]:
    outward = _outward_in_base(settled.base)
    solutions: dict[float, np.ndarray] = {}
    predictions: dict[float, np.ndarray] = {}
    for distance in POST_SETTLE_CANDIDATE_DISTANCES_M:
        solution = _solve_translation(node, settled.arm, outward, distance)
        solutions[distance] = solution
        predictions[distance] = _predicted_corners(settled, solution, node)
    scene = runtime.environment_preflight._read_scene()
    distance, predicted_clearance = choose_shortest_clearance_candidate(
        predictions, scene.shelf_triangles
    )
    target = solutions[distance]
    # Only the selected shortest leg reaches the expensive/live preflight.
    _preflight_translation(node, runtime, settled, target)
    moved, cancelled, watch = _run_arm_leg_with_watchdog(
        node,
        runtime,
        step18,
        settled,
        target,
        POST_SETTLE_DURATION_PER_5MM_S * (distance / 0.005),
        POST_SETTLE_EVENT,
        allow_settle_trigger=False,
    )
    if not moved or cancelled:
        raise RuntimeError(
            f'post-settle clearance leg stopped unsafely: {watch.reason}'
        )
    if watch.reason is not None:
        raise RuntimeError(
            f'post-settle clearance watchdog rejected the leg: {watch.reason}'
        )
    endpoint = node._wait_for_retained_endpoint(
        target,
        command=POST_SETTLE_EVENT,
        phase=POST_SETTLE_EVENT,
        leg=1,
    )
    if endpoint is None:
        raise RuntimeError(
            'post-settle clearance endpoint was not reached safely'
        )
    _fresh_cage_gate(node)
    after = _scene_sample(
        node,
        runtime.environment_preflight,
        newer_than=settled.observed_at,
    )
    if float(np.max(np.abs(after.arm - target))) > (
        POST_SETTLE_ENDPOINT_ARM_LIMIT_RAD
    ):
        raise RuntimeError(
            'post-settle measured endpoint differs from its plan'
        )
    left, right, palm = _exact_contacts(node)
    rigid = edge_leg_metrics(settled, after)
    actual_scene = runtime.environment_preflight._read_scene()
    clearance = padded_book_shelf_clearance_guard(
        after.corners, actual_scene.shelf_triangles
    )
    neighbour = non_target_book_motion_guard(
        step18.all_book_corners, after.all_book_corners
    )
    checks = (
        (left, 'left_contact_missing'),
        (right, 'right_contact_missing'),
        (palm, 'palm_contact_missing'),
        (not _hazard_is_latched(node), 'unexpected_contact'),
        (
            abs(rigid['hand_outward_progress_m'] - distance)
            <= POST_SETTLE_PROGRESS_TOLERANCE_M,
            'post_settle_hand_progress',
        ),
        (
            abs(rigid['book_outward_progress_m'] - distance)
            <= POST_SETTLE_PROGRESS_TOLERANCE_M,
            'post_settle_book_progress',
        ),
        (
            rigid['book_hand_translation_mismatch_m']
            <= EDGE_HAND_BOOK_MISMATCH_LIMIT_M,
            'post_settle_nonrigid_translation',
        ),
        (
            rigid['book_rotation_rad'] <= EDGE_BOOK_ROTATION_LIMIT_RAD,
            'post_settle_nonrigid_rotation',
        ),
        (clearance.safe, clearance.reason),
        (neighbour.safe, neighbour.reason),
    )
    for accepted, reason in checks:
        if not accepted:
            raise RuntimeError(f'post-settle endpoint gate failed: {reason}')
    _emit(
        'post_settle_clearance',
        passed=True,
        selected_distance_m=distance,
        predicted_clearance=predicted_clearance.metrics,
        actual_clearance=clearance.metrics,
        **rigid,
        **neighbour.metrics,
    )
    return after, distance, clearance


def _run(runtime: SimpleNamespace) -> None:
    if runtime.BOOK != TARGET_BOOK_MODEL:
        raise RuntimeError('runtime target is not the audited seed-101 book')
    ProbeNode, ProbeNavigation = _probe_node_types(runtime)
    node = ProbeNode()
    nav = ProbeNavigation()
    executor = runtime.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    spin_errors: list[BaseException] = []

    def spin() -> None:
        try:
            executor.spin()
        except BaseException as exc:
            spin_errors.append(exc)

    thread = threading.Thread(
        target=spin, name='rigid-palm-natural-settle-probe', daemon=True
    )
    thread.start()
    stage = 'created'
    try:
        deadline = time.monotonic() + 30.0
        while (
            len(node.joints) < 8
            or nav.pose is None
            or nav.last_front_scan_time is None
            or nav.last_rear_scan_time is None
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8 or nav.pose is None:
            raise RuntimeError('live robot/navigation state is unavailable')

        node._target_book_model = TARGET_BOOK_MODEL
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        node._gravity_supported_payload = False
        node._transport_lock_engaged = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        node.reset_probe_audit()

        _fresh_cage_gate(node)
        step18 = _scene_sample(node, runtime.environment_preflight)
        resume = natural_settle_step18_resume_guard(
            step18.book,
            step18.base,
            step18.arm,
            aperture_m=step18.aperture_m,
        )
        _emit(
            'natural_settle_step18_resume_guard',
            passed=resume.safe,
            reason=resume.reason,
            **resume.metrics,
        )
        if not resume.safe:
            raise RuntimeError(
                'natural settle requires the newer zero-angle step-18 cage: '
                f'{resume.reason}'
            )
        support = initial_shelf_support_geometry_guard(
            step18.corners,
            step18.shelf_triangles,
        )
        _emit(
            'natural_settle_initial_shelf_support',
            passed=support.safe,
            reason=support.reason,
            **support.metrics,
        )
        if not support.safe:
            raise RuntimeError(
                'step-18 shelf-support geometry is unavailable: '
                f'{support.reason}'
            )
        _preflight_leg(node, runtime.environment_preflight, step18.arm)
        if _hazard_is_latched(node):
            raise RuntimeError('contact hazard is latched at step-18 resume')
        stage = 'step18_verified'
        _emit(
            'natural_settle_resume_verified',
            diagnostic_only=True,
            gazebo_truth_used=True,
            exact_step18=True,
            aperture_m=step18.aperture_m,
            shelf_overlap_m=float(step18.book.maximum[0] - SHELF_FRONT_X_M),
            shelf_support_geometry=support.metrics,
            book_position=step18.book.position.tolist(),
            solution=step18.arm.tolist(),
        )

        stage = 'edge_transfer'
        trigger, trigger_reason, edge_legs = _edge_to_settle_trigger(
            node, runtime, step18
        )
        stage = 'natural_settle'
        settled, envelope, stability = _wait_for_single_settle(
            node, runtime, step18, trigger
        )
        stage = 'post_settle_clearance'
        final, distance, clearance = _post_settle_clearance_leg(
            node, runtime, step18, settled
        )
        stage = 'complete'
        nav._publish_zero()
        _emit(
            'result',
            passed=True,
            stage='rigid_palm_natural_settle_clearance',
            diagnostic_only=True,
            gazebo_truth_used=True,
            settle_trigger=trigger_reason,
            edge_legs=edge_legs,
            settle_metrics=envelope.metrics,
            stability_metrics=stability.metrics,
            post_settle_distance_m=distance,
            clearance_metrics=clearance.metrics,
            book_position=final.book.position.tolist(),
            book_minimum=final.book.minimum.tolist(),
            book_maximum=final.book.maximum.tolist(),
            gripper_opened=False,
            base_motion_commanded=False,
            next_motion_authorized=False,
        )
    except Exception as exc:
        nav._publish_zero()
        if nav.goal is not None:
            nav._finish_goal('cancelled', reason='diagnostic_failure')
        _emit(
            'result',
            passed=False,
            stage=stage,
            diagnostic_only=True,
            gazebo_truth_used=True,
            reason=f'{type(exc).__name__}: {exc}',
            gripper_opened=False,
            base_motion_commanded=False,
            next_motion_authorized=False,
        )
        raise
    finally:
        nav._publish_zero()
        node._cancel.set()
        executor.shutdown()
        thread.join(timeout=3.0)
        node.destroy_node()
        nav.destroy_node()
        if spin_errors:
            raise RuntimeError(f'executor spin failed: {spin_errors[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Diagnostic-only step-18 rigid-palm natural shelf-edge '
            'settle proof'
        )
    )
    parser.parse_args()
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
