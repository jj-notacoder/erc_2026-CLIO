#!/usr/bin/env python3
"""Diagnostic-only continuation of the seed-101 rigid-palm cage.

This module deliberately uses Gazebo entity poses and the temporary palm
contact selector.  It is engineering evidence, not production competition
logic.  Importing it is side-effect free: controllers are contacted only from
``main()`` after all resume gates pass.  On every failure the base is stopped,
the gripper remains closed, and no recovery motion is attempted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from erc_phase1_solution.live_rigid_palm_support_probe import (
    BOOK_HALF_EXTENTS_M,
    CAGE_MEASURED_TARGET_TOLERANCE_M,
    CAGE_PRELOAD_M,
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    LEFT_FINGER_TOKENS,
    PALM_COLLISION_TOKEN,
    RIGHT_FINGER_TOKENS,
    SCORED_GEOMETRY_TOKENS,
    BookSnapshot,
    _load_runtime,
    _probe_node_type,
    _unexpected_pairs,
    quaternion_distance,
    relative_book_pose,
    rotation_matrix_distance,
    sampled_joint_segment,
    world_hand_pose,
)


TARGET_BOOK_MODEL = 'book_col_3_row_2_red'
CAGED_EXTRACTION_EVENT = 'rigid_caged_extraction'

# This is intentionally a same-world resume checkpoint, not a reusable pose.
EXPECTED_CAGED_BOOK_POSITION = np.asarray(
    [2.7678814197, -0.1485013434, 1.5769974698], dtype=float
)
EXPECTED_CAGED_JOINTS = np.asarray(
    [
        0.3499999541,
        0.3725895429,
        0.5322174528,
        0.4704197043,
        -1.6127080998,
        1.5256023673,
        0.7971682680,
        -1.6710828239,
    ],
    dtype=float,
)
EXPECTED_CAGED_APERTURE_M = CAGE_PRELOAD_M

SHELF_FRONT_X_M = 2.755
SHELF_CLEARANCE_M = 0.020
EXTRACTION_DISTANCE_M = 0.120
EXTRACTION_STEP_M = 0.005
EXTRACTION_DURATION_S = 0.48
CONTACT_DWELL_S = 0.18
SHELF_EDGE_STOP_STEP = 18
SHELF_OVERLAP_RESTORE_MINIMUM_START_M = 0.003
SHELF_OVERLAP_RESTORE_TARGET_M = 0.008

RESUME_JOINT_TOLERANCE_RAD = 0.006
# The palm-supported book creeps a few millimetres over a long paused Gazebo
# session.  Keep this same-world gate narrow, but leave enough room for that
# settling; live authorization still requires exact target contact at the palm
# and both fingertips plus a fresh current-scene mesh preflight.
RESUME_BOOK_POSITION_TOLERANCE_M = 0.012
RESUME_BOOK_VERTICAL_TOLERANCE_M = 0.003
RESUME_BOOK_ROTATION_TOLERANCE_RAD = math.radians(2.0)
RESUME_BASE_POSITION_TOLERANCE_M = 0.005
RESUME_BASE_YAW_TOLERANCE_RAD = 0.008
RESUME_RELATIVE_POSITION_TOLERANCE_M = 0.004
RESUME_RELATIVE_ROTATION_TOLERANCE_RAD = math.radians(3.0)
TANGENT_SHELF_SUPPORT_MARGIN_M = 0.008

# A stopped extraction can be resumed only when the independently observed
# hand and book translations quantize to the same interior 5 mm checkpoint.
# Keep the residual strictly below half a step so the inferred integer is
# unique; the tighter hand cross-track bound catches a wrong IK branch while
# the book allowance covers the sub-millimetre cage settling seen live.
INTERMEDIATE_EXTRACTION_MINIMUM_STEP = 1
INTERMEDIATE_EXTRACTION_MAXIMUM_STEP = 17
INTERMEDIATE_STEP_RESIDUAL_LIMIT_M = 0.00225
INTERMEDIATE_HAND_CROSS_TRACK_LIMIT_M = 0.0010
INTERMEDIATE_BOOK_CROSS_TRACK_LIMIT_M = 0.0030
INTERMEDIATE_HAND_ROTATION_LIMIT_RAD = 0.004
INTERMEDIATE_BASE_POSITION_LIMIT_M = 0.00075
INTERMEDIATE_BASE_YAW_LIMIT_RAD = 0.0010
PREFLIGHT_STABILITY_MAX_RETRIES = 3

STEP_BASE_POSITION_LIMIT_M = 0.00035
STEP_BASE_YAW_LIMIT_RAD = 0.0005
STEP_HAND_BOOK_MISMATCH_M = 0.0025
STEP_AABB_MISMATCH_M = 0.0030
STEP_BOOK_ROTATION_LIMIT_RAD = math.radians(2.0)
STEP_HAND_ROTATION_LIMIT_RAD = 0.004
STEP_MINIMUM_PROGRESS_M = 0.003
STEP_MAXIMUM_PROGRESS_M = 0.007
CUMULATIVE_ATTACHMENT_POSITION_LIMIT_M = 0.0035
CUMULATIVE_ATTACHMENT_AABB_LIMIT_M = 0.0040
CUMULATIVE_ATTACHMENT_ROTATION_LIMIT_RAD = math.radians(3.0)

RETREAT_DISTANCE_DEFAULT_M = 0.10
RETREAT_DISTANCE_MAXIMUM_M = 0.40
SHELF_EDGE_RETREAT_MAXIMUM_M = 0.005
RETREAT_POSITION_TOLERANCE_M = 0.005
RETREAT_RELATIVE_POSITION_LIMIT_M = 0.003
RETREAT_RELATIVE_ROTATION_LIMIT_RAD = math.radians(3.0)
RETREAT_ARM_ERROR_LIMIT_RAD = 0.006
RETREAT_POLL_S = 0.05
SCENE_WAIT_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class GuardResult:
    safe: bool
    reason: str
    metrics: dict[str, float]


def _nearest_extraction_step(progress_m: float) -> int:
    """Return the unique nearest nonnegative 5 mm checkpoint index."""

    progress = float(progress_m)
    if not math.isfinite(progress) or progress < 0.0:
        raise ValueError('extraction progress must be finite and nonnegative')
    return int(math.floor(progress / EXTRACTION_STEP_M + 0.5))


def _angle_error(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def _book_transform(snapshot: BookSnapshot) -> np.ndarray:
    from erc_phase1_solution.live_rigid_palm_support_probe import quaternion_matrix

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quaternion_matrix(snapshot.quaternion)
    transform[:3, 3] = snapshot.position
    return transform


def _quaternion_from_rotation(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a normalized xyzw quaternion for a proper rotation matrix."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError('rotation must be a finite 3x3 matrix')
    # Eigen-decomposition of Davenport's symmetric K matrix is stable near pi.
    k = np.asarray(
        [
            [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], matrix[0, 1] + matrix[1, 0], matrix[0, 2] + matrix[2, 0], matrix[2, 1] - matrix[1, 2]],
            [matrix[0, 1] + matrix[1, 0], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], matrix[1, 2] + matrix[2, 1], matrix[0, 2] - matrix[2, 0]],
            [matrix[0, 2] + matrix[2, 0], matrix[1, 2] + matrix[2, 1], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], matrix[1, 0] - matrix[0, 1]],
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1], matrix.trace()],
        ],
        dtype=float,
    ) / 3.0
    values, vectors = np.linalg.eigh(k)
    quaternion = vectors[:, int(np.argmax(values))]
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion / np.linalg.norm(quaternion)


def _coherent_scene(
    runtime: SimpleNamespace,
    *,
    newer_than: float | None = None,
) -> tuple[BookSnapshot, np.ndarray, float, np.ndarray]:
    """Read base and target from one complete Gazebo Pose_V frame."""

    deadline = time.monotonic() + SCENE_WAIT_TIMEOUT_S
    while True:
        scene = runtime.environment_preflight._read_scene()
        stamp = float(scene.observed_at)
        if newer_than is None or stamp > float(newer_than) + 1e-12:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Gazebo scene timestamp did not advance')
        time.sleep(0.02)
    base_transform = np.asarray(scene.base_transform, dtype=float)
    base = np.asarray(
        [
            base_transform[0, 3],
            base_transform[1, 3],
            math.atan2(base_transform[1, 0], base_transform[0, 0]),
        ],
        dtype=float,
    )
    corners = np.asarray(scene.books[TARGET_BOOK_MODEL].corners, dtype=float)
    center = np.mean(corners, axis=0)
    rotation = np.column_stack(
        (
            (corners[4] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[0]),
            (corners[2] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[1]),
            (corners[1] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[2]),
        )
    )
    book = BookSnapshot(
        center,
        _quaternion_from_rotation(rotation),
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )
    return book, base, stamp, corners


def _attached_corners(
    node: Any,
    book: BookSnapshot,
    base: Sequence[float],
    arm: Sequence[float],
) -> np.ndarray:
    """Return an inflated observed OBB in the grasp-link frame."""
    world_corners = _vertically_ordered_book_corners(
        book,
        padding_m=float(node.carried_book_padding),
    )
    hand = world_hand_pose(base, node.chain.forward(arm))
    return (world_corners - hand[:3, 3]) @ hand[:3, :3]


def _vertically_ordered_book_corners(
    snapshot: BookSnapshot,
    *,
    padding_m: float = 0.0,
) -> np.ndarray:
    """Return the same OBB with ordered edge 0->1 pointing world-up."""

    from erc_phase1_solution.live_rigid_palm_support_probe import quaternion_matrix

    padding = float(padding_m)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError('book padding must be finite and non-negative')
    physical_basis = quaternion_matrix(snapshot.quaternion)
    vertical_index = int(np.argmax(np.abs(physical_basis[2, :])))
    permutation = [
        index for index in range(3) if index != vertical_index
    ] + [vertical_index]
    ordered_basis = physical_basis[:, permutation].copy()
    ordered_half_extents = (
        BOOK_HALF_EXTENTS_M + padding
    )[permutation]

    # Standard sign order changes z between corners 0 and 1.  Make that axis
    # point upward, then compensate on a horizontal axis if the permutation
    # or vertical flip changed handedness.  Axis sign flips and permutations
    # preserve the physical set of eight OBB corners.
    if ordered_basis[2, 2] < 0.0:
        ordered_basis[:, 2] *= -1.0
    if float(np.linalg.det(ordered_basis)) < 0.0:
        ordered_basis[:, 0] *= -1.0
    if not math.isclose(
        float(np.linalg.det(ordered_basis)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError('ordered book basis is not a proper rotation')

    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )
    return (
        signs * ordered_half_extents
    ) @ ordered_basis.T + snapshot.position


def _book_corners(
    snapshot: BookSnapshot,
    *,
    padding_m: float = 0.0,
) -> np.ndarray:
    """Reconstruct ordered OBB corners from one validated book snapshot."""

    from erc_phase1_solution.live_rigid_palm_support_probe import quaternion_matrix

    padding = float(padding_m)
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError('book padding must be finite and non-negative')
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )
    return (
        signs * (BOOK_HALF_EXTENTS_M + padding)
    ) @ quaternion_matrix(snapshot.quaternion).T + snapshot.position


def cumulative_attachment_guard(
    reference_book: BookSnapshot,
    current_book: BookSnapshot,
    reference_hand_world: Sequence[Sequence[float]],
    current_hand_world: Sequence[Sequence[float]],
) -> GuardResult:
    """Prove the book remains at one rigid transform from the palm."""

    try:
        first_hand = np.asarray(reference_hand_world, dtype=float)
        second_hand = np.asarray(current_hand_world, dtype=float)
        if (
            first_hand.shape != (4, 4)
            or second_hand.shape != (4, 4)
            or not np.all(np.isfinite(first_hand))
            or not np.all(np.isfinite(second_hand))
        ):
            raise ValueError('hand transforms must be finite 4x4 matrices')
        reference_relative = np.linalg.inv(first_hand) @ _book_transform(
            reference_book
        )
        current_relative = np.linalg.inv(second_hand) @ _book_transform(
            current_book
        )
        position_error = float(
            np.linalg.norm(
                current_relative[:3, 3] - reference_relative[:3, 3]
            )
        )
        rotation_error = rotation_matrix_distance(
            reference_relative[:3, :3], current_relative[:3, :3]
        )
        hand_motion = second_hand @ np.linalg.inv(first_hand)
        expected_corners = (
            _book_corners(reference_book) @ hand_motion[:3, :3].T
            + hand_motion[:3, 3]
        )
        current_corners = _book_corners(current_book)
        corner_error = float(
            np.max(np.linalg.norm(current_corners - expected_corners, axis=1))
        )
        expected_minimum = np.min(expected_corners, axis=0)
        expected_maximum = np.max(expected_corners, axis=0)
        aabb_error = float(
            max(
                np.max(np.abs(current_book.minimum - expected_minimum)),
                np.max(np.abs(current_book.maximum - expected_maximum)),
            )
        )
        metrics = {
            'cumulative_book_hand_position_error_m': position_error,
            'cumulative_book_hand_rotation_error_rad': rotation_error,
            'cumulative_book_corner_error_m': corner_error,
            'cumulative_book_aabb_error_m': aabb_error,
        }
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    checks = (
        (
            position_error <= CUMULATIVE_ATTACHMENT_POSITION_LIMIT_M,
            'cumulative_book_position_slip',
        ),
        (
            rotation_error <= CUMULATIVE_ATTACHMENT_ROTATION_LIMIT_RAD,
            'cumulative_book_rotation_slip',
        ),
        (
            corner_error <= CUMULATIVE_ATTACHMENT_AABB_LIMIT_M,
            'cumulative_book_corner_slip',
        ),
        (
            aabb_error <= CUMULATIVE_ATTACHMENT_AABB_LIMIT_M,
            'cumulative_book_aabb_slip',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _unexpected_scored_contact_pairs(
    message: Any,
    *,
    target_book: str = TARGET_BOOK_MODEL,
) -> set[tuple[str, str]]:
    """Catch robot/scored-geometry contacts, including finger-to-shelf."""

    target = str(target_book).lower()
    faults: set[tuple[str, str]] = set()
    for contact in getattr(message, 'contacts', ()):
        names = (
            str(getattr(getattr(contact, 'collision1', None), 'name', '')),
            str(getattr(getattr(contact, 'collision2', None), 'name', '')),
        )
        lowered = tuple(name.lower() for name in names)
        combined = ' '.join(lowered)
        involves_robot = any('tiago_pro::' in name for name in lowered)
        involves_scored_geometry = any(
            token in combined for token in SCORED_GEOMETRY_TOKENS
        )
        if not involves_robot or not involves_scored_geometry:
            continue
        recognized_cage_link = any(
            PALM_COLLISION_TOKEN in name
            or (
                'gripper_left_' in name
                and any(
                    token in name
                    for token in (*LEFT_FINGER_TOKENS, *RIGHT_FINGER_TOKENS)
                )
                and 'base_finger_' not in name
            )
            for name in lowered
        )
        exact_target = any(target in name for name in lowered)
        concrete_other_book = any(
            'book_col_' in name and target not in name for name in lowered
        )
        if not (
            exact_target
            and recognized_cage_link
            and not concrete_other_book
        ):
            faults.add(tuple(sorted(names)))
    return faults


def _probe_hazard_reason(node: Any) -> str | None:
    payload = getattr(node, '_payload_hazard_latched', None)
    if payload is not None:
        return str(payload)
    if bool(getattr(node, '_target_robot_contact_latched', False)):
        return 'payload_robot_contact'
    if _unexpected_pairs(node):
        return 'unexpected_scored_contact'
    return None


def _outward_world(base_pose: Sequence[float]) -> np.ndarray:
    base = np.asarray(base_pose, dtype=float)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError('base pose must contain finite x, y, and yaw')
    return np.asarray([-math.cos(base[2]), -math.sin(base[2]), 0.0])


def extraction_step_guard(
    before_book: BookSnapshot,
    after_book: BookSnapshot,
    before_hand_world: Sequence[Sequence[float]],
    after_hand_world: Sequence[Sequence[float]],
    before_base: Sequence[float],
    after_base: Sequence[float],
    *,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    measured_aperture_m: float,
) -> GuardResult:
    """Pure fail-closed endpoint gate for one caged extraction leg."""

    try:
        first_hand = np.asarray(before_hand_world, dtype=float)
        second_hand = np.asarray(after_hand_world, dtype=float)
        first_base = np.asarray(before_base, dtype=float)
        second_base = np.asarray(after_base, dtype=float)
        aperture = float(measured_aperture_m)
        if (
            first_hand.shape != (4, 4)
            or second_hand.shape != (4, 4)
            or first_base.shape != (3,)
            or second_base.shape != (3,)
            or not all(
                np.all(np.isfinite(value))
                for value in (first_hand, second_hand, first_base, second_base)
            )
            or not math.isfinite(aperture)
        ):
            raise ValueError('non-finite extraction observation')
        hand_delta = second_hand[:3, 3] - first_hand[:3, 3]
        book_delta = after_book.position - before_book.position
        outward = _outward_world(first_base)
        progress = float(np.dot(book_delta, outward))
        mismatch = float(np.linalg.norm(book_delta - hand_delta))
        minimum_mismatch = float(
            np.max(np.abs((after_book.minimum - before_book.minimum) - hand_delta))
        )
        maximum_mismatch = float(
            np.max(np.abs((after_book.maximum - before_book.maximum) - hand_delta))
        )
        book_rotation = quaternion_distance(
            before_book.quaternion, after_book.quaternion
        )
        hand_rotation = rotation_matrix_distance(
            first_hand[:3, :3], second_hand[:3, :3]
        )
        base_drift = float(np.linalg.norm(second_base[:2] - first_base[:2]))
        base_yaw_drift = _angle_error(second_base[2], first_base[2])
        metrics = {
            'outward_progress_m': progress,
            'book_hand_translation_mismatch_m': mismatch,
            'book_minimum_translation_mismatch_m': minimum_mismatch,
            'book_maximum_translation_mismatch_m': maximum_mismatch,
            'book_rotation_rad': book_rotation,
            'hand_rotation_rad': hand_rotation,
            'base_drift_m': base_drift,
            'base_yaw_drift_rad': base_yaw_drift,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})

    checks = (
        (left_contact, 'left_contact_missing'),
        (right_contact, 'right_contact_missing'),
        (palm_contact, 'palm_contact_missing'),
        (not unexpected_contacts, 'unexpected_contact'),
        (
            abs(aperture - EXPECTED_CAGED_APERTURE_M)
            <= CAGE_MEASURED_TARGET_TOLERANCE_M,
            'cage_aperture_changed',
        ),
        (STEP_MINIMUM_PROGRESS_M <= progress <= STEP_MAXIMUM_PROGRESS_M,
         'book_progress_out_of_bounds'),
        (mismatch <= STEP_HAND_BOOK_MISMATCH_M, 'book_did_not_follow_hand'),
        (minimum_mismatch <= STEP_AABB_MISMATCH_M, 'book_minimum_did_not_follow'),
        (maximum_mismatch <= STEP_AABB_MISMATCH_M, 'book_maximum_did_not_follow'),
        (book_rotation <= STEP_BOOK_ROTATION_LIMIT_RAD, 'book_rotated'),
        (hand_rotation <= STEP_HAND_ROTATION_LIMIT_RAD, 'hand_orientation_changed'),
        (base_drift <= STEP_BASE_POSITION_LIMIT_M, 'base_moved_during_arm_step'),
        (base_yaw_drift <= STEP_BASE_YAW_LIMIT_RAD, 'base_rotated_during_arm_step'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def retreat_sample_guard(
    reference_book: BookSnapshot,
    current_book: BookSnapshot,
    reference_base: Sequence[float],
    current_base: Sequence[float],
    reference_arm: Sequence[float],
    current_arm: Sequence[float],
    *,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    measured_aperture_m: float,
) -> GuardResult:
    """Pure carried-base gate; book pose must remain fixed in the base frame."""

    try:
        first_base = np.asarray(reference_base, dtype=float)
        second_base = np.asarray(current_base, dtype=float)
        first_arm = np.asarray(reference_arm, dtype=float)
        second_arm = np.asarray(current_arm, dtype=float)
        first_position, first_rotation = relative_book_pose(first_base, reference_book)
        second_position, second_rotation = relative_book_pose(second_base, current_book)
        position_error = float(np.linalg.norm(second_position - first_position))
        rotation_error = rotation_matrix_distance(first_rotation, second_rotation)
        arm_error = float(np.max(np.abs(second_arm - first_arm)))
        aperture_error = abs(float(measured_aperture_m) - EXPECTED_CAGED_APERTURE_M)
        travelled = float(np.dot(second_base[:2] - first_base[:2], _outward_world(first_base)[:2]))
        metrics = {
            'base_outward_travel_m': travelled,
            'book_base_position_error_m': position_error,
            'book_base_rotation_error_rad': rotation_error,
            'arm_error_rad': arm_error,
            'aperture_error_m': aperture_error,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    checks = (
        (left_contact, 'left_contact_missing'),
        (right_contact, 'right_contact_missing'),
        (palm_contact, 'palm_contact_missing'),
        (not unexpected_contacts, 'unexpected_contact'),
        (position_error <= RETREAT_RELATIVE_POSITION_LIMIT_M,
         'book_shifted_relative_to_base'),
        (rotation_error <= RETREAT_RELATIVE_ROTATION_LIMIT_RAD,
         'book_rotated_relative_to_base'),
        (arm_error <= RETREAT_ARM_ERROR_LIMIT_RAD, 'arm_moved_during_retreat'),
        (aperture_error <= CAGE_MEASURED_TARGET_TOLERANCE_M,
         'cage_aperture_changed'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({'event': event, **fields}, sort_keys=True), flush=True)


def _fresh_cage_gate(node: Any) -> tuple[bool, bool, bool]:
    node._clear_target_contact_samples(reset_robot_contact=False)
    node.clear_probe_finger_evidence()
    node.clear_probe_palm_evidence()
    started_ns = int(node.get_clock().now().nanoseconds)
    if not node._wait_sim_duration(CONTACT_DWELL_S):
        raise RuntimeError('fresh cage-contact dwell was interrupted')
    left, right = node.probe_exact_finger_sides(max_age=0.22)
    palm = node.probe_exact_palm_contact(max_age=0.22, since_ns=started_ns)
    if not left or not right or not palm:
        raise RuntimeError(
            'fresh exact-target palm plus bilateral cage contact is unavailable'
        )
    hazard = _probe_hazard_reason(node)
    if hazard is not None:
        raise RuntimeError(f'payload/contact hazard is latched: {hazard}')
    return left, right, palm


def _expected_relative_book_to_hand(node: Any) -> np.ndarray:
    expected_hand = world_hand_pose(
        EXPECTED_RELEASED_BASE_POSE,
        node.chain.forward(EXPECTED_CAGED_JOINTS),
    )
    expected_book = BookSnapshot(
        EXPECTED_CAGED_BOOK_POSITION,
        EXPECTED_RELEASED_BOOK_QUATERNION,
        EXPECTED_CAGED_BOOK_POSITION - np.asarray([0.0805, 0.0155, 0.1255]),
        EXPECTED_CAGED_BOOK_POSITION + np.asarray([0.0805, 0.0155, 0.1255]),
    )
    return np.linalg.inv(expected_hand) @ _book_transform(expected_book)


def intermediate_extraction_resume_guard(
    book: BookSnapshot,
    base: Sequence[float],
    current_hand_world: Sequence[Sequence[float]],
    expected_hand_world: Sequence[Sequence[float]],
    *,
    aperture_m: float,
    observed_at: float,
    reference_time: float,
    max_state_age_seconds: float,
    future_tolerance_seconds: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Recognize exactly one safe interior 5 mm extraction checkpoint.

    The hand and book are independently projected along the recorded outward
    direction and must round to the same integer step in ``[1, 17]``.  Their
    cross-track motion, relative transform, orientation, base, aperture,
    freshness, and exact three-point cage contacts are all checked before the
    caller may construct any remaining motion.
    """

    try:
        current_hand = np.asarray(current_hand_world, dtype=float)
        expected_hand = np.asarray(expected_hand_world, dtype=float)
        current_base = np.asarray(base, dtype=float)
        aperture = float(aperture_m)
        stamp = float(observed_at)
        now = float(reference_time)
        maximum_age = float(max_state_age_seconds)
        future_tolerance = float(future_tolerance_seconds)
        if (
            current_hand.shape != (4, 4)
            or expected_hand.shape != (4, 4)
            or current_base.shape != (3,)
            or not np.all(np.isfinite(current_hand))
            or not np.all(np.isfinite(expected_hand))
            or not np.all(np.isfinite(current_base))
            or not all(
                math.isfinite(value)
                for value in (
                    aperture,
                    stamp,
                    now,
                    maximum_age,
                    future_tolerance,
                )
            )
            or maximum_age <= 0.0
            or future_tolerance < 0.0
        ):
            raise ValueError('intermediate resume observation is malformed')

        outward = _outward_world(EXPECTED_RELEASED_BASE_POSE)
        hand_delta = current_hand[:3, 3] - expected_hand[:3, 3]
        book_delta = book.position - EXPECTED_CAGED_BOOK_POSITION
        hand_progress = float(np.dot(hand_delta, outward))
        book_progress = float(np.dot(book_delta, outward))
        hand_step = _nearest_extraction_step(hand_progress)
        book_step = _nearest_extraction_step(book_progress)
        hand_residual = abs(
            hand_progress - hand_step * EXTRACTION_STEP_M
        )
        book_residual = abs(
            book_progress - book_step * EXTRACTION_STEP_M
        )
        hand_cross_track = float(
            np.linalg.norm(hand_delta - hand_progress * outward)
        )
        book_cross_track = float(
            np.linalg.norm(book_delta - book_progress * outward)
        )
        hand_rotation = rotation_matrix_distance(
            expected_hand[:3, :3], current_hand[:3, :3]
        )
        book_rotation = quaternion_distance(
            EXPECTED_RELEASED_BOOK_QUATERNION, book.quaternion
        )
        expected_book = BookSnapshot(
            EXPECTED_CAGED_BOOK_POSITION,
            EXPECTED_RELEASED_BOOK_QUATERNION,
            EXPECTED_CAGED_BOOK_POSITION - np.asarray(
                [0.0805, 0.0155, 0.1255], dtype=float
            ),
            EXPECTED_CAGED_BOOK_POSITION + np.asarray(
                [0.0805, 0.0155, 0.1255], dtype=float
            ),
        )
        expected_relative = np.linalg.inv(expected_hand) @ _book_transform(
            expected_book
        )
        current_relative = np.linalg.inv(current_hand) @ _book_transform(book)
        relative_position_error = float(
            np.linalg.norm(
                current_relative[:3, 3] - expected_relative[:3, 3]
            )
        )
        relative_rotation_error = rotation_matrix_distance(
            expected_relative[:3, :3], current_relative[:3, :3]
        )
        base_drift = float(
            np.linalg.norm(current_base[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
        )
        base_yaw_drift = _angle_error(
            current_base[2], EXPECTED_RELEASED_BASE_POSE[2]
        )
        aperture_error = abs(aperture - EXPECTED_CAGED_APERTURE_M)
        scene_age = now - stamp
        metrics = {
            'inferred_extraction_step': float(hand_step),
            'hand_inferred_step': float(hand_step),
            'book_inferred_step': float(book_step),
            'hand_outward_progress_m': hand_progress,
            'book_outward_progress_m': book_progress,
            'hand_step_residual_m': hand_residual,
            'book_step_residual_m': book_residual,
            'hand_cross_track_m': hand_cross_track,
            'book_cross_track_m': book_cross_track,
            'hand_rotation_error_rad': hand_rotation,
            'book_rotation_error_rad': book_rotation,
            'book_hand_position_error_m': relative_position_error,
            'book_hand_rotation_error_rad': relative_rotation_error,
            'base_drift_m': base_drift,
            'base_yaw_drift_rad': base_yaw_drift,
            'aperture_error_m': aperture_error,
            'scene_age_s': scene_age,
        }
    except (
        AttributeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})

    checks = (
        (left_contact, 'left_contact_missing'),
        (right_contact, 'right_contact_missing'),
        (palm_contact, 'palm_contact_missing'),
        (not unexpected_contacts, 'unexpected_contact'),
        (
            -future_tolerance <= scene_age <= maximum_age,
            'stale_or_future_scene',
        ),
        (
            aperture_error <= CAGE_MEASURED_TARGET_TOLERANCE_M,
            'cage_aperture_changed',
        ),
        (
            base_drift <= INTERMEDIATE_BASE_POSITION_LIMIT_M,
            'base_not_at_recorded_pose',
        ),
        (
            base_yaw_drift <= INTERMEDIATE_BASE_YAW_LIMIT_RAD,
            'base_yaw_not_at_recorded_pose',
        ),
        (
            INTERMEDIATE_EXTRACTION_MINIMUM_STEP
            <= hand_step
            <= INTERMEDIATE_EXTRACTION_MAXIMUM_STEP,
            'hand_not_at_interior_extraction_step',
        ),
        (
            INTERMEDIATE_EXTRACTION_MINIMUM_STEP
            <= book_step
            <= INTERMEDIATE_EXTRACTION_MAXIMUM_STEP,
            'book_not_at_interior_extraction_step',
        ),
        (hand_step == book_step, 'book_hand_step_disagreement'),
        (
            hand_residual <= INTERMEDIATE_STEP_RESIDUAL_LIMIT_M,
            'hand_not_on_5mm_checkpoint',
        ),
        (
            book_residual <= INTERMEDIATE_STEP_RESIDUAL_LIMIT_M,
            'book_not_on_5mm_checkpoint',
        ),
        (
            hand_cross_track <= INTERMEDIATE_HAND_CROSS_TRACK_LIMIT_M,
            'hand_cross_track_motion',
        ),
        (
            book_cross_track <= INTERMEDIATE_BOOK_CROSS_TRACK_LIMIT_M,
            'book_cross_track_motion',
        ),
        (
            hand_rotation <= INTERMEDIATE_HAND_ROTATION_LIMIT_RAD,
            'hand_orientation_changed',
        ),
        (
            book_rotation <= RESUME_BOOK_ROTATION_TOLERANCE_RAD,
            'book_orientation_changed',
        ),
        (
            relative_position_error <= RESUME_RELATIVE_POSITION_TOLERANCE_M,
            'book_not_caged_at_recorded_offset',
        ),
        (
            relative_rotation_error <= RESUME_RELATIVE_ROTATION_TOLERANCE_RAD,
            'book_not_caged_at_recorded_orientation',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _classify_resume_state(
    node: Any,
    book: BookSnapshot,
    base: np.ndarray,
    arm: np.ndarray,
    aperture: float,
) -> str:
    if abs(aperture - EXPECTED_CAGED_APERTURE_M) > CAGE_MEASURED_TARGET_TOLERANCE_M:
        raise RuntimeError('gripper is not at the proven 30 mm cage')
    expected_hand_rotation = node.chain.forward(EXPECTED_CAGED_JOINTS)[:3, :3]
    current_hand_base = node.chain.forward(arm)
    if rotation_matrix_distance(
        expected_hand_rotation, current_hand_base[:3, :3]
    ) > RESUME_RELATIVE_ROTATION_TOLERANCE_RAD:
        raise RuntimeError('current palm orientation is not the proven orientation')

    current_hand = world_hand_pose(base, current_hand_base)
    relative = np.linalg.inv(current_hand) @ _book_transform(book)
    expected_relative = _expected_relative_book_to_hand(node)
    relative_position_error = float(
        np.linalg.norm(relative[:3, 3] - expected_relative[:3, 3])
    )
    relative_rotation_error = rotation_matrix_distance(
        relative[:3, :3], expected_relative[:3, :3]
    )

    at_tangent = bool(
        np.linalg.norm(
            book.position[:2] - EXPECTED_CAGED_BOOK_POSITION[:2]
        ) <= RESUME_BOOK_POSITION_TOLERANCE_M
        and abs(
            float(book.position[2] - EXPECTED_CAGED_BOOK_POSITION[2])
        ) <= RESUME_BOOK_VERTICAL_TOLERANCE_M
        and np.linalg.norm(base[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
        <= RESUME_BASE_POSITION_TOLERANCE_M
        and _angle_error(base[2], EXPECTED_RELEASED_BASE_POSE[2])
        <= RESUME_BASE_YAW_TOLERANCE_RAD
        and quaternion_distance(
            book.quaternion, EXPECTED_RELEASED_BOOK_QUATERNION
        ) <= RESUME_BOOK_ROTATION_TOLERANCE_RAD
        and float(book.position[0])
        >= SHELF_FRONT_X_M + TANGENT_SHELF_SUPPORT_MARGIN_M
        and relative_position_error <= RESUME_RELATIVE_POSITION_TOLERANCE_M
        and relative_rotation_error <= RESUME_RELATIVE_ROTATION_TOLERANCE_RAD
    )
    if at_tangent:
        return 'tangent_caged'

    if (
        book.maximum[0] <= SHELF_FRONT_X_M - SHELF_CLEARANCE_M
        and np.linalg.norm(relative[:3, 3] - expected_relative[:3, 3])
        <= RESUME_RELATIVE_POSITION_TOLERANCE_M
        and rotation_matrix_distance(
            relative[:3, :3], expected_relative[:3, :3]
        ) <= RESUME_RELATIVE_ROTATION_TOLERANCE_RAD
    ):
        return 'already_extracted_caged'
    raise RuntimeError('state is neither the recorded tangent cage nor a verified extracted cage')


def _is_transient_preflight_stability_rejection(result: Any) -> bool:
    """Recognize only live-adapter rejections caused by moving inputs."""

    if bool(getattr(result, 'safe', False)):
        return False
    if str(getattr(result, 'code', '')) != 'live_adapter_error':
        return False
    detail = str(getattr(result, 'detail', ''))
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    patterns = (
        rf'arm moved {number} rad during geometric preflight',
        rf'gripper moved {number} m during geometric preflight',
        rf'robot base moved during geometric preflight: {number} m, '
        rf'{number} rad',
        rf"book '[^']+' moved {number} m during preflight",
        rf'loaded palm-relative finger geometry moved during preflight: '
        rf'{number} m at \S+',
    )
    return any(re.fullmatch(pattern, detail) is not None for pattern in patterns)


def _preflight_leg(node: Any, environment: Any, target: np.ndarray) -> None:
    last_result: Any = None
    for attempt in range(PREFLIGHT_STABILITY_MAX_RETRIES + 1):
        # Re-read the measured start and let the live adapter capture a fresh
        # before/after Gazebo scene on every attempt.  No controller call is
        # reachable from this retry loop.
        samples = sampled_joint_segment(node._measured_left_solution(), target)
        for index, sample in enumerate(samples):
            collision = node._robot_self_collision(sample)
            if collision is not None:
                raise RuntimeError(
                    'self-collision at extraction preflight sample '
                    f'{index}: {collision}'
                )
        result = environment(
            node=node,
            event=CAGED_EXTRACTION_EVENT,
            joint_samples=tuple(sample.copy() for sample in samples),
        )
        last_result = result
        if bool(getattr(result, 'safe', False)):
            return
        if not _is_transient_preflight_stability_rejection(result):
            break
        if attempt >= PREFLIGHT_STABILITY_MAX_RETRIES:
            break
        _emit(
            'extraction_preflight_retry',
            retry=attempt + 1,
            maximum_retries=PREFLIGHT_STABILITY_MAX_RETRIES,
            code=str(getattr(result, 'code', 'invalid_result')),
            detail=str(getattr(result, 'detail', repr(result))),
            controller_motion_issued=False,
        )
        if not node._wait_sim_duration(CONTACT_DWELL_S):
            raise RuntimeError(
                'extraction preflight settling dwell was interrupted'
            )
    raise RuntimeError(
        'nine-link extraction preflight rejected the leg: '
        f'{getattr(last_result, "code", "invalid_result")}: '
        f'{getattr(last_result, "detail", repr(last_result))}'
    )


def _extraction_tangent_reference_pose(
    current_pose: Sequence[Sequence[float]],
    outward_in_base: Sequence[float],
    starting_step: int,
) -> np.ndarray:
    """Reconstruct the global step-zero pose from a verified checkpoint."""

    pose = np.asarray(current_pose, dtype=float)
    outward = np.asarray(outward_in_base, dtype=float)
    step = int(starting_step)
    if (
        pose.shape != (4, 4)
        or outward.shape != (3,)
        or not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(outward))
        or step < 0
        or step > INTERMEDIATE_EXTRACTION_MAXIMUM_STEP
    ):
        raise ValueError('invalid extraction checkpoint reference')
    reference = pose.copy()
    reference[:3, 3] -= outward * (EXTRACTION_STEP_M * step)
    return reference


def _execute_extraction(
    node: Any,
    runtime: SimpleNamespace,
    initial_book: BookSnapshot,
    base_start: np.ndarray,
    initial_scene_stamp: float,
    *,
    starting_step: int = 0,
    maximum_steps: int | None = None,
    require_shelf_clearance: bool = True,
) -> BookSnapshot:
    start_arm = np.asarray(node._measured_left_solution(), dtype=float)
    current_pose = node.chain.forward(start_arm)
    previous_solution = start_arm
    previous_book = initial_book
    previous_base = base_start.copy()
    previous_hand = world_hand_pose(previous_base, current_pose)
    reference_book = previous_book
    reference_hand = previous_hand
    previous_scene_stamp = float(initial_scene_stamp)
    outward = _outward_world(base_start)
    yaw = float(base_start[2])
    base_rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    outward_base = np.zeros(3, dtype=float)
    outward_base[:2] = base_rotation.T @ outward[:2]
    full_steps = int(round(EXTRACTION_DISTANCE_M / EXTRACTION_STEP_M))
    steps = full_steps if maximum_steps is None else int(maximum_steps)
    start_step = int(starting_step)
    if not 0 <= start_step < steps <= full_steps:
        raise ValueError(
            'extraction steps must satisfy '
            f'0 <= starting_step < maximum_steps <= {full_steps}'
        )
    fixed_pose = _extraction_tangent_reference_pose(
        current_pose, outward_base, start_step
    )

    for index in range(start_step + 1, steps + 1):
        position = (
            fixed_pose[:3, 3]
            + outward_base * (EXTRACTION_STEP_M * index)
        )
        target_pose = runtime.pose_matrix(position, fixed_pose[:3, :3])
        solution, _ = node.chain.solve(
            target_pose,
            [previous_solution],
            position_tolerance=0.00035,
            orientation_tolerance=0.002,
            max_iterations=320,
            fixed_positions={'torso_lift_joint': float(start_arm[0])},
        )
        if solution is None:
            raise RuntimeError(f'fixed-orientation extraction IK failed at step {index}')
        solution = np.asarray(solution, dtype=float)
        if float(np.max(np.abs(solution - previous_solution))) > 0.16:
            raise RuntimeError(f'extraction IK branch jump at step {index}')
        attached_corners = _attached_corners(
            node,
            previous_book,
            previous_base,
            previous_solution,
        )
        if not node._carried_robot_transition_is_safe(
            previous_solution,
            solution,
            attached_corners,
        ):
            diagnostic: dict[str, Any] = {
                'maximum_joint_delta_rad': float(
                    np.max(np.abs(solution[1:] - previous_solution[1:]))
                ),
                'orientation_delta_rad': float(
                    np.linalg.norm(
                        node.chain.pose_error(
                            node.chain.forward(previous_solution),
                            node.chain.forward(solution),
                        )[3:]
                    )
                ),
            }
            for label, candidate in (
                ('start', previous_solution),
                ('end', solution),
            ):
                candidate_hand = node.chain.forward(candidate)
                world_corners = (
                    attached_corners @ candidate_hand[:3, :3].T
                    + candidate_hand[:3, 3]
                )
                diagnostic[f'{label}_payload_robot_collision'] = (
                    node._carried_robot_collision(candidate, world_corners)
                )
                diagnostic[f'{label}_robot_self_collision'] = (
                    node._robot_self_collision(candidate)
                )
            raise RuntimeError(
                'carried robot/payload sweep rejected extraction step '
                f'{index}: {json.dumps(diagnostic, sort_keys=True)}'
            )
        node._held_book_corners = attached_corners.copy()
        _preflight_leg(node, runtime.environment_preflight, solution)
        _fresh_cage_gate(node)
        legs = ((solution, EXTRACTION_DURATION_S, CAGED_EXTRACTION_EVENT),)
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        moved, contact_loss = node._send_retained_arm_trajectory(
            goal, duration, legs, 'rigid_palm_extract'
        )
        if not moved:
            reason = 'contact_loss' if contact_loss else 'controller_failure'
            raise RuntimeError(f'extraction step {index} failed: {reason}')
        endpoint = node._wait_for_retained_endpoint(
            solution,
            command='rigid_palm_extract',
            phase=CAGED_EXTRACTION_EVENT,
            leg=index,
        )
        if endpoint is None:
            raise RuntimeError(f'extraction endpoint {index} was not reached safely')
        left, right, palm = _fresh_cage_gate(node)
        current_book, current_base, current_scene_stamp, _ = _coherent_scene(
            runtime,
            newer_than=previous_scene_stamp,
        )
        measured_solution = np.asarray(
            node._measured_left_solution(), dtype=float
        )
        current_hand = world_hand_pose(
            current_base, node.chain.forward(measured_solution)
        )
        unexpected = bool(_unexpected_pairs(node)) or bool(
            getattr(node, '_target_robot_contact_latched', False)
        ) or getattr(node, '_payload_hazard_latched', None) is not None
        guard = extraction_step_guard(
            previous_book,
            current_book,
            previous_hand,
            current_hand,
            previous_base,
            current_base,
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=unexpected,
            measured_aperture_m=float(node.joints['gripper_left_finger_joint']),
        )
        cumulative = cumulative_attachment_guard(
            reference_book,
            current_book,
            reference_hand,
            current_hand,
        )
        _emit(
            'extraction_step',
            step=index,
            steps=steps,
            passed=guard.safe and cumulative.safe,
            reason=(guard.reason if not guard.safe else cumulative.reason),
            **guard.metrics,
            **cumulative.metrics,
            book_position=current_book.position.tolist(),
            book_maximum=current_book.maximum.tolist(),
            solution=np.asarray(endpoint, dtype=float).tolist(),
            scene_stamp=current_scene_stamp,
        )
        if not guard.safe:
            raise RuntimeError(f'extraction endpoint gate failed: {guard.reason}')
        if not cumulative.safe:
            raise RuntimeError(
                f'cumulative attachment gate failed: {cumulative.reason}'
            )
        previous_solution = measured_solution
        previous_book = current_book
        previous_base = current_base
        previous_hand = current_hand
        previous_scene_stamp = current_scene_stamp

    if (
        require_shelf_clearance
        and previous_book.maximum[0] > SHELF_FRONT_X_M - SHELF_CLEARANCE_M
    ):
        raise RuntimeError('extraction ended before the whole book cleared the shelf face')
    return previous_book


def _restore_shelf_overlap(
    node: Any,
    runtime: SimpleNamespace,
    book: BookSnapshot,
    base: np.ndarray,
    scene_stamp: float,
) -> BookSnapshot:
    """Move one closed-cage 5 mm leg inward to regain shelf support."""

    arm = np.asarray(node._measured_left_solution(), dtype=float)
    aperture = float(node.joints.get('gripper_left_finger_joint', math.nan))
    hand_base = np.asarray(node.chain.forward(arm), dtype=float)
    hand_world = world_hand_pose(base, hand_base)
    expected_rotation = node.chain.forward(EXPECTED_CAGED_JOINTS)[:3, :3]
    expected_relative = _expected_relative_book_to_hand(node)
    relative = np.linalg.inv(hand_world) @ _book_transform(book)
    overlap = float(book.maximum[0] - SHELF_FRONT_X_M)
    if not (
        abs(aperture - EXPECTED_CAGED_APERTURE_M)
        <= CAGE_MEASURED_TARGET_TOLERANCE_M
        and SHELF_OVERLAP_RESTORE_MINIMUM_START_M
        <= overlap
        < SHELF_OVERLAP_RESTORE_TARGET_M
        and np.linalg.norm(base[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
        <= RESUME_BASE_POSITION_TOLERANCE_M
        and _angle_error(base[2], EXPECTED_RELEASED_BASE_POSE[2])
        <= RESUME_BASE_YAW_TOLERANCE_RAD
        and quaternion_distance(
            book.quaternion, EXPECTED_RELEASED_BOOK_QUATERNION
        ) <= RESUME_BOOK_ROTATION_TOLERANCE_RAD
        and rotation_matrix_distance(
            expected_rotation, hand_base[:3, :3]
        ) <= RESUME_RELATIVE_ROTATION_TOLERANCE_RAD
        and np.linalg.norm(relative[:3, 3] - expected_relative[:3, 3])
        <= RESUME_RELATIVE_POSITION_TOLERANCE_M
        and rotation_matrix_distance(
            relative[:3, :3], expected_relative[:3, :3]
        ) <= RESUME_RELATIVE_ROTATION_TOLERANCE_RAD
    ):
        raise RuntimeError('state is not the guarded low-overlap closed cage')

    _preflight_leg(node, runtime.environment_preflight, arm)
    _fresh_cage_gate(node)
    outward = _outward_world(base)
    yaw = float(base[2])
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    outward_base = np.zeros(3, dtype=float)
    outward_base[:2] = rotation.T @ outward[:2]
    target_position = hand_base[:3, 3] - outward_base * EXTRACTION_STEP_M
    solution, _ = node.chain.solve(
        runtime.pose_matrix(target_position, hand_base[:3, :3]),
        [arm],
        position_tolerance=0.00035,
        orientation_tolerance=0.002,
        max_iterations=320,
        fixed_positions={'torso_lift_joint': float(arm[0])},
    )
    if solution is None:
        raise RuntimeError('shelf-overlap restore IK failed')
    solution = np.asarray(solution, dtype=float)
    if float(np.max(np.abs(solution - arm))) > 0.16:
        raise RuntimeError('shelf-overlap restore IK branch jump')
    attached_corners = _attached_corners(node, book, base, arm)
    if not node._carried_robot_transition_is_safe(
        arm, solution, attached_corners
    ):
        raise RuntimeError('carried payload gate rejected shelf-overlap restore')
    node._held_book_corners = attached_corners.copy()
    _preflight_leg(node, runtime.environment_preflight, solution)
    _fresh_cage_gate(node)
    legs = ((solution, EXTRACTION_DURATION_S, CAGED_EXTRACTION_EVENT),)
    goal, duration = node._make_retained_arm_trajectory_goal(legs)
    moved, contact_loss = node._send_retained_arm_trajectory(
        goal, duration, legs, 'rigid_palm_restore_overlap'
    )
    if not moved:
        reason = 'contact_loss' if contact_loss else 'controller_failure'
        raise RuntimeError(f'shelf-overlap restore failed: {reason}')
    endpoint = node._wait_for_retained_endpoint(
        solution,
        command='rigid_palm_restore_overlap',
        phase=CAGED_EXTRACTION_EVENT,
        leg=1,
    )
    if endpoint is None:
        raise RuntimeError('shelf-overlap restore endpoint was not reached')
    left, right, palm = _fresh_cage_gate(node)
    current_book, current_base, current_stamp, _ = _coherent_scene(
        runtime, newer_than=scene_stamp
    )
    measured = np.asarray(node._measured_left_solution(), dtype=float)
    current_hand = world_hand_pose(
        current_base, node.chain.forward(measured)
    )
    # Reverse the observation order so the symmetric rigid-motion checks see
    # the commanded inward displacement as positive progress.
    guard = extraction_step_guard(
        current_book,
        book,
        current_hand,
        hand_world,
        current_base,
        base,
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=(
            bool(_unexpected_pairs(node))
            or bool(getattr(node, '_target_robot_contact_latched', False))
            or getattr(node, '_payload_hazard_latched', None) is not None
        ),
        measured_aperture_m=float(
            node.joints['gripper_left_finger_joint']
        ),
    )
    attachment = cumulative_attachment_guard(
        book, current_book, hand_world, current_hand
    )
    restored_overlap = float(current_book.maximum[0] - SHELF_FRONT_X_M)
    _emit(
        'shelf_overlap_restore',
        passed=(
            guard.safe
            and attachment.safe
            and restored_overlap >= SHELF_OVERLAP_RESTORE_TARGET_M
        ),
        reason=(guard.reason if not guard.safe else attachment.reason),
        start_shelf_overlap_m=overlap,
        shelf_overlap_m=restored_overlap,
        scene_stamp=current_stamp,
        solution=np.asarray(endpoint, dtype=float).tolist(),
        **guard.metrics,
        **attachment.metrics,
    )
    if not guard.safe:
        raise RuntimeError(f'shelf-overlap endpoint gate failed: {guard.reason}')
    if not attachment.safe:
        raise RuntimeError(
            f'shelf-overlap attachment gate failed: {attachment.reason}'
        )
    if restored_overlap < SHELF_OVERLAP_RESTORE_TARGET_M:
        raise RuntimeError('shelf-overlap restore did not regain 8 mm support')
    return current_book


def _run_retreat(
    node: Any,
    nav: Any,
    runtime: SimpleNamespace,
    retreat_distance: float,
) -> tuple[BookSnapshot, float]:
    reference_book, reference_base, scene_stamp, _ = _coherent_scene(runtime)
    reference_arm = np.asarray(node._measured_left_solution(), dtype=float)
    attached_corners = _attached_corners(
        node,
        reference_book,
        reference_base,
        reference_arm,
    )
    if not node._carried_robot_transition_is_safe(
        reference_arm,
        reference_arm,
        attached_corners,
    ):
        raise RuntimeError('carried robot/payload state rejected before retreat')
    node._held_book_corners = attached_corners.copy()
    _fresh_cage_gate(node)
    if nav.pose is None:
        raise RuntimeError('navigation odometry is unavailable')
    nav.position_tolerance = min(float(nav.position_tolerance), RETREAT_POSITION_TOLERANCE_M)
    nav.probe_terminal_event = None
    outward = _outward_world(reference_base)
    goal = (
        float(nav.pose[0] + retreat_distance * outward[0]),
        float(nav.pose[1] + retreat_distance * outward[1]),
        float(nav.pose[2]),
    )
    nav._accept_goal(goal, profile='carried_retreat')
    last_emit = -1.0
    while nav.goal is not None:
        current_book, current_base, scene_stamp, _ = _coherent_scene(
            runtime,
            newer_than=scene_stamp,
        )
        left, right = node.probe_exact_finger_sides(max_age=0.22)
        palm = node.probe_exact_palm_contact(max_age=0.22)
        guard = retreat_sample_guard(
            reference_book,
            current_book,
            reference_base,
            current_base,
            reference_arm,
            node._measured_left_solution(),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=bool(_unexpected_pairs(node))
            or bool(node._target_robot_contact_latched)
            or getattr(node, '_payload_hazard_latched', None) is not None,
            measured_aperture_m=float(node.joints['gripper_left_finger_joint']),
        )
        travelled = guard.metrics.get('base_outward_travel_m', 0.0)
        if travelled - last_emit >= 0.01 or not guard.safe:
            _emit(
                'retreat_sample',
                passed=guard.safe,
                reason=guard.reason,
                requested_distance_m=retreat_distance,
                **guard.metrics,
            )
            last_emit = travelled
        if not guard.safe:
            nav._finish_goal('cancelled', reason=guard.reason)
            raise RuntimeError(f'carried retreat gate failed: {guard.reason}')
        time.sleep(RETREAT_POLL_S)
    nav._publish_zero()
    if getattr(nav, 'probe_terminal_event', None) != 'reached':
        raise RuntimeError(
            f'navigation retreat ended as {getattr(nav, "probe_terminal_event", None)!r}'
        )
    final_book, final_base, _, _ = _coherent_scene(
        runtime,
        newer_than=scene_stamp,
    )
    left, right, palm = _fresh_cage_gate(node)
    final_guard = retreat_sample_guard(
        reference_book,
        final_book,
        reference_base,
        final_base,
        reference_arm,
        node._measured_left_solution(),
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=bool(_unexpected_pairs(node))
        or bool(node._target_robot_contact_latched)
        or getattr(node, '_payload_hazard_latched', None) is not None,
        measured_aperture_m=float(node.joints['gripper_left_finger_joint']),
    )
    travelled = float(final_guard.metrics.get('base_outward_travel_m', math.nan))
    if not final_guard.safe:
        raise RuntimeError(f'final retreat gate failed: {final_guard.reason}')
    if not math.isfinite(travelled) or abs(travelled - retreat_distance) > 0.012:
        raise RuntimeError(
            f'retreat travelled {travelled:.6f} m, requested {retreat_distance:.6f} m'
        )
    return final_book, travelled


def _probe_types(runtime: SimpleNamespace) -> tuple[type, type]:
    BaseProbe = _probe_node_type(
        runtime.ManipulationNode, TARGET_BOOK_MODEL, preflight_only=False
    )

    class ExtractProbeNode(BaseProbe):
        def _on_command(self, message: Any) -> None:
            self._publish_status('rejected', reason='diagnostic_owns_controller')

        def _run_command(self, command: str) -> None:
            self._publish_status('rejected', reason='diagnostic_owns_controller')

        def _on_contacts(self, message: Any) -> None:
            # The inherited audit historically returned early for contacts
            # with no book name.  Add the missing robot-to-shelf/table/bin
            # latch without weakening its exact-target finger/palm policy.
            super()._on_contacts(message)
            faults = _unexpected_scored_contact_pairs(message)
            if faults:
                self._with_probe_lock(
                    lambda: self.probe_unexpected_pairs.update(faults)
                )

        def _payload_hazard_reason(self, *, max_age: float = 0.20) -> str | None:
            reason = super()._payload_hazard_reason(max_age=max_age)
            if reason is not None:
                return reason
            unexpected = self._with_probe_lock(
                lambda: bool(self.probe_unexpected_pairs)
            )
            return 'unexpected_scored_contact' if unexpected else None

    class RetreatProbeNavigation(runtime.NavigationNode):
        def __init__(self) -> None:
            self.probe_terminal_event: str | None = None
            super().__init__()

        def _on_goal(self, message: Any) -> None:
            self._publish_status('rejected', reason='diagnostic_owns_controller')

        def _on_command(self, message: Any) -> None:
            self._publish_status('rejected', reason='diagnostic_owns_controller')

        def _finish_goal(self, event: str, **fields: Any) -> None:
            if self.goal is not None:
                self.probe_terminal_event = str(event)
            super()._finish_goal(event, **fields)

    return ExtractProbeNode, RetreatProbeNavigation


def _run(
    runtime: SimpleNamespace,
    retreat_distance: float,
    *,
    stop_at_step18: bool = False,
    retreat_from_step18: bool = False,
    restore_shelf_overlap: bool = False,
) -> None:
    if runtime.BOOK != TARGET_BOOK_MODEL:
        raise RuntimeError('runtime target is not the audited seed-101 book')
    ExtractProbeNode, RetreatProbeNavigation = _probe_types(runtime)
    node = ExtractProbeNode()
    nav = RetreatProbeNavigation()
    executor = runtime.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    spin_errors: list[BaseException] = []

    def spin() -> None:
        try:
            executor.spin()
        except BaseException as exc:
            spin_errors.append(exc)

    thread = threading.Thread(target=spin, name='rigid-palm-extract-probe', daemon=True)
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
        # The proven cage ends at the 30 mm preload.  The former 29 mm
        # transport lock exceeds the fingertip-overlap cap and is not claimed.
        node._transport_lock_engaged = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        node.reset_probe_audit()

        book, base, scene_stamp, _ = _coherent_scene(runtime)
        arm = np.asarray(node._measured_left_solution(), dtype=float)
        aperture = float(node.joints.get('gripper_left_finger_joint', math.nan))
        if restore_shelf_overlap:
            stage = 'restoring_shelf_overlap'
            book = _restore_shelf_overlap(
                node, runtime, book, base, scene_stamp
            )
            stage = 'complete'
            _emit(
                'result',
                passed=True,
                stage='rigid_palm_shelf_overlap_restored',
                shelf_overlap_m=float(book.maximum[0] - SHELF_FRONT_X_M),
                book_position=book.position.tolist(),
                book_minimum=book.minimum.tolist(),
                book_maximum=book.maximum.tolist(),
                gripper_opened=False,
                base_motion_commanded=False,
                next_motion_authorized=False,
            )
            return
        starting_step = 0
        try:
            resume = _classify_resume_state(node, book, base, arm, aperture)
        except RuntimeError as classification_error:
            # A previous fail-closed run can stop at any completed interior
            # 5 mm checkpoint.  Reuse the existing independent hand/book
            # quantization guard before constructing only the remaining legs.
            left, right, palm = _fresh_cage_gate(node)
            book, base, scene_stamp, _ = _coherent_scene(
                runtime,
                newer_than=scene_stamp,
            )
            arm = np.asarray(node._measured_left_solution(), dtype=float)
            aperture = float(
                node.joints.get('gripper_left_finger_joint', math.nan)
            )
            left, right = node.probe_exact_finger_sides(max_age=0.22)
            palm = node.probe_exact_palm_contact(max_age=0.22)
            expected_hand = world_hand_pose(
                EXPECTED_RELEASED_BASE_POSE,
                node.chain.forward(EXPECTED_CAGED_JOINTS),
            )
            current_hand = world_hand_pose(
                base,
                node.chain.forward(arm),
            )
            config = runtime.environment_preflight.config
            intermediate = intermediate_extraction_resume_guard(
                book,
                base,
                current_hand,
                expected_hand,
                aperture_m=aperture,
                observed_at=scene_stamp,
                reference_time=(
                    float(node.get_clock().now().nanoseconds) * 1e-9
                ),
                max_state_age_seconds=float(config.max_state_age_seconds),
                future_tolerance_seconds=float(
                    config.future_tolerance_seconds
                ),
                left_contact=left,
                right_contact=right,
                palm_contact=palm,
                unexpected_contacts=_probe_hazard_reason(node) is not None,
            )
            _emit(
                'intermediate_resume_guard',
                passed=intermediate.safe,
                reason=intermediate.reason,
                original_classification_error=str(classification_error),
                **intermediate.metrics,
            )
            if not intermediate.safe:
                raise RuntimeError(
                    f'{classification_error}; intermediate checkpoint '
                    f'rejected: {intermediate.reason}'
                ) from classification_error
            starting_step = int(
                intermediate.metrics['inferred_extraction_step']
            )
            resume = 'intermediate_caged'
        _preflight_leg(node, runtime.environment_preflight, arm)
        left, right, palm = _fresh_cage_gate(node)
        stage = 'resume_verified'
        _emit(
            'resume_verified',
            resume_state=resume,
            left_exact_target_contact=left,
            right_exact_target_contact=right,
            exact_target_palm_contact=palm,
            aperture_m=aperture,
            book_position=book.position.tolist(),
            solution=arm.tolist(),
        )

        shelf_edge_mode = bool(stop_at_step18 or retreat_from_step18)
        if shelf_edge_mode and resume not in {
            'tangent_caged',
            'intermediate_caged',
        }:
            raise RuntimeError(
                'step-18 operation requires a verified tangent or interior cage '
                'checkpoint'
            )

        if resume in {'tangent_caged', 'intermediate_caged'}:
            stage = 'extracting'
            book = _execute_extraction(
                node,
                runtime,
                book,
                base,
                scene_stamp,
                starting_step=starting_step,
                maximum_steps=(SHELF_EDGE_STOP_STEP if shelf_edge_mode else None),
                require_shelf_clearance=not shelf_edge_mode,
            )
            stage = 'shelf_edge_step18' if shelf_edge_mode else 'extracted'
        else:
            _emit('extraction_skipped', reason='already_extracted_caged')

        if stop_at_step18:
            left, right, palm = _fresh_cage_gate(node)
            _emit(
                'result',
                passed=True,
                stage='rigid_palm_extraction_step18',
                resume_state=resume,
                extraction_step=SHELF_EDGE_STOP_STEP,
                shelf_overlap_m=float(book.maximum[0] - SHELF_FRONT_X_M),
                left_exact_target_contact=left,
                right_exact_target_contact=right,
                exact_target_palm_contact=palm,
                book_position=book.position.tolist(),
                book_minimum=book.minimum.tolist(),
                book_maximum=book.maximum.tolist(),
                gripper_opened=False,
                base_motion_commanded=False,
                next_motion_authorized=False,
            )
            return

        stage = 'retreating'
        final_book, travelled = _run_retreat(
            node, nav, runtime, retreat_distance
        )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=(
                'rigid_palm_step18_carried_retreat'
                if retreat_from_step18
                else 'rigid_palm_extraction_and_carried_retreat'
            ),
            resume_state=resume,
            extraction_skipped=(resume == 'already_extracted_caged'),
            extraction_stopped_at_step18=bool(retreat_from_step18),
            requested_retreat_distance_m=retreat_distance,
            measured_retreat_distance_m=travelled,
            book_position=final_book.position.tolist(),
            book_minimum=final_book.minimum.tolist(),
            book_maximum=final_book.maximum.tolist(),
            gripper_opened=False,
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
            reason=f'{type(exc).__name__}: {exc}',
            gripper_opened=False,
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
            'Diagnostic-only same-world rigid-palm extraction and carried retreat'
        )
    )
    parser.add_argument(
        '--retreat-distance',
        type=float,
        default=RETREAT_DISTANCE_DEFAULT_M,
        help='outward carried-base retreat in metres (default: 0.10)',
    )
    parser.add_argument(
        '--stop-at-step-18',
        action='store_true',
        help=(
            'stop with the three-sided cage at the shelf-supported 90 mm '
            'extraction checkpoint; do not command the mobile base'
        ),
    )
    parser.add_argument(
        '--retreat-from-step-18',
        action='store_true',
        help=(
            'extract the mechanically caged book to the shelf-supported '
            '90 mm checkpoint, then immediately test the guarded base retreat'
        ),
    )
    parser.add_argument(
        '--restore-shelf-overlap',
        action='store_true',
        help=(
            'from a verified low-overlap shelf-edge cage, move one guarded '
            '5 mm step inward and stop without opening or base motion'
        ),
    )
    arguments = parser.parse_args()
    selected_special_modes = sum(
        bool(value)
        for value in (
            arguments.stop_at_step_18,
            arguments.retreat_from_step_18,
            arguments.restore_shelf_overlap,
        )
    )
    if selected_special_modes > 1:
        parser.error(
            '--stop-at-step-18, --retreat-from-step-18, and '
            '--restore-shelf-overlap are mutually exclusive'
        )
    distance = float(arguments.retreat_distance)
    if not math.isfinite(distance) or not 0.0 < distance <= RETREAT_DISTANCE_MAXIMUM_M:
        parser.error(
            f'--retreat-distance must be in (0, {RETREAT_DISTANCE_MAXIMUM_M}]'
        )
    if (
        arguments.retreat_from_step_18
        and distance > SHELF_EDGE_RETREAT_MAXIMUM_M
    ):
        parser.error(
            '--retreat-from-step-18 is a bounded shelf-edge diagnostic; '
            f'--retreat-distance must be <= {SHELF_EDGE_RETREAT_MAXIMUM_M}'
        )
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(
            runtime,
            distance,
            stop_at_step18=bool(arguments.stop_at_step_18),
            retreat_from_step18=bool(arguments.retreat_from_step_18),
            restore_shelf_overlap=bool(arguments.restore_shelf_overlap),
        )
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
