#!/usr/bin/env python3
"""Diagnostic-only 30-to-31 mm shelf-edge aperture evidence probe.

This executable is intentionally tied to the recorded seed-101 step-18
checkpoint.  It reads Gazebo entity truth and temporary contact sensors, so it
is engineering evidence only and must never be imported into the competition
mission.  The sole forward command it can publish is a slow 30-to-31 mm left
gripper opening while the arm and base remain fixed.  It never commands the
arm, torso, or base.

Before publishing, all six passive finger-link poses are measured relative to
the palm.  The ideal URDF aperture delta is then applied from those measured
poses, densely sampled, and checked against the complete shelf/book scene and
robot collision model.  A failure after dispatch may publish one slow return
to 30 mm only when a new scene proves that the book, palm, arm, and base are
still at the unchanged supported pose.  Otherwise the probe stops without a
recovery motion.
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

from erc_phase1_solution.kinematics import triangle_meshes_intersect
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    CONTACT_DWELL_S,
    EXPECTED_CAGED_JOINTS,
    GuardResult,
    RESUME_RELATIVE_POSITION_TOLERANCE_M,
    RESUME_RELATIVE_ROTATION_TOLERANCE_RAD,
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
    _angle_error,
    _book_transform,
    _coherent_scene,
    _emit,
    _expected_relative_book_to_hand,
    _fresh_cage_gate,
    _probe_hazard_reason,
    _probe_types,
    _unexpected_pairs,
    rotation_matrix_distance,
    world_hand_pose,
)
from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (
    EDGE_RESUME_BOOK_POSITION_TOLERANCE_M,
    EDGE_RESUME_BOOK_ROTATION_TOLERANCE_RAD,
    EXPECTED_STEP18_BOOK_POSITION,
    EXPECTED_STEP18_BOOK_MAXIMUM_X_M,
    EXPECTED_STEP18_JOINTS,
    _inflated_ordered_corners,
    _moving_arm_environment_collision,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    BookSnapshot,
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    _load_runtime,
    quaternion_distance,
)
from erc_phase1_solution.rigid_palm_live_preflight import (
    MAX_DENSE_SAMPLES,
    MOVING_GRIPPER_LINKS,
    _measured_finger_transforms_relative_to_palm,
    _measured_robot_state,
    _node_time_seconds,
    _require_stable_measured_finger_geometry,
    _require_stable_preflight_inputs,
    conservative_tool_vertex_displacement,
)
from erc_phase1_solution.rigid_palm_preflight import (
    LEFT_GRIPPER_COLLISION_LINKS,
    PALM_COLLISION_LINK,
    GripperSample,
    PreflightResult,
    ProbePhase,
    preflight_gripper_sweep,
    world_gripper_surfaces,
)


APERTURE_EVIDENCE_EVENT = 'rigid_palm_aperture_evidence'
START_APERTURE_M = 0.030
TARGET_APERTURE_M = 0.031
APERTURE_ENDPOINT_TOLERANCE_M = 0.00035
APERTURE_REVERSE_TOLERANCE_M = 0.00010
SLOW_COMMAND_DURATION_S = 2.0
COMMAND_TIMEOUT_SIM_S = 5.0
COMMAND_WALL_FAILSAFE_S = 45.0

TARGET_STEP_MOTION_LIMIT_M = 0.00050
TARGET_CUMULATIVE_MOTION_LIMIT_M = 0.00100
TARGET_ROTATION_LIMIT_RAD = math.radians(0.5)
MINIMUM_SHELF_OVERLAP_M = 0.008
ARM_HOLD_LIMIT_RAD = 0.004
BASE_HOLD_LIMIT_M = 0.00075
BASE_YAW_HOLD_LIMIT_RAD = 0.0010

RESUME_APERTURE_TOLERANCE_M = 0.00020
RESUME_POSITION_TOLERANCE_M = 0.00100
RESUME_ROTATION_TOLERANCE_RAD = math.radians(0.5)
RESUME_BASE_POSITION_TOLERANCE_M = 0.00100
RESUME_BASE_YAW_TOLERANCE_RAD = 0.0010
RESUME_MAXIMUM_SCENE_AGE_S = 0.25
RESUME_FUTURE_TOLERANCE_S = 0.02

# Recovery is deliberately stricter than the evidence endpoint.  It is not a
# general error handler: it merely removes the one-millimetre opening when the
# physical supported pose is still indistinguishable from its baseline.
RECLOSE_BOOK_POSITION_LIMIT_M = 0.00050
RECLOSE_BOOK_ROTATION_LIMIT_RAD = math.radians(0.35)


@dataclass(frozen=True)
class DenseApertureSample:
    """One aperture and its measured-seeded world gripper transforms."""

    aperture_m: float
    transforms: Mapping[str, np.ndarray]


class ApertureEvidenceFailure(RuntimeError):
    """Failure raised after the only forward gripper command was dispatched."""


def _finite_vector(
    value: Sequence[float], length: int, label: str
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{label} must contain {length} finite values')
    return vector


def step18_aperture_resume_guard(
    book: BookSnapshot,
    base: Sequence[float],
    arm: Sequence[float],
    *,
    aperture_m: float,
    observed_at: float,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Strictly recognize the one checkpoint allowed to run this probe."""

    try:
        base_pose = _finite_vector(base, 3, 'base pose')
        arm_state = _finite_vector(
            arm, len(EXPECTED_STEP18_JOINTS), 'arm state'
        )
        aperture = float(aperture_m)
        stamp = float(observed_at)
        now = float(reference_time)
        if not np.all(np.isfinite((aperture, stamp, now))):
            raise ValueError('aperture and scene times must be finite')
        age = now - stamp
        book_position_error = float(
            np.linalg.norm(book.position - EXPECTED_STEP18_BOOK_POSITION)
        )
        book_rotation_error = quaternion_distance(
            book.quaternion,
            EXPECTED_RELEASED_BOOK_QUATERNION,
        )
        book_maximum_error = abs(
            float(book.maximum[0]) - EXPECTED_STEP18_BOOK_MAXIMUM_X_M
        )
        shelf_overlap = float(book.maximum[0] - SHELF_FRONT_X_M)
        arm_error = float(np.max(np.abs(arm_state - EXPECTED_STEP18_JOINTS)))
        base_error = float(
            np.linalg.norm(base_pose[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
        )
        base_yaw_error = _angle_error(
            float(base_pose[2]), float(EXPECTED_RELEASED_BASE_POSE[2])
        )
        aperture_error = abs(aperture - START_APERTURE_M)
    except (TypeError, ValueError):
        return GuardResult(
            False,
            'invalid_observation',
            {'invalid_observation': 1.0},
        )

    metrics = {
        'scene_age_s': age,
        'book_position_error_m': book_position_error,
        'book_rotation_error_rad': book_rotation_error,
        'book_maximum_x_error_m': book_maximum_error,
        'shelf_overlap_m': shelf_overlap,
        'arm_checkpoint_error_rad': arm_error,
        'base_checkpoint_error_m': base_error,
        'base_checkpoint_yaw_error_rad': base_yaw_error,
        'aperture_error_m': aperture_error,
    }
    checks = (
        (
            age <= RESUME_MAXIMUM_SCENE_AGE_S,
            'scene_stale',
        ),
        (
            age >= -RESUME_FUTURE_TOLERANCE_S,
            'scene_from_future',
        ),
        (
            book_position_error <= RESUME_POSITION_TOLERANCE_M,
            'wrong_step18_book_position',
        ),
        (
            book_rotation_error <= RESUME_ROTATION_TOLERANCE_RAD,
            'wrong_step18_book_rotation',
        ),
        (
            book_maximum_error <= RESUME_POSITION_TOLERANCE_M,
            'wrong_step18_book_extent',
        ),
        (arm_error <= ARM_HOLD_LIMIT_RAD, 'wrong_step18_arm'),
        (
            base_error <= RESUME_BASE_POSITION_TOLERANCE_M,
            'wrong_step18_base',
        ),
        (
            base_yaw_error <= RESUME_BASE_YAW_TOLERANCE_RAD,
            'wrong_step18_base_yaw',
        ),
        (
            aperture_error <= RESUME_APERTURE_TOLERANCE_M,
            'wrong_step18_aperture',
        ),
        (shelf_overlap >= MINIMUM_SHELF_OVERLAP_M, 'shelf_overlap_lost'),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (bool(left_contact), 'left_target_contact_missing'),
        (bool(right_contact), 'right_target_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def measured_shelf_edge_resume_guard(
    node: Any,
    book: BookSnapshot,
    base: Sequence[float],
    arm: Sequence[float],
    *,
    aperture_m: float,
    observed_at: float,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Recognize the measured-pose equivalent of the audited shelf edge."""

    try:
        base_pose = _finite_vector(base, 3, 'base pose')
        arm_state = _finite_vector(
            arm, len(EXPECTED_STEP18_JOINTS), 'arm state'
        )
        aperture = float(aperture_m)
        stamp = float(observed_at)
        now = float(reference_time)
        if not np.all(np.isfinite((aperture, stamp, now))):
            raise ValueError('aperture and scene times must be finite')
        hand_base = np.asarray(node.chain.forward(arm_state), dtype=float)
        hand_world = world_hand_pose(base_pose, hand_base)
        relative = np.linalg.inv(hand_world) @ _book_transform(book)
        expected_relative = _expected_relative_book_to_hand(node)
        age = now - stamp
        book_position_error = float(
            np.linalg.norm(book.position - EXPECTED_STEP18_BOOK_POSITION)
        )
        book_maximum_error = abs(
            float(book.maximum[0]) - EXPECTED_STEP18_BOOK_MAXIMUM_X_M
        )
        book_rotation_error = quaternion_distance(
            book.quaternion, EXPECTED_RELEASED_BOOK_QUATERNION
        )
        shelf_overlap = float(book.maximum[0] - SHELF_FRONT_X_M)
        base_error = float(
            np.linalg.norm(base_pose[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
        )
        base_yaw_error = _angle_error(
            float(base_pose[2]), float(EXPECTED_RELEASED_BASE_POSE[2])
        )
        expected_hand_rotation = node.chain.forward(
            EXPECTED_CAGED_JOINTS
        )[:3, :3]
        hand_rotation_error = rotation_matrix_distance(
            hand_base[:3, :3], expected_hand_rotation
        )
        relative_position_error = float(
            np.linalg.norm(
                relative[:3, 3] - expected_relative[:3, 3]
            )
        )
        relative_rotation_error = rotation_matrix_distance(
            relative[:3, :3], expected_relative[:3, :3]
        )
        aperture_error = abs(aperture - START_APERTURE_M)
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return GuardResult(
            False, 'invalid_observation', {'invalid_observation': 1.0}
        )
    metrics = {
        'scene_age_s': age,
        'book_position_error_m': book_position_error,
        'book_maximum_x_error_m': book_maximum_error,
        'book_rotation_error_rad': book_rotation_error,
        'shelf_overlap_m': shelf_overlap,
        'base_checkpoint_error_m': base_error,
        'base_checkpoint_yaw_error_rad': base_yaw_error,
        'hand_rotation_error_rad': hand_rotation_error,
        'book_hand_position_error_m': relative_position_error,
        'book_hand_rotation_error_rad': relative_rotation_error,
        'aperture_error_m': aperture_error,
    }
    checks = (
        (age <= RESUME_MAXIMUM_SCENE_AGE_S, 'scene_stale'),
        (age >= -RESUME_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (
            book_position_error <= EDGE_RESUME_BOOK_POSITION_TOLERANCE_M,
            'wrong_shelf_edge_book_position',
        ),
        (
            book_maximum_error <= EDGE_RESUME_BOOK_POSITION_TOLERANCE_M,
            'wrong_shelf_edge_book_extent',
        ),
        (
            book_rotation_error <= EDGE_RESUME_BOOK_ROTATION_TOLERANCE_RAD,
            'wrong_shelf_edge_book_rotation',
        ),
        (shelf_overlap >= MINIMUM_SHELF_OVERLAP_M, 'shelf_overlap_lost'),
        (
            base_error <= RESUME_BASE_POSITION_TOLERANCE_M,
            'wrong_shelf_edge_base',
        ),
        (
            base_yaw_error <= RESUME_BASE_YAW_TOLERANCE_RAD,
            'wrong_shelf_edge_base_yaw',
        ),
        (
            hand_rotation_error <= RESUME_RELATIVE_ROTATION_TOLERANCE_RAD,
            'wrong_palm_orientation',
        ),
        (
            relative_position_error <= RESUME_RELATIVE_POSITION_TOLERANCE_M,
            'book_not_caged_at_expected_offset',
        ),
        (
            relative_rotation_error <= RESUME_RELATIVE_ROTATION_TOLERANCE_RAD,
            'book_not_caged_at_expected_orientation',
        ),
        (
            aperture_error <= RESUME_APERTURE_TOLERANCE_M,
            'wrong_shelf_edge_aperture',
        ),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (bool(left_contact), 'left_target_contact_missing'),
        (bool(right_contact), 'right_target_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def aperture_evidence_guard(
    reference_book: BookSnapshot,
    previous_book: BookSnapshot,
    current_book: BookSnapshot,
    reference_base: Sequence[float],
    current_base: Sequence[float],
    reference_arm: Sequence[float],
    current_arm: Sequence[float],
    *,
    previous_aperture_m: float,
    measured_aperture_m: float,
    endpoint: bool,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Validate one observed sample of the one-millimetre opening.

    ``left_contact`` and ``right_contact`` are required only at the endpoint;
    the exact target palm contact is required throughout because it is the
    front half of the shelf-plus-palm support bridge.
    """

    try:
        base0 = _finite_vector(reference_base, 3, 'reference base pose')
        base1 = _finite_vector(current_base, 3, 'current base pose')
        arm0 = np.asarray(reference_arm, dtype=float)
        arm1 = np.asarray(current_arm, dtype=float)
        if (
            arm0.ndim != 1
            or arm0.shape != arm1.shape
            or not len(arm0)
            or not np.all(np.isfinite(arm0))
            or not np.all(np.isfinite(arm1))
        ):
            raise ValueError('arm states must be equal-length finite vectors')
        previous_aperture = float(previous_aperture_m)
        aperture = float(measured_aperture_m)
        if not math.isfinite(previous_aperture) or not math.isfinite(aperture):
            raise ValueError('aperture observations must be finite')

        step_motion = float(
            np.linalg.norm(current_book.position - previous_book.position)
        )
        cumulative_motion = float(
            np.linalg.norm(current_book.position - reference_book.position)
        )
        cumulative_rotation = quaternion_distance(
            current_book.quaternion,
            reference_book.quaternion,
        )
        shelf_overlap = float(current_book.maximum[0] - SHELF_FRONT_X_M)
        arm_error = float(np.max(np.abs(arm1 - arm0)))
        base_error = float(np.linalg.norm(base1[:2] - base0[:2]))
        base_yaw_error = _angle_error(float(base1[2]), float(base0[2]))
        aperture_reverse = max(0.0, previous_aperture - aperture)
        endpoint_error = abs(aperture - TARGET_APERTURE_M)
    except (TypeError, ValueError):
        return GuardResult(
            False,
            'invalid_observation',
            {'invalid_observation': 1.0},
        )

    metrics = {
        'target_step_motion_m': step_motion,
        'target_cumulative_motion_m': cumulative_motion,
        'target_cumulative_rotation_rad': cumulative_rotation,
        'shelf_overlap_m': shelf_overlap,
        'arm_hold_error_rad': arm_error,
        'base_hold_error_m': base_error,
        'base_yaw_hold_error_rad': base_yaw_error,
        'aperture_m': aperture,
        'aperture_reverse_m': aperture_reverse,
        'aperture_endpoint_error_m': endpoint_error,
    }
    checks = (
        (step_motion <= TARGET_STEP_MOTION_LIMIT_M, 'target_step_motion'),
        (
            cumulative_motion <= TARGET_CUMULATIVE_MOTION_LIMIT_M,
            'target_cumulative_motion',
        ),
        (
            cumulative_rotation <= TARGET_ROTATION_LIMIT_RAD,
            'target_rotation',
        ),
        (shelf_overlap >= MINIMUM_SHELF_OVERLAP_M, 'shelf_overlap_lost'),
        (arm_error <= ARM_HOLD_LIMIT_RAD, 'arm_moved'),
        (base_error <= BASE_HOLD_LIMIT_M, 'base_moved'),
        (base_yaw_error <= BASE_YAW_HOLD_LIMIT_RAD, 'base_rotated'),
        (
            aperture_reverse <= APERTURE_REVERSE_TOLERANCE_M,
            'aperture_reversed',
        ),
        (
            aperture <= TARGET_APERTURE_M + APERTURE_ENDPOINT_TOLERANCE_M,
            'aperture_overshoot',
        ),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
        (
            not endpoint or endpoint_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'aperture_endpoint_missed',
        ),
        (not endpoint or bool(left_contact), 'left_target_contact_missing'),
        (not endpoint or bool(right_contact), 'right_target_contact_missing'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def reclose_authorization_guard(
    reference_book: BookSnapshot,
    current_book: BookSnapshot,
    reference_base: Sequence[float],
    current_base: Sequence[float],
    reference_arm: Sequence[float],
    current_arm: Sequence[float],
    *,
    measured_aperture_m: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Permit the sole recovery command only from an unchanged support pose."""

    try:
        base0 = _finite_vector(reference_base, 3, 'reference base pose')
        base1 = _finite_vector(current_base, 3, 'current base pose')
        arm0 = np.asarray(reference_arm, dtype=float)
        arm1 = np.asarray(current_arm, dtype=float)
        if (
            arm0.ndim != 1
            or arm0.shape != arm1.shape
            or not len(arm0)
            or not np.all(np.isfinite(arm0))
            or not np.all(np.isfinite(arm1))
        ):
            raise ValueError('arm states must be equal-length finite vectors')
        aperture = float(measured_aperture_m)
        if not math.isfinite(aperture):
            raise ValueError('aperture must be finite')
        book_motion = float(
            np.linalg.norm(current_book.position - reference_book.position)
        )
        book_rotation = quaternion_distance(
            current_book.quaternion, reference_book.quaternion
        )
        shelf_overlap = float(current_book.maximum[0] - SHELF_FRONT_X_M)
        arm_error = float(np.max(np.abs(arm1 - arm0)))
        base_error = float(np.linalg.norm(base1[:2] - base0[:2]))
        base_yaw_error = _angle_error(float(base1[2]), float(base0[2]))
    except (TypeError, ValueError):
        return GuardResult(
            False,
            'invalid_observation',
            {'invalid_observation': 1.0},
        )

    metrics = {
        'target_cumulative_motion_m': book_motion,
        'target_cumulative_rotation_rad': book_rotation,
        'shelf_overlap_m': shelf_overlap,
        'arm_hold_error_rad': arm_error,
        'base_hold_error_m': base_error,
        'base_yaw_hold_error_rad': base_yaw_error,
        'aperture_m': aperture,
    }
    checks = (
        (book_motion <= RECLOSE_BOOK_POSITION_LIMIT_M, 'target_not_unchanged'),
        (
            book_rotation <= RECLOSE_BOOK_ROTATION_LIMIT_RAD,
            'target_rotation_not_unchanged',
        ),
        (shelf_overlap >= MINIMUM_SHELF_OVERLAP_M, 'shelf_overlap_lost'),
        (arm_error <= ARM_HOLD_LIMIT_RAD, 'arm_moved'),
        (base_error <= BASE_HOLD_LIMIT_M, 'base_moved'),
        (base_yaw_error <= BASE_YAW_HOLD_LIMIT_RAD, 'base_rotated'),
        (
            START_APERTURE_M - APERTURE_ENDPOINT_TOLERANCE_M
            <= aperture
            <= TARGET_APERTURE_M + APERTURE_ENDPOINT_TOLERANCE_M,
            'aperture_outside_reclose_band',
        ),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _rigid_transform(
    value: Sequence[Sequence[float]], label: str
) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f'{label} must be a finite 4x4 transform')
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
    ):
        raise ValueError(f'{label} is not a proper rigid transform')
    return transform


def measured_seeded_relative_transforms(
    gripper_kinematics: Any,
    measured_start: Mapping[str, Sequence[Sequence[float]]],
    start_aperture_m: float,
    aperture_m: float,
) -> dict[str, np.ndarray]:
    """Apply the ideal aperture delta from six measured passive-link poses."""

    if set(measured_start) != set(MOVING_GRIPPER_LINKS):
        raise ValueError(
            'measured aperture seed must contain all six passive links'
        )
    start = float(start_aperture_m)
    aperture = float(aperture_m)
    if not math.isfinite(start) or not math.isfinite(aperture):
        raise ValueError('apertures must be finite')
    ideal_start = gripper_kinematics.relative_transforms(start)
    ideal_sample = gripper_kinematics.relative_transforms(aperture)
    result = {
        link: _rigid_transform(ideal_sample[link], f'ideal {link}')
        for link in LEFT_GRIPPER_COLLISION_LINKS
    }
    for link in MOVING_GRIPPER_LINKS:
        measured = _rigid_transform(measured_start[link], f'measured {link}')
        # Left-multiply the measured link by the ideal parent-frame motion.
        # This keeps sample zero exactly measured while preserving the URDF's
        # separating displacement in the palm frame.
        delta = ideal_sample[link] @ np.linalg.inv(ideal_start[link])
        result[link] = _rigid_transform(
            delta @ measured,
            f'measured-seeded {link}',
        )
    return result


def dense_measured_seeded_aperture_sweep(
    *,
    environment: Any,
    node: Any,
    scene: Any,
    arm_positions: Sequence[float],
    measured_start: Mapping[str, Sequence[Sequence[float]]],
    start_aperture_m: float,
    target_aperture_m: float,
) -> tuple[DenseApertureSample, ...]:
    """Densify a stationary-arm aperture sweep below the mesh-step cap."""

    arm = np.asarray(arm_positions, dtype=float)
    if arm.ndim != 1 or not len(arm) or not np.all(np.isfinite(arm)):
        raise ValueError('arm state must be a finite vector')
    start = float(start_aperture_m)
    target = float(target_aperture_m)
    if (
        not np.all(np.isfinite((start, target)))
        or abs(start - target) < 1e-12
        or min(start, target)
        < START_APERTURE_M - APERTURE_ENDPOINT_TOLERANCE_M
        or max(start, target)
        > TARGET_APERTURE_M + APERTURE_ENDPOINT_TOLERANCE_M
    ):
        raise ValueError(
            'aperture sweep is outside the diagnostic 30-31 mm band'
        )
    links = node.chain.link_transforms(arm)
    if PALM_COLLISION_LINK not in links:
        raise ValueError('arm FK omitted the left gripper palm')
    world_palm = _rigid_transform(
        np.asarray(scene.base_transform, dtype=float)
        @ links[PALM_COLLISION_LINK],
        'world palm',
    )

    def transforms(aperture: float) -> dict[str, np.ndarray]:
        relative = measured_seeded_relative_transforms(
            environment.gripper_kinematics,
            measured_start,
            start,
            aperture,
        )
        return {link: world_palm @ relative[link] for link in relative}

    first_transforms = transforms(start)
    dense = [DenseApertureSample(start, first_transforms)]

    def append(
        first: float,
        first_world: Mapping[str, np.ndarray],
        second: float,
        second_world: Mapping[str, np.ndarray],
        depth: int,
    ) -> None:
        displacement = conservative_tool_vertex_displacement(
            environment.model,
            first_world,
            second_world,
        )
        if displacement <= environment.config.max_vertex_step_m + 1e-12:
            dense.append(DenseApertureSample(second, second_world))
            if len(dense) > MAX_DENSE_SAMPLES:
                raise ValueError('dense aperture sample cap was exceeded')
            return
        if depth >= 24:
            raise ValueError('aperture sweep could not be safely densified')
        midpoint = 0.5 * (first + second)
        midpoint_world = transforms(midpoint)
        append(first, first_world, midpoint, midpoint_world, depth + 1)
        append(midpoint, midpoint_world, second, second_world, depth + 1)

    append(start, first_transforms, target, transforms(target), 0)
    return tuple(dense)


def _tool_robot_collision(
    node: Any,
    scene: Any,
    arm: np.ndarray,
    transforms: Mapping[str, np.ndarray],
    model: Any,
) -> tuple[str, str] | None:
    """Check the gripper against non-adjacent official robot meshes."""

    tool = world_gripper_surfaces(model, transforms)
    robot_base = node._world_collision_surfaces(arm)
    base = np.asarray(scene.base_transform, dtype=float)
    watertight = node._watertight_collision_links()
    for tool_link, tool_surface in tool.items():
        tool_bounds = np.asarray(
            [
                np.min(tool_surface, axis=(0, 1)),
                np.max(tool_surface, axis=(0, 1)),
            ]
        )
        for robot_link, base_surface in robot_base.items():
            if (
                tool_link == PALM_COLLISION_LINK
                and robot_link == 'arm_left_7_link'
            ):
                continue
            robot_surface = (
                np.asarray(base_surface, dtype=float) @ base[:3, :3].T
                + base[:3, 3]
            )
            robot_bounds = np.asarray(
                [
                    np.min(robot_surface, axis=(0, 1)),
                    np.max(robot_surface, axis=(0, 1)),
                ]
            )
            if np.any(tool_bounds[1] < robot_bounds[0]) or np.any(
                robot_bounds[1] < tool_bounds[0]
            ):
                continue
            if triangle_meshes_intersect(
                tool_surface,
                robot_surface,
                first_watertight=all(
                    mesh.watertight
                    for mesh in model.meshes_for_link(tool_link)
                ),
                second_watertight=robot_link in watertight,
            ):
                return tool_link, robot_link
    return None


def preflight_measured_seeded_aperture_sweep(
    *,
    node: Any,
    environment: Any,
    target_aperture_m: float,
) -> PreflightResult:
    """Capture a stable seed and preflight the complete static scene."""

    try:
        before_state = _measured_robot_state(node)
        start = float(before_state.aperture_m)
        target = float(target_aperture_m)
        if abs(start - START_APERTURE_M) > APERTURE_ENDPOINT_TOLERANCE_M and (
            target > start
        ):
            raise ValueError(
                'opening preflight did not start at the 30 mm cage'
            )
        scene = environment._read_scene()
        reference_time = _node_time_seconds(node)
        measured = _measured_finger_transforms_relative_to_palm(
            node, scene, before_state
        )
        dense = dense_measured_seeded_aperture_sweep(
            environment=environment,
            node=node,
            scene=scene,
            arm_positions=before_state.positions,
            measured_start=measured,
            start_aperture_m=start,
            target_aperture_m=target,
        )
        self_collision = node._robot_self_collision(before_state.positions)
        if self_collision is not None:
            return PreflightResult(
                False,
                'dense_robot_self_collision',
                f'stationary arm has self-collision: {self_collision}',
            )
        target_world = _inflated_ordered_corners(
            scene.books[TARGET_BOOK_MODEL].corners,
            float(node.carried_book_padding),
        )
        inverse_base = np.linalg.inv(
            np.asarray(scene.base_transform, dtype=float)
        )
        target_base = (
            target_world @ inverse_base[:3, :3].T + inverse_base[:3, 3]
        )
        payload_collision = node._carried_robot_collision(
            before_state.positions, target_base
        )
        if payload_collision is not None:
            return PreflightResult(
                False,
                'target_robot_collision',
                f'static target intersects {payload_collision}',
            )
        environment_collision = _moving_arm_environment_collision(
            node,
            scene,
            before_state.positions,
            padding_m=environment.config.collision_padding_m,
        )
        if environment_collision is not None:
            return PreflightResult(
                False,
                'arm_environment_collision',
                f'stationary arm intersects {environment_collision}',
            )
        for index, sample in enumerate(dense):
            collision = _tool_robot_collision(
                node,
                scene,
                before_state.positions,
                sample.transforms,
                environment.model,
            )
            if collision is not None:
                return PreflightResult(
                    False,
                    'tool_robot_collision',
                    f'dense sample {index} intersects robot: {collision}',
                    sample_index=index,
                    link=collision[0],
                    obstacle=collision[1],
                )
        samples = tuple(
            GripperSample(
                transforms=sample.transforms,
                measured_aperture_m=sample.aperture_m,
                expected_aperture_m=sample.aperture_m,
                observed_at=scene.observed_at,
                phase=ProbePhase.CAGE,
            )
            for sample in dense
        )
        result = preflight_gripper_sweep(
            model=environment.model,
            samples=samples,
            shelf_triangles=scene.shelf_triangles,
            books=scene.books,
            target_book=TARGET_BOOK_MODEL,
            expected_book_names=scene.expected_book_names,
            reference_time=reference_time,
            config=environment.config,
        )
        if not result.safe:
            return result
        after_scene = environment._read_scene()
        after_state = _measured_robot_state(node)
        after_measured = _measured_finger_transforms_relative_to_palm(
            node, after_scene, after_state
        )
        _require_stable_measured_finger_geometry(
            environment.model, measured, after_measured
        )
        _require_stable_preflight_inputs(
            scene, after_scene, before_state, after_state
        )
        return PreflightResult(
            True,
            'clear',
            f'{len(dense)} measured-seeded aperture states passed preflight',
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return PreflightResult(False, 'live_adapter_error', str(error))


def _measured_aperture(node: Any) -> float:
    lock = getattr(node, '_lock', None)
    if lock is None:
        value = node.joints.get('gripper_left_finger_joint', math.nan)
    else:
        with lock:
            value = node.joints.get('gripper_left_finger_joint', math.nan)
    aperture = float(value)
    if not math.isfinite(aperture):
        raise RuntimeError('measured left aperture is unavailable')
    return aperture


def _publish_slow_aperture(node: Any, target_aperture_m: float) -> None:
    target = float(target_aperture_m)
    if target not in {START_APERTURE_M, TARGET_APERTURE_M}:
        raise ValueError('probe may command only 30 mm or 31 mm')
    if not bool(getattr(node, '_probe_actuation_enabled', False)):
        raise RuntimeError('probe actuation is disabled')
    from rclpy.duration import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    message = JointTrajectory()
    message.joint_names = ['gripper_left_finger_joint']
    point = JointTrajectoryPoint()
    point.positions = [target]
    point.time_from_start = Duration(seconds=SLOW_COMMAND_DURATION_S).to_msg()
    message.points = [point]
    node.gripper_pub.publish(message)


def _contact_evidence(
    node: Any, *, require_bilateral: bool
) -> tuple[bool, bool, bool]:
    node._clear_target_contact_samples(reset_robot_contact=False)
    node.clear_probe_finger_evidence()
    node.clear_probe_palm_evidence()
    started_ns = int(node.get_clock().now().nanoseconds)
    if not node._wait_sim_duration(CONTACT_DWELL_S):
        raise RuntimeError('fresh contact dwell was interrupted')
    left, right = node.probe_exact_finger_sides(max_age=0.22)
    palm = node.probe_exact_palm_contact(max_age=0.22, since_ns=started_ns)
    if not palm or (require_bilateral and (not left or not right)):
        raise RuntimeError(
            'required fresh exact-target contact is unavailable'
        )
    hazard = _probe_hazard_reason(node)
    if hazard is not None:
        raise RuntimeError(f'payload/contact hazard is latched: {hazard}')
    return left, right, palm


def _monitor_opening(
    node: Any,
    runtime: SimpleNamespace,
    reference_book: BookSnapshot,
    reference_base: np.ndarray,
    reference_arm: np.ndarray,
    initial_scene_stamp: float,
) -> tuple[BookSnapshot, np.ndarray, np.ndarray, float, float]:
    previous_book = reference_book
    previous_aperture = START_APERTURE_M
    scene_stamp = float(initial_scene_stamp)
    started_sim = node.get_clock().now().nanoseconds / 1e9
    wall_deadline = time.monotonic() + COMMAND_WALL_FAILSAFE_S
    while time.monotonic() < wall_deadline:
        if (
            node.get_clock().now().nanoseconds / 1e9 - started_sim
            > COMMAND_TIMEOUT_SIM_S
        ):
            break
        book, base, stamp, _ = _coherent_scene(runtime, newer_than=scene_stamp)
        arm = np.asarray(node._measured_left_solution(), dtype=float)
        aperture = _measured_aperture(node)
        guard = aperture_evidence_guard(
            reference_book,
            previous_book,
            book,
            reference_base,
            base,
            reference_arm,
            arm,
            previous_aperture_m=previous_aperture,
            measured_aperture_m=aperture,
            endpoint=False,
            left_contact=False,
            right_contact=False,
            palm_contact=node.probe_palm_latched(),
            unexpected_contacts=(
                bool(_unexpected_pairs(node))
                or _probe_hazard_reason(node) is not None
            ),
        )
        _emit(
            'aperture_motion_sample',
            passed=guard.safe,
            reason=guard.reason,
            scene_stamp=stamp,
            **guard.metrics,
        )
        if not guard.safe:
            raise ApertureEvidenceFailure(
                f'aperture motion guard failed: {guard.reason}'
            )
        previous_book = book
        previous_aperture = aperture
        scene_stamp = stamp
        if abs(aperture - TARGET_APERTURE_M) <= APERTURE_ENDPOINT_TOLERANCE_M:
            return book, base, arm, aperture, scene_stamp
    raise ApertureEvidenceFailure('31 mm aperture endpoint timed out')


def _verified_reclose(
    node: Any,
    runtime: SimpleNamespace,
    reference_book: BookSnapshot,
    reference_base: np.ndarray,
    reference_arm: np.ndarray,
    scene_stamp: float,
) -> bool:
    """Attempt the sole recovery, or return false without moving."""

    try:
        book, base, stamp, _ = _coherent_scene(runtime, newer_than=scene_stamp)
        arm = np.asarray(node._measured_left_solution(), dtype=float)
        aperture = _measured_aperture(node)
        if abs(aperture - START_APERTURE_M) <= APERTURE_ENDPOINT_TOLERANCE_M:
            return True
        _, _, palm = _contact_evidence(node, require_bilateral=False)
        authorization = reclose_authorization_guard(
            reference_book,
            book,
            reference_base,
            base,
            reference_arm,
            arm,
            measured_aperture_m=aperture,
            palm_contact=palm,
            unexpected_contacts=(
                bool(_unexpected_pairs(node))
                or _probe_hazard_reason(node) is not None
            ),
        )
        _emit(
            'reclose_authorization',
            passed=authorization.safe,
            reason=authorization.reason,
            **authorization.metrics,
        )
        if not authorization.safe:
            return False
        result = preflight_measured_seeded_aperture_sweep(
            node=node,
            environment=runtime.environment_preflight,
            target_aperture_m=START_APERTURE_M,
        )
        if not result.safe:
            _emit(
                'reclose_preflight',
                passed=False,
                code=result.code,
                detail=result.detail,
            )
            return False
        _publish_slow_aperture(node, START_APERTURE_M)
        started_sim = node.get_clock().now().nanoseconds / 1e9
        wall_deadline = time.monotonic() + COMMAND_WALL_FAILSAFE_S
        while time.monotonic() < wall_deadline:
            if (
                node.get_clock().now().nanoseconds / 1e9 - started_sim
                > COMMAND_TIMEOUT_SIM_S
            ):
                break
            if abs(_measured_aperture(node) - START_APERTURE_M) <= (
                APERTURE_ENDPOINT_TOLERANCE_M
            ):
                _fresh_cage_gate(node)
                final_book, final_base, _, _ = _coherent_scene(
                    runtime, newer_than=stamp
                )
                final_arm = np.asarray(
                    node._measured_left_solution(), dtype=float
                )
                final = reclose_authorization_guard(
                    reference_book,
                    final_book,
                    reference_base,
                    final_base,
                    reference_arm,
                    final_arm,
                    measured_aperture_m=_measured_aperture(node),
                    palm_contact=True,
                    unexpected_contacts=(
                        bool(_unexpected_pairs(node))
                        or _probe_hazard_reason(node) is not None
                    ),
                )
                return bool(final.safe)
            time.sleep(0.02)
        return False
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _run(runtime: SimpleNamespace) -> None:
    if runtime.BOOK != TARGET_BOOK_MODEL:
        raise RuntimeError('runtime target is not the audited seed-101 book')
    ProbeNode, ProbeNavigation = _probe_types(runtime)
    node = ProbeNode()
    nav = ProbeNavigation()
    executor = runtime.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    spin_errors: list[BaseException] = []

    def spin() -> None:
        try:
            executor.spin()
        except BaseException as error:
            spin_errors.append(error)

    thread = threading.Thread(
        target=spin,
        name='rigid-palm-aperture-evidence-probe',
        daemon=True,
    )
    thread.start()
    stage = 'created'
    opening_dispatched = False
    baseline_book: BookSnapshot | None = None
    baseline_base: np.ndarray | None = None
    baseline_arm: np.ndarray | None = None
    last_scene_stamp = -math.inf
    try:
        deadline = time.monotonic() + 30.0
        while (
            len(node.joints) < 8
            or nav.pose is None
            or node.gripper_pub.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8 or nav.pose is None:
            raise RuntimeError('live robot/navigation state is unavailable')
        if node.gripper_pub.get_subscription_count() < 1:
            raise RuntimeError('left gripper command relay is unavailable')

        node._target_book_model = TARGET_BOOK_MODEL
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        node._gravity_supported_payload = False
        node._transport_lock_engaged = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        node.reset_probe_audit()

        left, right, palm = _fresh_cage_gate(node)
        book, base, scene_stamp, _ = _coherent_scene(runtime)
        arm = np.asarray(node._measured_left_solution(), dtype=float)
        aperture = _measured_aperture(node)
        resume = measured_shelf_edge_resume_guard(
            node,
            book,
            base,
            arm,
            aperture_m=aperture,
            observed_at=scene_stamp,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=(
                bool(_unexpected_pairs(node))
                or _probe_hazard_reason(node) is not None
            ),
        )
        if not resume.safe:
            raise RuntimeError(
                f'measured shelf-edge resume rejected: {resume.reason}'
            )
        baseline_book = book
        baseline_base = base.copy()
        baseline_arm = arm.copy()
        last_scene_stamp = scene_stamp
        stage = 'resume_verified'
        _emit(
            'aperture_resume_verified',
            left_exact_target_contact=left,
            right_exact_target_contact=right,
            exact_target_palm_contact=palm,
            aperture_m=aperture,
            arm=arm.tolist(),
            book_position=book.position.tolist(),
            **resume.metrics,
        )

        result = preflight_measured_seeded_aperture_sweep(
            node=node,
            environment=runtime.environment_preflight,
            target_aperture_m=TARGET_APERTURE_M,
        )
        if not result.safe:
            raise RuntimeError(
                f'aperture preflight rejected: {result.code}: {result.detail}'
            )
        # A stable measured preflight can take long enough for contact samples
        # to age out.  Demand a new exact three-point cage immediately before
        # the only forward command.
        _fresh_cage_gate(node)
        current_book, current_base, current_stamp, _ = _coherent_scene(
            runtime, newer_than=scene_stamp
        )
        current_arm = np.asarray(node._measured_left_solution(), dtype=float)
        precommand = reclose_authorization_guard(
            baseline_book,
            current_book,
            baseline_base,
            current_base,
            baseline_arm,
            current_arm,
            measured_aperture_m=_measured_aperture(node),
            palm_contact=True,
            unexpected_contacts=(
                bool(_unexpected_pairs(node))
                or _probe_hazard_reason(node) is not None
            ),
        )
        if not precommand.safe:
            raise RuntimeError(
                f'pre-command unchanged-pose gate failed: {precommand.reason}'
            )
        last_scene_stamp = current_stamp
        stage = 'opening'
        _publish_slow_aperture(node, TARGET_APERTURE_M)
        opening_dispatched = True
        book, base, arm, aperture, scene_stamp = _monitor_opening(
            node,
            runtime,
            baseline_book,
            baseline_base,
            baseline_arm,
            current_stamp,
        )
        last_scene_stamp = scene_stamp
        left, right, palm = _fresh_cage_gate(node)
        final_book, final_base, final_stamp, _ = _coherent_scene(
            runtime, newer_than=scene_stamp
        )
        final_arm = np.asarray(node._measured_left_solution(), dtype=float)
        final_aperture = _measured_aperture(node)
        final_guard = aperture_evidence_guard(
            baseline_book,
            book,
            final_book,
            baseline_base,
            final_base,
            baseline_arm,
            final_arm,
            previous_aperture_m=aperture,
            measured_aperture_m=final_aperture,
            endpoint=True,
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=(
                bool(_unexpected_pairs(node))
                or _probe_hazard_reason(node) is not None
            ),
        )
        if not final_guard.safe:
            raise ApertureEvidenceFailure(
                f'31 mm endpoint gate failed: {final_guard.reason}'
            )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=APERTURE_EVIDENCE_EVENT,
            measured_aperture_m=final_aperture,
            book_position=final_book.position.tolist(),
            scene_stamp=final_stamp,
            gripper_reclosed=False,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
            **final_guard.metrics,
        )
    except Exception as error:
        nav._publish_zero()
        reclosed = False
        if (
            opening_dispatched
            and baseline_book is not None
            and baseline_base is not None
            and baseline_arm is not None
        ):
            reclosed = _verified_reclose(
                node,
                runtime,
                baseline_book,
                baseline_base,
                baseline_arm,
                last_scene_stamp,
            )
        _emit(
            'result',
            passed=False,
            stage=stage,
            reason=f'{type(error).__name__}: {error}',
            gripper_reclosed=reclosed,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
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
            'Diagnostic-only seed-101 step-18 30-to-31 mm aperture probe'
        )
    )
    parser.add_argument(
        '--confirm-step18-aperture-evidence',
        action='store_true',
        help='explicitly authorize the single guarded 30-to-31 mm command',
    )
    arguments = parser.parse_args()
    if not arguments.confirm_step18_aperture_evidence:
        parser.error('--confirm-step18-aperture-evidence is required')
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
