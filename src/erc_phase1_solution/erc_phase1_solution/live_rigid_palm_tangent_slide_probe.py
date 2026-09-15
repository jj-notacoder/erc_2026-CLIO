#!/usr/bin/env python3
"""Diagnostic-only continuous-contact tangent-slide evidence probe.

This executable is intentionally tied to the recorded seed-101 step-18
shelf-supported cage.  It uses Gazebo entity truth and temporary exact-contact
sensors, so its output is engineering evidence only and must never be used by
the competition mission as perception or proof of a grasp.

The only allowed route keeps the base fixed, opens the left gripper from a
verified 30 or 31 mm cage to 47 mm in one-millimetre stages, translates the
unchanged-orientation palm along its horizontalized local tangent axes, and
re-cages to 30.3 mm.  The tangent translation is -4.184678 mm along local y,
followed by +61 mm along local z.  Every future state is dynamically solved
from the measured step-18 arm state.  All six passive finger-link poses are
measured and seed a dense full-scene mesh sweep before the first command and
again before every leg.

No navigation goal is possible here.  A failed opening may reverse toward the
30.3 mm cage only while the book, palm, arm, and base are freshly proven
unchanged and the complete reverse sweep is clear.  There is no fallback arm
motion and no recovery after tangent translation begins.
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

from erc_phase1_solution.live_rigid_palm_aperture_evidence_probe import (
    APERTURE_ENDPOINT_TOLERANCE_M,
    ARM_HOLD_LIMIT_RAD,
    BASE_HOLD_LIMIT_M,
    BASE_YAW_HOLD_LIMIT_RAD,
    MINIMUM_SHELF_OVERLAP_M,
    START_APERTURE_M,
    TARGET_CUMULATIVE_MOTION_LIMIT_M,
    TARGET_STEP_MOTION_LIMIT_M,
    _tool_robot_collision,
    measured_seeded_relative_transforms,
)
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    CONTACT_DWELL_S,
    GuardResult,
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
    _angle_error,
    _book_transform,
    _coherent_scene,
    _emit,
    _fresh_cage_gate,
    _probe_types,
    _unexpected_pairs,
)
from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (
    EXPECTED_STEP18_BOOK_MAXIMUM_X_M,
    EXPECTED_STEP18_BOOK_POSITION,
    EXPECTED_STEP18_JOINTS,
    _inflated_ordered_corners,
    _moving_arm_environment_collision,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
    rotation_matrix_distance,
    world_hand_pose,
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
    densify_joint_samples,
)
from erc_phase1_solution.rigid_palm_preflight import (
    LEFT_GRIPPER_COLLISION_LINKS,
    PALM_COLLISION_LINK,
    GripperSample,
    PreflightResult,
    ProbePhase,
    preflight_gripper_sweep,
)


TANGENT_SLIDE_EVENT = 'rigid_palm_continuous_contact_tangent_slide'
EVIDENCE_APERTURE_M = 0.031
OPEN_APERTURE_M = 0.047
# The measured deep endpoint produces 1.103 mm of fingertip/book overlap at
# exactly 30.0 mm, just beyond the audited 1.0 mm cage cap.  Keeping 0.3 mm of
# extra aperture retains approximately 0.8 mm of bilateral engagement without
# weakening that collision bound.
RECAGE_APERTURE_M = 0.0303
APERTURE_STAGE_M = 0.001
APERTURE_STAGE_DURATION_S = 0.55
APERTURE_COMMAND_TIMEOUT_S = 3.0
APERTURE_COMMAND_WALL_TIMEOUT_S = 45.0
APERTURE_SETTLED_POSITION_TOLERANCE_M = 0.00005
APERTURE_SETTLED_SAMPLE_DELTA_M = 0.000025

LOCAL_Y_RECENTER_M = -0.004184678
LOCAL_Y_STAGE_LIMIT_M = 0.001
LOCAL_Y_RECENTER_ABS_LIMIT_M = 0.0065
LOCAL_Z_INSERTION_M = 0.061
LOCAL_Z_STAGE_LIMIT_M = 0.005
ARM_STAGE_DURATION_S = 0.55

IK_POSITION_TOLERANCE_M = 0.00020
IK_ORIENTATION_TOLERANCE_RAD = 0.00050
IK_MAXIMUM_JOINT_STEP_RAD = 0.080
HAND_ENDPOINT_POSITION_LIMIT_M = 0.00035
HAND_ENDPOINT_ROTATION_LIMIT_RAD = 0.0010

RESUME_BOOK_HORIZONTAL_LIMIT_M = 0.0040
RESUME_BOOK_VERTICAL_LIMIT_M = 0.0020
RESUME_ROTATION_LIMIT_RAD = math.radians(2.0)
RESUME_BASE_POSITION_LIMIT_M = 0.0010
RESUME_BASE_YAW_LIMIT_RAD = 0.0010
RESUME_SCENE_AGE_LIMIT_S = 0.25
RESUME_FUTURE_TOLERANCE_S = 0.02
RESUME_APERTURE_LIMIT_M = 0.00035
RESUME_TORSO_LIMIT_M = 0.0040
RESUME_CAGE_CENTER_X_RANGE_M = (0.073, 0.084)
RESUME_CAGE_CENTER_Y_ABS_LIMIT_M = 0.0065
RESUME_CAGE_CENTER_Z_RANGE_M = (0.055, 0.067)
RESUME_CAGE_ROTATION = np.diag([-1.0, -1.0, 1.0])
RESUME_CAGE_ROTATION_LIMIT_RAD = math.radians(2.0)

BOOK_STEP_ROTATION_LIMIT_RAD = math.radians(0.5)
BOOK_CUMULATIVE_ROTATION_LIMIT_RAD = math.radians(1.0)
STABILITY_SAMPLE_COUNT = 3
STABILITY_MINIMUM_DURATION_S = 0.04
STABILITY_BOOK_CENTER_SPAN_M = 0.00035
STABILITY_BOOK_ROTATION_SPAN_RAD = math.radians(0.35)
STABILITY_ARM_SPAN_RAD = 0.0020
STABILITY_BASE_SPAN_M = 0.00025
STABILITY_BASE_YAW_SPAN_RAD = 0.00040
STABILITY_APERTURE_SPAN_M = 0.00025

RECOVERY_BOOK_POSITION_LIMIT_M = 0.00050
RECOVERY_BOOK_ROTATION_LIMIT_RAD = math.radians(0.35)
MAXIMUM_ROUTE_SAMPLES = MAX_DENSE_SAMPLES


@dataclass(frozen=True)
class TangentWaypoint:
    """One dynamically solved fixed-orientation tangent translation."""

    phase: str
    index: int
    local_y_m: float
    local_z_m: float
    positions: np.ndarray
    target_pose: np.ndarray


@dataclass(frozen=True)
class RouteGeometrySample:
    """One dense arm/aperture state and all nine world tool transforms."""

    positions: np.ndarray
    aperture_m: float
    transforms: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class StageObservation:
    """One coherent target/base/arm sample used by diagnostic pose gates."""

    book: BookSnapshot
    base: np.ndarray
    arm: np.ndarray
    hand_base: np.ndarray
    aperture_m: float
    observed_at: float


class TangentSlideFailure(RuntimeError):
    """Fail-closed terminal error for the diagnostic tangent slide."""


def _finite_vector(
    value: Sequence[float], length: int, label: str
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{label} must contain {length} finite values')
    return vector


def _proper_transform(
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
        raise ValueError(f'{label} must be a proper rigid transform')
    return transform


def classify_step18_resume(
    book: BookSnapshot,
    base: Sequence[float],
    arm: Sequence[float],
    hand_base: Sequence[Sequence[float]],
    *,
    aperture_m: float,
    observed_at: float,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> tuple[GuardResult, float | None]:
    """Accept the fresh step-18 cage or its proven 31 mm palm support.

    At 30 mm the target must still have both finger contacts.  The deliberate
    30-to-31 mm evidence step creates a one-millimetre clearance at each jaw,
    so requiring bilateral contact at that endpoint would reject the exact
    physical state that this route is designed to resume.  The 31 mm state is
    instead authorized only by fresh palm contact plus the same measured
    book/hand, shelf-overlap, arm, base, and scene-freshness checks.
    """

    try:
        base_pose = _finite_vector(base, 3, 'base pose')
        arm_state = _finite_vector(
            arm, len(EXPECTED_STEP18_JOINTS), 'arm state'
        )
        hand = _proper_transform(hand_base, 'measured hand pose')
        aperture = float(aperture_m)
        stamp = float(observed_at)
        now = float(reference_time)
        if not np.all(np.isfinite((aperture, stamp, now))):
            raise ValueError('resume scalars must be finite')
        candidates = (START_APERTURE_M, EVIDENCE_APERTURE_M)
        nominal = min(candidates, key=lambda value: abs(aperture - value))
        aperture_error = abs(aperture - nominal)
        age = now - stamp
        position_delta = (
            book.position - EXPECTED_STEP18_BOOK_POSITION
        )
        horizontal_error = float(np.linalg.norm(position_delta[:2]))
        vertical_error = abs(float(position_delta[2]))
        rotation_error = quaternion_distance(
            book.quaternion, EXPECTED_RELEASED_BOOK_QUATERNION
        )
        maximum_error = abs(
            float(book.maximum[0]) - EXPECTED_STEP18_BOOK_MAXIMUM_X_M
        )
        shelf_overlap = float(book.maximum[0] - SHELF_FRONT_X_M)
        torso_error = abs(float(arm_state[0] - EXPECTED_STEP18_JOINTS[0]))
        base_error = float(
            np.linalg.norm(base_pose[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
        )
        base_yaw_error = _angle_error(
            float(base_pose[2]), float(EXPECTED_RELEASED_BASE_POSE[2])
        )
        hand_world = world_hand_pose(base_pose, hand)
        hand_from_book = np.linalg.inv(hand_world) @ _book_transform(book)
        cage_center = hand_from_book[:3, 3]
        cage_rotation_error = rotation_matrix_distance(
            hand_from_book[:3, :3], RESUME_CAGE_ROTATION
        )
    except (AttributeError, TypeError, ValueError):
        return (
            GuardResult(False, 'invalid_observation', {}),
            None,
        )

    metrics = {
        'scene_age_s': age,
        'book_horizontal_error_m': horizontal_error,
        'book_vertical_error_m': vertical_error,
        'book_rotation_error_rad': rotation_error,
        'book_maximum_x_error_m': maximum_error,
        'shelf_overlap_m': shelf_overlap,
        'torso_checkpoint_error_m': torso_error,
        'base_checkpoint_error_m': base_error,
        'base_checkpoint_yaw_error_rad': base_yaw_error,
        'aperture_error_m': aperture_error,
        'nominal_aperture_m': nominal,
        'bilateral_contact_required': nominal == START_APERTURE_M,
        'cage_center_x_m': float(cage_center[0]),
        'cage_center_y_m': float(cage_center[1]),
        'cage_center_z_m': float(cage_center[2]),
        'cage_rotation_error_rad': cage_rotation_error,
    }
    checks = (
        (age <= RESUME_SCENE_AGE_LIMIT_S, 'scene_stale'),
        (age >= -RESUME_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (
            horizontal_error <= RESUME_BOOK_HORIZONTAL_LIMIT_M,
            'wrong_step18_book_horizontal',
        ),
        (
            vertical_error <= RESUME_BOOK_VERTICAL_LIMIT_M,
            'wrong_step18_book_vertical',
        ),
        (rotation_error <= RESUME_ROTATION_LIMIT_RAD, 'wrong_book_rotation'),
        (maximum_error <= RESUME_BOOK_HORIZONTAL_LIMIT_M, 'wrong_book_extent'),
        (shelf_overlap >= MINIMUM_SHELF_OVERLAP_M, 'shelf_overlap_lost'),
        (torso_error <= RESUME_TORSO_LIMIT_M, 'wrong_step18_torso'),
        (base_error <= RESUME_BASE_POSITION_LIMIT_M, 'wrong_step18_base'),
        (
            base_yaw_error <= RESUME_BASE_YAW_LIMIT_RAD,
            'wrong_step18_base_yaw',
        ),
        (aperture_error <= RESUME_APERTURE_LIMIT_M, 'wrong_aperture'),
        (
            RESUME_CAGE_CENTER_X_RANGE_M[0]
            <= cage_center[0]
            <= RESUME_CAGE_CENTER_X_RANGE_M[1],
            'book_not_above_palm',
        ),
        (
            abs(float(cage_center[1]))
            <= RESUME_CAGE_CENTER_Y_ABS_LIMIT_M,
            'book_not_centered_between_fingers',
        ),
        (
            RESUME_CAGE_CENTER_Z_RANGE_M[0]
            <= cage_center[2]
            <= RESUME_CAGE_CENTER_Z_RANGE_M[1],
            'book_not_at_step18_palm_depth',
        ),
        (
            cage_rotation_error <= RESUME_CAGE_ROTATION_LIMIT_RAD,
            'book_hand_rotation_mismatch',
        ),
        (
            nominal != START_APERTURE_M or bool(left_contact),
            'left_target_contact_missing',
        ),
        (
            nominal != START_APERTURE_M or bool(right_contact),
            'right_target_contact_missing',
        ),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics), None
    return GuardResult(True, 'ok', metrics), float(nominal)


def tangent_stage_guard(
    reference: StageObservation,
    previous: StageObservation,
    current: StageObservation,
    *,
    expected_arm: Sequence[float],
    expected_hand_base: Sequence[Sequence[float]],
    expected_aperture_m: float,
    reference_time: float,
    palm_contact: bool,
    left_contact: bool = False,
    right_contact: bool = False,
    require_bilateral: bool = False,
    unexpected_contacts: bool = False,
) -> GuardResult:
    """Gate one Gazebo-truth sample as a vision-equivalent pose check."""

    try:
        expected_q = _finite_vector(
            expected_arm, len(reference.arm), 'expected arm'
        )
        expected_hand = _proper_transform(
            expected_hand_base, 'expected hand pose'
        )
        expected_aperture = float(expected_aperture_m)
        now = float(reference_time)
        if not np.all(np.isfinite((expected_aperture, now))):
            raise ValueError('expected stage scalars must be finite')
        step_motion = float(
            np.linalg.norm(current.book.position - previous.book.position)
        )
        cumulative_motion = float(
            np.linalg.norm(current.book.position - reference.book.position)
        )
        step_rotation = quaternion_distance(
            previous.book.quaternion, current.book.quaternion
        )
        cumulative_rotation = quaternion_distance(
            reference.book.quaternion, current.book.quaternion
        )
        shelf_overlap = float(current.book.maximum[0] - SHELF_FRONT_X_M)
        base_error = float(
            np.linalg.norm(current.base[:2] - reference.base[:2])
        )
        base_yaw_error = _angle_error(
            float(current.base[2]), float(reference.base[2])
        )
        arm_error = float(np.max(np.abs(current.arm - expected_q)))
        hand_position_error = float(
            np.linalg.norm(
                current.hand_base[:3, 3] - expected_hand[:3, 3]
            )
        )
        hand_rotation_error = rotation_matrix_distance(
            current.hand_base[:3, :3], expected_hand[:3, :3]
        )
        aperture_error = abs(current.aperture_m - expected_aperture)
        scene_age = now - current.observed_at
    except (AttributeError, TypeError, ValueError):
        return GuardResult(False, 'invalid_observation', {})

    metrics = {
        'target_step_motion_m': step_motion,
        'target_cumulative_motion_m': cumulative_motion,
        'target_step_rotation_rad': step_rotation,
        'target_cumulative_rotation_rad': cumulative_rotation,
        'shelf_overlap_m': shelf_overlap,
        'base_hold_error_m': base_error,
        'base_yaw_hold_error_rad': base_yaw_error,
        'arm_endpoint_error_rad': arm_error,
        'hand_endpoint_position_error_m': hand_position_error,
        'hand_endpoint_rotation_error_rad': hand_rotation_error,
        'aperture_endpoint_error_m': aperture_error,
        'scene_age_s': scene_age,
    }
    checks = (
        (scene_age <= RESUME_SCENE_AGE_LIMIT_S, 'scene_stale'),
        (scene_age >= -RESUME_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (step_motion <= TARGET_STEP_MOTION_LIMIT_M, 'target_step_motion'),
        (
            cumulative_motion <= TARGET_CUMULATIVE_MOTION_LIMIT_M,
            'target_cumulative_motion',
        ),
        (
            step_rotation <= BOOK_STEP_ROTATION_LIMIT_RAD,
            'target_step_rotation',
        ),
        (
            cumulative_rotation <= BOOK_CUMULATIVE_ROTATION_LIMIT_RAD,
            'target_cumulative_rotation',
        ),
        (shelf_overlap >= MINIMUM_SHELF_OVERLAP_M, 'shelf_overlap_lost'),
        (base_error <= BASE_HOLD_LIMIT_M, 'base_moved'),
        (base_yaw_error <= BASE_YAW_HOLD_LIMIT_RAD, 'base_rotated'),
        (arm_error <= ARM_HOLD_LIMIT_RAD, 'arm_endpoint_missed'),
        (
            hand_position_error <= HAND_ENDPOINT_POSITION_LIMIT_M,
            'hand_endpoint_position_missed',
        ),
        (
            hand_rotation_error <= HAND_ENDPOINT_ROTATION_LIMIT_RAD,
            'hand_endpoint_rotation_missed',
        ),
        (
            aperture_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'aperture_endpoint_missed',
        ),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (
            not require_bilateral or bool(left_contact),
            'left_target_contact_missing',
        ),
        (
            not require_bilateral or bool(right_contact),
            'right_target_contact_missing',
        ),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def stage_stability_guard(
    samples: Sequence[StageObservation],
) -> GuardResult:
    """Require a short stationary multi-frame diagnostic pose window."""

    try:
        observations = tuple(samples)
        if len(observations) < STABILITY_SAMPLE_COUNT:
            return GuardResult(False, 'insufficient_stability_samples', {})
        duration = float(
            observations[-1].observed_at - observations[0].observed_at
        )
        centers = np.asarray(
            [sample.book.position for sample in observations], dtype=float
        )
        arms = np.asarray([sample.arm for sample in observations], dtype=float)
        bases = np.asarray(
            [sample.base for sample in observations], dtype=float
        )
        apertures = np.asarray(
            [sample.aperture_m for sample in observations], dtype=float
        )
        if not (
            np.all(np.isfinite(centers))
            and np.all(np.isfinite(arms))
            and np.all(np.isfinite(bases))
            and np.all(np.isfinite(apertures))
            and math.isfinite(duration)
        ):
            raise ValueError('stability observations must be finite')
        center_span = float(
            max(
                np.linalg.norm(left - right)
                for left in centers
                for right in centers
            )
        )
        rotation_span = float(
            max(
                quaternion_distance(
                    left.book.quaternion, right.book.quaternion
                )
                for left in observations
                for right in observations
            )
        )
        arm_span = float(np.max(np.ptp(arms, axis=0)))
        base_span = float(
            max(
                np.linalg.norm(left[:2] - right[:2])
                for left in bases
                for right in bases
            )
        )
        base_yaw_span = float(
            max(
                _angle_error(float(left[2]), float(right[2]))
                for left in bases
                for right in bases
            )
        )
        aperture_span = float(np.ptp(apertures))
    except (AttributeError, TypeError, ValueError):
        return GuardResult(False, 'invalid_stability_observation', {})

    metrics = {
        'stability_duration_s': duration,
        'stability_book_center_span_m': center_span,
        'stability_book_rotation_span_rad': rotation_span,
        'stability_arm_span_rad': arm_span,
        'stability_base_span_m': base_span,
        'stability_base_yaw_span_rad': base_yaw_span,
        'stability_aperture_span_m': aperture_span,
    }
    checks = (
        (
            duration >= STABILITY_MINIMUM_DURATION_S,
            'stability_window_too_short',
        ),
        (
            center_span <= STABILITY_BOOK_CENTER_SPAN_M,
            'target_not_stable',
        ),
        (
            rotation_span <= STABILITY_BOOK_ROTATION_SPAN_RAD,
            'target_rotation_not_stable',
        ),
        (arm_span <= STABILITY_ARM_SPAN_RAD, 'arm_not_stable'),
        (base_span <= STABILITY_BASE_SPAN_M, 'base_not_stable'),
        (
            base_yaw_span <= STABILITY_BASE_YAW_SPAN_RAD,
            'base_yaw_not_stable',
        ),
        (
            aperture_span <= STABILITY_APERTURE_SPAN_M,
            'aperture_not_stable',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def opening_recovery_guard(
    reference: StageObservation,
    current: StageObservation,
    *,
    reference_time: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Authorize reverse jaw motion only at the unchanged step-18 pose."""

    result = tangent_stage_guard(
        reference,
        reference,
        current,
        expected_arm=reference.arm,
        expected_hand_base=reference.hand_base,
        expected_aperture_m=current.aperture_m,
        reference_time=reference_time,
        palm_contact=palm_contact,
        unexpected_contacts=unexpected_contacts,
    )
    if not result.safe:
        return result
    position_error = float(
        np.linalg.norm(current.book.position - reference.book.position)
    )
    rotation_error = quaternion_distance(
        current.book.quaternion, reference.book.quaternion
    )
    metrics = dict(result.metrics)
    metrics['recovery_book_position_error_m'] = position_error
    metrics['recovery_book_rotation_error_rad'] = rotation_error
    if position_error > RECOVERY_BOOK_POSITION_LIMIT_M:
        return GuardResult(False, 'target_not_unchanged', metrics)
    if rotation_error > RECOVERY_BOOK_ROTATION_LIMIT_RAD:
        return GuardResult(False, 'target_rotation_not_unchanged', metrics)
    if not (
        RECAGE_APERTURE_M - APERTURE_ENDPOINT_TOLERANCE_M
        <= current.aperture_m
        <= OPEN_APERTURE_M + APERTURE_ENDPOINT_TOLERANCE_M
    ):
        return GuardResult(False, 'aperture_outside_recovery_band', metrics)
    return GuardResult(True, 'ok', metrics)


def bounded_stage_targets(
    start: float, target: float, maximum_step: float
) -> tuple[float, ...]:
    """Partition a scalar leg into monotonic, bounded endpoint targets."""

    first = float(start)
    last = float(target)
    limit = float(maximum_step)
    if not np.all(np.isfinite((first, last, limit))) or limit <= 0.0:
        raise ValueError('stage inputs must be finite with a positive limit')
    distance = abs(last - first)
    if distance <= 1e-12:
        return ()
    direction = math.copysign(1.0, last - first)
    full_steps = int(math.floor(distance / limit + 1e-12))
    values = [
        first + direction * limit * index
        for index in range(1, full_steps + 1)
    ]
    if not values or abs(values[-1] - last) > 1e-12:
        values.append(last)
    else:
        values[-1] = last
    return tuple(float(value) for value in values)


def opening_stage_targets(nominal_start_m: float) -> tuple[float, ...]:
    """Return exact commanded millimetre checkpoints through 47 mm."""

    start = float(nominal_start_m)
    if start not in {START_APERTURE_M, EVIDENCE_APERTURE_M}:
        raise ValueError('opening must start at the 30 or 31 mm checkpoint')
    first_mm = int(round(start * 1000.0)) + 1
    return tuple(
        millimetres / 1000.0
        for millimetres in range(first_mm, 48)
    )


def recage_stage_targets() -> tuple[float, ...]:
    """Return bounded 47-to-30.3 mm cage checkpoints."""

    return bounded_stage_targets(
        OPEN_APERTURE_M,
        RECAGE_APERTURE_M,
        APERTURE_STAGE_M,
    )


def fixed_height_tangent_pose(
    reference_pose: Sequence[Sequence[float]],
    local_y_m: float,
    local_z_m: float,
) -> np.ndarray:
    """Translate on local y/z while fixing palm-local height/orientation."""

    reference = _proper_transform(reference_pose, 'reference hand pose')
    local_y = float(local_y_m)
    local_z = float(local_z_m)
    if not np.all(np.isfinite((local_y, local_z))):
        raise ValueError('local tangent translations must be finite')
    if abs(local_y) > LOCAL_Y_RECENTER_ABS_LIMIT_M + 1e-12:
        raise ValueError('local-y recenter is outside its audited interval')
    if local_z < -1e-12 or local_z > LOCAL_Z_INSERTION_M + 1e-12:
        raise ValueError('local-z insertion is outside its audited interval')
    result = reference.copy()
    result[:3, 3] = (
        reference[:3, 3]
        + reference[:3, :3] @ np.asarray([0.0, local_y, local_z])
    )
    # This is exactly T0 @ Trans(0, local_y, local_z).  Local x is the palm
    # height/normal coordinate and remains fixed; no world-axis translation is
    # substituted for the measured tangent basis.
    return result


def solve_tangent_waypoints(
    chain: Any,
    measured_step18: Sequence[float],
    *,
    local_y_recenter_m: float = LOCAL_Y_RECENTER_M,
) -> tuple[TangentWaypoint, ...]:
    """Dynamically solve the complete tangent route from measured q18."""

    start = _finite_vector(
        measured_step18, len(EXPECTED_STEP18_JOINTS), 'measured step-18 arm'
    )
    lower = np.asarray(chain.lower, dtype=float)
    upper = np.asarray(chain.upper, dtype=float)
    if (
        lower.shape != start.shape
        or upper.shape != start.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(start < lower - 1e-9)
        or np.any(start > upper + 1e-9)
    ):
        raise ValueError('measured step-18 arm is outside its hard limits')
    reference_pose = _proper_transform(
        chain.forward(start), 'measured step-18 hand pose'
    )
    recenter = float(local_y_recenter_m)
    if (
        not math.isfinite(recenter)
        or abs(recenter) > LOCAL_Y_RECENTER_ABS_LIMIT_M
    ):
        raise ValueError('measured local-y recenter is outside its safe band')
    recenter_values = bounded_stage_targets(
        0.0, recenter, LOCAL_Y_STAGE_LIMIT_M
    )
    insertion_values = bounded_stage_targets(
        0.0, LOCAL_Z_INSERTION_M, LOCAL_Z_STAGE_LIMIT_M
    )
    specifications = [
        ('recenter', index, value, 0.0)
        for index, value in enumerate(recenter_values, start=1)
    ] + [
        ('insert', index, recenter, value)
        for index, value in enumerate(insertion_values, start=1)
    ]
    previous = start.copy()
    waypoints: list[TangentWaypoint] = []
    for phase, index, local_y, local_z in specifications:
        target_pose = fixed_height_tangent_pose(
            reference_pose, local_y, local_z
        )
        solution, _ = chain.solve(
            target_pose,
            [previous, start],
            position_tolerance=IK_POSITION_TOLERANCE_M,
            orientation_tolerance=IK_ORIENTATION_TOLERANCE_RAD,
            max_iterations=600,
            fixed_positions={'torso_lift_joint': float(start[0])},
        )
        if solution is None:
            raise RuntimeError(
                f'dynamic tangent IK failed at {phase} stage {index}'
            )
        positions = _finite_vector(
            solution, len(start), f'{phase} IK solution'
        )
        achieved = _proper_transform(
            chain.forward(positions), f'{phase} achieved hand pose'
        )
        error = chain.pose_error(achieved, target_pose)
        maximum_step = float(
            np.max(np.abs(positions[1:] - previous[1:]))
        )
        if (
            maximum_step > IK_MAXIMUM_JOINT_STEP_RAD
            or float(np.linalg.norm(error[:3])) > IK_POSITION_TOLERANCE_M
            or float(np.linalg.norm(error[3:]))
            > IK_ORIENTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                f'dynamic tangent IK is discontinuous or inexact at '
                f'{phase} stage {index}'
            )
        waypoints.append(
            TangentWaypoint(
                phase=phase,
                index=index,
                local_y_m=local_y,
                local_z_m=local_z,
                positions=positions,
                target_pose=target_pose,
            )
        )
        previous = positions
    return tuple(waypoints)


def _world_transforms_from_relative(
    node: Any,
    scene: Any,
    positions: Sequence[float],
    relative: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, np.ndarray]:
    q = _finite_vector(
        positions, len(EXPECTED_STEP18_JOINTS), 'planned arm state'
    )
    if set(relative) != set(LEFT_GRIPPER_COLLISION_LINKS):
        raise ValueError(
            'relative transforms must contain all nine tool links'
        )
    links = node.chain.link_transforms(q)
    if PALM_COLLISION_LINK not in links:
        raise ValueError('arm FK omitted the left gripper palm')
    world_palm = _proper_transform(
        np.asarray(scene.base_transform, dtype=float)
        @ links[PALM_COLLISION_LINK],
        'planned world palm',
    )
    return {
        link: world_palm @ _proper_transform(relative[link], link)
        for link in LEFT_GRIPPER_COLLISION_LINKS
    }


def _dense_aperture_segment(
    *,
    environment: Any,
    node: Any,
    scene: Any,
    positions: Sequence[float],
    measured_start: Mapping[str, Sequence[Sequence[float]]],
    start_aperture_m: float,
    target_aperture_m: float,
) -> tuple[RouteGeometrySample, ...]:
    """Densify an aperture-only segment from all six measured link poses."""

    q = _finite_vector(
        positions, len(EXPECTED_STEP18_JOINTS), 'aperture arm state'
    )
    start = float(start_aperture_m)
    target = float(target_aperture_m)
    if (
        not np.all(np.isfinite((start, target)))
        or abs(start - target) <= 1e-12
        or min(start, target)
        < RECAGE_APERTURE_M - APERTURE_ENDPOINT_TOLERANCE_M
        or max(start, target)
        > OPEN_APERTURE_M + APERTURE_ENDPOINT_TOLERANCE_M
    ):
        raise ValueError('aperture segment is outside the 30-to-47 mm route')

    def transforms(aperture: float) -> dict[str, np.ndarray]:
        relative = measured_seeded_relative_transforms(
            environment.gripper_kinematics,
            measured_start,
            start,
            aperture,
        )
        return _world_transforms_from_relative(node, scene, q, relative)

    first_transforms = transforms(start)
    dense = [RouteGeometrySample(q.copy(), start, first_transforms)]

    def append(
        first: float,
        first_world: Mapping[str, np.ndarray],
        second: float,
        second_world: Mapping[str, np.ndarray],
        depth: int,
    ) -> None:
        displacement = conservative_tool_vertex_displacement(
            environment.model, first_world, second_world
        )
        if displacement <= environment.config.max_vertex_step_m + 1e-12:
            dense.append(RouteGeometrySample(q.copy(), second, second_world))
            if len(dense) > MAXIMUM_ROUTE_SAMPLES:
                raise ValueError('dense aperture sample cap was exceeded')
            return
        if depth >= 24:
            raise ValueError('aperture segment could not be safely densified')
        midpoint = 0.5 * (first + second)
        midpoint_world = transforms(midpoint)
        append(first, first_world, midpoint, midpoint_world, depth + 1)
        append(midpoint, midpoint_world, second, second_world, depth + 1)

    append(start, first_transforms, target, transforms(target), 0)
    return tuple(dense)


def _dense_arm_segment(
    *,
    environment: Any,
    node: Any,
    scene: Any,
    joint_samples: Sequence[Sequence[float]],
    aperture_m: float,
    measured_relative: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[RouteGeometrySample, ...]:
    """Densify a fixed-aperture arm route from measured passive geometry."""

    aperture = float(aperture_m)
    if not math.isfinite(aperture):
        raise ValueError('arm-route aperture must be finite')
    if set(measured_relative) != set(MOVING_GRIPPER_LINKS):
        raise ValueError('arm route requires all six measured passive links')
    ideal = environment.gripper_kinematics.relative_transforms(aperture)
    relative = {
        link: _proper_transform(ideal[link], f'ideal {link}')
        for link in LEFT_GRIPPER_COLLISION_LINKS
    }
    for link in MOVING_GRIPPER_LINKS:
        relative[link] = _proper_transform(
            measured_relative[link], f'measured {link}'
        )

    def transforms(positions: np.ndarray) -> dict[str, np.ndarray]:
        return _world_transforms_from_relative(
            node, scene, positions, relative
        )

    dense = densify_joint_samples(
        model=environment.model,
        joint_samples=joint_samples,
        transform_factory=transforms,
        max_vertex_step_m=environment.config.max_vertex_step_m,
        max_samples=MAXIMUM_ROUTE_SAMPLES,
    )
    return tuple(
        RouteGeometrySample(
            sample.positions.copy(), aperture, sample.transforms
        )
        for sample in dense
    )


def _join_dense_segments(
    *segments: Sequence[RouteGeometrySample],
) -> tuple[RouteGeometrySample, ...]:
    """Join already dense segments and verify every boundary displacement."""

    joined: list[RouteGeometrySample] = []
    for segment in segments:
        current = tuple(segment)
        if not current:
            continue
        if joined:
            first = current[0]
            previous = joined[-1]
            same_state = bool(
                abs(first.aperture_m - previous.aperture_m) <= 1e-12
                and np.max(
                    np.abs(first.positions - previous.positions)
                ) <= 1e-12
            )
            if same_state:
                current = current[1:]
        joined.extend(current)
        if len(joined) > MAXIMUM_ROUTE_SAMPLES:
            raise ValueError('complete dense route exceeds the sample cap')
    if not joined:
        raise ValueError('dense route is empty')
    return tuple(joined)


def build_entire_dense_route(
    *,
    environment: Any,
    node: Any,
    scene: Any,
    measured_start: Mapping[str, Sequence[Sequence[float]]],
    start_positions: Sequence[float],
    start_aperture_m: float,
    waypoints: Sequence[TangentWaypoint],
) -> tuple[RouteGeometrySample, ...]:
    """Build opening, tangent translation, and re-cage before first motion."""

    q0 = _finite_vector(
        start_positions, len(EXPECTED_STEP18_JOINTS), 'route start arm'
    )
    planned = tuple(waypoints)
    if not planned:
        raise ValueError('complete tangent route has no arm waypoints')
    opening = _dense_aperture_segment(
        environment=environment,
        node=node,
        scene=scene,
        positions=q0,
        measured_start=measured_start,
        start_aperture_m=start_aperture_m,
        target_aperture_m=OPEN_APERTURE_M,
    )
    relative_open = measured_seeded_relative_transforms(
        environment.gripper_kinematics,
        measured_start,
        start_aperture_m,
        OPEN_APERTURE_M,
    )
    open_measured = {
        link: relative_open[link] for link in MOVING_GRIPPER_LINKS
    }
    arm = _dense_arm_segment(
        environment=environment,
        node=node,
        scene=scene,
        joint_samples=(q0, *(waypoint.positions for waypoint in planned)),
        aperture_m=OPEN_APERTURE_M,
        measured_relative=open_measured,
    )
    closing = _dense_aperture_segment(
        environment=environment,
        node=node,
        scene=scene,
        positions=planned[-1].positions,
        measured_start=open_measured,
        start_aperture_m=OPEN_APERTURE_M,
        target_aperture_m=RECAGE_APERTURE_M,
    )
    return _join_dense_segments(opening, arm, closing)


def _evaluate_dense_route(
    *,
    node: Any,
    environment: Any,
    scene: Any,
    samples: Sequence[RouteGeometrySample],
    reference_time: float,
) -> PreflightResult:
    """Check self, payload, robot, shelf, and all books at every sample."""

    dense = tuple(samples)
    if not dense:
        return PreflightResult(False, 'missing_samples', 'route is empty')
    if len(dense) > MAXIMUM_ROUTE_SAMPLES:
        return PreflightResult(
            False, 'too_many_samples', 'route exceeds dense sample cap'
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
    previous: RouteGeometrySample | None = None
    for index, sample in enumerate(dense):
        if previous is not None:
            displacement = conservative_tool_vertex_displacement(
                environment.model,
                previous.transforms,
                sample.transforms,
            )
            if displacement > environment.config.max_vertex_step_m + 1e-12:
                return PreflightResult(
                    False,
                    'undersampled_route',
                    f'dense sample {index - 1}->{index} moves '
                    f'{displacement:.9f} m',
                    sample_index=index,
                )
        self_collision = node._robot_self_collision(sample.positions)
        if self_collision is not None:
            return PreflightResult(
                False,
                'robot_self_collision',
                f'dense sample {index}: {self_collision}',
                sample_index=index,
            )
        payload_collision = node._carried_robot_collision(
            sample.positions, target_base
        )
        if payload_collision is not None:
            return PreflightResult(
                False,
                'target_robot_collision',
                f'dense sample {index}: {payload_collision}',
                sample_index=index,
                obstacle=payload_collision,
            )
        environment_collision = _moving_arm_environment_collision(
            node,
            scene,
            sample.positions,
            padding_m=environment.config.collision_padding_m,
        )
        if environment_collision is not None:
            return PreflightResult(
                False,
                'arm_environment_collision',
                f'dense sample {index}: {environment_collision}',
                sample_index=index,
                link=environment_collision[0],
                obstacle=environment_collision[1],
            )
        tool_robot = _tool_robot_collision(
            node,
            scene,
            sample.positions,
            sample.transforms,
            environment.model,
        )
        if tool_robot is not None:
            return PreflightResult(
                False,
                'tool_robot_collision',
                f'dense sample {index}: {tool_robot}',
                sample_index=index,
                link=tool_robot[0],
                obstacle=tool_robot[1],
            )
        previous = sample

    gripper_samples = tuple(
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
        samples=gripper_samples,
        shelf_triangles=scene.shelf_triangles,
        books=scene.books,
        target_book=TARGET_BOOK_MODEL,
        expected_book_names=scene.expected_book_names,
        reference_time=reference_time,
        config=environment.config,
    )
    if not result.safe and result.sample_index is not None:
        index = int(result.sample_index)
        if 0 <= index < len(dense):
            sample = dense[index]
            start_hand = node.chain.forward(dense[0].positions)
            sample_hand = node.chain.forward(sample.positions)
            local_delta = (
                start_hand[:3, :3].T
                @ (sample_hand[:3, 3] - start_hand[:3, 3])
            )
            return PreflightResult(
                result.safe,
                result.code,
                (
                    f'{result.detail}; route aperture '
                    f'{sample.aperture_m:.9f} m, local hand delta '
                    f'{np.asarray(local_delta).tolist()}'
                ),
                result.sample_index,
                result.link,
                result.obstacle,
            )
    return result


def _capture_and_preflight(
    *,
    node: Any,
    environment: Any,
    sample_factory: Any,
) -> PreflightResult:
    """Capture measured geometry, evaluate, then prove inputs stayed fixed."""

    try:
        before_state = _measured_robot_state(node)
        scene = environment._read_scene()
        reference_time = _node_time_seconds(node)
        measured = _measured_finger_transforms_relative_to_palm(
            node, scene, before_state
        )
        samples = sample_factory(before_state, scene, measured)
        result = _evaluate_dense_route(
            node=node,
            environment=environment,
            scene=scene,
            samples=samples,
            reference_time=reference_time,
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
            f'{len(samples)} measured-seeded states passed full preflight',
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return PreflightResult(False, 'live_adapter_error', str(error))


def preflight_entire_tangent_route(
    *,
    node: Any,
    environment: Any,
    start_positions: Sequence[float],
    start_aperture_m: float,
    waypoints: Sequence[TangentWaypoint],
) -> PreflightResult:
    """Preflight every future opening/translation/re-cage state at once."""

    expected_q = np.asarray(start_positions, dtype=float)
    expected_aperture = float(start_aperture_m)

    def build(before: Any, scene: Any, measured: Any) -> Any:
        if (
            before.positions.shape != expected_q.shape
            or np.max(np.abs(before.positions - expected_q))
            > ARM_HOLD_LIMIT_RAD
            or abs(before.aperture_m - expected_aperture)
            > APERTURE_ENDPOINT_TOLERANCE_M
        ):
            raise ValueError('measured route start changed before preflight')
        return build_entire_dense_route(
            environment=environment,
            node=node,
            scene=scene,
            measured_start=measured,
            start_positions=before.positions,
            start_aperture_m=before.aperture_m,
            waypoints=waypoints,
        )

    return _capture_and_preflight(
        node=node, environment=environment, sample_factory=build
    )


def preflight_aperture_leg(
    *, node: Any, environment: Any, target_aperture_m: float
) -> PreflightResult:
    """Re-measure all six passive links and preflight one aperture leg."""

    target = float(target_aperture_m)

    def build(before: Any, scene: Any, measured: Any) -> Any:
        return _dense_aperture_segment(
            environment=environment,
            node=node,
            scene=scene,
            positions=before.positions,
            measured_start=measured,
            start_aperture_m=before.aperture_m,
            target_aperture_m=target,
        )

    return _capture_and_preflight(
        node=node, environment=environment, sample_factory=build
    )


def preflight_arm_leg(
    *, node: Any, environment: Any, target_positions: Sequence[float]
) -> PreflightResult:
    """Re-measure passive links and preflight one fixed-aperture arm leg."""

    target = np.asarray(target_positions, dtype=float)

    def build(before: Any, scene: Any, measured: Any) -> Any:
        if target.shape != before.positions.shape:
            raise ValueError('arm preflight target shape is invalid')
        return _dense_arm_segment(
            environment=environment,
            node=node,
            scene=scene,
            joint_samples=(before.positions, target),
            aperture_m=before.aperture_m,
            measured_relative=measured,
        )

    return _capture_and_preflight(
        node=node, environment=environment, sample_factory=build
    )


def _stage_observation(
    node: Any,
    runtime: SimpleNamespace,
    *,
    newer_than: float | None = None,
) -> StageObservation:
    book, base, stamp, _ = _coherent_scene(
        runtime, newer_than=newer_than
    )
    arm = _finite_vector(
        node._measured_left_solution(),
        len(EXPECTED_STEP18_JOINTS),
        'measured arm',
    )
    aperture = float(
        node.joints.get('gripper_left_finger_joint', math.nan)
    )
    if not math.isfinite(aperture):
        raise RuntimeError('measured left aperture is unavailable')
    hand = _proper_transform(
        node.chain.forward(arm), 'measured base-frame hand pose'
    )
    return StageObservation(book, base, arm, hand, aperture, stamp)


def _has_unexpected_contact(node: Any) -> bool:
    return bool(
        _unexpected_pairs(node)
        or getattr(node, '_target_robot_contact_latched', False)
        or getattr(node, '_payload_hazard_latched', None) is not None
    )


def _fresh_palm_gate(
    node: Any, *, require_bilateral: bool = False
) -> tuple[bool, bool, bool]:
    """Demand a new exact-target palm sample and optional finger pair."""

    node._clear_target_contact_samples(reset_robot_contact=False)
    node.clear_probe_finger_evidence()
    node.clear_probe_palm_evidence()
    started_ns = int(node.get_clock().now().nanoseconds)
    if not node._wait_sim_duration(CONTACT_DWELL_S):
        raise RuntimeError('fresh tangent contact dwell was interrupted')
    left, right = node.probe_exact_finger_sides(max_age=0.22)
    palm = node.probe_exact_palm_contact(
        max_age=0.22, since_ns=started_ns
    )
    if not palm:
        raise RuntimeError('fresh exact-target palm contact is unavailable')
    if require_bilateral and (not left or not right):
        raise RuntimeError('fresh bilateral target contact is unavailable')
    if _has_unexpected_contact(node):
        raise RuntimeError('unexpected target/robot/scored contact is latched')
    return left, right, palm


def _probe_node_type(runtime: SimpleNamespace) -> type:
    """Use the diagnostic parser but retain on fresh palm, not open fingers."""

    BaseProbe, _ = _probe_types(runtime)

    class TangentProbeNode(BaseProbe):
        def __init__(self) -> None:
            self._tangent_palm_support_active = False
            super().__init__()

        def _payload_hazard_reason(
            self, *, max_age: float = 0.20
        ) -> str | None:
            if not self._tangent_palm_support_active:
                return super()._payload_hazard_reason(max_age=max_age)
            latched = getattr(self, '_payload_hazard_latched', None)
            if latched is not None:
                return str(latched)
            if bool(getattr(self, '_target_robot_contact_latched', False)):
                return 'payload_robot_contact'
            if bool(_unexpected_pairs(self)):
                return 'unexpected_scored_contact'
            if not self.probe_exact_palm_contact(max_age=max_age):
                return 'exact_target_palm_contact_lost'
            return None

    return TangentProbeNode


def _build_aperture_trajectory(
    target_aperture_m: float,
    duration_s: float = APERTURE_STAGE_DURATION_S,
) -> Any:
    """Construct the sole allowed gripper message with rclpy Duration."""

    target = float(target_aperture_m)
    duration = float(duration_s)
    if (
        not math.isfinite(target)
        or target
        < RECAGE_APERTURE_M - APERTURE_ENDPOINT_TOLERANCE_M
        or target > OPEN_APERTURE_M + APERTURE_ENDPOINT_TOLERANCE_M
        or not math.isfinite(duration)
        or duration <= 0.0
    ):
        raise ValueError('aperture trajectory target/duration is invalid')
    from rclpy.duration import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    message = JointTrajectory()
    message.joint_names = ['gripper_left_finger_joint']
    point = JointTrajectoryPoint()
    point.positions = [target]
    point.time_from_start = Duration(seconds=duration).to_msg()
    message.points = [point]
    return message


def _publish_aperture(
    node: Any,
    target_aperture_m: float,
    duration_s: float = APERTURE_STAGE_DURATION_S,
) -> None:
    if not bool(getattr(node, '_probe_actuation_enabled', False)):
        raise RuntimeError('diagnostic probe actuation is disabled')
    node.gripper_pub.publish(
        _build_aperture_trajectory(target_aperture_m, duration_s)
    )


def aperture_endpoint_is_settled(
    first_aperture_m: float,
    second_aperture_m: float,
    target_aperture_m: float,
) -> bool:
    """Require two quiet, tightly converged samples before endpoint handoff."""

    first = float(first_aperture_m)
    second = float(second_aperture_m)
    target = float(target_aperture_m)
    if not np.all(np.isfinite((first, second, target))):
        return False
    return bool(
        abs(first - target) <= APERTURE_SETTLED_POSITION_TOLERANCE_M
        and abs(second - target) <= APERTURE_SETTLED_POSITION_TOLERANCE_M
        and abs(second - first) <= APERTURE_SETTLED_SAMPLE_DELTA_M
    )


def _monitor_aperture_leg(
    *,
    node: Any,
    runtime: SimpleNamespace,
    reference: StageObservation,
    previous_stage: StageObservation,
    target_aperture_m: float,
    expected_arm: Sequence[float],
    expected_hand_base: Sequence[Sequence[float]],
) -> StageObservation:
    """Monitor a one-millimetre jaw leg with fresh palm and pose checks."""

    target = float(target_aperture_m)
    direction = math.copysign(1.0, target - previous_stage.aperture_m)
    previous = previous_stage
    stamp = previous.observed_at
    started_ns = int(node.get_clock().now().nanoseconds)
    wall_deadline = time.monotonic() + APERTURE_COMMAND_WALL_TIMEOUT_S
    simulated_timeout_ns = int(APERTURE_COMMAND_TIMEOUT_S * 1e9)
    settled_candidate: StageObservation | None = None
    while time.monotonic() < wall_deadline:
        now_ns = int(node.get_clock().now().nanoseconds)
        if now_ns < started_ns:
            raise TangentSlideFailure('simulation clock moved backwards')
        if now_ns - started_ns >= simulated_timeout_ns:
            break
        current = _stage_observation(node, runtime, newer_than=stamp)
        palm = node.probe_exact_palm_contact(max_age=0.18)
        guard = tangent_stage_guard(
            reference,
            previous,
            current,
            expected_arm=expected_arm,
            expected_hand_base=expected_hand_base,
            expected_aperture_m=current.aperture_m,
            reference_time=_node_time_seconds(node),
            palm_contact=palm,
            unexpected_contacts=_has_unexpected_contact(node),
        )
        if direction > 0.0:
            monotonic = bool(
                current.aperture_m
                >= previous.aperture_m - 0.00010
                and current.aperture_m
                <= target + APERTURE_ENDPOINT_TOLERANCE_M
            )
        else:
            monotonic = bool(
                current.aperture_m
                <= previous.aperture_m + 0.00010
                and current.aperture_m
                >= target - APERTURE_ENDPOINT_TOLERANCE_M
            )
        _emit(
            'tangent_aperture_motion_sample',
            passed=guard.safe and monotonic,
            reason=guard.reason if not guard.safe else (
                'ok' if monotonic else 'aperture_direction_or_overshoot'
            ),
            target_aperture_m=target,
            measured_aperture_m=current.aperture_m,
            **guard.metrics,
        )
        if not guard.safe:
            raise TangentSlideFailure(
                f'aperture motion gate failed: {guard.reason}'
            )
        if not monotonic:
            raise TangentSlideFailure(
                'aperture reversed or overshot its one-millimetre stage'
            )
        previous = current
        stamp = current.observed_at
        if abs(current.aperture_m - target) <= (
            APERTURE_SETTLED_POSITION_TOLERANCE_M
        ):
            if (
                settled_candidate is not None
                and aperture_endpoint_is_settled(
                    settled_candidate.aperture_m,
                    current.aperture_m,
                    target,
                )
            ):
                return current
            settled_candidate = current
        else:
            settled_candidate = None
    reason = (
        'simulated-time timeout'
        if int(node.get_clock().now().nanoseconds) - started_ns
        >= simulated_timeout_ns
        else 'wall-clock failsafe timeout'
    )
    raise TangentSlideFailure(
        f'aperture endpoint {target:.6f} m hit {reason}'
    )


def _stable_stage_endpoint(
    *,
    node: Any,
    runtime: SimpleNamespace,
    reference: StageObservation,
    previous_stage: StageObservation,
    first_sample: StageObservation,
    expected_arm: Sequence[float],
    expected_hand_base: Sequence[Sequence[float]],
    expected_aperture_m: float,
    require_bilateral: bool = False,
) -> StageObservation:
    """Collect a short stable window and apply a final fresh-contact gate."""

    _fresh_palm_gate(node, require_bilateral=require_bilateral)
    observations = [first_sample]
    stamp = first_sample.observed_at
    while len(observations) < STABILITY_SAMPLE_COUNT:
        if not node._wait_sim_duration(0.03):
            raise RuntimeError('stage stability dwell was interrupted')
        if not node.probe_exact_palm_contact(max_age=0.22):
            raise RuntimeError('palm contact aged out during stability dwell')
        sample = _stage_observation(node, runtime, newer_than=stamp)
        observations.append(sample)
        stamp = sample.observed_at
    stability = stage_stability_guard(observations)
    final = observations[-1]
    left, right = node.probe_exact_finger_sides(max_age=0.22)
    palm = node.probe_exact_palm_contact(max_age=0.22)
    endpoint = tangent_stage_guard(
        reference,
        previous_stage,
        final,
        expected_arm=expected_arm,
        expected_hand_base=expected_hand_base,
        expected_aperture_m=expected_aperture_m,
        reference_time=_node_time_seconds(node),
        palm_contact=palm,
        left_contact=left,
        right_contact=right,
        require_bilateral=require_bilateral,
        unexpected_contacts=_has_unexpected_contact(node),
    )
    if not stability.safe:
        raise TangentSlideFailure(
            f'stage stability gate failed: {stability.reason}'
        )
    if not endpoint.safe:
        raise TangentSlideFailure(
            f'stage endpoint gate failed: {endpoint.reason}'
        )
    _emit(
        'tangent_stage_stable',
        passed=True,
        expected_aperture_m=expected_aperture_m,
        **stability.metrics,
        **endpoint.metrics,
    )
    return final


def _execute_aperture_stage(
    *,
    node: Any,
    runtime: SimpleNamespace,
    reference: StageObservation,
    previous: StageObservation,
    target_aperture_m: float,
    phase: str,
    dispatch_hook: Any | None = None,
    expected_arm: Sequence[float] | None = None,
    expected_hand_base: Sequence[Sequence[float]] | None = None,
) -> StageObservation:
    """Preflight, execute, and observe exactly one small aperture stage."""

    preflight = preflight_aperture_leg(
        node=node,
        environment=runtime.environment_preflight,
        target_aperture_m=target_aperture_m,
    )
    if not preflight.safe:
        raise RuntimeError(
            f'{phase} preflight rejected: '
            f'{preflight.code}: {preflight.detail}'
        )
    _fresh_palm_gate(node)
    if dispatch_hook is not None:
        dispatch_hook()
    held_arm = previous.arm if expected_arm is None else expected_arm
    held_hand = (
        previous.hand_base
        if expected_hand_base is None
        else expected_hand_base
    )
    _publish_aperture(node, target_aperture_m)
    reached = _monitor_aperture_leg(
        node=node,
        runtime=runtime,
        reference=reference,
        previous_stage=previous,
        target_aperture_m=target_aperture_m,
        expected_arm=held_arm,
        expected_hand_base=held_hand,
    )
    final = _stable_stage_endpoint(
        node=node,
        runtime=runtime,
        reference=reference,
        previous_stage=previous,
        first_sample=reached,
        expected_arm=held_arm,
        expected_hand_base=held_hand,
        expected_aperture_m=target_aperture_m,
    )
    _emit(
        'tangent_aperture_stage',
        phase=phase,
        target_aperture_m=target_aperture_m,
        measured_aperture_m=final.aperture_m,
        preflight_detail=preflight.detail,
    )
    return final


def _execute_arm_stage(
    *,
    node: Any,
    runtime: SimpleNamespace,
    reference: StageObservation,
    previous: StageObservation,
    waypoint: TangentWaypoint,
) -> StageObservation:
    """Preflight and execute one dynamic IK leg under the palm watchdog."""

    preflight = preflight_arm_leg(
        node=node,
        environment=runtime.environment_preflight,
        target_positions=waypoint.positions,
    )
    if not preflight.safe:
        raise RuntimeError(
            f'{waypoint.phase} stage {waypoint.index} preflight rejected: '
            f'{preflight.code}: {preflight.detail}'
        )
    _fresh_palm_gate(node)
    legs = (
        (
            waypoint.positions,
            ARM_STAGE_DURATION_S,
            f'tangent_{waypoint.phase}',
        ),
    )
    goal, duration = node._make_retained_arm_trajectory_goal(legs)
    moved, contact_loss = node._send_retained_arm_trajectory(
        goal, duration, legs, TANGENT_SLIDE_EVENT
    )
    if not moved:
        reason = 'palm_contact_loss' if contact_loss else 'controller_failure'
        raise TangentSlideFailure(
            f'{waypoint.phase} stage {waypoint.index} failed: {reason}'
        )
    endpoint = node._wait_for_retained_endpoint(
        waypoint.positions,
        command=TANGENT_SLIDE_EVENT,
        phase=waypoint.phase,
        leg=waypoint.index,
        arm_tolerance=ARM_HOLD_LIMIT_RAD,
    )
    if endpoint is None:
        raise TangentSlideFailure(
            f'{waypoint.phase} stage {waypoint.index} endpoint was not held'
        )
    first = _stage_observation(
        node, runtime, newer_than=previous.observed_at
    )
    final = _stable_stage_endpoint(
        node=node,
        runtime=runtime,
        reference=reference,
        previous_stage=previous,
        first_sample=first,
        expected_arm=waypoint.positions,
        expected_hand_base=waypoint.target_pose,
        expected_aperture_m=OPEN_APERTURE_M,
    )
    _emit(
        'tangent_arm_stage',
        phase=waypoint.phase,
        stage=waypoint.index,
        local_y_m=waypoint.local_y_m,
        local_z_m=waypoint.local_z_m,
        solution=final.arm.tolist(),
        preflight_detail=preflight.detail,
    )
    return final


def _verified_opening_reclose(
    *,
    node: Any,
    runtime: SimpleNamespace,
    reference: StageObservation,
    newer_than: float,
) -> bool:
    """Attempt the sole fallback: verified incremental close during opening."""

    try:
        current = _stage_observation(node, runtime, newer_than=newer_than)
        _, _, palm = _fresh_palm_gate(node)
        authorization = opening_recovery_guard(
            reference,
            current,
            reference_time=_node_time_seconds(node),
            palm_contact=palm,
            unexpected_contacts=_has_unexpected_contact(node),
        )
        _emit(
            'tangent_opening_reclose_authorization',
            passed=authorization.safe,
            reason=authorization.reason,
            **authorization.metrics,
        )
        if not authorization.safe:
            return False
        if abs(current.aperture_m - RECAGE_APERTURE_M) <= (
            APERTURE_ENDPOINT_TOLERANCE_M
        ):
            _fresh_cage_gate(node)
            return True
        complete = preflight_aperture_leg(
            node=node,
            environment=runtime.environment_preflight,
            target_aperture_m=RECAGE_APERTURE_M,
        )
        if not complete.safe:
            return False
        targets = bounded_stage_targets(
            current.aperture_m,
            RECAGE_APERTURE_M,
            APERTURE_STAGE_M,
        )
        for target in targets:
            current = _execute_aperture_stage(
                node=node,
                runtime=runtime,
                reference=reference,
                previous=current,
                target_aperture_m=target,
                phase='opening_recovery_reclose',
                expected_arm=reference.arm,
                expected_hand_base=reference.hand_base,
            )
            _, _, palm = _fresh_palm_gate(node)
            unchanged = opening_recovery_guard(
                reference,
                current,
                reference_time=_node_time_seconds(node),
                palm_contact=palm,
                unexpected_contacts=_has_unexpected_contact(node),
            )
            if not unchanged.safe:
                return False
        _fresh_cage_gate(node)
        return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def endpoint_book_hand_transforms(
    book: BookSnapshot,
    base: Sequence[float],
    hand_base: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return measured hand_T_book and its explicit inverse."""

    hand_world = world_hand_pose(
        _finite_vector(base, 3, 'endpoint base'),
        _proper_transform(hand_base, 'endpoint hand'),
    )
    book_world = _proper_transform(_book_transform(book), 'endpoint book')
    hand_from_book = np.linalg.inv(hand_world) @ book_world
    book_from_hand = np.linalg.inv(hand_from_book)
    return hand_from_book, book_from_hand


def measured_local_y_recenter(
    book: BookSnapshot,
    base: Sequence[float],
    hand_base: Sequence[Sequence[float]],
) -> float:
    """Return the bounded palm-local shift that centers under the live book."""

    hand_from_book, _ = endpoint_book_hand_transforms(book, base, hand_base)
    recenter = float(hand_from_book[1, 3])
    if (
        not math.isfinite(recenter)
        or abs(recenter) > LOCAL_Y_RECENTER_ABS_LIMIT_M
    ):
        raise ValueError('measured book is outside the lateral recenter band')
    return recenter


def _run(runtime: SimpleNamespace) -> None:
    if runtime.BOOK != TARGET_BOOK_MODEL:
        raise RuntimeError('runtime target is not the audited seed-101 book')
    ProbeNode = _probe_node_type(runtime)
    node = ProbeNode()
    executor = runtime.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    spin_errors: list[BaseException] = []

    def spin() -> None:
        try:
            executor.spin()
        except BaseException as error:
            spin_errors.append(error)

    thread = threading.Thread(
        target=spin,
        name='rigid-palm-continuous-contact-tangent-slide',
        daemon=True,
    )
    thread.start()
    stage = 'created'
    opening_phase = False
    opening_motion_dispatched = False
    reference: StageObservation | None = None
    last_stamp = -math.inf

    def mark_opening_dispatched() -> None:
        nonlocal opening_motion_dispatched
        opening_motion_dispatched = True

    try:
        deadline = time.monotonic() + 30.0
        while (
            len(node.joints) < 8
            or node.gripper_pub.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('live torso/left-arm state is unavailable')
        if node.gripper_pub.get_subscription_count() < 1:
            raise RuntimeError('left gripper trajectory relay is unavailable')

        node._target_book_model = TARGET_BOOK_MODEL
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        node._gravity_supported_payload = True
        node._transport_lock_engaged = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        node.reset_probe_audit()

        left, right, palm = _fresh_palm_gate(node)
        reference = _stage_observation(node, runtime)
        resume, nominal_start = classify_step18_resume(
            reference.book,
            reference.base,
            reference.arm,
            reference.hand_base,
            aperture_m=reference.aperture_m,
            observed_at=reference.observed_at,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_has_unexpected_contact(node),
        )
        if not resume.safe or nominal_start is None:
            raise RuntimeError(f'step-18 resume rejected: {resume.reason}')
        local_y_recenter = measured_local_y_recenter(
            reference.book,
            reference.base,
            reference.hand_base,
        )
        waypoints = solve_tangent_waypoints(
            node.chain,
            reference.arm,
            local_y_recenter_m=local_y_recenter,
        )
        complete = preflight_entire_tangent_route(
            node=node,
            environment=runtime.environment_preflight,
            start_positions=reference.arm,
            start_aperture_m=reference.aperture_m,
            waypoints=waypoints,
        )
        _emit(
            'tangent_complete_route_preflight',
            passed=complete.safe,
            code=complete.code,
            detail=complete.detail,
            sample_index=complete.sample_index,
            link=complete.link,
            obstacle=complete.obstacle,
        )
        if not complete.safe:
            raise RuntimeError(
                f'complete tangent route preflight rejected: '
                f'{complete.code}: {complete.detail}'
            )

        # Re-prove the exact cage after the potentially lengthy complete-route
        # mesh pass.  No command exists before this second gate succeeds.
        left, right, palm = _fresh_palm_gate(node)
        current = _stage_observation(
            node, runtime, newer_than=reference.observed_at
        )
        unchanged = tangent_stage_guard(
            reference,
            reference,
            current,
            expected_arm=reference.arm,
            expected_hand_base=reference.hand_base,
            expected_aperture_m=nominal_start,
            reference_time=_node_time_seconds(node),
            palm_contact=palm,
            left_contact=left,
            right_contact=right,
            require_bilateral=(nominal_start == START_APERTURE_M),
            unexpected_contacts=_has_unexpected_contact(node),
        )
        if not unchanged.safe:
            raise RuntimeError(
                f'post-preflight step-18 gate failed: {unchanged.reason}'
            )
        last_stamp = current.observed_at
        _emit(
            'tangent_slide_resume_verified',
            measured_aperture_m=current.aperture_m,
            complete_preflight_detail=complete.detail,
            dynamic_waypoint_count=len(waypoints),
            measured_local_y_recenter_m=local_y_recenter,
            audited_q18_reference=EXPECTED_STEP18_JOINTS.tolist(),
            measured_q18=current.arm.tolist(),
            diagnostic_truth_and_contacts_only=True,
            **resume.metrics,
        )

        node._tangent_palm_support_active = True
        opening_phase = True
        stage = 'opening_to_47mm'
        for target in opening_stage_targets(nominal_start):
            current = _execute_aperture_stage(
                node=node,
                runtime=runtime,
                reference=reference,
                previous=current,
                target_aperture_m=target,
                phase='opening',
                dispatch_hook=mark_opening_dispatched,
                expected_arm=reference.arm,
                expected_hand_base=reference.hand_base,
            )
            last_stamp = current.observed_at
        opening_phase = False

        stage = 'tangent_translation'
        for waypoint in waypoints:
            current = _execute_arm_stage(
                node=node,
                runtime=runtime,
                reference=reference,
                previous=current,
                waypoint=waypoint,
            )
            last_stamp = current.observed_at

        stage = 'recaging_to_30_3mm'
        for target in recage_stage_targets():
            current = _execute_aperture_stage(
                node=node,
                runtime=runtime,
                reference=reference,
                previous=current,
                target_aperture_m=target,
                phase='recage',
                expected_arm=waypoints[-1].positions,
                expected_hand_base=waypoints[-1].target_pose,
            )
            last_stamp = current.observed_at

        left, right, palm = _fresh_cage_gate(node)
        final = _stage_observation(
            node, runtime, newer_than=current.observed_at
        )
        final_guard = tangent_stage_guard(
            reference,
            current,
            final,
            expected_arm=waypoints[-1].positions,
            expected_hand_base=waypoints[-1].target_pose,
            expected_aperture_m=RECAGE_APERTURE_M,
            reference_time=_node_time_seconds(node),
            palm_contact=palm,
            left_contact=left,
            right_contact=right,
            require_bilateral=True,
            unexpected_contacts=_has_unexpected_contact(node),
        )
        if not final_guard.safe:
            raise TangentSlideFailure(
                f'final three-point cage failed: {final_guard.reason}'
            )
        hand_from_book, book_from_hand = endpoint_book_hand_transforms(
            final.book, final.base, final.hand_base
        )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=TANGENT_SLIDE_EVENT,
            measured_aperture_m=final.aperture_m,
            book_position=final.book.position.tolist(),
            measured_endpoint_q=final.arm.tolist(),
            hand_T_book=hand_from_book.tolist(),
            book_T_hand=book_from_hand.tolist(),
            scene_stamp=final.observed_at,
            base_navigation_commanded=False,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
            **final_guard.metrics,
        )
    except Exception as error:
        reclosed = False
        if (
            opening_phase
            and opening_motion_dispatched
            and reference is not None
        ):
            reclosed = _verified_opening_reclose(
                node=node,
                runtime=runtime,
                reference=reference,
                newer_than=last_stamp,
            )
        _emit(
            'result',
            passed=False,
            stage=stage,
            reason=f'{type(error).__name__}: {error}',
            opening_reclosed=reclosed,
            base_navigation_commanded=False,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
        )
        raise
    finally:
        node._tangent_palm_support_active = False
        node._cancel.set()
        executor.shutdown()
        thread.join(timeout=3.0)
        node.destroy_node()
        if spin_errors:
            raise RuntimeError(f'executor spin failed: {spin_errors[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Diagnostic-only seed-101 continuous-contact tangent slide'
        )
    )
    parser.add_argument(
        '--confirm-diagnostic-tangent-slide',
        action='store_true',
        help='authorize the guarded diagnostic route after all preflights',
    )
    arguments = parser.parse_args()
    if not arguments.confirm_diagnostic_tangent_slide:
        parser.error('--confirm-diagnostic-tangent-slide is required')
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
