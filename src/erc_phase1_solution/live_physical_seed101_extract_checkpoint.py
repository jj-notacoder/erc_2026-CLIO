#!/usr/bin/env python3
"""Resume the exact seed-101 pressure grasp and extract with the left arm.

This launcher is deliberately checkpoint-specific.  It starts only while the
partially extracted pressure-grasp world is paused, reconstructs the held state
from fresh bilateral force/contact evidence, lifts the book 5 mm to unload
shelf friction, follows an offline-baked 245 mm extraction route, and pauses
again with the gripper closed.  It never solves IK live, opens the gripper,
drives the base, commands the right arm/head, or mutates an entity pose.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

from live_physical_seed101_pick_checkpoint import (
    BOOK,
    BOOK_HALF_EXTENTS_M,
    BOOK_LAYOUT,
    EntityPose,
    EXPECTED_SEED,
    GateResult,
    LEFT_GRIPPER_JOINTS,
    NOMINAL_BOOK_QUATERNION,
    OFFICIAL_PASSIVE_RIGHT_GRIPPER_Q,
    PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
    PASSIVE_JOINT_STATIONARY_LIMIT_RAD,
    Pose2,
    RIGHT_GRIPPER_JOINTS,
    SHELF,
    book_maximum_world_x,
    book_minimum_world_z,
    book_poses_from_dynamic_pose,
    certificate_asset_gate,
    dense_arm_waypoints,
    entity_pose_from_dynamic_pose,
    expected_joint_gate,
    quaternion_distance,
    read_dynamic_pose_message,
    read_full_pose_message,
    scene_gate,
    shelf_gate,
    stationary_joint_gate,
    _cmd_vel_gate,
    _emit,
    _joint_generation_snapshot,
    _joint_snapshot,
    _require,
    _wait_for_joint_update,
)
from erc_phase1_solution.seed101_loaded_extraction_certificate import (
    ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP,
    AUDIT,
    CHECKPOINT_BASE_WORLD_XYYAW,
    CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M,
    CHECKPOINT_BOOK_POSITION_WORLD_M,
    CHECKPOINT_BOOK_QUATERNION_XYZW,
    CHECKPOINT_GRIPPER_MASTER_M,
    CHECKPOINT_HEAD_Q2,
    CHECKPOINT_LEFT_Q8,
    CHECKPOINT_RIGHT_Q7,
    LOADED_RECOVERY_Q8,
    MAXIMUM_RELATIVE_ROTATION_SLIP_PER_LEG_RAD,
    MAXIMUM_RELATIVE_TRANSLATION_SLIP_PER_LEG_M,
    MICRO_LIFT_COUNT,
    MICRO_LIFT_STEP_M,
    MICRO_LIFT_TOTAL_M,
    MINIMUM_MICRO_LIFT_DURATION_S,
    MINIMUM_OUTWARD_DURATION_S,
    OUTWARD_COUNT,
    OUTWARD_STEP_M,
    OUTWARD_TOTAL_M,
    RECOVERY_ROUTE,
    validate_certificate,
)


WORLD_STATS_TOPIC = '/world/erc_world/stats'
WORLD_CONTROL_SERVICE = '/world/erc_world/control'
DYNAMIC_POSE_TOPIC = '/world/erc_world/dynamic_pose/info'

CHECKPOINT_ARM_LIMIT_RAD = 0.00025
CHECKPOINT_GRIPPER_LIMIT_M = 0.00025
CHECKPOINT_BASE_POSITION_LIMIT_M = 0.00035
CHECKPOINT_BASE_YAW_LIMIT_RAD = 0.00050
CHECKPOINT_BOOK_POSITION_LIMIT_M = 0.00075
CHECKPOINT_BOOK_ROTATION_LIMIT_RAD = 0.0030
CHECKPOINT_STABILITY_TRANSLATION_LIMIT_M = 0.00050
CHECKPOINT_STABILITY_ROTATION_LIMIT_RAD = 0.0020

PAYLOAD_APERTURE_DRIFT_LIMIT_M = 0.00035
PAYLOAD_ATTACHMENT_POSITION_LIMIT_M = 0.0025
PAYLOAD_TOTAL_ROTATION_LIMIT_RAD = 0.015
PAYLOAD_TOTAL_CORNER_DISPLACEMENT_LIMIT_M = 0.005
PAYLOAD_LEG_TRANSLATION_SLIP_LIMIT_M = (
    MAXIMUM_RELATIVE_TRANSLATION_SLIP_PER_LEG_M
)
PAYLOAD_LEG_ROTATION_LIMIT_RAD = MAXIMUM_RELATIVE_ROTATION_SLIP_PER_LEG_RAD
PAYLOAD_CONTINUOUS_TRANSLATION_LIMIT_M = 0.004
PAYLOAD_SHELF_CONTINUOUS_TRANSLATION_LIMIT_M = 0.00075
PAYLOAD_CONTINUOUS_ROTATION_LIMIT_RAD = 0.025
PAYLOAD_CONTINUOUS_CORNER_DISPLACEMENT_LIMIT_M = 0.008
PAYLOAD_INWARD_REGRESSION_LIMIT_M = 0.00075
PAYLOAD_CONSERVATIVE_RADIUS_M = 0.150
PAYLOAD_TARGET_POSE_MAXIMUM_WALL_AGE_S = 0.25
PAYLOAD_BASE_POSITION_LIMIT_M = 0.00035
PAYLOAD_BASE_YAW_LIMIT_RAD = 0.00050
FINAL_SHELF_CLEARANCE_REQUIRED_M = 0.020
MICRO_LIFT_LEG_SIM_SECONDS = max(1.0, MINIMUM_MICRO_LIFT_DURATION_S)
OUTWARD_LEG_SIM_SECONDS = max(1.0, MINIMUM_OUTWARD_DURATION_S)
PRESSURE_LOCK_MINIMUM_WIDTH_M = 0.028

AUDIT_DENSE_SAMPLES = int(AUDIT['dense_samples'])
AUDIT_SELF_AABB_MARGIN_M = float(AUDIT['minimum_self_aabb_clearance_m'])
AUDIT_PADDED_PAYLOAD_ROBOT_MARGIN_M = float(
    AUDIT['minimum_15mm_padded_payload_robot_aabb_clearance_m']
)
AUDIT_CLOSED_GRIPPER_SHELF_MARGIN_M = float(
    AUDIT['minimum_closed_gripper_shelf_triangle_aabb_clearance_m']
)
ROUTE_DENSE_MAXIMUM_INCREMENT_RAD = float(
    AUDIT['maximum_joint_increment_rad']
)
ROUTE_FINAL_SHELF_CLEARANCE_M = float(
    AUDIT['final_book_shelf_face_clearance_m']
)
ROUTE_BOOK_MAXIMUM_START_WORLD_X_M = max(
    corner[0] for corner in CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M
)
ROUTE_SHELF_FRONT_WORLD_X_M = (
    ROUTE_BOOK_MAXIMUM_START_WORLD_X_M
    + float(RECOVERY_ROUTE[0]['expected_book_shelf_face_clearance_m'])
)
ROUTE_FIRST_CLEAR_STEP = next(
    index
    for index, row in enumerate(RECOVERY_ROUTE)
    if float(row['expected_book_shelf_face_clearance_m']) >= 0.0
)
ROUTE_SUPPORT_FLOOR_WORLD_Z_M = (
    min(corner[2] for corner in CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M)
    - float(RECOVERY_ROUTE[0]['expected_book_floor_signed_m'])
)


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def resume_checkpoint_gate(
    arm_q8: Sequence[float],
    gripper_m: float,
    base: Pose2,
    book: EntityPose,
    *,
    expected_arm_q8: Sequence[float] = CHECKPOINT_LEFT_Q8,
    expected_gripper_m: float = CHECKPOINT_GRIPPER_MASTER_M,
    expected_base_xyyaw: Sequence[float] = CHECKPOINT_BASE_WORLD_XYYAW,
    expected_book_position: Sequence[float] = CHECKPOINT_BOOK_POSITION_WORLD_M,
    expected_book_quaternion: Sequence[float] = (
        CHECKPOINT_BOOK_QUATERNION_XYZW
    ),
    arm_limit_rad: float = CHECKPOINT_ARM_LIMIT_RAD,
    gripper_limit_m: float = CHECKPOINT_GRIPPER_LIMIT_M,
    base_position_limit_m: float = CHECKPOINT_BASE_POSITION_LIMIT_M,
    base_yaw_limit_rad: float = CHECKPOINT_BASE_YAW_LIMIT_RAD,
    book_position_limit_m: float = CHECKPOINT_BOOK_POSITION_LIMIT_M,
    book_rotation_limit_rad: float = CHECKPOINT_BOOK_ROTATION_LIMIT_RAD,
) -> GateResult:
    """Bind execution to the exact paused pressure-pick checkpoint."""
    arm = expected_joint_gate(
        arm_q8,
        expected_arm_q8,
        limit=float(arm_limit_rad),
        failure_reason='held_checkpoint_left_arm_mismatch',
    )
    gripper = expected_joint_gate(
        (float(gripper_m),),
        (float(expected_gripper_m),),
        limit=float(gripper_limit_m),
        failure_reason='held_checkpoint_gripper_mismatch',
    )
    base_position_error = math.hypot(
        base.x - float(expected_base_xyyaw[0]),
        base.y - float(expected_base_xyyaw[1]),
    )
    base_yaw_error = abs(math.atan2(
        math.sin(base.yaw - float(expected_base_xyyaw[2])),
        math.cos(base.yaw - float(expected_base_xyyaw[2])),
    ))
    book_position_error = _distance(
        book.position,
        expected_book_position,
    )
    book_rotation_error = quaternion_distance(
        book.quaternion,
        expected_book_quaternion,
    )
    metrics = {
        'maximum_left_joint_error_rad': float(
            arm.metrics.get('maximum_joint_error', math.inf)
        ),
        'gripper_error_m': float(
            gripper.metrics.get('maximum_joint_error', math.inf)
        ),
        'base_position_error_m': base_position_error,
        'base_yaw_error_rad': base_yaw_error,
        'book_position_error_m': book_position_error,
        'book_rotation_error_rad': book_rotation_error,
    }
    for accepted, reason in (
        (arm.ok, arm.reason),
        (gripper.ok, gripper.reason),
        (
            base_position_error <= float(base_position_limit_m),
            'held_checkpoint_base_position_mismatch',
        ),
        (
            base_yaw_error <= float(base_yaw_limit_rad),
            'held_checkpoint_base_yaw_mismatch',
        ),
        (
            book_position_error <= float(book_position_limit_m),
            'held_checkpoint_book_position_mismatch',
        ),
        (
            book_rotation_error <= float(book_rotation_limit_rad),
            'held_checkpoint_book_rotation_mismatch',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'certified_held_checkpoint', metrics)


def held_stability_gate(
    first: EntityPose,
    second: EntityPose,
    *,
    translation_limit_m: float = CHECKPOINT_STABILITY_TRANSLATION_LIMIT_M,
    rotation_limit_rad: float = CHECKPOINT_STABILITY_ROTATION_LIMIT_RAD,
) -> GateResult:
    translation = _distance(first.position, second.position)
    rotation = quaternion_distance(first.quaternion, second.quaternion)
    metrics = {
        'held_stability_translation_m': translation,
        'held_stability_rotation_rad': rotation,
    }
    if translation > float(translation_limit_m):
        return GateResult(False, 'held_book_not_stable_before_resume', metrics)
    if rotation > float(rotation_limit_rad):
        return GateResult(False, 'held_book_rotated_before_resume', metrics)
    return GateResult(True, 'held_book_stable_before_resume', metrics)


def resume_seed101_scene_gate(
    first: Mapping[str, EntityPose],
    second: Mapping[str, EntityPose],
) -> GateResult:
    """Apply nominal seed gates to every book except the already-held target."""
    if set(first) != set(BOOK_LAYOUT) or set(second) != set(BOOK_LAYOUT):
        return GateResult(False, 'seed101_book_set_mismatch', {
            'first_book_count': float(len(first)),
            'second_book_count': float(len(second)),
        })
    nominal_target = EntityPose(
        BOOK_LAYOUT[BOOK],
        NOMINAL_BOOK_QUATERNION,
    )
    sanitized_first = dict(first)
    sanitized_second = dict(second)
    sanitized_first[BOOK] = nominal_target
    sanitized_second[BOOK] = nominal_target
    result = scene_gate(sanitized_first, sanitized_second)
    return GateResult(
        result.ok,
        'exact_seed101_non_target_scene_stable' if result.ok else result.reason,
        result.metrics,
    )


def extraction_progress_gate(
    start: EntityPose,
    previous: EntityPose,
    observed: EntityPose,
    step: int,
) -> GateResult:
    """Require the physical book to follow one certified recovery endpoint."""
    if not 1 <= int(step) < len(LOADED_RECOVERY_Q8):
        return GateResult(False, 'loaded_extraction_step_out_of_range', {})

    route = RECOVERY_ROUTE[int(step)]
    previous_route = RECOVERY_ROUTE[int(step) - 1]
    phase = str(route['phase'])
    lift = float(route['lift_world_z_m'])
    outward = float(route['outward_world_minus_x_m'])
    previous_lift = float(previous_route['lift_world_z_m'])
    previous_outward = float(previous_route['outward_world_minus_x_m'])
    expected = (
        start.position[0] - outward,
        start.position[1],
        start.position[2] + lift,
    )
    planned_delta = (
        -(outward - previous_outward),
        0.0,
        lift - previous_lift,
    )
    observed_delta = tuple(
        after - before
        for before, after in zip(previous.position, observed.position)
    )
    leg_translation_slip = _distance(observed_delta, planned_delta)
    leg_rotation_slip = quaternion_distance(
        previous.quaternion,
        observed.quaternion,
    )
    outward_progress = previous.position[0] - observed.position[0]
    lift_progress = observed.position[2] - previous.position[2]
    cross_track = (
        math.hypot(observed_delta[0], observed_delta[1])
        if phase == 'micro_lift'
        else math.hypot(observed_delta[1], observed_delta[2])
    )
    attachment_error = _distance(observed.position, expected)
    total_rotation_error = quaternion_distance(
        start.quaternion,
        observed.quaternion,
    )
    total_corner_displacement = (
        attachment_error
        + 2.0 * PAYLOAD_CONSERVATIVE_RADIUS_M
        * math.sin(0.5 * total_rotation_error)
    )
    maximum_world_x = book_maximum_world_x(observed)
    shelf_clearance = ROUTE_SHELF_FRONT_WORLD_X_M - maximum_world_x
    metrics = {
        'step': float(step),
        'expected_lift_m': lift,
        'expected_outward_m': outward,
        'lift_progress_m': lift_progress,
        'outward_progress_m': outward_progress,
        'leg_cross_track_error_m': cross_track,
        'leg_translation_slip_m': leg_translation_slip,
        'leg_rotation_slip_rad': leg_rotation_slip,
        'attachment_position_error_m': attachment_error,
        'attachment_rotation_error_rad': total_rotation_error,
        'attachment_corner_displacement_m': total_corner_displacement,
        'book_maximum_world_x_m': maximum_world_x,
        'shelf_face_clearance_m': shelf_clearance,
    }
    if phase not in ('micro_lift', 'outward'):
        return GateResult(False, 'loaded_extraction_phase_invalid', metrics)
    for accepted, reason in (
        (
            leg_translation_slip <= PAYLOAD_LEG_TRANSLATION_SLIP_LIMIT_M,
            f'book_did_not_follow_{phase}_step',
        ),
        (
            cross_track <= PAYLOAD_LEG_TRANSLATION_SLIP_LIMIT_M,
            f'book_{phase}_cross_track_motion_exceeded_limit',
        ),
        (
            leg_rotation_slip <= PAYLOAD_LEG_ROTATION_LIMIT_RAD,
            f'book_rotated_during_{phase}_step',
        ),
        (
            attachment_error <= PAYLOAD_ATTACHMENT_POSITION_LIMIT_M,
            'book_hand_attachment_translation_mismatch',
        ),
        (
            total_rotation_error <= PAYLOAD_TOTAL_ROTATION_LIMIT_RAD,
            'book_hand_attachment_rotation_mismatch',
        ),
        (
            total_corner_displacement
            <= PAYLOAD_TOTAL_CORNER_DISPLACEMENT_LIMIT_M,
            'book_hand_attachment_corner_displacement_mismatch',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    if (
        int(step) == len(LOADED_RECOVERY_Q8) - 1
        and shelf_clearance < FINAL_SHELF_CLEARANCE_REQUIRED_M
    ):
        return GateResult(False, 'book_not_shelf_clear_at_extraction_endpoint', metrics)
    return GateResult(True, 'loaded_extraction_step_verified', metrics)


def _world_rotation_vector(
    before: Sequence[float],
    after: Sequence[float],
) -> Tuple[float, float, float]:
    """Return the shortest before-to-after rotation vector in world axes."""

    bx, by, bz, bw = (float(value) for value in before)
    ax, ay, az, aw = (float(value) for value in after)
    before_norm = math.sqrt(bx * bx + by * by + bz * bz + bw * bw)
    after_norm = math.sqrt(ax * ax + ay * ay + az * az + aw * aw)
    if before_norm <= 0.0 or after_norm <= 0.0:
        return (math.inf, math.inf, math.inf)
    bx, by, bz, bw = (
        bx / before_norm,
        by / before_norm,
        bz / before_norm,
        bw / before_norm,
    )
    ax, ay, az, aw = (
        ax / after_norm,
        ay / after_norm,
        az / after_norm,
        aw / after_norm,
    )
    # q_delta = q_after * conjugate(q_before).  Its vector is therefore in
    # world coordinates rather than the book's rotating local frame.
    dx = -aw * bx + ax * bw - ay * bz + az * by
    dy = -aw * by + ax * bz + ay * bw - az * bx
    dz = -aw * bz - ax * by + ay * bx + az * bw
    dw = aw * bw + ax * bx + ay * by + az * bz
    if dw < 0.0:
        dx, dy, dz, dw = -dx, -dy, -dz, -dw
    vector_norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if vector_norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, dw))
    scale = angle / vector_norm
    return (dx * scale, dy * scale, dz * scale)


def book_minimum_world_z_at_x(
    pose: EntityPose,
    world_x_m: float,
) -> float:
    """Return the lowest point where the book OBB crosses a world-X plane."""

    qx, qy, qz, qw = (float(value) for value in pose.quaternion)
    magnitude = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if magnitude < 1e-9:
        raise ValueError('book quaternion magnitude is zero')
    qx, qy, qz, qw = (
        qx / magnitude,
        qy / magnitude,
        qz / magnitude,
        qw / magnitude,
    )
    rotation = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )
    corners = []
    for bits in range(8):
        local = tuple(
            half_extent if bits & (1 << axis) else -half_extent
            for axis, half_extent in enumerate(BOOK_HALF_EXTENTS_M)
        )
        corners.append(tuple(
            float(pose.position[row])
            + sum(rotation[row][axis] * local[axis] for axis in range(3))
            for row in range(3)
        ))
    intersections = []
    for first_index, first in enumerate(corners):
        for bit in (1, 2, 4):
            second_index = first_index ^ bit
            if second_index <= first_index:
                continue
            second = corners[second_index]
            denominator = second[0] - first[0]
            if abs(denominator) <= 1e-12:
                if abs(first[0] - float(world_x_m)) <= 1e-9:
                    intersections.extend((first[2], second[2]))
                continue
            fraction = (float(world_x_m) - first[0]) / denominator
            if -1e-9 <= fraction <= 1.0 + 1e-9:
                intersections.append(
                    first[2] + fraction * (second[2] - first[2])
                )
    if not intersections:
        return math.inf
    return min(float(value) for value in intersections)


def lip_tangent_progress_gate(
    start: EntityPose,
    previous: EntityPose,
    observed: EntityPose,
    *,
    support_floor_world_z_m: float,
    final: bool = False,
    minimum_nominal_improvement_rad: float = 0.010,
    minimum_z_improvement_m: float = 0.001,
    minimum_rigid_shelf_face_clearance_m: float = 0.0015,
    maximum_step_rotation_rad: float = 0.015,
    maximum_final_nominal_error_rad: float = math.inf,
    minimum_final_remaining_world_y_rad: float = -math.inf,
    maximum_final_remaining_world_y_rad: float = math.inf,
) -> GateResult:
    """Accept a bounded shelf-lip tangent only when it unloads the book."""

    start_nominal_error = quaternion_distance(
        start.quaternion,
        NOMINAL_BOOK_QUATERNION,
    )
    previous_nominal_error = quaternion_distance(
        previous.quaternion,
        NOMINAL_BOOK_QUATERNION,
    )
    observed_nominal_error = quaternion_distance(
        observed.quaternion,
        NOMINAL_BOOK_QUATERNION,
    )
    remaining_nominal_vector = _world_rotation_vector(
        observed.quaternion,
        NOMINAL_BOOK_QUATERNION,
    )
    remaining_world_y = remaining_nominal_vector[1]
    relative_vector = _world_rotation_vector(
        previous.quaternion,
        observed.quaternion,
    )
    relative_rotation = math.sqrt(sum(value * value for value in relative_vector))
    axis_y_fraction = (
        1.0
        if relative_rotation <= 0.001
        else abs(relative_vector[1]) / relative_rotation
    )
    minimum_z_improvement = (
        book_minimum_world_z(observed) - book_minimum_world_z(start)
    )
    shelf_face_bottom = book_minimum_world_z_at_x(
        observed,
        ROUTE_SHELF_FRONT_WORLD_X_M,
    )
    shelf_overlap = (
        book_maximum_world_x(observed) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    outward_motion = start.position[0] - observed.position[0]
    lateral_motion = abs(observed.position[1] - start.position[1])
    nominal_improvement = start_nominal_error - observed_nominal_error
    rotating_toward_nominal = (
        observed_nominal_error <= previous_nominal_error + 0.001
    )
    path_a = bool(
        nominal_improvement >= float(minimum_nominal_improvement_rad)
        and minimum_z_improvement >= float(minimum_z_improvement_m)
    )
    path_b = bool(
        shelf_face_bottom
        >= float(support_floor_world_z_m)
        + float(minimum_rigid_shelf_face_clearance_m)
    )
    metrics = {
        'start_nominal_rotation_error_rad': start_nominal_error,
        'observed_nominal_rotation_error_rad': observed_nominal_error,
        'remaining_nominal_world_y_rad': remaining_world_y,
        'nominal_rotation_improvement_rad': nominal_improvement,
        'relative_rotation_rad': relative_rotation,
        'relative_rotation_axis_y_fraction': axis_y_fraction,
        'minimum_z_improvement_m': minimum_z_improvement,
        'minimum_required_nominal_improvement_rad': float(
            minimum_nominal_improvement_rad
        ),
        'minimum_required_z_improvement_m': float(minimum_z_improvement_m),
        'shelf_face_bottom_world_z_m': shelf_face_bottom,
        'shelf_face_bottom_signed_m': (
            shelf_face_bottom - float(support_floor_world_z_m)
        ),
        'shelf_overlap_m': shelf_overlap,
        'outward_motion_m': outward_motion,
        'lateral_motion_m': lateral_motion,
        'final_orientation_and_height_path': float(path_a),
        'final_rigid_unload_path': float(path_b),
    }
    for accepted, reason in (
        (
            observed_nominal_error <= start_nominal_error + 0.003,
            'lip_tangent_rotated_away_from_nominal',
        ),
        (
            relative_rotation <= float(maximum_step_rotation_rad),
            'lip_tangent_step_rotation_exceeded_limit',
        ),
        (
            axis_y_fraction >= 0.90,
            'lip_tangent_rotation_axis_unexpected',
        ),
        (
            rotating_toward_nominal,
            'lip_tangent_step_rotated_away_from_nominal',
        ),
        (
            lateral_motion <= 0.002,
            'lip_tangent_lateral_motion_exceeded_limit',
        ),
        (
            outward_motion <= 0.00075,
            'lip_tangent_moved_book_outward',
        ),
        (
            shelf_overlap >= 0.065,
            'lip_tangent_lost_required_shelf_overlap',
        ),
        (
            shelf_face_bottom
            >= float(support_floor_world_z_m) - 0.00075,
            'lip_tangent_shelf_face_penetration',
        ),
        (
            not final or path_a or path_b,
            'lip_tangent_did_not_unload_shelf_lip',
        ),
        (
            not final
            or observed_nominal_error
            <= float(maximum_final_nominal_error_rad),
            'lip_tangent_final_nominal_error_exceeded_limit',
        ),
        (
            not final
            or remaining_world_y
            >= float(minimum_final_remaining_world_y_rad),
            'lip_tangent_crossed_past_upright',
        ),
        (
            not final
            or remaining_world_y
            <= float(maximum_final_remaining_world_y_rad),
            'lip_tangent_remaining_world_y_error_exceeded_limit',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'lip_tangent_progress_verified', metrics)


def diagonal_peel_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Gate one shelf-supported peel step from measured physical response."""

    planned = tuple(float(value) for value in planned_delta_world_m)
    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    translation_slip = _distance(observed_delta, planned)
    outward_progress = start.position[0] - observed.position[0]
    start_maximum_x = book_maximum_world_x(start)
    observed_maximum_x = book_maximum_world_x(observed)
    deepest_edge_regression = observed_maximum_x - start_maximum_x
    rotation_vector = _world_rotation_vector(
        start.quaternion,
        observed.quaternion,
    )
    rotation = math.sqrt(sum(value * value for value in rotation_vector))
    axis_y = 1.0 if rotation <= 1e-12 else rotation_vector[1] / rotation
    yaw_component = abs(rotation_vector[2])
    floor_signed = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    limits = {
        'min_book_center_outward_progress_m': 0.00050,
        'max_book_deepest_extent_increase_m': 0.00015,
        'max_book_from_hand_translation_change_m': 0.00050,
        'max_incremental_rotation_rad': 0.00200,
        'min_abs_rotation_axis_dot_world_y': 0.95,
        'max_yaw_component_rad': 0.00150,
        'min_floor_signed_distance_m': -0.00010,
        'max_floor_signed_distance_m': 0.00050,
    }
    minimum_deepest_outward = None
    require_axis_alignment = True
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })
        if 'min_book_deepest_extent_outward_progress_m' in policy:
            minimum_deepest_outward = float(
                policy['min_book_deepest_extent_outward_progress_m']
            )
        require_axis_alignment = (
            'min_abs_rotation_axis_dot_world_y' in policy
            and float(policy['min_abs_rotation_axis_dot_world_y']) > 0.0
        )
    metrics = {
        'planned_world_dx_m': planned[0],
        'planned_world_dy_m': planned[1],
        'planned_world_dz_m': planned[2],
        'observed_world_dx_m': observed_delta[0],
        'observed_world_dy_m': observed_delta[1],
        'observed_world_dz_m': observed_delta[2],
        'book_hand_translation_change_m': translation_slip,
        'book_center_outward_progress_m': outward_progress,
        'book_deepest_edge_regression_m': deepest_edge_regression,
        'book_incremental_rotation_rad': rotation,
        'book_incremental_rotation_axis_y': axis_y,
        'book_incremental_yaw_component_rad': yaw_component,
        'book_floor_signed_m': floor_signed,
        'book_maximum_world_x_m': observed_maximum_x,
    }
    for accepted, reason in (
        (
            outward_progress
            >= limits['min_book_center_outward_progress_m'],
            'diagonal_peel_lacked_outward_progress',
        ),
        (
            deepest_edge_regression
            <= limits['max_book_deepest_extent_increase_m'],
            'diagonal_peel_pushed_deep_edge_inward',
        ),
        (
            minimum_deepest_outward is None
            or -deepest_edge_regression >= minimum_deepest_outward,
            'shelf_slide_lacked_deep_edge_outward_progress',
        ),
        (
            translation_slip
            <= limits['max_book_from_hand_translation_change_m'],
            'diagonal_peel_book_hand_translation_mismatch',
        ),
        (
            rotation <= limits['max_incremental_rotation_rad'],
            'diagonal_peel_rotation_exceeded_limit',
        ),
        (
            not require_axis_alignment
            or rotation <= 1e-12
            or abs(axis_y)
            >= limits['min_abs_rotation_axis_dot_world_y'],
            'diagonal_peel_rotation_axis_unexpected',
        ),
        (
            yaw_component <= limits['max_yaw_component_rad'],
            'diagonal_peel_yaw_exceeded_limit',
        ),
        (
            floor_signed >= limits['min_floor_signed_distance_m'],
            'diagonal_peel_shelf_penetration_exceeded_limit',
        ),
        (
            floor_signed <= limits['max_floor_signed_distance_m'],
            'diagonal_peel_lost_shelf_support',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'diagonal_peel_step_verified', metrics)


def practical_full_pull_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    support_floor_world_z_m: float,
    final: bool = False,
    maximum_position_error_m: float = 0.015,
    maximum_lateral_error_m: float = 0.010,
    maximum_vertical_error_m: float = 0.012,
    maximum_rotation_rad: float = 0.12,
    minimum_floor_signed_m: float = -0.002,
    minimum_final_shelf_clearance_m: float = 0.010,
    minimum_observed_vertical_progress_m: float = -math.inf,
) -> GateResult:
    """Require useful full-pull progress while tolerating ordinary compliance."""

    planned = tuple(float(value) for value in planned_delta_world_m)
    if len(planned) != 3 or not all(math.isfinite(value) for value in planned):
        return GateResult(False, 'practical_pull_plan_invalid', {})
    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    planned_outward = -planned[0]
    observed_outward = -observed_delta[0]
    observed_vertical = observed_delta[2]
    position_error = _distance(observed_delta, planned)
    lateral_error = abs(observed_delta[1] - planned[1])
    vertical_error = abs(observed_delta[2] - planned[2])
    rotation = quaternion_distance(start.quaternion, observed.quaternion)
    floor_signed = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    shelf_clearance = ROUTE_SHELF_FRONT_WORLD_X_M - book_maximum_world_x(
        observed
    )
    progress_tolerance = 0.012
    metrics = {
        'planned_outward_m': planned_outward,
        'observed_outward_m': observed_outward,
        'observed_vertical_m': observed_vertical,
        'book_hand_position_error_m': position_error,
        'lateral_error_m': lateral_error,
        'vertical_error_m': vertical_error,
        'book_rotation_from_start_rad': rotation,
        'book_floor_signed_m': floor_signed,
        'book_shelf_clearance_m': shelf_clearance,
    }
    for accepted, reason in (
        (
            planned_outward >= 0.0,
            'practical_pull_direction_invalid',
        ),
        (
            observed_outward >= planned_outward - progress_tolerance,
            'practical_pull_lacked_outward_progress',
        ),
        (
            observed_outward <= planned_outward + progress_tolerance,
            'practical_pull_outward_overshoot',
        ),
        (
            position_error <= float(maximum_position_error_m),
            'practical_pull_book_not_following_hand',
        ),
        (
            lateral_error <= float(maximum_lateral_error_m),
            'practical_pull_lateral_drift',
        ),
        (
            vertical_error <= float(maximum_vertical_error_m),
            'practical_pull_vertical_drift',
        ),
        (
            rotation <= float(maximum_rotation_rad),
            'practical_pull_book_rotation_exceeded_limit',
        ),
        (
            not final
            or observed_vertical
            >= float(minimum_observed_vertical_progress_m),
            'practical_pull_lacked_vertical_progress',
        ),
        (
            floor_signed >= float(minimum_floor_signed_m),
            'practical_pull_shelf_penetration',
        ),
        (
            not final
            or shelf_clearance >= float(minimum_final_shelf_clearance_m),
            'practical_pull_book_not_clear_of_shelf',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_full_pull_progress_verified', metrics)


def practical_upright_extract_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Verify the final fixed-attitude lift-then-outward extraction pose."""

    planned = tuple(float(value) for value in planned_delta_world_m)
    if len(planned) != 3 or not all(math.isfinite(value) for value in planned):
        return GateResult(False, 'practical_upright_extract_plan_invalid', {})
    limits = {
        'minimum_planned_lift_m': 0.00075,
        'maximum_planned_lift_m': 0.00125,
        'minimum_planned_outward_m': 0.110,
        'maximum_planned_outward_m': 0.120,
        'maximum_planned_lateral_m': 1e-6,
        'maximum_position_error_m': 0.0010,
        'maximum_lateral_error_m': 0.00050,
        'maximum_vertical_error_m': 0.00075,
        'maximum_rotation_from_start_rad': 0.002,
        'minimum_final_floor_clearance_m': 0.00050,
        'minimum_final_shelf_clearance_m': 0.020,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })

    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    planned_outward = -planned[0]
    observed_outward = -observed_delta[0]
    planned_lift = planned[2]
    observed_lift = observed_delta[2]
    position_error = _distance(observed_delta, planned)
    lateral_error = abs(observed_delta[1] - planned[1])
    vertical_error = abs(observed_lift - planned_lift)
    rotation = quaternion_distance(start.quaternion, observed.quaternion)
    floor_clearance = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    shelf_clearance = (
        ROUTE_SHELF_FRONT_WORLD_X_M - book_maximum_world_x(observed)
    )
    metrics = {
        'planned_outward_m': planned_outward,
        'observed_outward_m': observed_outward,
        'planned_lift_m': planned_lift,
        'observed_lift_m': observed_lift,
        'book_hand_position_error_m': position_error,
        'lateral_error_m': lateral_error,
        'vertical_error_m': vertical_error,
        'book_rotation_from_start_rad': rotation,
        'book_floor_clearance_m': floor_clearance,
        'book_shelf_clearance_m': shelf_clearance,
    }
    checks = (
        (
            abs(planned[1]) <= limits['maximum_planned_lateral_m']
            and limits['minimum_planned_lift_m'] <= planned_lift
            <= limits['maximum_planned_lift_m']
            and limits['minimum_planned_outward_m'] <= planned_outward
            <= limits['maximum_planned_outward_m'],
            'practical_upright_extract_direction_invalid',
        ),
        (
            position_error <= limits['maximum_position_error_m'],
            'practical_upright_extract_book_not_following_hand',
        ),
        (
            lateral_error <= limits['maximum_lateral_error_m'],
            'practical_upright_extract_lateral_drift',
        ),
        (
            vertical_error <= limits['maximum_vertical_error_m'],
            'practical_upright_extract_lift_error',
        ),
        (
            rotation <= limits['maximum_rotation_from_start_rad'],
            'practical_upright_extract_rotation_drift',
        ),
        (
            floor_clearance >= limits['minimum_final_floor_clearance_m'],
            'practical_upright_extract_final_height_insufficient',
        ),
        (
            shelf_clearance >= limits['minimum_final_shelf_clearance_m'],
            'practical_upright_extract_book_not_clear_of_shelf',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_upright_extract_progress_verified', metrics)


def practical_outward_slide_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Verify that the upright book slid outward instead of pivoting on the lip."""

    planned = tuple(float(value) for value in planned_delta_world_m)
    if len(planned) != 3 or not all(math.isfinite(value) for value in planned):
        return GateResult(False, 'practical_outward_slide_plan_invalid', {})
    limits = {
        'minimum_center_outward_progress_m': 0.0025,
        'maximum_center_outward_progress_m': 0.0055,
        'minimum_maximum_x_outward_progress_m': 0.0020,
        'minimum_shelf_overlap_m': 0.065,
        'maximum_rotation_from_start_rad': 0.008,
        'maximum_nominal_rotation_growth_rad': 0.004,
        'minimum_floor_signed_m': -0.0005,
        'minimum_shelf_face_bottom_signed_m': -0.00025,
        'minimum_center_vertical_progress_m': -0.00025,
        'maximum_center_vertical_progress_m': 0.0015,
        'maximum_lateral_motion_m': 0.001,
        'maximum_book_from_hand_translation_error_m': 0.0025,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })

    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    center_outward = -observed_delta[0]
    maximum_x_outward = (
        book_maximum_world_x(start) - book_maximum_world_x(observed)
    )
    shelf_overlap = (
        book_maximum_world_x(observed) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    rotation = quaternion_distance(start.quaternion, observed.quaternion)
    start_nominal = quaternion_distance(
        start.quaternion,
        NOMINAL_BOOK_QUATERNION,
    )
    observed_nominal = quaternion_distance(
        observed.quaternion,
        NOMINAL_BOOK_QUATERNION,
    )
    nominal_growth = observed_nominal - start_nominal
    floor_signed = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    shelf_face_bottom_signed = (
        book_minimum_world_z_at_x(observed, ROUTE_SHELF_FRONT_WORLD_X_M)
        - float(support_floor_world_z_m)
    )
    vertical_progress = observed_delta[2]
    lateral_motion = abs(observed_delta[1])
    translation_error = _distance(observed_delta, planned)
    metrics = {
        'planned_world_dx_m': planned[0],
        'planned_world_dy_m': planned[1],
        'planned_world_dz_m': planned[2],
        'observed_world_dx_m': observed_delta[0],
        'observed_world_dy_m': observed_delta[1],
        'observed_world_dz_m': observed_delta[2],
        'book_center_outward_progress_m': center_outward,
        'book_maximum_x_outward_progress_m': maximum_x_outward,
        'book_shelf_overlap_m': shelf_overlap,
        'book_rotation_from_start_rad': rotation,
        'book_start_nominal_error_rad': start_nominal,
        'book_observed_nominal_error_rad': observed_nominal,
        'book_nominal_rotation_growth_rad': nominal_growth,
        'book_floor_signed_m': floor_signed,
        'book_shelf_face_bottom_signed_m': shelf_face_bottom_signed,
        'book_center_vertical_progress_m': vertical_progress,
        'book_lateral_motion_m': lateral_motion,
        'book_from_hand_translation_error_m': translation_error,
    }
    for accepted, reason in (
        (
            planned[0] < 0.0
            and abs(planned[1]) <= 1e-12
            and planned[2] >= 0.0
            and abs(planned[2] + 0.125 * planned[0]) <= 1e-12,
            'practical_outward_slide_direction_invalid',
        ),
        (
            center_outward >= limits['minimum_center_outward_progress_m'],
            'practical_outward_slide_lacked_center_progress',
        ),
        (
            center_outward <= limits['maximum_center_outward_progress_m'],
            'practical_outward_slide_center_overshoot',
        ),
        (
            maximum_x_outward
            >= limits['minimum_maximum_x_outward_progress_m'],
            'practical_outward_slide_lip_pivot_detected',
        ),
        (
            shelf_overlap >= limits['minimum_shelf_overlap_m'],
            'practical_outward_slide_lost_required_shelf_overlap',
        ),
        (
            rotation <= limits['maximum_rotation_from_start_rad'],
            'practical_outward_slide_rotation_exceeded_limit',
        ),
        (
            nominal_growth <= limits['maximum_nominal_rotation_growth_rad'],
            'practical_outward_slide_rotated_away_from_nominal',
        ),
        (
            floor_signed >= limits['minimum_floor_signed_m'],
            'practical_outward_slide_shelf_penetration',
        ),
        (
            shelf_face_bottom_signed
            >= limits['minimum_shelf_face_bottom_signed_m'],
            'practical_outward_slide_shelf_face_penetration',
        ),
        (
            vertical_progress
            >= limits['minimum_center_vertical_progress_m'],
            'practical_outward_slide_dropped',
        ),
        (
            vertical_progress
            <= limits['maximum_center_vertical_progress_m'],
            'practical_outward_slide_rose_unexpectedly',
        ),
        (
            lateral_motion <= limits['maximum_lateral_motion_m'],
            'practical_outward_slide_lateral_drift',
        ),
        (
            translation_error
            <= limits['maximum_book_from_hand_translation_error_m'],
            'practical_outward_slide_book_not_following_hand',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_outward_slide_progress_verified', metrics)


def practical_radial_unload_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    radial_unit_world: Sequence[float],
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Require support clearance from a torque-neutral shelf-lip unload."""

    planned = tuple(float(value) for value in planned_delta_world_m)
    radial = tuple(float(value) for value in radial_unit_world)
    if (
        len(planned) != 3
        or len(radial) != 3
        or not all(math.isfinite(value) for value in (*planned, *radial))
    ):
        return GateResult(False, 'practical_radial_unload_plan_invalid', {})
    radial_norm = math.sqrt(sum(value * value for value in radial))
    if radial_norm <= 1e-12:
        return GateResult(False, 'practical_radial_unload_axis_invalid', {})
    unit = tuple(value / radial_norm for value in radial)
    planned_distance = sum(value * axis for value, axis in zip(planned, unit))
    planned_cross_track = math.sqrt(sum(
        (value - planned_distance * axis) ** 2
        for value, axis in zip(planned, unit)
    ))
    limits = {
        'minimum_projected_progress_m': 0.001,
        'maximum_projected_progress_m': 0.0035,
        'maximum_cross_track_motion_m': 0.00075,
        'maximum_rotation_from_start_rad': 0.004,
        'maximum_nominal_rotation_growth_rad': 0.003,
        'minimum_floor_signed_m': 0.0008,
        'minimum_shelf_face_bottom_signed_m': 0.0005,
        'minimum_shelf_overlap_m': 0.065,
        'maximum_lateral_motion_m': 0.001,
        'maximum_book_from_hand_translation_error_m': 0.00075,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })

    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    projected_progress = sum(
        value * axis for value, axis in zip(observed_delta, unit)
    )
    cross_track = math.sqrt(sum(
        (value - projected_progress * axis) ** 2
        for value, axis in zip(observed_delta, unit)
    ))
    rotation = quaternion_distance(start.quaternion, observed.quaternion)
    start_nominal = quaternion_distance(
        start.quaternion,
        NOMINAL_BOOK_QUATERNION,
    )
    observed_nominal = quaternion_distance(
        observed.quaternion,
        NOMINAL_BOOK_QUATERNION,
    )
    nominal_growth = observed_nominal - start_nominal
    floor_signed = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    shelf_face_bottom_signed = (
        book_minimum_world_z_at_x(observed, ROUTE_SHELF_FRONT_WORLD_X_M)
        - float(support_floor_world_z_m)
    )
    shelf_overlap = (
        book_maximum_world_x(observed) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    lateral_motion = abs(observed_delta[1])
    translation_error = _distance(observed_delta, planned)
    metrics = {
        'planned_radial_distance_m': planned_distance,
        'planned_radial_cross_track_m': planned_cross_track,
        'radial_unit_world_x': unit[0],
        'radial_unit_world_y': unit[1],
        'radial_unit_world_z': unit[2],
        'observed_world_dx_m': observed_delta[0],
        'observed_world_dy_m': observed_delta[1],
        'observed_world_dz_m': observed_delta[2],
        'book_radial_projected_progress_m': projected_progress,
        'book_radial_cross_track_m': cross_track,
        'book_rotation_from_start_rad': rotation,
        'book_start_nominal_error_rad': start_nominal,
        'book_observed_nominal_error_rad': observed_nominal,
        'book_nominal_rotation_growth_rad': nominal_growth,
        'book_floor_signed_m': floor_signed,
        'book_shelf_face_bottom_signed_m': shelf_face_bottom_signed,
        'book_shelf_overlap_m': shelf_overlap,
        'book_lateral_motion_m': lateral_motion,
        'book_from_hand_translation_error_m': translation_error,
    }
    for accepted, reason in (
        (
            planned_distance > 0.0 and planned_cross_track <= 1e-9,
            'practical_radial_unload_direction_invalid',
        ),
        (
            projected_progress >= limits['minimum_projected_progress_m'],
            'practical_radial_unload_lacked_progress',
        ),
        (
            projected_progress <= limits['maximum_projected_progress_m'],
            'practical_radial_unload_progress_overshoot',
        ),
        (
            cross_track <= limits['maximum_cross_track_motion_m'],
            'practical_radial_unload_cross_track_motion',
        ),
        (
            rotation <= limits['maximum_rotation_from_start_rad'],
            'practical_radial_unload_rotation_exceeded_limit',
        ),
        (
            nominal_growth <= limits['maximum_nominal_rotation_growth_rad'],
            'practical_radial_unload_rotated_away_from_nominal',
        ),
        (
            floor_signed >= limits['minimum_floor_signed_m'],
            'practical_radial_unload_failed_to_lift_bottom',
        ),
        (
            shelf_face_bottom_signed
            >= limits['minimum_shelf_face_bottom_signed_m'],
            'practical_radial_unload_failed_to_clear_lip',
        ),
        (
            shelf_overlap >= limits['minimum_shelf_overlap_m'],
            'practical_radial_unload_lost_required_shelf_overlap',
        ),
        (
            lateral_motion <= limits['maximum_lateral_motion_m'],
            'practical_radial_unload_lateral_drift',
        ),
        (
            translation_error
            <= limits['maximum_book_from_hand_translation_error_m'],
            'practical_radial_unload_book_not_following_hand',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_radial_unload_progress_verified', metrics)


def practical_current_ray_release_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    radial_unit_world: Sequence[float],
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Verify one bounded rigid release along the current lip-to-grasp ray.

    The book remains rolled, so its outer overhang is intentionally below the
    infinite shelf-floor plane.  The relevant support checks are the section
    at the shelf face and the deepest in-bay lower edge; global minimum Z is a
    diagnostic only.
    """

    try:
        planned = tuple(float(value) for value in planned_delta_world_m)
        radial = tuple(float(value) for value in radial_unit_world)
        floor = float(support_floor_world_z_m)
    except (TypeError, ValueError):
        planned = ()
        radial = ()
        floor = math.nan
    if (
        len(planned) != 3
        or len(radial) != 3
        or not all(math.isfinite(value) for value in (*planned, *radial, floor))
    ):
        return GateResult(False, 'practical_current_ray_plan_invalid', {})
    radial_norm = math.sqrt(sum(value * value for value in radial))
    if radial_norm <= 1e-12:
        return GateResult(False, 'practical_current_ray_axis_invalid', {})
    unit = tuple(value / radial_norm for value in radial)
    planned_progress = sum(
        value * axis for value, axis in zip(planned, unit)
    )
    planned_tangent = math.sqrt(sum(
        (value - planned_progress * axis) ** 2
        for value, axis in zip(planned, unit)
    ))
    if (
        planned_progress <= 0.0
        or planned_progress > 0.00101
        or planned_tangent > 1e-6
    ):
        return GateResult(False, 'practical_current_ray_direction_invalid', {
            'planned_radial_progress_m': planned_progress,
            'planned_tangent_m': planned_tangent,
        })

    limits = {
        'minimum_projected_progress_m': 0.00065,
        'maximum_projected_progress_m': 0.00135,
        'maximum_tangent_motion_m': 0.00025,
        'maximum_translation_error_m': 0.00045,
        'maximum_rotation_from_start_rad': 0.002,
        'maximum_corner_mismatch_m': 0.001,
        'maximum_nominal_rotation_growth_rad': 0.002,
        'minimum_shelf_face_bottom_signed_m': 0.0005,
        'minimum_deep_edge_signed_m': 0.005,
        'minimum_shelf_overlap_m': 0.065,
        'maximum_lateral_motion_m': 0.0005,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })

    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    projected_progress = sum(
        value * axis for value, axis in zip(observed_delta, unit)
    )
    tangent_motion = math.sqrt(sum(
        (value - projected_progress * axis) ** 2
        for value, axis in zip(observed_delta, unit)
    ))
    translation_error = _distance(observed_delta, planned)
    rotation = quaternion_distance(start.quaternion, observed.quaternion)
    corner_mismatch = translation_error + 0.15 * rotation
    nominal_growth = (
        quaternion_distance(observed.quaternion, NOMINAL_BOOK_QUATERNION)
        - quaternion_distance(start.quaternion, NOMINAL_BOOK_QUATERNION)
    )
    shelf_face_bottom_signed = (
        book_minimum_world_z_at_x(observed, ROUTE_SHELF_FRONT_WORLD_X_M)
        - floor
    )
    deep_edge_signed = book_deepest_extent_lower_world_z(observed) - floor
    shelf_overlap = (
        book_maximum_world_x(observed) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    lateral_motion = abs(observed_delta[1])
    metrics = {
        'planned_radial_progress_m': planned_progress,
        'planned_tangent_m': planned_tangent,
        'radial_unit_world_x': unit[0],
        'radial_unit_world_y': unit[1],
        'radial_unit_world_z': unit[2],
        'observed_world_dx_m': observed_delta[0],
        'observed_world_dy_m': observed_delta[1],
        'observed_world_dz_m': observed_delta[2],
        'book_radial_projected_progress_m': projected_progress,
        'book_radial_tangent_motion_m': tangent_motion,
        'book_translation_error_m': translation_error,
        'book_rotation_from_start_rad': rotation,
        'book_corner_mismatch_m': corner_mismatch,
        'book_nominal_rotation_growth_rad': nominal_growth,
        'book_shelf_face_bottom_signed_m': shelf_face_bottom_signed,
        'book_deep_edge_signed_m': deep_edge_signed,
        'book_global_minimum_z_m': book_minimum_world_z(observed),
        'book_shelf_overlap_m': shelf_overlap,
        'book_lateral_motion_m': lateral_motion,
    }
    for accepted, reason in (
        (
            projected_progress >= limits['minimum_projected_progress_m'],
            'practical_current_ray_lacked_progress',
        ),
        (
            projected_progress <= limits['maximum_projected_progress_m'],
            'practical_current_ray_progress_overshoot',
        ),
        (
            tangent_motion <= limits['maximum_tangent_motion_m'],
            'practical_current_ray_tangent_motion',
        ),
        (
            translation_error <= limits['maximum_translation_error_m'],
            'practical_current_ray_book_did_not_follow_hand',
        ),
        (
            rotation <= limits['maximum_rotation_from_start_rad'],
            'practical_current_ray_rotation_exceeded_limit',
        ),
        (
            corner_mismatch <= limits['maximum_corner_mismatch_m'],
            'practical_current_ray_corner_mismatch',
        ),
        (
            nominal_growth <= limits['maximum_nominal_rotation_growth_rad'],
            'practical_current_ray_rotated_away_from_nominal',
        ),
        (
            shelf_face_bottom_signed
            >= limits['minimum_shelf_face_bottom_signed_m'],
            'practical_current_ray_failed_to_clear_shelf_face',
        ),
        (
            deep_edge_signed >= limits['minimum_deep_edge_signed_m'],
            'practical_current_ray_deep_edge_not_clear',
        ),
        (
            shelf_overlap >= limits['minimum_shelf_overlap_m'],
            'practical_current_ray_lost_required_shelf_overlap',
        ),
        (
            lateral_motion <= limits['maximum_lateral_motion_m'],
            'practical_current_ray_lateral_drift',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_current_ray_release_verified', metrics)


def book_deepest_extent_lower_world_z(pose: EntityPose) -> float:
    """Return the lower Z coordinate at the book's deepest (+world-X) corner."""

    qx, qy, qz, qw = (float(value) for value in pose.quaternion)
    magnitude = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if magnitude < 1e-9:
        raise ValueError('book quaternion magnitude is zero')
    qx, qy, qz, qw = (
        qx / magnitude,
        qy / magnitude,
        qz / magnitude,
        qw / magnitude,
    )
    rotation = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )
    corners = []
    for bits in range(8):
        local = tuple(
            half_extent if bits & (1 << axis) else -half_extent
            for axis, half_extent in enumerate(BOOK_HALF_EXTENTS_M)
        )
        corners.append(tuple(
            float(pose.position[row])
            + sum(rotation[row][axis] * local[axis] for axis in range(3))
            for row in range(3)
        ))
    maximum_x = max(corner[0] for corner in corners)
    return min(
        corner[2]
        for corner in corners
        if maximum_x - corner[0] <= 1e-9
    )


def practical_lip_roll_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    expected_position_world_m: Sequence[float],
    expected_quaternion_xyzw: Sequence[float],
    expected_signed_world_y_rotation_rad: float,
    support_floor_world_z_m: float,
    minimum_deep_edge_signed_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Require a coordinated outward roll about the shelf-front lip.

    During this supported maneuver the outer overhang intentionally descends
    below the shelf-floor plane.  Safety is therefore bound to the section at
    the shelf face and to the rising deepest in-bay edge, never to global
    minimum Z.
    """

    try:
        expected_position = tuple(
            float(value) for value in expected_position_world_m
        )
        expected_quaternion = tuple(
            float(value) for value in expected_quaternion_xyzw
        )
        expected_signed_y = float(expected_signed_world_y_rotation_rad)
        floor = float(support_floor_world_z_m)
        minimum_deep = float(minimum_deep_edge_signed_m)
    except (TypeError, ValueError):
        expected_position = ()
        expected_quaternion = ()
        expected_signed_y = math.nan
        floor = math.nan
        minimum_deep = math.nan
    if (
        len(expected_position) != 3
        or len(expected_quaternion) != 4
        or not all(math.isfinite(value) for value in (
            *expected_position,
            *expected_quaternion,
            expected_signed_y,
            floor,
            minimum_deep,
        ))
        or expected_signed_y >= 0.0
    ):
        return GateResult(False, 'practical_lip_roll_plan_invalid', {})

    limits = {
        'maximum_center_arc_error_m': 0.002,
        'maximum_orientation_error_rad': 0.006,
        'maximum_signed_y_rotation_error_rad': 0.006,
        'maximum_cross_axis_rotation_rad': 0.003,
        'minimum_shelf_face_bottom_signed_m': -0.0005,
        'maximum_shelf_face_bottom_signed_m': 0.00075,
        'minimum_shelf_overlap_m': 0.068,
        'maximum_shelf_overlap_m': 0.072,
        'maximum_lateral_motion_m': 0.0005,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })

    rotation_vector = _world_rotation_vector(
        start.quaternion,
        observed.quaternion,
    )
    signed_y_error = abs(rotation_vector[1] - expected_signed_y)
    cross_axis_rotation = math.hypot(
        rotation_vector[0],
        rotation_vector[2],
    )
    center_arc_error = _distance(
        observed.position,
        expected_position,
    )
    orientation_error = quaternion_distance(
        observed.quaternion,
        expected_quaternion,
    )
    shelf_face_bottom_signed = (
        book_minimum_world_z_at_x(observed, ROUTE_SHELF_FRONT_WORLD_X_M)
        - floor
    )
    deep_edge_signed = book_deepest_extent_lower_world_z(observed) - floor
    shelf_overlap = (
        book_maximum_world_x(observed) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    lateral_motion = abs(observed.position[1] - start.position[1])
    metrics = {
        'expected_signed_world_y_rotation_rad': expected_signed_y,
        'observed_world_rotation_x_rad': rotation_vector[0],
        'observed_world_rotation_y_rad': rotation_vector[1],
        'observed_world_rotation_z_rad': rotation_vector[2],
        'signed_world_y_rotation_error_rad': signed_y_error,
        'cross_axis_rotation_rad': cross_axis_rotation,
        'book_center_arc_error_m': center_arc_error,
        'book_orientation_error_from_arc_target_rad': orientation_error,
        'book_shelf_face_bottom_signed_m': shelf_face_bottom_signed,
        'book_deep_edge_signed_m': deep_edge_signed,
        'book_global_minimum_z_m': book_minimum_world_z(observed),
        'book_shelf_overlap_m': shelf_overlap,
        'book_lateral_motion_m': lateral_motion,
    }
    for accepted, reason in (
        (
            center_arc_error <= limits['maximum_center_arc_error_m'],
            'practical_lip_roll_center_left_arc',
        ),
        (
            orientation_error <= limits['maximum_orientation_error_rad'],
            'practical_lip_roll_attitude_mismatch',
        ),
        (
            signed_y_error
            <= limits['maximum_signed_y_rotation_error_rad'],
            'practical_lip_roll_wrong_signed_rotation',
        ),
        (
            cross_axis_rotation
            <= limits['maximum_cross_axis_rotation_rad'],
            'practical_lip_roll_cross_axis_rotation',
        ),
        (
            shelf_face_bottom_signed
            >= limits['minimum_shelf_face_bottom_signed_m'],
            'practical_lip_roll_penetrated_shelf_face',
        ),
        (
            shelf_face_bottom_signed
            <= limits['maximum_shelf_face_bottom_signed_m'],
            'practical_lip_roll_lost_lip_support',
        ),
        (
            deep_edge_signed >= minimum_deep,
            'practical_lip_roll_deep_edge_did_not_rise',
        ),
        (
            shelf_overlap >= limits['minimum_shelf_overlap_m'],
            'practical_lip_roll_lost_required_shelf_overlap',
        ),
        (
            shelf_overlap <= limits['maximum_shelf_overlap_m'],
            'practical_lip_roll_moved_deeper_into_shelf',
        ),
        (
            lateral_motion <= limits['maximum_lateral_motion_m'],
            'practical_lip_roll_lateral_drift',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_lip_roll_progress_verified', metrics)


def practical_reseat_support_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    upright_position_world_m: Sequence[float],
    expected_position_world_m: Sequence[float],
    expected_quaternion_xyzw: Sequence[float],
    expected_signed_world_y_rotation_rad: float,
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Certify the reverse lip roll followed by a shelf-supported inward reseat.

    The overhanging edge is intentionally below the shelf plane during the roll,
    so this endpoint gate uses the final global bottom only after the continuous
    face-section watchdog has carried the book back upright.
    """

    try:
        upright_position = tuple(
            float(value) for value in upright_position_world_m
        )
        expected_position = tuple(
            float(value) for value in expected_position_world_m
        )
        expected_quaternion = tuple(
            float(value) for value in expected_quaternion_xyzw
        )
        expected_signed_y = float(expected_signed_world_y_rotation_rad)
        floor = float(support_floor_world_z_m)
    except (TypeError, ValueError):
        upright_position = ()
        expected_position = ()
        expected_quaternion = ()
        expected_signed_y = math.nan
        floor = math.nan
    if (
        len(upright_position) != 3
        or len(expected_position) != 3
        or len(expected_quaternion) != 4
        or not all(math.isfinite(value) for value in (
            *upright_position,
            *expected_position,
            *expected_quaternion,
            expected_signed_y,
            floor,
        ))
        or expected_signed_y <= 0.0
    ):
        return GateResult(False, 'practical_reseat_support_plan_invalid', {})

    limits = {
        'maximum_center_error_m': 0.0015,
        'maximum_orientation_error_rad': 0.006,
        'maximum_signed_y_rotation_error_rad': 0.006,
        'maximum_cross_axis_rotation_growth_rad': 0.002,
        'minimum_global_bottom_signed_m': -0.0005,
        'maximum_global_bottom_signed_m': 0.00075,
        'minimum_inward_from_upright_m': 0.018,
        'maximum_inward_from_upright_m': 0.022,
        'minimum_center_depth_from_shelf_face_m': 0.008,
        'minimum_shelf_overlap_m': 0.088,
        'maximum_shelf_overlap_m': 0.095,
        'maximum_lateral_motion_m': 0.00030,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })

    rotation_vector = _world_rotation_vector(
        start.quaternion,
        observed.quaternion,
    )
    signed_y_error = abs(rotation_vector[1] - expected_signed_y)
    cross_axis_growth = math.hypot(rotation_vector[0], rotation_vector[2])
    center_error = _distance(observed.position, expected_position)
    orientation_error = quaternion_distance(
        observed.quaternion,
        expected_quaternion,
    )
    global_bottom_signed = book_minimum_world_z(observed) - floor
    inward_from_upright = observed.position[0] - upright_position[0]
    center_depth = observed.position[0] - ROUTE_SHELF_FRONT_WORLD_X_M
    shelf_overlap = (
        book_maximum_world_x(observed) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    lateral_motion = abs(observed.position[1] - start.position[1])
    metrics = {
        'expected_signed_world_y_rotation_rad': expected_signed_y,
        'observed_world_rotation_x_rad': rotation_vector[0],
        'observed_world_rotation_y_rad': rotation_vector[1],
        'observed_world_rotation_z_rad': rotation_vector[2],
        'signed_world_y_rotation_error_rad': signed_y_error,
        'cross_axis_rotation_growth_rad': cross_axis_growth,
        'book_center_endpoint_error_m': center_error,
        'book_orientation_endpoint_error_rad': orientation_error,
        'book_global_bottom_signed_m': global_bottom_signed,
        'book_center_inward_from_upright_m': inward_from_upright,
        'book_center_depth_from_shelf_face_m': center_depth,
        'book_shelf_overlap_m': shelf_overlap,
        'book_lateral_motion_m': lateral_motion,
    }
    for accepted, reason in (
        (
            center_error <= limits['maximum_center_error_m'],
            'practical_reseat_center_endpoint_mismatch',
        ),
        (
            orientation_error <= limits['maximum_orientation_error_rad'],
            'practical_reseat_orientation_endpoint_mismatch',
        ),
        (
            signed_y_error
            <= limits['maximum_signed_y_rotation_error_rad'],
            'practical_reseat_reverse_roll_incomplete',
        ),
        (
            cross_axis_growth
            <= limits['maximum_cross_axis_rotation_growth_rad'],
            'practical_reseat_cross_axis_rotation_growth',
        ),
        (
            global_bottom_signed
            >= limits['minimum_global_bottom_signed_m'],
            'practical_reseat_shelf_penetration',
        ),
        (
            global_bottom_signed
            <= limits['maximum_global_bottom_signed_m'],
            'practical_reseat_not_shelf_supported',
        ),
        (
            inward_from_upright
            >= limits['minimum_inward_from_upright_m'],
            'practical_reseat_insufficient_inward_progress',
        ),
        (
            inward_from_upright
            <= limits['maximum_inward_from_upright_m'],
            'practical_reseat_inward_overshoot',
        ),
        (
            center_depth
            >= limits['minimum_center_depth_from_shelf_face_m'],
            'practical_reseat_com_support_depth_insufficient',
        ),
        (
            shelf_overlap >= limits['minimum_shelf_overlap_m'],
            'practical_reseat_shelf_overlap_insufficient',
        ),
        (
            shelf_overlap <= limits['maximum_shelf_overlap_m'],
            'practical_reseat_shelf_overlap_overshoot',
        ),
        (
            lateral_motion <= limits['maximum_lateral_motion_m'],
            'practical_reseat_lateral_drift',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_reseat_support_verified', metrics)


def practical_fixed_attitude_detach_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Require the rolled book to translate clear of the lip without pivoting.

    The outer overhang is still below the infinite shelf-floor plane, but it is
    outside the shelf face.  Therefore this gate checks the face section and
    deepest in-bay edge rather than rejecting the safe global minimum Z.
    """

    try:
        planned = tuple(float(value) for value in planned_delta_world_m)
        floor = float(support_floor_world_z_m)
    except (TypeError, ValueError):
        planned = ()
        floor = math.nan
    valid_plans = (
        (-0.0025, 0.0, 0.0015),
        (-0.005, 0.0, 0.003),
    )
    if (
        len(planned) != 3
        or not all(math.isfinite(value) for value in (*planned, floor))
        or not any(
            all(abs(value - expected) <= 1e-12 for value, expected in zip(
                planned,
                candidate,
            ))
            for candidate in valid_plans
        )
    ):
        return GateResult(False, 'practical_detach_plan_invalid', {})

    limits = {
        'minimum_projected_progress_m': 0.0045,
        'maximum_projected_progress_m': 0.0075,
        'maximum_cross_track_motion_m': 0.001,
        'maximum_translation_error_m': 0.001,
        'maximum_rotation_from_start_rad': 0.004,
        'maximum_yaw_from_start_rad': 0.003,
        'minimum_outward_progress_m': 0.0035,
        'minimum_vertical_progress_m': 0.0015,
        'maximum_vertical_progress_m': 0.0045,
        'minimum_shelf_face_bottom_signed_m': 0.0015,
        'maximum_shelf_face_bottom_signed_m': 0.005,
        'minimum_deep_edge_signed_m': 0.0075,
        'minimum_overlap_reduction_m': 0.0035,
        'minimum_shelf_overlap_m': 0.062,
        'maximum_shelf_overlap_m': 0.0685,
        'maximum_lateral_motion_m': 0.0005,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })

    planned_distance = math.sqrt(sum(value * value for value in planned))
    if planned_distance <= 1e-12:
        return GateResult(False, 'practical_detach_axis_invalid', {})
    direction = tuple(value / planned_distance for value in planned)
    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    projected_progress = sum(
        value * axis for value, axis in zip(observed_delta, direction)
    )
    cross_track = math.sqrt(sum(
        (value - projected_progress * axis) ** 2
        for value, axis in zip(observed_delta, direction)
    ))
    translation_error = _distance(observed_delta, planned)
    rotation_vector = _world_rotation_vector(
        start.quaternion,
        observed.quaternion,
    )
    rotation = math.sqrt(sum(value * value for value in rotation_vector))
    yaw = abs(rotation_vector[2])
    outward_progress = -observed_delta[0]
    vertical_progress = observed_delta[2]
    shelf_face_bottom_signed = (
        book_minimum_world_z_at_x(observed, ROUTE_SHELF_FRONT_WORLD_X_M)
        - floor
    )
    deep_edge_signed = book_deepest_extent_lower_world_z(observed) - floor
    start_overlap = (
        book_maximum_world_x(start) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    observed_overlap = (
        book_maximum_world_x(observed) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    overlap_reduction = start_overlap - observed_overlap
    lateral_motion = abs(observed_delta[1])
    metrics = {
        'planned_detach_distance_m': planned_distance,
        'book_detach_projected_progress_m': projected_progress,
        'book_detach_cross_track_m': cross_track,
        'book_detach_translation_error_m': translation_error,
        'book_detach_rotation_rad': rotation,
        'book_detach_yaw_rad': yaw,
        'book_outward_progress_m': outward_progress,
        'book_vertical_progress_m': vertical_progress,
        'book_shelf_face_bottom_signed_m': shelf_face_bottom_signed,
        'book_deep_edge_signed_m': deep_edge_signed,
        'book_global_minimum_z_m': book_minimum_world_z(observed),
        'book_start_shelf_overlap_m': start_overlap,
        'book_observed_shelf_overlap_m': observed_overlap,
        'book_shelf_overlap_reduction_m': overlap_reduction,
        'book_lateral_motion_m': lateral_motion,
    }
    for accepted, reason in (
        (
            projected_progress >= limits['minimum_projected_progress_m'],
            'practical_detach_lacked_progress',
        ),
        (
            projected_progress <= limits['maximum_projected_progress_m'],
            'practical_detach_progress_overshoot',
        ),
        (
            cross_track <= limits['maximum_cross_track_motion_m'],
            'practical_detach_cross_track_motion',
        ),
        (
            translation_error <= limits['maximum_translation_error_m'],
            'practical_detach_book_did_not_follow_hand',
        ),
        (
            rotation <= limits['maximum_rotation_from_start_rad'],
            'practical_detach_attitude_changed',
        ),
        (
            yaw <= limits['maximum_yaw_from_start_rad'],
            'practical_detach_yaw_changed',
        ),
        (
            outward_progress >= limits['minimum_outward_progress_m'],
            'practical_detach_lacked_outward_progress',
        ),
        (
            vertical_progress >= limits['minimum_vertical_progress_m'],
            'practical_detach_lacked_vertical_progress',
        ),
        (
            vertical_progress <= limits['maximum_vertical_progress_m'],
            'practical_detach_rose_unexpectedly',
        ),
        (
            shelf_face_bottom_signed
            >= limits['minimum_shelf_face_bottom_signed_m'],
            'practical_detach_failed_to_clear_shelf_face',
        ),
        (
            shelf_face_bottom_signed
            <= limits['maximum_shelf_face_bottom_signed_m'],
            'practical_detach_shelf_face_height_overshoot',
        ),
        (
            deep_edge_signed >= limits['minimum_deep_edge_signed_m'],
            'practical_detach_deep_edge_not_clear',
        ),
        (
            overlap_reduction >= limits['minimum_overlap_reduction_m'],
            'practical_detach_overlap_did_not_decrease',
        ),
        (
            observed_overlap >= limits['minimum_shelf_overlap_m'],
            'practical_detach_lost_bounded_shelf_overlap',
        ),
        (
            observed_overlap <= limits['maximum_shelf_overlap_m'],
            'practical_detach_retained_too_much_overlap',
        ),
        (
            lateral_motion <= limits['maximum_lateral_motion_m'],
            'practical_detach_lateral_drift',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_fixed_attitude_detach_verified', metrics)


def inward_reseat_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    support_floor_world_z_m: float,
    stable_reference: EntityPose,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Gate a shelf-supported inward return to a certified stable pose.

    Reseating is recovery, not extraction progress.  It therefore has its own
    signs and reference checks instead of weakening or sign-flipping the
    outward shelf gate.
    """

    try:
        planned = tuple(float(value) for value in planned_delta_world_m)
    except (TypeError, ValueError):
        planned = ()
    if (
        len(planned) != 3
        or not all(math.isfinite(value) for value in planned)
        or abs(planned[0] - 0.001) > 1e-12
        or abs(planned[1]) > 1e-12
        or abs(planned[2]) > 1e-12
    ):
        return GateResult(
            False,
            'reverse_reseat_plan_invalid',
            {'planned_world_delta_m': planned},
        )

    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    translation_mismatch = _distance(observed_delta, planned)
    cross_track = math.hypot(observed_delta[1], observed_delta[2])
    center_inward = observed_delta[0]
    start_maximum_x = book_maximum_world_x(start)
    observed_maximum_x = book_maximum_world_x(observed)
    stable_maximum_x = book_maximum_world_x(stable_reference)
    deepest_inward = observed_maximum_x - start_maximum_x
    reference_depth_overshoot = observed_maximum_x - stable_maximum_x
    stable_position_error = _distance(
        observed.position,
        stable_reference.position,
    )
    incremental_rotation_vector = _world_rotation_vector(
        start.quaternion,
        observed.quaternion,
    )
    incremental_rotation = math.sqrt(sum(
        value * value for value in incremental_rotation_vector
    ))
    stable_rotation_vector = _world_rotation_vector(
        stable_reference.quaternion,
        observed.quaternion,
    )
    stable_rotation = math.sqrt(sum(
        value * value for value in stable_rotation_vector
    ))
    stable_yaw = abs(stable_rotation_vector[2])
    start_stable_rotation = quaternion_distance(
        stable_reference.quaternion,
        start.quaternion,
    )
    rotation_growth = stable_rotation - start_stable_rotation
    floor_signed = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    limits = {
        'min_book_center_inward_progress_m': 0.00070,
        'min_book_deepest_edge_inward_progress_m': 0.00065,
        'max_book_from_hand_translation_change_m': 0.00018,
        'max_cross_track_motion_m': 0.00018,
        'max_reference_depth_overshoot_m': 0.00015,
        'max_stable_reference_position_error_m': 0.00025,
        'max_incremental_rotation_rad': 0.00120,
        'max_absolute_rotation_to_stable_reference_rad': 0.00080,
        'max_absolute_yaw_to_stable_reference_rad': 0.00080,
        'max_absolute_rotation_growth_from_start_rad': 0.00005,
        'min_floor_signed_distance_m': -0.00010,
        'max_floor_signed_distance_m': 0.00025,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })
    metrics = {
        'planned_world_dx_m': planned[0],
        'planned_world_dy_m': planned[1],
        'planned_world_dz_m': planned[2],
        'observed_world_dx_m': observed_delta[0],
        'observed_world_dy_m': observed_delta[1],
        'observed_world_dz_m': observed_delta[2],
        'book_center_inward_progress_m': center_inward,
        'book_deepest_edge_inward_progress_m': deepest_inward,
        'book_hand_translation_change_m': translation_mismatch,
        'book_cross_track_motion_m': cross_track,
        'book_incremental_rotation_rad': incremental_rotation,
        'book_stable_reference_rotation_rad': stable_rotation,
        'book_stable_reference_yaw_component_rad': stable_yaw,
        'book_rotation_growth_from_start_rad': rotation_growth,
        'book_floor_signed_m': floor_signed,
        'book_reference_depth_overshoot_m': reference_depth_overshoot,
        'book_stable_reference_position_error_m': stable_position_error,
        'book_maximum_world_x_m': observed_maximum_x,
    }
    for accepted, reason in (
        (
            center_inward >= limits['min_book_center_inward_progress_m'],
            'reverse_reseat_lacked_inward_progress',
        ),
        (
            deepest_inward
            >= limits['min_book_deepest_edge_inward_progress_m'],
            'reverse_reseat_lacked_deep_edge_inward_progress',
        ),
        (
            translation_mismatch
            <= limits['max_book_from_hand_translation_change_m'],
            'reverse_reseat_book_hand_translation_mismatch',
        ),
        (
            cross_track <= limits['max_cross_track_motion_m'],
            'reverse_reseat_cross_track_motion_exceeded_limit',
        ),
        (
            reference_depth_overshoot
            <= limits['max_reference_depth_overshoot_m'],
            'reverse_reseat_exceeded_reference_depth',
        ),
        (
            stable_position_error
            <= limits['max_stable_reference_position_error_m'],
            'reverse_reseat_stable_reference_position_mismatch',
        ),
        (
            incremental_rotation <= limits['max_incremental_rotation_rad'],
            'reverse_reseat_incremental_rotation_exceeded_limit',
        ),
        (
            stable_rotation
            <= limits['max_absolute_rotation_to_stable_reference_rad'],
            'reverse_reseat_absolute_rotation_exceeded_limit',
        ),
        (
            stable_yaw
            <= limits['max_absolute_yaw_to_stable_reference_rad'],
            'reverse_reseat_absolute_yaw_exceeded_limit',
        ),
        (
            rotation_growth
            <= limits['max_absolute_rotation_growth_from_start_rad'],
            'reverse_reseat_rotation_growth_exceeded_limit',
        ),
        (
            floor_signed >= limits['min_floor_signed_distance_m'],
            'reverse_reseat_shelf_penetration_exceeded_limit',
        ),
        (
            floor_signed <= limits['max_floor_signed_distance_m'],
            'reverse_reseat_lost_shelf_support',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'reverse_reseat_step_verified', metrics)


def stable_reseat_state_gate(
    observed: EntityPose,
    *,
    stable_reference: EntityPose,
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Independently certify a stationary post-reseat book state."""

    position_error = _distance(
        observed.position,
        stable_reference.position,
    )
    maximum_x_error = abs(
        book_maximum_world_x(observed)
        - book_maximum_world_x(stable_reference)
    )
    rotation_vector = _world_rotation_vector(
        stable_reference.quaternion,
        observed.quaternion,
    )
    rotation = math.sqrt(sum(value * value for value in rotation_vector))
    yaw = abs(rotation_vector[2])
    floor_signed = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    limits = {
        'max_stable_reference_position_error_m': 0.00025,
        'max_stable_reference_maximum_x_error_m': 0.00030,
        'max_absolute_rotation_to_stable_reference_rad': 0.00080,
        'max_absolute_yaw_to_stable_reference_rad': 0.00080,
        'min_floor_signed_distance_m': -0.00010,
        'max_floor_signed_distance_m': 0.00025,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })
    metrics = {
        'book_stable_reference_position_error_m': position_error,
        'book_stable_reference_maximum_x_error_m': maximum_x_error,
        'book_stable_reference_rotation_rad': rotation,
        'book_stable_reference_yaw_component_rad': yaw,
        'book_floor_signed_m': floor_signed,
        'book_maximum_world_x_m': book_maximum_world_x(observed),
    }
    for accepted, reason in (
        (
            position_error
            <= limits['max_stable_reference_position_error_m'],
            'post_reseat_stable_position_mismatch',
        ),
        (
            maximum_x_error
            <= limits['max_stable_reference_maximum_x_error_m'],
            'post_reseat_stable_depth_mismatch',
        ),
        (
            rotation
            <= limits['max_absolute_rotation_to_stable_reference_rad'],
            'post_reseat_absolute_rotation_exceeded_limit',
        ),
        (
            yaw <= limits['max_absolute_yaw_to_stable_reference_rad'],
            'post_reseat_absolute_yaw_exceeded_limit',
        ),
        (
            floor_signed >= limits['min_floor_signed_distance_m'],
            'post_reseat_shelf_penetration_exceeded_limit',
        ),
        (
            floor_signed <= limits['max_floor_signed_distance_m'],
            'post_reseat_lost_shelf_support',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'post_reseat_stable_state_verified', metrics)


def micro_outward_progress_gate(
    start: EntityPose,
    observed: EntityPose,
    *,
    planned_delta_world_m: Sequence[float],
    stable_reference: EntityPose,
    support_floor_world_z_m: float,
    policy: Optional[Mapping[str, object]] = None,
) -> GateResult:
    """Gate one independently paused 0.5 mm shelf-outward rung."""

    try:
        planned = tuple(float(value) for value in planned_delta_world_m)
    except (TypeError, ValueError):
        planned = ()
    if (
        len(planned) != 3
        or not all(math.isfinite(value) for value in planned)
        or abs(planned[0] + 0.0005) > 1e-12
        or abs(planned[1]) > 1e-12
        or abs(planned[2]) > 1e-12
    ):
        return GateResult(
            False,
            'micro_outward_plan_invalid',
            {'planned_world_delta_m': planned},
        )
    observed_delta = tuple(
        after - before
        for before, after in zip(start.position, observed.position)
    )
    center_outward = -observed_delta[0]
    deepest_outward = (
        book_maximum_world_x(start) - book_maximum_world_x(observed)
    )
    translation_mismatch = _distance(observed_delta, planned)
    cross_track = math.hypot(observed_delta[1], observed_delta[2])
    lateral_y = abs(observed_delta[1])
    incremental_rotation_vector = _world_rotation_vector(
        start.quaternion,
        observed.quaternion,
    )
    incremental_rotation = math.sqrt(sum(
        value * value for value in incremental_rotation_vector
    ))
    incremental_yaw = abs(incremental_rotation_vector[2])
    stable_rotation_vector = _world_rotation_vector(
        stable_reference.quaternion,
        observed.quaternion,
    )
    stable_rotation = math.sqrt(sum(
        value * value for value in stable_rotation_vector
    ))
    stable_yaw = abs(stable_rotation_vector[2])
    floor_signed = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    limits = {
        'min_book_center_outward_progress_m': 0.00030,
        'max_book_center_outward_progress_m': 0.00065,
        'min_book_deepest_edge_outward_progress_m': 0.00025,
        'max_book_deepest_edge_outward_progress_m': 0.00065,
        'max_book_from_hand_translation_change_m': 0.00020,
        'max_cross_track_motion_m': 0.00010,
        'max_lateral_y_motion_m': 0.00007,
        'max_incremental_rotation_rad': 0.00080,
        'max_incremental_yaw_component_rad': 0.00045,
        'max_absolute_rotation_to_stable_reference_rad': 0.00120,
        'max_absolute_yaw_to_stable_reference_rad': 0.00090,
        'min_floor_signed_distance_m': -0.00010,
        'max_floor_signed_distance_m': 0.00025,
    }
    if policy is not None:
        limits.update({
            key: float(policy[key])
            for key in tuple(limits)
            if key in policy
        })
    metrics = {
        'planned_world_dx_m': planned[0],
        'planned_world_dy_m': planned[1],
        'planned_world_dz_m': planned[2],
        'observed_world_dx_m': observed_delta[0],
        'observed_world_dy_m': observed_delta[1],
        'observed_world_dz_m': observed_delta[2],
        'book_center_outward_progress_m': center_outward,
        'book_deepest_edge_outward_progress_m': deepest_outward,
        'book_hand_translation_change_m': translation_mismatch,
        'book_cross_track_motion_m': cross_track,
        'book_lateral_y_motion_m': lateral_y,
        'book_incremental_rotation_rad': incremental_rotation,
        'book_incremental_yaw_component_rad': incremental_yaw,
        'book_stable_reference_rotation_rad': stable_rotation,
        'book_stable_reference_yaw_component_rad': stable_yaw,
        'book_floor_signed_m': floor_signed,
        'book_maximum_world_x_m': book_maximum_world_x(observed),
    }
    for accepted, reason in (
        (
            center_outward >= limits['min_book_center_outward_progress_m'],
            'micro_outward_lacked_center_progress',
        ),
        (
            center_outward <= limits['max_book_center_outward_progress_m'],
            'micro_outward_center_overshoot',
        ),
        (
            deepest_outward
            >= limits['min_book_deepest_edge_outward_progress_m'],
            'micro_outward_lacked_deep_edge_progress',
        ),
        (
            deepest_outward
            <= limits['max_book_deepest_edge_outward_progress_m'],
            'micro_outward_deep_edge_overshoot',
        ),
        (
            translation_mismatch
            <= limits['max_book_from_hand_translation_change_m'],
            'micro_outward_book_hand_translation_mismatch',
        ),
        (
            cross_track <= limits['max_cross_track_motion_m'],
            'micro_outward_cross_track_exceeded_limit',
        ),
        (
            lateral_y <= limits['max_lateral_y_motion_m'],
            'micro_outward_lateral_motion_exceeded_limit',
        ),
        (
            incremental_rotation <= limits['max_incremental_rotation_rad'],
            'micro_outward_incremental_rotation_exceeded_limit',
        ),
        (
            incremental_yaw
            <= limits['max_incremental_yaw_component_rad'],
            'micro_outward_incremental_yaw_exceeded_limit',
        ),
        (
            stable_rotation
            <= limits['max_absolute_rotation_to_stable_reference_rad'],
            'micro_outward_absolute_rotation_exceeded_limit',
        ),
        (
            stable_yaw
            <= limits['max_absolute_yaw_to_stable_reference_rad'],
            'micro_outward_absolute_yaw_exceeded_limit',
        ),
        (
            floor_signed >= limits['min_floor_signed_distance_m'],
            'micro_outward_shelf_penetration_exceeded_limit',
        ),
        (
            floor_signed <= limits['max_floor_signed_distance_m'],
            'micro_outward_lost_shelf_support',
        ),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'micro_outward_rung_verified', metrics)


def stable_reference_orientation_gate(
    observed: EntityPose,
    stable_reference: EntityPose,
    *,
    maximum_rotation_rad: float,
    maximum_yaw_component_rad: float,
) -> GateResult:
    """Bound a checkpoint against the long-term stable book orientation."""

    rotation_vector = _world_rotation_vector(
        stable_reference.quaternion,
        observed.quaternion,
    )
    rotation = math.sqrt(sum(value * value for value in rotation_vector))
    yaw = abs(rotation_vector[2])
    metrics = {
        'stable_reference_rotation_rad': rotation,
        'stable_reference_yaw_component_rad': yaw,
        'maximum_rotation_rad': float(maximum_rotation_rad),
        'maximum_yaw_component_rad': float(maximum_yaw_component_rad),
    }
    if rotation > float(maximum_rotation_rad):
        return GateResult(
            False,
            'checkpoint_absolute_rotation_exceeded_limit',
            metrics,
        )
    if yaw > float(maximum_yaw_component_rad):
        return GateResult(
            False,
            'checkpoint_absolute_yaw_exceeded_limit',
            metrics,
        )
    return GateResult(True, 'checkpoint_stable_orientation_verified', metrics)


def supported_book_state_gate(
    observed: EntityPose,
    stable_reference: EntityPose,
    *,
    support_floor_world_z_m: float,
    maximum_rotation_rad: float,
    maximum_yaw_component_rad: float,
    minimum_floor_signed_distance_m: float,
    maximum_floor_signed_distance_m: float,
) -> GateResult:
    """Verify stable orientation and continued shelf support without motion."""

    orientation = stable_reference_orientation_gate(
        observed,
        stable_reference,
        maximum_rotation_rad=maximum_rotation_rad,
        maximum_yaw_component_rad=maximum_yaw_component_rad,
    )
    floor_signed = (
        book_minimum_world_z(observed) - float(support_floor_world_z_m)
    )
    metrics = {
        **orientation.metrics,
        'book_floor_signed_m': floor_signed,
        'minimum_floor_signed_distance_m': float(
            minimum_floor_signed_distance_m
        ),
        'maximum_floor_signed_distance_m': float(
            maximum_floor_signed_distance_m
        ),
    }
    if not orientation.ok:
        return GateResult(False, orientation.reason, metrics)
    if floor_signed < float(minimum_floor_signed_distance_m):
        return GateResult(
            False,
            'supported_book_shelf_penetration_exceeded_limit',
            metrics,
        )
    if floor_signed > float(maximum_floor_signed_distance_m):
        return GateResult(
            False,
            'supported_book_lost_shelf_support',
            metrics,
        )
    return GateResult(True, 'supported_book_state_verified', metrics)


def bilateral_pressure_retention_gate(
    reference_force_n: Sequence[float],
    observed_force_n: Sequence[float],
    *,
    minimum_each_side_retention_fraction: float,
    maximum_normalized_balance_change: float,
) -> GateResult:
    """Require retained bilateral pressure without hiding a side-to-side shift."""

    try:
        reference = tuple(float(value) for value in reference_force_n)
        observed = tuple(float(value) for value in observed_force_n)
    except (TypeError, ValueError):
        reference = ()
        observed = ()
    valid = bool(
        len(reference) == 2
        and len(observed) == 2
        and all(math.isfinite(value) and value > 0.0 for value in reference)
        and all(math.isfinite(value) and value >= 0.0 for value in observed)
        and math.isfinite(float(minimum_each_side_retention_fraction))
        and 0.0 <= float(minimum_each_side_retention_fraction) <= 1.0
        and math.isfinite(float(maximum_normalized_balance_change))
        and 0.0 <= float(maximum_normalized_balance_change) <= 1.0
    )
    if not valid:
        return GateResult(
            False,
            'bilateral_pressure_retention_input_invalid',
            {
                'reference_force_n': reference,
                'observed_force_n': observed,
            },
        )
    retention = tuple(
        after / before for before, after in zip(reference, observed)
    )
    reference_total = sum(reference)
    observed_total = sum(observed)
    reference_balance = reference[0] / reference_total
    observed_balance = (
        observed[0] / observed_total if observed_total > 0.0 else math.nan
    )
    balance_change = abs(observed_balance - reference_balance)
    metrics = {
        'reference_left_force_n': reference[0],
        'reference_right_force_n': reference[1],
        'observed_left_force_n': observed[0],
        'observed_right_force_n': observed[1],
        'left_force_retention_fraction': retention[0],
        'right_force_retention_fraction': retention[1],
        'reference_normalized_left_balance': reference_balance,
        'observed_normalized_left_balance': observed_balance,
        'normalized_balance_change': balance_change,
        'minimum_each_side_retention_fraction': float(
            minimum_each_side_retention_fraction
        ),
        'maximum_normalized_balance_change': float(
            maximum_normalized_balance_change
        ),
    }
    if min(retention) < float(minimum_each_side_retention_fraction):
        return GateResult(
            False,
            'bilateral_force_weakened_after_shelf_step',
            metrics,
        )
    if balance_change > float(maximum_normalized_balance_change):
        return GateResult(
            False,
            'bilateral_force_balance_shifted_after_shelf_step',
            metrics,
        )
    return GateResult(True, 'bilateral_force_retention_verified', metrics)


def practical_preload_probe_gate(
    start_book: EntityPose,
    observed_book: EntityPose,
    *,
    start_arm_q8: Sequence[float],
    observed_arm_q8: Sequence[float],
    start_base: Pose2,
    observed_base: Pose2,
    start_aperture_m: float,
    commanded_aperture_m: float,
    observed_aperture_m: float,
    baseline_force_n: Sequence[float],
    observed_force_n: Sequence[float],
    support_floor_world_z_m: float,
) -> GateResult:
    """Accept exactly one bounded pressure-only 0.10 mm preload probe."""

    baseline = tuple(float(value) for value in baseline_force_n)
    observed_force = tuple(float(value) for value in observed_force_n)
    start_arm = tuple(float(value) for value in start_arm_q8)
    observed_arm = tuple(float(value) for value in observed_arm_q8)
    certified_target = 0.028908127502464856
    inputs = (
        *baseline,
        *observed_force,
        *start_arm,
        *observed_arm,
        float(start_aperture_m),
        float(commanded_aperture_m),
        float(observed_aperture_m),
        float(support_floor_world_z_m),
    )
    if (
        len(baseline) != 2
        or len(observed_force) != 2
        or len(start_arm) != 8
        or len(observed_arm) != 8
        or not all(math.isfinite(value) for value in inputs)
        or min(baseline) <= 0.0
        or min(observed_force) < 0.0
        or abs(float(commanded_aperture_m) - certified_target) > 1e-12
    ):
        return GateResult(False, 'practical_preload_probe_input_invalid', {})

    translation = _distance(start_book.position, observed_book.position)
    rotation = quaternion_distance(
        start_book.quaternion,
        observed_book.quaternion,
    )
    corner_metric = translation + 0.15 * rotation
    lateral = abs(observed_book.position[1] - start_book.position[1])
    floor = float(support_floor_world_z_m)
    start_face = (
        book_minimum_world_z_at_x(start_book, ROUTE_SHELF_FRONT_WORLD_X_M)
        - floor
    )
    observed_face = (
        book_minimum_world_z_at_x(observed_book, ROUTE_SHELF_FRONT_WORLD_X_M)
        - floor
    )
    start_deep = book_deepest_extent_lower_world_z(start_book) - floor
    observed_deep = book_deepest_extent_lower_world_z(observed_book) - floor
    start_overlap = (
        book_maximum_world_x(start_book) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    observed_overlap = (
        book_maximum_world_x(observed_book) - ROUTE_SHELF_FRONT_WORLD_X_M
    )
    arm_error = max(
        abs(after - before)
        for before, after in zip(start_arm, observed_arm)
    )
    base_translation = math.hypot(
        observed_base.x - start_base.x,
        observed_base.y - start_base.y,
    )
    base_yaw = abs(math.atan2(
        math.sin(observed_base.yaw - start_base.yaw),
        math.cos(observed_base.yaw - start_base.yaw),
    ))
    aperture_decrement = float(start_aperture_m) - float(observed_aperture_m)
    aperture_error = abs(
        float(observed_aperture_m) - float(commanded_aperture_m)
    )
    baseline_total = sum(baseline)
    observed_total = sum(observed_force)
    required_total_gain = max(0.20, 0.05 * baseline_total)
    required_each = tuple(max(2.0, 0.90 * value) for value in baseline)
    baseline_balance = baseline[0] / baseline_total
    observed_balance = (
        observed_force[0] / observed_total
        if observed_total > 0.0 else math.nan
    )
    balance_change = abs(observed_balance - baseline_balance)
    metrics = {
        'book_translation_m': translation,
        'book_rotation_rad': rotation,
        'book_corner_metric_m': corner_metric,
        'book_lateral_motion_m': lateral,
        'book_start_shelf_face_signed_m': start_face,
        'book_shelf_face_signed_m': observed_face,
        'book_start_deep_edge_signed_m': start_deep,
        'book_deep_edge_signed_m': observed_deep,
        'book_deep_edge_regression_m': start_deep - observed_deep,
        'book_shelf_overlap_change_m': observed_overlap - start_overlap,
        'maximum_left_arm_error_rad': arm_error,
        'base_translation_m': base_translation,
        'base_yaw_rad': base_yaw,
        'commanded_aperture_m': float(commanded_aperture_m),
        'observed_aperture_m': float(observed_aperture_m),
        'aperture_decrement_m': aperture_decrement,
        'aperture_endpoint_error_m': aperture_error,
        'baseline_left_force_n': baseline[0],
        'baseline_right_force_n': baseline[1],
        'observed_left_force_n': observed_force[0],
        'observed_right_force_n': observed_force[1],
        'required_left_force_n': required_each[0],
        'required_right_force_n': required_each[1],
        'observed_total_force_n': observed_total,
        'required_total_force_gain_n': required_total_gain,
        'observed_total_force_gain_n': observed_total - baseline_total,
        'normalized_balance_change': balance_change,
    }
    for accepted, reason in (
        (arm_error <= 0.00025, 'practical_preload_left_arm_moved'),
        (base_translation <= 0.00025, 'practical_preload_base_moved'),
        (base_yaw <= 0.00025, 'practical_preload_base_rotated'),
        (aperture_error <= 0.00006, 'practical_preload_aperture_mismatch'),
        (
            0.00005 <= aperture_decrement <= 0.00010,
            'practical_preload_aperture_decrement_invalid',
        ),
        (
            float(observed_aperture_m) >= 0.02885,
            'practical_preload_aperture_below_hard_floor',
        ),
        (translation <= 0.00020, 'practical_preload_book_translation'),
        (rotation <= 0.0015, 'practical_preload_book_rotation'),
        (corner_metric <= 0.00025, 'practical_preload_book_corner_motion'),
        (lateral <= 0.00010, 'practical_preload_book_lateral_motion'),
        (abs(observed_face) <= 0.00025, 'practical_preload_face_motion'),
        (observed_deep >= 0.0065, 'practical_preload_deep_edge_too_low'),
        (
            start_deep - observed_deep <= 0.00025,
            'practical_preload_deep_edge_regressed',
        ),
        (
            abs(observed_overlap - start_overlap) <= 0.00030,
            'practical_preload_shelf_overlap_changed',
        ),
        (
            observed_force[0] >= required_each[0],
            'practical_preload_left_force_below_target',
        ),
        (
            observed_force[1] >= required_each[1],
            'practical_preload_right_force_below_target',
        ),
        (
            observed_total - baseline_total >= required_total_gain,
            'practical_preload_sustained_force_gain_missing',
        ),
        (max(observed_force) <= 8.0, 'practical_preload_force_overload'),
        (balance_change <= 0.15, 'practical_preload_force_balance_shift'),
    ):
        if not accepted:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'practical_preload_probe_verified', metrics)


def consecutive_soft_limit_gate(
    value: float,
    generation: int,
    *,
    soft_limit: float,
    hard_limit: float,
    required_consecutive_samples: int,
    previous_generation: int,
    previous_consecutive_samples: int,
) -> Tuple[GateResult, int, int]:
    """Debounce one soft-limit crossing while retaining a hard fail-safe."""

    inputs_valid = bool(
        math.isfinite(float(value))
        and float(value) >= 0.0
        and int(generation) >= 0
        and math.isfinite(float(soft_limit))
        and float(soft_limit) >= 0.0
        and math.isfinite(float(hard_limit))
        and float(hard_limit) >= float(soft_limit)
        and int(required_consecutive_samples) >= 1
        and int(previous_consecutive_samples) >= 0
    )
    if not inputs_valid:
        return (
            GateResult(False, 'debounced_limit_input_invalid', {}),
            int(previous_consecutive_samples),
            int(previous_generation),
        )
    metrics = {
        'observed_value': float(value),
        'soft_limit': float(soft_limit),
        'hard_limit': float(hard_limit),
        'consecutive_samples_required': int(required_consecutive_samples),
    }
    if float(value) > float(hard_limit):
        return (
            GateResult(False, 'debounced_limit_hard_trip', metrics),
            int(required_consecutive_samples),
            int(generation),
        )
    count = int(previous_consecutive_samples)
    latest_generation = int(previous_generation)
    if int(generation) != int(previous_generation):
        count = count + 1 if float(value) > float(soft_limit) else 0
        latest_generation = int(generation)
    metrics['consecutive_samples'] = count
    if count >= int(required_consecutive_samples):
        return (
            GateResult(False, 'debounced_limit_sustained_trip', metrics),
            count,
            latest_generation,
        )
    return (
        GateResult(True, 'debounced_limit_clear', metrics),
        count,
        latest_generation,
    )


def final_extraction_geometry_gate(
    start: EntityPose,
    observed: EntityPose,
) -> GateResult:
    expected = (
        start.position[0] - OUTWARD_TOTAL_M,
        start.position[1],
        start.position[2] + MICRO_LIFT_TOTAL_M,
    )
    attachment_error = _distance(observed.position, expected)
    rotation_error = quaternion_distance(
        start.quaternion,
        observed.quaternion,
    )
    corner_displacement = (
        attachment_error
        + 2.0 * PAYLOAD_CONSERVATIVE_RADIUS_M
        * math.sin(0.5 * rotation_error)
    )
    maximum_world_x = book_maximum_world_x(observed)
    shelf_clearance = ROUTE_SHELF_FRONT_WORLD_X_M - maximum_world_x
    floor_clearance = (
        book_minimum_world_z(observed) - ROUTE_SUPPORT_FLOOR_WORLD_Z_M
    )
    metrics = {
        'final_attachment_position_error_m': attachment_error,
        'final_attachment_rotation_error_rad': rotation_error,
        'final_attachment_corner_displacement_m': corner_displacement,
        'book_maximum_world_x_m': maximum_world_x,
        'final_shelf_clearance_m': shelf_clearance,
        'final_floor_clearance_m': floor_clearance,
    }
    if attachment_error > PAYLOAD_ATTACHMENT_POSITION_LIMIT_M:
        return GateResult(
            False,
            'final_book_hand_attachment_translation_mismatch',
            metrics,
        )
    if rotation_error > PAYLOAD_TOTAL_ROTATION_LIMIT_RAD:
        return GateResult(
            False,
            'final_book_hand_attachment_rotation_mismatch',
            metrics,
        )
    if corner_displacement > PAYLOAD_TOTAL_CORNER_DISPLACEMENT_LIMIT_M:
        return GateResult(
            False,
            'final_book_hand_attachment_corner_displacement_mismatch',
            metrics,
        )
    if shelf_clearance < FINAL_SHELF_CLEARANCE_REQUIRED_M:
        return GateResult(
            False,
            'book_not_shelf_clear_at_extraction_endpoint',
            metrics,
        )
    if floor_clearance < 0.003:
        return GateResult(
            False,
            'book_not_lifted_clear_of_support_floor',
            metrics,
        )
    return GateResult(True, 'final_loaded_extraction_geometry_verified', metrics)


def world_paused_gate(message: str, expected: bool) -> GateResult:
    text = str(message)
    matches = re.findall(r'(?m)^paused:\s*(true|false)\s*$', text)
    if len(matches) > 1:
        return GateResult(False, 'world_pause_state_unavailable', {})
    if matches:
        observed = matches[0] == 'true'
    elif 'sim_time {' in text and re.search(r'(?m)^iterations:\s*\d+\s*$', text):
        # Gazebo Transport's protobuf text format omits scalar fields at their
        # default value; an unpaused ``false`` therefore has no ``paused``
        # line at all in an otherwise complete WorldStatistics message.
        observed = False
    else:
        return GateResult(False, 'world_pause_state_unavailable', {})
    return GateResult(
        observed is bool(expected),
        'world_pause_state_verified' if observed is bool(expected)
        else 'world_pause_state_mismatch',
        {'world_paused': float(observed)},
    )


def world_stats_snapshot(message: str) -> Tuple[float, int, bool]:
    """Parse the exact simulation time, iteration, and pause state."""

    text = str(message)
    sim_block = re.search(r'(?ms)^sim_time\s*\{(.*?)^\}', text)
    iterations_match = re.search(r'(?m)^iterations:\s*(\d+)\s*$', text)
    pause = world_paused_gate(text, True)
    if sim_block is None or iterations_match is None:
        raise RuntimeError('Gazebo world statistics were incomplete')
    seconds_match = re.search(
        r'(?m)^\s*sec:\s*(-?\d+)\s*$',
        sim_block.group(1),
    )
    nanoseconds_match = re.search(
        r'(?m)^\s*nsec:\s*(-?\d+)\s*$',
        sim_block.group(1),
    )
    if seconds_match is None:
        raise RuntimeError('Gazebo world simulation time was unavailable')
    seconds = int(seconds_match.group(1))
    nanoseconds = (
        int(nanoseconds_match.group(1))
        if nanoseconds_match is not None else 0
    )
    if nanoseconds < 0 or nanoseconds >= 1_000_000_000:
        raise RuntimeError('Gazebo world simulation time was malformed')
    return (
        float(seconds) + float(nanoseconds) / 1e9,
        int(iterations_match.group(1)),
        bool(pause.ok),
    )


def _quaternion_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> Tuple[float, float, float, float]:
    """Convert a serialized Gazebo XYZ/RPY pose to an XYZW quaternion."""

    cr = math.cos(0.5 * float(roll))
    sr = math.sin(0.5 * float(roll))
    cp = math.cos(0.5 * float(pitch))
    sp = math.sin(0.5 * float(pitch))
    cy = math.cos(0.5 * float(yaw))
    sy = math.sin(0.5 * float(yaw))
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def passive_scene_reference_gate(
    message: str,
    reference: Mapping[str, object],
    *,
    translation_limit_m: float,
    rotation_limit_rad: float,
) -> GateResult:
    """Bind every certified non-target book and arena prop to its pose."""

    maximum_translation = 0.0
    maximum_rotation = 0.0
    checked = 0
    try:
        for name, serialized_pose in reference.items():
            values = tuple(float(value) for value in str(serialized_pose).split())
            if len(values) != 6 or not all(math.isfinite(value) for value in values):
                raise ValueError(f'malformed passive pose for {name!r}')
            expected = EntityPose(
                tuple(values[:3]),
                _quaternion_from_rpy(*values[3:]),
            )
            observed = entity_pose_from_dynamic_pose(message, str(name))
            maximum_translation = max(
                maximum_translation,
                _distance(observed.position, expected.position),
            )
            maximum_rotation = max(
                maximum_rotation,
                quaternion_distance(observed.quaternion, expected.quaternion),
            )
            checked += 1
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return GateResult(False, 'passive_scene_reference_unavailable', {
            'checked_entity_count': float(checked),
            'reference_entity_count': float(len(reference)),
            'error_length': float(len(str(exc))),
        })
    metrics = {
        'checked_entity_count': float(checked),
        'maximum_passive_translation_error_m': maximum_translation,
        'maximum_passive_rotation_error_rad': maximum_rotation,
        'passive_translation_limit_m': float(translation_limit_m),
        'passive_rotation_limit_rad': float(rotation_limit_rad),
    }
    if maximum_translation > float(translation_limit_m):
        return GateResult(False, 'passive_scene_translation_changed', metrics)
    if maximum_rotation > float(rotation_limit_rad):
        return GateResult(False, 'passive_scene_rotation_changed', metrics)
    return GateResult(True, 'certified_passive_scene_verified', metrics)


def read_world_stats_message() -> str:
    completed = subprocess.run(
        ['gz', 'topic', '-e', '-t', WORLD_STATS_TOPIC, '-n', '1'],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    return completed.stdout


def read_exact_seed101_dynamic_pose_message(
    *,
    attempts: int = 5,
) -> str:
    """Retry transient partial transport samples, never accept one."""
    last_reason = 'Gazebo dynamic pose was unavailable'
    for _ in range(max(1, int(attempts))):
        try:
            message = read_dynamic_pose_message()
            book_poses_from_dynamic_pose(message)
            entity_pose_from_dynamic_pose(message, 'tiago_pro')
            return message
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            last_reason = str(exc)
            time.sleep(0.03)
    raise RuntimeError(
        f'exact seed-101 Gazebo dynamic pose unavailable:{last_reason}'
    )


def read_target_base_dynamic_pose_message(
    *,
    attempts: int = 5,
) -> str:
    """Require one complete target/base sample for the paused endpoint."""
    last_reason = 'Gazebo target/base pose was unavailable'
    for _ in range(max(1, int(attempts))):
        try:
            message = read_dynamic_pose_message()
            entity_pose_from_dynamic_pose(message, BOOK)
            entity_pose_from_dynamic_pose(message, 'tiago_pro')
            return message
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            last_reason = str(exc)
            time.sleep(0.03)
    raise RuntimeError(
        f'complete target/base Gazebo pose unavailable:{last_reason}'
    )


def set_world_paused(paused: bool) -> bool:
    """Use world control only; never mutate a model or link pose."""
    completed = subprocess.run(
        [
            'gz', 'service', '-s', WORLD_CONTROL_SERVICE,
            '--reqtype', 'gz.msgs.WorldControl',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', f'pause: {str(bool(paused)).lower()}',
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=7.0,
    )
    response = f'{completed.stdout}\n{completed.stderr}'
    return completed.returncode == 0 and bool(
        re.search(r'(?m)^data:\s*true\s*$', response)
    )


def set_world_paused_confirmed(paused: bool, *, attempts: int = 3) -> bool:
    """Request a pause transition and confirm the resulting world state."""
    for _ in range(max(1, int(attempts))):
        try:
            requested = set_world_paused(paused)
            confirmed = (
                requested
                and world_paused_gate(
                    read_world_stats_message(),
                    paused,
                ).ok
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            confirmed = False
        if confirmed:
            return True
        time.sleep(0.05)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Resume the strict one-left-arm seed-101 held extraction',
    )
    parser.add_argument(
        '--confirm-seed',
        type=int,
        required=True,
        choices=(EXPECTED_SEED,),
    )
    parser.add_argument(
        '--stage',
        choices=(
            'loaded-recovery',
            'diagonal-peel',
            'outward-probe',
            'outward-scaleup',
            'outward-scaleup-continue',
            'outward-settle-resample',
            'outward-reverse-reseat',
            'outward-post-reseat-settle',
            'outward-halfmillimeter',
            'outward-halfmillimeter-midroute-settle',
            'practical-preload-resume',
            'practical-preloaded-ray-resume',
            'practical-force-release-resume',
            'practical-reseat-support',
            'practical-upright-direct-pull',
            'practical-tighten-direct-pull',
            'practical-full-pull',
            'practical-full-pull-resume',
        ),
        default='loaded-recovery',
        help=(
            'Run the original shelf-clear route or exactly one guarded '
            'post-slip diagonal peel / shelf-outward probe step'
        ),
    )
    args = parser.parse_args()

    diagonal_peel = args.stage == 'diagonal-peel'
    outward_probe = args.stage == 'outward-probe'
    outward_scaleup = args.stage == 'outward-scaleup'
    outward_scaleup_continue = args.stage == 'outward-scaleup-continue'
    settle_resample = args.stage == 'outward-settle-resample'
    reverse_reseat = args.stage == 'outward-reverse-reseat'
    post_reseat_settle = args.stage == 'outward-post-reseat-settle'
    halfmillimeter_outward = args.stage == 'outward-halfmillimeter'
    halfmillimeter_midroute_settle = (
        args.stage == 'outward-halfmillimeter-midroute-settle'
    )
    practical_preload_resume = args.stage == 'practical-preload-resume'
    practical_preloaded_ray_resume = (
        args.stage == 'practical-preloaded-ray-resume'
    )
    practical_force_release_resume = (
        args.stage == 'practical-force-release-resume'
    )
    practical_reseat_support = args.stage == 'practical-reseat-support'
    practical_upright_direct_pull = (
        args.stage == 'practical-upright-direct-pull'
    )
    practical_tighten_direct_pull = (
        args.stage == 'practical-tighten-direct-pull'
    )
    practical_upright_extract = (
        practical_upright_direct_pull or practical_tighten_direct_pull
    )
    practical_full_pull_initial = args.stage == 'practical-full-pull'
    practical_full_pull_resume = args.stage in (
        'practical-full-pull-resume',
        'practical-preloaded-ray-resume',
        'practical-force-release-resume',
        'practical-reseat-support',
        'practical-upright-direct-pull',
        'practical-tighten-direct-pull',
    )
    practical_full_pull = (
        practical_full_pull_initial or practical_full_pull_resume
    )
    shelf_probe = (
        diagonal_peel
        or outward_probe
        or outward_scaleup
        or outward_scaleup_continue
        or settle_resample
        or reverse_reseat
        or post_reseat_settle
        or halfmillimeter_outward
        or halfmillimeter_midroute_settle
        or practical_preload_resume
        or practical_full_pull
    )
    active_final_dwell_s = 0.20
    active_final_endpoint_policy: Optional[Mapping[str, object]] = None
    active_supported_book_final_policy: Optional[Mapping[str, object]] = None
    active_minimum_force_n = (0.0, 0.0)
    active_minimum_total_force_n = 0.0
    active_maximum_force_n = (math.inf, math.inf)
    active_continuous_force_retention_fraction = 0.0
    active_settle_reference: Optional[EntityPose] = None
    active_settle_planned_delta = (0.0, 0.0, 0.0)
    active_settle_rotation_growth_limit_rad = 0.0
    active_reseat_reference: Optional[EntityPose] = None
    active_reseat_planned_delta = (0.0, 0.0, 0.0)
    active_continuous_stable_quaternion: Optional[Tuple[float, ...]] = None
    active_continuous_absolute_rotation_limit_rad = math.inf
    active_continuous_absolute_yaw_limit_rad = math.inf
    active_continuous_progress_regression_limit_m = (
        PAYLOAD_INWARD_REGRESSION_LIMIT_M
    )
    active_continuous_progress_overshoot_limit_m = (
        PAYLOAD_SHELF_CONTINUOUS_TRANSLATION_LIMIT_M
    )
    active_continuous_cross_track_limit_m = (
        PAYLOAD_SHELF_CONTINUOUS_TRANSLATION_LIMIT_M
    )
    active_continuous_stationary_translation_limit_m = (
        PAYLOAD_CONTINUOUS_TRANSLATION_LIMIT_M
    )
    active_continuous_lateral_y_limit_m = math.inf
    active_continuous_relative_rotation_limit_rad: Optional[float] = None
    active_continuous_relative_yaw_limit_rad = math.inf
    active_continuous_relative_yaw_hard_limit_rad = math.inf
    active_continuous_relative_yaw_consecutive_samples = 1
    active_checkpoint_stable_reference: Optional[EntityPose] = None
    active_checkpoint_absolute_rotation_limit_rad = math.inf
    active_checkpoint_absolute_yaw_limit_rad = math.inf
    active_maximum_force_balance_change = 1.0
    active_dense_maximum_increment_rad = ROUTE_DENSE_MAXIMUM_INCREMENT_RAD
    active_checkpoint_arm_limit_rad = CHECKPOINT_ARM_LIMIT_RAD
    active_checkpoint_gripper_limit_m = CHECKPOINT_GRIPPER_LIMIT_M
    active_checkpoint_book_position_limit_m = CHECKPOINT_BOOK_POSITION_LIMIT_M
    active_checkpoint_book_rotation_limit_rad = CHECKPOINT_BOOK_ROTATION_LIMIT_RAD
    active_startup_stability_translation_limit_m = (
        CHECKPOINT_STABILITY_TRANSLATION_LIMIT_M
    )
    active_startup_stability_rotation_limit_rad = (
        CHECKPOINT_STABILITY_ROTATION_LIMIT_RAD
    )
    active_final_stability_translation_limit_m = (
        CHECKPOINT_STABILITY_TRANSLATION_LIMIT_M
    )
    active_final_stability_rotation_limit_rad = (
        CHECKPOINT_STABILITY_ROTATION_LIMIT_RAD
    )
    active_pressure_probe_translation_limit_m = (
        PAYLOAD_SHELF_CONTINUOUS_TRANSLATION_LIMIT_M
        if shelf_probe else PAYLOAD_CONTINUOUS_TRANSLATION_LIMIT_M
    )
    active_continuous_corner_displacement_limit_m = (
        PAYLOAD_CONTINUOUS_CORNER_DISPLACEMENT_LIMIT_M
    )
    active_practical_maximum_position_error_m = 0.015
    active_practical_maximum_lateral_error_m = 0.010
    active_practical_maximum_vertical_error_m = 0.012
    active_practical_maximum_rotation_rad = 0.12
    active_practical_minimum_floor_signed_m = -0.002
    active_practical_minimum_final_shelf_clearance_m = 0.010
    active_practical_minimum_vertical_progress_m = -math.inf
    active_resume_progress_mode = 'lip_tangent'
    active_radial_unit_world: Optional[Tuple[float, float, float]] = None
    active_reseat_upright_position: Optional[Tuple[float, float, float]] = None
    active_reseat_expected_signed_y_rotation_rad: Optional[float] = None
    active_continuous_route_goal = False
    active_continuous_route_duration_s = 15.0
    active_continuous_route_use_row_durations = False
    active_upright_extract_lift_m = 0.0
    active_upright_extract_outward_m = 0.0
    active_upright_extract_unload_outward_limit_m = 0.00050
    active_upright_extract_minimum_retained_lift_m = 0.0
    active_upright_extract_minimum_pull_floor_clearance_m = 0.0
    active_upright_extract_lift_regression_limit_m = 0.00050
    active_upright_extract_outward_regression_limit_m = 0.00050
    active_upright_extract_clearance_regression_limit_m = 0.00050
    active_upright_extract_attachment_translation_limit_m = 0.00050
    active_upright_extract_attachment_rotation_limit_rad = 0.002
    active_upright_extract_attachment_corner_limit_m = 0.00075
    active_integrated_preload_target_m: Optional[float] = None
    active_integrated_preload_use_pressure_ladder = False
    active_pre_pull_proof_q8: Optional[Tuple[float, ...]] = None
    active_pre_pull_proof_duration_s = 0.0
    active_pre_pull_proof_minimum_book_rise_m = 0.0
    active_pre_pull_proof_maximum_attachment_translation_m = 0.0
    active_pre_pull_proof_maximum_attachment_rotation_rad = 0.0
    active_pre_pull_proof_maximum_attachment_corner_m = 0.0
    active_integrated_preload_policy: Mapping[str, float] = {
        'minimum_each_force_n': 2.10,
        'minimum_total_force_n': 4.40,
        'minimum_each_force_gain_n': 0.35,
        'minimum_weaker_stronger_ratio': 0.85,
        'maximum_each_force_n': 8.0,
        'maximum_book_translation_m': 0.00010,
        'maximum_book_rotation_rad': 0.00050,
        'maximum_book_corner_motion_m': 0.00015,
        'maximum_book_lateral_motion_m': 0.00005,
        'maximum_arm_error_rad': 0.00025,
        'maximum_base_translation_m': 0.00025,
        'maximum_base_yaw_error_rad': 0.00025,
        'maximum_aperture_error_m': 0.00003,
        'live_minimum_each_force_n': 1.80,
        'live_minimum_total_force_n': 3.80,
        'live_force_retention_fraction': 0.75,
        'live_maximum_balance_change': 0.15,
        'maximum_launch_delay_s': 0.15,
        'close_step_m': 0.00010,
        'maximum_close_steps': 1.0,
        'step_motion_s': 0.32,
            'minimum_confirmation_s': 0.12,
            'early_balance_check_total_force_n': math.inf,
            'early_minimum_weaker_stronger_ratio': 0.0,
            'minimum_book_floor_signed_m': -math.inf,
            'minimum_shelf_overlap_m': -math.inf,
            'minimum_shelf_com_depth_m': -math.inf,
    }
    if diagonal_peel:
        from erc_phase1_solution.seed101_diagonal_peel_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as PEEL_CORNERS,
            AUDIT as PEEL_AUDIT,
            CHECKPOINT_BASE_WORLD_XYYAW as PEEL_BASE_WORLD_XYYAW,
            CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M,
            CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M as PEEL_BOOK_BOUNDS,
            CHECKPOINT_BOOK_POSITION_WORLD_M as PEEL_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as PEEL_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as PEEL_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as PEEL_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as PEEL_LEFT_Q8,
            CHECKPOINT_RIGHT_Q7 as PEEL_RIGHT_Q7,
            ENDPOINT_POLICY as PEEL_ENDPOINT_POLICY,
            MINIMUM_DURATION_S as PEEL_MINIMUM_DURATION_S,
            PLANNED_WORLD_DELTA_M as PEEL_WORLD_DELTA_M,
            ROUTE as PEEL_ROUTE,
            ROUTE_Q8 as PEEL_ROUTE_Q8,
            validate_certificate as validate_peel_certificate,
        )

        active_route = PEEL_ROUTE
        active_q8 = PEEL_ROUTE_Q8
        active_checkpoint_left_q8 = PEEL_LEFT_Q8
        active_checkpoint_gripper_m = PEEL_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = PEEL_BASE_WORLD_XYYAW
        active_checkpoint_book_position = PEEL_BOOK_POSITION
        active_checkpoint_book_quaternion = PEEL_BOOK_QUATERNION
        active_checkpoint_right_q7 = PEEL_RIGHT_Q7
        active_checkpoint_head_q2 = PEEL_HEAD_Q2
        active_attached_book_corners = PEEL_CORNERS
        active_validate_certificate = validate_peel_certificate
        active_audit = PEEL_AUDIT
        active_peel_delta = PEEL_WORLD_DELTA_M
        active_peel_duration = max(1.2, PEEL_MINIMUM_DURATION_S)
        active_support_floor_world_z = (
            min(float(row[2]) for row in PEEL_BOOK_BOUNDS)
            - CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M
        )
        active_endpoint_policy = PEEL_ENDPOINT_POLICY
        active_certificate_reason = 'diagonal_peel_certificate'
        active_result_stage = 'strict_pressure_pick_diagonal_peel'
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = 0.50
    elif outward_probe:
        from erc_phase1_solution.seed101_pure_outward_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as OUTWARD_CORNERS,
            AUDIT as OUTWARD_AUDIT,
            CHECKPOINT_BASE_WORLD_POSITION_M as OUTWARD_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as OUTWARD_BASE_YAW,
            CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M as OUTWARD_FLOOR_SIGNED,
            CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M as OUTWARD_BOOK_BOUNDS,
            CHECKPOINT_BOOK_POSITION_WORLD_M as OUTWARD_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as OUTWARD_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as OUTWARD_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as OUTWARD_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as OUTWARD_LEFT_Q8,
            CHECKPOINT_RIGHT_Q7 as OUTWARD_RIGHT_Q7,
            ENDPOINT_POLICY as OUTWARD_ENDPOINT_POLICY,
            MINIMUM_DURATION_S as OUTWARD_MINIMUM_DURATION_S,
            PLANNED_WORLD_DELTA_M as OUTWARD_WORLD_DELTA_M,
            ROUTE as OUTWARD_ROUTE,
            ROUTE_Q8 as OUTWARD_ROUTE_Q8,
            validate_certificate as validate_outward_certificate,
        )

        active_route = OUTWARD_ROUTE
        active_q8 = OUTWARD_ROUTE_Q8
        active_checkpoint_left_q8 = OUTWARD_LEFT_Q8
        active_checkpoint_gripper_m = OUTWARD_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = (
            float(OUTWARD_BASE_POSITION[0]),
            float(OUTWARD_BASE_POSITION[1]),
            float(OUTWARD_BASE_YAW),
        )
        active_checkpoint_book_position = OUTWARD_BOOK_POSITION
        active_checkpoint_book_quaternion = OUTWARD_BOOK_QUATERNION
        active_checkpoint_right_q7 = OUTWARD_RIGHT_Q7
        active_checkpoint_head_q2 = OUTWARD_HEAD_Q2
        active_attached_book_corners = OUTWARD_CORNERS
        active_validate_certificate = validate_outward_certificate
        active_audit = OUTWARD_AUDIT
        active_peel_delta = OUTWARD_WORLD_DELTA_M
        active_peel_duration = max(1.5, OUTWARD_MINIMUM_DURATION_S)
        active_support_floor_world_z = (
            min(float(row[2]) for row in OUTWARD_BOOK_BOUNDS)
            - OUTWARD_FLOOR_SIGNED
        )
        active_endpoint_policy = {
            'min_book_center_outward_progress_m': float(
                OUTWARD_ENDPOINT_POLICY[
                    'minimum_book_center_outward_progress_m'
                ]
            ),
            'min_book_deepest_extent_outward_progress_m': float(
                OUTWARD_ENDPOINT_POLICY[
                    'minimum_deepest_extent_outward_progress_m'
                ]
            ),
            'max_book_deepest_extent_increase_m': float(
                OUTWARD_ENDPOINT_POLICY[
                    'maximum_deepest_extent_inward_motion_m'
                ]
            ),
            'max_book_from_hand_translation_change_m': float(
                OUTWARD_ENDPOINT_POLICY[
                    'maximum_book_from_hand_translation_change_m'
                ]
            ),
            'max_incremental_rotation_rad': float(
                OUTWARD_ENDPOINT_POLICY[
                    'maximum_book_incremental_rotation_rad'
                ]
            ),
            'min_floor_signed_distance_m': float(
                OUTWARD_ENDPOINT_POLICY['floor_signed_min_m']
            ),
            'max_floor_signed_distance_m': float(
                OUTWARD_ENDPOINT_POLICY['floor_signed_max_m']
            ),
        }
        active_certificate_reason = 'pure_outward_certificate'
        active_result_stage = 'strict_pressure_pick_shelf_outward_probe'
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = 0.50
    elif outward_scaleup:
        from erc_phase1_solution.seed101_outward_scaleup_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as SCALE_CORNERS,
            AUDIT as SCALE_AUDIT,
            CHECKPOINT_BASE_WORLD_POSITION_M as SCALE_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as SCALE_BASE_YAW,
            CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M as SCALE_FLOOR_SIGNED,
            CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M as SCALE_BOOK_BOUNDS,
            CHECKPOINT_BOOK_POSITION_WORLD_M as SCALE_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as SCALE_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as SCALE_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as SCALE_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as SCALE_LEFT_Q8,
            CHECKPOINT_RIGHT_Q7 as SCALE_RIGHT_Q7,
            ENDPOINT_POLICY as SCALE_ENDPOINT_POLICY,
            MINIMUM_DURATION_S as SCALE_MINIMUM_DURATION_S,
            ROUTE as SCALE_ROUTE,
            ROUTE_Q8 as SCALE_ROUTE_Q8,
            validate_certificate as validate_scale_certificate,
        )

        active_route = SCALE_ROUTE
        active_q8 = SCALE_ROUTE_Q8
        active_checkpoint_left_q8 = SCALE_LEFT_Q8
        active_checkpoint_gripper_m = SCALE_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = (
            float(SCALE_BASE_POSITION[0]),
            float(SCALE_BASE_POSITION[1]),
            float(SCALE_BASE_YAW),
        )
        active_checkpoint_book_position = SCALE_BOOK_POSITION
        active_checkpoint_book_quaternion = SCALE_BOOK_QUATERNION
        active_checkpoint_right_q7 = SCALE_RIGHT_Q7
        active_checkpoint_head_q2 = SCALE_HEAD_Q2
        active_attached_book_corners = SCALE_CORNERS
        active_validate_certificate = validate_scale_certificate
        active_audit = SCALE_AUDIT
        active_peel_delta = tuple(SCALE_ROUTE[-1]['world_delta_m'])
        active_peel_duration = max(2.5, SCALE_MINIMUM_DURATION_S)
        active_support_floor_world_z = (
            min(float(row[2]) for row in SCALE_BOOK_BOUNDS)
            - SCALE_FLOOR_SIGNED
        )
        active_endpoint_policy = {
            'max_book_deepest_extent_increase_m': 0.00015,
            'min_floor_signed_distance_m': float(
                SCALE_ENDPOINT_POLICY['floor_signed_min_m']
            ),
            'max_floor_signed_distance_m': float(
                SCALE_ENDPOINT_POLICY['floor_signed_max_m']
            ),
        }
        active_certificate_reason = 'outward_scaleup_certificate'
        active_result_stage = 'strict_pressure_pick_outward_scaleup'
        active_pause_every_endpoint = bool(
            SCALE_ENDPOINT_POLICY['pause_after_every_endpoint']
        )
        active_minimum_force_retention_fraction = float(
            SCALE_ENDPOINT_POLICY[
                'minimum_each_side_force_retention_fraction'
            ]
        )
    elif outward_scaleup_continue:
        from erc_phase1_solution.seed101_outward_scaleup_continue_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as CONTINUE_CORNERS,
            AUDIT as CONTINUE_AUDIT,
            CHECKPOINT_BASE_WORLD_POSITION_M as CONTINUE_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as CONTINUE_BASE_YAW,
            CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M as CONTINUE_FLOOR_SIGNED,
            CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M as CONTINUE_BOOK_BOUNDS,
            CHECKPOINT_BOOK_POSITION_WORLD_M as CONTINUE_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as CONTINUE_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as CONTINUE_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as CONTINUE_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as CONTINUE_LEFT_Q8,
            CHECKPOINT_RIGHT_Q7 as CONTINUE_RIGHT_Q7,
            ENDPOINT_POLICY as CONTINUE_ENDPOINT_POLICY,
            MINIMUM_DURATION_S as CONTINUE_MINIMUM_DURATION_S,
            ROUTE as CONTINUE_ROUTE,
            ROUTE_Q8 as CONTINUE_ROUTE_Q8,
            validate_certificate as validate_continue_certificate,
        )

        active_route = CONTINUE_ROUTE
        active_q8 = CONTINUE_ROUTE_Q8
        active_checkpoint_left_q8 = CONTINUE_LEFT_Q8
        active_checkpoint_gripper_m = CONTINUE_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = (
            float(CONTINUE_BASE_POSITION[0]),
            float(CONTINUE_BASE_POSITION[1]),
            float(CONTINUE_BASE_YAW),
        )
        active_checkpoint_book_position = CONTINUE_BOOK_POSITION
        active_checkpoint_book_quaternion = CONTINUE_BOOK_QUATERNION
        active_checkpoint_right_q7 = CONTINUE_RIGHT_Q7
        active_checkpoint_head_q2 = CONTINUE_HEAD_Q2
        active_attached_book_corners = CONTINUE_CORNERS
        active_validate_certificate = validate_continue_certificate
        active_audit = CONTINUE_AUDIT
        active_peel_delta = tuple(CONTINUE_ROUTE[-1]['world_delta_m'])
        active_peel_duration = max(2.5, CONTINUE_MINIMUM_DURATION_S)
        active_support_floor_world_z = (
            min(float(row[2]) for row in CONTINUE_BOOK_BOUNDS)
            - CONTINUE_FLOOR_SIGNED
        )
        active_endpoint_policy = {
            'max_book_deepest_extent_increase_m': 0.00015,
            'min_floor_signed_distance_m': float(
                CONTINUE_ENDPOINT_POLICY['floor_signed_min_m']
            ),
            'max_floor_signed_distance_m': float(
                CONTINUE_ENDPOINT_POLICY['floor_signed_max_m']
            ),
        }
        active_certificate_reason = 'outward_scaleup_continue_certificate'
        active_result_stage = 'strict_pressure_pick_outward_scaleup_continue'
        active_pause_every_endpoint = bool(
            CONTINUE_ENDPOINT_POLICY['pause_after_every_endpoint']
        )
        active_minimum_force_retention_fraction = float(
            CONTINUE_ENDPOINT_POLICY[
                'minimum_each_side_force_retention_fraction'
            ]
        )
    elif reverse_reseat:
        from erc_phase1_solution.seed101_outward_scaleup_continue_certificate import (
            CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M as RESEAT_FLOOR_SIGNED,
            CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M as RESEAT_BOOK_BOUNDS,
        )
        from erc_phase1_solution.seed101_outward_reverse_reseat_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as RESEAT_CORNERS,
            AUDIT as RESEAT_AUDIT,
            CHECKPOINT_BASE_WORLD_POSITION_M as RESEAT_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as RESEAT_BASE_YAW,
            CHECKPOINT_BOOK_POSITION_WORLD_M as RESEAT_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as RESEAT_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as RESEAT_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as RESEAT_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as RESEAT_LEFT_Q8,
            CHECKPOINT_RIGHT_Q7 as RESEAT_RIGHT_Q7,
            ENDPOINT_POLICY as RESEAT_ENDPOINT_POLICY,
            EXECUTION_POLICY as RESEAT_EXECUTION_POLICY,
            MINIMUM_DURATION_S as RESEAT_MINIMUM_DURATION_S,
            MINIMUM_FORCE_N as RESEAT_MINIMUM_FORCE_N,
            PLANNED_WORLD_DELTA_M as RESEAT_WORLD_DELTA_M,
            ROUTE as RESEAT_ROUTE,
            ROUTE_Q8 as RESEAT_ROUTE_Q8,
            STABLE_REFERENCE_BOOK_POSITION_WORLD_M as RESEAT_REFERENCE_POSITION,
            STABLE_REFERENCE_BOOK_QUATERNION_XYZW as RESEAT_REFERENCE_QUATERNION,
            validate_certificate as validate_reseat_certificate,
        )

        active_route = RESEAT_ROUTE
        active_q8 = RESEAT_ROUTE_Q8
        active_checkpoint_left_q8 = RESEAT_LEFT_Q8
        active_checkpoint_gripper_m = RESEAT_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = (
            float(RESEAT_BASE_POSITION[0]),
            float(RESEAT_BASE_POSITION[1]),
            float(RESEAT_BASE_YAW),
        )
        active_checkpoint_book_position = RESEAT_BOOK_POSITION
        active_checkpoint_book_quaternion = RESEAT_BOOK_QUATERNION
        active_checkpoint_right_q7 = RESEAT_RIGHT_Q7
        active_checkpoint_head_q2 = RESEAT_HEAD_Q2
        active_attached_book_corners = RESEAT_CORNERS
        active_validate_certificate = validate_reseat_certificate
        active_audit = RESEAT_AUDIT
        active_peel_delta = tuple(RESEAT_WORLD_DELTA_M)
        active_peel_duration = max(2.5, RESEAT_MINIMUM_DURATION_S)
        active_support_floor_world_z = (
            min(float(row[2]) for row in RESEAT_BOOK_BOUNDS)
            - RESEAT_FLOOR_SIGNED
        )
        active_endpoint_policy = {
            'min_book_center_inward_progress_m': float(
                RESEAT_ENDPOINT_POLICY[
                    'minimum_book_center_cumulative_inward_progress_m'
                ]
            ),
            'min_book_deepest_edge_inward_progress_m': float(
                RESEAT_ENDPOINT_POLICY[
                    'minimum_book_maximum_x_cumulative_inward_return_m'
                ]
            ),
            'max_book_from_hand_translation_change_m': float(
                RESEAT_ENDPOINT_POLICY[
                    'maximum_book_from_hand_translation_change_m'
                ]
            ),
            'max_cross_track_motion_m': float(
                RESEAT_ENDPOINT_POLICY['maximum_cross_track_motion_m']
            ),
            'max_reference_depth_overshoot_m': float(
                RESEAT_ENDPOINT_POLICY['maximum_reference_depth_overshoot_m']
            ),
            'max_stable_reference_position_error_m': float(
                RESEAT_ENDPOINT_POLICY[
                    'maximum_stable_reference_position_error_m'
                ]
            ),
            'max_incremental_rotation_rad': float(
                RESEAT_ENDPOINT_POLICY['maximum_incremental_rotation_rad']
            ),
            'max_absolute_rotation_to_stable_reference_rad': float(
                RESEAT_ENDPOINT_POLICY[
                    'maximum_absolute_rotation_to_stable_reference_rad'
                ]
            ),
            'max_absolute_yaw_to_stable_reference_rad': float(
                RESEAT_ENDPOINT_POLICY[
                    'maximum_absolute_yaw_to_stable_reference_rad'
                ]
            ),
            'max_absolute_rotation_growth_from_start_rad': float(
                RESEAT_ENDPOINT_POLICY[
                    'maximum_absolute_rotation_growth_from_start_rad'
                ]
            ),
            'min_floor_signed_distance_m': float(
                RESEAT_ENDPOINT_POLICY['floor_signed_min_m']
            ),
            'max_floor_signed_distance_m': float(
                RESEAT_ENDPOINT_POLICY['floor_signed_max_m']
            ),
        }
        active_certificate_reason = 'outward_reverse_reseat_certificate'
        active_result_stage = 'strict_pressure_pick_outward_reverse_reseat'
        active_pause_every_endpoint = bool(
            RESEAT_ENDPOINT_POLICY['pause_after_endpoint']
        )
        active_minimum_force_retention_fraction = float(
            RESEAT_ENDPOINT_POLICY[
                'minimum_each_side_force_retention_fraction'
            ]
        )
        active_minimum_force_n = tuple(
            float(value) for value in RESEAT_MINIMUM_FORCE_N
        )
        active_reseat_reference = EntityPose(
            tuple(float(value) for value in RESEAT_REFERENCE_POSITION),
            tuple(float(value) for value in RESEAT_REFERENCE_QUATERNION),
        )
        active_reseat_planned_delta = tuple(
            float(value) for value in RESEAT_WORLD_DELTA_M
        )
        active_continuous_stable_quaternion = tuple(
            float(value) for value in RESEAT_REFERENCE_QUATERNION
        )
        active_continuous_absolute_rotation_limit_rad = float(
            RESEAT_EXECUTION_POLICY[
                'maximum_continuous_absolute_rotation_rad'
            ]
        )
        active_continuous_absolute_yaw_limit_rad = float(
            RESEAT_EXECUTION_POLICY['maximum_continuous_absolute_yaw_rad']
        )
    elif halfmillimeter_outward:
        from erc_phase1_solution.seed101_outward_halfmillimeter_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as HALF_MM_CORNERS,
            AUDIT as HALF_MM_AUDIT,
            CHECKPOINT_BASE_WORLD_POSITION_M as HALF_MM_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as HALF_MM_BASE_YAW,
            CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M as HALF_MM_FLOOR_SIGNED,
            CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M as HALF_MM_BOOK_BOUNDS,
            CHECKPOINT_BOOK_POSITION_WORLD_M as HALF_MM_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as HALF_MM_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as HALF_MM_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as HALF_MM_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as HALF_MM_LEFT_Q8,
            CHECKPOINT_POLICY as HALF_MM_CHECKPOINT_POLICY,
            CHECKPOINT_RIGHT_Q7 as HALF_MM_RIGHT_Q7,
            CONTINUOUS_POLICY as HALF_MM_CONTINUOUS_POLICY,
            ENDPOINT_POLICY as HALF_MM_ENDPOINT_POLICY,
            FINAL_DWELL_POLICY as HALF_MM_FINAL_DWELL_POLICY,
            MINIMUM_DURATION_S as HALF_MM_MINIMUM_DURATION_S,
            PLANNED_WORLD_DELTA_M as HALF_MM_WORLD_DELTA_M,
            PRESSURE_POLICY as HALF_MM_PRESSURE_POLICY,
            ROUTE as HALF_MM_ROUTE,
            ROUTE_Q8 as HALF_MM_ROUTE_Q8,
            STABLE_REFERENCE_BOOK_POSITION_WORLD_M as HALF_MM_REFERENCE_POSITION,
            STABLE_REFERENCE_BOOK_QUATERNION_XYZW as HALF_MM_REFERENCE_QUATERNION,
            validate_certificate as validate_half_mm_certificate,
        )

        active_route = HALF_MM_ROUTE
        active_q8 = HALF_MM_ROUTE_Q8
        active_checkpoint_left_q8 = HALF_MM_LEFT_Q8
        active_checkpoint_gripper_m = HALF_MM_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = (
            float(HALF_MM_BASE_POSITION[0]),
            float(HALF_MM_BASE_POSITION[1]),
            float(HALF_MM_BASE_YAW),
        )
        active_checkpoint_book_position = HALF_MM_BOOK_POSITION
        active_checkpoint_book_quaternion = HALF_MM_BOOK_QUATERNION
        active_checkpoint_right_q7 = HALF_MM_RIGHT_Q7
        active_checkpoint_head_q2 = HALF_MM_HEAD_Q2
        active_attached_book_corners = HALF_MM_CORNERS
        active_validate_certificate = validate_half_mm_certificate
        active_audit = HALF_MM_AUDIT
        active_peel_delta = tuple(HALF_MM_WORLD_DELTA_M)
        active_peel_duration = max(2.5, HALF_MM_MINIMUM_DURATION_S)
        active_support_floor_world_z = (
            min(float(row[2]) for row in HALF_MM_BOOK_BOUNDS)
            - HALF_MM_FLOOR_SIGNED
        )
        active_endpoint_policy = {
            'min_book_center_outward_progress_m': float(
                HALF_MM_ENDPOINT_POLICY[
                    'minimum_book_center_outward_progress_m'
                ]
            ),
            'max_book_center_outward_progress_m': float(
                HALF_MM_ENDPOINT_POLICY[
                    'maximum_book_center_outward_progress_m'
                ]
            ),
            'min_book_deepest_edge_outward_progress_m': float(
                HALF_MM_ENDPOINT_POLICY[
                    'minimum_deepest_extent_outward_progress_m'
                ]
            ),
            'max_book_deepest_edge_outward_progress_m': float(
                HALF_MM_ENDPOINT_POLICY[
                    'maximum_book_center_outward_progress_m'
                ]
            ),
            'max_book_from_hand_translation_change_m': float(
                HALF_MM_ENDPOINT_POLICY[
                    'maximum_book_from_hand_translation_mismatch_m'
                ]
            ),
            'max_cross_track_motion_m': float(
                HALF_MM_ENDPOINT_POLICY['maximum_cross_track_motion_m']
            ),
            'max_lateral_y_motion_m': float(
                HALF_MM_ENDPOINT_POLICY['maximum_absolute_world_y_motion_m']
            ),
            'max_incremental_rotation_rad': float(
                HALF_MM_ENDPOINT_POLICY['maximum_incremental_rotation_rad']
            ),
            'max_incremental_yaw_component_rad': float(
                HALF_MM_ENDPOINT_POLICY['maximum_incremental_yaw_rad']
            ),
            'max_absolute_rotation_to_stable_reference_rad': float(
                HALF_MM_ENDPOINT_POLICY[
                    'maximum_absolute_rotation_to_stable_reference_rad'
                ]
            ),
            'max_absolute_yaw_to_stable_reference_rad': float(
                HALF_MM_ENDPOINT_POLICY[
                    'maximum_absolute_yaw_to_stable_reference_rad'
                ]
            ),
            'min_floor_signed_distance_m': float(
                HALF_MM_ENDPOINT_POLICY['floor_signed_min_m']
            ),
            'max_floor_signed_distance_m': float(
                HALF_MM_ENDPOINT_POLICY['floor_signed_max_m']
            ),
        }
        active_final_endpoint_policy = {
            **active_endpoint_policy,
            'max_incremental_rotation_rad': float(
                HALF_MM_FINAL_DWELL_POLICY[
                    'maximum_relative_rotation_from_start_rad'
                ]
            ),
            'max_incremental_yaw_component_rad': float(
                HALF_MM_FINAL_DWELL_POLICY[
                    'maximum_relative_yaw_from_start_rad'
                ]
            ),
            'max_absolute_rotation_to_stable_reference_rad': float(
                HALF_MM_FINAL_DWELL_POLICY[
                    'maximum_absolute_rotation_to_stable_reference_rad'
                ]
            ),
            'max_absolute_yaw_to_stable_reference_rad': float(
                HALF_MM_FINAL_DWELL_POLICY[
                    'maximum_absolute_yaw_to_stable_reference_rad'
                ]
            ),
        }
        active_certificate_reason = 'outward_halfmillimeter_certificate'
        active_result_stage = 'strict_pressure_pick_outward_halfmillimeter'
        active_pause_every_endpoint = bool(
            HALF_MM_ENDPOINT_POLICY['pause_after_endpoint']
        )
        active_minimum_force_retention_fraction = float(
            HALF_MM_PRESSURE_POLICY[
                'minimum_post_to_pre_force_ratio_each_side'
            ]
        )
        active_maximum_force_balance_change = float(
            HALF_MM_PRESSURE_POLICY[
                'maximum_left_right_balance_change_fraction'
            ]
        )
        emergency_fraction = float(
            HALF_MM_PRESSURE_POLICY['emergency_minimum_fraction']
        )
        active_minimum_force_n = (
            emergency_fraction
            * float(HALF_MM_PRESSURE_POLICY['historical_minimum_left_force_n']),
            emergency_fraction
            * float(HALF_MM_PRESSURE_POLICY['historical_minimum_right_force_n']),
        )
        active_continuous_force_retention_fraction = emergency_fraction
        active_checkpoint_stable_reference = EntityPose(
            tuple(float(value) for value in HALF_MM_REFERENCE_POSITION),
            tuple(float(value) for value in HALF_MM_REFERENCE_QUATERNION),
        )
        active_checkpoint_absolute_rotation_limit_rad = float(
            HALF_MM_CHECKPOINT_POLICY[
                'maximum_absolute_rotation_to_stable_reference_rad'
            ]
        )
        active_checkpoint_absolute_yaw_limit_rad = float(
            HALF_MM_CHECKPOINT_POLICY[
                'maximum_absolute_yaw_to_stable_reference_rad'
            ]
        )
        active_checkpoint_book_position_limit_m = float(
            HALF_MM_CHECKPOINT_POLICY['maximum_book_position_error_m']
        )
        active_checkpoint_book_rotation_limit_rad = float(
            HALF_MM_CHECKPOINT_POLICY['maximum_book_rotation_error_rad']
        )
        active_checkpoint_gripper_limit_m = 0.00015
        active_startup_stability_translation_limit_m = 0.00010
        active_startup_stability_rotation_limit_rad = 0.00020
        active_dense_maximum_increment_rad = 0.0005
        active_continuous_progress_regression_limit_m = abs(float(
            HALF_MM_CONTINUOUS_POLICY[
                'minimum_book_center_outward_progress_m'
            ]
        ))
        active_continuous_progress_overshoot_limit_m = (
            float(HALF_MM_CONTINUOUS_POLICY[
                'maximum_book_center_outward_progress_m'
            ]) - abs(float(HALF_MM_WORLD_DELTA_M[0]))
        )
        active_continuous_cross_track_limit_m = float(
            HALF_MM_CONTINUOUS_POLICY['maximum_cross_track_motion_m']
        )
        active_continuous_stationary_translation_limit_m = float(
            HALF_MM_CONTINUOUS_POLICY['maximum_cross_track_motion_m']
        )
        active_continuous_lateral_y_limit_m = float(
            HALF_MM_CONTINUOUS_POLICY['maximum_absolute_world_y_motion_m']
        )
        active_continuous_relative_rotation_limit_rad = float(
            HALF_MM_CONTINUOUS_POLICY[
                'maximum_relative_rotation_from_start_rad'
            ]
        )
        active_continuous_relative_yaw_limit_rad = float(
            HALF_MM_CONTINUOUS_POLICY[
                'maximum_relative_yaw_from_start_rad'
            ]
        )
        active_continuous_relative_yaw_hard_limit_rad = (
            active_continuous_relative_yaw_limit_rad
        )
        active_continuous_stable_quaternion = tuple(
            float(value) for value in HALF_MM_REFERENCE_QUATERNION
        )
        active_continuous_absolute_rotation_limit_rad = float(
            HALF_MM_CONTINUOUS_POLICY[
                'maximum_absolute_rotation_to_stable_reference_rad'
            ]
        )
        active_continuous_absolute_yaw_limit_rad = float(
            HALF_MM_CONTINUOUS_POLICY[
                'maximum_absolute_yaw_to_stable_reference_rad'
            ]
        )
        active_final_dwell_s = float(
            HALF_MM_FINAL_DWELL_POLICY['duration_s']
        )
        active_final_stability_translation_limit_m = float(
            HALF_MM_FINAL_DWELL_POLICY['maximum_translation_change_m']
        )
        active_final_stability_rotation_limit_rad = float(
            HALF_MM_FINAL_DWELL_POLICY['maximum_rotation_change_rad']
        )
        active_settle_reference = active_checkpoint_stable_reference
        active_settle_rotation_growth_limit_rad = float(
            HALF_MM_FINAL_DWELL_POLICY[
                'maximum_rotation_growth_from_immediate_endpoint_rad'
            ]
        )
        active_pressure_probe_translation_limit_m = float(
            HALF_MM_CONTINUOUS_POLICY['maximum_cross_track_motion_m']
        )
    elif practical_preload_resume:
        from erc_phase1_solution.seed101_outward_halfmillimeter_midroute_settle_certificate import (
            CHECKPOINT_HEAD_Q2 as PRELOAD_HEAD_Q2,
            CHECKPOINT_RIGHT_Q7 as PRELOAD_RIGHT_Q7,
            SUPPORT_FLOOR_WORLD_Z_M as PRELOAD_SUPPORT_FLOOR_WORLD_Z,
        )

        # Exact paused t=429 state after the interrupted preload probe.  This
        # stage owns no arm trajectory: it finishes the original single
        # 0.10 mm probe target (89.856 micrometres remain), validates the
        # resulting pressure/pose, then pauses.
        preload_q8 = (
            0.349999999634771,
            -0.24044475004723945,
            0.6676138876618526,
            0.11156520946627407,
            -1.8338054820345138,
            0.20053325555822715,
            1.3660477999392484,
            0.21500607647999087,
        )
        preload_book_position = (
            2.7340873751806165,
            -0.15308011472986996,
            1.57541429393124,
        )
        preload_book_quaternion = (
            -0.0032278865295478957,
            0.6720979687136833,
            0.0029297069421557595,
            0.7404494027391244,
        )
        active_preload_target_m = 0.028908127502464856
        active_q8 = (preload_q8,)
        active_route = ({
            'row': 0,
            'phase': 'checkpoint',
            'q8': preload_q8,
            'world_delta_m': (0.0, 0.0, 0.0),
            'minimum_duration_s': 0.0,
        },)

        def validate_practical_preload_resume() -> Tuple[bool, Mapping[str, float]]:
            source_ok, _ = validate_certificate()
            decrement = 0.028997983972560686 - active_preload_target_m
            valid = bool(
                source_ok
                and len(active_q8) == 1
                and len(active_q8[0]) == 8
                and abs(decrement - 0.000089856470095830) <= 1e-12
                and active_preload_target_m >= 0.02885
                and tuple(active_route[0]['q8']) == active_q8[0]
                and tuple(active_route[0]['world_delta_m']) == (0.0, 0.0, 0.0)
            )
            return valid, {
                'route_rows': 1.0,
                'gripper_command_count': 1.0,
                'commanded_aperture_decrement_m': decrement,
                'left_arm_trajectory_command_count': 0.0,
            }

        active_checkpoint_left_q8 = preload_q8
        active_checkpoint_gripper_m = 0.028997983972560686
        active_checkpoint_base_xyyaw = (
            2.00272231193355,
            -0.14965265677478828,
            -0.785274375808831,
        )
        active_checkpoint_book_position = preload_book_position
        active_checkpoint_book_quaternion = preload_book_quaternion
        active_checkpoint_right_q7 = tuple(PRELOAD_RIGHT_Q7)
        active_checkpoint_head_q2 = tuple(PRELOAD_HEAD_Q2)
        active_attached_book_corners = (
            (0.0449142770971, 0.0295283307033, -0.113014264402),
            (-0.0870730139995, 0.0295179405334, 0.133925717944),
            (0.0445186628034, -0.0304699834665, -0.113228241301),
            (-0.0874686282931, -0.0304803736364, 0.133711741045),
            (0.212476010691, 0.0281040633578, -0.0234540226783),
            (0.0804887195943, 0.0280936731879, 0.223485959668),
            (0.212080396397, -0.031894250812, -0.0236679995771),
            (0.0800931053007, -0.0319046409819, 0.223271982769),
        )
        active_validate_certificate = validate_practical_preload_resume
        active_audit = {
            'dense_samples': 1,
            'left_arm_trajectory_command_count': 0,
            'gripper_command_count': 1,
        }
        active_peel_delta = (0.0, 0.0, 0.0)
        active_peel_duration = 0.0
        active_support_floor_world_z = float(PRELOAD_SUPPORT_FLOOR_WORLD_Z)
        active_endpoint_policy = {}
        active_certificate_reason = 'practical_single_preload_probe'
        active_result_stage = 'pressure_only_single_preload_probe'
        active_pause_every_endpoint = True
        active_minimum_force_retention_fraction = 0.90
        active_maximum_force_balance_change = 0.15
        active_minimum_force_n = (0.75, 0.75)
        active_continuous_force_retention_fraction = 0.50
        active_continuous_stationary_translation_limit_m = 0.00020
        active_continuous_relative_rotation_limit_rad = 0.0015
        active_continuous_stable_quaternion = preload_book_quaternion
        active_continuous_absolute_rotation_limit_rad = 0.0015
        active_continuous_absolute_yaw_limit_rad = 0.0015
        active_continuous_corner_displacement_limit_m = 0.00025
        active_checkpoint_arm_limit_rad = 0.00025
        active_checkpoint_gripper_limit_m = 0.00015
        active_checkpoint_book_position_limit_m = 0.00050
        active_checkpoint_book_rotation_limit_rad = 0.003
        active_startup_stability_translation_limit_m = 0.00015
        active_startup_stability_rotation_limit_rad = 0.001
        active_pressure_probe_translation_limit_m = 0.00020
    elif practical_force_release_resume:
        from erc_phase1_solution.seed101_outward_halfmillimeter_midroute_settle_certificate import (
            CHECKPOINT_HEAD_Q2 as FORCE_RELEASE_HEAD_Q2,
            CHECKPOINT_RIGHT_Q7 as FORCE_RELEASE_RIGHT_Q7,
            SUPPORT_FLOOR_WORLD_Z_M as FORCE_RELEASE_SUPPORT_FLOOR_WORLD_Z,
        )

        force_release_q8 = (
            (
                0.3499999999979404,
                -0.2404443178314248,
                0.6676159772553971,
                0.11156534976314103,
                -1.8338041761017398,
                0.20053347502378444,
                1.3660486741473885,
                0.21500577855677022,
            ),
            (
                0.3499999999979404,
                -0.2401544781486073,
                0.6690113483469025,
                0.11165993153157976,
                -1.832931294777718,
                0.20067988070258658,
                1.3666321918848536,
                0.21480694441235473,
            ),
        )
        force_release_delta = (
            -0.000306874485863,
            0.000000031962363,
            0.000398921692966,
        )
        force_release_unit = (
            -0.609725295749,
            0.000063505641,
            0.792612805656,
        )
        force_release_book_positions = (
            (2.73413244516856, -0.1530762669598357, 1.5754258360781601),
            (2.733825570682697, -0.1530762359974731, 1.5758247577711257),
        )
        force_release_quaternion = (
            -0.0030427135623240327,
            0.6723965193841996,
            0.0027638695661045334,
            0.7401797238773924,
        )
        active_integrated_preload_target_m = 0.028808125290533
        active_q8 = force_release_q8
        active_route = (
            {
                'row': 0, 'phase': 'checkpoint', 'q8': active_q8[0],
                'world_delta_m': (0.0, 0.0, 0.0),
                'minimum_duration_s': 0.0,
            },
            {
                'row': 1, 'phase': 'diagonal_peel', 'q8': active_q8[1],
                'world_delta_m': force_release_delta,
                'minimum_duration_s': 0.90,
            },
        )

        def validate_practical_force_release() -> Tuple[bool, Mapping[str, float]]:
            source_ok, _ = validate_certificate()
            distance = math.sqrt(sum(value * value for value in force_release_delta))
            unit_norm = math.sqrt(sum(value * value for value in force_release_unit))
            projected = sum(
                value * axis
                for value, axis in zip(force_release_delta, force_release_unit)
            )
            expected_error = _distance(
                tuple(
                    after - before
                    for before, after in zip(*force_release_book_positions)
                ),
                force_release_delta,
            )
            valid = bool(
                source_ok
                and len(active_q8) == 2
                and abs(distance - 0.000503299580979) <= 1e-12
                and abs(unit_norm - 1.0) <= 1e-9
                and abs(projected - distance) <= 1e-9
                and expected_error <= 2e-9
                and active_route[1]['minimum_duration_s'] == 0.90
                and active_integrated_preload_target_m == 0.028808125290533
            )
            return valid, {
                'route_rows': 2.0,
                'gripper_command_count': 1.0,
                'arm_command_count': 1.0,
                'planned_radial_distance_m': projected,
                'expected_book_delta_error_m': expected_error,
            }

        active_checkpoint_left_q8 = active_q8[0]
        active_checkpoint_gripper_m = 0.02890812750196105
        active_checkpoint_base_xyyaw = (
            2.0027194861347604,
            -0.14964961370729221,
            -0.785271844284074,
        )
        active_checkpoint_book_position = force_release_book_positions[0]
        active_checkpoint_book_quaternion = force_release_quaternion
        active_checkpoint_right_q7 = tuple(FORCE_RELEASE_RIGHT_Q7)
        active_checkpoint_head_q2 = tuple(FORCE_RELEASE_HEAD_Q2)
        active_attached_book_corners = (
            (0.04509169000341823, 0.02947705308510616, -0.11301663835918535),
            (0.21257851763758517, 0.02815739294647313, -0.023314792970356338),
            (0.044725388459423476, -0.030521499622881806, -0.11321537468111902),
            (0.21221221609359048, -0.03184115976151483, -0.023513529292290003),
            (-0.08710375610455823, 0.029466545922232856, 0.13381197407849915),
            (0.08038307152960877, 0.028146885783599824, 0.22351381946732815),
            (-0.08747005764855297, -0.03053200678575511, 0.13361323775656547),
            (0.08001676998561402, -0.03185166692438814, 0.2233150831453945),
        )
        active_validate_certificate = validate_practical_force_release
        active_audit = {
            'dense_samples': 5,
            'maximum_dense_joint_increment_rad': 0.000348842773,
            'maximum_tool_mesh_vertex_step_m': 0.000125852686,
            'minimum_self_aabb_clearance_m': 0.004119769615,
            'minimum_15mm_padded_payload_robot_aabb_clearance_m': 0.016716685116,
            'minimum_robot_shelf_triangle_aabb_clearance_m': 0.099292426761,
            'minimum_closed_gripper_shelf_triangle_aabb_clearance_m': 0.083269010202,
            'exact_narrow_phase_collision_count': 0,
        }
        active_peel_delta = force_release_delta
        active_peel_duration = 0.90
        active_support_floor_world_z = float(FORCE_RELEASE_SUPPORT_FLOOR_WORLD_Z)
        active_endpoint_policy = {
            'minimum_projected_progress_m': 0.000325,
            'maximum_projected_progress_m': 0.000625,
            'maximum_tangent_motion_m': 0.00010,
            'maximum_translation_error_m': 0.00018,
            'maximum_rotation_from_start_rad': 0.00075,
            'maximum_corner_mismatch_m': 0.00025,
            'maximum_nominal_rotation_growth_rad': 0.00075,
            'minimum_shelf_face_bottom_signed_m': 0.00025,
            'minimum_deep_edge_signed_m': 0.0070,
            'minimum_shelf_overlap_m': 0.0702,
            'maximum_lateral_motion_m': 0.00010,
        }
        active_certificate_reason = 'practical_integrated_force_release_route'
        active_result_stage = 'practical_integrated_initial_lip_unload'
        active_resume_progress_mode = 'current_ray_release'
        active_radial_unit_world = force_release_unit
        active_pause_every_endpoint = True
        active_minimum_force_retention_fraction = 0.75
        active_maximum_force_balance_change = 0.15
        active_minimum_force_n = (0.75, 0.75)
        active_continuous_force_retention_fraction = 0.30
        active_continuous_progress_regression_limit_m = 0.000125
        active_continuous_progress_overshoot_limit_m = 0.00015
        active_continuous_cross_track_limit_m = 0.00010
        active_continuous_stationary_translation_limit_m = 0.00015
        active_continuous_lateral_y_limit_m = 0.00010
        active_continuous_relative_rotation_limit_rad = 0.00075
        active_continuous_relative_yaw_limit_rad = 0.00075
        active_continuous_relative_yaw_hard_limit_rad = 0.00075
        active_continuous_corner_displacement_limit_m = 0.00025
        active_continuous_stable_quaternion = force_release_quaternion
        active_continuous_absolute_rotation_limit_rad = 0.00075
        active_continuous_absolute_yaw_limit_rad = 0.00075
        active_checkpoint_arm_limit_rad = 0.00050
        active_checkpoint_gripper_limit_m = 0.00010
        active_checkpoint_book_position_limit_m = 0.00050
        active_checkpoint_book_rotation_limit_rad = 0.002
        active_startup_stability_translation_limit_m = 0.00010
        active_startup_stability_rotation_limit_rad = 0.00050
        active_final_dwell_s = 0.25
        active_final_stability_translation_limit_m = 0.000075
        active_final_stability_rotation_limit_rad = 0.00050
        active_pressure_probe_translation_limit_m = 0.00010
        active_dense_maximum_increment_rad = 0.0005
        active_practical_minimum_floor_signed_m = -math.inf
    elif practical_tighten_direct_pull:
        # Exact serialized t=443.948 shelf-supported checkpoint.  The arm has
        # reached the first +0.5 mm unload anchor, but the book is still resting
        # on the shelf.  This stage is intentionally fail-closed until both the
        # single pressure-controlled close target and the post-close retained
        # lift/pull route have independent mechanical audits.  No provisional
        # close decrement or unaudited arm target may be commanded.
        tighten_pull_checkpoint_q8 = (
            0.3499999999526234, -0.24041126471596883,
            0.725251772095361, 0.1072685278430156,
            -1.7094837786185086, 0.21562861418024012,
            1.3933218964031304, 0.2003353325961208,
        )
        tighten_pull_checkpoint_book_position = (
            2.7659433941862375,
            -0.15307368111886094,
            1.5768953138088597,
        )
        tighten_pull_checkpoint_book_quaternion = (
            -0.0026764969443423,
            0.7073903903212613,
            0.0026736680033139625,
            0.7068129339115469,
        )
        tighten_pull_proof_q8 = (
            0.3499999999526234, -0.24033291715122465,
            0.7259671591229399, 0.1072826034236589,
            -1.708530429396505, 0.21563614445837595,
            1.3931066095898266, 0.20020976023365333,
        )
        tighten_pull_close_audit_ready = True
        tighten_pull_route_audit_ready = False
        # The close audit permits at most sixteen 50 um master-joint steps.
        # This value is a hard floor, not an unconditional endpoint: runtime
        # stops at the first fresh balanced pressure window meeting the policy.
        tighten_pull_close_target_m: Optional[float] = 0.02800812526340508
        tighten_pull_route_rows: Tuple[
            Tuple[
                Tuple[float, ...],
                Tuple[float, ...],
                Tuple[float, ...],
                float,
            ],
            ...,
        ] = ()
        tighten_pull_q8 = (tighten_pull_proof_q8,) + tuple(
            tuple(row[0]) for row in tighten_pull_route_rows
        )
        tighten_pull_book_positions = (
            tighten_pull_checkpoint_book_position,
        ) + tuple(tuple(row[1]) for row in tighten_pull_route_rows)
        tighten_pull_book_quaternions = (
            tighten_pull_checkpoint_book_quaternion,
        ) + tuple(tuple(row[2]) for row in tighten_pull_route_rows)
        tighten_pull_durations = (0.0,) + tuple(
            float(row[3]) for row in tighten_pull_route_rows
        )
        active_q8 = tighten_pull_q8
        active_route = tuple(
            {
                'row': index,
                'phase': 'checkpoint' if index == 0 else 'upright_extract',
                'q8': q8,
                'world_delta_m': tuple(
                    after - before
                    for before, after in zip(
                        tighten_pull_book_positions[0],
                        tighten_pull_book_positions[index],
                    )
                ),
                'expected_book_position_world_m': (
                    tighten_pull_book_positions[index]
                ),
                'expected_book_quaternion_xyzw': (
                    tighten_pull_book_quaternions[index]
                ),
                'minimum_duration_s': tighten_pull_durations[index],
            }
            for index, q8 in enumerate(active_q8)
        )
        active_integrated_preload_target_m = tighten_pull_close_target_m
        active_integrated_preload_use_pressure_ladder = True
        active_integrated_preload_policy = {
            'minimum_each_force_n': 3.0,
            'minimum_total_force_n': 6.2,
            'minimum_each_force_gain_n': 0.0,
            'minimum_weaker_stronger_ratio': 0.80,
            'maximum_each_force_n': 8.0,
            'maximum_book_translation_m': 0.00010,
            'maximum_book_rotation_rad': 0.00050,
            'maximum_book_corner_motion_m': 0.00015,
            'maximum_book_lateral_motion_m': 0.000075,
            'maximum_arm_error_rad': 0.00025,
            'maximum_base_translation_m': 0.00025,
            'maximum_base_yaw_error_rad': 0.00025,
            'maximum_aperture_error_m': 0.00003,
            'live_minimum_each_force_n': 2.2,
            'live_minimum_total_force_n': 4.8,
            'live_force_retention_fraction': 0.65,
            'live_maximum_balance_change': 0.25,
            'maximum_launch_delay_s': 0.15,
            'close_step_m': 0.00005,
            'maximum_close_steps': 16.0,
            'step_motion_s': 0.35,
            'minimum_confirmation_s': 0.12,
            'early_balance_check_total_force_n': 3.0,
            'early_minimum_weaker_stronger_ratio': 0.60,
            'minimum_book_floor_signed_m': -0.00015,
            'minimum_shelf_overlap_m': 0.090,
            'minimum_shelf_com_depth_m': 0.0095,
        }

        def validate_practical_tighten_direct_pull(
        ) -> Tuple[bool, Mapping[str, float]]:
            target = active_integrated_preload_target_m
            route_duration = sum(tighten_pull_durations[1:])
            valid = bool(
                tighten_pull_close_audit_ready
                and tighten_pull_route_audit_ready
                and target is not None
                and math.isfinite(float(target))
                and abs(float(target) - 0.02800812526340508) <= 1e-15
                and active_integrated_preload_use_pressure_ladder
                and abs(float(active_integrated_preload_policy[
                    'close_step_m'
                ]) - 0.00005) <= 1e-15
                and int(active_integrated_preload_policy[
                    'maximum_close_steps'
                ]) == 16
                and float(active_integrated_preload_policy[
                    'maximum_each_force_n'
                ]) == 8.0
                and len(active_q8) >= 3
                and len(active_route) == len(active_q8)
                and active_q8[0] == tighten_pull_proof_q8
                and all(len(row) == 8 for row in active_q8)
                and all(
                    str(row['phase']) == 'upright_extract'
                    for row in active_route[1:]
                )
                and all(
                    duration > 0.0
                    for duration in tighten_pull_durations[1:]
                )
                and math.isfinite(route_duration)
                and route_duration >= 9.5
                and int(active_audit['dense_samples']) > 0
                and all(
                    int(active_audit[key]) == 0
                    for key in (
                        'exact_self_collision_count',
                        'exact_payload_robot_collision_count',
                        'exact_robot_shelf_collision_count',
                        'exact_tool_shelf_collision_count',
                        'exact_tool_non_target_collision_count',
                    )
                )
            )
            return valid, {
                'close_audit_ready': float(tighten_pull_close_audit_ready),
                'route_audit_ready': float(tighten_pull_route_audit_ready),
                'close_target_available': float(target is not None),
                'route_rows': float(len(active_q8)),
                'planned_duration_s': float(route_duration),
            }

        active_checkpoint_left_q8 = tighten_pull_checkpoint_q8
        active_pre_pull_proof_q8 = tighten_pull_proof_q8
        active_pre_pull_proof_duration_s = 1.0
        active_pre_pull_proof_minimum_book_rise_m = 0.00015
        active_pre_pull_proof_maximum_attachment_translation_m = 0.00015
        active_pre_pull_proof_maximum_attachment_rotation_rad = 0.001
        active_pre_pull_proof_maximum_attachment_corner_m = 0.00020
        active_checkpoint_gripper_m = 0.028808125263405076
        active_checkpoint_base_xyyaw = (
            2.0027193050832457,
            -0.14965014887002814,
            -0.7852664040221294,
        )
        active_checkpoint_book_position = tighten_pull_checkpoint_book_position
        active_checkpoint_book_quaternion = (
            tighten_pull_checkpoint_book_quaternion
        )
        active_checkpoint_right_q7 = (
            6.780151578542734e-10,
            4.531243610760511e-6,
            6.125259060072998e-10,
            -7.5193484624349175e-6,
            7.70818735051203e-11,
            6.328960313840125e-12,
            -4.231577895613703e-11,
        )
        active_checkpoint_head_q2 = (
            1.7021273850460704e-12,
            -2.0842366268737576e-12,
        )
        active_attached_book_corners: Tuple[Tuple[float, ...], ...] = ()
        active_validate_certificate = validate_practical_tighten_direct_pull
        active_audit = {
            'dense_samples': 0,
            'maximum_dense_joint_increment_rad': math.inf,
            'maximum_tool_mesh_vertex_step_m': math.inf,
            'minimum_self_aabb_clearance_m': -math.inf,
            'minimum_15mm_padded_payload_robot_aabb_clearance_m': -math.inf,
            'minimum_robot_shelf_triangle_aabb_clearance_m': -math.inf,
            'minimum_closed_gripper_shelf_triangle_aabb_clearance_m': -math.inf,
            'exact_self_collision_count': 1,
            'exact_payload_robot_collision_count': 1,
            'exact_robot_shelf_collision_count': 1,
            'exact_tool_shelf_collision_count': 1,
            'exact_tool_non_target_collision_count': 1,
        }
        active_peel_delta = (0.0, 0.0, 0.0)
        active_peel_duration = 13.75
        active_support_floor_world_z = 1.4518584668636323
        active_endpoint_policy = {
            'minimum_planned_lift_m': 0.0010,
            'maximum_planned_lift_m': 0.00150,
            'minimum_planned_outward_m': 0.110,
            'maximum_planned_outward_m': 0.120,
            'maximum_planned_lateral_m': 1e-6,
            'maximum_position_error_m': 0.0010,
            'maximum_lateral_error_m': 0.00050,
            'maximum_vertical_error_m': 0.00050,
            'maximum_rotation_from_start_rad': 0.003,
            'minimum_final_floor_clearance_m': 0.00050,
            'minimum_final_shelf_clearance_m': 0.020,
        }
        active_certificate_reason = 'practical_tighten_direct_pull_route'
        active_result_stage = 'practical_tighten_direct_pull'
        active_resume_progress_mode = 'upright_extract'
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = 0.35
        active_maximum_force_balance_change = 0.50
        active_minimum_force_n = (0.75, 0.75)
        active_minimum_total_force_n = 2.0
        active_maximum_force_n = (8.0, 8.0)
        active_continuous_force_retention_fraction = 0.35
        active_continuous_progress_regression_limit_m = 0.00050
        active_continuous_progress_overshoot_limit_m = 0.0010
        active_continuous_cross_track_limit_m = 0.00075
        active_continuous_stationary_translation_limit_m = 0.00015
        active_continuous_lateral_y_limit_m = 0.00050
        active_continuous_relative_rotation_limit_rad = 0.003
        active_continuous_relative_yaw_limit_rad = 0.002
        active_continuous_relative_yaw_hard_limit_rad = 0.003
        active_continuous_relative_yaw_consecutive_samples = 3
        active_continuous_corner_displacement_limit_m = 0.00075
        active_continuous_stable_quaternion = active_checkpoint_book_quaternion
        active_continuous_absolute_rotation_limit_rad = 0.003
        active_continuous_absolute_yaw_limit_rad = 0.003
        active_checkpoint_arm_limit_rad = 0.00050
        active_checkpoint_gripper_limit_m = 0.00010
        active_checkpoint_book_position_limit_m = 0.00050
        active_checkpoint_book_rotation_limit_rad = 0.002
        active_startup_stability_translation_limit_m = 0.00015
        active_startup_stability_rotation_limit_rad = 0.001
        active_final_dwell_s = 0.15
        active_final_stability_translation_limit_m = 0.00015
        active_final_stability_rotation_limit_rad = 0.00050
        active_pressure_probe_translation_limit_m = 0.00020
        active_dense_maximum_increment_rad = 0.001
        active_practical_minimum_floor_signed_m = -math.inf
        active_continuous_route_goal = True
        active_continuous_route_duration_s = 13.75
        active_continuous_route_use_row_durations = True
        # The separate proof goal consumes the first 0.25 mm of the audited
        # 1.50 mm hand unload; the continuous route begins from that proof.
        active_upright_extract_lift_m = 0.00125
        active_upright_extract_outward_m = 0.115
        active_upright_extract_unload_outward_limit_m = 0.00050
        active_upright_extract_minimum_retained_lift_m = 0.00100
        active_upright_extract_minimum_pull_floor_clearance_m = 0.00075
        active_upright_extract_lift_regression_limit_m = 0.00050
        active_upright_extract_outward_regression_limit_m = 0.00050
        active_upright_extract_clearance_regression_limit_m = 0.00050
        active_upright_extract_attachment_translation_limit_m = 0.00050
        active_upright_extract_attachment_rotation_limit_rad = 0.003
        active_upright_extract_attachment_corner_limit_m = 0.00075
    elif practical_upright_direct_pull:
        # Exact paused t=442.002 shelf-supported upright grasp.  Route rows
        # remain deliberately empty until the fixed-attitude lift/pull IK and
        # dense collision audit are copied in below.  The certificate gate is
        # therefore fail-closed: selecting this stage cannot command motion
        # while ``direct_pull_audit_ready`` is false.
        direct_pull_checkpoint_q8 = (
            0.34999999993561803, -0.24057269673040427,
            0.7237778544522296, 0.10723945055600038,
            -1.7114464525441615, 0.21561304830488387,
            1.3937639299045266, 0.20059394716571077,
        )
        direct_pull_checkpoint_book_position = (
            2.7659402693713537,
            -0.1530734234119139,
            1.5768992473836831,
        )
        direct_pull_checkpoint_book_quaternion = (
            -0.0026783532032646712,
            0.7072857476137667,
            0.0026824938114637827,
            0.7069176061419783,
        )
        # Exact audited semantic anchors.  Times are cumulative; the runtime
        # route below converts them to per-leg durations before densification.
        direct_pull_timed_q8 = (
            (0.0, direct_pull_checkpoint_q8),
            (0.75, (0.34999999993561803, -0.2404160248731255,
                    0.7252083028068279, 0.10726767240668897,
                    -1.7095417397782164, 0.21562815865696125,
                    1.393335005878419, 0.20034296223924505)),
            (1.5, (0.34999999993561803, -0.24025934696715154,
                   0.7266391516623629, 0.10729582763712024,
                   -1.7076345221899139, 0.21564319024584067,
                   1.3929039891293353, 0.20009177120017133)),
            (1.75, (0.34999999993561803, -0.23894718717796892,
                    0.7284239867335335, 0.10781959616650942,
                    -1.7125653229088482, 0.21650859976971681,
                    1.3998442397386601, 0.20016561827605395)),
            (2.0, (0.34999999993561803, -0.23762937652820798,
                   0.7302580832120062, 0.10834816509730023,
                   -1.7174392738687847, 0.2173813677980374,
                   1.406778385156261, 0.2002409048711512)),
            (2.5, (0.34999999993561803, -0.23497766931168754,
                   0.7340740964428887, 0.10942081976276301,
                   -1.7270160737602205, 0.21914851291358953,
                   1.420627798248249, 0.20039542139015457)),
            (3.0, (0.34999999993561803, -0.23230336956606526,
                   0.7380864799429229, 0.11051287078862766,
                   -1.736365012473794, 0.22094494825029717,
                   1.4344514577745695, 0.20055622476497728)),
            (3.5, (0.34999999993561803, -0.2296064576549097,
                   0.7422942438143436, 0.11162442046262687,
                   -1.7454872157089505, 0.2227705354975503,
                   1.4482495929838188, 0.20072395181439623)),
            (4.0, (0.34999999993561803, -0.2268869350657342,
                   0.7466964487744193, 0.11275557974655989,
                   -1.7543834532084712, 0.22462513051837843,
                   1.4620220781576765, 0.2008991642727326)),
            (4.5, (0.34999999993561803, -0.22414481658826466,
                   0.7512921405082869, 0.1139064667336745,
                   -1.7630543839218715, 0.22650857816950576,
                   1.475768663493805, 0.20108239329357894)),
            (5.0, (0.34999999993561803, -0.2213801309537223,
                   0.7560803442849621, 0.11507720631665128,
                   -1.7715005633260947, 0.22842070978685966,
                   1.489488977113443, 0.20127413930818316)),
            (5.5, (0.34999999993561803, -0.21859295148009733,
                   0.7610600482304798, 0.11626792847577293,
                   -1.7797221794200875, 0.23036132737082285,
                   1.5031821223699808, 0.2014748139830197)),
            (6.0, (0.34999999993561803, -0.21578327760611213,
                   0.766230229611057, 0.11747877260605313,
                   -1.7877201687270456, 0.2323302523814708,
                   1.5168483068345904, 0.20168497469160485)),
            (6.5, (0.34999999993561803, -0.21295121185420446,
                   0.7715898269775919, 0.11870988063357603,
                   -1.7954945195253962, 0.23432724918055325,
                   1.530486390022866, 0.2019049658856999)),
            (7.0, (0.34999999993561803, -0.21009684378204796,
                   0.7771377353098012, 0.11996140007705897,
                   -1.803045440416595, 0.23635206983399357,
                   1.5440955309578805, 0.2021351613905006)),
            (7.5, (0.34999999993561803, -0.20722027902264598,
                   0.7828728061619988, 0.12123348260570863,
                   -1.810373068667794, 0.23840444030309774,
                   1.5576747761409713, 0.2023759019587639)),
            (8.0, (0.34999999993561803, -0.2043216394464705,
                   0.7887938438320699, 0.12252628327074418,
                   -1.8174774759021974, 0.24048405794654218,
                   1.5712230615003209, 0.2026274945448673)),
            (8.5, (0.34999999993561803, -0.20140106322334606,
                   0.7948996017852941, 0.12383995965301872,
                   -1.824358673606111, 0.2425905890388361,
                   1.5847392144139605, 0.2028902115206944)),
            (9.0, (0.34999999993561803, -0.19845870477511793,
                   0.8011887793390536, 0.12517467092172477,
                   -1.8310166184519625, 0.24472366631616757,
                   1.598221955808412, 0.20316428984233104)),
            (9.5, (0.34999999993561803, -0.195494734606672,
                   0.8076600186112461, 0.12653057679608215,
                   -1.8374512174487645, 0.2468828865637601,
                   1.6116699023456111, 0.20344993018240803)),
            (10.0, (0.34999999993561803, -0.19250933901900333,
                    0.8143119017370258, 0.12790783642067538,
                    -1.8436623329299477, 0.2490678082509792,
                    1.625081568711744, 0.20374729603428263)),
            (10.5, (0.34999999993561803, -0.18950271968660373,
                    0.8211429483568915, 0.12930660714178766,
                    -1.849649787386113, 0.2512779492350709,
                    1.6384553700164572, 0.20405651280698173)),
            (11.0, (0.34999999993561803, -0.18647509310439836,
                    0.8281516133811319, 0.13072704319833678,
                    -1.855413368149299, 0.2535127845422969,
                    1.651789624313027, 0.2043776669174291)),
            (11.5, (0.34999999993561803, -0.18342668988921093,
                    0.8353362850338032, 0.13216929431918092,
                    -1.8609528319331405, 0.2557717442494499,
                    1.6650825552448858, 0.20471080489870125)),
            (12.0, (0.34999999993561803, -0.1803577539407563,
                    0.8426952831808846, 0.13363350424124387,
                    -1.8662679092325296, 0.25805421147798235,
                    1.678332294825595, 0.20505593253225657)),
            (12.5, (0.34999999993561803, -0.1772685414549797,
                    0.8502268579457716, 0.1351198091499527,
                    -1.871358308584636, 0.2603595205221846,
                    1.6915368863554217, 0.20541301401973217)),
            (13.0, (0.34999999993561803, -0.17415931978549895,
                    0.8579291886148896, 0.13662833604731936,
                    -1.8762237206921286, 0.26268695513274437,
                    1.7046942874763975, 0.2057819712089717)),
        )
        direct_pull_expected_book_poses = (
            (2.7659402693713537, -0.1530734234119139, 1.5768992473836831,
             -0.0026783532032646834, 0.7072857476137667,
             0.0026824938114638707, 0.7069176061419783),
            (2.7659402571728036, -0.15307342192474954, 1.5773992192695987,
             -0.0026783481894821796, 0.7072857453379005,
             0.002682495153337488, 0.7069176084329338),
            (2.765940257074804, -0.15307342191830983, 1.5778992190559926,
             -0.0026783481623425696, 0.707285745328979,
             0.0026824951604203806, 0.706917608441936),
            (2.7634403218550667, -0.15307342754915557, 1.5778993082876898,
             -0.0026783667067502123, 0.7072857267385185,
             0.00268249267431748, 0.7069176269812504),
            (2.7609403206887504, -0.15307342747988087, 1.5778993063738573,
             -0.0026783663650067403, 0.70728572673698,
             0.0026824927099644426, 0.7069176269839493),
            (2.7559402771498647, -0.1530734241620837, 1.5778992569038957,
             -0.002678355467408315, 0.7072857446816435,
             0.0026824935640452617, 0.7069176090679891),
            (2.750940276738498, -0.15307342412873784, 1.577899256180218,
             -0.002678355329315418, 0.7072857446745348,
             0.0026824935833996214, 0.7069176090755512),
            (2.7459402763685508, -0.1530734240989464, 1.5778992555025873,
             -0.002678355203352882, 0.7072857446598088,
             0.0026824936019193397, 0.706917609090692),
            (2.740940276036206, -0.1530734240723782, 1.5778992548666035,
             -0.002678355088336915, 0.7072857446381322,
             0.0026824936196599104, 0.7069176091127479),
            (2.7359402757381948, -0.15307342404875504, 1.5778992542685846,
             -0.0026783549832633934, 0.7072857446100965,
             0.00268249363666096, 0.7069176091411321),
            (2.7309402754715815, -0.15307342402782403, 1.5778992537052343,
             -0.0026783548872277496, 0.7072857445762506,
             0.0026824936529590794, 0.7069176091752976),
            (2.725940347350079, -0.153073430142814, 1.577899314906022,
             -0.0026783716532051646, 0.7072857024965619,
             0.0026824928168123, 0.7069176512165475),
            (2.720940345261427, -0.1530734300474491, 1.5778993098460006,
             -0.0026783709328035794, 0.7072857019987582,
             0.002682492923409347, 0.706917651716935),
            (2.7159403433903178, -0.15307342996653536, 1.5778993050103434,
             -0.002678370263989365, 0.7072857014642391,
             0.002682493025763613, 0.7069176522538781),
            (2.7109403417231914, -0.15307342989912784, 1.577899300386343,
             -0.0026783696431785867, 0.7072857008959753,
             0.0026824931238534078, 0.7069176528244177),
            (2.705940340245213, -0.15307342984402447, 1.577899295959519,
             -0.002678369066269948, 0.7072857002977809,
             0.002682493217790566, 0.7069176534247528),
            (2.700940338942615, -0.15307342980009167, 1.5778992917166514,
             -0.0026783685294763117, 0.7072856996733464,
             0.0026824933076925912, 0.7069176540512052),
            (2.6959403378025937, -0.15307342976625632, 1.5778992876456577,
             -0.002678368029295392, 0.7072856990262469,
             0.002682493393682387, 0.7069176547002103),
            (2.6909403368132017, -0.15307342974150057, 1.577899283735466,
             -0.0026783675624855565, 0.7072856983599507,
             0.0026824934758907233, 0.7069176553683101),
            (2.6859403359632665, -0.15307342972485624, 1.5778992799759142,
             -0.002678367126043654, 0.7072856976778213,
             0.0026824935544582733, 0.7069176560521501),
            (2.6809403352423127, -0.15307342971540025, 1.5778992763576491,
             -0.0026783667171856734, 0.7072856969831199,
             0.002682493629536091, 0.7069176567484773),
            (2.675940334640499, -0.15307342971225127, 1.5778992728720431,
             -0.0026783663333310053, 0.7072856962790064,
             0.0026824937012878748, 0.7069176574541396),
            (2.670940334148561, -0.15307342971456642, 1.5778992695111123,
             -0.002678365972087191, 0.7072856955685356,
             0.0026824937698907048, 0.7069176581660885),
            (2.665940333757766, -0.15307342972153828, 1.5778992662674431,
             -0.0026783656312389915, 0.7072856948546551,
             0.002682493835537139, 0.706917658881383),
            (2.6609403334598687, -0.15307342973239413, 1.5778992631341249,
             -0.00267836530873471, 0.7072856941401989,
             0.002682493898435484, 0.7069176595971945),
            (2.6559403332470826, -0.15307342974639265, 1.5778992601046937,
             -0.0026783650026789337, 0.7072856934278827,
             0.002682493958810998, 0.7069176603108119),
            (2.650940333112049, -0.15307342976282287, 1.5778992571730754,
             -0.0026783647113231813, 0.7072856927202956,
             0.00268249401690736, 0.7069176610196507),
        )
        direct_pull_route_rows = tuple(
            (
                tuple(direct_pull_timed_q8[index][1]),
                tuple(direct_pull_expected_book_poses[index][:3]),
                tuple(direct_pull_expected_book_poses[index][3:]),
                float(
                    direct_pull_timed_q8[index][0]
                    - direct_pull_timed_q8[index - 1][0]
                ),
            )
            for index in range(1, len(direct_pull_timed_q8))
        )
        direct_pull_audit_ready = True
        direct_pull_q8 = (direct_pull_checkpoint_q8,) + tuple(
            tuple(row[0]) for row in direct_pull_route_rows
        )
        direct_pull_book_positions = (
            direct_pull_checkpoint_book_position,
        ) + tuple(tuple(row[1]) for row in direct_pull_route_rows)
        direct_pull_book_quaternions = (
            direct_pull_checkpoint_book_quaternion,
        ) + tuple(tuple(row[2]) for row in direct_pull_route_rows)
        direct_pull_durations = (0.0,) + tuple(
            float(row[3]) for row in direct_pull_route_rows
        )
        active_q8 = direct_pull_q8
        active_route = tuple(
            {
                'row': index,
                'phase': 'checkpoint' if index == 0 else 'upright_extract',
                'q8': q8,
                'world_delta_m': tuple(
                    after - before
                    for before, after in zip(
                        direct_pull_book_positions[0],
                        direct_pull_book_positions[index],
                    )
                ),
                'expected_book_position_world_m': (
                    direct_pull_book_positions[index]
                ),
                'expected_book_quaternion_xyzw': (
                    direct_pull_book_quaternions[index]
                ),
                'minimum_duration_s': direct_pull_durations[index],
            }
            for index, q8 in enumerate(active_q8)
        )

        def validate_practical_upright_direct_pull(
        ) -> Tuple[bool, Mapping[str, float]]:
            final_delta = tuple(active_route[-1]['world_delta_m'])
            maximum_endpoint_increment = max(
                (
                    abs(after - before)
                    for first, second in zip(active_q8, active_q8[1:])
                    for before, after in zip(first, second)
                ),
                default=math.inf,
            )
            total_duration = sum(direct_pull_durations[1:])
            maximum_semantic_speed = max(
                (
                    _distance(first, second) / duration
                    for first, second, duration in zip(
                        direct_pull_book_positions,
                        direct_pull_book_positions[1:],
                        direct_pull_durations[1:],
                    )
                    if duration > 0.0
                ),
                default=math.inf,
            )
            planned_outward = -float(final_delta[0])
            planned_lift = float(final_delta[2])
            expected_terminal_q8 = (
                0.34999999993561803, -0.17415931978549895,
                0.8579291886148896, 0.13662833604731936,
                -1.8762237206921286, 0.26268695513274437,
                1.7046942874763975, 0.2057819712089717,
            )
            expected_times = (
                0.0, 0.75, 1.5, 1.75, 2.0,
                2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
                7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
                11.0, 11.5, 12.0, 12.5, 13.0,
            )
            valid = bool(
                direct_pull_audit_ready
                and len(active_q8) == 27
                and all(len(row) == 8 for row in active_q8)
                and active_q8[0] == direct_pull_checkpoint_q8
                and active_q8[-1] == expected_terminal_q8
                and tuple(row[0] for row in direct_pull_timed_q8)
                == expected_times
                and tuple(tuple(row['q8']) for row in active_route) == active_q8
                and all(
                    str(row['phase']) == 'upright_extract'
                    for row in active_route[1:]
                )
                and all(duration > 0.0 for duration in direct_pull_durations[1:])
                and abs(total_duration - 13.0) <= 1e-12
                and maximum_semantic_speed <= 0.010001
                and abs(float(final_delta[0]) + 0.11499993625930482)
                <= 1e-12
                and abs(float(final_delta[1]) + 6.350908965391255e-09)
                <= 1e-12
                and abs(float(final_delta[2]) - 0.0010000097893922977)
                <= 1e-12
                and maximum_endpoint_increment < 0.03
                and int(active_audit['dense_samples']) == 743
                and abs(
                    float(active_audit['maximum_dense_joint_increment_rad'])
                    - 0.000953608794151
                ) <= 1e-15
                and abs(
                    float(active_audit['maximum_tool_mesh_vertex_step_m'])
                    - 0.000249963336226
                ) <= 1e-15
                and abs(
                    float(active_audit['minimum_self_aabb_clearance_m'])
                    - 0.00189257074840532
                ) <= 1e-15
                and abs(
                    float(active_audit[
                        'minimum_15mm_padded_payload_robot_aabb_clearance_m'
                    ]) - 0.014845574039602738
                ) <= 1e-15
                and abs(
                    float(active_audit[
                        'minimum_robot_shelf_triangle_aabb_clearance_m'
                    ]) - 0.1035437662073111
                ) <= 1e-15
                and abs(
                    float(active_audit[
                        'minimum_closed_gripper_shelf_triangle_aabb_clearance_m'
                    ]) - 0.049754663770916974
                ) <= 1e-15
                and all(
                    int(active_audit[key]) == 0
                    for key in (
                        'exact_self_collision_count',
                        'exact_payload_robot_collision_count',
                        'exact_robot_shelf_collision_count',
                        'exact_tool_shelf_collision_count',
                        'exact_tool_non_target_collision_count',
                    )
                )
            )
            return valid, {
                'route_rows': float(len(active_q8)),
                'command_rows': float(len(active_q8) - 1),
                'maximum_endpoint_joint_increment_rad': maximum_endpoint_increment,
                'planned_vertical_unload_m': planned_lift,
                'planned_outward_pull_m': planned_outward,
                'planned_duration_s': total_duration,
                'maximum_semantic_speed_mps': maximum_semantic_speed,
                'audit_ready': float(bool(direct_pull_audit_ready)),
            }

        active_checkpoint_left_q8 = direct_pull_checkpoint_q8
        active_checkpoint_gripper_m = 0.02880812527215087
        active_checkpoint_base_xyyaw = (
            2.0027192328936767,
            -0.14965017200838537,
            -0.7852672751520583,
        )
        active_checkpoint_book_position = direct_pull_checkpoint_book_position
        active_checkpoint_book_quaternion = (
            direct_pull_checkpoint_book_quaternion
        )
        active_checkpoint_right_q7 = (
            9.924745032858155e-10,
            4.530508096942584e-06,
            9.202162336843538e-10,
            -7.519306446098959e-06,
            8.419283776961269e-11,
            1.99155251979796e-12,
            -4.1162641168189454e-11,
        )
        active_checkpoint_head_q2 = (
            -9.653750839196988e-13,
            -2.0842702525367732e-12,
        )
        # Exact t=442.002 15 mm padded target corners in the measured hand frame.
        active_attached_book_corners = (
            (0.045179252373324694, 0.029416628560084705, -0.11311705609693608),
            (0.21258750268172294, 0.028219402567323557, -0.023266934346307375),
            (0.044847341835047974, -0.03058218022359277, -0.11329810839404802),
            (0.21225559214344664, -0.03177940621635391, -0.023447986643419128),
            (-0.08723414847704707, 0.02940465620220602, 0.13359470108520857),
            (0.08017410183135155, 0.028207430209444873, 0.2234448228358375),
            (-0.0875660590153234, -0.030594152581471457, 0.13341364878809683),
            (0.07984219129307485, -0.031791378574232594, 0.22326377053872554),
        )
        active_validate_certificate = validate_practical_upright_direct_pull
        active_audit = {
            'dense_samples': 743,
            'maximum_dense_joint_increment_rad': 0.000953608794151,
            'maximum_tool_mesh_vertex_step_m': 0.000249963336226,
            'minimum_self_aabb_clearance_m': 0.00189257074840532,
            'minimum_15mm_padded_payload_robot_aabb_clearance_m': (
                0.014845574039602738
            ),
            'minimum_robot_shelf_triangle_aabb_clearance_m': (
                0.1035437662073111
            ),
            'minimum_closed_gripper_shelf_triangle_aabb_clearance_m': (
                0.049754663770916974
            ),
            'exact_self_collision_count': 0,
            'exact_payload_robot_collision_count': 0,
            'exact_robot_shelf_collision_count': 0,
            'exact_tool_shelf_collision_count': 0,
            'exact_tool_non_target_collision_count': 0,
        }
        active_peel_delta = tuple(active_route[-1]['world_delta_m'])
        active_peel_duration = 13.0
        active_support_floor_world_z = 1.4518584668636323
        active_endpoint_policy = {
            'minimum_planned_lift_m': 0.00075,
            'maximum_planned_lift_m': 0.00125,
            'minimum_planned_outward_m': 0.110,
            'maximum_planned_outward_m': 0.120,
            'maximum_planned_lateral_m': 1e-6,
            'maximum_position_error_m': 0.0010,
            'maximum_lateral_error_m': 0.00050,
            'maximum_vertical_error_m': 0.00050,
            'maximum_rotation_from_start_rad': 0.003,
            'minimum_final_floor_clearance_m': 0.00050,
            'minimum_final_shelf_clearance_m': 0.020,
        }
        active_certificate_reason = 'practical_upright_direct_pull_route'
        active_result_stage = 'practical_upright_direct_pull'
        active_resume_progress_mode = 'upright_extract'
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = 0.35
        active_maximum_force_balance_change = 0.50
        active_minimum_force_n = (0.75, 0.75)
        active_minimum_total_force_n = 2.0
        active_maximum_force_n = (8.0, 8.0)
        active_continuous_force_retention_fraction = 0.35
        active_continuous_progress_regression_limit_m = 0.00050
        active_continuous_progress_overshoot_limit_m = 0.0010
        active_continuous_cross_track_limit_m = 0.00075
        active_continuous_stationary_translation_limit_m = 0.00015
        active_continuous_lateral_y_limit_m = 0.00050
        active_continuous_relative_rotation_limit_rad = 0.003
        active_continuous_relative_yaw_limit_rad = 0.002
        active_continuous_relative_yaw_hard_limit_rad = 0.003
        active_continuous_relative_yaw_consecutive_samples = 3
        active_continuous_corner_displacement_limit_m = 0.00075
        active_continuous_stable_quaternion = active_checkpoint_book_quaternion
        active_continuous_absolute_rotation_limit_rad = 0.003
        active_continuous_absolute_yaw_limit_rad = 0.003
        active_checkpoint_arm_limit_rad = 0.00050
        active_checkpoint_gripper_limit_m = 0.00010
        active_checkpoint_book_position_limit_m = 0.00050
        active_checkpoint_book_rotation_limit_rad = 0.002
        active_startup_stability_translation_limit_m = 0.00015
        active_startup_stability_rotation_limit_rad = 0.001
        active_final_dwell_s = 0.15
        active_final_stability_translation_limit_m = 0.00015
        active_final_stability_rotation_limit_rad = 0.00050
        active_pressure_probe_translation_limit_m = 0.00020
        active_dense_maximum_increment_rad = 0.001
        active_practical_minimum_floor_signed_m = -math.inf
        active_continuous_route_goal = True
        active_continuous_route_duration_s = 13.0
        active_continuous_route_use_row_durations = True
        active_upright_extract_lift_m = 0.001
        active_upright_extract_outward_m = 0.115
        active_upright_extract_unload_outward_limit_m = 0.00050
        active_upright_extract_minimum_retained_lift_m = 0.00050
        active_upright_extract_minimum_pull_floor_clearance_m = 0.00075
        active_upright_extract_lift_regression_limit_m = 0.00050
        active_upright_extract_outward_regression_limit_m = 0.00050
        active_upright_extract_clearance_regression_limit_m = 0.00050
        active_upright_extract_attachment_translation_limit_m = 0.00050
        active_upright_extract_attachment_rotation_limit_rad = 0.003
        active_upright_extract_attachment_corner_limit_m = 0.00075
    elif practical_reseat_support:
        # Exact paused t=434.908 fallback after the integrated preload failed
        # only its force-gain gate.  Keep the verified bilateral grasp closed,
        # reverse-roll about the shelf lip until upright, then continue 20 mm
        # inward in the same controller goal so the COM reaches shelf support.
        reseat_roll_angle = 0.095534859256539
        reseat_q8 = (
            (
                0.34999999273665033, -0.24044432248334158,
                0.6676160003475275, 0.11156535331678025,
                -1.833804139360826, 0.20053347307908062,
                1.366048674831002, 0.2150057787933296,
            ),
            (
                0.34999999273665033, -0.23768137484512697,
                0.6856936833403152, 0.11161670951542088,
                -1.8136723542380904, 0.20643987112781226,
                1.387899526249279, 0.211437091770734,
            ),
            (
                0.34999999273665033, -0.23500276973172843,
                0.7036957970344727, 0.11163467853153856,
                -1.7928738708351892, 0.21211066749913693,
                1.4090305668497878, 0.2079606827808901,
            ),
            (
                0.34999999273665033, -0.23240794328367775,
                0.7216239171957798, 0.11161914561042555,
                -1.7714137967308887, 0.21755473148030585,
                1.429447170131566, 0.2045604722092228,
            ),
            (
                0.34999999273665033, -0.22989608351721053,
                0.7394811067465481, 0.11157017677530075,
                -1.749295936726769, 0.2227800381739543,
                1.4491550950484875, 0.20122144960535815,
            ),
            (
                0.34999999273665033, -0.23259945341666574,
                0.7352609388359334, 0.11045845807578938,
                -1.740174081861882, 0.2209440815276427,
                1.4353447650064965, 0.20105433513722096,
            ),
            (
                0.34999999273665033, -0.23528000576339408,
                0.7312363987110466, 0.1093661719440839,
                -1.7308258126184033, 0.21913751374830154,
                1.4215094764027005, 0.200894446417222,
            ),
            (
                0.34999999273665033, -0.23793774731385695,
                0.727408400518999, 0.10829320459943167,
                -1.7212502658159812, 0.21736046228828867,
                1.4076492379184669, 0.20074119131531096,
            ),
            (
                0.34999999273665033, -0.2405726972852116,
                0.7237778540761038, 0.10723945033785719,
                -1.7114464526297708, 0.21561304833127284,
                1.3937639298876034, 0.20059394713888837,
            ),
        )
        reseat_book_positions = (
            (2.734151530462136, -0.15307647193124602, 1.5754305466158172),
            (2.737108556222991, -0.15307647193124602, 1.5758931951292479),
            (2.7400757873477843, -0.15307647193124602, 1.5762850936485047),
            (2.743051531313884, -0.15307647193124602, 1.5766060186328232),
            (2.746034090742892, -0.15307647193124602, 1.5768557870250786),
            (2.751034090742892, -0.15307647193124602, 1.5768557870250786),
            (2.756034090742892, -0.15307647193124602, 1.5768557870250786),
            (2.761034090742892, -0.15307647193124602, 1.5768557870250786),
            (2.766034090742892, -0.15307647193124602, 1.5768557870250786),
        )
        reseat_book_quaternions = (
            (-0.0030265451236343006, 0.6725307672249471,
             0.0027506100956723864, 0.7400578634840252),
            (-0.0029934827086545673, 0.681320269101648,
             0.0027865556798304845, 0.7319740194023107),
            (-0.0029599934042906432, 0.6900126104404667,
             0.0028221038836948073, 0.7237857912406468),
            (-0.002926081986327092, 0.6986065516590828,
             0.0028572496378688225, 0.7154943466916572),
            (-0.002891753290744524, 0.7071008672076501,
             0.002891987930347848, 0.7071008681672689),
        ) + (
            (-0.002891753290744524, 0.7071008672076501,
             0.002891987930347848, 0.7071008681672689),
        ) * 4
        reseat_signed_y_rotations = (
            0.0,
            0.25 * reseat_roll_angle,
            0.50 * reseat_roll_angle,
            0.75 * reseat_roll_angle,
            reseat_roll_angle,
            reseat_roll_angle,
            reseat_roll_angle,
            reseat_roll_angle,
            reseat_roll_angle,
        )
        reseat_phases = (
            'checkpoint',
            'diagonal_peel', 'diagonal_peel', 'diagonal_peel', 'diagonal_peel',
            'shelf_inward', 'shelf_inward', 'shelf_inward', 'shelf_inward',
        )
        reseat_minimum_durations = (
            0.0, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65, 0.65, 0.65,
        )
        active_q8 = reseat_q8
        active_route = tuple(
            {
                'row': index,
                'phase': reseat_phases[index],
                'q8': q8,
                'world_delta_m': tuple(
                    after - before
                    for before, after in zip(
                        reseat_book_positions[0],
                        reseat_book_positions[index],
                    )
                ),
                'expected_book_position_world_m': reseat_book_positions[index],
                'expected_book_quaternion_xyzw': reseat_book_quaternions[index],
                'expected_signed_world_y_rotation_rad': (
                    reseat_signed_y_rotations[index]
                ),
                'minimum_duration_s': reseat_minimum_durations[index],
            }
            for index, q8 in enumerate(active_q8)
        )

        def validate_practical_reseat_support() -> Tuple[bool, Mapping[str, float]]:
            source_ok, _ = validate_certificate()
            final_delta = tuple(active_route[-1]['world_delta_m'])
            maximum_endpoint_increment = max(
                abs(after - before)
                for first, second in zip(active_q8, active_q8[1:])
                for before, after in zip(first, second)
            )
            valid = bool(
                source_ok
                and len(active_q8) == 9
                and all(len(row) == 8 for row in active_q8)
                and tuple(row['phase'] for row in active_route) == reseat_phases
                and tuple(row['minimum_duration_s'] for row in active_route)
                == reseat_minimum_durations
                and tuple(tuple(row['q8']) for row in active_route) == active_q8
                and tuple(
                    tuple(row['expected_book_position_world_m'])
                    for row in active_route
                ) == reseat_book_positions
                and tuple(
                    tuple(row['expected_book_quaternion_xyzw'])
                    for row in active_route
                ) == reseat_book_quaternions
                and abs(final_delta[0] - 0.03188256028075598) <= 1e-12
                and abs(final_delta[1]) <= 1e-12
                and abs(final_delta[2] - 0.0014252404092613968) <= 1e-12
                and abs(reseat_signed_y_rotations[-1] - reseat_roll_angle) <= 1e-15
                and maximum_endpoint_increment < 0.023
            )
            return valid, {
                'route_rows': float(len(active_q8)),
                'command_rows': float(len(active_q8) - 1),
                'maximum_endpoint_joint_increment_rad': maximum_endpoint_increment,
                'planned_reverse_roll_rad': reseat_roll_angle,
                'planned_inward_from_upright_m': 0.020,
                'planned_final_center_depth_m': 0.011018540742892,
                'audited_dense_states': 288.0,
                'maximum_dense_joint_increment_rad': 0.000499071490798,
                'maximum_full_tool_vertex_step_m': 0.000178843992068,
                'minimum_self_aabb_clearance_m': 0.002642156561332,
                'minimum_padded_payload_robot_aabb_clearance_m': 0.016865807503711,
                'minimum_robot_shelf_aabb_clearance_m': 0.099279542878381,
                'minimum_full_gripper_shelf_aabb_clearance_m': 0.049740299773641,
                'exact_narrow_phase_collision_count': 0.0,
            }

        active_checkpoint_left_q8 = active_q8[0]
        active_checkpoint_gripper_m = 0.02880813260941073
        active_checkpoint_base_xyyaw = (
            2.0027195148347796,
            -0.14964958382447013,
            -0.7852714147650421,
        )
        active_checkpoint_book_position = reseat_book_positions[0]
        active_checkpoint_book_quaternion = reseat_book_quaternions[0]
        active_checkpoint_right_q7 = (
            9.4989315444127e-07,
            3.0243413967052386e-06,
            8.046789091558862e-07,
            -7.445534058304117e-06,
            2.5429898449800823e-08,
            1.4969459015825713e-09,
            -4.60363339585441e-10,
        )
        active_checkpoint_head_q2 = (
            5.1484347686901044e-11,
            -2.1052812478162052e-12,
        )
        active_attached_book_corners = (
            (0.0451704973160912, 0.0294744314186955, -0.11301644362666335),
            (0.2126248012128214, 0.028162792676870507, -0.023253780395938787),
            (0.04480654103736347, -0.030524138826618225, -0.11321419181333463),
            (0.21226084493409358, -0.03183577756844324, -0.023451528582609873),
            (-0.0871145405999253, 0.029463523697734063, 0.13376416465644247),
            (0.08033976329680481, 0.028151884955909038, 0.22352682788716724),
            (-0.08747849687865314, -0.030535046547579676, 0.13356641646977138),
            (0.0799758070180771, -0.03184668528940468, 0.22332907970049595),
        )
        active_validate_certificate = validate_practical_reseat_support
        active_audit = {
            'dense_samples': 288,
            'maximum_dense_joint_increment_rad': 0.000499071490798,
            'maximum_tool_mesh_vertex_step_m': 0.000178843992068,
            'minimum_self_aabb_clearance_m': 0.002642156561332,
            'minimum_15mm_padded_payload_robot_aabb_clearance_m': 0.016865807503711,
            'minimum_robot_shelf_triangle_aabb_clearance_m': 0.099279542878381,
            'minimum_closed_gripper_shelf_triangle_aabb_clearance_m': 0.049740299773641,
            'exact_self_collision_count': 0,
            'exact_payload_robot_collision_count': 0,
            'exact_robot_shelf_collision_count': 0,
            'exact_tool_shelf_collision_count': 0,
            'exact_tool_non_target_collision_count': 0,
        }
        active_peel_delta = tuple(active_route[-1]['world_delta_m'])
        active_peel_duration = 5.20
        active_support_floor_world_z = 1.4518584668636323
        active_endpoint_policy = {
            'maximum_center_error_m': 0.0005,
            'maximum_orientation_error_rad': 0.002,
            'maximum_signed_y_rotation_error_rad': 0.006,
            'maximum_cross_axis_rotation_growth_rad': 0.002,
            'minimum_global_bottom_signed_m': -0.0005,
            'maximum_global_bottom_signed_m': 0.00075,
            'minimum_inward_from_upright_m': 0.018,
            'maximum_inward_from_upright_m': 0.022,
            'minimum_center_depth_from_shelf_face_m': 0.008,
            'minimum_shelf_overlap_m': 0.088,
            'maximum_shelf_overlap_m': 0.095,
            'maximum_lateral_motion_m': 0.00030,
        }
        active_certificate_reason = 'practical_reverse_roll_inward_reseat_route'
        active_result_stage = 'practical_shelf_supported_reseat'
        active_resume_progress_mode = 'reseat_support'
        active_reseat_upright_position = reseat_book_positions[4]
        active_reseat_expected_signed_y_rotation_rad = reseat_roll_angle
        active_pause_every_endpoint = True
        active_minimum_force_retention_fraction = 0.35
        active_maximum_force_balance_change = 0.50
        active_minimum_force_n = (0.75, 0.75)
        active_minimum_total_force_n = 2.0
        active_maximum_force_n = (8.0, 8.0)
        active_continuous_force_retention_fraction = 0.35
        active_continuous_progress_regression_limit_m = 0.0010
        active_continuous_progress_overshoot_limit_m = 0.0020
        active_continuous_cross_track_limit_m = 0.0015
        active_continuous_stationary_translation_limit_m = 0.00015
        active_continuous_lateral_y_limit_m = 0.00030
        active_continuous_relative_rotation_limit_rad = 0.105
        active_continuous_relative_yaw_limit_rad = 0.002
        active_continuous_relative_yaw_hard_limit_rad = 0.002
        active_continuous_corner_displacement_limit_m = 0.018
        active_continuous_stable_quaternion = active_checkpoint_book_quaternion
        active_continuous_absolute_rotation_limit_rad = 0.105
        active_continuous_absolute_yaw_limit_rad = 0.002
        active_checkpoint_arm_limit_rad = 0.00050
        active_checkpoint_gripper_limit_m = 0.00010
        active_checkpoint_book_position_limit_m = 0.00050
        active_checkpoint_book_rotation_limit_rad = 0.002
        active_startup_stability_translation_limit_m = 0.00015
        active_startup_stability_rotation_limit_rad = 0.001
        active_final_dwell_s = 0.15
        active_final_stability_translation_limit_m = 0.00015
        active_final_stability_rotation_limit_rad = 0.00050
        active_pressure_probe_translation_limit_m = 0.00020
        active_dense_maximum_increment_rad = 0.0005
        active_practical_minimum_floor_signed_m = -math.inf
        active_continuous_route_goal = True
        active_continuous_route_duration_s = 5.20
    elif practical_preloaded_ray_resume:
        from erc_phase1_solution.seed101_outward_halfmillimeter_midroute_settle_certificate import (
            CHECKPOINT_HEAD_Q2 as PRELOADED_RAY_HEAD_Q2,
            CHECKPOINT_RIGHT_Q7 as PRELOADED_RAY_RIGHT_Q7,
            SUPPORT_FLOOR_WORLD_Z_M as PRELOADED_RAY_SUPPORT_FLOOR_WORLD_Z,
        )

        preloaded_ray_q8 = (
            (
                0.3499999988392275,
                -0.24044475382107156,
                0.6676138893594549,
                0.11156520858818651,
                -1.8338054771041465,
                0.20053325534035754,
                1.3660478000157725,
                0.21500607651893114,
            ),
            (
                0.3499999988392275,
                -0.2401544781486073,
                0.6690113483469025,
                0.11165993153157976,
                -1.832931294777718,
                0.20067988070258658,
                1.3666321918848536,
                0.21480694441235473,
            ),
        )
        preloaded_ray_delta = (
            -0.00030733587746811775,
            0.00000003177841573326584,
            0.0003995190238834656,
        )
        preloaded_ray_unit = (
            -0.6097276499543101,
            0.00006304561284526662,
            0.792610994691876,
        )
        preloaded_ray_book_positions = (
            (2.7341036829738408, -0.15308141397448111, 1.5754179335693757),
            (2.733796347096373, -0.15308138219606537, 1.5758174525932591),
        )
        preloaded_ray_book_quaternion = (
            -0.0032366551102167348,
            0.6722323288829678,
            0.0029382296466011859,
            0.7403273511594872,
        )
        active_q8 = preloaded_ray_q8
        active_route = (
            {
                'row': 0,
                'phase': 'checkpoint',
                'q8': active_q8[0],
                'world_delta_m': (0.0, 0.0, 0.0),
                'expected_book_position_world_m': preloaded_ray_book_positions[0],
                'minimum_duration_s': 0.0,
            },
            {
                'row': 1,
                'phase': 'diagonal_peel',
                'q8': active_q8[1],
                'world_delta_m': preloaded_ray_delta,
                'expected_book_position_world_m': preloaded_ray_book_positions[1],
                'minimum_duration_s': 1.50,
            },
        )

        def validate_practical_preloaded_ray() -> Tuple[bool, Mapping[str, float]]:
            source_ok, _ = validate_certificate()
            distance = math.sqrt(sum(value * value for value in preloaded_ray_delta))
            radial_norm = math.sqrt(sum(value * value for value in preloaded_ray_unit))
            projection = sum(
                value * axis
                for value, axis in zip(preloaded_ray_delta, preloaded_ray_unit)
            )
            expected_error = _distance(
                tuple(
                    after - before
                    for before, after in zip(*preloaded_ray_book_positions)
                ),
                preloaded_ray_delta,
            )
            valid = bool(
                source_ok
                and len(active_q8) == 2
                and all(len(row) == 8 for row in active_q8)
                and abs(distance - 0.0005040543552373719) <= 1e-12
                and abs(radial_norm - 1.0) <= 1e-9
                and abs(projection - distance) <= 1e-9
                and expected_error <= 1e-12
                and active_route[1]['minimum_duration_s'] == 1.50
            )
            return valid, {
                'route_rows': 2.0,
                'command_rows': 1.0,
                'planned_radial_distance_m': projection,
                'planned_tangent_distance_m': math.sqrt(max(
                    0.0,
                    distance * distance - projection * projection,
                )),
                'expected_book_delta_error_m': expected_error,
                'left_gripper_command_count': 0.0,
            }

        active_checkpoint_left_q8 = active_q8[0]
        active_checkpoint_gripper_m = 0.028908125290532976
        active_checkpoint_base_xyyaw = (
            2.0027193332970663,
            -0.14964961732503781,
            -0.785272743923487,
        )
        active_checkpoint_book_position = preloaded_ray_book_positions[0]
        active_checkpoint_book_quaternion = preloaded_ray_book_quaternion
        active_checkpoint_right_q7 = tuple(PRELOADED_RAY_RIGHT_Q7)
        active_checkpoint_head_q2 = tuple(PRELOADED_RAY_HEAD_Q2)
        active_attached_book_corners = (
            (0.04500333550114752, 0.029530847399264, -0.11301141779118576),
            (0.21252912955426528, 0.028111920669402227, -0.023383883105646777),
            (0.04460927087141879, -0.03046747939132119, -0.1132247142931336),
            (0.2121350649245371, -0.03188640612118293, -0.02359717960759458),
            (-0.08708309738740853, 0.02952068819885107, 0.1338755485386868),
            (0.08044269666570975, 0.0281017614689893, 0.22350308322422585),
            (-0.08747716201713678, -0.030477638591734118, 0.133662252036739),
            (0.08004863203598105, -0.031896565321595856, 0.223289786722278),
        )
        active_validate_certificate = validate_practical_preloaded_ray
        active_audit = {
            'dense_samples': 5,
            'maximum_tool_mesh_vertex_step_m': 0.000126037613,
            'maximum_dense_joint_increment_rad': 0.000349364747,
            'minimum_self_aabb_clearance_m': 0.004119755488,
            'minimum_15mm_padded_payload_robot_aabb_clearance_m': (
                0.016628676454
            ),
            'minimum_robot_shelf_triangle_aabb_clearance_m': 0.099292468680,
            'minimum_closed_gripper_shelf_triangle_aabb_clearance_m': (
                0.083276831832
            ),
            'exact_narrow_phase_collision_count': 0,
            'left_gripper_command_count': 0,
            'fixed_gripper_width_m': active_checkpoint_gripper_m,
        }
        active_peel_delta = preloaded_ray_delta
        active_peel_duration = 1.50
        active_support_floor_world_z = float(
            PRELOADED_RAY_SUPPORT_FLOOR_WORLD_Z
        )
        active_endpoint_policy = {
            'minimum_projected_progress_m': 0.000325,
            'maximum_projected_progress_m': 0.000625,
            'maximum_tangent_motion_m': 0.00010,
            'maximum_translation_error_m': 0.00018,
            'maximum_rotation_from_start_rad': 0.00075,
            'maximum_corner_mismatch_m': 0.00025,
            'maximum_nominal_rotation_growth_rad': 0.00075,
            'minimum_shelf_face_bottom_signed_m': 0.00025,
            'minimum_deep_edge_signed_m': 0.0070,
            'minimum_shelf_overlap_m': 0.0702,
            'maximum_lateral_motion_m': 0.00010,
        }
        active_certificate_reason = 'practical_preloaded_current_ray_route'
        active_result_stage = 'pressure_preloaded_initial_lip_unload'
        active_resume_progress_mode = 'current_ray_release'
        active_radial_unit_world = preloaded_ray_unit
        active_pause_every_endpoint = True
        active_minimum_force_retention_fraction = 0.90
        active_maximum_force_balance_change = 0.15
        # Permit a short live contact-force dip while the grasp unloads from
        # the shelf.  The 90% endpoint-retention gate below still requires the
        # settled force to recover to roughly 2.1 N on each jaw.
        active_minimum_force_n = (1.6, 1.6)
        active_continuous_force_retention_fraction = 0.50
        active_continuous_progress_regression_limit_m = 0.000125
        active_continuous_progress_overshoot_limit_m = 0.00015
        active_continuous_cross_track_limit_m = 0.00010
        active_continuous_stationary_translation_limit_m = 0.00025
        active_continuous_lateral_y_limit_m = 0.00010
        active_continuous_relative_rotation_limit_rad = 0.00075
        active_continuous_relative_yaw_limit_rad = 0.00075
        active_continuous_relative_yaw_hard_limit_rad = 0.00075
        active_continuous_corner_displacement_limit_m = 0.00040
        active_continuous_stable_quaternion = preloaded_ray_book_quaternion
        active_continuous_absolute_rotation_limit_rad = 0.00075
        active_continuous_absolute_yaw_limit_rad = 0.00075
        active_checkpoint_arm_limit_rad = 0.00050
        active_checkpoint_gripper_limit_m = 0.00010
        active_checkpoint_book_position_limit_m = 0.00050
        active_checkpoint_book_rotation_limit_rad = 0.002
        active_startup_stability_translation_limit_m = 0.00015
        active_startup_stability_rotation_limit_rad = 0.001
        active_final_dwell_s = 0.25
        active_final_stability_translation_limit_m = 0.000075
        active_final_stability_rotation_limit_rad = 0.0005
        active_pressure_probe_translation_limit_m = 0.00025
        active_dense_maximum_increment_rad = 0.0005
        active_practical_minimum_floor_signed_m = -math.inf
    elif practical_full_pull_resume:
        from erc_phase1_solution.seed101_loaded_extraction_certificate import (
            validate_certificate as validate_practical_resume_source,
        )
        from erc_phase1_solution.seed101_outward_halfmillimeter_midroute_settle_certificate import (
            CHECKPOINT_HEAD_Q2 as PRACTICAL_RESUME_HEAD_Q2,
            CHECKPOINT_RIGHT_Q7 as PRACTICAL_RESUME_RIGHT_Q7,
            PRESSURE_POLICY as PRACTICAL_RESUME_PRESSURE_POLICY,
            SUPPORT_FLOOR_WORLD_Z_M as PRACTICAL_RESUME_SUPPORT_FLOOR_WORLD_Z,
        )

        # The fixed-attitude detach attempt reconfirmed an almost ideal pivot
        # about the shelf lip.  From the exact serialized t=426.394 stop,
        # command only 1 mm along the current lip-to-grasp ray.  This bounded
        # release has negligible torque about the active support point and is
        # accepted only if the payload follows rigidly without another pivot.
        practical_resume_q8 = (
            (
                0.3499999803877476,
                -0.24073025673155485,
                0.6662386472184129,
                0.1114721058928922,
                -1.8346662668796916,
                0.2003892496563243,
                1.3654737474261083,
                0.2152016822311962,
            ),
            (
                0.3499999803877476,
                -0.2401544781486073,
                0.6690113483469025,
                0.11165993153157976,
                -1.832931294777718,
                0.20067988070258658,
                1.3666321918848536,
                0.21480694441235473,
            ),
        )
        practical_resume_radial_unit_world = (
            -0.6090842841361629,
            0.0,
            0.7931055004337935,
        )
        practical_resume_world_deltas = (
            (0.0, 0.0, 0.0),
            (
                -0.0006090842841361629,
                0.0,
                0.0007931055004337935,
            ),
        )
        practical_resume_book_positions = (
            (2.7343636672162033, -0.15307902418999403, 1.5754569665152978),
            (2.733754579437, -0.153079024190, 1.576250069331),
        )
        practical_resume_book_quaternion = (
            -0.0031800709452264736,
            0.6728200000433938,
            0.002891374112927694,
            0.7397937379068101,
        )
        active_q8 = practical_resume_q8
        active_route = tuple(
            {
                'row': index,
                'phase': 'checkpoint' if index == 0 else 'diagonal_peel',
                'q8': q8,
                'world_delta_m': practical_resume_world_deltas[index],
                'expected_book_position_world_m': practical_resume_book_positions[index],
                'expected_book_quaternion_xyzw': practical_resume_book_quaternion,
                'minimum_duration_s': 0.0 if index == 0 else 1.50,
            }
            for index, q8 in enumerate(active_q8)
        )

        def validate_practical_full_pull_resume() -> Tuple[bool, Mapping[str, float]]:
            source_ok, _source_metrics = validate_practical_resume_source()
            maximum_increment = max(
                abs(after - before)
                for first, second in zip(active_q8, active_q8[1:])
                for before, after in zip(first[1:], second[1:])
            )
            torso_span = max(row[0] for row in active_q8) - min(
                row[0] for row in active_q8
            )
            radial_norm = math.sqrt(sum(
                value * value
                for value in practical_resume_radial_unit_world
            ))
            planned = practical_resume_world_deltas[-1]
            planned_radial = sum(
                value * axis
                for value, axis in zip(
                    planned,
                    practical_resume_radial_unit_world,
                )
            )
            expected_delta_error = _distance(
                tuple(
                    after - before
                    for before, after in zip(
                        practical_resume_book_positions[0],
                        practical_resume_book_positions[-1],
                    )
                ),
                planned,
            )
            valid = bool(
                source_ok
                and len(active_q8) == 2
                and all(len(row) == 8 for row in active_q8)
                and torso_span <= 1e-12
                and maximum_increment < 0.003
                and abs(radial_norm - 1.0) <= 1e-12
                and abs(planned_radial - 0.001) <= 1e-12
                and expected_delta_error <= 1e-8
                and tuple(
                    tuple(row['world_delta_m']) for row in active_route
                ) == practical_resume_world_deltas
                and tuple(row['phase'] for row in active_route) == (
                    'checkpoint',
                    'diagonal_peel',
                )
                and tuple(tuple(row['q8']) for row in active_route) == active_q8
                and tuple(row['minimum_duration_s'] for row in active_route)
                == (0.0, 1.50)
                and tuple(
                    tuple(row['expected_book_position_world_m'])
                    for row in active_route
                ) == practical_resume_book_positions
                and tuple(
                    tuple(row['expected_book_quaternion_xyzw'])
                    for row in active_route
                ) == (practical_resume_book_quaternion,) * 2
            )
            return valid, {
                'route_rows': float(len(active_q8)),
                'command_rows': float(len(active_q8) - 1),
                'maximum_endpoint_joint_increment_rad': maximum_increment,
                'planned_radial_distance_m': planned_radial,
                'planned_book_center_outward_distance_m': -planned[0],
                'planned_book_center_lift_m': planned[2],
                'expected_book_delta_error_m': expected_delta_error,
                'audited_dense_states': 7.0,
                'maximum_dense_joint_increment_rad': 0.000693175282,
                'maximum_tool_mesh_vertex_step_m': 0.000249984141,
                'minimum_self_aabb_clearance_m': 0.004110495732,
                'minimum_padded_payload_robot_aabb_clearance_m': (
                    0.016755061480
                ),
                'minimum_arm_gripper_shelf_aabb_clearance_m': (
                    0.099180431119
                ),
                'minimum_full_tool_shelf_aabb_clearance_m': (
                    0.082980442506
                ),
                'exact_narrow_phase_collision_count': 0.0,
            }

        active_checkpoint_left_q8 = active_q8[0]
        active_checkpoint_gripper_m = 0.029008119437545218
        active_checkpoint_base_xyyaw = (
            2.002722170986392,
            -0.14965270565695227,
            -0.785274948452815,
        )
        active_checkpoint_book_position = practical_resume_book_positions[0]
        active_checkpoint_book_quaternion = practical_resume_book_quaternion
        active_checkpoint_right_q7 = tuple(PRACTICAL_RESUME_RIGHT_Q7)
        active_checkpoint_head_q2 = tuple(PRACTICAL_RESUME_HEAD_Q2)
        active_attached_book_corners = (
            (0.045469755019, 0.029510239173, -0.113426875310),
            (0.212853251211, 0.028118716868, -0.023533444693),
            (0.045083722416, -0.030488151618, -0.113636830724),
            (0.212467218609, -0.031879673923, -0.023743400107),
            (-0.087008397126, 0.029499400809, 0.133250118119),
            (0.080375099066, 0.028107878504, 0.223143548735),
            (-0.087394429729, -0.030498989983, 0.133040162705),
            (0.079989066464, -0.031890512288, 0.222933593321),
        )
        active_validate_certificate = validate_practical_full_pull_resume
        active_audit = {
            'dense_samples': 7,
            'maximum_dense_joint_increment_rad': 0.000693175282,
            'maximum_tool_mesh_vertex_step_m': 0.000249984141,
            'minimum_self_aabb_clearance_m': 0.004110495732,
            'minimum_15mm_padded_payload_robot_aabb_clearance_m': (
                0.016755061480
            ),
            'minimum_closed_gripper_shelf_triangle_aabb_clearance_m': (
                0.082980442506
            ),
            'minimum_robot_shelf_triangle_aabb_clearance_m': 0.099180431119,
            'exact_self_collision_count': 0,
            'exact_payload_robot_collision_count': 0,
            'exact_robot_shelf_collision_count': 0,
            'exact_tool_shelf_collision_count': 0,
        }
        active_peel_delta = tuple(active_route[-1]['world_delta_m'])
        active_peel_duration = 1.50
        active_support_floor_world_z = float(
            PRACTICAL_RESUME_SUPPORT_FLOOR_WORLD_Z
        )
        active_endpoint_policy = {
            'minimum_projected_progress_m': 0.00065,
            'maximum_projected_progress_m': 0.00135,
            'maximum_tangent_motion_m': 0.00025,
            'maximum_translation_error_m': 0.00045,
            'maximum_rotation_from_start_rad': 0.002,
            'maximum_nominal_rotation_growth_rad': 0.002,
            'minimum_shelf_face_bottom_signed_m': 0.0005,
            'minimum_deep_edge_signed_m': 0.005,
            'minimum_shelf_overlap_m': 0.065,
            'maximum_lateral_motion_m': 0.0005,
        }
        active_certificate_reason = 'practical_current_ray_release_route'
        active_result_stage = 'pressure_controlled_current_ray_release'
        active_resume_progress_mode = 'current_ray_release'
        active_radial_unit_world = practical_resume_radial_unit_world
        active_pause_every_endpoint = True
        active_minimum_force_retention_fraction = 0.60
        active_maximum_force_balance_change = 0.50
        active_minimum_force_n = (
            0.5 * float(
                PRACTICAL_RESUME_PRESSURE_POLICY[
                    'historical_minimum_left_force_n'
                ]
            ),
            0.5 * float(
                PRACTICAL_RESUME_PRESSURE_POLICY[
                    'historical_minimum_right_force_n'
                ]
            ),
        )
        active_continuous_force_retention_fraction = 0.30
        active_continuous_progress_regression_limit_m = 0.00025
        active_continuous_progress_overshoot_limit_m = 0.00035
        active_continuous_cross_track_limit_m = 0.00025
        active_continuous_stationary_translation_limit_m = 0.00075
        active_continuous_lateral_y_limit_m = 0.0005
        active_continuous_relative_rotation_limit_rad = 0.002
        active_continuous_relative_yaw_limit_rad = 0.0015
        active_continuous_relative_yaw_hard_limit_rad = 0.002
        active_continuous_relative_yaw_consecutive_samples = 3
        active_continuous_corner_displacement_limit_m = 0.00075
        active_continuous_stable_quaternion = active_checkpoint_book_quaternion
        active_continuous_absolute_rotation_limit_rad = 0.002
        active_continuous_absolute_yaw_limit_rad = 0.002
        active_checkpoint_arm_limit_rad = 0.001
        active_checkpoint_gripper_limit_m = 0.001
        active_checkpoint_book_position_limit_m = 0.0015
        active_checkpoint_book_rotation_limit_rad = 0.006
        active_startup_stability_translation_limit_m = 0.001
        active_startup_stability_rotation_limit_rad = 0.003
        active_final_dwell_s = 0.25
        active_final_stability_translation_limit_m = 0.00075
        active_final_stability_rotation_limit_rad = 0.001
        active_pressure_probe_translation_limit_m = 0.00075
        active_dense_maximum_increment_rad = 0.0005
        active_practical_maximum_position_error_m = 0.015
        active_practical_maximum_lateral_error_m = 0.005
        active_practical_maximum_vertical_error_m = 0.015
        active_practical_maximum_rotation_rad = 0.002
        # The rolled outer overhang intentionally descends below this plane;
        # only the current-ray gate's face/deep-edge checks are relevant.
        active_practical_minimum_floor_signed_m = -math.inf
        active_practical_minimum_final_shelf_clearance_m = -0.100
        active_practical_minimum_vertical_progress_m = -math.inf
        active_continuous_route_goal = False
    elif practical_full_pull:
        from erc_phase1_solution.seed101_loaded_extraction_certificate import (
            AUDIT as PRACTICAL_PULL_AUDIT,
            LOADED_RECOVERY_Q8 as PRACTICAL_SOURCE_Q8,
            RECOVERY_ROUTE as PRACTICAL_SOURCE_ROUTE,
            validate_certificate as validate_practical_source_certificate,
        )
        from erc_phase1_solution.seed101_outward_halfmillimeter_midroute_settle_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as PRACTICAL_CORNERS,
            CHECKPOINT_BASE_WORLD_POSITION_M as PRACTICAL_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as PRACTICAL_BASE_YAW,
            CHECKPOINT_BOOK_POSITION_WORLD_M as PRACTICAL_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as PRACTICAL_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as PRACTICAL_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as PRACTICAL_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as PRACTICAL_LEFT_Q8,
            CHECKPOINT_RIGHT_Q7 as PRACTICAL_RIGHT_Q7,
            PRESSURE_POLICY as PRACTICAL_PRESSURE_POLICY,
            SUPPORT_FLOOR_WORLD_Z_M as PRACTICAL_SUPPORT_FLOOR_WORLD_Z,
            validate_certificate as validate_practical_checkpoint_certificate,
        )

        # The current hand is already 2.298 mm outward and 1.500 mm upward
        # from the original full-route checkpoint.  Join the previously
        # collision-audited route at its first 5 mm-out / 5 mm-up row, then
        # follow every remaining 5 mm endpoint through complete shelf clear.
        current_hand_delta_from_source_world = (
            -0.00229765636,
            0.00000137786,
            0.00149961646,
        )
        # dense_arm_waypoints intentionally requires an exactly constant torso
        # coordinate.  The live checkpoint and the baked route differ there by
        # only 0.16 micrometres, so anchor the route bookkeeping to the baked
        # torso value while retaining the measured seven arm joints.  The
        # checkpoint gate below still validates the unmodified live q8.
        practical_route_anchor_q8 = (
            float(PRACTICAL_SOURCE_Q8[6][0]),
            *(float(value) for value in PRACTICAL_LEFT_Q8[1:]),
        )
        practical_route_rows = [{
            'row': 0,
            'phase': 'checkpoint',
            'q8': practical_route_anchor_q8,
            'world_delta_m': (0.0, 0.0, 0.0),
            'minimum_duration_s': 0.0,
        }]
        for source_index in range(6, len(PRACTICAL_SOURCE_ROUTE)):
            source_row = PRACTICAL_SOURCE_ROUTE[source_index]
            source_delta = (
                -float(source_row['outward_world_minus_x_m']),
                0.0,
                float(source_row['lift_world_z_m']),
            )
            practical_route_rows.append({
                'row': len(practical_route_rows),
                'source_row': source_index,
                'phase': (
                    'diagonal_peel'
                    if source_index == 6 else 'shelf_outward'
                ),
                'q8': tuple(
                    float(value) for value in PRACTICAL_SOURCE_Q8[source_index]
                ),
                'world_delta_m': tuple(
                    source - current
                    for source, current in zip(
                        source_delta,
                        current_hand_delta_from_source_world,
                    )
                ),
                'minimum_duration_s': 0.65,
            })
        active_route = tuple(practical_route_rows)
        active_q8 = tuple(tuple(row['q8']) for row in active_route)

        def validate_practical_full_pull() -> Tuple[bool, Mapping[str, float]]:
            source_ok, _source_metrics = validate_practical_source_certificate()
            checkpoint_ok, _checkpoint_metrics = (
                validate_practical_checkpoint_certificate()
            )
            first_increment = max(
                abs(after - before)
                for before, after in zip(active_q8[0], active_q8[1])
            )
            valid = bool(
                source_ok
                and checkpoint_ok
                and len(active_q8) == 50
                and len(active_route) == 50
                and active_q8[0][1:] == tuple(PRACTICAL_LEFT_Q8[1:])
                and abs(
                    active_q8[0][0] - float(PRACTICAL_LEFT_Q8[0])
                ) < 1e-6
                and active_q8[-1] == tuple(PRACTICAL_SOURCE_Q8[-1])
                and first_increment < 0.020
                and float(active_route[-1]['world_delta_m'][0]) < -0.240
                and float(active_route[-1]['world_delta_m'][2]) > 0.003
            )
            return valid, {
                'route_rows': float(len(active_q8)),
                'command_rows': float(len(active_q8) - 1),
                'first_transition_maximum_joint_increment_rad': first_increment,
                'planned_outward_distance_m': float(
                    -active_route[-1]['world_delta_m'][0]
                ),
                'planned_lift_distance_m': float(
                    active_route[-1]['world_delta_m'][2]
                ),
            }

        active_checkpoint_left_q8 = tuple(PRACTICAL_LEFT_Q8)
        active_checkpoint_gripper_m = float(PRACTICAL_GRIPPER_MASTER_M)
        active_checkpoint_base_xyyaw = (
            float(PRACTICAL_BASE_POSITION[0]),
            float(PRACTICAL_BASE_POSITION[1]),
            float(PRACTICAL_BASE_YAW),
        )
        active_checkpoint_book_position = tuple(PRACTICAL_BOOK_POSITION)
        active_checkpoint_book_quaternion = tuple(PRACTICAL_BOOK_QUATERNION)
        active_checkpoint_right_q7 = tuple(PRACTICAL_RIGHT_Q7)
        active_checkpoint_head_q2 = tuple(PRACTICAL_HEAD_Q2)
        active_attached_book_corners = PRACTICAL_CORNERS
        active_validate_certificate = validate_practical_full_pull
        active_audit = PRACTICAL_PULL_AUDIT
        active_peel_delta = tuple(active_route[-1]['world_delta_m'])
        active_peel_duration = 0.65
        active_support_floor_world_z = float(PRACTICAL_SUPPORT_FLOOR_WORLD_Z)
        active_endpoint_policy = {}
        active_certificate_reason = 'practical_full_pull_route'
        active_result_stage = 'pressure_controlled_practical_full_pull'
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = 0.35
        active_maximum_force_balance_change = 0.45
        active_minimum_force_n = (
            0.5 * float(
                PRACTICAL_PRESSURE_POLICY['historical_minimum_left_force_n']
            ),
            0.5 * float(
                PRACTICAL_PRESSURE_POLICY['historical_minimum_right_force_n']
            ),
        )
        active_continuous_force_retention_fraction = 0.35
        active_continuous_progress_regression_limit_m = 0.010
        active_continuous_progress_overshoot_limit_m = 0.015
        active_continuous_cross_track_limit_m = 0.015
        active_continuous_stationary_translation_limit_m = 0.006
        active_continuous_lateral_y_limit_m = 0.010
        active_continuous_relative_rotation_limit_rad = 0.12
        active_continuous_relative_yaw_limit_rad = 0.12
        active_continuous_relative_yaw_hard_limit_rad = 0.12
        active_checkpoint_arm_limit_rad = 0.001
        active_checkpoint_gripper_limit_m = 0.001
        active_checkpoint_book_position_limit_m = 0.004
        active_checkpoint_book_rotation_limit_rad = 0.03
        active_startup_stability_translation_limit_m = 0.003
        active_startup_stability_rotation_limit_rad = 0.03
        active_final_dwell_s = 0.25
        active_final_stability_translation_limit_m = 0.005
        active_final_stability_rotation_limit_rad = 0.05
        active_pressure_probe_translation_limit_m = 0.008
        active_dense_maximum_increment_rad = 0.002
    elif halfmillimeter_midroute_settle:
        from erc_phase1_solution.seed101_outward_halfmillimeter_midroute_settle_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as MIDROUTE_CORNERS,
            AUDIT as MIDROUTE_AUDIT,
            CHECKPOINT_BASE_QUATERNION_XYZW as MIDROUTE_BASE_QUATERNION,
            CHECKPOINT_BASE_WORLD_POSITION_M as MIDROUTE_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as MIDROUTE_BASE_YAW,
            CHECKPOINT as MIDROUTE_CHECKPOINT,
            CHECKPOINT_BOOK_POSITION_WORLD_M as MIDROUTE_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as MIDROUTE_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_LEFT_GEOMETRY_JOINTS as MIDROUTE_LEFT_GRIPPER_GEOMETRY,
            CHECKPOINT_GRIPPER_MASTER_M as MIDROUTE_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as MIDROUTE_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as MIDROUTE_LEFT_Q8,
            CHECKPOINT_POLICY as MIDROUTE_CHECKPOINT_POLICY,
            CHECKPOINT_RIGHT_Q7 as MIDROUTE_RIGHT_Q7,
            CHECKPOINT_RIGHT_GRIPPER_Q8 as MIDROUTE_RIGHT_GRIPPER_Q8,
            CONTINUOUS_POLICY as MIDROUTE_CONTINUOUS_POLICY,
            EXECUTION_POLICY as MIDROUTE_EXECUTION_POLICY,
            FINAL_DWELL_POLICY as MIDROUTE_FINAL_DWELL_POLICY,
            PASSIVE_SCENE_REFERENCE as MIDROUTE_PASSIVE_SCENE_REFERENCE,
            PHYSICS_STEP_SIZE_S as MIDROUTE_PHYSICS_STEP_SIZE_S,
            PRESSURE_POLICY as MIDROUTE_PRESSURE_POLICY,
            RELATIVE_YAW_DEBOUNCE_POLICY as MIDROUTE_YAW_POLICY,
            ROUTE as MIDROUTE_ROUTE,
            ROUTE_Q8 as MIDROUTE_ROUTE_Q8,
            SETTLE_DURATION_S as MIDROUTE_SETTLE_DURATION_S,
            SETTLE_STEP_COUNT as MIDROUTE_SETTLE_STEP_COUNT,
            SETTLE_STABILITY_ROTATION_LIMIT_RAD as MIDROUTE_STABILITY_ROTATION_LIMIT,
            SETTLE_STABILITY_TRANSLATION_LIMIT_M as MIDROUTE_STABILITY_TRANSLATION_LIMIT,
            STABLE_REFERENCE_BOOK_POSITION_WORLD_M as MIDROUTE_REFERENCE_POSITION,
            STABLE_REFERENCE_BOOK_QUATERNION_XYZW as MIDROUTE_REFERENCE_QUATERNION,
            SUPPORT_FLOOR_WORLD_Z_M as MIDROUTE_SUPPORT_FLOOR_WORLD_Z,
            validate_certificate as validate_midroute_certificate,
        )

        active_route = MIDROUTE_ROUTE
        active_q8 = MIDROUTE_ROUTE_Q8
        active_checkpoint_left_q8 = MIDROUTE_LEFT_Q8
        active_checkpoint_gripper_m = MIDROUTE_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = (
            float(MIDROUTE_BASE_POSITION[0]),
            float(MIDROUTE_BASE_POSITION[1]),
            float(MIDROUTE_BASE_YAW),
        )
        active_checkpoint_book_position = MIDROUTE_BOOK_POSITION
        active_checkpoint_book_quaternion = MIDROUTE_BOOK_QUATERNION
        active_checkpoint_right_q7 = MIDROUTE_RIGHT_Q7
        active_checkpoint_head_q2 = MIDROUTE_HEAD_Q2
        active_attached_book_corners = MIDROUTE_CORNERS
        active_validate_certificate = validate_midroute_certificate
        active_audit = MIDROUTE_AUDIT
        active_peel_delta = (0.0, 0.0, 0.0)
        active_peel_duration = 0.0
        active_support_floor_world_z = float(MIDROUTE_SUPPORT_FLOOR_WORLD_Z)
        active_endpoint_policy = {}
        active_supported_book_final_policy = {
            'maximum_rotation_rad': float(
                MIDROUTE_FINAL_DWELL_POLICY[
                    'maximum_absolute_rotation_to_stable_reference_rad'
                ]
            ),
            'maximum_yaw_component_rad': float(
                MIDROUTE_FINAL_DWELL_POLICY[
                    'maximum_absolute_yaw_to_stable_reference_rad'
                ]
            ),
            'minimum_floor_signed_distance_m': float(
                MIDROUTE_FINAL_DWELL_POLICY['floor_signed_min_m']
            ),
            'maximum_floor_signed_distance_m': float(
                MIDROUTE_FINAL_DWELL_POLICY['floor_signed_max_m']
            ),
        }
        active_certificate_reason = (
            'outward_halfmillimeter_midroute_settle_certificate'
        )
        active_result_stage = (
            'strict_pressure_pick_outward_halfmillimeter_midroute_settle'
        )
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = float(
            MIDROUTE_PRESSURE_POLICY[
                'minimum_post_to_pre_force_ratio_each_side'
            ]
        )
        active_maximum_force_balance_change = float(
            MIDROUTE_PRESSURE_POLICY[
                'maximum_left_right_balance_change_fraction'
            ]
        )
        emergency_fraction = float(
            MIDROUTE_PRESSURE_POLICY['emergency_minimum_fraction']
        )
        active_minimum_force_n = (
            emergency_fraction * float(
                MIDROUTE_PRESSURE_POLICY['historical_minimum_left_force_n']
            ),
            emergency_fraction * float(
                MIDROUTE_PRESSURE_POLICY['historical_minimum_right_force_n']
            ),
        )
        active_continuous_force_retention_fraction = emergency_fraction
        active_checkpoint_stable_reference = EntityPose(
            tuple(float(value) for value in MIDROUTE_REFERENCE_POSITION),
            tuple(float(value) for value in MIDROUTE_REFERENCE_QUATERNION),
        )
        active_checkpoint_absolute_rotation_limit_rad = float(
            MIDROUTE_CHECKPOINT_POLICY[
                'maximum_absolute_rotation_to_stable_reference_rad'
            ]
        )
        active_checkpoint_absolute_yaw_limit_rad = float(
            MIDROUTE_CHECKPOINT_POLICY[
                'maximum_absolute_yaw_to_stable_reference_rad'
            ]
        )
        active_checkpoint_book_position_limit_m = float(
            MIDROUTE_CHECKPOINT_POLICY['maximum_book_position_error_m']
        )
        active_checkpoint_book_rotation_limit_rad = float(
            MIDROUTE_CHECKPOINT_POLICY['maximum_book_rotation_error_rad']
        )
        active_checkpoint_arm_limit_rad = 0.00015
        active_checkpoint_gripper_limit_m = 0.00015
        active_startup_stability_translation_limit_m = 0.00010
        active_startup_stability_rotation_limit_rad = 0.00020
        active_continuous_stationary_translation_limit_m = float(
            MIDROUTE_CONTINUOUS_POLICY[
                'maximum_stationary_translation_from_checkpoint_m'
            ]
        )
        active_continuous_relative_rotation_limit_rad = float(
            MIDROUTE_CONTINUOUS_POLICY[
                'maximum_relative_rotation_from_checkpoint_rad'
            ]
        )
        active_continuous_relative_yaw_limit_rad = float(
            MIDROUTE_YAW_POLICY['relative_yaw_soft_limit_rad']
        )
        active_continuous_relative_yaw_hard_limit_rad = float(
            MIDROUTE_YAW_POLICY['relative_yaw_immediate_hard_limit_rad']
        )
        active_continuous_relative_yaw_consecutive_samples = int(
            MIDROUTE_YAW_POLICY['consecutive_samples_required']
        )
        active_continuous_stable_quaternion = tuple(
            float(value) for value in MIDROUTE_REFERENCE_QUATERNION
        )
        active_continuous_absolute_rotation_limit_rad = float(
            MIDROUTE_CONTINUOUS_POLICY[
                'maximum_absolute_rotation_to_stable_reference_rad'
            ]
        )
        active_continuous_absolute_yaw_limit_rad = float(
            MIDROUTE_CONTINUOUS_POLICY[
                'maximum_absolute_yaw_to_stable_reference_rad'
            ]
        )
        active_final_dwell_s = float(MIDROUTE_SETTLE_DURATION_S)
        active_final_stability_translation_limit_m = float(
            MIDROUTE_STABILITY_TRANSLATION_LIMIT
        )
        active_final_stability_rotation_limit_rad = float(
            MIDROUTE_STABILITY_ROTATION_LIMIT
        )
        active_pressure_probe_translation_limit_m = float(
            MIDROUTE_CONTINUOUS_POLICY[
                'maximum_stationary_translation_from_checkpoint_m'
            ]
        )
    elif post_reseat_settle:
        from erc_phase1_solution.seed101_outward_post_reseat_settle_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as POST_RESEAT_CORNERS,
            AUDIT as POST_RESEAT_AUDIT,
            CHECKPOINT_BASE_WORLD_POSITION_M as POST_RESEAT_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as POST_RESEAT_BASE_YAW,
            CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M as POST_RESEAT_FLOOR_SIGNED,
            CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M as POST_RESEAT_BOOK_BOUNDS,
            CHECKPOINT_BOOK_POSITION_WORLD_M as POST_RESEAT_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as POST_RESEAT_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as POST_RESEAT_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as POST_RESEAT_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as POST_RESEAT_LEFT_Q8,
            CHECKPOINT_RIGHT_Q7 as POST_RESEAT_RIGHT_Q7,
            ENDPOINT_POLICY as POST_RESEAT_ENDPOINT_POLICY,
            EXECUTION_POLICY as POST_RESEAT_EXECUTION_POLICY,
            MINIMUM_FORCE_N as POST_RESEAT_MINIMUM_FORCE_N,
            ROUTE as POST_RESEAT_ROUTE,
            ROUTE_Q8 as POST_RESEAT_ROUTE_Q8,
            SETTLE_DURATION_S as POST_RESEAT_DURATION_S,
            SETTLE_ROTATION_GROWTH_LIMIT_RAD as POST_RESEAT_GROWTH_LIMIT,
            SETTLE_STABILITY_ROTATION_LIMIT_RAD as POST_RESEAT_STABILITY_ROTATION_LIMIT,
            SETTLE_STABILITY_TRANSLATION_LIMIT_M as POST_RESEAT_STABILITY_TRANSLATION_LIMIT,
            STARTUP_STABILITY_ROTATION_LIMIT_RAD as POST_RESEAT_STARTUP_ROTATION_LIMIT,
            STARTUP_STABILITY_TRANSLATION_LIMIT_M as POST_RESEAT_STARTUP_TRANSLATION_LIMIT,
            STABLE_REFERENCE_BOOK_POSITION_WORLD_M as POST_RESEAT_REFERENCE_POSITION,
            STABLE_REFERENCE_BOOK_QUATERNION_XYZW as POST_RESEAT_REFERENCE_QUATERNION,
            validate_certificate as validate_post_reseat_certificate,
        )

        active_route = POST_RESEAT_ROUTE
        active_q8 = POST_RESEAT_ROUTE_Q8
        active_checkpoint_left_q8 = POST_RESEAT_LEFT_Q8
        active_checkpoint_gripper_m = POST_RESEAT_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = (
            float(POST_RESEAT_BASE_POSITION[0]),
            float(POST_RESEAT_BASE_POSITION[1]),
            float(POST_RESEAT_BASE_YAW),
        )
        active_checkpoint_book_position = POST_RESEAT_BOOK_POSITION
        active_checkpoint_book_quaternion = POST_RESEAT_BOOK_QUATERNION
        active_checkpoint_right_q7 = POST_RESEAT_RIGHT_Q7
        active_checkpoint_head_q2 = POST_RESEAT_HEAD_Q2
        active_attached_book_corners = POST_RESEAT_CORNERS
        active_validate_certificate = validate_post_reseat_certificate
        active_audit = POST_RESEAT_AUDIT
        active_peel_delta = (0.0, 0.0, 0.0)
        active_peel_duration = 0.0
        active_support_floor_world_z = (
            min(float(row[2]) for row in POST_RESEAT_BOOK_BOUNDS)
            - POST_RESEAT_FLOOR_SIGNED
        )
        active_endpoint_policy = {
            'max_stable_reference_position_error_m': float(
                POST_RESEAT_ENDPOINT_POLICY[
                    'maximum_stable_reference_position_error_m'
                ]
            ),
            'max_stable_reference_maximum_x_error_m': float(
                POST_RESEAT_ENDPOINT_POLICY[
                    'maximum_stable_reference_maximum_x_error_m'
                ]
            ),
            'max_absolute_rotation_to_stable_reference_rad': float(
                POST_RESEAT_ENDPOINT_POLICY[
                    'maximum_absolute_rotation_to_stable_reference_rad'
                ]
            ),
            'max_absolute_yaw_to_stable_reference_rad': float(
                POST_RESEAT_ENDPOINT_POLICY[
                    'maximum_absolute_yaw_to_stable_reference_rad'
                ]
            ),
            'min_floor_signed_distance_m': float(
                POST_RESEAT_ENDPOINT_POLICY['floor_signed_min_m']
            ),
            'max_floor_signed_distance_m': float(
                POST_RESEAT_ENDPOINT_POLICY['floor_signed_max_m']
            ),
        }
        active_certificate_reason = 'outward_post_reseat_settle_certificate'
        active_result_stage = 'strict_pressure_pick_outward_post_reseat_settle'
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = 0.0
        active_final_dwell_s = float(POST_RESEAT_DURATION_S)
        active_minimum_force_n = tuple(
            float(value) for value in POST_RESEAT_MINIMUM_FORCE_N
        )
        active_reseat_reference = EntityPose(
            tuple(float(value) for value in POST_RESEAT_REFERENCE_POSITION),
            tuple(float(value) for value in POST_RESEAT_REFERENCE_QUATERNION),
        )
        active_settle_reference = active_reseat_reference
        active_settle_rotation_growth_limit_rad = float(
            POST_RESEAT_GROWTH_LIMIT
        )
        active_continuous_stable_quaternion = tuple(
            float(value) for value in POST_RESEAT_REFERENCE_QUATERNION
        )
        active_continuous_absolute_rotation_limit_rad = float(
            POST_RESEAT_EXECUTION_POLICY[
                'maximum_continuous_absolute_rotation_rad'
            ]
        )
        active_continuous_absolute_yaw_limit_rad = float(
            POST_RESEAT_EXECUTION_POLICY[
                'maximum_continuous_absolute_yaw_rad'
            ]
        )
        active_checkpoint_arm_limit_rad = 0.00015
        active_checkpoint_gripper_limit_m = 0.00015
        active_checkpoint_book_position_limit_m = 0.00010
        active_checkpoint_book_rotation_limit_rad = 0.00020
        active_final_stability_translation_limit_m = float(
            POST_RESEAT_STABILITY_TRANSLATION_LIMIT
        )
        active_final_stability_rotation_limit_rad = float(
            POST_RESEAT_STABILITY_ROTATION_LIMIT
        )
        active_startup_stability_translation_limit_m = float(
            POST_RESEAT_STARTUP_TRANSLATION_LIMIT
        )
        active_startup_stability_rotation_limit_rad = float(
            POST_RESEAT_STARTUP_ROTATION_LIMIT
        )
        active_pressure_probe_translation_limit_m = 0.00010
    elif settle_resample:
        from erc_phase1_solution.seed101_outward_scaleup_continue_certificate import (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP as SETTLE_CORNERS,
            CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M as SETTLE_FLOOR_SIGNED,
            CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M as SETTLE_BOOK_BOUNDS,
        )
        from erc_phase1_solution.seed101_outward_scaleup_settle_resample_certificate import (
            AUDIT as SETTLE_AUDIT,
            CHECKPOINT_BASE_WORLD_POSITION_M as SETTLE_BASE_POSITION,
            CHECKPOINT_BASE_WORLD_YAW_RAD as SETTLE_BASE_YAW,
            CHECKPOINT_BOOK_POSITION_WORLD_M as SETTLE_BOOK_POSITION,
            CHECKPOINT_BOOK_QUATERNION_XYZW as SETTLE_BOOK_QUATERNION,
            CHECKPOINT_GRIPPER_MASTER_M as SETTLE_GRIPPER_MASTER_M,
            CHECKPOINT_HEAD_Q2 as SETTLE_HEAD_Q2,
            CHECKPOINT_LEFT_Q8 as SETTLE_LEFT_Q8,
            CHECKPOINT_RIGHT_Q7 as SETTLE_RIGHT_Q7,
            ENDPOINT_POLICY as SETTLE_ENDPOINT_POLICY,
            MINIMUM_FORCE_N as SETTLE_MINIMUM_FORCE_N,
            REFERENCE_BOOK_POSITION_WORLD_M as SETTLE_REFERENCE_POSITION,
            REFERENCE_BOOK_QUATERNION_XYZW as SETTLE_REFERENCE_QUATERNION,
            REFERENCE_PLANNED_WORLD_DELTA_M as SETTLE_REFERENCE_DELTA,
            ROUTE as SETTLE_ROUTE,
            ROUTE_Q8 as SETTLE_ROUTE_Q8,
            SETTLE_DURATION_S,
            SETTLE_ROTATION_GROWTH_LIMIT_RAD,
            validate_certificate as validate_settle_certificate,
        )

        active_route = SETTLE_ROUTE
        active_q8 = SETTLE_ROUTE_Q8
        active_checkpoint_left_q8 = SETTLE_LEFT_Q8
        active_checkpoint_gripper_m = SETTLE_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = (
            float(SETTLE_BASE_POSITION[0]),
            float(SETTLE_BASE_POSITION[1]),
            float(SETTLE_BASE_YAW),
        )
        active_checkpoint_book_position = SETTLE_BOOK_POSITION
        active_checkpoint_book_quaternion = SETTLE_BOOK_QUATERNION
        active_checkpoint_right_q7 = SETTLE_RIGHT_Q7
        active_checkpoint_head_q2 = SETTLE_HEAD_Q2
        active_attached_book_corners = SETTLE_CORNERS
        active_validate_certificate = validate_settle_certificate
        active_audit = SETTLE_AUDIT
        active_peel_delta = (0.0, 0.0, 0.0)
        active_peel_duration = 0.0
        active_support_floor_world_z = (
            min(float(row[2]) for row in SETTLE_BOOK_BOUNDS)
            - SETTLE_FLOOR_SIGNED
        )
        active_endpoint_policy = {
            'min_book_center_outward_progress_m': float(
                SETTLE_ENDPOINT_POLICY[
                    'minimum_book_center_cumulative_outward_progress_m'
                ]
            ),
            'min_book_deepest_extent_outward_progress_m': float(
                SETTLE_ENDPOINT_POLICY[
                    'minimum_deepest_extent_cumulative_outward_progress_m'
                ]
            ),
            'max_book_deepest_extent_increase_m': 0.00015,
            'max_book_from_hand_translation_change_m': float(
                SETTLE_ENDPOINT_POLICY[
                    'maximum_book_from_hand_translation_change_m'
                ]
            ),
            'max_incremental_rotation_rad': float(
                SETTLE_ENDPOINT_POLICY['maximum_cumulative_rotation_rad']
            ),
            'max_yaw_component_rad': float(
                SETTLE_ENDPOINT_POLICY[
                    'maximum_cumulative_yaw_component_rad'
                ]
            ),
            'min_floor_signed_distance_m': float(
                SETTLE_ENDPOINT_POLICY['floor_signed_min_m']
            ),
            'max_floor_signed_distance_m': float(
                SETTLE_ENDPOINT_POLICY['floor_signed_max_m']
            ),
        }
        active_certificate_reason = 'outward_settle_resample_certificate'
        active_result_stage = 'strict_pressure_pick_outward_settle_resample'
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = 0.0
        active_final_dwell_s = float(SETTLE_DURATION_S)
        active_minimum_force_n = tuple(
            float(value) for value in SETTLE_MINIMUM_FORCE_N
        )
        active_settle_reference = EntityPose(
            tuple(float(value) for value in SETTLE_REFERENCE_POSITION),
            tuple(float(value) for value in SETTLE_REFERENCE_QUATERNION),
        )
        active_settle_planned_delta = tuple(
            float(value) for value in SETTLE_REFERENCE_DELTA
        )
        active_settle_rotation_growth_limit_rad = float(
            SETTLE_ROTATION_GROWTH_LIMIT_RAD
        )
    else:
        active_route = RECOVERY_ROUTE
        active_q8 = LOADED_RECOVERY_Q8
        active_checkpoint_left_q8 = CHECKPOINT_LEFT_Q8
        active_checkpoint_gripper_m = CHECKPOINT_GRIPPER_MASTER_M
        active_checkpoint_base_xyyaw = CHECKPOINT_BASE_WORLD_XYYAW
        active_checkpoint_book_position = CHECKPOINT_BOOK_POSITION_WORLD_M
        active_checkpoint_book_quaternion = CHECKPOINT_BOOK_QUATERNION_XYZW
        active_checkpoint_right_q7 = CHECKPOINT_RIGHT_Q7
        active_checkpoint_head_q2 = CHECKPOINT_HEAD_Q2
        active_attached_book_corners = (
            ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP
        )
        active_validate_certificate = validate_certificate
        active_audit = AUDIT
        active_peel_delta = (0.0, 0.0, 0.0)
        active_peel_duration = 0.0
        active_support_floor_world_z = ROUTE_SUPPORT_FLOOR_WORLD_Z_M
        active_endpoint_policy = {}
        active_certificate_reason = 'loaded_extraction_certificate'
        active_result_stage = 'strict_pressure_pick_shelf_clear_extraction'
        active_pause_every_endpoint = False
        active_minimum_force_retention_fraction = 0.0

    def active_checkpoint_gate(
        arm_q8: Sequence[float],
        gripper_m: float,
        base: Pose2,
        book: EntityPose,
    ) -> GateResult:
        checkpoint = resume_checkpoint_gate(
            arm_q8,
            gripper_m,
            base,
            book,
            expected_arm_q8=active_checkpoint_left_q8,
            expected_gripper_m=active_checkpoint_gripper_m,
            expected_base_xyyaw=active_checkpoint_base_xyyaw,
            expected_book_position=active_checkpoint_book_position,
            expected_book_quaternion=active_checkpoint_book_quaternion,
            arm_limit_rad=active_checkpoint_arm_limit_rad,
            gripper_limit_m=active_checkpoint_gripper_limit_m,
            book_position_limit_m=active_checkpoint_book_position_limit_m,
            book_rotation_limit_rad=active_checkpoint_book_rotation_limit_rad,
        )
        if not checkpoint.ok or active_checkpoint_stable_reference is None:
            return checkpoint
        orientation = stable_reference_orientation_gate(
            book,
            active_checkpoint_stable_reference,
            maximum_rotation_rad=(
                active_checkpoint_absolute_rotation_limit_rad
            ),
            maximum_yaw_component_rad=(
                active_checkpoint_absolute_yaw_limit_rad
            ),
        )
        return GateResult(
            orientation.ok,
            (
                'resume_checkpoint_and_stable_orientation_verified'
                if orientation.ok else orientation.reason
            ),
            {**checkpoint.metrics, **orientation.metrics},
        )

    def route_world_delta(row: Mapping[str, object]) -> Tuple[float, float, float]:
        if 'world_delta_m' in row:
            values = tuple(float(value) for value in row['world_delta_m'])
            if len(values) != 3:
                raise RuntimeError('certified route world delta is malformed')
            return values
        return (
            -float(row.get('outward_world_minus_x_m', 0.0)),
            0.0,
            float(row.get('lift_world_z_m', 0.0)),
        )

    def progress_policy(row: Mapping[str, object]) -> Mapping[str, object]:
        policy = dict(active_endpoint_policy)
        row_policy = row.get('progress_policy')
        if isinstance(row_policy, Mapping):
            policy.update({
                str(key): float(value)
                for key, value in row_policy.items()
            })
        row_mappings = {
            'minimum_center_outward_progress_m': (
                'min_book_center_outward_progress_m'
            ),
            'minimum_book_center_cumulative_outward_progress_m': (
                'min_book_center_outward_progress_m'
            ),
            'minimum_deepest_extent_outward_progress_m': (
                'min_book_deepest_extent_outward_progress_m'
            ),
            'minimum_deepest_extent_cumulative_outward_progress_m': (
                'min_book_deepest_extent_outward_progress_m'
            ),
            'maximum_cumulative_translation_change_m': (
                'max_book_from_hand_translation_change_m'
            ),
            'maximum_cumulative_rotation_rad': (
                'max_incremental_rotation_rad'
            ),
            'minimum_book_center_cumulative_inward_progress_m': (
                'min_book_center_inward_progress_m'
            ),
            'minimum_book_maximum_x_cumulative_inward_return_m': (
                'min_book_deepest_edge_inward_progress_m'
            ),
            'maximum_absolute_rotation_to_stable_reference_rad': (
                'max_absolute_rotation_to_stable_reference_rad'
            ),
        }
        for source, destination in row_mappings.items():
            if source in row:
                policy[destination] = float(row[source])
        return policy

    def reseat_support_progress(
        observed: EntityPose,
        row: Mapping[str, object],
    ) -> GateResult:
        if active_reseat_upright_position is None:
            return GateResult(
                False,
                'practical_reseat_upright_reference_unavailable',
                {},
            )
        return practical_reseat_support_progress_gate(
            start_book,
            observed,
            upright_position_world_m=active_reseat_upright_position,
            expected_position_world_m=row['expected_book_position_world_m'],
            expected_quaternion_xyzw=row['expected_book_quaternion_xyzw'],
            expected_signed_world_y_rotation_rad=float(
                row['expected_signed_world_y_rotation_rad']
            ),
            support_floor_world_z_m=active_support_floor_world_z,
            policy=progress_policy(row),
        )

    import numpy as np
    import rclpy
    from geometry_msgs.msg import Twist
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.double_v_pb2 import Double_V
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.serialized_map_pb2 import SerializedStepMap
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.msgs10.world_stats_pb2 import WorldStatistics
    from gz.transport13 import Node as GzTransportNode
    from google.protobuf import text_format
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from ament_index_python.packages import (
        get_package_prefix,
        get_package_share_directory,
    )
    from erc_phase1_solution.common import decode_event
    from erc_phase1_solution.manipulation_node import HEAD_JOINTS, ManipulationNode
    from erc_phase1_solution.motion_profiles import (
        ARM_JOINTS,
        IK_JOINTS,
        RIGHT_ARM_JOINTS,
    )
    from erc_phase1_solution.live_rigid_palm_support_probe import (
        quaternion_matrix,
        rotation_matrix_distance,
        world_hand_pose,
    )

    class StrictHeldNode(ManipulationNode):
        """Pressure-retained left-arm node with no generic motion path."""

        def __init__(self) -> None:
            self._strict_joint_state_wall_times = {}
            self._strict_joint_state_generations = {}
            self._strict_joint_state_counter = 0
            self._strict_entity_lock = threading.Lock()
            self._strict_entity_poses = {}
            self._strict_entity_wall_ns = {}
            self._strict_entity_generations = {}
            self._strict_world_stats_lock = threading.Lock()
            self._strict_world_sim_time_ns = -1
            self._strict_world_iterations = -1
            self._strict_world_paused = False
            self._strict_world_stats_wall_ns = 0
            self._strict_hazard_lock = threading.Lock()
            self._strict_last_payload_hazard = {}
            self._strict_payload_target_reference: Optional[EntityPose] = None
            self._strict_payload_motion_phase = 'stationary'
            self._strict_payload_planned_delta = (0.0, 0.0, 0.0)
            self._strict_payload_rotation_limit_rad = (
                PAYLOAD_CONTINUOUS_ROTATION_LIMIT_RAD
            )
            self._strict_payload_minimum_force_n = (0.0, 0.0)
            self._strict_payload_minimum_total_force_n = 0.0
            self._strict_payload_maximum_force_n = (math.inf, math.inf)
            self._strict_payload_live_reference_force_n = (0.0, 0.0)
            self._strict_payload_live_retention_fraction = 0.0
            self._strict_payload_maximum_balance_change = 1.0
            self._strict_payload_stable_quaternion = None
            self._strict_payload_absolute_rotation_limit_rad = math.inf
            self._strict_payload_absolute_yaw_limit_rad = math.inf
            self._strict_payload_progress_regression_limit_m = (
                PAYLOAD_INWARD_REGRESSION_LIMIT_M
            )
            self._strict_payload_progress_overshoot_limit_m = (
                PAYLOAD_SHELF_CONTINUOUS_TRANSLATION_LIMIT_M
            )
            self._strict_payload_cross_track_limit_m = (
                PAYLOAD_SHELF_CONTINUOUS_TRANSLATION_LIMIT_M
            )
            self._strict_payload_stationary_translation_limit_m = (
                PAYLOAD_CONTINUOUS_TRANSLATION_LIMIT_M
            )
            self._strict_payload_lateral_y_limit_m = math.inf
            self._strict_payload_relative_yaw_limit_rad = math.inf
            self._strict_payload_relative_yaw_hard_limit_rad = math.inf
            self._strict_payload_relative_yaw_consecutive_samples_required = 1
            self._strict_payload_relative_yaw_consecutive_samples = 0
            self._strict_payload_relative_yaw_last_generation = -1
            self._strict_payload_max_relative_yaw_rad = 0.0
            self._strict_payload_max_consecutive_soft_yaw_samples = 0
            self._strict_reseat_last_generation = -1
            self._strict_reseat_last_signed_y_rad = 0.0
            self._strict_reseat_last_deep_edge_signed_m = math.inf
            self._strict_upright_extract_hand_from_book = None
            self._strict_upright_extract_last_generation = -1
            self._strict_upright_extract_maximum_lift_m = 0.0
            self._strict_upright_extract_maximum_outward_m = 0.0
            self._strict_upright_extract_maximum_clearance_m = -math.inf
            self._strict_payload_pose_probe_translation_limit_m = (
                PAYLOAD_CONTINUOUS_TRANSLATION_LIMIT_M
            )
            self._strict_payload_base_reference: Optional[Pose2] = None
            self._strict_payload_gripper_reference: Optional[float] = None
            super().__init__()
            self._gz_pose_node = GzTransportNode()
            self._gz_pose_node.subscribe(
                Pose_V,
                DYNAMIC_POSE_TOPIC,
                self._on_gz_dynamic_pose,
            )
            self._gz_pose_node.subscribe(
                WorldStatistics,
                WORLD_STATS_TOPIC,
                self._on_gz_world_stats,
            )

        def _on_joint_state(self, message: object) -> None:
            super()._on_joint_state(message)
            observed_at = time.monotonic()
            names = tuple(str(name) for name in getattr(message, 'name', ()))
            with self._lock:
                self._strict_joint_state_counter += 1
                generation = self._strict_joint_state_counter
                for name in names:
                    self._strict_joint_state_wall_times[name] = observed_at
                    self._strict_joint_state_generations[name] = generation

        def _on_command(self, message: object) -> None:
            payload = decode_event(message.data)
            command = str(payload.get('event', message.data)).strip().lower()
            if command in ('cancel', 'abort', 'stop'):
                super()._on_command(message)
                return
            self._publish_status(
                'rejected',
                command=command,
                reason='strict_held_extraction_owns_actuators',
            )

        def _on_gz_dynamic_pose(self, message: object) -> None:
            observed_at = time.monotonic_ns()
            updates = {}
            for pose in getattr(message, 'pose', ()):
                name = str(getattr(pose, 'name', ''))
                if name != 'tiago_pro' and not name.startswith('book_'):
                    continue
                position = getattr(pose, 'position', None)
                orientation = getattr(pose, 'orientation', None)
                values = (
                    float(getattr(position, 'x', math.nan)),
                    float(getattr(position, 'y', math.nan)),
                    float(getattr(position, 'z', math.nan)),
                    float(getattr(orientation, 'x', math.nan)),
                    float(getattr(orientation, 'y', math.nan)),
                    float(getattr(orientation, 'z', math.nan)),
                    float(getattr(orientation, 'w', math.nan)),
                )
                if all(math.isfinite(value) for value in values):
                    updates[name] = EntityPose(
                        tuple(values[:3]),
                        tuple(values[3:]),
                    )
            if not updates:
                return
            with self._strict_entity_lock:
                for name, pose in updates.items():
                    self._strict_entity_poses[name] = pose
                    self._strict_entity_wall_ns[name] = observed_at
                    self._strict_entity_generations[name] = (
                        int(self._strict_entity_generations.get(name, 0)) + 1
                    )

        def _on_gz_world_stats(self, message: object) -> None:
            sim_time = getattr(message, 'sim_time', None)
            seconds = int(getattr(sim_time, 'sec', -1))
            nanoseconds = int(getattr(sim_time, 'nsec', -1))
            iterations = int(getattr(message, 'iterations', -1))
            if (
                seconds < 0
                or nanoseconds < 0
                or nanoseconds >= 1_000_000_000
                or iterations < 0
            ):
                return
            with self._strict_world_stats_lock:
                self._strict_world_sim_time_ns = (
                    seconds * 1_000_000_000 + nanoseconds
                )
                self._strict_world_iterations = iterations
                self._strict_world_paused = bool(
                    getattr(message, 'paused', False)
                )
                self._strict_world_stats_wall_ns = time.monotonic_ns()

        def strict_world_stats(
            self,
            *,
            maximum_wall_age_s: float = 0.50,
        ) -> Tuple[int, int, bool]:
            with self._strict_world_stats_lock:
                sim_time_ns = int(self._strict_world_sim_time_ns)
                iterations = int(self._strict_world_iterations)
                paused = bool(self._strict_world_paused)
                stamp_ns = int(self._strict_world_stats_wall_ns)
            age = (
                math.inf
                if stamp_ns <= 0
                else (time.monotonic_ns() - stamp_ns) / 1e9
            )
            if sim_time_ns < 0 or iterations < 0:
                raise RuntimeError('Gazebo world statistics unavailable')
            if age < 0.0 or age > float(maximum_wall_age_s):
                raise RuntimeError('Gazebo world statistics stale')
            return sim_time_ns, iterations, paused

        def request_run_to_sim_time_once(self, target_sim_time_ns: int) -> bool:
            request = WorldControl()
            request.pause = False
            request.run_to_sim_time.sec = int(target_sim_time_ns) // 1_000_000_000
            request.run_to_sim_time.nsec = int(target_sim_time_ns) % 1_000_000_000
            requested, response = self._gz_pose_node.request(
                WORLD_CONTROL_SERVICE,
                request,
                WorldControl,
                Boolean,
                5000,
            )
            return bool(requested and getattr(response, 'data', False))

        def paused_serialized_joint_positions(
            self,
            names: Sequence[str],
        ) -> Tuple[Mapping[str, float], Tuple[int, int, bool]]:
            """Read exact joint doubles from Gazebo's paused state service."""

            requested = False
            response = None
            for _attempt in range(2):
                requested, response = self._gz_pose_node.request(
                    '/world/erc_world/state',
                    Empty(),
                    Empty,
                    SerializedStepMap,
                    5000,
                )
                if requested:
                    break
                time.sleep(0.05)
            if not requested:
                # gz-transport's Python request occasionally times out after
                # the world has paused even though the service remains fully
                # responsive.  The CLI uses a separate transport client and
                # has proven reliable at the same checkpoint.  This fallback
                # is read-only and still produces the identical protobuf type
                # consumed by the extraction below.
                try:
                    completed = subprocess.run(
                        [
                            'gz', 'service',
                            '-s', '/world/erc_world/state',
                            '--reqtype', 'gz.msgs.Empty',
                            '--reptype', 'gz.msgs.SerializedStepMap',
                            '--timeout', '5000',
                            '--req', '',
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=7.0,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise RuntimeError(
                        'Gazebo serialized state request and CLI fallback '
                        f'failed:{exc}'
                    ) from exc
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip()
                    raise RuntimeError(
                        'Gazebo serialized state request and CLI fallback '
                        f'failed:{detail}'
                    )
                response = SerializedStepMap()
                try:
                    text_format.Parse(completed.stdout, response)
                except text_format.ParseError as exc:
                    raise RuntimeError(
                        'Gazebo serialized state CLI response malformed'
                    ) from exc
            assert response is not None
            stats = response.stats
            sim_time_ns = (
                int(stats.sim_time.sec) * 1_000_000_000
                + int(stats.sim_time.nsec)
            )
            state_stats = (
                sim_time_ns,
                int(stats.iterations),
                bool(stats.paused),
            )
            # gz-sim 8's stable component type id for JointPosition.  The
            # component payload is a gz.msgs.Double_V containing one value.
            joint_position_type = 8319580315957903596
            wanted = {str(name): str(name).encode('utf-8') for name in names}
            observed = {}
            for entity in response.state.entities.values():
                raw_components = tuple(
                    component.component
                    for component in entity.components.values()
                )
                matching = tuple(
                    name for name, raw_name in wanted.items()
                    if raw_name in raw_components
                )
                if not matching:
                    continue
                component = entity.components.get(joint_position_type)
                if component is None:
                    raise RuntimeError('Gazebo joint-position component missing')
                values = Double_V()
                values.ParseFromString(component.component)
                if len(values.data) != 1 or not math.isfinite(float(values.data[0])):
                    raise RuntimeError('Gazebo joint-position component malformed')
                for name in matching:
                    if name in observed:
                        raise RuntimeError(f'duplicate Gazebo joint entity:{name}')
                    observed[name] = float(values.data[0])
            missing = tuple(name for name in wanted if name not in observed)
            if missing:
                raise RuntimeError(f'Gazebo serialized state omitted joints:{missing}')
            return observed, state_stats

        def strict_entity_pose_with_generation(
            self,
            name: str,
            *,
            maximum_wall_age_s: float = PAYLOAD_TARGET_POSE_MAXIMUM_WALL_AGE_S,
        ) -> Tuple[EntityPose, int]:
            """Return one wall-fresh pose and its matching callback generation."""

            with self._strict_entity_lock:
                pose = self._strict_entity_poses.get(str(name))
                stamp_ns = int(self._strict_entity_wall_ns.get(str(name), 0))
                generation = int(
                    self._strict_entity_generations.get(str(name), 0)
                )
            age = (
                math.inf
                if stamp_ns <= 0
                else (time.monotonic_ns() - stamp_ns) / 1e9
            )
            if pose is None:
                raise RuntimeError(f'Gazebo dynamic pose omitted {name!r}')
            if age < 0.0 or age > float(maximum_wall_age_s):
                raise RuntimeError(f'Gazebo dynamic pose for {name!r} is stale')
            return pose, generation

        def strict_entity_pose(
            self,
            name: str,
            *,
            maximum_wall_age_s: float = PAYLOAD_TARGET_POSE_MAXIMUM_WALL_AGE_S,
        ) -> EntityPose:
            pose, _generation = self.strict_entity_pose_with_generation(
                name,
                maximum_wall_age_s=maximum_wall_age_s,
            )
            return pose

        def strict_entity_generation(self, name: str) -> int:
            with self._strict_entity_lock:
                return int(self._strict_entity_generations.get(str(name), 0))

        def wait_for_strict_entity_pose(
            self,
            name: str,
            previous_generation: int,
            *,
            timeout: float = 1.0,
        ) -> EntityPose:
            deadline = time.monotonic() + float(timeout)
            while time.monotonic() < deadline:
                if self.strict_entity_generation(name) > int(previous_generation):
                    return self.strict_entity_pose(name)
                if self._cancel.is_set():
                    break
                time.sleep(0.005)
            raise RuntimeError(
                f'no fresh Gazebo dynamic pose arrived for {name!r}'
            )

        def target_robot_contact_latched(self) -> bool:
            with self._lock:
                return bool(self._target_robot_contact_latched)

        def set_strict_payload_motion(
            self,
            reference: EntityPose,
            phase: str,
            planned_delta: Sequence[float] = (0.0, 0.0, 0.0),
            maximum_rotation_rad: float = PAYLOAD_CONTINUOUS_ROTATION_LIMIT_RAD,
        ) -> None:
            phase_name = str(phase)
            if phase_name not in (
                'stationary',
                'micro_lift',
                'outward',
                'diagonal_peel',
                'shelf_outward',
                'shelf_inward',
                'upright_extract',
            ):
                raise RuntimeError(f'invalid strict payload phase:{phase_name}')
            upright_hand_from_book = None
            upright_initial_clearance = -math.inf
            if phase_name == 'upright_extract':
                base_pose = self.strict_entity_pose('tiago_pro').planar
                arm_q8 = _joint_snapshot(self, IK_JOINTS)
                hand_world = world_hand_pose(
                    (base_pose.x, base_pose.y, base_pose.yaw),
                    self.chain.forward(arm_q8),
                )
                book_world = np.eye(4, dtype=float)
                book_world[:3, :3] = quaternion_matrix(reference.quaternion)
                book_world[:3, 3] = np.asarray(reference.position, dtype=float)
                upright_hand_from_book = np.linalg.inv(hand_world) @ book_world
                upright_initial_clearance = (
                    ROUTE_SHELF_FRONT_WORLD_X_M
                    - book_maximum_world_x(reference)
                )
            with self._strict_hazard_lock:
                self._strict_payload_target_reference = reference
                self._strict_payload_motion_phase = phase_name
                self._strict_payload_planned_delta = tuple(
                    float(value) for value in planned_delta
                )
                self._strict_payload_rotation_limit_rad = float(
                    maximum_rotation_rad
                )
                self._strict_payload_relative_yaw_consecutive_samples = 0
                self._strict_payload_relative_yaw_last_generation = -1
                self._strict_payload_max_relative_yaw_rad = 0.0
                self._strict_payload_max_consecutive_soft_yaw_samples = 0
                self._strict_reseat_last_generation = -1
                self._strict_reseat_last_signed_y_rad = 0.0
                self._strict_reseat_last_deep_edge_signed_m = math.inf
                if upright_hand_from_book is not None:
                    self._strict_upright_extract_hand_from_book = (
                        upright_hand_from_book
                    )
                    self._strict_upright_extract_last_generation = -1
                    self._strict_upright_extract_maximum_lift_m = 0.0
                    self._strict_upright_extract_maximum_outward_m = 0.0
                    self._strict_upright_extract_maximum_clearance_m = float(
                        upright_initial_clearance
                    )
                self._strict_last_payload_hazard = {}

        @staticmethod
        def _strict_entity_transform(pose: EntityPose) -> object:
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = quaternion_matrix(pose.quaternion)
            transform[:3, 3] = np.asarray(pose.position, dtype=float)
            return transform

        def strict_upright_extract_attachment_gate(
            self,
            target: EntityPose,
            arm_q8: Sequence[float],
            base: Pose2,
        ) -> GateResult:
            """Compare the measured book/hand transform with the launch anchor."""

            with self._strict_hazard_lock:
                anchor = self._strict_upright_extract_hand_from_book
            if anchor is None:
                return GateResult(
                    False,
                    'practical_upright_extract_attachment_anchor_unavailable',
                    {},
                )
            try:
                hand_world = world_hand_pose(
                    (base.x, base.y, base.yaw),
                    self.chain.forward(tuple(float(value) for value in arm_q8)),
                )
                relative = (
                    np.linalg.inv(hand_world)
                    @ self._strict_entity_transform(target)
                )
                translation_error = float(np.linalg.norm(
                    relative[:3, 3] - anchor[:3, 3]
                ))
                rotation_error = float(rotation_matrix_distance(
                    relative[:3, :3],
                    anchor[:3, :3],
                ))
                corner_error = float(
                    translation_error
                    + 2.0 * PAYLOAD_CONSERVATIVE_RADIUS_M
                    * math.sin(0.5 * rotation_error)
                )
            except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                return GateResult(
                    False,
                    'practical_upright_extract_attachment_measurement_invalid',
                    {},
                )
            metrics = {
                'book_hand_relative_translation_error_m': translation_error,
                'book_hand_relative_rotation_error_rad': rotation_error,
                'book_hand_relative_corner_error_m': corner_error,
            }
            for accepted, reason in (
                (
                    translation_error
                    <= active_upright_extract_attachment_translation_limit_m,
                    'practical_upright_extract_attachment_translation_slip',
                ),
                (
                    rotation_error
                    <= active_upright_extract_attachment_rotation_limit_rad,
                    'practical_upright_extract_attachment_rotation_slip',
                ),
                (
                    corner_error <= active_upright_extract_attachment_corner_limit_m,
                    'practical_upright_extract_attachment_corner_slip',
                ),
            ):
                if not accepted:
                    return GateResult(False, reason, metrics)
            return GateResult(
                True,
                'practical_upright_extract_attachment_verified',
                metrics,
            )

        def strict_relative_yaw_gate(
            self,
            yaw_error_rad: float,
            pose_generation: int,
        ) -> Optional[Mapping[str, object]]:
            """Debounce a soft yaw limit while preserving an immediate hard trip."""

            with self._strict_hazard_lock:
                result, count, generation = consecutive_soft_limit_gate(
                    float(yaw_error_rad),
                    int(pose_generation),
                    soft_limit=float(
                        self._strict_payload_relative_yaw_limit_rad
                    ),
                    hard_limit=float(
                        self._strict_payload_relative_yaw_hard_limit_rad
                    ),
                    required_consecutive_samples=max(
                        1,
                        int(
                            self._strict_payload_relative_yaw_consecutive_samples_required
                        ),
                    ),
                    previous_generation=int(
                        self._strict_payload_relative_yaw_last_generation
                    ),
                    previous_consecutive_samples=int(
                        self._strict_payload_relative_yaw_consecutive_samples
                    ),
                )
                self._strict_payload_relative_yaw_consecutive_samples = count
                self._strict_payload_relative_yaw_last_generation = generation
                self._strict_payload_max_relative_yaw_rad = max(
                    float(self._strict_payload_max_relative_yaw_rad),
                    float(yaw_error_rad),
                )
                self._strict_payload_max_consecutive_soft_yaw_samples = max(
                    int(self._strict_payload_max_consecutive_soft_yaw_samples),
                    int(count),
                )
            if result.ok:
                return None
            return {
                'kind': result.reason,
                'yaw_error_rad': float(yaw_error_rad),
                'yaw_limit_rad': float(
                    result.metrics.get('hard_limit', math.nan)
                    if result.reason == 'debounced_limit_hard_trip'
                    else result.metrics.get('soft_limit', math.nan)
                ),
                'consecutive_samples': int(
                    result.metrics.get('consecutive_samples', count)
                ),
                'consecutive_samples_required': int(
                    result.metrics.get('consecutive_samples_required', 1)
                ),
            }

        def strict_payload_observed_metrics(self) -> Mapping[str, object]:
            with self._strict_hazard_lock:
                return {
                    'maximum_observed_relative_yaw_rad': float(
                        self._strict_payload_max_relative_yaw_rad
                    ),
                    'maximum_consecutive_soft_yaw_samples': int(
                        self._strict_payload_max_consecutive_soft_yaw_samples
                    ),
                }

        def _follow(
            self,
            client: object,
            joints: Sequence[str],
            positions: Sequence[float],
            duration: float,
        ) -> bool:
            del client, joints, positions, duration
            raise RuntimeError('strict held extraction rejected inherited motion')

        def _payload_hazard_reason(
            self,
            *,
            max_age: float = 0.20,
        ) -> Optional[str]:
            def hazard(reason: str, **fields: object) -> str:
                with self._strict_hazard_lock:
                    self._strict_last_payload_hazard = {
                        'reason': str(reason),
                        **fields,
                    }
                return str(reason)

            reason = super()._payload_hazard_reason(max_age=max_age)
            if reason is not None:
                return hazard(reason)
            if not bool(getattr(self, '_payload_monitor_enabled', False)):
                return None
            evidence, model, identity_reason = self._adaptive_pressure_evidence(
                minimum_width=PRESSURE_LOCK_MINIMUM_WIDTH_M,
                baseline_effort=math.nan,
            )
            if not evidence.verified or model != BOOK or identity_reason is not None:
                return hazard(
                    f'payload_pressure:{identity_reason or evidence.reason}',
                    measured_width_m=float(evidence.width),
                    left_force_n=float(evidence.left_force),
                    right_force_n=float(evidence.right_force),
                    left_samples=int(evidence.left_samples),
                    right_samples=int(evidence.right_samples),
                    target_model=model,
                )
            with self._strict_hazard_lock:
                reference_target = self._strict_payload_target_reference
                motion_phase = self._strict_payload_motion_phase
                planned_delta = self._strict_payload_planned_delta
                rotation_limit = self._strict_payload_rotation_limit_rad
                minimum_force_n = self._strict_payload_minimum_force_n
                minimum_total_force_n = (
                    self._strict_payload_minimum_total_force_n
                )
                maximum_force_n = self._strict_payload_maximum_force_n
                live_reference_force_n = (
                    self._strict_payload_live_reference_force_n
                )
                live_retention_fraction = (
                    self._strict_payload_live_retention_fraction
                )
                maximum_balance_change = (
                    self._strict_payload_maximum_balance_change
                )
                stable_quaternion = self._strict_payload_stable_quaternion
                absolute_rotation_limit = (
                    self._strict_payload_absolute_rotation_limit_rad
                )
                absolute_yaw_limit = (
                    self._strict_payload_absolute_yaw_limit_rad
                )
                progress_regression_limit = (
                    self._strict_payload_progress_regression_limit_m
                )
                progress_overshoot_limit = (
                    self._strict_payload_progress_overshoot_limit_m
                )
                cross_track_limit = (
                    self._strict_payload_cross_track_limit_m
                )
                stationary_translation_limit = (
                    self._strict_payload_stationary_translation_limit_m
                )
                lateral_y_limit = self._strict_payload_lateral_y_limit_m
            minimum_left_force, minimum_right_force = minimum_force_n
            maximum_left_force, maximum_right_force = maximum_force_n
            if float(evidence.left_force) > float(maximum_left_force):
                return hazard(
                    'payload_left_force_above_absolute_maximum',
                    left_force_n=float(evidence.left_force),
                    maximum_left_force_n=float(maximum_left_force),
                )
            if float(evidence.right_force) > float(maximum_right_force):
                return hazard(
                    'payload_right_force_above_absolute_maximum',
                    right_force_n=float(evidence.right_force),
                    maximum_right_force_n=float(maximum_right_force),
                )
            if float(evidence.left_force) < float(minimum_left_force):
                return hazard(
                    'payload_left_force_below_absolute_minimum',
                    left_force_n=float(evidence.left_force),
                    minimum_left_force_n=float(minimum_left_force),
                )
            if float(evidence.right_force) < float(minimum_right_force):
                return hazard(
                    'payload_right_force_below_absolute_minimum',
                    right_force_n=float(evidence.right_force),
                    minimum_right_force_n=float(minimum_right_force),
                )
            if (
                float(evidence.left_force) + float(evidence.right_force)
                < float(minimum_total_force_n)
            ):
                return hazard(
                    'payload_total_force_below_absolute_minimum',
                    total_force_n=(
                        float(evidence.left_force)
                        + float(evidence.right_force)
                    ),
                    minimum_total_force_n=float(minimum_total_force_n),
                )
            if float(live_retention_fraction) > 0.0:
                retention = bilateral_pressure_retention_gate(
                    live_reference_force_n,
                    (float(evidence.left_force), float(evidence.right_force)),
                    minimum_each_side_retention_fraction=float(
                        live_retention_fraction
                    ),
                    maximum_normalized_balance_change=float(
                        maximum_balance_change
                    ),
                )
                if not retention.ok:
                    return hazard(
                        f'payload_{retention.reason}',
                        **retention.metrics,
                    )
            reference_base = self._strict_payload_base_reference
            reference_gripper = self._strict_payload_gripper_reference
            if (
                reference_target is None
                or reference_base is None
                or reference_gripper is None
            ):
                return hazard('strict_payload_reference_unavailable')
            with self._lock:
                gripper = self.joints.get('gripper_left_finger_joint')
            if gripper is None or not math.isfinite(float(gripper)):
                return hazard('payload_aperture_unavailable')
            if abs(float(gripper) - reference_gripper) > PAYLOAD_APERTURE_DRIFT_LIMIT_M:
                return hazard(
                    'payload_aperture_drift',
                    measured_width_m=float(gripper),
                    reference_width_m=float(reference_gripper),
                )
            try:
                target, target_generation = (
                    self.strict_entity_pose_with_generation(BOOK)
                )
                base = self.strict_entity_pose('tiago_pro').planar
            except RuntimeError:
                return hazard('payload_dynamic_pose_stale')
            delta = tuple(
                after - before
                for before, after in zip(
                    reference_target.position,
                    target.position,
                )
            )
            if (
                practical_upright_extract
                and motion_phase == 'upright_extract'
            ):
                try:
                    arm_q8 = _joint_snapshot(self, IK_JOINTS)
                except RuntimeError:
                    return hazard('practical_upright_extract_arm_state_stale')
                with self._lock:
                    joint_stamps = tuple(
                        self._strict_joint_state_wall_times.get(name)
                        for name in IK_JOINTS
                    )
                if any(stamp is None for stamp in joint_stamps):
                    return hazard(
                        'practical_upright_extract_arm_state_freshness_unavailable'
                    )
                joint_age = max(
                    time.monotonic() - float(stamp)
                    for stamp in joint_stamps
                    if stamp is not None
                )
                if joint_age < 0.0 or joint_age > 0.10:
                    return hazard(
                        'practical_upright_extract_arm_state_stale',
                        joint_state_wall_age_s=float(joint_age),
                        maximum_joint_state_wall_age_s=0.10,
                    )
                attachment = self.strict_upright_extract_attachment_gate(
                    target,
                    arm_q8,
                    base,
                )
                if not attachment.ok:
                    return hazard(attachment.reason, **attachment.metrics)
                lift = float(delta[2])
                outward = -float(delta[0])
                lateral = abs(float(delta[1]))
                floor_clearance = (
                    book_minimum_world_z(target) - active_support_floor_world_z
                )
                shelf_clearance = (
                    ROUTE_SHELF_FRONT_WORLD_X_M
                    - book_maximum_world_x(target)
                )
                lift_regressed = False
                outward_regressed = False
                clearance_regressed = False
                with self._strict_hazard_lock:
                    if (
                        target_generation
                        > self._strict_upright_extract_last_generation
                    ):
                        lift_regressed = bool(
                            lift
                            < self._strict_upright_extract_maximum_lift_m
                            - active_upright_extract_lift_regression_limit_m
                        )
                        outward_regressed = bool(
                            outward
                            < self._strict_upright_extract_maximum_outward_m
                            - active_upright_extract_outward_regression_limit_m
                        )
                        clearance_regressed = bool(
                            shelf_clearance
                            < self._strict_upright_extract_maximum_clearance_m
                            - active_upright_extract_clearance_regression_limit_m
                        )
                        self._strict_upright_extract_last_generation = (
                            target_generation
                        )
                        self._strict_upright_extract_maximum_lift_m = max(
                            self._strict_upright_extract_maximum_lift_m,
                            lift,
                        )
                        self._strict_upright_extract_maximum_outward_m = max(
                            self._strict_upright_extract_maximum_outward_m,
                            outward,
                        )
                        self._strict_upright_extract_maximum_clearance_m = max(
                            self._strict_upright_extract_maximum_clearance_m,
                            shelf_clearance,
                        )
                upright_metrics = {
                    'book_lift_m': lift,
                    'book_outward_progress_m': outward,
                    'book_lateral_motion_m': lateral,
                    'book_floor_clearance_m': floor_clearance,
                    'book_shelf_clearance_m': shelf_clearance,
                    **attachment.metrics,
                }
                for accepted, reason_name in (
                    (
                        not lift_regressed,
                        'practical_upright_extract_lift_regressed',
                    ),
                    (
                        not outward_regressed,
                        'practical_upright_extract_outward_regressed',
                    ),
                    (
                        not clearance_regressed,
                        'practical_upright_extract_clearance_regressed',
                    ),
                    (
                        -0.00025
                        <= lift
                        <= active_upright_extract_lift_m + 0.00075,
                        'practical_upright_extract_lift_out_of_bounds',
                    ),
                    (
                        -0.00025
                        <= outward
                        <= active_upright_extract_outward_m + 0.0010,
                        'practical_upright_extract_progress_out_of_bounds',
                    ),
                    (
                        lateral <= active_continuous_lateral_y_limit_m,
                        'practical_upright_extract_lateral_drift',
                    ),
                    (
                        floor_clearance >= -0.00050,
                        'practical_upright_extract_shelf_penetration',
                    ),
                    (
                        outward
                        <= active_upright_extract_unload_outward_limit_m
                        or (
                            lift
                            >= active_upright_extract_minimum_retained_lift_m
                            and floor_clearance
                            >= active_upright_extract_minimum_pull_floor_clearance_m
                        ),
                        'practical_upright_extract_pulled_before_unloaded',
                    ),
                ):
                    if not accepted:
                        return hazard(reason_name, **upright_metrics)
            if practical_reseat_support and motion_phase == 'shelf_inward':
                expected_signed_y = active_reseat_expected_signed_y_rotation_rad
                if expected_signed_y is None:
                    return hazard('practical_reseat_watchdog_plan_unavailable')
                relative_rotation = _world_rotation_vector(
                    reference_target.quaternion,
                    target.quaternion,
                )
                signed_y = float(relative_rotation[1])
                cross_axis_growth = math.hypot(
                    relative_rotation[0],
                    relative_rotation[2],
                )
                face_bottom_signed = (
                    book_minimum_world_z_at_x(
                        target,
                        ROUTE_SHELF_FRONT_WORLD_X_M,
                    )
                    - active_support_floor_world_z
                )
                deep_edge_signed = (
                    book_deepest_extent_lower_world_z(target)
                    - active_support_floor_world_z
                )
                overlap = (
                    book_maximum_world_x(target)
                    - ROUTE_SHELF_FRONT_WORLD_X_M
                )
                global_bottom_signed = (
                    book_minimum_world_z(target)
                    - active_support_floor_world_z
                )
                regression = False
                deep_edge_rise = False
                with self._strict_hazard_lock:
                    if target_generation > self._strict_reseat_last_generation:
                        regression = bool(
                            signed_y
                            < self._strict_reseat_last_signed_y_rad - 0.002
                        )
                        deep_edge_rise = bool(
                            math.isfinite(
                                self._strict_reseat_last_deep_edge_signed_m
                            )
                            and signed_y < float(expected_signed_y) - 0.002
                            and deep_edge_signed
                            > self._strict_reseat_last_deep_edge_signed_m + 0.0005
                        )
                        self._strict_reseat_last_generation = target_generation
                        self._strict_reseat_last_signed_y_rad = signed_y
                        self._strict_reseat_last_deep_edge_signed_m = (
                            deep_edge_signed
                        )
                reseat_metrics = {
                    'signed_world_y_rotation_rad': signed_y,
                    'expected_signed_world_y_rotation_rad': float(
                        expected_signed_y
                    ),
                    'cross_axis_rotation_growth_rad': cross_axis_growth,
                    'book_shelf_face_bottom_signed_m': face_bottom_signed,
                    'book_deep_edge_signed_m': deep_edge_signed,
                    'book_global_bottom_signed_m': global_bottom_signed,
                    'book_shelf_overlap_m': overlap,
                    'book_lateral_motion_m': abs(delta[1]),
                }
                if regression:
                    return hazard(
                        'practical_reseat_reverse_roll_regressed',
                        **reseat_metrics,
                    )
                if deep_edge_rise:
                    return hazard(
                        'practical_reseat_deep_edge_reversed',
                        **reseat_metrics,
                    )
                for accepted, reason_name in (
                    (
                        -0.002 <= signed_y <= float(expected_signed_y) + 0.006,
                        'practical_reseat_signed_roll_out_of_bounds',
                    ),
                    (
                        cross_axis_growth <= 0.002,
                        'practical_reseat_cross_axis_rotation_growth',
                    ),
                    (
                        -0.00050 <= face_bottom_signed <= 0.00075,
                        'practical_reseat_lost_shelf_lip_contact',
                    ),
                    (
                        deep_edge_signed >= -0.00025,
                        'practical_reseat_deep_edge_penetrated_shelf',
                    ),
                    (
                        overlap >= 0.069,
                        'practical_reseat_shelf_overlap_insufficient',
                    ),
                    (
                        abs(delta[1]) <= 0.00030,
                        'practical_reseat_lateral_drift',
                    ),
                    (
                        signed_y >= float(expected_signed_y) - 0.002
                        or overlap <= 0.0725,
                        'practical_reseat_moved_inward_before_upright',
                    ),
                    (
                        signed_y < float(expected_signed_y) - 0.002
                        or -0.00050 <= global_bottom_signed <= 0.00075,
                        'practical_reseat_upright_floor_support_invalid',
                    ),
                ):
                    if not accepted:
                        return hazard(reason_name, **reseat_metrics)
            if motion_phase == 'micro_lift':
                progress = delta[2]
                planned_progress = MICRO_LIFT_STEP_M
                cross_track = math.hypot(delta[0], delta[1])
            elif motion_phase == 'outward':
                progress = -delta[0]
                planned_progress = OUTWARD_STEP_M
                cross_track = math.hypot(delta[1], delta[2])
            elif motion_phase in (
                'diagonal_peel',
                'shelf_outward',
                'shelf_inward',
            ):
                planned_progress = math.sqrt(sum(
                    value * value for value in planned_delta
                ))
                if planned_progress <= 0.0:
                    return hazard('diagonal_peel_plan_invalid')
                direction = tuple(
                    value / planned_progress for value in planned_delta
                )
                progress = sum(
                    value * axis for value, axis in zip(delta, direction)
                )
                cross_track = math.sqrt(sum(
                    (value - progress * axis) ** 2
                    for value, axis in zip(delta, direction)
                ))
            elif motion_phase == 'upright_extract':
                # The dedicated piecewise lift-then-pull watchdog above has
                # already checked progress, clearance, and hand/book coupling.
                # Do not project its L-shaped route onto a single chord here.
                progress = 0.0
                planned_progress = 0.0
                cross_track = 0.0
            elif motion_phase == 'stationary':
                progress = _distance(target.position, reference_target.position)
                planned_progress = 0.0
                cross_track = progress
            else:
                return hazard(
                    'payload_motion_phase_invalid',
                    motion_phase=str(motion_phase),
                )
            shelf_motion = motion_phase in (
                'diagonal_peel',
                'shelf_outward',
                'shelf_inward',
            )
            regression_limit = (
                float(progress_regression_limit)
                if shelf_motion else PAYLOAD_INWARD_REGRESSION_LIMIT_M
            )
            overshoot_limit = (
                float(stationary_translation_limit)
                if motion_phase == 'stationary'
                else (
                    float(progress_overshoot_limit)
                    if shelf_motion else PAYLOAD_CONTINUOUS_TRANSLATION_LIMIT_M
                )
            )
            if (
                progress < -regression_limit
                or progress > planned_progress + overshoot_limit
            ):
                return hazard(
                    'payload_motion_progress_out_of_bounds',
                    motion_phase=motion_phase,
                    progress_m=float(progress),
                    planned_progress_m=float(planned_progress),
                )
            translation_limit = (
                float(stationary_translation_limit)
                if motion_phase == 'stationary'
                else (
                    float(cross_track_limit)
                    if shelf_motion else PAYLOAD_CONTINUOUS_TRANSLATION_LIMIT_M
                )
            )
            if cross_track > translation_limit:
                return hazard(
                    'payload_leg_cross_track_motion',
                    motion_phase=motion_phase,
                    cross_track_error_m=float(cross_track),
                )
            if shelf_motion and abs(delta[1]) > float(lateral_y_limit):
                return hazard(
                    'payload_leg_lateral_y_motion',
                    motion_phase=motion_phase,
                    lateral_y_motion_m=float(abs(delta[1])),
                    lateral_y_limit_m=float(lateral_y_limit),
                )
            if motion_phase == 'stationary':
                translation_error = progress
            else:
                longitudinal_error = max(
                    0.0,
                    -progress,
                    progress - planned_progress,
                )
                translation_error = math.hypot(
                    cross_track,
                    longitudinal_error,
                )
            if translation_error > translation_limit:
                return hazard(
                    'payload_continuous_translation_error',
                    motion_phase=motion_phase,
                    translation_error_m=float(translation_error),
                )
            rotation_error = quaternion_distance(
                target.quaternion,
                reference_target.quaternion,
            )
            relative_rotation_vector = _world_rotation_vector(
                reference_target.quaternion,
                target.quaternion,
            )
            relative_yaw = abs(relative_rotation_vector[2])
            yaw_fault = self.strict_relative_yaw_gate(
                relative_yaw,
                target_generation,
            )
            if yaw_fault is not None:
                return hazard(
                    'payload_leg_yaw_drift',
                    motion_phase=motion_phase,
                    **yaw_fault,
                )
            if stable_quaternion is not None:
                absolute_rotation_vector = _world_rotation_vector(
                    stable_quaternion,
                    target.quaternion,
                )
                absolute_rotation = math.sqrt(sum(
                    value * value for value in absolute_rotation_vector
                ))
                absolute_yaw = abs(absolute_rotation_vector[2])
                if absolute_rotation > absolute_rotation_limit:
                    return hazard(
                        'payload_absolute_rotation_exceeded_limit',
                        absolute_rotation_rad=float(absolute_rotation),
                        absolute_rotation_limit_rad=float(
                            absolute_rotation_limit
                        ),
                    )
                if absolute_yaw > absolute_yaw_limit:
                    return hazard(
                        'payload_absolute_yaw_exceeded_limit',
                        absolute_yaw_rad=float(absolute_yaw),
                        absolute_yaw_limit_rad=float(absolute_yaw_limit),
                    )
            if rotation_error > rotation_limit:
                return hazard(
                    'payload_leg_rotation_drift',
                    motion_phase=motion_phase,
                    rotation_error_rad=float(rotation_error),
                    rotation_limit_rad=float(rotation_limit),
                )
            corner_displacement = (
                translation_error
                + 2.0 * PAYLOAD_CONSERVATIVE_RADIUS_M
                * math.sin(0.5 * rotation_error)
            )
            if (
                corner_displacement
                > active_continuous_corner_displacement_limit_m
            ):
                return hazard(
                    'payload_continuous_corner_displacement',
                    motion_phase=motion_phase,
                    corner_displacement_m=float(corner_displacement),
                    corner_displacement_limit_m=float(
                        active_continuous_corner_displacement_limit_m
                    ),
                    translation_error_m=float(translation_error),
                    rotation_error_rad=float(rotation_error),
                )
            if math.hypot(
                base.x - reference_base.x,
                base.y - reference_base.y,
            ) > PAYLOAD_BASE_POSITION_LIMIT_M:
                return hazard('base_moved_during_loaded_extraction')
            yaw_error = abs(math.atan2(
                math.sin(base.yaw - reference_base.yaw),
                math.cos(base.yaw - reference_base.yaw),
            ))
            if yaw_error > PAYLOAD_BASE_YAW_LIMIT_RAD:
                return hazard('base_rotated_during_loaded_extraction')
            return None

        def strict_last_payload_hazard(self) -> Mapping[str, object]:
            with self._strict_hazard_lock:
                return dict(self._strict_last_payload_hazard)

        def strict_pose_only_hazard_reason(self) -> Optional[str]:
            """Check physical pose safety while contact samples are refreshed.

            The inherited retention probe intentionally suppresses contact-loss
            latching after clearing its sample epoch.  This independent path
            omits only that unavailable contact evidence; aperture, target
            pose, absolute orientation, and base constraints remain live.
            """

            def hazard(reason: str, **fields: object) -> str:
                with self._strict_hazard_lock:
                    self._strict_last_payload_hazard = {
                        'reason': str(reason),
                        **fields,
                    }
                return str(reason)

            if not bool(getattr(self, '_payload_monitor_enabled', False)):
                return None
            if self.target_robot_contact_latched():
                return hazard('payload_robot_contact')
            with self._strict_hazard_lock:
                reference_target = self._strict_payload_target_reference
                rotation_limit = self._strict_payload_rotation_limit_rad
                stable_quaternion = self._strict_payload_stable_quaternion
                absolute_rotation_limit = (
                    self._strict_payload_absolute_rotation_limit_rad
                )
                absolute_yaw_limit = (
                    self._strict_payload_absolute_yaw_limit_rad
                )
                translation_limit = (
                    self._strict_payload_pose_probe_translation_limit_m
                )
            reference_base = self._strict_payload_base_reference
            reference_gripper = self._strict_payload_gripper_reference
            if (
                reference_target is None
                or reference_base is None
                or reference_gripper is None
            ):
                return hazard('strict_payload_reference_unavailable')
            with self._lock:
                gripper = self.joints.get('gripper_left_finger_joint')
            if gripper is None or not math.isfinite(float(gripper)):
                return hazard('payload_aperture_unavailable')
            aperture_error = abs(float(gripper) - float(reference_gripper))
            if aperture_error > PAYLOAD_APERTURE_DRIFT_LIMIT_M:
                return hazard(
                    'payload_aperture_drift',
                    measured_width_m=float(gripper),
                    reference_width_m=float(reference_gripper),
                )
            try:
                target, target_generation = (
                    self.strict_entity_pose_with_generation(BOOK)
                )
                base = self.strict_entity_pose('tiago_pro').planar
            except RuntimeError:
                return hazard('payload_dynamic_pose_stale')
            translation_error = _distance(
                target.position,
                reference_target.position,
            )
            if translation_error > float(translation_limit):
                return hazard(
                    'payload_pose_probe_translation_drift',
                    translation_error_m=float(translation_error),
                    translation_limit_m=float(translation_limit),
                )
            rotation_error = quaternion_distance(
                target.quaternion,
                reference_target.quaternion,
            )
            if rotation_error > float(rotation_limit):
                return hazard(
                    'payload_pose_probe_rotation_drift',
                    rotation_error_rad=float(rotation_error),
                    rotation_limit_rad=float(rotation_limit),
                )
            relative_rotation_vector = _world_rotation_vector(
                reference_target.quaternion,
                target.quaternion,
            )
            relative_yaw = abs(relative_rotation_vector[2])
            yaw_fault = self.strict_relative_yaw_gate(
                relative_yaw,
                target_generation,
            )
            if yaw_fault is not None:
                return hazard(
                    'payload_pose_probe_yaw_drift',
                    **yaw_fault,
                )
            if stable_quaternion is not None:
                absolute_rotation_vector = _world_rotation_vector(
                    stable_quaternion,
                    target.quaternion,
                )
                absolute_rotation = math.sqrt(sum(
                    value * value for value in absolute_rotation_vector
                ))
                absolute_yaw = abs(absolute_rotation_vector[2])
                if absolute_rotation > float(absolute_rotation_limit):
                    return hazard(
                        'payload_absolute_rotation_exceeded_limit',
                        absolute_rotation_rad=float(absolute_rotation),
                        absolute_rotation_limit_rad=float(
                            absolute_rotation_limit
                        ),
                    )
                if absolute_yaw > float(absolute_yaw_limit):
                    return hazard(
                        'payload_absolute_yaw_exceeded_limit',
                        absolute_yaw_rad=float(absolute_yaw),
                        absolute_yaw_limit_rad=float(absolute_yaw_limit),
                    )
            corner_displacement = (
                translation_error
                + 2.0 * PAYLOAD_CONSERVATIVE_RADIUS_M
                * math.sin(0.5 * rotation_error)
            )
            if (
                corner_displacement
                > active_continuous_corner_displacement_limit_m
            ):
                return hazard(
                    'payload_continuous_corner_displacement',
                    corner_displacement_m=float(corner_displacement),
                    corner_displacement_limit_m=float(
                        active_continuous_corner_displacement_limit_m
                    ),
                    translation_error_m=float(translation_error),
                    rotation_error_rad=float(rotation_error),
                )
            if math.hypot(
                base.x - reference_base.x,
                base.y - reference_base.y,
            ) > PAYLOAD_BASE_POSITION_LIMIT_M:
                return hazard('base_moved_during_loaded_extraction')
            yaw_error = abs(math.atan2(
                math.sin(base.yaw - reference_base.yaw),
                math.cos(base.yaw - reference_base.yaw),
            ))
            if yaw_error > PAYLOAD_BASE_YAW_LIMIT_RAD:
                return hazard('base_rotated_during_loaded_extraction')
            return None

        def cancel_active_goals(self) -> bool:
            self._cancel.set()
            with self._lock:
                goal_handles = tuple(self._goal_handles)
            all_confirmed = True
            for goal_handle in goal_handles:
                try:
                    result_future = goal_handle.get_result_async()
                    confirmed = (
                        result_future.done()
                        or self._cancel_goal_and_confirm(goal_handle, result_future)
                    )
                except Exception:
                    confirmed = False
                    try:
                        goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                all_confirmed = all_confirmed and confirmed
                if confirmed:
                    with self._lock:
                        if goal_handle in self._goal_handles:
                            self._goal_handles.remove(goal_handle)
            return all_confirmed

    class BaseStopNode(Node):
        def __init__(self) -> None:
            super().__init__('strict_held_extraction_base_stop')
            self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        def publish_zero(self) -> None:
            message = Twist()
            self.cmd_pub.publish(message)
            self.cmd_pub.publish(message)

    certificate_ok, certificate_metrics = active_validate_certificate()
    _require(GateResult(
        certificate_ok,
        f'{active_certificate_reason}_verified'
        if certificate_ok else f'{active_certificate_reason}_invalid',
        certificate_metrics,
    ))

    bringup_share = Path(get_package_share_directory('erc_bringup'))
    bringup_prefix = Path(get_package_prefix('erc_bringup'))
    description_share = Path(get_package_share_directory('erc_description'))
    asset_paths = {
        'simulation_launch': bringup_share / 'launch' / 'simulation.launch.py',
        'trajectory_controller_params': bringup_share / 'config' / 'controller_params.yaml',
        'controller_manager_params': bringup_share / 'config' / 'gazebo_controller_manager_cfg.yaml',
        'gripper_command_clamp': bringup_prefix / 'lib' / 'erc_bringup' / 'gripper_command_clamp.py',
        'world_sdf': description_share / 'worlds' / 'erc_world.sdf',
        'book_sdf': description_share / 'models' / 'book' / 'sdf' / 'erc_book.sdf',
        'shelf_sdf': description_share / 'models' / 'shelf' / 'sdf' / 'erc_shelf.sdf',
        'shelf_mesh': description_share / 'models' / 'shelf' / 'meshes' / 'erc_base_shelf.STL',
        'table_sdf': description_share / 'models' / 'table' / 'sdf' / 'erc_table.sdf',
        'table_mesh': description_share / 'models' / 'table' / 'meshes' / 'erc_base_table.STL',
        'collection_bin_sdf': description_share / 'models' / 'collection_bin' / 'sdf' / 'erc_collection_bin.sdf',
        'collection_bin_mesh': description_share / 'models' / 'collection_bin' / 'meshes' / 'erc_base_collection_bin.STL',
        'robot_urdf': description_share / 'urdf' / 'tiago_pro.urdf',
    }
    urdf_root = ET.fromstring(asset_paths['robot_urdf'].read_bytes())
    collision_packages = sorted({
        str(mesh.attrib['filename'])[len('package://'):].split('/', 1)[0]
        for mesh in urdf_root.findall('.//collision/geometry/mesh')
        if str(mesh.attrib.get('filename', '')).startswith('package://')
    })
    package_shares = {
        package: Path(get_package_share_directory(package))
        for package in collision_packages
    }
    _require(certificate_asset_gate(asset_paths, package_shares))

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = StrictHeldNode()
    stop = BaseStopNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(stop)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    monitor_stop = threading.Event()
    monitor_suspended = threading.Event()
    monitor_quiesced = threading.Event()
    monitor_fault = threading.Event()
    monitor_reason = {'value': ''}
    monitor_thread: Optional[threading.Thread] = None
    world_unpaused = False
    right_reference: Optional[Tuple[float, ...]] = None
    right_gripper_reference: Optional[Tuple[float, ...]] = None
    head_reference: Optional[Tuple[float, ...]] = None

    def pause_or_raise() -> None:
        nonlocal world_unpaused
        stop.publish_zero()
        if not set_world_paused_confirmed(True):
            raise RuntimeError('Gazebo world pause was not confirmed')
        world_unpaused = False

    def latch_passive_fault(reason: str) -> None:
        if monitor_fault.is_set():
            return
        monitor_reason['value'] = reason
        monitor_fault.set()
        node._cancel.set()
        stop.publish_zero()

    def passive_monitor() -> None:
        assert (
            right_reference is not None
            and right_gripper_reference is not None
            and head_reference is not None
        )
        while not monitor_stop.is_set():
            if monitor_suspended.is_set():
                monitor_quiesced.set()
                while (
                    monitor_suspended.is_set()
                    and not monitor_stop.is_set()
                ):
                    time.sleep(0.01)
                monitor_quiesced.clear()
                continue
            try:
                if not stationary_joint_gate(
                    right_reference,
                    _joint_snapshot(node, RIGHT_ARM_JOINTS),
                ).ok:
                    latch_passive_fault('right_arm_moved')
                    return
                if not stationary_joint_gate(
                    right_gripper_reference,
                    _joint_snapshot(node, RIGHT_GRIPPER_JOINTS),
                    limit=PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
                ).ok:
                    latch_passive_fault('right_gripper_moved')
                    return
                if not stationary_joint_gate(
                    head_reference,
                    _joint_snapshot(node, HEAD_JOINTS),
                ).ok:
                    latch_passive_fault('head_moved')
                    return
                with node._strict_hazard_lock:
                    payload_motion_phase = str(
                        node._strict_payload_motion_phase
                    )
                if (
                    bool(getattr(node, '_payload_monitor_enabled', False))
                    and payload_motion_phase == 'stationary'
                ):
                    with node._lock:
                        pressure_refresh_active = bool(
                            getattr(node, '_retention_probe_active', False)
                        )
                    payload_reason = (
                        node.strict_pose_only_hazard_reason()
                        if pressure_refresh_active
                        else node._payload_hazard_reason(max_age=0.15)
                    )
                    if payload_reason is not None:
                        latch_passive_fault(
                            f'payload_hazard:{payload_reason}'
                        )
                        return
            except Exception as exc:
                latch_passive_fault(f'passive_joint_monitor_failed:{exc}')
                return
            time.sleep(0.01)

    def require_monitor_clear() -> None:
        if monitor_fault.is_set():
            raise RuntimeError(
                monitor_reason['value'] or 'passive_joint_monitor_failed'
            )
        if node._cancel.is_set():
            raise RuntimeError('strict held extraction was cancelled')
        if bool(getattr(node, '_payload_monitor_enabled', False)):
            hazard = node._payload_hazard_reason(max_age=0.15)
            if hazard is not None:
                node._cancel.set()
                raise RuntimeError(f'payload_hazard:{hazard}')

    def pressure_evidence(stage: str) -> Tuple[float, float, float]:
        pose_probe_stop = threading.Event()
        pose_probe_fault = {'reason': ''}
        pose_probe_thread: Optional[threading.Thread] = None

        def pose_probe_watchdog() -> None:
            while not pose_probe_stop.is_set():
                reason = node.strict_pose_only_hazard_reason()
                if reason is not None:
                    pose_probe_fault['reason'] = str(reason)
                    node._cancel.set()
                    stop.publish_zero()
                    return
                time.sleep(0.005)

        if bool(getattr(node, '_payload_monitor_enabled', False)):
            pose_probe_thread = threading.Thread(
                target=pose_probe_watchdog,
                daemon=True,
            )
            pose_probe_thread.start()
        try:
            retained = node._fresh_retention_probe(
                'strict_held_extraction',
                stage,
                require_new_sample=True,
            )
            if pose_probe_thread is not None and not pose_probe_fault['reason']:
                reason = node.strict_pose_only_hazard_reason()
                if reason is not None:
                    pose_probe_fault['reason'] = str(reason)
                    node._cancel.set()
                    stop.publish_zero()
        finally:
            pose_probe_stop.set()
            if pose_probe_thread is not None:
                pose_probe_thread.join(timeout=0.25)
        if pose_probe_fault['reason']:
            raise RuntimeError(
                f"payload pose hazard during pressure refresh:{stage}:"
                f"{pose_probe_fault['reason']}"
            )
        if not retained:
            raise RuntimeError(f'fresh bilateral retention failed:{stage}')
        evidence, model, identity_reason = node._adaptive_pressure_evidence(
            minimum_width=PRESSURE_LOCK_MINIMUM_WIDTH_M,
            baseline_effort=math.nan,
        )
        minimum_left_force, minimum_right_force = active_minimum_force_n
        maximum_left_force, maximum_right_force = active_maximum_force_n
        absolute_force_ok = bool(
            float(evidence.left_force) >= float(minimum_left_force)
            and float(evidence.right_force) >= float(minimum_right_force)
            and float(evidence.left_force) <= float(maximum_left_force)
            and float(evidence.right_force) <= float(maximum_right_force)
            and (
                float(evidence.left_force) + float(evidence.right_force)
                >= float(active_minimum_total_force_n)
            )
        )
        _emit(
            'pressure_evidence',
            stage=stage,
            passed=bool(
                evidence.verified
                and model == BOOK
                and identity_reason is None
                and absolute_force_ok
            ),
            reason=(
                identity_reason
                or (
                    evidence.reason
                    if absolute_force_ok
                    else 'absolute_pressure_out_of_bounds'
                )
            ),
            target_model=model,
            measured_width_m=float(evidence.width),
            left_force_n=float(evidence.left_force),
            right_force_n=float(evidence.right_force),
            left_samples=int(evidence.left_samples),
            right_samples=int(evidence.right_samples),
            minimum_left_force_n=float(minimum_left_force),
            minimum_right_force_n=float(minimum_right_force),
            minimum_total_force_n=float(active_minimum_total_force_n),
            maximum_left_force_n=float(maximum_left_force),
            maximum_right_force_n=float(maximum_right_force),
        )
        if not (
            evidence.verified
            and model == BOOK
            and identity_reason is None
            and absolute_force_ok
            and not node.target_robot_contact_latched()
        ):
            raise RuntimeError(f'fresh bilateral pressure failed:{stage}')
        return (
            float(evidence.left_force),
            float(evidence.right_force),
            float(evidence.width),
        )

    def upright_extract_strong_pressure_gate(
        pressure: Sequence[float],
        *,
        stage: str,
    ) -> GateResult:
        """Require the audited balanced launch grasp for unsupported carry."""

        left_force = float(pressure[0])
        right_force = float(pressure[1])
        stronger = max(left_force, right_force)
        weaker = min(left_force, right_force)
        ratio = weaker / stronger if stronger > 0.0 else 0.0
        post_tighten = bool(
            practical_tighten_direct_pull
            and stage not in ('initial_checkpoint', 'resume_transport_lock')
        )
        minimum_each_force_n = 2.2 if post_tighten else 1.5
        minimum_total_force_n = 4.8 if post_tighten else 3.5
        ok = bool(
            left_force >= minimum_each_force_n
            and right_force >= minimum_each_force_n
            and left_force + right_force >= minimum_total_force_n
            and stronger <= 8.0
            and ratio >= 0.80
        )
        return GateResult(
            ok,
            f'practical_upright_extract_{stage}_pressure_verified'
            if ok else f'practical_upright_extract_{stage}_pressure_failed',
            {
                'left_force_n': left_force,
                'right_force_n': right_force,
                'total_force_n': left_force + right_force,
                'weaker_stronger_ratio': ratio,
                'minimum_each_force_n': minimum_each_force_n,
                'minimum_total_force_n': minimum_total_force_n,
                'minimum_weaker_stronger_ratio': 0.80,
                'maximum_each_force_n': 8.0,
            },
        )

    def run_midroute_settle_once() -> None:
        """Advance the certified paused state by one exact 0.300 s run-to."""

        nonlocal monitor_thread, world_unpaused
        expected_start_ns = int(round(
            float(MIDROUTE_CHECKPOINT['world_sim_time_s']) * 1e9
        ))
        expected_start_iterations = int(MIDROUTE_CHECKPOINT['world_iterations'])
        physics_step_ns = int(round(float(MIDROUTE_PHYSICS_STEP_SIZE_S) * 1e9))
        settle_steps = int(MIDROUTE_SETTLE_STEP_COUNT)
        settle_duration_ns = physics_step_ns * settle_steps
        target_sim_time_ns = expected_start_ns + settle_duration_ns
        target_iterations = expected_start_iterations + settle_steps
        _require(GateResult(
            bool(
                MIDROUTE_EXECUTION_POLICY.get('run_to_sim_time_required') is True
                and int(MIDROUTE_EXECUTION_POLICY[
                    'maximum_resume_request_count'
                ]) == 1
                and settle_duration_ns == int(round(
                    float(MIDROUTE_SETTLE_DURATION_S) * 1e9
                ))
            ),
            'midroute_exact_run_to_policy_verified',
            {
                'physics_step_size_s': physics_step_ns / 1e9,
                'settle_steps': float(settle_steps),
                'settle_duration_s': settle_duration_ns / 1e9,
                'run_to_request_limit': float(MIDROUTE_EXECUTION_POLICY[
                    'maximum_resume_request_count'
                ]),
            },
        ))
        physics_element = ET.parse(asset_paths['world_sdf']).find(
            './/physics/max_step_size'
        )
        installed_step_s = (
            float(physics_element.text)
            if physics_element is not None and physics_element.text is not None
            else math.nan
        )
        _require(GateResult(
            math.isclose(
                installed_step_s,
                float(MIDROUTE_PHYSICS_STEP_SIZE_S),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            'midroute_physics_step_verified',
            {
                'installed_physics_step_s': installed_step_s,
                'certified_physics_step_s': float(
                    MIDROUTE_PHYSICS_STEP_SIZE_S
                ),
            },
        ))

        start_stats_message = read_world_stats_message()
        start_sim_time_s, start_iterations, start_paused = world_stats_snapshot(
            start_stats_message
        )
        start_sim_time_ns = int(round(start_sim_time_s * 1e9))
        exact_epoch = bool(
            start_paused
            and start_sim_time_ns == expected_start_ns
            and start_iterations == expected_start_iterations
            and expected_start_ns % physics_step_ns == 0
        )
        _require(GateResult(
            exact_epoch,
            'midroute_exact_start_epoch_verified'
            if exact_epoch else 'midroute_exact_start_epoch_mismatch',
            {
                'observed_sim_time_s': start_sim_time_ns / 1e9,
                'expected_sim_time_s': expected_start_ns / 1e9,
                'observed_iterations': float(start_iterations),
                'expected_iterations': float(expected_start_iterations),
                'world_paused': float(start_paused),
            },
        ))

        required_joints = (
            *IK_JOINTS,
            *RIGHT_ARM_JOINTS,
            *RIGHT_GRIPPER_JOINTS,
            *HEAD_JOINTS,
            *LEFT_GRIPPER_JOINTS,
        )
        serialized_joints, serialized_stats = (
            node.paused_serialized_joint_positions(required_joints)
        )
        _require(GateResult(
            serialized_stats
            == (expected_start_ns, expected_start_iterations, True),
            'midroute_serialized_state_epoch_verified'
            if serialized_stats
            == (expected_start_ns, expected_start_iterations, True)
            else 'midroute_serialized_state_epoch_mismatch',
            {
                'serialized_sim_time_s': serialized_stats[0] / 1e9,
                'serialized_iterations': float(serialized_stats[1]),
                'serialized_world_paused': float(serialized_stats[2]),
            },
        ))
        expected_left_gripper = tuple(
            float(MIDROUTE_LEFT_GRIPPER_GEOMETRY[name])
            for name in LEFT_GRIPPER_JOINTS
        )
        joint_groups = (
            (
                IK_JOINTS,
                tuple(float(value) for value in MIDROUTE_LEFT_Q8),
                active_checkpoint_arm_limit_rad,
                'midroute_serialized_left_arm_mismatch',
            ),
            (
                RIGHT_ARM_JOINTS,
                tuple(float(value) for value in MIDROUTE_RIGHT_Q7),
                PASSIVE_JOINT_STATIONARY_LIMIT_RAD,
                'midroute_serialized_right_arm_mismatch',
            ),
            (
                RIGHT_GRIPPER_JOINTS,
                tuple(float(value) for value in MIDROUTE_RIGHT_GRIPPER_Q8),
                PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
                'midroute_serialized_right_gripper_mismatch',
            ),
            (
                HEAD_JOINTS,
                tuple(float(value) for value in MIDROUTE_HEAD_Q2),
                PASSIVE_JOINT_STATIONARY_LIMIT_RAD,
                'midroute_serialized_head_mismatch',
            ),
            (
                LEFT_GRIPPER_JOINTS,
                expected_left_gripper,
                active_checkpoint_gripper_limit_m,
                'midroute_serialized_left_gripper_mismatch',
            ),
        )
        for names, expected, limit, failure_reason in joint_groups:
            _require(expected_joint_gate(
                tuple(serialized_joints[name] for name in names),
                expected,
                limit=float(limit),
                failure_reason=failure_reason,
            ))

        checkpoint_book = EntityPose(
            tuple(float(value) for value in MIDROUTE_BOOK_POSITION),
            tuple(float(value) for value in MIDROUTE_BOOK_QUATERNION),
        )
        checkpoint_base = EntityPose(
            tuple(float(value) for value in MIDROUTE_BASE_POSITION),
            tuple(float(value) for value in MIDROUTE_BASE_QUATERNION),
        )
        start_dynamic_message = read_exact_seed101_dynamic_pose_message()
        start_full_message = read_full_pose_message()
        start_books = book_poses_from_dynamic_pose(start_dynamic_message)
        start_book = start_books[BOOK]
        start_base_pose = entity_pose_from_dynamic_pose(
            start_dynamic_message,
            'tiago_pro',
        )
        _require(active_checkpoint_gate(
            tuple(serialized_joints[name] for name in IK_JOINTS),
            float(serialized_joints[LEFT_GRIPPER_JOINTS[0]]),
            start_base_pose.planar,
            start_book,
        ))
        _require(held_stability_gate(
            checkpoint_base,
            start_base_pose,
            translation_limit_m=active_checkpoint_book_position_limit_m,
            rotation_limit_rad=active_checkpoint_book_rotation_limit_rad,
        ))
        passive_translation_limit = float(
            MIDROUTE_CONTINUOUS_POLICY[
                'maximum_passive_scene_translation_from_reference_m'
            ]
        )
        passive_rotation_limit = float(
            MIDROUTE_CONTINUOUS_POLICY[
                'maximum_passive_scene_rotation_from_reference_rad'
            ]
        )
        _require(passive_scene_reference_gate(
            start_full_message,
            MIDROUTE_PASSIVE_SCENE_REFERENCE,
            translation_limit_m=passive_translation_limit,
            rotation_limit_rad=passive_rotation_limit,
        ))
        _require(resume_seed101_scene_gate(start_books, start_books))

        passive_book_reference = {}
        for name, serialized_pose in MIDROUTE_PASSIVE_SCENE_REFERENCE.items():
            if not str(name).startswith('book_'):
                continue
            values = tuple(float(value) for value in str(serialized_pose).split())
            passive_book_reference[str(name)] = EntityPose(
                tuple(values[:3]),
                _quaternion_from_rpy(*values[3:]),
            )

        with node._lock:
            node._target_book_model = BOOK
            node._held_book_corners = np.asarray(
                active_attached_book_corners,
                dtype=float,
            ).copy()
            node._gravity_supported_payload = False
            node._transport_lock_engaged = False
            node._payload_hazard_latched = None
            node._payload_monitor_enabled = True
            # Suppress the inherited pressure timer only during the bounded
            # fresh-contact acquisition grace.  Pose checks remain explicit.
            node._retention_probe_active = True
            node._payload_robot_watchdog_enabled = True
            node._strict_payload_base_reference = checkpoint_base.planar
            node._strict_payload_gripper_reference = float(
                MIDROUTE_GRIPPER_MASTER_M
            )
        reference_force = (
            float(MIDROUTE_PRESSURE_POLICY['preleg_left_force_n']),
            float(MIDROUTE_PRESSURE_POLICY['preleg_right_force_n']),
        )
        with node._strict_hazard_lock:
            node._strict_payload_minimum_force_n = tuple(
                float(value) for value in active_minimum_force_n
            )
            node._strict_payload_minimum_total_force_n = float(
                active_minimum_total_force_n
            )
            node._strict_payload_maximum_force_n = tuple(
                float(value) for value in active_maximum_force_n
            )
            node._strict_payload_live_reference_force_n = reference_force
            node._strict_payload_live_retention_fraction = float(
                active_continuous_force_retention_fraction
            )
            node._strict_payload_maximum_balance_change = float(
                active_maximum_force_balance_change
            )
            node._strict_payload_stable_quaternion = tuple(
                float(value) for value in MIDROUTE_REFERENCE_QUATERNION
            )
            node._strict_payload_absolute_rotation_limit_rad = float(
                active_continuous_absolute_rotation_limit_rad
            )
            node._strict_payload_absolute_yaw_limit_rad = float(
                active_continuous_absolute_yaw_limit_rad
            )
            node._strict_payload_stationary_translation_limit_m = float(
                active_continuous_stationary_translation_limit_m
            )
            node._strict_payload_relative_yaw_limit_rad = float(
                active_continuous_relative_yaw_limit_rad
            )
            node._strict_payload_relative_yaw_hard_limit_rad = float(
                active_continuous_relative_yaw_hard_limit_rad
            )
            node._strict_payload_relative_yaw_consecutive_samples_required = int(
                active_continuous_relative_yaw_consecutive_samples
            )
            node._strict_payload_pose_probe_translation_limit_m = float(
                active_pressure_probe_translation_limit_m
            )
        node.set_strict_payload_motion(
            checkpoint_book,
            'stationary',
            maximum_rotation_rad=float(
                active_continuous_relative_rotation_limit_rad
            ),
        )
        pose_generations_before = {
            name: node.strict_entity_generation(name)
            for name in (BOOK, 'tiago_pro', *passive_book_reference)
        }
        with node._lock:
            joint_generations_before = {
                name: int(node._strict_joint_state_generations.get(name, 0))
                for name in required_joints
            }
        node._clear_target_contact_samples(reset_robot_contact=True)
        stop.publish_zero()
        _require(_cmd_vel_gate(stop))

        monitor_ready = threading.Event()
        observation_lock = threading.Lock()
        observations = {
            'fresh_pose_verified': False,
            'fresh_joint_feedback_verified': False,
            'fresh_pressure': None,
        }
        acquisition_deadline_ns = int(round(
            float(MIDROUTE_PRESSURE_POLICY[
                'fresh_pressure_acquisition_deadline_s'
            ]) * 1e9
        ))

        def latch_midroute_fault(reason: str) -> None:
            latch_passive_fault(f'midroute_settle:{reason}')

        def midroute_monitor() -> None:
            monitor_ready.set()
            while not monitor_stop.is_set():
                try:
                    sim_time_ns, _iterations, _paused = node.strict_world_stats(
                        maximum_wall_age_s=0.75
                    )
                except RuntimeError:
                    time.sleep(0.001)
                    continue
                elapsed_ns = sim_time_ns - expected_start_ns
                if elapsed_ns < 0:
                    latch_midroute_fault('simulation_time_moved_backward')
                    return
                if sim_time_ns > target_sim_time_ns:
                    latch_midroute_fault('simulation_time_exceeded_certified_target')
                    return
                if elapsed_ns <= 0:
                    time.sleep(0.001)
                    continue

                try:
                    observed_book, book_generation = (
                        node.strict_entity_pose_with_generation(BOOK)
                    )
                    observed_base, base_generation = (
                        node.strict_entity_pose_with_generation('tiago_pro')
                    )
                    pose_is_fresh = bool(
                        book_generation > pose_generations_before[BOOK]
                        and base_generation > pose_generations_before['tiago_pro']
                    )
                except RuntimeError:
                    pose_is_fresh = False
                    observed_book = checkpoint_book
                    observed_base = checkpoint_base
                    book_generation = pose_generations_before[BOOK]
                if not pose_is_fresh:
                    if elapsed_ns >= 50_000_000:
                        latch_midroute_fault('fresh_dynamic_pose_not_received')
                        return
                else:
                    with observation_lock:
                        observations['fresh_pose_verified'] = True
                    book_motion = held_stability_gate(
                        checkpoint_book,
                        observed_book,
                        translation_limit_m=float(
                            active_continuous_stationary_translation_limit_m
                        ),
                        rotation_limit_rad=float(
                            active_continuous_relative_rotation_limit_rad
                        ),
                    )
                    if not book_motion.ok:
                        latch_midroute_fault(book_motion.reason)
                        return
                    supported = supported_book_state_gate(
                        observed_book,
                        active_checkpoint_stable_reference,
                        support_floor_world_z_m=active_support_floor_world_z,
                        maximum_rotation_rad=float(
                            active_continuous_absolute_rotation_limit_rad
                        ),
                        maximum_yaw_component_rad=float(
                            active_continuous_absolute_yaw_limit_rad
                        ),
                        minimum_floor_signed_distance_m=float(
                            MIDROUTE_FINAL_DWELL_POLICY['floor_signed_min_m']
                        ),
                        maximum_floor_signed_distance_m=float(
                            MIDROUTE_FINAL_DWELL_POLICY['floor_signed_max_m']
                        ),
                    )
                    if not supported.ok:
                        latch_midroute_fault(supported.reason)
                        return
                    relative_yaw = abs(_world_rotation_vector(
                        checkpoint_book.quaternion,
                        observed_book.quaternion,
                    )[2])
                    yaw_fault = node.strict_relative_yaw_gate(
                        relative_yaw,
                        book_generation,
                    )
                    if yaw_fault is not None:
                        latch_midroute_fault('payload_leg_yaw_drift')
                        return
                    if (
                        math.hypot(
                            observed_base.planar.x - checkpoint_base.planar.x,
                            observed_base.planar.y - checkpoint_base.planar.y,
                        ) > PAYLOAD_BASE_POSITION_LIMIT_M
                        or abs(math.atan2(
                            math.sin(
                                observed_base.planar.yaw
                                - checkpoint_base.planar.yaw
                            ),
                            math.cos(
                                observed_base.planar.yaw
                                - checkpoint_base.planar.yaw
                            ),
                        )) > PAYLOAD_BASE_YAW_LIMIT_RAD
                    ):
                        latch_midroute_fault('base_moved_during_settle')
                        return

                passive_ready = True
                maximum_passive_translation = 0.0
                maximum_passive_rotation = 0.0
                for name, expected_pose in passive_book_reference.items():
                    try:
                        observed_pose, generation = (
                            node.strict_entity_pose_with_generation(name)
                        )
                    except RuntimeError:
                        passive_ready = False
                        break
                    if generation <= pose_generations_before[name]:
                        passive_ready = False
                        break
                    maximum_passive_translation = max(
                        maximum_passive_translation,
                        _distance(observed_pose.position, expected_pose.position),
                    )
                    maximum_passive_rotation = max(
                        maximum_passive_rotation,
                        quaternion_distance(
                            observed_pose.quaternion,
                            expected_pose.quaternion,
                        ),
                    )
                if not passive_ready:
                    if elapsed_ns >= 50_000_000:
                        latch_midroute_fault('fresh_passive_scene_not_received')
                        return
                elif (
                    maximum_passive_translation > passive_translation_limit
                    or maximum_passive_rotation > passive_rotation_limit
                ):
                    latch_midroute_fault('passive_scene_changed_during_settle')
                    return

                try:
                    current_generations = _joint_generation_snapshot(
                        node,
                        required_joints,
                    )
                    joints_are_fresh = all(
                        after > joint_generations_before[name]
                        for name, after in zip(required_joints, current_generations)
                    )
                    if joints_are_fresh:
                        for names, expected, limit, failure_reason in joint_groups:
                            gate = expected_joint_gate(
                                _joint_snapshot(node, names),
                                expected,
                                limit=float(limit),
                                failure_reason=failure_reason,
                            )
                            if not gate.ok:
                                latch_midroute_fault(gate.reason)
                                return
                except RuntimeError:
                    joints_are_fresh = False
                if not joints_are_fresh:
                    if elapsed_ns >= 50_000_000:
                        latch_midroute_fault('fresh_joint_feedback_not_received')
                        return
                else:
                    with observation_lock:
                        observations['fresh_joint_feedback_verified'] = True

                if node.target_robot_contact_latched():
                    latch_midroute_fault('payload_robot_contact')
                    return

                with observation_lock:
                    pressure_acquired = observations['fresh_pressure'] is not None
                if not pressure_acquired:
                    evidence, model, identity_reason = (
                        node._adaptive_pressure_evidence(
                            minimum_width=PRESSURE_LOCK_MINIMUM_WIDTH_M,
                            baseline_effort=math.nan,
                        )
                    )
                    if identity_reason is not None:
                        latch_midroute_fault(
                            f'pressure_identity:{identity_reason}'
                        )
                        return
                    exact_pressure = bool(
                        evidence.verified
                        and model == BOOK
                        and float(evidence.left_force)
                        >= float(active_minimum_force_n[0])
                        and float(evidence.right_force)
                        >= float(active_minimum_force_n[1])
                    )
                    if exact_pressure:
                        retention = bilateral_pressure_retention_gate(
                            reference_force,
                            (
                                float(evidence.left_force),
                                float(evidence.right_force),
                            ),
                            minimum_each_side_retention_fraction=float(
                                active_continuous_force_retention_fraction
                            ),
                            maximum_normalized_balance_change=float(
                                active_maximum_force_balance_change
                            ),
                        )
                        if not retention.ok:
                            latch_midroute_fault(retention.reason)
                            return
                        with observation_lock:
                            observations['fresh_pressure'] = (
                                float(evidence.left_force),
                                float(evidence.right_force),
                                float(evidence.width),
                                int(evidence.left_samples),
                                int(evidence.right_samples),
                                elapsed_ns,
                            )
                        with node._lock:
                            node._transport_lock_engaged = True
                            node._retention_probe_active = False
                    elif elapsed_ns >= acquisition_deadline_ns:
                        latch_midroute_fault(
                            'fresh_bilateral_pressure_acquisition_timeout'
                        )
                        return

                if joints_are_fresh and pose_is_fresh:
                    with observation_lock:
                        pressure_acquired = (
                            observations['fresh_pressure'] is not None
                        )
                    hazard = (
                        node._payload_hazard_reason(max_age=0.15)
                        if pressure_acquired
                        else node.strict_pose_only_hazard_reason()
                    )
                    if hazard is not None:
                        latch_midroute_fault(f'payload_hazard:{hazard}')
                        return
                if sim_time_ns >= target_sim_time_ns:
                    return
                time.sleep(0.001)

        monitor_thread = threading.Thread(
            target=midroute_monitor,
            name='midroute-settle-monitor',
            daemon=True,
        )
        monitor_thread.start()
        if not monitor_ready.wait(timeout=1.0):
            raise RuntimeError('midroute settle monitor failed to arm')
        # Exactly one request is authorized.  Gazebo runs to this absolute,
        # step-aligned time and automatically pauses there.
        world_unpaused = True
        if not node.request_run_to_sim_time_once(target_sim_time_ns):
            raise RuntimeError('Gazebo run-to-simulation-time request failed')

        run_wall_deadline = time.monotonic() + 5.0
        while time.monotonic() < run_wall_deadline:
            if monitor_fault.is_set():
                pause_or_raise()
                raise RuntimeError(
                    monitor_reason['value'] or 'midroute settle monitor failed'
                )
            try:
                sim_time_ns, iterations, paused = node.strict_world_stats(
                    maximum_wall_age_s=0.75
                )
            except RuntimeError:
                time.sleep(0.001)
                continue
            if sim_time_ns > target_sim_time_ns or iterations > target_iterations:
                pause_or_raise()
                raise RuntimeError('Gazebo exceeded certified midroute endpoint')
            if (
                sim_time_ns == target_sim_time_ns
                and iterations == target_iterations
                and paused
            ):
                world_unpaused = False
                break
            time.sleep(0.001)
        else:
            pause_or_raise()
            raise RuntimeError('Gazebo did not reach the certified run-to endpoint')

        monitor_stop.set()
        monitor_thread.join(timeout=1.0)
        if monitor_thread.is_alive():
            raise RuntimeError('midroute settle monitor did not quiesce')
        if monitor_fault.is_set():
            raise RuntimeError(
                monitor_reason['value'] or 'midroute settle monitor failed'
            )
        with observation_lock:
            fresh_pose_verified = bool(observations['fresh_pose_verified'])
            fresh_joint_feedback_verified = bool(
                observations['fresh_joint_feedback_verified']
            )
            acquired_pressure = observations['fresh_pressure']
        if not fresh_pose_verified:
            raise RuntimeError('fresh midroute target pose was never verified')
        if not fresh_joint_feedback_verified:
            raise RuntimeError('fresh midroute joint feedback was never verified')
        if acquired_pressure is None:
            raise RuntimeError('fresh midroute bilateral pressure was never verified')

        final_joint_values = {
            name: value
            for name, value in zip(
                required_joints,
                _joint_snapshot(node, required_joints),
            )
        }
        final_evidence, final_model, final_identity = (
            node._adaptive_pressure_evidence(
                minimum_width=PRESSURE_LOCK_MINIMUM_WIDTH_M,
                baseline_effort=math.nan,
            )
        )
        final_stats_message = read_world_stats_message()
        final_sim_time_s, final_iterations, final_paused = world_stats_snapshot(
            final_stats_message
        )
        final_sim_time_ns = int(round(final_sim_time_s * 1e9))
        exact_final_epoch = bool(
            final_paused
            and final_sim_time_ns == target_sim_time_ns
            and final_iterations == target_iterations
        )
        _require(GateResult(
            exact_final_epoch,
            'midroute_exact_final_epoch_verified'
            if exact_final_epoch else 'midroute_exact_final_epoch_mismatch',
            {
                'final_sim_time_s': final_sim_time_ns / 1e9,
                'expected_final_sim_time_s': target_sim_time_ns / 1e9,
                'final_iterations': float(final_iterations),
                'expected_final_iterations': float(target_iterations),
                'world_paused': float(final_paused),
            },
        ))
        final_dynamic_message = read_exact_seed101_dynamic_pose_message()
        final_full_message = read_full_pose_message()
        final_books = book_poses_from_dynamic_pose(final_dynamic_message)
        final_book = final_books[BOOK]
        final_base_pose = entity_pose_from_dynamic_pose(
            final_dynamic_message,
            'tiago_pro',
        )
        _require(held_stability_gate(
            start_book,
            final_book,
            translation_limit_m=float(
                MIDROUTE_FINAL_DWELL_POLICY['maximum_translation_change_m']
            ),
            rotation_limit_rad=float(
                MIDROUTE_FINAL_DWELL_POLICY['maximum_rotation_change_rad']
            ),
        ))
        _require(active_checkpoint_gate(
            tuple(final_joint_values[name] for name in IK_JOINTS),
            float(final_joint_values[LEFT_GRIPPER_JOINTS[0]]),
            final_base_pose.planar,
            final_book,
        ))
        _require(held_stability_gate(
            checkpoint_base,
            final_base_pose,
            translation_limit_m=active_checkpoint_book_position_limit_m,
            rotation_limit_rad=active_checkpoint_book_rotation_limit_rad,
        ))
        _require(supported_book_state_gate(
            final_book,
            active_checkpoint_stable_reference,
            support_floor_world_z_m=active_support_floor_world_z,
            **active_supported_book_final_policy,
        ))
        for names, expected, limit, failure_reason in joint_groups:
            _require(expected_joint_gate(
                tuple(final_joint_values[name] for name in names),
                expected,
                limit=float(limit),
                failure_reason=failure_reason.replace('serialized', 'final'),
            ))
        _require(passive_scene_reference_gate(
            final_full_message,
            MIDROUTE_PASSIVE_SCENE_REFERENCE,
            translation_limit_m=float(
                MIDROUTE_FINAL_DWELL_POLICY[
                    'maximum_passive_scene_translation_from_reference_m'
                ]
            ),
            rotation_limit_rad=float(
                MIDROUTE_FINAL_DWELL_POLICY[
                    'maximum_passive_scene_rotation_from_reference_rad'
                ]
            ),
        ))
        _require(resume_seed101_scene_gate(start_books, final_books))
        exact_final_pressure = bool(
            final_evidence.verified
            and final_model == BOOK
            and final_identity is None
            and not node.target_robot_contact_latched()
            and float(final_evidence.left_force)
            >= float(active_minimum_force_n[0])
            and float(final_evidence.right_force)
            >= float(active_minimum_force_n[1])
        )
        _require(GateResult(
            exact_final_pressure,
            'midroute_final_exact_bilateral_pressure_verified'
            if exact_final_pressure
            else 'midroute_final_exact_bilateral_pressure_failed',
            {
                'left_force_n': float(final_evidence.left_force),
                'right_force_n': float(final_evidence.right_force),
                'left_samples': float(final_evidence.left_samples),
                'right_samples': float(final_evidence.right_samples),
                'measured_width_m': float(final_evidence.width),
            },
        ))
        _require(bilateral_pressure_retention_gate(
            reference_force,
            (
                float(final_evidence.left_force),
                float(final_evidence.right_force),
            ),
            minimum_each_side_retention_fraction=float(
                active_minimum_force_retention_fraction
            ),
            maximum_normalized_balance_change=float(
                active_maximum_force_balance_change
            ),
        ))
        _emit(
            'result',
            passed=True,
            stage=active_result_stage,
            target_model=BOOK,
            route_steps=0,
            run_to_sim_time_request_count=1,
            resumed_sim_duration_s=settle_duration_ns / 1e9,
            resumed_physics_steps=settle_steps,
            pressure_acquired_after_s=float(acquired_pressure[5]) / 1e9,
            measured_gripper_width_m=float(final_evidence.width),
            left_force_n=float(final_evidence.left_force),
            right_force_n=float(final_evidence.right_force),
            **node.strict_payload_observed_metrics(),
            final_book_position=list(final_book.position),
            final_book_quaternion=list(final_book.quaternion),
            final_base_position=list(final_base_pose.position),
            final_base_quaternion=list(final_base_pose.quaternion),
            lift_distance_m=0.0,
            extraction_distance_m=0.0,
            reseat_distance_m=0.0,
            gripper_reopened=False,
            left_arm_trajectory_commanded=False,
            gripper_trajectory_commanded=False,
            base_motion_commanded=False,
            zero_cmd_vel_published=True,
            right_arm_commanded=False,
            head_commanded=False,
            gazebo_entity_pose_mutation_used=False,
            automatic_next_stage_started=False,
            gazebo_paused=True,
        )

    try:
        _require(world_paused_gate(read_world_stats_message(), True))

        discovery_deadline = time.monotonic() + 20.0
        while time.monotonic() < discovery_deadline:
            if (
                node.arm_client.server_is_ready()
                and node.gripper_pub.get_subscription_count() > 0
                and stop.cmd_pub.get_subscription_count() > 0
            ):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError('held extraction controller discovery failed')

        if halfmillimeter_midroute_settle:
            run_midroute_settle_once()
            return

        with node._lock:
            node._target_book_model = BOOK
            node._held_book_corners = None
            node._gravity_supported_payload = False
            node._transport_lock_engaged = False
            node._payload_hazard_latched = None
            node._payload_monitor_enabled = False
            node._retention_probe_active = False
        node._clear_target_contact_samples(reset_robot_contact=True)
        stop.publish_zero()
        _require(_cmd_vel_gate(stop))
        # Be pessimistic before requesting motion so even an asynchronous
        # interruption in the service/assignment window triggers cleanup pause.
        world_unpaused = True
        if not set_world_paused_confirmed(False):
            raise RuntimeError('Gazebo world unpause was not confirmed')

        required_joints = (
            *IK_JOINTS,
            *RIGHT_ARM_JOINTS,
            *RIGHT_GRIPPER_JOINTS,
            *HEAD_JOINTS,
            *LEFT_GRIPPER_JOINTS,
        )
        ready_deadline = time.monotonic() + 40.0
        while time.monotonic() < ready_deadline:
            ready = (
                all(name in node.joints for name in required_joints)
                and node.strict_entity_generation(BOOK) > 0
                and node.strict_entity_generation('tiago_pro') > 0
            )
            if ready:
                try:
                    _joint_snapshot(node, required_joints)
                    node.strict_entity_pose(BOOK)
                    node.strict_entity_pose('tiago_pro')
                except RuntimeError:
                    ready = False
            if ready:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError('fresh held-state feedback did not arrive')

        arm = _joint_snapshot(node, IK_JOINTS)
        gripper = _joint_snapshot(node, LEFT_GRIPPER_JOINTS)[0]
        right_reference = _joint_snapshot(node, RIGHT_ARM_JOINTS)
        right_gripper_reference = _joint_snapshot(node, RIGHT_GRIPPER_JOINTS)
        head_reference = _joint_snapshot(node, HEAD_JOINTS)
        first_message = read_exact_seed101_dynamic_pose_message()
        first_full_message = read_full_pose_message()
        first_books = book_poses_from_dynamic_pose(first_message)
        first_book = first_books[BOOK]
        first_base = entity_pose_from_dynamic_pose(first_message, 'tiago_pro').planar
        first_shelf = entity_pose_from_dynamic_pose(first_full_message, SHELF)
        _require(resume_seed101_scene_gate(first_books, first_books))
        _require(active_checkpoint_gate(arm, gripper, first_base, first_book))
        _require(expected_joint_gate(
            right_reference,
            active_checkpoint_right_q7,
            limit=PASSIVE_JOINT_STATIONARY_LIMIT_RAD,
            failure_reason='held_checkpoint_right_arm_mismatch',
        ))
        _require(expected_joint_gate(
            right_gripper_reference,
            OFFICIAL_PASSIVE_RIGHT_GRIPPER_Q,
            limit=PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
            failure_reason='held_checkpoint_right_gripper_mismatch',
        ))
        _require(expected_joint_gate(
            head_reference,
            active_checkpoint_head_q2,
            limit=PASSIVE_JOINT_STATIONARY_LIMIT_RAD,
            failure_reason='held_checkpoint_head_mismatch',
        ))

        if not node._wait_sim_duration(0.20):
            raise RuntimeError('held checkpoint stability dwell was interrupted')
        second_message = read_exact_seed101_dynamic_pose_message()
        second_full_message = read_full_pose_message()
        second_books = book_poses_from_dynamic_pose(second_message)
        second_book = second_books[BOOK]
        second_base = entity_pose_from_dynamic_pose(second_message, 'tiago_pro').planar
        second_shelf = entity_pose_from_dynamic_pose(second_full_message, SHELF)
        _require(resume_seed101_scene_gate(first_books, second_books))
        _require(shelf_gate(first_shelf, second_shelf))
        _require(held_stability_gate(
            first_book,
            second_book,
            translation_limit_m=active_startup_stability_translation_limit_m,
            rotation_limit_rad=active_startup_stability_rotation_limit_rad,
        ))
        _require(active_checkpoint_gate(
            _joint_snapshot(node, IK_JOINTS),
            _joint_snapshot(node, LEFT_GRIPPER_JOINTS)[0],
            second_base,
            second_book,
        ))

        monitor_thread = threading.Thread(target=passive_monitor, daemon=True)
        monitor_thread.start()
        require_monitor_clear()

        # Reconstruct pressure from a fresh contact epoch before authorizing
        # the lower transport-lock width used by the already-held checkpoint.
        node._clear_target_contact_samples(reset_robot_contact=True)
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('fresh pressure acquisition dwell was interrupted')
        initial_evidence, initial_model, initial_identity = (
            node._adaptive_pressure_evidence(
                minimum_width=PRESSURE_LOCK_MINIMUM_WIDTH_M,
                baseline_effort=math.nan,
            )
        )
        initial_absolute_force_ok = bool(
            float(initial_evidence.left_force) >= float(active_minimum_force_n[0])
            and float(initial_evidence.right_force)
            >= float(active_minimum_force_n[1])
            and float(initial_evidence.left_force) <= float(active_maximum_force_n[0])
            and float(initial_evidence.right_force)
            <= float(active_maximum_force_n[1])
            and float(initial_evidence.left_force)
            + float(initial_evidence.right_force)
            >= float(active_minimum_total_force_n)
        )
        if practical_upright_extract:
            _require(upright_extract_strong_pressure_gate(
                (
                    float(initial_evidence.left_force),
                    float(initial_evidence.right_force),
                ),
                stage='resume_acquisition',
            ))
        _emit(
            'pressure_evidence',
            stage='resume_acquisition',
            passed=bool(
                initial_evidence.verified
                and initial_model == BOOK
                and initial_identity is None
                and initial_absolute_force_ok
            ),
            reason=(
                initial_identity
                or (
                    initial_evidence.reason
                    if initial_absolute_force_ok
                    else 'absolute_pressure_out_of_bounds'
                )
            ),
            target_model=initial_model,
            measured_width_m=float(initial_evidence.width),
            left_force_n=float(initial_evidence.left_force),
            right_force_n=float(initial_evidence.right_force),
            left_samples=int(initial_evidence.left_samples),
            right_samples=int(initial_evidence.right_samples),
            minimum_left_force_n=float(active_minimum_force_n[0]),
            minimum_right_force_n=float(active_minimum_force_n[1]),
            minimum_total_force_n=float(active_minimum_total_force_n),
            maximum_left_force_n=float(active_maximum_force_n[0]),
            maximum_right_force_n=float(active_maximum_force_n[1]),
        )
        if not (
            initial_evidence.verified
            and initial_model == BOOK
            and initial_identity is None
            and initial_absolute_force_ok
            and not node.target_robot_contact_latched()
        ):
            raise RuntimeError('held checkpoint lacked fresh bilateral pressure')
        with node._lock:
            node._transport_lock_engaged = True
        resume_transport_pressure = pressure_evidence(
            'resume_transport_lock'
        )
        if practical_upright_extract:
            _require(upright_extract_strong_pressure_gate(
                resume_transport_pressure,
                stage='resume_transport_lock',
            ))
        require_monitor_clear()

        start_book = node.strict_entity_pose(BOOK)
        start_base = node.strict_entity_pose('tiago_pro').planar
        start_gripper = _joint_snapshot(node, LEFT_GRIPPER_JOINTS)[0]
        _require(active_checkpoint_gate(
            _joint_snapshot(node, IK_JOINTS),
            start_gripper,
            start_base,
            start_book,
        ))
        with node._lock:
            node._held_book_corners = np.asarray(
                active_attached_book_corners,
                dtype=float,
            ).copy()
            node._gravity_supported_payload = False
            node._payload_hazard_latched = None
            node._payload_robot_watchdog_enabled = True
            node._strict_payload_base_reference = start_base
            node._strict_payload_gripper_reference = float(start_gripper)
        with node._strict_hazard_lock:
            node._strict_payload_minimum_force_n = tuple(
                float(value) for value in active_minimum_force_n
            )
            node._strict_payload_minimum_total_force_n = float(
                active_minimum_total_force_n
            )
            node._strict_payload_maximum_force_n = tuple(
                float(value) for value in active_maximum_force_n
            )
            node._strict_payload_stable_quaternion = (
                tuple(active_continuous_stable_quaternion)
                if active_continuous_stable_quaternion is not None
                else None
            )
            node._strict_payload_absolute_rotation_limit_rad = float(
                active_continuous_absolute_rotation_limit_rad
            )
            node._strict_payload_absolute_yaw_limit_rad = float(
                active_continuous_absolute_yaw_limit_rad
            )
            node._strict_payload_progress_regression_limit_m = float(
                active_continuous_progress_regression_limit_m
            )
            node._strict_payload_progress_overshoot_limit_m = float(
                active_continuous_progress_overshoot_limit_m
            )
            node._strict_payload_cross_track_limit_m = float(
                active_continuous_cross_track_limit_m
            )
            node._strict_payload_stationary_translation_limit_m = float(
                active_continuous_stationary_translation_limit_m
            )
            node._strict_payload_lateral_y_limit_m = float(
                active_continuous_lateral_y_limit_m
            )
            node._strict_payload_relative_yaw_limit_rad = float(
                active_continuous_relative_yaw_limit_rad
            )
            node._strict_payload_relative_yaw_hard_limit_rad = float(
                active_continuous_relative_yaw_hard_limit_rad
            )
            node._strict_payload_relative_yaw_consecutive_samples_required = (
                int(active_continuous_relative_yaw_consecutive_samples)
            )
            node._strict_payload_maximum_balance_change = float(
                active_maximum_force_balance_change
            )
            if active_continuous_force_retention_fraction > 0.0:
                node._strict_payload_live_reference_force_n = (
                    float(resume_transport_pressure[0]),
                    float(resume_transport_pressure[1]),
                )
                node._strict_payload_live_retention_fraction = float(
                    active_continuous_force_retention_fraction
                )
            node._strict_payload_pose_probe_translation_limit_m = float(
                active_pressure_probe_translation_limit_m
            )
        node.set_strict_payload_motion(
            start_book,
            'stationary',
            maximum_rotation_rad=(
                float(active_continuous_relative_rotation_limit_rad)
                if active_continuous_relative_rotation_limit_rad is not None
                else PAYLOAD_CONTINUOUS_ROTATION_LIMIT_RAD
            ),
        )
        with node._lock:
            node._payload_monitor_enabled = True

        integrated_pre_leg_pressure: Optional[Tuple[float, float, float]] = None
        integrated_launch_deadline_ns: Optional[int] = None
        if practical_force_release_resume or practical_tighten_direct_pull:
            integrated_stage_name = (
                'tighten_direct_pull'
                if practical_tighten_direct_pull
                else 'force_release'
            )
            _require(GateResult(
                active_integrated_preload_target_m is not None
                and math.isfinite(float(active_integrated_preload_target_m)),
                f'{integrated_stage_name}_close_target_verified'
                if active_integrated_preload_target_m is not None
                and math.isfinite(float(active_integrated_preload_target_m))
                else f'{integrated_stage_name}_close_target_unavailable',
                {
                    'close_target_available': float(
                        active_integrated_preload_target_m is not None
                    ),
                },
            ))
            if practical_tighten_direct_pull:
                with node._strict_hazard_lock:
                    node._strict_payload_pose_probe_translation_limit_m = float(
                        active_integrated_preload_policy[
                            'maximum_book_translation_m'
                        ]
                    )
                node.set_strict_payload_motion(
                    start_book,
                    'stationary',
                    maximum_rotation_rad=float(
                        active_integrated_preload_policy[
                            'maximum_book_rotation_rad'
                        ]
                    ),
                )
            monitor_suspended.set()
            if not monitor_quiesced.wait(timeout=1.0):
                raise RuntimeError(
                    f'{integrated_stage_name} monitor failed to quiesce'
                )
            baseline_force = (
                float(resume_transport_pressure[0]),
                float(resume_transport_pressure[1]),
            )
            effort_baseline = node._adaptive_effort_baseline(
                int(node.get_clock().now().nanoseconds)
            )
            _require(GateResult(
                math.isfinite(float(effort_baseline)),
                f'{integrated_stage_name}_effort_baseline_verified'
                if math.isfinite(float(effort_baseline))
                else f'{integrated_stage_name}_effort_baseline_unavailable',
                {'effort_baseline': float(effort_baseline)},
            ))
            _require(GateResult(
                bool(
                    int(node.adaptive_contact_samples) >= 3
                    and float(node.adaptive_contact_min_span) >= 0.05
                    and 0.05 <= float(node.adaptive_confirmation_seconds) <= 0.15
                ),
                f'{integrated_stage_name}_fresh_evidence_policy_verified',
                {
                    'minimum_samples': float(node.adaptive_contact_samples),
                    'minimum_span_s': float(node.adaptive_contact_min_span),
                    'confirmation_s': float(node.adaptive_confirmation_seconds),
                },
            ))
            original_force_maximum = float(node.adaptive_contact_force_maximum)
            original_step_motion = float(node.adaptive_step_motion)
            with node._lock:
                node._adaptive_close_active = True
                node._adaptive_overload_latched = None
                node._adaptive_effort_baseline_value = float(effort_baseline)
                node.adaptive_contact_force_maximum = min(
                    original_force_maximum,
                    float(active_integrated_preload_policy[
                        'maximum_each_force_n'
                    ]),
                )
                if active_integrated_preload_use_pressure_ladder:
                    node.adaptive_step_motion = max(
                        original_step_motion,
                        float(active_integrated_preload_policy[
                            'step_motion_s'
                        ]),
                    )
            pose_stop = threading.Event()
            pose_fault = {'value': ''}

            def force_release_pose_watchdog() -> None:
                while not pose_stop.is_set():
                    reason = node.strict_pose_only_hazard_reason()
                    if reason is None and practical_tighten_direct_pull:
                        try:
                            live_arm = _joint_snapshot(node, IK_JOINTS)
                            arm_error = max(
                                abs(after - before)
                                for before, after in zip(
                                    active_checkpoint_left_q8,
                                    live_arm,
                                )
                            )
                            if arm_error > float(
                                active_integrated_preload_policy[
                                    'maximum_arm_error_rad'
                                ]
                            ):
                                reason = 'tighten_direct_pull_arm_drift'
                        except RuntimeError:
                            reason = 'tighten_direct_pull_arm_state_stale'
                    if reason is not None:
                        pose_fault['value'] = str(reason)
                        node._cancel.set()
                        stop.publish_zero()
                        return
                    time.sleep(0.005)

            pose_thread = threading.Thread(
                target=force_release_pose_watchdog,
                daemon=True,
            )
            pose_thread.start()
            try:
                if active_integrated_preload_use_pressure_ladder:
                    close_step_m = float(
                        active_integrated_preload_policy['close_step_m']
                    )
                    maximum_close_steps = int(
                        active_integrated_preload_policy[
                            'maximum_close_steps'
                        ]
                    )
                    close_floor_m = float(active_integrated_preload_target_m)
                    close_targets = tuple(
                        max(
                            close_floor_m,
                            float(start_gripper) - close_step_m * step,
                        )
                        for step in range(1, maximum_close_steps + 1)
                    )
                else:
                    close_targets = (float(active_integrated_preload_target_m),)
                evidence = None
                evidence_model = None
                evidence_identity = None
                final_commanded_preload_target_m = math.nan
                target_force_reached = False
                for close_index, close_target_m in enumerate(
                    close_targets,
                    start=1,
                ):
                    final_commanded_preload_target_m = float(close_target_m)
                    if active_integrated_preload_use_pressure_ladder:
                        with node._lock:
                            # The intentional aperture change is at most one
                            # audited 50 um step from this live reference.
                            node._strict_payload_gripper_reference = float(
                                close_target_m
                            )
                    command_completed = node._command_adaptive_gripper_step(
                        close_target_m
                    )
                    overload_reason = node._adaptive_overload_reason()
                    interrupt_reason = (
                        pose_fault['value']
                        or overload_reason
                        or ('' if command_completed else 'command_interrupted')
                    )
                    _emit(
                        f'{integrated_stage_name}_preload_command',
                        passed=not bool(interrupt_reason),
                        reason=interrupt_reason or 'command_completed',
                        close_step=close_index,
                        target_aperture_m=float(close_target_m),
                    )
                    if interrupt_reason:
                        raise RuntimeError(
                            f'{integrated_stage_name}_preload_failed:'
                            f'{interrupt_reason}'
                        )
                    node._clear_target_contact_samples()
                    confirmation_s = max(
                        float(node.adaptive_confirmation_seconds),
                        float(active_integrated_preload_policy[
                            'minimum_confirmation_s'
                        ]),
                    )
                    if not node._wait_sim_duration(confirmation_s):
                        raise RuntimeError(
                            f'{integrated_stage_name}_fresh_confirmation_'
                            'interrupted'
                        )
                    evidence, evidence_model, evidence_identity = (
                        node._adaptive_pressure_evidence(
                            minimum_width=(
                                float(close_target_m)
                                - float(active_integrated_preload_policy[
                                    'maximum_aperture_error_m'
                                ])
                            ),
                            baseline_effort=float(effort_baseline),
                        )
                    )
                    overload_reason = node._adaptive_overload_reason()
                    if pose_fault['value'] or overload_reason is not None:
                        raise RuntimeError(
                            f'{integrated_stage_name}_preload_failed:'
                            f'{pose_fault["value"] or overload_reason}'
                        )
                    step_book = node.strict_entity_pose(BOOK)
                    step_base = node.strict_entity_pose('tiago_pro').planar
                    step_arm = _joint_snapshot(node, IK_JOINTS)
                    step_translation = _distance(
                        start_book.position,
                        step_book.position,
                    )
                    step_rotation = quaternion_distance(
                        start_book.quaternion,
                        step_book.quaternion,
                    )
                    step_corner_motion = (
                        step_translation + 0.15 * step_rotation
                    )
                    step_lateral = abs(
                        step_book.position[1] - start_book.position[1]
                    )
                    step_floor_signed = (
                        book_minimum_world_z(step_book)
                        - float(active_support_floor_world_z)
                    )
                    step_shelf_overlap = (
                        book_maximum_world_x(step_book)
                        - ROUTE_SHELF_FRONT_WORLD_X_M
                    )
                    step_shelf_com_depth = (
                        float(step_book.position[0])
                        - ROUTE_SHELF_FRONT_WORLD_X_M
                    )
                    step_arm_error = max(
                        abs(after - before)
                        for before, after in zip(
                            active_checkpoint_left_q8,
                            step_arm,
                        )
                    )
                    step_base_translation = math.hypot(
                        step_base.x - start_base.x,
                        step_base.y - start_base.y,
                    )
                    step_base_yaw = abs(math.atan2(
                        math.sin(step_base.yaw - start_base.yaw),
                        math.cos(step_base.yaw - start_base.yaw),
                    ))
                    step_left_force = float(evidence.left_force)
                    step_right_force = float(evidence.right_force)
                    step_total_force = step_left_force + step_right_force
                    step_aperture_error = abs(
                        float(evidence.width) - float(close_target_m)
                    )
                    step_force_ratio = min(
                        step_left_force,
                        step_right_force,
                    ) / max(step_left_force, step_right_force, 1e-12)
                    step_pose_ok = bool(
                        step_aperture_error <= float(
                            active_integrated_preload_policy[
                                'maximum_aperture_error_m'
                            ]
                        )
                        and step_translation <= float(
                            active_integrated_preload_policy[
                                'maximum_book_translation_m'
                            ]
                        )
                        and step_rotation <= float(
                            active_integrated_preload_policy[
                                'maximum_book_rotation_rad'
                            ]
                        )
                        and step_corner_motion <= float(
                            active_integrated_preload_policy[
                                'maximum_book_corner_motion_m'
                            ]
                        )
                        and step_lateral <= float(
                            active_integrated_preload_policy[
                                'maximum_book_lateral_motion_m'
                            ]
                        )
                        and step_floor_signed >= float(
                            active_integrated_preload_policy[
                                'minimum_book_floor_signed_m'
                            ]
                        )
                        and step_shelf_overlap >= float(
                            active_integrated_preload_policy[
                                'minimum_shelf_overlap_m'
                            ]
                        )
                        and step_shelf_com_depth >= float(
                            active_integrated_preload_policy[
                                'minimum_shelf_com_depth_m'
                            ]
                        )
                        and step_arm_error <= float(
                            active_integrated_preload_policy[
                                'maximum_arm_error_rad'
                            ]
                        )
                        and step_base_translation <= float(
                            active_integrated_preload_policy[
                                'maximum_base_translation_m'
                            ]
                        )
                        and step_base_yaw <= float(
                            active_integrated_preload_policy[
                                'maximum_base_yaw_error_rad'
                            ]
                        )
                    )
                    early_balance_ok = bool(
                        step_total_force
                        <= float(active_integrated_preload_policy[
                            'early_balance_check_total_force_n'
                        ])
                        or step_force_ratio >= float(
                            active_integrated_preload_policy[
                                'early_minimum_weaker_stronger_ratio'
                            ]
                        )
                    )
                    target_force_reached = bool(
                        evidence.verified
                        and evidence_model == BOOK
                        and evidence_identity is None
                        and int(evidence.left_samples) >= 3
                        and int(evidence.right_samples) >= 3
                        and step_left_force >= float(
                            active_integrated_preload_policy[
                                'minimum_each_force_n'
                            ]
                        )
                        and step_right_force >= float(
                            active_integrated_preload_policy[
                                'minimum_each_force_n'
                            ]
                        )
                        and step_total_force >= float(
                            active_integrated_preload_policy[
                                'minimum_total_force_n'
                            ]
                        )
                        and step_force_ratio >= float(
                            active_integrated_preload_policy[
                                'minimum_weaker_stronger_ratio'
                            ]
                        )
                    )
                    _emit(
                        f'{integrated_stage_name}_preload_step',
                        passed=bool(step_pose_ok and early_balance_ok),
                        close_step=close_index,
                        target_force_reached=target_force_reached,
                        commanded_aperture_m=float(close_target_m),
                        measured_aperture_m=float(evidence.width),
                        aperture_error_m=step_aperture_error,
                        left_force_n=step_left_force,
                        right_force_n=step_right_force,
                        total_force_n=step_total_force,
                        weaker_stronger_force_ratio=step_force_ratio,
                        book_translation_m=step_translation,
                        book_rotation_rad=step_rotation,
                        book_corner_motion_m=step_corner_motion,
                        book_lateral_motion_m=step_lateral,
                        book_floor_signed_m=step_floor_signed,
                        book_shelf_overlap_m=step_shelf_overlap,
                        book_shelf_com_depth_m=step_shelf_com_depth,
                        maximum_arm_error_rad=step_arm_error,
                        base_translation_m=step_base_translation,
                        base_yaw_error_rad=step_base_yaw,
                    )
                    if not step_pose_ok:
                        raise RuntimeError(
                            f'{integrated_stage_name}_step_pose_gate_failed'
                        )
                    if not early_balance_ok:
                        raise RuntimeError(
                            f'{integrated_stage_name}_step_force_imbalance'
                        )
                    if target_force_reached:
                        break
                if evidence is None or not target_force_reached:
                    raise RuntimeError(
                        f'{integrated_stage_name}_target_force_not_reached'
                    )
            finally:
                pose_stop.set()
                pose_thread.join(timeout=0.25)
                with node._lock:
                    node._adaptive_close_active = False
                    node.adaptive_contact_force_maximum = original_force_maximum
                    node.adaptive_step_motion = original_step_motion

            post_book = node.strict_entity_pose(BOOK)
            post_base = node.strict_entity_pose('tiago_pro').planar
            post_arm = _joint_snapshot(node, IK_JOINTS)
            post_gripper = _joint_snapshot(node, LEFT_GRIPPER_JOINTS)[0]
            preload_translation = _distance(
                start_book.position,
                post_book.position,
            )
            preload_rotation = quaternion_distance(
                start_book.quaternion,
                post_book.quaternion,
            )
            preload_corner_motion = preload_translation + 0.15 * preload_rotation
            preload_lateral = abs(post_book.position[1] - start_book.position[1])
            arm_error = max(
                abs(after - before)
                for before, after in zip(
                    active_checkpoint_left_q8,
                    post_arm,
                )
            )
            base_translation = math.hypot(
                post_base.x - start_base.x,
                post_base.y - start_base.y,
            )
            base_yaw = abs(math.atan2(
                math.sin(post_base.yaw - start_base.yaw),
                math.cos(post_base.yaw - start_base.yaw),
            ))
            left_force = float(evidence.left_force)
            right_force = float(evidence.right_force)
            weaker_stronger_ratio = min(left_force, right_force) / max(
                left_force,
                right_force,
                1e-12,
            )
            preload_metrics = {
                'measured_aperture_m': float(evidence.width),
                'target_aperture_m': float(final_commanded_preload_target_m),
                'left_force_n': left_force,
                'right_force_n': right_force,
                'left_force_gain_n': left_force - baseline_force[0],
                'right_force_gain_n': right_force - baseline_force[1],
                'total_force_n': left_force + right_force,
                'weaker_stronger_force_ratio': weaker_stronger_ratio,
                'left_samples': float(evidence.left_samples),
                'right_samples': float(evidence.right_samples),
                'book_translation_m': preload_translation,
                'book_rotation_rad': preload_rotation,
                'book_corner_motion_m': preload_corner_motion,
                'book_lateral_motion_m': preload_lateral,
                'maximum_arm_error_rad': arm_error,
                'base_translation_m': base_translation,
                'base_yaw_error_rad': base_yaw,
            }
            preload_ok = bool(
                evidence.verified
                and evidence_model == BOOK
                and evidence_identity is None
                and not node.target_robot_contact_latched()
                and int(evidence.left_samples) >= 3
                and int(evidence.right_samples) >= 3
                and abs(float(evidence.width) - float(
                    final_commanded_preload_target_m
                )) <= float(active_integrated_preload_policy[
                    'maximum_aperture_error_m'
                ])
                and left_force >= float(active_integrated_preload_policy[
                    'minimum_each_force_n'
                ])
                and right_force >= float(active_integrated_preload_policy[
                    'minimum_each_force_n'
                ])
                and left_force + right_force >= float(
                    active_integrated_preload_policy['minimum_total_force_n']
                )
                and left_force - baseline_force[0] >= float(
                    active_integrated_preload_policy[
                        'minimum_each_force_gain_n'
                    ]
                )
                and right_force - baseline_force[1] >= float(
                    active_integrated_preload_policy[
                        'minimum_each_force_gain_n'
                    ]
                )
                and weaker_stronger_ratio >= float(
                    active_integrated_preload_policy[
                        'minimum_weaker_stronger_ratio'
                    ]
                )
                and max(left_force, right_force) <= float(
                    active_integrated_preload_policy['maximum_each_force_n']
                )
                and preload_translation <= float(
                    active_integrated_preload_policy[
                        'maximum_book_translation_m'
                    ]
                )
                and preload_rotation <= float(
                    active_integrated_preload_policy[
                        'maximum_book_rotation_rad'
                    ]
                )
                and preload_corner_motion <= float(
                    active_integrated_preload_policy[
                        'maximum_book_corner_motion_m'
                    ]
                )
                and preload_lateral <= float(
                    active_integrated_preload_policy[
                        'maximum_book_lateral_motion_m'
                    ]
                )
                and arm_error <= float(active_integrated_preload_policy[
                    'maximum_arm_error_rad'
                ])
                and base_translation <= float(
                    active_integrated_preload_policy[
                        'maximum_base_translation_m'
                    ]
                )
                and base_yaw <= float(active_integrated_preload_policy[
                    'maximum_base_yaw_error_rad'
                ])
            )
            _require(GateResult(
                preload_ok,
                f'{integrated_stage_name}_preload_verified'
                if preload_ok
                else f'{integrated_stage_name}_preload_gate_failed',
                {
                    **preload_metrics,
                    **{
                        f'policy_{key}': float(value)
                        for key, value in active_integrated_preload_policy.items()
                    },
                },
            ))

            integrated_pre_leg_pressure = (
                left_force,
                right_force,
                float(evidence.width),
            )
            active_minimum_force_n = (
                float(active_integrated_preload_policy[
                    'live_minimum_each_force_n'
                ]),
            ) * 2
            active_minimum_total_force_n = float(
                active_integrated_preload_policy['live_minimum_total_force_n']
            )
            active_continuous_force_retention_fraction = float(
                active_integrated_preload_policy[
                    'live_force_retention_fraction'
                ]
            )
            start_book = post_book
            start_base = post_base
            start_gripper = float(evidence.width)
            with node._lock:
                node._strict_payload_base_reference = start_base
                node._strict_payload_gripper_reference = start_gripper
            with node._strict_hazard_lock:
                node._strict_payload_minimum_force_n = active_minimum_force_n
                node._strict_payload_minimum_total_force_n = (
                    active_minimum_total_force_n
                )
                node._strict_payload_live_reference_force_n = (
                    left_force,
                    right_force,
                )
                node._strict_payload_live_retention_fraction = float(
                    active_integrated_preload_policy[
                        'live_force_retention_fraction'
                    ]
                )
                node._strict_payload_maximum_balance_change = float(
                    active_integrated_preload_policy[
                        'live_maximum_balance_change'
                    ]
                )
                node._strict_payload_maximum_force_n = (
                    float(active_integrated_preload_policy[
                        'maximum_each_force_n'
                    ]),
                ) * 2
                node._strict_payload_stable_quaternion = tuple(
                    post_book.quaternion
                )
                node._strict_payload_pose_probe_translation_limit_m = float(
                    active_pressure_probe_translation_limit_m
                )
            node.set_strict_payload_motion(
                post_book,
                'stationary',
                maximum_rotation_rad=0.00075,
            )
            if not practical_tighten_direct_pull:
                integrated_launch_deadline_ns = (
                    int(node.get_clock().now().nanoseconds)
                    + int(round(1e9 * float(active_integrated_preload_policy[
                        'maximum_launch_delay_s'
                    ])))
                )
            monitor_suspended.clear()
            require_monitor_clear()

        if practical_preload_resume:
            # One command only.  There is deliberately no retry, second rung,
            # recovery motion, or opening path in this checkpoint stage.
            # The generic payload monitor treats moving gripper feedback as
            # invalid retention evidence, so quiesce it before the intentional
            # aperture change.  A dedicated pose-only watchdog remains live.
            monitor_suspended.set()
            if not monitor_quiesced.wait(timeout=1.0):
                raise RuntimeError('preload monitor failed to quiesce')
            baseline_force = (
                float(resume_transport_pressure[0]),
                float(resume_transport_pressure[1]),
            )
            effort_baseline = node._adaptive_effort_baseline(
                int(node.get_clock().now().nanoseconds)
            )
            _require(GateResult(
                math.isfinite(float(effort_baseline)),
                'practical_preload_effort_baseline_verified'
                if math.isfinite(float(effort_baseline))
                else 'practical_preload_effort_baseline_unavailable',
                {'effort_baseline': float(effort_baseline)},
            ))
            original_force_maximum = float(
                node.adaptive_contact_force_maximum
            )
            with node._lock:
                node._adaptive_close_active = True
                node._adaptive_overload_latched = None
                node._adaptive_effort_baseline_value = float(effort_baseline)
                node.adaptive_contact_force_maximum = min(
                    original_force_maximum,
                    8.0,
                )
            preload_pose_stop = threading.Event()
            preload_pose_reason = {'value': ''}

            def preload_pose_watchdog() -> None:
                while not preload_pose_stop.is_set():
                    reason = node.strict_pose_only_hazard_reason()
                    if reason is not None:
                        preload_pose_reason['value'] = str(reason)
                        node._cancel.set()
                        stop.publish_zero()
                        return
                    time.sleep(0.005)

            preload_pose_thread = threading.Thread(
                target=preload_pose_watchdog,
                daemon=True,
            )
            preload_pose_thread.start()
            try:
                command_completed = node._command_adaptive_gripper_step(
                    active_preload_target_m
                )
                if not preload_pose_reason['value']:
                    final_pose_reason = node.strict_pose_only_hazard_reason()
                    if final_pose_reason is not None:
                        preload_pose_reason['value'] = str(final_pose_reason)
                _require(GateResult(
                    not preload_pose_reason['value'],
                    'practical_preload_pose_watchdog_clear'
                    if not preload_pose_reason['value']
                    else (
                        'practical_preload_pose_watchdog:'
                        f"{preload_pose_reason['value']}"
                    ),
                    {},
                ))
                overload_reason = node._adaptive_overload_reason()
                _require(GateResult(
                    bool(command_completed and overload_reason is None),
                    'practical_preload_command_completed'
                    if command_completed and overload_reason is None
                    else f'practical_preload_{overload_reason or "interrupted"}',
                    {
                        'commanded_aperture_m': float(
                            active_preload_target_m
                        ),
                    },
                ))
                final_pressure = pressure_evidence(
                    'practical_single_preload_endpoint'
                )
                post_dwell_overload = node._adaptive_overload_reason()
                _require(GateResult(
                    post_dwell_overload is None,
                    'practical_preload_post_dwell_overload_clear'
                    if post_dwell_overload is None
                    else f'practical_preload_{post_dwell_overload}',
                    {},
                ))
            finally:
                preload_pose_stop.set()
                preload_pose_thread.join(timeout=0.25)
                with node._lock:
                    node._adaptive_close_active = False
                    node.adaptive_contact_force_maximum = (
                        original_force_maximum
                    )

            live_arm = _joint_snapshot(node, IK_JOINTS)
            live_gripper = _joint_snapshot(node, LEFT_GRIPPER_JOINTS)[0]
            live_book = node.strict_entity_pose(BOOK)
            live_base = node.strict_entity_pose('tiago_pro').planar
            live_gate = practical_preload_probe_gate(
                start_book,
                live_book,
                start_arm_q8=active_checkpoint_left_q8,
                observed_arm_q8=live_arm,
                start_base=start_base,
                observed_base=live_base,
                start_aperture_m=start_gripper,
                commanded_aperture_m=active_preload_target_m,
                observed_aperture_m=live_gripper,
                baseline_force_n=baseline_force,
                observed_force_n=final_pressure[:2],
                support_floor_world_z_m=active_support_floor_world_z,
            )

            with node._lock:
                node._payload_monitor_enabled = False
            pause_or_raise()
            serialized_joints, serialized_stats = (
                node.paused_serialized_joint_positions(required_joints)
            )
            _require(GateResult(
                bool(serialized_stats[2]),
                'practical_preload_serialized_pause_verified'
                if serialized_stats[2]
                else 'practical_preload_serialized_pause_failed',
                {
                    'paused_sim_time_s': serialized_stats[0] / 1e9,
                    'paused_iterations': float(serialized_stats[1]),
                },
            ))
            paused_arm = tuple(
                float(serialized_joints[name]) for name in IK_JOINTS
            )
            paused_gripper = float(
                serialized_joints[LEFT_GRIPPER_JOINTS[0]]
            )
            _require(expected_joint_gate(
                paused_arm,
                active_checkpoint_left_q8,
                limit=0.00025,
                failure_reason='practical_preload_paused_left_arm_moved',
            ))
            _require(expected_joint_gate(
                tuple(serialized_joints[name] for name in RIGHT_ARM_JOINTS),
                right_reference,
                limit=PASSIVE_JOINT_STATIONARY_LIMIT_RAD,
                failure_reason='practical_preload_paused_right_arm_moved',
            ))
            _require(expected_joint_gate(
                tuple(
                    serialized_joints[name] for name in RIGHT_GRIPPER_JOINTS
                ),
                right_gripper_reference,
                limit=PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
                failure_reason='practical_preload_paused_right_gripper_moved',
            ))
            _require(expected_joint_gate(
                tuple(serialized_joints[name] for name in HEAD_JOINTS),
                head_reference,
                limit=PASSIVE_JOINT_STATIONARY_LIMIT_RAD,
                failure_reason='practical_preload_paused_head_moved',
            ))
            paused_message = read_exact_seed101_dynamic_pose_message()
            paused_book = book_poses_from_dynamic_pose(paused_message)[BOOK]
            paused_base_pose = entity_pose_from_dynamic_pose(
                paused_message,
                'tiago_pro',
            )
            paused_gate = practical_preload_probe_gate(
                start_book,
                paused_book,
                start_arm_q8=active_checkpoint_left_q8,
                observed_arm_q8=paused_arm,
                start_base=start_base,
                observed_base=paused_base_pose.planar,
                start_aperture_m=start_gripper,
                commanded_aperture_m=active_preload_target_m,
                observed_aperture_m=paused_gripper,
                baseline_force_n=baseline_force,
                observed_force_n=final_pressure[:2],
                support_floor_world_z_m=active_support_floor_world_z,
            )
            _require(live_gate)
            _require(paused_gate)
            _emit(
                'result',
                passed=True,
                stage=active_result_stage,
                target_model=BOOK,
                route_steps=0,
                gripper_command_count=1,
                commanded_aperture_m=float(active_preload_target_m),
                measured_gripper_width_m=paused_gripper,
                left_force_n=float(final_pressure[0]),
                right_force_n=float(final_pressure[1]),
                final_book_position=list(paused_book.position),
                final_book_quaternion=list(paused_book.quaternion),
                final_base_position=list(paused_base_pose.position),
                final_base_quaternion=list(paused_base_pose.quaternion),
                gripper_reopened=False,
                left_arm_trajectory_commanded=False,
                gripper_trajectory_commanded=True,
                base_motion_commanded=False,
                right_arm_commanded=False,
                head_commanded=False,
                automatic_next_stage_started=False,
                gazebo_entity_pose_mutation_used=False,
                gazebo_paused=True,
                **paused_gate.metrics,
            )
            return

        if practical_tighten_direct_pull:
            if active_pre_pull_proof_q8 is None:
                raise RuntimeError('tighten direct-pull proof route unavailable')
            proof_start_book = start_book
            proof_start_base = start_base
            proof_start_q8 = _joint_snapshot(node, IK_JOINTS)
            _require(expected_joint_gate(
                proof_start_q8,
                active_checkpoint_left_q8,
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='tighten_direct_pull_proof_start_mismatch',
            ))
            node.set_strict_payload_motion(
                proof_start_book,
                'upright_extract',
                (0.0, 0.0, 0.00025),
                maximum_rotation_rad=(
                    active_pre_pull_proof_maximum_attachment_rotation_rad
                ),
            )
            proof_dense = dense_arm_waypoints(
                proof_start_q8,
                active_pre_pull_proof_q8,
                maximum_increment=active_dense_maximum_increment_rad,
            )
            proof_segment_duration_s = (
                active_pre_pull_proof_duration_s / len(proof_dense)
            )
            proof_legs = tuple(
                (
                    q8,
                    proof_segment_duration_s,
                    'tighten_direct_pull_proof_lift',
                )
                for q8 in proof_dense
            )
            proof_generation = _joint_generation_snapshot(node, ARM_JOINTS)
            proof_goal, proof_duration_s = (
                node._make_retained_arm_trajectory_goal(proof_legs)
            )
            proof_moved, proof_payload_lost = (
                node._send_retained_arm_trajectory(
                    proof_goal,
                    proof_duration_s,
                    proof_legs,
                    'tighten_direct_pull_proof_lift',
                )
            )
            if not proof_moved:
                proof_hazard = node.strict_last_payload_hazard()
                raise RuntimeError(
                    'tighten_direct_pull_proof_retention_lost'
                    if proof_payload_lost
                    else 'tighten_direct_pull_proof_trajectory_failed:'
                    f'{proof_hazard.get("reason", "unknown")}'
                )
            proof_measured_q8 = node._wait_for_retained_endpoint(
                active_pre_pull_proof_q8,
                command='tighten_direct_pull_proof_lift',
                phase='tighten_direct_pull_proof_lift',
                leg=0,
                settle_timeout=2.0,
                torso_tolerance=active_checkpoint_arm_limit_rad,
                arm_tolerance=active_checkpoint_arm_limit_rad,
            )
            if proof_measured_q8 is None:
                raise RuntimeError(
                    'tighten_direct_pull_proof_endpoint_not_verified'
                )
            _wait_for_joint_update(
                node,
                ARM_JOINTS,
                proof_generation,
                timeout=2.0,
            )
            proof_live_q8 = _joint_snapshot(node, IK_JOINTS)
            _require(expected_joint_gate(
                proof_live_q8,
                active_pre_pull_proof_q8,
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='tighten_direct_pull_proof_joint_mismatch',
            ))
            if not node._wait_sim_duration(0.12):
                raise RuntimeError('tighten direct-pull proof settle interrupted')
            proof_book_generation = node.strict_entity_generation(BOOK)
            proof_base_generation = node.strict_entity_generation('tiago_pro')
            proof_book = node.wait_for_strict_entity_pose(
                BOOK,
                proof_book_generation,
            )
            proof_base = node.wait_for_strict_entity_pose(
                'tiago_pro',
                proof_base_generation,
            ).planar
            proof_attachment = node.strict_upright_extract_attachment_gate(
                proof_book,
                proof_live_q8,
                proof_base,
            )
            proof_rise_m = float(
                proof_book.position[2] - proof_start_book.position[2]
            )
            proof_rotation_rad = quaternion_distance(
                proof_start_book.quaternion,
                proof_book.quaternion,
            )
            proof_gate_ok = bool(
                proof_attachment.ok
                and proof_rise_m
                >= active_pre_pull_proof_minimum_book_rise_m
                and float(proof_attachment.metrics.get(
                    'book_hand_relative_translation_error_m',
                    math.inf,
                )) <= active_pre_pull_proof_maximum_attachment_translation_m
                and float(proof_attachment.metrics.get(
                    'book_hand_relative_rotation_error_rad',
                    math.inf,
                )) <= active_pre_pull_proof_maximum_attachment_rotation_rad
                and float(proof_attachment.metrics.get(
                    'book_hand_relative_corner_error_m',
                    math.inf,
                )) <= active_pre_pull_proof_maximum_attachment_corner_m
                and proof_rotation_rad
                <= active_pre_pull_proof_maximum_attachment_rotation_rad
            )
            _require(GateResult(
                proof_gate_ok,
                'tighten_direct_pull_proof_lift_verified'
                if proof_gate_ok
                else 'tighten_direct_pull_proof_lift_failed',
                {
                    'book_rise_m': proof_rise_m,
                    'minimum_book_rise_m': (
                        active_pre_pull_proof_minimum_book_rise_m
                    ),
                    'book_rotation_rad': proof_rotation_rad,
                    **proof_attachment.metrics,
                },
            ))
            proof_pressure = pressure_evidence(
                'tighten_direct_pull_proof_endpoint'
            )
            _require(upright_extract_strong_pressure_gate(
                proof_pressure,
                stage='tighten_direct_pull_proof_endpoint',
            ))
            integrated_pre_leg_pressure = proof_pressure
            start_book = proof_book
            start_base = proof_base
            start_gripper = float(proof_pressure[2])
            with node._lock:
                node._strict_payload_base_reference = start_base
                node._strict_payload_gripper_reference = start_gripper
            with node._strict_hazard_lock:
                node._strict_payload_live_reference_force_n = (
                    float(proof_pressure[0]),
                    float(proof_pressure[1]),
                )
            node.set_strict_payload_motion(
                start_book,
                'stationary',
                maximum_rotation_rad=(
                    active_pre_pull_proof_maximum_attachment_rotation_rad
                ),
            )
            integrated_launch_deadline_ns = (
                int(node.get_clock().now().nanoseconds)
                + int(round(1e9 * float(active_integrated_preload_policy[
                    'maximum_launch_delay_s'
                ])))
            )
            require_monitor_clear()

        previous_q8 = active_q8[0]
        previous_book = start_book
        last_pre_leg_pressure: Optional[Tuple[float, float, float]] = None
        last_immediate_endpoint_book: Optional[EntityPose] = None
        for index, target_q8 in enumerate(active_q8[1:], start=1):
            if active_continuous_route_goal:
                if index > 1:
                    break
                # Cross the shelf lip without stop/start dwells.  The complete
                # semantic-anchor route is still retained in ``active_q8`` for
                # the audited shape; this iteration sends all of its bounded
                # subdivisions as one controller goal.
                index = len(active_q8) - 1
                target_q8 = active_q8[-1]
            route_row = active_route[index]
            cumulative_world_delta = route_world_delta(route_row)
            previous_world_delta = route_world_delta(
                active_route[0]
                if active_continuous_route_goal
                else active_route[index - 1]
            )
            leg_world_delta = tuple(
                after - before
                for before, after in zip(
                    previous_world_delta,
                    cumulative_world_delta,
                )
            )
            route_gate_policy = progress_policy(route_row)
            motion_phase = str(route_row['phase'])
            require_monitor_clear()
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                previous_q8,
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='loaded_extraction_start_endpoint_mismatch',
            ))
            require_monitor_clear()
            if integrated_pre_leg_pressure is not None:
                pre_leg_pressure = integrated_pre_leg_pressure
                integrated_pre_leg_pressure = None
                _emit(
                    'pressure_evidence_reused_for_immediate_launch',
                    stage=f'pre_extraction_step_{index}',
                    left_force_n=float(pre_leg_pressure[0]),
                    right_force_n=float(pre_leg_pressure[1]),
                    measured_width_m=float(pre_leg_pressure[2]),
                )
            else:
                pre_leg_pressure = (
                    pressure_evidence(f'pre_extraction_step_{index}')
                    if shelf_probe else (math.nan, math.nan, start_gripper)
                )
            last_pre_leg_pressure = pre_leg_pressure
            if practical_upright_extract:
                _require(upright_extract_strong_pressure_gate(
                    pre_leg_pressure,
                    stage='direct_pull_launch',
                ))
            elif practical_reseat_support:
                launch_pressure_ok = bool(
                    float(pre_leg_pressure[0]) >= 1.25
                    and float(pre_leg_pressure[1]) >= 1.25
                    and float(pre_leg_pressure[0]) + float(pre_leg_pressure[1])
                    >= 3.0
                    and max(
                        float(pre_leg_pressure[0]),
                        float(pre_leg_pressure[1]),
                    ) <= 8.0
                )
                _require(GateResult(
                    launch_pressure_ok,
                    'practical_reseat_launch_pressure_verified'
                    if launch_pressure_ok
                    else 'practical_reseat_launch_pressure_failed',
                    {
                        'left_force_n': float(pre_leg_pressure[0]),
                        'right_force_n': float(pre_leg_pressure[1]),
                        'minimum_each_force_n': 1.25,
                        'minimum_total_force_n': 3.0,
                        'maximum_each_force_n': 8.0,
                    },
                ))
            if active_continuous_force_retention_fraction > 0.0:
                with node._strict_hazard_lock:
                    node._strict_payload_live_reference_force_n = (
                        float(pre_leg_pressure[0]),
                        float(pre_leg_pressure[1]),
                    )
                    node._strict_payload_live_retention_fraction = float(
                        active_continuous_force_retention_fraction
                    )
                require_monitor_clear()
            node.set_strict_payload_motion(
                previous_book,
                motion_phase,
                leg_world_delta,
                (
                    float(active_continuous_relative_rotation_limit_rad)
                    if active_continuous_relative_rotation_limit_rad is not None
                    else (
                        max(
                            0.0015,
                            2.5 * float(route_gate_policy.get(
                                'max_incremental_rotation_rad',
                                PAYLOAD_CONTINUOUS_ROTATION_LIMIT_RAD,
                            )),
                        )
                        if shelf_probe
                        else PAYLOAD_CONTINUOUS_ROTATION_LIMIT_RAD
                    )
                ),
            )

            if active_continuous_route_goal:
                resume_legs = []
                uniform_semantic_duration = (
                    active_continuous_route_duration_s / (len(active_q8) - 1)
                )
                for semantic_index, (first_q8, second_q8) in enumerate(
                    zip(active_q8, active_q8[1:]),
                    start=1,
                ):
                    semantic_duration = (
                        float(active_route[semantic_index][
                            'minimum_duration_s'
                        ])
                        if active_continuous_route_use_row_durations
                        else uniform_semantic_duration
                    )
                    if not math.isfinite(semantic_duration) or semantic_duration <= 0.0:
                        raise RuntimeError(
                            'continuous route semantic duration is invalid'
                        )
                    semantic_dense = dense_arm_waypoints(
                        first_q8,
                        second_q8,
                        maximum_increment=active_dense_maximum_increment_rad,
                    )
                    dense_duration = semantic_duration / len(semantic_dense)
                    resume_legs.extend(
                        (
                            q8,
                            dense_duration,
                            f'loaded_extraction_step_{semantic_index}',
                        )
                        for q8 in semantic_dense
                    )
                dense = tuple(leg[0] for leg in resume_legs)
                legs = tuple(resume_legs)
            else:
                dense = dense_arm_waypoints(
                    previous_q8,
                    target_q8,
                    maximum_increment=active_dense_maximum_increment_rad,
                )
                leg_duration = (
                    max(
                        active_peel_duration,
                        float(route_row.get(
                            'minimum_duration_s',
                            active_peel_duration,
                        )),
                    )
                    if motion_phase in (
                        'diagonal_peel',
                        'shelf_outward',
                        'shelf_inward',
                    )
                    else (
                        MICRO_LIFT_LEG_SIM_SECONDS
                        if motion_phase == 'micro_lift'
                        else OUTWARD_LEG_SIM_SECONDS
                    )
                )
                segment_duration = leg_duration / len(dense)
                legs = tuple(
                    (q8, segment_duration, f'loaded_extraction_step_{index}')
                    for q8 in dense
                )
            previous_generation = _joint_generation_snapshot(node, ARM_JOINTS)
            goal, total_duration = node._make_retained_arm_trajectory_goal(legs)
            if integrated_launch_deadline_ns is not None:
                launch_now_ns = int(node.get_clock().now().nanoseconds)
                maximum_launch_delay_s = float(
                    active_integrated_preload_policy[
                        'maximum_launch_delay_s'
                    ]
                )
                launch_delay_s = (
                    launch_now_ns
                    - (
                        integrated_launch_deadline_ns
                        - int(round(1e9 * maximum_launch_delay_s))
                    )
                ) / 1e9
                _require(GateResult(
                    launch_now_ns <= integrated_launch_deadline_ns,
                    'integrated_preload_immediate_launch_verified'
                    if launch_now_ns <= integrated_launch_deadline_ns
                    else 'integrated_preload_immediate_launch_deadline_missed',
                    {
                        'post_preload_launch_delay_s': launch_delay_s,
                        'maximum_launch_delay_s': maximum_launch_delay_s,
                    },
                ))
                integrated_launch_deadline_ns = None
            moved, payload_lost = node._send_retained_arm_trajectory(
                goal,
                total_duration,
                legs,
                'strict_held_extraction',
            )
            if not moved:
                hazard_fields = node.strict_last_payload_hazard()
                hazard_reason = str(hazard_fields.get('reason', ''))
                observed_retention_loss = hazard_reason.startswith((
                    'payload_pressure:',
                    'payload_left_force_below_absolute_minimum',
                    'payload_right_force_below_absolute_minimum',
                    'payload_bilateral_force_',
                    'payload_robot_contact',
                    'contact_lost',
                ))
                _emit(
                    'loaded_extraction_stopped',
                    step=index,
                    payload_lost=bool(payload_lost and observed_retention_loss),
                    safety_gate_stopped=bool(payload_lost),
                    **node.strict_payload_observed_metrics(),
                    **hazard_fields,
                )
                raise RuntimeError(
                    'payload_retention_lost_during_loaded_extraction'
                    if payload_lost and observed_retention_loss
                    else (
                        f'loaded_extraction_safety_stop:{hazard_reason}'
                        if payload_lost
                        else 'loaded_extraction_trajectory_failed'
                    )
                )
            measured = node._wait_for_retained_endpoint(
                target_q8,
                command='strict_held_extraction',
                phase='loaded_extraction',
                leg=index,
                settle_timeout=2.0,
                torso_tolerance=active_checkpoint_arm_limit_rad,
                arm_tolerance=active_checkpoint_arm_limit_rad,
            )
            if measured is None:
                raise RuntimeError('loaded_extraction_endpoint_not_verified')
            _wait_for_joint_update(
                node,
                ARM_JOINTS,
                previous_generation,
                timeout=2.0,
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                target_q8,
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='loaded_extraction_endpoint_mismatch',
            ))
            if not node._wait_sim_duration(0.15):
                raise RuntimeError('loaded extraction endpoint settle interrupted')
            book_generation = node.strict_entity_generation(BOOK)
            base_generation = node.strict_entity_generation('tiago_pro')
            observed_book = node.wait_for_strict_entity_pose(
                BOOK,
                book_generation,
            )
            observed_base = node.wait_for_strict_entity_pose(
                'tiago_pro',
                base_generation,
            ).planar
            if (
                practical_full_pull_resume
                and active_resume_progress_mode == 'upright_extract'
            ):
                progress_result = practical_upright_extract_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    support_floor_world_z_m=active_support_floor_world_z,
                    policy=active_endpoint_policy,
                )
                _require(node.strict_upright_extract_attachment_gate(
                    observed_book,
                    _joint_snapshot(node, IK_JOINTS),
                    observed_base,
                ))
            elif (
                practical_full_pull_resume
                and active_resume_progress_mode == 'reseat_support'
            ):
                progress_result = reseat_support_progress(
                    observed_book,
                    route_row,
                )
            elif (
                practical_full_pull_resume
                and active_resume_progress_mode == 'current_ray_release'
            ):
                if active_radial_unit_world is None:
                    raise RuntimeError('current-ray release axis unavailable')
                progress_result = practical_current_ray_release_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    radial_unit_world=active_radial_unit_world,
                    support_floor_world_z_m=active_support_floor_world_z,
                    policy=route_gate_policy,
                )
            elif (
                practical_full_pull_resume
                and active_resume_progress_mode == 'fixed_attitude_detach'
            ):
                progress_result = practical_fixed_attitude_detach_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    support_floor_world_z_m=active_support_floor_world_z,
                    policy=route_gate_policy,
                )
            elif (
                practical_full_pull_resume
                and active_resume_progress_mode == 'lip_roll'
            ):
                progress_result = practical_lip_roll_progress_gate(
                    start_book,
                    observed_book,
                    expected_position_world_m=route_row[
                        'expected_book_position_world_m'
                    ],
                    expected_quaternion_xyzw=route_row[
                        'expected_book_quaternion_xyzw'
                    ],
                    expected_signed_world_y_rotation_rad=float(route_row[
                        'expected_signed_world_y_rotation_rad'
                    ]),
                    support_floor_world_z_m=active_support_floor_world_z,
                    minimum_deep_edge_signed_m=float(route_row[
                        'minimum_deep_edge_signed_m'
                    ]),
                    policy=route_gate_policy,
                )
            elif (
                practical_full_pull_resume
                and active_resume_progress_mode == 'lip_tangent'
            ):
                progress_result = lip_tangent_progress_gate(
                    start_book,
                previous_book,
                observed_book,
                support_floor_world_z_m=active_support_floor_world_z,
                final=bool(route_row.get(
                    'require_lip_unload_progress',
                    index == len(active_q8) - 1,
                )),
                minimum_nominal_improvement_rad=float(route_row.get(
                    'minimum_nominal_improvement_rad',
                    0.010,
                )),
                minimum_z_improvement_m=float(route_row.get(
                    'minimum_z_improvement_m',
                    0.001,
                )),
                maximum_step_rotation_rad=float(
                    active_continuous_relative_rotation_limit_rad
                ),
                maximum_final_nominal_error_rad=float(route_row.get(
                    'maximum_final_nominal_error_rad',
                    math.inf,
                )),
                minimum_final_remaining_world_y_rad=float(route_row.get(
                    'minimum_final_remaining_world_y_rad',
                    -math.inf,
                )),
                maximum_final_remaining_world_y_rad=float(route_row.get(
                    'maximum_final_remaining_world_y_rad',
                    math.inf,
                )),
            )
            elif (
                practical_full_pull_resume
                and active_resume_progress_mode == 'radial_unload'
            ):
                if active_radial_unit_world is None:
                    raise RuntimeError('radial unload axis unavailable')
                progress_result = practical_radial_unload_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    radial_unit_world=active_radial_unit_world,
                    support_floor_world_z_m=active_support_floor_world_z,
                    policy=route_gate_policy,
                )
            elif practical_full_pull_resume:
                progress_result = practical_outward_slide_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    support_floor_world_z_m=active_support_floor_world_z,
                    policy=route_gate_policy,
                )
            elif practical_full_pull:
                progress_result = practical_full_pull_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    support_floor_world_z_m=active_support_floor_world_z,
                    final=index == len(active_q8) - 1,
                    maximum_position_error_m=(
                        active_practical_maximum_position_error_m
                    ),
                    maximum_lateral_error_m=(
                        active_practical_maximum_lateral_error_m
                    ),
                    maximum_vertical_error_m=(
                        active_practical_maximum_vertical_error_m
                    ),
                    maximum_rotation_rad=(
                        active_practical_maximum_rotation_rad
                    ),
                    minimum_floor_signed_m=(
                        active_practical_minimum_floor_signed_m
                    ),
                    minimum_final_shelf_clearance_m=(
                        active_practical_minimum_final_shelf_clearance_m
                    ),
                    minimum_observed_vertical_progress_m=(
                        active_practical_minimum_vertical_progress_m
                    ),
                )
            elif halfmillimeter_outward:
                if active_checkpoint_stable_reference is None:
                    raise RuntimeError(
                        'half-millimeter stable reference unavailable'
                    )
                progress_result = micro_outward_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    support_floor_world_z_m=active_support_floor_world_z,
                    stable_reference=active_checkpoint_stable_reference,
                    policy=route_gate_policy,
                )
            elif reverse_reseat:
                if active_reseat_reference is None:
                    raise RuntimeError('reverse reseat reference unavailable')
                progress_result = inward_reseat_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    support_floor_world_z_m=active_support_floor_world_z,
                    stable_reference=active_reseat_reference,
                    policy=route_gate_policy,
                )
            elif shelf_probe:
                progress_result = diagonal_peel_progress_gate(
                    start_book,
                    observed_book,
                    planned_delta_world_m=cumulative_world_delta,
                    support_floor_world_z_m=active_support_floor_world_z,
                    policy=route_gate_policy,
                )
            else:
                progress_result = extraction_progress_gate(
                    start_book,
                    previous_book,
                    observed_book,
                    index,
                )
            _require(progress_result)
            last_immediate_endpoint_book = observed_book
            node.set_strict_payload_motion(observed_book, 'stationary')
            post_leg_pressure = pressure_evidence(
                f'post_extraction_step_{index}'
            )
            if practical_upright_extract:
                _require(upright_extract_strong_pressure_gate(
                    post_leg_pressure,
                    stage='direct_pull_immediate_endpoint',
                ))
            if active_minimum_force_retention_fraction > 0.0:
                _require(bilateral_pressure_retention_gate(
                    pre_leg_pressure[:2],
                    post_leg_pressure[:2],
                    minimum_each_side_retention_fraction=(
                        active_minimum_force_retention_fraction
                    ),
                    maximum_normalized_balance_change=(
                        active_maximum_force_balance_change
                    ),
                ))
            _require(expected_joint_gate(
                (math.hypot(observed_base.x - start_base.x,
                            observed_base.y - start_base.y),),
                (0.0,),
                limit=PAYLOAD_BASE_POSITION_LIMIT_M,
                failure_reason='base_moved_during_loaded_extraction',
            ))
            require_monitor_clear()
            progress_metrics = dict(progress_result.metrics)
            progress_metrics.pop('step', None)
            _emit(
                'loaded_extraction_endpoint',
                step=index,
                phase=motion_phase,
                cumulative_lift_m=float(
                    route_row.get(
                        'lift_world_z_m',
                        route_row.get('world_delta_m', (0.0, 0.0, 0.0))[2],
                    )
                ),
                cumulative_outward_m=float(
                    route_row.get(
                        'outward_world_minus_x_m',
                        -route_row.get('world_delta_m', (0.0, 0.0, 0.0))[0],
                    )
                ),
                route_steps=len(active_q8) - 1,
                book_position=list(observed_book.position),
                book_quaternion=list(observed_book.quaternion),
                measured_q8=[float(value) for value in measured],
                **progress_metrics,
            )
            previous_q8 = target_q8
            previous_book = observed_book
            if (
                active_pause_every_endpoint
                and index < len(active_q8) - 1
            ):
                passive_names = (
                    *RIGHT_ARM_JOINTS,
                    *RIGHT_GRIPPER_JOINTS,
                    *HEAD_JOINTS,
                )
                monitor_suspended.set()
                if not monitor_quiesced.wait(timeout=1.0):
                    raise RuntimeError(
                        'passive monitor did not quiesce before Gazebo pause'
                    )
                node._payload_monitor_enabled = False
                pause_or_raise()
                paused_endpoint_message = read_target_base_dynamic_pose_message()
                paused_endpoint_book = entity_pose_from_dynamic_pose(
                    paused_endpoint_message,
                    BOOK,
                )
                paused_endpoint_base = entity_pose_from_dynamic_pose(
                    paused_endpoint_message,
                    'tiago_pro',
                ).planar
                if (
                    practical_full_pull_resume
                    and active_resume_progress_mode == 'reseat_support'
                ):
                    _require(reseat_support_progress(
                        paused_endpoint_book,
                        route_row,
                    ))
                elif (
                    practical_full_pull_resume
                    and active_resume_progress_mode == 'current_ray_release'
                ):
                    if active_radial_unit_world is None:
                        raise RuntimeError('current-ray release axis unavailable')
                    _require(practical_current_ray_release_progress_gate(
                        start_book,
                        paused_endpoint_book,
                        planned_delta_world_m=cumulative_world_delta,
                        radial_unit_world=active_radial_unit_world,
                        support_floor_world_z_m=active_support_floor_world_z,
                        policy=route_gate_policy,
                    ))
                elif (
                    practical_full_pull_resume
                    and active_resume_progress_mode == 'fixed_attitude_detach'
                ):
                    _require(practical_fixed_attitude_detach_progress_gate(
                        start_book,
                        paused_endpoint_book,
                        planned_delta_world_m=cumulative_world_delta,
                        support_floor_world_z_m=active_support_floor_world_z,
                        policy=route_gate_policy,
                    ))
                elif (
                    practical_full_pull_resume
                    and active_resume_progress_mode == 'lip_roll'
                ):
                    _require(practical_lip_roll_progress_gate(
                        start_book,
                        paused_endpoint_book,
                        expected_position_world_m=route_row[
                            'expected_book_position_world_m'
                        ],
                        expected_quaternion_xyzw=route_row[
                            'expected_book_quaternion_xyzw'
                        ],
                        expected_signed_world_y_rotation_rad=float(route_row[
                            'expected_signed_world_y_rotation_rad'
                        ]),
                        support_floor_world_z_m=active_support_floor_world_z,
                        minimum_deep_edge_signed_m=float(route_row[
                            'minimum_deep_edge_signed_m'
                        ]),
                        policy=route_gate_policy,
                    ))
                elif (
                    practical_full_pull_resume
                    and active_resume_progress_mode == 'lip_tangent'
                ):
                    _require(lip_tangent_progress_gate(
                        start_book,
                        previous_book,
                        paused_endpoint_book,
                        support_floor_world_z_m=active_support_floor_world_z,
                        final=bool(route_row.get(
                            'require_lip_unload_progress',
                            False,
                        )),
                        minimum_nominal_improvement_rad=float(route_row.get(
                            'minimum_nominal_improvement_rad',
                            0.010,
                        )),
                        minimum_z_improvement_m=float(route_row.get(
                            'minimum_z_improvement_m',
                            0.001,
                        )),
                        maximum_step_rotation_rad=float(
                            active_continuous_relative_rotation_limit_rad
                        ),
                        maximum_final_nominal_error_rad=float(route_row.get(
                            'maximum_final_nominal_error_rad',
                            math.inf,
                        )),
                        minimum_final_remaining_world_y_rad=float(route_row.get(
                            'minimum_final_remaining_world_y_rad',
                            -math.inf,
                        )),
                        maximum_final_remaining_world_y_rad=float(route_row.get(
                            'maximum_final_remaining_world_y_rad',
                            math.inf,
                        )),
                    ))
                elif (
                    practical_full_pull_resume
                    and active_resume_progress_mode == 'radial_unload'
                ):
                    if active_radial_unit_world is None:
                        raise RuntimeError('radial unload axis unavailable')
                    _require(practical_radial_unload_progress_gate(
                        start_book,
                        paused_endpoint_book,
                        planned_delta_world_m=cumulative_world_delta,
                        radial_unit_world=active_radial_unit_world,
                        support_floor_world_z_m=active_support_floor_world_z,
                        policy=route_gate_policy,
                    ))
                elif practical_full_pull_resume:
                    _require(practical_outward_slide_progress_gate(
                        start_book,
                        paused_endpoint_book,
                        planned_delta_world_m=cumulative_world_delta,
                        support_floor_world_z_m=active_support_floor_world_z,
                        policy=route_gate_policy,
                    ))
                elif halfmillimeter_outward:
                    if active_checkpoint_stable_reference is None:
                        raise RuntimeError(
                            'half-millimeter stable reference unavailable'
                        )
                    _require(micro_outward_progress_gate(
                        start_book,
                        paused_endpoint_book,
                        planned_delta_world_m=cumulative_world_delta,
                        support_floor_world_z_m=(
                            active_support_floor_world_z
                        ),
                        stable_reference=(
                            active_checkpoint_stable_reference
                        ),
                        policy=route_gate_policy,
                    ))
                elif reverse_reseat:
                    if active_reseat_reference is None:
                        raise RuntimeError(
                            'reverse reseat reference unavailable'
                        )
                    _require(inward_reseat_progress_gate(
                        start_book,
                        paused_endpoint_book,
                        planned_delta_world_m=cumulative_world_delta,
                        support_floor_world_z_m=active_support_floor_world_z,
                        stable_reference=active_reseat_reference,
                        policy=route_gate_policy,
                    ))
                else:
                    _require(diagonal_peel_progress_gate(
                        start_book,
                        paused_endpoint_book,
                        planned_delta_world_m=cumulative_world_delta,
                        support_floor_world_z_m=active_support_floor_world_z,
                        policy=route_gate_policy,
                    ))
                if (
                    math.hypot(
                        paused_endpoint_base.x - start_base.x,
                        paused_endpoint_base.y - start_base.y,
                    ) > PAYLOAD_BASE_POSITION_LIMIT_M
                ):
                    raise RuntimeError(
                        'base_moved_at_paused_shelf_step_endpoint'
                    )
                _emit(
                    'paused_shelf_step_endpoint',
                    step=index,
                    route_steps=len(active_q8) - 1,
                    book_position=list(paused_endpoint_book.position),
                    book_quaternion=list(paused_endpoint_book.quaternion),
                    gazebo_paused=True,
                )
                # Snapshot only after the monitor has quiesced and the world is
                # confirmed paused.  A pre-pause baseline can be overtaken by a
                # final in-flight joint-state sample and would let that stale
                # sample satisfy the post-unpause freshness check.
                passive_generations = _joint_generation_snapshot(
                    node,
                    passive_names,
                )
                # Be pessimistic across the service call, matching the initial
                # unpause path: interruption anywhere below must re-pause.
                world_unpaused = True
                if not set_world_paused_confirmed(False):
                    raise RuntimeError(
                        'Gazebo world re-unpause was not confirmed'
                    )
                resume_generation = node.strict_entity_generation(BOOK)
                if not node._wait_sim_duration(0.05):
                    require_monitor_clear()
                    raise RuntimeError(
                        'intermediate shelf-step resume was interrupted'
                    )
                resumed_book = node.wait_for_strict_entity_pose(
                    BOOK,
                    resume_generation,
                )
                _wait_for_joint_update(
                    node,
                    passive_names,
                    passive_generations,
                    timeout=2.0,
                )
                _require(held_stability_gate(
                    paused_endpoint_book,
                    resumed_book,
                ))
                previous_book = resumed_book
                node.set_strict_payload_motion(previous_book, 'stationary')
                _require(stationary_joint_gate(
                    right_reference,
                    _joint_snapshot(node, RIGHT_ARM_JOINTS),
                ))
                _require(stationary_joint_gate(
                    right_gripper_reference,
                    _joint_snapshot(node, RIGHT_GRIPPER_JOINTS),
                    limit=PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
                ))
                _require(stationary_joint_gate(
                    head_reference,
                    _joint_snapshot(node, HEAD_JOINTS),
                ))
                monitor_suspended.clear()
                node._payload_monitor_enabled = True
                require_monitor_clear()

        final_before_dwell = node.strict_entity_pose(BOOK)
        if not node._wait_sim_duration(active_final_dwell_s):
            require_monitor_clear()
            raise RuntimeError('final loaded extraction stability dwell interrupted')
        final_generation = node.strict_entity_generation(BOOK)
        final_book = node.wait_for_strict_entity_pose(BOOK, final_generation)
        _require(held_stability_gate(
            final_before_dwell,
            final_book,
            translation_limit_m=active_final_stability_translation_limit_m,
            rotation_limit_rad=active_final_stability_rotation_limit_rad,
        ))
        if (
            practical_full_pull_resume
            and active_resume_progress_mode == 'upright_extract'
        ):
            final_geometry = practical_upright_extract_progress_gate(
                start_book,
                final_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                support_floor_world_z_m=active_support_floor_world_z,
                policy=active_endpoint_policy,
            )
            final_q8 = _joint_snapshot(node, IK_JOINTS)
            final_base_pose = node.strict_entity_pose('tiago_pro').planar
            _require(node.strict_upright_extract_attachment_gate(
                final_book,
                final_q8,
                final_base_pose,
            ))
            _require(expected_joint_gate(
                final_q8,
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_upright_extract_final_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'reseat_support'
        ):
            final_geometry = reseat_support_progress(
                final_book,
                active_route[-1],
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_reseat_final_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'current_ray_release'
        ):
            if active_radial_unit_world is None:
                raise RuntimeError('current-ray release axis unavailable')
            final_geometry = practical_current_ray_release_progress_gate(
                start_book,
                final_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                radial_unit_world=active_radial_unit_world,
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_current_ray_final_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'fixed_attitude_detach'
        ):
            final_geometry = practical_fixed_attitude_detach_progress_gate(
                start_book,
                final_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_detach_final_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'lip_roll'
        ):
            final_row = active_route[-1]
            final_geometry = practical_lip_roll_progress_gate(
                start_book,
                final_book,
                expected_position_world_m=final_row[
                    'expected_book_position_world_m'
                ],
                expected_quaternion_xyzw=final_row[
                    'expected_book_quaternion_xyzw'
                ],
                expected_signed_world_y_rotation_rad=float(final_row[
                    'expected_signed_world_y_rotation_rad'
                ]),
                support_floor_world_z_m=active_support_floor_world_z,
                minimum_deep_edge_signed_m=float(final_row[
                    'minimum_deep_edge_signed_m'
                ]),
                policy=progress_policy(final_row),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_lip_roll_final_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'lip_tangent'
        ):
            final_geometry = lip_tangent_progress_gate(
                start_book,
                final_before_dwell,
                final_book,
                support_floor_world_z_m=active_support_floor_world_z,
                final=True,
                minimum_nominal_improvement_rad=float(
                    active_route[-1].get(
                        'minimum_nominal_improvement_rad',
                        0.010,
                    )
                ),
                minimum_z_improvement_m=float(active_route[-1].get(
                    'minimum_z_improvement_m',
                    0.001,
                )),
                maximum_step_rotation_rad=float(
                    active_continuous_relative_rotation_limit_rad
                ),
                maximum_final_nominal_error_rad=float(
                    active_route[-1].get(
                        'maximum_final_nominal_error_rad',
                        math.inf,
                    )
                ),
                minimum_final_remaining_world_y_rad=float(
                    active_route[-1].get(
                        'minimum_final_remaining_world_y_rad',
                        -math.inf,
                    )
                ),
                maximum_final_remaining_world_y_rad=float(
                    active_route[-1].get(
                        'maximum_final_remaining_world_y_rad',
                        math.inf,
                    )
                ),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_lip_tangent_final_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'radial_unload'
        ):
            if active_radial_unit_world is None:
                raise RuntimeError('radial unload axis unavailable')
            final_geometry = practical_radial_unload_progress_gate(
                start_book,
                final_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                radial_unit_world=active_radial_unit_world,
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_radial_unload_final_joint_mismatch',
            ))
        elif practical_full_pull_resume:
            final_geometry = practical_outward_slide_progress_gate(
                start_book,
                final_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_outward_slide_final_joint_mismatch',
            ))
        elif practical_full_pull:
            final_geometry = practical_full_pull_progress_gate(
                start_book,
                final_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                support_floor_world_z_m=active_support_floor_world_z,
                final=True,
                maximum_position_error_m=(
                    active_practical_maximum_position_error_m
                ),
                maximum_lateral_error_m=(
                    active_practical_maximum_lateral_error_m
                ),
                maximum_vertical_error_m=(
                    active_practical_maximum_vertical_error_m
                ),
                maximum_rotation_rad=active_practical_maximum_rotation_rad,
                minimum_floor_signed_m=(
                    active_practical_minimum_floor_signed_m
                ),
                minimum_final_shelf_clearance_m=(
                    active_practical_minimum_final_shelf_clearance_m
                ),
                minimum_observed_vertical_progress_m=(
                    active_practical_minimum_vertical_progress_m
                ),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_full_pull_final_joint_mismatch',
            ))
        elif halfmillimeter_outward:
            if active_checkpoint_stable_reference is None:
                raise RuntimeError(
                    'half-millimeter stable reference unavailable'
                )
            final_geometry = micro_outward_progress_gate(
                start_book,
                final_book,
                planned_delta_world_m=active_peel_delta,
                support_floor_world_z_m=active_support_floor_world_z,
                stable_reference=active_checkpoint_stable_reference,
                policy=(
                    active_final_endpoint_policy
                    if active_final_endpoint_policy is not None
                    else progress_policy(active_route[-1])
                ),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='halfmillimeter_final_joint_mismatch',
            ))
        elif halfmillimeter_midroute_settle:
            if (
                active_checkpoint_stable_reference is None
                or active_supported_book_final_policy is None
            ):
                raise RuntimeError('midroute settle policy unavailable')
            final_geometry = supported_book_state_gate(
                final_book,
                active_checkpoint_stable_reference,
                support_floor_world_z_m=active_support_floor_world_z,
                **active_supported_book_final_policy,
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='midroute_settle_left_arm_moved',
            ))
            _require(expected_joint_gate(
                (_joint_snapshot(node, LEFT_GRIPPER_JOINTS)[0],),
                (active_checkpoint_gripper_m,),
                limit=active_checkpoint_gripper_limit_m,
                failure_reason='midroute_settle_gripper_moved',
            ))
        elif reverse_reseat:
            if active_reseat_reference is None:
                raise RuntimeError('reverse reseat reference unavailable')
            final_geometry = inward_reseat_progress_gate(
                start_book,
                final_book,
                planned_delta_world_m=active_reseat_planned_delta,
                support_floor_world_z_m=active_support_floor_world_z,
                stable_reference=active_reseat_reference,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=CHECKPOINT_ARM_LIMIT_RAD,
                failure_reason='reverse_reseat_final_joint_mismatch',
            ))
        elif post_reseat_settle:
            if active_reseat_reference is None:
                raise RuntimeError('post-reseat stable reference unavailable')
            final_geometry = stable_reseat_state_gate(
                final_book,
                stable_reference=active_reseat_reference,
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='post_reseat_settle_left_arm_moved',
            ))
            _require(expected_joint_gate(
                (_joint_snapshot(node, LEFT_GRIPPER_JOINTS)[0],),
                (active_checkpoint_gripper_m,),
                limit=active_checkpoint_gripper_limit_m,
                failure_reason='post_reseat_settle_gripper_moved',
            ))
        elif shelf_probe:
            final_route_row = active_route[-1]
            geometry_reference = (
                active_settle_reference
                if active_settle_reference is not None
                else start_book
            )
            geometry_delta = (
                active_settle_planned_delta
                if active_settle_reference is not None
                else route_world_delta(final_route_row)
            )
            final_geometry = diagonal_peel_progress_gate(
                geometry_reference,
                final_book,
                planned_delta_world_m=geometry_delta,
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(final_route_row),
            )
        else:
            final_geometry = final_extraction_geometry_gate(
                start_book,
                final_book,
            )
        if active_settle_reference is not None:
            rotation_growth_start = (
                last_immediate_endpoint_book
                if (
                    halfmillimeter_outward
                    and last_immediate_endpoint_book is not None
                )
                else final_before_dwell
            )
            rotation_before_settle = quaternion_distance(
                active_settle_reference.quaternion,
                rotation_growth_start.quaternion,
            )
            rotation_after_settle = quaternion_distance(
                active_settle_reference.quaternion,
                final_book.quaternion,
            )
            rotation_growth = rotation_after_settle - rotation_before_settle
            _require(GateResult(
                rotation_growth <= active_settle_rotation_growth_limit_rad,
                'settle_rotation_growth_verified'
                if rotation_growth <= active_settle_rotation_growth_limit_rad
                else 'settle_rotation_growth_exceeded_limit',
                {
                    'rotation_before_settle_rad': rotation_before_settle,
                    'rotation_after_settle_rad': rotation_after_settle,
                    'rotation_growth_rad': rotation_growth,
                    'rotation_growth_limit_rad': (
                        active_settle_rotation_growth_limit_rad
                    ),
                },
            ))
        _require(final_geometry)
        final_pressure = pressure_evidence('post_extraction_stability')
        if practical_upright_extract:
            _require(upright_extract_strong_pressure_gate(
                final_pressure,
                stage='direct_pull_final',
            ))
        if active_minimum_force_retention_fraction > 0.0:
            retention_reference_pressure = (
                last_pre_leg_pressure
                if last_pre_leg_pressure is not None
                else resume_transport_pressure
            )
            _require(bilateral_pressure_retention_gate(
                retention_reference_pressure[:2],
                final_pressure[:2],
                minimum_each_side_retention_fraction=(
                    active_minimum_force_retention_fraction
                ),
                maximum_normalized_balance_change=(
                    active_maximum_force_balance_change
                ),
            ))
        if active_minimum_force_n != (0.0, 0.0):
            minimum_left, minimum_right = active_minimum_force_n
            forces_ok = bool(
                final_pressure[0] >= minimum_left
                and final_pressure[1] >= minimum_right
                and final_pressure[0] <= active_maximum_force_n[0]
                and final_pressure[1] <= active_maximum_force_n[1]
                and final_pressure[0] + final_pressure[1]
                >= active_minimum_total_force_n
            )
            _require(GateResult(
                forces_ok,
                'settle_absolute_pressure_verified'
                if forces_ok else 'settle_absolute_pressure_out_of_bounds',
                {
                    'left_force_n': final_pressure[0],
                    'right_force_n': final_pressure[1],
                    'minimum_left_force_n': minimum_left,
                    'minimum_right_force_n': minimum_right,
                    'minimum_total_force_n': active_minimum_total_force_n,
                    'maximum_left_force_n': active_maximum_force_n[0],
                    'maximum_right_force_n': active_maximum_force_n[1],
                },
            ))
        require_monitor_clear()
        monitor_suspended.set()
        if not monitor_quiesced.wait(timeout=1.0):
            raise RuntimeError(
                'passive monitor did not quiesce before final Gazebo pause'
            )
        node._payload_monitor_enabled = False
        pause_or_raise()
        require_monitor_clear()
        paused_joint_names = tuple(dict.fromkeys((
            *IK_JOINTS,
            *LEFT_GRIPPER_JOINTS,
            *RIGHT_ARM_JOINTS,
            *RIGHT_GRIPPER_JOINTS,
            *HEAD_JOINTS,
        )))
        paused_joint_values, paused_state_stats = (
            node.paused_serialized_joint_positions(paused_joint_names)
        )
        if not paused_state_stats[2]:
            raise RuntimeError('serialized endpoint state was not paused')

        def paused_joint_snapshot(names: Sequence[str]) -> Tuple[float, ...]:
            return tuple(float(paused_joint_values[name]) for name in names)

        assert (
            right_reference is not None
            and right_gripper_reference is not None
            and head_reference is not None
        )
        _require(stationary_joint_gate(
            right_reference,
            paused_joint_snapshot(RIGHT_ARM_JOINTS),
        ))
        _require(stationary_joint_gate(
            right_gripper_reference,
            paused_joint_snapshot(RIGHT_GRIPPER_JOINTS),
            limit=PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
        ))
        _require(stationary_joint_gate(
            head_reference,
            paused_joint_snapshot(HEAD_JOINTS),
        ))
        _require(expected_joint_gate(
            (paused_joint_snapshot(LEFT_GRIPPER_JOINTS)[0],),
            (start_gripper,),
            limit=active_checkpoint_gripper_limit_m,
            failure_reason='paused_left_gripper_aperture_mismatch',
        ))
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=0.25)

        # Certify the state that is actually left for the next stage, after the
        # pause has taken effect.  ROS time is frozen, so the last force and
        # contact samples remain age-valid; Gazebo's paused pose stream gives
        # an independent final geometry snapshot.
        require_monitor_clear()
        paused_evidence, paused_model, paused_identity = (
            node._adaptive_pressure_evidence(
                minimum_width=PRESSURE_LOCK_MINIMUM_WIDTH_M,
                baseline_effort=math.nan,
            )
        )
        if not (
            paused_evidence.verified
            and paused_model == BOOK
            and paused_identity is None
            and not node.target_robot_contact_latched()
        ):
            raise RuntimeError('paused endpoint pressure was not verified')
        if practical_upright_extract:
            _require(upright_extract_strong_pressure_gate(
                (
                    float(paused_evidence.left_force),
                    float(paused_evidence.right_force),
                ),
                stage='direct_pull_paused',
            ))
        if active_minimum_force_retention_fraction > 0.0:
            retention_reference_pressure = (
                last_pre_leg_pressure
                if last_pre_leg_pressure is not None
                else resume_transport_pressure
            )
            _require(bilateral_pressure_retention_gate(
                retention_reference_pressure[:2],
                (
                    float(paused_evidence.left_force),
                    float(paused_evidence.right_force),
                ),
                minimum_each_side_retention_fraction=(
                    active_minimum_force_retention_fraction
                ),
                maximum_normalized_balance_change=(
                    active_maximum_force_balance_change
                ),
            ))
        paused_message = read_target_base_dynamic_pose_message()
        paused_book = entity_pose_from_dynamic_pose(paused_message, BOOK)
        paused_base = entity_pose_from_dynamic_pose(paused_message, 'tiago_pro')
        if (
            settle_resample
            or reverse_reseat
            or post_reseat_settle
            or halfmillimeter_outward
            or halfmillimeter_midroute_settle
        ):
            paused_books = book_poses_from_dynamic_pose(
                read_exact_seed101_dynamic_pose_message()
            )
            _require(resume_seed101_scene_gate(first_books, paused_books))
        if (
            practical_full_pull_resume
            and active_resume_progress_mode == 'upright_extract'
        ):
            paused_geometry = practical_upright_extract_progress_gate(
                start_book,
                paused_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                support_floor_world_z_m=active_support_floor_world_z,
                policy=active_endpoint_policy,
            )
            paused_q8 = paused_joint_snapshot(IK_JOINTS)
            _require(node.strict_upright_extract_attachment_gate(
                paused_book,
                paused_q8,
                paused_base.planar,
            ))
            _require(expected_joint_gate(
                paused_q8,
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_upright_extract_paused_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'reseat_support'
        ):
            paused_geometry = reseat_support_progress(
                paused_book,
                active_route[-1],
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_reseat_paused_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'current_ray_release'
        ):
            if active_radial_unit_world is None:
                raise RuntimeError('current-ray release axis unavailable')
            paused_geometry = practical_current_ray_release_progress_gate(
                start_book,
                paused_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                radial_unit_world=active_radial_unit_world,
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_current_ray_paused_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'fixed_attitude_detach'
        ):
            paused_geometry = practical_fixed_attitude_detach_progress_gate(
                start_book,
                paused_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_detach_paused_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'lip_roll'
        ):
            final_row = active_route[-1]
            paused_geometry = practical_lip_roll_progress_gate(
                start_book,
                paused_book,
                expected_position_world_m=final_row[
                    'expected_book_position_world_m'
                ],
                expected_quaternion_xyzw=final_row[
                    'expected_book_quaternion_xyzw'
                ],
                expected_signed_world_y_rotation_rad=float(final_row[
                    'expected_signed_world_y_rotation_rad'
                ]),
                support_floor_world_z_m=active_support_floor_world_z,
                minimum_deep_edge_signed_m=float(final_row[
                    'minimum_deep_edge_signed_m'
                ]),
                policy=progress_policy(final_row),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_lip_roll_paused_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'lip_tangent'
        ):
            paused_geometry = lip_tangent_progress_gate(
                start_book,
                final_book,
                paused_book,
                support_floor_world_z_m=active_support_floor_world_z,
                final=True,
                minimum_nominal_improvement_rad=float(
                    active_route[-1].get(
                        'minimum_nominal_improvement_rad',
                        0.010,
                    )
                ),
                minimum_z_improvement_m=float(active_route[-1].get(
                    'minimum_z_improvement_m',
                    0.001,
                )),
                maximum_step_rotation_rad=float(
                    active_continuous_relative_rotation_limit_rad
                ),
                maximum_final_nominal_error_rad=float(
                    active_route[-1].get(
                        'maximum_final_nominal_error_rad',
                        math.inf,
                    )
                ),
                minimum_final_remaining_world_y_rad=float(
                    active_route[-1].get(
                        'minimum_final_remaining_world_y_rad',
                        -math.inf,
                    )
                ),
                maximum_final_remaining_world_y_rad=float(
                    active_route[-1].get(
                        'maximum_final_remaining_world_y_rad',
                        math.inf,
                    )
                ),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_lip_tangent_paused_joint_mismatch',
            ))
        elif (
            practical_full_pull_resume
            and active_resume_progress_mode == 'radial_unload'
        ):
            if active_radial_unit_world is None:
                raise RuntimeError('radial unload axis unavailable')
            paused_geometry = practical_radial_unload_progress_gate(
                start_book,
                paused_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                radial_unit_world=active_radial_unit_world,
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_radial_unload_paused_joint_mismatch',
            ))
        elif practical_full_pull_resume:
            paused_geometry = practical_outward_slide_progress_gate(
                start_book,
                paused_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_outward_slide_paused_joint_mismatch',
            ))
        elif practical_full_pull:
            paused_geometry = practical_full_pull_progress_gate(
                start_book,
                paused_book,
                planned_delta_world_m=route_world_delta(active_route[-1]),
                support_floor_world_z_m=active_support_floor_world_z,
                final=True,
                maximum_position_error_m=(
                    active_practical_maximum_position_error_m
                ),
                maximum_lateral_error_m=(
                    active_practical_maximum_lateral_error_m
                ),
                maximum_vertical_error_m=(
                    active_practical_maximum_vertical_error_m
                ),
                maximum_rotation_rad=active_practical_maximum_rotation_rad,
                minimum_floor_signed_m=(
                    active_practical_minimum_floor_signed_m
                ),
                minimum_final_shelf_clearance_m=(
                    active_practical_minimum_final_shelf_clearance_m
                ),
                minimum_observed_vertical_progress_m=(
                    active_practical_minimum_vertical_progress_m
                ),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='practical_full_pull_paused_joint_mismatch',
            ))
        elif halfmillimeter_outward:
            if active_checkpoint_stable_reference is None:
                raise RuntimeError(
                    'half-millimeter stable reference unavailable'
                )
            paused_geometry = micro_outward_progress_gate(
                start_book,
                paused_book,
                planned_delta_world_m=active_peel_delta,
                support_floor_world_z_m=active_support_floor_world_z,
                stable_reference=active_checkpoint_stable_reference,
                policy=(
                    active_final_endpoint_policy
                    if active_final_endpoint_policy is not None
                    else progress_policy(active_route[-1])
                ),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='halfmillimeter_paused_joint_mismatch',
            ))
        elif halfmillimeter_midroute_settle:
            if (
                active_checkpoint_stable_reference is None
                or active_supported_book_final_policy is None
            ):
                raise RuntimeError('midroute settle policy unavailable')
            paused_geometry = supported_book_state_gate(
                paused_book,
                active_checkpoint_stable_reference,
                support_floor_world_z_m=active_support_floor_world_z,
                **active_supported_book_final_policy,
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='midroute_settle_paused_left_arm_moved',
            ))
            _require(expected_joint_gate(
                (paused_joint_snapshot(LEFT_GRIPPER_JOINTS)[0],),
                (active_checkpoint_gripper_m,),
                limit=active_checkpoint_gripper_limit_m,
                failure_reason='midroute_settle_paused_gripper_moved',
            ))
        elif reverse_reseat:
            if active_reseat_reference is None:
                raise RuntimeError('reverse reseat reference unavailable')
            paused_geometry = inward_reseat_progress_gate(
                start_book,
                paused_book,
                planned_delta_world_m=active_reseat_planned_delta,
                support_floor_world_z_m=active_support_floor_world_z,
                stable_reference=active_reseat_reference,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=CHECKPOINT_ARM_LIMIT_RAD,
                failure_reason='reverse_reseat_paused_joint_mismatch',
            ))
        elif post_reseat_settle:
            if active_reseat_reference is None:
                raise RuntimeError('post-reseat stable reference unavailable')
            paused_geometry = stable_reseat_state_gate(
                paused_book,
                stable_reference=active_reseat_reference,
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
            _require(expected_joint_gate(
                paused_joint_snapshot(IK_JOINTS),
                active_q8[-1],
                limit=active_checkpoint_arm_limit_rad,
                failure_reason='post_reseat_settle_paused_left_arm_moved',
            ))
            _require(expected_joint_gate(
                (paused_joint_snapshot(LEFT_GRIPPER_JOINTS)[0],),
                (active_checkpoint_gripper_m,),
                limit=active_checkpoint_gripper_limit_m,
                failure_reason='post_reseat_settle_paused_gripper_moved',
            ))
        elif shelf_probe:
            geometry_reference = (
                active_settle_reference
                if active_settle_reference is not None
                else start_book
            )
            geometry_delta = (
                active_settle_planned_delta
                if active_settle_reference is not None
                else route_world_delta(active_route[-1])
            )
            paused_geometry = diagonal_peel_progress_gate(
                geometry_reference,
                paused_book,
                planned_delta_world_m=geometry_delta,
                support_floor_world_z_m=active_support_floor_world_z,
                policy=progress_policy(active_route[-1]),
            )
        else:
            paused_geometry = final_extraction_geometry_gate(
                start_book,
                paused_book,
            )
        if active_settle_reference is not None:
            rotation_growth_start = (
                last_immediate_endpoint_book
                if (
                    halfmillimeter_outward
                    and last_immediate_endpoint_book is not None
                )
                else final_book
            )
            rotation_before_pause = quaternion_distance(
                active_settle_reference.quaternion,
                rotation_growth_start.quaternion,
            )
            rotation_after_pause = quaternion_distance(
                active_settle_reference.quaternion,
                paused_book.quaternion,
            )
            pause_rotation_growth = rotation_after_pause - rotation_before_pause
            _require(GateResult(
                pause_rotation_growth
                <= active_settle_rotation_growth_limit_rad,
                'paused_settle_rotation_growth_verified'
                if pause_rotation_growth
                <= active_settle_rotation_growth_limit_rad
                else 'paused_settle_rotation_growth_exceeded_limit',
                {
                    'rotation_before_pause_rad': rotation_before_pause,
                    'rotation_after_pause_rad': rotation_after_pause,
                    'rotation_growth_rad': pause_rotation_growth,
                    'rotation_growth_limit_rad': (
                        active_settle_rotation_growth_limit_rad
                    ),
                },
            ))
        _require(paused_geometry)
        if active_minimum_force_n != (0.0, 0.0):
            minimum_left, minimum_right = active_minimum_force_n
            paused_forces_ok = bool(
                float(paused_evidence.left_force) >= minimum_left
                and float(paused_evidence.right_force) >= minimum_right
                and float(paused_evidence.left_force)
                <= active_maximum_force_n[0]
                and float(paused_evidence.right_force)
                <= active_maximum_force_n[1]
                and (
                    float(paused_evidence.left_force)
                    + float(paused_evidence.right_force)
                    >= active_minimum_total_force_n
                )
            )
            _require(GateResult(
                paused_forces_ok,
                'paused_settle_absolute_pressure_verified'
                if paused_forces_ok
                else 'paused_settle_absolute_pressure_out_of_bounds',
                {
                    'left_force_n': float(paused_evidence.left_force),
                    'right_force_n': float(paused_evidence.right_force),
                    'minimum_left_force_n': minimum_left,
                    'minimum_right_force_n': minimum_right,
                    'minimum_total_force_n': active_minimum_total_force_n,
                    'maximum_left_force_n': active_maximum_force_n[0],
                    'maximum_right_force_n': active_maximum_force_n[1],
                },
            ))
        paused_base_planar = paused_base.planar
        if (
            math.hypot(
                paused_base_planar.x - start_base.x,
                paused_base_planar.y - start_base.y,
            ) > PAYLOAD_BASE_POSITION_LIMIT_M
            or abs(math.atan2(
                math.sin(paused_base_planar.yaw - start_base.yaw),
                math.cos(paused_base_planar.yaw - start_base.yaw),
            )) > PAYLOAD_BASE_YAW_LIMIT_RAD
        ):
            raise RuntimeError('base moved at paused extraction endpoint')
        _emit(
            'result',
            passed=True,
            stage=active_result_stage,
            target_model=BOOK,
            lift_distance_m=(
                0.0
                if (
                    reverse_reseat
                    or post_reseat_settle
                    or halfmillimeter_midroute_settle
                )
                else (
                    float(active_peel_delta[2])
                    if shelf_probe else MICRO_LIFT_TOTAL_M
                )
            ),
            extraction_distance_m=(
                0.0
                if (
                    reverse_reseat
                    or post_reseat_settle
                    or halfmillimeter_midroute_settle
                )
                else (
                    float(-active_peel_delta[0])
                    if shelf_probe else OUTWARD_TOTAL_M
                )
            ),
            reseat_distance_m=(
                float(active_peel_delta[0]) if reverse_reseat else 0.0
            ),
            route_steps=len(active_q8) - 1,
            measured_gripper_width_m=float(paused_evidence.width),
            left_force_n=float(paused_evidence.left_force),
            right_force_n=float(paused_evidence.right_force),
            **node.strict_payload_observed_metrics(),
            final_book_position=list(paused_book.position),
            final_book_quaternion=list(paused_book.quaternion),
            final_base_position=list(paused_base.position),
            final_base_quaternion=list(paused_base.quaternion),
            final_shelf_clearance_m=float(
                paused_geometry.metrics.get(
                    'final_shelf_clearance_m',
                    ROUTE_SHELF_FRONT_WORLD_X_M
                    - book_maximum_world_x(paused_book),
                )
            ),
            offline_dense_samples=int(active_audit.get(
                'dense_samples',
                active_audit.get('dense_joint_samples', 0),
            )),
            minimum_self_margin_m=float(active_audit.get(
                'minimum_self_aabb_clearance_m',
                active_audit.get(
                    'inherited_full_route_minimum_nonadjacent_robot_aabb_clearance_m',
                    active_audit.get(
                        'minimum_nonadjacent_robot_aabb_clearance_m',
                        math.nan,
                    ),
                ),
            )),
            minimum_padded_payload_robot_margin_m=float(active_audit.get(
                'minimum_15mm_padded_payload_robot_aabb_clearance_m',
                active_audit.get(
                    'inherited_conservative_minimum_15mm_padded_payload_robot_clearance_m',
                    math.nan,
                ),
            )),
            minimum_closed_gripper_shelf_margin_m=float(active_audit.get(
                'minimum_closed_gripper_shelf_triangle_aabb_clearance_m',
                active_audit.get(
                    'inherited_minimum_closed_gripper_shelf_clearance_after_tolerance_m',
                    active_audit.get(
                        'minimum_closed_gripper_shelf_clearance_after_tolerance_m',
                        math.nan,
                    ),
                ),
            )),
            gripper_reopened=False,
            left_arm_trajectory_commanded=bool(len(active_q8) > 1),
            gripper_trajectory_commanded=False,
            base_motion_commanded=False,
            zero_cmd_vel_published=True,
            right_arm_commanded=False,
            head_commanded=False,
            gazebo_entity_pose_mutation_used=False,
            gazebo_paused=True,
        )
    except Exception as exc:
        node._cancel.set()
        stop.publish_zero()
        goals_cancelled = node.cancel_active_goals()
        pause_ok = set_world_paused_confirmed(True)
        world_unpaused = not pause_ok
        _emit(
            'result',
            passed=False,
            reason=str(exc),
            controller_goals_terminal=bool(goals_cancelled),
            gazebo_pause_confirmed=bool(pause_ok),
            gripper_reopened=False,
            base_motion_commanded=False,
            zero_cmd_vel_published=True,
            right_arm_commanded=False,
            head_commanded=False,
            gazebo_entity_pose_mutation_used=False,
        )
        raise
    finally:
        monitor_stop.set()
        node._cancel.set()
        stop.publish_zero()
        goals_cancelled = node.cancel_active_goals()
        pause_ok = True
        if world_unpaused:
            pause_ok = set_world_paused_confirmed(True)
        _emit(
            'cleanup',
            passed=bool(goals_cancelled and pause_ok),
            controller_goals_terminal=bool(goals_cancelled),
            gazebo_paused=bool(pause_ok),
        )
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        try:
            node._gz_pose_node.unsubscribe(DYNAMIC_POSE_TOPIC)
        except Exception:
            pass
        executor.shutdown(timeout_sec=2.0)
        executor_thread.join(timeout=2.0)
        node.destroy_node()
        stop.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
