#!/usr/bin/env python3
"""One-command 30.000-to-30.250 mm shelf-cage aperture diagnostic.

This executable is deliberately isolated from the competition mission.  Its
only possible command is one slow left-gripper aperture trajectory.  It never
commands either arm, the base, navigation, or the right gripper, and it has no
automatic recovery path.  Gazebo truth and diagnostic contacts make its result
engineering evidence rather than competition-valid sensing.

The command is admitted only while the naturally tilted target has fresh exact
left-tip, right-tip, and palm contacts and a measured shelf-line/local-palm
support envelope.  The envelope is geometric evidence only: raw contact-wrench
signs are intentionally not interpreted as proof of support or force closure.
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
    preflight_measured_seeded_aperture_sweep,
)
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    GuardResult,
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
    _angle_error,
    _emit,
    _fresh_cage_gate,
    _unexpected_pairs,
)
from erc_phase1_solution.live_rigid_palm_natural_settle_probe import (
    EXPECTED_CAGED_APERTURE_M,
    SCENE_WAIT_TIMEOUT_S,
    WorldSample,
    _exact_contacts,
    _probe_node_types,
    _scene_sample,
    non_target_book_motion_guard,
)
from erc_phase1_solution.live_rigid_palm_natural_tangent_slip_probe import (
    BASE_STEP_LIMIT_M,
    BASE_YAW_LIMIT_RAD,
    RESUME_BASE_CUMULATIVE_LIMIT_M,
    RIGHT_ARM_HOLD_LIMIT_RAD,
    SHELF_SECTION_EXPECTED_X_SPAN_M,
    SHELF_SECTION_EXPECTED_Y_SPAN_M,
    SHELF_SECTION_MINIMUM_Y_OVERLAP_M,
    SHELF_SECTION_X_SPAN_TOLERANCE_M,
    SHELF_SECTION_Y_SPAN_TOLERANCE_M,
    SHELF_SOLID_PENETRATION_LIMIT_M,
    _right_arm_state,
    palm_pressure_interface_guard,
    shelf_lip_guard,
    shelf_solid_penetration_m,
)
from erc_phase1_solution.live_rigid_palm_post_natural_settle_probe import (
    _inset_convex_polygon,
    _minimum_inward_slack,
    _raw_palm_support_geometry,
    shelf_edge_proximity_geometry,
)
from erc_phase1_solution.live_rigid_palm_post_slide_transport_probe import (
    SUPPORT_POLYGON_INSET_M,
    _convex_hull,
    _palm_local_triangles,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
)
from erc_phase1_solution.motion_profiles import RIGHT_ARM_JOINTS
from erc_phase1_solution.rigid_palm_live_preflight import _node_time_seconds
from erc_phase1_solution.rigid_palm_preflight import PALM_COLLISION_LINK


MICRO_APERTURE_EVENT = 'rigid_palm_natural_micro_aperture_evidence'
START_APERTURE_M = 0.030000
TARGET_APERTURE_M = 0.030250
COMMAND_DURATION_S = 0.90
COMMAND_TIMEOUT_SIM_S = 2.0
COMMAND_WALL_FAILSAFE_S = 30.0

START_APERTURE_TOLERANCE_M = 0.00010
ENDPOINT_APERTURE_TOLERANCE_M = 0.00010
MINIMUM_APERTURE_PROGRESS_M = 0.00015
MAXIMUM_APERTURE_PROGRESS_M = 0.00035
APERTURE_REVERSE_LIMIT_M = 0.00005

BOOK_STEP_LIMIT_M = 0.00010
BOOK_CUMULATIVE_LIMIT_M = 0.00020
BOOK_ROTATION_LIMIT_RAD = math.radians(0.20)
BOOK_CORNER_LIMIT_M = 0.00050
ARM_HOLD_LIMIT_RAD = 0.0020

SHELF_EDGE_MISMATCH_LIMIT_M = 0.00025
COMBINED_SUPPORT_INSET_M = 0.003
COMBINED_SUPPORT_MINIMUM_X_SPAN_M = 0.120
COMBINED_SUPPORT_MINIMUM_Y_SPAN_M = 0.020
PALM_FACE_MAXIMUM_Z_SPAN_M = 0.00010
PALM_SHELF_CONTACT_HEIGHT_LIMIT_M = 0.00050

STABLE_DWELL_S = 0.30
STABLE_DWELL_MINIMUM_S = 0.25
STABLE_DWELL_MAXIMUM_S = 0.50

# Exact paused checkpoint after the one-millimetre tangent probe was stopped.
# This is a recognition bound only; no arm target is ever generated here.
EXPECTED_PARTIAL_TANGENT_Q = np.asarray(
    [
        0.349999986553275,
        0.2973449613572264,
        0.5495005599538189,
        0.5439278332152697,
        -1.834784238207804,
        1.288973481011598,
        0.8490026408837357,
        -1.3827522632905376,
    ],
    dtype=float,
)
RESUME_ARM_TOLERANCE_RAD = 0.0015
EXPECTED_PARTIAL_TANGENT_BASE = np.asarray(
    [2.126330, -0.091395, 0.002495], dtype=float
)
RESUME_BASE_POSITION_TOLERANCE_M = 0.0010
RESUME_BASE_YAW_TOLERANCE_RAD = 0.0010


@dataclass(frozen=True)
class MicroObservation:
    """One coherent world sample plus the measured idle right arm."""

    world: WorldSample
    right_arm: np.ndarray


@dataclass(frozen=True)
class CombinedSupportAssessment:
    """Shelf-line plus finite local-palm support-envelope evidence."""

    safe: bool
    reason: str
    hull_xy: np.ndarray
    local_palm_feature_xy: np.ndarray
    metrics: Mapping[str, float]


def _finite_vector(value: Sequence[float], length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{label} must contain {length} finite values')
    return result


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _convex_intersection(
    subject: Sequence[Sequence[float]],
    clip: Sequence[Sequence[float]],
) -> np.ndarray:
    """Intersect two finite convex CCW polygons with edge clipping."""

    output = _convex_hull(np.asarray(subject, dtype=float))
    boundary = _convex_hull(np.asarray(clip, dtype=float))
    if output.shape[1:] != (2,) or boundary.shape[1:] != (2,):
        raise ValueError('support polygons must be two-dimensional')
    for first, second in zip(boundary, np.roll(boundary, -1, axis=0)):
        incoming = output
        if not len(incoming):
            break
        output_points: list[np.ndarray] = []
        start = incoming[-1]
        start_inside = _cross_2d(second - first, start - first) >= -1e-10
        for finish in incoming:
            finish_inside = (
                _cross_2d(second - first, finish - first) >= -1e-10
            )
            if finish_inside != start_inside:
                travel = finish - start
                denominator = _cross_2d(travel, second - first)
                if abs(denominator) <= 1e-14:
                    raise ValueError('support polygon clipping is singular')
                fraction = _cross_2d(first - start, second - first) / denominator
                output_points.append(start + fraction * travel)
            if finish_inside:
                output_points.append(finish.copy())
            start, start_inside = finish, finish_inside
        output = np.asarray(output_points, dtype=float)
    if output.ndim != 2 or output.shape[1:] != (2,) or len(output) < 3:
        raise ValueError('book OBB has no finite local palm feature')
    return _convex_hull(output)


def combined_shelf_palm_support_guard(
    book_corners_world: Sequence[Sequence[float]],
    shelf_triangles_world: Sequence[Sequence[Sequence[float]]],
    palm_triangles_local: Sequence[Sequence[Sequence[float]]],
    palm_world_transform: Sequence[Sequence[float]],
    outward_world: Sequence[float],
    *,
    polygon_inset_m: float,
) -> CombinedSupportAssessment:
    """Require the COM inside an inset shelf-line/local-palm XY hull.

    The local palm feature is the convex XY intersection of the selected palm
    face and target OBB.  This avoids treating remote parts of the palm mesh as
    support.  The shelf contribution is only the measured OBB section at the
    shelf front edge.  No wrench direction or force-closure claim is made.
    """

    empty = np.empty((0, 2), dtype=float)
    try:
        corners = np.asarray(book_corners_world, dtype=float)
        if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
            raise ValueError('target corners are malformed')
        edge = shelf_edge_proximity_geometry(
            corners,
            shelf_triangles_world,
            shelf_front_x_m=SHELF_FRONT_X_M,
            proximity_limit_m=SHELF_EDGE_MISMATCH_LIMIT_M,
        )
        if not edge.available:
            return CombinedSupportAssessment(False, edge.reason, empty, empty, edge.metrics)
        x_span = float(edge.section_maximum_x_m - edge.section_minimum_x_m)
        y_span = float(edge.section_maximum_y_m - edge.section_minimum_y_m)
        penetration = shelf_solid_penetration_m(
            corners, shelf_plane_z_m=edge.shelf_plane_z_m
        )
        metrics: dict[str, float] = {
            **edge.metrics,
            'shelf_edge_section_x_span_m': x_span,
            'shelf_edge_section_y_span_m': y_span,
            'shelf_solid_penetration_m': penetration,
        }
        shelf_checks = (
            (edge.geometry_consistent, edge.reason),
            (edge.section_edge_distance_m <= SHELF_EDGE_MISMATCH_LIMIT_M, 'shelf_edge_mismatch'),
            (abs(x_span - SHELF_SECTION_EXPECTED_X_SPAN_M) <= SHELF_SECTION_X_SPAN_TOLERANCE_M, 'shelf_section_x_span_wrong'),
            (abs(y_span - SHELF_SECTION_EXPECTED_Y_SPAN_M) <= SHELF_SECTION_Y_SPAN_TOLERANCE_M, 'shelf_section_y_span_wrong'),
            (edge.section_shelf_y_overlap_m >= SHELF_SECTION_MINIMUM_Y_OVERLAP_M, 'shelf_section_y_overlap_too_small'),
            (penetration <= SHELF_SOLID_PENETRATION_LIMIT_M, 'target_shelf_penetration'),
        )
        for accepted, reason in shelf_checks:
            if not accepted:
                return CombinedSupportAssessment(False, reason, empty, empty, metrics)

        raw = _raw_palm_support_geometry(
            palm_triangles_local,
            palm_world_transform,
            corners,
            outward_world,
            polygon_inset_m=float(polygon_inset_m),
        )
        palm_world = np.asarray(raw.polygon_world, dtype=float)
        palm_face_z_span = float(np.ptp(palm_world[:, 2]))
        palm_shelf_height_error = abs(
            float(np.mean(palm_world[:, 2])) - edge.shelf_plane_z_m
        )
        metrics.update(
            {
                'local_palm_face_z_span_m': palm_face_z_span,
                'local_palm_shelf_contact_height_error_m': palm_shelf_height_error,
            }
        )
        if palm_face_z_span > PALM_FACE_MAXIMUM_Z_SPAN_M:
            return CombinedSupportAssessment(
                False, 'selected_palm_face_not_horizontal', empty, empty, metrics
            )
        if palm_shelf_height_error > PALM_SHELF_CONTACT_HEIGHT_LIMIT_M:
            return CombinedSupportAssessment(
                False, 'palm_shelf_contact_height_mismatch', empty, empty, metrics
            )
        palm_xy = _convex_hull(palm_world[:, :2])
        book_xy = _convex_hull(corners[:, :2])
        local_feature = _convex_intersection(palm_xy, book_xy)
        if not np.all(np.isfinite(local_feature)):
            raise ValueError('local palm feature is not finite')

        shelf_line = np.asarray(
            [
                [SHELF_FRONT_X_M, edge.section_minimum_y_m],
                [SHELF_FRONT_X_M, edge.section_maximum_y_m],
            ],
            dtype=float,
        )
        hull = _convex_hull(np.vstack((local_feature, shelf_line)))
        inset_hull = _inset_convex_polygon(hull, COMBINED_SUPPORT_INSET_M)
        com_xy = np.mean(corners, axis=0)[:2]
        inset_slack = _minimum_inward_slack(inset_hull, com_xy)
        original_slack = _minimum_inward_slack(hull, com_xy)
        combined_x_span = float(np.ptp(hull[:, 0]))
        combined_y_span = float(np.ptp(hull[:, 1]))
        metrics.update(
            {
                'local_palm_feature_vertex_count': float(len(local_feature)),
                'local_palm_feature_x_span_m': float(np.ptp(local_feature[:, 0])),
                'local_palm_feature_y_span_m': float(np.ptp(local_feature[:, 1])),
                'combined_support_hull_vertex_count': float(len(hull)),
                'combined_support_x_span_m': combined_x_span,
                'combined_support_y_span_m': combined_y_span,
                'combined_support_com_original_slack_m': original_slack,
                'combined_support_com_inset_slack_m': inset_slack,
                'combined_support_required_inset_m': COMBINED_SUPPORT_INSET_M,
                'raw_wrench_support_proven': 0.0,
                'force_closure_proven': 0.0,
            }
        )
        checks = (
            (combined_x_span >= COMBINED_SUPPORT_MINIMUM_X_SPAN_M, 'combined_support_x_span_too_small'),
            (combined_y_span >= COMBINED_SUPPORT_MINIMUM_Y_SPAN_M, 'combined_support_y_span_too_small'),
            (original_slack >= COMBINED_SUPPORT_INSET_M, 'combined_support_inset_slack_too_small'),
            (inset_slack >= -1e-9, 'target_com_outside_inset_combined_support'),
        )
        for accepted, reason in checks:
            if not accepted:
                return CombinedSupportAssessment(False, reason, hull, local_feature, metrics)
        return CombinedSupportAssessment(True, 'ok', hull, local_feature, metrics)
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return CombinedSupportAssessment(
            False,
            f'invalid_combined_support_geometry:{exc}',
            empty,
            empty,
            {'invalid_combined_support_geometry': 1.0},
        )


def _base_world(base: Sequence[float]) -> np.ndarray:
    pose = _finite_vector(base, 3, 'base pose')
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    result[:2, 3] = pose[:2]
    return result


def _outward_world(base: Sequence[float]) -> np.ndarray:
    pose = _finite_vector(base, 3, 'base pose')
    return np.asarray([-math.cos(pose[2]), -math.sin(pose[2]), 0.0], dtype=float)


def _support_for_sample(node: Any, environment: Any, sample: WorldSample) -> CombinedSupportAssessment:
    links = node.chain.link_transforms(sample.arm)
    if PALM_COLLISION_LINK not in links:
        return CombinedSupportAssessment(
            False, 'palm_fk_unavailable', np.empty((0, 2)), np.empty((0, 2)), {}
        )
    palm_world = _base_world(sample.base) @ np.asarray(links[PALM_COLLISION_LINK], dtype=float)
    pressure = palm_pressure_interface_guard(
        _palm_local_triangles(environment.model),
        palm_world,
        sample.corners,
        _outward_world(sample.base),
        polygon_inset_m=max(SUPPORT_POLYGON_INSET_M, float(environment.config.collision_padding_m)),
    )
    if not pressure.safe:
        return CombinedSupportAssessment(
            False, pressure.reason, np.empty((0, 2)), np.empty((0, 2)), pressure.metrics
        )
    combined = combined_shelf_palm_support_guard(
        sample.corners,
        sample.shelf_triangles,
        _palm_local_triangles(environment.model),
        palm_world,
        _outward_world(sample.base),
        polygon_inset_m=max(SUPPORT_POLYGON_INSET_M, float(environment.config.collision_padding_m)),
    )
    return CombinedSupportAssessment(
        combined.safe,
        combined.reason,
        combined.hull_xy,
        combined.local_palm_feature_xy,
        {**pressure.metrics, **combined.metrics},
    )


def _observation(node: Any, environment: Any, *, newer_than: float | None = None) -> MicroObservation:
    return MicroObservation(
        _scene_sample(node, environment, newer_than=newer_than),
        _right_arm_state(node),
    )


def micro_state_guard(
    reference: MicroObservation,
    previous: MicroObservation,
    current: MicroObservation,
    *,
    support: CombinedSupportAssessment,
    endpoint: bool,
    require_contacts: bool = False,
    left_contact: bool = False,
    right_contact: bool = False,
    palm_contact: bool = False,
    unexpected_contacts: bool = False,
) -> GuardResult:
    """Validate a dwell, in-flight sample, or exact-contact endpoint."""

    try:
        book_step = float(np.linalg.norm(current.world.book.position - previous.world.book.position))
        book_motion = float(np.linalg.norm(current.world.book.position - reference.world.book.position))
        book_rotation = quaternion_distance(current.world.book.quaternion, reference.world.book.quaternion)
        corner_motion = float(np.max(np.linalg.norm(current.world.corners - reference.world.corners, axis=1)))
        base_step = float(np.linalg.norm(current.world.base[:2] - previous.world.base[:2]))
        base_motion = float(np.linalg.norm(current.world.base[:2] - reference.world.base[:2]))
        base_yaw = _angle_error(current.world.base[2], reference.world.base[2])
        arm_hold = float(np.max(np.abs(current.world.arm - reference.world.arm)))
        right_hold = float(np.max(np.abs(current.right_arm - reference.right_arm)))
        progress = float(current.world.aperture_m - reference.world.aperture_m)
        reverse = max(0.0, float(previous.world.aperture_m - current.world.aperture_m))
        endpoint_error = abs(float(current.world.aperture_m - TARGET_APERTURE_M))
    except (AttributeError, TypeError, ValueError):
        return GuardResult(False, 'invalid_observation', {'invalid_observation': 1.0})
    metrics = {
        'book_step_m': book_step,
        'book_cumulative_m': book_motion,
        'book_rotation_rad': book_rotation,
        'book_maximum_corner_motion_m': corner_motion,
        'base_step_m': base_step,
        'base_cumulative_m': base_motion,
        'base_yaw_error_rad': base_yaw,
        'left_arm_hold_error_rad': arm_hold,
        'right_arm_hold_error_rad': right_hold,
        'aperture_progress_m': progress,
        'aperture_reverse_m': reverse,
        'aperture_endpoint_error_m': endpoint_error,
        **support.metrics,
    }
    checks = (
        (book_step <= BOOK_STEP_LIMIT_M, 'target_step_motion'),
        (book_motion <= BOOK_CUMULATIVE_LIMIT_M, 'target_cumulative_motion'),
        (book_rotation <= BOOK_ROTATION_LIMIT_RAD, 'target_rotation'),
        (corner_motion <= BOOK_CORNER_LIMIT_M, 'target_corner_motion'),
        (base_step <= BASE_STEP_LIMIT_M, 'base_step_motion'),
        (base_motion <= RESUME_BASE_CUMULATIVE_LIMIT_M, 'base_cumulative_motion'),
        (base_yaw <= BASE_YAW_LIMIT_RAD, 'base_rotated'),
        (arm_hold <= ARM_HOLD_LIMIT_RAD, 'left_arm_moved'),
        (right_hold <= RIGHT_ARM_HOLD_LIMIT_RAD, 'right_arm_moved'),
        (reverse <= APERTURE_REVERSE_LIMIT_M, 'aperture_reversed'),
        (progress <= MAXIMUM_APERTURE_PROGRESS_M, 'aperture_progress_overshoot'),
        (support.safe, support.reason),
        (not unexpected_contacts, 'unexpected_contact'),
        (not require_contacts or left_contact, 'left_target_contact_missing'),
        (not require_contacts or right_contact, 'right_target_contact_missing'),
        (not require_contacts or palm_contact, 'palm_target_contact_missing'),
        (not endpoint or progress >= MINIMUM_APERTURE_PROGRESS_M, 'aperture_progress_too_small'),
        (not endpoint or endpoint_error <= ENDPOINT_APERTURE_TOLERANCE_M, 'aperture_endpoint_missed'),
        (not endpoint or left_contact, 'left_target_contact_missing'),
        (not endpoint or right_contact, 'right_target_contact_missing'),
        (not endpoint or palm_contact, 'palm_target_contact_missing'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def resume_guard(sample: MicroObservation, *, support: CombinedSupportAssessment, reference_time: float, left_contact: bool, right_contact: bool, palm_contact: bool, unexpected_contacts: bool) -> GuardResult:
    try:
        age = float(reference_time) - float(sample.world.observed_at)
        arm_error = float(np.max(np.abs(sample.world.arm - EXPECTED_PARTIAL_TANGENT_Q)))
        aperture_error = abs(float(sample.world.aperture_m - EXPECTED_CAGED_APERTURE_M))
        base_position_error = float(np.linalg.norm(
            sample.world.base[:2] - EXPECTED_PARTIAL_TANGENT_BASE[:2]
        ))
        base_yaw_error = _angle_error(
            sample.world.base[2], EXPECTED_PARTIAL_TANGENT_BASE[2]
        )
    except (AttributeError, TypeError, ValueError):
        return GuardResult(False, 'invalid_observation', {'invalid_observation': 1.0})
    metrics = {
        'scene_age_s': age,
        'partial_tangent_arm_error_rad': arm_error,
        'aperture_start_error_m': aperture_error,
        'partial_tangent_base_position_error_m': base_position_error,
        'partial_tangent_base_yaw_error_rad': base_yaw_error,
        **support.metrics,
    }
    checks = (
        (-0.02 <= age <= 0.25, 'scene_timestamp_invalid'),
        (arm_error <= RESUME_ARM_TOLERANCE_RAD, 'wrong_partial_tangent_arm'),
        (base_position_error <= RESUME_BASE_POSITION_TOLERANCE_M, 'wrong_partial_tangent_base'),
        (base_yaw_error <= RESUME_BASE_YAW_TOLERANCE_RAD, 'wrong_partial_tangent_base_yaw'),
        (aperture_error <= START_APERTURE_TOLERANCE_M, 'wrong_start_aperture'),
        (left_contact, 'left_target_contact_missing'),
        (right_contact, 'right_target_contact_missing'),
        (palm_contact, 'palm_target_contact_missing'),
        (not unexpected_contacts, 'unexpected_contact'),
        (support.safe, support.reason),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _unexpected_or_latched(node: Any) -> bool:
    return bool(
        _unexpected_pairs(node)
        or getattr(node, '_target_robot_contact_latched', False)
        or getattr(node, '_payload_hazard_latched', None) is not None
        or getattr(node, 'probe_transfer_watchdog_reason', None) is not None
    )


def _require_recent_contacts(
    node: Any, *, max_age: float = 0.12, wait_wall_s: float = 0.20
) -> tuple[bool, bool, bool]:
    """Poll age-bounded exact evidence without clearing the contact epoch."""

    deadline = time.monotonic() + float(wait_wall_s)
    while True:
        contacts = _exact_contacts(node, max_age=float(max_age))
        if all(contacts):
            return contacts
        if time.monotonic() >= deadline:
            raise RuntimeError('recent exact bilateral+palm contact is unavailable')
        time.sleep(0.01)


def _stable_dwell(node: Any, environment: Any, reference: MicroObservation) -> tuple[MicroObservation, GuardResult]:
    if not (STABLE_DWELL_MINIMUM_S <= STABLE_DWELL_S <= STABLE_DWELL_MAXIMUM_S):
        raise RuntimeError('configured dwell is outside the audited 0.25--0.50 s window')
    previous = reference
    stamp = reference.world.observed_at
    final_guard = GuardResult(False, 'dwell_not_started', {})
    while True:
        current = _observation(node, environment, newer_than=stamp)
        support = _support_for_sample(node, environment, current.world)
        left, right, palm = _exact_contacts(node, max_age=0.12)
        final_guard = micro_state_guard(
            reference,
            previous,
            current,
            support=support,
            endpoint=False,
            require_contacts=True,
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_or_latched(node),
        )
        neighbours = non_target_book_motion_guard(reference.world.all_book_corners, current.world.all_book_corners)
        if not final_guard.safe:
            raise RuntimeError(f'stable dwell failed: {final_guard.reason}')
        if not neighbours.safe:
            raise RuntimeError(f'stable dwell neighbour failed: {neighbours.reason}')
        if current.world.observed_at - reference.world.observed_at >= STABLE_DWELL_S:
            return current, GuardResult(True, 'ok', {**final_guard.metrics, **neighbours.metrics})
        previous, stamp = current, current.world.observed_at


def _publish_micro_aperture(node: Any) -> None:
    if not bool(getattr(node, '_probe_actuation_enabled', False)):
        raise RuntimeError('probe actuation is disabled')
    from rclpy.duration import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    message = JointTrajectory()
    message.joint_names = ['gripper_left_finger_joint']
    point = JointTrajectoryPoint()
    point.positions = [TARGET_APERTURE_M]
    point.time_from_start = Duration(seconds=COMMAND_DURATION_S).to_msg()
    message.points = [point]
    node.gripper_pub.publish(message)


def _monitor_command(node: Any, environment: Any, reference: MicroObservation) -> tuple[MicroObservation, GuardResult]:
    previous = reference
    stamp = reference.world.observed_at
    start_sim = node.get_clock().now().nanoseconds / 1e9
    wall_deadline = time.monotonic() + COMMAND_WALL_FAILSAFE_S
    while time.monotonic() < wall_deadline:
        if node.get_clock().now().nanoseconds / 1e9 - start_sim > COMMAND_TIMEOUT_SIM_S:
            break
        current = _observation(node, environment, newer_than=stamp)
        support = _support_for_sample(node, environment, current.world)
        left, right, palm = _exact_contacts(node, max_age=0.12)
        guard = micro_state_guard(
            reference,
            previous,
            current,
            support=support,
            endpoint=False,
            require_contacts=True,
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_or_latched(node),
        )
        neighbours = non_target_book_motion_guard(reference.world.all_book_corners, current.world.all_book_corners)
        _emit('micro_aperture_motion_sample', passed=guard.safe and neighbours.safe, reason=guard.reason if not guard.safe else neighbours.reason, **guard.metrics, **neighbours.metrics)
        if not guard.safe:
            node.latch_transfer_watchdog(guard.reason)
            raise RuntimeError(f'micro-aperture watchdog fault: {guard.reason}')
        if not neighbours.safe:
            node.latch_transfer_watchdog(neighbours.reason)
            raise RuntimeError(f'micro-aperture neighbour fault: {neighbours.reason}')
        if abs(current.world.aperture_m - TARGET_APERTURE_M) <= ENDPOINT_APERTURE_TOLERANCE_M:
            return current, guard
        previous, stamp = current, current.world.observed_at
    node.latch_transfer_watchdog('micro_aperture_endpoint_timeout')
    raise RuntimeError('micro-aperture endpoint timed out')


def _run(runtime: SimpleNamespace) -> None:
    if runtime.BOOK != TARGET_BOOK_MODEL:
        raise RuntimeError('runtime target is not the audited seed-101 book')
    ProbeNode, _ = _probe_node_types(runtime)
    node = ProbeNode()
    executor = runtime.MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    spin_errors: list[BaseException] = []

    def spin() -> None:
        try:
            executor.spin()
        except BaseException as exc:
            spin_errors.append(exc)

    thread = threading.Thread(target=spin, name='rigid-palm-micro-aperture', daemon=True)
    thread.start()
    stage = 'created'
    command_count = 0
    try:
        deadline = time.monotonic() + 30.0
        while (len(node.joints) < len(EXPECTED_PARTIAL_TANGENT_Q) + len(RIGHT_ARM_JOINTS) or node.gripper_pub.get_subscription_count() < 1) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < len(EXPECTED_PARTIAL_TANGENT_Q) + len(RIGHT_ARM_JOINTS):
            raise RuntimeError('live robot joint state is unavailable')
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
        reference = _observation(node, runtime.environment_preflight)
        support = _support_for_sample(node, runtime.environment_preflight, reference.world)
        resume = resume_guard(reference, support=support, reference_time=_node_time_seconds(node), left_contact=left, right_contact=right, palm_contact=palm, unexpected_contacts=_unexpected_or_latched(node))
        if not resume.safe:
            raise RuntimeError(f'micro-aperture resume rejected: {resume.reason}')

        stage = 'measured_seeded_dense_preflight'
        preflight = preflight_measured_seeded_aperture_sweep(node=node, environment=runtime.environment_preflight, target_aperture_m=TARGET_APERTURE_M)
        _emit('micro_aperture_dense_preflight', passed=preflight.safe, code=preflight.code, detail=preflight.detail, full_captured_scene_checked=True, measured_passive_links_checked=True)
        if not preflight.safe:
            raise RuntimeError(f'micro-aperture preflight rejected: {preflight.code}: {preflight.detail}')

        left, right, palm = _fresh_cage_gate(node)
        fresh = _observation(node, runtime.environment_preflight, newer_than=reference.world.observed_at)
        fresh_support = _support_for_sample(node, runtime.environment_preflight, fresh.world)
        unchanged = micro_state_guard(reference, reference, fresh, support=fresh_support, endpoint=False, unexpected_contacts=_unexpected_or_latched(node))
        if not unchanged.safe:
            raise RuntimeError(f'post-preflight state changed: {unchanged.reason}')
        fresh_resume = resume_guard(fresh, support=fresh_support, reference_time=_node_time_seconds(node), left_contact=left, right_contact=right, palm_contact=palm, unexpected_contacts=_unexpected_or_latched(node))
        if not fresh_resume.safe:
            raise RuntimeError(f'post-preflight exact gate failed: {fresh_resume.reason}')

        stage = 'stable_dispatch_dwell'
        dwell_endpoint, dwell = _stable_dwell(node, runtime.environment_preflight, fresh)
        left, right, palm = _require_recent_contacts(node)
        dispatch = _observation(node, runtime.environment_preflight, newer_than=dwell_endpoint.world.observed_at)
        dispatch_support = _support_for_sample(node, runtime.environment_preflight, dispatch.world)
        dispatch_guard = micro_state_guard(fresh, dwell_endpoint, dispatch, support=dispatch_support, endpoint=False, require_contacts=True, left_contact=left, right_contact=right, palm_contact=palm, unexpected_contacts=_unexpected_or_latched(node))
        if not dispatch_guard.safe or not (left and right and palm):
            reason = dispatch_guard.reason if not dispatch_guard.safe else 'dispatch_exact_contact_missing'
            raise RuntimeError(f'dispatch gate failed: {reason}')

        stage = 'single_micro_aperture_command'
        # Non-clearing final timestamp check: clearing the epoch here could
        # briefly manufacture a contact-loss fault as the watchdog arms.
        _require_recent_contacts(node, max_age=0.12, wait_wall_s=0.12)
        node.begin_transfer_watchdog()
        command_count += 1
        try:
            _publish_micro_aperture(node)
            reached, _ = _monitor_command(
                node, runtime.environment_preflight, dispatch
            )
        finally:
            watchdog_reason = node.end_transfer_watchdog()
        if watchdog_reason is not None:
            raise RuntimeError(f'micro-aperture watchdog latched: {watchdog_reason}')
        left, right, palm = _require_recent_contacts(node)
        endpoint = _observation(node, runtime.environment_preflight, newer_than=reached.world.observed_at)
        endpoint_support = _support_for_sample(node, runtime.environment_preflight, endpoint.world)
        endpoint_guard = micro_state_guard(dispatch, reached, endpoint, support=endpoint_support, endpoint=True, require_contacts=True, left_contact=left, right_contact=right, palm_contact=palm, unexpected_contacts=_unexpected_or_latched(node))
        neighbours = non_target_book_motion_guard(dispatch.world.all_book_corners, endpoint.world.all_book_corners)
        if not endpoint_guard.safe:
            raise RuntimeError(f'micro-aperture endpoint rejected: {endpoint_guard.reason}')
        if not neighbours.safe:
            raise RuntimeError(f'micro-aperture endpoint neighbour rejected: {neighbours.reason}')

        stage = 'complete'
        _emit('result', passed=True, stage=MICRO_APERTURE_EVENT, diagnostic_only=True, gazebo_truth_used=True, left_gripper_command_count=command_count, command_duration_s=COMMAND_DURATION_S, measured_start_aperture_m=dispatch.world.aperture_m, measured_endpoint_aperture_m=endpoint.world.aperture_m, exact_left_contact=left, exact_right_contact=right, exact_palm_contact=palm, combined_shelf_palm_support_geometry=True, stable_dispatch_dwell_s=STABLE_DWELL_S, automatic_recovery_commanded=False, left_arm_motion_commanded=False, base_motion_commanded=False, navigation_commanded=False, right_arm_motion_commanded=False, raw_wrench_support_proven=False, force_closure_proven=False, gravity_support_authorized=False, next_motion_authorized=False, **dwell.metrics, **endpoint_guard.metrics, **neighbours.metrics)
    except Exception as exc:
        _emit('result', passed=False, stage=stage, reason=f'{type(exc).__name__}: {exc}', diagnostic_only=True, left_gripper_command_count=command_count, automatic_recovery_commanded=False, left_arm_motion_commanded=False, base_motion_commanded=False, navigation_commanded=False, right_arm_motion_commanded=False, raw_wrench_support_proven=False, force_closure_proven=False, gravity_support_authorized=False, next_motion_authorized=False)
        raise
    finally:
        node._cancel.set()
        executor.shutdown()
        thread.join(timeout=3.0)
        node.destroy_node()
        if spin_errors:
            raise RuntimeError(f'executor spin failed: {spin_errors[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Diagnostic-only 30.000-to-30.250 mm natural shelf-cage aperture probe')
    parser.add_argument('--confirm-diagnostic-0p25mm-aperture', action='store_true', help='authorize exactly one guarded 0.25 mm left-gripper aperture command')
    arguments = parser.parse_args()
    if not arguments.confirm_diagnostic_0p25mm_aperture:
        parser.error('--confirm-diagnostic-0p25mm-aperture is required')
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
