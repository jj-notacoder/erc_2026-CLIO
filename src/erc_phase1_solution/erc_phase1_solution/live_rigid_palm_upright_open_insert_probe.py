#!/usr/bin/env python3
"""Guarded upright 30-to-47 mm opening and one 0.25 mm palm step.

This is an isolated, single-run seed-101 engineering probe.  It resumes only
the recorded upright step-18 state at 30 mm, opens the left gripper in the
existing preflighted one-millimetre stages, recentres the palm from the fresh
measured hand-to-book transform, then reanchors once more and commands exactly
one +0.25 mm palm-local-z step.  It stops at that endpoint.

The target must retain both measured shelf support and a finite *local* palm
feature throughout.  The insertion endpoint must also show genuine hand/book
relative progress: a book that follows at least 80 percent of the hand motion
is rejected.  Gazebo poses and diagnostic contact sensors make this evidence,
not competition-valid perception.  There is no base, navigation, right-arm,
right-gripper, model-pose, or recovery command in this executable.
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

from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    GuardResult,
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
    _angle_error,
    _emit,
    _unexpected_pairs,
)
from erc_phase1_solution.live_rigid_palm_micro_aperture_probe import (
    _convex_intersection,
)
from erc_phase1_solution.live_rigid_palm_natural_settle_probe import (
    SCENE_WAIT_TIMEOUT_S,
    WorldSample,
    _exact_contacts,
    _scene_sample,
    non_target_book_motion_guard,
)
from erc_phase1_solution.live_rigid_palm_natural_tangent_slip_probe import (
    BASE_STEP_LIMIT_M,
    BASE_YAW_LIMIT_RAD,
    RESUME_BASE_CUMULATIVE_LIMIT_M,
    RIGHT_ARM_HOLD_LIMIT_RAD,
    _right_arm_state,
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
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
    rotation_matrix_distance,
    world_hand_pose,
)
from erc_phase1_solution.live_rigid_palm_tangent_slide_probe import (
    APERTURE_ENDPOINT_TOLERANCE_M,
    ARM_HOLD_LIMIT_RAD,
    HAND_ENDPOINT_POSITION_LIMIT_M,
    HAND_ENDPOINT_ROTATION_LIMIT_RAD,
    IK_ORIENTATION_TOLERANCE_RAD,
    IK_POSITION_TOLERANCE_M,
    OPEN_APERTURE_M,
    START_APERTURE_M,
    StageObservation,
    _execute_aperture_stage,
    _fresh_palm_gate,
    _probe_node_type,
    classify_step18_resume,
    fixed_height_tangent_pose,
    measured_local_y_recenter,
    opening_stage_targets,
    preflight_arm_leg,
)
from erc_phase1_solution.motion_profiles import RIGHT_ARM_JOINTS
from erc_phase1_solution.rigid_palm_live_preflight import _node_time_seconds
from erc_phase1_solution.rigid_palm_preflight import PALM_COLLISION_LINK


UPRIGHT_OPEN_INSERT_EVENT = 'rigid_palm_upright_open_insert_once'

RECENTER_MAXIMUM_M = 0.0010
RECENTER_SKIP_THRESHOLD_M = 0.00005
INSERT_LOCAL_Z_M = 0.00025
ARM_COMMAND_DURATION_S = 0.55
IK_MAXIMUM_JOINT_STEP_RAD = 0.012
PREFLIGHT_START_ARM_TOLERANCE_RAD = 0.00050

UPRIGHT_ROTATION_LIMIT_RAD = math.radians(2.0)
SHELF_OVERLAP_RANGE_M = (0.008, 0.015)
SHELF_SECTION_X_SPAN_RANGE_M = (0.150, 0.170)
SHELF_SECTION_Y_SPAN_RANGE_M = (0.025, 0.035)
SHELF_SECTION_MINIMUM_Y_OVERLAP_M = 0.025
SHELF_SOLID_PENETRATION_LIMIT_M = 0.00050
PALM_FACE_MAXIMUM_Z_SPAN_M = 0.00010
PALM_SHELF_HEIGHT_LIMIT_M = 0.00050
LOCAL_PALM_MINIMUM_X_SPAN_M = 0.00050
LOCAL_PALM_MINIMUM_Y_SPAN_M = 0.010
COMBINED_SUPPORT_INSET_M = 0.003
COMBINED_SUPPORT_MINIMUM_X_SPAN_M = 0.120
COMBINED_SUPPORT_MINIMUM_Y_SPAN_M = 0.020

OPENING_BOOK_CUMULATIVE_LIMIT_M = 0.0010
OPENING_BOOK_ROTATION_LIMIT_RAD = math.radians(1.0)
MOTION_BOOK_STEP_LIMIT_M = 0.00020
MOTION_BOOK_LEG_LIMIT_M = 0.00020
MOTION_BOOK_GLOBAL_LIMIT_M = 0.0010
MOTION_BOOK_ROTATION_LIMIT_RAD = math.radians(0.25)
MOTION_BOOK_OFF_AXIS_LIMIT_M = 0.00020
HAND_PROGRESS_FRACTION_RANGE = (0.60, 1.40)
HAND_OFF_AXIS_LIMIT_M = 0.00020
MINIMUM_RELATIVE_PROGRESS_FRACTION = 0.20
MAXIMUM_BOOK_FOLLOW_FRACTION = 0.80

SCENE_AGE_LIMIT_S = 0.25
SCENE_FUTURE_TOLERANCE_S = 0.02


@dataclass(frozen=True)
class ProbeObservation:
    """One coherent scene/left-state sample plus measured idle right arm."""

    world: WorldSample
    right_arm: np.ndarray


@dataclass(frozen=True)
class UprightCombinedSupportAssessment:
    """Finite shelf-line plus local-palm support for the upright target."""

    safe: bool
    reason: str
    hull_xy: np.ndarray
    local_palm_feature_xy: np.ndarray
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ReanchoredWaypoint:
    """One IK endpoint solved relative to a fresh measured hand pose."""

    phase: str
    start_positions: np.ndarray
    positions: np.ndarray
    start_hand_base: np.ndarray
    target_hand_base: np.ndarray
    command_local_m: np.ndarray


def _finite_vector(
    value: Sequence[float], length: int, label: str
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{label} must contain {length} finite values')
    return result


def _proper_transform(
    value: Sequence[Sequence[float]], label: str
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f'{label} must be a finite 4x4 transform')
    rotation = result[:3, :3]
    if (
        not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
    ):
        raise ValueError(f'{label} must be a proper rigid transform')
    return result


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
    return np.asarray(
        [-math.cos(float(pose[2])), -math.sin(float(pose[2])), 0.0],
        dtype=float,
    )


def upright_combined_support_guard(
    book: BookSnapshot,
    book_corners_world: Sequence[Sequence[float]],
    shelf_triangles_world: Sequence[Sequence[Sequence[float]]],
    palm_triangles_local: Sequence[Sequence[Sequence[float]]],
    palm_world_transform: Sequence[Sequence[float]],
    outward_world: Sequence[float],
    *,
    polygon_inset_m: float,
) -> UprightCombinedSupportAssessment:
    """Require an upright shelf line and a real local palm overlap feature.

    The shelf contributes only its measured front-edge section.  The palm
    contributes only the convex XY intersection between its selected upward
    face and the target OBB; remote palm mesh cannot enlarge the envelope.
    """

    empty = np.empty((0, 2), dtype=float)
    try:
        corners = np.asarray(book_corners_world, dtype=float)
        if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
            raise ValueError('target corners are malformed')
        tilt = quaternion_distance(
            book.quaternion, EXPECTED_RELEASED_BOOK_QUATERNION
        )
        edge = shelf_edge_proximity_geometry(
            corners,
            shelf_triangles_world,
            shelf_front_x_m=SHELF_FRONT_X_M,
            proximity_limit_m=SHELF_OVERLAP_RANGE_M[1],
        )
        if not edge.available:
            return UprightCombinedSupportAssessment(
                False, edge.reason, empty, empty, edge.metrics
            )
        x_span = float(
            edge.section_maximum_x_m - edge.section_minimum_x_m
        )
        y_span = float(
            edge.section_maximum_y_m - edge.section_minimum_y_m
        )
        signed_overlap = float(edge.section_rear_signed_overlap_m)
        penetration = shelf_solid_penetration_m(
            corners, shelf_plane_z_m=edge.shelf_plane_z_m
        )
        metrics: dict[str, float] = {
            **edge.metrics,
            'upright_rotation_error_rad': tilt,
            'shelf_section_x_span_m': x_span,
            'shelf_section_y_span_m': y_span,
            'shelf_signed_overlap_m': signed_overlap,
            'shelf_solid_penetration_m': penetration,
        }
        shelf_checks = (
            (tilt <= UPRIGHT_ROTATION_LIMIT_RAD, 'target_not_upright'),
            (edge.geometry_consistent, edge.reason),
            (
                SHELF_OVERLAP_RANGE_M[0]
                <= signed_overlap
                <= SHELF_OVERLAP_RANGE_M[1],
                'shelf_overlap_outside_upright_band',
            ),
            (
                SHELF_SECTION_X_SPAN_RANGE_M[0]
                <= x_span
                <= SHELF_SECTION_X_SPAN_RANGE_M[1],
                'upright_shelf_section_x_span_wrong',
            ),
            (
                SHELF_SECTION_Y_SPAN_RANGE_M[0]
                <= y_span
                <= SHELF_SECTION_Y_SPAN_RANGE_M[1],
                'upright_shelf_section_y_span_wrong',
            ),
            (
                edge.section_shelf_y_overlap_m
                >= SHELF_SECTION_MINIMUM_Y_OVERLAP_M,
                'shelf_section_y_overlap_too_small',
            ),
            (
                penetration <= SHELF_SOLID_PENETRATION_LIMIT_M,
                'target_shelf_penetration',
            ),
        )
        for accepted, reason in shelf_checks:
            if not accepted:
                return UprightCombinedSupportAssessment(
                    False, reason, empty, empty, metrics
                )

        raw = _raw_palm_support_geometry(
            palm_triangles_local,
            palm_world_transform,
            corners,
            outward_world,
            polygon_inset_m=float(polygon_inset_m),
        )
        palm_world = np.asarray(raw.polygon_world, dtype=float)
        if (
            palm_world.ndim != 2
            or palm_world.shape[1:] != (3,)
            or len(palm_world) < 3
            or not np.all(np.isfinite(palm_world))
        ):
            raise ValueError('selected palm face is malformed')
        palm_face_z_span = float(np.ptp(palm_world[:, 2]))
        palm_shelf_height_error = abs(
            float(np.mean(palm_world[:, 2])) - edge.shelf_plane_z_m
        )
        metrics.update(
            {
                'local_palm_face_z_span_m': palm_face_z_span,
                'local_palm_shelf_height_error_m': palm_shelf_height_error,
            }
        )
        if palm_face_z_span > PALM_FACE_MAXIMUM_Z_SPAN_M:
            return UprightCombinedSupportAssessment(
                False,
                'selected_palm_face_not_horizontal',
                empty,
                empty,
                metrics,
            )
        if palm_shelf_height_error > PALM_SHELF_HEIGHT_LIMIT_M:
            return UprightCombinedSupportAssessment(
                False,
                'palm_shelf_contact_height_mismatch',
                empty,
                empty,
                metrics,
            )

        palm_xy = _convex_hull(palm_world[:, :2])
        book_xy = _convex_hull(corners[:, :2])
        local_feature = _convex_intersection(palm_xy, book_xy)
        local_x_span = float(np.ptp(local_feature[:, 0]))
        local_y_span = float(np.ptp(local_feature[:, 1]))
        metrics.update(
            {
                'local_palm_feature_vertex_count': float(len(local_feature)),
                'local_palm_feature_x_span_m': local_x_span,
                'local_palm_feature_y_span_m': local_y_span,
            }
        )
        if local_x_span < LOCAL_PALM_MINIMUM_X_SPAN_M:
            return UprightCombinedSupportAssessment(
                False, 'local_palm_feature_x_span_too_small', empty,
                local_feature, metrics
            )
        if local_y_span < LOCAL_PALM_MINIMUM_Y_SPAN_M:
            return UprightCombinedSupportAssessment(
                False, 'local_palm_feature_y_span_too_small', empty,
                local_feature, metrics
            )

        shelf_line = np.asarray(
            [
                [SHELF_FRONT_X_M, edge.section_minimum_y_m],
                [SHELF_FRONT_X_M, edge.section_maximum_y_m],
            ],
            dtype=float,
        )
        hull = _convex_hull(np.vstack((local_feature, shelf_line)))
        inset_hull = _inset_convex_polygon(
            hull, COMBINED_SUPPORT_INSET_M
        )
        com_xy = np.mean(corners, axis=0)[:2]
        original_slack = _minimum_inward_slack(hull, com_xy)
        inset_slack = _minimum_inward_slack(inset_hull, com_xy)
        combined_x_span = float(np.ptp(hull[:, 0]))
        combined_y_span = float(np.ptp(hull[:, 1]))
        metrics.update(
            {
                'combined_support_hull_vertex_count': float(len(hull)),
                'combined_support_x_span_m': combined_x_span,
                'combined_support_y_span_m': combined_y_span,
                'combined_support_com_original_slack_m': original_slack,
                'combined_support_com_inset_slack_m': inset_slack,
                'combined_support_required_inset_m': (
                    COMBINED_SUPPORT_INSET_M
                ),
                'raw_wrench_support_proven': 0.0,
                'force_closure_proven': 0.0,
            }
        )
        checks = (
            (
                combined_x_span >= COMBINED_SUPPORT_MINIMUM_X_SPAN_M,
                'combined_support_x_span_too_small',
            ),
            (
                combined_y_span >= COMBINED_SUPPORT_MINIMUM_Y_SPAN_M,
                'combined_support_y_span_too_small',
            ),
            (
                original_slack >= COMBINED_SUPPORT_INSET_M,
                'combined_support_inset_slack_too_small',
            ),
            (
                inset_slack >= -1e-9,
                'target_com_outside_inset_combined_support',
            ),
        )
        for accepted, reason in checks:
            if not accepted:
                return UprightCombinedSupportAssessment(
                    False, reason, hull, local_feature, metrics
                )
        return UprightCombinedSupportAssessment(
            True, 'ok', hull, local_feature, metrics
        )
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return UprightCombinedSupportAssessment(
            False,
            f'invalid_upright_combined_support_geometry:{exc}',
            empty,
            empty,
            {'invalid_upright_combined_support_geometry': 1.0},
        )


def _observation(
    node: Any,
    environment: Any,
    *,
    newer_than: float | None = None,
) -> ProbeObservation:
    return ProbeObservation(
        _scene_sample(node, environment, newer_than=newer_than),
        _right_arm_state(node),
    )


def _stage_observation_from_world(
    node: Any, sample: WorldSample
) -> StageObservation:
    return StageObservation(
        sample.book,
        sample.base.copy(),
        sample.arm.copy(),
        _proper_transform(
            node.chain.forward(sample.arm), 'base-frame measured hand'
        ),
        float(sample.aperture_m),
        float(sample.observed_at),
    )


def _support_for_sample(
    node: Any, environment: Any, sample: WorldSample
) -> UprightCombinedSupportAssessment:
    links = node.chain.link_transforms(sample.arm)
    if PALM_COLLISION_LINK not in links:
        return UprightCombinedSupportAssessment(
            False,
            'palm_fk_unavailable',
            np.empty((0, 2)),
            np.empty((0, 2)),
            {},
        )
    palm_world = _base_world(sample.base) @ _proper_transform(
        links[PALM_COLLISION_LINK], 'base-frame palm link'
    )
    return upright_combined_support_guard(
        sample.book,
        sample.corners,
        sample.shelf_triangles,
        _palm_local_triangles(environment.model),
        palm_world,
        _outward_world(sample.base),
        polygon_inset_m=max(
            SUPPORT_POLYGON_INSET_M,
            float(environment.config.collision_padding_m),
        ),
    )


def _unexpected_or_latched(node: Any) -> bool:
    return bool(
        _unexpected_pairs(node)
        or getattr(node, '_target_robot_contact_latched', False)
        or getattr(node, '_payload_hazard_latched', None) is not None
        or getattr(node, 'upright_transfer_watchdog_reason', None) is not None
    )


def opening_checkpoint_guard(
    reference: ProbeObservation,
    current: ProbeObservation,
    *,
    expected_aperture_m: float,
    support: UprightCombinedSupportAssessment,
    reference_time: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Close every one-millimetre opening stage on the full support state."""

    try:
        now = float(reference_time)
        expected_aperture = float(expected_aperture_m)
        world = current.world
        scene_age = now - world.observed_at
        book_motion = float(
            np.linalg.norm(world.book.position - reference.world.book.position)
        )
        book_rotation = quaternion_distance(
            world.book.quaternion, reference.world.book.quaternion
        )
        upright_error = quaternion_distance(
            world.book.quaternion, EXPECTED_RELEASED_BOOK_QUATERNION
        )
        shelf_overlap = float(world.book.maximum[0] - SHELF_FRONT_X_M)
        base_motion = float(
            np.linalg.norm(world.base[:2] - reference.world.base[:2])
        )
        base_yaw = _angle_error(
            float(world.base[2]), float(reference.world.base[2])
        )
        arm_hold = float(
            np.max(np.abs(world.arm - reference.world.arm))
        )
        hand_motion = float(
            np.linalg.norm(
                world.hand_world[:3, 3]
                - reference.world.hand_world[:3, 3]
            )
        )
        hand_rotation = rotation_matrix_distance(
            world.hand_world[:3, :3], reference.world.hand_world[:3, :3]
        )
        aperture_error = abs(world.aperture_m - expected_aperture)
        right_hold = float(
            np.max(np.abs(current.right_arm - reference.right_arm))
        )
        neighbour = non_target_book_motion_guard(
            reference.world.all_book_corners, world.all_book_corners
        )
    except (AttributeError, TypeError, ValueError):
        return GuardResult(False, 'invalid_observation', {})
    metrics = {
        'scene_age_s': scene_age,
        'target_cumulative_motion_m': book_motion,
        'target_cumulative_rotation_rad': book_rotation,
        'upright_rotation_error_rad': upright_error,
        'shelf_overlap_m': shelf_overlap,
        'base_cumulative_motion_m': base_motion,
        'base_cumulative_yaw_motion_rad': base_yaw,
        'left_arm_hold_error_rad': arm_hold,
        'left_hand_hold_motion_m': hand_motion,
        'left_hand_hold_rotation_rad': hand_rotation,
        'aperture_endpoint_error_m': aperture_error,
        'right_arm_hold_error_rad': right_hold,
        **support.metrics,
        **neighbour.metrics,
    }
    checks = (
        (scene_age <= SCENE_AGE_LIMIT_S, 'scene_stale'),
        (scene_age >= -SCENE_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (
            book_motion <= OPENING_BOOK_CUMULATIVE_LIMIT_M,
            'target_cumulative_motion',
        ),
        (
            book_rotation <= OPENING_BOOK_ROTATION_LIMIT_RAD,
            'target_cumulative_rotation',
        ),
        (upright_error <= UPRIGHT_ROTATION_LIMIT_RAD, 'target_not_upright'),
        (
            SHELF_OVERLAP_RANGE_M[0]
            <= shelf_overlap
            <= SHELF_OVERLAP_RANGE_M[1],
            'shelf_overlap_outside_upright_band',
        ),
        (base_motion <= RESUME_BASE_CUMULATIVE_LIMIT_M, 'base_moved'),
        (base_yaw <= BASE_YAW_LIMIT_RAD, 'base_rotated'),
        (arm_hold <= ARM_HOLD_LIMIT_RAD, 'left_arm_moved_during_opening'),
        (
            hand_motion <= HAND_ENDPOINT_POSITION_LIMIT_M,
            'left_hand_moved_during_opening',
        ),
        (
            hand_rotation <= HAND_ENDPOINT_ROTATION_LIMIT_RAD,
            'left_hand_rotated_during_opening',
        ),
        (
            aperture_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'aperture_endpoint_missed',
        ),
        (right_hold <= RIGHT_ARM_HOLD_LIMIT_RAD, 'right_arm_moved'),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (support.safe, support.reason),
        (neighbour.safe, neighbour.reason),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def solve_reanchored_tangent_waypoint(
    chain: Any,
    measured_arm: Sequence[float],
    *,
    phase: str,
    local_y_m: float,
    local_z_m: float,
) -> ReanchoredWaypoint:
    """Solve one local displacement from this call's measured arm state."""

    start = _finite_vector(measured_arm, 8, 'measured arm')
    local_y = float(local_y_m)
    local_z = float(local_z_m)
    if not np.all(np.isfinite((local_y, local_z))):
        raise ValueError('local command must be finite')
    if abs(local_y) > RECENTER_MAXIMUM_M + 1e-12:
        raise ValueError('measured local-y recenter exceeds one millimetre')
    if local_z not in {0.0, INSERT_LOCAL_Z_M}:
        raise ValueError('local-z command is not the sole 0.25 mm step')
    if (abs(local_y) <= 1e-12) == (abs(local_z) <= 1e-12):
        raise ValueError('waypoint must contain exactly one tangent axis')
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
        raise ValueError('measured arm is outside its hard limits')
    start_hand = _proper_transform(
        chain.forward(start), 'reanchor hand pose'
    )
    target_hand = fixed_height_tangent_pose(
        start_hand, local_y, local_z
    )
    solution, _ = chain.solve(
        target_hand,
        [start],
        position_tolerance=IK_POSITION_TOLERANCE_M,
        orientation_tolerance=IK_ORIENTATION_TOLERANCE_RAD,
        max_iterations=600,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solution is None:
        raise RuntimeError(f'dynamic {phase} IK failed')
    positions = _finite_vector(solution, len(start), f'{phase} IK solution')
    achieved = _proper_transform(
        chain.forward(positions), f'{phase} achieved hand pose'
    )
    error = np.asarray(chain.pose_error(achieved, target_hand), dtype=float)
    joint_step = float(np.max(np.abs(positions[1:] - start[1:])))
    if (
        joint_step > IK_MAXIMUM_JOINT_STEP_RAD
        or float(np.linalg.norm(error[:3])) > IK_POSITION_TOLERANCE_M
        or float(np.linalg.norm(error[3:])) > IK_ORIENTATION_TOLERANCE_RAD
    ):
        raise RuntimeError(f'dynamic {phase} IK is discontinuous or inexact')
    return ReanchoredWaypoint(
        str(phase),
        start.copy(),
        positions,
        start_hand,
        target_hand,
        np.asarray([0.0, local_y, local_z], dtype=float),
    )


def motion_envelope_guard(
    pipeline_reference: ProbeObservation,
    leg_reference: ProbeObservation,
    previous: ProbeObservation,
    current: ProbeObservation,
    *,
    support: UprightCombinedSupportAssessment,
    reference_time: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Apply only hard physical caps while an arm leg is in flight."""

    try:
        world = current.world
        scene_age = float(reference_time) - world.observed_at
        step_motion = float(
            np.linalg.norm(world.book.position - previous.world.book.position)
        )
        leg_motion = float(
            np.linalg.norm(
                world.book.position - leg_reference.world.book.position
            )
        )
        global_motion = float(
            np.linalg.norm(
                world.book.position - pipeline_reference.world.book.position
            )
        )
        rotation = quaternion_distance(
            world.book.quaternion, leg_reference.world.book.quaternion
        )
        upright_error = quaternion_distance(
            world.book.quaternion, EXPECTED_RELEASED_BOOK_QUATERNION
        )
        shelf_overlap = float(world.book.maximum[0] - SHELF_FRONT_X_M)
        base_step = float(
            np.linalg.norm(world.base[:2] - previous.world.base[:2])
        )
        base_motion = float(
            np.linalg.norm(
                world.base[:2] - pipeline_reference.world.base[:2]
            )
        )
        base_yaw = _angle_error(
            float(world.base[2]),
            float(pipeline_reference.world.base[2]),
        )
        torso_hold = abs(
            float(world.arm[0] - leg_reference.world.arm[0])
        )
        aperture_error = abs(world.aperture_m - OPEN_APERTURE_M)
        right_hold = float(
            np.max(
                np.abs(current.right_arm - pipeline_reference.right_arm)
            )
        )
        neighbour = non_target_book_motion_guard(
            pipeline_reference.world.all_book_corners,
            world.all_book_corners,
        )
    except (AttributeError, TypeError, ValueError):
        return GuardResult(False, 'invalid_observation', {})
    metrics = {
        'scene_age_s': scene_age,
        'target_step_motion_m': step_motion,
        'target_leg_motion_m': leg_motion,
        'target_global_motion_m': global_motion,
        'target_leg_rotation_rad': rotation,
        'upright_rotation_error_rad': upright_error,
        'shelf_overlap_m': shelf_overlap,
        'base_step_motion_m': base_step,
        'base_cumulative_motion_m': base_motion,
        'base_cumulative_yaw_motion_rad': base_yaw,
        'torso_hold_error_m': torso_hold,
        'aperture_hold_error_m': aperture_error,
        'right_arm_hold_error_rad': right_hold,
        **support.metrics,
        **neighbour.metrics,
    }
    checks = (
        (scene_age <= SCENE_AGE_LIMIT_S, 'scene_stale'),
        (scene_age >= -SCENE_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (step_motion <= MOTION_BOOK_STEP_LIMIT_M, 'target_step_motion'),
        (leg_motion <= MOTION_BOOK_LEG_LIMIT_M, 'target_leg_motion'),
        (global_motion <= MOTION_BOOK_GLOBAL_LIMIT_M, 'target_global_motion'),
        (
            rotation <= MOTION_BOOK_ROTATION_LIMIT_RAD,
            'target_leg_rotation',
        ),
        (upright_error <= UPRIGHT_ROTATION_LIMIT_RAD, 'target_not_upright'),
        (
            SHELF_OVERLAP_RANGE_M[0]
            <= shelf_overlap
            <= SHELF_OVERLAP_RANGE_M[1],
            'shelf_overlap_outside_upright_band',
        ),
        (base_step <= BASE_STEP_LIMIT_M, 'base_step_motion'),
        (base_motion <= RESUME_BASE_CUMULATIVE_LIMIT_M, 'base_moved'),
        (base_yaw <= BASE_YAW_LIMIT_RAD, 'base_rotated'),
        (torso_hold <= PREFLIGHT_START_ARM_TOLERANCE_RAD, 'torso_moved'),
        (
            aperture_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'aperture_moved',
        ),
        (right_hold <= RIGHT_ARM_HOLD_LIMIT_RAD, 'right_arm_moved'),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (support.safe, support.reason),
        (neighbour.safe, neighbour.reason),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def relative_progress_endpoint_guard(
    pipeline_reference: ProbeObservation,
    leg_reference: ProbeObservation,
    previous: ProbeObservation,
    current: ProbeObservation,
    waypoint: ReanchoredWaypoint,
    *,
    support: UprightCombinedSupportAssessment,
    reference_time: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Require hand progress that is not mostly copied by the target book."""

    envelope = motion_envelope_guard(
        pipeline_reference,
        leg_reference,
        previous,
        current,
        support=support,
        reference_time=reference_time,
        palm_contact=palm_contact,
        unexpected_contacts=unexpected_contacts,
    )
    if not envelope.safe:
        return envelope
    try:
        command_local = _finite_vector(
            waypoint.command_local_m, 3, 'local tangent command'
        )
        command_distance = float(np.linalg.norm(command_local))
        if command_distance <= 1e-12:
            raise ValueError('local tangent command is zero')
        expected_delta_world = (
            leg_reference.world.hand_world[:3, :3] @ command_local
        )
        axis = expected_delta_world / np.linalg.norm(expected_delta_world)
        hand_delta = (
            current.world.hand_world[:3, 3]
            - leg_reference.world.hand_world[:3, 3]
        )
        book_delta = (
            current.world.book.position
            - leg_reference.world.book.position
        )
        hand_progress = float(np.dot(hand_delta, axis))
        book_progress = float(np.dot(book_delta, axis))
        relative_progress = hand_progress - book_progress
        hand_off_axis = float(
            np.linalg.norm(hand_delta - hand_progress * axis)
        )
        book_off_axis = float(
            np.linalg.norm(book_delta - book_progress * axis)
        )
        follow_fraction = max(0.0, book_progress) / max(
            hand_progress, 1e-12
        )
        arm_error = float(
            np.max(np.abs(current.world.arm - waypoint.positions))
        )
        target_hand_world = world_hand_pose(
            leg_reference.world.base, waypoint.target_hand_base
        )
        hand_position_error = float(
            np.linalg.norm(
                current.world.hand_world[:3, 3]
                - target_hand_world[:3, 3]
            )
        )
        hand_rotation_error = rotation_matrix_distance(
            current.world.hand_world[:3, :3],
            target_hand_world[:3, :3],
        )
        minimum_hand = HAND_PROGRESS_FRACTION_RANGE[0] * command_distance
        maximum_hand = HAND_PROGRESS_FRACTION_RANGE[1] * command_distance
        minimum_relative = (
            MINIMUM_RELATIVE_PROGRESS_FRACTION * command_distance
        )
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return GuardResult(False, 'invalid_relative_progress', envelope.metrics)
    metrics = {
        **envelope.metrics,
        'command_distance_m': command_distance,
        'hand_axis_progress_m': hand_progress,
        'book_axis_progress_m': book_progress,
        'hand_book_relative_progress_m': relative_progress,
        'book_follow_fraction': follow_fraction,
        'hand_off_axis_motion_m': hand_off_axis,
        'book_off_axis_motion_m': book_off_axis,
        'arm_endpoint_error_rad': arm_error,
        'hand_endpoint_position_error_m': hand_position_error,
        'hand_endpoint_rotation_error_rad': hand_rotation_error,
        'minimum_relative_progress_m': minimum_relative,
    }
    checks = (
        (
            minimum_hand <= hand_progress <= maximum_hand,
            'hand_axis_progress_out_of_bounds',
        ),
        (hand_off_axis <= HAND_OFF_AXIS_LIMIT_M, 'hand_off_axis_motion'),
        (
            book_off_axis <= MOTION_BOOK_OFF_AXIS_LIMIT_M,
            'book_off_axis_motion',
        ),
        (
            relative_progress + 1e-12 >= minimum_relative,
            'insufficient_hand_book_relative_progress',
        ),
        (
            follow_fraction
            < MAXIMUM_BOOK_FOLLOW_FRACTION - 1e-12,
            'target_followed_hand_too_far',
        ),
        (arm_error <= ARM_HOLD_LIMIT_RAD, 'arm_endpoint_missed'),
        (
            hand_position_error <= HAND_ENDPOINT_POSITION_LIMIT_M,
            'hand_endpoint_position_missed',
        ),
        (
            hand_rotation_error <= HAND_ENDPOINT_ROTATION_LIMIT_RAD,
            'hand_endpoint_rotation_missed',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _probe_type(runtime: SimpleNamespace) -> type:
    """Add one hard latch without changing the tangent probe's palm policy."""

    BaseProbe = _probe_node_type(runtime)

    class UprightOpenInsertProbeNode(BaseProbe):
        def __init__(self) -> None:
            self.upright_transfer_watchdog_active = False
            self.upright_transfer_watchdog_reason: str | None = None
            super().__init__()

        def reset_probe_audit(self) -> None:
            super().reset_probe_audit()

            def reset_extra() -> None:
                self.upright_transfer_watchdog_active = False
                self.upright_transfer_watchdog_reason = None

            self._with_probe_lock(reset_extra)

        def begin_upright_transfer_watchdog(self) -> None:
            def begin() -> None:
                self.upright_transfer_watchdog_reason = None
                self.upright_transfer_watchdog_active = True

            self._with_probe_lock(begin)

        def latch_upright_transfer_watchdog(self, reason: str) -> None:
            value = str(reason)

            def latch() -> None:
                if self.upright_transfer_watchdog_reason is None:
                    self.upright_transfer_watchdog_reason = value

            self._with_probe_lock(latch)

        def end_upright_transfer_watchdog(self) -> str | None:
            def end() -> str | None:
                self.upright_transfer_watchdog_active = False
                return self.upright_transfer_watchdog_reason

            return self._with_probe_lock(end)

        def _payload_hazard_reason(
            self, *, max_age: float = 0.20
        ) -> str | None:
            reason = super()._payload_hazard_reason(max_age=max_age)
            active, latched = self._with_probe_lock(
                lambda: (
                    bool(self.upright_transfer_watchdog_active),
                    self.upright_transfer_watchdog_reason,
                )
            )
            if reason is not None:
                if active:
                    self.latch_upright_transfer_watchdog(reason)
                return reason
            if active and latched is not None:
                return str(latched)
            return None

    return UprightOpenInsertProbeNode


def _watch_arm_leg(
    node: Any,
    environment: Any,
    pipeline_reference: ProbeObservation,
    leg_reference: ProbeObservation,
    stop: threading.Event,
    result: dict[str, Any],
) -> None:
    """Continuously latch support, target, neighbour, base, and right faults."""

    previous = leg_reference
    stamp = leg_reference.world.observed_at
    try:
        while not stop.is_set():
            current = _observation(node, environment, newer_than=stamp)
            stamp = current.world.observed_at
            result['latest'] = current
            _, _, palm = _exact_contacts(node, max_age=0.18)
            support = _support_for_sample(node, environment, current.world)
            guard = motion_envelope_guard(
                pipeline_reference,
                leg_reference,
                previous,
                current,
                support=support,
                reference_time=_node_time_seconds(node),
                palm_contact=palm,
                unexpected_contacts=_unexpected_or_latched(node),
            )
            if guard.safe:
                previous = current
                continue
            result['reason'] = guard.reason
            result['sample'] = current
            result['metrics'] = dict(guard.metrics)
            node.latch_upright_transfer_watchdog(guard.reason)
            return
    except BaseException as exc:
        reason = f'watchdog_error:{type(exc).__name__}:{exc}'
        result['reason'] = reason
        result['sample'] = result.get('latest')
        result['metrics'] = {}
        node.latch_upright_transfer_watchdog(reason)


def _execute_guarded_arm_leg(
    *,
    node: Any,
    runtime: SimpleNamespace,
    pipeline_reference: ProbeObservation,
    planning_reference: ProbeObservation,
    waypoint: ReanchoredWaypoint,
) -> ProbeObservation:
    """Preflight and execute one arm-only leg, with no recovery command."""

    preflight = preflight_arm_leg(
        node=node,
        environment=runtime.environment_preflight,
        target_positions=waypoint.positions,
    )
    _emit(
        'upright_arm_leg_preflight',
        phase=waypoint.phase,
        passed=preflight.safe,
        code=preflight.code,
        detail=preflight.detail,
        sample_index=preflight.sample_index,
        link=preflight.link,
        obstacle=preflight.obstacle,
    )
    if not preflight.safe:
        raise RuntimeError(
            f'{waypoint.phase} preflight rejected: '
            f'{preflight.code}: {preflight.detail}'
        )

    _fresh_palm_gate(node)
    confirmed = _observation(
        node,
        runtime.environment_preflight,
        newer_than=planning_reference.world.observed_at,
    )
    if float(
        np.max(np.abs(confirmed.world.arm - waypoint.start_positions))
    ) > PREFLIGHT_START_ARM_TOLERANCE_RAD:
        raise RuntimeError(
            f'{waypoint.phase} measured reanchor changed during preflight'
        )
    support = _support_for_sample(
        node, runtime.environment_preflight, confirmed.world
    )
    _, _, palm = _exact_contacts(node, max_age=0.18)
    unchanged = motion_envelope_guard(
        pipeline_reference,
        confirmed,
        confirmed,
        confirmed,
        support=support,
        reference_time=_node_time_seconds(node),
        palm_contact=palm,
        unexpected_contacts=_unexpected_or_latched(node),
    )
    if not unchanged.safe:
        raise RuntimeError(
            f'{waypoint.phase} post-preflight gate failed: '
            f'{unchanged.reason}'
        )

    stop = threading.Event()
    result: dict[str, Any] = {
        'reason': None,
        'sample': None,
        'latest': None,
        'metrics': {},
    }
    monitor = threading.Thread(
        target=_watch_arm_leg,
        args=(
            node,
            runtime.environment_preflight,
            pipeline_reference,
            confirmed,
            stop,
            result,
        ),
        name=f'upright-{waypoint.phase}-hard-watchdog',
        daemon=True,
    )
    node.begin_upright_transfer_watchdog()
    monitor.start()
    endpoint: ProbeObservation | None = None
    endpoint_guard = GuardResult(False, 'endpoint_not_measured', {})
    try:
        legs = ((
            waypoint.positions,
            ARM_COMMAND_DURATION_S,
            f'{UPRIGHT_OPEN_INSERT_EVENT}_{waypoint.phase}',
        ),)
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        moved, contact_loss = node._send_retained_arm_trajectory(
            goal, duration, legs, UPRIGHT_OPEN_INSERT_EVENT
        )
        if not moved or contact_loss:
            reason = (
                'palm_contact_or_watchdog_loss'
                if contact_loss
                else 'left_arm_controller_failure'
            )
            node.latch_upright_transfer_watchdog(reason)
        if moved and not contact_loss and (
            node.upright_transfer_watchdog_reason is None
        ):
            held = node._wait_for_retained_endpoint(
                waypoint.positions,
                command=UPRIGHT_OPEN_INSERT_EVENT,
                phase=waypoint.phase,
                leg=1,
                arm_tolerance=ARM_HOLD_LIMIT_RAD,
            )
            if held is None:
                node.latch_upright_transfer_watchdog(
                    f'{waypoint.phase}_endpoint_not_retained'
                )
            else:
                endpoint = _observation(
                    node,
                    runtime.environment_preflight,
                    newer_than=confirmed.world.observed_at,
                )
                _, _, palm = _exact_contacts(node, max_age=0.18)
                endpoint_support = _support_for_sample(
                    node, runtime.environment_preflight, endpoint.world
                )
                previous = result.get('latest') or confirmed
                endpoint_guard = relative_progress_endpoint_guard(
                    pipeline_reference,
                    confirmed,
                    previous,
                    endpoint,
                    waypoint,
                    support=endpoint_support,
                    reference_time=_node_time_seconds(node),
                    palm_contact=palm,
                    unexpected_contacts=_unexpected_or_latched(node),
                )
                if not endpoint_guard.safe:
                    node.latch_upright_transfer_watchdog(
                        endpoint_guard.reason
                    )
    finally:
        stop.set()
        monitor.join(timeout=SCENE_WAIT_TIMEOUT_S + 4.0)
        latched = node.end_upright_transfer_watchdog()
    if monitor.is_alive():
        raise RuntimeError(f'{waypoint.phase} hard watchdog did not stop')
    reason = (
        latched
        or result.get('reason')
        or getattr(node, 'upright_transfer_watchdog_reason', None)
    )
    if reason is not None:
        raise RuntimeError(
            f'{waypoint.phase} hard gate rejected the leg: {reason}'
        )
    if endpoint is None or not endpoint_guard.safe:
        raise RuntimeError(f'{waypoint.phase} endpoint was not proven')
    _emit(
        'upright_arm_leg_endpoint',
        phase=waypoint.phase,
        passed=True,
        measured_endpoint_q=endpoint.world.arm.tolist(),
        **endpoint_guard.metrics,
    )
    return endpoint


def _initial_resume(
    node: Any, runtime: SimpleNamespace
) -> ProbeObservation:
    left, right, palm = _fresh_palm_gate(node, require_bilateral=True)
    observation = _observation(node, runtime.environment_preflight)
    hand_base = _proper_transform(
        node.chain.forward(observation.world.arm), 'resume hand pose'
    )
    resume, nominal = classify_step18_resume(
        observation.world.book,
        observation.world.base,
        observation.world.arm,
        hand_base,
        aperture_m=observation.world.aperture_m,
        observed_at=observation.world.observed_at,
        reference_time=_node_time_seconds(node),
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=_unexpected_or_latched(node),
    )
    if not resume.safe or nominal != START_APERTURE_M:
        reason = resume.reason if not resume.safe else 'resume_is_not_30mm'
        raise RuntimeError(f'upright 30 mm resume rejected: {reason}')
    support = _support_for_sample(
        node, runtime.environment_preflight, observation.world
    )
    checkpoint = opening_checkpoint_guard(
        observation,
        observation,
        expected_aperture_m=START_APERTURE_M,
        support=support,
        reference_time=_node_time_seconds(node),
        palm_contact=palm,
        unexpected_contacts=_unexpected_or_latched(node),
    )
    if not checkpoint.safe:
        raise RuntimeError(
            f'initial finite support gate rejected: {checkpoint.reason}'
        )
    _emit(
        'upright_open_insert_resume_verified',
        passed=True,
        measured_aperture_m=observation.world.aperture_m,
        opening_target_count=len(opening_stage_targets(START_APERTURE_M)),
        one_arm_only=True,
        **resume.metrics,
        **checkpoint.metrics,
    )
    return observation


def _run(runtime: SimpleNamespace) -> None:
    if runtime.BOOK != TARGET_BOOK_MODEL:
        raise RuntimeError('runtime target is not the audited seed-101 book')
    ProbeNode = _probe_type(runtime)
    node = ProbeNode()
    executor = runtime.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    spin_errors: list[BaseException] = []

    def spin() -> None:
        try:
            executor.spin()
        except BaseException as exc:
            spin_errors.append(exc)

    thread = threading.Thread(
        target=spin,
        name='rigid-palm-upright-open-insert-once',
        daemon=True,
    )
    thread.start()
    stage = 'created'
    try:
        deadline = time.monotonic() + 30.0
        while (
            len(node.joints) < 8 + len(RIGHT_ARM_JOINTS)
            or node.gripper_pub.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8 + len(RIGHT_ARM_JOINTS):
            raise RuntimeError('live left/right joint state is unavailable')
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
        node._tangent_palm_support_active = True

        stage = 'resume_30mm'
        pipeline_reference = _initial_resume(node, runtime)
        current = pipeline_reference
        tangent_reference = _stage_observation_from_world(
            node, pipeline_reference.world
        )
        tangent_current = tangent_reference

        stage = 'opening_30_to_47mm'
        for index, target in enumerate(
            opening_stage_targets(START_APERTURE_M), start=1
        ):
            tangent_reached = _execute_aperture_stage(
                node=node,
                runtime=runtime,
                reference=tangent_reference,
                previous=tangent_current,
                target_aperture_m=target,
                phase='upright_opening',
                expected_arm=pipeline_reference.world.arm,
                expected_hand_base=tangent_reference.hand_base,
            )
            current = _observation(
                node,
                runtime.environment_preflight,
                newer_than=tangent_reached.observed_at,
            )
            _, _, palm = _exact_contacts(node, max_age=0.18)
            support = _support_for_sample(
                node, runtime.environment_preflight, current.world
            )
            checkpoint = opening_checkpoint_guard(
                pipeline_reference,
                current,
                expected_aperture_m=target,
                support=support,
                reference_time=_node_time_seconds(node),
                palm_contact=palm,
                unexpected_contacts=_unexpected_or_latched(node),
            )
            _emit(
                'upright_opening_checkpoint',
                passed=checkpoint.safe,
                stage_index=index,
                target_aperture_m=target,
                measured_aperture_m=current.world.aperture_m,
                reason=checkpoint.reason,
                **checkpoint.metrics,
            )
            if not checkpoint.safe:
                raise RuntimeError(
                    f'opening checkpoint {index} rejected: '
                    f'{checkpoint.reason}'
                )
            tangent_current = _stage_observation_from_world(
                node, current.world
            )

        stage = 'measured_local_y_recenter'
        recenter = measured_local_y_recenter(
            current.world.book,
            current.world.base,
            node.chain.forward(current.world.arm),
        )
        if abs(recenter) > RECENTER_MAXIMUM_M:
            raise RuntimeError(
                'measured local-y recenter exceeds the isolated one-mm band'
            )
        if abs(recenter) > RECENTER_SKIP_THRESHOLD_M:
            recenter_waypoint = solve_reanchored_tangent_waypoint(
                node.chain,
                current.world.arm,
                phase='measured_local_y_recenter',
                local_y_m=recenter,
                local_z_m=0.0,
            )
            current = _execute_guarded_arm_leg(
                node=node,
                runtime=runtime,
                pipeline_reference=pipeline_reference,
                planning_reference=current,
                waypoint=recenter_waypoint,
            )
        else:
            _emit(
                'upright_arm_leg_endpoint',
                phase='measured_local_y_recenter',
                passed=True,
                skipped_already_centered=True,
                measured_local_y_recenter_m=recenter,
            )

        # This fresh frame is the only anchor for the one insertion target.
        stage = 'dynamic_reanchor_for_0p25mm_insert'
        insertion_anchor = _observation(
            node,
            runtime.environment_preflight,
            newer_than=current.world.observed_at,
        )
        _, _, palm = _exact_contacts(node, max_age=0.18)
        anchor_support = _support_for_sample(
            node, runtime.environment_preflight, insertion_anchor.world
        )
        anchor_guard = motion_envelope_guard(
            pipeline_reference,
            insertion_anchor,
            insertion_anchor,
            insertion_anchor,
            support=anchor_support,
            reference_time=_node_time_seconds(node),
            palm_contact=palm,
            unexpected_contacts=_unexpected_or_latched(node),
        )
        if not anchor_guard.safe:
            raise RuntimeError(
                f'insertion reanchor rejected: {anchor_guard.reason}'
            )
        insertion_waypoint = solve_reanchored_tangent_waypoint(
            node.chain,
            insertion_anchor.world.arm,
            phase='single_0p25mm_local_z_insert',
            local_y_m=0.0,
            local_z_m=INSERT_LOCAL_Z_M,
        )
        _emit(
            'upright_insertion_reanchored',
            passed=True,
            scene_stamp=insertion_anchor.world.observed_at,
            measured_anchor_q=insertion_anchor.world.arm.tolist(),
            target_q=insertion_waypoint.positions.tolist(),
            local_y_m=0.0,
            local_z_m=INSERT_LOCAL_Z_M,
            **anchor_guard.metrics,
        )

        stage = 'single_0p25mm_local_z_insert'
        final = _execute_guarded_arm_leg(
            node=node,
            runtime=runtime,
            pipeline_reference=pipeline_reference,
            planning_reference=insertion_anchor,
            waypoint=insertion_waypoint,
        )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=UPRIGHT_OPEN_INSERT_EVENT,
            measured_aperture_m=final.world.aperture_m,
            measured_endpoint_q=final.world.arm.tolist(),
            book_position=final.world.book.position.tolist(),
            measured_local_y_recenter_m=recenter,
            local_z_step_count=1,
            local_z_step_m=INSERT_LOCAL_Z_M,
            left_arm_only=True,
            base_navigation_commanded=False,
            right_arm_commanded=False,
            recovery_commanded=False,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
        )
    except Exception as exc:
        _emit(
            'result',
            passed=False,
            stage=stage,
            reason=f'{type(exc).__name__}: {exc}',
            local_z_step_limit=1,
            left_arm_only=True,
            base_navigation_commanded=False,
            right_arm_commanded=False,
            recovery_commanded=False,
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
            'One-arm upright 30-to-47 mm opening plus one 0.25 mm palm step'
        )
    )
    parser.add_argument(
        '--confirm-upright-open-insert-once',
        action='store_true',
        help='authorize this isolated guarded diagnostic after all gates',
    )
    arguments = parser.parse_args()
    if not arguments.confirm_upright_open_insert_once:
        parser.error('--confirm-upright-open-insert-once is required')
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
