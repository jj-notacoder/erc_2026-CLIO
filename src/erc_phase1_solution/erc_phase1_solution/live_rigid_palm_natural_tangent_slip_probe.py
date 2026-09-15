#!/usr/bin/env python3
"""One-command diagnostic for the naturally tilted three-point cage.

This executable characterizes one very small, book-tangent palm motion from
the completed sixth natural-settle leg.  It does not try to recover, carry,
or place the book.  The sole command translates the measured left hand 1 mm
along the freshly measured target-book +Z axis while holding hand attitude.
The book must remain shelf supported (at most 0.25 mm / 0.25 degree motion).

Before dispatch, the measured arm/tool is densely checked against one fresh
full-scene capture with the target OBB fixed.  Fresh exact left-tip,
right-tip, and palm contact, the 30 mm aperture, the q6/base checkpoint, and
the measured shelf-lip section are required both before and after preflight.
All watchdog triggers are hard faults.  The probe sends no base, gripper, or
right-arm command and always terminates with ``next_motion_authorized=false``.

Gazebo poses and diagnostic contact sensors make this an engineering probe,
not competition-valid manipulation logic.  Importing the module has no ROS
or controller side effects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import math
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from erc_phase1_solution.kinematics import oriented_box_from_corners

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
    SETTLE_OFF_AXIS_TILT_LIMIT_RAD,
    WorldSample,
    _exact_contacts,
    _probe_node_types,
    _scene_sample,
    _solve_translation,
    non_target_book_motion_guard,
    settle_attitude_metrics,
)
from erc_phase1_solution.live_rigid_palm_post_natural_settle_probe import (
    BOX_EDGES,
    PALM_FACE_GAP_LIMIT_M,
    PALM_FACE_PENETRATION_LIMIT_M,
    _inset_convex_polygon,
    palm_bearing_support_guard,
    shelf_edge_proximity_geometry,
)
from erc_phase1_solution.live_rigid_palm_post_slide_transport_probe import (
    SUPPORT_POLYGON_INSET_M,
    _palm_local_triangles,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
    quaternion_matrix,
    rotation_matrix_distance,
    world_hand_pose,
)
from erc_phase1_solution.live_rigid_palm_tangent_slide_probe import (
    _dense_arm_segment,
    _evaluate_dense_route,
)
from erc_phase1_solution.motion_profiles import RIGHT_ARM_JOINTS
from erc_phase1_solution.rigid_palm_live_preflight import (
    WorldScene,
    _measured_finger_transforms_relative_to_palm,
    _measured_robot_state,
    _node_time_seconds,
    _require_stable_measured_finger_geometry,
    _require_stable_preflight_inputs,
)
from erc_phase1_solution.rigid_palm_preflight import (
    PALM_COLLISION_LINK,
    PreflightResult,
)


RECOVERY_EVENT = 'rigid_palm_natural_one_mm_tangent_slip'

# Measured endpoint after natural-settle edge leg 6.  It is only a checkpoint;
# the commanded target is solved from the fresh measured q and book attitude.
EXPECTED_Q6 = np.asarray(
    [
        0.3499999113228583,
        0.2973261096251118,
        0.5503904898581027,
        0.5435045355624630,
        -1.8352491242077136,
        1.2874581105613256,
        0.8490750268921373,
        -1.3812040013106470,
    ],
    dtype=float,
)
EXPECTED_TANGENT_TARGET_Q = np.asarray(
    [
        0.3499999316,
        0.2973388208,
        0.5491059388,
        0.5441176328,
        -1.8345508940,
        1.2896759010,
        0.8489639293,
        -1.3834807600,
    ],
    dtype=float,
)
EXPECTED_TANGENT_WORLD_DELTA_M = np.asarray(
    [0.000899386, 0.0, -0.000437156], dtype=float
)
EXPECTED_TANGENT_HAND_LOCAL_DELTA_M = np.asarray(
    [-0.000438808, 0.000000866, 0.000898581], dtype=float
)

TANGENT_DISTANCE_M = 0.001
TANGENT_DURATION_S = 0.35
TANGENT_HAND_PROGRESS_RANGE_M = (0.00070, 0.00130)
TANGENT_HAND_OFF_AXIS_LIMIT_M = 0.00020
TANGENT_HAND_ROTATION_LIMIT_RAD = 0.0010
TANGENT_LOCAL_VECTOR_TOLERANCE_M = 0.00020
TANGENT_WORLD_DIRECTION_ALIGNMENT_MINIMUM = math.cos(math.radians(3.0))
TANGENT_TARGET_Q_TOLERANCE_RAD = 0.0040
TANGENT_MAXIMUM_JOINT_STEP_RAD = 0.010

RESUME_ARM_TOLERANCE_RAD = 0.0030
PREFLIGHT_START_ARM_TOLERANCE_RAD = 0.00050
RESUME_BASE_CHECKPOINT_LIMIT_M = 0.005
RESUME_BASE_CHECKPOINT_YAW_LIMIT_RAD = 0.008
RESUME_BASE_CUMULATIVE_LIMIT_M = 0.00075
BASE_STEP_LIMIT_M = 0.00035
BASE_YAW_LIMIT_RAD = 0.0010
APERTURE_LIMIT_M = 0.00035
RIGHT_ARM_HOLD_LIMIT_RAD = 0.0020

NATURAL_TILT_MINIMUM_RAD = math.radians(25.0)
NATURAL_TILT_MAXIMUM_RAD = math.radians(30.0)
BOOK_TRANSLATION_LIMIT_M = 0.00025
BOOK_ROTATION_LIMIT_RAD = math.radians(0.25)
BOOK_CORNER_MOTION_LIMIT_M = 0.0010
POST_PREFLIGHT_BOOK_TRANSLATION_LIMIT_M = 0.00020
POST_PREFLIGHT_BOOK_ROTATION_LIMIT_RAD = math.radians(0.20)
PREFLIGHT_TARGET_CORNER_STABILITY_M = 0.00020

SHELF_EDGE_MISMATCH_LIMIT_M = 0.00050
SHELF_SECTION_EXPECTED_X_SPAN_M = 0.12870
SHELF_SECTION_X_SPAN_TOLERANCE_M = 0.00250
SHELF_SECTION_EXPECTED_Y_SPAN_M = 0.03000
SHELF_SECTION_Y_SPAN_TOLERANCE_M = 0.00200
SHELF_SECTION_MINIMUM_Y_OVERLAP_M = 0.02800
SHELF_SOLID_PENETRATION_LIMIT_M = 0.00050

ENDPOINT_ARM_LIMIT_RAD = 0.0040
ENDPOINT_HAND_POSITION_LIMIT_M = 0.00035
ENDPOINT_HAND_ROTATION_LIMIT_RAD = 0.0010


@dataclass(frozen=True)
class RecoveryObservation:
    """One coherent scene/left-state sample plus measured idle right arm."""

    world: WorldSample
    right_arm: np.ndarray


@dataclass(frozen=True)
class TangentWaypoint:
    """Dynamically anchored endpoint for the sole arm command."""

    positions: np.ndarray
    target_hand_base: np.ndarray
    book_axis_world: np.ndarray
    expected_delta_world: np.ndarray
    expected_delta_hand_local: np.ndarray


@dataclass(frozen=True)
class TangentPreflight:
    """Bounded fixed-target full-scene preflight evidence."""

    safe: bool
    reason: str
    route_sample_count: int
    fixed_target_hypothesis_count: int
    result: PreflightResult
    maximum_palm_gap_m: float = math.inf
    maximum_palm_penetration_m: float = math.inf


@dataclass(frozen=True)
class WatchResult:
    """First hard watchdog fault and latest coherent observation."""

    reason: str | None
    sample: RecoveryObservation | None
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ExecutedTangentLeg:
    """Terminal evidence gathered before the hard watchdog is disarmed."""

    moved: bool
    cancelled: bool
    watch: WatchResult
    endpoint: RecoveryObservation | None
    endpoint_guard: GuardResult
    neighbour_guard: GuardResult
    left_contact: bool
    right_contact: bool
    palm_contact: bool


def _finite_vector(
    value: Sequence[float], length: int, label: str
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{label} must contain {length} finite values')
    return vector


def _upright_reference(book: BookSnapshot) -> BookSnapshot:
    return BookSnapshot(
        np.asarray(book.position, dtype=float).copy(),
        EXPECTED_RELEASED_BOOK_QUATERNION.copy(),
        np.asarray(book.minimum, dtype=float).copy(),
        np.asarray(book.maximum, dtype=float).copy(),
    )


def measured_book_positive_z(book: BookSnapshot) -> np.ndarray:
    """Return the normalized world direction of the measured book +Z axis."""

    rotation = quaternion_matrix(book.quaternion)
    direction = np.asarray(rotation[:, 2], dtype=float)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('measured book +Z axis is invalid')
    return direction / norm


def _right_arm_state(node: Any) -> np.ndarray:
    try:
        values = [float(node.joints[name]) for name in RIGHT_ARM_JOINTS]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('measured right-arm state is unavailable') from exc
    return _finite_vector(values, len(RIGHT_ARM_JOINTS), 'right arm')


def _observation(
    node: Any,
    environment: Any,
    *,
    newer_than: float | None = None,
) -> RecoveryObservation:
    return RecoveryObservation(
        _scene_sample(node, environment, newer_than=newer_than),
        _right_arm_state(node),
    )


def shelf_solid_penetration_m(
    corners_world: Sequence[Sequence[float]],
    *,
    shelf_plane_z_m: float,
    shelf_front_x_m: float = SHELF_FRONT_X_M,
) -> float:
    """Depth below the shelf top in the OBB portion behind its front edge."""

    corners = np.asarray(corners_world, dtype=float)
    plane = float(shelf_plane_z_m)
    front = float(shelf_front_x_m)
    if (
        corners.shape != (8, 3)
        or not np.all(np.isfinite(corners))
        or not math.isfinite(plane)
        or not math.isfinite(front)
    ):
        raise ValueError('shelf penetration geometry is malformed')
    candidates = [point for point in corners if point[0] >= front]
    for first_index, second_index in BOX_EDGES:
        first = corners[first_index]
        second = corners[second_index]
        distances = (float(first[0] - front), float(second[0] - front))
        if distances[0] * distances[1] < 0.0:
            fraction = distances[0] / (distances[0] - distances[1])
            candidates.append(first + fraction * (second - first))
        elif abs(distances[0]) <= 1e-12:
            candidates.append(first)
        elif abs(distances[1]) <= 1e-12:
            candidates.append(second)
    if not candidates:
        return 0.0
    minimum_z = min(float(point[2]) for point in candidates)
    return max(0.0, plane - minimum_z)


def shelf_lip_guard(
    corners_world: Sequence[Sequence[float]],
    shelf_triangles_world: Sequence[Sequence[Sequence[float]]],
) -> GuardResult:
    """Require the measured 128.7-by-30 mm section at the shelf lip."""

    edge = shelf_edge_proximity_geometry(
        corners_world,
        shelf_triangles_world,
        shelf_front_x_m=SHELF_FRONT_X_M,
        proximity_limit_m=SHELF_EDGE_MISMATCH_LIMIT_M,
    )
    metrics = dict(edge.metrics)
    if not edge.available:
        return GuardResult(False, edge.reason, metrics)
    x_span = float(edge.section_maximum_x_m - edge.section_minimum_x_m)
    y_span = float(edge.section_maximum_y_m - edge.section_minimum_y_m)
    try:
        penetration = shelf_solid_penetration_m(
            corners_world,
            shelf_plane_z_m=edge.shelf_plane_z_m,
        )
    except (TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_shelf_geometry:{exc}', metrics)
    metrics.update(
        {
            'shelf_edge_section_x_span_m': x_span,
            'shelf_edge_section_y_span_m': y_span,
            'shelf_solid_penetration_m': penetration,
        }
    )
    checks = (
        (edge.geometry_consistent, edge.reason),
        (
            edge.section_edge_distance_m <= SHELF_EDGE_MISMATCH_LIMIT_M,
            'shelf_edge_mismatch',
        ),
        (
            abs(x_span - SHELF_SECTION_EXPECTED_X_SPAN_M)
            <= SHELF_SECTION_X_SPAN_TOLERANCE_M,
            'shelf_section_x_span_wrong',
        ),
        (
            abs(y_span - SHELF_SECTION_EXPECTED_Y_SPAN_M)
            <= SHELF_SECTION_Y_SPAN_TOLERANCE_M,
            'shelf_section_y_span_wrong',
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
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def palm_pressure_interface_guard(
    palm_triangles_local: Sequence[Sequence[Sequence[float]]],
    palm_world_transform: Sequence[Sequence[float]],
    book_corners_world: Sequence[Sequence[float]],
    outward_world: Sequence[float],
    *,
    polygon_inset_m: float,
) -> GuardResult:
    """Bound gap/penetration without mislabelling the pressure cage support."""

    assessment = palm_bearing_support_guard(
        palm_triangles_local,
        palm_world_transform,
        book_corners_world,
        outward_world,
        polygon_inset_m=polygon_inset_m,
        maximum_face_gap_m=PALM_FACE_GAP_LIMIT_M,
        maximum_face_penetration_m=PALM_FACE_PENETRATION_LIMIT_M,
    )
    metrics = dict(assessment.metrics)
    raw = assessment.raw_geometry
    if raw is None:
        return GuardResult(False, 'palm_pressure_geometry_unavailable', metrics)
    try:
        minimum, maximum, vertex_count = clipped_palm_book_signed_interval(
            book_corners_world,
            raw.polygon_world,
            raw.support_normal_world,
            raw.plane_offset_m,
            raw.depth_world,
            raw.lateral_world,
            polygon_inset_m=polygon_inset_m,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return GuardResult(
            False, f'palm_contact_section_unavailable:{exc}', metrics
        )
    gap = max(0.0, minimum)
    # If the entire local OBB slice is behind the palm plane, measure the
    # nearest boundary.  If it straddles the plane, measure the actual local
    # incursion.  Far corners outside the finite palm prism never contribute.
    penetration = (
        max(0.0, -maximum)
        if maximum < 0.0
        else max(0.0, -minimum)
    )
    metrics.update(
        {
            'palm_pressure_face_gap_m': gap,
            'palm_pressure_face_penetration_m': penetration,
            'palm_clipped_minimum_signed_m': minimum,
            'palm_clipped_maximum_signed_m': maximum,
            'palm_clipped_polytope_vertex_count': float(vertex_count),
            'global_obb_minimum_signed_m_diagnostic_only': float(
                raw.book_minimum_signed_plane_distance_m
            ),
            'gravity_bearing_support_proven': float(assessment.safe),
            'pressure_cage_only': float(not assessment.safe),
        }
    )
    if gap > PALM_FACE_GAP_LIMIT_M:
        return GuardResult(False, 'book_palm_gap_exceeds_limit', metrics)
    if penetration > PALM_FACE_PENETRATION_LIMIT_M:
        return GuardResult(
            False, 'book_palm_penetration_exceeds_limit', metrics
        )
    return GuardResult(True, 'ok', metrics)


def clipped_palm_book_signed_interval(
    book_corners_world: Sequence[Sequence[float]],
    palm_polygon_world: Sequence[Sequence[float]],
    palm_normal_world: Sequence[float],
    palm_plane_offset_m: float,
    palm_depth_world: Sequence[float],
    palm_lateral_world: Sequence[float],
    *,
    polygon_inset_m: float,
) -> tuple[float, float, int]:
    """Clip an OBB by the finite palm prism and return its signed interval.

    The old global minimum over all eight book corners is meaningless for a
    tilted book: its lowest remote corner may be centimetres behind the palm
    plane while lying completely outside the palm pad.  Here both the OBB and
    every edge half-space of the inset finite palm polygon are linear
    constraints in world XYZ.  Enumerating triples of constraint planes gives
    every vertex of their convex intersection, from which the exact local
    signed interval follows.
    """

    corners = np.asarray(book_corners_world, dtype=float)
    polygon_world = np.asarray(palm_polygon_world, dtype=float)
    normal = np.asarray(palm_normal_world, dtype=float)
    depth = np.asarray(palm_depth_world, dtype=float)
    lateral = np.asarray(palm_lateral_world, dtype=float)
    offset = float(palm_plane_offset_m)
    inset = float(polygon_inset_m)
    if (
        corners.shape != (8, 3)
        or polygon_world.ndim != 2
        or polygon_world.shape[1] != 3
        or len(polygon_world) < 3
        or normal.shape != (3,)
        or depth.shape != (3,)
        or lateral.shape != (3,)
        or not np.all(np.isfinite(corners))
        or not np.all(np.isfinite(polygon_world))
        or not np.all(np.isfinite(normal))
        or not np.all(np.isfinite(depth))
        or not np.all(np.isfinite(lateral))
        or not math.isfinite(offset)
        or not math.isfinite(inset)
        or inset < 0.0
    ):
        raise ValueError('finite palm clipping geometry is malformed')
    normal_norm = float(np.linalg.norm(normal))
    depth_norm = float(np.linalg.norm(depth))
    lateral_norm = float(np.linalg.norm(lateral))
    if min(normal_norm, depth_norm, lateral_norm) <= 1e-12:
        raise ValueError('finite palm clipping basis is degenerate')
    normal = normal / normal_norm
    depth = depth / depth_norm
    lateral = lateral / lateral_norm
    if (
        abs(float(np.dot(normal, depth))) > 1e-6
        or abs(float(np.dot(normal, lateral))) > 1e-6
        or abs(float(np.dot(depth, lateral))) > 1e-6
    ):
        raise ValueError('finite palm clipping basis is not orthogonal')

    polygon_2d = np.column_stack(
        (polygon_world @ depth, polygon_world @ lateral)
    )
    polygon_2d = _inset_convex_polygon(polygon_2d, inset)
    center, axes, half_extents = oriented_box_from_corners(corners)
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    for axis, half_extent in zip(axes.T, half_extents):
        rows.extend((axis, -axis))
        center_projection = float(np.dot(axis, center))
        bounds.extend(
            (center_projection + half_extent, -center_projection + half_extent)
        )
    for first, second in zip(
        polygon_2d, np.roll(polygon_2d, -1, axis=0)
    ):
        edge = second - first
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            raise ValueError('finite palm polygon has a zero-length edge')
        inward = np.asarray([-edge[1], edge[0]], dtype=float) / length
        # inward dot ([depth.p, lateral.p] - first) >= 0
        world_row = -(inward[0] * depth + inward[1] * lateral)
        rows.append(world_row)
        bounds.append(-float(np.dot(inward, first)))
    matrix = np.asarray(rows, dtype=float)
    vector = np.asarray(bounds, dtype=float)
    tolerance = 2e-9
    candidates: list[np.ndarray] = []
    for indices in combinations(range(len(matrix)), 3):
        active = matrix[list(indices)]
        if abs(float(np.linalg.det(active))) <= 1e-11:
            continue
        point = np.linalg.solve(active, vector[list(indices)])
        if np.all(matrix @ point <= vector + tolerance):
            candidates.append(point)
    if not candidates:
        raise ValueError('book OBB does not overlap the inset finite palm')
    unique = np.unique(np.round(np.asarray(candidates), decimals=11), axis=0)
    signed = unique @ normal - offset
    return float(np.min(signed)), float(np.max(signed)), len(unique)


def recovery_checkpoint_guard(
    sample: WorldSample,
    *,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    shelf: GuardResult,
) -> GuardResult:
    """Recognize only the q6, 30 mm, 25--30 degree pressure cage."""

    try:
        now = float(reference_time)
        attitude = settle_attitude_metrics(
            _upright_reference(sample.book), sample.book, sample.base
        )
        arm_error = float(np.max(np.abs(sample.arm - EXPECTED_Q6)))
        base_error = float(np.linalg.norm(
            sample.base[:2] - EXPECTED_RELEASED_BASE_POSE[:2]
        ))
        base_yaw = _angle_error(
            sample.base[2], EXPECTED_RELEASED_BASE_POSE[2]
        )
        aperture_error = abs(
            float(sample.aperture_m - EXPECTED_CAGED_APERTURE_M)
        )
        scene_age = now - float(sample.observed_at)
        if not math.isfinite(now):
            raise ValueError('reference time is not finite')
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    metrics = {
        'scene_age_s': scene_age,
        'arm_q6_error_rad': arm_error,
        'base_checkpoint_error_m': base_error,
        'base_checkpoint_yaw_error_rad': base_yaw,
        'aperture_error_m': aperture_error,
        'natural_tilt_rad': attitude['tilt_rad'],
        'off_axis_tilt_rad': attitude['off_axis_tilt_rad'],
        'exact_left_contact': float(bool(left_contact)),
        'exact_right_contact': float(bool(right_contact)),
        'exact_palm_contact': float(bool(palm_contact)),
        **shelf.metrics,
    }
    checks = (
        (scene_age <= 0.25, 'scene_stale'),
        (scene_age >= -0.02, 'scene_from_future'),
        (left_contact, 'left_target_contact_missing'),
        (right_contact, 'right_target_contact_missing'),
        (palm_contact, 'palm_target_contact_missing'),
        (not unexpected_contacts, 'unexpected_contact'),
        (arm_error <= RESUME_ARM_TOLERANCE_RAD, 'wrong_q6_arm'),
        (
            base_error <= RESUME_BASE_CHECKPOINT_LIMIT_M,
            'wrong_base_checkpoint',
        ),
        (
            base_yaw <= RESUME_BASE_CHECKPOINT_YAW_LIMIT_RAD,
            'wrong_base_yaw',
        ),
        (aperture_error <= APERTURE_LIMIT_M, 'wrong_cage_aperture'),
        (
            NATURAL_TILT_MINIMUM_RAD
            <= attitude['tilt_rad']
            <= NATURAL_TILT_MAXIMUM_RAD,
            'book_tilt_outside_25_to_30_degrees',
        ),
        (
            attitude['off_axis_tilt_rad']
            <= SETTLE_OFF_AXIS_TILT_LIMIT_RAD,
            'settle_off_axis_tilt',
        ),
        (shelf.safe, shelf.reason),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def solve_book_tangent_waypoint(
    node: Any,
    start: WorldSample,
) -> TangentWaypoint:
    """Solve fresh q6 to a fixed-attitude 1 mm measured-book +Z move."""

    q = _finite_vector(start.arm, len(EXPECTED_Q6), 'measured q6 arm')
    if float(np.max(np.abs(q - EXPECTED_Q6))) > RESUME_ARM_TOLERANCE_RAD:
        raise ValueError('measured arm is not near the q6 checkpoint')
    axis_world = measured_book_positive_z(start.book)
    cosine, sine = math.cos(start.base[2]), math.sin(start.base[2])
    world_from_base = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    axis_base = world_from_base.T @ axis_world
    positions = _solve_translation(node, q, axis_base, TANGENT_DISTANCE_M)
    target_hand_base = np.asarray(node.chain.forward(positions), dtype=float)
    start_hand_base = np.asarray(node.chain.forward(q), dtype=float)
    start_hand_world = world_hand_pose(start.base, start_hand_base)
    target_hand_world = world_hand_pose(start.base, target_hand_base)
    delta_world = target_hand_world[:3, 3] - start_hand_world[:3, 3]
    delta_local = start_hand_world[:3, :3].T @ delta_world
    progress = float(np.dot(delta_world, axis_world))
    off_axis = float(np.linalg.norm(delta_world - progress * axis_world))
    alignment = float(np.dot(
        delta_world / max(float(np.linalg.norm(delta_world)), 1e-15),
        axis_world,
    ))
    rotation = rotation_matrix_distance(
        start_hand_world[:3, :3], target_hand_world[:3, :3]
    )
    checks = (
        (
            TANGENT_HAND_PROGRESS_RANGE_M[0]
            <= progress
            <= TANGENT_HAND_PROGRESS_RANGE_M[1],
            'tangent IK progress is outside 0.7--1.3 mm',
        ),
        (
            off_axis <= TANGENT_HAND_OFF_AXIS_LIMIT_M,
            'tangent IK has excessive off-axis motion',
        ),
        (
            alignment >= TANGENT_WORLD_DIRECTION_ALIGNMENT_MINIMUM,
            'tangent IK points away from measured book +Z',
        ),
        (
            rotation <= TANGENT_HAND_ROTATION_LIMIT_RAD,
            'tangent IK changed hand attitude',
        ),
        (
            float(np.max(np.abs(positions - q)))
            <= TANGENT_MAXIMUM_JOINT_STEP_RAD,
            'tangent IK left the local joint branch',
        ),
        (
            float(np.max(np.abs(
                positions - EXPECTED_TANGENT_TARGET_Q
            ))) <= TANGENT_TARGET_Q_TOLERANCE_RAD,
            'tangent IK does not match the audited branch',
        ),
        (
            float(np.linalg.norm(
                delta_local - EXPECTED_TANGENT_HAND_LOCAL_DELTA_M
            )) <= TANGENT_LOCAL_VECTOR_TOLERANCE_M,
            'tangent IK hand-local vector is not the audited direction',
        ),
        (
            float(np.linalg.norm(
                delta_world - EXPECTED_TANGENT_WORLD_DELTA_M
            )) <= TANGENT_LOCAL_VECTOR_TOLERANCE_M,
            'tangent IK world vector is not the audited direction',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            raise RuntimeError(reason)
    return TangentWaypoint(
        positions.copy(),
        target_hand_base.copy(),
        axis_world.copy(),
        delta_world.copy(),
        delta_local.copy(),
    )


def _outward_world(base: Sequence[float]) -> np.ndarray:
    pose = _finite_vector(base, 3, 'base pose')
    return np.asarray(
        [-math.cos(pose[2]), -math.sin(pose[2]), 0.0], dtype=float
    )


def _pressure_for_sample(
    node: Any,
    environment: Any,
    sample: WorldSample,
) -> GuardResult:
    links = node.chain.link_transforms(sample.arm)
    if PALM_COLLISION_LINK not in links:
        return GuardResult(False, 'palm FK is unavailable', {})
    cosine, sine = math.cos(sample.base[2]), math.sin(sample.base[2])
    base_world = np.eye(4, dtype=float)
    base_world[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    base_world[:2, 3] = sample.base[:2]
    return palm_pressure_interface_guard(
        _palm_local_triangles(environment.model),
        base_world @ np.asarray(links[PALM_COLLISION_LINK], dtype=float),
        sample.corners,
        _outward_world(sample.base),
        polygon_inset_m=max(
            SUPPORT_POLYGON_INSET_M,
            float(environment.config.collision_padding_m),
        ),
    )


def _require_stable_fixed_target_preflight_inputs(
    before_scene: Any,
    after_scene: Any,
    before_state: Any,
    after_state: Any,
) -> None:
    """Keep generic stability limits, with one target-specific allowance.

    The naturally settling target can move slightly more than the generic
    80-micrometre whole-world jitter limit while a dense check runs.  Its
    maximum OBB-corner drift is separately capped at 0.20 mm; every other
    book, the base, arm, and aperture retain the generic strict thresholds.
    """

    if before_scene.expected_book_names != after_scene.expected_book_names:
        raise ValueError('book identities changed during geometric preflight')
    if (
        TARGET_BOOK_MODEL not in before_scene.books
        or TARGET_BOOK_MODEL not in after_scene.books
    ):
        raise ValueError('target book is missing during geometric preflight')
    before_corners = np.asarray(
        before_scene.books[TARGET_BOOK_MODEL].corners, dtype=float
    )
    after_corners = np.asarray(
        after_scene.books[TARGET_BOOK_MODEL].corners, dtype=float
    )
    if (
        before_corners.shape != (8, 3)
        or after_corners.shape != (8, 3)
        or not np.all(np.isfinite(before_corners))
        or not np.all(np.isfinite(after_corners))
    ):
        raise ValueError('target book geometry changed shape during preflight')
    target_corner_drift = float(np.max(np.linalg.norm(
        after_corners - before_corners, axis=1
    )))
    if target_corner_drift > PREFLIGHT_TARGET_CORNER_STABILITY_M:
        raise ValueError(
            f'target book moved {target_corner_drift:.9f} m during preflight'
        )

    masked_books = dict(after_scene.books)
    masked_books[TARGET_BOOK_MODEL] = before_scene.books[TARGET_BOOK_MODEL]
    masked_after = WorldScene(
        base_transform=after_scene.base_transform,
        shelf_triangles=after_scene.shelf_triangles,
        books=masked_books,
        model_link_transforms=after_scene.model_link_transforms,
        expected_book_names=after_scene.expected_book_names,
        observed_at=after_scene.observed_at,
    )
    _require_stable_preflight_inputs(
        before_scene, masked_after, before_state, after_state
    )


def preflight_fixed_book_tangent(
    *,
    node: Any,
    environment: Any,
    expected_start_arm: Sequence[float],
    target_arm: Sequence[float],
) -> TangentPreflight:
    """Bounded dense full-scene proof with exactly one fixed target OBB."""

    failed = PreflightResult(False, 'not_evaluated', 'not evaluated')
    try:
        expected = _finite_vector(
            expected_start_arm, len(EXPECTED_Q6), 'expected preflight start'
        )
        target = _finite_vector(
            target_arm, len(EXPECTED_Q6), 'tangent target arm'
        )
        before = _measured_robot_state(node)
        if (
            float(np.max(np.abs(before.positions - expected)))
            > PREFLIGHT_START_ARM_TOLERANCE_RAD
            or abs(before.aperture_m - EXPECTED_CAGED_APERTURE_M)
            > APERTURE_LIMIT_M
        ):
            raise ValueError('measured tangent start changed before preflight')
        scene = environment._read_scene()
        reference_time = _node_time_seconds(node)
        measured = _measured_finger_transforms_relative_to_palm(
            node, scene, before
        )
        dense = _dense_arm_segment(
            environment=environment,
            node=node,
            scene=scene,
            joint_samples=(before.positions, target),
            aperture_m=before.aperture_m,
            measured_relative=measured,
        )
        result = _evaluate_dense_route(
            node=node,
            environment=environment,
            scene=scene,
            samples=dense,
            reference_time=reference_time,
        )
        if not result.safe:
            return TangentPreflight(
                False,
                f'{result.code}: {result.detail}',
                len(dense),
                1,
                result,
            )

        target_corners = np.asarray(
            scene.books[TARGET_BOOK_MODEL].corners, dtype=float
        )
        shelf = shelf_lip_guard(target_corners, scene.shelf_triangles)
        if not shelf.safe:
            result = PreflightResult(
                False, 'fixed_target_shelf_geometry', shelf.reason
            )
            return TangentPreflight(False, shelf.reason, len(dense), 1, result)

        palm_triangles = _palm_local_triangles(environment.model)
        outward = _outward_world(np.asarray([
            scene.base_transform[0, 3],
            scene.base_transform[1, 3],
            math.atan2(
                scene.base_transform[1, 0], scene.base_transform[0, 0]
            ),
        ]))
        gaps: list[float] = []
        penetrations: list[float] = []
        for index, sample in enumerate(dense):
            if PALM_COLLISION_LINK not in sample.transforms:
                raise ValueError('dense route omitted palm transform')
            interface = palm_pressure_interface_guard(
                palm_triangles,
                sample.transforms[PALM_COLLISION_LINK],
                target_corners,
                outward,
                polygon_inset_m=max(
                    SUPPORT_POLYGON_INSET_M,
                    float(environment.config.collision_padding_m),
                ),
            )
            if not interface.safe:
                result = PreflightResult(
                    False,
                    'fixed_target_palm_interface',
                    f'dense sample {index}: {interface.reason}',
                    sample_index=index,
                    link=PALM_COLLISION_LINK,
                    obstacle=TARGET_BOOK_MODEL,
                )
                return TangentPreflight(
                    False, interface.reason, len(dense), 1, result
                )
            gap = float(interface.metrics['palm_pressure_face_gap_m'])
            penetration = float(
                interface.metrics['palm_pressure_face_penetration_m']
            )
            if gap > (
                PALM_FACE_GAP_LIMIT_M - PREFLIGHT_TARGET_CORNER_STABILITY_M
            ):
                result = PreflightResult(
                    False,
                    'fixed_target_palm_gap_margin',
                    f'dense sample {index}: insufficient target-drift margin',
                    sample_index=index,
                    link=PALM_COLLISION_LINK,
                    obstacle=TARGET_BOOK_MODEL,
                )
                return TangentPreflight(
                    False, result.detail, len(dense), 1, result
                )
            if penetration > (
                PALM_FACE_PENETRATION_LIMIT_M
                - PREFLIGHT_TARGET_CORNER_STABILITY_M
            ):
                result = PreflightResult(
                    False,
                    'fixed_target_palm_penetration_margin',
                    f'dense sample {index}: insufficient target-drift margin',
                    sample_index=index,
                    link=PALM_COLLISION_LINK,
                    obstacle=TARGET_BOOK_MODEL,
                )
                return TangentPreflight(
                    False, result.detail, len(dense), 1, result
                )
            gaps.append(gap)
            penetrations.append(penetration)

        after_scene = environment._read_scene()
        after = _measured_robot_state(node)
        after_measured = _measured_finger_transforms_relative_to_palm(
            node, after_scene, after
        )
        _require_stable_measured_finger_geometry(
            environment.model, measured, after_measured
        )
        # Do not weaken this gate: it rejects even sub-millimetre creep that
        # occurred while the pure geometry computation was running.
        _require_stable_fixed_target_preflight_inputs(
            scene, after_scene, before, after
        )
        return TangentPreflight(
            True,
            'ok',
            len(dense),
            1,
            result,
            max(gaps, default=0.0),
            max(penetrations, default=0.0),
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return TangentPreflight(False, str(exc), 0, 1, failed)


def static_book_motion_guard(
    reference: RecoveryObservation,
    previous: RecoveryObservation,
    current: RecoveryObservation,
    *,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    shelf: GuardResult,
    palm_interface: GuardResult,
    reference_time: float,
) -> GuardResult:
    """Hard live bounds for the fixed-book one-millimetre experiment."""

    try:
        first = reference.world
        prior = previous.world
        now = current.world
        book_translation = float(np.linalg.norm(
            now.book.position - first.book.position
        ))
        book_rotation = quaternion_distance(
            first.book.quaternion, now.book.quaternion
        )
        corner_motion = float(np.max(np.linalg.norm(
            now.corners - first.corners, axis=1
        )))
        base_step = float(np.linalg.norm(now.base[:2] - prior.base[:2]))
        base_cumulative = float(np.linalg.norm(
            now.base[:2] - first.base[:2]
        ))
        base_yaw = _angle_error(now.base[2], first.base[2])
        aperture_error = abs(
            now.aperture_m - EXPECTED_CAGED_APERTURE_M
        )
        right_step = float(np.max(np.abs(
            current.right_arm - previous.right_arm
        )))
        right_cumulative = float(np.max(np.abs(
            current.right_arm - reference.right_arm
        )))
        attitude = settle_attitude_metrics(
            _upright_reference(now.book), now.book, first.base
        )
        scene_age = float(reference_time) - float(now.observed_at)
        if not math.isfinite(scene_age):
            raise ValueError('scene age is not finite')
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    metrics = {
        'book_translation_m': book_translation,
        'book_rotation_rad': book_rotation,
        'book_maximum_corner_motion_m': corner_motion,
        'base_step_m': base_step,
        'base_cumulative_m': base_cumulative,
        'base_yaw_error_rad': base_yaw,
        'aperture_error_m': aperture_error,
        'right_arm_step_rad': right_step,
        'right_arm_cumulative_rad': right_cumulative,
        'natural_tilt_rad': attitude['tilt_rad'],
        'off_axis_tilt_rad': attitude['off_axis_tilt_rad'],
        'scene_age_s': scene_age,
        **shelf.metrics,
        **palm_interface.metrics,
    }
    checks = (
        (scene_age <= 0.25, 'scene_stale'),
        (scene_age >= -0.02, 'scene_from_future'),
        (left_contact, 'left_target_contact_lost'),
        (right_contact, 'right_target_contact_lost'),
        (palm_contact, 'palm_target_contact_lost'),
        (not unexpected_contacts, 'unexpected_contact'),
        (
            book_translation <= BOOK_TRANSLATION_LIMIT_M,
            'fixed_book_translation_exceeded',
        ),
        (
            book_rotation <= BOOK_ROTATION_LIMIT_RAD,
            'fixed_book_rotation_exceeded',
        ),
        (
            corner_motion <= BOOK_CORNER_MOTION_LIMIT_M,
            'fixed_book_corner_motion_exceeded',
        ),
        (base_step <= BASE_STEP_LIMIT_M, 'base_step_exceeded'),
        (
            base_cumulative <= RESUME_BASE_CUMULATIVE_LIMIT_M,
            'base_cumulative_motion_exceeded',
        ),
        (base_yaw <= BASE_YAW_LIMIT_RAD, 'base_yaw_exceeded'),
        (aperture_error <= APERTURE_LIMIT_M, 'cage_aperture_changed'),
        (right_step <= RIGHT_ARM_HOLD_LIMIT_RAD, 'right_arm_moved'),
        (right_cumulative <= RIGHT_ARM_HOLD_LIMIT_RAD, 'right_arm_moved'),
        (
            NATURAL_TILT_MINIMUM_RAD
            <= attitude['tilt_rad']
            <= NATURAL_TILT_MAXIMUM_RAD,
            'book_tilt_outside_25_to_30_degrees',
        ),
        (
            attitude['off_axis_tilt_rad']
            <= SETTLE_OFF_AXIS_TILT_LIMIT_RAD,
            'settle_off_axis_tilt',
        ),
        (shelf.safe, shelf.reason),
        (palm_interface.safe, palm_interface.reason),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def endpoint_guard(
    reference: RecoveryObservation,
    current: RecoveryObservation,
    waypoint: TangentWaypoint,
    *,
    previous: RecoveryObservation | None = None,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    shelf: GuardResult,
    palm_interface: GuardResult,
    reference_time: float,
) -> GuardResult:
    """Freshly prove the sole endpoint; never authorize another motion."""

    hold = static_book_motion_guard(
        reference,
        reference if previous is None else previous,
        current,
        left_contact=left_contact,
        right_contact=right_contact,
        palm_contact=palm_contact,
        unexpected_contacts=unexpected_contacts,
        shelf=shelf,
        palm_interface=palm_interface,
        reference_time=reference_time,
    )
    metrics = dict(hold.metrics)
    try:
        hand_delta = (
            current.world.hand_world[:3, 3]
            - reference.world.hand_world[:3, 3]
        )
        progress = float(np.dot(hand_delta, waypoint.book_axis_world))
        off_axis = float(np.linalg.norm(
            hand_delta - progress * waypoint.book_axis_world
        ))
        arm_error = float(np.max(np.abs(
            current.world.arm - waypoint.positions
        )))
        expected_world = world_hand_pose(
            reference.world.base, waypoint.target_hand_base
        )
        hand_position_error = float(np.linalg.norm(
            current.world.hand_world[:3, 3] - expected_world[:3, 3]
        ))
        hand_rotation_error = rotation_matrix_distance(
            current.world.hand_world[:3, :3], expected_world[:3, :3]
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_endpoint:{exc}', metrics)
    metrics.update(
        {
            'hand_book_tangent_progress_m': progress,
            'hand_book_tangent_off_axis_m': off_axis,
            'arm_endpoint_error_rad': arm_error,
            'hand_endpoint_position_error_m': hand_position_error,
            'hand_endpoint_rotation_error_rad': hand_rotation_error,
        }
    )
    if not hold.safe:
        return GuardResult(False, hold.reason, metrics)
    checks = (
        (
            TANGENT_HAND_PROGRESS_RANGE_M[0]
            <= progress
            <= TANGENT_HAND_PROGRESS_RANGE_M[1],
            'tangent_hand_progress_out_of_bounds',
        ),
        (
            off_axis <= TANGENT_HAND_OFF_AXIS_LIMIT_M,
            'tangent_hand_off_axis_motion',
        ),
        (arm_error <= ENDPOINT_ARM_LIMIT_RAD, 'arm_endpoint_missed'),
        (
            hand_position_error <= ENDPOINT_HAND_POSITION_LIMIT_M,
            'hand_endpoint_position_missed',
        ),
        (
            hand_rotation_error <= ENDPOINT_HAND_ROTATION_LIMIT_RAD,
            'hand_endpoint_rotation_missed',
        ),
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


def _watch_one_tangent_leg(
    node: Any,
    runtime: SimpleNamespace,
    reference: RecoveryObservation,
    stop: threading.Event,
    result: dict[str, Any],
) -> None:
    """Latch only hard physical faults while the one leg is in flight."""

    previous = reference
    stamp = reference.world.observed_at
    try:
        while not stop.is_set():
            current = _observation(
                node,
                runtime.environment_preflight,
                newer_than=stamp,
            )
            stamp = current.world.observed_at
            result['latest'] = current
            left, right, palm = _exact_contacts(node)
            shelf = shelf_lip_guard(
                current.world.corners, current.world.shelf_triangles
            )
            interface = _pressure_for_sample(
                node, runtime.environment_preflight, current.world
            )
            guard = static_book_motion_guard(
                reference,
                previous,
                current,
                left_contact=left,
                right_contact=right,
                palm_contact=palm,
                unexpected_contacts=_unexpected_or_latched(node),
                shelf=shelf,
                palm_interface=interface,
                reference_time=_node_time_seconds(node),
            )
            neighbour = non_target_book_motion_guard(
                reference.world.all_book_corners,
                current.world.all_book_corners,
            )
            if not guard.safe:
                reason, metrics = guard.reason, guard.metrics
            elif not neighbour.safe:
                reason, metrics = neighbour.reason, neighbour.metrics
            else:
                previous = current
                continue
            result['reason'] = reason
            result['sample'] = current
            result['metrics'] = {**guard.metrics, **neighbour.metrics}
            node.latch_transfer_watchdog(reason)
            return
    except BaseException as exc:
        reason = f'watchdog_error:{type(exc).__name__}:{exc}'
        result['reason'] = reason
        result['sample'] = result.get('latest')
        result['metrics'] = {}
        node.latch_transfer_watchdog(reason)


def _run_one_tangent_leg(
    node: Any,
    runtime: SimpleNamespace,
    reference: RecoveryObservation,
    waypoint: TangentWaypoint,
) -> ExecutedTangentLeg:
    stop = threading.Event()
    result: dict[str, Any] = {
        'reason': None,
        'sample': None,
        'latest': None,
        'metrics': {},
    }
    moved = False
    cancelled = False
    endpoint_sample: RecoveryObservation | None = None
    endpoint_result = GuardResult(False, 'endpoint_not_remeasured', {})
    neighbour_result = GuardResult(False, 'endpoint_not_remeasured', {})
    left = right = palm = False
    node.begin_transfer_watchdog()
    monitor = threading.Thread(
        target=_watch_one_tangent_leg,
        args=(node, runtime, reference, stop, result),
        name='natural-tangent-slip-hard-watchdog',
        daemon=True,
    )
    monitor.start()
    try:
        # This is the only trajectory goal constructed by the executable.
        legs = ((waypoint.positions, TANGENT_DURATION_S, RECOVERY_EVENT),)
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        moved, cancelled = node._send_retained_arm_trajectory(
            goal, duration, legs, RECOVERY_EVENT
        )
        if moved and not cancelled and (
            getattr(node, 'probe_transfer_watchdog_reason', None) is None
        ):
            endpoint = node._wait_for_retained_endpoint(
                waypoint.positions,
                command=RECOVERY_EVENT,
                phase='single_one_mm_book_tangent_leg',
                leg=1,
                arm_tolerance=ENDPOINT_ARM_LIMIT_RAD,
            )
            if endpoint is None:
                node.latch_transfer_watchdog(
                    'one_tangent_endpoint_not_retained'
                )
            else:
                # Do not call _fresh_cage_gate here: it clears the exact
                # contact caches while the concurrently armed watchdog reads
                # them.  A non-clearing, age-bounded exact poll plus a newer
                # coherent scene frame closes the endpoint instead.
                endpoint_sample = _observation(
                    node,
                    runtime.environment_preflight,
                    newer_than=reference.world.observed_at,
                )
                left, right, palm = _exact_contacts(node, max_age=0.18)
                endpoint_shelf = shelf_lip_guard(
                    endpoint_sample.world.corners,
                    endpoint_sample.world.shelf_triangles,
                )
                endpoint_pressure = _pressure_for_sample(
                    node,
                    runtime.environment_preflight,
                    endpoint_sample.world,
                )
                endpoint_result = endpoint_guard(
                    reference,
                    endpoint_sample,
                    waypoint,
                    previous=(result.get('latest') or reference),
                    left_contact=left,
                    right_contact=right,
                    palm_contact=palm,
                    unexpected_contacts=_unexpected_or_latched(node),
                    shelf=endpoint_shelf,
                    palm_interface=endpoint_pressure,
                    reference_time=_node_time_seconds(node),
                )
                neighbour_result = non_target_book_motion_guard(
                    reference.world.all_book_corners,
                    endpoint_sample.world.all_book_corners,
                )
                if not endpoint_result.safe:
                    node.latch_transfer_watchdog(endpoint_result.reason)
                elif not neighbour_result.safe:
                    node.latch_transfer_watchdog(neighbour_result.reason)
    finally:
        stop.set()
        monitor.join(timeout=SCENE_WAIT_TIMEOUT_S + 4.0)
        reason = node.end_transfer_watchdog()
    if monitor.is_alive():
        raise RuntimeError('hard watchdog did not stop')
    # Re-read the late latch after joining: a final hard sample must not be
    # lost between _send_retained_arm_trajectory returning and end_watchdog.
    late_reason = getattr(node, 'probe_transfer_watchdog_reason', None)
    reason = reason or late_reason or result.get('reason')
    return ExecutedTangentLeg(
        moved,
        cancelled,
        WatchResult(
            None if reason is None else str(reason),
            endpoint_sample or result.get('sample') or result.get('latest'),
            dict(result.get('metrics', {})),
        ),
        endpoint_sample,
        endpoint_result,
        neighbour_result,
        left,
        right,
        palm,
    )


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

    thread = threading.Thread(
        target=spin, name='rigid-palm-natural-tangent-slip', daemon=True
    )
    thread.start()
    stage = 'created'
    arm_command_count = 0
    try:
        deadline = time.monotonic() + 30.0
        while (
            len(node.joints) < len(EXPECTED_Q6) + len(RIGHT_ARM_JOINTS)
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < len(EXPECTED_Q6):
            raise RuntimeError('live robot joint state is unavailable')

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
        shelf = shelf_lip_guard(
            reference.world.corners, reference.world.shelf_triangles
        )
        pressure = _pressure_for_sample(
            node, runtime.environment_preflight, reference.world
        )
        resume = recovery_checkpoint_guard(
            reference.world,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_or_latched(node),
            shelf=shelf,
        )
        if not pressure.safe:
            raise RuntimeError(
                f'initial palm pressure interface failed: {pressure.reason}'
            )
        if not resume.safe:
            raise RuntimeError(
                f'natural q6 resume rejected: {resume.reason}'
            )
        waypoint = solve_book_tangent_waypoint(node, reference.world)

        stage = 'fixed_book_dense_preflight'
        preflight = preflight_fixed_book_tangent(
            node=node,
            environment=runtime.environment_preflight,
            expected_start_arm=reference.world.arm,
            target_arm=waypoint.positions,
        )
        _emit(
            'natural_tangent_fixed_book_preflight',
            passed=preflight.safe,
            reason=preflight.reason,
            route_sample_count=preflight.route_sample_count,
            fixed_target_hypothesis_count=(
                preflight.fixed_target_hypothesis_count
            ),
            result_code=preflight.result.code,
            result_detail=preflight.result.detail,
            maximum_palm_gap_m=preflight.maximum_palm_gap_m,
            maximum_palm_penetration_m=(
                preflight.maximum_palm_penetration_m
            ),
            full_captured_scene_checked=True,
            measured_passive_links_checked=True,
            target_book_assumed_fixed=True,
        )
        if not preflight.safe:
            raise RuntimeError(
                f'fixed-book dense preflight rejected: {preflight.reason}'
            )

        # The full-scene checker retains its strict before/after stability
        # gate.  Then obtain new exact evidence and one newer scene frame.
        left, right, palm = _fresh_cage_gate(node)
        current = _observation(
            node,
            runtime.environment_preflight,
            newer_than=reference.world.observed_at,
        )
        current_shelf = shelf_lip_guard(
            current.world.corners, current.world.shelf_triangles
        )
        current_pressure = _pressure_for_sample(
            node, runtime.environment_preflight, current.world
        )
        unchanged = static_book_motion_guard(
            reference,
            reference,
            current,
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_or_latched(node),
            shelf=current_shelf,
            palm_interface=current_pressure,
            reference_time=_node_time_seconds(node),
        )
        if (
            unchanged.metrics.get('book_translation_m', math.inf)
            > POST_PREFLIGHT_BOOK_TRANSLATION_LIMIT_M
            or unchanged.metrics.get('book_rotation_rad', math.inf)
            > POST_PREFLIGHT_BOOK_ROTATION_LIMIT_RAD
        ):
            unchanged = GuardResult(
                False, 'book_changed_after_preflight', unchanged.metrics
            )
        if not unchanged.safe:
            raise RuntimeError(
                f'post-preflight fresh start rejected: {unchanged.reason}'
            )
        if float(np.max(np.abs(
            current.world.arm - reference.world.arm
        ))) > PREFLIGHT_START_ARM_TOLERANCE_RAD:
            raise RuntimeError('arm changed after tangent planning')

        stage = 'single_one_mm_book_tangent_leg'
        arm_command_count += 1
        executed = _run_one_tangent_leg(
            node, runtime, current, waypoint
        )
        if (
            not executed.moved
            or executed.cancelled
            or executed.watch.reason is not None
        ):
            raise RuntimeError(
                'one tangent leg stopped on hard fault: '
                f'moved={executed.moved}, cancelled={executed.cancelled}, '
                f'reason={executed.watch.reason}'
            )
        if executed.endpoint is None:
            raise RuntimeError('fresh endpoint sample is unavailable')
        if not executed.endpoint_guard.safe:
            raise RuntimeError(
                'fresh tangent endpoint rejected: '
                f'{executed.endpoint_guard.reason}'
            )
        if not executed.neighbour_guard.safe:
            raise RuntimeError(
                'fresh neighbour endpoint rejected: '
                f'{executed.neighbour_guard.reason}'
            )
        final = executed.endpoint
        final_guard = executed.endpoint_guard
        neighbour = executed.neighbour_guard
        left = executed.left_contact
        right = executed.right_contact
        palm = executed.palm_contact
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=RECOVERY_EVENT,
            diagnostic_only=True,
            gazebo_truth_used=True,
            arm_command_count=arm_command_count,
            measured_start_q=current.world.arm.tolist(),
            measured_endpoint_q=final.world.arm.tolist(),
            planned_endpoint_q=waypoint.positions.tolist(),
            book_position=final.world.book.position.tolist(),
            book_quaternion=final.world.book.quaternion.tolist(),
            measured_aperture_m=final.world.aperture_m,
            exact_left_contact=left,
            exact_right_contact=right,
            exact_palm_contact=palm,
            fixed_book_slip_characterization=True,
            gravity_support_authorized=False,
            base_motion_commanded=False,
            gripper_motion_commanded=False,
            right_arm_motion_commanded=False,
            next_motion_authorized=False,
            **final_guard.metrics,
            **neighbour.metrics,
        )
    except Exception as exc:
        _emit(
            'result',
            passed=False,
            stage=stage,
            reason=f'{type(exc).__name__}: {exc}',
            diagnostic_only=True,
            arm_command_count=arm_command_count,
            automatic_recovery_commanded=False,
            gravity_support_authorized=False,
            base_motion_commanded=False,
            gripper_motion_commanded=False,
            right_arm_motion_commanded=False,
            next_motion_authorized=False,
        )
        raise
    finally:
        node._cancel.set()
        executor.shutdown()
        thread.join(timeout=3.0)
        node.destroy_node()
        if spin_errors:
            raise RuntimeError(f'executor spin failed: {spin_errors[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Diagnostic-only one-millimetre book-tangent slip probe from '
            'the natural q6 pressure cage'
        )
    )
    parser.add_argument(
        '--confirm-diagnostic-one-mm-tangent-slip',
        action='store_true',
        help=(
            'authorize exactly one guarded left-arm 1 mm tangent command; '
            'no continuation is authorized'
        ),
    )
    arguments = parser.parse_args()
    if not arguments.confirm_diagnostic_one_mm_tangent_slip:
        parser.error(
            '--confirm-diagnostic-one-mm-tangent-slip is required'
        )
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
