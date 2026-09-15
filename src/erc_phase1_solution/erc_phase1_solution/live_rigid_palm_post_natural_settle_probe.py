#!/usr/bin/env python3
"""Diagnostic continuation from the one-arm natural-settle cage.

This module is deliberately outside the competition mission.  It consumes
Gazebo poses and temporary exact-contact sensors to inspect the physical
seed-101 state.  Inspection is the default and cannot actuate the robot.  The
only optional live motion is a densely preflighted, fixed-orientation sequence
of 5 mm *left-arm-only* outward legs.  It stops once the measured target OBB is
20 mm clear of the shelf; it never moves the base, opens the gripper, compacts
the arm, navigates, or places the book.

Unlike the older post-tangent probe, this continuation does not infer bearing
support from the centre-of-mass projection alone.  The selected upward palm
face must also be within a tight gap / penetration band of the measured book
OBB.  The production ``_gravity_supported_payload`` state remains false until
that complete, fresh, stationary three-point proof has passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np

from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    GuardResult,
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
    _angle_error,
    _emit,
    _fresh_cage_gate,
)
from erc_phase1_solution.live_rigid_palm_natural_settle_probe import (
    EXPECTED_CAGED_APERTURE_M,
    SETTLE_FINAL_TILT_MAXIMUM_RAD,
    SETTLE_FINAL_TILT_MINIMUM_RAD,
    SETTLE_OFF_AXIS_TILT_LIMIT_RAD,
    _exact_contacts,
    _probe_node_types,
    settle_attitude_metrics,
)
from erc_phase1_solution.live_rigid_palm_post_slide_transport_probe import (
    ARM_BASE_CUMULATIVE_LIMIT_M,
    ARM_BASE_YAW_LIMIT_RAD,
    ARM_CUMULATIVE_ATTACHMENT_POSITION_LIMIT_M,
    ARM_CUMULATIVE_ROTATION_LIMIT_RAD,
    ARM_CUMULATIVE_CORNER_ERROR_LIMIT_M,
    ARM_ENDPOINT_JOINT_LIMIT_RAD,
    MINIMUM_SUPPORT_DEPTH_MARGIN_M,
    MINIMUM_SUPPORT_LATERAL_MARGIN_M,
    MINIMUM_SHELF_CLEARANCE_M,
    SCENE_AGE_LIMIT_S,
    SCENE_FUTURE_TOLERANCE_S,
    SUPPORT_POLYGON_INSET_M,
    SUPPORT_SURFACE_LAYER_M,
    AttachmentAnchor,
    PostSlideTransportFailure,
    SupportAssessment,
    TransportObservation,
    TransportWaypoint,
    _attachment_errors,
    _base_transform_from_pose,
    _convex_hull,
    _outward_world,
    _palm_local_triangles,
    _transport_observation,
    _unexpected_contact,
    arm_transport_step_guard,
    capture_attachment_anchor,
    finite_palm_support_guard,
    preflight_attached_arm_route,
    shelf_clearance_m,
    solve_outward_waypoints,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
    rotation_matrix_distance,
)
from erc_phase1_solution.live_rigid_palm_tangent_slide_probe import (
    ARM_STAGE_DURATION_S,
    IK_MAXIMUM_JOINT_STEP_RAD,
    IK_ORIENTATION_TOLERANCE_RAD,
    IK_POSITION_TOLERANCE_M,
    _dense_arm_segment,
    _evaluate_dense_route,
    _proper_transform,
)
from erc_phase1_solution.rigid_palm_live_preflight import (
    MeasuredRobotState,
    _measured_finger_transforms_relative_to_palm,
    _node_time_seconds,
)
from erc_phase1_solution.rigid_palm_preflight import (
    PALM_COLLISION_LINK,
    PreflightResult,
)


POST_NATURAL_EVENT = 'rigid_palm_post_natural_arm_clearance'

STATIONARY_DWELL_S = 0.50
STATIONARY_BOOK_TRANSLATION_LIMIT_M = 0.00050
STATIONARY_BOOK_ROTATION_LIMIT_RAD = math.radians(0.50)
STATIONARY_CORNER_LIMIT_M = 0.00075
STATIONARY_ARM_LIMIT_RAD = 0.0040
STATIONARY_HAND_TRANSLATION_LIMIT_M = 0.00075
STATIONARY_HAND_ROTATION_LIMIT_RAD = math.radians(0.50)
STATIONARY_BASE_TRANSLATION_LIMIT_M = 0.00035
STATIONARY_BASE_YAW_LIMIT_RAD = 0.00050
STATIONARY_APERTURE_SPAN_LIMIT_M = 0.00035

NATURAL_APERTURE_LIMIT_M = 0.00035
NATURAL_BASE_POSITION_LIMIT_M = 0.005
NATURAL_BASE_YAW_LIMIT_RAD = 0.008
# A 25 degree settle can make the axis-aligned rear extent much larger even
# while the lower edge remains close to the shelf.  This is only a same-world
# recognition cap; the dense mesh preflight must still prove every extraction
# sample and the terminal 20 mm clearance.
NATURAL_INITIAL_MAXIMUM_SHELF_OVERLAP_M = 0.100

PALM_FACE_GAP_LIMIT_M = 0.00075
PALM_FACE_PENETRATION_LIMIT_M = 0.00100
NATURAL_MAXIMUM_ARM_OUTWARD_M = 0.130
DISPATCH_ARM_CHANGE_LIMIT_RAD = 0.00050

BOX_EDGES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
)
GEOMETRY_TOLERANCE_M = 1e-8

PALM_REPOSITION_QUANTUM_M = 0.001
PALM_REPOSITION_MAXIMUM_M = 0.060
PALM_REPOSITION_MINIMUM_SLACK_GAIN_M = 0.00050
SHELF_EDGE_PROXIMITY_LIMIT_M = 0.00075
# The official shelf is placed with ``1.5708`` rather than an exact pi/2
# rotation.  Its nominally horizontal 300 mm top triangles consequently span
# about 1.102 micrometres in world z.  Keep this read-only classifier tight,
# but do not reject the official mesh for that authored-pose round-off.
SHELF_HORIZONTAL_TRIANGLE_SPAN_LIMIT_M = 0.000005


@dataclass(frozen=True)
class RawPalmSupportGeometry:
    """Unthresholded selected-face geometry retained on support failures."""

    projection_world: np.ndarray
    polygon_world: np.ndarray
    signed_slacks_m: np.ndarray
    inset_slacks_m: np.ndarray
    support_normal_world: np.ndarray
    depth_world: np.ndarray
    lateral_world: np.ndarray
    plane_offset_m: float
    book_minimum_signed_plane_distance_m: float
    book_com_height_m: float
    support_plane_span_m: float
    recovery_translation_world: np.ndarray
    recovery_direction_world: np.ndarray
    recovery_distance_m: float

    @property
    def metrics(self) -> dict[str, float]:
        result = {
            'raw_com_projection_world_x_m': float(self.projection_world[0]),
            'raw_com_projection_world_y_m': float(self.projection_world[1]),
            'raw_com_projection_world_z_m': float(self.projection_world[2]),
            'raw_palm_hull_vertex_count': float(len(self.polygon_world)),
            'raw_palm_min_signed_slack_m': float(
                np.min(self.signed_slacks_m)
            ),
            'raw_palm_min_inset_slack_m': float(
                np.min(self.inset_slacks_m)
            ),
            'raw_support_recovery_distance_m': float(
                self.recovery_distance_m
            ),
            'raw_support_recovery_translation_world_x_m': float(
                self.recovery_translation_world[0]
            ),
            'raw_support_recovery_translation_world_y_m': float(
                self.recovery_translation_world[1]
            ),
            'raw_support_recovery_translation_world_z_m': float(
                self.recovery_translation_world[2]
            ),
            'raw_support_recovery_direction_world_x': float(
                self.recovery_direction_world[0]
            ),
            'raw_support_recovery_direction_world_y': float(
                self.recovery_direction_world[1]
            ),
            'raw_support_recovery_direction_world_z': float(
                self.recovery_direction_world[2]
            ),
            'raw_support_plane_normal_world_x': float(
                self.support_normal_world[0]
            ),
            'raw_support_plane_normal_world_y': float(
                self.support_normal_world[1]
            ),
            'raw_support_plane_normal_world_z': float(
                self.support_normal_world[2]
            ),
            'raw_support_plane_up_alignment': float(
                self.support_normal_world[2]
            ),
            'raw_support_plane_tilt_rad': math.acos(float(np.clip(
                self.support_normal_world[2], -1.0, 1.0
            ))),
            'raw_support_depth_world_x': float(self.depth_world[0]),
            'raw_support_depth_world_y': float(self.depth_world[1]),
            'raw_support_depth_world_z': float(self.depth_world[2]),
            'raw_support_lateral_world_x': float(self.lateral_world[0]),
            'raw_support_lateral_world_y': float(self.lateral_world[1]),
            'raw_support_lateral_world_z': float(self.lateral_world[2]),
            'raw_support_plane_offset_m': float(self.plane_offset_m),
            'raw_book_minimum_signed_plane_distance_m': float(
                self.book_minimum_signed_plane_distance_m
            ),
            'raw_book_com_height_above_palm_m': float(
                self.book_com_height_m
            ),
            'raw_support_plane_span_m': float(self.support_plane_span_m),
        }
        for index, (vertex, signed, inset) in enumerate(zip(
            self.polygon_world,
            self.signed_slacks_m,
            self.inset_slacks_m,
        )):
            result[f'raw_palm_hull_{index}_world_x_m'] = float(vertex[0])
            result[f'raw_palm_hull_{index}_world_y_m'] = float(vertex[1])
            result[f'raw_palm_hull_{index}_world_z_m'] = float(vertex[2])
            result[f'raw_palm_hull_edge_{index}_signed_slack_m'] = float(
                signed
            )
            result[f'raw_palm_hull_edge_{index}_inset_slack_m'] = float(inset)
        return result


@dataclass(frozen=True)
class PalmBearingAssessment:
    """COM support plus measured book-to-selected-palm-plane proximity."""

    safe: bool
    reason: str
    support: SupportAssessment
    signed_face_clearance_m: float
    face_gap_m: float
    face_penetration_m: float
    book_com_height_m: float
    support_plane_span_m: float
    contact_patch_inset_overlap: bool = False
    contact_patch_vertex_count: int = 0
    contact_patch_best_inset_slack_m: float = -math.inf
    raw_geometry: RawPalmSupportGeometry | None = None

    @property
    def metrics(self) -> dict[str, float]:
        result = {
            'support_depth_margin_m': float(self.support.depth_margin_m),
            'support_lateral_margin_m': float(self.support.lateral_margin_m),
            'support_area_m2': float(self.support.support_area_m2),
            'palm_face_signed_clearance_m': float(
                self.signed_face_clearance_m
            ),
            'palm_face_gap_m': float(self.face_gap_m),
            'palm_face_penetration_m': float(self.face_penetration_m),
            'book_com_height_above_palm_m': float(self.book_com_height_m),
            'support_plane_span_m': float(self.support_plane_span_m),
            'contact_patch_inset_overlap': float(
                self.contact_patch_inset_overlap
            ),
            'contact_patch_vertex_count': float(
                self.contact_patch_vertex_count
            ),
            'contact_patch_best_inset_slack_m': float(
                self.contact_patch_best_inset_slack_m
            ),
        }
        if self.raw_geometry is not None:
            result.update(self.raw_geometry.metrics)
            # Raw geometry replaces sentinel infinities from the older finite
            # support helper; live JSON should contain measurements, not NaN.
            result = {
                key: value
                for key, value in result.items()
                if math.isfinite(float(value))
            }
        return result


@dataclass(frozen=True)
class PalmRepositionCandidate:
    """One fixed-book, fixed-orientation, collision-preflighted IK candidate."""

    translation_distance_m: float
    translation_world: np.ndarray
    positions: np.ndarray
    target_hand_base: np.ndarray
    maximum_incremental_joint_change_rad: float
    ik_position_error_m: float
    ik_orientation_error_rad: float
    inset_slack_gain_m: float
    bearing: PalmBearingAssessment
    preflight: PreflightResult

    @property
    def metrics(self) -> dict[str, float]:
        return {
            'candidate_translation_distance_m': float(
                self.translation_distance_m
            ),
            'candidate_translation_world_x_m': float(
                self.translation_world[0]
            ),
            'candidate_translation_world_y_m': float(
                self.translation_world[1]
            ),
            'candidate_translation_world_z_m': float(
                self.translation_world[2]
            ),
            'candidate_maximum_incremental_joint_change_rad': float(
                self.maximum_incremental_joint_change_rad
            ),
            'candidate_ik_position_error_m': float(self.ik_position_error_m),
            'candidate_ik_orientation_error_rad': float(
                self.ik_orientation_error_rad
            ),
            'candidate_inset_slack_gain_m': float(self.inset_slack_gain_m),
            **self.bearing.metrics,
        }


@dataclass(frozen=True)
class PalmRepositionPlan:
    """Read-only result of the bounded palm-under-book candidate search."""

    safe: bool
    reason: str
    candidate: PalmRepositionCandidate | None
    attempts: int
    ik_solutions: int
    support_candidates: int
    collision_preflights: int
    last_preflight_code: str | None = None

    @property
    def metrics(self) -> dict[str, float]:
        result = {
            'candidate_attempt_count': float(self.attempts),
            'candidate_ik_solution_count': float(self.ik_solutions),
            'candidate_full_support_count': float(self.support_candidates),
            'candidate_collision_preflight_count': float(
                self.collision_preflights
            ),
        }
        if self.candidate is not None:
            result.update(self.candidate.metrics)
        return result


@dataclass(frozen=True)
class ShelfEdgeProximity:
    """Geometry-only OBB section near a shelf top/front edge."""

    available: bool
    reason: str
    geometry_consistent: bool
    shelf_plane_z_m: float = math.nan
    section_minimum_x_m: float = math.nan
    section_maximum_x_m: float = math.nan
    section_minimum_y_m: float = math.nan
    section_maximum_y_m: float = math.nan
    section_rear_signed_overlap_m: float = math.nan
    section_edge_distance_m: float = math.nan
    section_shelf_y_overlap_m: float = math.nan
    section_vertex_count: int = 0

    @property
    def metrics(self) -> dict[str, float]:
        values = {
            'shelf_edge_geometry_available': float(self.available),
            'shelf_edge_support_geometry_consistent': float(
                self.geometry_consistent
            ),
            'shelf_edge_plane_z_m': float(self.shelf_plane_z_m),
            'shelf_edge_section_minimum_x_m': float(
                self.section_minimum_x_m
            ),
            'shelf_edge_section_maximum_x_m': float(
                self.section_maximum_x_m
            ),
            'shelf_edge_section_minimum_y_m': float(
                self.section_minimum_y_m
            ),
            'shelf_edge_section_maximum_y_m': float(
                self.section_maximum_y_m
            ),
            'shelf_edge_section_rear_signed_overlap_m': float(
                self.section_rear_signed_overlap_m
            ),
            'shelf_edge_section_distance_m': float(
                self.section_edge_distance_m
            ),
            'shelf_edge_section_shelf_y_overlap_m': float(
                self.section_shelf_y_overlap_m
            ),
            'shelf_edge_section_vertex_count': float(
                self.section_vertex_count
            ),
        }
        return {
            key: value
            for key, value in values.items()
            if math.isfinite(value)
        }


def _invalid_bearing(
    reason: str,
    support: SupportAssessment,
    raw_geometry: RawPalmSupportGeometry | None = None,
) -> PalmBearingAssessment:
    if raw_geometry is None:
        signed_clearance = math.nan
        gap = math.inf
        penetration = math.inf
        com_height = math.nan
        plane_span = math.inf
    else:
        signed_clearance = float(
            raw_geometry.book_minimum_signed_plane_distance_m
        )
        gap = max(0.0, signed_clearance)
        penetration = max(0.0, -signed_clearance)
        com_height = float(raw_geometry.book_com_height_m)
        plane_span = float(raw_geometry.support_plane_span_m)
    return PalmBearingAssessment(
        False,
        reason,
        support,
        signed_clearance,
        gap,
        penetration,
        com_height,
        plane_span,
        raw_geometry=raw_geometry,
    )


def _inset_convex_polygon(
    polygon: Sequence[Sequence[float]], inset_m: float
) -> np.ndarray:
    """Offset every CCW support edge inward by the requested distance."""

    points = np.asarray(polygon, dtype=float)
    inset = float(inset_m)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.all(np.isfinite(points))
        or not math.isfinite(inset)
        or inset < 0.0
    ):
        raise ValueError('inset polygon is malformed')
    # STL face triangulation can leave a mathematically collinear seam vertex
    # in the convex hull with sub-nanometre numerical bow.  Treat only that
    # negligible seam as collinear before intersecting offset edge lines;
    # otherwise two almost-parallel adjacent lines can manufacture a remote
    # vertex and make a perfectly valid rectangular support face disappear.
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index in range(len(points)):
            previous = points[index - 1]
            current = points[index]
            following = points[(index + 1) % len(points)]
            chord = following - previous
            length = float(np.linalg.norm(chord))
            if length <= 1e-12:
                continue
            offset = current - previous
            distance = abs(float(
                chord[0] * offset[1] - chord[1] * offset[0]
            )) / length
            between = float(np.dot(
                current - previous, current - following
            )) <= 1e-12
            if distance <= 1e-9 and between:
                points = np.delete(points, index, axis=0)
                changed = True
                break
    signed_area = 0.5 * float(np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - points[:, 1] * np.roll(points[:, 0], -1)
    ))
    if abs(signed_area) <= 1e-14:
        raise ValueError('inset polygon is degenerate')
    if signed_area < 0.0:
        points = points[::-1].copy()

    inward: list[np.ndarray] = []
    offsets: list[float] = []
    for first, second in zip(points, np.roll(points, -1, axis=0)):
        edge = second - first
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            raise ValueError('inset polygon has a zero-length edge')
        normal = np.asarray([-edge[1], edge[0]], dtype=float) / length
        inward.append(normal)
        offsets.append(float(np.dot(normal, first) + inset))

    result: list[np.ndarray] = []
    for index in range(len(points)):
        previous = (index - 1) % len(points)
        matrix = np.vstack((inward[previous], inward[index]))
        if abs(float(np.linalg.det(matrix))) <= 1e-12:
            raise ValueError('inset polygon has parallel adjacent edges')
        result.append(np.linalg.solve(
            matrix,
            np.asarray([offsets[previous], offsets[index]], dtype=float),
        ))
    eroded = np.asarray(result, dtype=float)
    for point in eroded:
        if any(
            float(np.dot(normal, point)) < offset - 1e-9
            for normal, offset in zip(inward, offsets)
        ):
            raise ValueError('support polygon disappears after inset')
    return eroded


def _minimum_inward_slack(
    polygon: np.ndarray, point: np.ndarray
) -> float:
    slacks: list[float] = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            continue
        inward = np.asarray([-edge[1], edge[0]], dtype=float) / length
        slacks.append(float(np.dot(inward, point - first)))
    if not slacks:
        raise ValueError('polygon has no usable edges')
    return min(slacks)


def _point_inside_convex_polygon(
    polygon: np.ndarray, point: np.ndarray
) -> bool:
    return _minimum_inward_slack(polygon, point) >= -1e-9


def _closest_point_on_convex_polygon(
    polygon: np.ndarray, point: np.ndarray
) -> np.ndarray:
    if _point_inside_convex_polygon(polygon, point):
        return point.copy()
    candidates: list[np.ndarray] = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        magnitude = float(np.dot(edge, edge))
        if magnitude <= 1e-16:
            continue
        fraction = float(np.dot(point - first, edge) / magnitude)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        candidates.append(first + fraction * edge)
    if not candidates:
        raise ValueError('support polygon has no closest point')
    return min(candidates, key=lambda value: float(np.linalg.norm(point - value)))


def _horizontal_obb_section(corners: np.ndarray, plane_z_m: float) -> np.ndarray:
    points: list[np.ndarray] = []
    for first, second in BOX_EDGES:
        start = corners[first]
        finish = corners[second]
        start_distance = float(start[2] - plane_z_m)
        finish_distance = float(finish[2] - plane_z_m)
        if abs(start_distance) <= GEOMETRY_TOLERANCE_M:
            points.append(start.copy())
        if abs(finish_distance) <= GEOMETRY_TOLERANCE_M:
            points.append(finish.copy())
        if start_distance * finish_distance < 0.0:
            fraction = start_distance / (start_distance - finish_distance)
            points.append(start + fraction * (finish - start))
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.unique(np.round(np.asarray(points), decimals=12), axis=0)


def shelf_edge_proximity_geometry(
    book_corners_world: Sequence[Sequence[float]],
    shelf_triangles_world: Sequence[Sequence[Sequence[float]]],
    *,
    shelf_front_x_m: float = SHELF_FRONT_X_M,
    proximity_limit_m: float = SHELF_EDGE_PROXIMITY_LIMIT_M,
) -> ShelfEdgeProximity:
    """Report an OBB/shelf-lip coincidence without claiming contact proof."""

    try:
        corners = np.asarray(book_corners_world, dtype=float)
        shelf = np.asarray(shelf_triangles_world, dtype=float)
        front = float(shelf_front_x_m)
        proximity = float(proximity_limit_m)
        if (
            corners.shape != (8, 3)
            or shelf.ndim != 3
            or shelf.shape[1:] != (3, 3)
            or not len(shelf)
            or not np.all(np.isfinite(corners))
            or not np.all(np.isfinite(shelf))
            or not math.isfinite(front)
            or not math.isfinite(proximity)
            or proximity < 0.0
        ):
            raise ValueError('shelf-edge geometry is malformed')

        horizontal: dict[float, list[np.ndarray]] = {}
        book_y = (float(np.min(corners[:, 1])), float(np.max(corners[:, 1])))
        book_z = (float(np.min(corners[:, 2])), float(np.max(corners[:, 2])))
        for triangle in shelf:
            cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            magnitude = float(np.linalg.norm(cross))
            if (
                magnitude <= 1e-12
                or abs(float(cross[2])) / magnitude < 0.999
                or float(np.ptp(triangle[:, 2]))
                > SHELF_HORIZONTAL_TRIANGLE_SPAN_LIMIT_M
                or float(np.max(triangle[:, 0])) < front - 0.002
                or float(np.min(triangle[:, 0])) > front + 0.002
                or float(np.max(triangle[:, 1])) < book_y[0]
                or float(np.min(triangle[:, 1])) > book_y[1]
            ):
                continue
            plane_z = float(np.mean(triangle[:, 2]))
            if plane_z < book_z[0] - proximity or plane_z > book_z[1] + proximity:
                continue
            key = round(plane_z, 7)
            horizontal.setdefault(key, []).append(triangle)

        candidates: list[tuple[float, ShelfEdgeProximity]] = []
        for triangles in horizontal.values():
            surface = np.concatenate(triangles, axis=0)
            plane_z = float(np.mean(surface[:, 2]))
            section = _horizontal_obb_section(corners, plane_z)
            if not len(section):
                continue
            minimum_x = float(np.min(section[:, 0]))
            maximum_x = float(np.max(section[:, 0]))
            minimum_y = float(np.min(section[:, 1]))
            maximum_y = float(np.max(section[:, 1]))
            y_overlap = max(
                0.0,
                min(maximum_y, float(np.max(surface[:, 1])))
                - max(minimum_y, float(np.min(surface[:, 1]))),
            )
            signed_overlap = maximum_x - front
            distance = abs(signed_overlap)
            consistent = bool(
                distance <= proximity + 1e-12
                and y_overlap > GEOMETRY_TOLERANCE_M
            )
            result = ShelfEdgeProximity(
                True,
                'geometry_consistent' if consistent else 'edge_not_near_section',
                consistent,
                plane_z,
                minimum_x,
                maximum_x,
                minimum_y,
                maximum_y,
                signed_overlap,
                distance,
                y_overlap,
                len(section),
            )
            candidates.append((distance, result))
        if not candidates:
            return ShelfEdgeProximity(
                False, 'no_relevant_shelf_top_section', False
            )
        return min(candidates, key=lambda item: item[0])[1]
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return ShelfEdgeProximity(False, f'invalid_geometry:{exc}', False)


def _raw_palm_support_geometry(
    palm_triangles_local: Sequence[Sequence[Sequence[float]]],
    palm_world_transform: Sequence[Sequence[float]],
    book_corners_world: Sequence[Sequence[float]],
    outward_world: Sequence[float],
    *,
    polygon_inset_m: float,
) -> RawPalmSupportGeometry:
    """Measure selected-face COM geometry without applying pass thresholds."""

    triangles = np.asarray(palm_triangles_local, dtype=float)
    palm = _proper_transform(palm_world_transform, 'world palm')
    corners = np.asarray(book_corners_world, dtype=float)
    outward = np.asarray(outward_world, dtype=float)
    inset = float(polygon_inset_m)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or not len(triangles)
        or corners.shape != (8, 3)
        or outward.shape != (3,)
        or not np.all(np.isfinite(triangles))
        or not np.all(np.isfinite(corners))
        or not np.all(np.isfinite(outward))
        or not math.isfinite(inset)
        or inset < 0.0
    ):
        raise ValueError('raw palm support geometry is malformed')

    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    alignment = palm[:3, :3].T @ up
    axis = int(np.argmax(np.abs(alignment)))
    sign = 1.0 if alignment[axis] >= 0.0 else -1.0
    normal = sign * palm[:3, axis]
    normal /= np.linalg.norm(normal)
    signed_coordinate = sign * triangles[:, :, axis]
    top = float(np.max(signed_coordinate))
    selected = triangles[
        np.all(
            signed_coordinate >= top - SUPPORT_SURFACE_LAYER_M,
            axis=1,
        )
    ]
    if not len(selected):
        raise ValueError('raw palm support surface is unavailable')
    world_vertices = (
        selected.reshape((-1, 3)) @ palm[:3, :3].T
        + palm[:3, 3]
    )
    depth = outward - normal * float(np.dot(outward, normal))
    depth_norm = float(np.linalg.norm(depth))
    # This is a diagnostic reconstruction, not the safety gate.  Preserve
    # finite geometry for a slanted pressure cage even when the production
    # support guard will reject its tangent alignment.
    if depth_norm <= 1e-9:
        raise ValueError('raw support outward direction is degenerate')
    depth /= depth_norm
    lateral = np.cross(normal, depth)
    lateral /= np.linalg.norm(lateral)
    coordinates = np.column_stack(
        (world_vertices @ depth, world_vertices @ lateral)
    )
    polygon_2d = _convex_hull(coordinates)
    recovery_polygon = _inset_convex_polygon(
        polygon_2d,
        inset + max(
            MINIMUM_SUPPORT_DEPTH_MARGIN_M,
            MINIMUM_SUPPORT_LATERAL_MARGIN_M,
        ),
    )
    plane_offset = float(np.dot(normal, palm[:3, 3]) + top)
    center = np.mean(corners, axis=0)
    book_signed = corners @ normal - plane_offset
    book_minimum_signed = float(np.min(book_signed))
    book_com_height = float(np.dot(center, normal) - plane_offset)
    gravity = np.asarray([0.0, 0.0, -1.0], dtype=float)
    denominator = float(np.dot(normal, gravity))
    if abs(denominator) <= 1e-9:
        raise ValueError('raw gravity is parallel to support')
    ray_distance = (
        plane_offset - float(np.dot(normal, center))
    ) / denominator
    projection = center + ray_distance * gravity
    point_2d = np.asarray(
        [np.dot(projection, depth), np.dot(projection, lateral)],
        dtype=float,
    )
    signed_slacks_list: list[float] = []
    for first, second in zip(polygon_2d, np.roll(polygon_2d, -1, axis=0)):
        edge = second - first
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            raise ValueError('raw support polygon has a zero-length edge')
        inward = np.asarray([-edge[1], edge[0]], dtype=float) / length
        signed_slacks_list.append(float(np.dot(inward, point_2d - first)))
    signed_slacks = np.asarray(signed_slacks_list, dtype=float)
    inset_slacks = signed_slacks - inset
    closest = _closest_point_on_convex_polygon(recovery_polygon, point_2d)
    recovery_2d = point_2d - closest
    recovery_world = recovery_2d[0] * depth + recovery_2d[1] * lateral
    recovery_distance = float(np.linalg.norm(recovery_world))
    recovery_direction = (
        np.zeros(3, dtype=float)
        if recovery_distance <= 1e-12
        else recovery_world / recovery_distance
    )
    polygon_world = (
        polygon_2d[:, 0, None] * depth
        + polygon_2d[:, 1, None] * lateral
        + normal * plane_offset
    )
    plane_span = float(np.ptp(polygon_world @ normal))
    return RawPalmSupportGeometry(
        projection,
        polygon_world,
        signed_slacks,
        inset_slacks,
        normal,
        depth,
        lateral,
        plane_offset,
        book_minimum_signed,
        book_com_height,
        plane_span,
        recovery_world,
        recovery_direction,
        recovery_distance,
    )


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _segments_intersect(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
) -> bool:
    direction = second - first
    other_direction = fourth - third
    offset = third - first
    denominator = _cross_2d(direction, other_direction)
    tolerance = 1e-10
    if abs(denominator) <= tolerance:
        if abs(_cross_2d(offset, direction)) > tolerance:
            return False
        magnitude = float(np.dot(direction, direction))
        if magnitude <= tolerance:
            return bool(np.linalg.norm(first - third) <= tolerance)
        start = float(np.dot(third - first, direction) / magnitude)
        finish = float(np.dot(fourth - first, direction) / magnitude)
        lower, upper = sorted((start, finish))
        return upper >= -tolerance and lower <= 1.0 + tolerance
    along_first = _cross_2d(offset, other_direction) / denominator
    along_second = _cross_2d(offset, direction) / denominator
    return bool(
        -tolerance <= along_first <= 1.0 + tolerance
        and -tolerance <= along_second <= 1.0 + tolerance
    )


def _ordered_contact_patch(points: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or len(values) == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError('contact patch is malformed')
    unique = np.unique(np.round(values, decimals=12), axis=0)
    if len(unique) <= 2:
        return unique
    try:
        return _convex_hull(unique)
    except ValueError:
        distances = np.linalg.norm(
            unique[:, None, :] - unique[None, :, :], axis=2
        )
        first, second = np.unravel_index(
            int(np.argmax(distances)), distances.shape
        )
        return np.asarray([unique[first], unique[second]], dtype=float)


def _patch_edges(patch: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if len(patch) < 2:
        return ()
    if len(patch) == 2:
        return ((patch[0], patch[1]),)
    return tuple(zip(patch, np.roll(patch, -1, axis=0)))


def _convex_patch_overlaps(
    patch: np.ndarray, polygon: np.ndarray
) -> bool:
    if any(_point_inside_convex_polygon(polygon, point) for point in patch):
        return True
    if len(patch) >= 3 and any(
        _point_inside_convex_polygon(patch, point) for point in polygon
    ):
        return True
    polygon_edges = _patch_edges(polygon)
    return any(
        _segments_intersect(first, second, third, fourth)
        for first, second in _patch_edges(patch)
        for third, fourth in polygon_edges
    )


def _obb_plane_contact_patch(
    corners: np.ndarray,
    signed: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    lateral: np.ndarray,
) -> np.ndarray:
    """Project the OBB/plane cross-section or its closest support feature."""

    minimum = float(np.min(signed))
    maximum = float(np.max(signed))
    world_points: list[np.ndarray] = []
    if minimum <= GEOMETRY_TOLERANCE_M and maximum >= -GEOMETRY_TOLERANCE_M:
        for first, second in BOX_EDGES:
            first_distance = float(signed[first])
            second_distance = float(signed[second])
            if abs(first_distance) <= GEOMETRY_TOLERANCE_M:
                world_points.append(corners[first])
            if abs(second_distance) <= GEOMETRY_TOLERANCE_M:
                world_points.append(corners[second])
            if first_distance * second_distance < 0.0:
                fraction = first_distance / (
                    first_distance - second_distance
                )
                world_points.append(
                    corners[first]
                    + fraction * (corners[second] - corners[first])
                )
    else:
        closest = np.flatnonzero(
            signed <= minimum + GEOMETRY_TOLERANCE_M
        )
        world_points.extend(corners[index] for index in closest)
    if not world_points:
        raise ValueError('OBB has no plane contact feature')
    projected = np.asarray([
        point - float(np.dot(point, normal)) * normal
        for point in world_points
    ])
    coordinates = np.column_stack((projected @ depth, projected @ lateral))
    return _ordered_contact_patch(coordinates)


def palm_bearing_support_guard(
    palm_triangles_local: Sequence[Sequence[Sequence[float]]],
    palm_world_transform: Sequence[Sequence[float]],
    book_corners_world: Sequence[Sequence[float]],
    outward_world: Sequence[float],
    *,
    polygon_inset_m: float = SUPPORT_POLYGON_INSET_M,
    maximum_face_gap_m: float = PALM_FACE_GAP_LIMIT_M,
    maximum_face_penetration_m: float = PALM_FACE_PENETRATION_LIMIT_M,
) -> PalmBearingAssessment:
    """Require the measured OBB to touch the same palm face supporting its COM.

    ``finite_palm_support_guard`` proves that the vertical COM projection lies
    inside an eroded, mesh-derived palm polygon.  It intentionally does not
    test how far the book is from that plane.  This wrapper closes that gap by
    measuring the minimum signed OBB clearance to the *selected* support plane.
    """

    try:
        raw_geometry = _raw_palm_support_geometry(
            palm_triangles_local,
            palm_world_transform,
            book_corners_world,
            outward_world,
            polygon_inset_m=polygon_inset_m,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        raw_geometry = None
    support = finite_palm_support_guard(
        palm_triangles_local,
        palm_world_transform,
        book_corners_world,
        outward_world,
        polygon_inset_m=polygon_inset_m,
    )
    if not support.safe:
        return _invalid_bearing(support.reason, support, raw_geometry)

    try:
        corners = np.asarray(book_corners_world, dtype=float)
        polygon = np.asarray(support.polygon_world, dtype=float)
        normal = np.asarray(support.support_normal_world, dtype=float)
        outward = np.asarray(outward_world, dtype=float)
        inset = float(polygon_inset_m)
        maximum_gap = float(maximum_face_gap_m)
        maximum_penetration = float(maximum_face_penetration_m)
        if (
            corners.shape != (8, 3)
            or polygon.ndim != 2
            or polygon.shape[1] != 3
            or len(polygon) < 3
            or normal.shape != (3,)
            or outward.shape != (3,)
            or not np.all(np.isfinite(corners))
            or not np.all(np.isfinite(polygon))
            or not np.all(np.isfinite(normal))
            or not np.all(np.isfinite(outward))
            or not math.isfinite(inset)
            or inset < 0.0
            or not math.isfinite(maximum_gap)
            or not math.isfinite(maximum_penetration)
            or maximum_gap < 0.0
            or maximum_penetration < 0.0
        ):
            raise ValueError('bearing geometry is malformed')
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError('bearing normal is degenerate')
        normal = normal / norm
        plane_coordinates = polygon @ normal
        plane_span = float(np.ptp(plane_coordinates))
        plane_offset = float(np.mean(plane_coordinates))
        signed = corners @ normal - plane_offset
        signed_clearance = float(np.min(signed))
        gap = max(0.0, signed_clearance)
        penetration = max(0.0, -signed_clearance)
        com_height = float(np.dot(np.mean(corners, axis=0), normal) - plane_offset)
        depth = outward - normal * float(np.dot(outward, normal))
        depth_norm = float(np.linalg.norm(depth))
        if depth_norm <= 0.95:
            raise ValueError('bearing outward direction is not tangent')
        depth /= depth_norm
        lateral = np.cross(normal, depth)
        lateral /= np.linalg.norm(lateral)
        polygon_2d = np.column_stack((polygon @ depth, polygon @ lateral))
        inset_polygon = _inset_convex_polygon(polygon_2d, inset)
        contact_patch = _obb_plane_contact_patch(
            corners, signed, normal, depth, lateral
        )
        patch_overlap = _convex_patch_overlaps(
            contact_patch, inset_polygon
        )
        best_patch_slack = max(
            _minimum_inward_slack(inset_polygon, point)
            for point in contact_patch
        )
        if patch_overlap:
            best_patch_slack = max(0.0, best_patch_slack)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return _invalid_bearing(
            'invalid_bearing_geometry', support, raw_geometry
        )

    assessment = PalmBearingAssessment(
        True,
        'ok',
        support,
        signed_clearance,
        gap,
        penetration,
        com_height,
        plane_span,
        patch_overlap,
        len(contact_patch),
        best_patch_slack,
        raw_geometry,
    )
    if plane_span > 1e-6:
        return PalmBearingAssessment(
            False, 'selected_support_face_not_planar', support,
            signed_clearance, gap, penetration, com_height, plane_span,
            patch_overlap, len(contact_patch), best_patch_slack,
            raw_geometry,
        )
    if gap > maximum_gap:
        return PalmBearingAssessment(
            False, 'book_not_on_selected_palm_face', support,
            signed_clearance, gap, penetration, com_height, plane_span,
            patch_overlap, len(contact_patch), best_patch_slack,
            raw_geometry,
        )
    if penetration > maximum_penetration:
        return PalmBearingAssessment(
            False, 'book_palm_penetration_exceeds_limit', support,
            signed_clearance, gap, penetration, com_height, plane_span,
            patch_overlap, len(contact_patch), best_patch_slack,
            raw_geometry,
        )
    if not patch_overlap:
        return PalmBearingAssessment(
            False, 'book_contact_patch_outside_inset_palm', support,
            signed_clearance, gap, penetration, com_height, plane_span,
            patch_overlap, len(contact_patch), best_patch_slack,
            raw_geometry,
        )
    return assessment


def _support_mode_label(
    bearing: PalmBearingAssessment,
    *,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
) -> str:
    exact_three = bool(left_contact and right_contact and palm_contact)
    if bearing.safe and exact_three:
        return 'gravity_bearing_three_point_cage'
    if exact_three:
        return 'three_point_pressure_cage_without_gravity_bearing'
    if bearing.safe:
        return 'gravity_bearing_without_exact_three_point_cage'
    return 'neither_gravity_bearing_nor_exact_three_point_cage'


def plan_fixed_book_palm_reposition(
    *,
    chain: Any,
    start_positions: Sequence[float],
    base_world_transform: Sequence[Sequence[float]],
    book_corners_world: Sequence[Sequence[float]],
    palm_triangles_local: Sequence[Sequence[Sequence[float]]],
    outward_world: Sequence[float],
    starting_bearing: PalmBearingAssessment,
    collision_preflight: Callable[[np.ndarray], PreflightResult],
    polygon_inset_m: float = SUPPORT_POLYGON_INSET_M,
    translation_quantum_m: float = PALM_REPOSITION_QUANTUM_M,
    maximum_translation_m: float = PALM_REPOSITION_MAXIMUM_M,
    minimum_slack_gain_m: float = PALM_REPOSITION_MINIMUM_SLACK_GAIN_M,
) -> PalmRepositionPlan:
    """Find the shortest fixed-book palm translation passing all pure gates."""

    try:
        start = np.asarray(start_positions, dtype=float)
        base_world = _proper_transform(
            base_world_transform, 'candidate-plan world base'
        )
        corners = np.asarray(book_corners_world, dtype=float)
        palm_triangles = np.asarray(palm_triangles_local, dtype=float)
        outward = np.asarray(outward_world, dtype=float)
        polygon_inset = float(polygon_inset_m)
        quantum = float(translation_quantum_m)
        maximum = float(maximum_translation_m)
        minimum_gain = float(minimum_slack_gain_m)
        raw = starting_bearing.raw_geometry
        if (
            start.ndim != 1
            or len(start) < 2
            or not np.all(np.isfinite(start))
            or corners.shape != (8, 3)
            or palm_triangles.ndim != 3
            or palm_triangles.shape[1:] != (3, 3)
            or outward.shape != (3,)
            or not np.all(np.isfinite(corners))
            or not np.all(np.isfinite(palm_triangles))
            or not np.all(np.isfinite(outward))
            or raw is None
            or not math.isfinite(polygon_inset)
            or polygon_inset < 0.0
            or not math.isfinite(quantum)
            or not math.isfinite(maximum)
            or not math.isfinite(minimum_gain)
            or quantum <= 0.0
            or maximum < quantum
            or maximum > PALM_REPOSITION_MAXIMUM_M + 1e-12
            or minimum_gain <= 0.0
        ):
            raise ValueError('fixed-book palm candidate inputs are malformed')
        direction_world = np.asarray(
            raw.recovery_direction_world, dtype=float
        )
        direction_norm = float(np.linalg.norm(direction_world))
        if direction_norm <= 1e-12:
            return PalmRepositionPlan(
                False, 'no_support_recovery_direction', None, 0, 0, 0, 0
            )
        direction_world /= direction_norm
        reference_hand = _proper_transform(
            chain.forward(start), 'candidate-plan start hand'
        )
        starting_slack = float(np.min(raw.inset_slacks_m))
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return PalmRepositionPlan(
            False,
            f'invalid_candidate_plan:{exc}',
            None,
            0,
            0,
            0,
            0,
        )

    maximum_steps = int(math.floor(maximum / quantum + 1e-12))
    previous_solution = start.copy()
    ik_solutions = 0
    support_candidates = 0
    collision_preflights = 0
    last_candidate: PalmRepositionCandidate | None = None
    last_preflight_code: str | None = None
    for index in range(1, maximum_steps + 1):
        distance = quantum * index
        translation_world = direction_world * distance
        target_hand = reference_hand.copy()
        target_hand[:3, 3] = (
            reference_hand[:3, 3]
            + base_world[:3, :3].T @ translation_world
        )
        try:
            solution, _ = chain.solve(
                target_hand,
                [previous_solution, start],
                position_tolerance=IK_POSITION_TOLERANCE_M,
                orientation_tolerance=IK_ORIENTATION_TOLERANCE_RAD,
                max_iterations=600,
                fixed_positions={'torso_lift_joint': float(start[0])},
            )
            if solution is None:
                continue
            positions = np.asarray(solution, dtype=float)
            if (
                positions.shape != start.shape
                or not np.all(np.isfinite(positions))
            ):
                continue
            achieved = _proper_transform(
                chain.forward(positions), 'candidate-plan achieved hand'
            )
            pose_error = np.asarray(
                chain.pose_error(achieved, target_hand), dtype=float
            )
            if pose_error.shape != (6,) or not np.all(np.isfinite(pose_error)):
                continue
            position_error = float(np.linalg.norm(pose_error[:3]))
            orientation_error = float(np.linalg.norm(pose_error[3:]))
            incremental_joint_change = float(np.max(np.abs(
                positions[1:] - previous_solution[1:]
            )))
        except Exception:
            # One failed numerical branch must not turn a read-only bounded
            # search into an exception or motion authority.
            continue
        if (
            position_error > IK_POSITION_TOLERANCE_M
            or orientation_error > IK_ORIENTATION_TOLERANCE_RAD
            or incremental_joint_change > IK_MAXIMUM_JOINT_STEP_RAD
        ):
            continue
        ik_solutions += 1
        previous_solution = positions
        try:
            links = chain.link_transforms(positions)
            palm_world = base_world @ _proper_transform(
                links[PALM_COLLISION_LINK], 'candidate-plan palm FK'
            )
            bearing = palm_bearing_support_guard(
                palm_triangles,
                palm_world,
                corners,
                outward,
                polygon_inset_m=polygon_inset,
            )
            if bearing.raw_geometry is None:
                continue
            slack_gain = float(
                np.min(bearing.raw_geometry.inset_slacks_m)
                - starting_slack
            )
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            continue
        if slack_gain < minimum_gain or not bearing.safe:
            continue
        support_candidates += 1
        try:
            preflight = collision_preflight(positions.copy())
            if not isinstance(preflight, PreflightResult):
                raise TypeError('collision preflight returned the wrong type')
        except Exception as exc:
            preflight = PreflightResult(
                False, 'candidate_preflight_error', str(exc)
            )
        collision_preflights += 1
        last_preflight_code = str(preflight.code)
        last_candidate = PalmRepositionCandidate(
            distance,
            translation_world,
            positions,
            target_hand,
            incremental_joint_change,
            position_error,
            orientation_error,
            slack_gain,
            bearing,
            preflight,
        )
        if preflight.safe:
            return PalmRepositionPlan(
                True,
                'clear',
                last_candidate,
                index,
                ik_solutions,
                support_candidates,
                collision_preflights,
                last_preflight_code,
            )

    if collision_preflights:
        reason = 'reposition_collision_preflight_rejected'
    elif support_candidates:
        reason = 'reposition_collision_preflight_unavailable'
    elif ik_solutions:
        reason = 'no_full_support_reposition'
    else:
        reason = 'no_reposition_ik'
    return PalmRepositionPlan(
        False,
        reason,
        last_candidate,
        maximum_steps,
        ik_solutions,
        support_candidates,
        collision_preflights,
        last_preflight_code,
    )


def _upright_reference(book: BookSnapshot) -> BookSnapshot:
    return BookSnapshot(
        np.asarray(book.position, dtype=float).copy(),
        EXPECTED_RELEASED_BOOK_QUATERNION.copy(),
        np.asarray(book.minimum, dtype=float).copy(),
        np.asarray(book.maximum, dtype=float).copy(),
    )


def natural_settled_cage_guard(
    reference: TransportObservation,
    current: TransportObservation,
    bearing: PalmBearingAssessment,
    *,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    minimum_dwell_s: float = STATIONARY_DWELL_S,
) -> GuardResult:
    """Recognize only the stable, exact three-point natural-settle endpoint."""

    try:
        now = float(reference_time)
        required_dwell = float(minimum_dwell_s)
        age = now - float(current.observed_at)
        dwell = float(current.observed_at - reference.observed_at)
        book_translation = float(np.linalg.norm(
            current.book.position - reference.book.position
        ))
        book_rotation = quaternion_distance(
            reference.book.quaternion, current.book.quaternion
        )
        corner_drift = float(np.max(np.linalg.norm(
            current.corners - reference.corners, axis=1
        )))
        arm_drift = float(np.max(np.abs(current.arm - reference.arm)))
        hand_translation = float(np.linalg.norm(
            current.hand_world[:3, 3] - reference.hand_world[:3, 3]
        ))
        hand_rotation = rotation_matrix_distance(
            reference.hand_world[:3, :3], current.hand_world[:3, :3]
        )
        base_translation = float(np.linalg.norm(
            current.base[:2] - reference.base[:2]
        ))
        base_yaw = _angle_error(current.base[2], reference.base[2])
        base_checkpoint_error = float(np.linalg.norm(
            current.base[:2] - EXPECTED_RELEASED_BASE_POSE[:2]
        ))
        base_checkpoint_yaw = _angle_error(
            current.base[2], EXPECTED_RELEASED_BASE_POSE[2]
        )
        aperture_span = abs(current.aperture_m - reference.aperture_m)
        aperture_error = abs(current.aperture_m - EXPECTED_CAGED_APERTURE_M)
        reference_aperture_error = abs(
            reference.aperture_m - EXPECTED_CAGED_APERTURE_M
        )
        clearance = shelf_clearance_m(current.corners)
        overlap = -clearance
        current_attitude = settle_attitude_metrics(
            _upright_reference(current.book), current.book, current.base
        )
        reference_attitude = settle_attitude_metrics(
            _upright_reference(reference.book), reference.book, reference.base
        )
        current_tilt = float(current_attitude['tilt_rad'])
        reference_tilt = float(reference_attitude['tilt_rad'])
        off_axis = float(current_attitude['off_axis_tilt_rad'])
        bearing_safe = bool(bearing.safe)
        bearing_reason = str(bearing.reason)
        bearing_metrics = dict(bearing.metrics)
        if (
            not math.isfinite(now)
            or not math.isfinite(required_dwell)
            or required_dwell < 0.0
        ):
            raise ValueError('resume timing is invalid')
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})

    metrics = {
        'scene_age_s': age,
        'stationary_dwell_s': dwell,
        'book_stationary_translation_m': book_translation,
        'book_stationary_rotation_rad': book_rotation,
        'book_corner_stationary_error_m': corner_drift,
        'arm_stationary_error_rad': arm_drift,
        'hand_stationary_translation_m': hand_translation,
        'hand_stationary_rotation_rad': hand_rotation,
        'base_stationary_translation_m': base_translation,
        'base_stationary_yaw_rad': base_yaw,
        'base_checkpoint_error_m': base_checkpoint_error,
        'base_checkpoint_yaw_error_rad': base_checkpoint_yaw,
        'aperture_span_m': aperture_span,
        'aperture_error_m': aperture_error,
        'reference_aperture_error_m': reference_aperture_error,
        'shelf_clearance_m': clearance,
        'shelf_overlap_m': overlap,
        'natural_tilt_rad': current_tilt,
        'reference_natural_tilt_rad': reference_tilt,
        'off_axis_tilt_rad': off_axis,
        'exact_left_contact': float(bool(left_contact)),
        'exact_right_contact': float(bool(right_contact)),
        'exact_palm_contact': float(bool(palm_contact)),
        'exact_three_point_cage': float(bool(
            left_contact and right_contact and palm_contact
        )),
        'gravity_bearing_support_proven': float(bearing_safe),
        'force_closure_proven': 0.0,
        'pressure_cage_without_gravity_bearing': float(bool(
            left_contact
            and right_contact
            and palm_contact
            and not bearing_safe
        )),
        **bearing_metrics,
    }
    checks = (
        (age <= SCENE_AGE_LIMIT_S, 'scene_stale'),
        (age >= -SCENE_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (dwell + 1e-12 >= required_dwell, 'stationary_dwell_too_short'),
        (bool(left_contact), 'left_target_contact_missing'),
        (bool(right_contact), 'right_target_contact_missing'),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
        (bearing_safe, bearing_reason),
        (
            reference_aperture_error <= NATURAL_APERTURE_LIMIT_M,
            'reference_cage_aperture_wrong',
        ),
        (
            aperture_error <= NATURAL_APERTURE_LIMIT_M,
            'wrong_cage_aperture',
        ),
        (
            aperture_span <= STATIONARY_APERTURE_SPAN_LIMIT_M,
            'cage_aperture_not_stationary',
        ),
        (
            SETTLE_FINAL_TILT_MINIMUM_RAD - 1e-12
            <= reference_tilt
            <= SETTLE_FINAL_TILT_MAXIMUM_RAD + 1e-12,
            'reference_not_naturally_settled',
        ),
        (
            SETTLE_FINAL_TILT_MINIMUM_RAD - 1e-12
            <= current_tilt
            <= SETTLE_FINAL_TILT_MAXIMUM_RAD + 1e-12,
            'book_not_naturally_settled',
        ),
        (
            off_axis <= SETTLE_OFF_AXIS_TILT_LIMIT_RAD,
            'settle_off_axis_tilt',
        ),
        (
            overlap <= NATURAL_INITIAL_MAXIMUM_SHELF_OVERLAP_M,
            'unexpected_deep_shelf_overlap',
        ),
        (
            base_checkpoint_error <= NATURAL_BASE_POSITION_LIMIT_M,
            'wrong_natural_settle_base',
        ),
        (
            base_checkpoint_yaw <= NATURAL_BASE_YAW_LIMIT_RAD,
            'wrong_natural_settle_base_yaw',
        ),
        (
            book_translation <= STATIONARY_BOOK_TRANSLATION_LIMIT_M,
            'book_not_stationary',
        ),
        (
            book_rotation <= STATIONARY_BOOK_ROTATION_LIMIT_RAD,
            'book_rotation_not_stationary',
        ),
        (
            corner_drift <= STATIONARY_CORNER_LIMIT_M,
            'book_obb_not_stationary',
        ),
        (arm_drift <= STATIONARY_ARM_LIMIT_RAD, 'arm_not_stationary'),
        (
            hand_translation <= STATIONARY_HAND_TRANSLATION_LIMIT_M,
            'hand_not_stationary',
        ),
        (
            hand_rotation <= STATIONARY_HAND_ROTATION_LIMIT_RAD,
            'hand_rotation_not_stationary',
        ),
        (
            base_translation <= STATIONARY_BASE_TRANSLATION_LIMIT_M,
            'base_not_stationary',
        ),
        (
            base_yaw <= STATIONARY_BASE_YAW_LIMIT_RAD,
            'base_yaw_not_stationary',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _bearing_for_observation(
    node: Any,
    environment: Any,
    observation: TransportObservation,
    scene: Any,
) -> PalmBearingAssessment:
    links = node.chain.link_transforms(observation.arm)
    palm_world = np.asarray(scene.base_transform, dtype=float) @ links[
        PALM_COLLISION_LINK
    ]
    return palm_bearing_support_guard(
        _palm_local_triangles(environment.model),
        palm_world,
        observation.corners,
        _outward_world(observation.base),
        polygon_inset_m=max(
            SUPPORT_POLYGON_INSET_M,
            float(environment.config.collision_padding_m),
        ),
    )


def _plan_live_fixed_book_reposition(
    node: Any,
    runtime: SimpleNamespace,
    observation: TransportObservation,
    scene: Any,
    bearing: PalmBearingAssessment,
) -> PalmRepositionPlan:
    """Bind one captured scene to the pure IK/collision candidate planner."""

    try:
        environment = runtime.environment_preflight
        state = MeasuredRobotState(
            observation.arm.copy(), float(observation.aperture_m)
        )
        measured_relative = _measured_finger_transforms_relative_to_palm(
            node, scene, state
        )
        reference_time = _node_time_seconds(node)

        def collision_preflight(positions: np.ndarray) -> PreflightResult:
            samples = _dense_arm_segment(
                environment=environment,
                node=node,
                scene=scene,
                joint_samples=(observation.arm, positions),
                aperture_m=observation.aperture_m,
                measured_relative=measured_relative,
            )
            return _evaluate_dense_route(
                node=node,
                environment=environment,
                scene=scene,
                samples=samples,
                reference_time=reference_time,
            )

        return plan_fixed_book_palm_reposition(
            chain=node.chain,
            start_positions=observation.arm,
            base_world_transform=scene.base_transform,
            book_corners_world=observation.corners,
            palm_triangles_local=_palm_local_triangles(environment.model),
            outward_world=_outward_world(observation.base),
            starting_bearing=bearing,
            collision_preflight=collision_preflight,
            polygon_inset_m=max(
                SUPPORT_POLYGON_INSET_M,
                float(environment.config.collision_padding_m),
            ),
        )
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return PalmRepositionPlan(
            False,
            f'live_candidate_adapter_error:{exc}',
            None,
            0,
            0,
            0,
            0,
        )


def _emit_fixed_book_reposition_plan(
    node: Any,
    runtime: SimpleNamespace,
    observation: TransportObservation,
    scene: Any,
    bearing: PalmBearingAssessment,
) -> None:
    plan = _plan_live_fixed_book_reposition(
        node, runtime, observation, scene, bearing
    )
    fields: dict[str, Any] = {
        'passed': plan.safe,
        'reason': plan.reason,
        'read_only_candidate_plan': True,
        'book_fixed_in_captured_scene': True,
        'translation_only_hypothesis': True,
        'wrist_rotation_planned': False,
        'force_closure_proven': False,
        'pressure_cage_is_not_carry_authority': True,
        'shelf_support_dynamics_proven': False,
        'candidate_is_not_motion_authority': True,
        'shelf_overlap_m': -shelf_clearance_m(observation.corners),
        'arm_motion_commanded': False,
        'base_motion_commanded': False,
        'gripper_opened': False,
        'collision_preflight_code': plan.last_preflight_code,
        **plan.metrics,
    }
    if plan.candidate is not None:
        fields.update({
            'candidate_q': plan.candidate.positions.tolist(),
            'candidate_preflight_detail': plan.candidate.preflight.detail,
        })
    _emit('post_natural_fixed_book_palm_reposition_plan', **fields)


def _emit_shelf_edge_proximity(
    observation: TransportObservation,
    scene: Any,
) -> ShelfEdgeProximity:
    assessment = shelf_edge_proximity_geometry(
        observation.corners,
        scene.shelf_triangles,
    )
    _emit(
        'post_natural_shelf_edge_proximity',
        geometry_available=assessment.available,
        reason=assessment.reason,
        geometry_only=True,
        shelf_contact_sensor_available=False,
        shelf_contact_proven=False,
        free_gravity_carry_proven=False,
        **assessment.metrics,
    )
    return assessment


def _stabilized_natural_endpoint(
    node: Any,
    runtime: SimpleNamespace,
) -> tuple[
    TransportObservation,
    Any,
    PalmBearingAssessment,
    GuardResult,
]:
    """Observe 0.5 simulated seconds without issuing any actuation command."""

    left, right, palm = _fresh_cage_gate(node)
    reference, current_scene = _transport_observation(node, runtime)
    current = reference
    bearing = _bearing_for_observation(
        node, runtime.environment_preflight, current, current_scene
    )
    _emit_shelf_edge_proximity(current, current_scene)
    initial = natural_settled_cage_guard(
        reference,
        current,
        bearing,
        reference_time=_node_time_seconds(node),
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=_unexpected_contact(node),
        minimum_dwell_s=0.0,
    )
    _emit(
        'post_natural_settle_inspection_sample',
        passed=initial.safe,
        reason=initial.reason,
        sample='initial',
        support_mode=_support_mode_label(
            bearing,
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
        ),
        arm_motion_commanded=False,
        gravity_supported_flag=False,
        **initial.metrics,
    )
    if not initial.safe:
        if initial.reason in (
            'book_com_projection_outside_palm',
            'support_depth_margin_too_small',
            'support_lateral_margin_too_small',
            'support_surface_too_tilted',
        ):
            _emit_fixed_book_reposition_plan(
                node, runtime, current, current_scene, bearing
            )
        raise RuntimeError(
            f'natural-settle initial cage rejected: {initial.reason}'
        )
    while current.observed_at - reference.observed_at < STATIONARY_DWELL_S:
        current, current_scene = _transport_observation(
            node, runtime, newer_than=current.observed_at
        )
        left, right, palm = _exact_contacts(node, max_age=0.18)
        bearing = _bearing_for_observation(
            node, runtime.environment_preflight, current, current_scene
        )
        intermediate = natural_settled_cage_guard(
            reference,
            current,
            bearing,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
            minimum_dwell_s=0.0,
        )
        if not intermediate.safe:
            _emit(
                'post_natural_settle_inspection_sample',
                passed=False,
                reason=intermediate.reason,
                sample='stationarity_dwell',
                support_mode=_support_mode_label(
                    bearing,
                    left_contact=left,
                    right_contact=right,
                    palm_contact=palm,
                ),
                arm_motion_commanded=False,
                gravity_supported_flag=False,
                **intermediate.metrics,
            )
            raise RuntimeError(
                f'natural-settle stability sample rejected: '
                f'{intermediate.reason}'
            )

    left, right, palm = _exact_contacts(node, max_age=0.18)
    final = natural_settled_cage_guard(
        reference,
        current,
        bearing,
        reference_time=_node_time_seconds(node),
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=_unexpected_contact(node),
    )
    if not final.safe:
        _emit(
            'post_natural_settle_inspection_sample',
            passed=False,
            reason=final.reason,
            sample='stationary_endpoint',
            support_mode=_support_mode_label(
                bearing,
                left_contact=left,
                right_contact=right,
                palm_contact=palm,
            ),
            arm_motion_commanded=False,
            gravity_supported_flag=False,
            **final.metrics,
        )
        raise RuntimeError(
            f'natural-settle stationary cage rejected: {final.reason}'
        )
    return current, current_scene, bearing, final


def _anchor_continuity_guard(
    reference: TransportObservation,
    current: TransportObservation,
    anchor: AttachmentAnchor,
) -> GuardResult:
    try:
        position, rotation, corner = _attachment_errors(anchor, current)
        arm_change = float(np.max(np.abs(current.arm - reference.arm)))
        base_change = float(np.linalg.norm(
            current.base[:2] - reference.base[:2]
        ))
        base_yaw = _angle_error(current.base[2], reference.base[2])
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})
    metrics = {
        'attachment_position_error_m': position,
        'attachment_rotation_error_rad': rotation,
        'attachment_corner_error_m': corner,
        'dispatch_arm_change_rad': arm_change,
        'dispatch_base_change_m': base_change,
        'dispatch_base_yaw_change_rad': base_yaw,
    }
    checks = (
        (
            position <= ARM_CUMULATIVE_ATTACHMENT_POSITION_LIMIT_M,
            'attachment_position_changed',
        ),
        (
            rotation <= ARM_CUMULATIVE_ROTATION_LIMIT_RAD,
            'attachment_rotation_changed',
        ),
        (
            corner <= ARM_CUMULATIVE_CORNER_ERROR_LIMIT_M,
            'attachment_obb_changed',
        ),
        (arm_change <= DISPATCH_ARM_CHANGE_LIMIT_RAD, 'arm_changed_after_plan'),
        (
            base_change <= ARM_BASE_CUMULATIVE_LIMIT_M,
            'base_changed_after_plan',
        ),
        (base_yaw <= ARM_BASE_YAW_LIMIT_RAD, 'base_yaw_changed_after_plan'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _plan_arm_only_clearance(
    chain: Any,
    start_positions: Sequence[float],
    base: Sequence[float],
    anchor: AttachmentAnchor,
    current_corners: Sequence[Sequence[float]],
) -> tuple[TransportWaypoint, ...]:
    """Return no motion when clear, otherwise the bounded 5 mm arm route."""

    if shelf_clearance_m(current_corners) >= MINIMUM_SHELF_CLEARANCE_M:
        return ()
    return solve_outward_waypoints(
        chain,
        start_positions,
        base,
        anchor,
        maximum_arm_outward_m=NATURAL_MAXIMUM_ARM_OUTWARD_M,
    )


def _execute_arm_only_clearance(
    *,
    node: Any,
    runtime: SimpleNamespace,
    reference: TransportObservation,
    anchor: AttachmentAnchor,
    waypoints: Sequence[TransportWaypoint],
) -> TransportObservation:
    """Execute only the guarded left-arm legs, stopping at 20 mm clearance."""

    planned = tuple(waypoints)
    if not planned:
        raise ValueError('post-natural arm clearance route is empty')
    previous = reference
    base_world = _base_transform_from_pose(reference.base)
    for waypoint in planned:
        preflight = preflight_attached_arm_route(
            node=node,
            environment=runtime.environment_preflight,
            anchor=anchor,
            targets=(waypoint.positions,),
            require_final_clearance=(waypoint.index == planned[-1].index),
        )
        if not preflight.safe:
            raise RuntimeError(
                f'outward step {waypoint.index} preflight rejected: '
                f'{preflight.code}: {preflight.detail}'
            )

        left, right, palm = _fresh_cage_gate(node)
        stationary, stationary_scene = _transport_observation(
            node, runtime, newer_than=previous.observed_at
        )
        bearing = _bearing_for_observation(
            node,
            runtime.environment_preflight,
            stationary,
            stationary_scene,
        )
        continuity = _anchor_continuity_guard(previous, stationary, anchor)
        if not bearing.safe:
            raise RuntimeError(
                f'outward step {waypoint.index} bearing support rejected: '
                f'{bearing.reason}'
            )
        if not left or not right or not palm or _unexpected_contact(node):
            raise RuntimeError(
                f'outward step {waypoint.index} pre-command cage changed'
            )
        if abs(
            stationary.aperture_m - EXPECTED_CAGED_APERTURE_M
        ) > NATURAL_APERTURE_LIMIT_M:
            raise RuntimeError(
                f'outward step {waypoint.index} aperture changed'
            )
        if not continuity.safe:
            raise RuntimeError(
                f'outward step {waypoint.index} pre-command state changed: '
                f'{continuity.reason}'
            )
        previous = stationary

        legs = ((
            waypoint.positions,
            ARM_STAGE_DURATION_S,
            POST_NATURAL_EVENT,
        ),)
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        node.begin_transfer_watchdog()
        try:
            node._post_natural_arm_motion_commanded = True
            moved, contact_loss = node._send_retained_arm_trajectory(
                goal, duration, legs, POST_NATURAL_EVENT
            )
            endpoint = None
            if moved:
                endpoint = node._wait_for_retained_endpoint(
                    waypoint.positions,
                    command=POST_NATURAL_EVENT,
                    phase='outward_arm_clearance',
                    leg=waypoint.index,
                    arm_tolerance=ARM_ENDPOINT_JOINT_LIMIT_RAD,
                )
            if endpoint is not None:
                left, right, palm = _fresh_cage_gate(node)
                current, current_scene = _transport_observation(
                    node, runtime, newer_than=previous.observed_at
                )
                bearing = _bearing_for_observation(
                    node,
                    runtime.environment_preflight,
                    current,
                    current_scene,
                )
                # Poll once more after the coherent endpoint scene while the
                # natural probe's exact-three latch is still armed.
                node._payload_hazard_reason(max_age=0.18)
        finally:
            transfer_reason = node.end_transfer_watchdog()
        if not moved:
            reason = (
                transfer_reason
                or ('exact_three_point_contact_lost' if contact_loss else None)
                or 'controller_failure'
            )
            raise PostSlideTransportFailure(
                f'outward step {waypoint.index} failed: {reason}'
            )
        if transfer_reason is not None:
            raise PostSlideTransportFailure(
                f'outward step {waypoint.index} watchdog failed: '
                f'{transfer_reason}'
            )
        if endpoint is None:
            raise PostSlideTransportFailure(
                f'outward step {waypoint.index} endpoint was not held'
            )
        expected_hand_world = base_world @ waypoint.target_pose
        step = arm_transport_step_guard(
            reference,
            previous,
            current,
            anchor,
            bearing.support,
            expected_hand_world=expected_hand_world,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
            expected_arm=waypoint.positions,
        )
        passed = bool(step.safe and bearing.safe)
        reason = step.reason if not step.safe else bearing.reason
        endpoint_metrics = {**step.metrics, **bearing.metrics}
        _emit(
            'post_natural_outward_step',
            passed=passed,
            reason=reason,
            step=waypoint.index,
            requested_outward_distance_m=waypoint.outward_distance_m,
            predicted_shelf_clearance_m=(
                waypoint.predicted_shelf_clearance_m
            ),
            measured_endpoint_q=current.arm.tolist(),
            preflight_detail=preflight.detail,
            exact_three_point_in_action_watchdog=True,
            diagnostic_truth_and_contacts_only=True,
            **endpoint_metrics,
        )
        if not passed:
            raise PostSlideTransportFailure(
                f'outward step {waypoint.index} gate failed: {reason}'
            )
        previous = current
        if shelf_clearance_m(previous.corners) >= MINIMUM_SHELF_CLEARANCE_M:
            return previous

    if shelf_clearance_m(previous.corners) < MINIMUM_SHELF_CLEARANCE_M:
        raise PostSlideTransportFailure(
            'arm route ended before 20 mm shelf clearance'
        )
    return previous


def _run(runtime: SimpleNamespace, *, inspect_only: bool) -> None:
    if runtime.BOOK != TARGET_BOOK_MODEL:
        raise RuntimeError('runtime target is not the audited seed-101 book')
    ProbeNode, ProbeNavigation = _probe_node_types(runtime)
    node = ProbeNode()
    nav = ProbeNavigation()
    executor = runtime.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    node._post_natural_arm_motion_commanded = False
    spin_errors: list[BaseException] = []

    def spin() -> None:
        try:
            executor.spin()
        except BaseException as error:
            spin_errors.append(error)

    thread = threading.Thread(
        target=spin,
        name='rigid-palm-post-natural-settle',
        daemon=True,
    )
    thread.start()
    stage = 'created'
    try:
        deadline = time.monotonic() + 30.0
        while (len(node.joints) < 8 or nav.pose is None) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if len(node.joints) < 8 or nav.pose is None:
            raise RuntimeError('live robot/base state is unavailable')
        if nav.goal is not None:
            raise RuntimeError('navigation is active; inspection requires idle base')

        node._target_book_model = TARGET_BOOK_MODEL
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        # This must remain false until the fresh bearing proof below passes.
        node._gravity_supported_payload = False
        node._transport_lock_engaged = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        node.reset_probe_audit()

        stage = 'stationary_bearing_inspection'
        reference, _, bearing, resume = _stabilized_natural_endpoint(
            node, runtime
        )
        _emit(
            'post_natural_settle_inspection',
            passed=True,
            reason='ok',
            inspect_only=inspect_only,
            gravity_supported_flag=False,
            measured_start_q=reference.arm.tolist(),
            measured_book_position=reference.book.position.tolist(),
            measured_book_minimum=reference.book.minimum.tolist(),
            measured_book_maximum=reference.book.maximum.tolist(),
            measured_book_quaternion=reference.book.quaternion.tolist(),
            diagnostic_truth_and_contacts_only=True,
            **resume.metrics,
        )
        if inspect_only:
            stage = 'complete'
            _emit(
                'result',
                passed=True,
                stage='post_natural_settle_inspection',
                inspection_only=True,
                arm_motion_commanded=False,
                base_motion_commanded=False,
                gripper_opened=False,
                compaction_commanded=False,
                placement_commanded=False,
                gravity_supported_flag=False,
                next_motion_authorized=False,
                **bearing.metrics,
            )
            return

        anchor = capture_attachment_anchor(
            reference.book,
            reference.corners,
            reference.hand_world,
            observed_at=reference.observed_at,
        )
        waypoints = _plan_arm_only_clearance(
            node.chain,
            reference.arm,
            reference.base,
            anchor,
            reference.corners,
        )
        if not waypoints:
            stage = 'complete'
            _emit(
                'result',
                passed=True,
                stage=POST_NATURAL_EVENT,
                inspection_only=False,
                already_clear_no_op=True,
                shelf_clearance_m=shelf_clearance_m(reference.corners),
                arm_motion_commanded=False,
                left_arm_only=True,
                base_motion_commanded=False,
                gripper_opened=False,
                compaction_commanded=False,
                placement_commanded=False,
                gravity_supported_flag=False,
                stopped_after_arm_clearance=True,
                next_motion_authorized=False,
                diagnostic_truth_and_contacts_only=True,
            )
            return
        stage = 'complete_route_preflight'
        complete = preflight_attached_arm_route(
            node=node,
            environment=runtime.environment_preflight,
            anchor=anchor,
            targets=tuple(waypoint.positions for waypoint in waypoints),
            require_final_clearance=True,
        )
        if not complete.safe:
            raise RuntimeError(
                f'complete post-natural arm route rejected: '
                f'{complete.code}: {complete.detail}'
            )

        # Close the planning interval with fresh exact contacts, a new scene,
        # the stronger bearing proof, and tight anchor continuity.  Only this
        # point may switch the inherited retention semantics to supported mode.
        left, right, palm = _fresh_cage_gate(node)
        dispatch, dispatch_scene = _transport_observation(
            node, runtime, newer_than=reference.observed_at
        )
        dispatch_bearing = _bearing_for_observation(
            node,
            runtime.environment_preflight,
            dispatch,
            dispatch_scene,
        )
        dispatch_resume = natural_settled_cage_guard(
            reference,
            dispatch,
            dispatch_bearing,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
            minimum_dwell_s=0.0,
        )
        continuity = _anchor_continuity_guard(reference, dispatch, anchor)
        if not dispatch_resume.safe:
            raise RuntimeError(
                f'post-preflight natural cage rejected: '
                f'{dispatch_resume.reason}'
            )
        if not continuity.safe:
            raise RuntimeError(
                f'post-preflight attachment changed: {continuity.reason}'
            )
        node._gravity_supported_payload = True
        _emit(
            'post_natural_arm_clearance_resume_verified',
            gravity_supported_flag=True,
            outward_step_count=len(waypoints),
            complete_preflight_detail=complete.detail,
            hand_T_book=anchor.hand_from_book.tolist(),
            book_T_hand=anchor.book_from_hand.tolist(),
            exact_three_point_in_action_watchdog=True,
            diagnostic_truth_and_contacts_only=True,
            **dispatch_resume.metrics,
            **continuity.metrics,
        )

        stage = 'left_arm_only_clearance'
        final = _execute_arm_only_clearance(
            node=node,
            runtime=runtime,
            reference=dispatch,
            anchor=anchor,
            waypoints=waypoints,
        )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=POST_NATURAL_EVENT,
            measured_endpoint_q=final.arm.tolist(),
            measured_book_position=final.book.position.tolist(),
            measured_book_minimum=final.book.minimum.tolist(),
            measured_book_maximum=final.book.maximum.tolist(),
            shelf_clearance_m=shelf_clearance_m(final.corners),
            arm_motion_commanded=True,
            left_arm_only=True,
            base_motion_commanded=False,
            gripper_opened=False,
            compaction_commanded=False,
            placement_commanded=False,
            stopped_after_arm_clearance=True,
            gravity_supported_flag=True,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
        )
    except Exception as error:
        _emit(
            'result',
            passed=False,
            stage=stage,
            reason=f'{type(error).__name__}: {error}',
            inspection_only=inspect_only,
            arm_motion_commanded=bool(
                getattr(node, '_post_natural_arm_motion_commanded', False)
            ),
            base_motion_commanded=False,
            gripper_opened=False,
            compaction_commanded=False,
            placement_commanded=False,
            gravity_supported_flag=bool(
                getattr(node, '_gravity_supported_payload', False)
            ),
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
        )
        raise
    finally:
        node._cancel.set()
        executor.shutdown()
        thread.join(timeout=3.0)
        nav.destroy_node()
        node.destroy_node()
        if spin_errors:
            raise RuntimeError(f'executor spin failed: {spin_errors[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Inspect or arm-clear the natural-settle three-point cage'
    )
    parser.add_argument(
        '--confirm-arm-only-clearance',
        action='store_true',
        help=(
            'after inspection, authorize only the guarded left-arm route to '
            '20 mm shelf clearance'
        ),
    )
    arguments = parser.parse_args()
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime, inspect_only=not arguments.confirm_arm_only_clearance)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
