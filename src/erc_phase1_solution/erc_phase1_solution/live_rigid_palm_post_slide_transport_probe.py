#!/usr/bin/env python3
"""Diagnostic-only continuation after the tangent-slide three-point cage.

This probe is deliberately excluded from the competition mission.  It reads
Gazebo entity poses and temporary exact-contact sensors, so a successful run
is engineering evidence rather than a competition-valid perception result.

The start state is discovered from the live robot; no recorded arm vector is
used.  Before any command, the probe requires a fresh 30 mm left--palm--right
cage and proves that the observed book centre of mass projects inside a
finite support polygon extracted from the *actual palm collision mesh and
measured palm transform*.  The actual book OBB is then re-anchored in the
measured grasp frame.

The only arm motion is a dynamically solved, fixed-orientation sequence of
5 mm outward steps.  It ends only after the target's rear/max-x edge is at
least 20 mm in front of the shelf face.  Every complete route and every leg
is densely preflighted against the shelf, all books, the full robot, and
self-collision while carrying the re-anchored OBB.  Only after padded shelf
clearance is proved may a short, slow, straight carried-retreat base goal be
issued.  The arm, aperture, exact contacts, support margins, and attachment
transform remain under continuous fail-closed observation.  The probe stops
there: it never opens, compacts, navigates to the bin, or authorizes a next
motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from erc_phase1_solution.kinematics import (
    oriented_box_intersects_triangles,
    triangle_meshes_intersect,
)
from erc_phase1_solution.live_rigid_palm_aperture_evidence_probe import (
    APERTURE_ENDPOINT_TOLERANCE_M,
    _tool_robot_collision,
)
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    GuardResult,
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
    _angle_error,
    _book_transform,
    _emit,
    _fresh_cage_gate,
    _outward_world,
    _probe_types,
    _quaternion_from_rotation,
    _unexpected_pairs,
)
from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (
    _inflated_ordered_corners,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
    rotation_matrix_distance,
    world_hand_pose,
)
from erc_phase1_solution.live_rigid_palm_tangent_slide_probe import (
    ARM_STAGE_DURATION_S,
    HAND_ENDPOINT_POSITION_LIMIT_M,
    HAND_ENDPOINT_ROTATION_LIMIT_RAD,
    IK_MAXIMUM_JOINT_STEP_RAD,
    IK_ORIENTATION_TOLERANCE_RAD,
    IK_POSITION_TOLERANCE_M,
    RECAGE_APERTURE_M,
    RouteGeometrySample,
    _dense_arm_segment,
    _finite_vector,
    _proper_transform,
)
from erc_phase1_solution.rigid_palm_live_preflight import (
    BOOK_HALF_EXTENTS_M,
    WorldScene,
    _measured_finger_transforms_relative_to_palm,
    _measured_robot_state,
    _node_time_seconds,
    _require_stable_measured_finger_geometry,
    _require_stable_preflight_inputs,
    conservative_tool_vertex_displacement,
)
from erc_phase1_solution.rigid_palm_preflight import (
    BookOBB,
    GripperSample,
    PALM_COLLISION_LINK,
    PreflightResult,
    ProbePhase,
    preflight_gripper_sweep,
)


POST_SLIDE_TRANSPORT_EVENT = 'rigid_palm_post_slide_transport'
OUTWARD_STEP_M = 0.005
MAXIMUM_ARM_OUTWARD_M = 0.100
MINIMUM_SHELF_CLEARANCE_M = 0.020

# The tangent insertion is intended to move the formerly unsupported gravity
# line only a few millimetres onto the finite pad.  These are *net* margins
# after eroding the mesh-derived polygon by collision padding.  They are large
# relative to the 0.5 mm swept-motion sampling and 0.35 mm endpoint limits,
# while remaining compatible with the audited 61 mm insertion geometry.
MINIMUM_SUPPORT_DEPTH_MARGIN_M = 0.003
MINIMUM_SUPPORT_LATERAL_MARGIN_M = 0.003
SUPPORT_POLYGON_INSET_M = 0.00075
SUPPORT_SURFACE_LAYER_M = 0.00050
SUPPORT_MAXIMUM_TILT_RAD = math.radians(12.0)
SUPPORT_MINIMUM_AREA_M2 = 0.00010

ARM_STEP_MINIMUM_PROGRESS_M = 0.003
ARM_STEP_MAXIMUM_PROGRESS_M = 0.007
ARM_STEP_ATTACHMENT_POSITION_LIMIT_M = 0.0020
ARM_CUMULATIVE_ATTACHMENT_POSITION_LIMIT_M = 0.0030
ARM_STEP_CORNER_ERROR_LIMIT_M = 0.0025
ARM_CUMULATIVE_CORNER_ERROR_LIMIT_M = 0.0035
ARM_STEP_ROTATION_LIMIT_RAD = math.radians(1.0)
ARM_CUMULATIVE_ROTATION_LIMIT_RAD = math.radians(2.0)
ARM_BASE_STEP_LIMIT_M = 0.00035
ARM_BASE_CUMULATIVE_LIMIT_M = 0.00075
ARM_BASE_YAW_LIMIT_RAD = 0.0010
ARM_ENDPOINT_JOINT_LIMIT_RAD = 0.0040
SCENE_AGE_LIMIT_S = 0.25
SCENE_FUTURE_TOLERANCE_S = 0.02

# Once the arm has created 20 mm of padded shelf separation, the base moves
# only far enough to leave a 50 mm rear-edge buffer.  The distance is computed
# from the observed OBB and measured outward direction, rounded up to 5 mm;
# it is not a recorded pose or arm-state constant.
BASE_FINAL_SHELF_CLEARANCE_M = 0.050
BASE_RETREAT_QUANTUM_M = 0.005
BASE_RETREAT_MAXIMUM_M = 0.080
BASE_RETREAT_POSITION_TOLERANCE_M = 0.005
BASE_RETREAT_DISTANCE_TOLERANCE_M = 0.012
BASE_RETREAT_LATERAL_LIMIT_M = 0.0010
BASE_RETREAT_YAW_LIMIT_RAD = 0.0010
BASE_RETREAT_ARM_LIMIT_RAD = 0.0040
BASE_RETREAT_SAMPLE_PROGRESS_LIMIT_M = 0.0070
BASE_RETREAT_ATTACHMENT_POSITION_LIMIT_M = 0.0030
BASE_RETREAT_ATTACHMENT_ROTATION_LIMIT_RAD = math.radians(2.0)
BASE_RETREAT_CORNER_ERROR_LIMIT_M = 0.0035
BASE_RETREAT_POLL_S = 0.03

BOX_FACES = np.asarray(
    (
        (0, 1, 3), (0, 3, 2),
        (4, 6, 7), (4, 7, 5),
        (0, 4, 5), (0, 5, 1),
        (2, 3, 7), (2, 7, 6),
        (0, 2, 6), (0, 6, 4),
        (1, 5, 7), (1, 7, 3),
    ),
    dtype=int,
)


@dataclass(frozen=True)
class SupportAssessment:
    """Finite mesh-derived support result and conservative margins."""

    safe: bool
    reason: str
    depth_margin_m: float
    lateral_margin_m: float
    raw_depth_margin_m: float
    raw_lateral_margin_m: float
    projection_world: np.ndarray
    support_normal_world: np.ndarray
    polygon_world: np.ndarray
    support_area_m2: float


@dataclass(frozen=True)
class AttachmentAnchor:
    """Actual observed target OBB expressed in the measured grasp frame."""

    hand_from_book: np.ndarray
    book_from_hand: np.ndarray
    corners_in_hand: np.ndarray
    observed_at: float


@dataclass(frozen=True)
class TransportObservation:
    """One coherent live state used by the post-slide motion gates."""

    book: BookSnapshot
    corners: np.ndarray
    base: np.ndarray
    arm: np.ndarray
    hand_base: np.ndarray
    hand_world: np.ndarray
    aperture_m: float
    observed_at: float


@dataclass(frozen=True)
class TransportWaypoint:
    """One dynamic fixed-orientation 5 mm extraction endpoint."""

    index: int
    outward_distance_m: float
    positions: np.ndarray
    target_pose: np.ndarray
    predicted_corners_world: np.ndarray
    predicted_shelf_clearance_m: float


@dataclass(frozen=True)
class TransportGeometrySample:
    """A dense candidate including robot, tool, base, and attached target."""

    positions: np.ndarray
    aperture_m: float
    transforms: Mapping[str, np.ndarray]
    base_transform: np.ndarray
    target_corners: np.ndarray


class PostSlideTransportFailure(RuntimeError):
    """Fail-closed terminal error for this diagnostic continuation."""


def _invalid_support(reason: str) -> SupportAssessment:
    nan3 = np.full(3, math.nan, dtype=float)
    return SupportAssessment(
        False,
        reason,
        -math.inf,
        -math.inf,
        -math.inf,
        -math.inf,
        nan3,
        nan3,
        np.empty((0, 3), dtype=float),
        0.0,
    )


def _convex_hull(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a counter-clockwise 2-D convex hull without duplicates."""

    values = np.asarray(points, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError('support points must be a finite (N, 2) array')
    unique = np.unique(np.round(values, decimals=12), axis=0)
    if len(unique) < 3:
        raise ValueError('support surface has fewer than three unique points')
    ordered = sorted((float(x), float(y)) for x, y in unique)

    def cross(origin: tuple[float, float], first: tuple[float, float],
              second: tuple[float, float]) -> float:
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-14:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-14:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        raise ValueError('support surface is collinear')
    return hull


def _polygon_area(polygon: np.ndarray) -> float:
    shifted = np.roll(polygon, -1, axis=0)
    return 0.5 * abs(float(np.sum(
        polygon[:, 0] * shifted[:, 1]
        - polygon[:, 1] * shifted[:, 0]
    )))


def _directional_polygon_margin(
    polygon: np.ndarray, point: np.ndarray, direction: np.ndarray
) -> float:
    """Distance from an interior point to the hull in either direction."""

    candidates: list[float] = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            continue
        inward = np.asarray([-edge[1], edge[0]], dtype=float) / length
        slack = float(np.dot(inward, point - first))
        rate = float(np.dot(inward, direction))
        if rate < -1e-12:
            candidates.append(-slack / rate)
    if not candidates:
        raise ValueError('support ray does not leave the polygon')
    return float(min(candidates))


def finite_palm_support_guard(
    palm_triangles_local: Sequence[Sequence[Sequence[float]]],
    palm_world_transform: Sequence[Sequence[float]],
    book_corners_world: Sequence[Sequence[float]],
    outward_world: Sequence[float],
    *,
    polygon_inset_m: float = SUPPORT_POLYGON_INSET_M,
    minimum_depth_margin_m: float = MINIMUM_SUPPORT_DEPTH_MARGIN_M,
    minimum_lateral_margin_m: float = MINIMUM_SUPPORT_LATERAL_MARGIN_M,
) -> SupportAssessment:
    """Prove the gravity projection is safely inside the finite palm pad.

    The candidate support face is not named or hard-coded.  It is the extreme
    face of the measured palm collision mesh along whichever measured palm
    axis points most nearly upward.  The face vertices form a convex support
    polygon; the gravity-line projection is ray-tested against every hull
    edge along the measured shelf-depth and lateral directions.
    """

    try:
        triangles = np.asarray(palm_triangles_local, dtype=float)
        palm = _proper_transform(palm_world_transform, 'world palm')
        corners = np.asarray(book_corners_world, dtype=float)
        outward = np.asarray(outward_world, dtype=float)
        inset = float(polygon_inset_m)
        minimum_depth = float(minimum_depth_margin_m)
        minimum_lateral = float(minimum_lateral_margin_m)
        if (
            triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or len(triangles) == 0
            or corners.shape != (8, 3)
            or outward.shape != (3,)
            or not np.all(np.isfinite(triangles))
            or not np.all(np.isfinite(corners))
            or not np.all(np.isfinite(outward))
            or not all(math.isfinite(value) and value >= 0.0 for value in (
                inset, minimum_depth, minimum_lateral
            ))
        ):
            raise ValueError('support geometry is malformed')

        up = np.asarray([0.0, 0.0, 1.0], dtype=float)
        alignment = palm[:3, :3].T @ up
        axis = int(np.argmax(np.abs(alignment)))
        sign = 1.0 if alignment[axis] >= 0.0 else -1.0
        normal = sign * palm[:3, axis]
        normal /= np.linalg.norm(normal)
        tilt = math.acos(float(np.clip(np.dot(normal, up), -1.0, 1.0)))
        if tilt > SUPPORT_MAXIMUM_TILT_RAD:
            return _invalid_support('support_surface_too_tilted')

        signed_coordinate = sign * triangles[:, :, axis]
        top = float(np.max(signed_coordinate))
        selected = triangles[
            np.all(
                signed_coordinate >= top - SUPPORT_SURFACE_LAYER_M,
                axis=1,
            )
        ]
        if not len(selected):
            return _invalid_support('support_surface_not_found')
        world_vertices = (
            selected.reshape((-1, 3)) @ palm[:3, :3].T
            + palm[:3, 3]
        )

        depth = outward - normal * float(np.dot(outward, normal))
        depth_norm = float(np.linalg.norm(depth))
        if depth_norm <= 0.95:
            return _invalid_support('outward_direction_not_tangent')
        depth /= depth_norm
        lateral = np.cross(normal, depth)
        lateral /= np.linalg.norm(lateral)
        coordinates = np.column_stack(
            (world_vertices @ depth, world_vertices @ lateral)
        )
        polygon = _convex_hull(coordinates)
        area = _polygon_area(polygon)
        if area < SUPPORT_MINIMUM_AREA_M2:
            return _invalid_support('support_surface_too_small')

        centre = np.mean(corners, axis=0)
        plane_offset = float(np.dot(normal, palm[:3, 3]) + top)
        gravity = np.asarray([0.0, 0.0, -1.0], dtype=float)
        denominator = float(np.dot(normal, gravity))
        if denominator >= -0.95:
            return _invalid_support('gravity_not_transverse_to_support')
        ray_distance = (
            plane_offset - float(np.dot(normal, centre))
        ) / denominator
        if ray_distance < -1e-6:
            return _invalid_support('book_com_below_support_plane')
        projection = centre + max(0.0, ray_distance) * gravity
        point = np.asarray(
            [np.dot(projection, depth), np.dot(projection, lateral)],
            dtype=float,
        )

        slacks: list[float] = []
        for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = second - first
            length = float(np.linalg.norm(edge))
            if length <= 1e-12:
                continue
            inward = np.asarray([-edge[1], edge[0]], dtype=float) / length
            slacks.append(float(np.dot(inward, point - first)))
        if not slacks or min(slacks) < -1e-9:
            return _invalid_support('book_com_projection_outside_palm')

        raw_depth = min(
            _directional_polygon_margin(
                polygon, point, np.asarray([1.0, 0.0])
            ),
            _directional_polygon_margin(
                polygon, point, np.asarray([-1.0, 0.0])
            ),
        )
        raw_lateral = min(
            _directional_polygon_margin(
                polygon, point, np.asarray([0.0, 1.0])
            ),
            _directional_polygon_margin(
                polygon, point, np.asarray([0.0, -1.0])
            ),
        )
        depth_margin = raw_depth - inset
        lateral_margin = raw_lateral - inset
        polygon_world = (
            polygon[:, 0, None] * depth
            + polygon[:, 1, None] * lateral
        )
        # Restore the support-plane normal offset lost by the two coordinates.
        polygon_world += normal * plane_offset
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return _invalid_support('invalid_support_geometry')

    if depth_margin < minimum_depth:
        reason = 'support_depth_margin_too_small'
        safe = False
    elif lateral_margin < minimum_lateral:
        reason = 'support_lateral_margin_too_small'
        safe = False
    else:
        reason = 'ok'
        safe = True
    return SupportAssessment(
        safe,
        reason,
        depth_margin,
        lateral_margin,
        raw_depth,
        raw_lateral,
        projection,
        normal,
        polygon_world,
        area,
    )


def _palm_local_triangles(model: Any) -> np.ndarray:
    meshes = tuple(model.meshes_for_link(PALM_COLLISION_LINK))
    if not meshes:
        raise ValueError('palm collision model is missing')
    return np.concatenate(
        [np.asarray(mesh.triangles, dtype=float) for mesh in meshes], axis=0
    )


def capture_attachment_anchor(
    book: BookSnapshot,
    book_corners_world: Sequence[Sequence[float]],
    hand_world: Sequence[Sequence[float]],
    *,
    observed_at: float,
) -> AttachmentAnchor:
    """Re-anchor the actual observed book pose and OBB in the grasp frame."""

    hand = _proper_transform(hand_world, 'measured world hand')
    book_world = _proper_transform(
        _book_transform(book), 'measured world book'
    )
    corners = np.asarray(book_corners_world, dtype=float)
    stamp = float(observed_at)
    if (
        corners.shape != (8, 3)
        or not np.all(np.isfinite(corners))
        or not math.isfinite(stamp)
    ):
        raise ValueError('attachment anchor requires a finite observed OBB')
    hand_from_book = np.linalg.inv(hand) @ book_world
    book_from_hand = np.linalg.inv(hand_from_book)
    corners_in_hand = (corners - hand[:3, 3]) @ hand[:3, :3]
    return AttachmentAnchor(
        hand_from_book=hand_from_book,
        book_from_hand=book_from_hand,
        corners_in_hand=corners_in_hand,
        observed_at=stamp,
    )


def predict_attached_corners(
    anchor: AttachmentAnchor,
    hand_world: Sequence[Sequence[float]],
) -> np.ndarray:
    hand = _proper_transform(hand_world, 'candidate world hand')
    local = np.asarray(anchor.corners_in_hand, dtype=float)
    if local.shape != (8, 3) or not np.all(np.isfinite(local)):
        raise ValueError('attachment anchor corners are malformed')
    return local @ hand[:3, :3].T + hand[:3, 3]


def _attachment_errors(
    anchor: AttachmentAnchor,
    observation: TransportObservation,
) -> tuple[float, float, float]:
    current_relative = (
        np.linalg.inv(observation.hand_world)
        @ _book_transform(observation.book)
    )
    position_error = float(np.linalg.norm(
        current_relative[:3, 3] - anchor.hand_from_book[:3, 3]
    ))
    rotation_error = rotation_matrix_distance(
        current_relative[:3, :3], anchor.hand_from_book[:3, :3]
    )
    predicted = predict_attached_corners(anchor, observation.hand_world)
    corner_error = float(np.max(np.linalg.norm(
        observation.corners - predicted, axis=1
    )))
    return position_error, rotation_error, corner_error


def shelf_clearance_m(corners: Sequence[Sequence[float]]) -> float:
    points = np.asarray(corners, dtype=float)
    if points.shape != (8, 3) or not np.all(np.isfinite(points)):
        raise ValueError('shelf clearance requires a finite OBB')
    return float(SHELF_FRONT_X_M - np.max(points[:, 0]))


def _base_transform_from_pose(base: Sequence[float]) -> np.ndarray:
    pose = _finite_vector(base, 3, 'base pose')
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    result[:2, 3] = pose[:2]
    return result


def solve_outward_waypoints(
    chain: Any,
    start_positions: Sequence[float],
    base: Sequence[float],
    anchor: AttachmentAnchor,
    *,
    maximum_arm_outward_m: float = MAXIMUM_ARM_OUTWARD_M,
) -> tuple[TransportWaypoint, ...]:
    """Dynamically solve 5 mm steps until the actual OBB clears the shelf."""

    start = np.asarray(start_positions, dtype=float)
    base_pose = _finite_vector(base, 3, 'measured base')
    maximum_outward = float(maximum_arm_outward_m)
    if start.ndim != 1 or len(start) < 2 or not np.all(np.isfinite(start)):
        raise ValueError('measured arm state is malformed')
    if (
        not math.isfinite(maximum_outward)
        or maximum_outward < OUTWARD_STEP_M
        or maximum_outward > 0.150
    ):
        raise ValueError('maximum outward distance is outside the 5--150 mm cap')
    reference_pose = _proper_transform(
        chain.forward(start), 'measured start hand pose'
    )
    base_world = _base_transform_from_pose(base_pose)
    outward_world = _outward_world(base_pose)
    outward_base = base_world[:3, :3].T @ outward_world
    previous = start.copy()
    waypoints: list[TransportWaypoint] = []
    maximum_steps = int(math.floor(
        maximum_outward / OUTWARD_STEP_M + 1e-12
    ))
    for index in range(1, maximum_steps + 1):
        target_pose = reference_pose.copy()
        target_pose[:3, 3] = (
            reference_pose[:3, 3]
            + outward_base * (OUTWARD_STEP_M * index)
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
            raise RuntimeError(f'post-slide outward IK failed at step {index}')
        positions = np.asarray(solution, dtype=float)
        if (
            positions.shape != start.shape
            or not np.all(np.isfinite(positions))
        ):
            raise RuntimeError(
                'post-slide outward IK returned a malformed state'
            )
        achieved = _proper_transform(
            chain.forward(positions), 'outward achieved hand pose'
        )
        error = np.asarray(
            chain.pose_error(achieved, target_pose), dtype=float
        )
        maximum_joint_step = float(
            np.max(np.abs(positions[1:] - previous[1:]))
        )
        if (
            maximum_joint_step > IK_MAXIMUM_JOINT_STEP_RAD
            or float(np.linalg.norm(error[:3])) > IK_POSITION_TOLERANCE_M
            or float(np.linalg.norm(error[3:])) > IK_ORIENTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                f'post-slide outward IK is discontinuous at step {index}'
            )
        hand_world = base_world @ achieved
        predicted = predict_attached_corners(anchor, hand_world)
        clearance = shelf_clearance_m(predicted)
        waypoints.append(TransportWaypoint(
            index=index,
            outward_distance_m=OUTWARD_STEP_M * index,
            positions=positions,
            target_pose=target_pose,
            predicted_corners_world=predicted,
            predicted_shelf_clearance_m=clearance,
        ))
        previous = positions
        if clearance >= MINIMUM_SHELF_CLEARANCE_M:
            return tuple(waypoints)
    raise RuntimeError(
        'dynamic outward route did not create 20 mm shelf clearance within '
        f'{maximum_outward:.3f} m'
    )


def post_slide_cage_guard(
    observation: TransportObservation,
    support: SupportAssessment,
    *,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Recognize a fresh dynamic 30 mm supported three-point cage."""

    try:
        now = float(reference_time)
        age = now - float(observation.observed_at)
        aperture_error = abs(observation.aperture_m - RECAGE_APERTURE_M)
        overlap = -shelf_clearance_m(observation.corners)
        if not math.isfinite(now):
            raise ValueError('reference time is non-finite')
    except (AttributeError, TypeError, ValueError):
        return GuardResult(False, 'invalid_observation', {})
    metrics = {
        'scene_age_s': age,
        'aperture_error_m': aperture_error,
        'shelf_overlap_m': overlap,
        'support_depth_margin_m': support.depth_margin_m,
        'support_lateral_margin_m': support.lateral_margin_m,
        'support_area_m2': support.support_area_m2,
    }
    checks = (
        (age <= SCENE_AGE_LIMIT_S, 'scene_stale'),
        (age >= -SCENE_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (
            aperture_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'wrong_cage_aperture',
        ),
        (bool(left_contact), 'left_target_contact_missing'),
        (bool(right_contact), 'right_target_contact_missing'),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
        (support.safe, support.reason),
        (
            overlap <= MINIMUM_SHELF_CLEARANCE_M,
            'unexpected_deep_shelf_overlap',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def arm_transport_step_guard(
    reference: TransportObservation,
    previous: TransportObservation,
    current: TransportObservation,
    anchor: AttachmentAnchor,
    support: SupportAssessment,
    *,
    expected_hand_world: Sequence[Sequence[float]],
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    expected_arm: Sequence[float],
) -> GuardResult:
    """Gate one completed 5 mm fixed-orientation carried arm step."""

    try:
        expected_hand = _proper_transform(
            expected_hand_world, 'expected arm-step hand'
        )
        expected_q = _finite_vector(
            expected_arm, len(current.arm), 'expected arm-step joints'
        )
        outward = _outward_world(reference.base)
        hand_delta = current.hand_world[:3, 3] - previous.hand_world[:3, 3]
        book_delta = current.book.position - previous.book.position
        progress = float(np.dot(book_delta, outward))
        hand_progress = float(np.dot(hand_delta, outward))
        step_mismatch = float(np.linalg.norm(book_delta - hand_delta))
        step_rotation = quaternion_distance(
            previous.book.quaternion, current.book.quaternion
        )
        cumulative_rotation = quaternion_distance(
            reference.book.quaternion, current.book.quaternion
        )
        attachment_position, attachment_rotation, corner_error = (
            _attachment_errors(anchor, current)
        )
        previous_position, _, _ = _attachment_errors(anchor, previous)
        predicted = predict_attached_corners(anchor, current.hand_world)
        step_corner_error = float(np.max(np.linalg.norm(
            (current.corners - previous.corners)
            - (
                predicted
                - predict_attached_corners(anchor, previous.hand_world)
            ),
            axis=1,
        )))
        base_step = float(np.linalg.norm(
            current.base[:2] - previous.base[:2]
        ))
        base_cumulative = float(np.linalg.norm(
            current.base[:2] - reference.base[:2]
        ))
        base_yaw = _angle_error(current.base[2], reference.base[2])
        hand_position_error = float(np.linalg.norm(
            current.hand_world[:3, 3] - expected_hand[:3, 3]
        ))
        hand_rotation_error = rotation_matrix_distance(
            current.hand_world[:3, :3], expected_hand[:3, :3]
        )
        arm_endpoint_error = float(np.max(np.abs(current.arm - expected_q)))
        aperture_error = abs(current.aperture_m - RECAGE_APERTURE_M)
        age = float(reference_time) - current.observed_at
        clearance = shelf_clearance_m(current.corners)
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return GuardResult(False, 'invalid_observation', {})
    metrics = {
        'book_outward_progress_m': progress,
        'hand_outward_progress_m': hand_progress,
        'book_hand_step_mismatch_m': step_mismatch,
        'book_step_rotation_rad': step_rotation,
        'book_cumulative_rotation_rad': cumulative_rotation,
        'attachment_previous_position_error_m': previous_position,
        'attachment_position_error_m': attachment_position,
        'attachment_rotation_error_rad': attachment_rotation,
        'attachment_step_corner_error_m': step_corner_error,
        'attachment_cumulative_corner_error_m': corner_error,
        'base_step_drift_m': base_step,
        'base_cumulative_drift_m': base_cumulative,
        'base_yaw_drift_rad': base_yaw,
        'hand_endpoint_position_error_m': hand_position_error,
        'hand_endpoint_rotation_error_rad': hand_rotation_error,
        'arm_endpoint_error_rad': arm_endpoint_error,
        'aperture_error_m': aperture_error,
        'scene_age_s': age,
        'shelf_clearance_m': clearance,
        'support_depth_margin_m': support.depth_margin_m,
        'support_lateral_margin_m': support.lateral_margin_m,
    }
    checks = (
        (age <= SCENE_AGE_LIMIT_S, 'scene_stale'),
        (age >= -SCENE_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (bool(left_contact), 'left_target_contact_missing'),
        (bool(right_contact), 'right_target_contact_missing'),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
        (
            aperture_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'cage_aperture_changed',
        ),
        (
            ARM_STEP_MINIMUM_PROGRESS_M <= progress
            <= ARM_STEP_MAXIMUM_PROGRESS_M,
            'book_progress_out_of_bounds',
        ),
        (
            ARM_STEP_MINIMUM_PROGRESS_M <= hand_progress
            <= ARM_STEP_MAXIMUM_PROGRESS_M,
            'hand_progress_out_of_bounds',
        ),
        (
            step_mismatch <= ARM_STEP_ATTACHMENT_POSITION_LIMIT_M,
            'book_did_not_follow_hand',
        ),
        (
            attachment_position
            <= ARM_CUMULATIVE_ATTACHMENT_POSITION_LIMIT_M,
            'cumulative_attachment_position_slip',
        ),
        (
            attachment_rotation <= ARM_CUMULATIVE_ROTATION_LIMIT_RAD,
            'cumulative_attachment_rotation_slip',
        ),
        (
            step_corner_error <= ARM_STEP_CORNER_ERROR_LIMIT_M,
            'step_attachment_corner_slip',
        ),
        (
            corner_error <= ARM_CUMULATIVE_CORNER_ERROR_LIMIT_M,
            'cumulative_attachment_corner_slip',
        ),
        (step_rotation <= ARM_STEP_ROTATION_LIMIT_RAD, 'book_step_rotated'),
        (
            cumulative_rotation <= ARM_CUMULATIVE_ROTATION_LIMIT_RAD,
            'book_cumulative_rotation',
        ),
        (base_step <= ARM_BASE_STEP_LIMIT_M, 'base_moved_during_arm_step'),
        (
            base_cumulative <= ARM_BASE_CUMULATIVE_LIMIT_M,
            'base_moved_during_arm_route',
        ),
        (base_yaw <= ARM_BASE_YAW_LIMIT_RAD, 'base_rotated_during_arm_route'),
        (
            hand_position_error <= HAND_ENDPOINT_POSITION_LIMIT_M,
            'hand_endpoint_position_missed',
        ),
        (
            hand_rotation_error <= HAND_ENDPOINT_ROTATION_LIMIT_RAD,
            'hand_endpoint_rotation_missed',
        ),
        (
            arm_endpoint_error <= ARM_ENDPOINT_JOINT_LIMIT_RAD,
            'arm_endpoint_missed',
        ),
        (support.safe, support.reason),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def choose_base_retreat_distance(
    book_corners_world: Sequence[Sequence[float]],
    outward_world: Sequence[float],
    *,
    target_clearance_m: float = BASE_FINAL_SHELF_CLEARANCE_M,
    quantum_m: float = BASE_RETREAT_QUANTUM_M,
    maximum_m: float = BASE_RETREAT_MAXIMUM_M,
) -> float:
    """Return the minimum bounded retreat reaching the rear-edge buffer."""

    outward = np.asarray(outward_world, dtype=float)
    target = float(target_clearance_m)
    quantum = float(quantum_m)
    maximum = float(maximum_m)
    current = shelf_clearance_m(book_corners_world)
    if (
        outward.shape != (3,)
        or not np.all(np.isfinite(outward))
        or not all(
            math.isfinite(value) for value in (target, quantum, maximum)
        )
        or target < MINIMUM_SHELF_CLEARANCE_M
        or quantum <= 0.0
        or maximum < quantum
        or float(outward[0]) >= -0.95
    ):
        raise ValueError('base-retreat geometry is invalid')
    required = max(quantum, (target - current) / (-float(outward[0])))
    distance = quantum * int(math.ceil(required / quantum - 1e-12))
    if distance > maximum + 1e-12:
        raise ValueError('required base retreat exceeds the diagnostic cap')
    return float(distance)


def base_retreat_sample_guard(
    reference: TransportObservation,
    previous: TransportObservation,
    current: TransportObservation,
    anchor: AttachmentAnchor,
    support: SupportAssessment,
    *,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Continuously gate the slow straight base retreat with the arm held."""

    try:
        outward = _outward_world(reference.base)
        base_delta = current.base[:2] - reference.base[:2]
        step_delta = current.base[:2] - previous.base[:2]
        travelled = float(np.dot(base_delta, outward[:2]))
        step_progress = float(np.dot(step_delta, outward[:2]))
        lateral = float(np.linalg.norm(
            base_delta - travelled * outward[:2]
        ))
        yaw_error = _angle_error(current.base[2], reference.base[2])
        arm_error = float(np.max(np.abs(current.arm - reference.arm)))
        aperture_error = abs(current.aperture_m - RECAGE_APERTURE_M)
        attachment_position, attachment_rotation, corner_error = (
            _attachment_errors(anchor, current)
        )
        age = float(reference_time) - current.observed_at
        clearance = shelf_clearance_m(current.corners)
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return GuardResult(False, 'invalid_observation', {})
    metrics = {
        'base_outward_travel_m': travelled,
        'base_step_outward_progress_m': step_progress,
        'base_lateral_error_m': lateral,
        'base_yaw_error_rad': yaw_error,
        'arm_hold_error_rad': arm_error,
        'aperture_error_m': aperture_error,
        'attachment_position_error_m': attachment_position,
        'attachment_rotation_error_rad': attachment_rotation,
        'attachment_corner_error_m': corner_error,
        'scene_age_s': age,
        'shelf_clearance_m': clearance,
        'support_depth_margin_m': support.depth_margin_m,
        'support_lateral_margin_m': support.lateral_margin_m,
    }
    checks = (
        (age <= SCENE_AGE_LIMIT_S, 'scene_stale'),
        (age >= -SCENE_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (bool(left_contact), 'left_target_contact_missing'),
        (bool(right_contact), 'right_target_contact_missing'),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
        (step_progress >= -0.00025, 'base_reversed_during_retreat'),
        (
            step_progress <= BASE_RETREAT_SAMPLE_PROGRESS_LIMIT_M,
            'base_sample_progress_too_large',
        ),
        (
            lateral <= BASE_RETREAT_LATERAL_LIMIT_M,
            'base_retreat_not_straight',
        ),
        (
            yaw_error <= BASE_RETREAT_YAW_LIMIT_RAD,
            'base_rotated_during_retreat',
        ),
        (arm_error <= BASE_RETREAT_ARM_LIMIT_RAD, 'arm_moved_during_retreat'),
        (
            aperture_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'cage_aperture_changed',
        ),
        (
            attachment_position <= BASE_RETREAT_ATTACHMENT_POSITION_LIMIT_M,
            'attachment_position_slip',
        ),
        (
            attachment_rotation <= BASE_RETREAT_ATTACHMENT_ROTATION_LIMIT_RAD,
            'attachment_rotation_slip',
        ),
        (
            corner_error <= BASE_RETREAT_CORNER_ERROR_LIMIT_M,
            'attachment_corner_slip',
        ),
        (
            clearance >= MINIMUM_SHELF_CLEARANCE_M,
            'shelf_clearance_lost',
        ),
        (support.safe, support.reason),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _box_triangles(corners: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(corners, dtype=float)
    if points.shape != (8, 3) or not np.all(np.isfinite(points)):
        raise ValueError('box triangles require a finite ordered OBB')
    return points[BOX_FACES]


def _target_other_book_collision(
    target_corners: np.ndarray,
    scene: WorldScene,
    padding_m: float,
) -> str | None:
    target = _inflated_ordered_corners(target_corners, padding_m)
    target_surface = _box_triangles(target)
    for name, book in scene.books.items():
        if name == TARGET_BOOK_MODEL:
            continue
        other = _inflated_ordered_corners(book.corners, padding_m)
        if oriented_box_intersects_triangles(
            other, target_surface, closed_surface=True
        ):
            return name
    return None


def _target_shelf_collision(
    target_corners: np.ndarray,
    scene: WorldScene,
    padding_m: float,
) -> bool:
    inflated = _inflated_ordered_corners(target_corners, padding_m)
    return oriented_box_intersects_triangles(
        inflated, scene.shelf_triangles, closed_surface=False
    )


def _scene_with_geometry(
    scene: WorldScene,
    base_transform: np.ndarray,
    target_corners: np.ndarray,
) -> WorldScene:
    books = dict(scene.books)
    books[TARGET_BOOK_MODEL] = BookOBB(
        name=TARGET_BOOK_MODEL,
        corners=np.asarray(target_corners, dtype=float),
        observed_at=scene.observed_at,
    )
    return replace(
        scene,
        base_transform=np.asarray(base_transform, dtype=float),
        books=books,
    )


def _robot_environment_collision(
    node: Any,
    scene: WorldScene,
    positions: np.ndarray,
    *,
    padding_m: float,
) -> tuple[str, str] | None:
    """Check every official robot collision link in a candidate world pose."""

    surfaces_base = node._world_collision_surfaces(positions)
    base = np.asarray(scene.base_transform, dtype=float)
    shelf = np.asarray(scene.shelf_triangles, dtype=float)
    watertight = node._watertight_collision_links()
    padded_books = {
        name: _inflated_ordered_corners(book.corners, padding_m)
        for name, book in scene.books.items()
    }
    for link, surface_base in surfaces_base.items():
        surface = (
            np.asarray(surface_base, dtype=float) @ base[:3, :3].T
            + base[:3, 3]
        )
        if triangle_meshes_intersect(
            surface,
            shelf,
            tolerance=padding_m,
            first_watertight=link in watertight,
        ):
            return link, 'shelf'
        for name, corners in padded_books.items():
            if oriented_box_intersects_triangles(
                corners,
                surface,
                closed_surface=link in watertight,
            ):
                return link, name
    return None


def _evaluate_transport_geometry(
    *,
    node: Any,
    environment: Any,
    scene: WorldScene,
    samples: Sequence[TransportGeometrySample],
    reference_time: float,
    allow_initial_shelf_support: bool,
    require_final_clearance: bool,
) -> PreflightResult:
    """Full dense shelf/all-book/robot/self/payload/tool preflight."""

    dense = tuple(samples)
    if not dense:
        return PreflightResult(False, 'missing_samples', 'route is empty')
    padding = float(environment.config.collision_padding_m)
    previous: TransportGeometrySample | None = None
    previous_clearance = -math.inf
    for index, sample in enumerate(dense):
        if previous is not None:
            tool_step = conservative_tool_vertex_displacement(
                environment.model, previous.transforms, sample.transforms
            )
            target_step = float(np.max(np.linalg.norm(
                sample.target_corners - previous.target_corners, axis=1
            )))
            maximum = float(environment.config.max_vertex_step_m) + 1e-12
            if tool_step > maximum or target_step > maximum:
                return PreflightResult(
                    False,
                    'undersampled_transport_route',
                    f'dense sample {index - 1}->{index} moves tool '
                    f'{tool_step:.9f} m and target {target_step:.9f} m',
                    sample_index=index,
                )
        candidate_scene = _scene_with_geometry(
            scene, sample.base_transform, sample.target_corners
        )
        self_collision = node._robot_self_collision(sample.positions)
        if self_collision is not None:
            return PreflightResult(
                False, 'robot_self_collision',
                f'dense sample {index}: {self_collision}', sample_index=index
            )
        inverse_base = np.linalg.inv(sample.base_transform)
        padded_target = _inflated_ordered_corners(
            sample.target_corners, float(node.carried_book_padding)
        )
        target_base = (
            padded_target @ inverse_base[:3, :3].T + inverse_base[:3, 3]
        )
        payload_robot = node._carried_robot_collision(
            sample.positions, target_base
        )
        if payload_robot is not None:
            return PreflightResult(
                False, 'target_robot_collision',
                f'dense sample {index}: {payload_robot}',
                sample_index=index, obstacle=payload_robot
            )
        robot_environment = _robot_environment_collision(
            node, candidate_scene, sample.positions, padding_m=padding
        )
        if robot_environment is not None:
            return PreflightResult(
                False, 'robot_environment_collision',
                f'dense sample {index}: {robot_environment}',
                sample_index=index,
                link=robot_environment[0], obstacle=robot_environment[1]
            )
        tool_robot = _tool_robot_collision(
            node,
            candidate_scene,
            sample.positions,
            sample.transforms,
            environment.model,
        )
        if tool_robot is not None:
            return PreflightResult(
                False, 'tool_robot_collision',
                f'dense sample {index}: {tool_robot}',
                sample_index=index, link=tool_robot[0], obstacle=tool_robot[1]
            )
        other_book = _target_other_book_collision(
            sample.target_corners, candidate_scene, padding
        )
        if other_book is not None:
            return PreflightResult(
                False, 'target_other_book_collision',
                f'dense sample {index}: {other_book}',
                sample_index=index, obstacle=other_book
            )
        clearance = shelf_clearance_m(sample.target_corners)
        shelf_collision = _target_shelf_collision(
            sample.target_corners, candidate_scene, padding
        )
        if shelf_collision:
            legacy_support = bool(
                allow_initial_shelf_support
                and clearance < padding
                and clearance + 1e-9 >= previous_clearance
            )
            if not legacy_support:
                return PreflightResult(
                    False, 'target_shelf_collision',
                    f'dense sample {index}: padded target still intersects '
                    'shelf '
                    f'at clearance {clearance:.9f} m',
                    sample_index=index, obstacle='shelf'
                )
        previous_clearance = max(previous_clearance, clearance)
        gripper = preflight_gripper_sweep(
            model=environment.model,
            samples=(GripperSample(
                transforms=sample.transforms,
                measured_aperture_m=sample.aperture_m,
                expected_aperture_m=RECAGE_APERTURE_M,
                observed_at=scene.observed_at,
                phase=ProbePhase.CAGE,
            ),),
            shelf_triangles=candidate_scene.shelf_triangles,
            books=candidate_scene.books,
            target_book=TARGET_BOOK_MODEL,
            expected_book_names=candidate_scene.expected_book_names,
            reference_time=reference_time,
            config=environment.config,
        )
        if not gripper.safe:
            return PreflightResult(
                False, f'gripper_{gripper.code}',
                f'dense sample {index}: {gripper.detail}',
                sample_index=index,
                link=gripper.link,
                obstacle=gripper.obstacle,
            )
        previous = sample
    final_clearance = shelf_clearance_m(dense[-1].target_corners)
    if require_final_clearance and final_clearance < MINIMUM_SHELF_CLEARANCE_M:
        return PreflightResult(
            False, 'route_ends_before_shelf_clearance',
            f'final clearance is only {final_clearance:.9f} m'
        )
    return PreflightResult(
        True, 'clear', f'{len(dense)} dynamic attached-target states passed'
    )


def _arm_geometry_samples(
    *,
    node: Any,
    environment: Any,
    scene: WorldScene,
    start_positions: np.ndarray,
    targets: Sequence[np.ndarray],
    measured_relative: Mapping[str, np.ndarray],
    anchor: AttachmentAnchor,
) -> tuple[TransportGeometrySample, ...]:
    dense: tuple[RouteGeometrySample, ...] = _dense_arm_segment(
        environment=environment,
        node=node,
        scene=scene,
        joint_samples=(start_positions, *targets),
        aperture_m=RECAGE_APERTURE_M,
        measured_relative=measured_relative,
    )
    result: list[TransportGeometrySample] = []
    for sample in dense:
        hand_world = (
            np.asarray(scene.base_transform, dtype=float)
            @ node.chain.forward(sample.positions)
        )
        result.append(TransportGeometrySample(
            positions=sample.positions,
            aperture_m=sample.aperture_m,
            transforms=sample.transforms,
            base_transform=np.asarray(scene.base_transform, dtype=float),
            target_corners=predict_attached_corners(anchor, hand_world),
        ))
    return tuple(result)


def preflight_attached_arm_route(
    *,
    node: Any,
    environment: Any,
    anchor: AttachmentAnchor,
    targets: Sequence[Sequence[float]],
    require_final_clearance: bool = False,
) -> PreflightResult:
    """Preflight the dynamic attached OBB using measured passive links."""

    try:
        before = _measured_robot_state(node)
        if abs(before.aperture_m - RECAGE_APERTURE_M) > (
            APERTURE_ENDPOINT_TOLERANCE_M
        ):
            raise ValueError('arm preflight did not start at the 30 mm cage')
        goals = tuple(np.asarray(target, dtype=float) for target in targets)
        if not goals or any(
            goal.shape != before.positions.shape
            or not np.all(np.isfinite(goal))
            for goal in goals
        ):
            raise ValueError('arm preflight targets are empty or malformed')
        scene = environment._read_scene()
        reference_time = _node_time_seconds(node)
        measured = _measured_finger_transforms_relative_to_palm(
            node, scene, before
        )
        current_hand = (
            scene.base_transform @ node.chain.forward(before.positions)
        )
        predicted = predict_attached_corners(anchor, current_hand)
        observed = np.asarray(
            scene.books[TARGET_BOOK_MODEL].corners, dtype=float
        )
        if float(np.max(np.linalg.norm(predicted - observed, axis=1))) > (
            ARM_CUMULATIVE_CORNER_ERROR_LIMIT_M
        ):
            raise ValueError(
                'live target no longer matches its attachment anchor'
            )
        samples = _arm_geometry_samples(
            node=node,
            environment=environment,
            scene=scene,
            start_positions=before.positions,
            targets=goals,
            measured_relative=measured,
            anchor=anchor,
        )
        result = _evaluate_transport_geometry(
            node=node,
            environment=environment,
            scene=scene,
            samples=samples,
            reference_time=reference_time,
            allow_initial_shelf_support=True,
            require_final_clearance=require_final_clearance,
        )
        if not result.safe:
            return result
        after_scene = environment._read_scene()
        after = _measured_robot_state(node)
        after_measured = _measured_finger_transforms_relative_to_palm(
            node, after_scene, after
        )
        _require_stable_measured_finger_geometry(
            environment.model, measured, after_measured
        )
        _require_stable_preflight_inputs(scene, after_scene, before, after)
        return result
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return PreflightResult(False, 'live_adapter_error', str(error))


def _base_geometry_samples(
    *,
    node: Any,
    environment: Any,
    scene: WorldScene,
    positions: np.ndarray,
    aperture_m: float,
    measured_relative: Mapping[str, np.ndarray],
    anchor: AttachmentAnchor,
    retreat_distance_m: float,
) -> tuple[TransportGeometrySample, ...]:
    distance = float(retreat_distance_m)
    step = float(environment.config.max_vertex_step_m)
    count = max(1, int(math.ceil(distance / step)))
    start_base = np.asarray(scene.base_transform, dtype=float)
    base_pose = np.asarray([
        start_base[0, 3], start_base[1, 3],
        math.atan2(start_base[1, 0], start_base[0, 0]),
    ])
    outward = _outward_world(base_pose)
    samples: list[TransportGeometrySample] = []
    for travelled in np.linspace(0.0, distance, count + 1):
        base = start_base.copy()
        base[:3, 3] = start_base[:3, 3] + outward * float(travelled)
        candidate_scene = replace(scene, base_transform=base)
        transforms = environment._world_transforms(
            node,
            candidate_scene,
            positions,
            aperture_m,
            measured_relative,
        )
        hand_world = base @ node.chain.forward(positions)
        samples.append(TransportGeometrySample(
            positions=positions.copy(),
            aperture_m=aperture_m,
            transforms=transforms,
            base_transform=base,
            target_corners=predict_attached_corners(anchor, hand_world),
        ))
    return tuple(samples)


def preflight_attached_base_retreat(
    *,
    node: Any,
    environment: Any,
    anchor: AttachmentAnchor,
    retreat_distance_m: float,
) -> PreflightResult:
    """Preflight every robot link while translating the fixed carried cage."""

    try:
        distance = float(retreat_distance_m)
        if not 0.0 < distance <= BASE_RETREAT_MAXIMUM_M:
            raise ValueError('base retreat is outside its diagnostic cap')
        before = _measured_robot_state(node)
        if abs(before.aperture_m - RECAGE_APERTURE_M) > (
            APERTURE_ENDPOINT_TOLERANCE_M
        ):
            raise ValueError('base preflight did not start at the 30 mm cage')
        scene = environment._read_scene()
        reference_time = _node_time_seconds(node)
        measured = _measured_finger_transforms_relative_to_palm(
            node, scene, before
        )
        current_hand = (
            scene.base_transform @ node.chain.forward(before.positions)
        )
        observed = np.asarray(
            scene.books[TARGET_BOOK_MODEL].corners, dtype=float
        )
        predicted = predict_attached_corners(anchor, current_hand)
        if (
            shelf_clearance_m(observed) < MINIMUM_SHELF_CLEARANCE_M
            or float(np.max(np.linalg.norm(predicted - observed, axis=1)))
            > ARM_CUMULATIVE_CORNER_ERROR_LIMIT_M
        ):
            raise ValueError(
                'base preflight requires a fully clear attached target'
            )
        samples = _base_geometry_samples(
            node=node,
            environment=environment,
            scene=scene,
            positions=before.positions,
            aperture_m=before.aperture_m,
            measured_relative=measured,
            anchor=anchor,
            retreat_distance_m=distance,
        )
        result = _evaluate_transport_geometry(
            node=node,
            environment=environment,
            scene=scene,
            samples=samples,
            reference_time=reference_time,
            allow_initial_shelf_support=False,
            require_final_clearance=True,
        )
        if not result.safe:
            return result
        after_scene = environment._read_scene()
        after = _measured_robot_state(node)
        after_measured = _measured_finger_transforms_relative_to_palm(
            node, after_scene, after
        )
        _require_stable_measured_finger_geometry(
            environment.model, measured, after_measured
        )
        _require_stable_preflight_inputs(scene, after_scene, before, after)
        return result
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return PreflightResult(False, 'live_adapter_error', str(error))


def _transport_observation(
    node: Any,
    runtime: SimpleNamespace,
    *,
    newer_than: float | None = None,
) -> tuple[TransportObservation, WorldScene]:
    deadline = time.monotonic() + 2.0
    while True:
        scene = runtime.environment_preflight._read_scene()
        stamp = float(scene.observed_at)
        if newer_than is None or stamp > float(newer_than) + 1e-12:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Gazebo scene timestamp did not advance')
        time.sleep(0.02)
    base_transform = np.asarray(scene.base_transform, dtype=float)
    base = np.asarray([
        base_transform[0, 3],
        base_transform[1, 3],
        math.atan2(base_transform[1, 0], base_transform[0, 0]),
    ], dtype=float)
    corners = np.asarray(scene.books[TARGET_BOOK_MODEL].corners, dtype=float)
    centre = np.mean(corners, axis=0)
    rotation = np.column_stack((
        (corners[4] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[0]),
        (corners[2] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[1]),
        (corners[1] - corners[0]) / (2.0 * BOOK_HALF_EXTENTS_M[2]),
    ))
    book = BookSnapshot(
        centre,
        _quaternion_from_rotation(rotation),
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )
    arm = np.asarray(node._measured_left_solution(), dtype=float)
    aperture = float(node.joints.get('gripper_left_finger_joint', math.nan))
    if not math.isfinite(aperture):
        raise RuntimeError('measured left aperture is unavailable')
    hand_base = _proper_transform(
        node.chain.forward(arm), 'measured base-frame hand'
    )
    hand_world = world_hand_pose(base, hand_base)
    observation = TransportObservation(
        book=book,
        corners=np.asarray(corners, dtype=float),
        base=np.asarray(base, dtype=float),
        arm=arm,
        hand_base=hand_base,
        hand_world=hand_world,
        aperture_m=aperture,
        observed_at=stamp,
    )
    return observation, scene


def _support_for_observation(
    node: Any,
    environment: Any,
    observation: TransportObservation,
    scene: WorldScene,
) -> SupportAssessment:
    links = node.chain.link_transforms(observation.arm)
    palm_world = np.asarray(scene.base_transform) @ links[PALM_COLLISION_LINK]
    return finite_palm_support_guard(
        _palm_local_triangles(environment.model),
        palm_world,
        observation.corners,
        _outward_world(observation.base),
        polygon_inset_m=max(
            SUPPORT_POLYGON_INSET_M,
            float(environment.config.collision_padding_m),
        ),
    )


def _unexpected_contact(node: Any) -> bool:
    return bool(
        _unexpected_pairs(node)
        or getattr(node, '_target_robot_contact_latched', False)
        or getattr(node, '_payload_hazard_latched', None) is not None
    )


def _execute_arm_route(
    *,
    node: Any,
    runtime: SimpleNamespace,
    reference: TransportObservation,
    anchor: AttachmentAnchor,
    waypoints: Sequence[TransportWaypoint],
) -> TransportObservation:
    previous = reference
    base_world = _base_transform_from_pose(reference.base)
    for waypoint in waypoints:
        preflight = preflight_attached_arm_route(
            node=node,
            environment=runtime.environment_preflight,
            anchor=anchor,
            targets=(waypoint.positions,),
            require_final_clearance=(waypoint.index == waypoints[-1].index),
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
        stationary_support = _support_for_observation(
            node,
            runtime.environment_preflight,
            stationary,
            stationary_scene,
        )
        stationary_cage = post_slide_cage_guard(
            stationary,
            stationary_support,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
        )
        anchor_position, anchor_rotation, anchor_corner = _attachment_errors(
            anchor, stationary
        )
        if not stationary_cage.safe:
            raise RuntimeError(
                f'outward step {waypoint.index} pre-command cage rejected: '
                f'{stationary_cage.reason}'
            )
        if (
            anchor_position > ARM_CUMULATIVE_ATTACHMENT_POSITION_LIMIT_M
            or anchor_rotation > ARM_CUMULATIVE_ROTATION_LIMIT_RAD
            or anchor_corner > ARM_CUMULATIVE_CORNER_ERROR_LIMIT_M
            or float(np.max(np.abs(stationary.arm - previous.arm)))
            > ARM_ENDPOINT_JOINT_LIMIT_RAD
            or float(np.linalg.norm(stationary.base[:2] - reference.base[:2]))
            > ARM_BASE_CUMULATIVE_LIMIT_M
            or _angle_error(stationary.base[2], reference.base[2])
            > ARM_BASE_YAW_LIMIT_RAD
        ):
            raise RuntimeError(
                f'outward step {waypoint.index} pre-command attachment/base '
                'state changed'
            )
        previous = stationary
        legs = ((
            waypoint.positions,
            ARM_STAGE_DURATION_S,
            POST_SLIDE_TRANSPORT_EVENT,
        ),)
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        moved, contact_loss = node._send_retained_arm_trajectory(
            goal, duration, legs, POST_SLIDE_TRANSPORT_EVENT
        )
        if not moved:
            reason = 'contact_loss' if contact_loss else 'controller_failure'
            raise PostSlideTransportFailure(
                f'outward step {waypoint.index} failed: {reason}'
            )
        endpoint = node._wait_for_retained_endpoint(
            waypoint.positions,
            command=POST_SLIDE_TRANSPORT_EVENT,
            phase='outward_arm_clearance',
            leg=waypoint.index,
            arm_tolerance=BASE_RETREAT_ARM_LIMIT_RAD,
        )
        if endpoint is None:
            raise PostSlideTransportFailure(
                f'outward step {waypoint.index} endpoint was not held'
            )
        left, right, palm = _fresh_cage_gate(node)
        current, scene = _transport_observation(
            node, runtime, newer_than=previous.observed_at
        )
        support = _support_for_observation(
            node, runtime.environment_preflight, current, scene
        )
        expected_hand_world = base_world @ waypoint.target_pose
        guard = arm_transport_step_guard(
            reference,
            previous,
            current,
            anchor,
            support,
            expected_hand_world=expected_hand_world,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
            expected_arm=waypoint.positions,
        )
        _emit(
            'post_slide_outward_step',
            passed=guard.safe,
            reason=guard.reason,
            step=waypoint.index,
            requested_outward_distance_m=waypoint.outward_distance_m,
            predicted_shelf_clearance_m=(
                waypoint.predicted_shelf_clearance_m
            ),
            measured_endpoint_q=current.arm.tolist(),
            preflight_detail=preflight.detail,
            diagnostic_truth_and_contacts_only=True,
            **guard.metrics,
        )
        if not guard.safe:
            raise PostSlideTransportFailure(
                f'outward step {waypoint.index} gate failed: {guard.reason}'
            )
        previous = current
    if shelf_clearance_m(previous.corners) < MINIMUM_SHELF_CLEARANCE_M:
        raise PostSlideTransportFailure(
            'arm route ended before shelf clearance'
        )
    return previous


def _run_base_retreat(
    *,
    node: Any,
    nav: Any,
    runtime: SimpleNamespace,
    reference: TransportObservation,
    anchor: AttachmentAnchor,
    retreat_distance_m: float,
) -> tuple[TransportObservation, float]:
    preflight = preflight_attached_base_retreat(
        node=node,
        environment=runtime.environment_preflight,
        anchor=anchor,
        retreat_distance_m=retreat_distance_m,
    )
    if not preflight.safe:
        raise RuntimeError(
            f'base retreat preflight rejected: '
            f'{preflight.code}: {preflight.detail}'
        )
    left, right, palm = _fresh_cage_gate(node)
    start, scene = _transport_observation(
        node, runtime, newer_than=reference.observed_at
    )
    start_support = _support_for_observation(
        node, runtime.environment_preflight, start, scene
    )
    start_guard = base_retreat_sample_guard(
        start,
        start,
        start,
        anchor,
        start_support,
        reference_time=_node_time_seconds(node),
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=_unexpected_contact(node),
    )
    if not start_guard.safe:
        raise RuntimeError(
            f'post-preflight base start gate failed: {start_guard.reason}'
        )
    if nav.pose is None:
        raise RuntimeError('navigation odometry is unavailable')
    nav.position_tolerance = min(
        float(nav.position_tolerance), BASE_RETREAT_POSITION_TOLERANCE_M
    )
    nav.probe_terminal_event = None
    outward = _outward_world(start.base)
    goal = (
        float(nav.pose[0] + retreat_distance_m * outward[0]),
        float(nav.pose[1] + retreat_distance_m * outward[1]),
        float(nav.pose[2]),
    )
    nav._accept_goal(goal, profile='carried_retreat')
    previous = start
    last_emit = -1.0
    while nav.goal is not None:
        current, current_scene = _transport_observation(
            node, runtime, newer_than=previous.observed_at
        )
        support = _support_for_observation(
            node, runtime.environment_preflight, current, current_scene
        )
        left, right = node.probe_exact_finger_sides(max_age=0.22)
        palm = node.probe_exact_palm_contact(max_age=0.22)
        guard = base_retreat_sample_guard(
            start,
            previous,
            current,
            anchor,
            support,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
        )
        travelled = guard.metrics.get('base_outward_travel_m', 0.0)
        if travelled - last_emit >= 0.005 or not guard.safe:
            _emit(
                'post_slide_base_retreat_sample',
                passed=guard.safe,
                reason=guard.reason,
                requested_distance_m=retreat_distance_m,
                diagnostic_truth_and_contacts_only=True,
                **guard.metrics,
            )
            last_emit = travelled
        if not guard.safe:
            nav._finish_goal('cancelled', reason=guard.reason)
            raise PostSlideTransportFailure(
                f'base retreat gate failed: {guard.reason}'
            )
        previous = current
        time.sleep(BASE_RETREAT_POLL_S)
    nav._publish_zero()
    if getattr(nav, 'probe_terminal_event', None) != 'reached':
        raise PostSlideTransportFailure(
            'base retreat ended as '
            f'{getattr(nav, "probe_terminal_event", None)!r}'
        )
    left, right, palm = _fresh_cage_gate(node)
    final, final_scene = _transport_observation(
        node, runtime, newer_than=previous.observed_at
    )
    support = _support_for_observation(
        node, runtime.environment_preflight, final, final_scene
    )
    guard = base_retreat_sample_guard(
        start,
        previous,
        final,
        anchor,
        support,
        reference_time=_node_time_seconds(node),
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=_unexpected_contact(node),
    )
    travelled = float(guard.metrics.get('base_outward_travel_m', math.nan))
    if not guard.safe:
        raise PostSlideTransportFailure(
            f'final base retreat gate failed: {guard.reason}'
        )
    if (
        not math.isfinite(travelled)
        or abs(travelled - retreat_distance_m)
        > BASE_RETREAT_DISTANCE_TOLERANCE_M
    ):
        raise PostSlideTransportFailure(
            f'base travelled {travelled:.6f} m, requested '
            f'{retreat_distance_m:.6f} m'
        )
    return final, travelled


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
        name='rigid-palm-post-slide-transport',
        daemon=True,
    )
    thread.start()
    stage = 'created'
    base_motion_commanded = False
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
        node._gravity_supported_payload = True
        node._transport_lock_engaged = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        node.reset_probe_audit()

        left, right, palm = _fresh_cage_gate(node)
        reference, scene = _transport_observation(node, runtime)
        support = _support_for_observation(
            node, runtime.environment_preflight, reference, scene
        )
        cage = post_slide_cage_guard(
            reference,
            support,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
        )
        if not cage.safe:
            raise RuntimeError(f'post-slide cage rejected: {cage.reason}')
        anchor = capture_attachment_anchor(
            reference.book,
            reference.corners,
            reference.hand_world,
            observed_at=reference.observed_at,
        )
        waypoints = solve_outward_waypoints(
            node.chain, reference.arm, reference.base, anchor
        )
        complete = preflight_attached_arm_route(
            node=node,
            environment=runtime.environment_preflight,
            anchor=anchor,
            targets=tuple(waypoint.positions for waypoint in waypoints),
            require_final_clearance=True,
        )
        if not complete.safe:
            raise RuntimeError(
                f'complete post-slide arm route rejected: '
                f'{complete.code}: {complete.detail}'
            )
        _emit(
            'post_slide_transport_resume_verified',
            measured_aperture_m=reference.aperture_m,
            measured_start_q=reference.arm.tolist(),
            hand_T_book=anchor.hand_from_book.tolist(),
            book_T_hand=anchor.book_from_hand.tolist(),
            book_corners_in_hand=anchor.corners_in_hand.tolist(),
            outward_step_count=len(waypoints),
            complete_preflight_detail=complete.detail,
            diagnostic_truth_and_contacts_only=True,
            **cage.metrics,
        )

        stage = 'fixed_orientation_arm_clearance'
        arm_final = _execute_arm_route(
            node=node,
            runtime=runtime,
            reference=reference,
            anchor=anchor,
            waypoints=waypoints,
        )
        current_scene = runtime.environment_preflight._read_scene()
        if _target_shelf_collision(
            arm_final.corners,
            current_scene,
            float(runtime.environment_preflight.config.collision_padding_m),
        ):
            raise RuntimeError('padded target is not fully shelf-clear')

        retreat_distance = choose_base_retreat_distance(
            arm_final.corners, _outward_world(arm_final.base)
        )
        stage = 'slow_straight_base_retreat'
        base_motion_commanded = True
        final, travelled = _run_base_retreat(
            node=node,
            nav=nav,
            runtime=runtime,
            reference=arm_final,
            anchor=anchor,
            retreat_distance_m=retreat_distance,
        )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=POST_SLIDE_TRANSPORT_EVENT,
            measured_aperture_m=final.aperture_m,
            measured_endpoint_q=final.arm.tolist(),
            book_position=final.book.position.tolist(),
            book_minimum=final.book.minimum.tolist(),
            book_maximum=final.book.maximum.tolist(),
            shelf_clearance_m=shelf_clearance_m(final.corners),
            base_retreat_requested_m=retreat_distance,
            base_retreat_measured_m=travelled,
            hand_T_book=anchor.hand_from_book.tolist(),
            book_T_hand=anchor.book_from_hand.tolist(),
            gripper_opened=False,
            compaction_commanded=False,
            mission_navigation_commanded=False,
            base_retreat_commanded=True,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
        )
    except Exception as error:
        nav._publish_zero()
        if nav.goal is not None:
            nav._finish_goal('cancelled', reason='diagnostic_failure')
        _emit(
            'result',
            passed=False,
            stage=stage,
            reason=f'{type(error).__name__}: {error}',
            gripper_opened=False,
            compaction_commanded=False,
            mission_navigation_commanded=False,
            base_retreat_commanded=base_motion_commanded,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
        )
        raise
    finally:
        nav._publish_zero()
        node._cancel.set()
        executor.shutdown()
        thread.join(timeout=3.0)
        nav.destroy_node()
        node.destroy_node()
        if spin_errors:
            raise RuntimeError(f'executor spin failed: {spin_errors[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Diagnostic-only post-tangent-slide extraction/retreat'
    )
    parser.add_argument(
        '--confirm-diagnostic-post-slide-transport',
        action='store_true',
        help='authorize guarded diagnostic arm clearance and base retreat',
    )
    arguments = parser.parse_args()
    if not arguments.confirm_diagnostic_post_slide_transport:
        parser.error('--confirm-diagnostic-post-slide-transport is required')
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
