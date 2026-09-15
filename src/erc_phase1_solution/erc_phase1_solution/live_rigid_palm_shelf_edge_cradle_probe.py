#!/usr/bin/env python3
"""Diagnostic-only shelf-edge transfer for the seed-101 rigid palm.

This probe deliberately uses Gazebo truth and the temporary palm contact
selector.  It is engineering evidence, not production competition logic.  It
stops the proven outward extraction at step 18, while about 10 mm of the book
still overlaps the shelf, then rotates the hand beneath the nominally
stationary book about the outer edge of the palm's top pad.  Every one-degree
leg is separately preflighted and observed.  Failure leaves the 30 mm cage
closed and commands no recovery motion.

This first-stage probe is deliberately capped at five degrees.  The outer-edge
pivot is collision-clear, but the finite palm pad does not put the book's
vertical gravity projection inside its support area.  Passing this probe is
only evidence about shelf-supported palm/finger/wedge behaviour; it does not
authorize shelf-edge crossing or transport.

Importing this module is side-effect free.  Controller clients are used only
from :func:`main` after all resume and preflight gates pass.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from erc_phase1_solution.kinematics import (
    oriented_box_from_corners,
    oriented_box_intersects_triangles,
    pose_matrix,
    triangle_meshes_intersect,
)
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    EXPECTED_CAGED_APERTURE_M,
    EXPECTED_CAGED_JOINTS,
    GuardResult,
    SHELF_FRONT_X_M,
    STEP_BASE_POSITION_LIMIT_M,
    STEP_BASE_YAW_LIMIT_RAD,
    TARGET_BOOK_MODEL,
    _angle_error,
    _classify_resume_state,
    _coherent_scene,
    _emit,
    _execute_extraction,
    _fresh_cage_gate,
    intermediate_extraction_resume_guard,
    _probe_types,
    _unexpected_pairs,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    CAGE_MEASURED_TARGET_TOLERANCE_M,
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
    rotation_matrix_distance,
    world_hand_pose,
)
from erc_phase1_solution.rigid_palm_live_preflight import (
    _measured_finger_transforms_relative_to_palm,
    _measured_robot_state,
    _node_time_seconds,
    _require_stable_measured_finger_geometry,
    _require_stable_preflight_inputs,
    densify_joint_samples,
)
from erc_phase1_solution.rigid_palm_preflight import (
    GripperSample,
    ProbePhase,
    preflight_gripper_sweep,
)


CRADLE_EVENT = 'rigid_palm_shelf_edge_cradle'
EXTRACTION_STOP_STEP = 18
CRADLE_STEP_RAD = math.radians(1.0)
CRADLE_DEFAULT_MAX_ANGLE_DEG = 5
CRADLE_MAX_ANGLE_DEG = 5
CRADLE_LEG_DURATION_S = 0.50

# The official palm mesh's load-bearing top facet spans local z from
# -11.537 mm to +11.463 mm at local x=-46.50 mm.  Pivot on the outer edge:
# R0*Ry(-angle) then leaves that edge fixed and lowers every point toward the
# inner (+z) edge.  The former +z/inner-edge pivot raised the opposite half of
# the pad into a nominally stationary book and exceeded the unchanged 0.5 mm
# palm-overlap cap.  This is mesh-derived geometry, not a fitted seed pose.
PALM_OUTER_TOP_EDGE_IN_GRASP_M = np.asarray(
    [-0.04650, 0.0, -0.011537], dtype=float
)

EXPECTED_STEP18_JOINTS = np.asarray(
    [
        0.34999998897070606,
        0.3822254566345725,
        0.5258644813930865,
        0.4355273966236304,
        -1.8212814698501696,
        1.327510835219393,
        0.8002464027363253,
        -1.3830228375779812,
    ],
    dtype=float,
)
EXPECTED_STEP18_BOOK_POSITION = np.asarray(
    [2.6846923463259027, -0.14421938761973457, 1.576999694477571],
    dtype=float,
)
EXPECTED_STEP18_BOOK_MAXIMUM_X_M = 2.7649729506091196

EDGE_RESUME_JOINT_TOLERANCE_RAD = 0.006
EDGE_RESUME_BOOK_POSITION_TOLERANCE_M = 0.004
EDGE_RESUME_BOOK_VERTICAL_TOLERANCE_M = 0.002
EDGE_RESUME_BOOK_ROTATION_TOLERANCE_RAD = math.radians(2.0)
EDGE_RESUME_BASE_POSITION_TOLERANCE_M = 0.005
EDGE_RESUME_BASE_YAW_TOLERANCE_RAD = 0.008

CRADLE_MINIMUM_SHELF_OVERLAP_M = 0.008
CRADLE_STEP_BOOK_HORIZONTAL_LIMIT_M = 0.00075
CRADLE_STEP_BOOK_VERTICAL_LIMIT_M = 0.00075
CRADLE_STEP_BOOK_DROP_LIMIT_M = 0.00050
CRADLE_STEP_BOOK_ROTATION_LIMIT_RAD = math.radians(1.0)
CRADLE_CUMULATIVE_BOOK_HORIZONTAL_LIMIT_M = 0.00150
CRADLE_CUMULATIVE_BOOK_VERTICAL_LIMIT_M = 0.00100
CRADLE_CUMULATIVE_BOOK_ROTATION_LIMIT_RAD = math.radians(2.0)
CRADLE_HAND_STEP_MINIMUM_RAD = math.radians(0.50)
CRADLE_HAND_STEP_MAXIMUM_RAD = math.radians(1.50)
CRADLE_HAND_ANGLE_ERROR_LIMIT_RAD = math.radians(0.25)
CRADLE_PIVOT_DRIFT_LIMIT_M = 0.00035
CRADLE_ENDPOINT_ARM_TOLERANCE_RAD = 0.004
CRADLE_BASE_CUMULATIVE_LIMIT_M = 0.00075
CRADLE_BASE_YAW_CUMULATIVE_LIMIT_RAD = 0.0010

MOVING_ARM_LINKS = tuple(f'arm_left_{index}_link' for index in range(1, 8))


@dataclass(frozen=True)
class CradleWaypoint:
    angle_deg: int
    positions: np.ndarray
    target_pose: np.ndarray


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=float,
    )


def pivoted_cradle_pose(
    reference_pose: Sequence[Sequence[float]],
    angle_rad: float,
    *,
    pivot_in_grasp: Sequence[float] = PALM_OUTER_TOP_EDGE_IN_GRASP_M,
) -> np.ndarray:
    """Rotate ``-local-y`` while leaving the selected palm point fixed."""

    reference = np.asarray(reference_pose, dtype=float)
    pivot = np.asarray(pivot_in_grasp, dtype=float)
    angle = float(angle_rad)
    if (
        reference.shape != (4, 4)
        or not np.all(np.isfinite(reference))
        or pivot.shape != (3,)
        or not np.all(np.isfinite(pivot))
        or not math.isfinite(angle)
        or angle < 0.0
        or angle > math.radians(CRADLE_MAX_ANGLE_DEG) + 1e-12
    ):
        raise ValueError('invalid shelf-edge cradle pose input')
    rotation = reference[:3, :3] @ _rotation_y(-angle)
    pivot_world = reference[:3, :3] @ pivot + reference[:3, 3]
    result = pose_matrix(pivot_world - rotation @ pivot, rotation)
    return result


def _solve_cradle_waypoints(node: Any, maximum_angle_deg: int) -> list[CradleWaypoint]:
    maximum = int(maximum_angle_deg)
    if not 1 <= maximum <= CRADLE_MAX_ANGLE_DEG:
        raise ValueError(
            f'cradle angle must be in [1, {CRADLE_MAX_ANGLE_DEG}] degrees'
        )
    reference = EXPECTED_STEP18_JOINTS.copy()
    reference_pose = node.chain.forward(reference)
    previous = reference
    waypoints: list[CradleWaypoint] = []
    for angle_deg in range(1, maximum + 1):
        target_pose = pivoted_cradle_pose(
            reference_pose, math.radians(angle_deg)
        )
        solution, _ = node.chain.solve(
            target_pose,
            [previous, reference],
            position_tolerance=0.00020,
            orientation_tolerance=0.00050,
            max_iterations=600,
            fixed_positions={'torso_lift_joint': float(reference[0])},
        )
        if solution is None:
            raise RuntimeError(f'cradle IK failed at {angle_deg} degrees')
        positions = np.asarray(solution, dtype=float)
        achieved = node.chain.forward(positions)
        error = node.chain.pose_error(achieved, target_pose)
        if (
            positions.shape != (8,)
            or not np.all(np.isfinite(positions))
            or float(np.max(np.abs(positions[1:] - previous[1:]))) > 0.08
            or float(np.linalg.norm(error[:3])) > 0.00020
            or float(np.linalg.norm(error[3:])) > 0.00050
        ):
            raise RuntimeError(
                f'cradle IK endpoint is not an exact continuous solution at '
                f'{angle_deg} degrees'
            )
        waypoints.append(CradleWaypoint(angle_deg, positions, target_pose))
        previous = positions
    return waypoints


def _inflated_ordered_corners(
    corners: Sequence[Sequence[float]], padding_m: float
) -> np.ndarray:
    points = np.asarray(corners, dtype=float)
    padding = float(padding_m)
    center, axes, half_extents = oriented_box_from_corners(points)
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
    return (signs * (half_extents + padding)) @ axes.T + center


def _bounds_overlap(first: np.ndarray, second: np.ndarray, padding: float) -> bool:
    return bool(
        np.all(first[1] + padding >= second[0])
        and np.all(second[1] + padding >= first[0])
    )


def _moving_arm_environment_collision(
    node: Any,
    scene: Any,
    positions: Sequence[float],
    *,
    padding_m: float,
) -> tuple[str, str] | None:
    """Check every moving left-arm mesh against shelf and every book."""

    transforms = node.chain.link_transforms(positions)
    shelf = np.asarray(scene.shelf_triangles, dtype=float)
    shelf_bounds = np.asarray(
        [np.min(shelf, axis=(0, 1)), np.max(shelf, axis=(0, 1))]
    )
    padded_books = {
        name: _inflated_ordered_corners(book.corners, padding_m)
        for name, book in scene.books.items()
    }
    book_bounds = {
        name: np.asarray([np.min(corners, axis=0), np.max(corners, axis=0)])
        for name, corners in padded_books.items()
    }
    for mesh in node.carried_collision_meshes:
        if mesh.link not in MOVING_ARM_LINKS:
            continue
        transform = scene.base_transform @ transforms[mesh.link]
        triangles = (
            np.asarray(mesh.triangles, dtype=float) @ transform[:3, :3].T
            + transform[:3, 3]
        )
        bounds = np.asarray(
            [np.min(triangles, axis=(0, 1)), np.max(triangles, axis=(0, 1))]
        )
        if _bounds_overlap(bounds, shelf_bounds, padding_m) and (
            triangle_meshes_intersect(
                triangles,
                shelf,
                tolerance=padding_m,
                first_watertight=bool(mesh.watertight),
            )
        ):
            return mesh.link, 'shelf'
        for name, corners in padded_books.items():
            if not _bounds_overlap(bounds, book_bounds[name], 0.0):
                continue
            if oriented_box_intersects_triangles(
                corners,
                triangles,
                closed_surface=bool(mesh.watertight),
            ):
                return mesh.link, name
    return None


def _preflight_cradle_path(
    node: Any,
    environment: Any,
    targets: Sequence[Sequence[float]],
) -> int:
    """Run a static-target, measured-finger dense full-mesh preflight."""

    before_state = _measured_robot_state(node)
    goals = tuple(np.asarray(target, dtype=float) for target in targets)
    if not goals or any(
        goal.shape != before_state.positions.shape
        or not np.all(np.isfinite(goal))
        for goal in goals
    ):
        raise RuntimeError('cradle preflight path is empty or malformed')
    if (
        float(np.max(np.abs(before_state.positions - goals[-1]))) <= 1e-12
        or abs(before_state.aperture_m - EXPECTED_CAGED_APERTURE_M)
        > CAGE_MEASURED_TARGET_TOLERANCE_M
    ):
        raise RuntimeError('cradle preflight start/target state is invalid')

    scene = environment._read_scene()
    reference_time = _node_time_seconds(node)
    measured_relative = _measured_finger_transforms_relative_to_palm(
        node, scene, before_state
    )

    def transforms(positions: np.ndarray) -> Mapping[str, np.ndarray]:
        return environment._world_transforms(
            node,
            scene,
            positions,
            before_state.aperture_m,
            measured_relative,
        )

    dense = densify_joint_samples(
        model=environment.model,
        joint_samples=(before_state.positions, *goals),
        transform_factory=transforms,
        max_vertex_step_m=environment.config.max_vertex_step_m,
    )

    target_world = _inflated_ordered_corners(
        scene.books[TARGET_BOOK_MODEL].corners,
        float(node.carried_book_padding),
    )
    inverse_base = np.linalg.inv(scene.base_transform)
    target_base = (
        target_world @ inverse_base[:3, :3].T + inverse_base[:3, 3]
    )
    for index, sample in enumerate(dense):
        self_collision = node._robot_self_collision(sample.positions)
        if self_collision is not None:
            raise RuntimeError(
                f'cradle dense sample {index} has self-collision: '
                f'{self_collision}'
            )
        payload_collision = node._carried_robot_collision(
            sample.positions, target_base
        )
        if payload_collision is not None:
            raise RuntimeError(
                f'cradle dense sample {index} puts the static target into '
                f'{payload_collision}'
            )
        environment_collision = _moving_arm_environment_collision(
            node,
            scene,
            sample.positions,
            padding_m=environment.config.collision_padding_m,
        )
        if environment_collision is not None:
            raise RuntimeError(
                f'cradle dense sample {index} has moving-arm collision: '
                f'{environment_collision}'
            )

    samples = tuple(
        GripperSample(
            transforms=sample.transforms,
            measured_aperture_m=before_state.aperture_m,
            expected_aperture_m=before_state.aperture_m,
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
        raise RuntimeError(
            'static-target cradle mesh preflight rejected the leg: '
            f'{result.code}: {result.detail}'
        )

    after_scene = environment._read_scene()
    after_state = _measured_robot_state(node)
    after_relative = _measured_finger_transforms_relative_to_palm(
        node, after_scene, after_state
    )
    _require_stable_measured_finger_geometry(
        environment.model, measured_relative, after_relative
    )
    _require_stable_preflight_inputs(
        scene, after_scene, before_state, after_state
    )
    return len(dense)


def _preflight_cradle_leg(
    node: Any,
    environment: Any,
    target: Sequence[float],
) -> int:
    return _preflight_cradle_path(node, environment, (target,))


def cradle_step_guard(
    reference_book: BookSnapshot,
    before_book: BookSnapshot,
    after_book: BookSnapshot,
    reference_hand_world: Sequence[Sequence[float]],
    before_hand_world: Sequence[Sequence[float]],
    after_hand_world: Sequence[Sequence[float]],
    reference_base: Sequence[float],
    before_base: Sequence[float],
    after_base: Sequence[float],
    *,
    expected_angle_rad: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    measured_aperture_m: float,
) -> GuardResult:
    """Require the hand to cradle while the shelf-supported book stays put."""

    try:
        reference_hand = np.asarray(reference_hand_world, dtype=float)
        first_hand = np.asarray(before_hand_world, dtype=float)
        second_hand = np.asarray(after_hand_world, dtype=float)
        base0 = np.asarray(reference_base, dtype=float)
        base1 = np.asarray(before_base, dtype=float)
        base2 = np.asarray(after_base, dtype=float)
        if any(
            hand.shape != (4, 4)
            for hand in (reference_hand, first_hand, second_hand)
        ):
            raise ValueError('hand transforms must be 4x4')
        if any(base.shape != (3,) for base in (base0, base1, base2)):
            raise ValueError('base poses must contain x, y, yaw')
        values = (*reference_hand.flat, *first_hand.flat, *second_hand.flat,
                  *base0, *base1, *base2, float(expected_angle_rad),
                  float(measured_aperture_m))
        if not np.all(np.isfinite(values)):
            raise ValueError('cradle observation is non-finite')

        step_delta = after_book.position - before_book.position
        cumulative_delta = after_book.position - reference_book.position
        step_horizontal = float(np.linalg.norm(step_delta[:2]))
        step_vertical = abs(float(step_delta[2]))
        step_drop = max(0.0, -float(step_delta[2]))
        step_rotation = quaternion_distance(
            before_book.quaternion, after_book.quaternion
        )
        cumulative_horizontal = float(np.linalg.norm(cumulative_delta[:2]))
        cumulative_vertical = abs(float(cumulative_delta[2]))
        cumulative_rotation = quaternion_distance(
            reference_book.quaternion, after_book.quaternion
        )
        hand_step = rotation_matrix_distance(
            first_hand[:3, :3], second_hand[:3, :3]
        )
        hand_angle = rotation_matrix_distance(
            reference_hand[:3, :3], second_hand[:3, :3]
        )
        pivot = PALM_OUTER_TOP_EDGE_IN_GRASP_M
        reference_pivot = reference_hand[:3, :3] @ pivot + reference_hand[:3, 3]
        current_pivot = second_hand[:3, :3] @ pivot + second_hand[:3, 3]
        pivot_drift = float(np.linalg.norm(current_pivot - reference_pivot))
        step_base_drift = float(np.linalg.norm(base2[:2] - base1[:2]))
        step_base_yaw = _angle_error(base2[2], base1[2])
        cumulative_base_drift = float(np.linalg.norm(base2[:2] - base0[:2]))
        cumulative_base_yaw = _angle_error(base2[2], base0[2])
        shelf_overlap = float(after_book.maximum[0] - SHELF_FRONT_X_M)
        aperture_error = abs(
            float(measured_aperture_m) - EXPECTED_CAGED_APERTURE_M
        )
        metrics = {
            'book_step_horizontal_m': step_horizontal,
            'book_step_vertical_m': step_vertical,
            'book_step_drop_m': step_drop,
            'book_step_rotation_rad': step_rotation,
            'book_cumulative_horizontal_m': cumulative_horizontal,
            'book_cumulative_vertical_m': cumulative_vertical,
            'book_cumulative_rotation_rad': cumulative_rotation,
            'hand_step_rotation_rad': hand_step,
            'hand_cumulative_rotation_rad': hand_angle,
            'hand_angle_error_rad': abs(hand_angle - float(expected_angle_rad)),
            'palm_pivot_drift_m': pivot_drift,
            'shelf_overlap_m': shelf_overlap,
            'base_step_drift_m': step_base_drift,
            'base_step_yaw_drift_rad': step_base_yaw,
            'base_cumulative_drift_m': cumulative_base_drift,
            'base_cumulative_yaw_drift_rad': cumulative_base_yaw,
            'aperture_error_m': aperture_error,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        return GuardResult(False, f'invalid_observation:{exc}', {})

    checks = (
        (left_contact, 'left_contact_missing'),
        (right_contact, 'right_contact_missing'),
        (palm_contact, 'palm_contact_missing'),
        (not unexpected_contacts, 'unexpected_contact'),
        (
            aperture_error <= CAGE_MEASURED_TARGET_TOLERANCE_M,
            'cage_aperture_changed',
        ),
        (
            CRADLE_HAND_STEP_MINIMUM_RAD <= hand_step
            <= CRADLE_HAND_STEP_MAXIMUM_RAD,
            'hand_step_angle_out_of_bounds',
        ),
        (
            abs(hand_angle - float(expected_angle_rad))
            <= CRADLE_HAND_ANGLE_ERROR_LIMIT_RAD,
            'hand_cumulative_angle_mismatch',
        ),
        (pivot_drift <= CRADLE_PIVOT_DRIFT_LIMIT_M, 'palm_pivot_moved'),
        (
            step_horizontal <= CRADLE_STEP_BOOK_HORIZONTAL_LIMIT_M,
            'book_shifted_during_cradle_step',
        ),
        (
            step_vertical <= CRADLE_STEP_BOOK_VERTICAL_LIMIT_M,
            'book_vertical_motion_during_cradle_step',
        ),
        (step_drop <= CRADLE_STEP_BOOK_DROP_LIMIT_M, 'book_dropped'),
        (
            step_rotation <= CRADLE_STEP_BOOK_ROTATION_LIMIT_RAD,
            'book_rotated_during_cradle_step',
        ),
        (
            cumulative_horizontal <= CRADLE_CUMULATIVE_BOOK_HORIZONTAL_LIMIT_M,
            'book_cumulative_horizontal_shift',
        ),
        (
            cumulative_vertical <= CRADLE_CUMULATIVE_BOOK_VERTICAL_LIMIT_M,
            'book_cumulative_vertical_shift',
        ),
        (
            cumulative_rotation <= CRADLE_CUMULATIVE_BOOK_ROTATION_LIMIT_RAD,
            'book_cumulative_rotation',
        ),
        (
            shelf_overlap >= CRADLE_MINIMUM_SHELF_OVERLAP_M,
            'shelf_support_overlap_lost',
        ),
        (step_base_drift <= STEP_BASE_POSITION_LIMIT_M, 'base_moved_during_step'),
        (step_base_yaw <= STEP_BASE_YAW_LIMIT_RAD, 'base_rotated_during_step'),
        (
            cumulative_base_drift <= CRADLE_BASE_CUMULATIVE_LIMIT_M,
            'base_cumulative_drift',
        ),
        (
            cumulative_base_yaw <= CRADLE_BASE_YAW_CUMULATIVE_LIMIT_RAD,
            'base_cumulative_rotation',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def _edge_resume_angle(
    node: Any,
    book: BookSnapshot,
    base: np.ndarray,
    arm: np.ndarray,
    aperture: float,
    waypoints: Sequence[CradleWaypoint],
) -> int:
    if abs(aperture - EXPECTED_CAGED_APERTURE_M) > CAGE_MEASURED_TARGET_TOLERANCE_M:
        raise RuntimeError('gripper is not at the proven 30 mm cage')
    if (
        np.linalg.norm(book.position[:2] - EXPECTED_STEP18_BOOK_POSITION[:2])
        > EDGE_RESUME_BOOK_POSITION_TOLERANCE_M
        or abs(float(book.position[2] - EXPECTED_STEP18_BOOK_POSITION[2]))
        > EDGE_RESUME_BOOK_VERTICAL_TOLERANCE_M
        or quaternion_distance(
            book.quaternion, EXPECTED_RELEASED_BOOK_QUATERNION
        ) > EDGE_RESUME_BOOK_ROTATION_TOLERANCE_RAD
        or float(book.maximum[0] - SHELF_FRONT_X_M)
        < CRADLE_MINIMUM_SHELF_OVERLAP_M
        or abs(float(book.maximum[0] - EXPECTED_STEP18_BOOK_MAXIMUM_X_M))
        > EDGE_RESUME_BOOK_POSITION_TOLERANCE_M
        or np.linalg.norm(base[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
        > EDGE_RESUME_BASE_POSITION_TOLERANCE_M
        or _angle_error(base[2], EXPECTED_RELEASED_BASE_POSE[2])
        > EDGE_RESUME_BASE_YAW_TOLERANCE_RAD
    ):
        raise RuntimeError('book/base is not at the supported step-18 checkpoint')
    candidates = [(0, EXPECTED_STEP18_JOINTS)] + [
        (waypoint.angle_deg, waypoint.positions) for waypoint in waypoints
    ]
    errors = [
        (float(np.max(np.abs(arm - positions))), angle)
        for angle, positions in candidates
    ]
    error, angle = min(errors)
    if error > EDGE_RESUME_JOINT_TOLERANCE_RAD:
        raise RuntimeError('arm is not at a recognized shelf-edge cradle checkpoint')
    return int(angle)


def _execute_cradle(
    node: Any,
    runtime: SimpleNamespace,
    waypoints: Sequence[CradleWaypoint],
    start_angle_deg: int,
) -> BookSnapshot:
    reference_book, reference_base, scene_stamp, _ = _coherent_scene(runtime)
    reference_hand = world_hand_pose(
        reference_base, node.chain.forward(EXPECTED_STEP18_JOINTS)
    )
    previous_book = reference_book
    previous_base = reference_base
    previous_arm = np.asarray(node._measured_left_solution(), dtype=float)
    previous_hand = world_hand_pose(
        previous_base, node.chain.forward(previous_arm)
    )

    for waypoint in waypoints:
        if waypoint.angle_deg <= start_angle_deg:
            continue
        dense_samples = _preflight_cradle_leg(
            node, runtime.environment_preflight, waypoint.positions
        )
        _fresh_cage_gate(node)
        legs = ((waypoint.positions, CRADLE_LEG_DURATION_S, CRADLE_EVENT),)
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        moved, contact_loss = node._send_retained_arm_trajectory(
            goal, duration, legs, 'rigid_palm_shelf_edge_cradle'
        )
        if not moved:
            reason = 'contact_loss' if contact_loss else 'controller_failure'
            raise RuntimeError(
                f'cradle step {waypoint.angle_deg} failed: {reason}'
            )
        endpoint = node._wait_for_retained_endpoint(
            waypoint.positions,
            command='rigid_palm_shelf_edge_cradle',
            phase=CRADLE_EVENT,
            leg=waypoint.angle_deg,
        )
        if endpoint is None:
            raise RuntimeError(
                f'cradle endpoint {waypoint.angle_deg} was not reached safely'
            )
        measured_arm = np.asarray(node._measured_left_solution(), dtype=float)
        if float(np.max(np.abs(measured_arm - waypoint.positions))) > (
            CRADLE_ENDPOINT_ARM_TOLERANCE_RAD
        ):
            raise RuntimeError('measured cradle endpoint differs from its plan')
        left, right, palm = _fresh_cage_gate(node)
        current_book, current_base, current_stamp, _ = _coherent_scene(
            runtime, newer_than=scene_stamp
        )
        current_hand = world_hand_pose(
            current_base, node.chain.forward(measured_arm)
        )
        unexpected = bool(_unexpected_pairs(node)) or bool(
            getattr(node, '_target_robot_contact_latched', False)
        ) or getattr(node, '_payload_hazard_latched', None) is not None
        guard = cradle_step_guard(
            reference_book,
            previous_book,
            current_book,
            reference_hand,
            previous_hand,
            current_hand,
            reference_base,
            previous_base,
            current_base,
            expected_angle_rad=math.radians(waypoint.angle_deg),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=unexpected,
            measured_aperture_m=float(
                node.joints['gripper_left_finger_joint']
            ),
        )
        _emit(
            'cradle_step',
            angle_deg=waypoint.angle_deg,
            passed=guard.safe,
            reason=guard.reason,
            dense_preflight_samples=dense_samples,
            solution=np.asarray(endpoint, dtype=float).tolist(),
            book_position=current_book.position.tolist(),
            book_maximum=current_book.maximum.tolist(),
            scene_stamp=current_stamp,
            **guard.metrics,
        )
        if not guard.safe:
            raise RuntimeError(
                f'cradle endpoint gate failed: {guard.reason}'
            )
        previous_book = current_book
        previous_base = current_base
        previous_arm = measured_arm
        previous_hand = current_hand
        scene_stamp = current_stamp
    return previous_book


def _run(runtime: SimpleNamespace, maximum_angle_deg: int) -> None:
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
        except BaseException as exc:
            spin_errors.append(exc)

    thread = threading.Thread(
        target=spin, name='rigid-palm-shelf-edge-cradle-probe', daemon=True
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

        waypoints = _solve_cradle_waypoints(node, maximum_angle_deg)
        left, right, palm = _fresh_cage_gate(node)
        book, base, scene_stamp, _ = _coherent_scene(runtime)
        arm = np.asarray(node._measured_left_solution(), dtype=float)
        aperture = float(node.joints.get('gripper_left_finger_joint', math.nan))
        intermediate_step: int | None = None
        try:
            resume = _classify_resume_state(node, book, base, arm, aperture)
        except RuntimeError:
            expected_hand = world_hand_pose(
                EXPECTED_RELEASED_BASE_POSE,
                node.chain.forward(EXPECTED_CAGED_JOINTS),
            )
            current_hand = world_hand_pose(base, node.chain.forward(arm))
            config = runtime.environment_preflight.config
            intermediate = intermediate_extraction_resume_guard(
                book,
                base,
                current_hand,
                expected_hand,
                aperture_m=aperture,
                observed_at=scene_stamp,
                reference_time=_node_time_seconds(node),
                max_state_age_seconds=float(config.max_state_age_seconds),
                future_tolerance_seconds=float(
                    config.future_tolerance_seconds
                ),
                left_contact=left,
                right_contact=right,
                palm_contact=palm,
                unexpected_contacts=(
                    bool(_unexpected_pairs(node))
                    or bool(
                        getattr(node, '_target_robot_contact_latched', False)
                    )
                    or getattr(node, '_payload_hazard_latched', None) is not None
                ),
            )
            if intermediate.safe:
                intermediate_step = int(
                    intermediate.metrics['inferred_extraction_step']
                )
                resume = 'intermediate_extraction_caged'
                _emit(
                    'intermediate_extraction_resume_verified',
                    step=intermediate_step,
                    left_exact_target_contact=left,
                    right_exact_target_contact=right,
                    exact_target_palm_contact=palm,
                    **intermediate.metrics,
                )
            else:
                _emit(
                    'intermediate_extraction_resume_rejected',
                    reason=intermediate.reason,
                    left_exact_target_contact=left,
                    right_exact_target_contact=right,
                    exact_target_palm_contact=palm,
                    **intermediate.metrics,
                )
                resume = 'shelf_edge_caged'

        if resume in {'tangent_caged', 'intermediate_extraction_caged'}:
            extraction_start_step = (
                0 if resume == 'tangent_caged' else int(intermediate_step)
            )
            stage = 'extracting_to_step18'
            book = _execute_extraction(
                node,
                runtime,
                book,
                base,
                scene_stamp,
                starting_step=extraction_start_step,
                maximum_steps=EXTRACTION_STOP_STEP,
                require_shelf_clearance=False,
            )
            book, base, scene_stamp, _ = _coherent_scene(runtime)
            arm = np.asarray(node._measured_left_solution(), dtype=float)
            aperture = float(node.joints['gripper_left_finger_joint'])
            start_angle = _edge_resume_angle(
                node, book, base, arm, aperture, waypoints
            )
            if start_angle != 0:
                raise RuntimeError('step-18 extraction ended at a cradle angle')
        elif resume == 'shelf_edge_caged':
            start_angle = _edge_resume_angle(
                node, book, base, arm, aperture, waypoints
            )
        else:
            raise RuntimeError(
                'fully extracted state is not valid for a shelf-edge probe'
            )

        if start_angle >= maximum_angle_deg:
            raise RuntimeError(
                f'current cradle angle {start_angle} is not below requested '
                f'{maximum_angle_deg}'
            )
        left, right, palm = _fresh_cage_gate(node)
        stage = 'edge_resume_verified'
        _emit(
            'edge_resume_verified',
            resume_state=resume,
            start_angle_deg=start_angle,
            target_angle_deg=maximum_angle_deg,
            left_exact_target_contact=left,
            right_exact_target_contact=right,
            exact_target_palm_contact=palm,
            aperture_m=aperture,
            shelf_overlap_m=float(book.maximum[0] - SHELF_FRONT_X_M),
            book_position=book.position.tolist(),
            solution=arm.tolist(),
        )

        remaining = [
            waypoint.positions
            for waypoint in waypoints
            if waypoint.angle_deg > start_angle
        ]
        # Prove the complete requested angular sweep before the first degree is
        # dispatched.  In particular, a later target-overlap rejection must
        # leave the hand at the known shelf-supported checkpoint rather than
        # strand it at an intermediate angle.
        route_samples = _preflight_cradle_path(
            node, runtime.environment_preflight, remaining
        )
        _emit(
            'cradle_route_preflight',
            start_angle_deg=start_angle,
            target_angle_deg=maximum_angle_deg,
            dense_samples=route_samples,
            passed=True,
        )

        stage = 'cradling'
        final_book = _execute_cradle(
            node, runtime, waypoints, start_angle
        )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage='rigid_palm_shelf_edge_cradle',
            start_angle_deg=start_angle,
            final_angle_deg=maximum_angle_deg,
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
            'Diagnostic-only seed-101 shelf-edge rigid-palm cradle probe'
        )
    )
    parser.add_argument(
        '--max-angle',
        type=int,
        default=CRADLE_DEFAULT_MAX_ANGLE_DEG,
        help=(
            'total integer cradle angle in degrees '
            f'(default {CRADLE_DEFAULT_MAX_ANGLE_DEG}, maximum '
            f'{CRADLE_MAX_ANGLE_DEG})'
        ),
    )
    arguments = parser.parse_args()
    if not 1 <= int(arguments.max_angle) <= CRADLE_MAX_ANGLE_DEG:
        parser.error(
            f'--max-angle must be an integer in [1, {CRADLE_MAX_ANGLE_DEG}]'
        )
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime, int(arguments.max_angle))
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
