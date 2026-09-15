#!/usr/bin/env python3
"""Diagnostic-only recovery from the partial 47 mm tangent-slide state.

The live tangent experiment established an important mechanical fact: the
target followed the palm through the first 5 mm local-z insertion.  Its pose
relative to the hand therefore did not deepen, even though its overlap with
the shelf increased.  Continuing the 61 mm insertion would repeat that
friction-locked translation and would not create the finite palm support that
the later transport plan requires.

This executable has exactly one arm move.  It verifies and retraces the exact
pre-insertion arm waypoint, a 4.961 mm local-z reverse from the measured
partial state toward the last known shelf-supported pose.  Before that move it
densely preflights the robot,
tool, shelf, and every book under a continuum of target-motion hypotheses:
the target may stay static, follow the hand rigidly, or slip by any sampled
fraction between those extremes.  The reverse is safe in either endpoint
case: a following target returns to about 11 mm of shelf overlap, while a
static target retains about 16 mm and gains 5 mm of palm-relative depth.

After a stable reverse endpoint is observed, the probe may close the left
gripper from 47 to 30.3 mm in one-millimetre-or-smaller stages.  It first runs
a new measured-finger, static-target dense preflight for the entire close and
then repeats that preflight before every stage.  Fresh exact palm evidence is
mandatory before every command and fresh bilateral evidence is mandatory at
the final cage.  Any ambiguity stops without another arm move.  The probe
never opens farther, commands the base, navigates, compacts, or places.

Gazebo entity truth and the temporary exact-contact sensors make this an
engineering recovery tool only.  It must not be imported by the competition
mission or treated as competition-valid grasp evidence.
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
)
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    GuardResult,
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
    _angle_error,
    _book_transform,
    _coherent_scene,
    _emit,
    _fresh_cage_gate,
)
from erc_phase1_solution.live_rigid_palm_post_slide_transport_probe import (
    _scene_with_geometry,
    _target_other_book_collision,
)
from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (
    _inflated_ordered_corners,
    _moving_arm_environment_collision,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    EXPECTED_RELEASED_BASE_POSE,
    BookSnapshot,
    _load_runtime,
    quaternion_distance,
    rotation_matrix_distance,
    world_hand_pose,
)
from erc_phase1_solution.live_rigid_palm_tangent_slide_probe import (
    HAND_ENDPOINT_POSITION_LIMIT_M,
    HAND_ENDPOINT_ROTATION_LIMIT_RAD,
    OPEN_APERTURE_M,
    RECAGE_APERTURE_M,
    RESUME_FUTURE_TOLERANCE_S,
    RESUME_SCENE_AGE_LIMIT_S,
    STABILITY_SAMPLE_COUNT,
    StageObservation,
    TangentSlideFailure,
    _dense_arm_segment,
    _execute_aperture_stage,
    _fresh_palm_gate,
    _has_unexpected_contact,
    _probe_node_type,
    _proper_transform,
    _stage_observation,
    _tool_robot_collision,
    endpoint_book_hand_transforms,
    preflight_aperture_leg,
    recage_stage_targets,
    stage_stability_guard,
    tangent_stage_guard,
)
from erc_phase1_solution.rigid_palm_live_preflight import (
    _measured_finger_transforms_relative_to_palm,
    _measured_robot_state,
    _node_time_seconds,
    _require_stable_measured_finger_geometry,
    _require_stable_preflight_inputs,
    conservative_tool_vertex_displacement,
)
from erc_phase1_solution.rigid_palm_preflight import (
    GripperSample,
    PreflightResult,
    ProbePhase,
    preflight_gripper_sweep,
)


RECOVERY_EVENT = 'rigid_palm_partial_tangent_reverse_recovery'

# Recorded settled state after the first +5 mm local-z tangent stage.  These
# values authorize only this one recovery checkpoint; they are not a reusable
# production trajectory.
EXPECTED_PARTIAL_ARM = np.asarray(
    [
        0.35,
        0.2968485642,
        0.5454354018,
        0.5471264568,
        -1.7999864487,
        1.3225042704,
        0.8410165557,
        -1.4308528438,
    ],
    dtype=float,
)
EXPECTED_PREINSERT_ARM = np.asarray(
    [
        0.3499999999982399,
        0.29737968776292767,
        0.5463611395968271,
        0.5454457923660042,
        -1.810538006913063,
        1.311954109279404,
        0.8428201001259721,
        -1.415712469932009,
    ],
    dtype=float,
)
EXPECTED_PARTIAL_BOOK_POSITION = np.asarray(
    [2.6908244593, -0.1476023926, 1.5769525266], dtype=float
)
EXPECTED_PARTIAL_BOOK_QUATERNION = np.asarray(
    [7.85257e-05, 0.7075192910, 2.23366e-05, 0.7066940258],
    dtype=float,
)
EXPECTED_PARTIAL_BOOK_MAXIMUM_X_M = 2.7709714847
EXPECTED_PARTIAL_HAND_FROM_BOOK_CENTER = np.asarray(
    [0.0784286376, -0.0008640196, 0.0589698569], dtype=float
)

REVERSE_LOCAL_Z_M = -0.00496108412
REVERSE_DURATION_S = 0.80
REVERSE_IK_POSITION_TOLERANCE_M = 0.00020
REVERSE_IK_ORIENTATION_TOLERANCE_RAD = 0.00050
REVERSE_IK_MAXIMUM_JOINT_STEP_RAD = 0.030
REVERSE_LOCAL_OFF_AXIS_LIMIT_M = 0.00010

RESUME_ARM_TOLERANCE_RAD = 0.0030
RESUME_BOOK_HORIZONTAL_TOLERANCE_M = 0.0015
RESUME_BOOK_VERTICAL_TOLERANCE_M = 0.0010
RESUME_BOOK_MAXIMUM_X_TOLERANCE_M = 0.0015
RESUME_BOOK_ROTATION_TOLERANCE_RAD = math.radians(0.6)
RESUME_BASE_POSITION_TOLERANCE_M = 0.0010
RESUME_BASE_YAW_TOLERANCE_RAD = 0.0010
RESUME_APERTURE_TOLERANCE_M = 0.00020
RESUME_RELATIVE_POSITION_TOLERANCE_M = np.asarray(
    [0.0030, 0.0025, 0.0020], dtype=float
)
RESUME_RELATIVE_ROTATION_TOLERANCE_RAD = math.radians(1.0)
EXPECTED_PARTIAL_HAND_FROM_BOOK_ROTATION = np.diag([-1.0, -1.0, 1.0])

PREFLIGHT_START_ARM_TOLERANCE_RAD = 0.0020
PREFLIGHT_TARGET_HYPOTHESIS_STEP_M = 0.00050
PREFLIGHT_TARGET_OFF_AXIS_LIMIT_M = 0.00040
PREFLIGHT_MAXIMUM_HYPOTHESES = 17

REVERSE_HAND_PROGRESS_RANGE_M = (0.00450, 0.00540)
REVERSE_BOOK_OUTWARD_RANGE_M = (-0.00025, 0.00565)
REVERSE_BOOK_OFF_AXIS_LIMIT_M = 0.00080
REVERSE_BOOK_ROTATION_LIMIT_RAD = math.radians(0.5)
REVERSE_BOOK_BOUNDS_MISMATCH_LIMIT_M = 0.00080
REVERSE_ATTACHED_MISMATCH_LIMIT_M = 0.00080
REVERSE_STATIC_MOTION_LIMIT_M = 0.00080
REVERSE_SUPPORT_GAIN_RANGE_M = (-0.00080, 0.00565)
REVERSE_MINIMUM_SHELF_OVERLAP_M = 0.008
START_UNCHANGED_BOOK_LIMIT_M = 0.00050
START_UNCHANGED_BOOK_ROTATION_LIMIT_RAD = math.radians(0.35)


@dataclass(frozen=True)
class ReverseWaypoint:
    """The dynamically solved sole arm endpoint."""

    positions: np.ndarray
    target_pose: np.ndarray
    local_z_m: float


@dataclass(frozen=True)
class HypothesisGeometrySample:
    """One dense reverse state under one target-follow fraction."""

    positions: np.ndarray
    aperture_m: float
    transforms: Mapping[str, np.ndarray]
    target_corners: np.ndarray


@dataclass(frozen=True)
class ReverseHypothesisPreflight:
    """Static, intermediate-slip, and rigid-follow preflight evidence."""

    safe: bool
    reason: str
    hypothesis_count: int
    route_sample_count: int
    static_result: PreflightResult
    following_result: PreflightResult


@dataclass(frozen=True)
class ReverseOutcome:
    """Physical classification of the single reverse endpoint."""

    safe: bool
    reason: str
    mode: str | None
    metrics: dict[str, float]


def _finite_vector(
    value: Sequence[float], length: int, label: str
) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{label} must contain {length} finite values')
    return vector


def _outward_world(base: Sequence[float]) -> np.ndarray:
    pose = _finite_vector(base, 3, 'base pose')
    return np.asarray(
        [-math.cos(float(pose[2])), -math.sin(float(pose[2])), 0.0],
        dtype=float,
    )


def classify_partial_tangent_resume(
    book: BookSnapshot,
    base: Sequence[float],
    arm: Sequence[float],
    hand_base: Sequence[Sequence[float]],
    *,
    aperture_m: float,
    observed_at: float,
    reference_time: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Accept only the recorded settled 47 mm, first-insertion checkpoint."""

    try:
        base_pose = _finite_vector(base, 3, 'base pose')
        arm_state = _finite_vector(
            arm, len(EXPECTED_PARTIAL_ARM), 'arm state'
        )
        hand = _proper_transform(hand_base, 'base-frame hand pose')
        aperture = float(aperture_m)
        stamp = float(observed_at)
        now = float(reference_time)
        if not np.all(np.isfinite((aperture, stamp, now))):
            raise ValueError('resume scalars must be finite')
        hand_world = world_hand_pose(base_pose, hand)
        relative = np.linalg.inv(hand_world) @ _book_transform(book)
        relative_position_error = np.abs(
            relative[:3, 3] - EXPECTED_PARTIAL_HAND_FROM_BOOK_CENTER
        )
        relative_rotation_error = rotation_matrix_distance(
            relative[:3, :3],
            EXPECTED_PARTIAL_HAND_FROM_BOOK_ROTATION,
        )
        book_horizontal_error = float(np.linalg.norm(
            book.position[:2] - EXPECTED_PARTIAL_BOOK_POSITION[:2]
        ))
        book_vertical_error = abs(float(
            book.position[2] - EXPECTED_PARTIAL_BOOK_POSITION[2]
        ))
        maximum_x_error = abs(float(
            book.maximum[0] - EXPECTED_PARTIAL_BOOK_MAXIMUM_X_M
        ))
        book_rotation_error = quaternion_distance(
            book.quaternion, EXPECTED_PARTIAL_BOOK_QUATERNION
        )
        arm_error = float(np.max(np.abs(
            arm_state - EXPECTED_PARTIAL_ARM
        )))
        base_position_error = float(np.linalg.norm(
            base_pose[:2] - EXPECTED_RELEASED_BASE_POSE[:2]
        ))
        base_yaw_error = _angle_error(
            float(base_pose[2]), float(EXPECTED_RELEASED_BASE_POSE[2])
        )
        aperture_error = abs(aperture - OPEN_APERTURE_M)
        shelf_overlap = float(book.maximum[0] - SHELF_FRONT_X_M)
        scene_age = now - stamp
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return GuardResult(False, 'invalid_partial_state', {})

    metrics = {
        'book_horizontal_error_m': book_horizontal_error,
        'book_vertical_error_m': book_vertical_error,
        'book_maximum_x_error_m': maximum_x_error,
        'book_rotation_error_rad': book_rotation_error,
        'arm_error_rad': arm_error,
        'base_position_error_m': base_position_error,
        'base_yaw_error_rad': base_yaw_error,
        'aperture_error_m': aperture_error,
        'shelf_overlap_m': shelf_overlap,
        'scene_age_s': scene_age,
        'relative_x_error_m': float(relative_position_error[0]),
        'relative_y_error_m': float(relative_position_error[1]),
        'relative_z_error_m': float(relative_position_error[2]),
        'relative_rotation_error_rad': relative_rotation_error,
    }
    checks = (
        (scene_age <= RESUME_SCENE_AGE_LIMIT_S, 'scene_stale'),
        (scene_age >= -RESUME_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (
            book_horizontal_error <= RESUME_BOOK_HORIZONTAL_TOLERANCE_M,
            'wrong_partial_book_position',
        ),
        (
            book_vertical_error <= RESUME_BOOK_VERTICAL_TOLERANCE_M,
            'wrong_partial_book_height',
        ),
        (
            maximum_x_error <= RESUME_BOOK_MAXIMUM_X_TOLERANCE_M,
            'wrong_partial_shelf_overlap',
        ),
        (
            book_rotation_error <= RESUME_BOOK_ROTATION_TOLERANCE_RAD,
            'wrong_partial_book_rotation',
        ),
        (arm_error <= RESUME_ARM_TOLERANCE_RAD, 'wrong_partial_arm'),
        (
            base_position_error <= RESUME_BASE_POSITION_TOLERANCE_M,
            'base_moved',
        ),
        (
            base_yaw_error <= RESUME_BASE_YAW_TOLERANCE_RAD,
            'base_rotated',
        ),
        (
            aperture_error <= RESUME_APERTURE_TOLERANCE_M,
            'wrong_aperture',
        ),
        (
            shelf_overlap >= REVERSE_MINIMUM_SHELF_OVERLAP_M,
            'insufficient_shelf_overlap',
        ),
        (
            bool(np.all(
                relative_position_error
                <= RESUME_RELATIVE_POSITION_TOLERANCE_M
            )),
            'wrong_book_hand_translation',
        ),
        (
            relative_rotation_error
            <= RESUME_RELATIVE_ROTATION_TOLERANCE_RAD,
            'wrong_book_hand_rotation',
        ),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def solve_reverse_waypoint(
    chain: Any,
    measured_arm: Sequence[float],
) -> ReverseWaypoint:
    """Validate and return the exact preceding executed arm waypoint."""

    start = _finite_vector(
        measured_arm, len(EXPECTED_PARTIAL_ARM), 'measured arm'
    )
    if (
        np.asarray(chain.lower).shape != start.shape
        or np.asarray(chain.upper).shape != start.shape
        or np.any(start < np.asarray(chain.lower, dtype=float) - 1e-9)
        or np.any(start > np.asarray(chain.upper, dtype=float) + 1e-9)
    ):
        raise ValueError('measured arm is outside its hard limits')
    if float(np.max(np.abs(start - EXPECTED_PARTIAL_ARM))) > (
        RESUME_ARM_TOLERANCE_RAD
    ):
        raise ValueError('measured arm is not the partial tangent checkpoint')
    reference = _proper_transform(
        chain.forward(start), 'measured partial hand pose'
    )
    positions = _finite_vector(
        EXPECTED_PREINSERT_ARM,
        len(start),
        'known pre-insertion arm waypoint',
    )
    target = _proper_transform(
        chain.forward(positions), 'known pre-insertion hand pose'
    )
    maximum_step = float(np.max(np.abs(positions[1:] - start[1:])))
    if maximum_step > REVERSE_IK_MAXIMUM_JOINT_STEP_RAD:
        raise RuntimeError('known reverse waypoint left the local branch')
    local_delta = (
        reference[:3, :3].T
        @ (target[:3, 3] - reference[:3, 3])
    )
    rotation_delta = rotation_matrix_distance(
        reference[:3, :3], target[:3, :3]
    )
    if (
        float(np.linalg.norm(local_delta[:2]))
        > REVERSE_LOCAL_OFF_AXIS_LIMIT_M
        or abs(float(local_delta[2] - REVERSE_LOCAL_Z_M))
        > REVERSE_IK_POSITION_TOLERANCE_M
        or rotation_delta > REVERSE_IK_ORIENTATION_TOLERANCE_RAD
    ):
        raise RuntimeError(
            'known pre-insertion waypoint is not the audited local-z reverse'
        )
    return ReverseWaypoint(positions, target, REVERSE_LOCAL_Z_M)


def target_follow_hypothesis_fractions(
    maximum_attached_corner_motion_m: float,
    *,
    maximum_gap_m: float = PREFLIGHT_TARGET_HYPOTHESIS_STEP_M,
) -> tuple[float, ...]:
    """Densely sample every slip ratio from static (0) to attached (1)."""

    motion = float(maximum_attached_corner_motion_m)
    gap = float(maximum_gap_m)
    if (
        not math.isfinite(motion)
        or motion < 0.0
        or not math.isfinite(gap)
        or gap <= 0.0
    ):
        raise ValueError('hypothesis motion and gap must be finite')
    count = max(1, int(math.ceil(motion / gap)))
    if count + 1 > PREFLIGHT_MAXIMUM_HYPOTHESES:
        raise ValueError('target-slip hypothesis cap was exceeded')
    return tuple(float(value) for value in np.linspace(0.0, 1.0, count + 1))


def build_target_hypothesis_routes(
    *,
    scene: Any,
    node: Any,
    dense_arm: Sequence[Any],
) -> tuple[tuple[float, tuple[HypothesisGeometrySample, ...]], ...]:
    """Attach a static-to-following target continuum to one dense arm leg."""

    dense = tuple(dense_arm)
    if not dense:
        raise ValueError('dense reverse arm route is empty')
    observed = np.asarray(
        scene.books[TARGET_BOOK_MODEL].corners, dtype=float
    )
    if observed.shape != (8, 3) or not np.all(np.isfinite(observed)):
        raise ValueError('live target OBB is malformed')
    base = _proper_transform(scene.base_transform, 'scene base transform')
    first_hand = base @ _proper_transform(
        node.chain.forward(dense[0].positions), 'first hand pose'
    )
    corners_in_hand = (
        (observed - first_hand[:3, 3]) @ first_hand[:3, :3]
    )

    attached_corners: list[np.ndarray] = []
    for sample in dense:
        hand = base @ _proper_transform(
            node.chain.forward(sample.positions), 'candidate hand pose'
        )
        attached_corners.append(
            corners_in_hand @ hand[:3, :3].T + hand[:3, 3]
        )
    maximum_motion = float(np.max(np.linalg.norm(
        attached_corners[-1] - observed, axis=1
    )))
    fractions = target_follow_hypothesis_fractions(maximum_motion)
    routes = []
    for fraction in fractions:
        samples = tuple(
            HypothesisGeometrySample(
                positions=np.asarray(sample.positions, dtype=float).copy(),
                aperture_m=float(sample.aperture_m),
                transforms=sample.transforms,
                target_corners=(
                    observed
                    + fraction * (candidate - observed)
                ),
            )
            for sample, candidate in zip(dense, attached_corners)
        )
        routes.append((fraction, samples))
    return tuple(routes)


def _evaluate_reverse_hypothesis(
    *,
    node: Any,
    environment: Any,
    scene: Any,
    samples: Sequence[HypothesisGeometrySample],
    reference_time: float,
) -> PreflightResult:
    """Full-scene preflight for one fixed target-follow fraction."""

    dense = tuple(samples)
    if not dense:
        return PreflightResult(False, 'missing_samples', 'route is empty')
    observed = np.asarray(
        scene.books[TARGET_BOOK_MODEL].corners, dtype=float
    )
    base = np.asarray(scene.base_transform, dtype=float)
    inverse_base = np.linalg.inv(base)
    start_center = np.mean(observed, axis=0)
    base_yaw = math.atan2(base[1, 0], base[0, 0])
    outward = _outward_world([base[0, 3], base[1, 3], base_yaw])
    previous: HypothesisGeometrySample | None = None
    for index, sample in enumerate(dense):
        if previous is not None:
            tool_step = conservative_tool_vertex_displacement(
                environment.model,
                previous.transforms,
                sample.transforms,
            )
            target_step = float(np.max(np.linalg.norm(
                sample.target_corners - previous.target_corners, axis=1
            )))
            limit = float(environment.config.max_vertex_step_m) + 1e-12
            if tool_step > limit or target_step > limit:
                return PreflightResult(
                    False,
                    'undersampled_reverse_route',
                    f'sample {index - 1}->{index} moves tool '
                    f'{tool_step:.9f} m and target {target_step:.9f} m',
                    sample_index=index,
                )
        center_delta = np.mean(sample.target_corners, axis=0) - start_center
        progress = float(np.dot(center_delta, outward))
        off_axis = float(np.linalg.norm(
            center_delta - progress * outward
        ))
        shelf_overlap = float(
            np.max(sample.target_corners[:, 0]) - SHELF_FRONT_X_M
        )
        if progress < -1e-6 or progress > 0.00550:
            return PreflightResult(
                False,
                'target_hypothesis_wrong_direction',
                f'sample {index} target outward progress is '
                f'{progress:.9f} m',
                sample_index=index,
            )
        if off_axis > PREFLIGHT_TARGET_OFF_AXIS_LIMIT_M:
            return PreflightResult(
                False,
                'target_hypothesis_off_axis',
                f'sample {index} target off-axis motion is '
                f'{off_axis:.9f} m',
                sample_index=index,
            )
        if shelf_overlap < REVERSE_MINIMUM_SHELF_OVERLAP_M:
            return PreflightResult(
                False,
                'target_shelf_support_lost',
                f'sample {index} leaves only '
                f'{shelf_overlap:.9f} m shelf overlap',
                sample_index=index,
                obstacle='shelf',
            )

        candidate_scene = _scene_with_geometry(
            scene, base, sample.target_corners
        )
        self_collision = node._robot_self_collision(sample.positions)
        if self_collision is not None:
            return PreflightResult(
                False,
                'robot_self_collision',
                f'sample {index}: {self_collision}',
                sample_index=index,
            )
        padded_target = _inflated_ordered_corners(
            sample.target_corners, float(node.carried_book_padding)
        )
        target_base = (
            padded_target @ inverse_base[:3, :3].T
            + inverse_base[:3, 3]
        )
        payload_robot = node._carried_robot_collision(
            sample.positions, target_base
        )
        if payload_robot is not None:
            return PreflightResult(
                False,
                'target_robot_collision',
                f'sample {index}: {payload_robot}',
                sample_index=index,
                obstacle=payload_robot,
            )
        arm_environment = _moving_arm_environment_collision(
            node,
            candidate_scene,
            sample.positions,
            padding_m=environment.config.collision_padding_m,
        )
        if arm_environment is not None:
            return PreflightResult(
                False,
                'arm_environment_collision',
                f'sample {index}: {arm_environment}',
                sample_index=index,
                link=arm_environment[0],
                obstacle=arm_environment[1],
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
                False,
                'tool_robot_collision',
                f'sample {index}: {tool_robot}',
                sample_index=index,
                link=tool_robot[0],
                obstacle=tool_robot[1],
            )
        other_book = _target_other_book_collision(
            sample.target_corners,
            candidate_scene,
            float(environment.config.collision_padding_m),
        )
        if other_book is not None:
            return PreflightResult(
                False,
                'target_other_book_collision',
                f'sample {index}: {other_book}',
                sample_index=index,
                obstacle=other_book,
            )
        gripper = preflight_gripper_sweep(
            model=environment.model,
            samples=(GripperSample(
                transforms=sample.transforms,
                measured_aperture_m=sample.aperture_m,
                expected_aperture_m=OPEN_APERTURE_M,
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
                False,
                f'gripper_{gripper.code}',
                f'sample {index}: {gripper.detail}',
                sample_index=index,
                link=gripper.link,
                obstacle=gripper.obstacle,
            )
        previous = sample
    return PreflightResult(
        True,
        'clear',
        f'{len(dense)} reverse states passed',
    )


def preflight_reverse_hypotheses(
    *,
    node: Any,
    environment: Any,
    expected_start_arm: Sequence[float],
    target_arm: Sequence[float],
) -> ReverseHypothesisPreflight:
    """Preflight static, rigid-following, and bounded-slip target routes."""

    failed = PreflightResult(False, 'not_evaluated', 'not evaluated')
    try:
        expected = _finite_vector(
            expected_start_arm,
            len(EXPECTED_PARTIAL_ARM),
            'expected reverse start arm',
        )
        target = _finite_vector(
            target_arm,
            len(EXPECTED_PARTIAL_ARM),
            'reverse target arm',
        )
        before = _measured_robot_state(node)
        if (
            before.positions.shape != expected.shape
            or np.max(np.abs(before.positions - expected))
            > PREFLIGHT_START_ARM_TOLERANCE_RAD
            or abs(before.aperture_m - OPEN_APERTURE_M)
            > RESUME_APERTURE_TOLERANCE_M
        ):
            raise ValueError('measured reverse start changed before preflight')
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
        routes = build_target_hypothesis_routes(
            scene=scene, node=node, dense_arm=dense
        )
        results: list[PreflightResult] = []
        for _, samples in routes:
            result = _evaluate_reverse_hypothesis(
                node=node,
                environment=environment,
                scene=scene,
                samples=samples,
                reference_time=reference_time,
            )
            results.append(result)
            if not result.safe:
                return ReverseHypothesisPreflight(
                    False,
                    f'target hypothesis {len(results) - 1} failed: '
                    f'{result.code}: {result.detail}',
                    len(routes),
                    len(dense),
                    results[0],
                    results[-1] if len(results) == len(routes) else failed,
                )
        after_scene = environment._read_scene()
        after = _measured_robot_state(node)
        after_measured = _measured_finger_transforms_relative_to_palm(
            node, after_scene, after
        )
        _require_stable_measured_finger_geometry(
            environment.model, measured, after_measured
        )
        _require_stable_preflight_inputs(
            scene, after_scene, before, after
        )
        return ReverseHypothesisPreflight(
            True,
            'ok',
            len(routes),
            len(dense),
            results[0],
            results[-1],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return ReverseHypothesisPreflight(
            False,
            str(error),
            0,
            0,
            failed,
            failed,
        )


def recovery_start_unchanged_guard(
    reference: StageObservation,
    current: StageObservation,
    *,
    reference_time: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Prove the state did not change during the expensive dual preflight."""

    result = classify_partial_tangent_resume(
        current.book,
        current.base,
        current.arm,
        current.hand_base,
        aperture_m=current.aperture_m,
        observed_at=current.observed_at,
        reference_time=reference_time,
        palm_contact=palm_contact,
        unexpected_contacts=unexpected_contacts,
    )
    if not result.safe:
        return result
    book_motion = float(np.linalg.norm(
        current.book.position - reference.book.position
    ))
    book_rotation = quaternion_distance(
        current.book.quaternion, reference.book.quaternion
    )
    arm_motion = float(np.max(np.abs(current.arm - reference.arm)))
    base_motion = float(np.linalg.norm(
        current.base[:2] - reference.base[:2]
    ))
    base_yaw = _angle_error(current.base[2], reference.base[2])
    aperture_motion = abs(current.aperture_m - reference.aperture_m)
    metrics = dict(result.metrics)
    metrics.update({
        'preflight_book_motion_m': book_motion,
        'preflight_book_rotation_rad': book_rotation,
        'preflight_arm_motion_rad': arm_motion,
        'preflight_base_motion_m': base_motion,
        'preflight_base_yaw_motion_rad': base_yaw,
        'preflight_aperture_motion_m': aperture_motion,
    })
    checks = (
        (
            book_motion <= START_UNCHANGED_BOOK_LIMIT_M,
            'book_changed_during_preflight',
        ),
        (
            book_rotation <= START_UNCHANGED_BOOK_ROTATION_LIMIT_RAD,
            'book_rotated_during_preflight',
        ),
        (
            arm_motion <= PREFLIGHT_START_ARM_TOLERANCE_RAD,
            'arm_changed_during_preflight',
        ),
        (
            base_motion <= BASE_HOLD_LIMIT_M,
            'base_changed_during_preflight',
        ),
        (
            base_yaw <= BASE_YAW_HOLD_LIMIT_RAD,
            'base_rotated_during_preflight',
        ),
        (
            aperture_motion <= RESUME_APERTURE_TOLERANCE_M,
            'aperture_changed_during_preflight',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def reverse_endpoint_guard(
    reference: StageObservation,
    current: StageObservation,
    *,
    expected_arm: Sequence[float],
    expected_hand_base: Sequence[Sequence[float]],
    reference_time: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> ReverseOutcome:
    """Accept static, attached, or bounded-slip outcomes of the reverse."""

    try:
        expected_q = _finite_vector(
            expected_arm, len(reference.arm), 'expected reverse arm'
        )
        expected_hand = _proper_transform(
            expected_hand_base, 'expected reverse hand'
        )
        now = float(reference_time)
        if not math.isfinite(now):
            raise ValueError('reference time is not finite')
        reference_hand_world = world_hand_pose(
            reference.base, reference.hand_base
        )
        current_hand_world = world_hand_pose(
            current.base, current.hand_base
        )
        outward = _outward_world(reference.base)
        hand_delta = (
            current_hand_world[:3, 3] - reference_hand_world[:3, 3]
        )
        book_delta = current.book.position - reference.book.position
        hand_progress = float(np.dot(hand_delta, outward))
        book_progress = float(np.dot(book_delta, outward))
        book_off_axis = float(np.linalg.norm(
            book_delta - book_progress * outward
        ))
        attached_mismatch = float(np.linalg.norm(book_delta - hand_delta))
        static_motion = float(np.linalg.norm(book_delta))
        support_gain = hand_progress - book_progress
        book_rotation = quaternion_distance(
            current.book.quaternion, reference.book.quaternion
        )
        minimum_mismatch = float(np.max(np.abs(
            (current.book.minimum - reference.book.minimum) - book_delta
        )))
        maximum_mismatch = float(np.max(np.abs(
            (current.book.maximum - reference.book.maximum) - book_delta
        )))
        shelf_overlap = float(current.book.maximum[0] - SHELF_FRONT_X_M)
        base_error = float(np.linalg.norm(
            current.base[:2] - reference.base[:2]
        ))
        base_yaw_error = _angle_error(
            current.base[2], reference.base[2]
        )
        arm_error = float(np.max(np.abs(current.arm - expected_q)))
        hand_position_error = float(np.linalg.norm(
            current.hand_base[:3, 3] - expected_hand[:3, 3]
        ))
        hand_rotation_error = rotation_matrix_distance(
            current.hand_base[:3, :3], expected_hand[:3, :3]
        )
        aperture_error = abs(current.aperture_m - OPEN_APERTURE_M)
        scene_age = now - current.observed_at
    except (AttributeError, TypeError, ValueError):
        return ReverseOutcome(False, 'invalid_observation', None, {})

    if attached_mismatch <= REVERSE_ATTACHED_MISMATCH_LIMIT_M:
        mode = 'following'
    elif static_motion <= REVERSE_STATIC_MOTION_LIMIT_M:
        mode = 'static'
    else:
        mode = 'bounded_slip'
    metrics = {
        'hand_outward_progress_m': hand_progress,
        'book_outward_progress_m': book_progress,
        'book_off_axis_motion_m': book_off_axis,
        'book_hand_translation_mismatch_m': attached_mismatch,
        'book_static_motion_m': static_motion,
        'palm_relative_support_gain_m': support_gain,
        'book_rotation_rad': book_rotation,
        'book_minimum_bounds_mismatch_m': minimum_mismatch,
        'book_maximum_bounds_mismatch_m': maximum_mismatch,
        'shelf_overlap_m': shelf_overlap,
        'base_hold_error_m': base_error,
        'base_yaw_hold_error_rad': base_yaw_error,
        'arm_endpoint_error_rad': arm_error,
        'hand_endpoint_position_error_m': hand_position_error,
        'hand_endpoint_rotation_error_rad': hand_rotation_error,
        'aperture_error_m': aperture_error,
        'scene_age_s': scene_age,
    }
    checks = (
        (scene_age <= RESUME_SCENE_AGE_LIMIT_S, 'scene_stale'),
        (scene_age >= -RESUME_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (
            REVERSE_HAND_PROGRESS_RANGE_M[0]
            <= hand_progress
            <= REVERSE_HAND_PROGRESS_RANGE_M[1],
            'reverse_hand_progress',
        ),
        (
            REVERSE_BOOK_OUTWARD_RANGE_M[0]
            <= book_progress
            <= REVERSE_BOOK_OUTWARD_RANGE_M[1],
            'target_reverse_progress',
        ),
        (
            book_off_axis <= REVERSE_BOOK_OFF_AXIS_LIMIT_M,
            'target_reverse_off_axis',
        ),
        (
            REVERSE_SUPPORT_GAIN_RANGE_M[0]
            <= support_gain
            <= REVERSE_SUPPORT_GAIN_RANGE_M[1],
            'target_support_gain_outside_envelope',
        ),
        (
            book_rotation <= REVERSE_BOOK_ROTATION_LIMIT_RAD,
            'target_rotated',
        ),
        (
            minimum_mismatch <= REVERSE_BOOK_BOUNDS_MISMATCH_LIMIT_M,
            'target_minimum_bounds_changed',
        ),
        (
            maximum_mismatch <= REVERSE_BOOK_BOUNDS_MISMATCH_LIMIT_M,
            'target_maximum_bounds_changed',
        ),
        (
            shelf_overlap >= REVERSE_MINIMUM_SHELF_OVERLAP_M,
            'shelf_overlap_lost',
        ),
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
            'aperture_changed',
        ),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
    )
    for accepted, reason in checks:
        if not accepted:
            return ReverseOutcome(False, reason, mode, metrics)
    return ReverseOutcome(True, 'ok', mode, metrics)


def _collect_stable_reverse_endpoint(
    *,
    node: Any,
    runtime: SimpleNamespace,
    reference: StageObservation,
    expected_arm: Sequence[float],
    expected_hand_base: Sequence[Sequence[float]],
) -> tuple[StageObservation, ReverseOutcome, GuardResult]:
    """Collect a sim-time stability window and fresh reverse evidence."""

    observations: list[StageObservation] = []
    stamp = reference.observed_at
    while len(observations) < STABILITY_SAMPLE_COUNT:
        if observations and not node._wait_sim_duration(0.03):
            raise TangentSlideFailure(
                'reverse stability dwell was interrupted'
            )
        if not node.probe_exact_palm_contact(max_age=0.22):
            raise TangentSlideFailure(
                'exact target palm contact was lost after reverse'
            )
        sample = _stage_observation(node, runtime, newer_than=stamp)
        observations.append(sample)
        stamp = sample.observed_at
    stability = stage_stability_guard(observations)
    final = observations[-1]
    palm = node.probe_exact_palm_contact(max_age=0.22)
    outcome = reverse_endpoint_guard(
        reference,
        final,
        expected_arm=expected_arm,
        expected_hand_base=expected_hand_base,
        reference_time=_node_time_seconds(node),
        palm_contact=palm,
        unexpected_contacts=_has_unexpected_contact(node),
    )
    if not stability.safe:
        raise TangentSlideFailure(
            f'reverse stability gate failed: {stability.reason}'
        )
    if not outcome.safe:
        raise TangentSlideFailure(
            f'reverse endpoint gate failed: {outcome.reason}'
        )
    return final, outcome, stability


def _recage_start_guard(
    reference: StageObservation,
    current: StageObservation,
    *,
    reference_time: float,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Require the reverse endpoint to remain fixed during close preflight."""

    return tangent_stage_guard(
        reference,
        reference,
        current,
        expected_arm=reference.arm,
        expected_hand_base=reference.hand_base,
        expected_aperture_m=OPEN_APERTURE_M,
        reference_time=reference_time,
        palm_contact=palm_contact,
        unexpected_contacts=unexpected_contacts,
    )


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
        name='rigid-palm-partial-tangent-reverse-recovery',
        daemon=True,
    )
    thread.start()
    stage = 'created'
    reverse_dispatched = False
    try:
        deadline = time.monotonic() + 30.0
        while (
            len(node.joints) < len(EXPECTED_PARTIAL_ARM)
            or node.gripper_pub.get_subscription_count() < 1
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < len(EXPECTED_PARTIAL_ARM):
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

        _, _, palm = _fresh_palm_gate(node)
        reference = _stage_observation(node, runtime)
        resume = classify_partial_tangent_resume(
            reference.book,
            reference.base,
            reference.arm,
            reference.hand_base,
            aperture_m=reference.aperture_m,
            observed_at=reference.observed_at,
            reference_time=_node_time_seconds(node),
            palm_contact=palm,
            unexpected_contacts=_has_unexpected_contact(node),
        )
        if not resume.safe:
            raise RuntimeError(
                f'partial tangent resume rejected: {resume.reason}'
            )
        waypoint = solve_reverse_waypoint(node.chain, reference.arm)

        stage = 'dual_hypothesis_preflight'
        preflight = preflight_reverse_hypotheses(
            node=node,
            environment=runtime.environment_preflight,
            expected_start_arm=reference.arm,
            target_arm=waypoint.positions,
        )
        _emit(
            'partial_tangent_reverse_preflight',
            passed=preflight.safe,
            reason=preflight.reason,
            hypothesis_count=preflight.hypothesis_count,
            route_sample_count=preflight.route_sample_count,
            static_code=preflight.static_result.code,
            following_code=preflight.following_result.code,
            static_target_checked=True,
            following_target_checked=True,
            bounded_slip_checked=True,
        )
        if not preflight.safe:
            raise RuntimeError(
                f'reverse dual-hypothesis preflight rejected: '
                f'{preflight.reason}'
            )

        # The geometry sweep may take appreciable wall time at low real-time
        # factor.  Re-read everything and require fresh palm evidence before
        # this executable's sole arm command.
        left, right, palm = _fresh_palm_gate(node)
        current = _stage_observation(
            node, runtime, newer_than=reference.observed_at
        )
        unchanged = recovery_start_unchanged_guard(
            reference,
            current,
            reference_time=_node_time_seconds(node),
            palm_contact=palm,
            unexpected_contacts=_has_unexpected_contact(node),
        )
        if not unchanged.safe:
            raise RuntimeError(
                f'post-preflight start gate failed: {unchanged.reason}'
            )
        _emit(
            'partial_tangent_resume_verified',
            measured_aperture_m=current.aperture_m,
            measured_arm=current.arm.tolist(),
            reverse_target_arm=waypoint.positions.tolist(),
            bilateral_present=bool(left and right),
            diagnostic_truth_and_contacts_only=True,
            **resume.metrics,
        )

        stage = 'single_5mm_reverse'
        node._tangent_palm_support_active = True
        legs = ((
            waypoint.positions,
            REVERSE_DURATION_S,
            'partial_tangent_reverse',
        ),)
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        reverse_dispatched = True
        moved, contact_loss = node._send_retained_arm_trajectory(
            goal, duration, legs, RECOVERY_EVENT
        )
        if not moved:
            reason = (
                'palm_contact_loss' if contact_loss else 'controller_failure'
            )
            raise TangentSlideFailure(f'single reverse failed: {reason}')
        endpoint = node._wait_for_retained_endpoint(
            waypoint.positions,
            command=RECOVERY_EVENT,
            phase='single_reverse',
            leg=1,
            arm_tolerance=ARM_HOLD_LIMIT_RAD,
        )
        if endpoint is None:
            raise TangentSlideFailure(
                'single reverse endpoint was not held'
            )
        reversed_state, outcome, stability = (
            _collect_stable_reverse_endpoint(
                node=node,
                runtime=runtime,
                reference=current,
                expected_arm=waypoint.positions,
                expected_hand_base=waypoint.target_pose,
            )
        )
        _emit(
            'partial_tangent_reverse_complete',
            passed=True,
            outcome_mode=outcome.mode,
            measured_arm=reversed_state.arm.tolist(),
            measured_aperture_m=reversed_state.aperture_m,
            no_second_arm_motion=True,
            **stability.metrics,
            **outcome.metrics,
        )

        stage = 'recage_complete_preflight'
        complete_close = preflight_aperture_leg(
            node=node,
            environment=runtime.environment_preflight,
            target_aperture_m=RECAGE_APERTURE_M,
        )
        _emit(
            'partial_tangent_recage_preflight',
            passed=complete_close.safe,
            code=complete_close.code,
            detail=complete_close.detail,
            static_target_checked=True,
        )
        if not complete_close.safe:
            raise RuntimeError(
                f'complete recage preflight rejected: '
                f'{complete_close.code}: {complete_close.detail}'
            )
        _, _, palm = _fresh_palm_gate(node)
        recage_reference = _stage_observation(
            node, runtime, newer_than=reversed_state.observed_at
        )
        close_start = _recage_start_guard(
            reversed_state,
            recage_reference,
            reference_time=_node_time_seconds(node),
            palm_contact=palm,
            unexpected_contacts=_has_unexpected_contact(node),
        )
        if not close_start.safe:
            raise RuntimeError(
                f'post-preflight recage start failed: {close_start.reason}'
            )

        stage = 'incremental_recage_to_30_3mm'
        current = recage_reference
        for target in recage_stage_targets():
            current = _execute_aperture_stage(
                node=node,
                runtime=runtime,
                reference=recage_reference,
                previous=current,
                target_aperture_m=target,
                phase='partial_tangent_recovery_recage',
                expected_arm=recage_reference.arm,
                expected_hand_base=recage_reference.hand_base,
            )

        left, right, palm = _fresh_cage_gate(node)
        final = _stage_observation(
            node, runtime, newer_than=current.observed_at
        )
        final_guard = tangent_stage_guard(
            recage_reference,
            current,
            final,
            expected_arm=recage_reference.arm,
            expected_hand_base=recage_reference.hand_base,
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
                f'final recovery cage failed: {final_guard.reason}'
            )
        hand_from_book, book_from_hand = endpoint_book_hand_transforms(
            final.book, final.base, final.hand_base
        )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=RECOVERY_EVENT,
            reverse_outcome_mode=outcome.mode,
            measured_aperture_m=final.aperture_m,
            book_position=final.book.position.tolist(),
            measured_endpoint_q=final.arm.tolist(),
            hand_T_book=hand_from_book.tolist(),
            book_T_hand=book_from_hand.tolist(),
            reverse_arm_commands=1,
            base_navigation_commanded=False,
            gripper_opening_commanded=False,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
            **final_guard.metrics,
        )
    except Exception as error:
        _emit(
            'result',
            passed=False,
            stage=stage,
            reason=f'{type(error).__name__}: {error}',
            reverse_dispatched=reverse_dispatched,
            automatic_arm_recovery_commanded=False,
            base_navigation_commanded=False,
            gripper_opening_commanded=False,
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
            'Diagnostic-only recovery from the partial 47 mm tangent state'
        )
    )
    parser.add_argument(
        '--confirm-diagnostic-reverse-and-recage',
        action='store_true',
        help=(
            'authorize one guarded 5 mm reverse and a separately '
            'preflighted incremental recage'
        ),
    )
    arguments = parser.parse_args()
    if not arguments.confirm_diagnostic_reverse_and_recage:
        parser.error(
            '--confirm-diagnostic-reverse-and-recage is required'
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
