#!/usr/bin/env python3
"""
Diagnostic-only supported compaction after post-slide base retreat.

This continuation deliberately consumes Gazebo truth and temporary exact
contact sensors, so it is engineering evidence and is never an authorization
for the competition mission.  It resumes only from a freshly observed,
stationary 30.3 mm left--palm--right cage whose target OBB is at least 20 mm
clear of the shelf.  The live target OBB is re-anchored in the measured hand
frame; no recorded joint vector, attachment transform, or cached production
route is accepted.

The existing production supported-staging and compact-goal solvers are used
only to propose joint-space routes from the measured state.  Each proposal is
then independently checked with the measured passive-finger geometry, the
attached target, every modeled robot link, self-collision, the shelf, and all
books.  The complete route and every individual controller leg are densely
preflighted before one continuous retained trajectory is sent.  During that
trajectory, fresh exact bilateral plus palm contact, aperture, unexpected
contacts, and base odometry are watched continuously.  The probe stops at the
compact cage; it never opens, navigates, or places the target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import threading
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    GuardResult,
    TARGET_BOOK_MODEL,
    _angle_error,
    _emit,
    _fresh_cage_gate,
    _probe_types as _transport_probe_types,
)
from erc_phase1_solution.live_rigid_palm_post_slide_transport_probe import (
    AttachmentAnchor,
    MINIMUM_SHELF_CLEARANCE_M,
    RECAGE_APERTURE_M,
    SCENE_AGE_LIMIT_S,
    SCENE_FUTURE_TOLERANCE_S,
    SupportAssessment,
    TransportGeometrySample,
    TransportObservation,
    _arm_geometry_samples,
    _attachment_errors,
    _evaluate_transport_geometry,
    _palm_local_triangles,
    _support_for_observation,
    _transport_observation,
    _unexpected_contact,
    capture_attachment_anchor,
    finite_palm_support_guard,
    predict_attached_corners,
    shelf_clearance_m,
)
from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (
    _inflated_ordered_corners,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    _load_runtime,
    quaternion_distance,
    rotation_matrix_distance,
)
from erc_phase1_solution.live_rigid_palm_tangent_slide_probe import (
    APERTURE_ENDPOINT_TOLERANCE_M,
)
from erc_phase1_solution.rigid_palm_live_preflight import (
    MOVING_GRIPPER_LINKS,
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


SUPPORTED_COMPACTION_EVENT = 'rigid_palm_supported_compaction'
MINIMUM_STATIONARY_DWELL_S = 0.10
RESUME_BASE_TRANSLATION_LIMIT_M = 0.00050
RESUME_BASE_YAW_LIMIT_RAD = 0.00080
RESUME_ARM_DRIFT_LIMIT_RAD = 0.0010
RESUME_HAND_TRANSLATION_LIMIT_M = 0.00075
RESUME_HAND_ROTATION_LIMIT_RAD = math.radians(0.30)
RESUME_BOOK_TRANSLATION_LIMIT_M = 0.00075
RESUME_BOOK_ROTATION_LIMIT_RAD = math.radians(0.40)
RESUME_CORNER_DRIFT_LIMIT_M = 0.0010
RESUME_ATTACHMENT_POSITION_LIMIT_M = 0.0020
RESUME_ATTACHMENT_ROTATION_LIMIT_RAD = math.radians(1.0)
RESUME_ATTACHMENT_CORNER_LIMIT_M = 0.0025

PREFLIGHT_START_ARM_LIMIT_RAD = 0.00050
PREFLIGHT_START_APERTURE_LIMIT_M = 0.00025
PREFLIGHT_ANCHOR_CORNER_LIMIT_M = 0.0025

WATCHDOG_BASE_TRANSLATION_LIMIT_M = 0.0010
WATCHDOG_BASE_YAW_LIMIT_RAD = 0.0015
WATCHDOG_CONTACT_MAX_AGE_S = 0.15

ENDPOINT_ARM_LIMIT_RAD = 0.0050
ENDPOINT_BASE_TRANSLATION_LIMIT_M = 0.0010
ENDPOINT_BASE_YAW_LIMIT_RAD = 0.0015
ENDPOINT_ATTACHMENT_POSITION_LIMIT_M = 0.0030
ENDPOINT_ATTACHMENT_ROTATION_LIMIT_RAD = math.radians(2.0)
ENDPOINT_ATTACHMENT_CORNER_LIMIT_M = 0.0035


@dataclass(frozen=True)
class SupportedCompactionPlan:
    """One measured-start route proposed by production-shaped helpers."""

    label: str
    positions: tuple[np.ndarray, ...]
    phases: tuple[str, ...]
    terminal_radius_m: float


@dataclass(frozen=True)
class CompactionPreflight:
    """Result of complete-route and repeated per-leg geometric proofs."""

    safe: bool
    code: str
    detail: str
    dense_samples: int = 0
    legs_checked: int = 0
    minimum_shelf_clearance_m: float = math.nan
    minimum_support_depth_margin_m: float = math.nan
    minimum_support_lateral_margin_m: float = math.nan


class SupportedCompactionFailure(RuntimeError):
    """Fail-closed terminal error for this diagnostic continuation."""


def measured_attached_envelope(
    anchor: AttachmentAnchor,
    padding_m: float,
) -> np.ndarray:
    """Inflate the measured OBB in its measured hand-frame orientation."""
    corners = np.asarray(anchor.corners_in_hand, dtype=float)
    padding = float(padding_m)
    if (
        corners.shape != (8, 3)
        or not np.all(np.isfinite(corners))
        or not math.isfinite(padding)
        or padding < 0.0
    ):
        raise ValueError('measured attachment envelope is malformed')
    return _inflated_ordered_corners(corners, padding)


def deep_cage_resume_guard(
    reference: TransportObservation,
    current: TransportObservation,
    anchor: AttachmentAnchor,
    support: SupportAssessment,
    *,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    navigation_idle: bool,
) -> GuardResult:
    """Prove that a shelf-clear post-retreat cage is fresh and stationary."""
    try:
        now = float(reference_time)
        age = now - float(current.observed_at)
        dwell = float(current.observed_at - reference.observed_at)
        aperture_error = abs(current.aperture_m - RECAGE_APERTURE_M)
        reference_aperture_error = abs(
            reference.aperture_m - RECAGE_APERTURE_M
        )
        base_translation = float(np.linalg.norm(
            current.base[:2] - reference.base[:2]
        ))
        base_yaw = _angle_error(current.base[2], reference.base[2])
        arm_drift = float(np.max(np.abs(current.arm - reference.arm)))
        hand_translation = float(np.linalg.norm(
            current.hand_world[:3, 3] - reference.hand_world[:3, 3]
        ))
        hand_rotation = rotation_matrix_distance(
            current.hand_world[:3, :3], reference.hand_world[:3, :3]
        )
        book_translation = float(np.linalg.norm(
            current.book.position - reference.book.position
        ))
        book_rotation = quaternion_distance(
            current.book.quaternion, reference.book.quaternion
        )
        corner_drift = float(np.max(np.linalg.norm(
            current.corners - reference.corners, axis=1
        )))
        attachment_position, attachment_rotation, attachment_corner = (
            _attachment_errors(anchor, current)
        )
        clearance = shelf_clearance_m(current.corners)
        if not math.isfinite(now):
            raise ValueError('reference time is non-finite')
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return GuardResult(False, 'invalid_observation', {})
    metrics = {
        'scene_age_s': age,
        'stationary_dwell_s': dwell,
        'measured_aperture_m': current.aperture_m,
        'aperture_error_m': aperture_error,
        'reference_aperture_error_m': reference_aperture_error,
        'shelf_clearance_m': clearance,
        'base_stationary_translation_m': base_translation,
        'base_stationary_yaw_rad': base_yaw,
        'arm_stationary_error_rad': arm_drift,
        'hand_stationary_translation_m': hand_translation,
        'hand_stationary_rotation_rad': hand_rotation,
        'book_stationary_translation_m': book_translation,
        'book_stationary_rotation_rad': book_rotation,
        'book_corner_stationary_error_m': corner_drift,
        'attachment_position_error_m': attachment_position,
        'attachment_rotation_error_rad': attachment_rotation,
        'attachment_corner_error_m': attachment_corner,
        'support_depth_margin_m': support.depth_margin_m,
        'support_lateral_margin_m': support.lateral_margin_m,
    }
    checks = (
        (age <= SCENE_AGE_LIMIT_S, 'scene_stale'),
        (age >= -SCENE_FUTURE_TOLERANCE_S, 'scene_from_future'),
        (dwell >= MINIMUM_STATIONARY_DWELL_S, 'stationary_dwell_too_short'),
        (bool(navigation_idle), 'navigation_still_active'),
        (bool(left_contact), 'left_target_contact_missing'),
        (bool(right_contact), 'right_target_contact_missing'),
        (bool(palm_contact), 'exact_target_palm_contact_missing'),
        (not bool(unexpected_contacts), 'unexpected_contact'),
        (
            reference_aperture_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'reference_cage_aperture_wrong',
        ),
        (
            aperture_error <= APERTURE_ENDPOINT_TOLERANCE_M,
            'wrong_cage_aperture',
        ),
        (
            clearance >= MINIMUM_SHELF_CLEARANCE_M,
            'target_not_shelf_clear',
        ),
        (support.safe, support.reason),
        (
            base_translation <= RESUME_BASE_TRANSLATION_LIMIT_M,
            'base_not_settled',
        ),
        (base_yaw <= RESUME_BASE_YAW_LIMIT_RAD, 'base_yaw_not_settled'),
        (arm_drift <= RESUME_ARM_DRIFT_LIMIT_RAD, 'arm_not_settled'),
        (
            hand_translation <= RESUME_HAND_TRANSLATION_LIMIT_M,
            'hand_not_settled',
        ),
        (
            hand_rotation <= RESUME_HAND_ROTATION_LIMIT_RAD,
            'hand_rotated_while_settling',
        ),
        (
            book_translation <= RESUME_BOOK_TRANSLATION_LIMIT_M,
            'book_not_settled',
        ),
        (
            book_rotation <= RESUME_BOOK_ROTATION_LIMIT_RAD,
            'book_rotated_while_settling',
        ),
        (
            corner_drift <= RESUME_CORNER_DRIFT_LIMIT_M,
            'book_obb_not_settled',
        ),
        (
            attachment_position <= RESUME_ATTACHMENT_POSITION_LIMIT_M,
            'attachment_position_changed',
        ),
        (
            attachment_rotation <= RESUME_ATTACHMENT_ROTATION_LIMIT_RAD,
            'attachment_rotation_changed',
        ),
        (
            attachment_corner <= RESUME_ATTACHMENT_CORNER_LIMIT_M,
            'attachment_obb_changed',
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def compaction_watchdog_reason(
    *,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
    measured_aperture_m: float,
    navigation_pose: Sequence[float] | None,
    reference_navigation_pose: Sequence[float],
    navigation_active: bool,
) -> str | None:
    """Return the first fail-closed fault during the retained trajectory."""
    try:
        aperture = float(measured_aperture_m)
        reference = np.asarray(reference_navigation_pose, dtype=float)
        pose = np.asarray(navigation_pose, dtype=float)
        if (
            reference.shape != (3,)
            or pose.shape != (3,)
            or not np.all(np.isfinite(reference))
            or not np.all(np.isfinite(pose))
            or not math.isfinite(aperture)
        ):
            return 'base_odometry_unavailable'
        translation = float(np.linalg.norm(pose[:2] - reference[:2]))
        yaw = _angle_error(float(pose[2]), float(reference[2]))
    except (TypeError, ValueError):
        return 'base_odometry_unavailable'
    checks = (
        (not bool(unexpected_contacts), 'unexpected_scored_contact'),
        (bool(left_contact), 'exact_left_target_contact_lost'),
        (bool(right_contact), 'exact_right_target_contact_lost'),
        (bool(palm_contact), 'exact_palm_target_contact_lost'),
        (not bool(navigation_active), 'navigation_became_active'),
        (
            abs(aperture - RECAGE_APERTURE_M)
            <= APERTURE_ENDPOINT_TOLERANCE_M,
            'cage_aperture_changed',
        ),
        (
            translation <= WATCHDOG_BASE_TRANSLATION_LIMIT_M,
            'base_moved_during_compaction',
        ),
        (yaw <= WATCHDOG_BASE_YAW_LIMIT_RAD, 'base_rotated_during_compaction'),
    )
    for accepted, reason in checks:
        if not accepted:
            return reason
    return None


def supported_compaction_candidates(
    node: Any,
    start_positions: Sequence[float],
    attached_corners_in_hand: Sequence[Sequence[float]],
) -> tuple[SupportedCompactionPlan, ...]:
    """Propose fresh direct and staged supported routes from measured state."""
    start = np.asarray(start_positions, dtype=float)
    attached = np.asarray(attached_corners_in_hand, dtype=float)
    if (
        start.shape != (8,)
        or not np.all(np.isfinite(start))
        or attached.shape != (8, 3)
        or not np.all(np.isfinite(attached))
    ):
        raise ValueError('supported compaction start geometry is malformed')

    proposals: list[SupportedCompactionPlan] = []

    def append_plan(
        label: str,
        sections: Sequence[tuple[Sequence[Sequence[float]], str]],
    ) -> None:
        previous = start.copy()
        route: list[np.ndarray] = []
        phases: list[str] = []
        for goals, phase in sections:
            requested = tuple(np.asarray(goal, dtype=float) for goal in goals)
            if not requested:
                continue
            try:
                planned = node._plan_carried_joint_route(
                    previous,
                    requested,
                    attached,
                    require_gravity_support=True,
                )
            except (RuntimeError, TypeError, ValueError):
                return
            if planned is None:
                return
            for position in planned:
                copied = np.asarray(position, dtype=float).copy()
                route.append(copied)
                phases.append(phase)
            previous = route[-1]
        if not route:
            return
        radius = float(node._carried_navigation_radius(route[-1], attached))
        limit = float(node.carried_navigation_radius_limit)
        if not math.isfinite(radius) or radius > limit:
            return
        proposals.append(SupportedCompactionPlan(
            label,
            tuple(route),
            tuple(phases),
            radius,
        ))

    try:
        direct_goals = tuple(node._supported_compact_goals(start))
        append_plan('direct_supported_compact', ((direct_goals, 'compact'),))
    except (RuntimeError, TypeError, ValueError):
        pass

    try:
        _, lowering, retraction = node._solve_supported_post_retreat_staging(
            start
        )
        staged_terminal = (
            np.asarray(retraction[-1], dtype=float)
            if retraction
            else np.asarray(lowering[-1], dtype=float)
        )
        staged_compact = tuple(node._supported_compact_goals(staged_terminal))
        append_plan(
            'supported_lower_retract_compact',
            (
                (lowering, 'supported_lowering'),
                (retraction, 'supported_retraction'),
                (staged_compact, 'compact'),
            ),
        )
    except (IndexError, RuntimeError, TypeError, ValueError):
        pass

    return tuple(proposals)


def _support_route_result(
    *,
    node: Any,
    environment: Any,
    samples: Sequence[TransportGeometrySample],
    outward_world: Sequence[float],
) -> tuple[PreflightResult, float, float, float]:
    """Check clearance and finite gravity support at every dense state."""
    palm_triangles = _palm_local_triangles(environment.model)
    minimum_clearance = math.inf
    minimum_depth = math.inf
    minimum_lateral = math.inf
    base = np.asarray(samples[0].base_transform, dtype=float)
    for index, sample in enumerate(samples):
        clearance = shelf_clearance_m(sample.target_corners)
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < MINIMUM_SHELF_CLEARANCE_M:
            return (
                PreflightResult(
                    False,
                    'shelf_clearance_lost',
                    f'dense sample {index} has only '
                    f'{clearance:.9f} m clearance',
                    sample_index=index,
                    obstacle='shelf',
                ),
                minimum_clearance,
                minimum_depth,
                minimum_lateral,
            )
        links = node.chain.link_transforms(sample.positions)
        if PALM_COLLISION_LINK not in links:
            return (
                PreflightResult(
                    False,
                    'palm_transform_missing',
                    f'dense sample {index} omitted the palm collision link',
                    sample_index=index,
                ),
                minimum_clearance,
                minimum_depth,
                minimum_lateral,
            )
        palm_world = base @ np.asarray(links[PALM_COLLISION_LINK], dtype=float)
        support = finite_palm_support_guard(
            palm_triangles,
            palm_world,
            sample.target_corners,
            outward_world,
            polygon_inset_m=max(
                float(environment.config.collision_padding_m),
                0.00075,
            ),
        )
        minimum_depth = min(minimum_depth, support.depth_margin_m)
        minimum_lateral = min(minimum_lateral, support.lateral_margin_m)
        if not support.safe:
            return (
                PreflightResult(
                    False,
                    f'support_{support.reason}',
                    f'dense sample {index}: {support.reason}',
                    sample_index=index,
                ),
                minimum_clearance,
                minimum_depth,
                minimum_lateral,
            )
        if not node._gravity_supported_transition_is_safe(
            sample.positions, sample.positions
        ):
            return (
                PreflightResult(
                    False,
                    'gravity_support_axis_lost',
                    f'dense sample {index} violates supported-carry attitude',
                    sample_index=index,
                ),
                minimum_clearance,
                minimum_depth,
                minimum_lateral,
            )
    return (
        PreflightResult(
            True,
            'clear',
            f'{len(samples)} dense states retain finite gravity support',
        ),
        minimum_clearance,
        minimum_depth,
        minimum_lateral,
    )


def preflight_supported_compaction(
    *,
    node: Any,
    environment: Any,
    anchor: AttachmentAnchor,
    start_positions: Sequence[float],
    plan: SupportedCompactionPlan,
    outward_world: Sequence[float],
) -> CompactionPreflight:
    """Measure once, then prove the whole route and every leg independently."""
    expected_start = np.asarray(start_positions, dtype=float)
    if not plan.positions:
        return CompactionPreflight(False, 'missing_route', 'route is empty')
    try:
        before = _measured_robot_state(node)
        if (
            before.positions.shape != expected_start.shape
            or float(np.max(np.abs(before.positions - expected_start)))
            > PREFLIGHT_START_ARM_LIMIT_RAD
            or abs(before.aperture_m - RECAGE_APERTURE_M)
            > PREFLIGHT_START_APERTURE_LIMIT_M
        ):
            raise ValueError(
                'measured compaction start changed before preflight'
            )
        scene = environment._read_scene()
        reference_time = _node_time_seconds(node)
        measured_relative = _measured_finger_transforms_relative_to_palm(
            node, scene, before
        )
        if set(measured_relative) != set(MOVING_GRIPPER_LINKS):
            raise ValueError('measured passive gripper geometry is incomplete')
        start_hand_world = (
            np.asarray(scene.base_transform, dtype=float)
            @ node.chain.forward(before.positions)
        )
        predicted_start = predict_attached_corners(anchor, start_hand_world)
        observed_start = np.asarray(
            scene.books[TARGET_BOOK_MODEL].corners, dtype=float
        )
        start_corner_error = float(np.max(np.linalg.norm(
            observed_start - predicted_start, axis=1
        )))
        if start_corner_error > PREFLIGHT_ANCHOR_CORNER_LIMIT_M:
            raise ValueError(
                'measured target no longer matches attachment anchor: '
                f'{start_corner_error:.9f} m'
            )

        full_samples = _arm_geometry_samples(
            node=node,
            environment=environment,
            scene=scene,
            start_positions=before.positions,
            targets=plan.positions,
            measured_relative=measured_relative,
            anchor=anchor,
        )
        full_geometry = _evaluate_transport_geometry(
            node=node,
            environment=environment,
            scene=scene,
            samples=full_samples,
            reference_time=reference_time,
            allow_initial_shelf_support=False,
            require_final_clearance=True,
        )
        if not full_geometry.safe:
            return CompactionPreflight(
                False,
                f'complete_{full_geometry.code}',
                full_geometry.detail,
                dense_samples=len(full_samples),
            )
        support_result, min_clearance, min_depth, min_lateral = (
            _support_route_result(
                node=node,
                environment=environment,
                samples=full_samples,
                outward_world=outward_world,
            )
        )
        if not support_result.safe:
            return CompactionPreflight(
                False,
                f'complete_{support_result.code}',
                support_result.detail,
                dense_samples=len(full_samples),
                minimum_shelf_clearance_m=min_clearance,
                minimum_support_depth_margin_m=min_depth,
                minimum_support_lateral_margin_m=min_lateral,
            )

        previous = before.positions
        repeated_samples = 0
        for index, target in enumerate(plan.positions):
            leg_samples = _arm_geometry_samples(
                node=node,
                environment=environment,
                scene=scene,
                start_positions=previous,
                targets=(target,),
                measured_relative=measured_relative,
                anchor=anchor,
            )
            repeated_samples += len(leg_samples)
            leg_geometry = _evaluate_transport_geometry(
                node=node,
                environment=environment,
                scene=scene,
                samples=leg_samples,
                reference_time=reference_time,
                allow_initial_shelf_support=False,
                require_final_clearance=True,
            )
            if not leg_geometry.safe:
                return CompactionPreflight(
                    False,
                    f'leg_{index}_{leg_geometry.code}',
                    leg_geometry.detail,
                    dense_samples=len(full_samples) + repeated_samples,
                    legs_checked=index,
                    minimum_shelf_clearance_m=min_clearance,
                    minimum_support_depth_margin_m=min_depth,
                    minimum_support_lateral_margin_m=min_lateral,
                )
            leg_support, clearance, depth, lateral = _support_route_result(
                node=node,
                environment=environment,
                samples=leg_samples,
                outward_world=outward_world,
            )
            min_clearance = min(min_clearance, clearance)
            min_depth = min(min_depth, depth)
            min_lateral = min(min_lateral, lateral)
            if not leg_support.safe:
                return CompactionPreflight(
                    False,
                    f'leg_{index}_{leg_support.code}',
                    leg_support.detail,
                    dense_samples=len(full_samples) + repeated_samples,
                    legs_checked=index,
                    minimum_shelf_clearance_m=min_clearance,
                    minimum_support_depth_margin_m=min_depth,
                    minimum_support_lateral_margin_m=min_lateral,
                )
            previous = np.asarray(target, dtype=float)

        after_scene = environment._read_scene()
        after = _measured_robot_state(node)
        after_relative = _measured_finger_transforms_relative_to_palm(
            node, after_scene, after
        )
        _require_stable_measured_finger_geometry(
            environment.model, measured_relative, after_relative
        )
        _require_stable_preflight_inputs(scene, after_scene, before, after)
        after_hand_world = (
            np.asarray(after_scene.base_transform, dtype=float)
            @ node.chain.forward(after.positions)
        )
        predicted_after = predict_attached_corners(anchor, after_hand_world)
        observed_after = np.asarray(
            after_scene.books[TARGET_BOOK_MODEL].corners, dtype=float
        )
        after_error = float(np.max(np.linalg.norm(
            observed_after - predicted_after, axis=1
        )))
        if after_error > PREFLIGHT_ANCHOR_CORNER_LIMIT_M:
            raise ValueError(
                'attachment moved during compaction preflight: '
                f'{after_error:.9f} m'
            )
        return CompactionPreflight(
            True,
            'clear',
            (
                f'{len(full_samples)} complete-route states and '
                f'{repeated_samples} repeated leg states passed'
            ),
            dense_samples=len(full_samples) + repeated_samples,
            legs_checked=len(plan.positions),
            minimum_shelf_clearance_m=min_clearance,
            minimum_support_depth_margin_m=min_depth,
            minimum_support_lateral_margin_m=min_lateral,
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return CompactionPreflight(False, 'live_adapter_error', str(error))


def compaction_endpoint_guard(
    reference: TransportObservation,
    current: TransportObservation,
    anchor: AttachmentAnchor,
    support: SupportAssessment,
    *,
    expected_arm: Sequence[float],
    navigation_radius_m: float,
    navigation_radius_limit_m: float,
    reference_time: float,
    left_contact: bool,
    right_contact: bool,
    palm_contact: bool,
    unexpected_contacts: bool,
) -> GuardResult:
    """Validate the measured compact cage before reporting diagnostic proof."""
    try:
        expected = np.asarray(expected_arm, dtype=float)
        radius = float(navigation_radius_m)
        radius_limit = float(navigation_radius_limit_m)
        age = float(reference_time) - current.observed_at
        arm_error = float(np.max(np.abs(current.arm - expected)))
        aperture_error = abs(current.aperture_m - RECAGE_APERTURE_M)
        base_translation = float(np.linalg.norm(
            current.base[:2] - reference.base[:2]
        ))
        base_yaw = _angle_error(current.base[2], reference.base[2])
        attachment_position, attachment_rotation, attachment_corner = (
            _attachment_errors(anchor, current)
        )
        clearance = shelf_clearance_m(current.corners)
        if (
            expected.shape != current.arm.shape
            or not math.isfinite(radius)
            or not math.isfinite(radius_limit)
        ):
            raise ValueError('endpoint target is malformed')
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return GuardResult(False, 'invalid_observation', {})
    metrics = {
        'scene_age_s': age,
        'arm_endpoint_error_rad': arm_error,
        'measured_aperture_m': current.aperture_m,
        'aperture_error_m': aperture_error,
        'base_translation_m': base_translation,
        'base_yaw_error_rad': base_yaw,
        'attachment_position_error_m': attachment_position,
        'attachment_rotation_error_rad': attachment_rotation,
        'attachment_corner_error_m': attachment_corner,
        'shelf_clearance_m': clearance,
        'navigation_radius_m': radius,
        'navigation_radius_limit_m': radius_limit,
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
        (arm_error <= ENDPOINT_ARM_LIMIT_RAD, 'compact_endpoint_missed'),
        (
            base_translation <= ENDPOINT_BASE_TRANSLATION_LIMIT_M,
            'base_moved_during_compaction',
        ),
        (
            base_yaw <= ENDPOINT_BASE_YAW_LIMIT_RAD,
            'base_rotated_during_compaction',
        ),
        (
            attachment_position <= ENDPOINT_ATTACHMENT_POSITION_LIMIT_M,
            'attachment_position_slip',
        ),
        (
            attachment_rotation <= ENDPOINT_ATTACHMENT_ROTATION_LIMIT_RAD,
            'attachment_rotation_slip',
        ),
        (
            attachment_corner <= ENDPOINT_ATTACHMENT_CORNER_LIMIT_M,
            'attachment_obb_slip',
        ),
        (
            clearance >= MINIMUM_SHELF_CLEARANCE_M,
            'target_lost_shelf_clearance',
        ),
        (support.safe, support.reason),
        (radius <= radius_limit, 'navigation_radius_exceeded'),
    )
    for accepted, reason in checks:
        if not accepted:
            return GuardResult(False, reason, metrics)
    return GuardResult(True, 'ok', metrics)


def build_compaction_goal(
    runtime: SimpleNamespace,
    legs: Sequence[tuple[Sequence[float], float, str]],
) -> tuple[Any, float]:
    """Build one ROS trajectory with cumulative ROS Duration values."""
    if not legs:
        raise ValueError('supported compaction trajectory requires a leg')
    goal = runtime.FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(runtime.ARM_JOINTS)
    elapsed = 0.0
    for positions, duration, _ in legs:
        q = np.asarray(positions, dtype=float)
        seconds = float(duration)
        if (
            q.shape != (8,)
            or not np.all(np.isfinite(q))
            or not math.isfinite(seconds)
            or seconds <= 0.0
        ):
            raise ValueError('supported compaction trajectory leg is invalid')
        elapsed += seconds
        point = runtime.JointTrajectoryPoint()
        point.positions = [float(value) for value in q[1:]]
        point.time_from_start = runtime.Duration(seconds=elapsed).to_msg()
        goal.trajectory.points.append(point)
    return goal, elapsed


def _load_compaction_runtime() -> SimpleNamespace:
    """Load ROS message classes only for a real diagnostic invocation."""
    runtime = _load_runtime()
    from control_msgs.action import FollowJointTrajectory
    from erc_phase1_solution.manipulation_node import ARM_JOINTS
    from rclpy.duration import Duration
    from trajectory_msgs.msg import JointTrajectoryPoint

    runtime.FollowJointTrajectory = FollowJointTrajectory
    runtime.ARM_JOINTS = ARM_JOINTS
    runtime.Duration = Duration
    runtime.JointTrajectoryPoint = JointTrajectoryPoint
    return runtime


def _navigation_pose(nav: Any) -> np.ndarray | None:
    value = getattr(nav, 'pose', None)
    if value is None:
        return None
    pose = np.asarray(value, dtype=float)
    return pose.copy() if pose.shape == (3,) else None


def _probe_types(runtime: SimpleNamespace) -> tuple[type, type]:
    BaseProbe, BaseNavigation = _transport_probe_types(runtime)

    class SupportedCompactionProbeNode(BaseProbe):
        def __init__(self) -> None:
            self._compaction_nav: Any | None = None
            self._compaction_reference_nav: np.ndarray | None = None
            super().__init__()

        def configure_compaction_watchdog(
            self,
            nav: Any,
            reference_pose: Sequence[float],
        ) -> None:
            self._compaction_nav = nav
            self._compaction_reference_nav = np.asarray(
                reference_pose, dtype=float
            ).copy()

        def _payload_hazard_reason(
            self, *, max_age: float = WATCHDOG_CONTACT_MAX_AGE_S
        ) -> str | None:
            inherited = super()._payload_hazard_reason(max_age=max_age)
            if inherited is not None:
                return inherited
            if (
                self._compaction_nav is None
                or self._compaction_reference_nav is None
            ):
                return None
            left, right = self.probe_exact_finger_sides(max_age=max_age)
            palm = self.probe_exact_palm_contact(max_age=max_age)
            joints = dict(self.joints)
            aperture = float(
                joints.get('gripper_left_finger_joint', math.nan)
            )
            return compaction_watchdog_reason(
                left_contact=left,
                right_contact=right,
                palm_contact=palm,
                unexpected_contacts=_unexpected_contact(self),
                measured_aperture_m=aperture,
                navigation_pose=_navigation_pose(self._compaction_nav),
                reference_navigation_pose=self._compaction_reference_nav,
                navigation_active=(
                    getattr(self._compaction_nav, 'goal', None) is not None
                ),
            )

        def _on_command(self, message: Any) -> None:
            self._publish_status(
                'rejected', reason='diagnostic_owns_controller'
            )

        def _run_command(self, command: str) -> None:
            self._publish_status(
                'rejected', reason='diagnostic_owns_controller'
            )

    class ObservationOnlyNavigation(BaseNavigation):
        def _on_goal(self, message: Any) -> None:
            self._publish_status(
                'rejected', reason='compaction_forbids_navigation'
            )

        def _on_command(self, message: Any) -> None:
            self._publish_status(
                'rejected', reason='compaction_forbids_navigation'
            )

    return SupportedCompactionProbeNode, ObservationOnlyNavigation


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
        name='rigid-palm-supported-compaction',
        daemon=True,
    )
    thread.start()
    stage = 'created'
    motion_commanded = False
    try:
        deadline = time.monotonic() + 30.0
        while (len(node.joints) < 8 or nav.pose is None) and (
            time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if len(node.joints) < 8 or nav.pose is None:
            raise RuntimeError('live robot/base state is unavailable')
        if nav.goal is not None:
            raise RuntimeError('navigation is still active')

        node._target_book_model = TARGET_BOOK_MODEL
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        node._gravity_supported_payload = True
        node._transport_lock_engaged = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        # Explicitly discard production caches; every value below is rebuilt
        # from the live endpoint and passed directly to pure planning helpers.
        node._cached_post_retreat_plan = None
        node._held_book_corners = None
        node._carried_staging_solution = None
        node.reset_probe_audit()

        reference, _ = _transport_observation(node, runtime)
        left, right, palm = _fresh_cage_gate(node)
        current, scene = _transport_observation(
            node, runtime, newer_than=reference.observed_at
        )
        anchor = capture_attachment_anchor(
            reference.book,
            reference.corners,
            reference.hand_world,
            observed_at=reference.observed_at,
        )
        support = _support_for_observation(
            node, runtime.environment_preflight, current, scene
        )
        resume = deep_cage_resume_guard(
            reference,
            current,
            anchor,
            support,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
            navigation_idle=(nav.goal is None),
        )
        if not resume.safe:
            raise RuntimeError(
                f'deep post-retreat cage rejected: {resume.reason}'
            )
        # Re-anchor at the end of the stationary proof, not at the older first
        # sample.  This measured transform is the sole attachment definition.
        anchor = capture_attachment_anchor(
            current.book,
            current.corners,
            current.hand_world,
            observed_at=current.observed_at,
        )
        attached = measured_attached_envelope(
            anchor, float(node.carried_book_padding)
        )
        candidates = supported_compaction_candidates(
            node, current.arm, attached
        )
        if not candidates:
            raise RuntimeError(
                'production-shaped helpers found no supported route at or '
                'below the navigation-radius limit'
            )
        _emit(
            'supported_compaction_resume_verified',
            measured_start_q=current.arm.tolist(),
            measured_base=current.base.tolist(),
            measured_book_position=current.book.position.tolist(),
            measured_book_minimum=current.book.minimum.tolist(),
            measured_book_maximum=current.book.maximum.tolist(),
            hand_T_book=anchor.hand_from_book.tolist(),
            book_T_hand=anchor.book_from_hand.tolist(),
            book_corners_in_hand=anchor.corners_in_hand.tolist(),
            candidate_count=len(candidates),
            cached_plan_used=False,
            diagnostic_truth_and_contacts_only=True,
            **resume.metrics,
        )

        stage = 'dense_preflight'
        selected: SupportedCompactionPlan | None = None
        selected_preflight: CompactionPreflight | None = None
        outward = np.asarray(
            [-math.cos(current.base[2]), -math.sin(current.base[2]), 0.0],
            dtype=float,
        )
        for candidate in candidates:
            report = preflight_supported_compaction(
                node=node,
                environment=runtime.environment_preflight,
                anchor=anchor,
                start_positions=current.arm,
                plan=candidate,
                outward_world=outward,
            )
            _emit(
                'supported_compaction_preflight',
                passed=report.safe,
                reason=report.code,
                detail=report.detail,
                route=candidate.label,
                controller_legs=len(candidate.positions),
                dense_samples=report.dense_samples,
                legs_checked=report.legs_checked,
                predicted_terminal_radius_m=candidate.terminal_radius_m,
                minimum_shelf_clearance_m=(
                    report.minimum_shelf_clearance_m
                ),
                minimum_support_depth_margin_m=(
                    report.minimum_support_depth_margin_m
                ),
                minimum_support_lateral_margin_m=(
                    report.minimum_support_lateral_margin_m
                ),
                diagnostic_truth_and_contacts_only=True,
            )
            if report.safe:
                selected = candidate
                selected_preflight = report
                break
        if selected is None or selected_preflight is None:
            raise RuntimeError(
                'every supported compaction route failed preflight'
            )

        if not node.arm_client.wait_for_server(
            timeout_sec=min(5.0, float(node.timeout))
        ):
            raise RuntimeError('left arm action server unavailable')
        # A fresh simulation-time dwell closes the potentially long geometric
        # planning window.  The measured start must still match the anchored
        # and preflighted endpoint before the single action is dispatched.
        left, right, palm = _fresh_cage_gate(node)
        dispatch, dispatch_scene = _transport_observation(
            node, runtime, newer_than=current.observed_at
        )
        dispatch_support = _support_for_observation(
            node,
            runtime.environment_preflight,
            dispatch,
            dispatch_scene,
        )
        dispatch_guard = deep_cage_resume_guard(
            current,
            dispatch,
            anchor,
            dispatch_support,
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
            navigation_idle=(nav.goal is None),
        )
        dispatch_arm_error = float(np.max(np.abs(
            dispatch.arm - current.arm
        )))
        if not dispatch_guard.safe or (
            dispatch_arm_error > PREFLIGHT_START_ARM_LIMIT_RAD
        ):
            reason = (
                dispatch_guard.reason
                if not dispatch_guard.safe
                else 'arm_changed_after_preflight'
            )
            raise RuntimeError(
                f'pre-dispatch measured state rejected: {reason}'
            )

        legs = tuple(
            (
                positions,
                node._transport_leg_duration(previous, positions),
                phase,
            )
            for previous, positions, phase in zip(
                (dispatch.arm, *selected.positions[:-1]),
                selected.positions,
                selected.phases,
            )
        )
        goal, total_duration = build_compaction_goal(runtime, legs)
        nav_reference = _navigation_pose(nav)
        if nav_reference is None:
            raise RuntimeError('base odometry disappeared before compaction')
        node.configure_compaction_watchdog(nav, nav_reference)
        stage = 'one_retained_trajectory'
        motion_commanded = True
        succeeded, contact_lost = node._send_retained_arm_trajectory(
            goal,
            total_duration,
            legs,
            SUPPORTED_COMPACTION_EVENT,
        )
        if not succeeded:
            reason = (
                'retention_or_base_fault'
                if contact_lost
                else 'controller_failure'
            )
            raise SupportedCompactionFailure(
                f'continuous compact trajectory failed: {reason}'
            )
        endpoint_q = node._wait_for_retained_endpoint(
            selected.positions[-1],
            command=SUPPORTED_COMPACTION_EVENT,
            phase='compact_cage_endpoint',
            leg=len(selected.positions) - 1,
            arm_tolerance=ENDPOINT_ARM_LIMIT_RAD,
        )
        if endpoint_q is None:
            raise SupportedCompactionFailure('compact endpoint was not held')

        left, right, palm = _fresh_cage_gate(node)
        final, final_scene = _transport_observation(
            node, runtime, newer_than=dispatch.observed_at
        )
        final_support = _support_for_observation(
            node, runtime.environment_preflight, final, final_scene
        )
        measured_attached = measured_attached_envelope(
            anchor, float(node.carried_book_padding)
        )
        measured_radius = float(node._carried_navigation_radius(
            final.arm, measured_attached
        ))
        endpoint = compaction_endpoint_guard(
            dispatch,
            final,
            anchor,
            final_support,
            expected_arm=selected.positions[-1],
            navigation_radius_m=measured_radius,
            navigation_radius_limit_m=float(
                node.carried_navigation_radius_limit
            ),
            reference_time=_node_time_seconds(node),
            left_contact=left,
            right_contact=right,
            palm_contact=palm,
            unexpected_contacts=_unexpected_contact(node),
        )
        if not endpoint.safe:
            raise SupportedCompactionFailure(
                f'compact endpoint gate failed: {endpoint.reason}'
            )
        final_anchor = capture_attachment_anchor(
            final.book,
            final.corners,
            final.hand_world,
            observed_at=final.observed_at,
        )
        stage = 'complete'
        _emit(
            'result',
            passed=True,
            stage=SUPPORTED_COMPACTION_EVENT,
            route=selected.label,
            controller_legs=len(selected.positions),
            retained_trajectory_duration_s=total_duration,
            measured_endpoint_q=final.arm.tolist(),
            measured_endpoint_radius_m=measured_radius,
            measured_book_position=final.book.position.tolist(),
            measured_book_minimum=final.book.minimum.tolist(),
            measured_book_maximum=final.book.maximum.tolist(),
            original_hand_T_book=anchor.hand_from_book.tolist(),
            measured_endpoint_hand_T_book=final_anchor.hand_from_book.tolist(),
            measured_endpoint_book_T_hand=final_anchor.book_from_hand.tolist(),
            preflight_detail=selected_preflight.detail,
            gripper_opened=False,
            navigation_commanded=False,
            placement_commanded=False,
            stopped_at_compact_cage=True,
            next_motion_authorized=False,
            diagnostic_truth_and_contacts_only=True,
            **endpoint.metrics,
        )
    except Exception as error:
        nav._publish_zero()
        _emit(
            'result',
            passed=False,
            stage=stage,
            reason=f'{type(error).__name__}: {error}',
            arm_motion_commanded=motion_commanded,
            gripper_opened=False,
            navigation_commanded=False,
            placement_commanded=False,
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
        description='Diagnostic-only supported compaction after base retreat'
    )
    parser.add_argument(
        '--confirm-diagnostic-supported-compaction',
        action='store_true',
        help=(
            'authorize one guarded diagnostic supported-compaction trajectory'
        ),
    )
    arguments = parser.parse_args()
    if not arguments.confirm_diagnostic_supported_compaction:
        parser.error('--confirm-diagnostic-supported-compaction is required')
    runtime = _load_compaction_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
