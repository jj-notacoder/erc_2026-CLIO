"""Focused boundary checks for the 0.5 mm shelf-step pure gates."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / 'live_physical_seed101_extract_checkpoint.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('seed101_micro_outward_gates', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
START_POSITION = (0.0, 0.0, 1.5)
PLANNED_DELTA = (-0.0005, 0.0, 0.0)

RELAXED_MICRO_POLICY = {
    'min_book_center_outward_progress_m': 0.0,
    'max_book_center_outward_progress_m': 0.001,
    'min_book_deepest_edge_outward_progress_m': 0.0,
    'max_book_deepest_edge_outward_progress_m': 0.001,
    'max_book_from_hand_translation_change_m': 0.001,
    'max_cross_track_motion_m': 0.001,
    'max_lateral_y_motion_m': 0.001,
    'max_incremental_rotation_rad': 0.1,
    'max_incremental_yaw_component_rad': 0.1,
    'max_absolute_rotation_to_stable_reference_rad': 0.1,
    'max_absolute_yaw_to_stable_reference_rad': 0.1,
    'min_floor_signed_distance_m': -0.01,
    'max_floor_signed_distance_m': 0.01,
}


def axis_angle_quaternion(axis: str, angle: float):
    vector = {
        'x': (1.0, 0.0, 0.0),
        'y': (0.0, 1.0, 0.0),
        'z': (0.0, 0.0, 1.0),
    }[axis]
    sine = math.sin(0.5 * angle)
    return (
        vector[0] * sine,
        vector[1] * sine,
        vector[2] * sine,
        math.cos(0.5 * angle),
    )


def micro_gate(
    *,
    observed_delta=PLANNED_DELTA,
    observed_quaternion=IDENTITY_QUATERNION,
    stable_quaternion=None,
    floor_signed_m=0.0,
    planned_delta=PLANNED_DELTA,
    policy=None,
):
    start = probe.EntityPose(START_POSITION, IDENTITY_QUATERNION)
    observed = probe.EntityPose(
        tuple(
            before + delta
            for before, delta in zip(START_POSITION, observed_delta)
        ),
        observed_quaternion,
    )
    stable_reference = probe.EntityPose(
        START_POSITION,
        observed_quaternion if stable_quaternion is None else stable_quaternion,
    )
    support_floor = probe.book_minimum_world_z(observed) - floor_signed_m
    return probe.micro_outward_progress_gate(
        start,
        observed,
        planned_delta_world_m=planned_delta,
        stable_reference=stable_reference,
        support_floor_world_z_m=support_floor,
        policy=policy,
    )


def test_micro_outward_gate_accepts_every_limit_at_exact_equality():
    baseline = micro_gate()
    assert baseline.ok

    metrics = baseline.metrics
    equality_policy = {
        'min_book_center_outward_progress_m': metrics[
            'book_center_outward_progress_m'
        ],
        'max_book_center_outward_progress_m': metrics[
            'book_center_outward_progress_m'
        ],
        'min_book_deepest_edge_outward_progress_m': metrics[
            'book_deepest_edge_outward_progress_m'
        ],
        'max_book_deepest_edge_outward_progress_m': metrics[
            'book_deepest_edge_outward_progress_m'
        ],
        'max_book_from_hand_translation_change_m': metrics[
            'book_hand_translation_change_m'
        ],
        'max_cross_track_motion_m': metrics['book_cross_track_motion_m'],
        'max_lateral_y_motion_m': metrics['book_lateral_y_motion_m'],
        'max_incremental_rotation_rad': metrics[
            'book_incremental_rotation_rad'
        ],
        'max_incremental_yaw_component_rad': metrics[
            'book_incremental_yaw_component_rad'
        ],
        'max_absolute_rotation_to_stable_reference_rad': metrics[
            'book_stable_reference_rotation_rad'
        ],
        'max_absolute_yaw_to_stable_reference_rad': metrics[
            'book_stable_reference_yaw_component_rad'
        ],
        'min_floor_signed_distance_m': metrics['book_floor_signed_m'],
        'max_floor_signed_distance_m': metrics['book_floor_signed_m'],
    }

    result = micro_gate(policy=equality_policy)
    assert result.ok
    assert result.reason == 'micro_outward_rung_verified'


@pytest.mark.parametrize(
    ('planned_delta',),
    (
        (None,),
        ((-0.0005, 0.0),),
        ((-0.0005, 0.0, 0.0, 0.0),),
        ((0.0005, 0.0, 0.0),),
        ((-0.0005, 1e-9, 0.0),),
        ((-0.0005, 0.0, 1e-9),),
        ((math.nan, 0.0, 0.0),),
    ),
)
def test_micro_outward_gate_rejects_invalid_plan(planned_delta):
    result = micro_gate(planned_delta=planned_delta)
    assert not result.ok
    assert result.reason == 'micro_outward_plan_invalid'


@pytest.mark.parametrize(
    ('expected_reason', 'gate_changes', 'policy_change'),
    (
        (
            'micro_outward_lacked_center_progress',
            {},
            {'min_book_center_outward_progress_m': 0.000500001},
        ),
        (
            'micro_outward_center_overshoot',
            {},
            {'max_book_center_outward_progress_m': 0.000499999},
        ),
        (
            'micro_outward_lacked_deep_edge_progress',
            {},
            {'min_book_deepest_edge_outward_progress_m': 0.000500001},
        ),
        (
            'micro_outward_deep_edge_overshoot',
            {},
            {'max_book_deepest_edge_outward_progress_m': 0.000499999},
        ),
        (
            'micro_outward_book_hand_translation_mismatch',
            {'observed_delta': (-0.00049, 0.0, 0.0)},
            {'max_book_from_hand_translation_change_m': 0.000009},
        ),
        (
            'micro_outward_cross_track_exceeded_limit',
            {'observed_delta': (-0.0005, 0.00005, 0.0)},
            {'max_cross_track_motion_m': 0.000049},
        ),
        (
            'micro_outward_lateral_motion_exceeded_limit',
            {'observed_delta': (-0.0005, 0.00005, 0.0)},
            {'max_lateral_y_motion_m': 0.000049},
        ),
        (
            'micro_outward_incremental_rotation_exceeded_limit',
            {'observed_quaternion': axis_angle_quaternion('x', 0.0004)},
            {'max_incremental_rotation_rad': 0.000399},
        ),
        (
            'micro_outward_incremental_yaw_exceeded_limit',
            {'observed_quaternion': axis_angle_quaternion('z', 0.0003)},
            {'max_incremental_yaw_component_rad': 0.000299},
        ),
        (
            'micro_outward_absolute_rotation_exceeded_limit',
            {'stable_quaternion': axis_angle_quaternion('x', 0.0004)},
            {'max_absolute_rotation_to_stable_reference_rad': 0.000399},
        ),
        (
            'micro_outward_absolute_yaw_exceeded_limit',
            {'stable_quaternion': axis_angle_quaternion('z', 0.0003)},
            {'max_absolute_yaw_to_stable_reference_rad': 0.000299},
        ),
        (
            'micro_outward_shelf_penetration_exceeded_limit',
            {'floor_signed_m': -0.000101},
            {'min_floor_signed_distance_m': -0.0001},
        ),
        (
            'micro_outward_lost_shelf_support',
            {'floor_signed_m': 0.000251},
            {'max_floor_signed_distance_m': 0.00025},
        ),
    ),
)
def test_micro_outward_gate_rejects_each_boundary(
    expected_reason,
    gate_changes,
    policy_change,
):
    policy = dict(RELAXED_MICRO_POLICY)
    policy.update(policy_change)
    result = micro_gate(policy=policy, **gate_changes)
    assert not result.ok
    assert result.reason == expected_reason


def test_stable_orientation_gate_accepts_zero_limits_at_exact_reference():
    reference = probe.EntityPose(START_POSITION, IDENTITY_QUATERNION)
    result = probe.stable_reference_orientation_gate(
        reference,
        reference,
        maximum_rotation_rad=0.0,
        maximum_yaw_component_rad=0.0,
    )
    assert result.ok
    assert result.reason == 'checkpoint_stable_orientation_verified'


@pytest.mark.parametrize(
    (
        'observed_quaternion',
        'maximum_rotation_rad',
        'maximum_yaw_component_rad',
        'expected_reason',
    ),
    (
        (
            axis_angle_quaternion('x', 0.001),
            0.000999,
            1.0,
            'checkpoint_absolute_rotation_exceeded_limit',
        ),
        (
            axis_angle_quaternion('z', 0.001),
            0.002,
            0.000999,
            'checkpoint_absolute_yaw_exceeded_limit',
        ),
        (
            (0.0, 0.0, 0.0, 0.0),
            0.002,
            0.002,
            'checkpoint_absolute_rotation_exceeded_limit',
        ),
    ),
)
def test_stable_orientation_gate_rejects_each_boundary(
    observed_quaternion,
    maximum_rotation_rad,
    maximum_yaw_component_rad,
    expected_reason,
):
    result = probe.stable_reference_orientation_gate(
        probe.EntityPose(START_POSITION, observed_quaternion),
        probe.EntityPose(START_POSITION, IDENTITY_QUATERNION),
        maximum_rotation_rad=maximum_rotation_rad,
        maximum_yaw_component_rad=maximum_yaw_component_rad,
    )
    assert not result.ok
    assert result.reason == expected_reason


def test_bilateral_pressure_gate_accepts_exact_retention_and_balance_limits():
    result = probe.bilateral_pressure_retention_gate(
        (1.0, 1.0),
        (1.0, 3.0),
        minimum_each_side_retention_fraction=1.0,
        maximum_normalized_balance_change=0.25,
    )
    assert result.ok
    assert result.reason == 'bilateral_force_retention_verified'
    assert result.metrics['left_force_retention_fraction'] == pytest.approx(1.0)
    assert result.metrics['normalized_balance_change'] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ('reference', 'observed', 'minimum_retention', 'maximum_balance'),
    (
        ((1.0,), (1.0, 1.0), 0.5, 0.2),
        ((1.0, 1.0), (1.0,), 0.5, 0.2),
        ((0.0, 1.0), (1.0, 1.0), 0.5, 0.2),
        ((-1.0, 1.0), (1.0, 1.0), 0.5, 0.2),
        ((math.nan, 1.0), (1.0, 1.0), 0.5, 0.2),
        ((1.0, 1.0), (-0.001, 1.0), 0.5, 0.2),
        ((1.0, 1.0), (math.inf, 1.0), 0.5, 0.2),
        ((1.0, 1.0), (1.0, 1.0), -0.001, 0.2),
        ((1.0, 1.0), (1.0, 1.0), 1.001, 0.2),
        ((1.0, 1.0), (1.0, 1.0), math.nan, 0.2),
        ((1.0, 1.0), (1.0, 1.0), 0.5, -0.001),
        ((1.0, 1.0), (1.0, 1.0), 0.5, 1.001),
        ((1.0, 1.0), (1.0, 1.0), 0.5, math.nan),
    ),
)
def test_bilateral_pressure_gate_rejects_invalid_inputs(
    reference,
    observed,
    minimum_retention,
    maximum_balance,
):
    result = probe.bilateral_pressure_retention_gate(
        reference,
        observed,
        minimum_each_side_retention_fraction=minimum_retention,
        maximum_normalized_balance_change=maximum_balance,
    )
    assert not result.ok
    assert result.reason == 'bilateral_pressure_retention_input_invalid'


def test_bilateral_pressure_gate_rejects_side_below_retention_boundary():
    result = probe.bilateral_pressure_retention_gate(
        (2.0, 2.0),
        (0.999999, 2.0),
        minimum_each_side_retention_fraction=0.5,
        maximum_normalized_balance_change=1.0,
    )
    assert not result.ok
    assert result.reason == 'bilateral_force_weakened_after_shelf_step'


def test_bilateral_pressure_gate_rejects_balance_just_over_boundary():
    result = probe.bilateral_pressure_retention_gate(
        (2.0, 2.0),
        (1.0, 2.0),
        minimum_each_side_retention_fraction=0.5,
        maximum_normalized_balance_change=(1.0 / 6.0) - 1e-12,
    )
    assert not result.ok
    assert result.reason == 'bilateral_force_balance_shifted_after_shelf_step'


def test_bilateral_pressure_gate_treats_zero_observed_force_as_weakened():
    result = probe.bilateral_pressure_retention_gate(
        (2.0, 2.0),
        (0.0, 0.0),
        minimum_each_side_retention_fraction=0.5,
        maximum_normalized_balance_change=1.0,
    )
    assert not result.ok
    assert result.reason == 'bilateral_force_weakened_after_shelf_step'
    assert math.isnan(result.metrics['observed_normalized_left_balance'])
