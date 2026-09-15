"""Focused tests for pure fail-closed adaptive-grasp evidence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from erc_phase1_solution.adaptive_grasp import (
    AdaptiveGraspEvidence,
    ForceSample,
    GripperFeedback,
    evaluate_bilateral_contact,
)


NOW_NS = 10_000_000_000
_UNSET = object()
_LIMITS = {
    'minimum_width': 0.0295,
    'maximum_width': 0.0500,
    'minimum_force': 0.50,
    'maximum_force': 3.00,
    'minimum_samples': 3,
    'maximum_age_seconds': 0.15,
    'maximum_gap_seconds': 0.06,
    'minimum_span_seconds': 0.08,
    'maximum_side_skew_seconds': 0.01,
    'maximum_velocity': 0.002,
    'maximum_effort': 2.0,
}


def _history(
    ages_ms=(100, 50, 0),
    forces=(0.75, 0.90, 1.10),
):
    return tuple(
        ForceSample(NOW_NS - int(age_ms * 1_000_000), force)
        for age_ms, force in zip(ages_ms, forces)
    )


def _evaluate(
    *,
    left=_UNSET,
    right=_UNSET,
    feedback=_UNSET,
    now_ns=NOW_NS,
    **overrides,
):
    limits = dict(_LIMITS)
    limits.update(overrides)
    return evaluate_bilateral_contact(
        _history() if left is _UNSET else left,
        _history(forces=(0.80, 0.95, 1.20))
        if right is _UNSET
        else right,
        GripperFeedback(NOW_NS, 0.035, 0.0)
        if feedback is _UNSET
        else feedback,
        now_ns,
        **limits,
    )


def test_valid_bilateral_window_reports_current_measurements():
    result = _evaluate()

    assert isinstance(result, AdaptiveGraspEvidence)
    assert result.verified
    assert result.reason == 'bilateral_contact_verified'
    assert result.width == pytest.approx(0.035)
    assert result.left_force == pytest.approx(1.10)
    assert result.right_force == pytest.approx(1.20)
    assert math.isnan(result.effort)
    assert math.isnan(result.effort_delta)
    assert result.left_samples == 3
    assert result.right_samples == 3


def test_evidence_records_are_frozen_and_effort_defaults_to_unavailable():
    sample = ForceSample(NOW_NS, 1.0)
    feedback = GripperFeedback(NOW_NS, 0.035, 0.0)
    result = _evaluate()

    assert math.isnan(feedback.effort)
    with pytest.raises(FrozenInstanceError):
        sample.force_newtons = 2.0
    with pytest.raises(FrozenInstanceError):
        feedback.position = 0.030
    with pytest.raises(FrozenInstanceError):
        result.verified = False


@pytest.mark.parametrize(
    ('feedback', 'overrides'),
    [
        (GripperFeedback(NOW_NS, 0.0295, 0.0), {}),
        (GripperFeedback(NOW_NS, 0.035, 0.002), {}),
        (GripperFeedback(NOW_NS, 0.035, -0.002), {}),
        (GripperFeedback(NOW_NS, 0.035, 0.0, 2.0), {}),
        (
            GripperFeedback(NOW_NS, 0.035, 0.0, 1.5),
            {'baseline_effort': 0.5, 'maximum_effort_delta': 1.0},
        ),
    ],
)
def test_feedback_safety_boundaries_are_inclusive(feedback, overrides):
    assert _evaluate(feedback=feedback, **overrides).verified


@pytest.mark.parametrize(
    ('left', 'right', 'overrides'),
    [
        (
            _history(forces=(0.50, 0.50, 0.50)),
            _history(forces=(0.50, 0.50, 0.50)),
            {},
        ),
        (
            _history(forces=(3.0, 3.0, 3.0)),
            _history(forces=(3.0, 3.0, 3.0)),
            {},
        ),
        (
            _history(ages_ms=(150, 75, 0)),
            _history(ages_ms=(150, 75, 0)),
            {
                'maximum_gap_seconds': 0.075,
                'minimum_span_seconds': 0.15,
            },
        ),
        (
            _history(ages_ms=(120, 60, 0)),
            _history(ages_ms=(120, 60, 0)),
            {'maximum_gap_seconds': 0.06},
        ),
        (
            _history(ages_ms=(80, 40, 0)),
            _history(ages_ms=(80, 40, 0)),
            {'minimum_span_seconds': 0.08},
        ),
        (
            _history(ages_ms=(110, 60, 10)),
            _history(ages_ms=(100, 50, 0)),
            {'maximum_side_skew_seconds': 0.01},
        ),
    ],
)
def test_force_and_timing_boundaries_are_inclusive(
    left,
    right,
    overrides,
):
    assert _evaluate(left=left, right=right, **overrides).verified


@pytest.mark.parametrize(
    ('feedback', 'now_ns', 'expected_reason'),
    [
        (None, NOW_NS, 'joint_feedback_unavailable'),
        (object(), NOW_NS, 'joint_feedback_invalid'),
        (
            GripperFeedback(NOW_NS - 151_000_000, 0.035, 0.0),
            NOW_NS,
            'joint_feedback_stale',
        ),
        (
            GripperFeedback(NOW_NS + 1, 0.035, 0.0),
            NOW_NS,
            'joint_feedback_invalid',
        ),
        (
            GripperFeedback(NOW_NS, math.nan, 0.0),
            NOW_NS,
            'joint_feedback_invalid',
        ),
        (
            GripperFeedback(NOW_NS, 0.035, math.inf),
            NOW_NS,
            'joint_feedback_invalid',
        ),
        (GripperFeedback(NOW_NS, 0.035, 0.0), -1, 'joint_feedback_invalid'),
    ],
)
def test_missing_stale_or_nonfinite_feedback_fails_closed(
    feedback,
    now_ns,
    expected_reason,
):
    result = _evaluate(feedback=feedback, now_ns=now_ns)

    assert not result.verified
    assert result.reason == expected_reason


@pytest.mark.parametrize(
    ('feedback', 'expected_reason'),
    [
        (GripperFeedback(NOW_NS, 0.02949, 0.0), 'width_too_narrow'),
        (GripperFeedback(NOW_NS, 0.0500, 0.0), 'width_too_wide'),
        (GripperFeedback(NOW_NS, 0.05001, 0.0), 'width_too_wide'),
        (GripperFeedback(NOW_NS, 0.035, 0.00201), 'joint_still_moving'),
        (GripperFeedback(NOW_NS, 0.035, -0.00201), 'joint_still_moving'),
    ],
)
def test_width_and_motion_gates_fail_closed(feedback, expected_reason):
    result = _evaluate(feedback=feedback)

    assert not result.verified
    assert result.reason == expected_reason


@pytest.mark.parametrize(
    'history',
    [
        None,
        (object(),),
        (ForceSample(NOW_NS, math.nan),),
        (ForceSample(NOW_NS, math.inf),),
        (ForceSample(NOW_NS, -0.01),),
        (ForceSample(True, 1.0),),
        (
            ForceSample(NOW_NS - 10, 1.0),
            ForceSample(NOW_NS - 10, 1.0),
        ),
        (
            ForceSample(NOW_NS, 1.0),
            ForceSample(NOW_NS - 10, 1.0),
        ),
        (ForceSample(NOW_NS + 1, 1.0),),
    ],
)
def test_malformed_force_history_fails_closed(history):
    result = _evaluate(left=history)

    assert not result.verified
    assert result.reason == 'force_history_invalid'


@pytest.mark.parametrize('side', ('left', 'right'))
def test_recent_force_overload_on_either_side_aborts(side):
    overloaded = _history(forces=(1.0, 3.01, 1.0))
    result = _evaluate(**{side: overloaded})

    assert not result.verified
    assert result.reason == 'force_overload'


def test_stale_overload_does_not_poison_a_fresh_safe_window():
    history = (
        ForceSample(NOW_NS - 200_000_000, 20.0),
        *_history(),
    )

    assert _evaluate(left=history, right=history).verified


@pytest.mark.parametrize(
    ('left', 'right', 'expected_reason'),
    [
        ((), (), 'bilateral_contact_missing'),
        (_history(), (), 'unilateral_contact'),
        ((), _history(), 'unilateral_contact'),
        (
            _history(ages_ms=(300, 250, 200)),
            _history(ages_ms=(300, 250, 200)),
            'bilateral_contact_stale',
        ),
    ],
)
def test_missing_unilateral_and_stale_contact_never_verify(
    left,
    right,
    expected_reason,
):
    result = _evaluate(left=left, right=right)

    assert not result.verified
    assert result.reason == expected_reason


def test_unilateral_result_retains_recent_force_count_for_travel_guard():
    result = _evaluate(left=_history(), right=())

    assert result.reason == 'unilateral_contact'
    assert result.left_samples == 3
    assert result.right_samples == 0


@pytest.mark.parametrize(
    ('left', 'right', 'expected_reason'),
    [
        (
            _history(forces=(1.0, 1.0, 0.49)),
            _history(),
            'left_force_below_minimum',
        ),
        (
            _history(),
            _history(forces=(1.0, 1.0, 0.49)),
            'right_force_below_minimum',
        ),
        (
            _history(ages_ms=(50, 0), forces=(1.0, 1.0)),
            _history(),
            'insufficient_force_samples',
        ),
        (
            _history(ages_ms=(150, 50, 0)),
            _history(),
            'force_sample_gap',
        ),
        (
            _history(ages_ms=(40, 20, 0)),
            _history(),
            'force_sample_span_too_short',
        ),
        (
            _history(ages_ms=(120, 70, 20)),
            _history(ages_ms=(100, 50, 0)),
            'bilateral_sample_skew',
        ),
    ],
)
def test_contact_must_be_sustained_and_synchronized(
    left,
    right,
    expected_reason,
):
    result = _evaluate(left=left, right=right)

    assert not result.verified
    assert result.reason == expected_reason


def test_latest_contact_must_be_within_the_inter_sample_gap_limit():
    ended = _history(
        ages_ms=(140, 100, 80),
        forces=(0.8, 0.9, 1.0),
    )

    result = _evaluate(
        left=ended,
        right=ended,
        minimum_span_seconds=0.05,
    )

    assert not result.verified
    assert result.reason == 'bilateral_contact_stale'


def test_high_rate_samples_can_use_the_full_qualifying_contact_span():
    high_rate = _history(
        ages_ms=(60, 40, 20, 0),
        forces=(0.8, 0.9, 1.0, 1.1),
    )

    result = _evaluate(
        left=high_rate,
        right=high_rate,
        minimum_span_seconds=0.05,
    )

    assert result.verified
    assert result.left_samples == 4
    assert result.right_samples == 4


def test_low_or_gapped_prefix_does_not_poison_a_complete_new_suffix():
    left = (
        ForceSample(NOW_NS - 190_000_000, 1.0),
        ForceSample(NOW_NS - 100_000_000, 0.25),
        ForceSample(NOW_NS - 90_000_000, 0.75),
        ForceSample(NOW_NS - 45_000_000, 0.90),
        ForceSample(NOW_NS, 1.10),
    )
    right = left

    result = _evaluate(left=left, right=right)

    assert result.verified
    assert result.left_samples == 3
    assert result.right_samples == 3


def test_nan_effort_is_unavailable_and_cannot_replace_contacts():
    feedback = GripperFeedback(NOW_NS, 0.035, 0.0)

    accepted = _evaluate(feedback=feedback)
    rejected = _evaluate(left=(), right=(), feedback=feedback)

    assert accepted.verified
    assert math.isnan(accepted.effort)
    assert math.isnan(accepted.effort_delta)
    assert not rejected.verified
    assert rejected.reason == 'bilateral_contact_missing'


def test_finite_effort_and_baseline_report_absolute_delta():
    result = _evaluate(
        feedback=GripperFeedback(NOW_NS, 0.035, 0.0, -0.8),
        baseline_effort=-0.3,
        maximum_effort_delta=0.6,
    )

    assert result.verified
    assert result.effort == pytest.approx(-0.8)
    assert result.effort_delta == pytest.approx(0.5)


@pytest.mark.parametrize(
    ('feedback', 'overrides', 'expected_reason'),
    [
        (
            GripperFeedback(NOW_NS, 0.035, 0.0, 2.01),
            {},
            'effort_overload',
        ),
        (
            GripperFeedback(NOW_NS, 0.035, 0.0, -2.01),
            {},
            'effort_overload',
        ),
        (
            GripperFeedback(NOW_NS, 0.035, 0.0, 1.51),
            {'baseline_effort': 0.5, 'maximum_effort_delta': 1.0},
            'effort_delta_overload',
        ),
        (
            GripperFeedback(NOW_NS, 0.035, 0.0, math.inf),
            {},
            'joint_feedback_invalid',
        ),
        (
            GripperFeedback(NOW_NS, 0.035, 0.0, 0.5),
            {'baseline_effort': math.inf},
            'joint_feedback_invalid',
        ),
    ],
)
def test_effort_is_only_an_optional_overload_guard(
    feedback,
    overrides,
    expected_reason,
):
    result = _evaluate(feedback=feedback, **overrides)

    assert not result.verified
    assert result.reason == expected_reason


@pytest.mark.parametrize(
    'overrides',
    [
        {'minimum_width': math.nan},
        {'minimum_width': 0.05, 'maximum_width': 0.05},
        {'minimum_width': 0.05, 'maximum_width': 0.04},
        {'minimum_force': 0.0},
        {'minimum_force': 2.0, 'maximum_force': 2.0},
        {'minimum_samples': 2},
        {'minimum_samples': True},
        {'maximum_age_seconds': 0.0},
        {'maximum_gap_seconds': 0.20},
        {'minimum_span_seconds': 0.20},
        {'maximum_side_skew_seconds': -0.01},
        {'maximum_velocity': -0.01},
        {'maximum_effort': math.nan},
        {'maximum_effort_delta': -0.01},
    ],
)
def test_invalid_safety_limits_raise_value_error(overrides):
    message = 'adaptive grasp|minimum_samples|effort'
    with pytest.raises(ValueError, match=message):
        _evaluate(**overrides)


def test_low_force_in_middle_resets_the_consecutive_run():
    history = _history(
        ages_ms=(120, 80, 40, 0),
        forces=(1.0, 0.49, 1.0, 1.0),
    )

    result = _evaluate(left=history)

    assert not result.verified
    assert result.reason == 'insufficient_force_samples'
    assert result.left_samples == 2


def test_result_preserves_feedback_on_an_evidence_failure():
    feedback = GripperFeedback(NOW_NS, 0.034, 0.0, 0.8)
    result = _evaluate(
        left=(),
        right=(),
        feedback=feedback,
        baseline_effort=0.5,
        maximum_effort_delta=1.0,
    )

    assert not result.verified
    assert result.width == pytest.approx(0.034)
    assert result.effort == pytest.approx(0.8)
    assert result.effort_delta == pytest.approx(0.3)
