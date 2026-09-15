"""Pure fail-closed evaluation of bilateral adaptive-grasp evidence.

The evaluator deliberately has no ROS dependencies.  Callers remain
responsible for associating force samples with the selected target and the
correct left/right fingers before passing them here.  Joint effort is only an
optional overload guard; it is never positive evidence of contact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class ForceSample:
    """One target-specific finger-force observation."""

    stamp_ns: int
    force_newtons: float


@dataclass(frozen=True)
class GripperFeedback:
    """One measured gripper state; effort may be unavailable as NaN."""

    stamp_ns: int
    position: float
    velocity: float
    effort: float = math.nan


@dataclass(frozen=True)
class AdaptiveGraspEvidence:
    """Outcome and measured evidence for one bilateral-contact decision."""

    verified: bool
    reason: str
    width: float
    left_force: float
    right_force: float
    effort: float
    effort_delta: float
    left_samples: int
    right_samples: int


@dataclass(frozen=True)
class _ForceHistory:
    """Validated complete and maximum-age-filtered force history."""

    samples: Tuple[ForceSample, ...]
    recent: Tuple[ForceSample, ...]


def _is_stamp(value: object) -> bool:
    """Return whether ``value`` is a non-negative integer timestamp."""
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and int(value) >= 0
    )


def _finite_number(value: object) -> Optional[float]:
    """Convert a real, finite, non-boolean scalar or return ``None``."""
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _evidence(
    verified: bool,
    reason: str,
    *,
    width: float,
    left_force: float,
    right_force: float,
    effort: float,
    effort_delta: float,
    left_samples: int,
    right_samples: int,
) -> AdaptiveGraspEvidence:
    """Build an evidence result while keeping every failure diagnostic."""
    return AdaptiveGraspEvidence(
        verified=verified,
        reason=reason,
        width=width,
        left_force=left_force,
        right_force=right_force,
        effort=effort,
        effort_delta=effort_delta,
        left_samples=left_samples,
        right_samples=right_samples,
    )


def _validate_limits(
    *,
    minimum_width: float,
    maximum_width: float,
    minimum_force: float,
    maximum_force: float,
    minimum_samples: int,
    maximum_age_seconds: float,
    maximum_gap_seconds: float,
    minimum_span_seconds: float,
    maximum_side_skew_seconds: float,
    maximum_velocity: float,
    maximum_effort: float,
    maximum_effort_delta: float,
) -> None:
    """Reject unsafe evaluator configuration instead of hiding it as data."""
    finite_values = (
        minimum_width,
        maximum_width,
        minimum_force,
        maximum_force,
        maximum_age_seconds,
        maximum_gap_seconds,
        minimum_span_seconds,
        maximum_side_skew_seconds,
        maximum_velocity,
    )
    if any(_finite_number(value) is None for value in finite_values):
        raise ValueError('adaptive grasp limits must be finite real numbers')
    if (
        minimum_width < 0.0
        or maximum_width <= minimum_width
        or minimum_force <= 0.0
        or maximum_force <= minimum_force
    ):
        raise ValueError('adaptive grasp width or force limits are invalid')
    if (
        not isinstance(minimum_samples, Integral)
        or isinstance(minimum_samples, bool)
        or int(minimum_samples) < 3
    ):
        raise ValueError(
            'minimum_samples must be an integer of at least three'
        )
    if (
        maximum_age_seconds <= 0.0
        or maximum_gap_seconds <= 0.0
        or maximum_gap_seconds > maximum_age_seconds
        or minimum_span_seconds <= 0.0
        or minimum_span_seconds > maximum_age_seconds
        or maximum_side_skew_seconds < 0.0
        or maximum_side_skew_seconds > maximum_age_seconds
        or maximum_velocity < 0.0
    ):
        raise ValueError(
            'adaptive grasp timing or velocity limits are invalid'
        )
    for value, name in (
        (maximum_effort, 'maximum_effort'),
        (maximum_effort_delta, 'maximum_effort_delta'),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or math.isnan(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f'{name} must be non-negative and not NaN')


def _prepare_history(
    values: object,
    *,
    side: str,
    now_ns: int,
    maximum_age_seconds: float,
) -> tuple[Optional[_ForceHistory], Optional[str]]:
    """Validate one ordered history and select its recent suffix."""
    try:
        samples = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        return None, f'malformed_{side}_force_history'
    previous_stamp = -1
    for sample in samples:
        if not isinstance(sample, ForceSample):
            return None, f'malformed_{side}_force_history'
        if not _is_stamp(sample.stamp_ns):
            return None, f'malformed_{side}_force_timestamp'
        stamp_ns = int(sample.stamp_ns)
        if stamp_ns <= previous_stamp:
            return None, f'nonmonotonic_{side}_force_history'
        if stamp_ns > now_ns:
            return None, f'future_{side}_force_sample'
        force = _finite_number(sample.force_newtons)
        if force is None:
            return None, f'nonfinite_{side}_force'
        if force < 0.0:
            return None, f'negative_{side}_force'
        previous_stamp = stamp_ns
    recent = tuple(
        sample
        for sample in samples
        if (now_ns - int(sample.stamp_ns)) / 1e9 <= maximum_age_seconds
    )
    return _ForceHistory(samples=samples, recent=recent), None


def _qualifying_suffix(
    samples: Sequence[ForceSample],
    *,
    minimum_force: float,
    maximum_gap_seconds: float,
) -> tuple[Tuple[ForceSample, ...], bool]:
    """Return the newest consecutive above-threshold run and gap status."""
    if not samples or float(samples[-1].force_newtons) < minimum_force:
        return (), False
    reversed_run = [samples[-1]]
    gap_blocked = False
    for previous, current in zip(
        reversed(samples[:-1]),
        reversed(samples[1:]),
    ):
        if float(previous.force_newtons) < minimum_force:
            break
        gap = (int(current.stamp_ns) - int(previous.stamp_ns)) / 1e9
        if gap > maximum_gap_seconds:
            gap_blocked = True
            break
        reversed_run.append(previous)
    return tuple(reversed(reversed_run)), gap_blocked


def evaluate_bilateral_contact(
    left_samples: Sequence[ForceSample],
    right_samples: Sequence[ForceSample],
    feedback: Optional[GripperFeedback],
    now_ns: int,
    *,
    minimum_width: float,
    maximum_width: float,
    minimum_force: float,
    maximum_force: float,
    minimum_samples: int,
    maximum_age_seconds: float,
    maximum_gap_seconds: float,
    minimum_span_seconds: float,
    maximum_side_skew_seconds: float,
    maximum_velocity: float,
    maximum_effort: float,
    baseline_effort: float = math.nan,
    maximum_effort_delta: float = math.inf,
) -> AdaptiveGraspEvidence:
    """Evaluate fresh, sustained and synchronized bilateral contact.

    Invalid configuration raises ``ValueError``.  Missing, malformed, stale,
    unsafe, or insufficient runtime evidence instead returns
    ``verified=False``.
    The newest consecutive run on each side must contain at least
    ``minimum_samples`` observations above the force floor, remain within the
    age and gap limits, span the required duration, and end with synchronized
    observations on both sides.
    """
    _validate_limits(
        minimum_width=minimum_width,
        maximum_width=maximum_width,
        minimum_force=minimum_force,
        maximum_force=maximum_force,
        minimum_samples=minimum_samples,
        maximum_age_seconds=maximum_age_seconds,
        maximum_gap_seconds=maximum_gap_seconds,
        minimum_span_seconds=minimum_span_seconds,
        maximum_side_skew_seconds=maximum_side_skew_seconds,
        maximum_velocity=maximum_velocity,
        maximum_effort=maximum_effort,
        maximum_effort_delta=maximum_effort_delta,
    )
    unknown = math.nan
    width = unknown
    effort = unknown
    effort_delta = unknown
    left_force = unknown
    right_force = unknown
    left_count = 0
    right_count = 0

    def failed(reason: str) -> AdaptiveGraspEvidence:
        return _evidence(
            False,
            reason,
            width=width,
            left_force=left_force,
            right_force=right_force,
            effort=effort,
            effort_delta=effort_delta,
            left_samples=left_count,
            right_samples=right_count,
        )

    if not _is_stamp(now_ns):
        return failed('joint_feedback_invalid')
    now_ns = int(now_ns)
    if feedback is None:
        return failed('joint_feedback_unavailable')
    if not isinstance(feedback, GripperFeedback):
        return failed('joint_feedback_invalid')
    if not _is_stamp(feedback.stamp_ns):
        return failed('joint_feedback_invalid')
    feedback_stamp = int(feedback.stamp_ns)
    if feedback_stamp > now_ns:
        return failed('joint_feedback_invalid')
    if (now_ns - feedback_stamp) / 1e9 > maximum_age_seconds:
        return failed('joint_feedback_stale')
    position = _finite_number(feedback.position)
    if position is None:
        return failed('joint_feedback_invalid')
    width = position
    velocity = _finite_number(feedback.velocity)
    if velocity is None:
        return failed('joint_feedback_invalid')

    raw_effort = feedback.effort
    if not isinstance(raw_effort, Real) or isinstance(raw_effort, bool):
        return failed('joint_feedback_invalid')
    if math.isinf(float(raw_effort)):
        return failed('joint_feedback_invalid')
    if not isinstance(baseline_effort, Real) or isinstance(
        baseline_effort,
        bool,
    ):
        return failed('joint_feedback_invalid')
    if math.isinf(float(baseline_effort)):
        return failed('joint_feedback_invalid')
    if math.isfinite(float(raw_effort)):
        effort = float(raw_effort)
        if abs(effort) > maximum_effort:
            return failed('effort_overload')
        if math.isfinite(float(baseline_effort)):
            effort_delta = abs(effort - float(baseline_effort))
            if effort_delta > maximum_effort_delta:
                return failed('effort_delta_overload')

    left, left_error = _prepare_history(
        left_samples,
        side='left',
        now_ns=now_ns,
        maximum_age_seconds=maximum_age_seconds,
    )
    if left_error is not None:
        return failed('force_history_invalid')
    right, right_error = _prepare_history(
        right_samples,
        side='right',
        now_ns=now_ns,
        maximum_age_seconds=maximum_age_seconds,
    )
    if right_error is not None:
        return failed('force_history_invalid')
    assert left is not None and right is not None
    if left.samples:
        left_force = float(left.samples[-1].force_newtons)
    if right.samples:
        right_force = float(right.samples[-1].force_newtons)

    if any(
        float(sample.force_newtons) > maximum_force
        for sample in left.recent
    ):
        return failed('force_overload')
    if any(
        float(sample.force_newtons) > maximum_force
        for sample in right.recent
    ):
        return failed('force_overload')
    left_run, left_gap = _qualifying_suffix(
        left.recent,
        minimum_force=minimum_force,
        maximum_gap_seconds=maximum_gap_seconds,
    )
    right_run, right_gap = _qualifying_suffix(
        right.recent,
        minimum_force=minimum_force,
        maximum_gap_seconds=maximum_gap_seconds,
    )
    left_count = len(left_run)
    right_count = len(right_run)
    if width < minimum_width:
        return failed('width_too_narrow')
    if width >= maximum_width:
        return failed('width_too_wide')
    if abs(velocity) > maximum_velocity:
        return failed('joint_still_moving')

    if not left.samples and not right.samples:
        return failed('bilateral_contact_missing')
    if not left.samples:
        return failed('unilateral_contact')
    if not right.samples:
        return failed('unilateral_contact')
    if not left.recent and not right.recent:
        return failed('bilateral_contact_stale')
    if not left.recent:
        return failed('unilateral_contact')
    if not right.recent:
        return failed('unilateral_contact')
    left_current = (
        now_ns - int(left.recent[-1].stamp_ns)
    ) / 1e9 <= maximum_gap_seconds
    right_current = (
        now_ns - int(right.recent[-1].stamp_ns)
    ) / 1e9 <= maximum_gap_seconds
    if not left_current and not right_current:
        return failed('bilateral_contact_stale')
    if not left_current or not right_current:
        return failed('unilateral_contact')

    if not left_run:
        return failed('left_force_below_minimum')
    if not right_run:
        return failed('right_force_below_minimum')
    if left_count < minimum_samples:
        return failed(
            'force_sample_gap' if left_gap else 'insufficient_force_samples'
        )
    if right_count < minimum_samples:
        return failed(
            'force_sample_gap' if right_gap else 'insufficient_force_samples'
        )

    left_window = left_run[-int(minimum_samples):]
    right_window = right_run[-int(minimum_samples):]
    left_span = (
        int(left_run[-1].stamp_ns) - int(left_run[0].stamp_ns)
    ) / 1e9
    right_span = (
        int(right_run[-1].stamp_ns) - int(right_run[0].stamp_ns)
    ) / 1e9
    if left_span < minimum_span_seconds:
        return failed('force_sample_span_too_short')
    if right_span < minimum_span_seconds:
        return failed('force_sample_span_too_short')
    if any(
        abs(int(left_sample.stamp_ns) - int(right_sample.stamp_ns)) / 1e9
        > maximum_side_skew_seconds
        for left_sample, right_sample in zip(left_window, right_window)
    ):
        return failed('bilateral_sample_skew')

    return _evidence(
        True,
        'bilateral_contact_verified',
        width=width,
        left_force=left_force,
        right_force=right_force,
        effort=effort,
        effort_delta=effort_delta,
        left_samples=left_count,
        right_samples=right_count,
    )
