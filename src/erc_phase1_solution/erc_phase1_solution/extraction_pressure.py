"""Pure, bounded pressure maintenance at checked extraction endpoints.

This module publishes nothing and provides no geometric or physical-retention
proof. The adapter owns public position/action commands, a single command lock,
exact named-contact aggregation, continuous fault latches and checked arm paths.
It must consume permits under that lock using the same immutable snapshot and
report publication/action failures. Contact loss never becomes reacquisition.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Optional, Tuple

from .adaptive_grasp import (
    AdaptiveGraspEvidence, ForceSample, GripperFeedback, evaluate_bilateral_contact,
)
from .runtime_utils import book_model_name


PRESSURE_DEFICIENCIES = frozenset({
    'bilateral_contact_stale', 'unilateral_contact',
    'left_force_below_minimum', 'right_force_below_minimum',
    'insufficient_force_samples', 'force_sample_span_too_short',
    'force_sample_gap', 'bilateral_sample_skew',
})


def _real(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _stamp(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class ExtractionPressureLimits:
    minimum_force: float = 3.5
    maximum_force: float = 20.0
    maximum_effort: float = 4.0
    maximum_effort_delta: float = 3.0
    minimum_width: float = .0175
    maximum_width: float = .05
    correction_step: float = .0000001
    maximum_corrections: int = 20
    correction_travel: float = .000002
    global_preload_travel: float = .000002
    motion_seconds: float = .20
    dwell_seconds: float = .20
    pressure_recheck_seconds: float = .05
    maximum_age_seconds: float = .15
    maximum_gap_seconds: float = .075
    minimum_span_seconds: float = .05
    maximum_side_skew_seconds: float = .05
    minimum_samples: int = 3
    gripper_position_tolerance: float = .000000025
    gripper_stationary_velocity: float = .000002
    arm_position_tolerance: float = .005
    torso_position_tolerance: float = .001
    arm_stationary_velocity: float = .001
    step_wall_seconds: float = 5.
    total_wall_seconds: float = 120.
    clock_stall_wall_seconds: float = 1.
    maximum_segment_distance: float = .001
    maximum_extraction_distance: float = .005

    def __post_init__(self):
        if not all(_real(value) and value > 0 for value in self.__dict__.values()):
            raise ValueError('Extraction pressure limits must be positive finite numbers')
        if not (
            .05 <= self.minimum_force < self.maximum_force <= 20.
            and self.maximum_effort <= 4. and self.maximum_effort_delta <= 3.
            and .016 <= self.minimum_width < self.maximum_width <= .069
            and self.correction_step <= .0000001
            and isinstance(self.maximum_corrections, int)
            and self.maximum_corrections <= 20
            and self.correction_travel <= .000002
            and self.global_preload_travel <= .000002
            and self.maximum_corrections*self.correction_step <= self.correction_travel+1e-15
            and .20 <= self.motion_seconds <= 1.
            and .20 <= self.dwell_seconds <= .5
            and .05 <= self.pressure_recheck_seconds <= .15
            and self.maximum_age_seconds <= .15
            and self.maximum_gap_seconds <= min(.075, self.maximum_age_seconds)
            and .05 <= self.minimum_span_seconds <= self.maximum_age_seconds
            and self.maximum_side_skew_seconds <= min(.05, self.maximum_age_seconds)
            and isinstance(self.minimum_samples, int) and self.minimum_samples >= 3
            and self.gripper_position_tolerance <= .000000025
            and self.gripper_position_tolerance < self.correction_step
            and self.gripper_stationary_velocity <= .000002
            and self.arm_position_tolerance <= .005
            and self.torso_position_tolerance <= .001
            and self.arm_stationary_velocity <= .001
            and self.step_wall_seconds <= 5. and self.total_wall_seconds <= 120.
            and self.clock_stall_wall_seconds <= 1.
            and self.maximum_segment_distance <= .001
            and self.maximum_segment_distance <= self.maximum_extraction_distance <= .01
        ):
            raise ValueError('Extraction pressure limits exceed the diagnostic bounds')


@dataclass(frozen=True)
class NamedForceSample:
    model: str
    stamp_ns: int
    force_newtons: float


@dataclass(frozen=True)
class ArmFeedback:
    stamp_ns: int
    positions: Tuple[float, ...]  # Torso, then the seven left arm joints.
    velocities: Tuple[float, ...]


@dataclass(frozen=True)
class ExtractionSnapshot:
    now_ns: int
    wall_seconds: float
    model: str
    gripper: GripperFeedback
    arm: ArmFeedback
    left: Tuple[NamedForceSample, ...]
    right: Tuple[NamedForceSample, ...]
    transform_valid: bool  # Invalidation input; contact is not a rigidity proof.
    hazard: Optional[str] = None


@dataclass(frozen=True)
class ExtractionDecision:
    action: str
    reason: str
    token: Optional[int] = None
    position: Optional[float] = None
    motion_seconds: Optional[float] = None
    evidence: Optional[AdaptiveGraspEvidence] = None
    snapshot: Optional[ExtractionSnapshot] = None


class ExtractionPressureMaintenance:
    """Endpoint-only corrections and single-use fresh arm permissions.

    `observe` never authorizes publishing a gripper command. A CLOSE_STEP request
    must be consumed with `authorize_correction`, under the adapter command lock.
    READY_FOR_ARM must similarly be consumed with `arm_started`; only its returned
    ARM_PERMIT authorizes a checked segment. Both consume the exact snapshot which
    produced the request (`decision.snapshot`), plus a freshly read live snapshot
    containing current clocks and faults, after rechecking all guards.
    Observe and consume must run without awaiting or releasing the adapter lock;
    a later callback must invalidate the permit through observe/stop first.
    Publication failure calls `stop`; action completion calls `arm_finished`.
    """
    def __init__(self, *, expected_model, acquired_position,
                 acquisition_reference_position, acquisition_microsteps_used,
                 arm_endpoint, started_ros_ns,
                 started_wall_seconds, baseline_effort=math.nan, limits=None):
        self.limits = limits or ExtractionPressureLimits()
        if not isinstance(expected_model, str) or book_model_name(expected_model) != expected_model:
            raise ValueError('A concrete expected book identity is required')
        if not all(_real(v) for v in (acquired_position, acquisition_reference_position,
                                     started_wall_seconds)) or not _stamp(started_ros_ns):
            raise ValueError('Invalid acquisition reference or clock')
        self.expected_model = expected_model
        self.acquired_position = float(acquired_position)
        self.acquisition_reference_position = float(acquisition_reference_position)
        if (not isinstance(acquisition_microsteps_used, int) or isinstance(acquisition_microsteps_used, bool)
                or not 0 <= acquisition_microsteps_used <= self.limits.maximum_corrections):
            raise ValueError('Acquisition microstep count must preserve the combined command budget')
        self.acquisition_microsteps_used = acquisition_microsteps_used
        self.minimum_position = max(
            self.limits.minimum_width,
            self.acquired_position-self.limits.correction_travel,
            self.acquisition_reference_position-self.limits.global_preload_travel)
        if not (self.minimum_position <= self.acquired_position < self.limits.maximum_width
                and self.acquired_position <= self.acquisition_reference_position+self.limits.gripper_position_tolerance):
            raise ValueError('Acquisition has exhausted or invalidated its immutable budget')
        if not isinstance(baseline_effort, Real) or isinstance(baseline_effort, bool) or math.isinf(baseline_effort):
            raise ValueError('Invalid baseline effort')
        self.baseline_effort = baseline_effort
        self.arm_endpoint = self._endpoint(arm_endpoint)
        self.target = self.acquired_position
        self.phase = 'settle'
        self.fault = None
        self.corrections = 0
        self.extraction_distance = 0.
        self._last_ros = int(started_ros_ns)
        self._last_wall = float(started_wall_seconds)
        self._clock_progress_wall = self._last_wall
        self._total_wall_end = self._last_wall+self.limits.total_wall_seconds
        self._step_wall_end = self._last_wall+self.limits.step_wall_seconds
        self._motion_end = int(started_ros_ns)
        self._stationary_epoch = None
        self._last_stationary_feedback = None
        self._recheck_end = None
        self._token = 0
        self._request = None
        self._request_snapshot = None
        self._pending_future = None
        self._pending_future_wall = None
        self._active_arm_token = None
        self._active_arm_distance = None

    @staticmethod
    def _endpoint(values):
        result = tuple(values)
        if len(result) != 8 or not all(_real(v) for v in result):
            raise ValueError('Arm endpoint requires eight finite joint positions')
        return result

    def stop(self, reason):
        if self.fault is None:
            self.fault = str(reason) or 'unspecified_stop'
        self.phase = 'stopped'
        self._request = self._request_snapshot = None
        return ExtractionDecision('STOP', self.fault)

    def _validate(self, s):
        if not isinstance(s, ExtractionSnapshot) or not _stamp(s.now_ns) or not _real(s.wall_seconds):
            return 'invalid_snapshot'
        if s.now_ns < self._last_ros or s.wall_seconds < self._last_wall:
            return 'clock_reversed'
        if s.now_ns > self._last_ros:
            self._clock_progress_wall = s.wall_seconds
        self._last_ros, self._last_wall = int(s.now_ns), s.wall_seconds
        if s.wall_seconds >= self._total_wall_end:
            return 'total_wall_timeout'
        if s.wall_seconds-self._clock_progress_wall >= self.limits.clock_stall_wall_seconds:
            return 'clock_stalled'
        if s.hazard:
            return str(s.hazard)
        if s.model != self.expected_model:
            return 'identity_changed'
        if s.transform_valid is not True:
            return 'payload_transform_ambiguous'
        g, a = s.gripper, s.arm
        if not isinstance(g, GripperFeedback) or not isinstance(a, ArmFeedback):
            return 'feedback_unavailable'
        if not _stamp(g.stamp_ns) or not _stamp(a.stamp_ns):
            return 'invalid_feedback_stamp'
        if not all(_real(v) for v in (g.position, g.velocity)):
            return 'invalid_gripper_feedback'
        if not (self.minimum_position-1e-15 <= g.position < self.limits.maximum_width):
            return 'measured_travel_or_width_limit'
        if (not isinstance(g.effort, Real) or isinstance(g.effort, bool)
                or math.isinf(g.effort)):
            return 'invalid_effort_feedback'
        if math.isfinite(g.effort):
            if abs(g.effort) > self.limits.maximum_effort:
                return 'effort_overload'
            if math.isfinite(self.baseline_effort) and abs(g.effort-self.baseline_effort) > self.limits.maximum_effort_delta:
                return 'effort_delta_overload'
        if (not isinstance(a.positions, tuple) or not isinstance(a.velocities, tuple)
                or len(a.positions) != 8 or len(a.velocities) != 8
                or not all(_real(v) for v in (*a.positions, *a.velocities))):
            return 'invalid_arm_feedback'
        stamps = [g.stamp_ns, a.stamp_ns]
        for side, history in (('left', s.left), ('right', s.right)):
            if not isinstance(history, tuple):
                return 'invalid_force_history'
            if not history:
                return 'contact_lost:'+side
            previous = -1
            for sample in history:
                if not isinstance(sample, NamedForceSample) or sample.model != self.expected_model:
                    return 'unexpected_contact_identity'
                if not _stamp(sample.stamp_ns) or sample.stamp_ns <= previous:
                    return 'invalid_force_chronology'
                if not _real(sample.force_newtons) or sample.force_newtons < 0:
                    return 'invalid_force'
                if sample.force_newtons > self.limits.maximum_force:
                    return 'force_overload'
                if sample.stamp_ns > s.now_ns+100_000_000:
                    return 'future_force_beyond_bound'
                previous = sample.stamp_ns
            stamps.append(history[-1].stamp_ns)
        for stamp in stamps:
            if s.now_ns-stamp > int(self.limits.maximum_age_seconds*1e9):
                return 'contact_or_feedback_stale'
            if stamp > s.now_ns+100_000_000:
                return 'future_feedback_beyond_bound'
        return None

    def _stationary(self, s):
        a, g, l = s.arm, s.gripper, self.limits
        return bool(
            abs(a.positions[0]-self.arm_endpoint[0]) <= l.torso_position_tolerance
            and max(abs(x-y) for x, y in zip(a.positions[1:], self.arm_endpoint[1:])) <= l.arm_position_tolerance
            and max(abs(v) for v in a.velocities) <= l.arm_stationary_velocity
            and abs(g.position-self.target) <= l.gripper_position_tolerance
            and abs(g.velocity) <= l.gripper_stationary_velocity)

    def _confirmed(self, s):
        error = self._validate(s)
        if error:
            self.stop(error)
            return None
        # Observe genuine mechanical changes even while clock ordering catches
        # up with one fixed future snapshot. Never hide a new outlier behind it.
        if self.phase != 'arm_running' and not self._stationary(s):
            self._stationary_epoch = self._last_stationary_feedback = None
            self._request = self._request_snapshot = None
            self._recheck_end = None
            self._pending_future = self._pending_future_wall = None
            self.phase = 'settle'
        latest = max(s.arm.stamp_ns, s.gripper.stamp_ns, s.left[-1].stamp_ns, s.right[-1].stamp_ns)
        if self._pending_future is None and latest > s.now_ns:
            self._pending_future, self._pending_future_wall = s, s.wall_seconds
        if self._pending_future is not None:
            fixed = self._pending_future
            last = max(fixed.arm.stamp_ns, fixed.gripper.stamp_ns,
                       fixed.left[-1].stamp_ns, fixed.right[-1].stamp_ns)
            if s.wall_seconds-self._pending_future_wall >= .5:
                self.stop('future_snapshot_timeout')
                return None
            if last > s.now_ns:
                return None
            self._pending_future = self._pending_future_wall = None
            return replace(fixed, now_ns=s.now_ns, wall_seconds=s.wall_seconds)
        return s

    def _pressure(self, s):
        epoch = self._stationary_epoch
        l = self.limits
        histories = [tuple(ForceSample(x.stamp_ns, x.force_newtons) for x in values
                           if x.stamp_ns >= epoch) for values in (s.left, s.right)]
        return evaluate_bilateral_contact(
            *histories, s.gripper, s.now_ns, minimum_width=self.minimum_position,
            maximum_width=l.maximum_width, minimum_force=l.minimum_force,
            maximum_force=l.maximum_force, minimum_samples=l.minimum_samples,
            maximum_age_seconds=l.maximum_age_seconds, maximum_gap_seconds=l.maximum_gap_seconds,
            minimum_span_seconds=l.minimum_span_seconds, maximum_side_skew_seconds=l.maximum_side_skew_seconds,
            maximum_velocity=l.gripper_stationary_velocity, maximum_effort=l.maximum_effort,
            baseline_effort=self.baseline_effort, maximum_effort_delta=l.maximum_effort_delta)

    def _request_for(self, action, s, *, evidence=None, position=None):
        if self._request is None or self._request_snapshot != s or self._request.action != action:
            self._token += 1
            self._request = ExtractionDecision(action, 'fresh_stationary_pressure' if action == 'READY_FOR_ARM'
                else 'bounded_pressure_correction', self._token, position,
                self.limits.motion_seconds if position is not None else None, evidence, s)
            self._request_snapshot = s
        return self._request

    def observe(self, snapshot):
        if self.fault:
            return ExtractionDecision('STOP', self.fault)
        s = self._confirmed(snapshot)
        if s is None:
            return ExtractionDecision('STOP' if self.fault else 'WAIT', self.fault or 'future_snapshot_pending')
        if s.wall_seconds >= self._step_wall_end:
            return self.stop('arm_segment_wall_timeout' if self.phase == 'arm_running'
                             else 'stationary_step_wall_timeout')
        if self.phase == 'arm_running':
            return ExtractionDecision('MONITOR', 'arm_segment_running')
        if s.now_ns < self._motion_end or not self._stationary(s):
            self._stationary_epoch = self._last_stationary_feedback = None
            return ExtractionDecision('WAIT', 'motion_or_settling')
        confirmed_ns = min(s.arm.stamp_ns, s.gripper.stamp_ns)
        if confirmed_ns < self._motion_end:
            return ExtractionDecision('WAIT', 'pre_motion_feedback')
        if (self._last_stationary_feedback is not None and
                (confirmed_ns < self._last_stationary_feedback or
                 confirmed_ns-self._last_stationary_feedback > int(self.limits.maximum_gap_seconds*1e9))):
            self._stationary_epoch = self._recheck_end = None
        self._last_stationary_feedback = confirmed_ns
        if self._stationary_epoch is None:
            self._stationary_epoch = confirmed_ns
        if confirmed_ns-self._stationary_epoch < int(self.limits.dwell_seconds*1e9):
            return ExtractionDecision('WAIT', 'stationary_dwell')
        evidence = self._pressure(s)
        # A frozen confirmed window must not ignore a known newer pressure
        # change just because that producer frame still leads local /clock.
        live_above = min(snapshot.left[-1].force_newtons, snapshot.right[-1].force_newtons) >= self.limits.minimum_force
        if ((evidence.verified and not live_above)
                or (not evidence.verified and live_above and snapshot != s)):
            self._request = self._request_snapshot = None
            return ExtractionDecision('WAIT', 'newer_pressure_transition_pending', evidence=evidence)
        if evidence.verified:
            if self.extraction_distance >= self.limits.maximum_extraction_distance-1e-15:
                self.phase = 'complete'
                self._request = self._request_snapshot = None
                return ExtractionDecision('PREFIX_VERIFIED', 'bounded_prefix_only', evidence=evidence)
            self.phase = 'ready'
            self._recheck_end = None
            return self._request_for('READY_FOR_ARM', s, evidence=evidence)
        self._request = self._request_snapshot = None
        if evidence.reason not in PRESSURE_DEFICIENCIES:
            return self.stop('pressure_evidence_invalid:'+evidence.reason)
        if self._recheck_end is None:
            self._recheck_end = s.now_ns+int(self.limits.pressure_recheck_seconds*1e9)
        if s.now_ns < self._recheck_end:
            return ExtractionDecision('WAIT', 'same_target_pressure_recheck', evidence=evidence)
        target = self.acquired_position-(self.corrections+1)*self.limits.correction_step
        if (self.acquisition_microsteps_used+self.corrections >= self.limits.maximum_corrections
                or target < self.minimum_position-1e-15):
            return self.stop('cumulative_correction_budget_exhausted')
        self.phase = 'correction_pending'
        return self._request_for('CLOSE_STEP', s, evidence=evidence, position=target)

    def _consume(self, action, token, snapshot, live):
        if self.fault:
            return ExtractionDecision('STOP', self.fault)
        error = self._validate(live)
        if error:
            return self.stop(error)
        if (not isinstance(token, int) or isinstance(token, bool)
                or self._request is None or self._request.action != action or self._request.token != token
                or self._request_snapshot != snapshot or not self._stationary(snapshot)
                or not self._stationary(live)):
            return self.stop('stale_or_reused_permission')
        # Age the immutable evidence against current clocks, not clocks embedded
        # in an old permit. Future live readings may invalidate physical safety,
        # but do not become evidence certifying this command.
        current_certificate = replace(snapshot, now_ns=live.now_ns, wall_seconds=live.wall_seconds)
        error = self._validate(current_certificate)
        if error:
            return self.stop(error)
        if live.wall_seconds >= self._step_wall_end:
            return self.stop('stationary_step_wall_timeout')
        latest_above = min(live.left[-1].force_newtons, live.right[-1].force_newtons) >= self.limits.minimum_force
        if action == 'READY_FOR_ARM':
            if not latest_above or not self._pressure(current_certificate).verified:
                return self.stop('pressure_permission_invalidated')
        elif latest_above:
            return self.stop('pressure_correction_superseded')
        request = self._request
        self._request = self._request_snapshot = None
        return request

    def authorize_correction(self, token, snapshot, *, live):
        request = self._consume('CLOSE_STEP', token, snapshot, live)
        if request.action == 'STOP':
            return request
        self.corrections += 1
        self.target = request.position
        self.phase = 'gripper_running'
        self._motion_end = live.now_ns+int(self.limits.motion_seconds*1e9)
        self._step_wall_end = min(self._total_wall_end, live.wall_seconds+self.limits.step_wall_seconds)
        self._stationary_epoch = self._last_stationary_feedback = self._recheck_end = None
        self._pending_future = self._pending_future_wall = None
        return request

    def arm_started(self, token, snapshot, *, distance, live):
        request = self._consume('READY_FOR_ARM', token, snapshot, live)
        if request.action == 'STOP':
            return request
        if (not _real(distance) or not 0 < distance <= self.limits.maximum_segment_distance
                or self.extraction_distance+distance > self.limits.maximum_extraction_distance+1e-15):
            return self.stop('extraction_distance_limit')
        self.phase = 'arm_running'
        self._step_wall_end = min(self._total_wall_end, live.wall_seconds+self.limits.step_wall_seconds)
        self._active_arm_token, self._active_arm_distance = token, float(distance)
        self._pending_future = self._pending_future_wall = None
        return ExtractionDecision('ARM_PERMIT', 'one_checked_segment', token=token)

    def arm_finished(self, token, snapshot, *, endpoint, terminal_confirmed, succeeded):
        if self.fault:
            return ExtractionDecision('STOP', self.fault)
        if self.phase != 'arm_running' or token != self._active_arm_token:
            return self.stop('unexpected_arm_terminal')
        error = self._validate(snapshot)
        if error:
            return self.stop(error)
        if terminal_confirmed is not True or succeeded is not True:
            return self.stop('unconfirmed_or_failed_arm_segment')
        try:
            self.arm_endpoint = self._endpoint(endpoint)
        except (ValueError, TypeError):
            return self.stop('invalid_arm_endpoint')
        self.extraction_distance += self._active_arm_distance
        self._active_arm_token = self._active_arm_distance = None
        self.phase = 'settle'
        self._motion_end = snapshot.now_ns
        self._step_wall_end = min(self._total_wall_end, snapshot.wall_seconds+self.limits.step_wall_seconds)
        self._stationary_epoch = self._last_stationary_feedback = self._recheck_end = None
        self._pending_future = self._pending_future_wall = None
        return ExtractionDecision('WAIT', 'fresh_post_arm_stationary_pressure_required')
