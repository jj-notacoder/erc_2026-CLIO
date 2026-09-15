"""Opt-in bounded fine closure for the unchanged official position gripper.

Derived from the calibration03 sequence; pressure acquisition is not retention.
The owner supplies normal sensor/contact identity and checked recovery. This
controller never commands arm/base motion and never clears an overload latch.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import math
import time

from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .adaptive_grasp import evaluate_bilateral_contact
from .runtime_utils import book_model_name


# These are well-formed failures to certify acquisition, not permission by
# themselves to move. A next unacquired probe additionally needs the independent
# live identity/stream/feedback/force/travel gates and a stationary fixed target.
# Unknown evaluator outcomes fail closed.
MICRO_PRESSURE_DEFICIENCIES = frozenset({
    'bilateral_contact_missing', 'bilateral_contact_stale', 'unilateral_contact',
    'left_force_below_minimum', 'right_force_below_minimum',
    'insufficient_force_samples', 'force_sample_span_too_short',
    'force_sample_gap', 'bilateral_sample_skew',
})


class _MicroEpochInvalidated(Exception):
    """Internal worker outcome, never a cleared actuator/fault stop."""


@dataclass(frozen=True)
class FineGripLimits:
    preclose: float = .025
    fine_start: float = .019
    floor: float = .0175
    coarse_step: float = .001
    fine_step: float = .000025
    touched_step: float = .000005
    motion_seconds: float = .20
    dwell_seconds: float = .20
    # Explicit slow-simulation allowance; motion and fresh ROS dwell unchanged.
    step_wall_seconds: float = 5.0
    preclose_wall_seconds: float = 10.0
    total_wall_seconds: float = 300.0
    feedback_age_seconds: float = .15
    stationary_velocity: float = .000002
    stationary_position_error: float = .00000025
    # Software policy for this simulated position mechanism; no asset change.
    maximum_force: float = 8.0
    micro_preload_steps: int = 20
    micro_preload_step: float = .0000001
    micro_preload_limit: float = .000002
    # Only preload inward trajectories use this; ordinary seek/holds are unchanged.
    micro_motion_seconds: float = .20
    micro_position_error: float = .000000025
    # Separate loaded microtarget envelope; ordinary closing remains stricter.
    micro_stationary_velocity: float = .000002
    minimum_force: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Fine-grip limits must be finite')
        if not (.069 > self.preclose >= self.fine_start > self.floor >= .016
                and 0 < self.coarse_step <= .001
                and 0 < self.touched_step <= .000005
                and self.touched_step <= self.fine_step <= .000025
                and .20 <= self.motion_seconds <= 1.
                and .20 <= self.micro_motion_seconds <= 1.
                and .15 <= self.dwell_seconds <= .5
                and 0 < self.step_wall_seconds <= 15.
                and 0 < self.preclose_wall_seconds <= 10.
                # Unscored slow-simulation diagnostics may opt into more wall
                # time; individual steps and all mechanical bounds stay fixed.
                and 0 < self.total_wall_seconds <= 1500.
                and 0 < self.feedback_age_seconds <= .15
                and 0 < self.stationary_velocity <= .000002
                and 0 < self.stationary_position_error <= .00000025
                and 0 < self.maximum_force <= 30.
                and isinstance(self.micro_preload_steps, int)
                and not isinstance(self.micro_preload_steps, bool)
                and 0 <= self.micro_preload_steps <= 300
                and 0 < self.micro_preload_step <= .0000005
                and 0 < self.micro_preload_limit <= .000075
                and self.micro_preload_steps * self.micro_preload_step <= self.micro_preload_limit + 1e-15
                and 0 < self.micro_position_error <= .000000050
                and self.micro_position_error < self.micro_preload_step
                and 0 < self.micro_stationary_velocity <= .000010
                and .05 <= self.minimum_force < self.maximum_force):
            raise ValueError('Fine-grip limits exceed bounded fine-grip profile')



def producer_stamp(message):
    stamp = getattr(getattr(message, 'header', None), 'stamp', None)
    if stamp is None:
        return None
    sec, nsec = getattr(stamp, 'sec', -1), getattr(stamp, 'nanosec', -1)
    if not isinstance(sec, int) or not isinstance(nsec, int):
        return None
    return sec * 1_000_000_000 + nsec if sec >= 0 and 0 <= nsec < 1_000_000_000 else None



class FineGripperClose:
    def __init__(self, node, limits=None):
        self.node = node
        self.limits = limits or FineGripLimits()
        self._fine_lock = node._adaptive_command_guard()
        self._fine_stop = None
        self._fine_fault = None
        self._fine_fault_snapshot = None
        self._fine_fault_reported = False
        self._fine_micro_active = False
        self._fine_side_last = {}
        self._fine_first_contact = None
        self._fine_coarse_contact_model = None
        self._fine_command = float(node.gripper_open)
        self._fine_motion_end_ns = 0
        self._fine_phase = 'idle'
        self._fine_wall_deadline = time.monotonic() + self.limits.total_wall_seconds
        self._fine_evidence = None
        self._fine_result = None
        self._fine_resettle_enabled = False
        self.expected_model = getattr(node, '_target_book_model', None)
        if self.expected_model is not None and not self._valid_model(self.expected_model):
            raise ValueError('Fine-grip target latch must be a concrete matching-colour book')
        self.active = True

    def __getattr__(self, name):
        return getattr(self.node, name)

    def _valid_model(self, model):
        return (isinstance(model, str) and book_model_name(model) == model
                and model.endswith('_' + self.target_colour.lower()))

    def _emit(self, kind, **fields):
        self.node._publish_status('fine_gripper_progress', stage=kind, **fields)

    def _publish_once_locked(self, position, duration, kind):
        """Caller owns _fine_lock: a stopped close can never overwrite its hold."""
        if kind == 'close':
            fault = (self._fine_fault or self._adaptive_overload_reason()
                     or getattr(self.node, '_adaptive_motion_halt_reason', None))
            if fault:
                self._hold(fault)
                return False
            if self._fine_stop is not None:
                return False
            error = self._feedback_error()
            if error is None and self._fine_first_contact is not None and self._fine_phase == 'coarse':
                error = 'coarse_command_after_contact'
            if (error is None and self._fine_coarse_contact_model is not None
                    and self._fine_phase == 'fine'
                    and not -1e-15 <= self._fine_command-position <= self.limits.touched_step+1e-15):
                error = 'coarse_contact_handoff_step_limit'
            if error:
                self._hold(error)
                return False
        if (not math.isfinite(position) or not 0 <= position <= .069
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError('Invalid public gripper position')
        message = JointTrajectory(joint_names=['gripper_left_finger_joint'])
        point = JointTrajectoryPoint(positions=[float(position)], velocities=[0.0])
        point.time_from_start = Duration(seconds=duration).to_msg()
        message.points = [point]
        self._fine_motion_end_ns = int(self.get_clock().now().nanoseconds) + int(duration * 1e9)
        self.node._adaptive_motion_started = True
        self._fine_command = float(position)
        self.gripper_pub.publish(message)
        self._emit('public_position_command', purpose=kind, position=position,
                   motion_seconds=duration, phase=self._fine_phase)
        return True

    def _hold(self, reason):
        """Normal first-touch hold is resumable; any fault is permanently latched."""
        with self._fine_lock:
            if reason not in ('bilateral_contact_observed', 'coarse_contact_observed'):
                self._fine_fault = self._fine_fault or str(reason)
                self._fine_stop = self._fine_fault
                if self._fine_fault_snapshot is None:
                    self._fine_fault_snapshot = self._fault_snapshot_locked(
                        self._fine_fault, feedback=self._latest_gripper_feedback(),
                        now=int(self.get_clock().now().nanoseconds))
                self.node._hold_adaptive_gripper(self._fine_fault)
                if not self._fine_fault_reported:
                    # Actuator interruption precedes diagnostic publication.
                    # Mark first so a publication exception cannot recursively
                    # emit another fault from the contact callback's handler.
                    self._fine_fault_reported = True
                    self._emit('fault_snapshot', **self._fine_fault_snapshot)
                return
            # Bilateral contact can arrive while the first coarse-contact hold
            # is settling. Upgrade that normal hold, never an independent fault.
            if self._fine_fault or (self._fine_stop is not None and not (
                    self._fine_stop == 'coarse_contact_observed'
                    and reason == 'bilateral_contact_observed')):
                return
            feedback, error = self.node._adaptive_motion_feedback()
            if error:
                self._hold(error)
                return
            # A fault can already be latched by a callback waiting for this lock.
            fault = (self._adaptive_overload_reason()
                     or getattr(self.node, '_adaptive_motion_halt_reason', None))
            if fault:
                self._hold(fault)
                return
            self._fine_stop = reason
            if reason == 'coarse_contact_observed':
                self._fine_coarse_contact_model = self.expected_model
            stage = ('first_coarse_contact_hold' if reason == 'coarse_contact_observed'
                     else 'first_bilateral_hold')
            self._publish_once_locked(float(feedback.position), .02, stage)
            self._emit(stage, measured_position=feedback.position)

    def observe_contacts(self, message):
        with self._fine_lock:
            self._observe_contacts_locked(message)

    def _observe_contacts_locked(self, message):
        if not self.active:
            return
        stamp = producer_stamp(message)
        now = int(self.get_clock().now().nanoseconds)
        reason = None
        by_side = {}
        if stamp is None or stamp <= 0 or not -100_000_000 <= now - stamp <= 150_000_000:
            reason = 'invalid_or_stale_contact_stamp'
        for contact in getattr(message, 'contacts', ()):
            a, b = contact.collision1.name, contact.collision2.name
            left_a, left_b = 'gripper_left_' in a, 'gripper_left_' in b
            if not (left_a or left_b) or (left_a and left_b):
                continue
            grip, other = (a, b) if left_a else (b, a)
            model = book_model_name(other)
            force = self.node._fine_contact_force_magnitude(contact, gripper_is_collision1=left_a)
            side = ('left' if any(s in grip for s in ('fingertip_left_', 'finger_left_'))
                    else 'right' if any(s in grip for s in ('fingertip_right_', 'finger_right_'))
                    else None)
            if self._valid_model(model) and self.expected_model is None:
                self.expected_model = model
            if model != self.expected_model or not self._valid_model(model):
                reason = 'unexpected_or_anonymous_hand_contact'
            elif side is None:
                reason = 'unexpected_target_palm_contact'
            elif force is None or not math.isfinite(force):
                reason = 'invalid_contact_force'
            else:
                # Duplicate collision records in one message are not extra load.
                pairs = by_side.setdefault(side, {})
                pair = tuple(sorted((a, b)))
                pairs[pair] = max(pairs.get(pair, 0.), force)
        feedback, feedback_error = self.node._adaptive_motion_feedback()
        if feedback_error:
            self._hold(feedback_error)
            return
        with self._fine_lock:
            for side, pairs in by_side.items():
                force = sum(pairs.values())
                if stamp is not None:
                    previous_side = self._fine_side_last.get(side)
                    if previous_side is None or stamp >= previous_side[0]:
                        self._fine_side_last[side] = (stamp, force)
                if self._fine_first_contact is None:
                    self._fine_first_contact = float(feedback.position)
                if force > self.limits.maximum_force:
                    reason = 'force_overload'
            overload = self._adaptive_overload_reason()
            if overload:
                reason = overload
            if reason:
                self._hold(reason)
            elif by_side and self._fine_phase == 'preclose':
                self._hold('contact_during_coarse_seek')
            else:
                if (not getattr(self, '_fine_micro_active', False)
                        and all(side in self._fine_side_last for side in ('left', 'right'))):
                    left, right = self._fine_side_last['left'], self._fine_side_last['right']
                    if (abs(left[0] - right[0]) <= 50_000_000
                            and 0 <= max(stamp, now) - min(left[0], right[0]) <= 150_000_000
                            and min(left[1], right[1]) >= self.adaptive_contact_force_minimum):
                        self._hold('bilateral_contact_observed')
                if by_side and self._fine_phase == 'coarse' and self._fine_stop is None:
                    self._hold('coarse_contact_observed')

    def _fault_snapshot_locked(self, reason, *, feedback, now):
        """Capture the fixed rejection inputs while the command lock is held."""
        sides = {}
        for side in ('left', 'right'):
            observation = self._fine_side_last.get(side)
            stamp, force = observation if observation is not None else (None, None)
            fresh = stamp is not None and stamp > 0 and -100_000_000 <= now-stamp <= 150_000_000
            finite_force = force is not None and math.isfinite(force) and force >= 0
            sides[side] = dict(producer_stamp_ns=stamp,
                               age_seconds=(now-stamp)/1e9 if stamp is not None else None,
                               force_newtons=force if finite_force else None,
                               force_finite_nonnegative=finite_force,
                               fresh_named_pair=bool(fresh and finite_force),
                               minimum_contact_force_met=bool(fresh and finite_force
                                   and force >= self.adaptive_contact_force_minimum))
        return dict(reason=str(reason), evaluation_ros_ns=now, phase=self._fine_phase,
                    commanded_position=self._fine_command,
                    command_motion_end_ros_ns=self._fine_motion_end_ns,
                    micro_active=bool(self._fine_micro_active),
                    micro_reference_position=getattr(self, '_fine_micro_reference', None),
                    feedback=asdict(feedback) if feedback is not None else None,
                    expected_model=self.expected_model,
                    latched_model=self._target_book_model,
                    side_observations=sides,
                    contact_minimum_force=self.adaptive_contact_force_minimum,
                    acquisition_minimum_force=self.limits.minimum_force,
                    maximum_force=self.limits.maximum_force,
                    acquired=False)

    def _feedback_error(self):
        with self._fine_lock:
            return self._feedback_error_locked()

    def _feedback_error_locked(self):
        # Preserve the very feedback/clock/side snapshot which rejected the
        # command, rather than newer callbacks arriving before _hold executes.
        feedback = self._latest_gripper_feedback()
        now = int(self.get_clock().now().nanoseconds)

        def reject(reason):
            if self._fine_fault_snapshot is None:
                self._fine_fault_snapshot = self._fault_snapshot_locked(
                    reason, feedback=feedback, now=now)
            return reason

        fault = (self._fine_fault or self._adaptive_overload_reason()
                 or getattr(self.node, '_adaptive_motion_halt_reason', None))
        if fault:
            return reject(fault)
        if (feedback is None or not all(math.isfinite(v) for v in
                                        (feedback.position, feedback.velocity))
                or not 0 <= feedback.position <= .069):
            return reject('joint_feedback_unavailable')
        if not -100_000_000 <= now - feedback.stamp_ns <= int(self.limits.feedback_age_seconds * 1e9):
            return reject('stale_joint_feedback')
        if self._cancel.is_set():
            return reject('cancelled')
        if time.monotonic() >= self._fine_wall_deadline:
            return reject('fine_grip_wall_timeout')
        if self._fine_coarse_contact_model is not None:
            if (self.expected_model != self._fine_coarse_contact_model
                    or not self._valid_model(self.expected_model)
                    or self._target_book_model not in (None, self.expected_model)):
                return reject('coarse_contact_identity_lost')
            if not any(self._observed_contact_sides(require_minimum_force=False, now=now)):
                return reject('contact_stream_lost')
        if self._fine_first_contact is not None:
            with self._fine_lock:
                last = max((v[0] for v in self._fine_side_last.values()), default=0)
            if not -100_000_000 <= now - last <= 150_000_000:
                return reject('contact_stream_lost')
            if self._fine_first_contact - feedback.position >= (
                    self.adaptive_unilateral_travel_limit - self.adaptive_endpoint_tolerance):
                return reject('unilateral_contact_travel_limit')
        if getattr(self, '_fine_micro_active', False):
            if self._target_book_model != self.expected_model:
                return reject('micro_preload_identity_lost')
            # During this bounded, still-unacquired probe, a pressure dip is
            # allowed only while an exact named pair keeps producing fresh
            # finite contact observations. Resume still requires >=.05 N, and
            # acquisition still requires the configured stationary pressure
            # window. This neither invents zero frames nor permits stale data.
            if not any(self._observed_contact_sides(require_minimum_force=False, now=now)):
                return reject('micro_preload_contact_lost')
            if self._fine_micro_reference - feedback.position > self.limits.micro_preload_limit + 1e-15:
                return reject('micro_preload_measured_travel_limit')
        return None

    def _wait_motion_and_stationary(self, target, *, allow_bilateral_hold=False,
                                    allow_coarse_hold=False,
                                    preclose=False, micro=False, wall_deadline=None):
        limits = self.limits
        wall_limit = limits.preclose_wall_seconds if preclose else limits.step_wall_seconds
        wall_end = min(self._fine_wall_deadline, time.monotonic() + wall_limit)
        if wall_deadline is not None:
            wall_end = min(wall_end, wall_deadline)
        stationary_since = None
        last_clock = int(self.get_clock().now().nanoseconds)
        last_progress_wall = time.monotonic()
        diagnostic_start_ns, diagnostic_start_wall = last_clock, last_progress_wall
        observed_samples = deque(maxlen=256)
        reset_samples = deque(maxlen=32)
        recorded_stamps = set()
        recorded_order = deque(maxlen=256)
        last_feedback_values = None
        last_distinct_stamp = None
        last_confirmed_ns = None
        diagnostics = dict(
            distinct_feedback_stamps=0, duplicate_feedback_polls=0,
            changed_feedback_same_stamp=0, stationary_resets=0,
            stationary_epoch_starts=0, out_of_order_feedback_stamps=0,
            max_stored_feedback_gap_ns=0,
            resets_by_reason={}, nonstationary_new_feedback_by_reason={},
            max_position_error_m=0., max_abs_velocity_mps=0., max_future_lag_ns=0,
            post_motion_max_position_error_m=0., post_motion_max_abs_velocity_mps=0.,
            longest_confirmed_stable_interval_seconds=0.,
        )

        def diagnostic_record(feedback, now, causes):
            nonlocal last_feedback_values, last_distinct_stamp
            position_delta = abs(feedback.position-target)
            velocity = abs(feedback.velocity)
            diagnostics['max_position_error_m'] = max(diagnostics['max_position_error_m'],position_delta)
            diagnostics['max_abs_velocity_mps'] = max(diagnostics['max_abs_velocity_mps'],velocity)
            diagnostics['max_future_lag_ns'] = max(diagnostics['max_future_lag_ns'],feedback.stamp_ns-now)
            if now >= self._fine_motion_end_ns:
                diagnostics['post_motion_max_position_error_m'] = max(
                    diagnostics['post_motion_max_position_error_m'],position_delta)
                diagnostics['post_motion_max_abs_velocity_mps'] = max(
                    diagnostics['post_motion_max_abs_velocity_mps'],velocity)
            values = (feedback.stamp_ns,feedback.position,feedback.velocity,feedback.effort)
            if (last_feedback_values is not None and values[0] == last_feedback_values[0]
                    and values[1:3] != last_feedback_values[1:3]):
                diagnostics['changed_feedback_same_stamp'] += 1
            last_feedback_values = values
            sample = dict(observed_ros_ns=now, feedback=asdict(feedback),
                          position_error_m=position_delta,
                          signed_feedback_age_ns=now-feedback.stamp_ns,
                          command_motion_end_ros_ns=self._fine_motion_end_ns,
                          nonstationary_causes=list(causes))
            if feedback.stamp_ns not in recorded_stamps:
                if last_distinct_stamp is not None:
                    gap = feedback.stamp_ns-last_distinct_stamp
                    diagnostics['out_of_order_feedback_stamps'] += int(gap < 0)
                    diagnostics['max_stored_feedback_gap_ns'] = max(
                        diagnostics['max_stored_feedback_gap_ns'],gap)
                last_distinct_stamp = feedback.stamp_ns
                if len(recorded_order) == recorded_order.maxlen:
                    recorded_stamps.discard(recorded_order[0])
                recorded_order.append(feedback.stamp_ns)
                recorded_stamps.add(feedback.stamp_ns)
                diagnostics['distinct_feedback_stamps'] += 1
                observed_samples.append(sample)
                for cause in causes:
                    counts = diagnostics['nonstationary_new_feedback_by_reason']
                    counts[cause] = counts.get(cause,0)+1
            else:
                diagnostics['duplicate_feedback_polls'] += 1
            return sample

        def diagnostic_snapshot(reason, *, include_samples=False):
            result = dict(diagnostics)
            result.update(
                reason=reason, target=target, phase=self._fine_phase, micro=bool(micro),
                wait_start_ros_ns=diagnostic_start_ns,
                command_motion_end_ros_ns=self._fine_motion_end_ns,
                elapsed_ros_seconds=(int(self.get_clock().now().nanoseconds)-diagnostic_start_ns)/1e9,
                elapsed_wall_seconds=time.monotonic()-diagnostic_start_wall,
                stationary_start_ros_ns=stationary_since,
                last_confirmed_stationary_ros_ns=last_confirmed_ns,
                required_dwell_seconds=limits.dwell_seconds,
                position_error_limit_m=(limits.micro_position_error if micro else limits.stationary_position_error),
                velocity_limit_mps=(limits.micro_stationary_velocity if micro else limits.stationary_velocity),
                dwell_completed=reason == 'stationary_endpoint',
                feedback_stamp_basis='Stored GripperFeedback timestamp; normal JointState callback may already clamp small future producer stamps',
                sampling_basis='Worker-observed distinct stored stamps; not an exhaustive sensor callback trace',
            )
            if include_samples:
                result.update(observed_feedback_samples=list(observed_samples),
                              observed_reset_samples=list(reset_samples))
            return result

        def emit_failure_diagnostic(reason):
            self._emit('stationarity_diagnostic', reason=reason,
                       diagnostic=diagnostic_snapshot(
                           reason,include_samples=reason not in (
                               'bilateral_contact_observed', 'coarse_contact_observed')))

        while time.monotonic() < wall_end:
            reason = self._feedback_error()
            if reason:
                self._hold(reason)
                emit_failure_diagnostic(reason)
                return False
            stop = self._fine_stop
            if stop and not ((allow_bilateral_hold and stop == 'bilateral_contact_observed')
                             or (allow_coarse_hold and stop == 'coarse_contact_observed')):
                emit_failure_diagnostic(stop)
                return False
            feedback = self._latest_gripper_feedback()
            now = int(self.get_clock().now().nanoseconds)
            if now < last_clock:
                self._hold('clock_reversed')
                emit_failure_diagnostic('clock_reversed')
                return False
            if now > last_clock:
                last_progress_wall, last_clock = time.monotonic(), now
            elif time.monotonic() - last_progress_wall > 1.:
                self._hold('clock_stalled')
                emit_failure_diagnostic('clock_stalled')
                return False
            position_error = limits.micro_position_error if micro else limits.stationary_position_error
            velocity_limit = limits.micro_stationary_velocity if micro else limits.stationary_velocity
            stationary = (now >= self._fine_motion_end_ns
                          and abs(feedback.position - target) <= position_error
                          and abs(feedback.velocity) <= velocity_limit
                          and feedback.stamp_ns <= now)
            causes = []
            if now < self._fine_motion_end_ns:
                causes.append('motion_not_finished')
            if abs(feedback.position-target) > position_error:
                causes.append('position_error')
            if abs(feedback.velocity) > velocity_limit:
                causes.append('velocity')
            if feedback.stamp_ns > now:
                causes.append('future_feedback')
            observed = diagnostic_record(feedback,now,causes)
            if stationary:
                if stationary_since is None:
                    stationary_since = now
                    diagnostics['stationary_epoch_starts'] += 1
                    if micro:
                        # Only post-motion stationary producer frames can
                        # certify this microtarget. Preserve identity/faults.
                        self._clear_target_contact_samples()
                        self._fine_micro_stationary_start_ns = stationary_since
                last_confirmed_ns = now
                diagnostics['longest_confirmed_stable_interval_seconds'] = max(
                    diagnostics['longest_confirmed_stable_interval_seconds'],
                    (now-stationary_since)/1e9)
                if now - stationary_since >= int(limits.dwell_seconds * 1e9):
                    self._emit('stationary_endpoint', target=target, feedback=asdict(feedback),
                               stationary_start_ros_ns=stationary_since,
                               stop_reason=self._fine_stop, phase=self._fine_phase,
                               stationarity_diagnostic=diagnostic_snapshot('stationary_endpoint'))
                    return True
            else:
                if stationary_since is not None:
                    diagnostics['stationary_resets'] += 1
                    for cause in causes:
                        counts = diagnostics['resets_by_reason']
                        counts[cause] = counts.get(cause,0)+1
                    reset_samples.append(dict(observed,
                        previous_stationary_start_ros_ns=stationary_since,
                        previous_confirmed_stable_interval_seconds=(
                            (last_confirmed_ns-stationary_since)/1e9
                            if last_confirmed_ns is not None else 0.)))
                stationary_since = None
                if micro:
                    self._fine_micro_stationary_start_ns = None
            time.sleep(.002)
        self._hold('step_wall_timeout')
        emit_failure_diagnostic('step_wall_timeout')
        return False

    def _step(self, target, phase, duration=None):
        error = self._feedback_error()
        if error:
            self._hold(error)
            return False
        with self._fine_lock:
            self._fine_phase = phase
            if not self._publish_once_locked(target, duration or self.limits.motion_seconds, 'close'):
                return False
        return self._wait_motion_and_stationary(target, preclose=(phase == 'preclose'))

    def _resume_coarse_contact(self):
        """Settle one measured hold before consuming only its normal stop.

        Contact may occur above the nominal empty-band boundary. The callback
        already interrupted that coarse target; this worker never republishes
        it and never clears a fault, contact history, or identity latch.
        """
        with self._fine_lock:
            error = self._feedback_error()
            if error:
                self._hold(error)
                return False
            if self._fine_stop == 'bilateral_contact_observed':
                return False
            if (self._fine_stop != 'coarse_contact_observed'
                    or self._fine_coarse_contact_model is None):
                self._hold('coarse_contact_invalid_resume_state')
                return False
            target = self._fine_command
            if target < self.limits.floor:
                self._hold('coarse_contact_hold_below_floor')
                return False
            self._fine_phase = 'coarse_contact_settle'
            self._emit('coarse_contact_settle_started', commanded_position=target,
                       expected_model=self.expected_model, acquired=False)
        if not self._wait_motion_and_stationary(target, allow_coarse_hold=True):
            return False
        with self._fine_lock:
            error = self._feedback_error()
            if error:
                self._hold(error)
                return False
            # A bilateral/fault callback may have won the lock after the final
            # dwell sample. Leave that stop for its existing handling path.
            if self._fine_stop != 'coarse_contact_observed':
                return False
            feedback = self._latest_gripper_feedback()
            now = int(self.get_clock().now().nanoseconds)
            if (self._fine_command != target or now < self._fine_motion_end_ns
                    or feedback.stamp_ns > now
                    or abs(feedback.position-target) > self.limits.stationary_position_error
                    or abs(feedback.velocity) > self.limits.stationary_velocity):
                self._hold('coarse_contact_endpoint_changed')
                return False
            self._fine_stop = None
            self._fine_phase = 'fine'
            self._emit('coarse_contact_handoff', commanded_position=target,
                       expected_model=self.expected_model,
                       maximum_next_step=self.limits.touched_step, acquired=False)
            return True

    def _observed_contact_sides(self, *, require_minimum_force=True, now=None):
        """Separate named-pair freshness from pressure sufficient for contact."""
        if now is None:
            now = self.get_clock().now().nanoseconds
        with self._fine_lock:
            observations = dict(self._fine_side_last)
        def fresh(side):
            stamp, force = observations.get(side, (0, 0.))
            return (stamp > 0 and -100_000_000 <= now - stamp <= 150_000_000
                    and math.isfinite(force)
                    and force >= 0
                    and (not require_minimum_force
                         or force >= self.adaptive_contact_force_minimum))
        return fresh('left'), fresh('right')

    def _micro_pressure_snapshot(self, step, target, *, wall_deadline=None,
                                 ros_deadline=None, _resettle_validation=False):
        """Evaluate one fixed producer snapshot at both contact and acquisition thresholds."""
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            self._hold('step_wall_timeout')
            return None
        model, raw_left, raw_right, identity_reason = self._adaptive_force_histories()
        epoch = getattr(self, '_fine_micro_stationary_start_ns', None)
        if epoch is None:
            self._hold('micro_preload_stationary_epoch_missing')
            return None
        # A callback arriving after reset may still carry an older producer
        # stamp. Such delayed data cannot certify this stationary microtarget.
        left = tuple(sample for sample in raw_left if sample.stamp_ns >= epoch)
        right = tuple(sample for sample in raw_right if sample.stamp_ns >= epoch)
        feedback = self._latest_gripper_feedback()
        # Check raw history/force/effort/width before treating measured movement
        # as a soft epoch invalidation. Otherwise a resettle could erase a hard
        # error from history before the evaluator had examined it.
        latest = max((sample.stamp_ns for sample in (*left, *right)), default=0)
        latest = max(latest, getattr(feedback, 'stamp_ns', 0))
        deadline = min(self._fine_wall_deadline, time.monotonic() + .5)
        if wall_deadline is not None:
            deadline = min(deadline, wall_deadline)
        now = int(self.get_clock().now().nanoseconds)
        while now < latest <= now + 100_000_000 and time.monotonic() < deadline:
            if ros_deadline is not None and now >= ros_deadline:
                self._hold('micro_pressure_recheck_timeout')
                return None
            error = self._feedback_error()
            if error is None and ros_deadline is not None:
                error = self._micro_stationarity_error(target)
            if error or self._fine_stop is not None:
                self._stop_micro(error or self._fine_stop)
                return None
            time.sleep(.002)
            now = int(self.get_clock().now().nanoseconds)
        if model != self.expected_model or identity_reason is not None:
            self._hold(identity_reason or 'micro_preload_identity_lost')
            return None
        limits = self.limits
        kwargs = dict(
            minimum_width=limits.floor, maximum_width=self.grasp_max_position,
            maximum_force=self.adaptive_contact_force_maximum,
            minimum_samples=self.adaptive_contact_samples,
            maximum_age_seconds=self.adaptive_contact_max_age,
            maximum_gap_seconds=self.adaptive_contact_max_gap,
            minimum_span_seconds=max(.05, self.adaptive_contact_min_span),
            maximum_side_skew_seconds=self.adaptive_contact_max_skew,
            maximum_velocity=limits.micro_stationary_velocity,
            maximum_effort=self.adaptive_effort_maximum,
            baseline_effort=self._adaptive_effort_baseline_value,
            maximum_effort_delta=self.adaptive_effort_delta_maximum,
        )
        raw_evidence = evaluate_bilateral_contact(raw_left, raw_right, feedback, now,
            minimum_force=self.adaptive_contact_force_minimum, **kwargs)
        if (not raw_evidence.verified and raw_evidence.reason not in MICRO_PRESSURE_DEFICIENCIES
                and raw_evidence.reason != 'joint_still_moving'):
            self._hold('micro_preload_evidence_rejected:' + raw_evidence.reason)
            return None
        legacy = evaluate_bilateral_contact(left, right, feedback, now,
            minimum_force=self.adaptive_contact_force_minimum, **kwargs)
        pressure = evaluate_bilateral_contact(left, right, feedback, now,
            minimum_force=limits.minimum_force, **kwargs)

        def spans(evidence):
            result = {}
            for side, samples, count in (('left', left, evidence.left_samples),
                                         ('right', right, evidence.right_samples)):
                recent = [sample for sample in samples
                          if 0 <= now - sample.stamp_ns <= int(self.adaptive_contact_max_age * 1e9)]
                window = recent[-count:] if count else []
                result[side] = dict(
                    samples=len(window),
                    first_producer_stamp_ns=window[0].stamp_ns if window else None,
                    last_producer_stamp_ns=window[-1].stamp_ns if window else None,
                    span_seconds=(window[-1].stamp_ns - window[0].stamp_ns) / 1e9 if window else 0.,
                )
            return result

        with self._fine_lock:
            error = self._feedback_error()
            if error is None and wall_deadline is not None and time.monotonic() >= wall_deadline:
                error = 'step_wall_timeout'
            if error is None and ros_deadline is not None:
                if int(self.get_clock().now().nanoseconds) > ros_deadline:
                    error = 'micro_pressure_recheck_timeout'
                else:
                    error = self._micro_stationarity_error(target)
            if error is None and (self._fine_resettle_enabled or _resettle_validation):
                error = self._micro_stationarity_error(target)
            if error is None and (feedback is None or not math.isfinite(feedback.position)
                    or abs(feedback.position-target) > limits.micro_position_error):
                error = 'micro_preload_pressure_endpoint_drift'
            if error is None and raw_evidence.reason == 'joint_still_moving':
                error = 'joint_still_moving'
            soft_validation = (_resettle_validation and error in (
                'joint_still_moving', 'micro_preload_pressure_endpoint_drift')
                and self._micro_can_resettle())
            if error or self._fine_stop is not None:
                if not soft_validation or self._fine_stop is not None:
                    self._stop_micro(error or self._fine_stop)
                    return None
            if _resettle_validation:
                return legacy, pressure
            self._fine_evidence = pressure
            self._emit('pressure_measurement', step=step,
                reference_position=self._fine_micro_reference, commanded_position=target,
                additional_commanded_closure=self._fine_micro_reference - target,
                feedback=asdict(feedback), evaluation_ros_ns=now,
                stationary_start_ros_ns=epoch,
                expected_model=model, legacy_minimum_force=self.adaptive_contact_force_minimum,
                minimum_force=limits.minimum_force,
                legacy_evidence=asdict(legacy), pressure_evidence=asdict(pressure),
                legacy_windows=spans(legacy), pressure_windows=spans(pressure),
                force_histories=dict(left=[asdict(sample) for sample in left],
                                     right=[asdict(sample) for sample in right]),
                raw_force_histories=dict(left=[asdict(sample) for sample in raw_left],
                                         right=[asdict(sample) for sample in raw_right]),
                retention_verified=False)
        return legacy, pressure

    @staticmethod
    def _micro_pressure_fault(evidence):
        for result in evidence:
            if not result.verified and result.reason not in MICRO_PRESSURE_DEFICIENCIES:
                return result.reason
        return None

    def _micro_stationarity_error(self, target):
        """Additional continuation guard; never certify acquisition here."""
        with self._fine_lock:
            error = self._feedback_error()
            if error:
                return error
            feedback = self._latest_gripper_feedback()
            now = int(self.get_clock().now().nanoseconds)
            reason = None
            if self._fine_resettle_enabled:
                if (self._fine_command != self._fine_step_command
                        or self._fine_motion_end_ns != self._fine_step_motion_end_ns):
                    reason = 'micro_preload_command_changed'
                elif now < self._fine_step_last_ros_ns:
                    reason = 'clock_reversed'
                self._fine_step_last_ros_ns = max(now, self._fine_step_last_ros_ns)
            if reason is None and abs(feedback.position - target) > self.limits.micro_position_error:
                reason = 'micro_preload_pressure_endpoint_drift'
            elif reason is None and (now < self._fine_motion_end_ns
                    or abs(feedback.velocity) > self.limits.micro_stationary_velocity):
                reason = 'joint_still_moving'
            soft = (reason in ('joint_still_moving','micro_preload_pressure_endpoint_drift')
                    and self._micro_can_resettle())
            if reason and not soft and self._fine_fault_snapshot is None:
                self._fine_fault_snapshot = self._fault_snapshot_locked(
                    reason, feedback=feedback, now=now)
            return reason

    def _micro_can_resettle(self):
        return (self._fine_resettle_enabled and self._fine_micro_active
                and not self._fine_fault and self._fine_stop is None
                and self._fine_command == self._fine_step_command
                and self._fine_motion_end_ns == self._fine_step_motion_end_ns
                and int(self.get_clock().now().nanoseconds) >= self._fine_motion_end_ns)

    def _stop_micro(self, reason):
        if (reason in ('joint_still_moving','micro_preload_pressure_endpoint_drift')
                and self._micro_can_resettle()):
            raise _MicroEpochInvalidated(reason)
        self._hold(reason)

    @staticmethod
    def _hard_before_disturbance(error, hard_reason):
        """Chronology/budget faults cannot be softened by a simultaneous twitch."""
        if error in (None, 'joint_still_moving', 'micro_preload_pressure_endpoint_drift'):
            return hard_reason
        return error

    def _resettle_micro_epoch(self, step, target, reason, *, wall_deadline):
        """Discard disturbed proof, never the command, reference or hard latches."""
        # Validate a fixed raw history snapshot before clearing its epoch. This
        # includes the normal bounded producer-clock catch-up, inside the same
        # absolute step/total deadline; no new close or hold is published.
        checked = self._micro_pressure_snapshot(step,target,wall_deadline=wall_deadline,
                                               _resettle_validation=True)
        if checked is None:
            return False
        with self._fine_lock:
            hard = self._feedback_error()
            if (not hard and (self._fine_command != self._fine_step_command
                    or self._fine_motion_end_ns != self._fine_step_motion_end_ns)):
                hard = 'micro_preload_command_changed'
            if time.monotonic() >= wall_deadline:
                hard = hard or 'step_wall_timeout'
            if hard or self._fine_stop is not None:
                self._hold(hard or self._fine_stop)
                return False
            old_epoch = self._fine_micro_stationary_start_ns
            self._fine_micro_stationary_start_ns = None
            self._fine_evidence = None
            self._emit('pressure_epoch_invalidated',step=step,reason=reason,
                commanded_position=self._fine_command,measurement_target=target,
                previous_stationary_start_ros_ns=old_epoch,
                step_wall_seconds_remaining=max(0.,wall_deadline-time.monotonic()),
                acquired=False)
        return self._wait_motion_and_stationary(target,micro=True,wall_deadline=wall_deadline)

    def _recheck_micro_pressure(self, step, target, evidence, *, wall_deadline):
        """One bounded observation hold, without a new command or epoch reset.

        Observe at the same target for 50 ms, then evaluate one fresh fixed
        snapshot. Its clock catch-up may use the remaining 100 ms; the entire
        recheck is capped at 150 ms ROS and the original step/total wall limits.
        """
        hard_reason = self._micro_pressure_fault(evidence)
        if hard_reason:
            self._hold('micro_preload_evidence_rejected:' + hard_reason)
            return None
        if evidence[1].verified:
            return evidence
        with self._fine_lock:
            error = self._micro_stationarity_error(target)
            if error or self._fine_stop is not None:
                self._stop_micro(error or self._fine_stop)
                return None
            started_ns = int(self.get_clock().now().nanoseconds)
            epoch = self._fine_micro_stationary_start_ns
            self._emit('pressure_recheck_started', step=step,
                commanded_position=target, stationary_start_ros_ns=epoch,
                recheck_start_ros_ns=started_ns, maximum_ros_seconds=.15,
                step_wall_seconds_remaining=max(0.,wall_deadline-time.monotonic()),
                legacy_reason=evidence[0].reason, pressure_reason=evidence[1].reason,
                acquired=False)
        end_ns = started_ns + 150_000_000
        sample_after_ns = started_ns + 50_000_000
        wall_end = min(wall_deadline, self._fine_wall_deadline)
        last_clock, last_progress_wall = started_ns, time.monotonic()
        final_evidence = None
        try:
            while True:
                with self._fine_lock:
                    error = self._micro_stationarity_error(target)
                    now = int(self.get_clock().now().nanoseconds)
                    if now < last_clock:
                        error = self._hard_before_disturbance(error, 'clock_reversed')
                    elif now > last_clock:
                        last_clock, last_progress_wall = now, time.monotonic()
                    elif time.monotonic()-last_progress_wall > 1.:
                        error = self._hard_before_disturbance(error, 'clock_stalled')
                    if time.monotonic() >= wall_end:
                        error = self._hard_before_disturbance(error, 'step_wall_timeout')
                    if now > end_ns:
                        error = self._hard_before_disturbance(error, 'micro_pressure_recheck_timeout')
                    if self._fine_micro_stationary_start_ns != epoch:
                        error = self._hard_before_disturbance(error, 'micro_preload_stationary_epoch_changed')
                    if error or self._fine_stop is not None:
                        self._stop_micro(error or self._fine_stop)
                        return None
                if now >= sample_after_ns:
                    break
                time.sleep(.002)
            final_evidence = self._micro_pressure_snapshot(
                step, target, wall_deadline=wall_end, ros_deadline=end_ns)
            if final_evidence is None:
                return None
            hard_reason = self._micro_pressure_fault(final_evidence)
            if hard_reason:
                self._hold('micro_preload_evidence_rejected:' + hard_reason)
                return None
            with self._fine_lock:
                error = self._micro_stationarity_error(target)
                if time.monotonic() >= wall_end:
                    error = self._hard_before_disturbance(error, 'step_wall_timeout')
                if int(self.get_clock().now().nanoseconds) > end_ns:
                    error = self._hard_before_disturbance(error, 'micro_pressure_recheck_timeout')
                if error or self._fine_stop is not None:
                    self._stop_micro(error or self._fine_stop)
                    return None
            return final_evidence
        finally:
            with self._fine_lock:
                self._emit('pressure_recheck_finished', step=step,
                    commanded_position=target, stationary_start_ros_ns=epoch,
                    recheck_start_ros_ns=started_ns,
                    recheck_elapsed_ros_seconds=(int(self.get_clock().now().nanoseconds)-started_ns)/1e9,
                    reason=self._fine_fault,
                    legacy_reason=final_evidence[0].reason if final_evidence else None,
                    pressure_reason=final_evidence[1].reason if final_evidence else None,
                    pressure_verified=bool(final_evidence and final_evidence[1].verified),
                    acquired=False)

    def _run_micro_preload(self, *, finalize=False):
        """Probe within one immutable travel bound (2um default, at most 25um)."""
        limits = self.limits
        if not limits.micro_preload_steps:
            return
        step_wall_deadline = min(self._fine_wall_deadline,
                                 time.monotonic() + limits.step_wall_seconds)
        # Tighten the initial held endpoint too: the ordinary .25 micrometre
        # settling test is not a certificate for the selected micro envelope.
        if not self._wait_motion_and_stationary(
                self._fine_command, allow_bilateral_hold=True, micro=True,
                wall_deadline=step_wall_deadline):
            return
        with self._fine_lock:
            error = self._feedback_error()
            if error:
                self._hold(error)
                return
            if self._fine_stop != 'bilateral_contact_observed':
                self._hold('micro_preload_invalid_resume_state')
                return
            if self._target_book_model != self.expected_model:
                self._hold('micro_preload_identity_lost')
                return
            if not any(self._observed_contact_sides()):
                self._hold('micro_preload_contact_lost')
                return
            feedback = self._latest_gripper_feedback()
            self._fine_micro_reference = float(feedback.position)
            self._fine_micro_active = True
            self._fine_phase = 'micro_preload'
            # Consume only the normal stop-to-measure state. The independent
            # fault latch and production overload latch are never cleared.
            self._fine_stop = None
            self._fine_resettle_enabled = True
            self._fine_step_command = self._fine_command
            self._fine_step_motion_end_ns = self._fine_motion_end_ns
            self._fine_step_last_ros_ns = int(self.get_clock().now().nanoseconds)
            self._emit('micro_preload_started', reference_position=self._fine_micro_reference,
                       maximum_additional_closure=limits.micro_preload_limit,
                       maximum_steps=limits.micro_preload_steps,
                       command_step_m=limits.micro_preload_step,
                       motion_seconds=limits.micro_motion_seconds,
                       position_error_limit_m=limits.micro_position_error,
                       velocity_limit_mps=limits.micro_stationary_velocity,
                       minimum_force=limits.minimum_force, acquired=False)
        try:
            step = 0
            while step <= limits.micro_preload_steps:
                target = self._fine_micro_reference - step * limits.micro_preload_step
                try:
                    evidence = self._micro_pressure_snapshot(
                        step, target, wall_deadline=step_wall_deadline)
                    if evidence is None:
                        break
                    if not evidence[1].verified:
                        evidence = self._recheck_micro_pressure(
                            step, target, evidence, wall_deadline=step_wall_deadline)
                        if evidence is None:
                            break
                    if evidence[1].verified and finalize:
                        # Final acquisition consumes freshly evaluated pressure
                        # on this epoch, not cached evidence from before a newer
                        # callback. Any newer measured disturbance below causes
                        # a same-target resettle, before publishing success.
                        evidence = self._micro_pressure_snapshot(
                            step,target,wall_deadline=step_wall_deadline)
                        if evidence is None:
                            break
                        if not evidence[1].verified:
                            evidence = self._recheck_micro_pressure(
                                step,target,evidence,wall_deadline=step_wall_deadline)
                            if evidence is None:
                                break
                    hard_reason = self._micro_pressure_fault(evidence)
                    if hard_reason:
                        self._stop_micro('joint_still_moving' if hard_reason == 'joint_still_moving'
                                         else 'micro_preload_evidence_rejected:' + hard_reason)
                        break
                    with self._fine_lock:
                        error = self._micro_stationarity_error(target)
                        if time.monotonic() >= step_wall_deadline:
                            error = ('step_wall_timeout' if error in (None,'joint_still_moving',
                                     'micro_preload_pressure_endpoint_drift') else error)
                        if error or self._fine_stop is not None:
                            self._stop_micro(error or self._fine_stop)
                            break
                        if evidence[1].verified:
                            self._fine_stop = 'micro_pressure_target_reached'
                            if finalize:
                                self._fine_result = self.node._publish_adaptive_close_result(
                                    evidence[1],verified=True,reason='fine_bilateral_pressure_confirmed',
                                    close_stage='fine_transport_lock',commanded_position=self._fine_command,
                                    acquisition_position=self._fine_command)
                            break
                        if step == limits.micro_preload_steps:
                            break
                        next_target = self._fine_micro_reference-(step+1)*limits.micro_preload_step
                        if (next_target < limits.floor or self._fine_micro_reference-next_target
                                > limits.micro_preload_limit+1e-15):
                            self._hold('micro_preload_target_limit')
                            break
                        # Only now does a new target receive a new step budget.
                        # A disturbance in the previous admission used its old
                        # absolute deadline and cannot earn another five seconds.
                        next_deadline = min(self._fine_wall_deadline,
                                            time.monotonic()+limits.step_wall_seconds)
                        if not self._publish_once_locked(next_target, limits.micro_motion_seconds, 'close'):
                            break
                        step += 1
                        step_wall_deadline = next_deadline
                        self._fine_step_command = self._fine_command
                        self._fine_step_motion_end_ns = self._fine_motion_end_ns
                        self._fine_step_last_ros_ns = int(self.get_clock().now().nanoseconds)
                    if not self._wait_motion_and_stationary(
                            next_target, micro=True, wall_deadline=step_wall_deadline):
                        break
                except _MicroEpochInvalidated as disturbance:
                    if not self._resettle_micro_epoch(
                            step,target,str(disturbance),wall_deadline=step_wall_deadline):
                        break
            with self._fine_lock:
                if self._fine_stop is None:
                    self._fine_stop = 'micro_preload_limit_reached'
                self._emit('micro_preload_finished', reason=self._fine_stop,
                           fault=getattr(self, '_fine_fault', None), acquired=False,
                           reference_position=self._fine_micro_reference,
                           commanded_position=self._fine_command)
        finally:
            self._fine_micro_active = False
            self._fine_resettle_enabled = False

    def run(self):
        """Return the normal five-field close result; faults use checked recovery."""
        node, limits = self.node, self.limits
        node._transport_lock_engaged = False
        node._clear_target_contact_samples()
        node._adaptive_effort_baseline_value = node._adaptive_effort_baseline(
            int(node.get_clock().now().nanoseconds))
        feedback, error = node._adaptive_motion_feedback()
        # The official contact bridge emits no empty free-space heartbeat.
        # Verify publisher readiness before contact; thereafter require fresh
        # producer frames instead of inventing zero-force observations.
        if node.count_publishers('/contacts') < 1:
            error = error or 'contact_publisher_unavailable'
        if node.gripper_pub.get_subscription_count() < 1:
            error = error or 'gripper_command_subscriber_unavailable'
        if (error or not node._gripper_open_confirmed or feedback is None
                or abs(feedback.position - node.gripper_open) > node.adaptive_start_tolerance):
            self._hold(error or 'open_start_not_confirmed')
        else:
            node._gripper_open_confirmed = False
            self._emit('started', limits=asdict(limits), retention_verified=False)
            q = limits.preclose
            if self._step(q, 'preclose', 1.0):
                while q > limits.floor + 1e-10:
                    if self._fine_stop == 'coarse_contact_observed':
                        if not self._resume_coarse_contact():
                            break
                        q = self._fine_command
                    if self._fine_stop is not None:
                        break
                    if q > limits.fine_start + 1e-10 and self._fine_first_contact is None:
                        target = max(limits.fine_start, q - limits.coarse_step)
                        phase, duration = 'coarse', .32
                    else:
                        step = limits.touched_step if self._fine_first_contact is not None else limits.fine_step
                        target = max(limits.floor, q - step)
                        phase, duration = 'fine', limits.motion_seconds
                    if not self._step(target, phase, duration):
                        if self._fine_stop == 'coarse_contact_observed':
                            continue
                        break
                    q = target
            if self._fine_stop == 'bilateral_contact_observed':
                if self._wait_motion_and_stationary(
                        self._fine_command, allow_bilateral_hold=True):
                    self._run_micro_preload(finalize=True)
        if self._fine_result is not None:
            return self._fine_result
        evidence = self._fine_evidence
        if evidence is None:
            evidence, _, _ = node._adaptive_pressure_evidence(
                minimum_width=limits.floor,
                baseline_effort=node._adaptive_effort_baseline_value)
        # Success is published only at the locked, freshly checked microstep
        # admission above. A stale cached pressure result cannot reach it here.
        verified = False
        return node._publish_adaptive_close_result(
            evidence, verified=verified,
            reason=('fine_bilateral_pressure_confirmed' if verified else
                    self._fine_fault or self._fine_stop or 'fine_width_floor'),
            close_stage='fine_transport_lock' if verified else 'fine_pressure_close',
            commanded_position=self._fine_command,
            acquisition_position=(self._fine_command if verified else None))
