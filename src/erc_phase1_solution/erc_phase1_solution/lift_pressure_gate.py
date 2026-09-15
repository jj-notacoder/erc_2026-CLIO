"""Single-use, read-only pressure admission for the initial shelf lift.

This gate neither closes the gripper nor creates a new acquisition budget.
The action request is published only while its final sensor snapshot is locked.
"""
from __future__ import annotations

import math
import re
import time

from .adaptive_grasp import evaluate_bilateral_contact
from .stock_gripper_close import enabled as _stock_diagnostic, contact_evidence as _stock_contact_evidence, measured_position_in_range as _stock_measured_position_in_range
from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS


class LiftPressureRejected(RuntimeError):
    """No arm request was sent; keep the checked stationary hold."""


class _PressureDeficient(LiftPressureRejected):
    """Healthy named contact has not yet supplied a qualifying force window."""


PRESSURE_DEFICIENCIES = frozenset({
    'unilateral_contact', 'left_force_below_minimum', 'right_force_below_minimum',
    'insufficient_force_samples', 'force_sample_span_too_short',
    'force_sample_gap', 'bilateral_sample_skew',
})


def held_contact_identity_fault(first, second, expected_model):
    """Inspect external left-hand pairs before target-only filtering."""
    names = (str(first), str(second))
    hand = ['gripper_left_' in name.lower() for name in names]
    if hand[0] == hand[1]:
        return None
    finger, other = names if hand[0] else names[::-1]
    finger_links = {f'gripper_left_{part}_{side}_link'
                    for part in ('base_finger', 'inner_finger', 'outer_finger', 'fingertip')
                    for side in ('left', 'right')}
    if not any(component in finger_links for component in finger.split('::')):
        return 'unintended_held_hand_contact'
    models = [component for component in other.split('::')
              if re.fullmatch(r'book_col_\d+_row_\d+_(red|green|yellow|blue)', component)]
    if models != [expected_model]:
        return 'unexpected_held_contact_identity'
    return None


class LiftPressureGate:
    def __init__(self, node, checked_hold, reference):
        self.node = node
        self.checked = checked_hold
        self.reference = reference
        self.used = False
        self.expected_model = getattr(node, '_target_book_model', None)
        self.minimum_force = float(getattr(node, 'lift_first_minimum_contact_force', 3.5))
        self.maximum_gripper_velocity = float(getattr(
            node, 'lift_first_maximum_gripper_velocity_mps', 2e-6))
        if not (math.isfinite(self.maximum_gripper_velocity)
                and 0 < self.maximum_gripper_velocity <= 1e-5):
            raise ValueError('lift-first gripper velocity must be within (0, 10e-6] m/s')
        self.stock_mode = _stock_diagnostic(node)
        if self.stock_mode:
            self.minimum_force = 0.0  # named contact evidence, not pressure qualification
            self.maximum_gripper_velocity = node.adaptive_velocity_tolerance
        self.master_geometry_tolerance = (float(node.adaptive_endpoint_tolerance)
            if self.stock_mode else .00001)
        self.baseline_effort = float(getattr(node, '_acquired_effort_baseline', math.nan))
        self.last_reason = None
        self.send_started = False
        self._last_clock = None
        self._last_input = None
        self._started_wall = None

    def _snapshot_locked(self):
        n = self.node
        histories = getattr(n, '_book_contact_force_samples', {})
        return dict(
            model=getattr(n, '_target_book_model', None),
            histories={key: (tuple(value[0]), tuple(value[1])) for key, value in histories.items()},
            feedback=tuple(getattr(n, '_gripper_feedback_samples', ())),
            joints=dict(n.joints), stamps=dict(getattr(n, '_joint_stamps_ns', {})),
            velocities=dict(getattr(n, '_joint_velocities', {})),
            odom=dict(getattr(n, '_staging_odom', {}) or {}),
            hazard=(getattr(n, '_payload_hazard_latched', None)
                    or getattr(n, '_held_grip_sensor_fault', None)
                    or ('payload_robot_contact' if getattr(n, '_target_robot_contact_latched', False) else None)
                    or getattr(n, '_adaptive_overload_latched', None)
                    or getattr(n, '_adaptive_motion_halt_reason', None)),
            active_arm=bool(getattr(n, '_goal_handles', ()) or getattr(n, '_pending_retained_acceptances', ())),
        )

    def _remember_input(self, snapshot, now, phase):
        # Each snapshot already owns copied dictionaries/immutable sample tuples.
        # Retain the evaluated input, never re-read sensors after a rejection.
        self._last_input = (snapshot, now, phase, time.monotonic())

    def _rejection_snapshot(self):
        try:
            return self._format_rejection_snapshot()
        except Exception as exc:
            # Malformed negative data must not hide the original stop reason.
            return dict(snapshot_export_error=str(exc),
                        maximum_gripper_velocity_mps=self.maximum_gripper_velocity)

    def _format_rejection_snapshot(self):
        if self._last_input is None:
            return None
        snapshot, now, phase, checked_wall = self._last_input
        def finite(value):
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
        def feedback_fields(sample):
            return dict(stamp_ns=sample.stamp_ns,
                age_seconds=(now-sample.stamp_ns)/1e9,
                position_m=finite(sample.position), velocity_mps=finite(sample.velocity),
                effort=finite(sample.effort))
        force_sides = {}
        for name, samples in zip(('left', 'right'),
                snapshot['histories'].get(self.expected_model, ((), ()))):
            force_sides[name] = dict(sample_count=len(samples),
                first_producer_stamp_ns=samples[0].stamp_ns if samples else None,
                last_producer_stamp_ns=samples[-1].stamp_ns if samples else None,
                latest_force_n=finite(samples[-1].force_newtons) if samples else None,
                latest_age_seconds=(now-samples[-1].stamp_ns)/1e9 if samples else None,
                recent_samples=[dict(producer_stamp_ns=s.stamp_ns,
                    force_n=finite(s.force_newtons)) for s in samples[-8:]])
        elapsed = None if self._started_wall is None else time.monotonic()-self._started_wall
        return dict(phase=phase, evaluation_ros_ns=now,
            elapsed_wall_seconds=elapsed, overall_wall_timeout_seconds=5.,
            input_evaluated_elapsed_wall_seconds=(None if self._started_wall is None
                                                  else checked_wall-self._started_wall),
            maximum_gripper_velocity_mps=self.maximum_gripper_velocity,
            minimum_force_n=self.minimum_force,
            maximum_force_n=(None if self.stock_mode else self.node.adaptive_contact_force_maximum),
            minimum_force_span_seconds=self.node.adaptive_contact_min_span,
            feedback=feedback_fields(snapshot['feedback'][-1]) if snapshot['feedback'] else None,
            recent_feedback=[feedback_fields(s) for s in snapshot['feedback'][-8:]],
            master_geometry=self._master_geometry_diagnostic(),
            force_sides=force_sides, expected_model=self.expected_model,
            observed_model=snapshot['model'], hazard=snapshot['hazard'],
            active_arm=snapshot['active_arm'],
            feedback_stamp_basis='Stored GripperFeedback; normal callback may clamp small future producer stamps',
            force_stamp_basis='Original contact producer stamps',
            history_limit_per_stream=8,
            snapshot_basis='Last evaluated gate input; exact input for a sensor/evaluator rejection, last checked input for a between-check clock/deadline failure; no later sensor read')

    def _hard_check(self, snapshot, now, *, allow_future, phase='hard_check'):
        self._remember_input(snapshot, now, phase)
        n = self.node
        if self._last_clock is not None and now < self._last_clock:
            raise LiftPressureRejected('clock_reversed')
        self._last_clock = now
        if n._cancel.is_set():
            raise LiftPressureRejected('cancelled')
        if snapshot['hazard']:
            raise LiftPressureRejected(str(snapshot['hazard']))
        if snapshot['active_arm']:
            raise LiftPressureRejected('arm_motion_pending')
        expected = self.expected_model
        if (not isinstance(expected, str)
                or re.fullmatch(r'book_col_\d+_row_\d+_(red|green|yellow|blue)', expected) is None
                or not expected.endswith('_' + n.target_colour)
                or snapshot['model'] != expected):
            raise LiftPressureRejected('target_identity_invalid')
        lead = 100_000_000 if allow_future else 0
        for model, sides in snapshot['histories'].items():
            for side in sides:
                previous = -1
                for sample in side:
                    stamp, force = sample.stamp_ns, sample.force_newtons
                    if (isinstance(stamp, bool) or not isinstance(stamp, int)
                            or stamp <= previous or stamp <= 0):
                        raise LiftPressureRejected('malformed_force_history')
                    previous = stamp
                    if not math.isfinite(force) or force < 0:
                        raise LiftPressureRejected('invalid_contact_force')
                    if stamp > now + lead:
                        raise LiftPressureRejected('future_contact_force')
                    if not self.stock_mode and force > n.adaptive_contact_force_maximum:
                        raise LiftPressureRejected('force_overload')
                    if now - stamp <= int(n.adaptive_contact_max_age * 1e9) and model != expected:
                        raise LiftPressureRejected('unexpected_contact_identity')
        sides = snapshot['histories'].get(expected, ((), ()))
        if len(sides) != 2 or any(not side for side in sides):
            raise LiftPressureRejected('named_contact_missing')
        if any(now-side[-1].stamp_ns > int(n.adaptive_contact_max_age*1e9) for side in sides):
            raise LiftPressureRejected('named_contact_stale')
        feedback = snapshot['feedback'][-1] if snapshot['feedback'] else None
        if feedback is None:
            raise LiftPressureRejected('missing_gripper_feedback')
        if (not all(math.isfinite(v) for v in (feedback.position, feedback.velocity, feedback.effort))
                or not 0 <= now - feedback.stamp_ns + lead <= 150_000_000 + lead):
            raise LiftPressureRejected('invalid_gripper_feedback')
        if abs(feedback.velocity) > self.maximum_gripper_velocity:
            raise LiftPressureRejected('gripper_not_stationary')
        width_valid = (_stock_measured_position_in_range(feedback.position) if self.stock_mode else
            n.grasp_min_position+n.grasp_min_margin <= feedback.position < n.grasp_max_position)
        if not width_valid:
            raise LiftPressureRejected('width_invalid')
        if not self.stock_mode and abs(feedback.effort) > n.adaptive_effort_maximum:
            raise LiftPressureRejected('effort_overload')
        if not math.isfinite(self.baseline_effort):
            raise LiftPressureRejected('acquisition_effort_baseline_missing')
        if not self.stock_mode and abs(feedback.effort-self.baseline_effort) > n.adaptive_effort_delta_maximum:
            raise LiftPressureRejected('effort_delta_overload')
        joints, stamps = snapshot['joints'], snapshot['stamps']
        for index, name in enumerate(IK_JOINTS):
            q, v, stamp = joints.get(name, math.nan), snapshot['velocities'].get(name, math.nan), stamps.get(name, 0)
            if (not math.isfinite(q) or not math.isfinite(v)
                    or not -lead <= now-stamp <= 150_000_000):
                raise LiftPressureRejected('arm_feedback_invalid:' + name)
            if abs(v) > .001 or abs(q-float(self.checked['left'][index])) > .0002:
                raise LiftPressureRejected('arm_not_stationary:' + name)
        context = self.checked['geometry_context']
        for names, values in ((RIGHT_ARM_JOINTS, context['right_positions']),
                              (('head_1_joint', 'head_2_joint'), context['head_positions'])):
            for name, checked in zip(names, values):
                q, stamp = joints.get(name, math.nan), stamps.get(name, 0)
                if (not math.isfinite(q) or not -lead <= now-stamp <= 350_000_000
                        or abs(q-float(checked)) > .001
                        or abs(q-float(self.reference['joints'][name])) > .001):
                    raise LiftPressureRejected('collision_context_changed:' + name)
        for name, checked in self.checked['fingers'].items():
            q, stamp = joints.get(name, math.nan), stamps.get(name, 0)
            tolerance = self.master_geometry_tolerance if name == 'gripper_left_finger_joint' else .001
            if (not math.isfinite(q) or not -lead <= now-stamp <= 350_000_000
                    or abs(q-float(checked)) > tolerance):
                raise LiftPressureRejected('finger_geometry_changed:' + name)
        reported = set(self.checked.get('reported_passive_joints', ()))
        modeled = set(self.checked.get('modeled_finger_joints', ()))
        if {name for name in reported | modeled if name in joints} != reported:
            raise LiftPressureRejected('finger_feedback_availability_changed')
        for name in reported:
            if (not math.isfinite(joints.get(name, math.nan))
                    or not -lead <= now-stamps.get(name, 0) <= 350_000_000):
                raise LiftPressureRejected('reported_passive_feedback_invalid:' + name)
        odom = snapshot['odom']
        pose = odom.get('pose', ())
        if (len(pose) != 3 or not all(math.isfinite(v) for v in pose)
                or not -lead <= now-odom.get('stamp_ns', 0) <= 350_000_000
                or not 0 <= odom.get('linear_speed', math.inf) <= .005
                or not 0 <= odom.get('angular_speed', math.inf) <= .008):
            raise LiftPressureRejected('base_feedback_invalid')
        original = self.reference['base_pose']
        yaw = math.atan2(math.sin(pose[2]-original[2]), math.cos(pose[2]-original[2]))
        if math.hypot(pose[0]-original[0], pose[1]-original[1]) > .002 or abs(yaw) > .005:
            raise LiftPressureRejected('registered_base_moved')

        interval_admission = self.checked.get('preclose_interval_admission')
        if interval_admission is not None:
            interval_admission.check(n, snapshot, now)

    def _evaluate(self, snapshot, now, *, phase='pressure_evaluation'):
        self._remember_input(snapshot, now, phase)
        n = self.node
        left, right = snapshot['histories'].get(self.expected_model, ((), ()))
        if self.stock_mode:
            return _stock_contact_evidence(left, right,
                snapshot['feedback'][-1] if snapshot['feedback'] else None, now,
                maximum_velocity=self.maximum_gripper_velocity)
        return evaluate_bilateral_contact(
            left, right, snapshot['feedback'][-1] if snapshot['feedback'] else None, now,
            minimum_width=n.grasp_min_position+n.grasp_min_margin,
            maximum_width=n.grasp_max_position,
            minimum_force=self.minimum_force, maximum_force=n.adaptive_contact_force_maximum,
            minimum_samples=n.adaptive_contact_samples,
            maximum_age_seconds=n.adaptive_contact_max_age,
            maximum_gap_seconds=n.adaptive_contact_max_gap,
            minimum_span_seconds=n.adaptive_contact_min_span,
            maximum_side_skew_seconds=n.adaptive_contact_max_skew,
            maximum_velocity=self.maximum_gripper_velocity, maximum_effort=n.adaptive_effort_maximum,
            baseline_effort=self.baseline_effort,
            maximum_effort_delta=n.adaptive_effort_delta_maximum,
        )

    def send(self, send_goal):
        """One permit, at most five wall seconds, no closure or motion retry."""
        if self.used:
            raise LiftPressureRejected('pressure_permit_already_consumed')
        self.used = True
        self._started_wall = time.monotonic()
        overall_deadline = self._started_wall + 5.
        waiting = False
        try:
            while True:
                try:
                    return self._attempt(send_goal, overall_deadline)
                except _PressureDeficient as exc:
                    if time.monotonic() >= overall_deadline:
                        raise LiftPressureRejected('fresh_pressure_wait_timeout:' + str(exc)) from exc
                    if not waiting:
                        self._status('lift_first_pressure_waiting', reason=str(exc),
                                     wall_timeout_seconds=5., read_only=True)
                        waiting = True
                    time.sleep(.002)
        except Exception as exc:
            if self.send_started:
                raise
            self.last_reason = str(exc)
            self._status('lift_first_pressure_rejected', reason=str(exc),
                         read_only=True, retained_stop=True, recovery_halted=True,
                         admission_snapshot=self._rejection_snapshot())
            if isinstance(exc, LiftPressureRejected):
                raise
            raise LiftPressureRejected(str(exc)) from exc

    def _master_geometry_diagnostic(self):
        master = 'gripper_left_finger_joint'
        checked = self.checked.get('fingers', {}).get(master)
        snapshot = self._last_input[0] if self._last_input is not None else None
        return dict(geometry_reference_master_position_m=self.checked.get(
                'geometry_reference_master_position_m', checked),
            master_comparison_reference_position_m=checked,
            current_master_position_m=(snapshot['joints'].get(master) if snapshot is not None else None),
            master_geometry_tolerance_m=self.master_geometry_tolerance,
            master_geometry_scope='nominal sampled geometry plus engineering admission tolerance; no geometry recomputation or passive deflection proof')

    def _status(self, event, **fields):
        # Diagnostics must never hide the future of an already-published goal.
        try:
            self.node._publish_status(event, command='pick',
                minimum_force=self.minimum_force,
                maximum_gripper_velocity_mps=self.maximum_gripper_velocity,
                control_mode=('stock_public_position' if self.stock_mode else 'pressure_admission'),
                contact_only=self.stock_mode, **self._master_geometry_diagnostic(), **fields)
        except Exception:
            pass

    @staticmethod
    def _require_pressure(evidence):
        if not evidence.verified:
            if evidence.reason in PRESSURE_DEFICIENCIES:
                raise _PressureDeficient(evidence.reason)
            raise LiftPressureRejected(evidence.reason)

    def _attempt(self, send_goal, overall_deadline):
        """Catch up to one fixed snapshot without holding callback locks."""
        n = self.node
        first_clock = int(n.get_clock().now().nanoseconds)
        deadline = min(overall_deadline, time.monotonic() + .5)
        with n._lock:
            fixed = self._snapshot_locked()
        try:
            self._hard_check(fixed, first_clock, allow_future=True, phase='initial_fixed')
            stamps = [s.stamp_ns for sides in fixed['histories'].values() for side in sides for s in side]
            stamps += [fixed['feedback'][-1].stamp_ns, *fixed['stamps'].values(), fixed['odom'].get('stamp_ns', 0)]
            latest = max(first_clock + 1, *stamps)
            now = first_clock
            while now < latest:
                if time.monotonic() >= deadline:
                    raise LiftPressureRejected('clock_catchup_timeout')
                with n._lock:
                    live = self._snapshot_locked()
                self._hard_check(live, now, allow_future=True, phase='catchup_live')
                time.sleep(.002)
                updated = int(n.get_clock().now().nanoseconds)
                if updated < now:
                    raise LiftPressureRejected('clock_reversed')
                now = updated
            with n._adaptive_command_guard():
                with n._lock:
                    now = int(n.get_clock().now().nanoseconds)
                    if now < first_clock or time.monotonic() >= deadline:
                        raise LiftPressureRejected('pressure_admission_deadline')
                    live = self._snapshot_locked()
                    self._hard_check(live, now, allow_future=True, phase='dispatch_live')
                    self._hard_check(fixed, now, allow_future=False, phase='dispatch_fixed')
                    evidence = self._evaluate(fixed, now, phase='fixed_pressure')
                    self._require_pressure(evidence)
                    # Late backfilled samples may invalidate the fixed suffix.
                    # Re-evaluate all currently confirmed producer observations;
                    # retain unconfirmed samples only as negative/hard guards.
                    confirmed = dict(live)
                    confirmed['histories'] = {model: tuple(tuple(
                        sample for sample in side if sample.stamp_ns <= now
                    ) for side in sides) for model, sides in live['histories'].items()}
                    confirmed['feedback'] = tuple(sample for sample in live['feedback']
                                                  if sample.stamp_ns <= now)
                    evidence = self._evaluate(confirmed, now, phase='confirmed_live_pressure')
                    self._require_pressure(evidence)
                    self._remember_input(live, now, 'latest_live_pressure')
                    for side in live['histories'].get(self.expected_model, ((), ())):
                        if not side or side[-1].force_newtons < self.minimum_force:
                            raise _PressureDeficient('pressure_changed_before_dispatch')
                        for sample in side:
                            if sample.stamp_ns > now and sample.force_newtons < self.minimum_force:
                                raise _PressureDeficient('pressure_changed_before_dispatch')
                    if time.monotonic() >= overall_deadline:
                        raise LiftPressureRejected('fresh_pressure_wait_timeout')
                    if n._cancel.is_set():
                        raise LiftPressureRejected('cancelled')
                    interval_admission = self.checked.get('preclose_interval_admission')
                    if interval_admission is not None:
                        interval_admission.consume(n, live, now)
                    self.send_started = True
                    result = send_goal()
            self._status('lift_first_pressure_admitted', left_force=evidence.left_force,
                right_force=evidence.right_force, model=self.expected_model,
                snapshot_stamp_ns=first_clock, dispatch_stamp_ns=now,
                width=evidence.width, read_only=True)
            return result
        except Exception as exc:
            # Do not translate unknown action acceptance into a safe rejection.
            raise
