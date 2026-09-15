"""Additional measured stop admission between normal PLACE transition legs0/1.

No motion publisher, scene relaxation, retry after a fault, or payload-monitor
suppression. The existing raw collector is exclusively borrowed until leg1's
sender returns; its bounded samples also explain a subsequent scene veto.
"""
from collections import deque
import math
import time

from .motion_profiles import IK_JOINTS, ARM_JOINTS
from .placement_scene_context import measured_scene_context
from .preopen_stationary import stationary_closed_sample, _attachment, _stock_retention_locked
from .release_evidence import stamp_valid
from .stock_gripper_close import enabled as stock_enabled


class TransitionStopRejected(RuntimeError):
    pass


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('place_transition_stop_enabled must be Boolean')
    return value


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def remember_scene_failure(node, now, odom, pose, speeds):
    """Best-effort exact failing-read diagnostics; never mask the original veto."""
    try:
        p = list(pose) if getattr(pose, 'shape', None) == (3,) else []
        s = [_finite(v) for v in speeds]
        predicates = []
        if len(p) != 3: predicates.append('pose_shape')
        elif any(_finite(v) is None for v in p): predicates.append('pose_nonfinite')
        if any(v is None for v in s): predicates.append('speed_nonfinite')
        if any(v is not None and v < 0 for v in s): predicates.append('speed_negative')
        if s[0] is not None and s[0] > .005: predicates.append('linear_speed_above_0.005')
        if s[1] is not None and s[1] > .008: predicates.append('angular_speed_above_0.008')
        details = dict(reason='placement_scene_base_not_stationary', evaluated_ros_ns=now,
            odom_stamp_ns=odom.get('stamp_ns'), odom_producer_stamp_ns=odom.get('producer_stamp_ns'),
            base_pose=[_finite(v) for v in p], linear_speed=s[0], angular_speed=s[1],
            failed_predicates=predicates)
        with node._lock:
            node._place_transition_scene_failure = details
    except Exception:
        pass


def diagnostic_fields(node, reason):
    if reason != 'placement_scene_base_not_stationary':
        return {}
    try:
        with node._lock:
            details = dict(getattr(node, '_place_transition_scene_failure', None) or {})
            owner = getattr(node, '_place_transition_stop_owner', None)
            raw = list(getattr(node, '_delivery_raw_samples', ())) if owner is not None else []
        window = []
        for sample in raw[-12:]:
            odom = sample.get('odom', {})
            window.append(dict(sequence=sample.get('sequence'), producer_stamp_ns=sample.get('producer_stamp_ns'),
                joint_velocities=[_finite(sample.get('velocities', {}).get(n)) for n in IK_JOINTS],
                odom_producer_stamp_ns=odom.get('stamp_ns'), linear_speed=_finite(odom.get('linear_speed')),
                angular_speed=_finite(odom.get('angular_speed'))))
        return dict(scene_stationarity_diagnostic=details, raw_velocity_window=window,
            scene_diagnostic_scope='latest failing scene read before this event; evaluated clock identifies the observation',
            velocity_window_scope='at most12 existing joint callbacks; no continuous coverage claim')
    except Exception:
        return {}


def _legs(legs):
    return tuple((tuple(float(v) for v in q), float(t), phase) for q, t, phase in legs)


def normal_options(node, identity, legs, arm_speed_scale, master):
    if not checked_enabled(getattr(node, 'place_transition_stop_enabled', False)):
        return {}
    if (not stock_enabled(node) or not getattr(node, 'delivery_evidence_enabled', False)
            or not getattr(node, 'table_scene_required', False) or not getattr(node, 'bin_scene_required', False)
            or getattr(node, 'loaded_place_speed_scale_cap', None) != 2.0
            or getattr(node, 'additional_arm_time_scale', None) != 2.0 or arm_speed_scale != 2.0):
        raise TransitionStopRejected('normal_registered_stock_cap2_place_required')
    route = _legs(legs)
    if (len(route) < 2 or any(len(q) != len(IK_JOINTS) or not all(math.isfinite(v) for v in q)
            for q, _, _ in route) or any(route[i][1:] != (.8, 'bin_transition') for i in (0, 1))):
        raise TransitionStopRejected('original_transition_prefix_required')
    return dict(transition_stop=Qualification(node, identity, route, master))


class Qualification:
    def __init__(self, node, identity, route, master):
        self.node, self.route = node, route
        self.master = float(master)
        if not math.isfinite(self.master) or not 0 <= self.master < .0685:
            raise TransitionStopRejected('invalid_closed_master_reference')
        self.identity = dict(identity or {})
        self.active = self.qualified = self.published = False
        self.publication_rejection = None
        with node._lock:
            self.reference = getattr(node, '_active_place_scene_reference', None)
            self.epoch = getattr(node, '_contact_epoch', None)
            self.target = getattr(node, '_target_book_model', None)
            self.attached = _attachment(node)
        if (self.reference is None or self.epoch is None or self.attached is None
                or not self.target or self.identity.get('target_model') != self.target
                or not self.identity.get('trial_id') or not self.identity.get('placement_attempt_id')):
            raise TransitionStopRejected('missing_registered_place_identity')

    def require_execution(self, node, legs, command, **options):
        if (node is not self.node or command != 'place' or _legs(legs) != self.route
                or options.get('arm_speed_scale') != 2.0 or options.get('leg_offset') != 0
                or options.get('fresh_retention_phases') or options.get('initial_pressure_gate') is not None
                or options.get('clearance_timing') is not None or options.get('withdrawal_timing') is not None
                or options.get('withdrawal_speed_scale') != 1.0 or self.active or self.qualified):
            raise TransitionStopRejected('transition_stop_execution_scope_changed')

    def _hard_locked(self, *, entering=False):
        n = self.node
        if n._cancel.is_set(): raise TransitionStopRejected('cancelled')
        if n._goal_handles or getattr(n, '_pending_retained_acceptances', ()):
            raise TransitionStopRejected('active_or_pending_goal')
        if (not checked_enabled(getattr(n, 'place_transition_stop_enabled', False))
                or n.loaded_place_speed_scale_cap != 2.0 or n.additional_arm_time_scale != 2.0
                or not stock_enabled(n) or not n.table_scene_required or not n.bin_scene_required
                or not n.delivery_evidence_enabled):
            raise TransitionStopRejected('transition_stop_options_changed')
        for key in ('_payload_hazard_latched', '_held_grip_sensor_fault', '_raw_contacts_first_failure', '_release_pose_fault_latched'):
            if getattr(n, key, None) is not None: raise TransitionStopRejected(str(getattr(n, key)))
        if getattr(n, '_target_robot_contact_latched', False): raise TransitionStopRejected('payload_robot_contact')
        if (n._active_place_scene_reference is not self.reference or n._contact_epoch != self.epoch
                or n._target_book_model != self.target or _attachment(n) != self.attached):
            raise TransitionStopRejected('retained_scene_identity_changed')
        if (not n._payload_monitor_enabled or n._retention_probe_active
                or getattr(n, '_gripper_open_confirmed', False)
                or getattr(n, '_release_pose_owner', None) is not None):
            raise TransitionStopRejected('retained_owner_state_changed')
        owner = getattr(n, '_place_transition_stop_owner', None)
        if entering:
            if owner is not None or getattr(n, '_delivery_measurement_active', False):
                raise TransitionStopRejected('raw_measurement_already_owned')
        elif owner is not self or not n._delivery_measurement_active:
            raise TransitionStopRejected('raw_measurement_ownership_changed')

    def _normal_clear(self):
        n = self.node
        fault = n._payload_hazard_reason(max_age=min(.15, n.grasp_contact_max_age))
        if fault is not None: raise TransitionStopRejected(fault)
        if not n._pinch_sample(max_age=min(.15, n.grasp_contact_max_age))[0]:
            raise TransitionStopRejected('closed_grip_not_retained')
        measured_scene_context(n, self.reference)

    def _sample(self, raw, now, *, pending=False):
        stamp = raw.get('producer_stamp_ns')
        if not stamp_valid(stamp): raise TransitionStopRejected('invalid_raw_joint_stamp')
        odom = raw.get('odom') or {}
        if odom and not stamp_valid(odom.get('stamp_ns')): raise TransitionStopRejected('invalid_raw_odom_stamp')
        for value in (odom.get('linear_speed', 0.), odom.get('angular_speed', 0.)):
            if _finite(value) is None or value < 0: raise TransitionStopRejected('invalid_raw_odom_speed')
        if odom.get('linear_speed', 0.) > .005 or odom.get('angular_speed', 0.) > .008:
            raise TransitionStopRejected('placement_scene_base_not_stationary')
        ok, reason = stationary_closed_sample(raw, self.route[0][0], self.master, now, pending=pending)
        if reason == 'joint_feedback_invalid': raise TransitionStopRejected(reason)
        return ok, reason

    def qualify(self):
        n = self.node
        self.started_wall = time.monotonic()
        self.deadline = self.started_wall + 3.
        simulated = bool(getattr(n.get_clock(), 'ros_time_is_active', False))
        total_deadline = self.started_wall + max(3., float(getattr(n, 'timeout', 120.)))
        self.entered = self.last_clock = int(n.get_clock().now().nanoseconds)
        progress_clock = self.entered
        progress_joint = progress_odom = None
        if not stamp_valid(self.entered): raise TransitionStopRejected('invalid_clock')
        self.first_joint = self.first_odom = None
        self.previous_joint = self.previous_odom = None
        self.seen = 0
        try:
            self._normal_clear()
            with n._lock:
                self._hard_locked(entering=True)
                n._delivery_raw_samples = deque(maxlen=128)
                n._delivery_raw_odom = None
                n._delivery_sample_sequence = 0
                n._place_transition_stop_owner = self
                n._delivery_measurement_active = True
                self.active = True
            while True:
                self._normal_clear()
                with n._lock:
                    self._hard_locked()
                    batch = list(n._delivery_raw_samples)
                now = int(n.get_clock().now().nanoseconds)
                if now < self.last_clock: raise TransitionStopRejected('clock_reversed')
                self.last_clock = now
                if now-self.entered >= 500_000_000 or time.monotonic() >= self.deadline:
                    raise TransitionStopRejected('transition_stop_deadline')
                if simulated and batch:
                    joint = batch[-1].get('producer_stamp_ns')
                    odom = (batch[-1].get('odom') or {}).get('stamp_ns')
                    if stamp_valid(joint) and stamp_valid(odom):
                        if progress_joint is None:
                            # Fresh producers can lag the current ROS clock.
                            # Observe their actual baseline without renewing
                            # the watchdog or counting pre-entry evidence.
                            progress_clock, progress_joint, progress_odom = now, joint, odom
                        elif (now > progress_clock and joint > progress_joint
                                and odom > progress_odom):
                            # Physical settling is bounded by the unchanged 0.5 s
                            # ROS window. Three wall seconds bounds missing clock
                            # or feedback progress, independently of Gazebo's rate.
                            self.deadline = min(total_deadline, time.monotonic()+3.)
                            progress_clock, progress_joint, progress_odom = now, joint, odom
                unseen = [s for s in batch if s['sequence'] > self.seen]
                if unseen and unseen[0]['sequence'] != self.seen+1:
                    raise TransitionStopRejected('raw_history_gap')
                for raw in unseen:
                    stamp = raw['producer_stamp_ns']; odom = raw.get('odom') or {}; os = odom.get('stamp_ns')
                    # Check negative evidence even while DDS clock delivery catches up.
                    self._sample(raw, now, pending=True)
                    if (self.previous_joint is not None and stamp < self.previous_joint
                            or os is not None and self.previous_odom is not None and os < self.previous_odom):
                        raise TransitionStopRejected('producer_stamp_reversed')
                    if stamp > now or (os is not None and os > now):
                        if max(stamp, os or 0)-now > 100_000_000:
                            raise TransitionStopRejected('producer_too_far_future')
                        break
                    ok, reason = self._sample(raw, now)
                    self.seen = raw['sequence']
                    if stamp <= self.entered or os is None or os <= self.entered:
                        ok = False
                    gap = (self.previous_joint is not None and stamp-self.previous_joint > 75_000_000
                           or os is not None and self.previous_odom is not None and os-self.previous_odom > 75_000_000)
                    if not ok or gap:
                        self.first_joint = self.first_odom = None
                    if ok and self.first_joint is None:
                        self.first_joint, self.first_odom = stamp, os
                    self.previous_joint, self.previous_odom = stamp, os
                if (self.first_joint is not None and self.previous_joint-self.first_joint >= 100_000_000
                        and self.previous_odom-self.first_odom >= 100_000_000):
                    with n._adaptive_command_guard():
                        self._normal_clear()
                        with n._lock:
                            self._hard_locked()
                            self._current_locked()
                            self.qualified = True
                    self._report(True)
                    return
                time.sleep(.002)
        except BaseException as exc:
            self._report(False, str(exc))
            self.close()
            raise

    def _current_locked(self):
        n = self.node; now = int(n.get_clock().now().nanoseconds)
        if now < self.last_clock or now-self.entered >= 500_000_000 or time.monotonic() >= self.deadline:
            raise TransitionStopRejected('transition_stop_admission_expired')
        raw = list(n._delivery_raw_samples)
        if not raw: raise TransitionStopRejected('raw_feedback_missing')
        newer = [s for s in raw if s['sequence'] > self.seen]
        last_joint, last_odom = self.previous_joint, self.previous_odom
        if newer and newer[0]['sequence'] != self.seen+1: raise TransitionStopRejected('raw_history_gap')
        for sample in newer:
            js=sample['producer_stamp_ns']; os=(sample.get('odom') or {}).get('stamp_ns')
            if (not stamp_valid(js) or not stamp_valid(os)
                    or not 0 <= js-last_joint <= 75_000_000 or not 0 <= os-last_odom <= 75_000_000):
                raise TransitionStopRejected('producer_chronology_changed')
            if not self._sample(sample, now, pending=True)[0]: raise TransitionStopRejected('newer_stopped_feedback_invalid')
            last_joint,last_odom=js,os
        latest=raw[-1]
        current=dict(latest, odom=dict(n._delivery_raw_odom or {}))
        current_odom_stamp=current['odom'].get('stamp_ns')
        if not stamp_valid(current_odom_stamp) or not 0 <= current_odom_stamp-last_odom <= 75_000_000:
            raise TransitionStopRejected('latest_odom_chronology_changed')
        # Actual qualification endpoint and newest unpaired odometry must be fresh.
        if (not 0 <= now-self.previous_joint <= 150_000_000
                or not 0 <= now-self.previous_odom <= 150_000_000
                or not self._sample(latest, now, pending=True)[0]
                or not self._sample(current, now, pending=True)[0]):
            raise TransitionStopRejected('stopped_feedback_no_longer_current')
        fault = _stock_retention_locked(n, now, min(.15, n.grasp_contact_max_age))
        if fault is not None: raise TransitionStopRejected(fault)

    def require_publication_locked(self, node, goal, command, leg_offset):
        try:
            self._require_publication_locked(node, goal, command, leg_offset)
        except TransitionStopRejected as exc:
            # Exact exception identity certifies failure before pending registration/send.
            self.publication_rejection = exc
            raise

    def _require_publication_locked(self, node, goal, command, leg_offset):
        if (node is not self.node or not self.qualified or self.published
                or command != 'place' or leg_offset != 1
                or tuple(goal.trajectory.joint_names) != tuple(ARM_JOINTS)
                or len(goal.trajectory.points) != 1
                or tuple(goal.trajectory.points[0].positions) != self.route[1][0][1:]):
            raise TransitionStopRejected('transition_stop_publication_scope_changed')
        self._hard_locked()
        self._current_locked()
        self.published = True

    def _report(self, verified, reason=None):
        try:
            self.node._publish_status('placement_transition_stop', **self.identity, command='place',
                after_leg=0, before_leg=1, verified=verified, reason=reason,
                stationary_joint_start_ns=self.first_joint, stationary_odom_start_ns=self.first_odom,
                latest_joint_stamp_ns=self.previous_joint, latest_odom_stamp_ns=self.previous_odom,
                elapsed_wall_seconds=time.monotonic()-self.started_wall,
                elapsed_ros_ns=self.last_clock-self.entered,
                book_stationarity_verified=False, physical_inside_verified=None,
                **diagnostic_fields(self.node, reason))
        except Exception:
            pass

    def close(self):
        if self.active:
            with self.node._lock:
                if getattr(self.node, '_place_transition_stop_owner', None) is self:
                    self.node._delivery_measurement_active = False
                    self.node._place_transition_stop_owner = None
            self.active = False
