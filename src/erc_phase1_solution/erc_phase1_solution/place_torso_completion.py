"""Registered loaded PLACE torso motion and measured arrival, before any arm leg.

The accepted plan supplies the same fixed arm and torso endpoints. Only the
torso duration changes; no motion/geometry shortcut or recovery is authorized.
"""
from collections import deque
from copy import deepcopy
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from .motion_profiles import IK_JOINTS
from .placement_scene_context import measured_scene_context
from .preopen_stationary import stationary_closed_sample, _stock_retention_locked
from .release_evidence import stamp_valid

JOINT = 'torso_lift_joint'
MASTER = 'gripper_left_finger_joint'


class PlaceTorsoRejected(RuntimeError):
    """Stop with the book retained; an unverified torso endpoint cannot recover."""


def _vector(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (8,) or not np.isfinite(q).all():
        raise PlaceTorsoRejected('place_torso_invalid_checked_joints')
    return q.copy()


def _signature(plan):
    from .scene_checked_place import SceneCheckedPlacePlan
    if type(plan) is not SceneCheckedPlacePlan:
        raise PlaceTorsoRejected('place_torso_requires_checked_plan')
    try:
        detail = plan.diagnostics
        if detail['registered_bin_scene'] is None or detail['whole_table_pose_modeled'] is not True:
            raise ValueError('registered scene required')
        first, last = (_vector(detail[key]) for key in ('actual_carry_start', 'torso_ready'))
        if not np.array_equal(first[1:], last[1:]) or not plan.setup or not plan.solutions:
            raise ValueError('fixed arm required')
        # All accepted route coordinates are bound, not just the torso target.
        route = tuple(_vector(q).tobytes() for q in (*plan.setup, *plan.solutions))
        master = float(detail['measured_master'])
        if not math.isfinite(master) or not 0 <= master < .0685:
            raise ValueError('closed master required')
        return first.tobytes(), last.tobytes(), route, master
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise PlaceTorsoRejected('place_torso_invalid_checked_plan') from error


def execute_registered_place_torso(node, plan, identity, resolve_package, *, planned_start=None):
    """No False return: any failed/uncertain motion or arrival is a retained stop."""
    owner = PlaceTorsoCompletion(node, plan, identity, resolve_package)
    try:
        owner.check()
        # _move_torso keeps its original, stricter completed-target hold first.
        # Its existing no-command result is marked explicitly by the owner.
        if not node._move_torso(owner.target[0], 2.2, allow_completed_hold=True,
                place_torso_owner=owner,
                **({'planned_start': planned_start} if planned_start is not None else {})):
            raise PlaceTorsoRejected('place_torso_action_failed')
        if owner.skipped:
            return True
        if not owner.sent:
            raise PlaceTorsoRejected('place_torso_action_not_admitted')
        owner.wait_for_arrival()
        return True
    except BaseException as error:
        owner.report('placement_torso_failed', verified=False, reason=str(error))
        raise
    finally:
        owner.close()


class PlaceTorsoCompletion:
    def __init__(self, node, plan, identity, resolve_package):
        from .empty_pickup_collision import joint_velocity_limits
        self.node, self.plan = node, plan
        self.signature = _signature(plan)
        self.start = np.frombuffer(self.signature[0], dtype=np.float64)
        self.target = np.frombuffer(self.signature[1], dtype=np.float64)
        self.master = self.signature[3]
        self.identity = dict(identity or {})
        self.active = self.sent = self.skipped = False
        self.motion = None
        self.last_reads = deque(maxlen=12)
        self.read_count = self.accepted_count = 0
        self.reasons = {}
        self.started = self.last_clock = self.first_joint = self.first_odom = None
        self.previous_joint = self.previous_odom = None
        self.started_wall = None
        urdf = Path(resolve_package('erc_description')) / 'urdf/tiago_pro.urdf'
        self.urdf_sha256 = hashlib.sha256(urdf.read_bytes()).hexdigest()
        self.velocity_limit = float(joint_velocity_limits(urdf)[0])
        with node._lock:
            self.reference = getattr(node, '_active_place_scene_reference', None)
            self.selected_reference = getattr(node, '_selected_place_scene_reference', None)
            self.bin = getattr(node, '_selected_place_bin_scene', None)
            self.table = getattr(node, '_selected_place_table_scene', None)
            self.reference_copy = deepcopy(self.reference)
            self.bin_copy, self.table_copy = deepcopy(self.bin), deepcopy(self.table)
            self.attached = node._held_book_corners
            self.attached_bytes = np.asarray(self.attached, dtype=np.float64).tobytes()
            self.epoch = getattr(node, '_contact_epoch', None)
            self.contact_guard = getattr(node, '_place_contact_guard', None)
            self.guard_correlation = deepcopy(getattr(self.contact_guard, 'correlation', None))
            self.target_model = getattr(node, '_target_book_model', None)
            self.bounds = (float(node.chain.lower[0]), float(node.chain.upper[0]))
        from .place_contact_guard import PlaceContactGuard
        if (type(self.contact_guard) is not PlaceContactGuard
                or type(self.contact_guard.started_ns) is not int or self.contact_guard.started_ns <= 0
                or any(self.guard_correlation.get(key) != value for key, value in self.identity.items())
                or not all(type(v) is dict for v in (self.reference, self.bin, self.table))
                or self.reference is not self.selected_reference
                or self.bin_copy != plan.diagnostics['registered_bin_scene']
                or self.table_copy != plan.diagnostics.get('table_scene')
                or np.asarray(self.attached).shape != (8, 3)
                or not np.isfinite(self.attached).all()
                or self.epoch is None or not self.target_model
                or self.identity.get('target_model') != self.target_model
                or not self.identity.get('trial_id') or not self.identity.get('placement_attempt_id')
                or not self.bounds[0] <= self.target[0] <= self.bounds[1]):
            raise PlaceTorsoRejected('place_torso_missing_registered_binding')

    def locked(self, *, collecting=False):
        n = self.node
        if n._cancel.is_set():
            raise PlaceTorsoRejected('place_torso_cancelled')
        if (not n.table_scene_required or not n.bin_scene_required or not n.delivery_evidence_enabled
                or getattr(n, '_active_place_scene_reference', None) is not self.reference
                or getattr(n, '_selected_place_scene_reference', None) is not self.selected_reference
                or getattr(n, '_selected_place_bin_scene', None) is not self.bin
                or getattr(n, '_selected_place_table_scene', None) is not self.table
                or self.reference != self.reference_copy or self.bin != self.bin_copy or self.table != self.table_copy
                or n._held_book_corners is not self.attached
                or np.asarray(self.attached).shape != (8, 3)
                or np.asarray(self.attached, dtype=np.float64).tobytes() != self.attached_bytes
                or n._contact_epoch != self.epoch or n._target_book_model != self.target_model
                or getattr(n, '_place_contact_guard', None) is not self.contact_guard
                or self.contact_guard.correlation != self.guard_correlation
                or _signature(self.plan) != self.signature
                or (float(n.chain.lower[0]), float(n.chain.upper[0])) != self.bounds):
            raise PlaceTorsoRejected('place_torso_checked_binding_changed')
        if n._goal_handles or getattr(n, '_pending_retained_acceptances', ()):
            raise PlaceTorsoRejected('place_torso_controller_not_idle')
        for name in ('_payload_hazard_latched', '_held_grip_sensor_fault',
                     '_raw_contacts_first_failure', '_release_pose_fault_latched'):
            if getattr(n, name, None) is not None:
                raise PlaceTorsoRejected(str(getattr(n, name)))
        if (not n._payload_monitor_enabled or n._retention_probe_active
                or getattr(n, '_target_robot_contact_latched', False)
                or getattr(n, '_gripper_open_confirmed', False)
                or getattr(n, '_release_pose_owner', None) is not None
                or getattr(n, '_place_transition_stop_owner', None) is not None):
            raise PlaceTorsoRejected('place_torso_retained_owner_changed')
        owner = getattr(n, '_place_torso_arrival_owner', None)
        if (collecting and (owner is not self or not n._delivery_measurement_active)
                or not collecting and (owner is not None or getattr(n, '_delivery_measurement_active', False))):
            raise PlaceTorsoRejected('place_torso_collector_ownership_changed')
        fault = _stock_retention_locked(n, int(n.get_clock().now().nanoseconds), min(.15, n.grasp_contact_max_age))
        if fault is not None:
            raise PlaceTorsoRejected(fault)

    def check(self):
        n = self.node
        from .place_contact_guard import require_clear
        require_clear(n)
        measured_scene_context(n, self.reference)
        fault = n._payload_hazard_reason(max_age=min(.15, n.grasp_contact_max_age))
        if fault is not None or not n._pinch_sample(max_age=min(.15, n.grasp_contact_max_age))[0]:
            raise PlaceTorsoRejected(fault or 'place_torso_grip_not_retained')
        with n._lock:
            self.locked(collecting=self.active)

    def hold_admitted(self):
        if self.sent or self.active:
            raise PlaceTorsoRejected('place_torso_hold_after_motion')
        # The existing completed-hold helper just performed its stricter fresh
        # 1 um / 1 um/s checks and generation/scene/retention admissions.
        self.check()
        self.skipped = True
        self.report('placement_torso_motion_admitted', verified=True,
                    controller_command_sent=False, completed_hold_preserved=True)

    def admit_locked(self, client, goal, duration):
        n = self.node
        self.locked()
        if (self.sent or self.skipped or self.active or client is not n.torso_client
                or duration != 2.2 or tuple(goal.trajectory.joint_names) != (JOINT,)
                or len(goal.trajectory.points) != 1):
            raise PlaceTorsoRejected('place_torso_command_scope_changed')
        point = goal.trajectory.points[0]
        if (tuple(point.positions) != (self.target[0],)
                or point.velocities or point.accelerations or point.effort):
            raise PlaceTorsoRejected('place_torso_goal_changed')
        now = int(n.get_clock().now().nanoseconds)
        actual = []
        for index, name in enumerate(IK_JOINTS):
            position = n.joints.get(name, math.nan)
            speed = n._joint_velocities.get(name, math.nan)
            stamp = n._joint_stamps_ns.get(name)
            if (not stamp_valid(stamp) or not -50_000_000 <= now-stamp <= 150_000_000
                    or not math.isfinite(position) or not math.isfinite(speed)
                    or abs(position-self.start[index]) > (.001 if index == 0 else .002)
                    or abs(speed) > .001):
                raise PlaceTorsoRejected('place_torso_fresh_checked_start_changed:'+name)
            actual.append(float(position))
        if not self.bounds[0] <= actual[0] <= self.bounds[1]:
            raise PlaceTorsoRejected('place_torso_start_outside_limits')
        delta = abs(Fraction(float(self.target[0]))-Fraction(actual[0]))
        required = delta*1_000_000_000/(Fraction(4, 5)*Fraction(self.velocity_limit))
        ns = max(2_200_000_000, -(-required.numerator//required.denominator))
        if ns > 2**31*1_000_000_000-1:
            raise PlaceTorsoRejected('place_torso_duration_overflow')
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1_000_000_000)
        if delta*1_000_000_000 > Fraction(4, 5)*Fraction(self.velocity_limit)*ns:
            raise PlaceTorsoRejected('place_torso_serialized_velocity_exceeded')
        self.motion = dict(fresh_start=actual, target=self.target.tolist(),
            start_producer_stamp_ns=n._joint_stamps_ns[JOINT], admitted_ros_ns=now,
            original_duration_ns=2_200_000_000, command_duration_ns=ns,
            urdf_velocity_limit=self.velocity_limit, velocity_headroom_factor=.8,
            urdf_sha256=self.urdf_sha256)
        self.sent = True
        # The same value owns the existing generic action watchdog.
        return ns/1_000_000_000

    def _sample(self, raw, now, *, pending=False):
        stamp = raw.get('producer_stamp_ns')
        odom = raw.get('odom') or {}
        if not stamp_valid(stamp) or odom and not stamp_valid(odom.get('stamp_ns')):
            raise PlaceTorsoRejected('place_torso_invalid_producer_stamp')
        for name, maximum in (('linear_speed', .005), ('angular_speed', .008)):
            value = odom.get(name, 0.)
            if not math.isfinite(value) or not 0 <= value <= maximum:
                raise PlaceTorsoRejected('place_torso_invalid_or_moving_base')
        ok, reason = stationary_closed_sample(raw, self.target, self.master, now, pending=pending)
        if reason == 'joint_feedback_invalid':
            raise PlaceTorsoRejected(reason)
        if not pending:
            self.read_count += 1
            self.accepted_count += int(ok)
            self.reasons[reason] = self.reasons.get(reason, 0)+1
            names = (*IK_JOINTS, MASTER)
            self.last_reads.append(dict(producer_stamp_ns=stamp, odom_stamp_ns=odom.get('stamp_ns'),
                evaluated_ros_ns=now, accepted=ok, reason=reason,
                positions=[raw.get('positions', {}).get(name) for name in names],
                errors=[raw.get('positions', {}).get(name, math.nan)-target
                        for name, target in zip(names, (*self.target, self.master))],
                velocities=[raw.get('velocities', {}).get(name) for name in names]))
        return ok

    def wait_for_arrival(self):
        n = self.node
        self.started = self.last_clock = int(n.get_clock().now().nanoseconds)
        self.started_wall = time.monotonic()
        self.deadline = self.started_wall+3.
        total_deadline = self.started_wall+max(3., float(n.timeout))
        if not stamp_valid(self.started):
            raise PlaceTorsoRejected('place_torso_invalid_clock')
        progress = None
        self.seen = 0
        self.report('placement_torso_arrival_started', verified=False)
        with n._adaptive_command_guard():
            self.check()
            with n._lock:
                self.locked()
                n._delivery_raw_samples = deque(maxlen=128)
                n._delivery_raw_odom = None
                n._delivery_sample_sequence = 0
                n._place_torso_arrival_owner = self
                n._delivery_measurement_active = self.active = True
        while True:
            self.check()
            with n._lock:
                self.locked(collecting=True)
                batch = list(n._delivery_raw_samples)
            now = int(n.get_clock().now().nanoseconds)
            if now < self.last_clock:
                raise PlaceTorsoRejected('place_torso_clock_reversed')
            self.last_clock = now
            if now-self.started >= 500_000_000 or time.monotonic() >= self.deadline:
                raise PlaceTorsoRejected('place_torso_arrival_deadline')
            if getattr(n.get_clock(), 'ros_time_is_active', False) and batch:
                js = batch[-1].get('producer_stamp_ns')
                os = (batch[-1].get('odom') or {}).get('stamp_ns')
                if stamp_valid(js) and stamp_valid(os):
                    if progress is None:
                        progress = now, js, os
                    elif all(a > b for a, b in zip((now, js, os), progress)):
                        self.deadline = min(total_deadline, time.monotonic()+3.)
                        progress = now, js, os
            unseen = [raw for raw in batch if raw['sequence'] > self.seen]
            if unseen and unseen[0]['sequence'] != self.seen+1:
                raise PlaceTorsoRejected('place_torso_raw_history_gap')
            for raw in unseen:
                js, os = raw['producer_stamp_ns'], (raw.get('odom') or {}).get('stamp_ns')
                self._sample(raw, now, pending=True)
                if (self.previous_joint is not None and js < self.previous_joint
                        or os is not None and self.previous_odom is not None and os < self.previous_odom):
                    raise PlaceTorsoRejected('place_torso_producer_reversed')
                if js > now or os is not None and os > now:
                    if max(js, os or 0)-now > 100_000_000:
                        raise PlaceTorsoRejected('place_torso_producer_future')
                    break
                ok = self._sample(raw, now)
                self.seen = raw['sequence']
                if js <= self.started or os is None or os <= self.started:
                    ok = False
                gap = (self.previous_joint is not None and js-self.previous_joint > 75_000_000
                       or os is not None and self.previous_odom is not None and os-self.previous_odom > 75_000_000)
                if not ok or gap:
                    self.first_joint = self.first_odom = None
                if ok and self.first_joint is None:
                    self.first_joint, self.first_odom = js, os
                self.previous_joint, self.previous_odom = js, os
            if (self.first_joint is not None and self.previous_joint-self.first_joint >= 100_000_000
                    and self.previous_odom-self.first_odom >= 100_000_000):
                with n._adaptive_command_guard():
                    self.check()
                    with n._lock:
                        self.locked(collecting=True)
                        self._current_locked()
                self.report('placement_torso_arrival_verified', verified=True)
                return
            time.sleep(.002)

    def _current_locked(self):
        n = self.node
        now = int(n.get_clock().now().nanoseconds)
        if now < self.last_clock or now-self.started >= 500_000_000 or time.monotonic() >= self.deadline:
            raise PlaceTorsoRejected('place_torso_arrival_expired')
        batch = list(n._delivery_raw_samples)
        if not batch:
            raise PlaceTorsoRejected('place_torso_raw_feedback_missing')
        js, os = self.previous_joint, self.previous_odom
        unseen = [raw for raw in batch if raw['sequence'] > self.seen]
        if unseen and unseen[0]['sequence'] != self.seen+1:
            raise PlaceTorsoRejected('place_torso_raw_history_gap')
        for raw in unseen:
            a, b = raw.get('producer_stamp_ns'), (raw.get('odom') or {}).get('stamp_ns')
            if (not stamp_valid(a) or not stamp_valid(b) or not 0 <= a-js <= 75_000_000
                    or not 0 <= b-os <= 75_000_000 or not self._sample(raw, now, pending=True)):
                raise PlaceTorsoRejected('place_torso_latest_feedback_changed')
            js, os = a, b
        latest = batch[-1]
        current = dict(latest, odom=dict(n._delivery_raw_odom or {}))
        current_os = current['odom'].get('stamp_ns')
        if (not stamp_valid(current_os) or not 0 <= current_os-os <= 75_000_000
                or not 0 <= now-self.previous_joint <= 150_000_000
                or not 0 <= now-self.previous_odom <= 150_000_000
                or not self._sample(latest, now, pending=True)
                or not self._sample(current, now, pending=True)):
            raise PlaceTorsoRejected('place_torso_final_feedback_not_current')

    def report(self, event, **fields):
        try:
            self.node._publish_status(event, command='place', **self.identity, **fields,
                motion=self.motion, target=self.target.tolist(),
                samples_evaluated=self.read_count, accepted_samples=self.accepted_count,
                reason_counts=dict(self.reasons), recent_samples=list(self.last_reads),
                reference_positions=[*self.target.tolist(), self.master],
                joint_names=[*IK_JOINTS, MASTER], position_tolerances=[.001, *([.002]*7), .0005],
                velocity_limits=[*([.001]*8), .0001],
                stationary_joint_start_ns=self.first_joint, stationary_odom_start_ns=self.first_odom,
                elapsed_ros_ns=None if self.started is None else self.last_clock-self.started,
                elapsed_wall_seconds=None if self.started_wall is None else time.monotonic()-self.started_wall,
                scope='Checked fixed-arm torso motion; fresh closed endpoint before any PLACE arm leg')
        except Exception:
            pass

    def close(self):
        if self.active:
            with self.node._lock:
                if getattr(self.node, '_place_torso_arrival_owner', None) is self:
                    self.node._delivery_measurement_active = False
                    self.node._place_torso_arrival_owner = None
            self.active = False
