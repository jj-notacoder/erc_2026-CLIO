"""Opt-in bounded closed-hand velocity admission before normal PLACE opening.

Uses the existing raw collectors, never an evaluator pose or a geometry cache.
The borrowed post-open bounds describe measured actuator/base velocities only;
they do not establish book stationarity, release dynamics or landing clearance.
"""
from collections import deque
import math
import time

from .motion_profiles import IK_JOINTS
from .placement_scene_context import measured_scene_context
from .release_evidence import stamp_valid
from .stock_gripper_close import enabled as stock_enabled, measured_position_in_range


def _stock_retention_locked(node, now_ns, max_age):
    """Recheck the existing selected-stock retention predicates under sensor lock.

    No lock acquisition, callbacks, waits, publication or force-policy changes.
    This fresh snapshot qualifies ordinary generation advances only; the caller
    still performs the original scene/retention checks and all hard interlocks.
    """
    guard = getattr(node, '_place_contact_guard', None)
    if guard is not None and guard.fault is not None:
        return 'place_scene_contact'
    raw_fault = getattr(node, '_held_grip_sensor_fault', None)
    if raw_fault:
        return str(raw_fault)
    history = getattr(node, '_gripper_feedback_samples', ())
    feedback = history[-1] if history else None
    if (feedback is None or not all(math.isfinite(v) for v in
            (feedback.position, feedback.velocity, feedback.effort))
            or not -100_000_000 <= now_ns-feedback.stamp_ns <= 150_000_000):
        return 'invalid_stock_retained_feedback'
    latched = getattr(node, '_payload_hazard_latched', None)
    if latched is not None:
        return str(latched)
    if bool(getattr(node, '_target_robot_contact_latched', False)):
        return 'payload_robot_contact'
    # Exact _contact_timestamp_is_recent arithmetic and supported-side policy.
    def recent(stamp):
        return stamp > 0 and -.10 <= (now_ns-stamp)/1e9 <= max_age
    left = recent(node._left_target_contact_ns)
    right = recent(node._right_target_contact_ns)
    retained = left if getattr(node, '_gravity_supported_payload', False) else left and right
    if not retained:
        return 'contact_lost'
    width = float(node.joints.get('gripper_left_finger_joint', 0.0))
    if not measured_position_in_range(width):
        return 'closed_grip_not_retained'
    return None


class PreopenStationaryRejected(RuntimeError):
    """Admission failed before any opening or unloaded return is authorized."""


def stationary_closed_sample(raw, goal, master, now_ns, *, pending=False):
    """Same arm/odom bounds as measured_pose_is_open; closed master is explicit."""
    earliest = -100_000_000 if pending else 0
    stamp = raw.get('producer_stamp_ns') if raw else None
    if not stamp_valid(stamp) or not earliest <= now_ns-stamp <= 150_000_000:
        return False, 'raw_joint_feedback_unconfirmed'
    q, v = raw.get('positions', {}), raw.get('velocities', {})
    names = (*IK_JOINTS, 'gripper_left_finger_joint')
    if any(not math.isfinite(q.get(n, math.nan)) or
           not math.isfinite(v.get(n, math.nan)) for n in names):
        return False, 'joint_feedback_invalid'
    if (abs(q['gripper_left_finger_joint']-master) > .0005 or
            abs(v['gripper_left_finger_joint']) > .0001):
        return False, 'master_not_at_stationary_closed_reference'
    for name, target in zip(IK_JOINTS, goal):
        tolerance = .001 if name == 'torso_lift_joint' else .002
        if abs(q[name]-target) > tolerance or abs(v[name]) > .001:
            return False, 'arm_not_at_stationary_checked_endpoint'
    odom = raw.get('odom')
    if (not odom or not stamp_valid(odom.get('stamp_ns')) or
            not earliest <= now_ns-odom['stamp_ns'] <= 150_000_000 or
            not 0 <= odom.get('linear_speed', math.inf) <= .005 or
            not 0 <= odom.get('angular_speed', math.inf) <= .008):
        return False, 'base_not_measured_stationary'
    return True, 'stationary_closed_at_checked_endpoint'


def _attachment(node):
    held = getattr(node, '_held_book_corners', None)
    if held is None:
        return None
    try:
        value = tuple(tuple(float(v) for v in row) for row in held)
    except (TypeError, ValueError, OverflowError):
        return None
    if (len(value) != 8 or any(len(row) != 3 for row in value) or
            any(not math.isfinite(v) for row in value for v in row)):
        return None
    return value


def require_stationary_closed_pose(node, identity, goal, master):
    """Require a new 100 ms raw span, or raise before _open_gripper is called.

    Ten wall seconds maximum, including a fixed-frame clock catch-up with a
    0.5 wall-second no-progress watchdog when simulation time is active;
    150 ms freshness, <=75 ms inter-sample gaps, <=100 ms future catch-up.
    Normal contact monitoring remains enabled. No lock spans waits or reports.
    A fresh final context check is additional to _open_gripper's existing check.
    """
    started_wall = time.monotonic()
    deadline = started_wall+10.
    simulated = bool(getattr(node.get_clock(), 'ros_time_is_active', False))
    owned = False
    last_clock = entered = int(node.get_clock().now().nanoseconds)
    first = previous = None
    seen = 0
    reason = 'measurement_timeout'
    fields = None
    diagnostics = dict(samples_seen=0, stationary_samples=0, batch_gap_resets=0,
        invalid_sample_resets=0, producer_gap_resets=0, future_waits=0, future_wait_expired=0,
        catchup_disturbances=0, final_attempts=0, final_raw_retries=0,
        generation_retries=0, generation_advances_revalidated=0,
        maximum_stationary_span_ns=0, last_final_veto=None)

    def reject(why):
        raise PreopenStationaryRejected(str(why))

    def report(verified, why=None, **extra):
        # Observability must neither authorize an opening nor mask a refusal.
        try:
            node._publish_status('placement_preopen_stationary',
                **dict(identity or {}), command='place', verified=verified,
                reason=why, basis='raw_actuated_feedback_before_open',
                book_stationarity_verified=False, physical_inside_verified=None,
                timing_diagnostics=dict(diagnostics,
                    elapsed_wall_seconds=max(0., time.monotonic()-started_wall),
                    elapsed_ros_ns=last_clock-entered, last_sample_reason=reason),
                **extra)
        except Exception:
            pass

    try:
        goal = tuple(float(v) for v in goal)
        master = float(master)
        if (len(goal) != len(IK_JOINTS) or not all(math.isfinite(v) for v in goal)
                or not math.isfinite(master) or not 0 <= master <= .069
                or not math.isfinite(node.gripper_open)
                or not master < node.gripper_open-.0005):
            reject('invalid_checked_closed_reference')
        if (not getattr(node, 'table_scene_required', False) or
                not getattr(node, 'bin_scene_required', False)):
            reject('registered_place_scene_required')
        with node._lock:
            reference = getattr(node, '_active_place_scene_reference', None)
            attached = _attachment(node)
            epoch = getattr(node, '_contact_epoch', None)
            target = getattr(node, '_target_book_model', None)
            if reference is None or attached is None or epoch is None or not target:
                reject('missing_retained_scene_identity')
            if identity is not None and identity.get('target_model') != target:
                reject('placement_target_identity_changed')
            if getattr(node, '_delivery_measurement_active', False):
                reject('raw_measurement_already_active')
            node._delivery_raw_samples = deque(maxlen=128)
            node._delivery_raw_odom = None
            node._delivery_sample_sequence = 0
            node._delivery_measurement_active = True
            owned = True

        def hard_fault_locked():
            # Command -> sensor ordering at final admission. No helper reentry.
            if node._cancel.is_set():
                return 'cancelled'
            if node._goal_handles or getattr(node, '_pending_retained_acceptances', ()):
                return 'active_or_pending_goal'
            for name in ('_payload_hazard_latched', '_held_grip_sensor_fault', '_raw_contacts_first_failure'):
                if getattr(node, name, None) is not None:
                    return str(getattr(node, name))
            if getattr(node, '_target_robot_contact_latched', False):
                return 'payload_robot_contact'
            if (getattr(node, '_contact_epoch', None) != epoch or
                    getattr(node, '_target_book_model', None) != target):
                return 'contact_identity_changed'
            if _attachment(node) != attached:
                return 'attachment_changed'
            if getattr(node, '_active_place_scene_reference', None) is not reference:
                return 'scene_reference_changed'
            if getattr(node, '_gripper_open_confirmed', False):
                return 'gripper_open_still_confirmed'
            if not getattr(node, '_payload_monitor_enabled', False):
                return 'payload_monitor_disabled'
            if getattr(node, '_retention_probe_active', False):
                return 'retention_probe_active'
            if not node._delivery_measurement_active:
                return 'raw_measurement_deactivated'
            return None

        def snapshot():
            with node._lock:
                fault = hard_fault_locked()
                return list(node._delivery_raw_samples), fault, getattr(node, '_contact_generation', 0)

        def retained_clear():
            # Both helpers acquire the plain sensor lock themselves.
            fault = node._payload_hazard_reason(max_age=min(.15, node.grasp_contact_max_age))
            if fault is not None:
                reject(fault)
            if not node._pinch_sample(max_age=min(.15, node.grasp_contact_max_age))[0]:
                reject('closed_grip_not_retained')

        def valid(sample, now, pending=False):
            return stationary_closed_sample(sample, goal, master, now, pending=pending)

        while time.monotonic() < deadline:
            batch, fault, _ = snapshot()
            if fault:
                reject(fault)
            retained_clear()
            now = int(node.get_clock().now().nanoseconds)
            if now < last_clock:
                reject('clock_reversed')
            last_clock = now
            batch = [sample for sample in batch if sample['sequence'] > seen]
            if not batch:
                time.sleep(.002)
                continue
            if batch[0]['sequence'] != seen+1:
                diagnostics['batch_gap_resets'] += 1
                first = previous = None
            fixed = batch[-1]
            future = max(fixed['producer_stamp_ns'], fixed['odom'].get('stamp_ns', 0))
            catchup = min(deadline, time.monotonic()+.5)
            disturbed = False
            if now < future <= now+100_000_000:
                diagnostics['future_waits'] += 1
            while now < future <= now+100_000_000 and time.monotonic() < catchup:
                live, fault, _ = snapshot()
                if fault:
                    reject(fault)
                retained_clear()
                disturbed |= any(not valid(sample, now, True)[0]
                    for sample in live if sample['sequence'] > seen)
                time.sleep(.002)
                updated = int(node.get_clock().now().nanoseconds)
                if updated < now:
                    reject('clock_reversed')
                if simulated and updated > now:
                    # A permitted 100 ms ROS lead takes more than 0.5 wall
                    # seconds below RTF 0.2. Keep waiting for this fixed frame
                    # while /clock advances, within the original total bound.
                    catchup = min(deadline, time.monotonic()+.5)
                now = updated
            last_clock = now
            if now < future:
                diagnostics['future_wait_expired'] += 1
            if disturbed:
                diagnostics['catchup_disturbances'] += 1
                first = previous = None
            for sample in batch:
                stamp = sample['producer_stamp_ns']
                ok, reason = valid(sample, now)
                if stamp <= entered:
                    ok = False
                if previous is not None and stamp < previous:
                    reject('joint_stamp_reversed')
                seen = sample['sequence']
                diagnostics['samples_seen'] += 1
                if not ok:
                    diagnostics['invalid_sample_resets'] += 1
                    first = previous = None
                    continue
                diagnostics['stationary_samples'] += 1
                if first is None or (previous is not None and stamp-previous > 75_000_000):
                    if previous is not None and stamp-previous > 75_000_000:
                        diagnostics['producer_gap_resets'] += 1
                    first = stamp
                previous = stamp
                diagnostics['maximum_stationary_span_ns'] = max(
                    diagnostics['maximum_stationary_span_ns'], previous-first)
            if first is not None and previous-first >= 100_000_000:
                diagnostics['final_attempts'] += 1
                with node._adaptive_command_guard():
                    _, fault, generation = snapshot()
                    if fault:
                        reject(fault)
                    retained_clear()
                    measured_scene_context(node, reference)
                    with node._lock:
                        fault = hard_fault_locked()
                        if fault:
                            reject(fault)
                        admitted_now = int(node.get_clock().now().nanoseconds)
                        if admitted_now < last_clock:
                            reject('clock_reversed')
                        live = list(node._delivery_raw_samples)
                        newer = [sample for sample in live if sample['sequence'] > seen]
                        latest = live[-1] if live else None
                        # A newer odometry callback need not wait for the next
                        # joint message before invalidating the final admission.
                        latest_with_odom = (None if latest is None else dict(latest,
                            odom=dict(getattr(node, '_delivery_raw_odom', None) or {})))
                        chronology = [fixed['producer_stamp_ns']] + [
                            sample['producer_stamp_ns'] for sample in newer]
                        final_checks = dict(
                            fixed_fresh=valid(fixed, admitted_now)[0],
                            latest_fresh=latest is not None and valid(latest, admitted_now, True)[0],
                            latest_odom_fresh=latest_with_odom is not None and valid(latest_with_odom, admitted_now, True)[0],
                            newer_samples_valid=all(valid(sample, admitted_now, True)[0] for sample in newer),
                            newer_gaps_valid=all(0 <= b-a <= 75_000_000 for a,b in zip(chronology, chronology[1:])),
                            newer_sequence_contiguous=not newer or newer[0]['sequence'] == seen+1,
                            within_wall_deadline=time.monotonic() < deadline)
                        final_ok = all(final_checks.values())
                        if not final_ok:
                            diagnostics['final_raw_retries'] += 1
                            diagnostics['last_final_veto'] = 'raw:' + ','.join(
                                name for name, value in final_checks.items() if not value)
                            diagnostics['last_final_raw_reasons'] = {
                                name: ('missing' if sample is None else valid(sample, admitted_now, pending)[1])
                                for name, sample, pending in (('fixed', fixed, False),
                                    ('latest', latest, True), ('latest_odom', latest_with_odom, True))}
                            diagnostics['last_final_raw_reasons']['newer_first_failure'] = next(
                                (valid(sample, admitted_now, True)[1] for sample in newer
                                 if not valid(sample, admitted_now, True)[0]), None)
                            first = previous = None
                        else:
                            current_generation = getattr(node, '_contact_generation', 0)
                            selected_stock = stock_enabled(node)
                            if selected_stock:
                                # Ordinary contact deliveries may advance the counter.
                                # Re-prove current retention under this same sensor lock;
                                # never approve using the earlier mixed-generation reads.
                                fault = _stock_retention_locked(node, admitted_now,
                                    min(.15, node.grasp_contact_max_age))
                                if fault:
                                    diagnostics['last_final_veto'] = 'retention:' + fault
                                    reject(fault)
                            if generation != current_generation and not selected_stock:
                                diagnostics['generation_retries'] += 1
                                diagnostics['last_final_veto'] = 'contact_generation_changed'
                            else:
                                if generation != current_generation:
                                    diagnostics['generation_advances_revalidated'] += 1
                                diagnostics['last_final_veto'] = None
                                fields = dict(producer_stamp_ns=previous, stationary_start_ns=first,
                                    planned_master=master, measured_master=fixed['positions']['gripper_left_finger_joint'],
                                    measured_master_velocity=fixed['velocities']['gripper_left_finger_joint'],
                                    measured_joint_positions=[fixed['positions'][n] for n in IK_JOINTS],
                                    measured_joint_velocities=[fixed['velocities'][n] for n in IK_JOINTS],
                                    odom_producer_stamp_ns=fixed['odom']['stamp_ns'],
                                    base_linear_speed=fixed['odom']['linear_speed'],
                                    base_angular_speed=fixed['odom']['angular_speed'],
                                    latest_negative_joint_stamp_ns=latest['producer_stamp_ns'],
                                    latest_negative_odom_stamp_ns=latest_with_odom['odom']['stamp_ns'],
                                    admitted_ros_ns=admitted_now,
                                    contact_epoch=epoch, contact_generation=current_generation,
                                    earlier_contact_generation=generation,
                                    current_stock_retention_revalidated=selected_stock)
                if fields is not None:
                    report(True, **fields)
                    return fields
            time.sleep(.002)
        reject('measurement_timeout:' + reason)
    except Exception as exc:
        report(False, str(exc))
        # The caller is outside recovery wrappers: any exception prevents open.
        raise
    finally:
        if owned:
            with node._lock:
                node._delivery_measurement_active = False
