"""Bounded observation of existing placement commands, using raw ROS epochs.

No actuator publisher or action client. The node owns the sensor lock.
"""
from collections import deque
import math
import time

from .release_evidence import measured_pose_is_open
from .motion_profiles import IK_JOINTS


def record_raw_odom(node, message):
    if (not getattr(node, '_delivery_measurement_active', False)
            and getattr(node, '_release_pose_owner', None) is None):
        return
    twist = message.twist.twist
    value = dict(stamp_ns=int(message.header.stamp.sec)*1_000_000_000
                 + int(message.header.stamp.nanosec),
                 linear_speed=math.hypot(twist.linear.x, twist.linear.y),
                 angular_speed=abs(float(twist.angular.z)))
    with node._lock:
        node._delivery_raw_odom = value


def record_raw_joints(node, message):
    """Called before ordinary feedback normalization; malformed replaces valid."""
    if (not getattr(node, '_delivery_measurement_active', False)
            and getattr(node, '_release_pose_owner', None) is None):
        return
    names = tuple(message.name)
    value = dict(producer_stamp_ns=int(message.header.stamp.sec)*1_000_000_000
                 + int(message.header.stamp.nanosec), positions={}, velocities={})
    if len(set(names)) == len(names):
        value['positions'] = {name: float(message.position[i])
                              if i < len(message.position) else math.nan
                              for i, name in enumerate(names)}
        value['velocities'] = {name: float(message.velocity[i])
                               if i < len(message.velocity) else math.nan
                               for i, name in enumerate(names)}
    with node._lock:
        if (not getattr(node, '_delivery_measurement_active', False)
            and getattr(node, '_release_pose_owner', None) is None):
            return
        node._delivery_sample_sequence += 1
        value['sequence'] = node._delivery_sample_sequence
        value['odom'] = dict(getattr(node, '_delivery_raw_odom', None) or {})
        node._delivery_raw_samples.append(value)
        if getattr(node, '_release_pose_owner', None) is not None:
            from .release_pose_finish import observe_joint_locked
            observe_joint_locked(node, value)


def observe_measured_open_pose(node, identity, goal, event):
    """100 ms of distinct raw feedback after the existing command completes.

    Bounds: 5 wall s overall, 0.5 wall s fixed-frame catch-up, 100 ms maximum
    future lead, 150 ms freshness, 75 ms maximum observed feedback gap.
    Raw callbacks are retained in a bounded ring so a between-poll outlier
    resets the span. This is actuator/checked endpoint evidence, not physical
    book detachment, full-hand clearance, or book/bin containment.
    """
    deadline = time.monotonic()+5.
    last_clock = int(node.get_clock().now().nanoseconds)
    entered = last_clock
    first = previous = None
    seen = 0
    reason = 'measurement_timeout'
    with node._lock:
        node._delivery_raw_samples = deque(maxlen=128)
        node._delivery_raw_odom = None
        node._delivery_sample_sequence = 0
        node._delivery_measurement_active = True

    def current_snapshot():
        with node._lock:
            return (list(node._delivery_raw_samples),
                    bool(node._goal_handles or
                         getattr(node, '_pending_retained_acceptances', ())))

    def valid(sample, now, pending=False):
        return measured_pose_is_open(sample, sample.get('odom'), goal, now,
            IK_JOINTS, pending_negative_check=pending, open_position=node.gripper_open)

    try:
        while time.monotonic() < deadline and not node._cancel.is_set():
            batch, active = current_snapshot()
            now = int(node.get_clock().now().nanoseconds)
            if now < last_clock:
                reason = 'clock_reversed'; break
            last_clock = now
            batch = [sample for sample in batch if sample['sequence'] > seen]
            if not batch:
                time.sleep(.002); continue
            if batch[0]['sequence'] != seen+1:
                first = previous = None  # dropped raw data cannot certify dwell
            fixed = batch[-1]
            future = max(fixed['producer_stamp_ns'], fixed['odom'].get('stamp_ns', 0))
            catchup = min(deadline, time.monotonic()+.5)
            disturbed = active
            while now < future <= now+100_000_000 and time.monotonic() < catchup:
                if node._cancel.is_set(): break
                live, live_active = current_snapshot()
                # New data can invalidate the pending span, never prove it.
                disturbed |= live_active or any(not valid(sample, now, True)[0]
                    for sample in live if sample['sequence'] > seen)
                time.sleep(.002)
                updated = int(node.get_clock().now().nanoseconds)
                if updated < now:
                    reason = 'clock_reversed'; break
                now = updated
            if reason == 'clock_reversed': break
            last_clock = now
            if disturbed: first = previous = None
            for sample in batch:
                stamp = sample['producer_stamp_ns']
                ok, reason = valid(sample, now)
                if stamp <= entered or active: ok = False
                if previous is not None and stamp < previous:
                    reason = 'joint_stamp_reversed'; break
                seen = sample['sequence']
                if not ok:
                    first = previous = None
                    continue
                if first is None or (previous is not None and stamp-previous > 75_000_000):
                    first = stamp
                previous = stamp
            if reason == 'joint_stamp_reversed': break
            if first is not None and previous-first >= 100_000_000:
                with node._lock:
                    admitted_now = int(node.get_clock().now().nanoseconds)
                    live = list(node._delivery_raw_samples)
                    newer = [sample for sample in live if sample['sequence'] > seen]
                    latest = live[-1] if live else None
                    chronology = [fixed['producer_stamp_ns']] + [
                        sample['producer_stamp_ns'] for sample in newer]
                    final_ok = (latest is not None and valid(fixed, admitted_now)[0]
                        and valid(latest, admitted_now, True)[0]
                        and all(valid(sample, admitted_now, True)[0] for sample in newer)
                        and all(0 <= b-a <= 75_000_000
                                for a,b in zip(chronology,chronology[1:]))
                        and (not newer or newer[0]['sequence'] == seen+1)
                        and not node._goal_handles
                        and not getattr(node, '_pending_retained_acceptances', ())
                        and not node._cancel.is_set() and admitted_now >= last_clock
                        and time.monotonic() < deadline)
                    if not final_ok:
                        first = previous = None
                        continue
                    fields = dict(identity, command='place', verified=True,
                        producer_stamp_ns=previous, stationary_start_ns=first,
                        master_position=fixed['positions']['gripper_left_finger_joint'],
                        master_velocity=fixed['velocities']['gripper_left_finger_joint'],
                        measured_joint_positions=[fixed['positions'][n] for n in IK_JOINTS],
                        basis='raw_actuated_feedback_at_checked_endpoint',
                        hand_clear_verified=False,physical_inside_verified=None)
                    if getattr(node, '_release_pose_owner', None) is not None:
                        fields['raw_sample_sequence'] = fixed['sequence']
                node._publish_status(event, **fields)
                return fields
            time.sleep(.002)
        if node._cancel.is_set(): reason = 'cancelled'
        fields = dict(identity, command='place', verified=False, reason=reason,
                      producer_stamp_ns=None, hand_clear_verified=False,
                      physical_inside_verified=None)
        node._publish_status(event, **fields)
        return fields
    finally:
        with node._lock:
            node._delivery_measurement_active = False
