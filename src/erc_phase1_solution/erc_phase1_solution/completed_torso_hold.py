"""Optional reuse of a known completed torso target; never a new hold command.

The owner keeps its planned nominal torso coordinate and all scene/probe checks.
The generation record is command provenance, not proof of controller ownership
against an external publisher. All state below is protected by node._lock.
"""
from dataclasses import dataclass, field
from functools import wraps
import math
import threading
import time

import numpy as np

JOINT = 'torso_lift_joint'


class TorsoHoldCancellation(threading.Event):
    """A later clear cannot make proof from before cancellation eligible again."""
    def __init__(self):
        super().__init__()
        self._hold_lock = threading.Lock()
        self._hold_generation = 0

    def set(self):
        with self._hold_lock:
            self._hold_generation += 1
            super().set()

    def clear(self):
        with self._hold_lock:
            super().clear()

    def snapshot(self):
        with self._hold_lock:
            return self._hold_generation, self.is_set()


@dataclass(eq=False)
class FollowToken:
    state: object
    generation: int
    cancel_generation: int
    torso: bool
    target: object
    dispatched: bool = False


@dataclass
class TorsoHoldState:
    generation: int = 0
    completed: object = None
    pending: set = field(default_factory=set)
    uncertain: bool = False


def checked_torso_hold_enabled(value):
    if type(value) is not bool:
        raise ValueError('settled_place_torso_skip_enabled must be Boolean')
    return value


def checked_torso_hold_retry_enabled(value):
    if type(value) is not bool:
        raise ValueError('settled_place_torso_retry_enabled must be Boolean')
    return value


def _cancel(node):
    event = node._cancel
    # A generic Event cannot prove that cancellation was never set and cleared.
    return event.snapshot() if type(event) is TorsoHoldCancellation else (None, True)


def track_completed_torso_hold(function):
    """Central lifetime tracking for every _follow call while the option is on."""
    @wraps(function)
    def tracked(node, client, names, positions, duration, **kwargs):
        if not getattr(node, 'settled_place_torso_skip_enabled', False):
            return function(node, client, names, positions, duration, **kwargs)
        with node._adaptive_command_guard():
            with node._lock:
                state = node._torso_hold_state
                joint_names = tuple(names)
                torso = client is node.torso_client or JOINT in joint_names
                target = None
                if torso:
                    state.generation += 1
                    state.completed = None  # Before server waits or any failure.
                    if client is node.torso_client and joint_names == (JOINT,):
                        try:
                            if len(positions) == 1 and math.isfinite(float(positions[0])):
                                target = float(positions[0])
                        except (TypeError, ValueError, OverflowError):
                            pass
                cancel_generation, _ = _cancel(node)
                token = FollowToken(state, state.generation, cancel_generation, torso, target)
                state.pending.add(token)
        succeeded = False
        try:
            succeeded = function(node, client, names, positions, duration,
                                 _torso_hold_token=token, **kwargs) is True
            return succeeded
        finally:
            with node._adaptive_command_guard():
                with node._lock:
                    state.pending.discard(token)
                    generation, cancelled = _cancel(node)
                    if not succeeded:
                        state.completed = None
                        state.generation += 1
                        # An attempted send with no proven success may still be
                        # accepted later. Never erase this uncertainty for a skip.
                        state.uncertain |= token.dispatched
                    elif (token.torso and token.target is not None
                          and state is node._torso_hold_state
                          and token.generation == state.generation
                          and generation == token.cancel_generation and not cancelled
                          and not state.pending and not state.uncertain):
                        state.completed = (state.generation, generation, token.target,
                                           int(node.get_clock().now().nanoseconds))

    return tracked


def require_follow_token_locked(node, token, goal):
    """Immediately before a send, with command then sensor ownership already held."""
    if token is None:
        return
    state = node._torso_hold_state
    generation, cancelled = _cancel(node)
    if (token.state is not state or token not in state.pending
            or (token.torso and token.generation != state.generation)
            or generation != token.cancel_generation or cancelled):
        raise RuntimeError('torso_hold_command_generation_changed')
    if token.torso and token.target is not None:
        # Record the serialized command actually submitted, not an earlier
        # caller-owned positions sequence that could have changed during waits.
        trajectory = goal.trajectory
        token.target = None
        if (tuple(trajectory.joint_names) == (JOINT,) and len(trajectory.points) == 1
                and len(trajectory.points[0].positions) == 1):
            value = float(trajectory.points[0].positions[0])
            if math.isfinite(value):
                token.target = value
    token.dispatched = True


def _measurement_locked(node, target, completed, retry=None):
    try:
        position = float(node.joints[JOINT])
        velocity = float(node._joint_velocities[JOINT])
        stamp = node._joint_stamps_ns[JOINT]
        now = int(node.get_clock().now().nanoseconds)
        lower, upper = float(node.chain.lower[0]), float(node.chain.upper[0])
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        if retry is not None:
            retry.reason = 'measurement_unavailable'
        return None
    if retry is not None:
        if now < retry.last_ros:
            retry.reason = 'clock_regressed'
            return None
        retry.last_ros = now
        if type(stamp) is int:
            if retry.last_producer is not None and stamp < retry.last_producer:
                retry.reason = 'measurement_regressed'
                return None
            retry.last_producer = stamp
    if (type(stamp) is not int or stamp <= 0 or stamp <= completed[3]
            or not all(math.isfinite(x) for x in (position, velocity, lower, upper, target))
            or not lower <= target <= upper or not lower <= position <= upper
            or abs(position-target) > 1e-6 or abs(velocity) > 1e-6
            or now < completed[3] or not -50_000_000 <= now-stamp <= 150_000_000):
        if retry is not None:
            if now < completed[3]:
                retry.reason = 'clock_regressed'
            elif type(stamp) is not int or stamp <= 0:
                retry.reason = 'measurement_stamp_invalid'
            elif not all(math.isfinite(x) for x in (position, velocity, lower, upper, target)):
                retry.reason = 'measurement_nonfinite'
            elif not lower <= target <= upper or not lower <= position <= upper:
                retry.reason = 'measurement_out_of_limits'
            elif stamp <= completed[3]:
                retry.reason = 'measurement_not_after_completion'
            elif now-stamp < -50_000_000:
                retry.reason = 'measurement_from_future'
            elif now-stamp > 150_000_000:
                retry.reason = 'measurement_stale'
            elif abs(position-target) > 1e-6:
                retry.reason = 'torso_position_not_settled'
            else:
                retry.reason = 'torso_velocity_not_settled'
        return None
    return dict(position=position, velocity=velocity, producer_stamp_ns=stamp,
                evaluated_ros_ns=now)


def _try_completed_torso_hold_once(node, target, *, require_contact, check_scene, retry=None):
    """True admits no new command; None takes the unchanged normal action.

    Hard faults raise or return False. No helper called below the sensor lock can
    acquire that non-reentrant lock. Delivery changes conservatively fall back.
    """
    if not getattr(node, 'settled_place_torso_skip_enabled', False):
        return None
    if (not getattr(node, 'table_scene_required', False)
            or not getattr(node, 'bin_scene_required', False)):
        return None
    reference = getattr(node, '_active_place_scene_reference', None)
    if reference is None:
        return None
    target = float(target)
    fields = None
    with node._adaptive_command_guard():
        with node._lock:
            if retry is not None and retry.anchor is not None and not retry.same_context_locked(node):
                return False
            state = node._torso_hold_state
            generation, cancelled = _cancel(node)
            completed = state.completed
            if cancelled:
                state.completed = None
                if retry is not None:
                    retry.reason = 'cancelled'
                return False
            if (completed is None or completed[:2] != (state.generation, generation)
                    or completed[2] != target or state.pending or state.uncertain
                    or getattr(node, '_goal_handles', ())
                    or getattr(node, '_pending_retained_acceptances', ())):
                if retry is not None:
                    retry.reason = 'completed_action_unavailable'
                return None
            if retry is not None and not retry.capture_locked(node, state, completed, reference):
                return False if retry.anchor is not None else None
            first = _measurement_locked(node, target, completed, retry)
            if first is None:
                return None
            held = getattr(node, '_held_book_corners', None)
            attached = None if held is None else np.asarray(held, dtype=float).copy()
            if (attached is None or attached.shape != (8, 3)
                    or not np.all(np.isfinite(attached))):
                if retry is not None:
                    retry.reason = 'held_geometry_invalid'
                return None
            epoch = int(getattr(node, '_contact_epoch', 0))
            delivery = int(getattr(node, '_contact_generation', 0))
            history = getattr(node, '_gripper_feedback_samples', ())
            feedback = history[-1] if history else None
        require_contact()
        check_scene(reference)
        reason = node._payload_hazard_reason(max_age=.15)
        if reason is not None:
            if retry is not None:
                retry.reason = 'payload_hazard'
            node._publish_status('payload_hazard', reason=reason)
            return False
        require_contact()
        with node._lock:
            if retry is not None and not retry.same_context_locked(node):
                return False
            generation, cancelled = _cancel(node)
            if cancelled:
                state.completed = None
                if retry is not None:
                    retry.reason = 'cancelled'
                return False
            for name in ('_payload_hazard_latched', '_held_grip_sensor_fault',
                         '_raw_contacts_first_failure'):
                if getattr(node, name, None) is not None:
                    if retry is not None:
                        retry.reason = 'payload_fault'
                    return False
            if getattr(node, '_target_robot_contact_latched', False):
                if retry is not None:
                    retry.reason = 'target_robot_contact'
                return False
            if (state is not node._torso_hold_state or state.completed is not completed
                    or completed[:2] != (state.generation, generation)
                    or state.pending or state.uncertain
                    or getattr(node, '_goal_handles', ())
                    or getattr(node, '_pending_retained_acceptances', ())):
                if retry is not None:
                    retry.reason = 'completed_action_changed'
                return None
            current = getattr(node, '_held_book_corners', None)
            if (current is not held or not np.array_equal(current, attached)
                    or int(getattr(node, '_contact_epoch', 0)) != epoch
                    or getattr(node, '_active_place_scene_reference', None) is not reference):
                if retry is not None:
                    retry.reason = 'held_context_changed'
                return False
            history = getattr(node, '_gripper_feedback_samples', ())
            if (int(getattr(node, '_contact_generation', 0)) != delivery
                    or (history[-1] if history else None) is not feedback):
                if retry is not None:
                    retry.reason = 'delivery_changed'
                return None
            latest = _measurement_locked(node, target, completed, retry)
            if (latest is None or latest['producer_stamp_ns'] < first['producer_stamp_ns']
                    or latest['evaluated_ros_ns'] < first['evaluated_ros_ns']):
                if retry is not None and latest is not None:
                    retry.reason = 'measurement_regressed'
                return None
            if retry is not None:
                limit = retry.limit(node)
                if limit == 'clock_regressed' or (retry.retries and limit):
                    retry.reason = limit
                    return False if limit == 'clock_regressed' else None
            fields = dict(command='place', target=target,
                          completed_action_ros_ns=completed[3], command_generation=completed[0],
                          cancel_generation=completed[1], **latest,
                          reason='completed_exact_torso_target_measured_stationary',
                          controller_command_sent=False,
                          scope='existing torso target retained; nominal plan and both retention probes unchanged')
        # No sensor lock spans diagnostics; cancellation invalidates later proof.
        try:
            node._publish_status('torso_motion_skipped', **fields)
        except Exception:
            pass
        if retry is not None:
            retry.reason = 'admitted'
        return not node._cancel.is_set()


class _HoldRetry:
    """Attempt-local provenance; no node state or caller-owned sensor mutation."""
    def __init__(self, node):
        self.wall_start = time.monotonic()
        self.ros_start = int(node.get_clock().now().nanoseconds)
        self.last_wall = self.wall_start
        self.last_ros = self.ros_start
        self.last_producer = None
        self.anchor = None
        self.retries = 0
        self.reason = 'ineligible'
        self.rejection_counts = {}
        self.waited = False

    def capture_locked(self, node, state, completed, reference):
        if self.anchor is not None:
            return self.same_context_locked(node)
        held = getattr(node, '_held_book_corners', None)
        attached = None if held is None else np.asarray(held, dtype=float).copy()
        if attached is None or attached.shape != (8, 3) or not np.all(np.isfinite(attached)):
            self.reason = 'held_geometry_invalid'
            return False
        self.anchor = (state, completed, reference, held, attached,
                       int(getattr(node, '_contact_epoch', 0)))
        stamp = getattr(node, '_joint_stamps_ns', {}).get(JOINT)
        self.last_producer = stamp if type(stamp) is int else None
        return self.same_context_locked(node)

    def same_context_locked(self, node):
        state, completed, reference, held, attached, epoch = self.anchor
        generation, cancelled = _cancel(node)
        if cancelled or generation != completed[1]:
            self.reason = 'cancelled'
            return False
        if (state is not node._torso_hold_state or state.completed is not completed
                or completed[:2] != (state.generation, generation)
                or state.pending or state.uncertain or getattr(node, '_goal_handles', ())
                or getattr(node, '_pending_retained_acceptances', ())):
            self.reason = 'completed_action_changed'
            return False
        if (getattr(node, '_active_place_scene_reference', None) is not reference
                or getattr(node, '_held_book_corners', None) is not held
                or not np.array_equal(held, attached)
                or int(getattr(node, '_contact_epoch', 0)) != epoch):
            self.reason = 'held_context_changed'
            return False
        if any(getattr(node, name, None) is not None for name in (
                '_payload_hazard_latched', '_held_grip_sensor_fault', '_raw_contacts_first_failure')):
            self.reason = 'payload_fault'
            return False
        if getattr(node, '_target_robot_contact_latched', False):
            self.reason = 'target_robot_contact'
            return False
        return True

    def limit(self, node):
        wall = time.monotonic()
        ros = int(node.get_clock().now().nanoseconds)
        if not math.isfinite(wall) or wall < self.last_wall or ros < self.last_ros:
            return 'clock_regressed'
        self.last_wall, self.last_ros = wall, ros
        if wall-self.wall_start >= 1.0:
            return 'wall_limit'
        if ros-self.ros_start >= 200_000_000:
            return 'ros_limit'
        return None

    def fallback(self, node, require_contact, check_scene):
        """After added waiting, check live payload/scene before the normal goal.

        This does not admit a hold or require a stationary torso. The original
        sender still performs its entry, post-server and pre-send admissions.
        """
        if not self.waited and not self.retries:
            return None
        with node._adaptive_command_guard():
            with node._lock:
                if not self.same_context_locked(node):
                    return False
                reference = self.anchor[2]
                producer = node._joint_stamps_ns.get(JOINT)
                if (type(producer) is int and self.last_producer is not None
                        and producer < self.last_producer):
                    self.reason = 'measurement_regressed'
                    return False
            # Expiry caused this fallback; only monotonicity still applies.
            if self.limit(node) == 'clock_regressed':
                self.reason = 'clock_regressed'
                return False
            require_contact()
            check_scene(reference)
            reason = node._payload_hazard_reason(max_age=.15)
            if reason is not None:
                self.reason = 'fallback_payload_hazard'
                node._publish_status('payload_hazard', reason=reason)
                return False
            require_contact()
            with node._lock:
                if not self.same_context_locked(node):
                    return False
                current = node._joint_stamps_ns.get(JOINT)
                if (type(current) is int and type(producer) is int and current < producer):
                    self.reason = 'measurement_regressed'
                    return False
                if self.limit(node) == 'clock_regressed':
                    self.reason = 'clock_regressed'
                    return False
        return None


def try_completed_torso_hold(node, target, *, require_contact, check_scene):
    """Retry only a transient rejection of the exact original hold admission.

    Bounds apply to added cooperative work, not OS scheduling. No ownership
    lock spans a wait, and a timeout requires the original normal torso action.
    """
    if not getattr(node, 'settled_place_torso_retry_enabled', False):
        return _try_completed_torso_hold_once(node, target,
            require_contact=require_contact, check_scene=check_scene)
    target = float(target)
    retry = _HoldRetry(node)
    outcome = None
    try:
        while True:
            outcome = _try_completed_torso_hold_once(node, target,
                require_contact=require_contact, check_scene=check_scene, retry=retry)
            if outcome is None and retry.reason in ('measurement_regressed', 'clock_regressed'):
                outcome = False
            if outcome is not None or retry.anchor is None:
                return outcome
            retry.rejection_counts[retry.reason] = retry.rejection_counts.get(retry.reason, 0)+1
            if retry.reason not in ('measurement_not_after_completion', 'measurement_from_future',
                    'measurement_stale', 'torso_position_not_settled',
                    'torso_velocity_not_settled', 'delivery_changed'):
                outcome = retry.fallback(node, require_contact, check_scene)
                return outcome
            if retry.retries >= 10:
                retry.reason = 'retry_limit'
                outcome = retry.fallback(node, require_contact, check_scene)
                return outcome
            # Wait for new input, never reconsider the same failed snapshot.
            with node._adaptive_command_guard():
                with node._lock:
                    if not retry.same_context_locked(node):
                        outcome = False
                        return outcome
                    stamp = node._joint_stamps_ns.get(JOINT)
                    history = getattr(node, '_gripper_feedback_samples', ())
                    feedback = history[-1] if history else None
            while True:
                with node._adaptive_command_guard():
                    with node._lock:
                        if not retry.same_context_locked(node):
                            outcome = False
                            return outcome
                        current = node._joint_stamps_ns.get(JOINT)
                        history = getattr(node, '_gripper_feedback_samples', ())
                        latest = history[-1] if history else None
                        if type(current) is int and type(stamp) is int and current < stamp:
                            retry.reason = 'measurement_regressed'
                            outcome = False
                            return outcome
                        fresh = (type(current) is int and type(stamp) is int and current > stamp) or latest is not feedback
                limit = retry.limit(node)
                if limit:
                    retry.reason = limit
                    outcome = (False if limit == 'clock_regressed'
                               else retry.fallback(node, require_contact, check_scene))
                    return outcome
                if fresh:
                    retry.retries += 1
                    break
                retry.waited = True
                node._cancel.wait(min(.01, max(0., 1.0-(time.monotonic()-retry.wall_start))))
    except Exception:
        retry.reason = 'gate_exception'
        outcome = False
        raise
    finally:
        try:
            node._publish_status('torso_hold_retry_completed', command='place',
                retries=retry.retries, attempts=retry.retries+1, reason=retry.reason,
                rejection_counts=dict(retry.rejection_counts),
                outcome='admitted' if outcome is True else 'failed' if outcome is False else 'fallback_required',
                wall_seconds=time.monotonic()-retry.wall_start,
                ros_seconds=(int(node.get_clock().now().nanoseconds)-retry.ros_start)/1e9,
                controller_command_sent=False, scope='diagnostic only; fallback action, if needed, follows this record')
        except Exception:
            pass
