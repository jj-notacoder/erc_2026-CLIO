"""Optional reuse of a known completed torso target; never a new hold command.

The owner keeps its planned nominal torso coordinate and all scene/probe checks.
The generation record is command provenance, not proof of controller ownership
against an external publisher. All state below is protected by node._lock.
"""
from dataclasses import dataclass, field
from functools import wraps
import math
import threading

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


def _measurement_locked(node, target, completed):
    try:
        position = float(node.joints[JOINT])
        velocity = float(node._joint_velocities[JOINT])
        stamp = node._joint_stamps_ns[JOINT]
        now = int(node.get_clock().now().nanoseconds)
        lower, upper = float(node.chain.lower[0]), float(node.chain.upper[0])
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if (type(stamp) is not int or stamp <= 0 or stamp <= completed[3]
            or not all(math.isfinite(x) for x in (position, velocity, lower, upper, target))
            or not lower <= target <= upper or not lower <= position <= upper
            or abs(position-target) > 1e-6 or abs(velocity) > 1e-6
            or now < completed[3] or not -50_000_000 <= now-stamp <= 150_000_000):
        return None
    return dict(position=position, velocity=velocity, producer_stamp_ns=stamp,
                evaluated_ros_ns=now)


def try_completed_torso_hold(node, target, *, require_contact, check_scene):
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
            state = node._torso_hold_state
            generation, cancelled = _cancel(node)
            completed = state.completed
            if cancelled:
                state.completed = None
                return False
            if (completed is None or completed[:2] != (state.generation, generation)
                    or completed[2] != target or state.pending or state.uncertain
                    or getattr(node, '_goal_handles', ())
                    or getattr(node, '_pending_retained_acceptances', ())):
                return None
            first = _measurement_locked(node, target, completed)
            if first is None:
                return None
            held = getattr(node, '_held_book_corners', None)
            attached = None if held is None else np.asarray(held, dtype=float).copy()
            if (attached is None or attached.shape != (8, 3)
                    or not np.all(np.isfinite(attached))):
                return None
            epoch = int(getattr(node, '_contact_epoch', 0))
            delivery = int(getattr(node, '_contact_generation', 0))
            history = getattr(node, '_gripper_feedback_samples', ())
            feedback = history[-1] if history else None
        require_contact()
        check_scene(reference)
        reason = node._payload_hazard_reason(max_age=.15)
        if reason is not None:
            node._publish_status('payload_hazard', reason=reason)
            return False
        require_contact()
        with node._lock:
            generation, cancelled = _cancel(node)
            if cancelled:
                state.completed = None
                return False
            for name in ('_payload_hazard_latched', '_held_grip_sensor_fault',
                         '_raw_contacts_first_failure'):
                if getattr(node, name, None) is not None:
                    return False
            if getattr(node, '_target_robot_contact_latched', False):
                return False
            if (state is not node._torso_hold_state or state.completed is not completed
                    or completed[:2] != (state.generation, generation)
                    or state.pending or state.uncertain
                    or getattr(node, '_goal_handles', ())
                    or getattr(node, '_pending_retained_acceptances', ())):
                return None
            current = getattr(node, '_held_book_corners', None)
            if (current is not held or not np.array_equal(current, attached)
                    or int(getattr(node, '_contact_epoch', 0)) != epoch
                    or getattr(node, '_active_place_scene_reference', None) is not reference):
                return False
            history = getattr(node, '_gripper_feedback_samples', ())
            if (int(getattr(node, '_contact_generation', 0)) != delivery
                    or (history[-1] if history else None) is not feedback):
                return None
            latest = _measurement_locked(node, target, completed)
            if (latest is None or latest['producer_stamp_ns'] < first['producer_stamp_ns']
                    or latest['evaluated_ros_ns'] < first['evaluated_ros_ns']):
                return None
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
        return not node._cancel.is_set()
