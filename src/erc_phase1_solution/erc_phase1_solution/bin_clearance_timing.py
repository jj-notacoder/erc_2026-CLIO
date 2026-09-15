"""Optional normal-PLACE clearance timing; geometry and watchdog stay original."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .motion_profiles import IK_JOINTS
from .place_contact_guard import PlaceContactGuard
from .release_evidence import AttemptIdentity
from .arm_velocity_admission import ArmVelocityAdmissionRejected


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('bin_clearance_timing_enabled must be boolean')
    return value


@dataclass(frozen=True)
class ClearanceTiming:
    identity: AttemptIdentity
    guard: PlaceContactGuard
    scene_reference: dict
    index: int
    target: tuple


def _route(legs):
    phases = [leg[2] for leg in legs]
    if phases.count('bin_clearance') != 1:
        raise RuntimeError('clearance timing requires one original connector')
    index = phases.index('bin_clearance')
    if (any(p != 'bin_transition' for p in phases[:index])
            or any(p != 'bin_approach' for p in phases[index + 1:])):
        raise RuntimeError('clearance timing requires the normal PLACE route')
    expected = {'bin_transition': .8, 'bin_clearance': 2.8, 'bin_approach': .65}
    for q, seconds, phase in legs:
        if (type(seconds) is bool or float(seconds) != expected[phase]
                or len(q) != 8 or not all(math.isfinite(float(v)) for v in q)):
            raise RuntimeError('clearance timing original route changed')
    return index, tuple(float(v) for v in legs[index][0])


def _scope_locked(node, timing):
    if (node._cancel.is_set() or getattr(node, '_goal_handles', ())
            or getattr(node, '_pending_retained_acceptances', ())):
        raise RuntimeError('clearance timing requires stopped owned actions')
    guard = getattr(node, '_place_contact_guard', None)
    if (guard is not timing.guard or not isinstance(guard, PlaceContactGuard)
            or not timing.identity.matches(guard.correlation) or guard.fault is not None
            or type(guard.started_ns) is not int or guard.started_ns <= 0
            or getattr(node, '_active_place_scene_reference', None) is not timing.scene_reference
            or getattr(node, '_target_book_model', None) != timing.identity.target_model
            or getattr(node, '_held_book_corners', None) is None
            or getattr(node, '_gripper_open_confirmed', False) is True):
        raise RuntimeError('clearance timing correlated loaded context changed')


def _scope(node, timing):
    with node._lock:
        _scope_locked(node, timing)


def normal_options(node, correlation, direct_empty_home, legs):
    """Called only at the normal accepted scene-plan PLACE call site."""
    if not checked_enabled(getattr(node, 'bin_clearance_timing_enabled', False)):
        return {}
    if (not getattr(node, 'delivery_evidence_enabled', False)
            or not getattr(node, 'table_scene_required', False)
            or not getattr(node, 'bin_scene_required', False)
            or direct_empty_home is None
            or getattr(node, '_active_place_scene_reference', None) is None):
        raise RuntimeError('clearance timing requires a measured registered PLACE')
    identity = AttemptIdentity(**correlation)
    index, target = _route(legs)
    with node._lock:
        timing = ClearanceTiming(identity, getattr(node, '_place_contact_guard', None),
            node._active_place_scene_reference, index, target)
        _scope_locked(node, timing)
    return {'clearance_timing': timing}


def validate_execution(node, timing, legs, command, *, leg_offset,
                       initial_pressure_gate, withdrawal_speed_scale,
                       fresh_retention_phases):
    if (type(timing) is not ClearanceTiming
            or not checked_enabled(getattr(node, 'bin_clearance_timing_enabled', False))
            or command != 'place' or leg_offset != 0
            or initial_pressure_gate is not None or withdrawal_speed_scale != 1.0
            or isinstance(withdrawal_speed_scale, bool) or fresh_retention_phases):
        raise RuntimeError('clearance timing is restricted to normal PLACE')
    if _route(legs) != (timing.index, timing.target):
        raise RuntimeError('clearance timing connector changed')
    _scope(node, timing)


def timing_input(node, timing, index, solution, duration, phase):
    if index != timing.index:
        return duration
    if (phase != 'bin_clearance' or duration != 2.8
            or tuple(float(v) for v in solution) != timing.target):
        raise RuntimeError('clearance timing connector changed before send')
    _scope(node, timing)
    reason = node._payload_hazard_reason(max_age=min(.15, float(node.grasp_contact_max_age)))
    if reason is not None:
        raise RuntimeError('clearance timing payload hazard: ' + str(reason))
    _scope(node, timing)
    node._publish_status('bin_clearance_timing', command='place', phase=phase, leg=index,
        nominal_watchdog_seconds=2.8, timing_input_seconds=.8,
        expected=list(timing.target), **timing.guard.correlation)
    return .8


def require_publication_locked(node, timing, goal, command):
    """Caller holds command then sensor lock through immediate publication."""
    try:
        if (type(timing) is not ClearanceTiming or command != 'place'
                or not checked_enabled(getattr(node, 'bin_clearance_timing_enabled', False))
                or len(goal.trajectory.points) != 1
                or tuple(goal.trajectory.points[0].positions) != timing.target[1:]):
            raise RuntimeError('clearance timing publication target changed')
        _scope_locked(node, timing)
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ArmVelocityAdmissionRejected('clearance timing publication rejected: ' + str(error)) from error


def require_endpoint(node, timing, index):
    """Existing retained convergence wait, then fresh measured acceptance.

    This is an endpoint check, not a velocity/stationarity certificate. The
    original wait's 2 ROS seconds and timeout+8 wall allowance remain intact.
    A miss aborts before any route index advances or recovery/open is selected.
    """
    _scope(node, timing)
    measured = node._wait_for_retained_endpoint(timing.target, command='place',
        phase='bin_clearance', leg=index)
    if measured is None:
        raise RuntimeError('bin_clearance_endpoint_unverified')
    with node._adaptive_command_guard():
        reason = node._payload_hazard_reason(max_age=min(.15, float(node.grasp_contact_max_age)))
        if reason is not None:
            raise RuntimeError('bin_clearance_endpoint_payload_hazard: ' + str(reason))
        with node._lock:
            _scope_locked(node, timing)
            now = node.get_clock().now().nanoseconds
            stamps = getattr(node, '_joint_stamps_ns', {})
            positions = [node.joints.get(n, math.nan) for n in IK_JOINTS]
            producers = [stamps.get(n) for n in IK_JOINTS]
            if (type(now) is not int or now <= 0
                    or any(type(s) is not int or s <= 0 or not -50_000_000 <= now-s <= 150_000_000
                           for s in producers)
                    or not all(math.isfinite(v) for v in positions)
                    or abs(positions[0]-timing.target[0]) > .001
                    or any(abs(v-t) > .005 for v,t in zip(positions[1:],timing.target[1:]))):
                raise RuntimeError('bin_clearance_endpoint_fresh_measurement_unverified')
    node._publish_status('bin_clearance_endpoint_verified', command='place',
        phase='bin_clearance', leg=index, expected=list(timing.target), measured=positions,
        producer_stamps_ns=producers, evaluated_ros_ns=now,
        torso_tolerance=.001, arm_tolerance=.005, stationary_verified=False,
        **timing.guard.correlation)
