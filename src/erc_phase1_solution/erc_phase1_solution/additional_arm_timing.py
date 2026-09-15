"""Optional owned retiming after fresh locked admission of the original goal.

Exact rational arithmetic on serialized position doubles and integer times keeps
every segment at <= 80% of the official velocity. This does not prove tracking,
acceleration or retention. The caller must re-admit the new goal under the same
command->sensor locks before dispatch, and retain its original wall watchdog.
"""
import copy
from fractions import Fraction
import math
import struct

from .arm_velocity_admission import ArmVelocityAdmissionRejected, check_serialized_arm_goal


MINIMUM_SEGMENT_NS = 87_500_000
VELOCITY_FRACTION = Fraction(4, 5)


def checked_additional_arm_time_scale(value):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 1.0 <= value <= 2.0):
        raise ValueError('additional_arm_time_scale must be finite numeric within [1, 2]')
    return float(value)


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _bits(values):
    return struct.pack('>' + 'd' * len(values), *values)


def require_retimed_arm_headroom(record):
    """Apply the exact 80% ceiling to the second, actual dispatch admission."""
    try:
        if record.get('record_complete') is not True:
            raise ValueError('complete retimed admission required')
        previous, previous_ns = record['start_positions'], 0
        limits = [Fraction.from_float(v) for v in record['velocity_limits']]
        if len(previous) != 7 or len(limits) != 7 or any(v <= 0 for v in limits) or not 1 <= len(record['points']) <= 256:
            raise ValueError('malformed retimed admission')
        for point in record['points']:
            duration = point['time_from_start_ns'] - previous_ns
            if duration <= 0 or len(point['positions']) != 7 or len(limits) != 7:
                raise ValueError('malformed retimed admission')
            for first, last, limit in zip(previous, point['positions'], limits):
                if abs(Fraction.from_float(last) - Fraction.from_float(first)) * 1_000_000_000 > VELOCITY_FRACTION * limit * duration:
                    raise ValueError('retimed arm exceeds 80-percent official velocity headroom')
            previous, previous_ns = point['positions'], point['time_from_start_ns']
        return record
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ArmVelocityAdmissionRejected(str(error), record) from error


def retime_admitted_arm_goal(goal, admission, factor, *, nominal_duration, legs=None,
                            minimum_segment_ns=MINIMUM_SEGMENT_NS):
    """Return new goal/phase legs plus diagnostics; input goal is never changed.

    Scale1 preserves exact input identity and does no new admission/retiming.
    Eligible callers already performed require_arm_velocity_locked, and must
    call it again on the returned goal before publication. No locks are acquired.
    """
    try:
        factor = checked_additional_arm_time_scale(factor)
        if (type(minimum_segment_ns) is not int or minimum_segment_ns < MINIMUM_SEGMENT_NS
                or minimum_segment_ns > 2_147_483_647 * 1_000_000_000 + 999_999_999):
            raise ValueError('additional timing floor must be a monotonic integer nanosecond bound')
        if factor == 1.0 and minimum_segment_ns != MINIMUM_SEGMENT_NS:
            raise ValueError('explicit additional timing floor requires enabled retiming')
        if factor == 1.0:
            return goal, legs, None
        if (type(nominal_duration) not in (int, float) or not math.isfinite(nominal_duration)
                or nominal_duration <= 0):
            raise ValueError('invalid nominal watchdog duration')
        if type(admission) is not dict or admission.get('record_complete') is not True:
            raise ValueError('complete prior locked velocity admission required')
        original = check_serialized_arm_goal(goal, admission['start_positions'], admission['velocity_limits'])
        # A mutated goal or record between the first gate and this operation is
        # a refusal, not permission to use an unrelated measured start.
        for key, value in original.items():
            if value != admission.get(key):
                raise ValueError('goal differs from prior locked velocity admission')
        points = original['points']
        if legs is not None:
            if len(legs) != len(points):
                raise ValueError('phase count does not match serialized points')
            elapsed = 0.0
            for (solution, seconds, phase), point in zip(legs, points):
                if (len(solution) != 8 or not isinstance(phase, str)
                        or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
                        or not all(math.isfinite(float(v)) for v in solution)
                        or _bits([float(v) for v in solution[1:]]) != _bits(point['positions'])):
                    raise ValueError('phase geometry does not match serialized point')
                elapsed += float(seconds)
                if int(elapsed * 1_000_000_000) != point['time_from_start_ns']:
                    raise ValueError('phase timing does not match serialized goal')
        previous = original['start_positions']
        previous_ns = 0
        new_total = 0
        durations, details = [], []
        scale = Fraction.from_float(factor)
        limits = [Fraction.from_float(v) for v in original['velocity_limits']]
        for point in points:
            old_segment = point['time_from_start_ns'] - previous_ns
            required = max(abs(Fraction.from_float(last) - Fraction.from_float(first))
                           * 1_000_000_000 / (VELOCITY_FRACTION * limit)
                           for first, last, limit in zip(previous, point['positions'], limits))
            segment = max(_ceil(Fraction(old_segment) / scale), _ceil(required), minimum_segment_ns)
            new_total += segment
            if new_total > 2_147_483_647 * 1_000_000_000 + 999_999_999:
                raise ValueError('retimed serialized duration overflow')
            durations.append(segment)
            details.append(dict(old_segment_ns=old_segment, new_segment_ns=segment,
                                velocity_minimum_ns=_ceil(required)))
            previous, previous_ns = point['positions'], point['time_from_start_ns']
        nominal_ns = Fraction.from_float(float(nominal_duration)) * 1_000_000_000
        if new_total > nominal_ns:
            raise ValueError('80-percent retiming exceeds unchanged nominal duration allowance')
        owned = copy.deepcopy(goal)
        if (owned is goal or owned.trajectory is goal.trajectory
                or any(a is b for a, b in zip(owned.trajectory.points, goal.trajectory.points))):
            raise ValueError('retiming requires an independent owned goal')
        elapsed = 0
        for point, segment in zip(owned.trajectory.points, durations):
            elapsed += segment
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(elapsed, 1_000_000_000)
        # The original gate checks all fields that affect interpolation. Restore
        # only times in this comparison; every position byte/order remains exact.
        final = check_serialized_arm_goal(owned, original['start_positions'], original['velocity_limits'])
        require_retimed_arm_headroom(final)
        if (owned.trajectory.header != goal.trajectory.header
                or final['joint_names'] != original['joint_names']
                or len(final['points']) != len(points)
                or any(_bits(a['positions']) != _bits(b['positions']) for a,b in zip(final['points'],points))):
            raise ValueError('retiming changed trajectory geometry or header')
        new_legs = (legs if legs is None else
                    tuple((solution, ns / 1_000_000_000, phase)
                          for (solution, _, phase), ns in zip(legs, durations)))
        diagnostic = dict(factor=factor, maximum_velocity_fraction=.8,
            minimum_segment_ns=minimum_segment_ns, old_total_ns=previous_ns,
            new_total_ns=new_total, nominal_duration_seconds=float(nominal_duration),
            segments=details, unchanged_nominal_watchdog=True)
        return owned, new_legs, diagnostic
    except ArmVelocityAdmissionRejected:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        record = dict(admission) if type(admission) is dict else {}
        record['additional_arm_timing_rejected'] = str(error)
        raise ArmVelocityAdmissionRejected(str(error), record) from error
