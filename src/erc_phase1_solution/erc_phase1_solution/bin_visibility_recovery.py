"""Bounded outward visibility recovery; remembered points never authorize place."""

from dataclasses import dataclass
import math
from typing import Tuple


RECOVERY_BACKOFF_M = 0.12
RECOVERY_TIMEOUT_NS = 35_000_000_000
REFERENCE_MAX_AGE_NS = 90_000_000_000
POSE_MAX_AGE_NS = 500_000_000
BIN_ASSOCIATION_DISTANCE_M = 0.18


@dataclass(frozen=True)
class BinApproachReference:
    """One verified bin surface and the goal derived from it, all in odom."""

    point: Tuple[float, float, float]
    stamp_ns: int
    goal: Tuple[float, float, float]


def _finite_vector(values, length: int):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid_bin_recovery_geometry') from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError('invalid_bin_recovery_geometry')
    return result


def _reference_is_current(reference: BinApproachReference, now_ns: int) -> None:
    if reference is None or reference.stamp_ns <= 0 or not (
        0 <= now_ns - reference.stamp_ns <= REFERENCE_MAX_AGE_NS
    ):
        raise ValueError('bin_recovery_reference_expired')
    _finite_vector(reference.point, 3)
    _finite_vector(reference.goal, 3)


def visibility_retreat_goal(reference, robot_pose, pose_stamp_ns, now_ns, attempts):
    """Authorize one 12 cm straight retreat from the reached approach goal.

    This only checks goal geometry. The existing navigator must still enforce
    scan freshness, obstacles, carried speed limits and stopping. It is not a
    collision certificate or an authorization to move toward a remembered bin.
    """
    if attempts != 0:
        raise ValueError('bin_visibility_retry_exhausted')
    _reference_is_current(reference, now_ns)
    if pose_stamp_ns <= 0 or not 0 <= now_ns - pose_stamp_ns <= POSE_MAX_AGE_NS:
        raise ValueError('bin_recovery_robot_pose_stale')
    x, y, yaw = _finite_vector(robot_pose, 3)
    gx, gy, gyaw = reference.goal
    if math.hypot(x-gx, y-gy) > 0.10:
        raise ValueError('bin_recovery_robot_displaced')
    if abs(math.atan2(math.sin(yaw-gyaw), math.cos(yaw-gyaw))) > 0.12:
        raise ValueError('bin_recovery_heading_changed')
    dx, dy = reference.point[0]-x, reference.point[1]-y
    distance = math.hypot(dx, dy)
    if not 0.45 <= distance <= 1.20:
        raise ValueError('bin_recovery_distance_invalid')
    bearing = math.atan2(dy, dx)-yaw
    if abs(math.atan2(math.sin(bearing), math.cos(bearing))) > 0.35:
        raise ValueError('bin_recovery_not_facing_bin')
    return (x-RECOVERY_BACKOFF_M*math.cos(yaw),
            y-RECOVERY_BACKOFF_M*math.sin(yaw), yaw)


def verify_reacquired_bin(reference, point, stamp_ns, now_ns, after_ns):
    """Check fresh observation association, after normal geometry verification.

    A caller must separately require positive same-frame geometry status from
    the three-frame detector. This function cannot validate a raw red point.
    """
    _reference_is_current(reference, now_ns)
    if stamp_ns <= after_ns or not 0 <= now_ns-stamp_ns <= POSE_MAX_AGE_NS:
        raise ValueError('bin_reacquisition_not_new_and_fresh')
    observed = _finite_vector(point, 3)
    if math.dist(observed, reference.point) > BIN_ASSOCIATION_DISTANCE_M:
        raise ValueError('bin_reacquisition_location_changed')
