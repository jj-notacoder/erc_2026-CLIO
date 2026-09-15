"""Choose a nearby measured PLACE height without changing motion tolerances.

The chosen value becomes the fixed torso coordinate of the entire plan and
the normal torso action. This does not skip a controller action, certify a
sweep with rounded joints, or change the configured nominal height.
"""
import math

_JOINT = 'torso_lift_joint'
_POSITION_WINDOW = 1e-6  # metres; maximum alternative-height selection/drift
_VELOCITY_WINDOW = 1e-6  # metres/second


def _measurement(node):
    """Read one atomic, fresh, finite, nearly stationary torso observation."""
    try:
        with node._lock:
            position = float(node.joints[_JOINT])
            velocity = float(node._joint_velocities[_JOINT])
            stamp = int(node._joint_stamps_ns[_JOINT])
        now = int(node.get_clock().now().nanoseconds)
        lower, upper = float(node.chain.lower[0]), float(node.chain.upper[0])
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(x) for x in (position, velocity, lower, upper))
            or not lower <= position <= upper
            or abs(velocity) > _VELOCITY_WINDOW
            or stamp <= 0 or not -50_000_000 <= now-stamp <= 350_000_000):
        return None
    return position, lower, upper


def choose_measured_place_height(node, carried_start, nominal_height):
    """Return the measured plan-start height, or preserve the nominal route."""
    sample = _measurement(node)
    if sample is None:
        return None
    try:
        candidate = float(carried_start[0])
        nominal = float(nominal_height)
    except (IndexError, TypeError, ValueError, OverflowError):
        return None
    position, lower, upper = sample
    if (not math.isfinite(candidate) or not math.isfinite(nominal)
            or not lower <= candidate <= upper
            or not lower <= nominal <= upper
            or abs(candidate-nominal) > _POSITION_WINDOW
            or abs(position-candidate) > _POSITION_WINDOW):
        return None
    return candidate


def require_planned_place_height(node, planned_height):
    """Reject a stale/moving/drifted start before the ordinary torso action."""
    sample = _measurement(node)
    try:
        target = float(planned_height)
    except (TypeError, ValueError, OverflowError):
        target = math.nan
    if (sample is None or not math.isfinite(target)
            or not sample[1] <= target <= sample[2]
            or abs(sample[0]-target) > _POSITION_WINDOW):
        raise RuntimeError('placement_measured_torso_changed')
