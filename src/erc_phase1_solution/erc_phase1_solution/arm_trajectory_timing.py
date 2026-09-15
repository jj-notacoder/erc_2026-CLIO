"""Optional ordinary arm timing, independent of geometry and safety waits."""
import math


def checked_arm_speed_scale(value):
    scale = float(value)
    if not math.isfinite(scale) or not 1.0 <= scale <= 3.0:
        raise ValueError('placement_transport_speed_scale must be finite and within [1.0, 3.0]')
    return scale


def scaled_arm_seconds(seconds, scale, *, minimum=0.0):
    scale = checked_arm_speed_scale(scale)
    seconds, minimum = float(seconds), float(minimum)
    if (not math.isfinite(seconds) or not math.isfinite(minimum)
            or seconds <= 0.0 or not 0.0 <= minimum <= seconds):
        raise ValueError('arm timing requires a positive duration and a valid unchanged floor')
    return max(minimum, seconds / scale)
