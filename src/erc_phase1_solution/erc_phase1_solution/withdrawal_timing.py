"""Optional four-leg withdrawal timing; no geometry, commands or sensor reads.

Range admission is a software bound, not acceleration, tracking or retention
proof. Each shortened goal still requires fresh locked joint-velocity admission.
"""
import math


def checked_withdrawal_speed_scale(value):
    if isinstance(value, bool):
        raise ValueError('withdrawal_speed_scale must be a finite number in [1.0, 3.0]')
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('withdrawal_speed_scale must be a finite number in [1.0, 3.0]') from error
    if not math.isfinite(scale) or not 1.0 <= scale <= 3.0:
        raise ValueError('withdrawal_speed_scale must be a finite number in [1.0, 3.0]')
    return scale


def withdrawal_leg_speed_scales(legs, *, command, speed_scale, arm_speed_scale,
                                leg_offset, initial_pressure_gate):
    """Admit only the exact initial-lift/four-withdrawal prefix, never recovery.

The caller supplies the existing first PICK segment after unchanged geometry
and pressure preflight. Do not copy/edit its joint arrays, legs or durations.
Additional carry phases retain scale1. Default callers need not enter here.
"""
    scale = checked_withdrawal_speed_scale(speed_scale)
    if (command != 'pick' or arm_speed_scale != 1.0 or leg_offset != 0
            or initial_pressure_gate is None):
        raise ValueError('withdrawal scaling requires the unstacked initial lift-first PICK segment')
    if not 5 <= len(legs) <= 256:
        raise ValueError('withdrawal scaling requires exactly four withdrawal legs after the lift')
    expected = ((1.0, 'initial_shelf_lift'), *((5.8, 'extraction'),) * 4)
    for leg, (duration, phase) in zip(legs[:5], expected):
        if len(leg) != 3 or leg[1] != duration or leg[2] != phase:
            raise ValueError('withdrawal scaling requires the unchanged 1s lift and four 5.8s extraction legs')
    if any(len(leg) != 3 or leg[2] in ('initial_shelf_lift', 'extraction') for leg in legs[5:]):
        raise ValueError('extra extraction/lift legs are not admitted for withdrawal scaling')
    return (1.0, scale, scale, scale, scale, *((1.0,) * (len(legs)-5)))
