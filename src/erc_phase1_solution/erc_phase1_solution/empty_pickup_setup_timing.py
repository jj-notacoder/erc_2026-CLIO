"""Opt-in retiming for already checked retracted setup, never book approach.

Geometry, command positions and the measured endpoint waits stay in the caller.
This helper adds no motion primitive, solver or geometry evaluator.
"""
import math

from .empty_pickup_collision import EmptyPickupCollision, _fixed
from .motion_profiles import ARM_JOINTS, IK_JOINTS


def checked_empty_pickup_setup_retiming_enabled(value):
    if type(value) is not bool:
        raise ValueError('empty_pickup_setup_retiming_enabled must be bool')
    return value


def move_empty_pickup_setup(node, guard, expected, solution, duration, *, phase):
    # Default and alternate/custom preparation routes retain the original call.
    if (not getattr(node, 'empty_pickup_setup_retiming_enabled', False)
            or type(guard) is not EmptyPickupCollision):
        return node._move_arm_solution(solution, duration)
    if guard.node is not node:
        raise RuntimeError('empty setup geometry belongs to another owner')
    durations = {'empty_setup_transition': 2.2, 'empty_setup_clearance': 2.8}
    if (phase not in durations or type(duration) not in (int, float)
            or not math.isfinite(duration) or duration != durations[phase]):
        raise ValueError('retiming only supports original empty setup durations')
    expected = _fixed(expected, len(IK_JOINTS))
    target = _fixed(solution, len(IK_JOINTS))
    if target[0] != expected[0]:
        raise RuntimeError('empty setup retiming cannot move the torso')

    def pre_send():
        if getattr(node, '_held_book_corners', None) is not None:
            raise RuntimeError('empty setup retiming requires no held payload')
        guard.require_fresh(expected, guard.open_aperture)

    pre_send()
    return node._follow(node.arm_client, ARM_JOINTS, target[1:], duration,
        trajectory_duration=duration, pre_send_check=pre_send,
        empty_pickup_setup=True)
