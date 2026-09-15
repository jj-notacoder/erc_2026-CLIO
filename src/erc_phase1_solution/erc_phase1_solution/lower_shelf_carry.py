"""Supported compact posture proposals for lower shelf grasps.

No admissibility bypass: callers must run their ordinary carried route,
robot/payload/shelf sweep, signed gravity support, detailed gripper/tool route,
and navigation-radius validators before using these proposed goals.
"""
import math
import numpy as np


def supported_progress_goals(node, start, target, powers, count):
    first = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    powers = np.asarray(powers, dtype=float)
    if (first.shape != (8,) or target.shape != (8,) or powers.shape != (6,)
            or not np.all(np.isfinite([first, target]))
            or not np.all(np.isfinite(powers)) or np.any(powers <= 0.)
            or type(count) is not int or count < 2 or target[0] != first[0]):
        raise ValueError('invalid supported progress proposal')
    if (np.any(target[1:] < node.chain.lower[1:] + .01)
            or np.any(target[1:] > node.chain.upper[1:] - .01)):
        raise ValueError('compact target exceeds joint margin')
    # Reparameterise by the fastest proximal joint so fractional powers never
    # make the first step disproportionately large near progress zero.
    powers = powers / np.min(powers)
    lower = float(node.chain.lower[-1]) + .01
    upper = float(node.chain.upper[-1]) - .01
    previous_q7 = float(first[-1])
    result = []
    for fraction in np.linspace(1. / count, 1., count):
        q = first.copy()
        q[1:7] += (target[1:7] - first[1:7]) * fraction ** powers
        q[-1] = 0.
        rotation = node.chain.forward(q)
        a, b = float(rotation[2, 1]), float(rotation[2, 2])
        optimum = math.atan2(b, a)
        candidates = [lower, upper]
        for period in range(-2, 3):
            angle = optimum + 2 * math.pi * period
            if lower <= angle <= upper:
                candidates.append(angle)
        scored = [(a * math.cos(angle) + b * math.sin(angle), angle)
                  for angle in candidates]
        maximum = max(support for support, _ in scored)
        if maximum < node.carried_supported_jaw_vertical_component:
            raise RuntimeError('no signed-support compact wrist angle')
        choices = [angle for support, angle in scored if maximum-support <= 1e-9]
        q[-1] = min(choices, key=lambda angle: (abs(angle-previous_q7), angle))
        # Check the physical signed convention on the production FK as well.
        if node.chain.forward(q)[2, 1] < node.carried_supported_jaw_vertical_component:
            raise RuntimeError('signed-support convention mismatch')
        result.append(q)
        previous_q7 = float(q[-1])
    return result


POSITIVE_WRIST_CARRY = np.array([
    .1, 2.802816760676171, -2.1641582001441426, -1.9901919374363823,
    -2.212871456022635, .6763034420257978, 1.3500515755859643,
    .8920708110865356,
])


def middle_compact_proposal(node, start, *, count=20):
    first = np.asarray(start, dtype=float)
    target = POSITIVE_WRIST_CARRY.copy()
    target[0] = first[0]
    goals = supported_progress_goals(node, first, target, np.ones(6), count)
    # Finish at the selected checked wrist posture, not merely the support optimum.
    goals[-1][-1] = target[-1]
    return goals
