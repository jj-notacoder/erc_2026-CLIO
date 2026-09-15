def coordinated_supported_goals(chain, start, goal, *, margin, minimum_support,
                                shoulder_progress_power=1., joint_progress_powers=None):
    """Twenty proximal states with nearby legal q7 maximizing SIGNED jaw-up.

    The endpoint remains the exact original IK solution. These are candidate
    goals only: the existing controller subdivision and full transition/support
    checks must subsequently accept every interpolation.
    """
    first = _finite(start, (8,), 'coordinator start')
    last = _finite(goal, (8,), 'coordinator goal')
    if (not math.isfinite(margin) or margin < 0
            or not math.isfinite(minimum_support) or not 0 < minimum_support <= 1
            or shoulder_progress_power not in (1., .5)):
        raise ValueError('invalid coordinator policy')
    powers = (np.asarray([shoulder_progress_power, 1., 1., 1., 1., 1.])
              if joint_progress_powers is None else
              _finite(joint_progress_powers, (6,), 'joint progress powers'))
    if not all(value in (.5, 1., 2.) for value in powers):
        raise ValueError('unsupported joint progress power')
    lower, upper = np.asarray(chain.lower), np.asarray(chain.upper)
    for q in (first, last):
        if (np.any(q < lower) or np.any(q > upper)
                or np.any(q[1:] < lower[1:]+margin)
                or np.any(q[1:] > upper[1:]-margin)):
            return None
    lo, hi = float(lower[-1]+margin), float(upper[-1]-margin)
    if lo > hi:
        return None
    previous = float(first[-1])
    goals = []
    for fraction in np.linspace(.05, 1., 20):
        q = first + (last-first)*fraction
        q[1:7] = first[1:7] + (last[1:7]-first[1:7]) * fraction**powers
        zero = q.copy(); zero[-1] = 0.
        quarter = zero.copy(); quarter[-1] = np.pi/2
        optimum = math.atan2(float(chain.forward(quarter)[2, 1]), float(chain.forward(zero)[2, 1]))
        options = [lo, hi]
        options.extend(optimum+k*2*np.pi for k in range(-2, 3) if lo <= optimum+k*2*np.pi <= hi)
        scored = []
        for roll in options:
            probe = q.copy(); probe[-1] = roll
            scored.append((float(chain.forward(probe)[2, 1]), float(roll)))
        maximum = max(support for support, _ in scored)
        if not math.isfinite(maximum) or maximum < minimum_support:
            return None
        roll = min((roll for support, roll in scored if maximum-support <= 1e-9),
                   key=lambda value: (abs(value-previous), value))
        if abs(roll-previous) > np.pi:
            return None
        q[-1] = roll
        goals.append(q); previous = roll
    if abs(float(last[-1])-previous) > np.pi:
        return None
    if not np.array_equal(goals[-1], last):
        goals.append(last.copy())
    return goals
