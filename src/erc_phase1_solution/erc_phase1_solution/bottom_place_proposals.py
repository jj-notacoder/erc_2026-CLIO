"""Extra bottom-book PLACE proposals; ordinary full admission remains mandatory.

No stored arm pose, column/colour choice, motion command or acceptance cache.
The current release-only Request owns target/contact/attachment/scene identity.
"""
import math
import re

import numpy as np


# Same canonical stock naming contract as rigid_palm_live_preflight, kept pure
# here so the competition planner does not import the Gazebo diagnostic adapter.
_STOCK_BOOK = re.compile(r'book_col_([1-5])_row_([2-5])_(red|green|yellow|blue)')
_RETAINED_FLAGS = ('_transport_lock_engaged', '_gravity_supported_payload',
                   '_payload_monitor_enabled')


def verified_bottom_request(node, request):
    """Scope extra proposals to a currently retained canonical bottom target.

    This is proposal eligibility, not a replacement for measured retention or
    the original Request's fresh scene/contact/held-object admission.
    """
    from .release_only_place_planning import Request
    if type(request) is not Request or request.node is not node:
        return False
    model = request.identity.target_model
    match = _STOCK_BOOK.fullmatch(model) if type(model) is str else None
    if match is None or match.group(2) != '5':
        return False
    request.require()
    with node._lock:
        request.require_locked()
        return (node._target_book_model == model
                and all(getattr(node, flag, False) is True for flag in _RETAINED_FLAGS))


def require_verified_bottom(node, request):
    if not verified_bottom_request(node, request):
        raise RuntimeError('bottom_place_proposal_retained_identity_changed')


def nearest_negative_wrist_goals(chain, start, goals, *, margin, minimum_support,
                                 checkpoint):
    """Change only intermediate q7 to the nearest sufficiently supported angle.

    On the official chain the final revolute wrist's signed jaw-up is exactly
    A*cos(q7)+B*sin(q7). Ordinary FK supplies A/B and verifies every proposal.
    A 0.005 waypoint reserve improves interpolation; it never changes the full
    dense body's original support threshold. The final IK endpoint is exact.
    """
    def finite(value, shape, label):
        result = np.asarray(value, dtype=float)
        if result.shape != shape or not np.all(np.isfinite(result)):
            raise ValueError('invalid ' + label)
        return result.copy()

    first = finite(start, (8,), 'nearest wrist start')
    if not isinstance(goals, (list, tuple)) or not goals or not callable(checkpoint):
        raise ValueError('invalid nearest wrist goals/checkpoint')
    proposed = [finite(q, (8,), 'nearest wrist goal') for q in goals]
    lower = finite(chain.lower, (8,), 'nearest wrist lower limits')
    upper = finite(chain.upper, (8,), 'nearest wrist upper limits')
    if (not math.isfinite(margin) or margin < 0.
            or not math.isfinite(minimum_support) or not 0. < minimum_support <= 1.
            or np.any(lower >= upper)):
        raise ValueError('invalid nearest wrist policy')
    checkpoint()
    # Final endpoint and start must remain the actual legal negative states.
    for q in (first, proposed[-1]):
        if (q[-1] >= 0. or np.any(q < lower) or np.any(q > upper)
                or np.any(q[1:] < lower[1:] + margin)
                or np.any(q[1:] > upper[1:] - margin)):
            return None
    lo = float(lower[-1] + margin)
    hi = min(-1e-12, float(upper[-1] - margin))
    threshold = max(.75, minimum_support) + .005
    if lo > hi or threshold > 1.:
        return None
    previous = float(first[-1])
    for q in proposed[:-1]:
        checkpoint()
        if (np.any(q[:-1] < lower[:-1]) or np.any(q[:-1] > upper[:-1])
                or np.any(q[1:7] < lower[1:7] + margin)
                or np.any(q[1:7] > upper[1:7] - margin)):
            return None
        zero = q.copy(); zero[-1] = 0.
        quarter = zero.copy(); quarter[-1] = np.pi / 2.
        a = float(chain.forward(zero)[2, 1])
        b = float(chain.forward(quarter)[2, 1])
        radius = math.hypot(a, b)
        if not math.isfinite(radius) or radius < threshold:
            return None
        optimum = math.atan2(b, a)
        width = math.acos(min(1., threshold / radius))
        options = []
        for period in range(-2, 3):
            low = max(lo, optimum - width + period * 2. * math.pi)
            high = min(hi, optimum + width + period * 2. * math.pi)
            if low > high:
                continue
            nearest = min(high, max(low, previous))
            inward = min(high, max(low, nearest + (1e-10 if nearest == low else -1e-10)))
            for angle in (nearest, inward):
                q[-1] = angle
                if chain.forward(q)[2, 1] >= threshold:
                    options.append(float(angle))
        if not options:
            return None
        q[-1] = min(options, key=lambda value: (abs(value - previous), value))
        previous = float(q[-1])
    checkpoint()
    return proposed
