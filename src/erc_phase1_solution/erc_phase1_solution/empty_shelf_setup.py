"""DRAFT bounded setup proposals; every returned leg receives full admission.

Coarse samples only prune/search proposal edges. A result is returned only after
all existing dense empty-body/tool checks AND dense registered shelf bounds pass.
No movement, changed collision tolerances, fabricated measured state, or cache of
admissibility across calls. Exact Cartesian clearance remains unchanged.
"""
import time
import numpy as np


def _folded_intermediate_proposals(first, last):
    """Bounded numerical proposals, admitted only by the caller's full guards.

    Turn arm5 while the arm is folded, then raise arm2 before changing the
    shoulder/elbow orientation. Preserve the measured wrist until the final
    clearance leg. The middle-row route needs this order to avoid sweeping
    the forearm into the shelf; its goal comes from the current IK solution.
    """
    for shoulder, elbow in ((.7, -2.1), (.5, -2.1), (.3, -2.1),
                            (.7, -2.25), (.5, -2.25), (.3, -2.25)):
        current = first.copy()
        route = []
        for joint, value in ((5, last[5]), (2, shoulder), (1, last[1]),
                             (3, last[3]), (4, elbow)):
            current = current.copy()
            current[joint] = value
            route.append(current)
        yield route


def plan_registered_empty_setup(node, first, last, *, guard, bounds, wall_budget=180., emit=None):
    first,last=np.asarray(first,dtype=float),np.asarray(last,dtype=float)
    if first.shape!=(8,) or last.shape!=(8,) or not np.all(np.isfinite(first)) or not np.all(np.isfinite(last)) or first[0]!=last[0]:
        raise ValueError('registered_setup_endpoint_shape')
    if not np.isfinite(wall_budget) or not 0 < wall_budget <= 300.:
        raise ValueError('registered_setup_wall_budget')
    began=time.monotonic();coarse_cache={};point_cache={};blocked=set();full_cache={}
    def deduplicate(route):
        result=[];previous=first
        for value in route:
            q=np.asarray(value,dtype=float)
            if np.array_equal(q,previous):continue
            result.append(q);previous=q
        if result and np.array_equal(result[-1],last):result.pop()
        return result
    def check_budget():
        if node._cancel.is_set():raise RuntimeError('registered_setup_cancelled')
        if time.monotonic()-began>wall_budget:raise RuntimeError('registered_setup_budget')
    def static(q):
        key=q.tobytes()
        if key not in point_cache:
            point_cache[key]=(bounds.sample(q,False) is None and guard.sample(q,guard.open_aperture) is None)
        return point_cache[key]
    def coarse(a,b):
        check_budget();a,b=np.asarray(a),np.asarray(b);key=(a.tobytes(),b.tobytes())
        if key in blocked:return False
        if key in coarse_cache:return coarse_cache[key]
        passed=node._retracted_transition_is_safe(a,b) and static(a) and static(b)
        if passed:
            passed=all(bounds.sample(a+(b-a)*fraction,False) is None for fraction in (.125,.25,.375,.5,.625,.75,.875))
        coarse_cache[key]=passed
        return passed
    def full(a,b):
        check_budget();key=(a.tobytes(),b.tobytes())
        if key not in full_cache:
            full_cache[key]=bounds.edge(a,b,allow_entry=False) is None and guard.retracted_edge(a,b)
        return full_cache[key]
    check_budget()
    if not static(first) or not static(last):return None
    # Common admitted lower-row proposal: turn the shoulder while the arm
    # stays folded, position joint2, then join the exact Cartesian clearance.
    # All values come from the measured initial state or freshly solved goal.
    joint1 = first.copy(); joint1[1] = last[1]
    joint12 = joint1.copy(); joint12[2] = last[2]
    # A checked shortcut coordinates joints1+2 in the first leg. Try it
    # first; retain the separate-joint proposal if that merged sweep rejects.
    # Keep the proven row3 prefixes first, then try a bounded family that
    # keeps the wrist folded during the row2 shoulder sweep. These are only
    # proposals: neither coarse samples nor a known joint order admit motion.
    proposals = [[joint12], [joint1, joint12]]
    proposals.extend(_folded_intermediate_proposals(first, last))
    for proposal_index, values in enumerate(proposals):
        prefix = deduplicate(values)
        previous = first
        coarse_ok = True
        for goal in [*prefix, last]:
            if not coarse(previous, goal):coarse_ok = False;break
            previous = goal
        if coarse_ok:
            previous = first; admitted = True
            for goal in [*prefix, last]:
                if not full(previous, goal):
                    blocked.add((previous.tobytes(), goal.tobytes()));admitted=False;break
                previous = goal
            if admitted:
                check_budget()
                if emit:emit('setup_fully_admitted_prefix',proposal_index=proposal_index,route=prefix,dense_edges=len(full_cache))
                check_budget()
                return [q.copy() for q in prefix]
    for attempt in range(8):
        route=node._plan_retracted_transition(first,last,edge_validator=coarse)
        if route is None:return None
        route=deduplicate(route)
        if emit:emit('setup_proposal',attempt=attempt,route=route,coarse_edges=len(coarse_cache))
        previous=first;admitted=True
        for goal in [*route,last]:
            goal=np.asarray(goal)
            if not full(previous,goal):
                blocked.add((previous.tobytes(),goal.tobytes()));admitted=False
                if emit:emit('setup_edge_rejected',attempt=attempt,from_q=previous,to_q=goal,body_reason=guard.last_rejection)
                break
            previous=goal
        if admitted:
            check_budget()
            if emit:emit('setup_fully_admitted',attempt=attempt,route=route,coarse_edges=len(coarse_cache),dense_edges=len(full_cache))
            check_budget()
            return [np.asarray(q).copy() for q in route]
    return None
