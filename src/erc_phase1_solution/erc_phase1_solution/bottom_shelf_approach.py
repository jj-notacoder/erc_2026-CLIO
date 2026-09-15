"""Draft: explicit empty bottom-row entry, preserving a supplied grasp/lift proposal.

No dispatch.  Actual empty-body/tool predicates and registered bay bounds admit
all motion.  The bottom-row redundant-joint schedule has nominal offline
kinematic evidence; fresh runtime guards remain mandatory for every proposal.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from .kinematics import pose_matrix
from .lift_first_extraction import _bay_values
from .motion_profiles import IK_JOINTS, shelf_pinch_orientations
from .open_gripper_approach import check_open_gripper_approach
from .rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS
from .shelf_cradle_geometry import check_cradle_tool_sweep, check_gripper_opening

# Backward from the unchanged grasp.  Values are offsets from the observed
# front, not a stored simulator position or an executable joint trajectory.
BACKWARD_ENTRY_SCHEDULE = ((0., .65), (-.0216666666667, .80), (-.045, 1.),
    (-.070, 1.2), (-.095, 1.4), (-.120, 1.5), (-.140, 1.5))


@dataclass(frozen=True)
class BottomShelfApproachPlan:
    positions: tuple
    solutions: tuple
    transition_waypoints: tuple
    extraction_solutions: tuple
    path_score: float
    metrics: object


def _frozen(value):
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _cancel(node):
    if node._cancel.is_set():
        raise RuntimeError('bottom_empty_approach_cancelled')


def _q(node, value):
    array = np.asarray(value, dtype=float)
    if (array.shape != (len(IK_JOINTS),) or not np.all(np.isfinite(array))
            or np.any(array < node.chain.lower) or np.any(array > node.chain.upper)):
        raise ValueError('bottom_empty_approach_joint_limits_or_shape')
    return array.copy()


from .empty_shelf_bounds import EmptyShelfBounds as _ShelfEntryBounds, _forward_vertices


def plan_bottom_shelf_approach(node, front, grasp_solution, *, empty_guard,
                              withdrawal_solutions, bay):
    """Rebuild and admit one supplied proposal; never dispatch or mutate it.

    The caller numerically proposes the grasp and its short withdrawal using
    the ordinary bottom-row IK branch.  This function preserves their exact
    values; the caller must then run its existing actual lift and carry guards
    with ``plan.extraction_solutions`` before admitting any command.
    """
    _cancel(node)
    point = np.asarray(front, dtype=float)
    if (point.shape != (3,) or not np.all(np.isfinite(point))
            or not .55 <= point[0] <= .80 or abs(point[1]) > .15
            or not .45 < point[2] < .78):
        raise ValueError('bottom_empty_front_outside_candidate_workspace')
    if tuple(node.chain.active_names) != tuple(IK_JOINTS):
        raise ValueError('bottom_empty_chain_order')
    if (not 0 < node.pick_position_tolerance <= .0005
            or not 0 < node.pick_orientation_tolerance <= .01
            or not 0 < node.cartesian_joint_step <= .40):
        raise ValueError('bottom_empty_precision_or_joint_step')
    if empty_guard.node is not node or not np.isclose(empty_guard.open_aperture, .069, atol=1e-12):
        raise ValueError('bottom_empty_guard_context')
    grasp = _q(node, grasp_solution)
    if abs(grasp[0]-.35) > 1e-9:
        raise ValueError('bottom_empty_requires_validated_torso')
    rotation = shelf_pinch_orientations(float(point[2]))[3]
    target = pose_matrix(point+[.025, 0., 0.], rotation)
    residual = node.chain.pose_error(node.chain.forward(grasp), target)
    if (np.linalg.norm(residual[:3]) > node.pick_position_tolerance
            or np.linalg.norm(residual[3:]) > node.pick_orientation_tolerance):
        raise ValueError('bottom_empty_grasp_proposal_residual')
    withdrawal = tuple(_q(node, q) for q in withdrawal_solutions)
    if len(withdrawal) != 1:
        raise ValueError('bottom_empty_requires_one_short_withdrawal')
    withdrawal_pose = node.chain.forward(withdrawal[0])
    if (abs(withdrawal[0][0]-.35) > 1e-9
            or not point[0]-.04 <= withdrawal_pose[0, 3] <= point[0]-.005
            or np.linalg.norm(withdrawal_pose[1:3, 3]-target[1:3, 3]) > .001):
        raise ValueError('bottom_empty_withdrawal_proposal_changed')
    bounds = _ShelfEntryBounds(node, point, grasp, empty_guard, bay)
    if not empty_guard.opening():
        raise RuntimeError('bottom_empty_opening:'+str(empty_guard.last_rejection))
    initial = np.asarray(empty_guard.start, dtype=float).copy()
    elevated = initial.copy(); elevated[0] = .35
    shelf_plane = float(point[0])-.065
    reason = check_gripper_opening(node, point, grasp, initial, shelf_plane,
        start_aperture=empty_guard.initial_aperture, end_aperture=empty_guard.open_aperture)
    if reason:
        raise RuntimeError('bottom_empty_opening_shelf:'+reason)
    if not empty_guard.edge(initial, elevated):
        raise RuntimeError('bottom_empty_torso:'+str(empty_guard.last_rejection))
    reason = bounds.edge(initial, elevated, allow_entry=False)
    if reason:
        raise RuntimeError(reason)

    previous = grasp.copy(); reverse = [previous]; reverse_positions = [target[:3, 3].copy()]
    total_score = 0.
    for x_offset, arm3 in BACKWARD_ENTRY_SCHEDULE:
        _cancel(node)
        position = point+[x_offset, 0., 0.]
        seed = previous.copy(); seed[3] = arm3
        solved, score = node.chain.solve(pose_matrix(position, rotation), [seed],
            position_tolerance=node.pick_position_tolerance,
            orientation_tolerance=node.pick_orientation_tolerance, max_iterations=220,
            fixed_positions={'torso_lift_joint': .35, 'arm_left_3_joint': arm3})
        if solved is None:
            raise RuntimeError(f'bottom_empty_reverse_ik:{x_offset}')
        q = _q(node, solved)
        residual = node.chain.pose_error(node.chain.forward(q), pose_matrix(position, rotation))
        if (abs(q[0]-.35) > 1e-9 or abs(q[3]-arm3) > 1e-9
                or np.linalg.norm(residual[:3]) > node.pick_position_tolerance
                or np.linalg.norm(residual[3:]) > node.pick_orientation_tolerance
                or np.max(np.abs(q[1:]-previous[1:])) > node.cartesian_joint_step):
            raise RuntimeError(f'bottom_empty_reverse_residual_or_step:{x_offset}')
        if not empty_guard.edge(q, previous):
            raise RuntimeError('bottom_empty_reverse_body:'+str(empty_guard.last_rejection))
        reason = bounds.edge(q, previous, allow_entry=True)
        if reason:
            raise RuntimeError(reason)
        total_score += float(score)
        reverse.append(q.copy()); reverse_positions.append(position.copy()); previous = q
    solutions = list(reversed(reverse)); positions = list(reversed(reverse_positions))
    opened = check_open_gripper_approach(node, point, solutions, torso_height=.35)
    if not opened.ok:
        raise RuntimeError('bottom_empty_approach_book:'+opened.reason)

    def setup_edge(first, last):
        _cancel(node)
        first, last = np.asarray(first), np.asarray(last)
        if not empty_guard.retracted_edge(first, last):
            return False
        reason = check_cradle_tool_sweep(node, point, grasp, first, last,
            shelf_plane, aperture=empty_guard.open_aperture, **empty_guard.context)
        return reason is None and bounds.edge(first, last, allow_entry=False) is None

    transition = node._plan_retracted_transition(elevated, solutions[0], edge_validator=setup_edge)
    if transition is None:
        raise RuntimeError('bottom_empty_setup_unavailable')
    # Recheck the returned route explicitly: a planner must not accidentally
    # hand back a route that bypassed its supplied admission callback.
    previous = elevated
    for goal in [*transition, solutions[0]]:
        if not setup_edge(previous, goal):
            raise RuntimeError('bottom_empty_setup_route_rejected')
        previous = np.asarray(goal)
    total_score += node._arm_route_cost(elevated, transition, solutions[0])
    total_score += sum(float(np.linalg.norm(last[1:]-first[1:]))
                       for first, last in zip(solutions, solutions[1:]))
    _cancel(node)
    return BottomShelfApproachPlan(tuple(_frozen(p) for p in positions),
        tuple(_frozen(q) for q in solutions), tuple(_frozen(q) for q in transition),
        tuple(_frozen(q) for q in withdrawal), float(total_score), MappingProxyType(dict(
            samples=bounds.samples, minimum_floor_m=bounds.minimum_floor,
            minimum_roof_m=bounds.minimum_roof, minimum_side_m=bounds.minimum_side,
            exact_grasp_preserved=True, exact_withdrawal_preserved=True,
            full_shelf_collision_certificate=False, dispatches_commands=False)))
