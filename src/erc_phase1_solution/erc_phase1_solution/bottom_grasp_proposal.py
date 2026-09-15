"""Unadmitted numerical proposals for the explicit bottom-row planner.

The legacy low clearance is used only to reproduce a useful grasp IK branch.
Its old approach is known to collide with the torso and is NEVER returned as
an executable approach.  Rebuild with bottom_shelf_approach and then require
ordinary lift/carry/recovery admission before any command.
"""
from dataclasses import dataclass
import numpy as np

from .kinematics import pose_matrix
from .motion_profiles import IK_JOINTS, shelf_pinch_orientations


@dataclass(frozen=True)
class BottomGraspProposal:
    grasp_position: np.ndarray
    grasp_solution: np.ndarray
    withdrawal_solutions: tuple
    rotation: np.ndarray
    numerical_solver_score: float


def _freeze(value):
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def propose_bottom_grasp(node, front, *, measured_start):
    """Use ordinary measured-current/HOME/OFFER/PREGRASP seed selection.

    No custom seed, stored grasp vector, stale simulator pose, weakened IK
    tolerance, skipped setup-selection option, or dispatch is used.  Numerical
    proposal selection retains the legacy link-origin check solely to preserve
    the previously investigated branch; it does not admit the old setup.
    """
    if node._cancel.is_set():
        raise RuntimeError('bottom_grasp_proposal_cancelled')
    point = np.asarray(front, dtype=float)
    start = np.asarray(measured_start, dtype=float)
    if (point.shape != (3,) or not np.all(np.isfinite(point))
            or not .565 < point[0] <= .80 or abs(point[1]) > .15
            or not .45 < point[2] < .78):
        raise ValueError('bottom_grasp_proposal_front_outside_workspace')
    if (start.shape != (len(IK_JOINTS),) or not np.all(np.isfinite(start))
            or np.any(start < node.chain.lower) or np.any(start > node.chain.upper)):
        raise ValueError('bottom_grasp_proposal_measured_start_invalid')
    if (not 0 < node.pick_position_tolerance <= .0005
            or not 0 < node.pick_orientation_tolerance <= .01
            or not 0 < node.cartesian_joint_step <= .40):
        raise ValueError('bottom_grasp_proposal_precision_or_joint_step')
    grasp = point+[.025, 0., 0.]
    pregrasp = grasp-[.14, 0., 0.]
    clearance = pregrasp.copy(); clearance[0] = .45
    positions = [clearance,
        *node._interpolate_positions(clearance, pregrasp, .06),
        *node._interpolate_positions(pregrasp, grasp, .06)]
    rotation = shelf_pinch_orientations(float(point[2]))[3]
    # Exact original source strategy.  The returned old setup and old approach
    # are discarded; only grasp and short withdrawal seed the new full plan.
    solutions, index, score, _ = node._solve_cartesian_path(
        positions, [rotation], .35, endpoint_first=False, first_valid=False,
        transition_start=start.copy(), position_tolerance=node.pick_position_tolerance,
        orientation_tolerance=node.pick_orientation_tolerance)
    if index != 0 or len(solutions) != len(positions) or len(solutions) < 2:
        raise RuntimeError('bottom_grasp_proposal_solver_shape')
    previous = None
    for position, raw in zip(positions, solutions):
        q = np.asarray(raw, dtype=float)
        if (q.shape != start.shape or not np.all(np.isfinite(q))
                or np.any(q < node.chain.lower) or np.any(q > node.chain.upper)
                or abs(q[0]-.35) > 1e-9):
            raise RuntimeError('bottom_grasp_proposal_solver_joint_limits')
        residual = node.chain.pose_error(node.chain.forward(q), pose_matrix(position, rotation))
        if (np.linalg.norm(residual[:3]) > node.pick_position_tolerance
                or np.linalg.norm(residual[3:]) > node.pick_orientation_tolerance
                or (previous is not None and
                    np.max(np.abs(q[1:]-previous[1:])) > node.cartesian_joint_step)):
            raise RuntimeError('bottom_grasp_proposal_solver_residual_or_step')
        previous = q
    if node._cancel.is_set():
        raise RuntimeError('bottom_grasp_proposal_cancelled')
    if not np.isfinite(score):
        raise RuntimeError('bottom_grasp_proposal_score_nonfinite')
    return BottomGraspProposal(_freeze(grasp), _freeze(solutions[-1]),
        (_freeze(solutions[-2]),), _freeze(rotation), float(score))


def propose_and_rebuild_bottom_approach(node, front, *, empty_guard, bay):
    """Only return after the actual empty entry has passed every admission.

    The caller must still run actual lift/carry/recovery with the returned
    approach.extraction_solutions and retain those admitted results for dispatch.
    """
    from .bottom_shelf_approach import plan_bottom_shelf_approach
    proposal = propose_bottom_grasp(node, front, measured_start=empty_guard.start)
    approach = plan_bottom_shelf_approach(node, front, proposal.grasp_solution,
        empty_guard=empty_guard, withdrawal_solutions=proposal.withdrawal_solutions, bay=bay)
    if (not np.array_equal(approach.solutions[-1], proposal.grasp_solution)
            or len(approach.extraction_solutions) != len(proposal.withdrawal_solutions)
            or any(not np.array_equal(actual, expected) for actual, expected in
                   zip(approach.extraction_solutions, proposal.withdrawal_solutions))):
        raise RuntimeError('bottom_grasp_proposal_changed_during_rebuild')
    return proposal, approach
