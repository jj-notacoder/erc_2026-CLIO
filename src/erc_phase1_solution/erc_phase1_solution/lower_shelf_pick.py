"""Plan a lower-shelf PICK completely before admitting any motion.

This module dispatches no commands.  Candidate geometry must pass the existing
empty-hand, open-tool, lift, loaded-volume, support and recovery predicates.
Candidates are enabled only after their complete approach and carried routes
have passed the existing geometry checks in independent offline fixtures.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .motion_profiles import HOME, _rotation_x, _rotation_y
from .open_gripper_approach import check_open_gripper_approach


@dataclass(frozen=True)
class LowerShelfCandidate:
    name: str
    torso_height: float
    pitch: float
    grasp_depth_offset: float = .025
    grasp_lateral_offset: float = -.001
    grasp_vertical_offset: float = -.015
    clearance_x: float = .46
    loaded_clearance_index: int = 1
    endpoint_first: bool = True
    carry_options: Mapping = field(default_factory=dict)


ROW_THREE_CANDIDATES = (
    LowerShelfCandidate('row3_supported_positive_wrist', .10, .25,
        grasp_lateral_offset=0., grasp_vertical_offset=0., clearance_x=.45,
        loaded_clearance_index=3, endpoint_first=False,
        carry_options={'extension_distance': .12, 'staging_vertical_offset': .18,
                       'compact_path': 'middle'}),
)


ROW_TWO_CANDIDATES = (
    LowerShelfCandidate('row2_supported_torso_010', .10, -.50),
)


@dataclass(frozen=True)
class LowerShelfPickPlan:
    candidate_name: str
    positions: tuple
    grasp: np.ndarray
    rotations: tuple
    solutions: tuple
    orientation_index: int
    path_score: float
    transition_waypoints: tuple
    pick_torso_height: float
    loaded_clearance_index: int
    lift_plan: object
    return_result: tuple
    cached_post_retreat_plan: dict
    unloaded_recovery_route: tuple
    grasp_depth_offset: float
    grasp_lateral_offset: float
    grasp_vertical_offset: float
    loaded_clearance_lift: float = 0.
    endpoint_first: bool = True
    staged_empty_gripper: bool = False
    deferred_return: bool = True
    carry_options: Mapping = field(default_factory=dict)


class LowerShelfPlanningCancelled(RuntimeError):
    pass


def _check_cancelled(node):
    event = getattr(node, '_cancel', None)
    if event is not None and event.is_set():
        raise LowerShelfPlanningCancelled('lower_shelf_pick_cancelled')


def _array(value):
    """Return an owned, read-only numerical snapshot."""
    array = np.asarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _candidates(front):
    # This is a measured-height band, not a column/color/seed lookup.
    if 1.10 <= float(front[2]) < 1.42:
        return ROW_TWO_CANDIDATES
    if .76 < float(front[2]) < 1.10:
        return ROW_THREE_CANDIDATES
    raise RuntimeError('lower_shelf_pick_row_has_no_validated_candidate')


def plan_lower_shelf_pick(node, front, *, empty_guard, lift_planner):
    """Select a complete guarded route, retaining its lift/carry preflight.

    ``lift_planner(grasp_solution, extraction_solutions)`` performs the caller's
    fresh sensor admission and ordinary lift planner. ``empty_guard`` is the
    caller's captured EmptyPickupCollision context.  ``node.pick_torso_height``
    is temporarily bound because the existing empty candidate predicate reads
    it; its original value is restored on every exit.  The caller must propagate
    ``plan.pick_torso_height`` locally through subsequent PICK dispatch.
    """
    node._cached_post_retreat_plan = None
    observed_front = _array(front)
    if (observed_front.shape != (3,) or not np.all(np.isfinite(observed_front))
            or not .30 < observed_front[0] < 1.30
            or abs(observed_front[1]) >= .70 or not .45 < observed_front[2] < 1.85):
        raise ValueError('lower_shelf_pick_front_outside_workspace')
    if empty_guard is None or not callable(lift_planner):
        raise ValueError('lower_shelf_pick_requires_empty_and_lift_guards')
    if not (0 < node.pick_position_tolerance <= .0005
            and 0 < node.pick_orientation_tolerance <= .01):
        raise ValueError('lower_shelf_pick_requires_precision_tolerances')
    specifications = tuple(_candidates(observed_front))
    if not specifications or len(specifications) > 8:
        raise ValueError('lower_shelf_pick_candidate_count_outside_bound')
    original_torso = node.pick_torso_height
    failures = []
    accepted_plan = None
    node._cached_post_retreat_plan = None

    def rejected(spec, stage, error):
        node._cached_post_retreat_plan = None
        entry = dict(candidate=getattr(spec, 'name', 'invalid_candidate'),
                     stage=stage, reason=str(error))
        failures.append(entry)
        node._publish_status('lower_shelf_pick_candidate_rejected', command='pick', **entry)

    try:
        for specification in specifications:
            _check_cancelled(node)
            selected = None
            stage = 'candidate_parameters'
            node._cached_post_retreat_plan = None
            try:
                spec = specification
                if not isinstance(spec, LowerShelfCandidate):
                    raise ValueError('lower_shelf_pick_candidate_type_invalid')
                values = np.asarray([spec.torso_height, spec.pitch,
                    spec.grasp_depth_offset, spec.grasp_lateral_offset,
                    spec.grasp_vertical_offset, spec.clearance_x], dtype=float)
                if (not np.all(np.isfinite(values))
                        or not float(node.chain.lower[0]) <= spec.torso_height <= float(node.chain.upper[0])
                        or not 0 < spec.grasp_depth_offset <= .06
                        or abs(spec.grasp_lateral_offset) > .01
                        or abs(spec.grasp_vertical_offset) > .05
                        or not .30 <= spec.clearance_x <= .65
                        or isinstance(spec.loaded_clearance_index, bool)
                        or not isinstance(spec.loaded_clearance_index, int)
                        or type(spec.endpoint_first) is not bool
                        or not isinstance(spec.carry_options, Mapping)
                        or 'defer_return' in spec.carry_options):
                    raise ValueError('lower_shelf_pick_candidate_parameters_invalid')
                node.pick_torso_height = float(spec.torso_height)
                stage = 'empty_opening'
                if not empty_guard.opening():
                    raise RuntimeError(str(empty_guard.last_rejection))
                elevated = np.asarray(empty_guard.start, dtype=float).copy()
                elevated[0] = spec.torso_height
                stage = 'empty_torso'
                if not empty_guard.edge(empty_guard.start, elevated):
                    raise RuntimeError(str(empty_guard.last_rejection))
                _check_cancelled(node)
                grasp = observed_front + [spec.grasp_depth_offset,
                    spec.grasp_lateral_offset, spec.grasp_vertical_offset]
                pregrasp = grasp.copy(); pregrasp[0] -= node.pregrasp_offset
                clearance = pregrasp.copy(); clearance[0] = spec.clearance_x
                if pregrasp[0] <= clearance[0]:
                    raise RuntimeError('lower_shelf_pick_too_close_for_staged_approach')
                positions = [clearance,
                    *node._interpolate_positions(clearance, pregrasp, node.cartesian_step),
                    *node._interpolate_positions(pregrasp, grasp, node.cartesian_step)]
                rotations = (_rotation_y(spec.pitch) @ _rotation_x(np.pi),)
                if not 0 <= spec.loaded_clearance_index < len(positions)-1:
                    raise ValueError('lower_shelf_pick_loaded_index_outside_route')

                def validate(solutions, transition):
                    nonlocal selected
                    _check_cancelled(node)
                    phase = 'empty_approach'
                    selected = None
                    node._cached_post_retreat_plan = None
                    try:
                        if not empty_guard.candidate(solutions, transition):
                            raise RuntimeError(str(empty_guard.last_rejection))
                        phase = 'open_gripper_approach'
                        opened = check_open_gripper_approach(node, observed_front, solutions,
                                                            torso_height=spec.torso_height)
                        if not opened.ok:
                            raise RuntimeError(opened.reason)
                        _check_cancelled(node)
                        extraction = list(reversed(solutions[spec.loaded_clearance_index:-1]))
                        phase = 'lift'
                        lift_plan = lift_planner(solutions[-1], extraction)
                        _check_cancelled(node)
                        phase = 'deferred_return'
                        return_result = node._plan_carried_return(
                            observed_front, solutions[-1], lift_plan.terminal,
                            rotations[0], spec.torso_height, defer_return=True,
                            **dict(spec.carry_options))
                        if node._cached_post_retreat_plan is None:
                            raise RuntimeError('lower_shelf_pick_deferred_cache_missing')
                        phase = 'extraction_volume'
                        previous = solutions[-1]
                        for goal in lift_plan.route:
                            _check_cancelled(node)
                            if not node._carried_robot_transition_is_safe(previous, goal,
                                                                         return_result[4]):
                                raise RuntimeError('lower_shelf_pick_extraction_sweep_rejected')
                            previous = goal
                        phase = 'unloaded_recovery'
                        terminal = np.asarray(return_result[3][-1] if return_result[3]
                            else return_result[1][-1] if return_result[1]
                            else return_result[0][-1] if return_result[0]
                            else lift_plan.terminal, dtype=float)
                        recovery = []
                        if not return_result[3]:
                            recovery_start = terminal.copy(); recovery_start[0] = HOME[0]
                            recovery = node._plan_retracted_transition(recovery_start, HOME)
                            if recovery is None:
                                raise RuntimeError('lower_shelf_pick_unloaded_recovery_unavailable')
                            recovery = [*recovery, HOME.copy()]
                        _check_cancelled(node)
                        selected = (lift_plan, return_result,
                                    deepcopy(node._cached_post_retreat_plan), recovery)
                        return True
                    except LowerShelfPlanningCancelled:
                        raise
                    except (RuntimeError, ValueError) as error:
                        rejected(spec, phase, error)
                        return False

                stage = 'cartesian_path'
                solutions, orientation_index, score, transition = node._solve_cartesian_path(
                    positions, rotations, spec.torso_height, endpoint_first=spec.endpoint_first,
                    first_valid=True, transition_start=elevated,
                    transition_edge_validator=empty_guard.retracted_edge,
                    setup_transition_planner=empty_guard.plan_transition,
                    candidate_validator=validate,
                    position_tolerance=node.pick_position_tolerance,
                    orientation_tolerance=node.pick_orientation_tolerance)
                if selected is None or orientation_index != 0:
                    raise RuntimeError('lower_shelf_pick_solver_did_not_admit_complete_candidate')
                lift_plan, return_result, cached, recovery = selected
                accepted_plan = LowerShelfPickPlan(
                    spec.name, tuple(_array(p) for p in positions), _array(grasp),
                    tuple(_array(r) for r in rotations), tuple(_array(q) for q in solutions),
                    int(orientation_index), float(score), tuple(_array(q) for q in transition),
                    float(spec.torso_height), int(spec.loaded_clearance_index),
                    lift_plan, return_result, cached, tuple(_array(q) for q in recovery),
                    float(spec.grasp_depth_offset), float(spec.grasp_lateral_offset),
                    float(spec.grasp_vertical_offset),
                    endpoint_first=spec.endpoint_first,
                    carry_options=MappingProxyType(deepcopy(dict(spec.carry_options))))
                node._cached_post_retreat_plan = cached
                node._publish_status('lower_shelf_pick_candidate_planned', command='pick',
                    candidate=spec.name, torso_height=spec.torso_height,
                    loaded_clearance_index=spec.loaded_clearance_index,
                    rejected_candidates=len(failures))
                return accepted_plan
            except LowerShelfPlanningCancelled:
                raise
            except (RuntimeError, ValueError) as error:
                rejected(specification, stage, error)
        raise RuntimeError('lower_shelf_pick_no_complete_route:' + ';'.join(
            entry['candidate'] + '/' + entry['stage'] + ':' + entry['reason']
            for entry in failures))
    finally:
        node.pick_torso_height = original_torso
        if accepted_plan is None:
            node._cached_post_retreat_plan = None
