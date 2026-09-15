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
from .empty_shelf_bounds import EmptyShelfBounds
from .empty_shelf_setup import plan_registered_empty_setup


@dataclass(frozen=True)
class LowerShelfCandidate:
    name: str
    torso_height: float
    pitch: float
    grasp_depth_offset: float = .025
    grasp_lateral_offset: float = -.001
    grasp_vertical_offset: float = -.015
    clearance_x: float = .46
    loaded_clearance_index: int | None = 1
    endpoint_first: bool = True
    carry_options: Mapping = field(default_factory=dict)
    approach_kind: str = 'ordinary'


ROW_THREE_CANDIDATES = (
    LowerShelfCandidate('row3_supported_positive_wrist', .10, .25,
        grasp_lateral_offset=0., grasp_vertical_offset=0., clearance_x=.45,
        loaded_clearance_index=3, endpoint_first=False,
        carry_options={'extension_distance': .12, 'staging_vertical_offset': .18,
                       'compact_path': 'middle'}),
)


ROW_TWO_CANDIDATES = (
    LowerShelfCandidate('row2_supported_torso_010', .10, -.50,
        grasp_vertical_offset=.005),
)

BOTTOM_CANDIDATES = (
    LowerShelfCandidate('bottom_redundant_supported', .35, 0.,
        grasp_lateral_offset=0., grasp_vertical_offset=0., clearance_x=.45,
        loaded_clearance_index=None, endpoint_first=False,
        carry_options={'extension_distance': .03, 'staging_vertical_offset': -.06,
                       'compact_path': 'bottom'}, approach_kind='bottom_redundant'),
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
    loaded_clearance_index: int | None
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
    extraction_solutions: tuple = ()
    shelf_geometry_metrics: Mapping = field(default_factory=dict)


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
    if .45 < float(front[2]) <= .76:
        return BOTTOM_CANDIDATES
    raise RuntimeError('lower_shelf_pick_row_has_no_validated_candidate')


def plan_lower_shelf_pick(node, front, *, empty_guard, lift_planner, bay=None):
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
    if bay is None:
        raise ValueError('lower_shelf_pick_requires_registered_bay')
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

    def planning_stage(spec, stage):
        node._publish_status('lower_shelf_planning_stage', command='pick',
            candidate=spec.name, stage=stage)

    node._publish_status('lower_shelf_planning_stage', command='pick',
        stage='captured_empty_context',
        measured_start=np.asarray(empty_guard.start).tolist(),
        measured_right=np.asarray(getattr(empty_guard, 'right', [])).tolist(),
        measured_head=np.asarray(getattr(empty_guard, 'head', [])).tolist())

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
                        or spec.approach_kind not in ('ordinary', 'bottom_redundant')
                        or (spec.approach_kind == 'ordinary' and (
                            isinstance(spec.loaded_clearance_index, bool)
                            or not isinstance(spec.loaded_clearance_index, int)))
                        or (spec.approach_kind == 'bottom_redundant' and (
                            spec.loaded_clearance_index is not None or bay is None
                            or spec.torso_height != .35 or spec.pitch != 0.
                            or spec.grasp_depth_offset != .025
                            or spec.grasp_lateral_offset != 0.
                            or spec.grasp_vertical_offset != 0.))
                        or type(spec.endpoint_first) is not bool
                        or not isinstance(spec.carry_options, Mapping)
                        or 'defer_return' in spec.carry_options):
                    raise ValueError('lower_shelf_pick_candidate_parameters_invalid')
                node.pick_torso_height = float(spec.torso_height)
                stage = 'empty_opening'
                planning_stage(spec, stage)
                if not empty_guard.opening():
                    raise RuntimeError(str(empty_guard.last_rejection))
                elevated = np.asarray(empty_guard.start, dtype=float).copy()
                elevated[0] = spec.torso_height
                stage = 'empty_torso'
                planning_stage(spec, stage)
                if not empty_guard.edge(empty_guard.start, elevated):
                    raise RuntimeError(str(empty_guard.last_rejection))
                bounds = EmptyShelfBounds(node, observed_front, None, empty_guard, bay)
                reason = bounds.edge(empty_guard.start, elevated, allow_entry=False)
                if reason:
                    raise RuntimeError(reason)
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
                if (spec.approach_kind == 'ordinary'
                        and not 0 <= spec.loaded_clearance_index < len(positions)-1):
                    raise ValueError('lower_shelf_pick_loaded_index_outside_route')

                def validate(solutions, transition, explicit_extraction=None):
                    nonlocal selected
                    _check_cancelled(node)
                    phase = 'empty_approach'
                    planning_stage(spec, phase)
                    selected = None
                    node._cached_post_retreat_plan = None
                    try:
                        if not empty_guard.candidate(solutions, transition):
                            raise RuntimeError(str(empty_guard.last_rejection))
                        phase = 'registered_shelf_approach'
                        planning_stage(spec, phase)
                        # Recheck the returned complete route independently of
                        # proposal-search pruning. These metrics cover only
                        # this route, excluding rejected search alternatives.
                        admitted_bounds = EmptyShelfBounds(
                            node, observed_front, solutions[-1], empty_guard, bay)
                        previous = elevated
                        for goal in (*transition, solutions[0]):
                            reason = admitted_bounds.edge(previous, goal, allow_entry=False)
                            if reason:
                                raise RuntimeError(reason)
                            previous = goal
                        for goal in solutions[1:]:
                            reason = admitted_bounds.edge(previous, goal, allow_entry=True)
                            if reason:
                                raise RuntimeError(reason)
                            previous = goal
                        phase = 'open_gripper_approach'
                        planning_stage(spec, phase)
                        opened = check_open_gripper_approach(node, observed_front, solutions,
                                                            torso_height=spec.torso_height)
                        if not opened.ok:
                            raise RuntimeError(opened.reason)
                        _check_cancelled(node)
                        extraction = (list(explicit_extraction)
                            if explicit_extraction is not None else
                            list(reversed(solutions[spec.loaded_clearance_index:-1])))
                        if not extraction:
                            raise RuntimeError('lower_shelf_pick_empty_withdrawal_proposal')
                        phase = 'lift'
                        planning_stage(spec, phase)
                        lift_plan = lift_planner(solutions[-1], extraction)
                        _check_cancelled(node)
                        phase = 'deferred_return'
                        planning_stage(spec, phase)
                        return_result = node._plan_carried_return(
                            observed_front, solutions[-1], lift_plan.terminal,
                            rotations[0], spec.torso_height, defer_return=True,
                            **dict(spec.carry_options))
                        if node._cached_post_retreat_plan is None:
                            raise RuntimeError('lower_shelf_pick_deferred_cache_missing')
                        phase = 'extraction_volume'
                        planning_stage(spec, phase)
                        previous = solutions[-1]
                        for goal in lift_plan.route:
                            _check_cancelled(node)
                            if not node._carried_robot_transition_is_safe(previous, goal,
                                                                         return_result[4]):
                                raise RuntimeError('lower_shelf_pick_extraction_sweep_rejected')
                            previous = goal
                        phase = 'unloaded_recovery'
                        planning_stage(spec, phase)
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
                                    deepcopy(node._cached_post_retreat_plan), recovery,
                                    tuple(_array(q) for q in extraction),
                                    dict(samples=admitted_bounds.samples,
                                        plane_base=admitted_bounds.plane_point.tolist(),
                                        normal_uncertainty_m=admitted_bounds.normal_uncertainty_m,
                                        minimum_floor_m=admitted_bounds.minimum_floor,
                                        minimum_roof_m=admitted_bounds.minimum_roof,
                                        minimum_side_m=admitted_bounds.minimum_side,
                                        minimum_back_m=admitted_bounds.minimum_back,
                                        full_shelf_mesh_certificate=False))
                        return True
                    except LowerShelfPlanningCancelled:
                        raise
                    except (RuntimeError, ValueError) as error:
                        _check_cancelled(node)
                        rejected(spec, phase, error)
                        return False

                if spec.approach_kind == 'bottom_redundant':
                    from .bottom_grasp_proposal import propose_and_rebuild_bottom_approach
                    stage = 'bottom_redundant_approach'
                    planning_stage(spec, stage)
                    proposal, approach = propose_and_rebuild_bottom_approach(
                        node, observed_front, empty_guard=empty_guard, bay=bay)
                    positions, solutions = approach.positions, approach.solutions
                    transition, score = approach.transition_waypoints, approach.path_score
                    grasp, rotations = proposal.grasp_position, (proposal.rotation,)
                    orientation_index = 0
                    if not validate(solutions, transition, approach.extraction_solutions):
                        raise RuntimeError('bottom_shelf_complete_candidate_rejected')
                else:
                    stage = 'cartesian_path'
                    planning_stage(spec, stage)

                    def strict_setup_edge(first, last):
                        return (bounds.edge(first, last, allow_entry=False) is None
                                and empty_guard.retracted_edge(first, last))

                    def setup(first, last):
                        return plan_registered_empty_setup(
                            node, first, last, guard=empty_guard, bounds=bounds,
                            emit=lambda event, **fields: node._publish_status(
                                'lower_shelf_planning_stage', command='pick',
                                candidate=spec.name, stage=event,
                                **{key: value for key, value in fields.items()
                                   if key in ('attempt', 'coarse_edges', 'dense_edges')}))

                    solutions, orientation_index, score, transition = node._solve_cartesian_path(
                        positions, rotations, spec.torso_height, endpoint_first=spec.endpoint_first,
                        first_valid=True, transition_start=elevated,
                        transition_edge_validator=strict_setup_edge,
                        setup_transition_planner=setup,
                        candidate_validator=validate,
                        position_tolerance=node.pick_position_tolerance,
                        orientation_tolerance=node.pick_orientation_tolerance)
                if selected is None or orientation_index != 0:
                    raise RuntimeError('lower_shelf_pick_solver_did_not_admit_complete_candidate')
                lift_plan, return_result, cached, recovery, extraction, shelf_metrics = selected
                accepted_plan = LowerShelfPickPlan(
                    spec.name, tuple(_array(p) for p in positions), _array(grasp),
                    tuple(_array(r) for r in rotations), tuple(_array(q) for q in solutions),
                    int(orientation_index), float(score), tuple(_array(q) for q in transition),
                    float(spec.torso_height), spec.loaded_clearance_index,
                    lift_plan, return_result, cached, tuple(_array(q) for q in recovery),
                    float(spec.grasp_depth_offset), float(spec.grasp_lateral_offset),
                    float(spec.grasp_vertical_offset),
                    endpoint_first=spec.endpoint_first,
                    carry_options=MappingProxyType(deepcopy(dict(spec.carry_options))),
                    extraction_solutions=extraction,
                    shelf_geometry_metrics=MappingProxyType(shelf_metrics))
                node._cached_post_retreat_plan = cached
                node._publish_status('lower_shelf_pick_candidate_planned', command='pick',
                    candidate=spec.name, torso_height=spec.torso_height,
                    loaded_clearance_index=spec.loaded_clearance_index,
                    rejected_candidates=len(failures),
                    shelf_geometry=dict(shelf_metrics))
                return accepted_plan
            except LowerShelfPlanningCancelled:
                raise
            except (RuntimeError, ValueError) as error:
                _check_cancelled(node)
                rejected(specification, stage, error)
        raise RuntimeError('lower_shelf_pick_no_complete_route:' + ';'.join(
            entry['candidate'] + '/' + entry['stage'] + ':' + entry['reason']
            for entry in failures))
    finally:
        node.pick_torso_height = original_torso
        if accepted_plan is None:
            node._cached_post_retreat_plan = None
