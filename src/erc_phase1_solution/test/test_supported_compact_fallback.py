"""Regression for shallow-grasp shelf clearance without relaxed carry gates."""

import numpy as np
import pytest

pytest.importorskip('rclpy')
from test_shutdown import _official_manipulation_planner
from erc_phase1_solution import manipulation_node as manipulation


FRONT = np.asarray([.6733843618688746, -.0564842549223933, 1.5842067119476184])


def shallow_pick():
    node = _official_manipulation_planner()
    node.joints['head_2_joint'] = .20
    node.carried_book_dimensions = np.asarray([.16, .02, .25])
    node.top_row_loaded_clearance_lift = 0.
    grasp = FRONT + [.025, -.001, -.015]
    pregrasp = grasp - [.14, 0., 0.]
    clearance = pregrasp.copy()
    clearance[0] = .46
    positions = [clearance,
                 *node._interpolate_positions(clearance, pregrasp, .06),
                 *node._interpolate_positions(pregrasp, grasp, .06)]
    rotations = manipulation.shelf_pinch_orientations(grasp[2])
    solutions, index, _, _ = node._solve_cartesian_path(
        positions, rotations, .35, endpoint_first=True)
    return node, solutions, rotations[index]


def test_current_official_shallow_grasp_finds_clear_compact_route():
    node, solutions, rotation = shallow_pick()
    route_attempts = []
    original = node._plan_carried_joint_route

    def record(start, goals, corners, **kwargs):
        result = original(start, goals, corners, **kwargs)
        if len(goals) == 20:
            route_attempts.append((result is not None, kwargs))
        return result

    node._plan_carried_joint_route = record
    node._plan_carried_return(FRONT, solutions[-1], solutions[1], rotation, .35)
    cache = node._cached_post_retreat_plan
    assert [passed for passed, _ in route_attempts] == [False, True]
    assert cache['compact_shoulder_progress_power'] == .8
    assert all(kwargs == {
        'post_retreat_shelf_front_x': pytest.approx(FRONT[0] + .25),
        'require_gravity_support': True,
    } for _, kwargs in route_attempts)
    assert cache['compact_radius'] < .45
    expected_terminal = manipulation.SUPPORTED_CARRY.copy()
    expected_terminal[0] = .35
    assert cache['terminal'] == pytest.approx(expected_terminal)
    compact = [q for q, phase in cache['legs'] if phase == 'compact_transport']
    assert len(compact) == 20
    first = cache['staging_terminal']
    max_x = -np.inf
    for goal in compact:
        for fraction in np.linspace(0., 1., node.carried_transition_samples):
            pose = node.chain.forward(first + (goal - first) * fraction)
            world = cache['attached_corners'] @ pose[:3, :3].T + pose[:3, 3]
            max_x = max(max_x, float(world[:, 0].max()))
        first = goal
    # More than 20mm clearance to the UNCHANGED shelf margin after compaction.
    assert cache['shelf_front_x'] - node.carried_shelf_margin - max_x > .020


def test_all_rejected_compact_candidates_still_fail_closed():
    node, solutions, rotation = shallow_pick()
    original = node._plan_carried_joint_route
    attempts = []

    def reject_compact(start, goals, corners, **kwargs):
        if len(goals) == 20:
            attempts.append((np.asarray(goals), kwargs))
            return None
        return original(start, goals, corners, **kwargs)

    node._plan_carried_joint_route = reject_compact
    with pytest.raises(RuntimeError, match='No payload-safe supported compact'):
        node._plan_carried_return(FRONT, solutions[-1], solutions[1], rotation, .35)
    assert len(attempts) == 3
    assert node._cached_post_retreat_plan is None
    expected_terminal = manipulation.SUPPORTED_CARRY.copy()
    expected_terminal[0] = .35
    for goals, kwargs in attempts:
        assert goals[-1] == pytest.approx(expected_terminal)
        assert kwargs['require_gravity_support'] is True
        assert kwargs['post_retreat_shelf_front_x'] == pytest.approx(FRONT[0] + .25)


@pytest.mark.parametrize('power', [0., .49, 1.01, float('nan'), float('inf')])
def test_shoulder_progress_is_bounded(power):
    node = object.__new__(manipulation.ManipulationNode)
    with pytest.raises(ValueError, match='shoulder progress power'):
        node._supported_compact_goals(manipulation.SUPPORTED_CARRY,
                                      shoulder_progress_power=power)
