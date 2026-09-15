"""A 20mm book needs accurate pick alignment without changing carry IK."""

import numpy as np
import pytest

pytest.importorskip('rclpy')
from test_shutdown import _official_manipulation_planner
from erc_phase1_solution import manipulation_node as manipulation


FRONT = np.asarray([.6922532156805421, -.06005257028383863, 1.5845481738629341])


def approach_input(node):
    grasp = FRONT + [.025, -.001, -.015]
    pregrasp = grasp - [.14, 0., 0.]
    clearance = pregrasp.copy()
    clearance[0] = .46
    return ([clearance,
             *node._interpolate_positions(clearance, pregrasp, .06),
             *node._interpolate_positions(pregrasp, grasp, .06)],
            manipulation.shelf_pinch_orientations(grasp[2]))


def test_pick_overrides_refine_same_endpoint_branch_and_leave_carry_defaults_alone():
    node = _official_manipulation_planner()
    positions, rotations = approach_input(node)
    coarse, coarse_index, _, _ = node._solve_cartesian_path(
        positions, rotations, .35, endpoint_first=True)
    coarse_accuracy = node._pick_path_accuracy(positions, rotations[coarse_index], coarse)
    assert coarse_accuracy['grasp_position_error_base_m'][1] > .0015
    assert coarse_accuracy['grasp_orientation_error_rad'] > .08

    precise, precise_index, _, _ = node._solve_cartesian_path(
        positions, rotations, .35, endpoint_first=True,
        position_tolerance=.0005, orientation_tolerance=.01)
    accuracy = node._pick_path_accuracy(positions, rotations[precise_index], precise)
    assert precise_index == coarse_index == 0
    assert np.max(np.abs(precise[-1] - coarse[-1])) < .15
    assert accuracy['approach_max_position_error_m'] <= .0005
    assert accuracy['approach_max_orientation_error_rad'] <= .01
    assert abs(accuracy['grasp_position_error_base_m'][1]) < .0005
    assert accuracy['actual_grasp_position'] == pytest.approx(node.chain.forward(precise[-1])[:3, 3])
    assert node.position_tolerance == .012
    assert node.orientation_tolerance == .10


@pytest.mark.parametrize('position,orientation', [
    (0., .01), (-.001, .01), (float('nan'), .01),
    (.0005, 0.), (.0005, -.01), (.0005, float('inf')),
])
def test_invalid_per_call_tolerances_reject_before_solver(position, orientation):
    node = object.__new__(manipulation.ManipulationNode)
    node.position_tolerance, node.orientation_tolerance = .012, .10
    with pytest.raises(ValueError, match='finite and positive'):
        node._solve_cartesian_path([[.7, 0., 1.5]], [np.eye(3)], .35,
                                  position_tolerance=position,
                                  orientation_tolerance=orientation)


def test_accuracy_diagnostics_reject_missing_path_and_nonfinite_fk():
    node = object.__new__(manipulation.ManipulationNode)
    with pytest.raises(RuntimeError, match='complete Cartesian path'):
        node._pick_path_accuracy([[.7, 0., 1.5]], np.eye(3), [])
    from types import SimpleNamespace
    node.chain = SimpleNamespace(forward=lambda q: np.full((4, 4), np.nan))
    with pytest.raises(RuntimeError, match='invalid FK'):
        node._pick_path_accuracy([[.7, 0., 1.5]], np.eye(3), [[0.]])
