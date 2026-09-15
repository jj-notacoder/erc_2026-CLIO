"""The selected precision pinch paths reach every active row at fixed torso.

These are offline kinematics and link-origin path checks; they do not establish
mesh clearance, physical grip retention, or delivery for any row.
"""
import numpy as np
import pytest

from erc_phase1_solution.motion_profiles import shelf_pinch_orientations


@pytest.mark.parametrize('row,height', [(1, 1.584), (2, 1.254), (3, .934), (4, .604)])
def test_runtime_fixed_torso_precision_pinch_path_reaches_each_row(row, height):
    pytest.importorskip('rclpy')
    from test_shutdown import _official_manipulation_planner
    node = _official_manipulation_planner()
    node.joints['head_2_joint'] = [.2, 0., -.4, -.7][row-1]
    top = row == 1
    front = np.array([.670, -.056, height])
    grasp = front + [.025, -.001 if top else 0., -.015 if top else 0.]
    pregrasp = grasp - [.14, 0., 0.]
    clearance = pregrasp.copy(); clearance[0] = .46 if top else .45
    positions = [clearance, *node._interpolate_positions(clearance, pregrasp, .06),
                 *node._interpolate_positions(pregrasp, grasp, .06)]
    rotations = shelf_pinch_orientations(grasp[2])
    solutions, index, _, _ = node._solve_cartesian_path(
        positions, rotations, .35, endpoint_first=top,
        position_tolerance=.0005, orientation_tolerance=.01)
    assert len(solutions) == len(positions)
    for q, point in zip(solutions, positions):
        pose = node.chain.forward(q)
        assert q[0] == .35
        assert np.linalg.norm(pose[:3, 3]-point) <= .0005
        error = node.chain.pose_error(pose, np.block([
            [rotations[index], np.array(point).reshape(3, 1)],
            [np.zeros((1, 3)), np.ones((1, 1))]]))
        assert np.linalg.norm(error[3:]) <= .01
