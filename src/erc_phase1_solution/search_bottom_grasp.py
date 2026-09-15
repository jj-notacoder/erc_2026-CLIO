#!/usr/bin/env python3
"""Temporary offline scan for a bottom-edge shelf cradle; remove before commit."""

import importlib.util

import numpy as np

from erc_phase1_solution.kinematics import pose_matrix


spec = importlib.util.spec_from_file_location(
    'test_shutdown_bottom',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

front = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)
floor_z = 1.45186
ceiling_z = 1.75186

for offset in (-0.070, -0.080, -0.090, -0.095, -0.100, -0.105):
    node = helpers._official_manipulation_planner()
    node.top_row_grasp_vertical_offset = offset
    try:
        _, _, _, solutions, _, score, transition = helpers._top_row_pick_plan(
            node, front
        )
    except Exception as exc:
        print('SOLVE_FAIL', offset, repr(exc), flush=True)
        continue
    grasp_solution = np.asarray(solutions[-1], dtype=float)
    extracted = np.asarray(solutions[-2], dtype=float)
    padded = node._attached_book_corners(front, grasp_solution)
    saved_padding = node.carried_book_padding
    node.carried_book_padding = 0.0
    physical = node._attached_book_corners(front, grasp_solution)
    node.carried_book_padding = saved_padding
    extracted_ok = node._carried_robot_transition_is_safe(
        grasp_solution, extracted, padded
    )
    extracted_pose = node.chain.forward(extracted)
    lift_target = extracted_pose[:3, 3] + np.asarray([0.0, 0.0, 0.025])
    lifted, lift_score = node.chain.solve(
        pose_matrix(lift_target, extracted_pose[:3, :3]),
        [extracted],
        position_tolerance=node.position_tolerance,
        orientation_tolerance=node.orientation_tolerance,
        max_iterations=240,
        fixed_positions={'torso_lift_joint': float(extracted[0])},
    )
    if lifted is None:
        print('LIFT_FAIL', offset, flush=True)
        continue
    lifted = np.asarray(lifted, dtype=float)
    cradled = lifted.copy()
    cradled[-1] += node.carried_cradle_roll
    roll = node._plan_carried_joint_route(
        lifted,
        [cradled],
        padded,
    )
    if not roll:
        print('ROLL_FAIL', offset, 'robot', flush=True)
        continue
    samples = []
    previous = lifted
    collision = None
    for waypoint in roll:
        for fraction in np.linspace(0.0, 1.0, 61):
            q = previous + (waypoint - previous) * fraction
            transform = node.chain.forward(q)
            physical_world = physical @ transform[:3, :3].T + transform[:3, 3]
            padded_world = padded @ transform[:3, :3].T + transform[:3, 3]
            samples.append((physical_world, padded_world, q))
            robot_hit = node._carried_robot_collision(q, padded_world)
            if robot_hit is not None:
                collision = robot_hit
                break
        if collision is not None:
            break
        previous = waypoint
    physical_min_z = min(float(np.min(value[0][:, 2])) for value in samples)
    physical_max_z = max(float(np.max(value[0][:, 2])) for value in samples)
    padded_min_z = min(float(np.min(value[1][:, 2])) for value in samples)
    padded_max_z = max(float(np.max(value[1][:, 2])) for value in samples)
    final_pose = node.chain.forward(roll[-1])
    local_y_vertical = float(final_pose[2, 1])
    print(
        'PASS',
        'offset', offset,
        'score', score,
        'transition', len(transition),
        'extract_ok', extracted_ok,
        'grasp', grasp_solution.tolist(),
        'extracted', extracted.tolist(),
        'lifted', lifted.tolist(),
        'lift_achieved', (
            node.chain.forward(lifted)[:3, 3] - extracted_pose[:3, 3]
        ).tolist(),
        'roll_legs', len(roll),
        'cradled', roll[-1].tolist(),
        'jaw_z', local_y_vertical,
        'robot_collision', collision,
        'physical_z', [physical_min_z, physical_max_z],
        'physical_aperture_margin', [
            physical_min_z - floor_z,
            ceiling_z - physical_max_z,
        ],
        'padded_aperture_margin', [
            padded_min_z - floor_z,
            ceiling_z - padded_max_z,
        ],
        'lift_score', lift_score,
        flush=True,
    )
