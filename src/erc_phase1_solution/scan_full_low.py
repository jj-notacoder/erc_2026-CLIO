#!/usr/bin/env python3
"""Temporary scan for a lower grasp cradled only after full extraction."""

import importlib.util

import numpy as np

from erc_phase1_solution.kinematics import pose_matrix


spec = importlib.util.spec_from_file_location(
    'test_shutdown_full_low',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

FRONT = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)

for offset in (-0.055, -0.060, -0.065, -0.070):
    node = helpers._official_manipulation_planner()
    node.top_row_grasp_vertical_offset = offset
    try:
        _, _, _, solutions, _, _, _ = helpers._top_row_pick_plan(node, FRONT)
    except Exception as exc:
        print('SOLVE_FAIL', offset, repr(exc), flush=True)
        continue
    grasp = np.asarray(solutions[-1], dtype=float)
    fully_extracted = np.asarray(solutions[1], dtype=float)
    corners = node._attached_book_corners(FRONT, grasp)
    previous = grasp
    extraction_ok = True
    for solution in reversed(solutions[1:-1]):
        if not node._carried_robot_transition_is_safe(
            previous,
            solution,
            corners,
        ):
            extraction_ok = False
            break
        previous = np.asarray(solution, dtype=float)
    if not extraction_ok:
        print('EXTRACT_FAIL', offset, flush=True)
        continue
    for lift in (0.0, 0.010, 0.025):
        staged = fully_extracted.copy()
        if lift > 0.0:
            pose = node.chain.forward(fully_extracted)
            target = pose[:3, 3] + np.asarray([0.0, 0.0, lift])
            solved, _ = node.chain.solve(
                pose_matrix(target, pose[:3, :3]),
                [fully_extracted],
                position_tolerance=node.position_tolerance,
                orientation_tolerance=node.orientation_tolerance,
                max_iterations=240,
                fixed_positions={
                    'torso_lift_joint': float(fully_extracted[0])
                },
            )
            if solved is None:
                continue
            staged = np.asarray(solved, dtype=float)
        stage_route = node._plan_carried_joint_route(
            fully_extracted,
            [staged],
            corners,
            shelf_front_x=float(FRONT[0]),
        )
        if not stage_route and not np.allclose(staged, fully_extracted):
            continue
        for roll in (-1.10, -1.30, -1.55):
            cradled = staged.copy()
            cradled[-1] += roll
            roll_route = node._plan_carried_joint_route(
                staged,
                [cradled],
                corners,
                shelf_front_x=float(FRONT[0]),
            )
            if not roll_route:
                continue
            transform = node.chain.forward(roll_route[-1])
            world = corners @ transform[:3, :3].T + transform[:3, 3]
            print(
                'PASS',
                'offset', offset,
                'lift', lift,
                'roll', roll,
                'extraction_legs', len(solutions) - 2,
                'stage_legs', len(stage_route),
                'roll_legs', len(roll_route),
                'jaw', float(transform[2, 1]),
                'bounds', [
                    np.min(world, axis=0).tolist(),
                    np.max(world, axis=0).tolist(),
                ],
                'staged', staged.tolist(),
                'terminal', roll_route[-1].tolist(),
                flush=True,
            )
