#!/usr/bin/env python3
"""Temporary search for a fully shelf-exterior lower-grasp cradle."""

import importlib.util
import itertools

import numpy as np

from erc_phase1_solution.kinematics import pose_matrix


spec = importlib.util.spec_from_file_location(
    'test_shutdown_outside_low',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

FRONT = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)
SHELF_FRONT_X = FRONT[0] - 0.065

for offset in (-0.055, -0.060, -0.065, -0.070):
    node = helpers._official_manipulation_planner()
    node.top_row_grasp_vertical_offset = offset
    _, _, _, solutions, _, _, _ = helpers._top_row_pick_plan(node, FRONT)
    grasp = np.asarray(solutions[-1], dtype=float)
    start = np.asarray(solutions[1], dtype=float)
    corners = node._attached_book_corners(FRONT, grasp)
    start_pose = node.chain.forward(start)
    passes = []
    for dx, dy, dz in itertools.product(
        (-0.04, -0.06, -0.08, -0.10, -0.12),
        (-0.10, -0.05, 0.0, 0.05, 0.10),
        (0.0, 0.02, 0.04, 0.06),
    ):
        target = start_pose[:3, 3] + np.asarray([dx, dy, dz])
        solved, score = node.chain.solve(
            pose_matrix(target, start_pose[:3, :3]),
            [start],
            position_tolerance=node.position_tolerance,
            orientation_tolerance=node.orientation_tolerance,
            max_iterations=280,
            fixed_positions={'torso_lift_joint': float(start[0])},
        )
        if solved is None:
            continue
        stage_route = node._plan_carried_joint_route(
            start,
            [solved],
            corners,
        )
        if not stage_route:
            continue
        staged = np.asarray(stage_route[-1], dtype=float)
        transform = node.chain.forward(staged)
        world = corners @ transform[:3, :3].T + transform[:3, 3]
        maximum_x = float(np.max(world[:, 0]))
        if maximum_x > SHELF_FRONT_X - 0.010:
            continue
        cradled = staged.copy()
        cradled[-1] -= 1.10
        roll_route = node._plan_carried_joint_route(
            staged,
            [cradled],
            corners,
        )
        if not roll_route:
            continue
        terminal = np.asarray(roll_route[-1], dtype=float)
        if not node._gravity_supported_transition_is_safe(terminal, terminal):
            continue
        radius = node._carried_navigation_radius(terminal, corners)
        passes.append(
            (
                radius,
                float(score),
                dx,
                dy,
                dz,
                len(stage_route),
                len(roll_route),
                maximum_x,
                staged,
                terminal,
            )
        )
    print('OFFSET', offset, 'TOTAL', len(passes), flush=True)
    for candidate in sorted(passes)[:10]:
        (
            radius,
            score,
            dx,
            dy,
            dz,
            stage_legs,
            roll_legs,
            maximum_x,
            staged,
            terminal,
        ) = candidate
        print(
            'PASS',
            'offset', offset,
            'dx', dx,
            'dy', dy,
            'dz', dz,
            'score', score,
            'stage_legs', stage_legs,
            'roll_legs', roll_legs,
            'maximum_x', maximum_x,
            'radius', radius,
            'staged', staged.tolist(),
            'terminal', terminal.tolist(),
            flush=True,
        )
