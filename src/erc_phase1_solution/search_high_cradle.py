#!/usr/bin/env python3
"""Temporary offline scan for an elevated pre-retreat cradle; remove later."""

import importlib.util
import itertools

import numpy as np

from erc_phase1_solution.kinematics import pose_matrix


spec = importlib.util.spec_from_file_location(
    'test_shutdown_high',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

front = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)
node = helpers._official_manipulation_planner()
node.top_row_grasp_vertical_offset = -0.015
_, _, _, solutions, _, _, _ = helpers._top_row_pick_plan(node, front)
start = np.asarray(solutions[1], dtype=float)
attached = node._attached_book_corners(front, solutions[-1])
start_pose = node.chain.forward(start)
shelf_front = float(front[0])
print('START', start.tolist(), start_pose[:3, 3].tolist(), flush=True)

passes = []
for dx, dy, dz in itertools.product(
    (-0.06, -0.03, 0.0, 0.03),
    (-0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16),
    (0.10, 0.12, 0.14, 0.16, 0.18),
):
    target = start_pose[:3, 3] + np.asarray([dx, dy, dz])
    staged, score = node.chain.solve(
        pose_matrix(target, start_pose[:3, :3]),
        [start],
        position_tolerance=node.position_tolerance,
        orientation_tolerance=node.orientation_tolerance,
        max_iterations=280,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if staged is None:
        continue
    staged_route = node._plan_carried_joint_route(
        start,
        [staged],
        attached,
        shelf_front_x=shelf_front,
    )
    if not staged_route:
        continue
    staged = np.asarray(staged_route[-1], dtype=float)
    cradled = staged.copy()
    cradled[-1] += node.carried_cradle_roll
    roll_route = node._plan_carried_joint_route(
        staged,
        [cradled],
        attached,
        shelf_front_x=shelf_front,
    )
    if not roll_route:
        continue
    terminal = np.asarray(roll_route[-1], dtype=float)
    if not node._gravity_supported_transition_is_safe(terminal, terminal):
        continue
    transform = node.chain.forward(terminal)
    world = attached @ transform[:3, :3].T + transform[:3, 3]
    radius = node._carried_navigation_radius(terminal, attached)
    passes.append((dx, dy, dz, float(score), staged, terminal, radius))
    print(
        'PASS', dx, dy, dz,
        'score', float(score),
        'stage_legs', len(staged_route),
        'roll_legs', len(roll_route),
        'origin', transform[:3, 3].tolist(),
        'support_center_z_estimate', float(transform[2, 3] - 0.13),
        'payload_bounds', [
            np.min(world, axis=0).tolist(),
            np.max(world, axis=0).tolist(),
        ],
        'radius', radius,
        'staged', staged.tolist(),
        'terminal', terminal.tolist(),
        flush=True,
    )

print('TOTAL', len(passes), flush=True)
