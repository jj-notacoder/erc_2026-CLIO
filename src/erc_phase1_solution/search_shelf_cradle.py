#!/usr/bin/env python3
"""Temporary offline search for a shelf-side supported-cradle pose."""

import importlib.util
import itertools

import numpy as np

from erc_phase1_solution.kinematics import pose_matrix


spec = importlib.util.spec_from_file_location(
    'test_shutdown_search',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

front = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)
node = helpers._official_manipulation_planner()
node.top_row_grasp_vertical_offset = 0.0
_, _, rotations, solutions, orientation_index, _, _ = helpers._top_row_pick_plan(
    node, front
)
start = np.asarray(solutions[1], dtype=float)
attached = node._attached_book_corners(front, solutions[-1])
shelf_front = float(front[0])
start_pose = node.chain.forward(start)
print(
    'START',
    'q', start.tolist(),
    'xyz', start_pose[:3, 3].tolist(),
    'payload_max_x', float(
        np.max(
            (
                attached @ start_pose[:3, :3].T
                + start_pose[:3, 3]
            )[:, 0]
        )
    ),
    'shelf_limit', shelf_front - node.carried_shelf_margin,
    flush=True,
)

results = []
for dx, dy, dz in itertools.product(
    (-0.06, -0.03, 0.0, 0.03, 0.06),
    (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12),
    (0.0, 0.03, 0.06, 0.09, 0.12),
):
    target_position = start_pose[:3, 3] + np.asarray([dx, dy, dz])
    candidate, score = node.chain.solve(
        pose_matrix(target_position, start_pose[:3, :3]),
        [start],
        position_tolerance=node.position_tolerance,
        orientation_tolerance=node.orientation_tolerance,
        max_iterations=240,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if candidate is None:
        continue
    translated = node._plan_carried_joint_route(
        start,
        [candidate],
        attached,
        shelf_front_x=shelf_front,
    )
    if not translated:
        continue
    candidate = np.asarray(translated[-1], dtype=float)
    cradle = node._solve_carried_cradle(candidate)
    rolled = node._plan_carried_joint_route(
        candidate,
        [cradle],
        attached,
        shelf_front_x=shelf_front,
    )
    if not rolled:
        continue
    final = np.asarray(rolled[-1], dtype=float)
    if not node._gravity_supported_transition_is_safe(final, final):
        continue
    final_pose = node.chain.forward(final)
    world_corners = attached @ final_pose[:3, :3].T + final_pose[:3, 3]
    collision = node._carried_robot_collision(final, world_corners)
    results.append(
        (
            abs(dx) + abs(dy) + abs(dz),
            dx,
            dy,
            dz,
            len(translated),
            len(rolled),
            float(score),
            float(np.max(world_corners[:, 0])),
            candidate,
            final,
            collision,
        )
    )
    print(
        'PASS', dx, dy, dz,
        'translation_legs', len(translated),
        'roll_legs', len(rolled),
        'score', score,
        'payload_max_x', float(np.max(world_corners[:, 0])),
        'collision', collision,
        'candidate', candidate.tolist(),
        'final', final.tolist(),
        flush=True,
    )

print('TOTAL', len(results), flush=True)
