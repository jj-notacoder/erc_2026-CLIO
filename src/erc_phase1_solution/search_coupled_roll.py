#!/usr/bin/env python3
"""Temporary coupled wrist-roll/rise search for a shelf-supported book."""

import importlib.util
import itertools

import numpy as np

from erc_phase1_solution.kinematics import (
    interpolate_joint_waypoints,
    pose_matrix,
)


spec = importlib.util.spec_from_file_location(
    'test_shutdown_coupled',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

FRONT = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)


def rotation_x(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=float,
    )


node = helpers._official_manipulation_planner()
node.top_row_grasp_vertical_offset = -0.065
_, _, _, solutions, _, _, _ = helpers._top_row_pick_plan(node, FRONT)
grasp = np.asarray(solutions[-1], dtype=float)
extracted = np.asarray(solutions[-2], dtype=float)
corners = node._attached_book_corners(FRONT, grasp)
extracted_pose = node.chain.forward(extracted)
lifted_target = extracted_pose[:3, 3] + np.asarray([0.0, 0.0, 0.025])
start, _ = node.chain.solve(
    pose_matrix(lifted_target, extracted_pose[:3, :3]),
    [extracted],
    position_tolerance=node.position_tolerance,
    orientation_tolerance=node.orientation_tolerance,
    max_iterations=240,
    fixed_positions={'torso_lift_joint': float(extracted[0])},
)
start = np.asarray(start, dtype=float)
start_pose = node.chain.forward(start)
stationary_book = corners @ start_pose[:3, :3].T + start_pose[:3, 3]
print('START', start.tolist(), start_pose[:3, 3].tolist(), flush=True)

passes = []
for roll, dx, dy, dz in itertools.product(
    (-1.10, -1.30, -1.55, 1.10, 1.30, 1.55),
    (-0.08, -0.04, 0.0, 0.04),
    (-0.08, -0.04, 0.0, 0.04, 0.08),
    (0.04, 0.06, 0.08, 0.10, 0.12, 0.14),
):
    target_position = start_pose[:3, 3] + np.asarray([dx, dy, dz])
    target_rotation = start_pose[:3, :3] @ rotation_x(roll)
    guess = start.copy()
    guess[-1] += roll
    solved, score = node.chain.solve(
        pose_matrix(target_position, target_rotation),
        [guess, start],
        position_tolerance=0.006,
        orientation_tolerance=0.08,
        max_iterations=320,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solved is None:
        continue
    solved = np.asarray(solved, dtype=float)
    waypoints = list(
        interpolate_joint_waypoints(
            start,
            solved,
            first_arm_index=1,
            maximum_joint_step=node.cartesian_joint_step,
            orientation_distance=lambda first, last: np.linalg.norm(
                node.chain.pose_error(
                    node.chain.forward(first),
                    node.chain.forward(last),
                )[3:]
            ),
            maximum_orientation_step=node.carried_orientation_step_limit,
        )
    )
    previous = start
    safe = True
    minimum_jaw = 1.0
    for waypoint in waypoints:
        for fraction in np.linspace(0.0, 1.0, 21):
            q = previous + (waypoint - previous) * fraction
            if (
                node._robot_self_collision(q) is not None
                or node._carried_robot_collision(q, stationary_book) is not None
            ):
                safe = False
                break
            minimum_jaw = min(
                minimum_jaw,
                abs(float(node.chain.forward(q)[2, 1])),
            )
        if not safe:
            break
        previous = waypoint
    if not safe:
        continue
    achieved = node.chain.forward(solved)
    jaw = float(achieved[2, 1])
    if abs(jaw) < 0.75:
        continue
    passes.append(
        (
            -float(achieved[2, 3]),
            float(score),
            roll,
            dx,
            dy,
            dz,
            len(waypoints),
            jaw,
            solved,
            achieved[:3, 3],
        )
    )

print('TOTAL', len(passes), flush=True)
for candidate in sorted(passes)[:30]:
    _, score, roll, dx, dy, dz, legs, jaw, solved, achieved = candidate
    print(
        'PASS',
        'roll', roll,
        'dx', dx,
        'dy', dy,
        'dz', dz,
        'score', score,
        'legs', legs,
        'jaw', jaw,
        'origin', achieved.tolist(),
        'solution', solved.tolist(),
        flush=True,
    )
