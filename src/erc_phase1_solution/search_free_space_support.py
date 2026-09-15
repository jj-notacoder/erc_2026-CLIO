#!/usr/bin/env python3
"""Temporary exact-ish search for a post-extraction bottom catch."""

import importlib.util
import itertools

import numpy as np

from erc_phase1_solution.kinematics import (
    interpolate_joint_waypoints,
    oriented_box_intersects_triangles,
    pose_matrix,
)


spec = importlib.util.spec_from_file_location(
    'test_shutdown_free_support',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

FRONT = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


node = helpers._official_manipulation_planner()
_, _, _, solutions, _, _, _ = helpers._top_row_pick_plan(node, FRONT)
grasp = np.asarray(solutions[-1], dtype=float)
start = np.asarray(solutions[1], dtype=float)
saved_padding = node.carried_book_padding
node.carried_book_padding = 0.0
physical_local = node._attached_book_corners(FRONT, grasp)
node.carried_book_padding = saved_padding
start_pose = node.chain.forward(start)
book = physical_local @ start_pose[:3, :3].T + start_pose[:3, 3]
book_min = np.min(book, axis=0)
book_max = np.max(book, axis=0)
print(
    'START', start.tolist(), start_pose[:3, 3].tolist(),
    'BOOK', book_min.tolist(), book_max.tolist(),
    flush=True,
)

passes = []
for roll, dx, dy, dz in itertools.product(
    (-1.10, -1.30, -1.55, 1.10, 1.30, 1.55),
    (-0.12, -0.08, -0.04, 0.0),
    (-0.04, 0.0, 0.04),
    (-0.14, -0.12, -0.10, -0.08, -0.06),
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
        max_iterations=240,
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
                node.chain.pose_error(node.chain.forward(first), node.chain.forward(last))[3:]
            ),
            maximum_orientation_step=node.carried_orientation_step_limit,
        )
    )
    safe = True
    previous = start
    for waypoint in waypoints:
        for fraction in np.linspace(0.0, 1.0, 21):
            q = previous + fraction * (waypoint - previous)
            if node._robot_self_collision(q) is not None:
                safe = False
                break
            surfaces = node._world_collision_surfaces(q)
            for link, triangles in surfaces.items():
                if link.startswith('gripper_left_'):
                    continue
                if oriented_box_intersects_triangles(
                    book,
                    triangles,
                    closed_surface=link in node._watertight_collision_links(),
                ):
                    safe = False
                    break
            if not safe:
                break
        if not safe:
            break
        previous = waypoint
    if not safe:
        continue

    surfaces = node._world_collision_surfaces(solved)
    lower = (
        'gripper_left_fingertip_left_link'
        if roll < 0.0
        else 'gripper_left_fingertip_right_link'
    )
    triangles = surfaces[lower]
    tip_min = np.min(triangles, axis=(0, 1))
    tip_max = np.max(triangles, axis=(0, 1))
    xy_overlap = bool(
        tip_max[0] >= book_min[0]
        and tip_min[0] <= book_max[0]
        and tip_max[1] >= book_min[1]
        and tip_min[1] <= book_max[1]
    )
    bottom_gap = float(book_min[2] - tip_max[2])
    intersects = oriented_box_intersects_triangles(
        book,
        triangles,
        closed_surface=lower in node._watertight_collision_links(),
    )
    if not xy_overlap or not -0.012 <= bottom_gap <= 0.015:
        continue
    achieved = node.chain.forward(solved)
    candidate = (
        abs(bottom_gap),
        float(score),
        roll,
        dx,
        dy,
        dz,
        len(waypoints),
        float(achieved[2, 1]),
        bottom_gap,
        bool(intersects),
        tip_min,
        tip_max,
        solved,
        achieved[:3, 3],
    )
    passes.append(candidate)
    print(
        'PASS', 'roll', roll, 'dx', dx, 'dy', dy, 'dz', dz,
        'gap', bottom_gap, 'intersects', bool(intersects),
        'legs', len(waypoints), 'jaw', float(achieved[2, 1]),
        'tip', tip_min.tolist(), tip_max.tolist(),
        'origin', achieved[:3, 3].tolist(), 'q', solved.tolist(),
        flush=True,
    )

print('TOTAL', len(passes), flush=True)
