#!/usr/bin/env python3
"""Temporary Gazebo-link mesh measurement for the edge regrasp."""

import subprocess
import re

import numpy as np

from erc_phase1_solution.kinematics import load_stl_triangles
from live_retreat_probe import _entity_pose, quaternion_matrix


text = subprocess.run(
    ['gz', 'topic', '-e', '-t', '/world/erc_world/dynamic_pose/info', '-n', '1'],
    check=True,
    capture_output=True,
    text=True,
    timeout=5.0,
).stdout
base_p, base_q = _entity_pose(text, 'tiago_pro')
base_r = quaternion_matrix(base_q)
book_p, book_q = _entity_pose(text, 'book_col_3_row_2_red')
half = np.asarray([.125, .015, .080])
signs = np.asarray(
    [(x, y, z) for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)]
)
book = (signs * half) @ quaternion_matrix(book_q).T + book_p
print('BASE', base_p.tolist(), base_q.tolist())
print('BOOK', np.min(book, axis=0).tolist(), np.max(book, axis=0).tolist())
for frame_name in (
    'gripper_left_base_link',
    'gripper_left_grasping_link',
    'arm_left_7_link',
):
    try:
        frame_p, frame_q = _entity_pose(text, frame_name)
    except RuntimeError:
        continue
    frame_world_p = base_r @ frame_p + base_p
    frame_world_r = base_r @ quaternion_matrix(frame_q)
    print(
        frame_name,
        'origin_local', frame_p.tolist(),
        'origin_world', frame_world_p.tolist(),
        'rotation_world', frame_world_r.tolist(),
    )
print(
    'AVAILABLE_GRIPPER_LINKS',
    sorted(set(re.findall(r'name: "([^"]*gripper_left[^"]*)"', text))),
)
root = '/opt/erc_ws/src/pal_pro_gripper/pal_pro_gripper_description/meshes'
for name, mesh in (
    ('gripper_left_inner_finger_left_link', 'inner_finger.stl'),
    ('gripper_left_outer_finger_left_link', 'outer_finger.stl'),
    ('gripper_left_fingertip_left_link', 'fingertip.stl'),
    ('gripper_left_inner_finger_right_link', 'inner_finger.stl'),
    ('gripper_left_outer_finger_right_link', 'outer_finger.stl'),
    ('gripper_left_fingertip_right_link', 'fingertip.stl'),
):
    local_p, local_q = _entity_pose(text, name)
    triangles = load_stl_triangles(f'{root}/{mesh}')
    in_model = triangles @ quaternion_matrix(local_q).T + local_p
    world = in_model @ base_r.T + base_p
    first = world[:, 1] - world[:, 0]
    second = world[:, 2] - world[:, 0]
    normals = np.cross(first, second)
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(lengths[:, None], 1e-12)
    upward = world[normals[:, 2] > 0.70]
    leading_upward = upward[
        np.max(upward[:, :, 0], axis=1)
        >= np.max(world[:, :, 0]) - 0.008
    ] if len(upward) else upward
    print(
        name,
        'origin_local', local_p.tolist(),
        'bounds', np.min(world, axis=(0, 1)).tolist(), np.max(world, axis=(0, 1)).tolist(),
        'upward', (
            None
            if len(upward) == 0
            else [
                np.min(upward, axis=(0, 1)).tolist(),
                np.max(upward, axis=(0, 1)).tolist(),
            ]
        ),
        'leading_upward', (
            None
            if len(leading_upward) == 0
            else [
                np.min(leading_upward, axis=(0, 1)).tolist(),
                np.max(leading_upward, axis=(0, 1)).tolist(),
            ]
        ),
    )
