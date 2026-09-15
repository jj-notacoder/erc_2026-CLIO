#!/usr/bin/env python3
"""Temporary padded-aperture scan for the lower one-arm cradle."""

import importlib.util

import numpy as np

from erc_phase1_solution.kinematics import pose_matrix


spec = importlib.util.spec_from_file_location(
    'test_shutdown_low',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

FRONT = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)
FLOOR = 1.45186
CEILING = 1.75186


def aperture_margin(node, start, end, corners, aperture_corners):
    first = np.asarray(start, dtype=float)
    last = np.asarray(end, dtype=float)
    if not node._carried_robot_transition_is_safe(first, last, corners):
        print('APERTURE_FAIL robot', first.tolist(), last.tolist(), flush=True)
        return None
    minimum_floor = float('inf')
    minimum_ceiling = float('inf')
    for fraction in np.linspace(0.0, 1.0, 61):
        q = first + (last - first) * fraction
        transform = node.chain.forward(q)
        world = (
            aperture_corners @ transform[:3, :3].T
            + transform[:3, 3]
        )
        minimum_floor = min(minimum_floor, float(np.min(world[:, 2]) - FLOOR))
        minimum_ceiling = min(
            minimum_ceiling,
            float(CEILING - np.max(world[:, 2])),
        )
        if (
            float(np.min(world[:, 0])) < 0.40
            or float(np.max(world[:, 0])) > FRONT[0] + 0.21
        ):
            print(
                'APERTURE_FAIL bounds', fraction,
                np.min(world, axis=0).tolist(),
                np.max(world, axis=0).tolist(),
                flush=True,
            )
            return None
    return minimum_floor, minimum_ceiling


for offset in (-0.065,):
    for lift in (0.025,):
        node = helpers._official_manipulation_planner()
        node.top_row_grasp_vertical_offset = offset
        try:
            _, _, _, solutions, _, _, _ = helpers._top_row_pick_plan(
                node,
                FRONT,
            )
        except Exception:
            continue
        grasp = np.asarray(solutions[-1], dtype=float)
        extracted = np.asarray(solutions[-2], dtype=float)
        corners = node._attached_book_corners(FRONT, grasp)
        saved_padding = node.carried_book_padding
        node.carried_book_padding = 0.0
        aperture_corners = node._attached_book_corners(FRONT, grasp)
        node.carried_book_padding = saved_padding
        extracted_margin = aperture_margin(
            node, grasp, extracted, corners, aperture_corners
        )
        pose = node.chain.forward(extracted)
        target = pose[:3, 3] + np.asarray([0.0, 0.0, lift])
        lifted, _ = node.chain.solve(
            pose_matrix(target, pose[:3, :3]),
            [extracted],
            position_tolerance=node.position_tolerance,
            orientation_tolerance=node.orientation_tolerance,
            max_iterations=240,
            fixed_positions={'torso_lift_joint': float(extracted[0])},
        )
        if lifted is None:
            continue
        lifted = np.asarray(lifted, dtype=float)
        lift_margin = aperture_margin(
            node, extracted, lifted, corners, aperture_corners
        )
        if extracted_margin is None or lift_margin is None:
            continue
        print(
            'LIFT_PASS', offset, lift,
            'extract', extracted_margin,
            'lift_margin', lift_margin,
            flush=True,
        )
        cradled = lifted.copy()
        cradled[-1] -= 1.55
        roll = node._plan_carried_joint_route(lifted, [cradled], corners)
        if not roll:
            continue
        previous = lifted
        roll_margins = []
        for waypoint in roll:
            margin = aperture_margin(
                node, previous, waypoint, corners, aperture_corners
            )
            if margin is None:
                break
            roll_margins.append(margin)
            previous = waypoint
        if len(roll_margins) != len(roll):
            continue
        print(
            'ROLL_PASS', offset, lift,
            'margin', (
                min(value[0] for value in roll_margins),
                min(value[1] for value in roll_margins),
            ),
            flush=True,
        )
        supported_pose = node.chain.forward(roll[-1])
        target_position = supported_pose[:3, 3].copy()
        target_position[0] = node.chain.forward(solutions[1])[0, 3]
        supported = []
        previous = np.asarray(roll[-1], dtype=float)
        for position in node._interpolate_positions(
            supported_pose[:3, 3],
            target_position,
            node.cartesian_step,
        ):
            solution, _ = node.chain.solve(
                pose_matrix(position, supported_pose[:3, :3]),
                [previous],
                position_tolerance=node.position_tolerance,
                orientation_tolerance=node.orientation_tolerance,
                max_iterations=240,
                fixed_positions={'torso_lift_joint': float(previous[0])},
            )
            if solution is None:
                supported = []
                break
            solution = np.asarray(solution, dtype=float)
            margin = aperture_margin(
                node, previous, solution, corners, aperture_corners
            )
            if (
                margin is None
                or not node._gravity_supported_transition_is_safe(
                    previous,
                    solution,
                )
            ):
                supported = []
                break
            supported.append((solution, margin))
            previous = solution
        if not supported:
            continue
        all_margins = [
            extracted_margin,
            lift_margin,
            *roll_margins,
            *(item[1] for item in supported),
        ]
        if any(margin is None for margin in all_margins):
            continue
        floor_margin = min(margin[0] for margin in all_margins)
        ceiling_margin = min(margin[1] for margin in all_margins)
        print(
            'PASS',
            'offset', offset,
            'lift', lift,
            'floor', floor_margin,
            'ceiling', ceiling_margin,
            'jaw', node.chain.forward(roll[-1])[2, 1],
            'roll_legs', len(roll),
            'supported_legs', len(supported),
            'lifted', lifted.tolist(),
            'terminal', supported[-1][0].tolist(),
            flush=True,
        )
