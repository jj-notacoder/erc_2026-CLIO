import importlib.util

import numpy as np


spec = importlib.util.spec_from_file_location(
    'test_shutdown',
    '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
node = module._official_manipulation_planner()
front = np.asarray(
    [0.6937779125006432, -0.05493238067016105, 1.5841850667634312]
)
grasp, positions, rotations, solutions, orientation_index, score, transition = (
    module._top_row_pick_plan(node, front)
)
np.set_printoptions(precision=12, suppress=True)
corners = node._attached_book_corners(front, solutions[-1])
for lowering_distance in (0.12, 0.18):
    node.retreat_distance = lowering_distance
    lowering = node._solve_fixed_orientation_lowering(
        solutions[1],
        rotations[orientation_index],
        0.35,
    )
    for retraction in (-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.02):
        for roll in (-0.5 * np.pi,):
            node.carried_cradle_retraction = retraction
            node.carried_cradle_roll = roll
            try:
                retracted, supported, staged = node._solve_carried_cradle(
                    lowering[-1],
                    rotations[orientation_index],
                    0.35,
                )
            except RuntimeError as error:
                print('FAIL', lowering_distance, retraction, roll, error)
                continue
            minimum_aabb_separation = float('inf')
            collision = None
            minimum_book_z = float('inf')
            maximum_payload_x = -float('inf')
            for fraction in np.linspace(0.0, 1.0, 101):
                solution = retracted + (supported - retracted) * fraction
                transform = node.chain.forward(solution)
                world_corners = corners @ transform[:3, :3].T + transform[:3, 3]
                triangles = node._world_collision_surfaces(solution)[
                    'arm_left_5_link'
                ]
                book_min = np.min(world_corners, axis=0)
                book_max = np.max(world_corners, axis=0)
                arm_min = np.min(triangles, axis=(0, 1))
                arm_max = np.max(triangles, axis=(0, 1))
                separation = np.maximum(
                    np.maximum(arm_min - book_max, book_min - arm_max),
                    0,
                )
                minimum_aabb_separation = min(
                    minimum_aabb_separation,
                    float(np.linalg.norm(separation)),
                )
                minimum_book_z = min(minimum_book_z, float(book_min[2]))
                maximum_payload_x = max(maximum_payload_x, float(book_max[0]))
                collision = collision or node._carried_robot_collision(
                    solution,
                    world_corners,
                )
            print(
                'OK',
                'down',
                lowering_distance,
                'retract',
                retraction,
                'roll',
                round(float(roll), 4),
                'origin',
                node.chain.forward(retracted)[:3, 3],
                'jawz',
                node.chain.forward(supported)[2, 1],
                'arm5_aabb_gap',
                minimum_aabb_separation,
                'book_min_z',
                minimum_book_z,
                'payload_max_x',
                maximum_payload_x,
                'collision',
                collision,
            )
