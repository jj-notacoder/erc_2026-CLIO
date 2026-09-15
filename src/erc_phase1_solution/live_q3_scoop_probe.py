#!/usr/bin/env python3
"""Temporary live q3 shelf-edge scoop calibration; remove before commit."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import (
    BASE_POSITION,
    BASE_YAW,
    BOOK,
    PickProbeNode,
    emit,
    set_model_pose,
)


Q_CLEAR = np.asarray(
    [
        0.35,
        0.6891889459,
        0.9450918793,
        -0.0670020205,
        -1.6019341416,
        0.2860138745,
        1.5017476599,
        -0.4766783535,
    ],
    dtype=float,
)
Q_DOWN = np.asarray(
    [
        0.35,
        0.8230698857,
        0.5902873546,
        0.0128847972,
        -2.0564081504,
        0.4114116760,
        1.6423975834,
        -0.5829860184,
    ],
    dtype=float,
)
Q_ROLL = Q_DOWN.copy()
Q_ROLL[-1] -= 1.55


class ScoopProbeNode(PickProbeNode):
    def _loaded_clearance_index(self, top_row, solutions):
        if top_row:
            return 3
        return super()._loaded_clearance_index(top_row, solutions)


def linear_route(node, start, delta, step):
    start = np.asarray(start, dtype=float)
    start_pose = node.chain.forward(start)
    rotation = start_pose[:3, :3]
    origin = start_pose[:3, 3]
    distance = float(np.linalg.norm(delta))
    count = max(1, int(math.ceil(distance / float(step))))
    previous = start
    route = []
    for fraction in np.linspace(1.0 / count, 1.0, count):
        target = origin + np.asarray(delta, dtype=float) * fraction
        solved, _ = node.chain.solve(
            pose_matrix(target, rotation),
            [previous],
            position_tolerance=0.002,
            orientation_tolerance=0.025,
            max_iterations=320,
            fixed_positions={'torso_lift_joint': float(start[0])},
        )
        if solved is None:
            raise RuntimeError(f'scoop IK failed at {target.tolist()}')
        solved = np.asarray(solved, dtype=float)
        if node._robot_self_collision(solved) is not None:
            raise RuntimeError('scoop route has a robot self-collision')
        route.append(solved)
        previous = solved
    return route


def move_route(node, route, phase, duration=0.55):
    for index, solution in enumerate(route):
        if not node._move_arm_solution(solution, duration):
            raise RuntimeError(f'{phase} failed at leg {index}')
        if not node._wait_sim_duration(0.05):
            raise RuntimeError(f'{phase} dwell failed at leg {index}')


def main() -> None:
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ScoopProbeNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 30.0
        while (
            len(node.joints) < 8
            or nav.pose is None
            or nav.last_front_scan_time is None
            or nav.last_rear_scan_time is None
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8 or nav.pose is None:
            raise RuntimeError('live robot state unavailable')

        node.gripper_transport_lock = 0.029
        node.top_row_grasp_vertical_offset = -0.015
        node.skip_return_preflight = True
        if not node._stow():
            raise RuntimeError('stow failed')
        base_quaternion = np.asarray(
            [0.0, 0.0, math.sin(BASE_YAW / 2.0), math.cos(BASE_YAW / 2.0)]
        )
        set_model_pose('tiago_pro', BASE_POSITION, base_quaternion)
        if not node._wait_sim_duration(0.40):
            raise RuntimeError('base settle failed')
        if not node._pick():
            raise RuntimeError('q3 shelf pick failed')

        held = emit('q3_held', node, nav, BOOK, None)
        _, _, held_minimum, held_maximum = physical_book_bounds()
        shelf_front_x = 2.755012
        held_center_x = 0.5 * float(held_minimum[0] + held_maximum[0])
        if not (
            float(held_minimum[0]) < shelf_front_x - 0.030
            and held_center_x > shelf_front_x + 0.020
        ):
            raise RuntimeError('q3 did not establish a stable supported overhang')

        if not node._open_gripper():
            raise RuntimeError('supported q3 release failed')
        # The normal release helper deliberately clears identity.  This
        # diagnostic knows the seeded model exactly and restores it solely so
        # the later fingertip contact sample can be attributed to that book.
        with node._lock:
            node._target_book_model = BOOK
        node.track_unexpected_contacts = True
        if not node._wait_sim_duration(0.50):
            raise RuntimeError('supported release settle failed')
        released = emit('q3_released', node, nav, BOOK, held)
        _, _, released_minimum, released_maximum = physical_book_bounds()
        released_center_x = 0.5 * float(
            released_minimum[0] + released_maximum[0]
        )
        if not (
            float(released_minimum[0]) < shelf_front_x - 0.030
            and released_center_x > shelf_front_x + 0.020
            and abs(float(released_minimum[2] - held_minimum[2])) < 0.010
        ):
            raise RuntimeError('released book did not remain shelf-supported')

        if not node._move_arm_solution(Q_CLEAR, 0.80):
            raise RuntimeError('unloaded clear move failed')
        if not node._command_gripper(0.018):
            raise RuntimeError('free-space hook close failed')
        if not node._move_arm_solution(Q_DOWN, 1.20):
            raise RuntimeError('outside-lip descent failed')
        if not node._move_arm_solution(Q_ROLL, 1.20):
            raise RuntimeError('outside-lip roll failed')

        # Shift toward the book centreline while advancing.  The live q3
        # bounds show this gives roughly 29 mm overlap across the 31 mm book
        # thickness, instead of catching only its y-edge.
        inward = linear_route(node, Q_ROLL, [0.030, -0.015, 0.0], 0.005)
        move_route(node, inward, 'hook insertion', duration=0.42)
        if not node._wait_sim_duration(0.30):
            raise RuntimeError('hook insertion settle failed')
        terminal = emit(
            'q3_hook_inserted',
            node,
            nav,
            BOOK,
            released,
            terminal_solution=np.asarray(inward[-1], dtype=float).tolist(),
            unexpected_contacts=sorted(node.unexpected_contact_pairs),
        )
        _, _, terminal_minimum, terminal_maximum = physical_book_bounds()
        terminal_left, terminal_right = node._target_contact_sides(max_age=0.20)
        print(
            json.dumps(
                {
                    'event': 'result',
                    'book_minimum': terminal_minimum.tolist(),
                    'book_maximum': terminal_maximum.tolist(),
                    'book_translation': float(np.linalg.norm(terminal[0] - released[0])),
                    'left_contact': bool(terminal_left),
                    'right_contact': bool(terminal_right),
                    'unexpected_contacts': sorted(node.unexpected_contact_pairs),
                    'terminal_solution': np.asarray(inward[-1], dtype=float).tolist(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        node.track_unexpected_contacts = False
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
