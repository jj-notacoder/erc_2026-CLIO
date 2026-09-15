#!/usr/bin/env python3
"""Temporary same-world contact-leveled fork airlift; remove before commit."""

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK, _rotation_distance, quaternion_matrix
from live_q3_fork_cam import cam_solution
from live_q3_fork_probe import solve_from_start
from live_q3_scoop_probe import ScoopProbeNode


SHELF_FLOOR_Z = 1.451858


def rotation_distance(first, second):
    return _rotation_distance(
        quaternion_matrix(first),
        quaternion_matrix(second),
    )


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ScoopProbeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('live robot state unavailable')
        node.track_unexpected_contacts = True
        with node._lock:
            node._target_book_model = BOOK
            node._clear_target_contact_samples_unlocked(
                reset_robot_contact=True,
                reset_target_model=False,
            )
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('fork airlift initial dwell failed')
        base_position, base_quaternion, _, _ = physical_book_bounds()
        airborne_solution = None
        final_sides = (False, False)
        for step in range(1, 41):
            left, right = node._target_contact_sides(max_age=0.24)
            if not (left or right):
                raise RuntimeError('fork airlift lost both contacts')
            current = node._measured_left_solution()
            if left and right:
                mode = 'bilateral_rise'
                solution = solve_from_start(
                    node,
                    current,
                    [0.0, 0.0, 0.00025],
                )
            else:
                mode = 'left_cam' if left else 'right_cam'
                solution = cam_solution(
                    node,
                    current,
                    -0.005 if left else 0.005,
                    rise=0.00032,
                )
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(solution, 0.25):
                raise RuntimeError(f'fork airlift motion failed at step {step}')
            if not node._wait_sim_duration(0.14):
                raise RuntimeError('fork airlift dwell failed')
            left, right = node._target_contact_sides(max_age=0.23)
            position, quaternion, minimum, maximum = physical_book_bounds()
            rotation = rotation_distance(base_quaternion, quaternion)
            print(
                json.dumps(
                    {
                        'event': 'fork_airlift_step',
                        'step': step,
                        'mode': mode,
                        'left': bool(left),
                        'right': bool(right),
                        'q7': float(solution[-1]),
                        'book_position': position.tolist(),
                        'book_minimum': minimum.tolist(),
                        'book_maximum': maximum.tolist(),
                        'minimum_floor_clearance': float(
                            minimum[2] - SHELF_FLOOR_Z
                        ),
                        'rotation': rotation,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                not (left or right)
                or rotation > np.deg2rad(10.0)
                or float(position[0] - base_position[0]) > 0.015
                or node.unexpected_contact_pairs
            ):
                raise RuntimeError(f'fork airlift gate failed at step {step}')
            final_sides = (bool(left), bool(right))
            if minimum[2] >= SHELF_FLOOR_Z + 0.002 and left and right:
                airborne_solution = solution
                break
        position, quaternion, minimum, maximum = physical_book_bounds()
        result = {
            'event': 'result',
            'airborne': airborne_solution is not None,
            'left': final_sides[0],
            'right': final_sides[1],
            'solution': (
                None
                if airborne_solution is None
                else np.asarray(airborne_solution, dtype=float).tolist()
            ),
            'book_position': position.tolist(),
            'book_quaternion': quaternion.tolist(),
            'book_minimum': minimum.tolist(),
            'book_maximum': maximum.tolist(),
            'minimum_floor_clearance': float(minimum[2] - SHELF_FLOOR_Z),
            'rotation': rotation_distance(base_quaternion, quaternion),
            'unexpected_contacts': sorted(node.unexpected_contact_pairs),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if airborne_solution is None:
            raise RuntimeError('fork did not lift the complete book off the shelf')
    finally:
        node.track_unexpected_contacts = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
