#!/usr/bin/env python3
"""Temporary same-world bilateral fork ratchet extraction; remove before commit."""

import argparse
import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK, _rotation_distance, quaternion_matrix
from live_q3_fork_probe import solve_from_start
from live_q3_scoop_probe import ScoopProbeNode


def rotation_distance(first, second):
    return _rotation_distance(
        quaternion_matrix(first),
        quaternion_matrix(second),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=3)
    args = parser.parse_args()
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
        if not node._command_gripper(0.018):
            raise RuntimeError('fork aperture restore failed')
        node._clear_target_contact_samples(reset_robot_contact=True)
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('ratchet initial dwell failed')
        if not any(node._target_contact_sides(max_age=0.25)):
            raise RuntimeError('ratchet requires a fresh fork contact')

        baseline_position, baseline_quaternion, _, _ = physical_book_bounds()
        for cycle in range(1, int(args.cycles) + 1):
            cycle_start = node._measured_left_solution()
            before_position, before_quaternion, _, _ = physical_book_bounds()

            reinsert = solve_from_start(node, cycle_start, [0.004, 0.0, 0.0])
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(reinsert, 0.42):
                raise RuntimeError(f'ratchet reinsert failed in cycle {cycle}')
            if not node._wait_sim_duration(0.16):
                raise RuntimeError('ratchet reinsert dwell failed')
            reinsert_left, reinsert_right = node._target_contact_sides(max_age=0.22)
            reinsert_position, reinsert_quaternion, _, _ = physical_book_bounds()
            reinsert_inward = float(reinsert_position[0] - before_position[0])
            if (
                not (reinsert_left or reinsert_right)
                or reinsert_inward > 0.001
                or node.unexpected_contact_pairs
            ):
                raise RuntimeError(f'ratchet reinsert gate failed in cycle {cycle}')

            # Sliding under can unload one edge.  Re-seat in 0.25 mm vertical
            # increments and require both exact contacts before every pull.
            seated = reinsert
            seated_left, seated_right = reinsert_left, reinsert_right
            for seat_height in (0.00025, 0.00050, 0.00075, 0.00100):
                if seated_left and seated_right:
                    break
                seated = solve_from_start(
                    node,
                    reinsert,
                    [0.0, 0.0, seat_height],
                )
                node._clear_target_contact_samples(reset_robot_contact=True)
                if not node._move_arm_solution(seated, 0.26):
                    raise RuntimeError(
                        f'ratchet re-seat failed in cycle {cycle}'
                    )
                if not node._wait_sim_duration(0.14):
                    raise RuntimeError('ratchet re-seat dwell failed')
                seated_left, seated_right = node._target_contact_sides(
                    max_age=0.22
                )
            if not (seated_left and seated_right):
                raise RuntimeError(
                    f'ratchet could not restore bilateral support in cycle {cycle}'
                )

            pull = solve_from_start(node, seated, [-0.005, 0.0, 0.001])
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(pull, 0.48):
                raise RuntimeError(f'ratchet pull failed in cycle {cycle}')
            if not node._wait_sim_duration(0.18):
                raise RuntimeError('ratchet pull dwell failed')
            left, right = node._target_contact_sides(max_age=0.24)
            position, quaternion, minimum, maximum = physical_book_bounds()
            pull_outward = float(reinsert_position[0] - position[0])
            pull_vertical = float(position[2] - reinsert_position[2])
            pull_rotation = rotation_distance(reinsert_quaternion, quaternion)
            cumulative_outward = float(baseline_position[0] - position[0])
            cumulative_rotation = rotation_distance(baseline_quaternion, quaternion)
            print(
                json.dumps(
                    {
                        'event': 'fork_ratchet_cycle',
                        'cycle': cycle,
                        'reinsert_left': bool(reinsert_left),
                        'reinsert_right': bool(reinsert_right),
                        'reinsert_inward': reinsert_inward,
                        'seated_left': bool(seated_left),
                        'seated_right': bool(seated_right),
                        'left': bool(left),
                        'right': bool(right),
                        'pull_outward': pull_outward,
                        'pull_vertical': pull_vertical,
                        'pull_rotation': pull_rotation,
                        'cumulative_outward': cumulative_outward,
                        'cumulative_rotation': cumulative_rotation,
                        'book_minimum': minimum.tolist(),
                        'book_maximum': maximum.tolist(),
                        'unexpected_contacts': sorted(node.unexpected_contact_pairs),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                not (left and right)
                or pull_outward < 0.0035
                or pull_vertical < -0.001
                or pull_rotation > np.deg2rad(2.0)
                or cumulative_rotation > np.deg2rad(8.0)
                or node.unexpected_contact_pairs
            ):
                raise RuntimeError(f'ratchet pull gate failed in cycle {cycle}')

        position, quaternion, minimum, maximum = physical_book_bounds()
        print(
            json.dumps(
                {
                    'event': 'result',
                    'cycles': int(args.cycles),
                    'cumulative_outward': float(baseline_position[0] - position[0]),
                    'cumulative_vertical': float(position[2] - baseline_position[2]),
                    'cumulative_rotation': rotation_distance(
                        baseline_quaternion, quaternion
                    ),
                    'book_minimum': minimum.tolist(),
                    'book_maximum': maximum.tolist(),
                    'shelf_clearance': float(2.755012 - maximum[0]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        node.track_unexpected_contacts = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
