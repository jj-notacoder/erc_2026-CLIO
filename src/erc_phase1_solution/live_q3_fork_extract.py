#!/usr/bin/env python3
"""Temporary same-world guarded bilateral-fork extraction; remove before commit."""

import argparse
import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import (
    BOOK,
    _rotation_distance,
    emit,
    quaternion_matrix,
)
from live_q3_fork_probe import solve_from_start
from live_q3_scoop_probe import ScoopProbeNode


def rotation_distance(first, second):
    return _rotation_distance(
        quaternion_matrix(first),
        quaternion_matrix(second),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ScoopProbeNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    executor.add_node(nav)
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
            raise RuntimeError('fork extraction initial dwell failed')
        initial_left, initial_right = node._target_contact_sides(max_age=0.25)
        if not (initial_left and initial_right):
            raise RuntimeError('fork extraction requires fresh bilateral support')

        start_q = node._measured_left_solution()
        baseline = emit('fork_extraction_start', node, nav, BOOK, None)
        base_position, base_quaternion, base_minimum, base_maximum = (
            physical_book_bounds()
        )
        previous_position = base_position
        previous_quaternion = base_quaternion
        for index in range(1, int(args.steps) + 1):
            solution = solve_from_start(
                node,
                start_q,
                [-0.005 * index, 0.0, 0.001 * index],
            )
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(solution, 0.48):
                raise RuntimeError(f'fork extraction failed at step {index}')
            if not node._wait_sim_duration(0.18):
                raise RuntimeError('fork extraction dwell failed')
            left, right = node._target_contact_sides(max_age=0.24)
            position, quaternion, minimum, maximum = physical_book_bounds()
            step_outward = float(previous_position[0] - position[0])
            step_vertical = float(position[2] - previous_position[2])
            step_rotation = rotation_distance(previous_quaternion, quaternion)
            cumulative_outward = float(base_position[0] - position[0])
            cumulative_rotation = rotation_distance(base_quaternion, quaternion)
            print(
                json.dumps(
                    {
                        'event': 'fork_extraction_step',
                        'step': index,
                        'left': bool(left),
                        'right': bool(right),
                        'step_outward': step_outward,
                        'step_vertical': step_vertical,
                        'step_rotation': step_rotation,
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
                or step_outward < 0.0035
                or step_vertical < -0.001
                or step_rotation > np.deg2rad(2.0)
                or cumulative_rotation > np.deg2rad(8.0)
                or node.unexpected_contact_pairs
            ):
                raise RuntimeError(f'fork extraction gate failed at step {index}')
            previous_position = position
            previous_quaternion = quaternion

        terminal = emit('fork_extraction_complete', node, nav, BOOK, baseline)
        position, quaternion, minimum, maximum = physical_book_bounds()
        result = {
            'event': 'result',
            'steps': int(args.steps),
            'left': True,
            'right': True,
            'cumulative_outward': float(base_position[0] - position[0]),
            'cumulative_vertical': float(position[2] - base_position[2]),
            'cumulative_rotation': rotation_distance(base_quaternion, quaternion),
            'book_minimum': minimum.tolist(),
            'book_maximum': maximum.tolist(),
            'shelf_clearance': float(2.755012 - maximum[0]),
            'unexpected_contacts': sorted(node.unexpected_contact_pairs),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
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
