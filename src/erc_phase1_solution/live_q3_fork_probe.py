#!/usr/bin/env python3
"""Temporary same-world bilateral under-fingertip fork probe; remove before commit."""

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK, emit
from live_q3_scoop_probe import ScoopProbeNode


def solve_from_start(node, start, offset):
    start = np.asarray(start, dtype=float)
    pose = node.chain.forward(start)
    target = pose[:3, 3] + np.asarray(offset, dtype=float)
    solved, _ = node.chain.solve(
        pose_matrix(target, pose[:3, :3]),
        [start],
        position_tolerance=0.0008,
        orientation_tolerance=0.015,
        max_iterations=360,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solved is None:
        raise RuntimeError(f'fork IK failed at {target.tolist()}')
    solved = np.asarray(solved, dtype=float)
    if node._robot_self_collision(solved) is not None:
        raise RuntimeError('fork route has a self-collision')
    return solved


def main():
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
            raise RuntimeError('fork aperture command failed')
        start = node._measured_left_solution()
        baseline = emit('bilateral_fork_start', node, nav, BOOK, None)

        inserted = start
        for distance in (0.005, 0.010, 0.015, 0.020):
            inserted = solve_from_start(node, start, [distance, 0.0, 0.0])
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(inserted, 0.38):
                raise RuntimeError(f'fork insertion failed at {distance:.3f} m')
            if not node._wait_sim_duration(0.08):
                raise RuntimeError('fork insertion dwell failed')
            left, right = node._target_contact_sides(max_age=0.18)
            if left or right:
                raise RuntimeError('fork contacted the book before vertical seating')
            if node.unexpected_contact_pairs:
                raise RuntimeError('fork insertion caused an unexpected collision')

        inserted_pose = emit('bilateral_fork_inserted', node, nav, BOOK, baseline)
        contact_solution = None
        contact_height = None
        contact_sides = (False, False)
        heights = [0.016, *np.arange(0.0165, 0.0221, 0.0005).tolist()]
        for index, height in enumerate(heights):
            solution = solve_from_start(node, start, [0.020, 0.0, float(height)])
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(solution, 0.60 if index == 0 else 0.30):
                raise RuntimeError(f'fork seating failed at {height:.4f} m')
            if not node._wait_sim_duration(0.12):
                raise RuntimeError('fork seating dwell failed')
            left, right = node._target_contact_sides(max_age=0.20)
            _, _, minimum, maximum = physical_book_bounds()
            print(
                json.dumps(
                    {
                        'event': 'fork_contact_step',
                        'height': float(height),
                        'left': bool(left),
                        'right': bool(right),
                        'book_minimum': minimum.tolist(),
                        'book_maximum': maximum.tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if node.unexpected_contact_pairs:
                raise RuntimeError('fork seating caused an unexpected collision')
            if left or right:
                contact_solution = solution
                contact_height = float(height)
                contact_sides = (bool(left), bool(right))
                break

        terminal = emit('bilateral_fork_contact', node, nav, BOOK, inserted_pose)
        _, _, minimum, maximum = physical_book_bounds()
        result = {
            'event': 'result',
            'contact_height': contact_height,
            'left': contact_sides[0],
            'right': contact_sides[1],
            'contact_solution': (
                None
                if contact_solution is None
                else np.asarray(contact_solution, dtype=float).tolist()
            ),
            'book_minimum': minimum.tolist(),
            'book_maximum': maximum.tolist(),
            'unexpected_contacts': sorted(node.unexpected_contact_pairs),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if not all(contact_sides):
            raise RuntimeError('fork did not establish bilateral support')
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
