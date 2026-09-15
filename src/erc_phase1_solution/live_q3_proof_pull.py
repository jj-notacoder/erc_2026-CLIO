#!/usr/bin/env python3
"""Temporary same-world shelf-supported hook proof pull; remove before commit."""

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
from live_q3_scoop_continue import solve_offset
from live_q3_scoop_probe import ScoopProbeNode


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
            raise RuntimeError('hook aperture restore failed')
        node._clear_target_contact_samples(reset_robot_contact=True)
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('hook contact settle failed')
        initial_left, initial_right = node._target_contact_sides(max_age=0.25)
        if not (initial_left or initial_right):
            raise RuntimeError('proof pull started without exact-book hook contact')
        start = node._measured_left_solution()
        baseline = emit('hook_proof_start', node, nav, BOOK, None)
        baseline_position, baseline_quaternion, baseline_minimum, _ = (
            physical_book_bounds()
        )
        target = solve_offset(node, start, [-0.005, 0.0, 0.001])
        node._clear_target_contact_samples(reset_robot_contact=True)
        if not node._move_arm_solution(target, 0.65):
            raise RuntimeError('hook proof motion failed')
        if not node._wait_sim_duration(0.25):
            raise RuntimeError('hook proof dwell failed')
        left, right = node._target_contact_sides(max_age=0.25)
        terminal = emit('hook_proof_complete', node, nav, BOOK, baseline)
        position, quaternion, minimum, maximum = physical_book_bounds()
        outward = float(baseline_position[0] - position[0])
        vertical = float(position[2] - baseline_position[2])
        rotation = _rotation_distance(
            quaternion_matrix(baseline_quaternion),
            quaternion_matrix(quaternion),
        )
        result = {
            'event': 'result',
            'initial_left': bool(initial_left),
            'initial_right': bool(initial_right),
            'left': bool(left),
            'right': bool(right),
            'book_outward_motion': outward,
            'book_vertical_motion': vertical,
            'book_rotation': rotation,
            'book_minimum': minimum.tolist(),
            'book_maximum': maximum.tolist(),
            'unexpected_contacts': sorted(node.unexpected_contact_pairs),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if (
            outward < 0.004
            or vertical < -0.001
            or rotation > np.deg2rad(3.0)
            or not (left or right)
            or node.unexpected_contact_pairs
        ):
            raise RuntimeError('unilateral hook failed the proof pull')
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
