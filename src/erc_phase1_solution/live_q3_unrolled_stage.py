#!/usr/bin/env python3
"""Temporary same-world two-fingertip fork staging; remove before commit."""

import json
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK, emit
from live_q3_scoop_probe import Q_DOWN, Q_ROLL, ScoopProbeNode


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
        # First retreat/lower while retaining the already-safe rolled attitude;
        # only unroll once the complete gripper is outside the shelf lip.
        if not node._move_arm_solution(Q_ROLL, 0.90):
            raise RuntimeError('rolled fork-clear move failed')
        if not node._move_arm_solution(Q_DOWN, 1.30):
            raise RuntimeError('outside-lip unroll failed')
        if not node._wait_sim_duration(0.25):
            raise RuntimeError('unrolled fork settle failed')
        left, right = node._target_contact_sides(max_age=0.20)
        terminal = emit('unrolled_fork_staged', node, nav, BOOK, None)
        position, quaternion, minimum, maximum = physical_book_bounds()
        result = {
            'event': 'result',
            'left': bool(left),
            'right': bool(right),
            'book_position': position.tolist(),
            'book_quaternion': quaternion.tolist(),
            'book_minimum': minimum.tolist(),
            'book_maximum': maximum.tolist(),
            'unexpected_contacts': sorted(node.unexpected_contact_pairs),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if left or right or node.unexpected_contact_pairs:
            raise RuntimeError('unrolled fork staging was not contact-free')
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
