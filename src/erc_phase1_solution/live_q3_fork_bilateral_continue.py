#!/usr/bin/env python3
"""Temporary same-world second-fingertip contact search; remove before commit."""

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK, emit
from live_q3_fork_probe import solve_from_start
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
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('initial fork-contact dwell failed')
        start = node._measured_left_solution()
        baseline = emit('second_fingertip_search_start', node, nav, BOOK, None)
        bilateral_solution = None
        for height in np.arange(0.00025, 0.00251, 0.00025):
            solution = solve_from_start(node, start, [0.0, 0.0, float(height)])
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(solution, 0.28):
                raise RuntimeError(f'second-contact motion failed at {height:.5f} m')
            if not node._wait_sim_duration(0.14):
                raise RuntimeError('second-contact dwell failed')
            left, right = node._target_contact_sides(max_age=0.22)
            _, _, minimum, maximum = physical_book_bounds()
            print(
                json.dumps(
                    {
                        'event': 'second_fingertip_step',
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
                raise RuntimeError('second-contact search caused an unexpected collision')
            if left and right:
                bilateral_solution = solution
                break
        terminal = emit('second_fingertip_search_complete', node, nav, BOOK, baseline)
        position, quaternion, minimum, maximum = physical_book_bounds()
        result = {
            'event': 'result',
            'bilateral': bilateral_solution is not None,
            'solution': (
                None
                if bilateral_solution is None
                else np.asarray(bilateral_solution, dtype=float).tolist()
            ),
            'book_position': position.tolist(),
            'book_quaternion': quaternion.tolist(),
            'book_minimum': minimum.tolist(),
            'book_maximum': maximum.tolist(),
            'unexpected_contacts': sorted(node.unexpected_contact_pairs),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if bilateral_solution is None:
            raise RuntimeError('second fingertip did not acquire the book')
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
