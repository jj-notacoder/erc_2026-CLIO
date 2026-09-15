#!/usr/bin/env python3
"""Temporary same-world adaptive fork leveling; remove before commit."""

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK
from live_q3_scoop_probe import ScoopProbeNode


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
        start = node._measured_left_solution()
        bilateral = None
        for delta in np.arange(0.01, 0.101, 0.01):
            solution = start.copy()
            solution[-1] += float(delta)
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(solution, 0.30):
                raise RuntimeError(f'fork leveling failed at {delta:.3f} rad')
            if not node._wait_sim_duration(0.15):
                raise RuntimeError('fork leveling dwell failed')
            left, right = node._target_contact_sides(max_age=0.22)
            position, quaternion, minimum, maximum = physical_book_bounds()
            print(
                json.dumps(
                    {
                        'event': 'fork_level_step',
                        'roll_delta': float(delta),
                        'left': bool(left),
                        'right': bool(right),
                        'book_position': position.tolist(),
                        'book_quaternion': quaternion.tolist(),
                        'book_minimum': minimum.tolist(),
                        'book_maximum': maximum.tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if node.unexpected_contact_pairs:
                raise RuntimeError('fork leveling caused an unexpected collision')
            if left and right:
                bilateral = solution
                break
        print(
            json.dumps(
                {
                    'event': 'result',
                    'bilateral': bilateral is not None,
                    'solution': (
                        None if bilateral is None else bilateral.tolist()
                    ),
                    'unexpected_contacts': sorted(node.unexpected_contact_pairs),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if bilateral is None:
            raise RuntimeError('fork leveling did not restore bilateral support')
    finally:
        node.track_unexpected_contacts = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
