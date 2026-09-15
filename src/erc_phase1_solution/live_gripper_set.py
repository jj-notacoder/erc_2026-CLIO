#!/usr/bin/env python3
"""Temporary same-world gripper-aperture probe; remove before commit."""

import argparse
import json
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK
from live_q3_scoop_probe import ScoopProbeNode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('aperture', type=float)
    args = parser.parse_args()
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ScoopProbeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('live robot state unavailable')
        with node._lock:
            node._target_book_model = BOOK
            node._clear_target_contact_samples_unlocked(
                reset_robot_contact=True,
                reset_target_model=False,
            )
        if not node._command_gripper(float(args.aperture)):
            raise RuntimeError('gripper command failed')
        node._clear_target_contact_samples(reset_robot_contact=True)
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('gripper aperture dwell failed')
        left, right = node._target_contact_sides(max_age=0.25)
        position, quaternion, minimum, maximum = physical_book_bounds()
        print(
            json.dumps(
                {
                    'event': 'gripper_aperture_result',
                    'commanded': float(args.aperture),
                    'measured': float(node.joints['gripper_left_finger_joint']),
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
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
