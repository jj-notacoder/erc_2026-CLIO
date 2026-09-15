#!/usr/bin/env python3
"""Temporary guarded same-world recovery to the known Q_EXT grasp."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_coupled_catch_probe import physical_book_bounds
from live_hook_extract_continue import Q_EXT
from live_lower_shelf_rescue import RescueProbeNode, retained_arm_leg
from live_pick_retreat_probe import BOOK, observed_attached_corners


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = RescueProbeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('joint state unavailable')
        node.gripper_transport_lock = 0.029
        node._target_book_model = BOOK
        node._transport_lock_engaged = True
        node._gravity_supported_payload = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        node._payload_monitor_enabled = True
        if not node._fresh_retention_probe('recover_qext', 'start', leg=0):
            raise RuntimeError('bilateral grasp unavailable at recovery start')

        start = node._measured_left_solution()
        before = physical_book_bounds()
        for index in range(1, 5):
            target = start + (Q_EXT - start) * (index / 4.0)
            corners, _ = observed_attached_corners(node, node.carried_book_padding)
            node._held_book_corners = corners
            retained_arm_leg(node, target, 0.45, 'recover_qext', index)
        if not node._wait_sim_duration(0.45):
            raise RuntimeError('recovery settle interrupted')
        after = physical_book_bounds()
        left, right = node._target_contact_sides(max_age=0.20)
        error = float(np.max(np.abs(node._measured_left_solution() - Q_EXT)))
        print(json.dumps({
            'event': 'result',
            'passed': bool(left and right and error < 0.015),
            'joint_error': error,
            'left': bool(left),
            'right': bool(right),
            'before_position': before[0].tolist(),
            'after_position': after[0].tolist(),
            'after_quaternion': after[1].tolist(),
            'after_minimum': after[2].tolist(),
            'after_maximum': after[3].tolist(),
            'solution': node._measured_left_solution().tolist(),
        }, sort_keys=True), flush=True)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
