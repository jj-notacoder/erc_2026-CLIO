#!/usr/bin/env python3
"""Temporary open-hand withdrawal after shelf-supported release."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_coupled_catch_probe import physical_book_bounds
from live_hook_extract_continue import Q_EXT
from live_lower_shelf_rescue import RescueProbeNode
from live_pick_retreat_probe import BOOK, _rotation_distance, quaternion_matrix


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
        node._target_book_model = BOOK
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        node._target_robot_contact_latched = False
        baseline = physical_book_bounds()
        start = node._measured_left_solution()
        for index in range(1, 5):
            target = start + (Q_EXT - start) * (index / 4.0)
            if node._robot_self_collision(target) is not None:
                raise RuntimeError(f'self collision on withdrawal leg {index}')
            if not node._move_arm_solution(target, 0.65):
                raise RuntimeError(f'controller failed on withdrawal leg {index}')
            current = physical_book_bounds()
            translation = float(np.linalg.norm(current[0] - baseline[0]))
            rotation = _rotation_distance(
                quaternion_matrix(current[1]),
                quaternion_matrix(baseline[1]),
            )
            print(json.dumps({
                'event': f'open_withdraw_{index}',
                'position': current[0].tolist(),
                'translation_m': translation,
                'rotation_rad': rotation,
                'robot_contact': bool(node._target_robot_contact_latched),
                'solution': node._measured_left_solution().tolist(),
            }, sort_keys=True), flush=True)
            if translation > 0.002 or rotation > 0.035 or node._target_robot_contact_latched:
                raise RuntimeError(f'book disturbed on withdrawal leg {index}')
        current = physical_book_bounds()
        error = float(np.max(np.abs(node._measured_left_solution() - Q_EXT)))
        passed = bool(error < 0.015 and float(current[0][0]) >= 2.763)
        print(json.dumps({
            'event': 'result',
            'passed': passed,
            'stage': 'open_withdrawal',
            'joint_error': error,
            'position': current[0].tolist(),
            'quaternion': current[1].tolist(),
            'minimum': current[2].tolist(),
            'maximum': current[3].tolist(),
            'solution': node._measured_left_solution().tolist(),
        }, sort_keys=True), flush=True)
        if not passed:
            raise RuntimeError('open withdrawal failed')
    finally:
        node._retention_probe_active = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
