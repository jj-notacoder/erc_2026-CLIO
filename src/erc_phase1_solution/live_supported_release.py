#!/usr/bin/env python3
"""Temporary staged release after the guarded palm reset."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_coupled_catch_probe import physical_book_bounds
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
        node._retention_probe_active = False
        baseline = physical_book_bounds()
        if float(baseline[0][0]) < 2.765 or float(baseline[2][2]) < 1.450:
            raise RuntimeError('book is not safely supported before release')
        previous = baseline
        for aperture in (0.040, 0.055, 0.069):
            if not node._command_gripper(aperture):
                raise RuntimeError(f'gripper failed at aperture {aperture}')
            current = physical_book_bounds()
            translation = float(np.linalg.norm(current[0] - previous[0]))
            rotation = _rotation_distance(
                quaternion_matrix(current[1]),
                quaternion_matrix(previous[1]),
            )
            left, right = node._target_contact_sides(max_age=0.25)
            print(json.dumps({
                'event': 'staged_release',
                'commanded': aperture,
                'measured': float(node.joints['gripper_left_finger_joint']),
                'position': current[0].tolist(),
                'minimum': current[2].tolist(),
                'maximum': current[3].tolist(),
                'translation_m': translation,
                'rotation_rad': rotation,
                'left': bool(left),
                'right': bool(right),
            }, sort_keys=True), flush=True)
            unsafe = (
                float(current[0][0]) < 2.763
                or float(current[2][2]) < 1.450
                or translation > 0.003
                or rotation > math.radians(3.0)
            )
            if unsafe:
                node._command_gripper(0.029)
                raise RuntimeError(f'book moved during staged release at {aperture}')
            previous = current
        if not node._wait_sim_duration(0.60):
            raise RuntimeError('open-gripper dwell interrupted')
        settled = physical_book_bounds()
        translation = float(np.linalg.norm(settled[0] - previous[0]))
        rotation = _rotation_distance(
            quaternion_matrix(settled[1]),
            quaternion_matrix(previous[1]),
        )
        passed = bool(
            float(settled[0][0]) >= 2.763
            and float(settled[2][2]) >= 1.450
            and translation <= 0.002
            and rotation <= math.radians(2.0)
            and float(node.joints['gripper_left_finger_joint']) >= 0.067
        )
        print(json.dumps({
            'event': 'result',
            'passed': passed,
            'stage': 'supported_release',
            'position': settled[0].tolist(),
            'quaternion': settled[1].tolist(),
            'minimum': settled[2].tolist(),
            'maximum': settled[3].tolist(),
            'translation_m': translation,
            'rotation_rad': rotation,
            'measured_aperture': float(node.joints['gripper_left_finger_joint']),
            'solution': node._measured_left_solution().tolist(),
        }, sort_keys=True), flush=True)
        if not passed:
            raise RuntimeError('released book did not remain stable')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
