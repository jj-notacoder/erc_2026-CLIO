#!/usr/bin/env python3
"""Temporary small-aperture cam load for the open bottom hook."""

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
            time.sleep(.05)
        if len(node.joints) < 8:
            raise RuntimeError('joint state unavailable')
        node._target_book_model = BOOK
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        node._target_robot_contact_latched = False
        baseline = physical_book_bounds()
        terminal = None
        for aperture in (.068, .067):
            if not node._command_gripper(aperture):
                raise RuntimeError(f'cam command failed at {aperture}')
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._wait_sim_duration(.22):
                raise RuntimeError('cam dwell interrupted')
            current = physical_book_bounds()
            left, right = node._target_contact_sides(max_age=.26)
            rise = float(current[2][2] - baseline[2][2])
            inward = float(current[0][0] - baseline[0][0])
            lateral = float(current[0][1] - baseline[0][1])
            rotation = _rotation_distance(
                quaternion_matrix(current[1]), quaternion_matrix(baseline[1])
            )
            print(json.dumps({
                'event': 'aperture_cam',
                'commanded': aperture,
                'measured': float(node.joints['gripper_left_finger_joint']),
                'left': bool(left),
                'right': bool(right),
                'book_rise_m': rise,
                'book_inward_m': inward,
                'book_lateral_m': lateral,
                'rotation_rad': rotation,
                'position': current[0].tolist(),
                'minimum': current[2].tolist(),
                'maximum': current[3].tolist(),
            }, sort_keys=True), flush=True)
            if (
                not left
                or inward > .00045
                or abs(lateral) > .0004
                or rise < -.0003
                or rotation > math.radians(1.0)
                or node._target_robot_contact_latched
            ):
                node._command_gripper(.069)
                raise RuntimeError(f'unsafe cam response at aperture {aperture}')
            terminal = current
            if right and rise >= .0001:
                break
        if terminal is None:
            raise RuntimeError('cam did not execute')
        left, right = node._target_contact_sides(max_age=.26)
        rise = float(terminal[2][2] - baseline[2][2])
        passed = bool(left and right and rise >= .0001)
        print(json.dumps({
            'event': 'result',
            'passed': passed,
            'stage': 'aperture_cam_load',
            'book_rise_m': rise,
            'left': bool(left),
            'right': bool(right),
            'aperture': float(node.joints['gripper_left_finger_joint']),
            'position': terminal[0].tolist(),
            'minimum': terminal[2].tolist(),
            'maximum': terminal[3].tolist(),
            'solution': node._measured_left_solution().tolist(),
        }, sort_keys=True), flush=True)
        if not passed:
            raise RuntimeError('aperture cam did not begin unloading the shelf')
    finally:
        node._retention_probe_active = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
