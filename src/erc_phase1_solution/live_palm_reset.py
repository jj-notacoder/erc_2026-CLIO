#!/usr/bin/env python3
"""Temporary guarded same-world palm push that returns the book to shelf support."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from live_coupled_catch_probe import physical_book_bounds
from live_lower_shelf_rescue import RescueProbeNode, retained_arm_leg
from live_pick_retreat_probe import (
    BOOK,
    _rotation_distance,
    observed_attached_corners,
    quaternion_matrix,
)


# The shelf deck starts at x=2.755 m.  A centre at or beyond 2.766 m leaves
# at least 11 mm of static-stability margin while remaining reachable by the
# guarded 17-step reset from the deterministic seed-101 pick.  The previous
# 2.775 m threshold exceeded that route's measured reach and made a safely
# supported reset report failure after its final guarded step.
TARGET_CENTER_X = 2.766


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
        if not node._fresh_retention_probe('palm_reset', 'start', leg=0):
            raise RuntimeError('bilateral grasp unavailable at reset start')

        start_q = node._measured_left_solution()
        start_pose = node.chain.forward(start_q)
        start_book = physical_book_bounds()
        start_y = float(start_book[0][1])
        start_z = float(start_book[0][2])
        start_rotation = quaternion_matrix(start_book[1])
        previous_q = start_q
        last_x = float(start_book[0][0])
        completed = False
        for index in range(1, 18):
            position = start_pose[:3, 3].copy()
            position[0] += min(0.005 * index, 0.082)
            solution, _ = node.chain.solve(
                pose_matrix(position, start_pose[:3, :3]),
                [previous_q],
                position_tolerance=0.00025,
                orientation_tolerance=0.002,
                max_iterations=300,
                fixed_positions={'torso_lift_joint': float(start_q[0])},
            )
            if solution is None:
                raise RuntimeError(f'palm-reset IK failed at step {index}')
            solution = np.asarray(solution, dtype=float)
            corners, _ = observed_attached_corners(node, node.carried_book_padding)
            node._held_book_corners = corners
            retained_arm_leg(node, solution, 0.38, 'palm_reset', index)
            current = physical_book_bounds()
            center = current[0]
            rotation = _rotation_distance(
                start_rotation,
                quaternion_matrix(current[1]),
            )
            left, right = node._target_contact_sides(max_age=0.20)
            print(json.dumps({
                'event': f'palm_reset_{index}',
                'position': center.tolist(),
                'minimum': current[2].tolist(),
                'maximum': current[3].tolist(),
                'rotation_from_start_rad': rotation,
                'left': bool(left),
                'right': bool(right),
                'solution': node._measured_left_solution().tolist(),
            }, sort_keys=True), flush=True)
            if not (left and right):
                raise RuntimeError('bilateral contact lost during palm reset')
            if (
                abs(float(center[1]) - start_y) > 0.002
                or float(center[2]) < start_z - 0.002
                or float(center[0]) < last_x - 0.001
                or rotation > math.radians(4.0)
                or node._target_robot_contact_latched
                or node.neighbor_contact_pairs
            ):
                raise RuntimeError(f'unsafe book motion during palm reset step {index}')
            last_x = float(center[0])
            previous_q = solution
            if last_x >= TARGET_CENTER_X:
                completed = True
                break
        if not completed:
            raise RuntimeError('palm reset did not return the COM behind the shelf edge')
        if not node._wait_sim_duration(0.50):
            raise RuntimeError('palm reset dwell interrupted')
        settled = physical_book_bounds()
        left, right = node._target_contact_sides(max_age=0.20)
        drift = float(np.linalg.norm(settled[0] - current[0]))
        passed = bool(
            left
            and right
            and float(settled[0][0]) >= TARGET_CENTER_X
            and drift <= 0.002
            and not node._target_robot_contact_latched
        )
        print(json.dumps({
            'event': 'result',
            'passed': passed,
            'stage': 'palm_reset_supported',
            'position': settled[0].tolist(),
            'quaternion': settled[1].tolist(),
            'minimum': settled[2].tolist(),
            'maximum': settled[3].tolist(),
            'drift': drift,
            'left': bool(left),
            'right': bool(right),
            'solution': node._measured_left_solution().tolist(),
        }, sort_keys=True), flush=True)
        if not passed:
            raise RuntimeError('palm reset did not settle safely')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
