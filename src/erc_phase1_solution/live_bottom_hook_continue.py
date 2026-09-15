#!/usr/bin/env python3
"""Temporary same-world bottom-hook continuation probe; remove before commit."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import (
    BOOK,
    PickProbeNode,
    _rotation_distance,
    observed_attached_corners,
    quaternion_matrix,
)


Q_EXT = np.asarray(
    [
        0.35,
        0.694356032812,
        1.002037268331,
        -0.064411156950,
        -1.678900821488,
        0.294235608298,
        1.630941424793,
        -0.443853762084,
    ],
    dtype=float,
)


def emit(event, node, baseline_position, baseline_rotation):
    position, quaternion, minimum, maximum = physical_book_bounds()
    rotation = quaternion_matrix(quaternion)
    left, right = node._target_contact_sides(max_age=0.15)
    payload = {
        'event': event,
        'solution': node._measured_left_solution().tolist(),
        'book_position': position.tolist(),
        'book_minimum': minimum.tolist(),
        'book_maximum': maximum.tolist(),
        'translation_m': float(np.linalg.norm(position - baseline_position)),
        'rotation_rad': _rotation_distance(baseline_rotation, rotation),
        'left': bool(left),
        'right': bool(right),
        'robot_contact': bool(node._target_robot_contact_latched),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return position, rotation


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = PickProbeNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
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
        corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        node._held_book_corners = corners
        node._payload_monitor_enabled = True
        if not node._fresh_retention_probe('bottom_hook', 'resume', leg=0):
            raise RuntimeError('same-world bilateral grasp is unavailable')

        baseline = physical_book_bounds()
        baseline_position = baseline[0]
        baseline_rotation = quaternion_matrix(baseline[1])
        emit('resume', node, baseline_position, baseline_rotation)

        # Undo the failed 3 mm pinch-only lift.  The book remained on its shelf
        # while the fingers slid, so returning the grasp link restores the
        # audited Q_EXT starting geometry without disturbing the payload.
        node._retention_probe_active = True
        try:
            moved = node._move_arm_solution(Q_EXT, 0.50)
        finally:
            node._retention_probe_active = False
        if not moved or not node._fresh_retention_probe(
            'bottom_hook', 'restore_q_ext', leg=1
        ):
            raise RuntimeError('failed to restore Q_EXT')

        # Slide down the stationary book in 10 mm fixed-orientation increments.
        # At 90 mm the calibrated lower fingertip sweep reaches the overhanging
        # bottom edge; geometric support can then replace pinch friction.
        for index in range(9):
            measured = node._measured_left_solution()
            achieved = node.chain.forward(measured)
            position = achieved[:3, 3].copy()
            position[2] -= 0.010
            solution, _ = node.chain.solve(
                pose_matrix(position, achieved[:3, :3]),
                [measured],
                position_tolerance=node.position_tolerance,
                orientation_tolerance=node.orientation_tolerance,
                max_iterations=240,
                fixed_positions={'torso_lift_joint': float(measured[0])},
            )
            if solution is None:
                raise RuntimeError(f'down-slide IK failed at step {index + 1}')
            solution = np.asarray(solution, dtype=float)
            if node._robot_self_collision(solution) is not None:
                raise RuntimeError(f'down-slide self collision at step {index + 1}')
            node._retention_probe_active = True
            try:
                moved = node._move_arm_solution(solution, 0.55)
            finally:
                node._retention_probe_active = False
            if not moved or not node._fresh_retention_probe(
                'bottom_hook', 'down_slide', leg=2 + index
            ):
                raise RuntimeError(
                    f'bilateral contact lost during down step {index + 1}'
                )
            current_position, current_rotation = emit(
                f'down_{(index + 1) * 10}mm',
                node,
                baseline_position,
                baseline_rotation,
            )
            if (
                np.linalg.norm(current_position - baseline_position) > 0.001
                or _rotation_distance(baseline_rotation, current_rotation)
                > math.radians(1.0)
                or node._target_robot_contact_latched
            ):
                raise RuntimeError('book moved while sliding to the bottom edge')

        print(
            json.dumps(
                {
                    'event': 'result',
                    'passed': True,
                    'stage': 'bottom_edge_slide',
                    'solution': node._measured_left_solution().tolist(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
