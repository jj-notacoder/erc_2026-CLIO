#!/usr/bin/env python3
"""Temporary real-shelf coordinated roll/rise catch probe; remove before commit."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.navigation_node import NavigationNode
from live_pick_retreat_probe import (
    BASE_POSITION,
    BASE_YAW,
    BOOK,
    PickProbeNode,
    _rotation_distance,
    emit,
    gazebo_poses,
    quaternion_matrix,
    set_model_pose,
)


ROUTE = [
    np.asarray(values, dtype=float)
    for values in (
        (
            0.35,
            0.6996603538396243,
            0.8186069489366229,
            -0.07508275060043502,
            -1.352591768687821,
            0.31361447040647233,
            1.1409944341927594,
            -0.5883022070197879,
        ),
        (
            0.35,
            0.6750556729512911,
            0.9111500490524177,
            -0.0824933915765194,
            -1.2981625731150341,
            0.2942532517864413,
            1.1714074374225603,
            -0.9178053120200729,
        ),
        (
            0.35,
            0.6504509920629578,
            1.0036931491682126,
            -0.08990403255260376,
            -1.2437333775422474,
            0.2748920331664103,
            1.2018204406523612,
            -1.2473084170203579,
        ),
        (
            0.35,
            0.6258463111746245,
            1.0962362492840074,
            -0.09731467352868814,
            -1.1893041819694605,
            0.25553081454637927,
            1.232233443882162,
            -1.576811522020643,
        ),
    )
]


def physical_book_bounds():
    position, quaternion = gazebo_poses(BOOK)['book']
    half = np.asarray([0.125, 0.015, 0.080], dtype=float)
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )
    corners = (signs * half) @ quaternion_matrix(quaternion).T + position
    return position, quaternion, np.min(corners, axis=0), np.max(corners, axis=0)


def main() -> None:
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = PickProbeNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 30.0
        while (
            len(node.joints) < 8
            or nav.pose is None
            or nav.last_front_scan_time is None
            or nav.last_rear_scan_time is None
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8 or nav.pose is None:
            raise RuntimeError('live robot state is unavailable')

        node.gripper_transport_lock = 0.029
        node.top_row_grasp_vertical_offset = -0.065
        node.skip_return_preflight = True
        node.bottom_cradle = True

        if not node._stow():
            raise RuntimeError('stow failed')
        base_quaternion = np.asarray(
            [0.0, 0.0, math.sin(BASE_YAW / 2.0), math.cos(BASE_YAW / 2.0)],
            dtype=float,
        )
        set_model_pose('tiago_pro', BASE_POSITION, base_quaternion)
        if not node._wait_sim_duration(0.40):
            raise RuntimeError('base teleport did not settle')
        if not node._pick():
            raise RuntimeError('lower real-shelf pick failed')

        baseline = emit(
            'coupled_pick_complete',
            node,
            nav,
            BOOK,
            None,
            measured_start=node._measured_left_solution().tolist(),
        )
        left, right = node._target_contact_sides(max_age=0.15)
        if not (left and right and node._target_book_model == BOOK):
            raise RuntimeError('real pick did not establish the exact bilateral lock')

        # The intentional transfer changes bilateral pinch into lower-fingertip
        # support.  During this diagnostic, cancel only on a scored robot hit;
        # final retention and pose stability are measured independently below.
        node._payload_monitor_enabled = False
        node._gravity_supported_payload = True
        original_hazard = node._payload_hazard_reason

        def robot_only_hazard(*, max_age=0.20):
            del max_age
            if node._payload_hazard_latched is not None:
                return str(node._payload_hazard_latched)
            if node._target_robot_contact_latched:
                return 'payload_robot_contact'
            return None

        node._payload_hazard_reason = robot_only_hazard
        legs = [
            (ROUTE[0], 0.45, 'catch_lift'),
            (ROUTE[1], 0.90, 'catch_roll'),
            (ROUTE[2], 0.90, 'catch_roll'),
            (ROUTE[3], 0.90, 'catch_roll'),
        ]
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        node._retention_probe_active = True
        node._payload_robot_watchdog_enabled = True
        try:
            moved, contact_fault = node._send_retained_arm_trajectory(
                goal,
                duration,
                legs,
                'coupled_catch_probe',
            )
        finally:
            node._payload_hazard_reason = original_hazard
            node._retention_probe_active = False
            node._payload_robot_watchdog_enabled = False
        if not moved:
            raise RuntimeError(
                f'coupled catch trajectory failed (contact_fault={contact_fault})'
            )

        if not node._wait_sim_duration(0.50):
            raise RuntimeError('first support dwell was interrupted')
        first = emit('coupled_catch_first_dwell', node, nav, BOOK, baseline)
        first_bounds = physical_book_bounds()
        first_left, first_right = node._target_contact_sides(max_age=0.15)

        if not node._wait_sim_duration(0.75):
            raise RuntimeError('second support dwell was interrupted')
        second = emit('coupled_catch_second_dwell', node, nav, BOOK, first)
        second_bounds = physical_book_bounds()
        second_left, second_right = node._target_contact_sides(max_age=0.15)
        stability_translation = float(np.linalg.norm(second[0] - first[0]))
        stability_rotation = _rotation_distance(first[1], second[1])
        initial_translation = float(np.linalg.norm(first[0] - baseline[0]))
        initial_rotation = _rotation_distance(baseline[1], first[1])
        result = {
            'event': 'result',
            'passed': bool(
                moved
                and first_left
                and second_left
                and stability_translation <= 0.005
                and stability_rotation <= math.radians(5.0)
                and float(second_bounds[2][2]) >= 1.49
                and not node._target_robot_contact_latched
            ),
            'moved': bool(moved),
            'contact_fault': bool(contact_fault),
            'first_left': bool(first_left),
            'first_right': bool(first_right),
            'second_left': bool(second_left),
            'second_right': bool(second_right),
            'initial_translation_m': initial_translation,
            'initial_rotation_rad': initial_rotation,
            'stability_translation_m': stability_translation,
            'stability_rotation_rad': stability_rotation,
            'first_minimum_z': float(first_bounds[2][2]),
            'second_minimum_z': float(second_bounds[2][2]),
            'second_position': second_bounds[0].tolist(),
            'robot_contact_latched': bool(node._target_robot_contact_latched),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if not result['passed']:
            raise RuntimeError('coordinated catch did not establish stable support')
    finally:
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
