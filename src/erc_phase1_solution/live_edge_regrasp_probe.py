#!/usr/bin/env python3
"""Temporary real-shelf bottom-edge regrasp probe; remove before commit."""

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
    BASE_POSITION,
    BASE_YAW,
    BOOK,
    PickProbeNode,
    _rotation_distance,
    emit,
    set_model_pose,
)


Q_OUT = np.asarray(
    [
        0.35,
        0.6734756853,
        0.8960508912,
        -0.0646125748,
        -1.3764016728,
        0.2838232384,
        1.2350771320,
        -0.5457885161,
    ],
    dtype=float,
)
Q_DOWN = np.asarray(
    [
        0.35,
        0.7545204806,
        0.6694487140,
        -0.0416973493,
        -1.6824786171,
        0.3423814816,
        1.3375676675,
        -0.6026645814,
    ],
    dtype=float,
)
Q_ROLL = Q_DOWN.copy()
Q_ROLL[-1] -= 1.55


def make_outward_route(node, start, *, lift=0.020, distance=0.18):
    start = np.asarray(start, dtype=float)
    start_pose = node.chain.forward(start)
    rotation = start_pose[:3, :3]
    previous = start
    position = start_pose[:3, 3].copy()
    targets = [position + np.asarray([0.0, 0.0, lift])]
    lifted = targets[-1].copy()
    for offset in np.arange(0.02, distance + 0.001, 0.02):
        targets.append(lifted + np.asarray([-float(offset), 0.0, 0.0]))
    route = []
    for target in targets:
        solved, score = node.chain.solve(
            pose_matrix(target, rotation),
            [previous],
            position_tolerance=0.003,
            orientation_tolerance=node.orientation_tolerance,
            max_iterations=280,
            fixed_positions={'torso_lift_joint': float(start[0])},
        )
        if solved is None:
            raise RuntimeError(f'edge-support IK failed at {target.tolist()}')
        solved = np.asarray(solved, dtype=float)
        if node._robot_self_collision(solved) is not None:
            raise RuntimeError('edge-support route has a robot self-collision')
        route.append(solved)
        previous = solved
    return route


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
        node.top_row_grasp_vertical_offset = -0.015
        node.skip_return_preflight = True
        # Reuse the probe override to stop at Cartesian solution index 5: one
        # extraction leg, while the book still has shelf support and overhang.
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
            raise RuntimeError('centered shelf pick failed')

        pick_pose = emit(
            'edge_pick_complete',
            node,
            nav,
            BOOK,
            None,
            measured_start=node._measured_left_solution().tolist(),
        )
        left, right = node._target_contact_sides(max_age=0.15)
        if not (left and right and node._target_book_model == BOOK):
            raise RuntimeError('centered pick did not establish exact bilateral contact')

        # Intentionally leave the book on its known shelf support before
        # moving the open hand around the exposed bottom/front corner.
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        if not node._command_gripper(0.069):
            raise RuntimeError('supported release failed')
        node._clear_target_contact_samples()
        if not node._wait_sim_duration(0.30):
            raise RuntimeError('supported release settle failed')
        release_pose = emit('edge_supported_release', node, nav, BOOK, pick_pose)
        release_bounds = physical_book_bounds()

        if not node._move_arm_solution(Q_OUT, 1.2):
            raise RuntimeError('edge outward-clear move failed')
        stage_pose = emit('edge_out_complete', node, nav, BOOK, release_pose)
        stage_bounds = physical_book_bounds()
        if node._target_robot_contact_latched:
            raise RuntimeError('edge outward-clear move caused a book/robot collision')

        if not node._move_arm_solution(Q_DOWN, 1.4):
            raise RuntimeError('edge descent failed')
        down_pose = emit('edge_down_complete', node, nav, BOOK, stage_pose)
        down_bounds = physical_book_bounds()
        if node._target_robot_contact_latched:
            raise RuntimeError('edge descent caused a book/robot collision')

        if not node._move_arm_solution(Q_ROLL, 1.4):
            raise RuntimeError('edge roll failed')
        roll_pose = emit('edge_roll_complete', node, nav, BOOK, down_pose)
        roll_bounds = physical_book_bounds()
        if node._target_robot_contact_latched:
            raise RuntimeError('edge roll caused a book/robot collision')

        # Close through the exposed corner only after the lower fingertip is
        # beneath the physical bottom face.
        if not node._command_gripper(0.018):
            raise RuntimeError('edge regrasp close failed')
        node._clear_target_contact_samples()
        if not node._wait_sim_duration(0.25):
            raise RuntimeError('edge contact reacquisition failed')
        close_pose = emit('edge_close_complete', node, nav, BOOK, roll_pose)
        close_bounds = physical_book_bounds()
        close_left, close_right = node._target_contact_sides(max_age=0.15)

        outward = make_outward_route(node, Q_ROLL)
        lift_goal, lift_duration = node._make_retained_arm_trajectory_goal(
            [(outward[0], 0.65, 'edge_support_lift')]
        )
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
        try:
            lifted, lift_fault = node._send_retained_arm_trajectory(
                lift_goal,
                lift_duration,
                [(outward[0], 0.65, 'edge_support_lift')],
                'edge_regrasp_probe',
            )
        finally:
            node._payload_hazard_reason = original_hazard
        if not lifted:
            raise RuntimeError(f'edge support lift failed (fault={lift_fault})')

        if not node._wait_sim_duration(0.50):
            raise RuntimeError('edge support lift dwell failed')
        lifted_pose = emit('edge_lift_complete', node, nav, BOOK, close_pose)
        lifted_bounds = physical_book_bounds()
        lift_left, lift_right = node._target_contact_sides(max_age=0.15)
        lifted_minimum_z = float(lifted_bounds[2][2])
        close_minimum_z = float(close_bounds[2][2])
        lift_delta_z = lifted_minimum_z - close_minimum_z
        lift_translation = float(np.linalg.norm(lifted_pose[0] - close_pose[0]))
        lift_rotation = _rotation_distance(close_pose[1], lifted_pose[1])
        print(
            json.dumps(
                {
                    'event': 'edge_lift_gate',
                    'close_left': bool(close_left),
                    'close_right': bool(close_right),
                    'lift_left': bool(lift_left),
                    'lift_right': bool(lift_right),
                    'close_minimum_z': close_minimum_z,
                    'lifted_minimum_z': lifted_minimum_z,
                    'lift_delta_z': lift_delta_z,
                    'translation_m': lift_translation,
                    'rotation_rad': lift_rotation,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            lift_delta_z < 0.006
            or lift_translation > 0.020
            or lift_rotation > math.radians(8.0)
            or not lift_left
            or node._target_robot_contact_latched
        ):
            raise RuntimeError('edge regrasp failed the physical lift gate')

        result = {
            'event': 'result',
            'passed': bool(
                lift_left
                and lifted_minimum_z >= close_minimum_z + 0.006
                and not node._target_robot_contact_latched
            ),
            'close_minimum_z': close_minimum_z,
            'lifted_minimum_z': lifted_minimum_z,
            'lift_left': bool(lift_left),
            'lift_right': bool(lift_right),
            'supported_release_minimum_z': float(release_bounds[2][2]),
            'robot_contact_latched': bool(node._target_robot_contact_latched),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if not result['passed']:
            raise RuntimeError('edge-supported lift failed')
    finally:
        node._retention_probe_active = False
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
