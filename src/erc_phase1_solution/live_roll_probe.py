#!/usr/bin/env python3
"""Temporary same-world cradle dynamics probe; remove before commit."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.manipulation_node import ManipulationNode
from erc_phase1_solution.navigation_node import NavigationNode
from live_retreat_probe import set_model_pose


PRE_ROLL = np.asarray(
    [
        0.35,
        0.678358206892,
        1.040598400040,
        -0.062552115375,
        -1.618660613295,
        0.282722401697,
        1.609463041905,
        -0.444189842088,
    ],
    dtype=float,
)


def set_book_pose(name: str, position: np.ndarray) -> None:
    half_pitch = 0.5 * 1.5708
    request = (
        f'name: "{name}", '
        f'position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}, '
        f'orientation: {{y: {math.sin(half_pitch)}, w: {math.cos(half_pitch)}}}'
    )
    completed = subprocess.run(
        [
            'gz',
            'service',
            '-s',
            '/world/erc_world/set_pose',
            '--reqtype',
            'gz.msgs.Pose',
            '--reptype',
            'gz.msgs.Boolean',
            '--timeout',
            '3000',
            '--req',
            request,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if 'true' not in completed.stdout:
        raise RuntimeError(f'book pose request failed: {completed.stdout!r}')


def emit(event: str, node: ManipulationNode, **fields) -> None:
    verified, width, left, right, plausible = node._pinch_sample()
    print(
        json.dumps(
            {
                'event': event,
                'verified': bool(verified),
                'width': float(width),
                'left': bool(left),
                'right': bool(right),
                'plausible': bool(plausible),
                'robot_contact_latched': bool(
                    getattr(node, '_target_robot_contact_latched', False)
                ),
                'target_model': getattr(node, '_target_book_model', None),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=1.5)
    parser.add_argument('--preload', type=float, default=0.0165)
    parser.add_argument('--lock', type=float, default=None)
    parser.add_argument('--extension', action='store_true')
    parser.add_argument('--staging', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--extension-distance', type=float, default=0.0)
    parser.add_argument('--roll', type=float, default=-1.30)
    parser.add_argument('--retreat-distance', type=float, default=0.0)
    parser.add_argument('--book', default='book_col_3_row_2_red')
    parser.add_argument('--base-x', type=float, default=2.126938006268)
    parser.add_argument('--base-y', type=float, default=-0.093344043799)
    parser.add_argument('--base-yaw', type=float, default=0.001089153)
    args = parser.parse_args()

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ManipulationNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.add_node(nav)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        deadline = time.monotonic() + 15.0
        while (
            (
                'gripper_left_finger_joint' not in node.joints
                or nav.pose is None
                or nav.last_front_scan_time is None
                or nav.last_rear_scan_time is None
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if 'gripper_left_finger_joint' not in node.joints:
            raise RuntimeError('joint states unavailable')

        safe = np.asarray([4.0, 2.0, 0.08], dtype=float)
        set_book_pose(args.book, safe)
        set_model_pose(
            'tiago_pro',
            np.asarray([args.base_x, args.base_y, 0.0], dtype=float),
            np.asarray(
                [
                    0.0,
                    0.0,
                    math.sin(0.5 * args.base_yaw),
                    math.cos(0.5 * args.base_yaw),
                ],
                dtype=float,
            ),
        )
        node._wait_sim_duration(0.30)
        if not node._command_gripper(0.020):
            raise RuntimeError('pre-open failed')
        if not node._move_arm_solution(PRE_ROLL, 2.0):
            raise RuntimeError('pre-roll arm move failed')

        grasp_pose = node.chain.forward(PRE_ROLL)
        centre_base = grasp_pose[:3, 3] + np.asarray([0.020, 0.0, 0.0])
        cosine = math.cos(args.base_yaw)
        sine = math.sin(args.base_yaw)
        centre_world = np.asarray(
            [
                args.base_x + cosine * centre_base[0] - sine * centre_base[1],
                args.base_y + sine * centre_base[0] + cosine * centre_base[1],
                centre_base[2],
            ],
            dtype=float,
        )
        set_book_pose(args.book, centre_world)
        node._target_robot_contact_latched = False
        if not node._command_gripper(args.preload):
            raise RuntimeError('preload failed')
        node._left_target_contact_ns = 0
        node._right_target_contact_ns = 0
        node._wait_sim_duration(0.20)
        emit(
            'pre_roll',
            node,
            book_position=[float(value) for value in centre_world],
            grasp_position=[
                float(value) for value in grasp_pose[:3, 3]
            ],
        )
        if args.lock is not None:
            if not node._command_gripper(args.lock):
                raise RuntimeError('transport lock failed')
            node._transport_lock_engaged = True
            node._left_target_contact_ns = 0
            node._right_target_contact_ns = 0
            node._wait_sim_duration(0.20)
            emit('post_lock', node, lock=args.lock)

        rolled = PRE_ROLL.copy()
        rolled[-1] += args.roll
        if not node._move_arm_solution(rolled, args.duration):
            raise RuntimeError('roll failed')
        node._gravity_supported_payload = True
        node._left_target_contact_ns = 0
        node._right_target_contact_ns = 0
        node._wait_sim_duration(0.20)
        emit('post_roll', node, duration=args.duration, preload=args.preload)
        if args.retreat_distance > 0.0:
            if nav.pose is None:
                raise RuntimeError('odometry unavailable before retreat')
            start = np.asarray(nav.pose, dtype=float)
            goal = (
                float(start[0] - args.retreat_distance * math.cos(start[2])),
                float(start[1] - args.retreat_distance * math.sin(start[2])),
                float(start[2]),
            )
            nav._accept_goal(goal, profile='carried_retreat')
            all_bilateral = True
            while nav.goal is not None:
                left, right = node._target_contact_sides(max_age=0.15)
                all_bilateral = all_bilateral and left and right
                time.sleep(0.02)
            node._wait_sim_duration(0.25)
            displacement = (
                math.hypot(nav.pose[0] - start[0], nav.pose[1] - start[1])
                if nav.pose is not None
                else math.nan
            )
            emit(
                'post_retreat',
                node,
                requested_distance=args.retreat_distance,
                actual_displacement=displacement,
                all_bilateral=bool(all_bilateral),
            )
        if args.extension or args.staging or args.compact:
            node.carried_cradle_extension = args.extension_distance
            extension, lowering, retraction = (
                node._solve_supported_post_retreat_staging(rolled)
            )
            staging = [
                *((solution, 'supported_cradle_extension') for solution in extension),
                *((solution, 'supported_cradle_lowering') for solution in lowering),
                *((solution, 'supported_cradle_retraction') for solution in retraction),
            ]
            if args.compact:
                compact_start = (
                    staging[-1][0] if staging else rolled
                )
                staging.extend(
                    (solution, 'compact_transport')
                    for solution in node._supported_compact_goals(compact_start)
                )
            if not args.staging:
                if args.compact:
                    pass
                else:
                    staging = [
                        (solution, 'supported_cradle_extension')
                        for solution in extension
                    ]
            legs = []
            previous = rolled.copy()
            for solution, phase in staging:
                legs.append(
                    (
                        solution,
                        node._transport_leg_duration(previous, solution),
                        phase,
                    )
                )
                previous = solution
            if not legs:
                emit('post_staging', node, moved=True, contact_lost=False, duration=0.0)
                return
            goal, total_duration = node._make_retained_arm_trajectory_goal(legs)
            if not node.arm_client.wait_for_server(timeout_sec=5.0):
                raise RuntimeError('arm server unavailable')
            moved, contact_lost = node._send_retained_arm_trajectory(
                goal,
                total_duration,
                legs,
                'probe_extension',
            )
            emit(
                'post_staging' if args.staging else 'post_extension',
                node,
                moved=bool(moved),
                contact_lost=bool(contact_lost),
                duration=total_duration,
            )
        node._wait_sim_duration(1.0)
        node._left_target_contact_ns = 0
        node._right_target_contact_ns = 0
        node._wait_sim_duration(0.20)
        emit('post_dwell', node, duration=args.duration, preload=args.preload)
    finally:
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
