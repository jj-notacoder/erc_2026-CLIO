#!/usr/bin/env python3
"""Temporary Gazebo-only vertical-pinch retreat probe; remove before commit."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import re
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

from erc_phase1_solution.manipulation_node import ManipulationNode
from erc_phase1_solution.motion_profiles import IK_JOINTS
from erc_phase1_solution.navigation_node import NavigationNode


# Loaded top-row clearance configuration produced by the 60 mm extraction.
VERTICAL_CLEARANCE = np.asarray(
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


def _protobuf_scalar(block: str, field: str, default: float = 0.0) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(field)}:\s*([^\s]+)", block)
    return default if match is None else float(match.group(1))


def _entity_pose(message: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
    marker = f'name: "{name}"'
    marker_index = message.find(marker)
    if marker_index < 0:
        raise RuntimeError(f'Gazebo pose message omitted {name!r}')
    start = message.rfind('pose {', 0, marker_index)
    next_pose = message.find('\npose {', marker_index)
    block = message[start:] if next_pose < 0 else message[start:next_pose]
    position_match = re.search(r'position\s*\{([^}]*)\}', block, re.DOTALL)
    orientation_match = re.search(r'orientation\s*\{([^}]*)\}', block, re.DOTALL)
    if position_match is None or orientation_match is None:
        raise RuntimeError(f'Gazebo pose for {name!r} was incomplete')
    position_block = position_match.group(1)
    orientation_block = orientation_match.group(1)
    position = np.asarray(
        [_protobuf_scalar(position_block, axis) for axis in ('x', 'y', 'z')],
        dtype=float,
    )
    quaternion = np.asarray(
        [
            _protobuf_scalar(orientation_block, axis)
            for axis in ('x', 'y', 'z', 'w')
        ],
        dtype=float,
    )
    return position, quaternion


def gazebo_poses(book: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    completed = subprocess.run(
        [
            'gz',
            'topic',
            '-e',
            '-t',
            '/world/erc_world/dynamic_pose/info',
            '-n',
            '1',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    return {
        'base': _entity_pose(completed.stdout, 'tiago_pro'),
        'book': _entity_pose(completed.stdout, book),
    }


def set_model_pose(
    name: str,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    request = (
        f'name: "{name}", '
        f'position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}, '
        'orientation: {'
        f'x: {quaternion[0]}, y: {quaternion[1]}, '
        f'z: {quaternion[2]}, w: {quaternion[3]}'
        '}'
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
        timeout=5.0,
    )
    if 'true' not in completed.stdout.lower():
        raise RuntimeError(f'Gazebo rejected pose for {name}: {completed.stdout!r}')


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    x, y, z, w = (quaternion / norm).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


class ProbeManipulationNode(ManipulationNode):
    def __init__(self) -> None:
        self.gripper_state_samples = deque(maxlen=20000)
        super().__init__()

    def _on_joint_state(self, message: JointState) -> None:
        super()._on_joint_state(message)
        try:
            index = list(message.name).index('gripper_left_finger_joint')
        except ValueError:
            return
        position = float(message.position[index]) if index < len(message.position) else math.nan
        velocity = float(message.velocity[index]) if index < len(message.velocity) else math.nan
        effort = float(message.effort[index]) if index < len(message.effort) else math.nan
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        self.gripper_state_samples.append((stamp_ns, position, velocity, effort))


def _current_arm(node: ProbeManipulationNode) -> np.ndarray:
    with node._lock:
        return np.asarray([node.joints[name] for name in IK_JOINTS], dtype=float)


def _gripper_relative_book(
    node: ProbeManipulationNode,
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    base_position, base_quaternion = poses['base']
    book_position, book_quaternion = poses['book']
    world_from_base = np.eye(4, dtype=float)
    world_from_base[:3, :3] = quaternion_matrix(base_quaternion)
    world_from_base[:3, 3] = base_position
    world_from_grasp = world_from_base @ node.chain.forward(_current_arm(node))
    relative_position = (
        world_from_grasp[:3, :3].T
        @ (book_position - world_from_grasp[:3, 3])
    )
    relative_rotation = (
        world_from_grasp[:3, :3].T @ quaternion_matrix(book_quaternion)
    )
    return relative_position, relative_rotation


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(first.T @ second) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _latest_gripper(node: ProbeManipulationNode) -> Dict[str, float]:
    if not node.gripper_state_samples:
        return {'position': math.nan, 'velocity': math.nan, 'effort': math.nan}
    _, position, velocity, effort = node.gripper_state_samples[-1]
    return {'position': position, 'velocity': velocity, 'effort': effort}


def emit(
    event: str,
    node: ProbeManipulationNode,
    nav: NavigationNode,
    book: str,
    baseline: Optional[Tuple[np.ndarray, np.ndarray]],
    **fields,
) -> Tuple[np.ndarray, np.ndarray]:
    poses = gazebo_poses(book)
    relative_position, relative_rotation = _gripper_relative_book(node, poses)
    left, right = node._target_contact_sides(max_age=0.15)
    payload = {
        'event': event,
        'sim_time': node.get_clock().now().nanoseconds / 1e9,
        'odom': None if nav.pose is None else list(nav.pose),
        'gazebo_base_position': poses['base'][0].tolist(),
        'gazebo_book_position': poses['book'][0].tolist(),
        'book_in_grasp': relative_position.tolist(),
        'bilateral': bool(left and right),
        'left': bool(left),
        'right': bool(right),
        'robot_contact_latched': bool(node._target_robot_contact_latched),
        'gripper': _latest_gripper(node),
        **fields,
    }
    if baseline is not None:
        payload['translation_drift_m'] = float(
            np.linalg.norm(relative_position - baseline[0])
        )
        payload['rotation_drift_rad'] = _rotation_distance(
            baseline[1], relative_rotation
        )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return relative_position, relative_rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=float, required=True)
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--direct-start', type=float, default=0.032)
    parser.add_argument('--teleport-into-lock', action='store_true')
    parser.add_argument('--distance', type=float, default=0.35)
    parser.add_argument('--book', default='book_col_3_row_2_red')
    parser.add_argument('--base-x', type=float, default=2.126938006268)
    parser.add_argument('--base-y', type=float, default=-0.093344043799)
    parser.add_argument('--base-yaw', type=float, default=0.001089153)
    args = parser.parse_args()

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ProbeManipulationNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    executor.add_node(nav)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        deadline = time.monotonic() + 20.0
        while (
            any(name not in node.joints for name in IK_JOINTS)
            or nav.pose is None
            or nav.last_front_scan_time is None
            or nav.last_rear_scan_time is None
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        if any(name not in node.joints for name in IK_JOINTS):
            raise RuntimeError('arm joint states unavailable')
        if nav.pose is None:
            raise RuntimeError('odometry unavailable')

        nav._publish_zero()
        safe_position = np.asarray([0.0, 4.0, 0.08], dtype=float)
        book_quaternion = np.asarray(
            [0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)],
            dtype=float,
        )
        set_model_pose(args.book, safe_position, book_quaternion)
        node._wait_sim_duration(0.35)
        if not node._open_gripper():
            raise RuntimeError('gripper open failed')

        base_quaternion = np.asarray(
            [0.0, 0.0, math.sin(args.base_yaw / 2.0), math.cos(args.base_yaw / 2.0)],
            dtype=float,
        )
        set_model_pose(
            'tiago_pro',
            np.asarray([args.base_x, args.base_y, 0.0], dtype=float),
            base_quaternion,
        )
        node._wait_sim_duration(0.30)
        if not node._move_arm_solution(VERTICAL_CLEARANCE, 2.0):
            raise RuntimeError('vertical clearance arm move failed')

        # Gazebo's model teleport deliberately does not reset the integrated
        # odom origin.  Seed the book from the actual Gazebo model transform,
        # not from odom, while navigation continues to use its native frame.
        measured_base_position, measured_base_quaternion = gazebo_poses(args.book)[
            'base'
        ]
        grasp_pose = node.chain.forward(VERTICAL_CLEARANCE)
        centre_base = grasp_pose[:3, 3] + np.asarray([0.020, 0.0, 0.0])
        centre_world = (
            measured_base_position
            + quaternion_matrix(measured_base_quaternion) @ centre_base
        )
        node._clear_target_contact_samples(
            reset_robot_contact=True,
            reset_target_model=True,
        )
        with node._lock:
            node._target_book_model = args.book
        if args.direct and args.teleport_into_lock:
            raise RuntimeError('direct and teleport-into-lock are mutually exclusive')
        if args.teleport_into_lock:
            if not node._command_gripper(args.lock):
                raise RuntimeError('empty transport lock command failed')
        elif args.direct:
            if args.direct_start <= args.lock:
                raise RuntimeError('direct-start must be wider than lock')
            if not node._command_gripper(args.direct_start):
                raise RuntimeError('direct open starting command failed')
        seed_reference = _gripper_relative_book(
            node,
            {
                'base': (measured_base_position, measured_base_quaternion),
                'book': (centre_world, book_quaternion),
            },
        )
        set_model_pose(args.book, centre_world, book_quaternion)
        node._wait_sim_duration(
            0.05 if args.teleport_into_lock else (0.02 if args.direct else 0.10)
        )
        if args.teleport_into_lock:
            pass
        elif args.direct:
            if not node._command_gripper(args.lock):
                raise RuntimeError('direct transport lock command failed')
        else:
            if not node._command_gripper(0.0155):
                raise RuntimeError('initial preload failed')
            node._clear_target_contact_samples(reset_robot_contact=True)
            node._wait_sim_duration(0.25)
            emit('preload', node, nav, args.book, None, command=0.0155)
            if abs(args.lock - 0.0155) > 1e-9:
                if not node._command_gripper(args.lock):
                    raise RuntimeError('transport lock command failed')
        node._transport_lock_engaged = True
        node._clear_target_contact_samples(reset_robot_contact=True)
        node._wait_sim_duration(0.05 if args.teleport_into_lock else 0.25)
        baseline = emit(
            'locked',
            node,
            nav,
            args.book,
            seed_reference if args.teleport_into_lock else None,
            command=args.lock,
            teleport_into_lock=bool(args.teleport_into_lock),
        )
        baseline_position = np.asarray(baseline[0], dtype=float)
        baseline_rotation = np.asarray(baseline[1], dtype=float)
        left, right = node._target_contact_sides(max_age=0.15)
        if not (left and right):
            print(
                json.dumps(
                    {
                        'event': 'result',
                        'command': args.lock,
                        'passed': False,
                        'reason': 'no_bilateral_contact_before_retreat',
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

        if nav.pose is None:
            raise RuntimeError('odometry unavailable before retreat')
        start = np.asarray(nav.pose, dtype=float)
        goal = (
            float(start[0] - args.distance * math.cos(start[2])),
            float(start[1] - args.distance * math.sin(start[2])),
            float(start[2]),
        )
        start_sim_ns = node.get_clock().now().nanoseconds
        nav._accept_goal(goal, profile='carried_retreat')
        milestones = [0.10, 0.20, 0.30]
        next_milestone = 0
        all_bilateral = True
        first_contact_loss = None
        maximum_speed = 0.0
        while nav.goal is not None:
            if nav.pose is None:
                time.sleep(0.02)
                continue
            displacement = float(math.hypot(nav.pose[0] - start[0], nav.pose[1] - start[1]))
            maximum_speed = max(
                maximum_speed,
                math.hypot(nav.last_command.linear.x, nav.last_command.linear.y),
            )
            left, right = node._target_contact_sides(max_age=0.15)
            if not (left and right):
                all_bilateral = False
                if first_contact_loss is None:
                    first_contact_loss = {
                        'sim_time': node.get_clock().now().nanoseconds / 1e9,
                        'displacement': displacement,
                        'left': bool(left),
                        'right': bool(right),
                    }
            if (
                next_milestone < len(milestones)
                and displacement >= milestones[next_milestone]
            ):
                emit(
                    'milestone',
                    node,
                    nav,
                    args.book,
                    (baseline_position, baseline_rotation),
                    command=args.lock,
                    requested_milestone=milestones[next_milestone],
                    displacement=displacement,
                )
                next_milestone += 1
            time.sleep(0.02)

        node._wait_sim_duration(0.25)
        final = emit(
            'retreat_complete',
            node,
            nav,
            args.book,
            (baseline_position, baseline_rotation),
            command=args.lock,
            requested_distance=args.distance,
            actual_displacement=float(
                math.hypot(nav.pose[0] - start[0], nav.pose[1] - start[1])
            ) if nav.pose is not None else math.nan,
        )
        end_sim_ns = node.get_clock().now().nanoseconds
        motion_samples = [
            sample
            for sample in node.gripper_state_samples
            if start_sim_ns <= sample[0] <= end_sim_ns
        ]
        efforts = [sample[3] for sample in motion_samples if math.isfinite(sample[3])]
        positions = [sample[1] for sample in motion_samples if math.isfinite(sample[1])]
        final_left, final_right = node._target_contact_sides(max_age=0.15)
        translation_drift = float(np.linalg.norm(final[0] - baseline_position))
        rotation_drift = _rotation_distance(baseline_rotation, final[1])
        passed = bool(
            all_bilateral
            and final_left
            and final_right
            and not node._target_robot_contact_latched
            and translation_drift <= 0.015
            and rotation_drift <= 0.15
        )
        print(
            json.dumps(
                {
                    'event': 'result',
                    'command': args.lock,
                    'passed': passed,
                    'all_bilateral': all_bilateral,
                    'final_bilateral': bool(final_left and final_right),
                    'first_contact_loss': first_contact_loss,
                    'robot_contact_latched': bool(node._target_robot_contact_latched),
                    'translation_drift_m': translation_drift,
                    'rotation_drift_rad': rotation_drift,
                    'maximum_commanded_speed': maximum_speed,
                    'effort_min': min(efforts) if efforts else math.nan,
                    'effort_max': max(efforts) if efforts else math.nan,
                    'effort_mean': float(np.mean(efforts)) if efforts else math.nan,
                    'position_min': min(positions) if positions else math.nan,
                    'position_max': max(positions) if positions else math.nan,
                    'position_mean': float(np.mean(positions)) if positions else math.nan,
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
