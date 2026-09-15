#!/usr/bin/env python3
"""Temporary guarded disengagement from the failed compliant hook."""

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
from live_lower_shelf_rescue import RescueProbeNode
from live_pick_retreat_probe import BOOK, _rotation_distance, quaternion_matrix


def solve_offset(node, start, offset):
    pose = node.chain.forward(start)
    solved, _ = node.chain.solve(
        pose_matrix(pose[:3, 3] + np.asarray(offset), pose[:3, :3]),
        [start],
        position_tolerance=.00003,
        orientation_tolerance=.0007,
        max_iterations=400,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solved is None:
        raise RuntimeError(f'disengage IK failed for {offset}')
    return np.asarray(solved, dtype=float)


def sample(node, baseline, event):
    node._clear_target_contact_samples(reset_robot_contact=True)
    if not node._wait_sim_duration(.18):
        raise RuntimeError('contact dwell interrupted')
    current = physical_book_bounds()
    left, right = node._target_contact_sides(max_age=.22)
    translation = float(np.linalg.norm(current[0] - baseline[0]))
    rotation = _rotation_distance(
        quaternion_matrix(current[1]), quaternion_matrix(baseline[1])
    )
    print(json.dumps({
        'event': event,
        'left': bool(left),
        'right': bool(right),
        'translation_m': translation,
        'rotation_rad': rotation,
        'position': current[0].tolist(),
        'solution': node._measured_left_solution().tolist(),
    }, sort_keys=True), flush=True)
    if translation > .0003 or rotation > math.radians(.5) or node._target_robot_contact_latched:
        raise RuntimeError(f'book moved during {event}')
    return bool(left), bool(right)


def move(node, target, baseline, event, duration=.30):
    if node._robot_self_collision(target) is not None:
        raise RuntimeError(f'self collision during {event}')
    if not node._move_arm_solution(target, duration):
        raise RuntimeError(f'controller failure during {event}')
    return sample(node, baseline, event)


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
        start = node._measured_left_solution()

        clear_q = None
        for index in range(1, 5):
            target = solve_offset(node, start, [0.0, 0.0, -.00025 * index])
            left, right = move(node, target, baseline, f'disengage_down_{index}')
            if not left and not right:
                clear_q = node._measured_left_solution()
                break
        if clear_q is None:
            raise RuntimeError('lower contact did not release within 1mm')

        outward_start = np.asarray(clear_q, dtype=float)
        terminal = outward_start
        for distance in (.005, .010, .020, .040, .056):
            target = solve_offset(node, outward_start, [-distance, 0.0, 0.0])
            left, right = move(
                node, target, baseline, f'disengage_out_{int(distance * 1000)}'
            )
            if left or right:
                raise RuntimeError('contact returned during outward withdrawal')
            terminal = node._measured_left_solution()

        roll_start = np.asarray(terminal, dtype=float)
        for index in range(1, 5):
            target = roll_start.copy()
            target[-1] = roll_start[-1] + 1.55 * (index / 4.0)
            left, right = move(
                node, target, baseline, f'disengage_unroll_{index}', duration=.48
            )
            if left or right:
                raise RuntimeError('contact returned during unroll')

        final_book = physical_book_bounds()
        print(json.dumps({
            'event': 'result',
            'passed': True,
            'stage': 'failed_hook_disengaged',
            'position': final_book[0].tolist(),
            'quaternion': final_book[1].tolist(),
            'minimum': final_book[2].tolist(),
            'maximum': final_book[3].tolist(),
            'aperture': float(node.joints['gripper_left_finger_joint']),
            'solution': node._measured_left_solution().tolist(),
        }, sort_keys=True), flush=True)
    finally:
        node._retention_probe_active = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
