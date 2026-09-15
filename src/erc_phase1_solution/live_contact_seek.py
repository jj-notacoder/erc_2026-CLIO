#!/usr/bin/env python3
"""Temporary micrometric contact seek from the open-hook CT pose."""

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
    start = np.asarray(start, dtype=float)
    pose = node.chain.forward(start)
    solved, _ = node.chain.solve(
        pose_matrix(pose[:3, 3] + np.asarray(offset), pose[:3, :3]),
        [start],
        position_tolerance=0.00002,
        orientation_tolerance=0.0005,
        max_iterations=400,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solved is None:
        raise RuntimeError(f'contact-seek IK failed for {offset}')
    return np.asarray(solved, dtype=float)


def move_sample(node, target, baseline, event):
    if node._robot_self_collision(target) is not None:
        raise RuntimeError(f'self collision during {event}')
    node._clear_target_contact_samples(reset_robot_contact=True)
    if not node._move_arm_solution(target, .26):
        raise RuntimeError(f'controller failed during {event}')
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
        'position': current[0].tolist(),
        'minimum': current[2].tolist(),
        'maximum': current[3].tolist(),
        'translation_m': translation,
        'rotation_rad': rotation,
        'robot_contact': bool(node._target_robot_contact_latched),
        'solution': node._measured_left_solution().tolist(),
    }, sort_keys=True), flush=True)
    if (
        translation > .0006
        or rotation > math.radians(.8)
        or node._target_robot_contact_latched
    ):
        raise RuntimeError(f'book moved during {event}')
    return bool(left), bool(right), current


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

        under_q = None
        under_height = None
        for index in range(1, 7):
            target = solve_offset(node, start, [0.0, 0.0, .0001 * index])
            left, right, _ = move_sample(
                node, target, baseline, f'under_seek_{index}'
            )
            if right:
                raise RuntimeError('upper finger contacted during underside seek')
            if left:
                under_q = node._measured_left_solution()
                under_height = .0001 * index
                break
        if under_q is None:
            raise RuntimeError('underside contact not found within 0.6mm')

        cage = None
        cage_distance = None
        side_start = np.asarray(under_q, dtype=float)
        for index in range(1, 9):
            target = solve_offset(node, side_start, [.0001 * index, 0.0, 0.0])
            left, right, current = move_sample(
                node, target, baseline, f'side_seek_{index}'
            )
            if not left:
                raise RuntimeError('underside contact lost during side seek')
            if right:
                cage = current
                cage_distance = .0001 * index
                break
        if cage is None:
            raise RuntimeError('side contact not found within 0.8mm')
        print(json.dumps({
            'event': 'result',
            'passed': True,
            'stage': 'open_hook_cage',
            'under_height': under_height,
            'side_distance': cage_distance,
            'position': cage[0].tolist(),
            'minimum': cage[2].tolist(),
            'maximum': cage[3].tolist(),
            'left': True,
            'right': True,
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
