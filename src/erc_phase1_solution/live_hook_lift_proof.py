#!/usr/bin/env python3
"""Temporary fine-grained proof that the open hook lifts the real book."""

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
from live_pick_retreat_probe import (
    BOOK,
    _rotation_distance,
    observed_attached_corners,
    quaternion_matrix,
)


def solve_offset(node, start, dz, dx=0.0):
    pose = node.chain.forward(start)
    position = pose[:3, 3].copy()
    position[0] += float(dx)
    position[2] += float(dz)
    solved, _ = node.chain.solve(
        pose_matrix(position, pose[:3, :3]),
        [start],
        position_tolerance=.00002,
        orientation_tolerance=.0005,
        max_iterations=400,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solved is None:
        raise RuntimeError(f'lift IK failed at dz={dz}')
    return np.asarray(solved, dtype=float)


def fresh_contacts(node):
    node._clear_target_contact_samples(reset_robot_contact=True)
    if not node._wait_sim_duration(.18):
        raise RuntimeError('contact dwell interrupted')
    return node._target_contact_sides(max_age=.22)


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
        left, right = fresh_contacts(node)
        if not left:
            raise RuntimeError('underside contact unavailable at lift start')
        if not right:
            reseat_start = node._measured_left_solution()
            contact_index = None
            for index in range(1, 13):
                target = solve_offset(node, reseat_start, 0.0, .00005 * index)
                if not node._move_arm_solution(target, .22):
                    raise RuntimeError('side re-seat controller failure')
                left, right = fresh_contacts(node)
                current = physical_book_bounds()
                translation = float(np.linalg.norm(current[0] - baseline[0]))
                print(json.dumps({
                    'event': f'cage_reseat_{index}',
                    'left': bool(left),
                    'right': bool(right),
                    'translation_m': translation,
                    'position': current[0].tolist(),
                    'solution': node._measured_left_solution().tolist(),
                }, sort_keys=True), flush=True)
                if not left or translation > .0005 or node._target_robot_contact_latched:
                    raise RuntimeError('cage became unsafe during side re-seat')
                if right and contact_index is None:
                    contact_index = index
                if contact_index is not None:
                    if not right:
                        raise RuntimeError('side contact relaxed during preload')
                    if index >= contact_index + 2:
                        break
            if contact_index is None or not right:
                raise RuntimeError('upper-side contact could not be re-seated')
        start_q = node._measured_left_solution()
        baseline = physical_book_bounds()
        corners, _ = observed_attached_corners(node, node.carried_book_padding)
        node._held_book_corners = corners
        node._gravity_supported_payload = True

        last = baseline
        for index in range(1, 3):
            dz = .00025 * index
            target = solve_offset(node, start_q, dz, .00005 * index)
            if node._robot_self_collision(target) is not None:
                raise RuntimeError(f'self collision at lift step {index}')
            if not node._gravity_supported_transition_is_safe(
                node._measured_left_solution(), target
            ):
                raise RuntimeError(f'geometric support failed at lift step {index}')
            if not node._move_arm_solution(target, .26):
                raise RuntimeError(f'controller failed at lift step {index}')
            left, right = fresh_contacts(node)
            current = physical_book_bounds()
            rise = float(current[2][2] - baseline[2][2])
            horizontal = float(np.linalg.norm(current[0][:2] - baseline[0][:2]))
            rotation = _rotation_distance(
                quaternion_matrix(current[1]), quaternion_matrix(baseline[1])
            )
            print(json.dumps({
                'event': f'hook_lift_{index}',
                'commanded_rise_m': dz,
                'book_rise_m': rise,
                'horizontal_drift_m': horizontal,
                'rotation_rad': rotation,
                'left': bool(left),
                'right': bool(right),
                'position': current[0].tolist(),
                'minimum': current[2].tolist(),
                'maximum': current[3].tolist(),
                'solution': node._measured_left_solution().tolist(),
            }, sort_keys=True), flush=True)
            if (
                not (left and right)
                or horizontal > .0007
                or rotation > math.radians(1.5)
                or float(current[2][2]) < float(last[2][2]) - .0002
                or node._target_robot_contact_latched
            ):
                raise RuntimeError(f'hook lift became unsafe at step {index}')
            last = current

        if not node._wait_sim_duration(.50):
            raise RuntimeError('loaded hook dwell interrupted')
        settled = physical_book_bounds()
        left, right = node._target_contact_sides(max_age=.22)
        drift = float(np.linalg.norm(settled[0] - last[0]))
        passed = bool(
            left
            and right
            and float(settled[2][2] - baseline[2][2]) >= .0001
            and drift <= .0006
            and not node._target_robot_contact_latched
        )
        print(json.dumps({
            'event': 'result',
            'passed': passed,
            'stage': 'open_hook_cam_load_0_5mm',
            'book_rise_m': float(settled[2][2] - baseline[2][2]),
            'drift_m': drift,
            'left': bool(left),
            'right': bool(right),
            'position': settled[0].tolist(),
            'minimum': settled[2].tolist(),
            'maximum': settled[3].tolist(),
            'solution': node._measured_left_solution().tolist(),
        }, sort_keys=True), flush=True)
        if not passed:
            raise RuntimeError('hook did not support the lifted book')
    finally:
        node._retention_probe_active = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
