#!/usr/bin/env python3
"""Temporary contact-preserving fork-level cam; remove before commit."""

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK
from live_q3_scoop_probe import ScoopProbeNode


def cam_solution(node, start, q7_delta, rise=0.00014):
    start = np.asarray(start, dtype=float)
    start_pose = node.chain.forward(start)
    rolled = start.copy()
    rolled[-1] += float(q7_delta)
    rotation = node.chain.forward(rolled)[:3, :3]
    target_position = start_pose[:3, 3] + np.asarray([0.0, 0.0, rise])
    solved, _ = node.chain.solve(
        pose_matrix(target_position, rotation),
        [rolled],
        position_tolerance=0.0005,
        orientation_tolerance=0.010,
        max_iterations=420,
        fixed_positions={
            'torso_lift_joint': float(start[0]),
            'arm_left_7_joint': float(rolled[-1]),
        },
    )
    if solved is None:
        raise RuntimeError('fork cam IK failed')
    solved = np.asarray(solved, dtype=float)
    if node._robot_self_collision(solved) is not None:
        raise RuntimeError('fork cam has a self-collision')
    return solved


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ScoopProbeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('live robot state unavailable')
        node.track_unexpected_contacts = True
        with node._lock:
            node._target_book_model = BOOK
            node._clear_target_contact_samples_unlocked(
                reset_robot_contact=True,
                reset_target_model=False,
            )
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('fork cam initial dwell failed')
        bilateral_solution = None
        for step in range(1, 31):
            left, right = node._target_contact_sides(max_age=0.24)
            if left and right:
                bilateral_solution = node._measured_left_solution()
                break
            if not (left or right):
                raise RuntimeError('fork cam lost both contacts')
            q7_delta = -0.01 if left else 0.01
            solution = cam_solution(
                node,
                node._measured_left_solution(),
                q7_delta,
            )
            node._clear_target_contact_samples(reset_robot_contact=True)
            if not node._move_arm_solution(solution, 0.30):
                raise RuntimeError(f'fork cam motion failed at step {step}')
            if not node._wait_sim_duration(0.16):
                raise RuntimeError('fork cam dwell failed')
            left, right = node._target_contact_sides(max_age=0.24)
            position, quaternion, minimum, maximum = physical_book_bounds()
            print(
                json.dumps(
                    {
                        'event': 'fork_cam_step',
                        'step': step,
                        'q7_delta': q7_delta,
                        'left': bool(left),
                        'right': bool(right),
                        'q7': float(solution[-1]),
                        'book_position': position.tolist(),
                        'book_quaternion': quaternion.tolist(),
                        'book_minimum': minimum.tolist(),
                        'book_maximum': maximum.tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if node.unexpected_contact_pairs:
                raise RuntimeError('fork cam caused an unexpected collision')
            if left and right:
                bilateral_solution = solution
                break
        print(
            json.dumps(
                {
                    'event': 'result',
                    'bilateral': bilateral_solution is not None,
                    'solution': (
                        None
                        if bilateral_solution is None
                        else np.asarray(bilateral_solution, dtype=float).tolist()
                    ),
                    'unexpected_contacts': sorted(node.unexpected_contact_pairs),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if bilateral_solution is None:
            raise RuntimeError('fork cam did not restore bilateral support')
    finally:
        node.track_unexpected_contacts = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
