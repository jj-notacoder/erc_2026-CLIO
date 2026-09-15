#!/usr/bin/env python3
"""Temporary same-world q3 hook contact search; remove before commit."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK, emit
from live_q3_scoop_probe import ScoopProbeNode


def solve_offset(node, start, offset):
    start = np.asarray(start, dtype=float)
    pose = node.chain.forward(start)
    target = pose[:3, 3] + np.asarray(offset, dtype=float)
    solved, _ = node.chain.solve(
        pose_matrix(target, pose[:3, :3]),
        [start],
        position_tolerance=0.0008,
        orientation_tolerance=0.015,
        max_iterations=360,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solved is None:
        raise RuntimeError(f'contact-search IK failed at {target.tolist()}')
    solved = np.asarray(solved, dtype=float)
    if node._robot_self_collision(solved) is not None:
        raise RuntimeError('contact-search solution has a self-collision')
    return solved


def main() -> None:
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ScoopProbeNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 30.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('live robot state unavailable')
        start = node._measured_left_solution()
        if abs(float(node.joints.get('gripper_left_finger_joint', 1.0)) - 0.018) > 0.003:
            raise RuntimeError('hook continuation did not start with the closed hook')
        node.track_unexpected_contacts = True
        with node._lock:
            node._target_book_model = BOOK
            node._clear_target_contact_samples_unlocked(
                reset_robot_contact=True,
                reset_target_model=False,
            )

        baseline = emit('hook_contact_search_start', node, nav, BOOK, None)
        _, _, baseline_minimum, baseline_maximum = physical_book_bounds()
        # The runtime mesh measurement leaves 13.78 mm beneath the nearer
        # fingertip.  Take one conservative 10 mm step, then approach in
        # 0.5 mm increments with a fresh exact-model contact sample each time.
        offsets = [0.010, *np.arange(0.0105, 0.0161, 0.0005).tolist()]
        previous = start
        contact_side = None
        contact_solution = None
        for index, total_z in enumerate(offsets):
            solution = solve_offset(node, start, [0.0, 0.0, float(total_z)])
            node._clear_target_contact_samples(reset_robot_contact=True)
            duration = 0.65 if index == 0 else 0.32
            if not node._move_arm_solution(solution, duration):
                raise RuntimeError(f'contact-search motion failed at {total_z:.4f} m')
            if not node._wait_sim_duration(0.12):
                raise RuntimeError('contact-search dwell failed')
            left, right = node._target_contact_sides(max_age=0.20)
            _, _, minimum, maximum = physical_book_bounds()
            print(
                json.dumps(
                    {
                        'event': 'hook_contact_step',
                        'total_z': float(total_z),
                        'left': bool(left),
                        'right': bool(right),
                        'book_minimum': minimum.tolist(),
                        'book_maximum': maximum.tolist(),
                        'unexpected_contacts': sorted(node.unexpected_contact_pairs),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if node.unexpected_contact_pairs:
                raise RuntimeError('contact search caused an unexpected collision')
            if left or right:
                contact_side = 'bilateral' if left and right else 'left' if left else 'right'
                contact_solution = solution
                break
            previous = solution

        terminal = emit('hook_contact_search_complete', node, nav, BOOK, baseline)
        _, _, terminal_minimum, terminal_maximum = physical_book_bounds()
        print(
            json.dumps(
                {
                    'event': 'result',
                    'contact_side': contact_side,
                    'contact_solution': (
                        None
                        if contact_solution is None
                        else np.asarray(contact_solution, dtype=float).tolist()
                    ),
                    'book_minimum': terminal_minimum.tolist(),
                    'book_maximum': terminal_maximum.tolist(),
                    'book_translation': float(np.linalg.norm(terminal[0] - baseline[0])),
                    'baseline_minimum': baseline_minimum.tolist(),
                    'baseline_maximum': baseline_maximum.tolist(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if contact_side is None:
            raise RuntimeError('hook contact not found within the calibrated limit')
    finally:
        node.track_unexpected_contacts = False
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
