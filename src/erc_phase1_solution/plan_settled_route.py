#!/usr/bin/env python3
"""Temporary live-state planner for the settled shelf/jaw cradle."""

import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from live_pick_retreat_probe import (
    BOOK,
    PickProbeNode,
    observed_attached_corners,
    settled_transition_is_safe,
)


def plan(node, start, corners, physical, targets):
    orientation = node.chain.forward(start)[:3, :3]
    previous = np.asarray(start, dtype=float)
    position = node.chain.forward(previous)[:3, 3]
    route = []
    for target in targets:
        target = np.asarray(target, dtype=float)
        for waypoint in node._interpolate_positions(
            position,
            target,
            0.020,
        ):
            solved, _ = node.chain.solve(
                pose_matrix(waypoint, orientation),
                [previous],
                position_tolerance=node.position_tolerance,
                orientation_tolerance=node.orientation_tolerance,
                max_iterations=320,
                fixed_positions={'torso_lift_joint': float(start[0])},
            )
            if solved is None:
                return None, f'ik:{waypoint.tolist()}'
            solved = np.asarray(solved, dtype=float)
            if not node._gravity_supported_transition_is_safe(previous, solved):
                return None, f'support:{waypoint.tolist()}'
            if not settled_transition_is_safe(
                node,
                previous,
                solved,
                corners,
                physical,
            ):
                return None, f'collision:{waypoint.tolist()}'
            route.append(solved)
            previous = solved
        position = target
    return route, ''


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = PickProbeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        start = node._measured_left_solution()
        corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        physical, _ = observed_attached_corners(node, 0.0)
        node._held_book_corners = corners
        node._gravity_supported_payload = True
        pose = node.chain.forward(start)
        origin = pose[:3, 3]
        print('START', start.tolist(), origin.tolist(), flush=True)
        candidates = []
        sequences = []
        for lift in (0.005, 0.010, 0.015, 0.020, 0.025, 0.030):
            sequences.append(
                (
                    f'direct_lift_{lift:.3f}',
                    [origin + np.asarray([0.0, 0.0, lift])],
                )
            )
        for label, targets in sequences:
            route, reason = plan(
                node,
                start,
                corners,
                physical,
                targets,
            )
            if not route:
                print('FAIL', label, reason, flush=True)
                continue
            terminal = route[-1]
            radius = node._carried_navigation_radius(terminal, corners)
            candidates.append(
                (
                    radius,
                    len(route),
                    label,
                    terminal,
                )
            )
            print(
                'PASS_NOW', label,
                'radius', radius,
                'legs', len(route),
                'terminal', terminal.tolist(),
                flush=True,
            )
        print('TOTAL', len(candidates), flush=True)
        for candidate in sorted(candidates)[:30]:
            radius, legs, label, terminal = candidate
            print(
                'PASS',
                'radius', radius,
                'legs', legs,
                'label', label,
                'terminal', terminal.tolist(),
                flush=True,
            )
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
