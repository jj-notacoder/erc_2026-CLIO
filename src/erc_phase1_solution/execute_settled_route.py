#!/usr/bin/env python3
"""Temporary same-world execution of the planned shelf-supported route."""

import argparse
import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.navigation_node import NavigationNode
from live_pick_retreat_probe import (
    BOOK,
    PickProbeNode,
    _rotation_distance,
    emit,
    observed_attached_corners,
)
from plan_settled_route import plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lift-only', action='store_true')
    args = parser.parse_args()
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

        node._clear_target_contact_samples(
            reset_robot_contact=True,
            reset_target_model=True,
        )
        with node._lock:
            node._target_book_model = BOOK
            node._transport_lock_engaged = True
            node._gravity_supported_payload = True
            node._payload_hazard_latched = None
            node._payload_monitor_enabled = False
        if not node._wait_sim_duration(0.25):
            raise RuntimeError('contact reacquisition wait failed')
        if not node._fresh_retention_probe(
            'probe',
            'settled_resume',
            leg=0,
        ):
            raise RuntimeError('settled resume has no lower-jaw contact')

        corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        physical, _ = observed_attached_corners(node, 0.0)
        with node._lock:
            node._held_book_corners = corners
            node._payload_monitor_enabled = True
        start = node._measured_left_solution()
        pose = node.chain.forward(start)
        origin = pose[:3, 3]
        targets = (
            [origin + np.asarray([0.0, 0.0, 0.020])]
            if args.lift_only
            else [
                np.asarray([0.60, origin[1], origin[2]], dtype=float),
                np.asarray([0.60, origin[1], origin[2] + 0.010], dtype=float),
                np.asarray([0.42, origin[1], origin[2] + 0.010], dtype=float),
            ]
        )
        route, reason = plan(node, start, corners, physical, targets)
        if not route:
            raise RuntimeError(f'settled route no longer plans: {reason}')
        print(
            json.dumps(
                {
                    'event': 'settled_route_ready',
                    'waypoints': len(route),
                    'start': start.tolist(),
                    'terminal': route[-1].tolist(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        baseline = emit('settled_route_start', node, nav, BOOK, None)
        node.track_unexpected_contacts = True

        legs = []
        previous = start
        for solution in route:
            legs.append(
                (
                    solution,
                    node._transport_leg_duration(previous, solution),
                    'supported_shelf_extraction',
                )
            )
            previous = solution
        goal, duration = node._make_retained_arm_trajectory_goal(legs)
        node._retention_probe_active = True
        node._payload_robot_watchdog_enabled = True
        terminal = None
        try:
            moved, _ = node._send_retained_arm_trajectory(
                goal,
                duration,
                legs,
                'probe',
            )
            if moved:
                terminal = node._wait_for_retained_endpoint(
                    route[-1],
                    command='probe',
                    phase='supported_shelf_extraction',
                    leg=len(route),
                )
        finally:
            node._retention_probe_active = False
            node._payload_robot_watchdog_enabled = False
        if not moved or terminal is None:
            raise RuntimeError('supported route controller execution failed')
        if not node._fresh_retention_probe(
            'probe',
            'supported_shelf_extraction',
            leg=len(route),
        ):
            raise RuntimeError('lower-jaw contact was lost after extraction')
        if not node._wait_sim_duration(1.0):
            raise RuntimeError('post-extraction dwell failed')
        after = emit('settled_route_complete', node, nav, BOOK, baseline)
        translation_drift = float(np.linalg.norm(after[0] - baseline[0]))
        rotation_drift = _rotation_distance(baseline[1], after[1])
        _, world = observed_attached_corners(node, 0.0)
        maximum_world_x = float(np.max(world[:, 0]))
        minimum_world_z = float(np.min(world[:, 2]))
        if args.lift_only:
            result = {
                'event': 'result',
                'passed': bool(
                    translation_drift <= 0.020
                    and rotation_drift <= math.radians(10.0)
                    and minimum_world_z >= 1.461
                    and not node._target_robot_contact_latched
                    and not node.unexpected_contact_pairs
                ),
                'lift_only': True,
                'translation_drift_m': translation_drift,
                'rotation_drift_rad': rotation_drift,
                'minimum_world_z': minimum_world_z,
                'left': bool(node._target_contact_sides(max_age=0.15)[0]),
                'unexpected_contacts': [
                    list(pair) for pair in sorted(node.unexpected_contact_pairs)
                ],
            }
            print(json.dumps(result, sort_keys=True), flush=True)
            if not result['passed']:
                raise RuntimeError('direct support-lift probe failed')
            return
        if (
            translation_drift > 0.020
            or rotation_drift > math.radians(10.0)
            or maximum_world_x > 2.750
            or node.unexpected_contact_pairs
        ):
            raise RuntimeError(
                'supported route failed its pose/contact gate: '
                f'{translation_drift:.6f} m, {rotation_drift:.6f} rad, '
                f'max_x={maximum_world_x:.6f}'
            )

        retreat_start = np.asarray(nav.pose, dtype=float)
        goal_pose = (
            float(retreat_start[0] - 0.35 * math.cos(retreat_start[2])),
            float(retreat_start[1] - 0.35 * math.sin(retreat_start[2])),
            float(retreat_start[2]),
        )
        nav._accept_goal(goal_pose, profile='carried_retreat')
        first_loss = None
        while nav.goal is not None:
            if node._payload_hazard_latched is not None:
                first_loss = str(node._payload_hazard_latched)
                nav._finish_goal('cancelled')
                break
            time.sleep(0.02)
        if not node._wait_sim_duration(0.5):
            raise RuntimeError('retreat settle failed')
        final = emit('settled_route_retreat_complete', node, nav, BOOK, after)
        final_translation = float(np.linalg.norm(final[0] - after[0]))
        final_rotation = _rotation_distance(after[1], final[1])
        displacement = float(
            math.hypot(
                nav.pose[0] - retreat_start[0],
                nav.pose[1] - retreat_start[1],
            )
        )
        left, _ = node._target_contact_sides(max_age=0.15)
        result = {
            'event': 'result',
            'passed': bool(
                first_loss is None
                and displacement >= 0.34
                and left
                and final_translation <= 0.020
                and final_rotation <= math.radians(10.0)
                and not node._target_robot_contact_latched
                and not node.unexpected_contact_pairs
            ),
            'displacement': displacement,
            'first_loss': first_loss,
            'left': bool(left),
            'translation_drift_m': final_translation,
            'rotation_drift_rad': final_rotation,
            'unexpected_contacts': [
                list(pair) for pair in sorted(node.unexpected_contact_pairs)
            ],
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if not result['passed']:
            raise RuntimeError('settled route/retreat probe failed')
    finally:
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
