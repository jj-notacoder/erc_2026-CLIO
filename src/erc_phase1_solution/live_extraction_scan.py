#!/usr/bin/env python3
"""Temporary real-shelf per-leg extraction pose scan; remove before commit."""

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import (
    BASE_POSITION,
    BASE_YAW,
    BOOK,
    PickProbeNode,
    emit,
    set_model_pose,
)


class ScanNode(PickProbeNode):
    def _execute_retained_arm_legs(
        self,
        legs,
        command,
        *,
        fresh_retention_phases=(),
        leg_offset=0,
    ):
        for index, (solution, duration, phase) in enumerate(legs):
            if not self._move_arm_solution(solution, duration):
                return False, index, False
            if not self._wait_sim_duration(0.10):
                return False, index, False
            position, quaternion, minimum, maximum = physical_book_bounds()
            left, right = self._target_contact_sides(max_age=0.15)
            print(
                json.dumps(
                    {
                        'event': 'extraction_leg',
                        'leg': index + leg_offset,
                        'phase': phase,
                        'solution': np.asarray(solution, dtype=float).tolist(),
                        'book_position_world': position.tolist(),
                        'book_minimum_world': minimum.tolist(),
                        'book_maximum_world': maximum.tolist(),
                        'left': bool(left),
                        'right': bool(right),
                        'robot_contact_latched': bool(
                            self._target_robot_contact_latched
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            retained = self._retention_after_leg(
                command,
                phase,
                index + leg_offset,
            )
            if not retained:
                return False, index + 1, True
        return True, len(legs), False


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = ScanNode()
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
        if len(node.joints) < 8:
            raise RuntimeError('live state unavailable')
        node.gripper_transport_lock = 0.029
        node.top_row_grasp_vertical_offset = -0.015
        node.skip_return_preflight = True
        if not node._stow():
            raise RuntimeError('stow failed')
        quaternion = np.asarray(
            [0.0, 0.0, math.sin(BASE_YAW / 2.0), math.cos(BASE_YAW / 2.0)]
        )
        set_model_pose('tiago_pro', BASE_POSITION, quaternion)
        if not node._wait_sim_duration(0.40):
            raise RuntimeError('base did not settle')
        picked = node._pick()
        emit('extraction_scan_complete', node, nav, BOOK, None)
        print(json.dumps({'event': 'result', 'picked': bool(picked)}), flush=True)
    finally:
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
