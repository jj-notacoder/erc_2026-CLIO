#!/usr/bin/env python3
"""Temporary live probe for the deferred top-row cradle; remove before commit."""

from __future__ import annotations

from types import SimpleNamespace
import importlib.util
import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_retreat_probe import (
    ProbeManipulationNode,
    emit,
    quaternion_matrix,
    set_model_pose,
)


BOOK = 'book_col_3_row_2_red'
FRONT = np.asarray(
    [0.6937779125, -0.05493238067, 1.58418506676],
    dtype=float,
)
BASE_POSITION = np.asarray([1.8183, -0.09362, 0.0], dtype=float)
BASE_YAW = 0.0


class VerboseProbeNode(ProbeManipulationNode):
    def _publish_status(self, event, **fields):
        print(
            json.dumps({'event': f'status:{event}', **fields}, sort_keys=True),
            flush=True,
        )
        return super()._publish_status(event, **fields)


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return an x/y/z/w unit quaternion for a proper rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=float,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=float,
            )
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=float,
            )
    return quaternion / np.linalg.norm(quaternion)


def _test_helpers():
    spec = importlib.util.spec_from_file_location(
        'test_shutdown_probe',
        '/opt/erc_ws/src/erc_phase1_solution/test/test_shutdown.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = VerboseProbeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    nav = SimpleNamespace(pose=None, last_command=SimpleNamespace())
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('joint states unavailable')

        # Reproduce the mission's parked-right-arm geometry before planning.
        set_model_pose(
            BOOK,
            np.asarray([4.0, 2.0, 0.08], dtype=float),
            np.asarray(
                [0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)],
                dtype=float,
            ),
        )
        if not node._stow():
            raise RuntimeError('stow failed')

        helpers = _test_helpers()
        (
            _grasp,
            _positions,
            rotations,
            solutions,
            orientation_index,
            _score,
            _transition,
        ) = helpers._top_row_pick_plan(node, FRONT)
        node._plan_carried_return(
            FRONT,
            solutions[-1],
            solutions[1],
            rotations[orientation_index],
            0.35,
        )
        cached_plan = node._cached_post_retreat_plan
        if not isinstance(cached_plan, dict):
            raise RuntimeError('deferred plan was not cached')
        cached_start = np.asarray(cached_plan['start'], dtype=float)
        attached = np.asarray(cached_plan['attached_corners'], dtype=float)

        base_quaternion = np.asarray(
            [0.0, 0.0, math.sin(BASE_YAW / 2.0), math.cos(BASE_YAW / 2.0)],
            dtype=float,
        )
        set_model_pose('tiago_pro', BASE_POSITION, base_quaternion)
        node._wait_sim_duration(0.30)
        if not node._move_torso(float(cached_start[0]), 2.5):
            raise RuntimeError('pick-height torso move failed')
        torso_deadline = time.monotonic() + 120.0
        while (
            abs(
                float(node._measured_left_solution()[0] - cached_start[0])
            )
            > 0.001
            and time.monotonic() < torso_deadline
        ):
            time.sleep(0.05)
        if (
            abs(float(node._measured_left_solution()[0] - cached_start[0]))
            > 0.001
        ):
            raise RuntimeError('pick-height torso failed to settle')
        if not node._move_arm_solution(cached_start, 2.0):
            raise RuntimeError('vertical loaded-clearance move failed')
        if not node._command_gripper(0.035):
            raise RuntimeError('direct preclose failed')

        grasp_pose = node.chain.forward(cached_start)
        centre_in_grasp = np.mean(attached, axis=0)
        centre_base = (
            centre_in_grasp @ grasp_pose[:3, :3].T
            + grasp_pose[:3, 3]
        )
        centre_world = (
            BASE_POSITION
            + quaternion_matrix(base_quaternion) @ centre_base
        )
        book_rotation_at_grasp = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        grasp_rotation = node.chain.forward(solutions[-1])[:3, :3]
        book_in_grasp_rotation = (
            grasp_rotation.T @ book_rotation_at_grasp
        )
        book_rotation_base = (
            grasp_pose[:3, :3] @ book_in_grasp_rotation
        )
        book_quaternion_world = _quaternion_from_matrix(
            quaternion_matrix(base_quaternion) @ book_rotation_base
        )
        node._payload_monitor_enabled = False
        node._clear_target_contact_samples(
            reset_robot_contact=True,
            reset_target_model=True,
        )
        with node._lock:
            node._target_book_model = BOOK
        set_model_pose(
            BOOK,
            centre_world,
            book_quaternion_world,
        )
        node._wait_sim_duration(0.02)
        if not node._command_gripper(0.029):
            raise RuntimeError('transport lock failed')
        node._transport_lock_engaged = True
        node._clear_target_contact_samples(reset_robot_contact=True)
        node._wait_sim_duration(0.25)
        baseline = emit('locked', node, nav, BOOK, None, command=0.029)
        verified, width, left, right, plausible = node._pinch_sample(
            max_age=0.15
        )
        if not (verified and left and right and plausible):
            raise RuntimeError(
                f'initial bilateral lock failed: {width}, {left}, {right}'
            )

        with node._lock:
            node._held_book_corners = attached.copy()
            node._cached_post_retreat_plan = cached_plan
            node._carried_staging_solution = cached_start.copy()
            node._post_retreat_shelf_front_x = float(
                cached_plan['shelf_front_x']
            )
            node._gravity_supported_payload = False
            node._supported_post_retreat_staging_required = True
            node._payload_hazard_latched = None
            node._payload_monitor_enabled = True

        measured_start = node._measured_left_solution()
        print(
            json.dumps(
                {
                    'event': 'dispatch_gate',
                    'cached_start': cached_start.tolist(),
                    'measured_start': measured_start.tolist(),
                    'torso_error': float(measured_start[0] - cached_start[0]),
                    'maximum_arm_error': float(
                        np.max(np.abs(measured_start[1:] - cached_start[1:]))
                    ),
                    'shelf_front_x': float(node._post_retreat_shelf_front_x),
                    'cached_shelf_front_x': float(cached_plan['shelf_front_x']),
                    'compact_radius': float(cached_plan['compact_radius']),
                    'radius_limit': float(node.carried_navigation_radius_limit),
                    'corner_match': bool(
                        np.allclose(
                            node._held_book_corners,
                            cached_plan['attached_corners'],
                            rtol=0.0,
                            atol=1e-12,
                        )
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        succeeded = node._compact_transport()
        node._wait_sim_duration(0.30)
        emit(
            'compact_complete',
            node,
            nav,
            BOOK,
            baseline,
            succeeded=bool(succeeded),
        )
        verified, width, left, right, plausible = node._pinch_sample(
            max_age=0.15
        )
        result = {
            'event': 'result',
            'passed': bool(
                succeeded
                and verified
                and left
                and plausible
                and not node._target_robot_contact_latched
                and node._gravity_supported_payload
            ),
            'succeeded': bool(succeeded),
            'verified': bool(verified),
            'width': float(width),
            'left': bool(left),
            'right': bool(right),
            'plausible': bool(plausible),
            'robot_contact_latched': bool(node._target_robot_contact_latched),
            'gravity_supported': bool(node._gravity_supported_payload),
            'cached_waypoints': len(cached_plan['legs']),
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if not result['passed']:
            raise RuntimeError('deferred cradle live probe failed')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
