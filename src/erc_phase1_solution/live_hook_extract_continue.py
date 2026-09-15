#!/usr/bin/env python3
"""Temporary same-world shelf-supported hook extraction; remove before commit."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_lift_pull_continue import LIFT_PULL
from live_lower_shelf_rescue import (
    RescueProbeNode,
    bounds_payload,
    retained_arm_leg,
)
from live_pick_retreat_probe import (
    BOOK,
    _rotation_distance,
    observed_attached_corners,
    quaternion_matrix,
)


Q_EXT = np.asarray(
    [
        0.3499999954440325,
        0.6943560328120175,
        1.0020372683310048,
        -0.06441115695003245,
        -1.6789008214877463,
        0.2942356082976997,
        1.6309414247928256,
        -0.443853762084003,
    ]
)
Q_DOWN117 = np.asarray(
    [
        0.3499999954440325,
        0.802154942936009,
        0.7428894229132217,
        -0.0029350068638443094,
        -2.043767745344703,
        0.39451955647937825,
        1.7622103764357544,
        -0.5076604195349552,
    ]
)
Q_ROLL = Q_DOWN117.copy()
Q_ROLL[-1] = -2.007660419534955
Q_TOUCH1 = np.asarray(
    [
        0.3499999954440325,
        0.8012520103631988,
        0.7450417755495916,
        -0.0036989905465110154,
        -2.041072585875235,
        0.39347079247157624,
        1.761330237592195,
        -2.006930543205059,
    ]
)
Q_UP10 = np.asarray(
    [
        0.3499999954440325,
        0.7931660776048669,
        0.7644381375336237,
        -0.010416899969261668,
        -2.016508744585927,
        0.3842453486763206,
        1.7532230056140923,
        -2.0004835211392473,
    ]
)
Q_OUT60_SUPPORT = np.asarray(
    [
        0.3499999954440325,
        0.8037438377842102,
        0.8773021763697442,
        0.003609340595114341,
        -2.07444678895004,
        0.4153192492035906,
        1.9166409015179318,
        -1.9467053720099061,
    ]
)


def fresh_contacts(node, seconds=0.16):
    node._clear_target_contact_samples()
    if not node._wait_sim_duration(seconds):
        raise RuntimeError('contact sampling dwell was interrupted')
    return node._target_contact_sides(max_age=0.15)


def stable_shelf_gate(node, baseline_position, baseline_rotation, phase):
    current = physical_book_bounds()
    translation = float(np.linalg.norm(current[0] - baseline_position))
    rotation = _rotation_distance(
        baseline_rotation,
        quaternion_matrix(current[1]),
    )
    left, right = node._target_contact_sides(max_age=0.15)
    print(
        json.dumps(
            {
                'event': phase,
                'position': current[0].tolist(),
                'minimum': current[2].tolist(),
                'maximum': current[3].tolist(),
                'translation_m': translation,
                'rotation_rad': rotation,
                'left': bool(left),
                'right': bool(right),
                'solution': node._measured_left_solution().tolist(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if (
        translation > 0.003
        or rotation > math.radians(5.0)
        or node._target_robot_contact_latched
        or node.neighbor_contact_pairs
    ):
        raise RuntimeError(f'shelf-supported book moved during {phase}')
    return current


def move_shelf_guided(node, target, duration, phase):
    if node._robot_self_collision(target) is not None:
        raise RuntimeError(f'self collision during {phase}')
    if not node._move_arm_solution(target, duration):
        raise RuntimeError(f'controller failure during {phase}')
    left, right = fresh_contacts(node)
    if not (left or right):
        raise RuntimeError(f'all target contact was lost during {phase}')
    if node._target_robot_contact_latched:
        raise RuntimeError(f'payload touched the robot during {phase}')
    return left, right


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = RescueProbeNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('joint state unavailable')

        node.gripper_transport_lock = 0.029
        node._target_book_model = BOOK
        node._transport_lock_engaged = True
        node._gravity_supported_payload = False
        node._payload_hazard_latched = None
        node._target_robot_contact_latched = False
        corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        node._held_book_corners = corners
        node._payload_monitor_enabled = True
        if not node._fresh_retention_probe('bottom_hook', 'resume', leg=0):
            raise RuntimeError('same-world bilateral grasp is unavailable')

        # The previous exact-lift experiment stopped at L8.  Reverse through
        # L4 to Q_EXT, letting the shelf settle the book back to its starting
        # upright pose before the intentional guided slide.
        for index, target in enumerate((LIFT_PULL[0][0], Q_EXT), start=1):
            corners, _ = observed_attached_corners(
                node,
                node.carried_book_padding,
            )
            node._held_book_corners = corners
            retained_arm_leg(
                node,
                np.asarray(target, dtype=float),
                0.62,
                'restore_shelf_support',
                index,
            )
        if not node._wait_sim_duration(0.30):
            raise RuntimeError('restore dwell failed')
        baseline = physical_book_bounds()
        baseline_position = baseline[0]
        baseline_rotation = quaternion_matrix(baseline[1])
        stable_shelf_gate(
            node,
            baseline_position,
            baseline_rotation,
            'restored_q_ext',
        )

        # During the shelf-guided slide, one jaw intentionally clears the book
        # for part of the path.  Suppress the bilateral pinch watchdog while
        # still requiring a fresh contact from at least one target finger at
        # every small endpoint and keeping robot-contact latching active.
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        start = node._measured_left_solution()
        start_pose = node.chain.forward(start)
        previous = start
        total_steps = 12
        for index in range(1, total_steps + 1):
            fraction = index / total_steps
            target_position = start_pose[:3, 3].copy()
            target_position[2] -= 0.117 * fraction
            solution, _ = node.chain.solve(
                pose_matrix(target_position, start_pose[:3, :3]),
                [previous],
                position_tolerance=0.00025,
                orientation_tolerance=0.002,
                max_iterations=300,
                fixed_positions={'torso_lift_joint': float(start[0])},
            )
            if solution is None:
                raise RuntimeError(f'bottom slide IK failed at step {index}')
            solution = np.asarray(solution, dtype=float)
            move_shelf_guided(
                node,
                solution,
                0.55,
                f'bottom_slide_{index}',
            )
            stable_shelf_gate(
                node,
                baseline_position,
                baseline_rotation,
                f'bottom_slide_{index}',
            )
            previous = solution

        if float(np.max(np.abs(previous - Q_DOWN117))) > 0.02:
            raise RuntimeError('bottom slide missed the audited terminal branch')

        # Rotate only q7 in four cage-preserving steps.  Negative roll moves
        # the physical left fingertip below the overhanging bottom while the
        # right finger continues guiding the vertical side.
        roll_start = node._measured_left_solution()
        for index in range(1, 5):
            target = roll_start.copy()
            target[-1] = roll_start[-1] - 1.5 * (index / 4.0)
            move_shelf_guided(node, target, 0.58, f'hook_roll_{index}')
            stable_shelf_gate(
                node,
                baseline_position,
                baseline_rotation,
                f'hook_roll_{index}',
            )

        if float(np.max(np.abs(node._measured_left_solution() - Q_ROLL))) > 0.02:
            raise RuntimeError('hook roll missed the audited terminal branch')

        # Rise no more than 1 mm in 0.25 mm increments and stop on the first
        # fresh left-under + right-side contact pair.
        touch_start = node._measured_left_solution()
        touched = False
        for index in range(1, 5):
            target = touch_start + (Q_TOUCH1 - touch_start) * (index / 4.0)
            left, right = move_shelf_guided(
                node,
                target,
                0.38,
                f'hook_touch_{index}',
            )
            stable_shelf_gate(
                node,
                baseline_position,
                baseline_rotation,
                f'hook_touch_{index}',
            )
            if left and right:
                touched = True
                break
        if not touched:
            raise RuntimeError('left-under/right-side hook contact was not established')

        corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        node._held_book_corners = corners
        node._gravity_supported_payload = True
        node._retention_probe_active = False
        node._payload_monitor_enabled = True
        if not node._fresh_retention_probe('bottom_hook', 'hook_loaded', leg=20):
            raise RuntimeError('lower fingertip did not accept the load')
        left, right = node._target_contact_sides(max_age=0.15)
        if not (left and right):
            raise RuntimeError('hook load lacks the right-side cage contact')

        # Lift 10 mm and then pull 60 mm with the lower finger carrying weight.
        lift_start = node._measured_left_solution()
        lift_bounds = physical_book_bounds()
        for index in range(1, 4):
            target = lift_start + (Q_UP10 - lift_start) * (index / 3.0)
            if not node._gravity_supported_transition_is_safe(
                node._measured_left_solution(), target
            ):
                raise RuntimeError('hook lift lost geometric support in preflight')
            corners, _ = observed_attached_corners(
                node,
                node.carried_book_padding,
            )
            node._held_book_corners = corners
            retained_arm_leg(node, target, 0.55, 'hook_lift', 20 + index)
            current = bounds_payload(f'hook_lift_{index}_bounds')
            if float(current[2][2] - lift_bounds[2][2]) < 0.0015:
                raise RuntimeError('book did not rise with the loaded hook')
            lift_bounds = current

        pull_start = node._measured_left_solution()
        pull_bounds = physical_book_bounds()
        for index in range(1, 4):
            target = pull_start + (
                Q_OUT60_SUPPORT - pull_start
            ) * (index / 3.0)
            if not node._gravity_supported_transition_is_safe(
                node._measured_left_solution(), target
            ):
                raise RuntimeError('hook pull lost geometric support in preflight')
            corners, _ = observed_attached_corners(
                node,
                node.carried_book_padding,
            )
            node._held_book_corners = corners
            retained_arm_leg(node, target, 0.65, 'supported_pull', 24 + index)
            current = bounds_payload(f'supported_pull_{index}_bounds')
            if (
                float(current[0][0] - pull_bounds[0][0]) > -0.015
                or abs(float(current[0][2] - pull_bounds[0][2])) > 0.003
            ):
                raise RuntimeError('book did not track the supported pull')
            pull_bounds = current

        if float(pull_bounds[3][0]) > 2.725:
            raise RuntimeError('supported hook pull did not clear the shelf lip')
        if not node._wait_sim_duration(0.50):
            raise RuntimeError('supported extraction dwell failed')
        stable = bounds_payload('supported_extraction_stable')
        left, _ = node._target_contact_sides(max_age=0.15)
        if (
            not left
            or np.linalg.norm(stable[0] - pull_bounds[0]) > 0.002
            or _rotation_distance(
                quaternion_matrix(stable[1]),
                quaternion_matrix(pull_bounds[1]),
            )
            > math.radians(3.0)
            or node._target_robot_contact_latched
        ):
            raise RuntimeError('supported extraction did not remain stable')

        print(
            json.dumps(
                {
                    'event': 'result',
                    'passed': True,
                    'stage': 'supported_extraction',
                    'solution': node._measured_left_solution().tolist(),
                    'left': bool(left),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    'event': 'result',
                    'passed': False,
                    'error': f'{type(exc).__name__}: {exc}',
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    finally:
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
