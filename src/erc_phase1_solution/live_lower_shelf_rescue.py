#!/usr/bin/env python3
"""Temporary live top-to-lower-shelf rescue probe; remove before commit."""

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
from live_pick_retreat_probe import (
    BASE_POSITION,
    BASE_YAW,
    BOOK,
    PickProbeNode,
    _rotation_distance,
    observed_attached_corners,
    quaternion_matrix,
    set_model_pose,
)


SHELF_FRONT_X = 2.755012
LOWER_FLOOR_Z = 1.121858
LOWER_CEILING_Z = 1.451858

Q_OUT = np.asarray(
    [
        0.35,
        0.6976965371,
        1.0927597449,
        -0.0575737442,
        -1.7368829347,
        0.3086309108,
        1.7748700580,
        -0.4069247503,
    ],
    dtype=float,
)
STAGE_TORSO = 0.25988
DOWN1 = np.asarray(
    [
        STAGE_TORSO,
        0.8000798785,
        0.8632861834,
        0.0209064889,
        -2.0952286779,
        0.4166281693,
        1.9297945755,
        -0.4578918082,
    ],
    dtype=float,
)
DOWN2 = np.asarray(
    [
        STAGE_TORSO,
        0.9075679899,
        0.6447975649,
        0.1670872022,
        -2.3656946133,
        0.5941396785,
        2.0374632163,
        -0.5411078738,
    ],
    dtype=float,
)
IN1 = np.asarray(
    [
        STAGE_TORSO,
        0.8570683150,
        0.4578206506,
        0.1449028627,
        -2.2697911427,
        0.5151234682,
        1.7686656274,
        -0.6533912812,
    ],
    dtype=float,
)
IN2 = np.asarray(
    [
        STAGE_TORSO,
        0.8126644759,
        0.3521836242,
        0.1327922235,
        -2.1077610798,
        0.4756764075,
        1.5116054253,
        -0.7437816905,
    ],
    dtype=float,
)


class RescueProbeNode(PickProbeNode):
    def __init__(self):
        self.shelf_contact_ns = 0
        self.neighbor_contact_pairs = set()
        super().__init__()

    def _on_contacts(self, message):
        super()._on_contacts(message)
        now_ns = self.get_clock().now().nanoseconds
        for contact in getattr(message, 'contacts', []):
            first = str(
                getattr(getattr(contact, 'collision1', None), 'name', '')
            )
            second = str(
                getattr(getattr(contact, 'collision2', None), 'name', '')
            )
            lowered = f'{first} {second}'.lower()
            if BOOK in lowered and 'erc_shelf' in lowered:
                self.shelf_contact_ns = now_ns
            if BOOK in lowered and 'book_col_' in lowered:
                other_books = [
                    token
                    for token in (first, second)
                    if 'book_col_' in token.lower() and BOOK not in token.lower()
                ]
                if other_books:
                    self.neighbor_contact_pairs.add(tuple(sorted((first, second))))


def bounds_payload(event: str):
    position, quaternion, minimum, maximum = physical_book_bounds()
    payload = {
        'event': event,
        'position': position.tolist(),
        'quaternion': quaternion.tolist(),
        'minimum': minimum.tolist(),
        'maximum': maximum.tolist(),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return position, quaternion, minimum, maximum


def require_retention(node: RescueProbeNode, phase: str, leg: int) -> None:
    if not node._fresh_retention_probe('lower_shelf_rescue', phase, leg=leg):
        raise RuntimeError(f'payload retention failed after {phase}')
    if node._target_robot_contact_latched:
        raise RuntimeError(f'payload touched the robot during {phase}')
    if node.neighbor_contact_pairs:
        raise RuntimeError(f'neighbor contact during {phase}')


def fresh_shelf_contact(node: RescueProbeNode, maximum_age: float = 0.15) -> bool:
    now_ns = node.get_clock().now().nanoseconds
    return bool(
        node.shelf_contact_ns > 0
        and -100_000_000
        <= now_ns - node.shelf_contact_ns
        <= int(maximum_age * 1e9)
    )


def retained_arm_leg(
    node: RescueProbeNode,
    target: np.ndarray,
    duration: float,
    phase: str,
    leg: int,
) -> None:
    start = node._measured_left_solution()
    corners = node._held_book_corners
    if corners is None:
        raise RuntimeError('retained payload envelope is unavailable')
    if not node._carried_robot_transition_is_safe(start, target, corners):
        raise RuntimeError(f'live carried-volume guard rejected {phase}')
    legs = [(target, duration, phase)]
    goal, total = node._make_retained_arm_trajectory_goal(legs)
    node._payload_robot_watchdog_enabled = True
    try:
        moved, contact_fault = node._send_retained_arm_trajectory(
            goal,
            total,
            legs,
            'lower_shelf_rescue',
        )
        endpoint = (
            node._wait_for_retained_endpoint(
                target,
                command='lower_shelf_rescue',
                phase=phase,
                leg=leg,
            )
            if moved
            else None
        )
    finally:
        node._payload_robot_watchdog_enabled = False
    if not moved or endpoint is None:
        raise RuntimeError(
            f'{phase} failed (contact_fault={bool(contact_fault)})'
        )
    require_retention(node, phase, leg)


def solve_cartesian_shift(
    node: RescueProbeNode,
    shift: np.ndarray,
) -> np.ndarray:
    measured = node._measured_left_solution()
    current_pose = node.chain.forward(measured)
    target_position = current_pose[:3, 3] + np.asarray(shift, dtype=float)
    solution, _ = node.chain.solve(
        pose_matrix(target_position, current_pose[:3, :3]),
        [measured],
        position_tolerance=node.position_tolerance,
        orientation_tolerance=node.orientation_tolerance,
        max_iterations=240,
        fixed_positions={'torso_lift_joint': float(measured[0])},
    )
    if solution is None:
        raise RuntimeError(f'Cartesian shift IK failed for {shift.tolist()}')
    return np.asarray(solution, dtype=float)


def main() -> None:
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = RescueProbeNode()
    nav = NavigationNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    threading.Thread(target=executor.spin, daemon=True).start()
    passed = False
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

        node.gripper_transport_lock = 0.029
        node.top_row_grasp_vertical_offset = -0.015
        node.skip_return_preflight = True

        if not node._stow():
            raise RuntimeError('stow failed')
        base_quaternion = np.asarray(
            [0.0, 0.0, math.sin(BASE_YAW / 2.0), math.cos(BASE_YAW / 2.0)],
            dtype=float,
        )
        set_model_pose('tiago_pro', BASE_POSITION, base_quaternion)
        if not node._wait_sim_duration(0.40):
            raise RuntimeError('base teleport did not settle')
        if not node._pick():
            raise RuntimeError('real top-row pick failed')
        require_retention(node, 'full_extraction', 0)
        start_position, start_quaternion, _, _ = bounds_payload(
            'full_extraction_bounds'
        )
        start_rotation = quaternion_matrix(start_quaternion)

        # Replace the ideal rigid transform with the physically observed one
        # for this diagnostic.  The live grasp has consistently slipped deeper
        # than the nominal transform during extraction.
        # The extracted book has settled about 4 mm into the shelf plane in
        # live physics.  Lift it clear in three small increments before the
        # horizontal pull so the weak vertical pinch does not have to overcome
        # shelf friction and gravity simultaneously.
        lift_previous_position = start_position
        for index in range(4):
            live_corners, _ = observed_attached_corners(
                node,
                node.carried_book_padding,
            )
            node._held_book_corners = live_corners
            lifted = solve_cartesian_shift(
                node,
                np.asarray([0.0, 0.0, 0.003], dtype=float),
            )
            retained_arm_leg(
                node,
                lifted,
                0.55,
                'top_bay_lift',
                1 + index,
            )
            lift_measurement = bounds_payload(
                f'top_bay_lift_{index + 1}_bounds'
            )
            if float(lift_measurement[0][2] - lift_previous_position[2]) < 0.0015:
                raise RuntimeError('book slipped instead of tracking the lift')
            lift_previous_position = lift_measurement[0]
        _, _, lifted_minimum, lifted_maximum = bounds_payload(
            'top_bay_lift_complete'
        )
        if (
            float(lifted_minimum[2]) < 1.454858
            or float(lifted_maximum[2]) > 1.731858
        ):
            raise RuntimeError('lifted book did not clear the top-bay aperture')

        # Pull 60 mm in three 20 mm fixed-orientation legs while elevated.
        # Re-anchor the diagnostic envelope after each leg so the next safety
        # gate follows any small physical settling instead of an ideal model.
        for index in range(3):
            live_corners, _ = observed_attached_corners(
                node,
                node.carried_book_padding,
            )
            node._held_book_corners = live_corners
            pulled = solve_cartesian_shift(
                node,
                np.asarray([-0.020, 0.0, 0.0], dtype=float),
            )
            retained_arm_leg(
                node,
                pulled,
                0.70,
                'elevated_pull_out',
                5 + index,
            )
            bounds_payload(f'elevated_pull_{index + 1}_bounds')

        # Return to the exact audited OUT60 height only after the payload is
        # outside the shelf lip, then continue with the static lower-bay route.
        live_corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        node._held_book_corners = live_corners
        retained_arm_leg(node, Q_OUT, 0.65, 'outside_lower_to_out60', 8)
        _, _, _, out_maximum = bounds_payload('nominal_out_bounds')
        extra_index = 3
        if float(out_maximum[0]) > 2.725:
            raise RuntimeError('book did not clear the upper shelf lip')
        require_retention(node, 'outward_clearance_dwell', 5)
        node.shelf_contact_ns = 0

        # Lower the torso in short retained increments while the book is fully
        # outside the shelf.  Each endpoint must re-establish a fresh bilateral
        # contact sample before the next step.
        torso_start = float(node._measured_left_solution()[0])
        for index, height in enumerate(
            np.linspace(torso_start, STAGE_TORSO, 6)[1:],
            start=6,
        ):
            if not node._move_torso(float(height), 0.70):
                raise RuntimeError('torso descent failed')
            if fresh_shelf_contact(node):
                raise RuntimeError('premature shelf contact during torso descent')
            require_retention(node, 'outside_torso_descent', index)
        bounds_payload('outside_torso_complete')

        # Re-anchor again after the vertical move, then use the densely audited
        # lower-bay arm route.  DOWN1 and IN1 are required safety waypoints.
        live_corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        node._held_book_corners = live_corners
        for index, (target, phase, duration) in enumerate(
            (
                (DOWN1, 'lower_down_1', 1.15),
                (DOWN2, 'lower_down_2', 1.15),
                (IN1, 'lower_in_1', 1.10),
                (IN2, 'lower_in_2', 1.10),
            ),
            start=20,
        ):
            if phase == 'lower_in_1':
                _, _, minimum, maximum = bounds_payload('pre_insertion_bounds')
                if (
                    float(minimum[2]) < LOWER_FLOOR_Z + 0.003
                    or float(maximum[2]) > LOWER_CEILING_Z - 0.030
                ):
                    raise RuntimeError('book is outside the lower-bay aperture')
            retained_arm_leg(node, target, duration, phase, index)
            if fresh_shelf_contact(node):
                raise RuntimeError(f'premature shelf contact during {phase}')
            bounds_payload(f'{phase}_bounds')

        position, quaternion, minimum, maximum = bounds_payload(
            'lower_insertion_complete'
        )
        if float(position[0]) < SHELF_FRONT_X + 0.020:
            raise RuntimeError('book COM is not safely inside the lower shelf')
        if float(minimum[2]) < LOWER_FLOOR_Z + 0.002:
            raise RuntimeError('book reached the shelf before the seating probe')

        node.shelf_contact_ns = 0
        seat_contact_ns = 0
        seat_height = float(node._measured_left_solution()[0])
        for index in range(28):
            seat_height -= 0.00025
            if not node._move_torso(seat_height, 0.18):
                raise RuntimeError('micro seating move failed')
            if fresh_shelf_contact(node):
                seat_contact_ns = node.shelf_contact_ns
                if node._target_robot_contact_latched:
                    raise RuntimeError('payload touched the robot while seating')
                if node.neighbor_contact_pairs:
                    raise RuntimeError('neighbor contact while seating')
                break
            require_retention(node, 'micro_seat', 30 + index)
        if seat_contact_ns <= 0:
            raise RuntimeError('no fresh lower-shelf contact was observed')

        seat_first = bounds_payload('seat_first_contact')
        if not node._wait_sim_duration(0.45):
            raise RuntimeError('seat stability dwell failed')
        seat_stable = bounds_payload('seat_stable')
        seat_translation = float(np.linalg.norm(seat_stable[0] - seat_first[0]))
        seat_rotation = _rotation_distance(
            quaternion_matrix(seat_first[1]),
            quaternion_matrix(seat_stable[1]),
        )
        total_rotation = _rotation_distance(
            start_rotation,
            quaternion_matrix(seat_stable[1]),
        )
        if (
            seat_translation > 0.001
            or seat_rotation > math.radians(2.0)
            or total_rotation > math.radians(3.0)
            or float(seat_stable[0][0]) < SHELF_FRONT_X + 0.020
            or node._target_robot_contact_latched
            or node.neighbor_contact_pairs
        ):
            raise RuntimeError('lower-shelf seat failed the stability gate')

        if not node._open_gripper():
            raise RuntimeError('release on the lower shelf failed')
        if not node._wait_sim_duration(0.35):
            raise RuntimeError('post-release dwell failed')
        release_first = bounds_payload('release_first_dwell')
        if not node._wait_sim_duration(0.50):
            raise RuntimeError('post-release stability dwell failed')
        release_stable = bounds_payload('release_stable')
        release_translation = float(
            np.linalg.norm(release_stable[0] - release_first[0])
        )
        release_rotation = _rotation_distance(
            quaternion_matrix(release_first[1]),
            quaternion_matrix(release_stable[1]),
        )
        if release_translation > 0.001 or release_rotation > math.radians(2.0):
            raise RuntimeError('released book is not stable on the lower shelf')

        # Withdraw the now-open gripper without disturbing the staged book.
        for target, phase in ((IN1, 'withdraw_1'), (DOWN2, 'withdraw_2')):
            if not node._move_arm_solution(target, 0.90):
                raise RuntimeError(f'{phase} failed')
        if not node._wait_sim_duration(0.35):
            raise RuntimeError('withdrawal dwell failed')
        withdrawn = bounds_payload('withdrawn_stable')
        withdrawal_translation = float(
            np.linalg.norm(withdrawn[0] - release_stable[0])
        )
        withdrawal_rotation = _rotation_distance(
            quaternion_matrix(release_stable[1]),
            quaternion_matrix(withdrawn[1]),
        )
        if (
            withdrawal_translation > 0.0015
            or withdrawal_rotation > math.radians(2.0)
            or node.neighbor_contact_pairs
        ):
            raise RuntimeError('gripper withdrawal disturbed the staged book')

        passed = True
        print(
            json.dumps(
                {
                    'event': 'result',
                    'passed': True,
                    'seat_height': seat_height,
                    'seat_translation_m': seat_translation,
                    'seat_rotation_rad': seat_rotation,
                    'total_rotation_rad': total_rotation,
                    'release_translation_m': release_translation,
                    'release_rotation_rad': release_rotation,
                    'withdrawal_translation_m': withdrawal_translation,
                    'withdrawal_rotation_rad': withdrawal_rotation,
                    'extra_pull_steps': extra_index,
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
        if not passed:
            time.sleep(0.05)


if __name__ == '__main__':
    main()
