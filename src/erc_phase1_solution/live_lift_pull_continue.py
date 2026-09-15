#!/usr/bin/env python3
"""Temporary same-world exact-lift lower-shelf continuation; remove before commit."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.navigation_node import NavigationNode
from live_coupled_catch_probe import physical_book_bounds
from live_lower_shelf_rescue import (
    DOWN1,
    DOWN2,
    IN1,
    IN2,
    LOWER_CEILING_Z,
    LOWER_FLOOR_Z,
    Q_OUT,
    SHELF_FRONT_X,
    STAGE_TORSO,
    RescueProbeNode,
    bounds_payload,
    fresh_shelf_contact,
    require_retention,
    retained_arm_leg,
)
from live_pick_retreat_probe import (
    BOOK,
    _rotation_distance,
    observed_attached_corners,
    quaternion_matrix,
)


LIFT_PULL = [
    (
        np.asarray(
            [
                0.349999995444,
                0.690945093032,
                1.011369178051,
                -0.066079702839,
                -1.664407685011,
                0.291724168917,
                1.625228901548,
                -0.442167076033,
            ]
        ),
        'exact_lift_4mm',
        'lift',
    ),
    (
        np.asarray(
            [
                0.349999995444,
                0.687532267972,
                1.020750220914,
                -0.067711511908,
                -1.649755579642,
                0.289254431100,
                1.619426960108,
                -0.440520336379,
            ]
        ),
        'exact_lift_8mm',
        'lift',
    ),
    (
        np.asarray(
            [
                0.349999995444,
                0.684117030397,
                1.030182740888,
                -0.069307191109,
                -1.634940644980,
                0.286825577466,
                1.613533787172,
                -0.438913630640,
            ]
        ),
        'exact_lift_12mm',
        'lift',
    ),
    (
        np.asarray(
            [
                0.349999995444,
                0.685430154660,
                1.057129129919,
                -0.067674267115,
                -1.657941764897,
                0.290775549369,
                1.661904193050,
                -0.426833409074,
            ]
        ),
        'elevated_pull_20mm',
        'pull',
    ),
    (
        np.asarray(
            [
                0.349999995444,
                0.686423527077,
                1.086643808115,
                -0.065631485366,
                -1.677335410354,
                0.295203658750,
                1.709292870468,
                -0.414954369543,
            ]
        ),
        'elevated_pull_40mm',
        'pull',
    ),
    (
        np.asarray(
            [
                0.349999995444,
                0.687091572691,
                1.118611117499,
                -0.063189143642,
                -1.693145689812,
                0.300031101533,
                1.755617858820,
                -0.403287386995,
            ]
        ),
        'elevated_pull_60mm',
        'pull',
    ),
]


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
        if not node._fresh_retention_probe('exact_lift', 'resume', leg=0):
            raise RuntimeError('same-world bilateral grasp is unavailable')

        previous = physical_book_bounds()
        baseline_rotation = quaternion_matrix(previous[1])
        measured_start = node._measured_left_solution()
        route_start = 0
        for route_index, (target, _, _) in enumerate(LIFT_PULL):
            if float(np.max(np.abs(measured_start - target))) <= 0.005:
                route_start = route_index + 1
        for index, (target, phase, mode) in enumerate(
            LIFT_PULL[route_start:],
            start=route_start + 1,
        ):
            corners, _ = observed_attached_corners(
                node,
                node.carried_book_padding,
            )
            node._held_book_corners = corners
            retained_arm_leg(node, target, 0.62, phase, index)
            current = bounds_payload(f'{phase}_bounds')
            translation = current[0] - previous[0]
            rotation = _rotation_distance(
                quaternion_matrix(previous[1]),
                quaternion_matrix(current[1]),
            )
            if mode == 'lift':
                if (
                    float(current[2][2] - previous[2][2]) < 0.0025
                    or float(np.linalg.norm(translation[:2])) > 0.002
                    or rotation > 0.04
                ):
                    raise RuntimeError(f'book did not track {phase}')
            elif (
                float(translation[0]) > -0.016
                or abs(float(translation[2])) > 0.002
                or rotation > 0.02
            ):
                raise RuntimeError(f'book did not track {phase}')
            previous = current

        if (
            float(previous[2][2]) < 1.4540
            or float(previous[3][2]) > 1.731858
            or float(previous[3][0]) > 2.725
        ):
            raise RuntimeError('elevated pull did not clear the top shelf')

        corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        node._held_book_corners = corners
        retained_arm_leg(node, Q_OUT, 0.65, 'outside_lower_to_out60', 7)
        lowered_out = bounds_payload('outside_lowered_bounds')
        if (
            float(lowered_out[3][0]) > 2.725
            or float(previous[0][2] - lowered_out[0][2]) < 0.010
        ):
            raise RuntimeError('outside descent did not reach audited OUT60')
        require_retention(node, 'outward_clearance_dwell', 8)
        node.shelf_contact_ns = 0

        torso_start = float(node._measured_left_solution()[0])
        for index, height in enumerate(
            np.linspace(torso_start, STAGE_TORSO, 6)[1:],
            start=9,
        ):
            if not node._move_torso(float(height), 0.70):
                raise RuntimeError('torso descent failed')
            if fresh_shelf_contact(node):
                raise RuntimeError('premature shelf contact during torso descent')
            require_retention(node, 'outside_torso_descent', index)

        corners, _ = observed_attached_corners(
            node,
            node.carried_book_padding,
        )
        node._held_book_corners = corners
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

        position, _, minimum, _ = bounds_payload('lower_insertion_complete')
        if float(position[0]) < SHELF_FRONT_X + 0.020:
            raise RuntimeError('book COM is not safely inside the lower shelf')
        if float(minimum[2]) < LOWER_FLOOR_Z + 0.002:
            raise RuntimeError('book reached shelf before the seating probe')

        node.shelf_contact_ns = 0
        seat_height = float(node._measured_left_solution()[0])
        for index in range(28):
            seat_height -= 0.00025
            if not node._move_torso(seat_height, 0.18):
                raise RuntimeError('micro seating move failed')
            if fresh_shelf_contact(node):
                if node._target_robot_contact_latched or node.neighbor_contact_pairs:
                    raise RuntimeError('unsafe contact while seating')
                break
            require_retention(node, 'micro_seat', 30 + index)
        else:
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
            baseline_rotation,
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
            raise RuntimeError('release on lower shelf failed')
        if not node._wait_sim_duration(0.35):
            raise RuntimeError('release dwell failed')
        release_first = bounds_payload('release_first_dwell')
        if not node._wait_sim_duration(0.50):
            raise RuntimeError('release stability dwell failed')
        release_stable = bounds_payload('release_stable')
        if (
            np.linalg.norm(release_stable[0] - release_first[0]) > 0.001
            or _rotation_distance(
                quaternion_matrix(release_first[1]),
                quaternion_matrix(release_stable[1]),
            )
            > math.radians(2.0)
        ):
            raise RuntimeError('released book is unstable')

        for target, phase in ((IN1, 'withdraw_1'), (DOWN2, 'withdraw_2')):
            if not node._move_arm_solution(target, 0.90):
                raise RuntimeError(f'{phase} failed')
        if not node._wait_sim_duration(0.35):
            raise RuntimeError('withdrawal dwell failed')
        withdrawn = bounds_payload('withdrawn_stable')
        if (
            np.linalg.norm(withdrawn[0] - release_stable[0]) > 0.0015
            or _rotation_distance(
                quaternion_matrix(release_stable[1]),
                quaternion_matrix(withdrawn[1]),
            )
            > math.radians(2.0)
        ):
            raise RuntimeError('withdrawal disturbed the staged book')

        print(
            json.dumps(
                {
                    'event': 'result',
                    'passed': True,
                    'seat_height': seat_height,
                    'seat_translation_m': seat_translation,
                    'seat_rotation_rad': seat_rotation,
                    'total_rotation_rad': total_rotation,
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
