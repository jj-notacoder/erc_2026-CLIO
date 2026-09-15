#!/usr/bin/env python3
"""Temporary real-shelf pick/retreat/compact probe; remove before commit."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_retreat_probe import (
    ProbeManipulationNode,
    _rotation_distance,
    emit,
    gazebo_poses,
    quaternion_matrix,
    set_model_pose,
)
from erc_phase1_solution.kinematics import pose_matrix
from erc_phase1_solution.navigation_node import NavigationNode


BOOK = 'book_col_3_row_2_red'
FRONT = np.asarray(
    [0.6927616089375521, -0.055868643690940314, 1.5836944979660614],
    dtype=float,
)
BASE_POSITION = np.asarray(
    [2.126938006268, -0.093344043799, 0.0],
    dtype=float,
)
BASE_YAW = 0.001089153
BAY_FLOOR_Z = 1.45186
BAY_CEILING_Z = 1.75186

ALT_SOLUTIONS = [
    np.asarray(values, dtype=float)
    for values in (
        [0.35, 3.3409075931, -1.9957193235, 1.1082417689, 0.9892997090, -0.9003201217, -1.4533579662, 2.0299520768],
        [0.35, 3.3665971496, -1.9197343893, 1.1576232041, 0.9660387049, -0.8524933308, -1.3738147292, 2.0075337951],
        [0.35, 3.4011215725, -1.8496449816, 1.2052973561, 0.9124553526, -0.8075781479, -1.2862820048, 1.9938210943],
        [0.35, 3.4472667206, -1.7865507107, 1.2485899291, 0.8237836630, -0.7702165835, -1.1889952019, 1.9906468117],
        [0.35, 3.5034632744, -1.7349966227, 1.2835225010, 0.7064077307, -0.7461316311, -1.0886723101, 2.0002908738],
        [0.35, 3.5683374425, -1.6869635605, 1.3104210558, 0.5674715830, -0.7329038489, -0.9787509756, 2.0172540676],
        [0.35, 3.6168362014, -1.6356976568, 1.3311425118, 0.4641886249, -0.7142357253, -0.8771637057, 2.0094996641],
    )
]
ALT_TRANSITION = [
    np.asarray(values, dtype=float)
    for values in (
        [0.35, 3.3409075931, -1.83, 0.47, -2.35, 0.0, -1.2, 0.0],
        [0.35, 3.3409075931, -1.9957193235, 0.47, -2.35, 0.0, -1.2, 0.0],
        [0.35, 3.3409075931, -1.9957193235, 1.1082417689, -2.35, 0.0, -1.2, 0.0],
        [0.35, 3.3409075931, -1.9957193235, 1.1082417689, -2.35, -0.9003201217, -1.2, 0.0],
        [0.35, 3.3409075931, -1.9957193235, 1.1082417689, 0.9892997090, -0.9003201217, -1.2, 0.0],
        [0.35, 3.3409075931, -1.9957193235, 1.1082417689, 0.9892997090, -0.9003201217, -1.4533579662, 0.0],
    )
]

# Highest fixed-orientation candidate from the exact carried-volume scan.  The
# +y translation gives arm_left_2 enough room to raise the still-pinched book
# roughly 11.4 cm after it has cleared the shelf face.
HIGH_CRADLE_STAGED = np.asarray(
    [
        0.35,
        0.4447527167,
        1.1212183373,
        -0.2442251487,
        -1.4511660791,
        0.0921473927,
        1.4977638625,
        -0.1678183941,
    ],
    dtype=float,
)


def bay_transition_is_safe(
    node,
    start,
    end,
    corners,
    *,
    aperture_corners=None,
) -> bool:
    """Conservative box-aperture check for this fixed top-row probe."""
    first = np.asarray(start, dtype=float)
    last = np.asarray(end, dtype=float)
    if not node._carried_robot_transition_is_safe(first, last, corners):
        print(
            json.dumps(
                {
                    'event': 'bay_transition_rejected',
                    'reason': 'carried_robot',
                    'start': first.tolist(),
                    'end': last.tolist(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return False
    checked_corners = np.asarray(
        corners if aperture_corners is None else aperture_corners,
        dtype=float,
    )
    for fraction in np.linspace(0.0, 1.0, 61):
        solution = first + (last - first) * fraction
        transform = node.chain.forward(solution)
        world = (
            checked_corners @ transform[:3, :3].T
            + transform[:3, 3]
        )
        if (
            float(np.min(world[:, 2])) < BAY_FLOOR_Z
            or float(np.max(world[:, 2])) > BAY_CEILING_Z
            or float(np.min(world[:, 0])) < 0.40
            or float(np.max(world[:, 0])) > FRONT[0] + 0.21
        ):
            print(
                json.dumps(
                    {
                        'event': 'bay_transition_rejected',
                        'reason': 'aperture',
                        'fraction': float(fraction),
                        'minimum': np.min(world, axis=0).tolist(),
                        'maximum': np.max(world, axis=0).tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return False
    return True


def observed_attached_corners(node, padding):
    """Build a diagnostic gripper-frame box from Gazebo's measured pose."""
    poses = gazebo_poses(BOOK)
    base_position, base_quaternion = poses['base']
    book_position, book_quaternion = poses['book']
    # Gazebo's book model uses local +x for its 250 mm height and local +z for
    # its 160 mm shelf depth.  The carried-volume checker intentionally treats
    # corners[1] - corners[0] as the vertical box axis, matching the logical
    # world-depth/world-width/world-height ordering produced by
    # ``_attached_book_corners``.  Reorder the measured model axes here (and
    # flip local x upward) so a physically upright book is not misclassified
    # as a 90-degree tilt after re-anchoring.
    half = np.asarray([0.080, 0.015, 0.125], dtype=float) + float(padding)
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )
    model_rotation = quaternion_matrix(book_quaternion)
    logical_rotation = np.column_stack(
        (model_rotation[:, 2], model_rotation[:, 1], -model_rotation[:, 0])
    )
    world = (signs * half) @ logical_rotation.T + book_position
    base = (world - base_position) @ quaternion_matrix(base_quaternion)
    grasp = node.chain.forward(node._measured_left_solution())
    attached = (base - grasp[:3, 3]) @ grasp[:3, :3]
    return attached, world


def settled_transition_is_safe(
    node,
    start,
    end,
    corners,
    aperture_corners,
):
    """Validate a shelf-supported lift/retraction after pose re-anchoring."""
    first = np.asarray(start, dtype=float)
    last = np.asarray(end, dtype=float)
    if not node._carried_robot_transition_is_safe(first, last, corners):
        return False
    shelf_front_x = float(FRONT[0] - 0.065)
    for fraction in np.linspace(0.0, 1.0, 61):
        q = first + (last - first) * fraction
        transform = node.chain.forward(q)
        world = (
            aperture_corners @ transform[:3, :3].T
            + transform[:3, 3]
        )
        if float(np.max(world[:, 0])) > shelf_front_x:
            if (
                float(np.min(world[:, 2])) < BAY_FLOOR_Z - 0.003
                or float(np.max(world[:, 2])) > BAY_CEILING_Z
            ):
                return False
    return True


class PickProbeNode(ProbeManipulationNode):
    skip_return_preflight = False
    alternate_branch = False
    bottom_cradle = False
    bottom_full_extraction = False
    high_cradle = False
    probe_solutions = None

    def __init__(self):
        self.track_unexpected_contacts = False
        self.unexpected_contact_pairs = set()
        super().__init__()

    def _on_contacts(self, message):
        super()._on_contacts(message)
        if not self.track_unexpected_contacts:
            return
        for contact in getattr(message, 'contacts', []):
            first = str(
                getattr(getattr(contact, 'collision1', None), 'name', '')
            )
            second = str(
                getattr(getattr(contact, 'collision2', None), 'name', '')
            )
            lowered = f'{first} {second}'.lower()
            involves_probe = BOOK in lowered or 'tiago_pro' in lowered
            scored_object = any(
                token in lowered
                for token in (
                    'erc_shelf',
                    'book_col_',
                    'erc_table',
                    'collection_bin',
                )
            )
            intentional = (
                BOOK in lowered
                and 'gripper_left' in lowered
                and 'erc_shelf' not in lowered
            )
            if involves_probe and scored_object and not intentional:
                pair = tuple(sorted((first, second)))
                if pair not in self.unexpected_contact_pairs:
                    self.unexpected_contact_pairs.add(pair)
                    print(
                        json.dumps(
                            {
                                'event': 'unexpected_contact',
                                'pair': pair,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    def _wait_for_perception_point(self, attribute: str) -> np.ndarray:
        if attribute != 'latest_book':
            raise RuntimeError(f'unexpected perception request: {attribute}')
        return FRONT.copy()

    def _plan_carried_return(
        self,
        front,
        grasp_solution,
        clearance_solution,
        rotation,
        torso_height,
    ):
        if not self.skip_return_preflight:
            return super()._plan_carried_return(
                front,
                grasp_solution,
                clearance_solution,
                rotation,
                torso_height,
            )
        # A focused retreat probe does not execute compact transport.  Preserve
        # the top-row torso/deferred-return state while avoiding minutes of
        # unrelated compact-route mesh planning on every dynamics experiment.
        self._cached_post_retreat_plan = {'probe_only': True}
        return (
            [],
            [],
            None,
            [np.asarray(clearance_solution, dtype=float).copy()],
            self._attached_book_corners(front, grasp_solution),
        )

    def _solve_cartesian_path(self, *args, **kwargs):
        if self.alternate_branch and kwargs.get('endpoint_first'):
            result = (
                [solution.copy() for solution in ALT_SOLUTIONS],
                0,
                0.0,
                [solution.copy() for solution in ALT_TRANSITION],
            )
        else:
            result = super()._solve_cartesian_path(*args, **kwargs)
        if kwargs.get('endpoint_first'):
            self.probe_solutions = [
                np.asarray(solution, dtype=float).copy()
                for solution in result[0]
            ]
        return result

    def _loaded_clearance_index(self, top_row, solutions):
        if (
            top_row
            and self.bottom_cradle
            and not self.bottom_full_extraction
        ):
            return len(solutions) - 2
        return super()._loaded_clearance_index(top_row, solutions)

    def _publish_status(self, event, **fields):
        print(
            json.dumps({'event': f'status:{event}', **fields}, sort_keys=True),
            flush=True,
        )
        return super()._publish_status(event, **fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=float, required=True)
    parser.add_argument('--speed', type=float, default=0.10)
    parser.add_argument('--acceleration', type=float, default=0.06)
    parser.add_argument('--vertical-offset', type=float, default=-0.015)
    parser.add_argument('--distance', type=float, default=0.35)
    parser.add_argument('--skip-compact', action='store_true')
    parser.add_argument(
        '--pick-only',
        action='store_true',
        help='stop after the guarded pick without commanding the mobile base',
    )
    parser.add_argument('--pre-retreat-cradle', action='store_true')
    parser.add_argument('--skip-return-preflight', action='store_true')
    parser.add_argument('--alternate-branch', action='store_true')
    parser.add_argument('--bottom-cradle', action='store_true')
    parser.add_argument('--bottom-full-extraction', action='store_true')
    parser.add_argument('--high-cradle', action='store_true')
    parser.add_argument('--roll-duration', type=float, default=0.75)
    parser.add_argument('--roll-radians', type=float, default=-1.10)
    parser.add_argument('--support-dwell', type=float, default=0.50)
    parser.add_argument('--cradle-only', action='store_true')
    parser.add_argument('--reanchor-settled', action='store_true')
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
            raise RuntimeError('robot state is unavailable')

        node.gripper_transport_lock = float(args.lock)
        node.top_row_grasp_vertical_offset = float(args.vertical_offset)
        node.skip_return_preflight = bool(args.skip_return_preflight)
        node.alternate_branch = bool(args.alternate_branch)
        node.bottom_cradle = bool(args.bottom_cradle)
        node.bottom_full_extraction = bool(args.bottom_full_extraction)
        node.high_cradle = bool(args.high_cradle)
        node.carried_cradle_transfer = float(args.roll_duration)
        nav.carried_retreat_max_speed = float(args.speed)
        nav.carried_retreat_acceleration = float(args.acceleration)
        nav.carried_retreat_braking_acceleration = float(args.acceleration)

        if not node._stow():
            raise RuntimeError('stow failed')
        base_quaternion = np.asarray(
            [
                0.0,
                0.0,
                math.sin(BASE_YAW / 2.0),
                math.cos(BASE_YAW / 2.0),
            ],
            dtype=float,
        )
        set_model_pose('tiago_pro', BASE_POSITION, base_quaternion)
        if not node._wait_sim_duration(0.40):
            raise RuntimeError('base teleport did not settle')

        picked = node._pick()
        if not picked:
            print(
                json.dumps(
                    {
                        'event': 'result',
                        'passed': False,
                        'reason': 'pick_failed',
                        'lock': args.lock,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        baseline = emit(
            'real_pick_complete',
            node,
            nav,
            BOOK,
            None,
            lock=args.lock,
            speed=args.speed,
            acceleration=args.acceleration,
            vertical_offset=args.vertical_offset,
            target_model=node._target_book_model,
        )
        node.track_unexpected_contacts = True
        left, right = node._target_contact_sides(max_age=0.15)
        if not (left and right and node._target_book_model == BOOK):
            raise RuntimeError('real pick did not establish the exact bilateral lock')

        if args.pick_only:
            print(
                json.dumps(
                    {
                        'event': 'result',
                        'passed': True,
                        'stage': 'guarded_pick_only',
                        'target_model': node._target_book_model,
                        'left_contact': left,
                        'right_contact': right,
                        'base_motion_commanded': False,
                        'gripper_opened': False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

        if args.pre_retreat_cradle:
            measured = node._measured_left_solution()
            shelf_front_x = float(FRONT[0])
            corners = node._held_book_corners
            if corners is None:
                raise RuntimeError('book envelope was not installed')
            aperture_corners = corners
            if args.bottom_cradle:
                if not node.probe_solutions:
                    raise RuntimeError('pick solutions were not retained')
                saved_padding = node.carried_book_padding
                try:
                    node.carried_book_padding = 0.0
                    aperture_corners = node._attached_book_corners(
                        FRONT,
                        node.probe_solutions[-1],
                    )
                finally:
                    node.carried_book_padding = saved_padding

            if args.bottom_cradle:
                # Either lift an early lower grasp inside the aperture or pull
                # a fully extracted lower grasp an additional 12 cm outward.
                measured_pose = node.chain.forward(measured)
                lift_target = measured_pose[:3, 3] + np.asarray(
                    (
                        [-0.12, 0.0, 0.0]
                        if args.bottom_full_extraction
                        else [0.0, 0.0, 0.025]
                    )
                )
                retracted, _ = node.chain.solve(
                    pose_matrix(lift_target, measured_pose[:3, :3]),
                    [measured],
                    position_tolerance=node.position_tolerance,
                    orientation_tolerance=node.orientation_tolerance,
                    max_iterations=240,
                    fixed_positions={
                        'torso_lift_joint': float(measured[0])
                    },
                )
                if retracted is None:
                    raise RuntimeError('bottom-cradle lift IK failed')
                retracted = np.asarray(retracted, dtype=float)
                if args.bottom_full_extraction:
                    if not node._carried_robot_transition_is_safe(
                        measured,
                        retracted,
                        corners,
                    ):
                        raise RuntimeError(
                            'outside-cradle staging failed robot safety'
                        )
                    transform = node.chain.forward(retracted)
                    staged_world = (
                        corners @ transform[:3, :3].T
                        + transform[:3, 3]
                    )
                    if float(np.max(staged_world[:, 0])) > FRONT[0] - 0.075:
                        raise RuntimeError(
                            'outside-cradle staging left the payload in the shelf'
                        )
                elif not bay_transition_is_safe(
                    node,
                    measured,
                    retracted,
                    corners,
                    aperture_corners=aperture_corners,
                ):
                    raise RuntimeError('bottom-cradle lift failed aperture safety')
            else:
                # Earlier diagnostic candidates retained for comparison.
                retracted = (
                    HIGH_CRADLE_STAGED.copy()
                    if args.high_cradle
                    else (
                        ALT_SOLUTIONS[1].copy()
                        if args.alternate_branch
                        else np.asarray(
                        [
                            0.35,
                            0.67851687341569,
                            1.1121799360699212,
                            -0.058910664741116704,
                            -1.69236452688237,
                            0.28933612213361865,
                            1.7466043534606153,
                            -0.40855261890557076,
                        ],
                        dtype=float,
                    )
                    )
                )
                if not node._carried_transition_is_safe(
                    measured,
                    retracted,
                    corners,
                    shelf_front_x,
                ):
                    raise RuntimeError(
                        'pre-retreat retraction failed exact safety check'
                    )

            cradled = retracted.copy()
            cradled[-1] += float(args.roll_radians)
            if args.bottom_cradle:
                roll_route = node._plan_carried_joint_route(
                    retracted,
                    [cradled],
                    corners,
                )
                roll_previous = retracted
                if not roll_route or any(
                    (
                        not node._carried_robot_transition_is_safe(
                            roll_previous
                            if index == 0
                            else roll_route[index - 1],
                            solution,
                            corners,
                        )
                        if args.bottom_full_extraction
                        else not bay_transition_is_safe(
                            node,
                            roll_previous
                            if index == 0
                            else roll_route[index - 1],
                            solution,
                            corners,
                            aperture_corners=aperture_corners,
                        )
                    )
                    for index, solution in enumerate(roll_route)
                ):
                    raise RuntimeError(
                        'bottom-cradle roll failed aperture safety'
                    )
            else:
                roll_route = node._plan_carried_joint_route(
                    retracted,
                    [cradled],
                    corners,
                    shelf_front_x=shelf_front_x,
                )
            if not roll_route:
                raise RuntimeError('pre-retreat cradle failed exact safety check')
            if args.high_cradle or args.bottom_full_extraction:
                stage_route = node._plan_carried_joint_route(
                    measured,
                    [retracted],
                    corners,
                    **(
                        {'shelf_front_x': shelf_front_x}
                        if args.high_cradle
                        else {}
                    ),
                )
                if not stage_route:
                    raise RuntimeError('high-cradle staging route was not safe')
                stage_legs = []
                stage_previous = measured
                for solution in stage_route:
                    stage_legs.append(
                        (
                            solution,
                            max(
                                0.50,
                                1.5 * node._transport_leg_duration(
                                    stage_previous,
                                    solution,
                                ),
                            ),
                            'high_cradle_lift',
                        )
                    )
                    stage_previous = solution
                stage_goal, stage_duration = (
                    node._make_retained_arm_trajectory_goal(stage_legs)
                )
                node._retention_probe_active = True
                node._payload_robot_watchdog_enabled = True
                try:
                    stage_moved, _ = node._send_retained_arm_trajectory(
                        stage_goal,
                        stage_duration,
                        stage_legs,
                        'probe',
                    )
                    stage_terminal = (
                        node._wait_for_retained_endpoint(
                            retracted,
                            command='probe',
                            phase='high_cradle_lift',
                            leg=len(stage_legs),
                        )
                        if stage_moved
                        else None
                    )
                finally:
                    node._retention_probe_active = False
                    node._payload_robot_watchdog_enabled = False
                if not stage_moved or stage_terminal is None:
                    raise RuntimeError('high-cradle staging controller failure')
            else:
                if (
                    (args.bottom_cradle or not args.alternate_branch)
                    and not node._move_arm_solution(retracted, 0.90)
                ):
                    raise RuntimeError('pre-retreat staging controller failure')
                if node._wait_for_retained_endpoint(
                    retracted,
                    command='probe',
                    phase='pre_retreat_retraction',
                    leg=0,
                ) is None:
                    raise RuntimeError(
                        'pre-retreat retraction missed its endpoint'
                    )
            if not node._fresh_retention_probe(
                'probe',
                'pre_cradle',
                leg=0,
            ):
                raise RuntimeError('bilateral contact lost before pre-retreat cradle')

            roll_legs = [
                (
                    np.asarray(solution, dtype=float),
                    node.carried_cradle_transfer / len(roll_route),
                    'mechanical_cradle',
                )
                for solution in roll_route
            ]
            roll_goal, roll_duration = node._make_retained_arm_trajectory_goal(
                roll_legs
            )
            node._retention_probe_active = True
            node._gravity_supported_payload = True
            node._payload_robot_watchdog_enabled = True
            cradle_terminal = None
            try:
                moved, _ = node._send_retained_arm_trajectory(
                    roll_goal,
                    roll_duration,
                    roll_legs,
                    'probe',
                )
                if moved:
                    cradle_terminal = node._wait_for_retained_endpoint(
                        cradled,
                        command='probe',
                        phase='mechanical_cradle',
                        leg=len(roll_legs),
                    )
            finally:
                node._retention_probe_active = False
                node._payload_robot_watchdog_enabled = False
            if not moved or cradle_terminal is None:
                raise RuntimeError('pre-retreat cradle controller failure')
            if not node._fresh_retention_probe(
                'probe',
                'mechanical_cradle',
                leg=len(roll_legs),
            ):
                raise RuntimeError('lower-jaw support was not established')

            if not node._wait_sim_duration(float(args.support_dwell)):
                raise RuntimeError('support dwell was interrupted')
            if node._payload_hazard_latched is not None:
                raise RuntimeError(
                    f'support dwell failed: {node._payload_hazard_latched}'
                )
            if not node._fresh_retention_probe(
                'probe',
                'mechanical_cradle_dwell',
                leg=len(roll_legs),
            ):
                raise RuntimeError('lower-jaw support did not survive the dwell')

            support_measurement = emit(
                'pre_retreat_cradle_complete',
                node,
                nav,
                BOOK,
                baseline,
                roll_waypoints=len(roll_route),
            )
            support_translation_drift = float(
                np.linalg.norm(support_measurement[0] - baseline[0])
            )
            support_rotation_drift = _rotation_distance(
                baseline[1],
                support_measurement[1],
            )
            stable_measurement = support_measurement
            if args.reanchor_settled:
                if not node._wait_sim_duration(0.50):
                    raise RuntimeError('settled-pose stability dwell failed')
                stable_measurement = emit(
                    'settled_pose_stability',
                    node,
                    nav,
                    BOOK,
                    support_measurement,
                )
                stability_translation = float(
                    np.linalg.norm(
                        stable_measurement[0] - support_measurement[0]
                    )
                )
                stability_rotation = _rotation_distance(
                    support_measurement[1],
                    stable_measurement[1],
                )
                if (
                    stability_translation > 0.005
                    or stability_rotation > math.radians(5.0)
                ):
                    raise RuntimeError('settled payload pose was not stable')
                corners, _ = observed_attached_corners(
                    node,
                    node.carried_book_padding,
                )
                aperture_corners, _ = observed_attached_corners(node, 0.0)
                node._held_book_corners = np.asarray(corners, dtype=float)
            elif args.bottom_full_extraction:
                if not node._wait_sim_duration(0.50):
                    raise RuntimeError('settled-pose stability dwell failed')
                stable_measurement = emit(
                    'settled_pose_stability',
                    node,
                    nav,
                    BOOK,
                    support_measurement,
                )
                stability_translation = float(
                    np.linalg.norm(
                        stable_measurement[0] - support_measurement[0]
                    )
                )
                stability_rotation = _rotation_distance(
                    support_measurement[1],
                    stable_measurement[1],
                )
                poses = gazebo_poses(BOOK)
                book_position, book_quaternion = poses['book']
                half = np.asarray([0.125, 0.015, 0.080], dtype=float)
                signs = np.asarray(
                    [
                        (x, y, z)
                        for x in (-1.0, 1.0)
                        for y in (-1.0, 1.0)
                        for z in (-1.0, 1.0)
                    ],
                    dtype=float,
                )
                actual_world = (
                    signs * half
                ) @ quaternion_matrix(book_quaternion).T + book_position
                maximum_world_x = float(np.max(actual_world[:, 0]))
                print(
                    json.dumps(
                        {
                            'event': 'settled_pose_gate',
                            'translation_change_m': stability_translation,
                            'rotation_change_rad': stability_rotation,
                            'maximum_world_x': maximum_world_x,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if (
                    stability_translation > 0.005
                    or stability_rotation > math.radians(5.0)
                    or maximum_world_x > 2.750
                ):
                    raise RuntimeError('outside cradle did not settle safely')
            elif (
                support_translation_drift > 0.020
                or support_rotation_drift > math.radians(10.0)
            ):
                raise RuntimeError(
                    'supported book pose drifted outside the rigid-envelope '
                    f'gate ({support_translation_drift:.6f} m, '
                    f'{support_rotation_drift:.6f} rad)'
                )
            if node.unexpected_contact_pairs:
                raise RuntimeError('cradle produced an unexpected scored contact')

            if args.cradle_only:
                print(
                    json.dumps(
                        {
                            'event': 'result',
                            'passed': True,
                            'pick': True,
                            'cradle': True,
                            'target_model': node._target_book_model,
                            'payload_hazard': node._payload_hazard_latched,
                            'unexpected_contacts': len(
                                node.unexpected_contact_pairs
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return

            if args.bottom_cradle and not args.bottom_full_extraction:
                if not node.probe_solutions:
                    raise RuntimeError('pick solutions were not retained')
                supported_start = np.asarray(roll_route[-1], dtype=float)
                supported_pose = node.chain.forward(supported_start)
                target_positions = []
                if args.reanchor_settled:
                    lifted_position = supported_pose[:3, 3].copy()
                    lifted_position[2] += 0.025
                    target_positions.append(lifted_position)
                    target_position = lifted_position.copy()
                    target_position[0] = 0.420
                    target_positions.append(target_position)
                else:
                    target_position = supported_pose[:3, 3].copy()
                    target_position[0] = node.chain.forward(
                        node.probe_solutions[1]
                    )[0, 3]
                    target_positions.append(target_position)
                supported_route = []
                supported_previous = supported_start
                supported_position = supported_pose[:3, 3].copy()
                for target_position in target_positions:
                    for position in node._interpolate_positions(
                        supported_position,
                        target_position,
                        node.cartesian_step,
                    ):
                        solution, _ = node.chain.solve(
                            pose_matrix(position, supported_pose[:3, :3]),
                            [supported_previous],
                            position_tolerance=node.position_tolerance,
                            orientation_tolerance=node.orientation_tolerance,
                            max_iterations=240,
                            fixed_positions={
                                'torso_lift_joint': float(supported_start[0])
                            },
                        )
                        if solution is None:
                            raise RuntimeError('supported retraction IK failed')
                        solution = np.asarray(solution, dtype=float)
                        transition_safe = (
                            settled_transition_is_safe(
                                node,
                                supported_previous,
                                solution,
                                corners,
                                aperture_corners,
                            )
                            if args.reanchor_settled
                            else bay_transition_is_safe(
                                node,
                                supported_previous,
                                solution,
                                corners,
                                aperture_corners=aperture_corners,
                            )
                        )
                        if (
                            float(
                                np.max(
                                    np.abs(
                                        solution[1:] - supported_previous[1:]
                                    )
                                )
                            )
                            > node.cartesian_joint_step
                            or not transition_safe
                            or not node._gravity_supported_transition_is_safe(
                                supported_previous,
                                solution,
                            )
                        ):
                            raise RuntimeError(
                                'supported retraction failed payload safety'
                            )
                        supported_route.append(solution)
                        supported_previous = solution
                    supported_position = target_position.copy()
                supported_legs = []
                supported_previous = supported_start
                for solution in supported_route:
                    supported_legs.append(
                        (
                            solution,
                            node._transport_leg_duration(
                                supported_previous,
                                solution,
                            ),
                            'supported_cradle_retraction',
                        )
                    )
                    supported_previous = solution
                supported_goal, supported_duration = (
                    node._make_retained_arm_trajectory_goal(supported_legs)
                )
                node._payload_robot_watchdog_enabled = True
                try:
                    moved, _ = node._send_retained_arm_trajectory(
                        supported_goal,
                        supported_duration,
                        supported_legs,
                        'probe',
                    )
                    supported_terminal = (
                        node._wait_for_retained_endpoint(
                            supported_route[-1],
                            command='probe',
                            phase='supported_cradle_retraction',
                            leg=len(roll_legs) + len(supported_legs),
                        )
                        if moved
                        else None
                    )
                finally:
                    node._payload_robot_watchdog_enabled = False
                if not moved or supported_terminal is None:
                    raise RuntimeError('supported retraction controller failure')
                if not node._fresh_retention_probe(
                    'probe',
                    'supported_cradle_retraction',
                    leg=len(roll_legs) + len(supported_legs),
                ):
                    raise RuntimeError(
                        'lower-jaw support was lost during retraction'
                    )
                supported_measurement = emit(
                    'supported_retraction_complete',
                    node,
                    nav,
                    BOOK,
                    baseline,
                    waypoints=len(supported_route),
                )
                if args.reanchor_settled:
                    supported_translation_drift = float(
                        np.linalg.norm(
                            supported_measurement[0] - stable_measurement[0]
                        )
                    )
                    supported_rotation_drift = _rotation_distance(
                        stable_measurement[1],
                        supported_measurement[1],
                    )
                    _, actual_world = observed_attached_corners(node, 0.0)
                    maximum_world_x = float(np.max(actual_world[:, 0]))
                    print(
                        json.dumps(
                            {
                                'event': 'supported_retraction_gate',
                                'translation_drift_m': (
                                    supported_translation_drift
                                ),
                                'rotation_drift_rad': supported_rotation_drift,
                                'maximum_world_x': maximum_world_x,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if (
                        supported_translation_drift > 0.020
                        or supported_rotation_drift > math.radians(10.0)
                        or maximum_world_x > 2.750
                    ):
                        raise RuntimeError(
                            'supported retraction did not carry the book clear'
                        )
            node._cached_post_retreat_plan = None
            node._supported_post_retreat_staging_required = False

        start = np.asarray(nav.pose, dtype=float)
        goal = (
            float(start[0] - args.distance * math.cos(start[2])),
            float(start[1] - args.distance * math.sin(start[2])),
            float(start[2]),
        )
        nav._accept_goal(goal, profile='carried_retreat')
        maximum_speed = 0.0
        first_loss = None
        while nav.goal is not None:
            if nav.pose is not None:
                displacement = float(
                    math.hypot(
                        nav.pose[0] - start[0],
                        nav.pose[1] - start[1],
                    )
                )
            else:
                displacement = math.nan
            maximum_speed = max(
                maximum_speed,
                math.hypot(
                    nav.last_command.linear.x,
                    nav.last_command.linear.y,
                ),
            )
            hazard = node._payload_hazard_latched
            if hazard is not None:
                first_loss = {
                    'reason': str(hazard),
                    'sim_time': node.get_clock().now().nanoseconds / 1e9,
                    'displacement': displacement,
                }
                nav._finish_goal('cancelled')
                break
            time.sleep(0.02)

        node._wait_sim_duration(0.20)
        retreat = emit(
            'real_retreat_complete',
            node,
            nav,
            BOOK,
            baseline,
            requested_distance=args.distance,
            actual_displacement=float(
                math.hypot(nav.pose[0] - start[0], nav.pose[1] - start[1])
            ) if nav.pose is not None else math.nan,
            maximum_commanded_speed=maximum_speed,
            first_loss=first_loss,
        )
        left, right = node._target_contact_sides(max_age=0.15)
        required_contact = left if args.pre_retreat_cradle else left and right
        retreat_ok = bool(
            first_loss is None
            and required_contact
            and node._payload_hazard_latched is None
            and not node._target_robot_contact_latched
            and not node.unexpected_contact_pairs
        )
        compact_ok = False
        if retreat_ok and not args.skip_compact:
            compact_ok = bool(node._compact_transport())
            node._wait_sim_duration(0.20)
            emit(
                'real_compact_complete',
                node,
                nav,
                BOOK,
                retreat,
                compact_ok=compact_ok,
            )
        elif retreat_ok:
            compact_ok = True

        result = {
            'event': 'result',
            'passed': bool(retreat_ok and compact_ok),
            'pick': bool(picked),
            'retreat': retreat_ok,
            'compact': compact_ok,
            'lock': args.lock,
            'speed': args.speed,
            'acceleration': args.acceleration,
            'vertical_offset': args.vertical_offset,
            'left': bool(left),
            'right': bool(right),
            'payload_hazard': node._payload_hazard_latched,
            'robot_contact_latched': bool(node._target_robot_contact_latched),
            'target_model': node._target_book_model,
            'unexpected_contacts': [
                list(pair) for pair in sorted(node.unexpected_contact_pairs)
            ],
        }
        print(json.dumps(result, sort_keys=True), flush=True)
        if not result['passed']:
            raise RuntimeError('real pick/retreat probe failed')
    finally:
        nav._publish_zero()
        executor.shutdown()
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
