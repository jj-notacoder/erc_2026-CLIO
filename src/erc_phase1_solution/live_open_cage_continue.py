#!/usr/bin/env python3
"""Temporary continuation from open U3 to an underside-and-spine cage."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_coupled_catch_probe import physical_book_bounds
from live_lower_shelf_rescue import RescueProbeNode
from live_pick_retreat_probe import BOOK, _rotation_distance, quaternion_matrix


U275 = np.asarray([.35,.769801018518,.736203665559,-.016415295074,-1.932558345609,.356754895435,1.643399068630,-2.077048707415])
C5 = np.asarray([.35,.768784718705,.729758732953,-.016929362356,-1.924700752122,.355303807774,1.629674194695,-2.081138682733])
C10 = np.asarray([.35,.767757452843,.723522030020,-.017410155670,-1.916626139298,.353922078131,1.615938786964,-2.085219127901])
C14 = np.asarray([.35,.766928467020,.718683044546,-.017771694835,-1.910011554485,.352866942396,1.604944935067,-2.088476547009])
C15 = np.asarray([.35,.766720305472,.717494474087,-.017859021267,-1.908335019118,.352610336611,1.602194327545,-2.089290411719])
C155 = np.asarray([.35,.766616131215,.716903108319,-.017902268107,-1.907494228767,.352483019215,1.600819457468,-2.089696993793])
CT = np.asarray([.35,.766398783340,.717476568002,-.018047021522,-1.906789950079,.352261750967,1.600614055340,-2.089508250364])
X_TARGETS = [
    np.asarray([.35,.766357160775,.717240857608,-.018064205131,-1.906453023027,.352211096908,1.600064396279,-2.089670691206]),
    np.asarray([.35,.766336669534,.717124028667,-.018072767176,-1.906286313932,.352185982937,1.599793519707,-2.089750851981]),
    np.asarray([.35,.766315855748,.717006326588,-.018081325101,-1.906117714165,.352160733575,1.599518695422,-2.089832064581]),
    np.asarray([.35,.766295033841,.716888696057,-.018089870192,-1.905949003691,.352135510908,1.599243814734,-2.089913288327]),
    np.asarray([.35,.766274208080,.716771148962,-.018098402816,-1.905780207474,.352110316132,1.598968930976,-2.089994508780]),
]


def inspect(node, baseline, event):
    node._clear_target_contact_samples(reset_robot_contact=True)
    if not node._wait_sim_duration(0.18):
        raise RuntimeError('contact dwell interrupted')
    current = physical_book_bounds()
    left, right = node._target_contact_sides(max_age=0.22)
    translation = float(np.linalg.norm(current[0] - baseline[0]))
    rotation = _rotation_distance(
        quaternion_matrix(current[1]), quaternion_matrix(baseline[1])
    )
    print(json.dumps({
        'event': event,
        'left': bool(left),
        'right': bool(right),
        'position': current[0].tolist(),
        'minimum': current[2].tolist(),
        'maximum': current[3].tolist(),
        'translation_m': translation,
        'rotation_rad': rotation,
        'robot_contact': bool(node._target_robot_contact_latched),
        'solution': node._measured_left_solution().tolist(),
    }, sort_keys=True), flush=True)
    if (
        translation > 0.0006
        or rotation > math.radians(0.8)
        or node._target_robot_contact_latched
    ):
        raise RuntimeError(f'book moved or robot contacted during {event}')
    return bool(left), bool(right), current


def move(node, target, duration, event, baseline):
    if node._robot_self_collision(target) is not None:
        raise RuntimeError(f'self collision during {event}')
    if not node._move_arm_solution(target, duration):
        raise RuntimeError(f'controller failed during {event}')
    return inspect(node, baseline, event)


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = RescueProbeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('joint state unavailable')
        node._target_book_model = BOOK
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        node._target_robot_contact_latched = False
        if float(node.joints['gripper_left_finger_joint']) < 0.067:
            raise RuntimeError('gripper is not fully open')
        baseline = physical_book_bounds()

        for target, duration, event in (
            (U275, .34, 'cage_lower_025'),
            (C5, .42, 'cage_in_5'),
            (C10, .42, 'cage_in_10'),
            (C14, .38, 'cage_in_14'),
            (C15, .32, 'cage_in_15'),
            (C155, .30, 'cage_in_15_5'),
        ):
            left, right, _ = move(node, target, duration, event, baseline)
            if left or right:
                raise RuntimeError(f'premature contact during {event}')

        left, right, _ = move(node, CT, .32, 'cage_under_touch', baseline)
        if not left or right:
            raise RuntimeError('expected isolated lower-left contact was not established')

        terminal = None
        for index, target in enumerate(X_TARGETS, start=1):
            left, right, current = move(
                node, target, .28, f'cage_side_touch_{index}', baseline
            )
            if not left:
                raise RuntimeError('underside support was lost during side approach')
            if right:
                terminal = current
                break
        if terminal is None:
            raise RuntimeError('upper-right spine contact was not established')
        print(json.dumps({
            'event': 'result',
            'passed': True,
            'stage': 'open_hook_cage',
            'position': terminal[0].tolist(),
            'minimum': terminal[2].tolist(),
            'maximum': terminal[3].tolist(),
            'left': True,
            'right': True,
            'aperture': float(node.joints['gripper_left_finger_joint']),
            'solution': node._measured_left_solution().tolist(),
        }, sort_keys=True), flush=True)
    finally:
        node._retention_probe_active = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
