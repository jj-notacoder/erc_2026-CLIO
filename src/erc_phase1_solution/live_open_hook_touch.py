#!/usr/bin/env python3
"""Temporary same-world open-hand route to first underside hook contact."""

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


HIGH = np.asarray([.35,.694973014385,1.006893992766,-.063808759991,-1.682065701134,.295259337612,1.638822520566,-.442359128450])
D1 = np.asarray([.35,.737201066950,.896045020690,-.040096002315,-1.848248971086,.330283539283,1.702658129373,-.465898993781])
D2 = np.asarray([.35,.780000589866,.789178933932,-.009240717607,-1.995494920878,.373526257204,1.755430606000,-.495597117909])
I20 = np.asarray([.35,.776217012742,.757948024149,-.012285986059,-1.969634343031,.365852368240,1.700696194264,-2.062442810025])
I35 = np.asarray([.35,.773224951853,.736670712202,-.014166373990,-1.947952227745,.360804233227,1.659498525468,-2.074934849053])
I40 = np.asarray([.35,.772204719133,.729992579622,-.014723149227,-1.940296866527,.359260567438,1.645751085541,-2.079070907508])
U2 = np.asarray([.35,.770456213788,.734507752814,-.015956036218,-1.934675073521,.357435486193,1.644042683435,-2.077598421993])
U3 = np.asarray([.35,.769582668372,.736769141302,-.016568007509,-1.931851798792,.356528485065,1.643183990112,-2.076865812763])


def sample(node, baseline, event, *, contacts_allowed=False):
    node._clear_target_contact_samples(reset_robot_contact=True)
    if not node._wait_sim_duration(0.14):
        raise RuntimeError('contact dwell interrupted')
    current = physical_book_bounds()
    left, right = node._target_contact_sides(max_age=0.18)
    translation = float(np.linalg.norm(current[0] - baseline[0]))
    rotation = _rotation_distance(
        quaternion_matrix(current[1]),
        quaternion_matrix(baseline[1]),
    )
    print(json.dumps({
        'event': event,
        'position': current[0].tolist(),
        'minimum': current[2].tolist(),
        'maximum': current[3].tolist(),
        'translation_m': translation,
        'rotation_rad': rotation,
        'left': bool(left),
        'right': bool(right),
        'robot_contact': bool(node._target_robot_contact_latched),
        'solution': node._measured_left_solution().tolist(),
    }, sort_keys=True), flush=True)
    if node._target_robot_contact_latched:
        raise RuntimeError(f'non-gripper robot contact during {event}')
    if translation > 0.0008 or rotation > math.radians(1.0):
        raise RuntimeError(f'book moved during {event}')
    if not contacts_allowed and (left or right):
        raise RuntimeError(f'premature gripper contact during {event}')
    return left, right, current


def move(node, baseline, target, duration, event):
    if node._robot_self_collision(target) is not None:
        raise RuntimeError(f'self collision during {event}')
    if not node._move_arm_solution(target, duration):
        raise RuntimeError(f'controller failed during {event}')
    return sample(node, baseline, event)


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
        if float(baseline[0][0]) < 2.763:
            raise RuntimeError('book is not shelf-supported')

        for target, duration, event in (
            (HIGH, .45, 'open_hook_high'),
            (D1, .65, 'open_hook_down_1'),
            (D2, .65, 'open_hook_down_2'),
        ):
            move(node, baseline, target, duration, event)

        roll_start = node._measured_left_solution()
        for index in range(1, 5):
            target = roll_start.copy()
            target[-1] = D2[-1] - 1.55 * (index / 4.0)
            move(node, baseline, target, .55, f'open_hook_roll_{index}')

        for target, duration, event in (
            (I20, .55, 'open_hook_insert_20'),
            (I35, .48, 'open_hook_insert_35'),
            (I40, .42, 'open_hook_insert_40'),
            (U2, .42, 'open_hook_up_2mm'),
        ):
            move(node, baseline, target, duration, event)

        touch_start = node._measured_left_solution()
        touched = False
        touch_state = None
        for index in range(1, 5):
            target = touch_start + (U3 - touch_start) * (index / 4.0)
            if node._robot_self_collision(target) is not None:
                raise RuntimeError(f'self collision during touch {index}')
            if not node._move_arm_solution(target, .34):
                raise RuntimeError(f'controller failed during touch {index}')
            left, right, current = sample(
                node,
                baseline,
                f'open_hook_touch_{index}',
                contacts_allowed=True,
            )
            if right:
                raise RuntimeError('upper finger contacted before underside seating')
            if left:
                touched = True
                touch_state = current
                break
        if not touched:
            raise RuntimeError('lower fingertip did not reach the underside')
        print(json.dumps({
            'event': 'result',
            'passed': True,
            'stage': 'open_hook_underside_touch',
            'position': touch_state[0].tolist(),
            'minimum': touch_state[2].tolist(),
            'maximum': touch_state[3].tolist(),
            'solution': node._measured_left_solution().tolist(),
            'aperture': float(node.joints['gripper_left_finger_joint']),
        }, sort_keys=True), flush=True)
    finally:
        node._retention_probe_active = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
