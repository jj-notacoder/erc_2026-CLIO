#!/usr/bin/env python3
"""Temporary collision-audited route to rigid gripper-base support."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from erc_phase1_solution.kinematics import pose_matrix
from live_coupled_catch_probe import physical_book_bounds
from live_hook_extract_continue import Q_EXT
from live_lower_shelf_rescue import RescueProbeNode
from live_open_hook_touch import D1, HIGH
from live_pick_retreat_probe import BOOK, _rotation_distance, quaternion_matrix


U260 = np.asarray([.35,.697351187207,1.061690013751,-.059854191621,-1.720810605813,.303940533832,1.729358644769,-.419189885638])
VFAR_HI = np.asarray([.35,.379294706031,.752071665989,.358453269586,-1.867268396732,.984559479831,.856536206464,-1.037809183768])
VFAR = np.asarray([.35,.413637756743,.613901603249,.403682837994,-2.027458764675,1.066868822073,.911817792474,-1.021105724351])
V58 = np.asarray([.35,.388867592051,.529467885533,.433868597509,-1.914378797115,1.241782052030,.831650403791,-1.254685724378])
V62 = np.asarray([.35,.381317684635,.512856422371,.443246040390,-1.836432401396,1.331440181664,.806342323957,-1.379518448179])
V65 = np.asarray([.35,.377139882245,.508195586664,.451922547376,-1.772059069625,1.399586802762,.796707938344,-1.475523881540])
V68 = np.asarray([.35,.374652507563,.510087002278,.462065241341,-1.702060018459,1.466908478494,.795150684498,-1.572536035926])
V70_LOW = np.asarray([.35,.373949124210,.515024715620,.469910600391,-1.651921868096,1.510466746480,.798676132414,-1.637346684743])


class RigidPalmProbeNode(RescueProbeNode):
    def __init__(self):
        self.rigid_contact_ns = 0
        self.rigid_pairs = set()
        self.unexpected_robot_contact_ns = 0
        self.unexpected_robot_pairs = set()
        super().__init__()

    def _on_contacts(self, message):
        now_ns = self.get_clock().now().nanoseconds
        for contact in getattr(message, 'contacts', []):
            first = getattr(getattr(contact, 'collision1', None), 'name', '')
            second = getattr(getattr(contact, 'collision2', None), 'name', '')
            names = (str(first), str(second))
            lowered = tuple(name.lower() for name in names)
            if not any('book' in name for name in lowered):
                continue
            pair = tuple(sorted(names))
            if any('gripper_left_base_link' in name for name in lowered):
                self.rigid_contact_ns = now_ns
                self.rigid_pairs.add(pair)
            for name in lowered:
                if name.startswith('tiago_pro::') and '::gripper_left_' not in name:
                    self.unexpected_robot_contact_ns = now_ns
                    self.unexpected_robot_pairs.add(pair)
        super()._on_contacts(message)

    def clear_probe_contacts(self):
        self.rigid_contact_ns = 0
        self.rigid_pairs.clear()
        self.unexpected_robot_contact_ns = 0
        self.unexpected_robot_pairs.clear()
        self._clear_target_contact_samples(reset_robot_contact=True)

    def fresh_rigid_contact(self, max_age=.25):
        if self.rigid_contact_ns <= 0:
            return False
        age = (self.get_clock().now().nanoseconds - self.rigid_contact_ns) / 1e9
        return 0.0 <= age <= max_age


def solve_offset(node, start, dz):
    pose = node.chain.forward(start)
    position = pose[:3, 3].copy()
    position[2] += float(dz)
    solved, _ = node.chain.solve(
        pose_matrix(position, pose[:3, :3]),
        [start],
        position_tolerance=.00003,
        orientation_tolerance=.0007,
        max_iterations=400,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solved is None:
        raise RuntimeError(f'rigid-palm touch IK failed at dz={dz}')
    return np.asarray(solved, dtype=float)


def inspect(node, baseline, event):
    if not node._wait_sim_duration(.17):
        raise RuntimeError('contact dwell interrupted')
    current = physical_book_bounds()
    left, right = node._target_contact_sides(max_age=.22)
    rigid = node.fresh_rigid_contact(.22)
    translation = float(np.linalg.norm(current[0] - baseline[0]))
    rotation = _rotation_distance(
        quaternion_matrix(current[1]), quaternion_matrix(baseline[1])
    )
    print(json.dumps({
        'event': event,
        'rigid': bool(rigid),
        'rigid_pairs': sorted(node.rigid_pairs),
        'left': bool(left),
        'right': bool(right),
        'translation_m': translation,
        'rotation_rad': rotation,
        'position': current[0].tolist(),
        'minimum': current[2].tolist(),
        'maximum': current[3].tolist(),
        'solution': node._measured_left_solution().tolist(),
    }, sort_keys=True), flush=True)
    if (
        translation > .0006
        or rotation > math.radians(.8)
        or node.unexpected_robot_contact_ns > 0
    ):
        raise RuntimeError(f'book or robot moved unsafely during {event}')
    return bool(rigid), bool(left), bool(right), current


def move(node, target, duration, baseline, event):
    if node._robot_self_collision(target) is not None:
        raise RuntimeError(f'self collision during {event}')
    node.clear_probe_contacts()
    if not node._move_arm_solution(target, duration):
        raise RuntimeError(f'controller failed during {event}')
    return inspect(node, baseline, event)


def main():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = RigidPalmProbeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(.05)
        if len(node.joints) < 8:
            raise RuntimeError('joint state unavailable')
        node._target_book_model = BOOK
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        if float(node.joints['gripper_left_finger_joint']) < .067:
            raise RuntimeError('gripper is not open')
        baseline = physical_book_bounds()

        for target, duration, event in (
            (D1, .55, 'rigid_return_d1'),
            (HIGH, .55, 'rigid_return_high'),
            (Q_EXT, .45, 'rigid_return_qext'),
            (U260, .60, 'rigid_u260'),
            (VFAR_HI, 1.20, 'rigid_vertical_far_high'),
            (VFAR, .70, 'rigid_vertical_far'),
            (V58, .50, 'rigid_insert_58'),
            (V62, .45, 'rigid_insert_62'),
            (V65, .40, 'rigid_insert_65'),
            (V68, .36, 'rigid_insert_68'),
            (V70_LOW, .34, 'rigid_insert_70_low'),
        ):
            rigid, left, right, _ = move(node, target, duration, baseline, event)
            if rigid or left or right:
                raise RuntimeError(f'premature gripper contact during {event}')

        touch_start = node._measured_left_solution()
        terminal = None
        for dz in (.001, .002, .003, .004, .005, .00525, .00550, .00575):
            target = solve_offset(node, touch_start, dz)
            rigid, left, right, current = move(
                node, target, .32, baseline, f'rigid_touch_{int(round(dz * 1000000))}um'
            )
            if left or right:
                raise RuntimeError('finger contacted before rigid palm support')
            if rigid:
                terminal = current
                break
        if terminal is None:
            raise RuntimeError('rigid palm did not contact within audited touch range')
        print(json.dumps({
            'event': 'result',
            'passed': True,
            'stage': 'rigid_palm_touch',
            'rigid_pairs': sorted(node.rigid_pairs),
            'position': terminal[0].tolist(),
            'minimum': terminal[2].tolist(),
            'maximum': terminal[3].tolist(),
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
