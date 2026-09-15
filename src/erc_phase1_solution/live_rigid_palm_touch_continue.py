#!/usr/bin/env python3
"""Temporary continuation from V65 to the deeper rigid-palm touch."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_coupled_catch_probe import physical_book_bounds
from live_pick_retreat_probe import BOOK
from live_rigid_palm_touch import RigidPalmProbeNode, V70_LOW, move


V705_LOW = np.asarray([.35,.373876355599,.516733063469,.472061125985,-1.638911196863,1.521109075047,.800145620698,-1.653542517342])
V710_LOW = np.asarray([.35,.373878203317,.518623469387,.474247055263,-1.625706006669,1.531643485168,.801831162366,-1.669719061621])
V710_U4 = np.asarray([.35,.372956005871,.528346486547,.471505214545,-1.616424321031,1.527301257255,.798504362357,-1.670678403714])
V710_U5 = np.asarray([.35,.372729064154,.530781702803,.470824516415,-1.614083212578,1.526235525289,.797672704588,-1.670935669871])
V710_TOUCH = np.asarray([.35,.372648636041,.531649146839,.470582682252,-1.613247656781,1.525858120652,.797376699454,-1.671029065894])


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
        baseline = physical_book_bounds()
        terminal = None
        for target, duration, event, may_touch in (
            (V70_LOW, .40, 'rigid_deep_lower_70', False),
            (V705_LOW, .32, 'rigid_deep_insert_70_5', False),
            (V710_LOW, .32, 'rigid_deep_insert_71', False),
            (V710_U4, .45, 'rigid_deep_up_4mm', False),
            (V710_U5, .32, 'rigid_deep_up_5mm', False),
            (V710_TOUCH, .30, 'rigid_deep_touch', True),
        ):
            rigid, left, right, current = move(
                node, target, duration, baseline, event
            )
            if left or right:
                raise RuntimeError(f'finger contact during {event}')
            if rigid and not may_touch:
                raise RuntimeError(f'premature rigid contact during {event}')
            if may_touch:
                terminal = current
        if terminal is None:
            raise RuntimeError('audited rigid-palm tangent was not reached')
        print(json.dumps({
            'event': 'result',
            'passed': True,
            'stage': 'rigid_palm_tangent_deep',
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
