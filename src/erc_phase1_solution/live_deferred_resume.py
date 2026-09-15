#!/usr/bin/env python3
"""Resume the temporary deferred-cradle probe from its locked start state."""

from __future__ import annotations

from types import SimpleNamespace
import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from live_deferred_cradle_probe import BOOK, FRONT, _test_helpers
from live_retreat_probe import ProbeManipulationNode, emit


class VerboseProbeNode(ProbeManipulationNode):
    def _publish_status(self, event, **fields):
        print(
            json.dumps({'event': f'status:{event}', **fields}, sort_keys=True),
            flush=True,
        )
        return super()._publish_status(event, **fields)


def main() -> None:
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node = VerboseProbeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    nav = SimpleNamespace(pose=None, last_command=SimpleNamespace())
    try:
        deadline = time.monotonic() + 20.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
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
        cached_start = np.asarray(cached_plan['start'], dtype=float)
        attached = np.asarray(cached_plan['attached_corners'], dtype=float)
        node._payload_monitor_enabled = False
        node._clear_target_contact_samples(
            reset_robot_contact=True,
            reset_target_model=True,
        )
        with node._lock:
            node._target_book_model = BOOK
        node._transport_lock_engaged = True
        node._wait_sim_duration(0.25)
        baseline = emit('resume_locked', node, nav, BOOK, None, command=0.029)
        verified, width, left, right, plausible = node._pinch_sample(
            max_age=0.15
        )
        measured_start = node._measured_left_solution()
        print(
            json.dumps(
                {
                    'event': 'resume_gate',
                    'verified': bool(verified),
                    'width': float(width),
                    'left': bool(left),
                    'right': bool(right),
                    'plausible': bool(plausible),
                    'torso_error': float(measured_start[0] - cached_start[0]),
                    'maximum_arm_error': float(
                        np.max(np.abs(measured_start[1:] - cached_start[1:]))
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not (verified and left and right and plausible):
            raise RuntimeError('resume state did not retain bilateral contact')
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
        succeeded = node._compact_transport()
        node._wait_sim_duration(0.30)
        emit(
            'resume_compact_complete',
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
            raise RuntimeError('deferred cradle resume probe failed')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
