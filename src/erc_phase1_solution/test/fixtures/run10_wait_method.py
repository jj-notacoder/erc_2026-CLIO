"""Frozen Run10 wait for behavior-preserving telemetry differential tests.
Source fine_gripper_close.py SHA256 d6bf009fc4b0229e035cbe0ab143fb3cad03b8c9015363d3778b5f249ec8af4a.
Only executed against fake clocks/nodes in tests.
"""
from dataclasses import asdict
import time

def _wait_motion_and_stationary(self, target, *, allow_bilateral_hold=False,
                                preclose=False, micro=False, wall_deadline=None):
    limits = self.limits
    wall_limit = limits.preclose_wall_seconds if preclose else limits.step_wall_seconds
    wall_end = min(self._fine_wall_deadline, time.monotonic() + wall_limit)
    if wall_deadline is not None:
        wall_end = min(wall_end, wall_deadline)
    stationary_since = None
    last_clock = int(self.get_clock().now().nanoseconds)
    last_progress_wall = time.monotonic()
    while time.monotonic() < wall_end:
        reason = self._feedback_error()
        if reason:
            self._hold(reason)
            return False
        stop = self._fine_stop
        if stop and not (allow_bilateral_hold and stop == 'bilateral_contact_observed'):
            return False
        feedback = self._latest_gripper_feedback()
        now = int(self.get_clock().now().nanoseconds)
        if now < last_clock:
            self._hold('clock_reversed')
            return False
        if now > last_clock:
            last_progress_wall, last_clock = time.monotonic(), now
        elif time.monotonic() - last_progress_wall > 1.:
            self._hold('clock_stalled')
            return False
        position_error = limits.micro_position_error if micro else limits.stationary_position_error
        stationary = (now >= self._fine_motion_end_ns
                      and abs(feedback.position - target) <= position_error
                      and abs(feedback.velocity) <= limits.stationary_velocity
                      and feedback.stamp_ns <= now)
        if stationary:
            if stationary_since is None:
                stationary_since = now
                if micro:
                    # Only post-motion stationary producer frames can
                    # certify this microtarget. Preserve identity/faults.
                    self._clear_target_contact_samples()
                    self._fine_micro_stationary_start_ns = stationary_since
            if now - stationary_since >= int(limits.dwell_seconds * 1e9):
                self._emit('stationary_endpoint', target=target, feedback=asdict(feedback),
                           stationary_start_ros_ns=stationary_since,
                           stop_reason=self._fine_stop, phase=self._fine_phase)
                return True
        else:
            stationary_since = None
            if micro:
                self._fine_micro_stationary_start_ns = None
        time.sleep(.002)
    self._hold('step_wall_timeout')
    return False
