"""Default-off initial serial stow timing for a controlled simulator screen.

Original opening,2.5s torso, left/right order and3.5s watchdogs stay in _follow.
Only the two arm endpoints receive2.0s timing, independent of PLACE scaling.
Fresh80% slope checks are velocity admission, not a tracking/retention proof.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from .motion_profiles import HOME, RIGHT_HOME, ARM_JOINTS, IK_JOINTS, RIGHT_ARM_JOINTS
from .completed_torso_hold import TorsoHoldCancellation, require_follow_token_locked

HEAD = ('head_1_joint', 'head_2_joint')
MASTERS = ('gripper_left_finger_joint', 'gripper_right_finger_joint')
NAMES = (*IK_JOINTS, *RIGHT_ARM_JOINTS, *HEAD, *MASTERS)
ARM_NAMES = (*ARM_JOINTS, *RIGHT_ARM_JOINTS)
MASTER_ENVELOPE = .0005
ENDPOINT_TOLERANCE = .005
STOP_VELOCITY = .01
NOMINAL_DURATION = 3.5
ARM_DURATION = 2.0


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('initial_stow_serial_timing_enabled must be Boolean')
    return value


@dataclass(frozen=True)
class Limits:
    names: tuple
    lower: tuple
    upper: tuple
    velocity: tuple
    urdf_sha256: str


def load_limits(path):
    raw = Path(path).read_bytes()
    root = ET.fromstring(raw)
    rows = []
    for name in ARM_NAMES:
        joints = [j for j in root.findall('joint') if j.get('name') == name]
        if len(joints) != 1 or joints[0].get('type') != 'revolute':
            raise ValueError('official arm joint missing/ambiguous:' + name)
        lim = joints[0].find('limit')
        if lim is None:
            raise ValueError('official arm limit absent:' + name)
        values = tuple(float(lim.get(k)) for k in ('lower', 'upper', 'velocity'))
        if not all(math.isfinite(x) for x in values) or not values[0] < values[1] or values[2] <= 0:
            raise ValueError('official arm limit invalid:' + name)
        rows.append(values)
    return Limits(tuple(ARM_NAMES), *(tuple(r[i] for r in rows) for i in range(3)),
                  hashlib.sha256(raw).hexdigest())


def begin_command(node, command):
    if not checked_enabled(getattr(node, 'initial_stow_serial_timing_enabled', False)):
        return None
    with node._lock:
        first = not getattr(node, '_initial_stow_serial_command_seen', False)
        node._initial_stow_serial_command_seen = True
        if getattr(node, '_initial_stow_serial_owner', None) is not None:
            raise RuntimeError('initial stow serial timing owner unresolved')
        # Only the very first ordinary command can use this option. Later stow
        # calls retain the original serial implementation, never a retry owner.
        if not first or command != 'stow' or node.dry_run:
            return None
        if getattr(node, '_right_parked', False):
            raise RuntimeError('initial stow already parked')
        owner = Owner(node)
        node._initial_stow_serial_owner = owner
        owner.require_empty_locked()
        return owner


class Owner:
    def __init__(self, node):
        self.node, self.cancel = node, node._cancel
        self.limits = node._initial_stow_serial_limits
        if type(self.limits) is not Limits:
            raise RuntimeError('initial stow official limits unavailable')
        if (type(node._cancel) is not TorsoHoldCancellation
                or not getattr(node, 'settled_place_torso_skip_enabled', False)):
            raise RuntimeError('initial stow timing requires original tracked sender')
        self.cancel_generation = self.cancel.snapshot()[0]
        self.models = (node.chain, node.right_chain, node.head_chain, node.carried_collision_meshes)
        self.tokens = (object(), object())
        self.side = 0
        self.acceptance = self.handle = self.result_future = None
        self.sent = self.late_registered = self.completed = self.cancelling = False
        self.last_ros = self.last_odom = None
        self.last_stamps = {}
        self.start = self.pre_send = None
        self.follow_token = self.returned_ns = None
        self.terminal_complete = False
        self.torso_completed_ns = None
        self.torso_settle_active = self.torso_settle_reported = False
        self.torso_settle_diagnostic = None
        self.torso_settle_wall_budget = 3.

    def require_empty_locked(self):
        n = self.node
        if (getattr(n, '_initial_stow_serial_owner', None) is not self
                or not checked_enabled(n.initial_stow_serial_timing_enabled)
                or n._cancel is not self.cancel or self.cancel.snapshot() != (self.cancel_generation, False)
                or n._initial_stow_serial_limits is not self.limits
                or any(a is not b for a, b in zip(self.models, (n.chain, n.right_chain, n.head_chain, n.carried_collision_meshes)))
                or not n._busy or self.completed):
            raise RuntimeError('initial stow owner/model/cancel changed')
        for field in ('_held_book_corners', '_active_place_scene_reference', '_place_contact_guard',
                      '_empty_torso_planning_owner', '_head_return_state', '_empty_head_timing_owner',
                      '_payload_hazard_latched', '_held_grip_sensor_fault', '_raw_contacts_first_failure',
                      '_release_pose_owner', '_release_pose_fault_latched'):
            if getattr(n, field, None) is not None:
                raise RuntimeError('initial stow occupied/fault:' + field)
        for field in ('_empty_arm_staged', '_target_robot_contact_latched', '_empty_arm_contact_latched',
                      '_adaptive_close_active', '_transport_lock_engaged', '_retention_probe_active',
                      '_empty_arm_motion_active', '_empty_pickup_geometry_active', '_pickup_parallel_geometry_active'):
            if getattr(n, field, False):
                raise RuntimeError('initial stow context active:' + field)
        own = (self.handle,) if self.handle is not None else ()
        if any(h not in own for h in n._goal_handles):
            raise RuntimeError('initial stow conflicting active goal')
        if not set(n._pending_retained_acceptances).issubset(self.tokens):
            raise RuntimeError('initial stow conflicting pending goal')

    def snapshot_locked(self, *, stationary=False, torso_after=None):
        self.require_empty_locked()
        if torso_after is not None and (
                type(torso_after) is not int or torso_after <= 0
                or torso_after != self.torso_completed_ns
                or self.start is not None or self.side != 0 or self.sent
                or self.acceptance is not None or self.handle is not None
                or self.follow_token is not None or self.node._goal_handles
                or self.node._pending_retained_acceptances):
            raise RuntimeError('initial stow torso settling phase identity changed')
        n = self.node
        now = int(n.get_clock().now().nanoseconds)
        if now < 0 or (self.last_ros is not None and now < self.last_ros):
            raise RuntimeError('initial stow ROS clock reversed')
        self.last_ros = now
        positions, velocities, stamps = {}, {}, {}
        for name in NAMES:
            q, v = n.joints.get(name), n._joint_velocities.get(name)
            stamp = n._joint_stamps_ns.get(name)
            if (isinstance(q, bool) or isinstance(v, bool) or q is None or v is None
                    or not math.isfinite(q) or not math.isfinite(v) or type(stamp) is not int
                    or stamp <= 0 or not 0 <= now-stamp <= 150_000_000
                    or stamp < self.last_stamps.get(name, stamp)):
                raise RuntimeError('initial stow invalid/stale/reversed joint:' + name)
            positions[name], velocities[name], stamps[name] = float(q), float(v), stamp
            self.last_stamps[name] = stamp
        for i, name in enumerate(ARM_NAMES):
            if not self.limits.lower[i] <= positions[name] <= self.limits.upper[i] or abs(velocities[name]) > self.limits.velocity[i]:
                raise RuntimeError('initial stow measured official arm limit:' + name)
            if stationary and abs(velocities[name]) > STOP_VELOCITY:
                raise RuntimeError('initial stow moving arm:' + name)
        odom = getattr(n, '_staging_odom', None)
        if not isinstance(odom, dict) or type(odom.get('stamp_ns')) is not int:
            raise RuntimeError('initial stow odometry unavailable')
        stamp = odom['stamp_ns']
        pose = tuple(odom.get('pose', ()))
        speeds = (odom.get('linear_speed'), odom.get('angular_speed'))
        if (len(pose) != 3 or any(isinstance(x, bool) or x is None or not math.isfinite(x) for x in (*pose, *speeds))
                or stamp <= 0 or not 0 <= now-stamp <= 150_000_000
                or (self.last_odom is not None and stamp < self.last_odom)
                or not 0 <= speeds[0] <= .005 or not 0 <= speeds[1] <= .008):
            raise RuntimeError('initial stow base not fresh/stationary')
        self.last_odom = stamp
        if torso_after is None and (abs(positions[IK_JOINTS[0]]-HOME[0]) > .002
                                    or abs(velocities[IK_JOINTS[0]]) > .001):
            raise RuntimeError('initial stow torso not at original stopped HOME')
        if any(abs(velocities[name]) > .001 for name in (*HEAD, *MASTERS)):
            raise RuntimeError('initial stow parked head/gripper moving')
        if abs(positions[MASTERS[0]]-n.gripper_open) > MASTER_ENVELOPE or not 0 <= positions[MASTERS[1]] <= .069:
            raise RuntimeError('initial stow gripper opening unmeasured')
        if self.start is not None:
            old = self.start
            yaw = math.atan2(math.sin(pose[2]-old['base'][2]), math.cos(pose[2]-old['base'][2]))
            if math.hypot(pose[0]-old['base'][0], pose[1]-old['base'][1]) > .002 or abs(yaw) > .005:
                raise RuntimeError('initial stow base drift')
            for name in (IK_JOINTS[0], *HEAD, *MASTERS):
                bound = .002 if name == IK_JOINTS[0] else MASTER_ENVELOPE if name in MASTERS else .003
                if abs(positions[name]-old['positions'][name]) > bound:
                    raise RuntimeError('initial stow parked joint drift:' + name)
            parked_names = RIGHT_ARM_JOINTS if self.side == 0 else ARM_JOINTS
            parked_values = ([old['positions'][name] for name in parked_names]
                             if self.side == 0 else HOME[1:])
            tolerance = .003 if self.side == 0 else ENDPOINT_TOLERANCE
            for name, target in zip(parked_names, parked_values):
                if (abs(positions[name]-target) > tolerance
                        or abs(velocities[name]) > STOP_VELOCITY):
                    raise RuntimeError('initial stow inactive arm moved:' + name)
        return dict(now=now, positions=positions, velocities=velocities, stamps=stamps,
                    base=pose, odom_stamp_ns=stamp)

    def check(self, stationary=False):
        with self.node._lock:
            return self.snapshot_locked(stationary=stationary)

    def report_torso_settle_failure(self, error):
        if not self.torso_settle_active or self.torso_settle_reported:
            return
        self.torso_settle_reported = True
        # One bounded diagnostic, using only the most recent validated sample.
        # A missing sample is explicit; malformed/stale inputs are never coerced.
        try:
            self.node._publish_status('initial_stow_torso_settle_refused',
                reason=str(error)[:256], completion_ros_ns=self.torso_completed_ns,
                last_valid_sample=self.torso_settle_diagnostic,
                position_tolerance_m=.002, velocity_limit_mps=.001,
                ros_budget_seconds=.75, wall_budget_seconds=self.torso_settle_wall_budget,
                wall_progress_timeout_seconds=3.,
                required_producer_span_ns=100_000_000)
        except Exception:
            pass  # Diagnostics cannot replace the original fault or ownership.

    def wait_stopped(self, *, after=None, torso_after=None):
        if torso_after is not None and after is not None:
            raise RuntimeError('initial stow ambiguous measured stop phase')
        first = None
        clock = self.node.get_clock()
        simulated = bool(getattr(clock, 'ros_time_is_active', False))
        progress_timeout = 3. if after is None else min(self.node.timeout, 8.)
        # Settling and producer spans are measured in ROS time. A slow Gazebo
        # clock must get the same settling interval as a real-time simulator.
        # Keep a wall watchdog for stalled clocks/feedback, and bound the whole
        # wait by the existing command timeout even if progress is very slow.
        wall_budget = max(progress_timeout, self.node.timeout) if simulated else progress_timeout
        started_wall = time.monotonic()
        wall = started_wall + wall_budget
        progress_wall = started_wall + progress_timeout
        progress = None
        start_ros = int(clock.now().nanoseconds)
        ros_budget = (750_000_000 if torso_after is not None
                      else 500_000_000 if after is None else 2_000_000_000)
        if torso_after is not None:
            self.torso_settle_active = True
            self.torso_settle_wall_budget = wall_budget
        while time.monotonic() < min(wall, progress_wall):
            if torso_after is None:
                snap = self.check()
            else:
                with self.node._lock:
                    snap = self.snapshot_locked(torso_after=torso_after)
                q = snap['positions'][IK_JOINTS[0]]
                v = snap['velocities'][IK_JOINTS[0]]
                self.torso_settle_diagnostic = dict(
                    sample_ros_ns=snap['now'], joint_producer_ns=snap['stamps'][IK_JOINTS[0]],
                    odom_producer_ns=snap['odom_stamp_ns'], q=q, v=v, target=float(HOME[0]),
                    error_m=q-float(HOME[0]), position_within=bool(abs(q-float(HOME[0])) <= .002),
                    velocity_within=abs(v) <= .001,
                    all_producers_after_completion=(all(snap['stamps'][name] > torso_after
                        for name in NAMES) and snap['odom_stamp_ns'] > torso_after),
                    arm_max_abs_velocity=max(abs(snap['velocities'][name]) for name in ARM_NAMES),
                    arms_within_stop_limit=all(abs(snap['velocities'][name]) <= STOP_VELOCITY for name in ARM_NAMES),
                    arm_stop_velocity_limit=STOP_VELOCITY,
                    wait_elapsed_ros_ns=snap['now']-start_ros,
                    # Saved window state before this sample's deadline/admission check.
                    first_good_ros_ns=None if first is None else first['now'],
                    minimum_joint_producer_span_ns=None if first is None else min(
                        snap['stamps'][name]-first['stamps'][name] for name in NAMES),
                    odom_producer_span_ns=None if first is None else snap['odom_stamp_ns']-first['odom_stamp_ns'])
            if snap['now']-start_ros > ros_budget:
                raise TimeoutError('initial stow measured stop ROS deadline')
            if simulated and (progress is None or (
                    snap['now'] > progress['now']
                    and all(snap['stamps'][name] > progress['stamps'][name] for name in NAMES)
                    and snap['odom_stamp_ns'] > progress['odom_stamp_ns'])):
                progress = snap
                progress_wall = time.monotonic() + progress_timeout
            good = all(abs(snap['velocities'][name]) <= STOP_VELOCITY for name in ARM_NAMES)
            if torso_after is not None:
                good = (good and abs(snap['positions'][IK_JOINTS[0]]-HOME[0]) <= .002
                        and abs(snap['velocities'][IK_JOINTS[0]]) <= .001
                        and all(snap['stamps'][name] > torso_after for name in NAMES)
                        and snap['odom_stamp_ns'] > torso_after)
            if after is not None:
                names, target = self.arm()
                good = (good and all(abs(snap['positions'][name]-q) <= ENDPOINT_TOLERANCE
                        for name, q in zip(names, target))
                        and all(snap['stamps'][name] > after for name in NAMES)
                        and snap['odom_stamp_ns'] > after)
            if good:
                if first is None:
                    first = snap
                elif (all(snap['stamps'][name]-first['stamps'][name] >= 100_000_000 for name in NAMES)
                      and snap['odom_stamp_ns']-first['odom_stamp_ns'] >= 100_000_000):
                    if torso_after is not None:
                        self.torso_settle_active = False
                    return snap
            else:
                first = None
            time.sleep(.02)
        raise TimeoutError('initial stow measured stop wall deadline')

    def arm(self):
        if self.side not in (0, 1):
            raise RuntimeError('initial stow serial sequence complete')
        return (ARM_JOINTS, HOME[1:]) if self.side == 0 else (RIGHT_ARM_JOINTS, RIGHT_HOME)

    def slope_locked(self, snapshot):
        names, target = self.arm()
        for name, end in zip(names, target):
            index = ARM_NAMES.index(name)
            if (not self.limits.lower[index] <= end <= self.limits.upper[index]
                    or abs(float(end)-snapshot['positions'][name])/ARM_DURATION > .8*self.limits.velocity[index]):
                raise RuntimeError('initial stow2s80percent official velocity gate:' + name)

    def follow_started(self, client, names, positions, duration, token):
        with self.node._lock:
            expected_names, target = self.arm()
            expected_client = self.node.arm_client if self.side == 0 else self.node.right_arm_client
            if (self.sent or self.start is None or token is None
                    or client is not expected_client or tuple(names) != tuple(expected_names)
                    or tuple(positions) != tuple(target) or duration != NOMINAL_DURATION):
                raise RuntimeError('initial stow original serial command identity changed')
            self.follow_token = token
            self.pre_send = self.snapshot_locked(stationary=True)
            self.slope_locked(self.pre_send)

    def command_duration(self):
        return ARM_DURATION

    def admit_locked(self, client, goal, token):
        names, target = self.arm()
        expected_client = self.node.arm_client if self.side == 0 else self.node.right_arm_client
        if (self.sent or self.pre_send is None or token is not self.follow_token
                or client is not expected_client or tuple(goal.trajectory.joint_names) != tuple(names)
                or len(goal.trajectory.points) != 1):
            raise RuntimeError('initial stow final publication identity changed')
        point = goal.trajectory.points[0]
        if (tuple(point.positions) != tuple(target) or point.velocities or point.accelerations or point.effort
                or (point.time_from_start.sec, point.time_from_start.nanosec) != (2, 0)
                or goal.trajectory.header.stamp.sec != 0 or goal.trajectory.header.stamp.nanosec != 0
                or goal.goal_time_tolerance.sec != 0 or goal.goal_time_tolerance.nanosec != 0
                or goal.path_tolerance or goal.goal_tolerance):
            raise RuntimeError('initial stow original goal path or2s timing changed')
        snap = self.snapshot_locked(stationary=True)
        if any(abs(snap['positions'][name]-self.pre_send['positions'][name]) > .001 for name in ARM_NAMES):
            raise RuntimeError('initial stow start changed during server wait')
        self.slope_locked(snap)
        require_follow_token_locked(self.node, token, goal)
        self.node._pending_retained_acceptances.add(self.tokens[self.side])
        self.sent = True  # Before publication: an exception is not proof of no send.

    def sent_locked(self, acceptance):
        self.acceptance = acceptance

    def wait_acceptance(self, acceptance, timeout):
        deadline = time.monotonic()+timeout
        while not acceptance.done():
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError('initial stow original acceptance watchdog')
            time.sleep(.02)
        handle = acceptance.result()
        if handle is None or type(handle.accepted) is not bool:
            raise RuntimeError('initial stow malformed acceptance')
        with self.node._lock:
            # Capture before the next fallible check or get_result_async call.
            self.handle = handle
            if not handle.accepted:
                self.node._pending_retained_acceptances.discard(self.tokens[self.side])
        self.check()
        return handle

    def accepted_locked(self, handle, result_future):
        if handle is not self.handle or not handle.accepted:
            raise RuntimeError('initial stow accepted goal identity changed')
        if handle not in self.node._goal_handles:
            raise RuntimeError('initial stow accepted handle not globally registered')
        self.result_future = result_future
        self.node._pending_retained_acceptances.discard(self.tokens[self.side])
        self.snapshot_locked()

    def terminal(self, future):
        self.check()
        result = future.result()
        if (future is not self.result_future or not future.done()
                or not self.node._valid_retained_terminal_result(result)
                or result.status != 4 or result.result.error_code != 0):
            raise RuntimeError('initial stow controller terminal failure')
        self.returned_ns = int(self.node.get_clock().now().nanoseconds)
        self.terminal_complete = True

    def finish_leg(self):
        if not self.terminal_complete or self.returned_ns is None:
            raise RuntimeError('initial stow missing original action completion')
        self.wait_stopped(after=self.returned_ns)
        with self.node._lock:
            snap = self.snapshot_locked(stationary=True)
            names, target = self.arm()
            if any(abs(snap['positions'][name]-q) > ENDPOINT_TOLERANCE for name, q in zip(names, target)):
                raise RuntimeError('initial stow endpoint changed after measured stop')
            self.side += 1
            self.sent = self.late_registered = self.terminal_complete = False
            self.acceptance = self.handle = self.result_future = None
            self.follow_token = self.pre_send = self.returned_ns = None
            if self.side == 2:
                self.node._initial_stow_serial_owner = None
                self.node._right_parked = True
                self.completed = True

    def cancel_owned(self):
        n = self.node
        n._cancel.set()
        if self.cancelling:
            return
        self.cancelling = True
        handle = self.handle
        if handle is None and self.acceptance is not None and not self.late_registered:
            self.late_registered = True
            try:
                self.acceptance.add_done_callback(
                    lambda future, token=self.tokens[self.side]: n._cancel_late_retained_goal(future, token))
            except Exception:
                pass
        if handle is None or not handle.accepted:
            return
        try:
            handle.cancel_goal_async()
        except Exception:
            pass
        try:
            result = self.result_future
            if result is None:
                result = handle.get_result_async()
                self.result_future = result
            n._cancel_retained_goal_and_confirm(handle, result)
            if result.done() and n._valid_retained_terminal_result(result.result()):
                with n._lock:
                    if handle in n._goal_handles:
                        n._goal_handles.remove(handle)
                    n._pending_retained_acceptances.discard(self.tokens[self.side])
        except Exception:
            pass


def run_stow(node, owner):
    if type(owner) is not Owner or owner.node is not node:
        raise RuntimeError('initial stow serial owner mismatch')
    try:
        with node._lock:
            owner.require_empty_locked()
        if not node._open_gripper():
            raise RuntimeError('initial stow original opening failed')
        if not node._follow(node.torso_client, ['torso_lift_joint'], [HOME[0]], 2.5):
            raise RuntimeError('initial stow original torso move failed')
        with node._lock:
            owner.require_empty_locked()
            completed_ns = node.get_clock().now().nanoseconds
            if (type(completed_ns) is not int or completed_ns <= 0
                    or (owner.last_ros is not None and completed_ns < owner.last_ros)):
                raise RuntimeError('initial stow invalid torso completion clock')
            owner.torso_completed_ns = completed_ns
            owner.last_ros = completed_ns
        owner.start = owner.wait_stopped(torso_after=completed_ns)
        for client, names, target in ((node.arm_client, ARM_JOINTS, HOME[1:]),
                                       (node.right_arm_client, RIGHT_ARM_JOINTS, RIGHT_HOME)):
            if not node._follow(client, names, target, NOMINAL_DURATION, initial_stow_owner=owner):
                raise RuntimeError('initial stow original serial arm action failed')
            owner.finish_leg()
        return True
    except BaseException as error:
        owner.cancel_owned()
        owner.report_torso_settle_failure(error)
        raise
