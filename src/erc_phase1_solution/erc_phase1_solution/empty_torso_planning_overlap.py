"""Default-off ownership of one prechecked empty torso ascent during planning.

No collision evaluator, alternate torso publisher or hold command lives here.
An unsuccessful owner remains an explicit command interlock.
"""
from contextlib import contextmanager
import copy
import math
import threading
import time

import numpy as np

from .completed_torso_hold import TorsoHoldCancellation
from .empty_pickup_collision import (
    EmptyPickupCollision, HEAD, MASTER, NAMES, wait_for_geometry_endpoint,
)
from .motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS


def checked_enabled(value):
    if type(value) is not bool:
        raise ValueError('empty_torso_planning_overlap_enabled must be Boolean')
    return value


class EmptyTorsoPlanningOwner:
    """One attempt: pending action, physical ascent, stop and planner join."""
    def __init__(self, node, checker, reference):
        if (type(checker) is not EmptyPickupCollision or checker.node is not node
                or type(node._cancel) is not TorsoHoldCancellation
                or not getattr(node, 'settled_place_torso_skip_enabled', False)):
            raise RuntimeError('empty torso overlap requires ordinary checker and tracked torso sender')
        self.node, self.checker = node, checker
        self.reference = copy.deepcopy(reference)
        self.frozen_vectors = {name: np.frombuffer(getattr(checker, name).tobytes(), dtype=float)
            for name in ('start', 'right', 'head', 'velocity_limits')}
        self.start = self.frozen_vectors['start']
        self.initial_aperture = float(checker.initial_aperture)
        self.open_aperture = float(checker.open_aperture)
        self.aperture_tolerance = float(node.adaptive_endpoint_tolerance)
        self.target = self.start.copy(); self.target[0] = float(node.pick_torso_height)
        if not (np.isfinite(self.target).all() and self.start[0] <= self.target[0]
                and float(node.chain.lower[0]) <= self.start[0]
                and self.target[0] <= float(node.chain.upper[0])):
            raise RuntimeError('empty torso overlap target outside admitted upward segment')
        self.models = tuple(getattr(node, n) for n in
            ('chain', 'right_chain', 'head_chain', 'carried_collision_meshes', '_shelf_cradle_geometry'))
        self.components = (checker.geometry, checker.screen)
        self.token = object()
        self.acceptance = self.handle = self.result_future = self.thread = None
        self.follow_token = None
        self.fault = None
        self.done = threading.Event()
        self.motion_succeeded = self.measured_stopped = self.joined = False
        self.stopped_stamp = None
        self.dispatched = self.late_callback_registered = False
        self.last_ros = None; self.last_stamps = {}
        self.cancel_generation = node._cancel.snapshot()[0]
        self.torso_state = node._torso_hold_state
        self.prior_generation = self.torso_state.generation
        self.contact_epoch = None
        self.target_model = None
        self.original_contact_guard = getattr(node, '_empty_arm_contact_guard', False)
        self.started_wall = time.monotonic()
        # Bounds only the new owner; the sender/endpoint retain their old bounds.
        self.wall_deadline = self.started_wall + min(90., float(node.timeout) + 50.)

    def _fault(self, error):
        with self.node._lock:
            if self.fault is None:
                self.fault = str(error)

    def _require_plan_locked(self):
        n, g = self.node, self.checker
        if (any(getattr(n, name) is not obj for name, obj in zip(
                ('chain', 'right_chain', 'head_chain', 'carried_collision_meshes', '_shelf_cradle_geometry'), self.models))
                or g.node is not n or any(a is not b for a, b in zip(
                    (g.geometry, g.screen), self.components))
                or any(getattr(g, name).dtype != value.dtype
                    or getattr(g, name).shape != value.shape
                    or getattr(g, name).tobytes() != value.tobytes()
                    for name, value in self.frozen_vectors.items())
                or g.initial_aperture != self.initial_aperture
                or g.open_aperture != self.open_aperture
                or float(n.adaptive_endpoint_tolerance) != self.aperture_tolerance
                or float(n.pick_torso_height) != float(self.target[0])
                or float(n.gripper_open) != self.open_aperture):
            raise RuntimeError('empty torso overlap model/plan changed')

    def _require_locked(self, *, before_send=False):
        """Additional locked re-admission; original measured function runs outside."""
        n, g = self.node, self.checker
        now = int(n.get_clock().now().nanoseconds)
        generation, cancelled = n._cancel.snapshot()
        if (getattr(n, '_empty_torso_planning_owner', None) is not self or self.fault
                or cancelled or generation != self.cancel_generation):
            raise RuntimeError('empty torso overlap owner/cancel invalidated')
        if self.last_ros is not None and now < self.last_ros:
            raise RuntimeError('empty torso overlap clock regressed')
        self.last_ros = now
        if (time.monotonic() >= self.wall_deadline
                or getattr(n, '_held_book_corners', None) is not None
                or getattr(n, '_active_place_scene_reference', None) is not None
                or getattr(n, '_payload_hazard_latched', None) is not None
                or getattr(n, '_held_grip_sensor_fault', None) is not None
                or getattr(n, '_target_robot_contact_latched', False)
                or getattr(n, '_empty_arm_contact_latched', False)
                or getattr(n, '_raw_contacts_first_failure', None) is not None):
            raise RuntimeError('empty torso overlap physical/context interlock')
        self._require_plan_locked()
        if self.contact_epoch != int(getattr(n, '_contact_epoch', 0)):
            raise RuntimeError('empty torso overlap contact epoch changed')
        if getattr(n, '_target_book_model', None) != self.target_model:
            raise RuntimeError('empty torso overlap target identity changed')
        if not getattr(n, '_empty_arm_contact_guard', False):
            raise RuntimeError('empty torso overlap contact guard removed')
        if self.torso_state is not n._torso_hold_state or self.torso_state.uncertain:
            raise RuntimeError('empty torso overlap torso ownership uncertain')
        expected_generation = self.prior_generation + (1 if self.follow_token is not None else 0)
        if self.torso_state.generation != expected_generation:
            raise RuntimeError('empty torso overlap torso generation changed')
        for name in NAMES:
            value, stamp = n.joints.get(name), n._joint_stamps_ns.get(name)
            if (value is None or not math.isfinite(value) or type(stamp) is not int
                    or not -50_000_000 <= now-stamp <= 350_000_000):
                raise RuntimeError('empty torso overlap stale joint:' + name)
            if name in self.last_stamps and stamp < self.last_stamps[name]:
                raise RuntimeError('empty torso overlap producer regressed:' + name)
            self.last_stamps[name] = stamp
        for name, expected in zip(IK_JOINTS[1:], self.start[1:]):
            if abs(n.joints[name]-float(expected)) > .008:
                raise RuntimeError('empty torso overlap parked left arm moved')
        for names in (RIGHT_ARM_JOINTS, HEAD):
            for name in names:
                if abs(n.joints[name]-float(self.reference['joints'][name])) > .001:
                    raise RuntimeError('empty torso overlap parked right/head moved')
        torso = n.joints[IK_JOINTS[0]]
        if not self.start[0]-.002 <= torso <= self.target[0]+.002:
            raise RuntimeError('empty torso overlap left checked torso interval')
        if before_send and abs(torso-self.start[0]) > .002:
            raise RuntimeError('empty torso overlap start moved before send')
        if abs(n.joints[MASTER]-self.open_aperture) > self.aperture_tolerance:
            raise RuntimeError('empty torso overlap gripper changed')
        odom = getattr(n, '_staging_odom', None)
        if (not isinstance(odom, dict) or not -50_000_000 <= now-odom.get('stamp_ns', 0) <= 350_000_000):
            raise RuntimeError('empty torso overlap odometry stale')
        pose = np.asarray(odom.get('pose'), dtype=float)
        speeds = np.asarray([odom.get('linear_speed'), odom.get('angular_speed')], dtype=float)
        if (pose.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(speeds).all()
                or np.any(speeds < 0) or speeds[0] > .005 or speeds[1] > .008):
            raise RuntimeError('empty torso overlap base moving')
        previous = self.reference['base_pose']
        yaw = math.atan2(math.sin(pose[2]-previous[2]), math.cos(pose[2]-previous[2]))
        if np.linalg.norm(pose[:2]-previous[:2]) > .002 or abs(yaw) > .005:
            raise RuntimeError('empty torso overlap registered base changed')

    def check(self):
        # This remains the original full feedback/reference validation.
        self.node._lift_first_measurements(self.reference)
        with self.node._lock:
            self._require_locked()

    def follow_started(self, client, names, positions, duration, token):
        with self.node._lock:
            if (self.follow_token is not None or token is None
                    or client is not self.node.torso_client
                    or tuple(names) != (IK_JOINTS[0],)
                    or tuple(positions) != (float(self.target[0]),) or duration != 2.5
                    or token.state is not self.torso_state or not token.torso
                    or token.generation != self.prior_generation+1
                    or token.cancel_generation != self.cancel_generation):
                raise RuntimeError('empty torso overlap original sender contract changed')
            self.follow_token = token
        self.check()

    def require_send_locked(self, client, goal, follow_token):
        n = self.node
        if self.dispatched or client is not n.torso_client or follow_token is None:
            raise RuntimeError('empty torso overlap sender/duplicate invalid')
        if (follow_token.state is not self.torso_state or not follow_token.torso
                or follow_token.generation != self.prior_generation+1
                or follow_token.cancel_generation != self.cancel_generation):
            raise RuntimeError('empty torso overlap follow token changed')
        self.follow_token = follow_token
        self._require_locked(before_send=True)
        points = goal.trajectory.points
        if (tuple(goal.trajectory.joint_names) != (IK_JOINTS[0],) or len(points) != 1
                or tuple(points[0].positions) != (float(self.target[0]),)
                or points[0].time_from_start.sec != 2 or points[0].time_from_start.nanosec != 500_000_000
                or points[0].velocities or points[0].accelerations or points[0].effort
                or n._goal_handles or n._pending_retained_acceptances):
            raise RuntimeError('empty torso overlap serialized goal/ownership changed')
        n._pending_retained_acceptances.add(self.token)
        self.dispatched = True

    def sent_locked(self, acceptance):
        self.acceptance = acceptance

    def _late(self):
        if self.acceptance is not None and not self.late_callback_registered:
            self.late_callback_registered = True
            self.acceptance.add_done_callback(
                lambda future: self.node._cancel_late_retained_goal(future, self.token))

    def wait_acceptance(self, acceptance, timeout):
        if acceptance is not self.acceptance:
            raise RuntimeError('empty torso overlap acceptance changed')
        deadline = min(self.wall_deadline, time.monotonic()+float(timeout))
        try:
            while not acceptance.done():
                self.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError('empty torso overlap acceptance timeout')
                time.sleep(.02)
            handle = acceptance.result()
            if handle is None or type(handle.accepted) is not bool:
                raise RuntimeError('empty torso overlap invalid acceptance')
            with self.node._lock:
                self.handle = handle
                if not handle.accepted:
                    self.node._pending_retained_acceptances.discard(self.token)
            self.check()
            return handle
        except BaseException:
            self._late()
            raise

    def accepted_locked(self, handle, result_future):
        if self.handle is not handle or self.result_future is not None:
            raise RuntimeError('empty torso overlap accepted handle changed')
        self.result_future = result_future
        self.node._pending_retained_acceptances.discard(self.token)

    def action_completed(self, result_future):
        self.check()
        if (result_future is not self.result_future or not result_future.done()
                or not self.node._valid_retained_terminal_result(result_future.result())):
            raise RuntimeError('empty torso overlap terminal result unresolved')

    def _stop_evidence(self):
        # No cancellation clearing, command or old-state substitution. Two fresh
        # producers after action completion must report an actually stopped axis.
        simulated = bool(getattr(self.node.get_clock(), 'ros_time_is_active', False))
        started_ns = int(self.node.get_clock().now().nanoseconds)
        progress_ns = started_ns
        progress_stamp = self.node._joint_stamps_ns[IK_JOINTS[0]]
        deadline = min(self.wall_deadline, time.monotonic()+5.)
        last = None
        while time.monotonic() < deadline:
            self.check()
            with self.node._lock:
                self._require_locked()
                now_ns = int(self.node.get_clock().now().nanoseconds)
                stamp = self.node._joint_stamps_ns[IK_JOINTS[0]]
                if simulated:
                    if now_ns-started_ns >= 5_000_000_000:
                        break
                    # Preserve five seconds of physical settling when Gazebo
                    # runs slowly; frozen clock/feedback still fails in five
                    # wall seconds, and the owner's total bound still applies.
                    if now_ns > progress_ns and stamp > progress_stamp:
                        deadline = min(self.wall_deadline, time.monotonic()+5.)
                        progress_ns, progress_stamp = now_ns, stamp
                velocity = self.node._joint_velocities.get(IK_JOINTS[0])
                position = self.node.joints[IK_JOINTS[0]]
                stopped = (velocity is not None and math.isfinite(velocity)
                    and abs(velocity) <= 1e-6 and abs(position-self.target[0]) <= .002)
                if stopped and last is not None and stamp > last:
                    self.measured_stopped = True
                    self.stopped_stamp = stamp
                    return
            last = stamp if stopped else None
            time.sleep(.02)
        raise TimeoutError('empty torso overlap measured stop unresolved')

    def _require_stopped_locked(self):
        # Revalidate the current producer in the same snapshot as all owner
        # guards; a stop recorded while planning was running is insufficient.
        self._require_locked()
        stamp = self.node._joint_stamps_ns[IK_JOINTS[0]]
        velocity = self.node._joint_velocities.get(IK_JOINTS[0])
        position = self.node.joints[IK_JOINTS[0]]
        if (not self.measured_stopped or self.stopped_stamp is None
                or stamp < self.stopped_stamp or velocity is None
                or not math.isfinite(velocity) or abs(velocity) > 1e-6
                or abs(position-self.target[0]) > .002):
            raise RuntimeError('empty torso overlap current measured stop unresolved')

    def _cancel_owned(self):
        n = self.node
        if self.handle is None:
            self._late()
            return
        if not self.handle.accepted:
            return
        terminal = self.result_future
        try:
            # A known accepted action must receive a stop request even if the
            # action client's result-future accessor itself is broken.
            try:
                self.handle.cancel_goal_async()
            except Exception:
                pass
            if terminal is None:
                terminal = self.handle.get_result_async()
                self.result_future = terminal
            if not (terminal.done() and n._valid_retained_terminal_result(terminal.result())):
                if not n._cancel_retained_goal_and_confirm(self.handle, terminal):
                    return
            if not (terminal.done() and n._valid_retained_terminal_result(terminal.result())):
                return
            # Terminal proof may remove action ownership, never the physical owner.
            with n._lock:
                if self.handle in n._goal_handles:
                    n._goal_handles.remove(self.handle)
                n._pending_retained_acceptances.discard(self.token)
        except Exception:
            return

    def _motion(self):
        try:
            if not self.node._move_torso(float(self.target[0]), 2.5, empty_torso_owner=self):
                raise RuntimeError('empty torso overlap original action failed')
            wait_for_geometry_endpoint(self.node, self.start, self.target,
                right=self.frozen_vectors['right'], head=self.frozen_vectors['head'],
                velocity_limits=self.frozen_vectors['velocity_limits'],
                aperture=self.open_aperture, phase='empty_setup_torso', command='pick',
                context_check=self.check)
            self._stop_evidence()
            self.motion_succeeded = True
        except BaseException as error:
            self._fault(error)
            self.node._cancel.set()
            self._cancel_owned()
        finally:
            self.done.set()

    def start_motion(self):
        n, g = self.node, self.checker
        n._lift_first_measurements(self.reference)
        g.require_fresh(self.start, self.initial_aperture)
        with n._adaptive_command_guard():
            with n._lock:
                self._require_plan_locked()
                if (getattr(n, '_empty_torso_planning_owner', None) is not None
                        or getattr(n, '_empty_pickup_geometry_active', False)
                        or getattr(n, '_pickup_parallel_geometry_active', False)
                        or n._goal_handles or n._pending_retained_acceptances
                        or n._held_book_corners is not None or n._cancel.is_set()
                        or getattr(n, '_active_place_scene_reference', None) is not None
                        or getattr(n, '_payload_hazard_latched', None) is not None
                        or getattr(n, '_held_grip_sensor_fault', None) is not None
                        or getattr(n, '_raw_contacts_first_failure', None) is not None
                        or getattr(n, '_target_robot_contact_latched', False)
                        or getattr(n, '_empty_arm_contact_latched', False)):
                    raise RuntimeError('empty torso overlap existing motion/pool/payload')
                n._empty_torso_planning_owner = self
                n._empty_arm_contact_guard = True
        try:
            if not n._open_gripper():
                raise RuntimeError('empty torso overlap opening failed')
            g.require_fresh(self.start, self.open_aperture)
            with n._lock:
                self.contact_epoch = int(getattr(n, '_contact_epoch', 0))
                self.target_model = getattr(n, '_target_book_model', None)
            self.check()
            self.thread = threading.Thread(target=self._motion,
                name='erc-empty-torso-planning', daemon=True)
            self.thread.start()
            n._publish_status('empty_torso_planning_overlap_started', command='pick',
                nominal_target=self.target.tolist(), nominal_duration=2.5,
                prefix='original empty pool closed; original opening/ascent samples retained')
        except BaseException as error:
            self._fault(error)
            n._cancel.set()
            self.done.set()
            raise

    def join(self):
        while not self.done.wait(.02):
            if time.monotonic() >= self.wall_deadline:
                self._fault('empty torso overlap owner deadline')
                self.node._cancel.set()
                self._cancel_owned()
                break
        if self.thread is not None:
            self.thread.join(timeout=3.)
        if (self.thread is not None and self.thread.is_alive()) or not self.done.is_set():
            raise RuntimeError('empty torso overlap worker unresolved')
        if self.fault or not self.motion_succeeded or not self.measured_stopped:
            raise RuntimeError(self.fault or 'empty torso overlap not physically joined')
        self.check()
        self.checker.require_fresh(self.target, self.open_aperture)
        with self.node._lock:
            self._require_stopped_locked()
            self.joined = True
        self.node._publish_status('empty_torso_planning_overlap_joined', command='pick',
            wall_seconds=time.monotonic()-self.started_wall,
            measured_stopped=True, original_endpoint_verified=True,
            scope='Owned torso action and fresh physical endpoint joined before arm motion')

    def close(self, planner_failed):
        try:
            self.join()
            if not planner_failed:
                with self.node._lock:
                    self._require_stopped_locked()
                    self.node._empty_torso_planning_owner = None
                    self.node._empty_arm_contact_guard = self.original_contact_guard
        except BaseException as error:
            self._fault(error)
            raise
        finally:
            if planner_failed or self.fault or not self.joined:
                self.node._cached_post_retreat_plan = None
                self._fault('empty torso overlap planner failed' if planner_failed else 'empty torso overlap join failed')
                # Keep this failed owner even if the action terminated: no proof revival.
                self.node._cancel.set()


@contextmanager
def planning_scope(node, checker, reference, *, ordinary):
    enabled = checked_enabled(getattr(node, 'empty_torso_planning_overlap_enabled', False))
    if not enabled or not ordinary or type(checker) is not EmptyPickupCollision:
        yield None
        return
    owner = EmptyTorsoPlanningOwner(node, checker, reference)
    owner.start_motion()
    try:
        yield owner
    except BaseException as planner_error:
        try:
            owner.close(planner_failed=True)
        except BaseException as closure_error:
            raise RuntimeError('empty torso overlap planning failed: '
                + str(planner_error) + '; closure: ' + str(closure_error)) from planner_error
        raise
    else:
        owner.close(planner_failed=False)
