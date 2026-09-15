"""Odom-frame holonomic waypoint controller with conservative safety limits."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .common import (
    SENSOR_QOS,
    RELIABLE_QOS,
    TRANSIENT_RELIABLE_QOS,
    clamp,
    decode_event,
    encode_event,
    normalize_angle,
    quaternion_from_yaw,
    yaw_from_quaternion,
)
from .runtime_utils import minimum_valid_range


def _checked_normal_translation_gain(value) -> float:
    if (
        type(value) not in (int, float)
        or not 0.85 <= value <= 1.8
        or not math.isfinite(value)
    ):
        raise ValueError('normal_translation_gain must be finite in [0.85, 1.8]')
    return float(value)


def _checked_normal_yaw_gain(value) -> float:
    if (
        type(value) not in (int, float)
        or not 1.35 <= value <= 2.0
        or not math.isfinite(value)
    ):
        raise ValueError('normal_yaw_gain must be finite in [1.35, 2.0]')
    return float(value)


EMPTY_ARM_PROFILES = frozenset(('empty_arm_staging', 'empty_arm_advance'))
EMPTY_ARM_STATE_MAX_AGE = 0.25
EMPTY_ARM_POSITION_TOLERANCE = 0.008
EMPTY_ARM_SETTLE_SECONDS = 0.35


class NavigationNode(Node):
    """Drive short collision-aware waypoints without assuming an unavailable map."""

    def __init__(self) -> None:
        super().__init__('erc_navigation')
        self._declare_parameters()
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.control_rate = float(self.get_parameter('control_rate_hz').value)
        self.max_vx = float(self.get_parameter('max_linear_speed').value)
        self.max_vy = float(self.get_parameter('max_lateral_speed').value)
        self.max_wz = float(self.get_parameter('max_angular_speed').value)
        self.linear_acceleration = float(
            self.get_parameter('linear_acceleration_limit').value
        )
        self.normal_linear_convergence_enabled = bool(
            self.get_parameter('normal_linear_convergence_enabled').value
        )
        self.normal_translation_gain = _checked_normal_translation_gain(
            self.get_parameter('normal_translation_gain').value
        )
        self.normal_yaw_gain = _checked_normal_yaw_gain(
            self.get_parameter('normal_yaw_gain').value
        )
        self.carried_retreat_max_speed = float(
            self.get_parameter('carried_retreat_max_speed').value
        )
        self.carried_retreat_acceleration = float(
            self.get_parameter('carried_retreat_acceleration_limit').value
        )
        self.carried_retreat_braking_acceleration = float(
            self.get_parameter('carried_retreat_braking_acceleration').value
        )
        self.carried_retreat_max_angular_speed = float(
            self.get_parameter('carried_retreat_max_angular_speed').value
        )
        self.carried_retreat_angular_acceleration = float(
            self.get_parameter(
                'carried_retreat_angular_acceleration_limit'
            ).value
        )
        self.angular_acceleration = float(
            self.get_parameter('angular_acceleration_limit').value
        )
        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.yaw_tolerance = float(self.get_parameter('yaw_tolerance').value)
        self.settle_seconds = float(self.get_parameter('settle_time_seconds').value)
        self.carried_retreat_settle_seconds = float(
            self.get_parameter('carried_retreat_settle_time_seconds').value
        )
        self.stop_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.goal_timeout = float(self.get_parameter('goal_timeout_seconds').value)
        self.sensor_stale_timeout = float(
            self.get_parameter('sensor_stale_timeout_seconds').value
        )

        self.pose: Optional[Tuple[float, float, float]] = None
        self.last_odom_time = None
        self.odom_stamp_ns = None
        self.measured_twist = None
        self.empty_arm_still_since_ns = None
        self.empty_arm_still_pose = None
        self.empty_arm_joint_states = {}
        self.prepared_joint_guard = {}
        self.empty_arm_anchor = None
        self.empty_arm_direction = None
        self.navigation_plan_id = None
        self.advance_bounds = None
        self.front_scan_message = None
        self.rear_scan_message = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_front_scan_time = None
        self.last_rear_scan_time = None
        self.goal: Optional[Tuple[float, float, float]] = None
        self.next_goal_profile = 'normal'
        self.goal_profile = 'normal'
        self.goal_started = None
        self.settle_started = None
        self.goal_id = 0
        self.blocked = False
        self.front_clearance = float('inf')
        self.rear_clearance = float('inf')
        self.last_command = Twist()
        self.executed_path = Path()
        self.executed_path.header.frame_id = 'odom'
        self._last_path_sample = None
        self._ready_announced = False
        if not (
            math.isfinite(self.carried_retreat_max_speed)
            and math.isfinite(self.carried_retreat_acceleration)
            and math.isfinite(self.carried_retreat_braking_acceleration)
            and 0.0 < self.carried_retreat_max_speed <= min(self.max_vx, self.max_vy)
            and 0.0 < self.carried_retreat_braking_acceleration
            <= self.carried_retreat_acceleration
            <= self.linear_acceleration
            and math.isfinite(self.carried_retreat_max_angular_speed)
            and 0.0 < self.carried_retreat_max_angular_speed <= self.max_wz
            and math.isfinite(self.carried_retreat_angular_acceleration)
            and 0.0 < self.carried_retreat_angular_acceleration
            <= self.angular_acceleration
            and math.isfinite(self.carried_retreat_settle_seconds)
            and 0.0 <= self.carried_retreat_settle_seconds
            <= self.settle_seconds
        ):
            raise ValueError('carried-retreat motion limits are invalid')

        odom_topic = str(self.get_parameter('odom_topic').value)
        front_topic = str(self.get_parameter('front_scan_topic').value)
        rear_topic = str(self.get_parameter('rear_scan_topic').value)
        cmd_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.create_subscription(Odometry, odom_topic, self._on_odom, SENSOR_QOS)
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, SENSOR_QOS)
        self.create_subscription(LaserScan, front_topic, self._on_front_scan, SENSOR_QOS)
        self.create_subscription(LaserScan, rear_topic, self._on_rear_scan, SENSOR_QOS)
        self.create_subscription(
            PoseStamped,
            '/erc/navigation/goal',
            self._on_goal,
            RELIABLE_QOS,
        )
        self.create_subscription(
            String,
            '/erc/navigation/command',
            self._on_command,
            RELIABLE_QOS,
        )
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.status_pub = self.create_publisher(
            String, '/erc/navigation/status', TRANSIENT_RELIABLE_QOS
        )
        self.planned_path_pub = self.create_publisher(
            Path, '/erc/navigation/planned_path', TRANSIENT_RELIABLE_QOS
        )
        self.executed_path_pub = self.create_publisher(
            Path, '/erc/navigation/executed_path', TRANSIENT_RELIABLE_QOS
        )
        self.timer = self.create_timer(1.0 / self.control_rate, self._control)

    def _declare_parameters(self) -> None:
        values = {
            'dry_run': False,
            'odom_topic': '/odom',
            'cmd_vel_topic': '/cmd_vel',
            'front_scan_topic': '/scan_front_raw',
            'rear_scan_topic': '/scan_rear_raw',
            'control_rate_hz': 20.0,
            'max_linear_speed': 0.42,
            'max_lateral_speed': 0.32,
            'max_angular_speed': 0.55,
            'linear_acceleration_limit': 0.55,
            'normal_linear_convergence_enabled': False,
            'normal_translation_gain': 0.85,
            'normal_yaw_gain': 1.35,
            'carried_retreat_max_speed': 0.10,
            'carried_retreat_acceleration_limit': 0.06,
            'carried_retreat_braking_acceleration': 0.06,
            'carried_retreat_max_angular_speed': 0.08,
            'carried_retreat_angular_acceleration_limit': 0.10,
            'angular_acceleration_limit': 0.8,
            'position_tolerance': 0.045,
            'yaw_tolerance': 0.045,
            'settle_time_seconds': 0.45,
            'carried_retreat_settle_time_seconds': 0.10,
            'emergency_stop_distance': 0.28,
            'goal_timeout_seconds': 45.0,
            'sensor_stale_timeout_seconds': 1.0,
        }
        for name, default in values.items():
            self.declare_parameter(name, default)

    @staticmethod
    def _minimum_valid_range(message: LaserScan) -> float:
        return minimum_valid_range(
            message.ranges, message.range_min, message.range_max
        )

    def _on_front_scan(self, message: LaserScan) -> None:
        self.front_clearance = self._minimum_valid_range(message)
        self.last_front_scan_time = self.get_clock().now()
        self.front_scan_message = message

    def _on_rear_scan(self, message: LaserScan) -> None:
        self.rear_clearance = self._minimum_valid_range(message)
        self.last_rear_scan_time = self.get_clock().now()
        self.rear_scan_message = message

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        self.pose = (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )
        self.last_odom_time = self.get_clock().now()
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        previous_stamp = getattr(self, 'odom_stamp_ns', None)
        self.odom_stamp_ns = stamp_ns
        velocity = message.twist.twist
        self.measured_twist = (
            float(velocity.linear.x), float(velocity.linear.y), float(velocity.angular.z)
        )
        reference = getattr(self, 'empty_arm_still_pose', None)
        stationary = (
            all(math.isfinite(value) for value in (*self.pose, *self.measured_twist))
            and math.hypot(*self.measured_twist[:2]) <= 0.003
            and abs(self.measured_twist[2]) <= 0.005
        )
        if not stationary or (previous_stamp is not None and stamp_ns < previous_stamp):
            self.empty_arm_still_since_ns = None
            self.empty_arm_still_pose = None
        elif (
            reference is None
            or math.hypot(self.pose[0] - reference[0], self.pose[1] - reference[1]) > 0.002
            or abs(normalize_angle(self.pose[2] - reference[2])) > 0.003
        ):
            self.empty_arm_still_since_ns = stamp_ns
            self.empty_arm_still_pose = self.pose

    def _on_joint_state(self, message: JointState) -> None:
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if not hasattr(self, 'empty_arm_joint_states'):
            self.empty_arm_joint_states = {}
        for name, position in zip(message.name, message.position):
            previous = self.empty_arm_joint_states.get(name)
            if previous is None or stamp_ns >= previous[1]:
                self.empty_arm_joint_states[name] = (float(position), stamp_ns)

    def _empty_arm_state_fault(self, now_ns: int) -> Optional[str]:
        stamp_ns = getattr(self, 'odom_stamp_ns', None)
        velocity = getattr(self, 'measured_twist', None)
        if (
            stamp_ns is None
            or not 0 <= now_ns - stamp_ns <= EMPTY_ARM_STATE_MAX_AGE * 1e9
            or self.pose is None or velocity is None
            or not all(math.isfinite(value) for value in (*self.pose, *velocity))
        ):
            return 'empty_arm_odometry_stale_or_invalid'
        return None

    def _empty_arm_base_still(self, seconds: float) -> bool:
        start_ns = getattr(self, 'empty_arm_still_since_ns', None)
        stamp_ns = getattr(self, 'odom_stamp_ns', None)
        return bool(
            start_ns is not None and stamp_ns is not None
            and stamp_ns - start_ns >= seconds * 1e9
        )

    def _prepared_joint_fault(self, expected, now_ns: int) -> Optional[str]:
        measured = getattr(self, 'empty_arm_joint_states', {})
        for name, target in expected.items():
            sample = measured.get(name)
            if (
                sample is None or not math.isfinite(sample[0])
                or not 0 <= now_ns - sample[1] <= EMPTY_ARM_STATE_MAX_AGE * 1e9
            ):
                return f'empty_arm_joint_state_stale_or_invalid:{name}'
            tolerance = (
                0.001 if name == 'torso_lift_joint'
                else 0.003 if name == 'gripper_left_finger_joint' else 0.005
            )
            if abs(sample[0] - target) > tolerance:
                return f'empty_arm_prepared_joint_drift:{name}'
        return None

    def _sensors_fresh(self, now) -> bool:
        stamps = (
            self.last_odom_time,
            self.last_front_scan_time,
            self.last_rear_scan_time,
        )
        return all(
            stamp is not None
            and (now - stamp).nanoseconds <= self.sensor_stale_timeout * 1e9
            for stamp in stamps
        )

    def _on_goal(self, message: PoseStamped) -> None:
        # The legacy two-message interface arms a profile before sending its
        # PoseStamped.  Consume that one-shot choice on every goal attempt so a
        # rejected or stale message cannot leak a payload profile into a later
        # unrelated goal.
        profile = getattr(self, 'next_goal_profile', 'normal')
        self.next_goal_profile = 'normal'
        if profile in EMPTY_ARM_PROFILES:
            self._publish_status('rejected', reason='empty_arm_profile_requires_atomic_command')
            return
        if message.header.frame_id not in ('', 'odom'):
            self._publish_status('rejected', reason='goal_frame_must_be_odom')
            return
        if self.pose is None:
            self._publish_status('rejected', reason='odometry_not_ready')
            return
        if self.goal is not None:
            self._publish_status(
                'rejected', goal_id=self.goal_id, reason='goal_already_active'
            )
            return
        goal = (
            float(message.pose.position.x),
            float(message.pose.position.y),
            yaw_from_quaternion(message.pose.orientation),
        )
        if not all(math.isfinite(value) for value in goal):
            self._publish_status('rejected', reason='goal_must_be_finite')
            return

        self._accept_goal(goal, profile=profile)

    def _accept_goal(
        self,
        goal: Tuple[float, float, float],
        *,
        profile: Optional[str] = None,
        prepared_joints=None,
        plan_id: Optional[str] = None,
        advance_bounds=None,
    ) -> None:
        """Accept one already-validated goal and atomically bind its profile."""
        self.goal_id += 1
        self.goal_profile = (
            getattr(self, 'next_goal_profile', 'normal')
            if profile is None
            else profile
        )
        self.next_goal_profile = 'normal'
        self.prepared_joint_guard = dict(prepared_joints or {})
        self.navigation_plan_id = plan_id
        self.advance_bounds = advance_bounds
        self.empty_arm_anchor = None
        self.empty_arm_direction = None
        if self.goal_profile == 'empty_arm_advance':
            self.empty_arm_anchor = tuple(self.pose)
            dx, dy = goal[0] - self.pose[0], goal[1] - self.pose[1]
            length = math.hypot(dx, dy)
            self.empty_arm_direction = (dx / length, dy / length, length)
        self.goal = goal
        self.goal_started = self.get_clock().now()
        self.settle_started = None
        self.blocked = False
        self.executed_path = Path()
        self.executed_path.header.frame_id = 'odom'
        self._last_path_sample = None
        self._publish_planned_path()
        extra = {'plan_id': plan_id} if plan_id is not None else {}
        self._publish_status(
            'accepted',
            goal_id=self.goal_id,
            goal=list(self.goal),
            profile=self.goal_profile,
            **extra,
        )
        if self.dry_run:
            self._finish_goal('reached', dry_run=True)

    def _on_command(self, message: String) -> None:
        payload = decode_event(message.data)
        event = payload.get('event', message.data.strip().lower())
        if event == 'navigate':
            profile = str(payload.get('profile', 'normal')).strip().lower()
            if profile not in ('normal', 'carried_retreat', *EMPTY_ARM_PROFILES):
                self._publish_status(
                    'rejected',
                    reason='invalid_navigation_profile',
                    profile=profile,
                )
                return
            if self.pose is None:
                self._publish_status('rejected', reason='odometry_not_ready')
                return
            if self.goal is not None:
                self._publish_status(
                    'rejected',
                    reason='goal_already_active',
                    goal_id=self.goal_id,
                )
                return
            try:
                goal = (
                    float(payload['x']),
                    float(payload['y']),
                    float(payload['yaw']),
                )
            except (KeyError, TypeError, ValueError):
                self._publish_status('rejected', reason='goal_must_be_finite')
                return
            if not all(math.isfinite(value) for value in goal):
                self._publish_status('rejected', reason='goal_must_be_finite')
                return
            if profile in EMPTY_ARM_PROFILES:
                plan_id = payload.get('plan_id')
                extra = {'plan_id': plan_id} if isinstance(plan_id, str) else {}
                if plan_id is not None and (not isinstance(plan_id, str) or not plan_id):
                    self._publish_status('rejected', reason='empty_arm_plan_id_invalid')
                    return
                expected = payload.get('prepared_joints', {})
                if not isinstance(expected, dict) or (
                    'prepared_joints' in payload and not expected
                ) or any(
                    not isinstance(name, str) or not name
                    or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for name, value in expected.items()
                ):
                    self._publish_status('rejected', reason='empty_arm_prepared_joints_invalid', **extra)
                    return
                now_ns = self.get_clock().now().nanoseconds
                fault = self._empty_arm_state_fault(now_ns) or self._prepared_joint_fault(expected, now_ns)
                if fault:
                    self._publish_status('rejected', reason=fault, **extra)
                    return
                if profile == 'empty_arm_advance':
                    stamp_ns = payload.get('prepared_stamp_ns')
                    bounds = payload.get('advance_bounds')
                    if not plan_id or not expected or (
                        isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int)
                        or not 0 <= now_ns - stamp_ns <= 1_000_000_000
                    ):
                        self._publish_status('rejected', reason='empty_arm_advance_certificate_missing_or_stale', **extra)
                        return
                    if (
                        not isinstance(bounds, list) or len(bounds) != 2
                        or any(not isinstance(row, list) or len(row) != 3 for row in bounds)
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for row in bounds for v in row)
                        or any(bounds[0][axis] >= bounds[1][axis] for axis in range(3))
                        or not all(bounds[0][axis] <= 0 <= bounds[1][axis] for axis in (0, 1))
                    ):
                        self._publish_status('rejected', reason='empty_arm_advance_bounds_invalid', **extra)
                        return
                    x, y, yaw = self.pose
                    dx, dy = goal[0] - x, goal[1] - y
                    along = math.cos(yaw) * dx + math.sin(yaw) * dy
                    lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
                    if (
                        along <= 0.0 or abs(lateral) > 0.01
                        or abs(normalize_angle(goal[2] - yaw)) > 0.01
                    ):
                        self._publish_status('rejected', reason='empty_arm_advance_goal_not_straight_forward', **extra)
                        return
                    if not self._empty_arm_base_still(0.20):
                        self._publish_status('rejected', reason='empty_arm_advance_base_not_settled', **extra)
                        return
                self._accept_goal(
                    goal, profile=profile, prepared_joints=expected, plan_id=plan_id,
                    advance_bounds=tuple(tuple(row) for row in bounds)
                    if profile == 'empty_arm_advance' else None,
                )
            else:
                self._accept_goal(goal, profile=profile)
            return
        if event == 'set_next_profile':
            profile = str(payload.get('profile', '')).strip().lower()
            if profile not in ('normal', 'carried_retreat'):
                self._publish_status(
                    'rejected',
                    reason='invalid_navigation_profile',
                    profile=profile,
                )
                return
            if self.goal is not None:
                self._publish_status(
                    'rejected',
                    reason='goal_already_active',
                    goal_id=self.goal_id,
                )
                return
            self.next_goal_profile = profile
            self._publish_status('profile_armed', profile=profile)
            return
        if event in ('cancel', 'stop', 'abort'):
            self.next_goal_profile = 'normal'
            if self.goal is not None:
                self._finish_goal('cancelled')
            else:
                self._publish_zero()
                self._publish_status(
                    'cancelled', goal_id=self.goal_id, active_goal=False
                )

    def _publish_status(self, event: str, **fields) -> None:
        message = String()
        message.data = encode_event(event, **fields)
        self.status_pub.publish(message)

    def _publish_planned_path(self) -> None:
        if self.pose is None or self.goal is None:
            return
        start_x, start_y, start_yaw = self.pose
        goal_x, goal_y, goal_yaw = self.goal
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        for index in range(41):
            fraction = index / 40.0
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = start_x + fraction * (goal_x - start_x)
            pose.pose.position.y = start_y + fraction * (goal_y - start_y)
            yaw = start_yaw + fraction * normalize_angle(goal_yaw - start_yaw)
            pose.pose.orientation = quaternion_from_yaw(yaw)
            path.poses.append(pose)
        self.planned_path_pub.publish(path)

    def _sample_executed_path(self, force: bool = False) -> None:
        if self.pose is None:
            return
        now = self.get_clock().now()
        if not force and self._last_path_sample is not None:
            if (now - self._last_path_sample).nanoseconds < 150_000_000:
                return
        self._last_path_sample = now
        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = now.to_msg()
        pose.pose.position.x, pose.pose.position.y, yaw = self.pose
        pose.pose.orientation = quaternion_from_yaw(yaw)
        self.executed_path.header.stamp = pose.header.stamp
        self.executed_path.poses.append(pose)
        if len(self.executed_path.poses) > 2500:
            self.executed_path.poses = self.executed_path.poses[-2500:]
        self.executed_path_pub.publish(self.executed_path)

    def _finish_goal(self, event: str, **fields) -> None:
        elapsed = 0.0
        if self.goal_started is not None:
            elapsed = (self.get_clock().now() - self.goal_started).nanoseconds / 1e9
        self._sample_executed_path(force=True)
        self._publish_zero()
        if getattr(self, 'navigation_plan_id', None) is not None:
            fields['plan_id'] = self.navigation_plan_id
        self._publish_status(event, goal_id=self.goal_id, elapsed_seconds=elapsed, **fields)
        self.goal = None
        self.goal_started = None
        self.settle_started = None
        self.goal_profile = 'normal'
        self.prepared_joint_guard = {}
        self.empty_arm_anchor = None
        self.empty_arm_direction = None
        self.navigation_plan_id = None
        self.advance_bounds = None

    def _advance_scan_fault(self, now_ns: int) -> Optional[str]:
        """Check a stopping sweep against both timestamped planar laser scans.

        This projects the whole certified body/tool bound to XY. It cannot
        establish clearance for obstacles outside the sensors' scan heights.
        """
        low, high = self.advance_bounds
        ux, uy, _ = self.empty_arm_direction
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        speed = math.hypot(self.last_command.linear.x, self.last_command.linear.y)
        travel = speed * speed / (2.0 * 0.06) + 0.06 * (
            EMPTY_ARM_STATE_MAX_AGE + 1.0 / self.control_rate
        )
        sweep_x, sweep_y = (c * ux + s * uy) * travel, (-s * ux + c * uy) * travel
        xmin, xmax = low[0] - 0.04 + min(0.0, sweep_x), high[0] + 0.04 + max(0.0, sweep_x)
        ymin, ymax = low[1] - 0.04 + min(0.0, sweep_y), high[1] + 0.04 + max(0.0, sweep_y)
        for label in ('front', 'rear'):
            scan = getattr(self, f'{label}_scan_message', None)
            if scan is None:
                return f'empty_arm_advance_scan_missing:{label}'
            stamp_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
            if not 0 <= now_ns - stamp_ns <= EMPTY_ARM_STATE_MAX_AGE * 1e9:
                return f'empty_arm_advance_scan_stale:{label}'
            if (
                not scan.header.frame_id or not scan.ranges
                or not all(math.isfinite(v) for v in (
                    scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max,
                ))
                or scan.range_max <= scan.range_min
            ):
                return f'empty_arm_advance_scan_invalid:{label}'
            try:
                tf = self.tf_buffer.lookup_transform(
                    'odom', scan.header.frame_id, Time.from_msg(scan.header.stamp),
                    timeout=Duration(seconds=0.0),
                ).transform
                q = tf.rotation
                norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
                if not math.isfinite(norm) or norm < 1e-9:
                    raise ValueError('invalid laser transform rotation')
                qx, qy, qz, qw = q.x/norm, q.y/norm, q.z/norm, q.w/norm
                tx, ty = float(tf.translation.x), float(tf.translation.y)
                if not math.isfinite(tx) or not math.isfinite(ty):
                    raise ValueError('invalid laser transform translation')
            except Exception:
                return f'empty_arm_advance_scan_transform_unavailable:{label}'
            observed = False
            for index, distance in enumerate(scan.ranges):
                if math.isinf(distance) and distance > 0:
                    observed = True  # A valid ray reporting no obstacle in range.
                    continue
                if not math.isfinite(distance) or not scan.range_min <= distance <= scan.range_max:
                    continue
                observed = True
                angle = scan.angle_min + index * scan.angle_increment
                lx, ly = distance * math.cos(angle), distance * math.sin(angle)
                ox = tx + (1 - 2*(qy*qy + qz*qz))*lx + 2*(qx*qy - qz*qw)*ly
                oy = ty + 2*(qx*qy + qz*qw)*lx + (1 - 2*(qx*qx + qz*qz))*ly
                bx, by = c*(ox-x) + s*(oy-y), -s*(ox-x) + c*(oy-y)
                if xmin <= bx <= xmax and ymin <= by <= ymax:
                    return f'empty_arm_advance_swept_footprint_obstacle:{label}'
            if not observed:
                return f'empty_arm_advance_scan_invalid:{label}'
        return None

    def _limit_empty_arm_command(self, desired: Twist) -> Twist:
        """Bound the complete translation vector, keeping advance yaw fixed."""
        result = Twist()
        dx = desired.linear.x - self.last_command.linear.x
        dy = desired.linear.y - self.last_command.linear.y
        delta = math.hypot(dx, dy)
        step = 0.06 / self.control_rate
        scale = min(1.0, step / delta) if delta else 1.0
        result.linear.x = self.last_command.linear.x + dx * scale
        result.linear.y = self.last_command.linear.y + dy * scale
        speed = math.hypot(result.linear.x, result.linear.y)
        if speed > 0.06:
            result.linear.x *= 0.06 / speed
            result.linear.y *= 0.06 / speed
        if self.goal_profile == 'empty_arm_staging':
            result.angular.z = clamp(
                self.last_command.angular.z + clamp(
                    desired.angular.z - self.last_command.angular.z,
                    -0.12 / self.control_rate, 0.12 / self.control_rate,
                ), -0.16, 0.16,
            )
        # Advance never inherits or generates an angular velocity command.
        return result

    def _control_empty_arm(self, now) -> None:
        """Control a certified empty-arm corridor; geometry is supplied upstream.

        The advance line joins the measured start to the accepted endpoint.
        Its endpoint may differ from the start-heading axis by at most 10 mm.
        No cross-track or heading recovery is attempted with the arm extended.
        """
        fault = self._empty_arm_state_fault(now.nanoseconds) or self._prepared_joint_fault(
            self.prepared_joint_guard, now.nanoseconds,
        )
        if fault:
            self._finish_goal('failed', reason=fault)
            return
        self._sample_executed_path()
        x, y, yaw = self.pose
        gx, gy, gyaw = self.goal
        dx, dy = gx - x, gy - y
        distance = math.hypot(dx, dy)
        advance = self.goal_profile == 'empty_arm_advance'
        remaining = distance
        if advance:
            anchor = self.empty_arm_anchor
            ux, uy, length = self.empty_arm_direction
            moved_x, moved_y = x - anchor[0], y - anchor[1]
            cross_track = -uy * moved_x + ux * moved_y
            remaining = length - (ux * moved_x + uy * moved_y)
            yaw_error = normalize_angle(anchor[2] - yaw)
            if abs(cross_track) > 0.015 or abs(yaw_error) > 0.02:
                self._finish_goal(
                    'failed', reason='empty_arm_advance_corridor_drift',
                    cross_track_m=cross_track, heading_error_rad=yaw_error,
                )
                return
            if remaining < -EMPTY_ARM_POSITION_TOLERANCE:
                self._finish_goal('failed', reason='empty_arm_advance_overshoot')
                return
            if remaining <= 0.003 and distance > EMPTY_ARM_POSITION_TOLERANCE:
                self._finish_goal('failed', reason='empty_arm_advance_endpoint_off_line')
                return
            fault = self._advance_scan_fault(now.nanoseconds)
            if fault:
                self._finish_goal('failed', reason=fault)
                return
        else:
            yaw_error = normalize_angle(gyaw - yaw)

        if min(self.front_clearance, self.rear_clearance) < self.stop_distance:
            self._finish_goal('failed', reason='empty_arm_navigation_obstacle')
            return
        in_tolerance = distance <= EMPTY_ARM_POSITION_TOLERANCE and (
            advance or abs(yaw_error) <= 0.008
        )
        if in_tolerance:
            self._publish_zero()
            if not self._empty_arm_base_still(0.0):
                self.settle_started = None
            elif self.settle_started is None:
                self.settle_started = now
            elif (
                self._empty_arm_base_still(EMPTY_ARM_SETTLE_SECONDS)
                and self.odom_stamp_ns - self.settle_started.nanoseconds
                >= EMPTY_ARM_SETTLE_SECONDS * 1e9
            ):
                self._finish_goal('reached', final_pose=list(self.pose), measured_settled=True)
            return
        self.settle_started = None
        desired = Twist()
        if distance > EMPTY_ARM_POSITION_TOLERANCE:
            speed = min(0.06, math.sqrt(2.0 * 0.06 * max(0.0, remaining - 0.004)))
            if advance:
                world_x, world_y = ux * speed, uy * speed
            else:
                world_x, world_y = dx * speed / distance, dy * speed / distance
            desired.linear.x = math.cos(yaw) * world_x + math.sin(yaw) * world_y
            desired.linear.y = -math.sin(yaw) * world_x + math.cos(yaw) * world_y
        if not advance:
            desired.angular.z = clamp(1.35 * yaw_error, -0.16, 0.16)
        self.last_command = self._limit_empty_arm_command(desired)
        self.cmd_pub.publish(self.last_command)

    def _limited_command(self, desired: Twist) -> Twist:
        dt = 1.0 / self.control_rate
        linear_acceleration = (
            self.carried_retreat_acceleration
            if getattr(self, 'goal_profile', 'normal') == 'carried_retreat'
            else self.linear_acceleration
        )
        linear_step = linear_acceleration * dt
        angular_acceleration = (
            self.carried_retreat_angular_acceleration
            if getattr(self, 'goal_profile', 'normal') == 'carried_retreat'
            else self.angular_acceleration
        )
        angular_step = angular_acceleration * dt
        result = Twist()
        delta_x = desired.linear.x - self.last_command.linear.x
        delta_y = desired.linear.y - self.last_command.linear.y
        if getattr(self, 'goal_profile', 'normal') == 'carried_retreat':
            # The payload limit is translational, not per axis.  Independent
            # x/y clamps can exceed it by sqrt(2) on a diagonal retreat and
            # briefly rotate the commanded direction during ramp-up.
            delta_norm = math.hypot(delta_x, delta_y)
            if delta_norm > linear_step:
                scale = linear_step / delta_norm
                delta_x *= scale
                delta_y *= scale
        else:
            delta_x = clamp(delta_x, -linear_step, linear_step)
            delta_y = clamp(delta_y, -linear_step, linear_step)
        result.linear.x = self.last_command.linear.x + delta_x
        result.linear.y = self.last_command.linear.y + delta_y
        result.angular.z = self.last_command.angular.z + clamp(
            desired.angular.z - self.last_command.angular.z,
            -angular_step,
            angular_step,
        )
        return result

    def _desired_translation(
        self,
        body_x: float,
        body_y: float,
        distance: float,
    ) -> Tuple[float, float]:
        """Return a normal goal command or a time-bounded carried retreat."""
        if distance <= 0.0:
            return 0.0, 0.0
        if getattr(self, 'goal_profile', 'normal') == 'carried_retreat':
            remaining = max(0.0, distance - self.position_tolerance)
            speed = min(
                self.carried_retreat_max_speed,
                math.sqrt(
                    2.0
                    * self.carried_retreat_braking_acceleration
                    * remaining
                ),
            )
            return speed * body_x / distance, speed * body_y / distance

        gain = (
            getattr(self, 'normal_translation_gain', 0.85)
            if getattr(self, 'goal_profile', 'normal') == 'normal'
            and getattr(self, 'normal_linear_convergence_enabled', False)
            else 0.85
        )
        vx = clamp(gain * body_x, -self.max_vx, self.max_vx)
        vy = clamp(gain * body_y, -self.max_vy, self.max_vy)
        if distance < 0.35 and not (
            getattr(self, 'goal_profile', 'normal') == 'normal'
            and getattr(self, 'normal_linear_convergence_enabled', False)
        ):
            scale = max(0.20, distance / 0.35)
            vx *= scale
            vy *= scale
        return vx, vy

    def _publish_zero(self) -> None:
        self.last_command = Twist()
        self.cmd_pub.publish(self.last_command)

    def _control(self) -> None:
        now = self.get_clock().now()
        if self.pose is None or self.last_odom_time is None:
            self._publish_zero()
            return
        if not self._sensors_fresh(now):
            self._publish_zero()
            if self.goal is not None:
                self._finish_goal('failed', reason='stale_navigation_sensor')
            return
        if (
            not self._ready_announced
            and self.cmd_pub.get_subscription_count() > 0
        ):
            self._publish_status('ready', pose=list(self.pose))
            self._ready_announced = True
        if self.goal is None:
            self._publish_zero()
            return
        if self.goal_started is not None:
            elapsed = (now - self.goal_started).nanoseconds / 1e9
            if elapsed > self.goal_timeout:
                self._finish_goal('failed', reason='timeout')
                return

        if getattr(self, 'goal_profile', 'normal') in EMPTY_ARM_PROFILES:
            self._control_empty_arm(now)
            return

        self._sample_executed_path()
        x, y, yaw = self.pose
        goal_x, goal_y, goal_yaw = self.goal
        dx, dy = goal_x - x, goal_y - y
        distance = math.hypot(dx, dy)
        yaw_error = normalize_angle(goal_yaw - yaw)
        minimum_clearance = min(self.front_clearance, self.rear_clearance)
        if distance <= self.position_tolerance and abs(yaw_error) <= self.yaw_tolerance:
            if (
                (
                    getattr(self, 'goal_profile', 'normal') == 'carried_retreat'
                    or (
                        getattr(self, 'goal_profile', 'normal') == 'normal'
                        and getattr(self, 'normal_linear_convergence_enabled', False)
                    )
                )
                and max(
                    abs(self.last_command.linear.x),
                    abs(self.last_command.linear.y),
                    abs(self.last_command.angular.z),
                ) > 0.005
            ):
                if minimum_clearance < self.stop_distance:
                    self._publish_zero()
                    if not self.blocked:
                        self.blocked = True
                        self._publish_status(
                            'blocked',
                            goal_id=self.goal_id,
                            clearance=minimum_clearance,
                        )
                    return
                self.settle_started = None
                self.last_command = self._limited_command(Twist())
                self.cmd_pub.publish(self.last_command)
                return
            self._publish_zero()
            if self.settle_started is None:
                self.settle_started = now
            else:
                settle_seconds = (
                    self.carried_retreat_settle_seconds
                    if getattr(self, 'goal_profile', 'normal')
                    == 'carried_retreat'
                    else self.settle_seconds
                )
                if (
                    (now - self.settle_started).nanoseconds / 1e9
                    >= settle_seconds
                ):
                    self._finish_goal('reached', final_pose=list(self.pose))
            return
        self.settle_started = None

        body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        desired = Twist()
        desired.linear.x, desired.linear.y = self._desired_translation(
            body_x,
            body_y,
            distance,
        )
        maximum_angular_speed = (
            self.carried_retreat_max_angular_speed
            if getattr(self, 'goal_profile', 'normal') == 'carried_retreat'
            else self.max_wz
        )
        yaw_gain = (
            getattr(self, 'normal_yaw_gain', 1.35)
            if getattr(self, 'goal_profile', 'normal') == 'normal'
            and getattr(self, 'normal_linear_convergence_enabled', False)
            else 1.35
        )
        desired.angular.z = clamp(
            yaw_gain * yaw_error,
            -maximum_angular_speed,
            maximum_angular_speed,
        )

        if minimum_clearance < self.stop_distance:
            self._publish_zero()
            if not self.blocked:
                self.blocked = True
                self._publish_status(
                    'blocked', goal_id=self.goal_id, clearance=minimum_clearance
                )
            return
        self.blocked = False
        self.last_command = self._limited_command(desired)
        self.cmd_pub.publish(self.last_command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        if rclpy.ok():
            node._publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
