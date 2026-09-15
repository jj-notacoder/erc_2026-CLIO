"""Callback-level checks for prepared empty-arm approach navigation."""

import math
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import String

from erc_phase1_solution import navigation_node as navigation


PREPARED = {'torso_lift_joint': 0.30, 'arm_left_1_joint': -0.4,
            'gripper_left_finger_joint': 0.0}
BOUNDS = [[-0.3968, -0.30777, 0.0], [0.52265, 0.28137, 1.7]]


def _node(pose=(0.0, 0.0, 0.0)):
    n = object.__new__(navigation.NavigationNode)
    n.now_ns = 1_000_000_000
    n.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=n.now_ns))
    n.pose = None
    n.goal = None
    n.goal_id = 0
    n.goal_started = None
    n.settle_started = None
    n.next_goal_profile = 'normal'
    n.goal_profile = 'normal'
    n.last_command = Twist()
    n.dry_run = False
    n.blocked = False
    n.control_rate = 20.0
    n.goal_timeout = 45.0
    n.sensor_stale_timeout = 1.0
    n.stop_distance = 0.28
    n._ready_announced = True
    n.statuses = []
    n.commands = []
    n.cmd_pub = SimpleNamespace(publish=n.commands.append)
    n._publish_status = lambda event, **kw: n.statuses.append((event, kw))
    n._publish_planned_path = lambda: None
    n._sample_executed_path = lambda force=False: None
    n.transforms = {}
    n.transform_requests = []

    def transform(target, source, stamp, **kwargs):
        n.transform_requests.append((target, source, stamp.nanoseconds))
        tf = TransformStamped()
        tf.transform.rotation.w = 1.0
        tf.transform.translation.x, tf.transform.translation.y = n.transforms.get(source, (0., 0.))
        return tf

    n.tf_buffer = SimpleNamespace(lookup_transform=transform)
    for stamp in (0.70, 0.80, 0.90, 1.00):
        _sensors(n, stamp, pose)
    return n


def _scan(n, frame, ranges=(float('inf'),), stamp_ns=None):
    msg = LaserScan()
    msg.header.stamp = Time(nanoseconds=n.now_ns if stamp_ns is None else stamp_ns).to_msg()
    msg.header.frame_id = frame
    msg.angle_min = 0.0
    msg.angle_increment = 0.01
    msg.range_min = 0.05
    msg.range_max = 10.0
    msg.ranges = list(ranges)
    return msg


def _sensors(n, seconds, pose, velocity=(0., 0., 0.), joints=None):
    n.now_ns = round(seconds * 1e9)
    msg = Odometry()
    msg.header.stamp = Time(nanoseconds=n.now_ns).to_msg()
    msg.pose.pose.position.x, msg.pose.pose.position.y = map(float, pose[:2])
    msg.pose.pose.orientation = navigation.quaternion_from_yaw(pose[2])
    msg.twist.twist.linear.x, msg.twist.twist.linear.y = map(float, velocity[:2])
    msg.twist.twist.angular.z = float(velocity[2])
    n._on_odom(msg)
    joint_msg = JointState()
    joint_msg.header.stamp = msg.header.stamp
    values = PREPARED if joints is None else joints
    joint_msg.name = list(values)
    joint_msg.position = list(values.values())
    n._on_joint_state(joint_msg)
    n._on_front_scan(_scan(n, 'front_laser'))
    n._on_rear_scan(_scan(n, 'rear_laser'))


def _request(n, profile='empty_arm_advance', goal=None, **changes):
    x, y, yaw = n.pose
    if goal is None:
        goal = (x + .08 * math.cos(yaw), y + .08 * math.sin(yaw), yaw)
    fields = dict(event='navigate', x=goal[0], y=goal[1], yaw=goal[2],
                  profile=profile, plan_id='prepared-7')
    if profile == 'empty_arm_advance':
        fields.update(prepared_joints=dict(PREPARED),
                      prepared_stamp_ns=n.now_ns, advance_bounds=BOUNDS)
    fields.update(changes)
    n._on_command(String(data=navigation.encode_event(**fields)))


@pytest.mark.parametrize('yaw', (0.0, 1.2, math.pi - 0.002, -math.pi + 0.002))
def test_advance_uses_measured_heading_and_one_atomic_certificate(yaw):
    n = _node((2., -1., yaw))
    _request(n)
    assert n.goal_profile == 'empty_arm_advance'
    assert n.empty_arm_anchor == pytest.approx((2., -1., yaw))
    assert n.statuses[-1][1]['plan_id'] == 'prepared-7'
    n._control()
    assert n.commands[-1].linear.x > 0
    assert abs(n.commands[-1].linear.y) < 1e-10
    assert n.commands[-1].angular.z == 0.0


@pytest.mark.parametrize(('changes', 'reason'), (
    ({'goal': (-.08, 0., 0.)}, 'not_straight_forward'),
    ({'goal': (.08, .011, 0.)}, 'not_straight_forward'),
    ({'goal': (.08, 0., .011)}, 'not_straight_forward'),
    ({'prepared_stamp_ns': -1}, 'certificate_missing_or_stale'),
    ({'prepared_stamp_ns': 1_000_000_001}, 'certificate_missing_or_stale'),
    ({'prepared_stamp_ns': True}, 'certificate_missing_or_stale'),
    ({'prepared_joints': {}}, 'prepared_joints_invalid'),
    ({'prepared_joints': {'torso_lift_joint': float('nan')}}, 'prepared_joints_invalid'),
    ({'plan_id': None}, 'certificate_missing_or_stale'),
    ({'advance_bounds': None}, 'bounds_invalid'),
    ({'advance_bounds': [[.1, -.3, 0.], [.5, .3, 1.7]]}, 'bounds_invalid'),
))
def test_invalid_advance_requests_never_move(changes, reason):
    n = _node()
    _request(n, **changes)
    assert n.goal is None
    assert n.statuses[-1][0] == 'rejected'
    assert reason in n.statuses[-1][1]['reason']
    assert n.commands == []


def test_advance_requires_measured_stationarity_not_only_zero_command():
    n = _node()
    _sensors(n, 1.05, (0., 0., 0.), velocity=(.02, 0., 0.))
    _request(n)
    assert n.statuses[-1][1]['reason'] == 'empty_arm_advance_base_not_settled'


@pytest.mark.parametrize(('pose', 'reason'), (
    ((.01, .016, 0.), 'corridor_drift'),
    ((.01, 0., .021), 'corridor_drift'),
    ((.09, 0., 0.), 'overshoot'),
    ((.08, .009, 0.), 'endpoint_off_line'),
))
def test_advance_drift_stops_without_turning_or_recovery(pose, reason):
    n = _node()
    _request(n)
    _sensors(n, 1.05, pose)
    n._control()
    assert n.goal is None
    assert n.statuses[-1][0] == 'failed'
    assert reason in n.statuses[-1][1]['reason']
    assert n.statuses[-1][1]['plan_id'] == 'prepared-7'
    assert all(cmd.angular.z == 0 for cmd in n.commands)
    assert n.commands[-1] == Twist()


def test_prepared_joint_drift_stops_and_clears_certificate():
    n = _node()
    _request(n)
    _sensors(n, 1.05, n.pose, joints={**PREPARED, 'arm_left_1_joint': -.39})
    n._control()
    assert n.statuses[-1][1]['reason'] == 'empty_arm_prepared_joint_drift:arm_left_1_joint'
    assert n.prepared_joint_guard == {}
    assert n.advance_bounds is None
    assert n.commands[-1] == Twist()


def test_old_odometry_fails_even_when_receipt_time_is_new():
    n = _node()
    _request(n)
    n.now_ns += 300_000_000
    n.last_odom_time = Time(nanoseconds=n.now_ns)
    n._control()
    assert n.statuses[-1][1]['reason'] == 'empty_arm_odometry_stale_or_invalid'


def test_front_laser_transform_and_extended_tool_rectangle_stop_before_body_limit():
    n = _node()
    n.transforms['front_laser'] = (.20, 0.)
    n._on_front_scan(_scan(n, 'front_laser', (.36,)))
    _request(n)
    assert n.front_clearance > n.stop_distance
    n._control()
    assert n.statuses[-1][1]['reason'] == 'empty_arm_advance_swept_footprint_obstacle:front'
    assert n.transform_requests[0] == ('odom', 'front_laser', 1_000_000_000)
    assert n.commands[-1] == Twist()


def test_realistic_shelf_distance_clears_extended_tool_rectangle():
    n = _node()
    n.transforms['front_laser'] = (.20, 0.)
    n._on_front_scan(_scan(n, 'front_laser', (.525,)))
    _request(n)
    n._control()
    assert n.goal is not None
    assert n.commands[-1].linear.x > 0


@pytest.mark.parametrize('cause', ('stale', 'transform'))
def test_missing_fresh_laser_geometry_fails_closed(cause):
    n = _node()
    _request(n)
    if cause == 'stale':
        n._on_front_scan(_scan(n, 'front_laser', stamp_ns=0))
    else:
        def unavailable(*args, **kwargs):
            raise RuntimeError('no transform')
        n.tf_buffer.lookup_transform = unavailable
    n._control()
    assert 'scan_' in n.statuses[-1][1]['reason']
    assert n.goal is None
    assert n.commands[-1] == Twist()


def test_staging_uses_tight_endpoint_and_waits_for_measured_stop():
    n = _node()
    _request(n, profile='empty_arm_staging', goal=(.08, 0., .02))
    _sensors(n, 1.05, (.04, 0., 0.))
    n._control()
    assert n.commands[-1].linear.x > 0  # Old45mm tolerance must not finish here.
    assert n.commands[-1].angular.z > 0
    _sensors(n, 1.10, (.076, 0., .02), velocity=(.02, 0., 0.))
    n._control()
    assert n.goal is not None
    assert n.commands[-1] == Twist()
    for tick in range(1, 10):
        _sensors(n, 1.10 + tick * .05, (.076, 0., .02))
        n._control()
        if n.goal is None:
            break
    assert n.statuses[-1][0] == 'reached'
    assert n.statuses[-1][1]['measured_settled'] is True
    assert n.now_ns >= 1_500_000_000


def test_eight_centimetre_advance_rollout_reaches_with_no_yaw_commands():
    n = _node()
    _request(n, goal=(.08, 0., .009))  # Accepted yaw tolerance never commands a turn.
    x = 0.
    for tick in range(1, 240):
        command = n.last_command
        x += command.linear.x * .05
        _sensors(n, 1.0 + tick * .05, (x, 0., 0.),
                 velocity=(command.linear.x, command.linear.y, command.angular.z))
        n._control()
        if n.goal is None:
            break
    assert n.statuses[-1][0] == 'reached'
    assert abs(x - .08) <= .008
    assert all(cmd.angular.z == 0.0 for cmd in n.commands)
    assert max(math.hypot(cmd.linear.x, cmd.linear.y) for cmd in n.commands) <= .06
    assert n.prepared_joint_guard == {}
    assert n.empty_arm_anchor is None


def test_empty_arm_profile_cannot_be_armed_for_an_unbound_pose_goal():
    n = _node()
    n._on_command(String(data=navigation.encode_event(
        'set_next_profile', profile='empty_arm_advance')))
    assert n.statuses[-1][1]['reason'] == 'invalid_navigation_profile'
    assert n.next_goal_profile == 'normal'
