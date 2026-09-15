"""ROS-aware regression coverage for payload-safe navigation profiles."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


pytest.importorskip('rclpy')

from geometry_msgs.msg import PoseStamped, Twist  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
from rclpy.time import Time  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from erc_phase1_solution import navigation_node  # noqa: E402


def _profile_node(profile: str = 'carried_retreat'):
    """Build the motion-profile portion of a node without a ROS graph."""
    node = object.__new__(navigation_node.NavigationNode)
    node.control_rate = 20.0
    node.max_vx = 0.42
    node.max_vy = 0.32
    node.linear_acceleration = 0.55
    node.angular_acceleration = 0.8
    node.carried_retreat_max_speed = 0.10
    node.carried_retreat_acceleration = 0.06
    node.carried_retreat_braking_acceleration = 0.06
    node.carried_retreat_max_angular_speed = 0.08
    node.carried_retreat_angular_acceleration = 0.10
    node.position_tolerance = 0.045
    node.yaw_tolerance = 0.045
    node.settle_seconds = 0.45
    node.carried_retreat_settle_seconds = 0.10
    node.goal_profile = profile
    node.last_command = Twist()
    return node


@pytest.mark.parametrize(
    ('body_x', 'body_y', 'distance', 'expected_speed'),
    (
        (-0.35, 0.0, 0.35, 0.10),
        (0.10, 0.0, 0.10, (2.0 * 0.06 * 0.055) ** 0.5),
        (0.046, 0.0, 0.046, (2.0 * 0.06 * 0.001) ** 0.5),
    ),
)
def test_carried_retreat_uses_capped_braking_speed(
    body_x,
    body_y,
    distance,
    expected_speed,
):
    node = _profile_node()

    vx, vy = node._desired_translation(body_x, body_y, distance)

    assert (vx * vx + vy * vy) ** 0.5 == pytest.approx(expected_speed)
    assert vx * body_x >= 0.0
    assert vy == pytest.approx(0.0)


def test_carried_retreat_uses_its_lower_acceleration_limit_only():
    carried = _profile_node('carried_retreat')
    normal = _profile_node('normal')
    desired = Twist()
    desired.linear.x = -1.0
    desired.linear.y = 1.0
    desired.angular.z = 1.0

    carried_command = carried._limited_command(desired)
    normal_command = normal._limited_command(desired)

    carried_axis_step = (0.06 / 20.0) / (2.0 ** 0.5)
    assert carried_command.linear.x == pytest.approx(-carried_axis_step)
    assert carried_command.linear.y == pytest.approx(carried_axis_step)
    assert (
        carried_command.linear.x ** 2 + carried_command.linear.y ** 2
    ) ** 0.5 <= 0.06 / 20.0 + 1e-12
    assert normal_command.linear.x == pytest.approx(-0.55 / 20.0)
    assert normal_command.linear.y == pytest.approx(0.55 / 20.0)
    assert carried_command.angular.z == pytest.approx(0.10 / 20.0)
    assert normal_command.angular.z == pytest.approx(0.8 / 20.0)


@pytest.mark.parametrize('goal_yaw', (-1.0, 1.0))
def test_carried_control_enforces_angular_speed_cap(goal_yaw):
    node = _profile_node()
    now = Time(nanoseconds=1_000_000_000)
    node.pose = (0.0, 0.0, 0.0)
    node.goal = (0.0, 0.0, goal_yaw)
    node.goal_id = 7
    node.goal_started = None
    node.settle_started = None
    node.last_odom_time = now
    node.last_front_scan_time = now
    node.last_rear_scan_time = now
    node.sensor_stale_timeout = 1.0
    node.front_clearance = 2.0
    node.rear_clearance = 2.0
    node.stop_distance = 0.28
    node.blocked = False
    node._ready_announced = True
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    node._sample_executed_path = lambda force=False: None
    published = []
    node.cmd_pub = SimpleNamespace(
        get_subscription_count=lambda: 1,
        publish=lambda message: published.append(message),
    )
    node._publish_status = lambda *args, **kwargs: None

    for _ in range(30):
        node._control()

    signed_limit = node.carried_retreat_max_angular_speed * (
        -1.0 if goal_yaw < 0.0 else 1.0
    )
    assert published
    assert all(
        abs(command.angular.z)
        <= node.carried_retreat_max_angular_speed + 1e-12
        for command in published
    )
    assert node.last_command.angular.z == pytest.approx(signed_limit)


def _lifecycle_node():
    node = object.__new__(navigation_node.NavigationNode)
    node.pose = (0.0, 0.0, 0.0)
    node.goal = None
    node.goal_id = 0
    node.goal_started = None
    node.settle_started = None
    node.next_goal_profile = 'normal'
    node.goal_profile = 'normal'
    node.blocked = False
    node.dry_run = False
    node.executed_path = Path()
    node._last_path_sample = None
    node.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=1_000_000_000))
    node._publish_planned_path = lambda: None
    node._sample_executed_path = lambda force=False: None
    node._publish_zero = lambda: None
    node.statuses = []
    node._publish_status = lambda event, **fields: node.statuses.append(
        (event, fields)
    )
    return node


def _goal(x: float = -0.35) -> PoseStamped:
    message = PoseStamped()
    message.header.frame_id = 'odom'
    message.pose.position.x = x
    message.pose.orientation.w = 1.0
    return message


def _command(event: str, **fields) -> String:
    message = String()
    message.data = navigation_node.encode_event(event, **fields)
    return message


def test_carried_retreat_profile_is_consumed_by_one_accepted_goal():
    node = _lifecycle_node()

    node._on_command(_command('set_next_profile', profile='carried_retreat'))
    assert node.next_goal_profile == 'carried_retreat'
    assert node.goal_profile == 'normal'

    node._on_goal(_goal())
    assert node.goal_profile == 'carried_retreat'
    assert node.next_goal_profile == 'normal'
    assert node.statuses[-1][0] == 'accepted'
    assert node.statuses[-1][1]['profile'] == 'carried_retreat'

    node._on_command(_command('cancel'))
    assert node.goal is None
    assert node.goal_profile == 'normal'
    assert node.next_goal_profile == 'normal'

    node._on_goal(_goal(-0.20))
    assert node.goal_profile == 'normal'
    assert node.statuses[-1][0] == 'accepted'
    assert node.statuses[-1][1]['profile'] == 'normal'


def test_profiled_navigation_command_binds_goal_and_profile_atomically():
    node = _lifecycle_node()
    node.next_goal_profile = 'carried_retreat'

    node._on_command(
        _command(
            'navigate',
            x=-0.35,
            y=0.02,
            yaw=-0.01,
            profile='carried_retreat',
        )
    )

    assert node.goal == pytest.approx((-0.35, 0.02, -0.01))
    assert node.goal_profile == 'carried_retreat'
    assert node.next_goal_profile == 'normal'
    assert node.statuses[-1] == (
        'accepted',
        {
            'goal_id': 1,
            'goal': [-0.35, 0.02, -0.01],
            'profile': 'carried_retreat',
        },
    )


def test_cancel_clears_an_armed_profile_without_an_active_goal():
    node = _lifecycle_node()
    node._on_command(_command('set_next_profile', profile='carried_retreat'))

    node._on_command(_command('cancel'))

    assert node.next_goal_profile == 'normal'
    assert node.goal_profile == 'normal'
    assert node.statuses[-1] == (
        'cancelled',
        {'goal_id': 0, 'active_goal': False},
    )


@pytest.mark.parametrize(
    ('mutate', 'reason'),
    (
        (
            lambda node, goal: setattr(goal.header, 'frame_id', 'map'),
            'goal_frame_must_be_odom',
        ),
        (lambda node, goal: setattr(node, 'pose', None), 'odometry_not_ready'),
        (
            lambda node, goal: setattr(node, 'goal', (1.0, 0.0, 0.0)),
            'goal_already_active',
        ),
        (
            lambda node, goal: setattr(goal.pose.position, 'x', float('nan')),
            'goal_must_be_finite',
        ),
    ),
)
def test_rejected_pose_goal_consumes_legacy_armed_profile(mutate, reason):
    node = _lifecycle_node()
    node.next_goal_profile = 'carried_retreat'
    goal = _goal()
    mutate(node, goal)

    node._on_goal(goal)

    assert node.next_goal_profile == 'normal'
    assert node.goal_profile == 'normal'
    assert node.statuses[-1][0] == 'rejected'
    assert node.statuses[-1][1]['reason'] == reason


def test_carried_in_tolerance_ramp_stops_immediately_for_low_clearance():
    node = _profile_node()
    now = Time(nanoseconds=1_000_000_000)
    node.pose = (0.0, 0.0, 0.0)
    node.goal = (0.01, 0.0, 0.0)
    node.goal_id = 4
    node.goal_started = None
    node.settle_started = None
    node.last_odom_time = now
    node.last_front_scan_time = now
    node.last_rear_scan_time = now
    node.sensor_stale_timeout = 1.0
    node.front_clearance = 0.10
    node.rear_clearance = 0.10
    node.stop_distance = 0.28
    node.blocked = False
    node._ready_announced = True
    node.get_clock = lambda: SimpleNamespace(now=lambda: now)
    node._sample_executed_path = lambda force=False: None
    published = []
    node.cmd_pub = SimpleNamespace(
        get_subscription_count=lambda: 1,
        publish=lambda message: published.append(message),
    )
    node.statuses = []
    node._publish_status = lambda event, **fields: node.statuses.append(
        (event, fields)
    )
    node.last_command.linear.x = 0.05

    node._control()

    assert published
    assert published[-1].linear.x == pytest.approx(0.0)
    assert node.last_command.linear.x == pytest.approx(0.0)
    assert node.blocked
    assert node.statuses[-1] == (
        'blocked',
        {'goal_id': 4, 'clearance': pytest.approx(0.10)},
    )


def test_carried_retreat_rollout_finishes_within_gravity_creep_window():
    node = _profile_node()
    dt = 1.0 / node.control_rate
    target_distance = 0.35
    position = 0.0
    elapsed = 0.0
    speed_deltas = []

    while target_distance - position > node.position_tolerance:
        distance = target_distance - position
        desired = Twist()
        desired.linear.x, desired.linear.y = node._desired_translation(
            distance,
            0.0,
            distance,
        )
        previous_speed = node.last_command.linear.x
        node.last_command = node._limited_command(desired)
        speed_deltas.append(abs(node.last_command.linear.x - previous_speed))
        position += node.last_command.linear.x * dt
        elapsed += dt
        assert elapsed < 10.0

    # Mirror the in-tolerance branch: ramp the remaining command below the
    # threshold before the controller publishes its final exact zero.
    while abs(node.last_command.linear.x) > 0.005:
        previous_speed = node.last_command.linear.x
        node.last_command = node._limited_command(Twist())
        speed_deltas.append(abs(node.last_command.linear.x - previous_speed))
        position += node.last_command.linear.x * dt
        elapsed += dt
        assert elapsed < 10.0

    node.last_command = Twist()
    elapsed += node.carried_retreat_settle_seconds

    assert position >= target_distance - node.position_tolerance
    assert max(speed_deltas) <= node.carried_retreat_acceleration * dt + 1e-12
    assert elapsed <= 5.0
    assert node.last_command.linear.x == 0.0
