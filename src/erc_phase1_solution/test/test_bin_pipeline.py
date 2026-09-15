"""ROS message wiring checks without a running ROS graph or simulation."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json

import numpy as np
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import TransformException

from erc_phase1_solution.bin_geometry import BinTracker
from erc_phase1_solution import perception_node
from erc_phase1_solution.manipulation_node import ManipulationNode
from erc_phase1_solution.mission_manager import MissionManager


def image_message(stamp_ns):
    message = Image()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    message.header.frame_id = 'camera_optical_frame'
    return message


@pytest.fixture
def bin_node(monkeypatch):
    fixture = np.load(Path(__file__).parent/'data'/'bin_geometry'/'real_bin_rgbd.npz')
    now = [10_100_000_000]
    transform = SimpleNamespace(translation=None, rotation=None)
    node = SimpleNamespace(
        camera_info=SimpleNamespace(k=fixture['k']),
        bin_rgb_frames=deque(), bin_depth_frames=deque(),
        bin_table_height=.73, last_bin_consumed_ns=-1, last_bin_verified_ns=-1,
        bin_tracker=BinTracker(), bin_pub=Mock(), saved_modes={'bin'},
        tf_buffer=Mock(), _publish_status=Mock(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now[0])),
    )
    node.tf_buffer.lookup_transform.return_value = SimpleNamespace(transform=transform)
    monkeypatch.setattr(perception_node, 'camera_transform', lambda *args: fixture['base_from_camera'])
    node._invalidate_bin = lambda reason, stamp: perception_node.PerceptionNode._invalidate_bin(
        node, reason, stamp,
    )

    def frame(stamp, depth_stamp=None):
        node.bin_rgb_frames.append((image_message(stamp), fixture['bgr']))
        node.bin_depth_frames.append((
            image_message(stamp if depth_stamp is None else depth_stamp), fixture['depth'],
        ))
        now[0] = stamp+100_000_000

    return node, now, frame


def test_bin_pipeline_needs_distinct_matched_frames_then_expires(bin_node):
    node, now, frame = bin_node
    for stamp in (10_000_000_000, 10_100_000_000):
        frame(stamp)
        perception_node.PerceptionNode._process_bin(node)
        perception_node.PerceptionNode._process_bin(node)
        node.bin_pub.publish.assert_not_called()
    frame(10_200_000_000)
    perception_node.PerceptionNode._process_bin(node)
    node.bin_pub.publish.assert_called_once()
    point = node.bin_pub.publish.call_args.args[0]
    assert point.header.stamp.nanosec == 200_000_000
    assert point.header.frame_id == 'camera_optical_frame'
    transform_args = node.tf_buffer.lookup_transform.call_args.args
    assert transform_args[:2] == ('base_footprint', 'camera_optical_frame')
    assert transform_args[2].nanoseconds == 10_200_000_000
    now[0] = 10_700_000_001
    perception_node.PerceptionNode._process_bin(node)
    assert node.bin_tracker.count == 0
    assert node._publish_status.call_args.args[0] == 'bin_invalid'


def test_missing_tf_or_stale_depth_cannot_publish_and_does_not_consume_rgb(bin_node):
    node, _, frame = bin_node
    frame(10_000_000_000, 9_800_000_000)
    perception_node.PerceptionNode._process_bin(node)
    node.bin_pub.publish.assert_not_called()
    assert node.last_bin_consumed_ns == -1
    frame(10_100_000_000)
    node.tf_buffer.lookup_transform.side_effect = TransformException('not yet available')
    perception_node.PerceptionNode._process_bin(node)
    assert node.last_bin_consumed_ns == -1
    node.tf_buffer.lookup_transform.side_effect = None
    perception_node.PerceptionNode._process_bin(node)
    assert node.last_bin_consumed_ns == 10_100_000_000
    assert node.bin_tracker.count == 1


def test_deprojection_uses_depth_frame_stamp_and_matching_verification(bin_node):
    node, _, frame = bin_node
    for stamp in (10_000_000_000, 10_100_000_000, 10_200_000_000):
        frame(stamp, stamp+20_000_000)
        node.bin_depth_frames[-1][0].header.frame_id = 'depth_optical_frame'
        perception_node.PerceptionNode._process_bin(node)
    point = node.bin_pub.publish.call_args.args[0]
    assert point.header.frame_id == 'depth_optical_frame'
    assert point.header.stamp.nanosec == 220_000_000
    args = node.tf_buffer.lookup_transform.call_args.args
    assert args[1] == 'depth_optical_frame'
    assert args[2].nanoseconds == 10_220_000_000
    assert node._publish_status.call_args.kwargs['observation_stamp_ns'] == 10_220_000_000


def test_reused_depth_cannot_count_as_another_confirmation_frame(bin_node):
    node, _, frame = bin_node
    frame(10_000_000_000)
    perception_node.PerceptionNode._process_bin(node)
    frame(10_020_000_000, 10_000_000_000)
    perception_node.PerceptionNode._process_bin(node)
    assert node.bin_tracker.count == 1
    assert node.last_bin_consumed_ns == 10_000_000_000
    node.bin_pub.publish.assert_not_called()


@pytest.mark.parametrize('receive, status, attribute', [
    (MissionManager._on_bin, MissionManager._on_perception_status, 'bin_point'),
    (ManipulationNode._on_bin, ManipulationNode._on_bin_status, 'latest_bin'),
])
def test_consumers_clear_rejected_points_and_reject_latched_replay(receive, status, attribute):
    node = SimpleNamespace(
        bin_point=None, latest_bin=None, ready={}, bin_invalidated_ns=-1,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=2_100_000_000)),
    )
    point = PointStamped()
    point.header.stamp.sec = 2
    point.header.frame_id = 'camera_optical_frame'
    point.point.z = 2.0
    receive(node, point)
    assert getattr(node, attribute) is None
    status(node, String(data=json.dumps({
        'mode': 'bin', 'event': 'bin_verified', 'bin_valid': True,
        'observation_stamp_ns': 2_000_000_000,
    })))
    assert getattr(node, attribute) is point
    status(node, String(data=json.dumps({
        'mode': 'bin', 'event': 'bin_invalid', 'bin_valid': False,
        'observation_stamp_ns': 2_000_000_000,
    })))
    assert getattr(node, attribute) is None
    receive(node, point)
    assert getattr(node, attribute) is None
    point = PointStamped()
    point.header.stamp.sec = 2
    point.header.stamp.nanosec = 50_000_000
    point.header.frame_id = 'camera_optical_frame'
    point.point.z = 2.0
    status(node, String(data=json.dumps({
        'mode': 'bin', 'event': 'bin_verified', 'bin_valid': True,
        'observation_stamp_ns': 2_050_000_000,
    })))
    assert getattr(node, attribute) is None
    receive(node, point)
    assert getattr(node, attribute) is point
    status(node, String(data=json.dumps({
        'mode': 'bin', 'event': 'bin_invalid', 'bin_valid': False,
        'observation_stamp_ns': 2_000_000_000,
    })))
    assert getattr(node, attribute) is point
