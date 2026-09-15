"""Focused tests for manipulation's volatile RGB-D target contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32, PointCloud
from std_msgs.msg import String
import yaml

from erc_phase1_solution import manipulation_node


STAMP_NS = 10_000_000_000


def _channel(name, values):
    result = ChannelFloat32(name=name)
    if isinstance(values, (float, int)):
        result.values = [float(values)] * 5
    else:
        result.values = [float(value) for value in values]
    return result


def _cloud(*, stamp_ns=STAMP_NS, sequence=3, center_x=0.0):
    cloud = PointCloud()
    cloud.header.frame_id = 'camera_depth_optical_frame'
    cloud.header.stamp.sec = stamp_ns // 1_000_000_000
    cloud.header.stamp.nanosec = stamp_ns % 1_000_000_000
    points = (
        (center_x, 0.0, 1.0),
        (center_x - 0.015, -0.125, 1.0),
        (center_x + 0.015, -0.125, 1.0),
        (center_x + 0.015, 0.125, 1.0),
        (center_x - 0.015, 0.125, 1.0),
    )
    cloud.points = [Point32(x=x, y=y, z=z) for x, y, z in points]
    cloud.channels = [
        _channel('point_role', range(5)),
        _channel('target_row', 2),
        _channel('track_generation', 4),
        _channel('sequence', sequence),
        _channel('detection_confidence', 0.96),
        _channel('quality_confidence', 0.85),
        _channel('identity_continuity_confidence', 0.92),
        _channel('depth_coverage', 0.93),
        _channel('plane_residual_m', 0.0002),
        _channel('center_uncertainty_m', 0.0001),
        _channel('extent_uncertainty_m', 0.0002),
        _channel('orientation_uncertainty_rad', 0.01),
        _channel('short_extent_m', 0.03),
        _channel('long_extent_m', 0.25),
        _channel('face_normal_x', 0.0),
        _channel('face_normal_y', 0.0),
        _channel('face_normal_z', -1.0),
        _channel('long_axis_x', 0.0),
        _channel('long_axis_y', 1.0),
        _channel('long_axis_z', 0.0),
        _channel('observed_rate_hz', 15.0),
    ]
    return cloud


def _status(*, stamp_ns=STAMP_NS, sequence=3):
    return manipulation_node._TargetTrackingStatus(
        True,
        '',
        track_id='red:row2:track4',
        track_generation=4,
        target_colour='red',
        row=2,
        stamp_ns=stamp_ns,
        sequence=sequence,
    )


class _Harness:
    _tracking_observation_from_cloud = (
        manipulation_node.ManipulationNode._tracking_observation_from_cloud
    )
    _transform_target_observation = (
        manipulation_node.ManipulationNode._transform_target_observation
    )
    _current_target_observation = (
        manipulation_node.ManipulationNode._current_target_observation
    )
    _guard_target_motion = manipulation_node.ManipulationNode._guard_target_motion
    _on_target_tracking = manipulation_node.ManipulationNode._on_target_tracking
    _on_target_tracking_status = (
        manipulation_node.ManipulationNode._on_target_tracking_status
    )

    def __init__(self, cloud=None, status=None, now_ns=STAMP_NS + 50_000_000):
        self.target_colour = 'red'
        self._lock = threading.Lock()
        self._latest_target_tracking = cloud
        self._target_tracking_status = status or _status()
        self.target_tracking_max_age = 0.15
        self.target_tracking_guard_wait = 0.04
        self.target_tracking_minimum_rate = 10.0
        self.target_tracking_minimum_depth_coverage = 0.65
        self.target_tracking_maximum_plane_residual = 0.002
        self.target_tracking_maximum_center_uncertainty = 0.0005
        self.target_tracking_maximum_extent_uncertainty = 0.001
        self.target_tracking_maximum_orientation_uncertainty = 0.05
        self.tf_timeout = 1.0
        self._cancel = threading.Event()
        self._now_ns = now_ns
        self.get_clock = lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=self._now_ns)
        )
        self.lookup_calls = []

        def lookup(target, source, stamp, *, timeout):
            self.lookup_calls.append((target, source, stamp.nanoseconds, timeout))
            half = np.sqrt(0.5)
            return SimpleNamespace(
                transform=SimpleNamespace(
                    translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                    rotation=SimpleNamespace(x=0.0, y=0.0, z=half, w=half),
                )
            )

        self.tf_buffer = SimpleNamespace(lookup_transform=lookup)


def test_status_decoder_preserves_identity_and_rejects_invalid_state():
    valid = manipulation_node._decode_target_tracking_status(
        manipulation_node.encode_event(
            'target_tracking',
            mode='books',
            target_colour='red',
            row=2,
            track_id='red:row2:track4',
            track_generation=4,
            stamp_ns=STAMP_NS,
            sequence=3,
        ),
        'red',
    )
    assert valid.available
    assert valid.track_id == 'red:row2:track4'
    assert valid.track_generation == 4
    assert valid.sequence == 3

    wrong_colour = manipulation_node._decode_target_tracking_status(
        manipulation_node.encode_event(
            'target_tracking',
            mode='books',
            target_colour='blue',
            row=2,
            track_id='blue:row2:track1',
            track_generation=1,
            stamp_ns=STAMP_NS,
            sequence=3,
        ),
        'red',
    )
    assert not wrong_colour.available
    assert 'colour' in wrong_colour.reason

    occluded = manipulation_node._decode_target_tracking_status(
        manipulation_node.encode_event(
            'target_tracking_unavailable',
            reason='target_occluded',
        ),
        'red',
    )
    assert not occluded.available
    assert occluded.reason == 'target_occluded'


def test_current_observation_transforms_all_five_points_at_message_time():
    node = _Harness(_cloud())

    observation = node._current_target_observation(target_frame='odom')

    assert node.lookup_calls[0][0:3] == (
        'odom',
        'camera_depth_optical_frame',
        STAMP_NS,
    )
    source = np.asarray(
        [[point.x, point.y, point.z] for point in node._latest_target_tracking.points]
    )
    expected = np.column_stack((-source[:, 1], source[:, 0], source[:, 2]))
    expected += np.asarray([1.0, 2.0, 3.0])
    actual = np.asarray([observation.center, *observation.corners])
    assert actual == pytest.approx(expected)
    assert observation.face_normal == pytest.approx((0.0, 0.0, -1.0))
    assert observation.long_axis == pytest.approx((-1.0, 0.0, 0.0))
    assert observation.track_id == 'red:row2:track4'
    assert observation.track_generation == 4
    assert observation.sequence == 3
    assert observation.stamp_ns == STAMP_NS


@pytest.mark.parametrize(
    ('channel_name', 'unsafe_value', 'reason'),
    (
        ('depth_coverage', 0.60, 'depth coverage'),
        ('center_uncertainty_m', 0.0006, 'center uncertainty'),
        ('extent_uncertainty_m', 0.0011, 'extent uncertainty'),
        ('orientation_uncertainty_rad', 0.051, 'orientation uncertainty'),
        ('observed_rate_hz', 9.9, 'rate'),
    ),
)
def test_current_observation_fails_closed_on_unsafe_quality(
    channel_name,
    unsafe_value,
    reason,
):
    cloud = _cloud()
    channel = next(item for item in cloud.channels if item.name == channel_name)
    channel.values = [unsafe_value] * 5
    node = _Harness(cloud)

    with pytest.raises(RuntimeError, match=reason):
        node._current_target_observation()


def test_current_observation_never_relaxes_the_150ms_freshness_cap():
    node = _Harness(_cloud(), now_ns=STAMP_NS + 151_000_000)

    with pytest.raises(RuntimeError, match='stale'):
        node._current_target_observation(maximum_age_seconds=2.0)


def test_current_observation_fails_closed_when_exact_time_tf_is_missing():
    node = _Harness(_cloud())

    def unavailable(*unused, **unused_keywords):
        raise manipulation_node.TransformException('transform unavailable')

    node.tf_buffer.lookup_transform = unavailable
    with pytest.raises(RuntimeError, match='cannot transform live target'):
        node._current_target_observation(target_frame='odom')


def test_unavailable_status_callback_clears_previous_geometry():
    node = _Harness(_cloud())
    message = String()
    message.data = manipulation_node.encode_event(
        'target_tracking_unavailable',
        reason='target_occluded',
    )

    node._on_target_tracking_status(message)

    assert node._latest_target_tracking is None
    with pytest.raises(RuntimeError, match='target_occluded'):
        node._current_target_observation()


def test_current_observation_rejects_generation_and_status_order_mismatch():
    cloud = _cloud(sequence=4)
    generation = next(
        item for item in cloud.channels if item.name == 'track_generation'
    )
    generation.values = [5.0] * 5
    node = _Harness(cloud)
    with pytest.raises(RuntimeError, match='generation'):
        node._current_target_observation()

    node = _Harness(_cloud(sequence=2), status=_status(sequence=3))
    with pytest.raises(RuntimeError, match='predates'):
        node._current_target_observation()


def test_motion_guard_requires_new_same_identity_and_uncertainty_bound():
    node = _Harness(_cloud())
    reference = node._current_target_observation(target_frame='odom')
    newer = replace(
        reference,
        stamp_ns=reference.stamp_ns + 50_000_000,
        sequence=reference.sequence + 1,
        center=(reference.center[0] + 0.0001, *reference.center[1:]),
        corners=tuple(
            (corner[0] + 0.0001, corner[1], corner[2])
            for corner in reference.corners
        ),
    )
    node._current_target_observation = lambda **unused: newer

    result = node._guard_target_motion(
        reference,
        maximum_center_motion_m=0.0005,
        maximum_corner_motion_m=0.0006,
        maximum_orientation_change_rad=0.03,
    )
    assert result.ok
    assert result.observation is newer

    node._current_target_observation = lambda **unused: replace(
        newer,
        track_id='red:row2:track5',
        track_generation=5,
    )
    with pytest.raises(RuntimeError, match='identity changed'):
        node._guard_target_motion(reference)


def test_manipulation_tracking_configuration_matches_producer_limits():
    configuration = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / 'config'
            / 'solution.yaml'
        ).read_text(encoding='utf-8')
    )
    producer = configuration['erc_perception']['ros__parameters']
    consumer = configuration['erc_manipulation']['ros__parameters']

    assert consumer['target_tracking_maximum_age_seconds'] <= 0.15
    assert consumer['target_tracking_minimum_rate_hz'] >= 10.0
    for suffix in (
        'minimum_depth_coverage',
        'maximum_plane_residual_m',
        'maximum_center_uncertainty_m',
        'maximum_orientation_uncertainty_rad',
    ):
        assert consumer[f'target_tracking_{suffix}'] == producer[
            f'target_tracking_{suffix}'
        ]
