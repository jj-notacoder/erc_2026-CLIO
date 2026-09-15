"""Book identity association from onboard marker/TF geometry, before grasping."""
from collections import deque
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from erc_phase1_solution.book_selection_context import (
    decode_context, make_context, select_registered_book,
)
from erc_phase1_solution.vision import BookDetection, select_target_book


NOW = 2_000_000_000


def fields(**changes):
    values = dict(marker=[2.755, 0., 2.26], normal=[-1., 0.],
                  observed_ns=1_000_000_000, dispatch_ns=NOW,
                  column=4, colour='blue')
    values.update(changes)
    return make_context(**values)


def decode(raw=None, **changes):
    return decode_context(fields() if raw is None else raw, NOW,
                          **dict(column=4, colour='blue', confirmed_row=None, **changes))


def book(colour='blue', x=10, y=10, row=3):
    return BookDetection(colour, (x, y, 12, 80), 900, .9, row)


@pytest.mark.parametrize('column', range(1, 6))
@pytest.mark.parametrize('colour', ['red', 'blue', 'green', 'yellow'])
@pytest.mark.parametrize('yaw', [0., .7, -2.2])
def test_every_column_colour_and_rotated_registration_excludes_neighbours(column, colour, yaw):
    normal = np.array([-np.cos(yaw), -np.sin(yaw)])
    tangent = np.array([normal[1], -normal[0]])
    marker = np.r_[np.array([2., -1.])+(3-column)*tangent, 2.26]
    context = decode_context(fields(marker=marker, normal=normal,
                                     column=column, colour=colour), NOW,
                             column=column, colour=colour, confirmed_row=None)
    for jitter in (-.25, 0., .25):
        for marker_error in (-.1999, 0., .1999):
            position = np.r_[marker[:2]+tangent*(jitter+marker_error)-normal*.065, .605]
            assert context.contains(position)
    for side in (-1, 1):
        neighbor = np.r_[marker[:2]+tangent*side*.55-normal*.065, 1.265]
        assert not context.contains(neighbor)


def test_neighbor_closer_to_image_center_cannot_replace_selected_bottom_row_book():
    selected = book(x=40, y=390, row=4)
    neighbor = book(x=160, y=210, row=2)
    assert select_target_book([neighbor, selected], 'blue',
                              reference_point=(160, 240)) is neighbor
    points = {selected.center: (2.82, .24, .605),
              neighbor.center: (2.82, -.75, 1.265)}
    chosen, center = select_registered_book([neighbor, selected], decode(),
                                           points.get, np.eye(4))
    assert chosen is selected and center == points[selected.center]


@pytest.mark.parametrize('height,row', [(1.595, 1), (1.265, 2), (.935, 3), (.605, 4)])
def test_confirmed_row_height_survives_local_crop_labels(height, row):
    raw = fields(confirmed_row=row, confirmed_point=[2.82, .02, height],
                 confirmed_stamp_ns=1_500_000_000)
    context = decode_context(raw, NOW, column=4, colour='blue', confirmed_row=row)
    assert context.contains([2.82, .02, height+.02])
    assert context.contains([2.82, .02, height-.02])
    assert not context.contains([2.82, .02, height+.33])
    assert not context.contains([2.82, .02, height-.33])
    # A local detector may number this single visible component as row 1.
    candidate = book(row=1)
    assert select_registered_book([candidate], context,
        lambda _: (2.82, .02, height), np.eye(4))[0] is candidate


@pytest.mark.parametrize('mutation', [
    {'schema': 2}, {'source': 'gazebo_model_pose'}, {'frame_id': 'base_footprint'},
    {'column': 2}, {'column': True}, {'colour': 'red'},
    {'marker_odom': [np.nan, 0., 0.]}, {'outward_normal_odom': [-.5, 0.]},
    {'observation_stamp_ns': NOW+1}, {'observation_stamp_ns': True},
    {'dispatch_stamp_ns': NOW+1}, {'dispatch_stamp_ns': 500_000_000},
    {'confirmed_row': 3}, {'confirmed_point_odom': [2.82, 0., .935]},
])
def test_invalid_or_mismatched_context_is_rejected(mutation):
    raw = fields(); raw.update(mutation)
    with pytest.raises(ValueError):
        decode(raw)


@pytest.mark.parametrize('raw', [None, {}, [], 'books'])
def test_missing_context_is_rejected(raw):
    with pytest.raises(ValueError, match='missing_or_incompatible'):
        decode_context(raw, NOW, column=4, colour='blue', confirmed_row=None)


def test_expired_static_registration_is_rejected_on_receipt_and_on_frame():
    with pytest.raises(ValueError, match='registration_stale'):
        decode_context(fields(), 122_000_000_001,
                       column=4, colour='blue', confirmed_row=None)
    with pytest.raises(ValueError, match='registration_stale'):
        decode().require_frame(122_000_000_001, 122_000_000_001, 122_000_000_001)


@pytest.mark.parametrize('rgb,depth,now', [
    (NOW-1, NOW, NOW), (NOW, NOW-1, NOW),
    (NOW+1, NOW, NOW), (NOW, NOW+1, NOW),
    (NOW, NOW, NOW+200_000_001),
])
def test_frames_must_follow_command_and_be_fresh_in_both_streams(rgb, depth, now):
    with pytest.raises(ValueError, match='frame_outside_epoch'):
        decode().require_frame(rgb, depth, now)


def test_missing_depth_and_multiple_same_colour_components_never_pick_arbitrarily():
    a, b = book(x=10), book(x=90)
    with pytest.raises(ValueError, match='target_outside_registered_bay'):
        select_registered_book([a], decode(), lambda _: None, np.eye(4))
    with pytest.raises(ValueError, match='ambiguous_candidates'):
        select_registered_book([a, b], decode(), lambda _: (2.82, .02, .935), np.eye(4))


def test_selection_applies_full_camera_to_odom_transform():
    matrix = np.array([[0., 0., 1., 2.], [-1., 0., 0., .1],
                       [0., -1., 0., 1.2], [0., 0., 0., 1.]])
    selected = book()
    assert select_registered_book([selected], decode(),
        lambda _: (.05, .265, .82), matrix)[0] is selected


def _mode_node():
    from erc_phase1_solution.perception_node import PerceptionNode
    node = NS(mode='books', target_column=4, target_colour='blue',
              confirmed_book_row=None, book_selection_context=decode(),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=NOW)),
              target_tracker=NS(reset=Mock()), bin_tracker=NS(reset=Mock()),
              last_tracking_depth_ns=NOW-10, last_tracking_status='',
              _publish_tracking_status=Mock(), _publish_status=Mock())
    for name in ('marker_history', 'book_history', 'bin_history',
                 'bin_rgb_frames', 'bin_depth_frames'):
        setattr(node, name, deque(['old']))
    node._on_mode = MethodType(PerceptionNode._on_mode, node)
    return node


def test_invalid_replacement_books_command_disables_previous_valid_context():
    from erc_phase1_solution.common import encode_event
    from std_msgs.msg import String
    node = _mode_node()
    node._on_mode(String(data=encode_event('books', shelf_column_number=1,
                                          book_colour='red')))
    assert node.mode == 'idle' and node.book_selection_context is None
    assert not node.book_history
    node.target_tracker.reset.assert_called_once()
    assert node._publish_status.call_args.args[0] == 'mode_rejected'
    assert node._publish_tracking_status.call_args.kwargs['reason'] == (
        'book_selection_context_missing_or_incompatible')


def test_same_target_new_mode_epoch_resets_previous_track_and_frame_history():
    from erc_phase1_solution.common import encode_event
    from std_msgs.msg import String
    node = _mode_node()
    node.book_selection_context = decode_context(fields(dispatch_ns=NOW-10), NOW,
        column=4, colour='blue', confirmed_row=None)
    node._on_mode(String(data=encode_event('books', shelf_column_number=4,
        book_colour='blue', book_selection_context=fields())))
    assert node.mode == 'books' and node.book_selection_context.dispatch_stamp_ns == NOW
    assert not node.book_history and node.last_tracking_depth_ns == -1
    node.target_tracker.reset.assert_called_once()


def test_mission_dispatches_observed_bay_and_original_confirmed_row_point():
    from erc_phase1_solution.mission_manager import MissionManager
    node = NS(target_column=4, target_colour='blue', row_confirmed=True, detected_row=3,
              target_marker_odom=[2.755, 0., 2.26], shelf_normal=[-1., 0.],
              _shelf_registration_stamp_ns=1_000_000_000,
              _confirmed_book_point_odom=[2.82, .02, .935],
              _confirmed_book_point_stamp_ns=1_500_000_000,
              mode_pub=object(), _command=Mock(), _abort=Mock(),
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=NOW)))
    MissionManager._perception_mode(node, 'books')
    payload = node._command.call_args.kwargs
    context = decode_context(payload['book_selection_context'], NOW,
                             column=4, colour='blue', confirmed_row=3)
    assert context.confirmed_point_odom == (2.82, .02, .935)
    assert context.observation_stamp_ns == 1_000_000_000
    assert context.confirmed_point_stamp_ns == 1_500_000_000
    assert payload['confirmed_row'] == 3
    node._abort.assert_not_called()


def test_delayed_tf_retries_matching_buffered_pairs_and_keeps_three_frame_track():
    from sensor_msgs.msg import Image
    from tf2_ros import TransformException
    from erc_phase1_solution.perception_node import PerceptionNode
    from erc_phase1_solution.target_tracking import TargetBookTracker
    from test_target_tracking import _observation
    clock = NS(ns=NOW, tf_ns=NOW-1)
    tracker = TargetBookTracker()
    node = NS(book_selection_context=decode(), last_tracking_depth_ns=-1,
              maximum_tracking_frame_skew=.04, tracking_maximum_age=.15,
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=clock.ns)),
              _tracking_unavailable=Mock())
    node._registered_book_frame = MethodType(PerceptionNode._registered_book_frame, node)
    def lookup(_target, _source, stamp):
        if stamp.nanoseconds > clock.tf_ns:
            raise TransformException('TF has not caught up')
        return NS(transform=NS(translation=NS(x=0., y=0., z=0.),
                               rotation=NS(x=0., y=0., z=0., w=1.)))
    node.tf_buffer = NS(lookup_transform=lookup)
    def update(stamp):
        message = Image(); message.header.frame_id = 'camera'
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        node.latest_rgb_message = node.latest_depth_message = message
        node.latest_rgb = np.full((2, 2, 3), stamp-NOW, dtype=np.int64)
        node.latest_depth = np.full((2, 2), stamp-NOW, dtype=np.int64)
    update(NOW)
    assert node._registered_book_frame() is None
    assert node.last_tracking_depth_ns == -1  # The pending pair is not consumed.
    observed = []
    for step in range(1, 4):
        clock.ns = NOW+step*66_000_000
        clock.tf_ns = clock.ns-33_000_000
        update(clock.ns)  # The latest image is newer than available exact TF.
        rgb_msg, rgb, depth_msg, depth, _ = node._registered_book_frame()
        actual = depth_msg.header.stamp.sec*1_000_000_000+depth_msg.header.stamp.nanosec
        expected = clock.ns-66_000_000
        assert actual == expected
        assert rgb_msg is depth_msg
        assert np.all(rgb == expected-NOW) and np.all(depth == expected-NOW)
        node.last_tracking_depth_ns = actual
        observed.append(actual)
        assert tracker.observe(_observation(actual)).ok
    assert len(set(observed)) == 3
    assert tracker.latest(clock.ns).ok


def test_tracking_cloud_uses_observation_header_not_newest_buffered_depth():
    from erc_phase1_solution.perception_node import PerceptionNode
    from test_target_tracking import _observation
    node = NS(target_tracking_pub=NS(publish=Mock()),
              _header=Mock(side_effect=AssertionError('latest header used')))
    observation = _observation(NOW)
    PerceptionNode._publish_target_tracking(node, observation)
    cloud = node.target_tracking_pub.publish.call_args.args[0]
    assert cloud.header.frame_id == observation.frame_id
    assert cloud.header.stamp.sec*1_000_000_000+cloud.header.stamp.nanosec == NOW


@pytest.mark.parametrize('stamp,accepted', [(NOW-1, False), (NOW, False),
    (NOW+1, True), (NOW+100_000_000, True), (NOW+100_000_001, False)])
def test_mission_reacquisition_accepts_only_new_book_epoch_and_no_future_point(stamp, accepted):
    from erc_phase1_solution.mission_manager import MissionManager
    from geometry_msgs.msg import PointStamped
    node = NS(_book_mode_epoch_ns=NOW, book_point=None,
              get_clock=lambda: NS(now=lambda: NS(nanoseconds=NOW+100_000_000)))
    point = PointStamped()
    point.header.stamp.sec, point.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    MissionManager._on_book(node, point)
    assert (node.book_point is point) is accepted
