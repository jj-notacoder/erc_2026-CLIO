"""Mission-to-pick propagation uses the original visual registration epoch."""
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import PointStamped
from erc_phase1_solution.mission_manager import MissionManager


def mission(enabled=True):
    node = object.__new__(MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1800.
    node.state = 'REACQUIRE_BOOK'
    node.book_point = None
    node._book_reacquire_after_ns = 49_500_000_000
    node._book_mode_epoch_ns = 49_500_000_000
    node.pick_attempts = 0
    node.target_book_model = 'old_book'
    node.lift_first_extraction_enabled = enabled
    node.target_marker_odom = [3., 1., 2.26]
    node.shelf_normal = [-1., 0.]
    node.target_physical_column = 2
    node._shelf_registration_stamp_ns = 20_000_000_000
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=50_000_000_000))
    node._manipulate = Mock()
    node._set_state = Mock()
    node._abort = Mock()
    # Book admission requires a fresh observation after the camera/mode epoch,
    # independently of whether the optional lift bay is attached to PICK.
    book = PointStamped()
    book.header.frame_id = 'head_front_camera_depth_optical_frame'
    book.header.stamp.sec = 49
    book.header.stamp.nanosec = 900_000_000
    book.point.x, book.point.y, book.point.z = .05, .10, .65
    node._on_book(book)
    assert node.book_point is book
    return node


def test_enabled_mission_preserves_marker_epoch_column_and_odom_frame_in_pick():
    node = mission()
    node._tick()
    args, fields = node._manipulate.call_args
    assert args == ('pick',)
    context = fields['shelf_bay_context']
    assert context['observation_stamp_ns'] == 20_000_000_000
    assert context['dispatch_stamp_ns'] == 50_000_000_000
    assert context['physical_column'] == 2
    assert context['marker_odom'] == [3., 1., 2.26]
    assert context['frame_id'] == 'odom'
    node._set_state.assert_called_once_with('PICK')
    node._abort.assert_not_called()
    assert node.target_book_model is None


def test_disabled_mission_retains_existing_pick_interface():
    node = mission(False)
    node._shelf_registration_stamp_ns = None
    node._tick()
    node._manipulate.assert_called_once_with('pick')
    node._abort.assert_not_called()


@pytest.mark.parametrize('field,value', [
    ('_shelf_registration_stamp_ns', None),
    ('_shelf_registration_stamp_ns', 51_000_000_000),
    ('target_marker_odom', None), ('target_physical_column', 6),
])
def test_enabled_mission_never_dispatches_lift_with_missing_or_invalid_registration(field, value):
    node = mission()
    setattr(node, field, value)
    node._tick()
    node._manipulate.assert_not_called()
    assert node._abort.call_args.args[0].startswith('lift_first_registration_invalid:')
