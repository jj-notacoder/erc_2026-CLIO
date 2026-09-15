"""Focused checks for randomized shelf-cell identity and dry-run gates."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest


pytest.importorskip('rclpy')

from geometry_msgs.msg import Point32  # noqa: E402
from sensor_msgs.msg import ChannelFloat32, PointCloud  # noqa: E402
from ros_gz_interfaces.msg import Contact, Contacts  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from erc_phase1_solution import mission_manager  # noqa: E402


def _marker_cloud(digits_by_physical_column, physical_columns=None):
    if physical_columns is None:
        physical_columns = range(1, 6)
    cloud = PointCloud()
    digit_channel = ChannelFloat32(name='digit')
    for physical_column in physical_columns:
        # simulation.launch.py places model column 1 at +2 m (robot-left),
        # then decrements y by its exact 1 m shelf-column spacing.
        lateral = 3.0 - float(physical_column)
        cloud.points.append(Point32(x=3.0, y=lateral, z=2.26))
        digit_channel.values.append(
            float(digits_by_physical_column[physical_column - 1])
        )
    cloud.channels = [digit_channel]
    return cloud


@pytest.mark.parametrize(
    ('target_digit', 'expected_physical_column'),
    ((5, 1), (1, 2), (4, 3), (2, 4), (3, 5)),
)
def test_randomized_marker_digit_maps_to_physical_book_model_column(
    target_digit,
    expected_physical_column,
):
    node = object.__new__(mission_manager.MissionManager)
    node.target_column = target_digit
    node.start_pose = (0.0, 0.0, 0.0)
    node.robot_pose = node.start_pose
    node._point_to_odom = lambda _header, point: np.asarray(
        [point.x, point.y, point.z],
        dtype=float,
    )
    digits = [5, 1, 4, 2, 3]

    _, normal, physical_column = node._shelf_geometry(_marker_cloud(digits))

    assert normal == pytest.approx([-1.0, 0.0])
    assert physical_column == expected_physical_column


def test_physical_column_mapping_does_not_require_all_five_markers():
    node = object.__new__(mission_manager.MissionManager)
    node.target_column = 3
    node.start_pose = (0.0, 0.0, 0.0)
    node.robot_pose = node.start_pose
    node._point_to_odom = lambda _header, point: np.asarray(
        [point.x, point.y, point.z],
        dtype=float,
    )

    _, _, physical_column = node._shelf_geometry(
        _marker_cloud([5, 1, 4, 2, 3], physical_columns=(1, 3, 5))
    )

    assert physical_column == 5


def _identity_node():
    node = object.__new__(mission_manager.MissionManager)
    node.state = 'PICK'
    node.finished = False
    node.target_colour = 'red'
    node.target_physical_column = 3
    node.detected_row = 1
    node.target_book_model = None
    node.manip_event = None
    node.identity_logs = []
    node._log = lambda event, **fields: node.identity_logs.append((event, fields))
    return node


def _latched_message(model=None):
    message = String()
    fields = {} if model is None else {'model': model}
    message.data = mission_manager.encode_event('target_book_latched', **fields)
    return message


def test_identity_requires_the_marker_selected_physical_column_and_active_row():
    node = _identity_node()

    node._on_manipulation_status(
        _latched_message('book_col_4_row_2_red')
    )
    node._on_manipulation_status(
        _latched_message('book_col_3_row_3_red')
    )
    node._on_manipulation_status(_latched_message())

    assert node.target_book_model is None
    rejected = [
        fields
        for event, fields in node.identity_logs
        if event == 'target_book_identity_rejected'
    ]
    assert [item['reason'] for item in rejected] == [
        'unexpected_model',
        'unexpected_model',
        'scoped_model_unavailable',
    ]
    assert all(
        item['expected_model'] == 'book_col_3_row_2_red'
        for item in rejected
    )

    node._on_manipulation_status(
        _latched_message('book_col_3_row_2_red')
    )

    assert node.target_book_model == 'book_col_3_row_2_red'
    assert node._target_identity_confirmed()


def test_expected_identity_uses_full_shelf_row_numbering_without_off_by_one():
    node = _identity_node()

    assert node._expected_target_book_model() == 'book_col_3_row_2_red'
    node.detected_row = 4
    assert node._expected_target_book_model() == 'book_col_3_row_5_red'
    node.target_book_model = 'book_col_3_row_4_red'
    assert not node._target_identity_confirmed()


def test_summary_distinguishes_expected_model_from_a_wrong_scoped_model(tmp_path):
    node = _identity_node()
    node.trial_id = 'identity-test'
    node.team_name = 'TEST_TEAM'
    node.target_column = 2
    node.dry_run = False
    node.wall_started = time.monotonic()
    node.column_confirmed = True
    node.row_confirmed = True
    node.pick_attempts = 1
    node.nav_goals = 0
    node.navigation_runs = []
    node.collision_episodes = 0
    node.bin_contact_confirmed = False
    node.summary_path = tmp_path / 'summary.json'
    node.target_book_model = 'book_col_4_row_2_red'

    node._write_summary(False, 'wrong_identity', log_event=False)

    summary = json.loads(node.summary_path.read_text(encoding='utf-8'))
    assert summary['expected_target_book_model'] == 'book_col_3_row_2_red'
    assert summary['target_book_model'] == 'book_col_4_row_2_red'
    assert not summary['target_identity_confirmed']
    assert not summary['dry_run']


def test_dry_run_pick_bypasses_physical_identity_without_fabricating_it():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'PICK'
    node.dry_run = True
    node.target_book_model = None
    node.robot_pose = (1.0, 2.0, 0.1)
    node.shelf_normal = np.asarray([1.0, 0.0])
    node.carried_shelf_retreat = 0.35
    node._manip_succeeded = lambda command: command == 'pick'
    node._manip_failed = lambda: False
    node._elapsed_state = lambda: 0.0
    calls = []
    node._perception_mode = lambda mode: calls.append(('mode', mode))
    node._navigate = lambda x, y, yaw, purpose, **kwargs: calls.append(
        ('navigate', x, y, yaw, purpose, kwargs)
    )
    node._set_state = lambda state, **fields: calls.append(
        ('state', state, fields)
    )
    node._abort = lambda reason: calls.append(('abort', reason))

    node._tick()

    assert node.target_book_model is None
    assert [call[0] for call in calls] == ['mode', 'navigate', 'state']
    assert calls[0] == ('mode', 'idle')
    assert calls[-1] == ('state', 'CLEAR_SHELF_WITH_BOOK', {})


def test_dry_run_completes_without_fabricating_bin_contact():
    node = object.__new__(mission_manager.MissionManager)
    node.finished = False
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.0
    node.state = 'WAIT_BIN_CONTACT'
    node.dry_run = True
    node.bin_contact_confirmed = False
    calls = []
    node._complete = lambda: calls.append('complete')
    node._abort = lambda reason: calls.append(('abort', reason))

    node._tick()

    assert calls == ['complete']
    assert not node.bin_contact_confirmed


def _bin_contact_message(book_name, reverse=False):
    contact = Contact()
    names = ('erc_collection_bin::collection_bin_base_link::bin_collision', book_name)
    if reverse:
        names = tuple(reversed(names))
    contact.collision1.name, contact.collision2.name = names
    message = Contacts()
    message.contacts = [contact]
    return message


@pytest.mark.parametrize('state', ['PLACE', 'WAIT_BIN_CONTACT'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('book_name,expected', [
    ('book_col_3_row_2_red::book_base_link::book_collision', True),
    ('book_col_4_row_2_red::book_base_link::book_collision', False),
    ('book_col_3_row_3_red::book_base_link::book_collision', False),
    ('book_col_3_row_2_blue::book_base_link::book_collision', False),
    ('base_link_book_collision', False),
    ('book_base_link::base_link_book_collision', False),
    ('erc_book_red::book_collision', False),
])
def test_bin_callback_only_confirms_exact_latched_book(state, reverse, book_name, expected):
    node = _identity_node()
    node.state = state
    node.target_book_model = 'book_col_3_row_2_red'
    node.bin_contact_confirmed = False

    node._on_bin_contacts(_bin_contact_message(book_name, reverse))

    assert node.bin_contact_confirmed is expected
    assert len(node.identity_logs) == int(expected)
    if expected:
        assert node.identity_logs[0][0] == 'bin_contact'


def test_bin_callback_requires_latched_identity_and_placement_state():
    node = _identity_node()
    node.bin_contact_confirmed = False
    message = _bin_contact_message('book_col_3_row_2_red::book_base_link::book_collision')
    node.state = 'PLACE'
    node._on_bin_contacts(message)
    assert not node.bin_contact_confirmed
    node.target_book_model = 'book_col_3_row_2_red'
    node.state = 'PICK'
    node._on_bin_contacts(message)
    assert not node.bin_contact_confirmed
    assert not node.identity_logs


def test_dry_run_summary_keeps_qualifier_without_anonymous_delivery(tmp_path):
    node = _identity_node()
    node.state = 'WAIT_BIN_CONTACT'
    node.dry_run = True
    node.bin_contact_confirmed = False
    node.trial_id = 'dry-identity-test'
    node.team_name = 'TEST_TEAM'
    node.target_column = 2
    node.wall_started = time.monotonic()
    node.trial_timeout = 1200.
    node.column_confirmed = node.row_confirmed = True
    node.pick_attempts = 1
    node.nav_goals = node.collision_episodes = 0
    node.navigation_runs = []
    node.summary_path = tmp_path / 'summary.json'
    completed = []
    node._complete = lambda: completed.append(True)
    node._on_bin_contacts(_bin_contact_message('base_link_book_collision'))

    node._tick()
    node._write_summary(True, '', log_event=False)

    summary = json.loads(node.summary_path.read_text(encoding='utf-8'))
    assert completed == [True]
    assert summary['success'] and summary['dry_run']
    assert not summary['bin_contact_confirmed']
    assert not summary['target_identity_confirmed']
    assert summary['target_book_model'] is None
