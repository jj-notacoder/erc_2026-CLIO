"""Launch-to-contact timing is separate from completion and simulation time."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip('rclpy')
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from ros_gz_interfaces.msg import Contact, Contacts
from erc_phase1_solution import mission_manager
from test_launch_timeout_override import PACKAGE_ROOT, _load_launch_module


TARGET = 'book_col_3_row_2_red'
UTC = '2026-09-11T01:00:00.000000+00:00'


def timing_node(**parameters):
    node = object.__new__(mission_manager.MissionManager)
    node.wall_started = 105.
    node.dry_run = False
    node.state = 'NAVIGATE_BIN'
    node.target_colour = 'red'
    node.target_book_model = TARGET
    node.bin_contact_confirmed = False
    node.logs = []
    node._log = lambda event, **fields: node.logs.append((event, fields))
    values = {'launch_origin_monotonic_seconds': 100.,
              'launch_origin_utc': UTC, 'launch_origin_basis': 'launcher_entry'}
    values.update(parameters)
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._initialize_trial_timing()
    return node


def contact_message(book=TARGET+'::book_base_link::book_collision', reverse=False):
    names = ['erc_collection_bin::collection_bin_base_link::bin_collision', book]
    if reverse:
        names.reverse()
    contact = Contact()
    contact.collision1.name, contact.collision2.name = names
    message = Contacts()
    message.contacts = [contact]
    message.header.stamp.sec = 42
    message.header.stamp.nanosec = 123
    return message


def test_delayed_initialization_contact_and_summary_have_distinct_intervals(monkeypatch, tmp_path):
    node = timing_node()
    monkeypatch.setattr(mission_manager, 'time', SimpleNamespace(monotonic=lambda: 120.))
    node._on_bin_contacts(contact_message())
    first = node.first_target_bin_contact.copy()
    assert first['receipt_monotonic_seconds'] == 120.
    assert first['producer_ros_time_ns'] == 42_000_000_123
    assert first['source_topic'] == '/bin_contacts'
    assert first['target_book_model'] == TARGET
    assert first['mission_state_at_receipt'] == 'NAVIGATE_BIN'
    assert not node.bin_contact_confirmed
    assert node.logs == []

    # A later placement-state report may confirm contact, but cannot move the
    # first receipt or claim that its timing alone proves placement.
    monkeypatch.setattr(mission_manager, 'time', SimpleNamespace(monotonic=lambda: 125.))
    node.state = 'WAIT_BIN_CONTACT'
    node._on_bin_contacts(contact_message(reverse=True))
    assert node.first_target_bin_contact == first
    assert node.bin_contact_confirmed

    node.trial_id, node.team_name, node.target_column = 'timing-test', 'TEAM', 2
    node.detected_row, node.target_physical_column = 1, 3
    node.column_confirmed = node.row_confirmed = True
    node.pick_attempts, node.nav_goals, node.collision_episodes = 1, 2, 0
    node.navigation_runs = []
    node.summary_path = tmp_path/'summary.json'
    monkeypatch.setattr(mission_manager, 'time', SimpleNamespace(monotonic=lambda: 130.))
    node._write_summary(True, '', log_event=False)
    summary = json.loads(node.summary_path.read_text())
    assert summary['elapsed_seconds'] == 25.
    assert summary['elapsed_seconds_basis'] == 'mission_initialization_to_summary_wall_monotonic'
    timing = summary['timing']
    assert timing['launch_to_first_target_bin_contact_wall_seconds'] == 20.
    assert timing['launch_to_first_target_bin_contact_status'] == 'recorded'
    assert timing['mission_initialization_to_summary_wall_seconds'] == 25.
    assert timing['first_target_bin_contact'] == json.loads(json.dumps(first))
    assert timing['placement_verified_by_contact_timer'] is False


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('book,expected', [
    (TARGET+'::book_base_link::book_collision', True),
    ('book_col_4_row_2_red::book_base_link::book_collision', False),
    ('book_col_3_row_3_red::book_base_link::book_collision', False),
    ('base_link_book_collision', False),
    ('erc_book_red::book_collision', False),
    ('prefix_'+TARGET+'::book_collision', False),
])
def test_timer_freezes_only_exact_named_target_bin_receipt(monkeypatch, book, expected, reverse):
    node = timing_node()
    monkeypatch.setattr(mission_manager, 'time', SimpleNamespace(monotonic=lambda: 120.))
    node._on_bin_contacts(contact_message(book, reverse))
    assert (node.first_target_bin_contact is not None) is expected
    assert not node.bin_contact_confirmed
    assert node.logs == []


def test_contact_before_identity_latch_does_not_start_contact_timer(monkeypatch):
    node = timing_node()
    node.target_book_model = None
    monkeypatch.setattr(mission_manager, 'time', SimpleNamespace(monotonic=lambda: 120.))
    node._on_bin_contacts(contact_message())
    assert node.first_target_bin_contact is None
    assert node._trial_timing(130.)['launch_to_first_target_bin_contact_status'] == 'pending_contact'


@pytest.mark.parametrize('parameters,reason', [
    ({'launch_origin_monotonic_seconds': -1.}, 'missing_or_invalid_monotonic_origin'),
    ({'launch_origin_monotonic_seconds': float('nan')}, 'missing_or_invalid_monotonic_origin'),
    ({'launch_origin_monotonic_seconds': float('inf')}, 'missing_or_invalid_monotonic_origin'),
    ({'launch_origin_monotonic_seconds': 'bad'}, 'missing_or_invalid_monotonic_origin'),
    ({'launch_origin_monotonic_seconds': 106.}, 'launch_origin_after_mission_initialization'),
    ({'launch_origin_basis': 'not_provided'}, 'missing_origin_basis'),
    ({'launch_origin_basis': ' '}, 'missing_origin_basis'),
    ({'launch_origin_utc': ''}, 'missing_or_invalid_origin_utc'),
    ({'launch_origin_utc': '2026-09-11T01:00:00'}, 'missing_or_invalid_origin_utc'),
    ({'launch_origin_utc': '2026-09-11T01:00:00+04:00'}, 'missing_or_invalid_origin_utc'),
])
def test_invalid_launch_origin_keeps_contact_evidence_without_scoring_interval(monkeypatch, parameters, reason):
    node = timing_node(**parameters)
    monkeypatch.setattr(mission_manager, 'time', SimpleNamespace(monotonic=lambda: 120.))
    node._on_bin_contacts(contact_message())
    result = node._trial_timing(130.)
    assert result['launch_origin']['unavailable_reason'] == reason
    assert result['launch_to_first_target_bin_contact_status'] == 'unavailable_launch_origin'
    assert result['launch_to_first_target_bin_contact_wall_seconds'] is None
    assert result['first_target_bin_contact']['receipt_monotonic_seconds'] == 120.


@pytest.mark.parametrize('dry_run', [False, True])
def test_missing_origin_fallback_and_dry_run_never_invent_launch_time(dry_run):
    node = object.__new__(mission_manager.MissionManager)
    node.wall_started = 105.
    node.dry_run = dry_run
    timing = node._trial_timing(130.)
    assert timing['launch_to_first_target_bin_contact_status'] == (
        'excluded_dry_run' if dry_run else 'unavailable_launch_origin')
    assert timing['launch_to_first_target_bin_contact_wall_seconds'] is None
    assert timing['mission_initialization_to_summary_wall_seconds'] == 25.


def test_dry_run_with_exact_contact_still_has_no_scoring_interval(monkeypatch):
    node = timing_node()
    node.dry_run = True
    monkeypatch.setattr(mission_manager, 'time', SimpleNamespace(monotonic=lambda: 120.))
    message = contact_message()
    message.header.stamp.sec = message.header.stamp.nanosec = 0
    node._on_bin_contacts(message)
    timing = node._trial_timing(130.)
    assert timing['first_target_bin_contact']['producer_ros_time_ns'] is None
    assert timing['launch_to_first_target_bin_contact_status'] == 'excluded_dry_run'
    assert timing['launch_to_first_target_bin_contact_wall_seconds'] is None


def test_backward_receipt_clock_is_exposed_without_negative_interval(monkeypatch):
    node = timing_node()
    monkeypatch.setattr(mission_manager, 'time', SimpleNamespace(monotonic=lambda: 99.))
    node._on_bin_contacts(contact_message())
    timing = node._trial_timing(130.)
    assert timing['launch_to_first_target_bin_contact_status'] == 'invalid_monotonic_order'
    assert timing['launch_to_first_target_bin_contact_wall_seconds'] is None


def test_launch_origin_captured_before_ros_imports(monkeypatch):
    events = []
    original_import = builtins.__import__

    def observe_import(name, *args, **kwargs):
        if name == 'time':
            return SimpleNamespace(monotonic=lambda: events.append('origin_capture') or 100.)
        if name.startswith(('launch', 'ament_index_python')):
            events.append('ros_import')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', observe_import)
    module = _load_launch_module()
    assert events.index('origin_capture') < events.index('ros_import')
    assert module._LAUNCH_ORIGIN_MONOTONIC_SECONDS == 100.
    assert module._LAUNCH_ORIGIN_BASIS == 'solution.launch.py_module_entry_before_ros_imports'


@pytest.mark.parametrize('override', [False, True])
def test_launch_origin_reaches_only_delayed_mission_node(monkeypatch, tmp_path, override):
    module = _load_launch_module()
    monkeypatch.setattr(module, '_LAUNCH_ORIGIN_MONOTONIC_SECONDS', 100.)
    monkeypatch.setattr(module, 'get_package_share_directory', lambda _: str(PACKAGE_ROOT))
    monkeypatch.setattr(module, '_writable_output', lambda name: str(tmp_path/name))
    context = LaunchContext()
    context.launch_configurations.update({'shelf_column_number': '2', 'book_colour': 'red'})
    if override:
        context.launch_configurations.update({
            'launch_origin_monotonic_seconds': '90.',
            'launch_origin_utc': UTC, 'launch_origin_basis': 'external_launcher'})
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    actions = module._launch_nodes(context)
    mission = actions[3].actions[0]
    for index, node in enumerate([*actions[:3], mission]):
        node._perform_substitutions(context)
        path, is_file = node._Node__expanded_parameter_arguments[-1]
        assert is_file
        parameters = yaml.safe_load(Path(path).read_text())['/**']['ros__parameters']
        if index < 3:
            assert not any(key.startswith('launch_origin_') for key in parameters)
        else:
            assert parameters['launch_origin_monotonic_seconds'] == (90. if override else 100.)
            assert parameters['launch_origin_basis'] == (
                'external_launcher' if override else module._LAUNCH_ORIGIN_BASIS)
            assert parameters['launch_origin_utc'] == (UTC if override else module._LAUNCH_ORIGIN_UTC)
