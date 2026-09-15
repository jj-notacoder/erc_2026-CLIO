"""Planning and guard evidence cannot impersonate command completion."""

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution.mission_manager import MissionManager, String, encode_event


@pytest.mark.parametrize('event,fields', [
    ('pick_approach_planned', {'command': 'pick', 'solutions': [[.35, .6]],
                             'target': [.67, -.05, 1.58], 'grasp_depth_offset': .025}),
    ('trajectory_endpoint_missed', {'command': 'pick', 'phase': 'extraction',
                                    'torso_error': .01, 'arm_error': .04}),
    ('ik_ready', {'command': 'pick', 'post_retreat_compaction_required': True}),
    ('gripper_closed', {'command': 'pick', 'transport_lock_engaged': False,
                        'reason': 'force_overload', 'force_peaks': {'right': 8.1}}),
    ('adaptive_gripper_interrupted', {'reason': 'force_overload',
                                     'hold_published': True, 'measured_position': .0183}),
    ('fine_gripper_progress', {'stage': 'pressure_measurement',
                               'pressure_evidence': {'verified': True},
                               'retention_verified': False}),
    ('preclose_aperture_geometry_started', {'predicted_reference': .01825,
        'observed_open_master': .069}),
    ('preclose_aperture_geometry_verified', {'reference_basis': 'predicted_nominal_master',
        'drift_is_proved_displacement_envelope': False}),
    ('preclose_aperture_geometry_reused', {'observed_closed_master': .0183,
        'pressure_or_freshness_verified': False, 'motion_permit': False}),
    ('placement_torso_motion_admitted', {'command': 'place', 'verified': True,
        'motion': {'command_duration_ns': 8928571429, 'urdf_velocity_limit': .035}}),
    ('placement_torso_arrival_started', {'command': 'place', 'verified': False}),
    ('placement_torso_arrival_verified', {'command': 'place', 'verified': True,
        'accepted_samples': 6, 'elapsed_ros_ns': 250000000}),
    ('placement_torso_failed', {'command': 'place', 'verified': False,
        'reason': 'place_torso_arrival_deadline', 'recent_samples': [{'errors': [.1]}]}),
    ('carried_recovery', {'command': 'pick', 'retained_stop': True,
                          'recovery_succeeded': False}),
])
def test_diagnostic_payload_is_retained_without_replacing_command_result(event, fields):
    node = object.__new__(MissionManager)
    terminal = {'event': 'succeeded', 'command': 'look_books'}
    node.manip_event = terminal
    logged = []
    node._log = lambda event, **fields: logged.append((event, fields))
    message = String(data=encode_event(event, **fields))
    node._on_manipulation_status(message)
    assert node.manip_event is terminal
    assert logged == [('manipulation_event', {'payload': {'event': event, **fields}})]


@pytest.mark.parametrize('reason', [
    'Open gripper approach rejected: open_approach_book_intersection; link=gripper_left_base_link',
    'No payload-safe supported compact transport route',
])
def test_guard_failure_keeps_its_exact_reason_in_log_and_command_result(reason):
    node = object.__new__(MissionManager)
    node.manip_event = None
    logged = []
    node._log = lambda event, **fields: logged.append((event, fields))
    payload = {'event': 'failed', 'command': 'pick', 'reason': reason}
    node._on_manipulation_status(String(data=encode_event(**payload)))
    assert node.manip_event == payload
    assert logged == [('manipulation_event', {'payload': payload})]
