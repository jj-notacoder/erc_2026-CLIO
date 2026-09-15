"""Installed selected settings are complete and portable; node defaults remain conservative."""
import ast
import hashlib
from pathlib import Path

import pytest
import yaml

from installed_profile_test_support import (
    restore_installed_profile_bytes, selected_profile_record,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'config/collision_quality.yaml'


def test_installed_profile_matches_complete_selected_override_mapping():
    selected = yaml.safe_load(CONFIG.read_bytes())
    actual = {node: value['ros__parameters'] for node, value in selected.items()}
    assert actual == selected_profile_record()['selected_overrides']
    assert all(set(value) == {'ros__parameters'} for value in selected.values())
    assert actual['erc_manipulation']['withdrawal_speed_scale'] == 3.0
    assert actual['erc_manipulation']['supported_compact_minimum_segment_seconds'] == .175
    assert actual['erc_manipulation']['lift_first_extraction_lift_m'] == .020
    assert actual['erc_manipulation']['stock_gripper_close_target_position'] == .0182
    assert actual['erc_manipulation']['placement_transport_speed_scale'] == 3.0
    assert actual['erc_navigation']['normal_linear_convergence_enabled'] is True
    assert actual['erc_manipulation']['additional_arm_time_scale'] == 2.0
    assert actual['erc_manipulation']['settled_place_torso_skip_enabled'] is True
    assert actual['erc_manipulation']['settled_place_torso_retry_enabled'] is True
    assert actual['erc_manipulation']['pickup_parallel_geometry_enabled'] is True
    assert actual['erc_manipulation']['empty_pickup_parallel_geometry_enabled'] is True
    assert actual['erc_manipulation']['geometry_process_workers'] == 8
    assert actual['erc_manipulation']['pickup_post_retreat_parallel_geometry_enabled'] is True
    assert actual['erc_manipulation']['empty_pickup_setup_retiming_enabled'] is True
    assert actual['erc_manipulation']['place_carried_volume_parallel_enabled'] is True
    assert actual['erc_manipulation']['place_finish_keep_torso_height_enabled'] is True
    assert actual['erc_navigation']['max_angular_speed'] == .70
    assert actual['erc_navigation']['normal_translation_gain'] == 1.8
    assert actual['erc_navigation']['normal_yaw_gain'] == 1.8
    assert actual['erc_manipulation']['head_return_overlap_enabled'] is False
    assert actual['erc_mission_manager']['head_return_overlap_enabled'] is False
    assert actual['erc_manipulation']['loaded_place_speed_scale_cap'] == 2.0
    assert actual['erc_manipulation']['initial_stow_serial_timing_enabled'] is True
    assert actual['erc_manipulation']['place_transition_stop_enabled'] is True
    assert actual['erc_manipulation']['release_only_clearance_timing_enabled'] is True
    assert actual['erc_manipulation']['bin_clearance_timing_enabled'] is False
    assert actual['erc_manipulation']['withdrawal_half_timing_enabled'] is True
    assert actual['erc_manipulation']['withdrawal_quarter_timing_enabled'] is True
    assert actual['erc_manipulation']['place_finish_at_release_enabled'] is True
    assert actual['erc_manipulation']['release_only_place_planning_enabled'] is True
    assert actual['erc_mission_manager'].get('marker_search_first_probe_enabled', False) is True
    assert actual['erc_manipulation'].get('completed_head_hold_recheck_enabled', False) is True
    assert actual['erc_manipulation'].get('compact_extension_half_timing_enabled', False) is True
    assert actual['erc_mission_manager']['place_finish_at_release_enabled'] is True
    assert actual['erc_mission_manager']['mission_raw_contacts_enabled'] is True
    assert actual['erc_manipulation']['empty_torso_planning_overlap_enabled'] is True
    assert actual['erc_manipulation']['empty_head_timing_enabled'] is False
    assert actual['erc_mission_manager']['empty_head_timing_enabled'] is False
    assert actual['erc_manipulation']['place_parallel_geometry_enabled'] is True


def test_complete_old_profile_is_restored_for_historical_assertions():
    restored = restore_installed_profile_bytes(CONFIG.read_bytes())
    assert hashlib.sha256(restored).hexdigest() == '98ef332abbec93aa4af1d9874cf01c2308a8dc8575a3b22ab790843ebebe4bea'
    actual = yaml.safe_load(restored)
    assert actual['erc_manipulation']['ros__parameters']['lift_first_extraction_lift_m'] == .005
    assert 'withdrawal_speed_scale' not in actual['erc_manipulation']['ros__parameters']


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'unrelated'])
def test_profile_inverse_rejects_unlisted_config_changes(damage):
    raw = CONFIG.read_bytes()
    token = b'    withdrawal_speed_scale: 3.0\n'
    assert raw.count(token) == 1
    if damage == 'missing': raw = raw.replace(token, b'', 1)
    elif damage == 'duplicate': raw += token
    else: raw += b'\n# unlisted config mutation\n'
    with pytest.raises(AssertionError): restore_installed_profile_bytes(raw)


def test_base_config_remains_exact():
    assert hashlib.sha256((ROOT/'config/solution.yaml').read_bytes()).hexdigest() == '3ab897b19c11995310bfbad442abc1ab8d8574e08cc866dcfd373f32b50e4f0d'


def test_actual_declarations_preserve_conservative_defaults():
    def defaults(file, owner):
        tree = ast.parse((ROOT/'erc_phase1_solution'/file).read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_declare_parameters')
        dictionaries = [n for n in ast.walk(method) if isinstance(n, ast.Dict)]
        values = next(n for n in dictionaries if any(isinstance(k,ast.Constant) and k.value == 'dry_run' for k in n.keys))
        keys = {'lift_first_extraction_lift_m', 'withdrawal_speed_scale',
                'supported_compact_minimum_segment_seconds', 'placement_transport_speed_scale',
                'normal_linear_convergence_enabled', 'normal_translation_gain', 'normal_yaw_gain', 'additional_arm_time_scale', 'place_parallel_geometry_enabled', 'settled_place_torso_skip_enabled', 'pickup_parallel_geometry_enabled', 'geometry_process_workers', 'loaded_place_speed_scale_cap', 'pickup_post_retreat_parallel_geometry_enabled', 'empty_pickup_setup_retiming_enabled', 'place_carried_volume_parallel_enabled', 'head_return_overlap_enabled', 'place_finish_keep_torso_height_enabled', 'max_angular_speed', 'empty_pickup_parallel_geometry_enabled', 'settled_place_torso_retry_enabled', 'bin_clearance_timing_enabled', 'empty_torso_planning_overlap_enabled', 'empty_head_timing_enabled', 'withdrawal_half_timing_enabled', 'withdrawal_quarter_timing_enabled', 'place_finish_at_release_enabled', 'release_only_place_planning_enabled', 'marker_search_first_probe_enabled', 'mission_raw_contacts_enabled', 'completed_head_hold_recheck_enabled', 'compact_extension_half_timing_enabled', 'place_transition_stop_enabled', 'initial_stow_serial_timing_enabled', 'release_only_clearance_timing_enabled'}
        return {k.value: ast.literal_eval(v) for k,v in zip(values.keys,values.values)
                if isinstance(k,ast.Constant) and k.value in keys}
    manipulation = defaults('manipulation_node.py', 'ManipulationNode')
    assert manipulation['lift_first_extraction_lift_m'] == .005
    assert manipulation['withdrawal_speed_scale'] == 1.0
    assert manipulation['supported_compact_minimum_segment_seconds'] == .35
    assert manipulation['placement_transport_speed_scale'] == 1.0
    assert manipulation['additional_arm_time_scale'] == 1.0
    assert manipulation['place_parallel_geometry_enabled'] is False
    assert manipulation['settled_place_torso_skip_enabled'] is False
    assert manipulation['settled_place_torso_retry_enabled'] is False
    assert manipulation['pickup_parallel_geometry_enabled'] is False
    assert manipulation['empty_pickup_parallel_geometry_enabled'] is False
    assert manipulation['pickup_post_retreat_parallel_geometry_enabled'] is False
    assert manipulation['empty_pickup_setup_retiming_enabled'] is False
    assert manipulation['place_carried_volume_parallel_enabled'] is False
    assert manipulation['head_return_overlap_enabled'] is False
    assert manipulation['place_finish_keep_torso_height_enabled'] is False
    assert defaults('mission_manager.py','MissionManager')['head_return_overlap_enabled'] is False
    assert manipulation['geometry_process_workers'] == 4
    assert manipulation['loaded_place_speed_scale_cap'] == 3.0
    assert manipulation['bin_clearance_timing_enabled'] is False
    assert manipulation['empty_torso_planning_overlap_enabled'] is False
    assert manipulation['empty_head_timing_enabled'] is False
    assert manipulation['withdrawal_half_timing_enabled'] is False
    assert manipulation['withdrawal_quarter_timing_enabled'] is False
    assert manipulation['place_finish_at_release_enabled'] is False
    assert manipulation['release_only_place_planning_enabled'] is False
    assert manipulation['completed_head_hold_recheck_enabled'] is False
    assert manipulation['compact_extension_half_timing_enabled'] is False
    assert manipulation['initial_stow_serial_timing_enabled'] is False
    assert manipulation['place_transition_stop_enabled'] is False
    assert manipulation['release_only_clearance_timing_enabled'] is False
    assert defaults('mission_manager.py','MissionManager')['marker_search_first_probe_enabled'] is False
    assert defaults('mission_manager.py','MissionManager')['place_finish_at_release_enabled'] is False
    assert defaults('mission_manager.py','MissionManager')['mission_raw_contacts_enabled'] is False
    assert defaults('mission_manager.py','MissionManager')['empty_head_timing_enabled'] is False
    navigation = defaults('navigation_node.py', 'NavigationNode')
    assert navigation['normal_linear_convergence_enabled'] is False
    assert navigation['normal_translation_gain'] == .85
    assert navigation['normal_yaw_gain'] == 1.35
    assert navigation['max_angular_speed'] == .55
