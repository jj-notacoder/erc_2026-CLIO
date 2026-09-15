"""Focused checks for the zero-command mid-route settle certificate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / 'erc_phase1_solution' / (
    'seed101_outward_halfmillimeter_midroute_settle_certificate.py'
)
SPEC = importlib.util.spec_from_file_location('half_mm_midroute_settle', MODULE)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_current_full_double_paused_checkpoint_is_bound():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT['world_paused'] is True
    assert certificate.CHECKPOINT['world_sim_time_s'] == 346.66
    assert certificate.CHECKPOINT['world_iterations'] == 173330
    assert certificate.CHECKPOINT_BOOK_POSITION_WORLD_M == (
        2.891342768706694,
        -0.15306102947940783,
        1.5775073067363485,
    )
    assert certificate.CHECKPOINT_BOOK_QUATERNION_XYZW == (
        -0.0037302502014055204,
        0.7099740343741349,
        0.0036603078421359346,
        0.704208461958851,
    )
    assert certificate.CHECKPOINT_BASE_WORLD_POSITION_M == (
        2.002648039946845,
        -0.1495205290917588,
        -1.6957408982813647e-07,
    )
    assert certificate.CHECKPOINT_BASE_QUATERNION_XYZW == (
        9.978588766464317e-08,
        2.3035211189417835e-07,
        -0.38256263060418283,
        0.923929560986737,
    )
    assert certificate.CHECKPOINT_BASE_WORLD_YAW_RAD == -0.7851366607105121
    assert certificate.CHECKPOINT_LEFT_Q8 == (
        0.349999950536834,
        -0.30949603866899605,
        0.6178645057946034,
        0.08206232481188921,
        -1.4826272057601544,
        0.17374047013531607,
        1.0387571269500129,
        0.21480420336300401,
    )
    assert certificate.CHECKPOINT_GRIPPER_MASTER_M == 0.029008131671675274
    assert metrics['checkpoint_reference_position_error_m'] < 4e-06
    assert metrics['checkpoint_reference_rotation_error_rad'] < 3e-06


def test_route_is_zero_command_and_settle_limits_are_exact():
    execution = certificate.EXECUTION_POLICY

    assert certificate.STAGE == 'outward-halfmillimeter-midroute-settle'
    assert len(certificate.ROUTE_Q8) == 1
    assert certificate.ROUTE_Q8 == (certificate.CHECKPOINT_LEFT_Q8,)
    assert certificate.COMMAND_Q8 == ()
    assert certificate.ROUTE[0]['world_delta_m'] == [0.0, 0.0, 0.0]
    assert certificate.SETTLE_DURATION_S == 0.3
    assert certificate.SETTLE_STABILITY_TRANSLATION_LIMIT_M == 0.0001
    assert certificate.SETTLE_STABILITY_ROTATION_LIMIT_RAD == 0.0002
    assert execution['minimum_resume_duration_s'] == 0.3
    assert execution['maximum_resume_duration_s'] == 0.3
    assert execution['trajectory_command_rows'] == 0
    for key in (
        'base_command_count',
        'gazebo_entity_pose_command_count',
        'gripper_command_count',
        'head_command_count',
        'left_arm_command_count',
        'right_arm_command_count',
    ):
        assert execution[key] == 0
    assert execution['world_resume_allowed_for_settle_only'] is True
    assert execution['repause_after_settle'] is True
    assert execution['gazebo_entity_pose_mutation_allowed'] is False
    assert execution['gripper_squeeze_allowed'] is False
    assert execution['automatic_next_stage_allowed'] is False


def test_orientation_debounce_and_absolute_gates_fail_closed():
    assert certificate.CHECKPOINT_POLICY == {
        'maximum_absolute_rotation_to_stable_reference_rad': 0.0009,
        'maximum_absolute_yaw_to_stable_reference_rad': 0.00065,
        'maximum_book_position_error_m': 0.0001,
        'maximum_book_rotation_error_rad': 0.0003,
    }
    assert certificate.CONTINUOUS_POLICY == {
        'maximum_absolute_rotation_to_stable_reference_rad': 0.0012,
        'maximum_absolute_yaw_to_stable_reference_rad': 0.0009,
        'maximum_passive_scene_rotation_from_reference_rad': 0.0002,
        'maximum_passive_scene_translation_from_reference_m': 0.0001,
        'maximum_relative_rotation_from_checkpoint_rad': 0.00085,
        'maximum_stationary_translation_from_checkpoint_m': 0.0001,
        'require_exact_target_bilateral_pressure': True,
        'require_non_target_scene_stable': True,
    }
    assert certificate.RELATIVE_YAW_DEBOUNCE_POLICY == {
        'consecutive_samples_required': 2,
        'distinct_pose_generations_required': True,
        'reset_below_soft_limit': True,
        'relative_yaw_immediate_hard_limit_rad': 0.00085,
        'relative_yaw_soft_limit_rad': 0.00045,
        'soft_limit_comparison': 'strictly_greater_than',
    }
    assert certificate.FINAL_DWELL_POLICY == {
        'duration_s': 0.3,
        'floor_signed_max_m': 0.00025,
        'floor_signed_min_m': -0.0001,
        'maximum_absolute_rotation_to_stable_reference_rad': 0.0009,
        'maximum_absolute_yaw_to_stable_reference_rad': 0.00065,
        'maximum_passive_scene_rotation_from_reference_rad': 0.0002,
        'maximum_passive_scene_translation_from_reference_m': 0.0001,
        'maximum_rotation_change_rad': 0.0002,
        'maximum_translation_change_m': 0.0001,
        'repause_after_dwell': True,
        'require_exact_target_bilateral_pressure': True,
        'require_non_target_scene_stable': True,
    }
    assert certificate.FLOOR_SIGNED_RANGE_M == (-0.0001, 0.00025)
    evidence = certificate.CHECKPOINT_STABLE_REFERENCE_EVIDENCE
    assert evidence['current_exact_absolute_rotation_rad'] < 0.0009
    assert evidence['current_exact_absolute_yaw_component_rad'] < 0.00065
    assert evidence['current_exact_floor_signed_distance_m'] < 0.000001


def test_pressure_scene_geometry_digests_and_inert_imports_are_bound():
    pressure = certificate.PRESSURE_POLICY
    audit = certificate.AUDIT

    assert certificate.MINIMUM_FORCE_N == (
        0.919798447906543,
        0.8461851825324634,
    )
    assert pressure['current_paused_left_target_contact'] is True
    assert pressure['current_paused_right_target_contact'] is True
    assert pressure['current_paused_left_force_n'] > certificate.MINIMUM_FORCE_N[0]
    assert pressure['current_paused_right_force_n'] > certificate.MINIMUM_FORCE_N[1]
    assert pressure[
        'require_fresh_bilateral_exact_target_pressure_after_resume'
    ] is True
    assert len(certificate.PASSIVE_SCENE_REFERENCE) == 22
    assert certificate.passive_scene_digest() == (
        certificate.PASSIVE_SCENE_REFERENCE_SHA256
    )
    assert len(certificate.ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP) == 8
    assert len(certificate.ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP) == 8
    assert audit['collision_free'] is True
    assert audit['zero_command_state_only'] is True
    assert audit['attached_corners_recomputed_from_current_exact_dynamic_pose'] is True
    assert audit['inherited_minimum_nonadjacent_robot_aabb_clearance_m'] > 0.002
    assert certificate.source_json_digest() == (
        certificate.CERTIFICATE_SOURCE_JSON_SHA256
    )
    assert certificate.semantic_digest() == certificate.CERTIFICATE_SEMANTIC_DIGEST
    assert certificate.route_q8_digest() == certificate.ROUTE_Q8_SHA256

    tree = ast.parse(MODULE.read_text(encoding='utf-8'))
    roots = {
        alias.name.split('.')[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots.update(
        (node.module or '').split('.')[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != '__future__'
    )
    assert roots <= {'hashlib', 'json', 'math', 'typing'}
    assert not ({'rclpy', 'subprocess', 'socket'} & roots)
