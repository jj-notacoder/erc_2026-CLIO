"""Focused offline checks for the exact-current 0.5 mm outward certificate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / 'erc_phase1_solution' / (
    'seed101_outward_halfmillimeter_certificate.py'
)
SPEC = importlib.util.spec_from_file_location('outward_halfmillimeter', MODULE)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_exact_paused_checkpoint_and_route_are_bound():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT['world_paused'] is True
    assert certificate.CHECKPOINT['world_sim_time_s'] == pytest.approx(345.362)
    assert certificate.CHECKPOINT['world_iterations'] == 172681
    assert certificate.CHECKPOINT_BOOK_POSITION_WORLD_M == pytest.approx((
        2.891381953941669,
        -0.15307094751992217,
        1.5775159685960052,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_WORLD_XYYAW == pytest.approx((
        2.0026505847001035,
        -0.14951724023818858,
        -0.7851469451686999,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_GRIPPER_MASTER_M == pytest.approx(
        0.029008139290647626, abs=1e-15
    )
    assert len(certificate.ROUTE_Q8) == 2
    assert certificate.ROUTE_Q8[0] == certificate.CHECKPOINT_LEFT_Q8
    assert certificate.ROUTE[1]['phase'] == 'shelf_outward'
    assert certificate.ROUTE[1]['minimum_duration_s'] == 2.5
    assert certificate.PLANNED_WORLD_DELTA_M == (-0.0005, 0.0, 0.0)
    assert metrics['route_rows'] == 2
    assert metrics['command_rows'] == 1


def test_endpoint_is_exact_fixed_orientation_ik_not_a_joint_midpoint():
    expected_endpoint = (
        0.34999997571431013,
        -0.30930585016382295,
        0.6177154645826406,
        0.0821239227235539,
        -1.484092155685582,
        0.17384098428524442,
        1.0401021615240358,
        0.21480741731946937,
    )

    assert certificate.ENDPOINT_Q8 == pytest.approx(expected_endpoint, abs=1e-15)
    assert certificate.AUDIT['endpoint_is_exact_ik_target'] is True
    assert certificate.AUDIT['joint_midpoint_used'] is False
    assert certificate.AUDIT['target_fk_position_error_m'] < 1e-12
    assert certificate.AUDIT['target_fk_orientation_error_rad'] < 1e-12
    assert certificate.AUDIT['dense_joint_samples'] == 5
    assert certificate.AUDIT['maximum_dense_joint_increment_rad'] <= 0.0005
    assert certificate.EXECUTION_POLICY['fixed_hand_orientation'] is True
    assert certificate.EXECUTION_POLICY['torso_fixed'] is True
    assert certificate.EXECUTION_POLICY['joint_midpoint_allowed'] is False


def test_reviewer_gates_and_pressure_fail_closed_values_are_exact():
    assert certificate.CHECKPOINT_POLICY == {
        'maximum_absolute_rotation_to_stable_reference_rad': 0.0008,
        'maximum_absolute_yaw_to_stable_reference_rad': 0.0006,
        'maximum_book_position_error_m': 0.0001,
        'maximum_book_rotation_error_rad': 0.0003,
    }
    assert certificate.CONTINUOUS_POLICY == {
        'maximum_absolute_rotation_to_stable_reference_rad': 0.0012,
        'maximum_absolute_world_y_motion_m': 0.00007,
        'maximum_absolute_yaw_to_stable_reference_rad': 0.0009,
        'maximum_book_center_outward_progress_m': 0.00065,
        'maximum_cross_track_motion_m': 0.0001,
        'maximum_relative_rotation_from_start_rad': 0.00085,
        'maximum_relative_yaw_from_start_rad': 0.00045,
        'minimum_book_center_outward_progress_m': -0.0001,
    }
    assert certificate.ENDPOINT_POLICY == {
        'floor_signed_max_m': 0.00025,
        'floor_signed_min_m': -0.0001,
        'maximum_absolute_rotation_to_stable_reference_rad': 0.0012,
        'maximum_absolute_world_y_motion_m': 0.00007,
        'maximum_absolute_yaw_to_stable_reference_rad': 0.0009,
        'maximum_book_center_outward_progress_m': 0.00065,
        'maximum_book_from_hand_translation_mismatch_m': 0.0002,
        'maximum_cross_track_motion_m': 0.0001,
        'maximum_incremental_rotation_rad': 0.0008,
        'maximum_incremental_yaw_rad': 0.00045,
        'minimum_book_center_outward_progress_m': 0.0003,
        'minimum_deepest_extent_outward_progress_m': 0.00025,
        'pause_after_endpoint': True,
    }
    assert certificate.FINAL_DWELL_POLICY == {
        'duration_s': 0.3,
        'maximum_absolute_rotation_to_stable_reference_rad': 0.0009,
        'maximum_absolute_yaw_to_stable_reference_rad': 0.00065,
        'maximum_relative_rotation_from_start_rad': 0.0006,
        'maximum_relative_yaw_from_start_rad': 0.00035,
        'maximum_rotation_change_rad': 0.000075,
        'maximum_rotation_growth_from_immediate_endpoint_rad': 0.00005,
        'maximum_translation_change_m': 0.00005,
        'repause_after_dwell': True,
    }
    assert certificate.MINIMUM_FORCE_N == pytest.approx((
        0.919798447906543,
        0.905239502936712,
    ), abs=1e-15)
    assert certificate.PRESSURE_POLICY[
        'minimum_post_to_pre_force_ratio_each_side'
    ] == 0.6
    assert certificate.PRESSURE_POLICY[
        'maximum_left_right_balance_change_fraction'
    ] == 0.3
    assert certificate.PRESSURE_POLICY['gripper_squeeze_allowed'] is False
    assert certificate.PRESSURE_POLICY['automatic_next_rung_allowed'] is False


def test_current_payload_geometry_collision_audit_and_inert_imports_are_bound():
    audit = certificate.AUDIT

    assert len(certificate.ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP) == 8
    assert len(certificate.ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP) == 8
    assert certificate.ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP[0] == pytest.approx((
        0.06545561248255072,
        0.014804266438389248,
        -0.11561228633125106,
    ), abs=1e-15)
    assert audit['collision_free'] is True
    assert audit['maximum_attached_corner_reconstruction_error_m'] < 2e-15
    assert audit['minimum_nonadjacent_robot_aabb_clearance_m'] > 0.002
    assert audit['minimum_15mm_padded_payload_robot_aabb_clearance_m'] > 0.1
    assert audit['minimum_closed_gripper_shelf_clearance_after_tolerance_m'] > 0.029
    assert audit['asset_count'] == 13
    assert audit['robot_collision_bundle_file_count'] == 24
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
    assert certificate.EXECUTION_POLICY['left_arm_only'] is True
    assert certificate.EXECUTION_POLICY['left_arm_command_count'] == 1
    assert certificate.EXECUTION_POLICY['gripper_command_count'] == 0
    assert certificate.EXECUTION_POLICY['base_motion_allowed'] is False
    assert certificate.EXECUTION_POLICY['right_arm_motion_allowed'] is False
    assert certificate.EXECUTION_POLICY['head_motion_allowed'] is False
    assert certificate.EXECUTION_POLICY[
        'gazebo_entity_pose_mutation_allowed'
    ] is False
