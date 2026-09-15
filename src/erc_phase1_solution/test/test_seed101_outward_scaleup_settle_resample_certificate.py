"""Focused checks for the zero-command settle / resample certificate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / 'erc_phase1_solution' / (
    'seed101_outward_scaleup_settle_resample_certificate.py'
)
SPEC = importlib.util.spec_from_file_location('scaleup_settle_resample', MODULE)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_exact_paused_checkpoint_is_bound():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT_BOOK_POSITION_WORLD_M == pytest.approx((
        2.8906623656072785,
        -0.15317203567178594,
        1.5775255289762071,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BOOK_QUATERNION_XYZW == pytest.approx((
        -0.0041921970354157952,
        0.71006386026635582,
        0.0041818392611605477,
        0.70411238594989312,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_WORLD_POSITION_M == pytest.approx((
        2.0026756657405862,
        -0.14949491213554689,
        -1.4878387192088314e-06,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_QUATERNION_XYZW == pytest.approx((
        -7.841987717116337e-07,
        -1.8111510164022872e-06,
        -0.38261328459685773,
        0.92390858554626965,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_WORLD_YAW_RAD == pytest.approx(
        -0.78524631099223996, abs=1e-15
    )
    assert certificate.CHECKPOINT_LEFT_Q8 == pytest.approx((
        0.34999960263322505,
        -0.3090982737139299,
        0.6175550756809001,
        0.08219257607711211,
        -1.4857083412082002,
        0.17395238960120715,
        1.0415884404391076,
        0.21481052689977764,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_GRIPPER_MASTER_M == pytest.approx(
        0.029008248114871713, abs=1e-15
    )
    assert certificate.CHECKPOINT_RIGHT_Q7 == pytest.approx((
        8.928280789134892e-06,
        -7.879013991233926e-06,
        7.637038921104753e-06,
        -6.830823170602171e-06,
        2.0678424885728507e-07,
        9.92501358043243e-10,
        -1.1777669089316835e-09,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_HEAD_Q2 == pytest.approx((
        -2.129024192630906e-10,
        -2.154373341162415e-12,
    ), abs=1e-15)
    assert certificate.CHECKPOINT['world_paused'] is True
    assert certificate.CHECKPOINT['world_sim_time_s'] == pytest.approx(338.362)
    assert certificate.CHECKPOINT['world_iterations'] == 169181
    assert metrics['route_rows'] == 1
    assert metrics['command_rows'] == 0


def test_reference_and_fail_closed_resample_limits_are_exact():
    policy = certificate.ENDPOINT_POLICY

    assert certificate.REFERENCE_BOOK_POSITION_WORLD_M == pytest.approx((
        2.891599226313614,
        -0.15309109581985894,
        1.5775425288657587,
    ), abs=1e-15)
    assert certificate.REFERENCE_BOOK_QUATERNION_XYZW == pytest.approx((
        -0.003867323503119562,
        0.7101335529054696,
        0.003847067873253381,
        0.704045865633451,
    ), abs=1e-15)
    assert certificate.REFERENCE_PLANNED_WORLD_DELTA_M == (
        -0.001, 0.0, 0.0
    )
    assert policy['maximum_cumulative_rotation_rad'] == pytest.approx(0.0008)
    assert policy['maximum_cumulative_yaw_component_rad'] == pytest.approx(
        0.0008
    )
    assert certificate.SETTLE_ROTATION_GROWTH_LIMIT_RAD == pytest.approx(
        0.00005
    )
    assert policy[
        'maximum_book_from_hand_translation_change_m'
    ] == pytest.approx(0.00018)
    assert policy[
        'minimum_book_center_cumulative_outward_progress_m'
    ] == pytest.approx(0.0007)
    assert policy[
        'minimum_deepest_extent_cumulative_outward_progress_m'
    ] == pytest.approx(0.00065)
    assert certificate.FLOOR_SIGNED_RANGE_M == pytest.approx(
        (-0.0001, 0.00025)
    )
    assert certificate.MINIMUM_FORCE_N == pytest.approx((1.031, 0.913))
    assert policy['require_fresh_bilateral_exact_target_pressure'] is True


def test_route_has_one_checkpoint_and_execution_commands_nothing():
    execution = certificate.EXECUTION_POLICY

    assert certificate.SETTLE_DURATION_S == pytest.approx(0.2)
    assert len(certificate.ROUTE) == 1
    assert certificate.ROUTE[0]['phase'] == 'checkpoint'
    assert certificate.ROUTE[0]['world_delta_m'] == [0.0, 0.0, 0.0]
    assert certificate.COMMAND_Q8 == ()
    assert execution['trajectory_command_rows'] == 0
    assert execution['base_command_count'] == 0
    assert execution['left_arm_command_count'] == 0
    assert execution['right_arm_command_count'] == 0
    assert execution['gripper_command_count'] == 0
    assert execution['head_command_count'] == 0
    assert execution['gazebo_entity_pose_mutation_allowed'] is False
    assert execution['world_resume_allowed_for_settle_only'] is True
    assert execution['repause_after_settle'] is True


def test_inherited_stationary_audit_and_digests_are_bound():
    audit = certificate.AUDIT

    assert audit['collision_free'] is True
    assert audit['zero_command_state_only'] is True
    assert audit['current_geometry_revalidation_required'] is True
    assert audit['asset_count'] == 13
    assert audit['robot_collision_bundle_file_count'] == 24
    assert audit['minimum_nonadjacent_robot_aabb_clearance_m'] > 0.002
    assert audit['minimum_15mm_padded_payload_robot_aabb_clearance_m'] > 0.099
    assert audit[
        'minimum_closed_gripper_shelf_clearance_after_tolerance_m'
    ] > 0.011
    assert certificate.source_json_digest() == (
        certificate.CERTIFICATE_SOURCE_JSON_SHA256
    )
    assert certificate.semantic_digest() == certificate.CERTIFICATE_SEMANTIC_DIGEST
    assert certificate.route_q8_digest() == certificate.ROUTE_Q8_SHA256


def test_module_is_inert():
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
