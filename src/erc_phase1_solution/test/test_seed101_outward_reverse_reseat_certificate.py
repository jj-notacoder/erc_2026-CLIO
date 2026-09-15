"""Focused checks for the exact seed-101 reverse-reseat certificate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / 'erc_phase1_solution' / (
    'seed101_outward_reverse_reseat_certificate.py'
)
SPEC = importlib.util.spec_from_file_location('scaleup_reverse_reseat', MODULE)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_exact_failed_settle_checkpoint_is_bound():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT_BOOK_POSITION_WORLD_M == pytest.approx((
        2.890661892167218,
        -0.15317035347670219,
        1.5775238509060145,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BOOK_QUATERNION_XYZW == pytest.approx((
        -0.004188475048371351,
        0.71005656164804254,
        0.004174723293775891,
        0.70411981055978623,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_WORLD_POSITION_M == pytest.approx((
        2.0026729276880491,
        -0.14949948904604238,
        -3.977426593568196e-07,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_QUATERNION_XYZW == pytest.approx((
        1.8196506007855779e-07,
        3.006148335564496e-07,
        -0.3826074383522628,
        0.923911006600417,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_WORLD_YAW_RAD == pytest.approx(
        -0.7852336555462195, abs=1e-15
    )
    assert certificate.CHECKPOINT_LEFT_Q8 == pytest.approx((
        0.34999953751554436,
        -0.30909861272542083,
        0.6175553554062443,
        0.08219275225689975,
        -1.4857088139359278,
        0.17395238931347662,
        1.0415889844466004,
        0.2148105279821962,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_GRIPPER_MASTER_M == pytest.approx(
        0.029008297335297823, abs=1e-15
    )
    assert certificate.CHECKPOINT_RIGHT_Q7 == pytest.approx((
        1.0360774759655906e-05,
        -1.0042776704687532e-05,
        8.783570150086181e-06,
        -6.745791004783519e-06,
        2.4298609371527956e-07,
        5.51294518680262e-09,
        -2.491546018600574e-09,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_HEAD_Q2 == pytest.approx((
        4.837166206112712e-12,
        -2.158367100045046e-12,
    ), abs=1e-15)
    assert certificate.CHECKPOINT['world_paused'] is True
    assert certificate.CHECKPOINT['world_sim_time_s'] == pytest.approx(339.416)
    assert certificate.CHECKPOINT['world_iterations'] == 169708
    assert metrics['route_rows'] == 2
    assert metrics['command_rows'] == 1


def test_reverse_route_targets_exact_previously_stable_checkpoint():
    assert certificate.PLANNED_WORLD_DELTA_M == (0.001, 0.0, 0.0)
    assert certificate.MINIMUM_DURATION_S == pytest.approx(2.5)
    assert certificate.ROUTE[0]['phase'] == 'checkpoint'
    assert certificate.ROUTE[1]['phase'] == 'shelf_inward'
    assert certificate.REVERSE_RESEAT_Q8 == pytest.approx((
        0.34999953751554436,
        -0.3095148967058795,
        0.6178794379281761,
        0.08205575594538433,
        -1.4824722381676605,
        0.17372994976888595,
        1.0386147095992404,
        0.2148038614518491,
    ), abs=1e-15)
    assert certificate.REVERSE_RESEAT_Q8 == certificate.TARGET_LEFT_Q8
    assert certificate.STABLE_REFERENCE_BOOK_POSITION_WORLD_M == pytest.approx((
        2.891599226313614,
        -0.15309109581985894,
        1.5775425288657587,
    ), abs=1e-15)
    assert certificate.STABLE_REFERENCE_BOOK_QUATERNION_XYZW == pytest.approx((
        -0.003867323503119562,
        0.7101335529054696,
        0.003847067873253381,
        0.704045865633451,
    ), abs=1e-15)


def test_execution_policy_is_left_arm_only_and_pressure_retaining():
    execution = certificate.EXECUTION_POLICY

    assert certificate.STAGE == 'outward-reverse-reseat'
    assert execution['left_arm_only'] is True
    assert execution['fixed_hand_orientation'] is True
    assert execution['torso_fixed'] is True
    assert execution['keep_current_gripper_pressure'] is True
    assert execution['pause_after_endpoint'] is True
    assert execution['base_motion_allowed'] is False
    assert execution['head_motion_allowed'] is False
    assert execution['right_arm_motion_allowed'] is False
    assert execution['gazebo_entity_pose_mutation_allowed'] is False
    assert execution[
        'maximum_continuous_absolute_rotation_rad'
    ] == pytest.approx(0.0012)
    assert execution[
        'maximum_continuous_absolute_yaw_rad'
    ] == pytest.approx(0.0012)


def test_endpoint_policy_and_pressure_retention_are_exact():
    policy = certificate.ENDPOINT_POLICY

    assert policy[
        'minimum_book_center_cumulative_inward_progress_m'
    ] == pytest.approx(0.0007)
    assert policy[
        'minimum_book_maximum_x_cumulative_inward_return_m'
    ] == pytest.approx(0.00065)
    assert policy[
        'maximum_book_from_hand_translation_change_m'
    ] == pytest.approx(0.00018)
    assert policy['maximum_incremental_rotation_rad'] == pytest.approx(0.0012)
    assert policy[
        'maximum_absolute_rotation_to_stable_reference_rad'
    ] == pytest.approx(0.0008)
    assert policy[
        'maximum_absolute_yaw_to_stable_reference_rad'
    ] == pytest.approx(0.0008)
    assert policy[
        'maximum_absolute_rotation_growth_from_start_rad'
    ] == pytest.approx(0.00005)
    assert policy['maximum_cross_track_motion_m'] == pytest.approx(0.00018)
    assert policy['maximum_reference_depth_overshoot_m'] == pytest.approx(
        0.00015
    )
    assert policy['maximum_stable_reference_position_error_m'] == pytest.approx(
        0.00025
    )
    assert certificate.FLOOR_SIGNED_RANGE_M == pytest.approx(
        (-0.0001, 0.00025)
    )
    assert policy['require_fresh_bilateral_exact_target_pressure'] is True
    assert policy['pause_after_endpoint'] is True
    assert certificate.MINIMUM_FORCE_RETENTION_FRACTION == pytest.approx(0.5)
    assert certificate.REFERENCE_PRESSURE_N == pytest.approx((
        2.07182345237312,
        1.810479005873424,
    ), abs=1e-15)
    assert certificate.MINIMUM_FORCE_N == pytest.approx((
        1.03591172618656,
        0.905239502936712,
    ), abs=1e-15)


def test_reverse_audit_and_digest_bindings_are_positive():
    valid, metrics = certificate.validate_certificate()
    audit = certificate.AUDIT

    assert valid
    assert audit['collision_free'] is True
    assert audit['reverse_of_source_route_rows'] == [1, 0]
    assert audit['current_geometry_revalidation_required'] is True
    assert audit['asset_count'] == 13
    assert audit['robot_collision_bundle_file_count'] == 24
    assert len(certificate.ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP) == 8
    assert len(certificate.ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP) == 8
    assert audit['minimum_nonadjacent_robot_aabb_clearance_m'] > 0.002
    assert audit['minimum_15mm_padded_payload_robot_aabb_clearance_m'] > 0.099
    assert audit[
        'minimum_closed_gripper_shelf_clearance_after_tolerance_m'
    ] > 0.011
    assert metrics['certified_start_joint_error_rad'] == pytest.approx(
        2.536561498067691e-06, abs=1e-15
    )
    assert metrics['maximum_joint_delta_rad'] == pytest.approx(
        0.0032365757682673024, abs=1e-15
    )
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
