"""Focused checks for the zero-command post-reseat settle certificate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / 'erc_phase1_solution' / (
    'seed101_outward_post_reseat_settle_certificate.py'
)
SPEC = importlib.util.spec_from_file_location('post_reseat_settle', MODULE)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_exact_paused_checkpoint_is_bound():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT_BOOK_POSITION_WORLD_M == pytest.approx((
        2.8913950882433226,
        -0.15308777230180326,
        1.5775144514170192,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BOOK_QUATERNION_XYZW == pytest.approx((
        -0.0038177587397655315,
        0.71003865841669134,
        0.0037855082890519117,
        0.70414217186518768,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_WORLD_POSITION_M == pytest.approx((
        2.0026561063355421,
        -0.14951695526078862,
        -7.2401578769641856e-08,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_QUATERNION_XYZW == pytest.approx((
        4.9649765503099407e-08,
        7.9098610137010558e-08,
        -0.382572923442866,
        0.92392529906284648,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BASE_WORLD_YAW_RAD == pytest.approx(
        -0.7851589413319934, abs=1e-15
    )
    assert certificate.CHECKPOINT_LEFT_Q8 == pytest.approx((
        0.3499999965486979,
        -0.30951489249886954,
        0.6178794341067166,
        0.08205576832470303,
        -1.4824724103134208,
        0.1737299607249708,
        1.038614878890176,
        0.21480386196601237,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_GRIPPER_MASTER_M == pytest.approx(
        0.0290081304296433, abs=1e-15
    )
    assert certificate.CHECKPOINT_RIGHT_Q7 == pytest.approx((
        7.034048053706401e-08,
        4.423280813918232e-06,
        5.8105495467438036e-08,
        -7.514830644261804e-06,
        1.7459349433664546e-09,
        1.166947670158128e-10,
        -1.0919903091720797e-10,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_HEAD_Q2 == pytest.approx((
        9.39826583820943e-12,
        -2.0856175280534895e-12,
    ), abs=1e-15)
    assert certificate.CHECKPOINT['world_paused'] is True
    assert certificate.CHECKPOINT['world_sim_time_s'] == pytest.approx(344.284)
    assert certificate.CHECKPOINT['world_iterations'] == 172142
    assert metrics['route_rows'] == 1
    assert metrics['command_rows'] == 0


def test_recomputed_current_geometry_and_stable_reference_are_exact():
    assert certificate.CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M[0] == pytest.approx(
        (2.81019889910172, -0.1689583865527535, 1.4518516705086513),
        abs=1e-15,
    )
    assert certificate.CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M[1] == pytest.approx(
        (2.9725912773849252, -0.13721715805085302, 1.703177232325387),
        abs=1e-15,
    )
    assert certificate.CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M == pytest.approx(
        -6.796354348637124e-06, abs=1e-15
    )
    assert certificate.SUPPORT_FLOOR_WORLD_Z_M == pytest.approx(
        1.451858466863, abs=1e-15
    )
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
    assert certificate.ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP[0] == (
        0.0654698981797365,
        0.014823390334779923,
        -0.11561033562445316,
    )
    assert certificate.ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP[-1] == (
        0.08887817228416191,
        -0.03180873270627125,
        0.20085845873628055,
    )
    assert len(certificate.ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP) == 8
    assert len(certificate.ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP) == 8
    assert certificate.AUDIT[
        'attached_corners_recomputed_from_current_checkpoint'
    ] is True


def test_fail_closed_stationary_geometry_and_pressure_limits_are_exact():
    endpoint = certificate.ENDPOINT_POLICY
    execution = certificate.EXECUTION_POLICY

    assert endpoint['maximum_stable_reference_position_error_m'] == 0.00025
    assert endpoint['maximum_stable_reference_maximum_x_error_m'] == 0.00030
    assert endpoint[
        'maximum_absolute_rotation_to_stable_reference_rad'
    ] == 0.0008
    assert endpoint[
        'maximum_absolute_yaw_to_stable_reference_rad'
    ] == 0.0008
    assert certificate.FLOOR_SIGNED_RANGE_M == (-0.0001, 0.00025)
    assert certificate.MINIMUM_FORCE_N == (
        1.03591172618656,
        0.905239502936712,
    )
    assert endpoint['require_fresh_bilateral_exact_target_pressure'] is True
    assert endpoint['require_base_stable'] is True
    assert endpoint['require_gripper_stable'] is True
    assert endpoint['require_head_stable'] is True
    assert endpoint['require_right_arm_stable'] is True
    assert endpoint['require_non_target_scene_stable'] is True
    assert execution['maximum_continuous_absolute_rotation_rad'] == 0.0012
    assert execution['maximum_continuous_absolute_yaw_rad'] == 0.0012
    assert certificate.SETTLE_ROTATION_GROWTH_LIMIT_RAD == 0.00005
    assert certificate.SETTLE_STABILITY_TRANSLATION_LIMIT_M == 0.00005
    assert certificate.SETTLE_STABILITY_ROTATION_LIMIT_RAD == 0.000075
    assert certificate.STARTUP_STABILITY_TRANSLATION_LIMIT_M == 0.00005
    assert certificate.STARTUP_STABILITY_ROTATION_LIMIT_RAD == 0.0002


def test_startup_and_final_stability_limits_remain_distinct_and_exact():
    """The looser startup sample must not weaken the final settle proof."""

    execution = certificate.EXECUTION_POLICY

    assert certificate.STARTUP_STABILITY_TRANSLATION_LIMIT_M == 0.00005
    assert certificate.STARTUP_STABILITY_ROTATION_LIMIT_RAD == 0.0002
    assert certificate.SETTLE_STABILITY_TRANSLATION_LIMIT_M == 0.00005
    assert certificate.SETTLE_STABILITY_ROTATION_LIMIT_RAD == 0.000075
    assert certificate.SETTLE_ROTATION_GROWTH_LIMIT_RAD == 0.00005
    assert (
        execution['maximum_startup_translation_change_m']
        == certificate.STARTUP_STABILITY_TRANSLATION_LIMIT_M
    )
    assert (
        execution['maximum_startup_rotation_change_rad']
        == certificate.STARTUP_STABILITY_ROTATION_LIMIT_RAD
    )
    assert (
        execution['maximum_short_dwell_translation_change_m']
        == certificate.SETTLE_STABILITY_TRANSLATION_LIMIT_M
    )
    assert (
        execution['maximum_short_dwell_rotation_change_rad']
        == certificate.SETTLE_STABILITY_ROTATION_LIMIT_RAD
    )
    assert (
        execution['maximum_rotation_growth_during_settle_rad']
        == certificate.SETTLE_ROTATION_GROWTH_LIMIT_RAD
    )
    assert (
        certificate.STARTUP_STABILITY_ROTATION_LIMIT_RAD
        > certificate.SETTLE_STABILITY_ROTATION_LIMIT_RAD
    )


def test_route_has_one_checkpoint_and_execution_commands_nothing():
    execution = certificate.EXECUTION_POLICY

    assert certificate.STAGE == 'outward-post-reseat-settle'
    assert certificate.SETTLE_DURATION_S == 0.4
    assert len(certificate.ROUTE) == 1
    assert certificate.ROUTE[0]['phase'] == 'checkpoint'
    assert certificate.ROUTE[0]['world_delta_m'] == [0.0, 0.0, 0.0]
    assert certificate.ROUTE_Q8 == (certificate.CHECKPOINT_LEFT_Q8,)
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


def test_audit_digests_and_inert_imports_are_bound():
    audit = certificate.AUDIT

    assert audit['collision_free'] is True
    assert audit['zero_command_state_only'] is True
    assert audit['current_geometry_revalidation_required'] is True
    assert audit[
        'inherited_collision_audit_applies_to_stationary_checkpoint_only'
    ] is True
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
