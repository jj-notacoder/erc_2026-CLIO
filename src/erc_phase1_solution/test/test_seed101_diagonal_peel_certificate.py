"""Focused pure checks for the exact seed-101 diagonal peel certificate."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest


MODULE = (
    Path(__file__).parents[1]
    / 'erc_phase1_solution'
    / 'seed101_diagonal_peel_certificate.py'
)
SPEC = importlib.util.spec_from_file_location(
    'seed101_diagonal_peel_certificate', MODULE
)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_exact_checkpoint_and_single_diagonal_peel_target_are_bound():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT_LEFT_Q8 == pytest.approx((
        0.3499998174377888,
        -0.31065478099691884,
        0.6171232383242858,
        0.08169638870230944,
        -1.4769923426656921,
        0.1731037765037025,
        1.0321934140278828,
        0.21509846856833278,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BOOK_POSITION_WORLD_M == pytest.approx((
        2.893645821378262,
        -0.15316229419440927,
        1.5773555286545407,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_GRIPPER_MASTER_M == pytest.approx(
        0.02900820842, abs=1e-15
    )
    assert certificate.ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP == (
        (0.0596105041931412, 0.0300944716983146, -0.136214299735423),
        (0.225737849631093, 0.0280366341074724, -0.0440325748535948),
        (0.0590427449893652, -0.0299020090531596, -0.1365304435706),
        (0.225170090427318, -0.0319598466440018, -0.0443487186887723),
        (-0.0762442592673814, 0.0300899741763813, 0.108619282979159),
        (0.0898830861705709, 0.0280321365855391, 0.200801007860987),
        (-0.0768120184711573, -0.0299065065750928, 0.108303139143981),
        (0.0893153269667949, -0.031964344165935, 0.200484864025809),
    )
    assert certificate.PLANNED_WORLD_DELTA_M == (-0.00075, 0.0, 0.0005)
    assert certificate.MINIMUM_DURATION_S >= 1.2
    assert certificate.DIAGONAL_PEEL_Q8 == pytest.approx((
        0.3499998174377888,
        -0.31014159591450025,
        0.6183859306984962,
        0.08185239525203032,
        -1.4775941032640676,
        0.17339864886753584,
        1.0341486100334276,
        0.2147926251332205,
    ), abs=1e-15)
    assert metrics['route_rows'] == 2
    assert metrics['maximum_joint_delta_rad'] == pytest.approx(
        0.0019551960055448347, abs=1e-15
    )


def test_focused_fk_and_inherited_collision_audit_is_positive_and_scoped():
    assert certificate.AUDIT[
        'diagnostic_waypoint_is_certified_neighborhood_subset'
    ] is True
    assert 'inherited conservative bounds' in certificate.AUDIT[
        'collision_coverage'
    ]
    assert certificate.AUDIT['target_fk_position_error_m'] < 1e-6
    assert certificate.AUDIT['target_fk_orientation_error_rad'] < 3e-6
    assert certificate.AUDIT[
        'inherited_full_route_minimum_nonadjacent_robot_aabb_clearance_m'
    ] == pytest.approx(0.002051274)
    assert certificate.AUDIT[
        'inherited_conservative_minimum_15mm_padded_payload_robot_clearance_m'
    ] == pytest.approx(0.00525063)
    assert certificate.ENDPOINT_POLICY[
        'require_fresh_bilateral_exact_target_pressure'
    ] is True
    assert certificate.ENDPOINT_POLICY[
        'min_book_center_outward_progress_m'
    ] == pytest.approx(0.00025)
    assert certificate.ENDPOINT_POLICY[
        'max_incremental_rotation_rad'
    ] == pytest.approx(0.002)
    assert certificate.ENDPOINT_POLICY[
        'max_book_from_hand_translation_change_m'
    ] == pytest.approx(0.0005)


def test_certificate_has_three_independent_digest_bindings():
    assert certificate.source_json_digest() == (
        certificate.CERTIFICATE_SOURCE_JSON_SHA256
    )
    assert certificate.semantic_digest() == certificate.CERTIFICATE_SEMANTIC_DIGEST
    assert certificate.route_q8_digest() == certificate.ROUTE_Q8_SHA256

    source_file_digest = hashlib.sha256(MODULE.read_bytes()).hexdigest()
    assert len(source_file_digest) == 64


def test_certificate_module_is_inert_and_has_no_runtime_control_imports():
    tree = ast.parse(MODULE.read_text(encoding='utf-8'))
    imported_roots = {
        alias.name.split('.')[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        (node.module or '').split('.')[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != '__future__'
    )
    assert imported_roots <= {'hashlib', 'json', 'math', 'typing'}
    assert not ({'rclpy', 'subprocess', 'socket'} & imported_roots)
