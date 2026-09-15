"""Focused checks for the exact seed-101 pure-outward certificate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


MODULE = (
    Path(__file__).parents[1]
    / 'erc_phase1_solution'
    / 'seed101_pure_outward_certificate.py'
)
SPEC = importlib.util.spec_from_file_location(
    'seed101_pure_outward_certificate', MODULE
)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_exact_checkpoint_and_outward_target_are_bound():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT_LEFT_Q8 == pytest.approx((
        0.3499998174377888,
        -0.31014159591450025,
        0.6183859306984962,
        0.08185239525203032,
        -1.4775941032640676,
        0.17339864886753584,
        1.0341486100334276,
        0.2147926251332205,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_BOOK_POSITION_WORLD_M == pytest.approx((
        2.893007288445724,
        -0.1531063074070775,
        1.577598577430115,
    ), abs=1e-15)
    assert certificate.CHECKPOINT_GRIPPER_MASTER_M == pytest.approx(
        0.029008128166538853, abs=1e-15
    )
    assert certificate.PLANNED_WORLD_DELTA_M == (-0.0005, 0.0, 0.0)
    assert certificate.MINIMUM_DURATION_S == 1.5
    assert certificate.PURE_OUTWARD_Q8 == pytest.approx((
        0.3499998174377888,
        -0.3099328933601225,
        0.618214728582325,
        0.0819199889501402,
        -1.479223208250511,
        0.17350872560919878,
        1.0356379503462736,
        0.21479652461032916,
    ), abs=1e-15)
    assert metrics['route_rows'] == 2
    assert metrics['maximum_joint_delta_rad'] == pytest.approx(
        0.0016291049864434193, abs=1e-15
    )


def test_endpoint_policy_is_narrow_and_fail_closed():
    policy = certificate.ENDPOINT_POLICY

    assert policy['require_fresh_bilateral_exact_target_pressure'] is True
    assert certificate.MAXIMUM_TRANSLATION_CHANGE_M == pytest.approx(0.00035)
    assert certificate.MAXIMUM_INCREMENTAL_ROTATION_RAD == pytest.approx(0.002)
    assert certificate.MINIMUM_CENTER_OUTWARD_PROGRESS_M == pytest.approx(
        0.00025
    )
    assert certificate.MINIMUM_DEEPEST_EXTENT_OUTWARD_PROGRESS_M == pytest.approx(
        0.0002
    )
    assert certificate.FLOOR_SIGNED_RANGE_M == pytest.approx(
        (-0.0001, 0.00025)
    )
    assert 'min_abs_rotation_axis_dot_world_y' not in policy


def test_collision_and_digest_bindings_are_positive():
    audit = certificate.AUDIT

    assert audit['collision_free'] is True
    assert audit['dense_joint_samples'] == 9
    assert audit['minimum_nonadjacent_robot_aabb_clearance_m'] > 0.002
    assert audit['minimum_15mm_padded_payload_robot_aabb_clearance_m'] > 0.09
    assert audit[
        'minimum_closed_gripper_shelf_clearance_after_tolerance_m'
    ] > 0.011
    assert certificate.source_json_digest() == (
        certificate.CERTIFICATE_SOURCE_JSON_SHA256
    )
    assert certificate.semantic_digest() == certificate.CERTIFICATE_SEMANTIC_DIGEST
    assert certificate.route_q8_digest() == certificate.ROUTE_Q8_SHA256


def test_certificate_module_is_inert():
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
