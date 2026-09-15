"""Focused checks for the seed-101 outward scale-up certificate."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


MODULE = (
    Path(__file__).parents[1]
    / 'erc_phase1_solution'
    / 'seed101_outward_scaleup_certificate.py'
)
SPEC = importlib.util.spec_from_file_location(
    'seed101_outward_scaleup_certificate', MODULE
)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_exact_checkpoint_and_three_scaleup_rows_are_bound():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT_LEFT_Q8 == pytest.approx((
        0.3499998174377888,
        -0.3099328933601225,
        0.618214728582325,
        0.0819199889501402,
        -1.479223208250511,
        0.17350872560919878,
        1.0356379503462736,
        0.21479652461032916,
    ), abs=1e-15)
    assert certificate.PLANNED_WORLD_DELTA_M == (
        (-0.001, 0.0, 0.0),
        (-0.002, 0.0, 0.0),
        (-0.005, 0.0, 0.0),
    )
    assert certificate.ROW_MINIMUM_DURATIONS_S == (2.5, 2.5, 4.5)
    assert tuple(row['phase'] for row in certificate.ROUTE) == (
        'checkpoint', 'shelf_outward', 'shelf_outward', 'shelf_outward'
    )
    assert metrics['route_rows'] == 4
    assert metrics['command_rows'] == 3


def test_endpoint_policy_pauses_and_scales_limits_per_leg():
    policy = certificate.ENDPOINT_POLICY

    assert policy['pause_after_every_endpoint'] is True
    assert policy['require_fresh_bilateral_exact_target_pressure'] is True
    assert policy['maximum_cumulative_translation_change_m'] == (
        [0.00015, 0.0002, 0.00035]
    )
    assert policy['maximum_cumulative_rotation_rad'] == (
        [0.0006, 0.001, 0.002]
    )
    assert policy['minimum_each_side_force_retention_fraction'] == 0.5
    assert policy['floor_signed_min_m'] == pytest.approx(-0.0001)
    assert policy['floor_signed_max_m'] == pytest.approx(0.00025)


def test_collision_audit_and_digests_are_bound():
    audit = certificate.AUDIT

    assert audit['collision_free'] is True
    assert audit['dense_joint_samples'] == 10
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


def test_certificate_is_inert():
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
