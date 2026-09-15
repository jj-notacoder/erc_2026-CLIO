"""Focused checks for the remaining seed-101 scale-up rungs."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).parents[1] / 'erc_phase1_solution' / (
    'seed101_outward_scaleup_continue_certificate.py'
)
SPEC = importlib.util.spec_from_file_location('scaleup_continue', MODULE)
assert SPEC is not None and SPEC.loader is not None
certificate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(certificate)


def test_checkpoint_route_and_durations_are_exact():
    valid, metrics = certificate.validate_certificate()

    assert valid
    assert certificate.CHECKPOINT_BOOK_POSITION_WORLD_M == pytest.approx((
        2.891599226313614, -0.15309109581985894, 1.5775425288657587
    ), abs=1e-15)
    assert certificate.PLANNED_WORLD_DELTA_M == (
        (-0.001, 0.0, 0.0), (-0.004, 0.0, 0.0)
    )
    assert certificate.ROW_MINIMUM_DURATIONS_S == (2.5, 4.5)
    assert tuple(row['phase'] for row in certificate.ROUTE) == (
        'checkpoint', 'shelf_outward', 'shelf_outward'
    )
    assert metrics['route_rows'] == 3
    assert metrics['command_rows'] == 2


def test_per_row_endpoint_limits_are_bound():
    policy = certificate.ENDPOINT_POLICY

    assert policy['pause_after_every_endpoint'] is True
    assert policy['require_fresh_bilateral_exact_target_pressure'] is True
    assert policy['maximum_cumulative_rotation_rad'] == [0.0008, 0.0018]
    assert policy['maximum_cumulative_translation_change_m'] == [
        0.00018, 0.00032
    ]
    assert policy['minimum_book_center_cumulative_outward_progress_m'] == [
        0.0007, 0.0028
    ]
    assert policy['minimum_deepest_extent_cumulative_outward_progress_m'] == [
        0.00065, 0.0026
    ]
    assert policy['minimum_each_side_force_retention_fraction'] == 0.5


def test_collision_and_digest_facts_are_positive():
    audit = certificate.AUDIT

    assert audit['collision_free'] is True
    assert audit['dense_joint_samples'] == 8
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
