"""Inert certificate for a zero-command seed-101 settle / resample.

The certificate binds the exact paused state left by the failed first rung of
the outward-scale-up continuation.  Its route contains only that checkpoint;
it authorizes no joint, gripper, base, head, or entity-pose command.  A live
caller may set up feedback and pressure monitoring, resume physics for at
least 0.20 s, repause, and apply the embedded fail-closed observation policy.
Importing this module performs no external operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    'ab4a2fa12c40deb03906110f1e5fb887e42b3c6335251d53fcfd8221dd6ba8a8'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    '33cce7939f5eb30e7ad0b797bbf7ffe71a7efe06a1678dc845fe2f5663951251'
)
CERTIFICATE_JSON = r'''{
  "audit": {
    "asset_count": 13,
    "collision_free": true,
    "current_geometry_revalidation_required": true,
    "dense_joint_samples": 8,
    "inherited_from_certificate_kind": "read_only_seed101_outward_scaleup_continue_certificate",
    "inherited_source_json_sha256": "c48b1d01d43bb2f461e4e4c83b42ad05037629f8c21afb6a5d886715d4e5be1c",
    "inherited_source_semantic_digest": "3f73080bd147c3b61e1060b3ac3348ef1d09d45dba77cd3cb7c4182619f0a1ee",
    "inherited_source_route_q8_sha256": "8a88bf78e9350e4fa2625735a61fb2f0aa2b6e4e037ed598df167384dfcfe7e3",
    "maximum_dense_joint_increment_rad": 0.002,
    "minimum_15mm_padded_payload_robot_aabb_clearance_m": 0.09931032286966568,
    "minimum_carried_book_other_book_aabb_clearance_m": 0.25845650009185483,
    "minimum_closed_gripper_shelf_clearance_after_tolerance_m": 0.011773523189314156,
    "minimum_nonadjacent_robot_aabb_clearance_m": 0.002157269827602226,
    "minimum_robot_bin_aabb_clearance_m": 2.3313741399156047,
    "minimum_robot_other_book_aabb_clearance_m": 0.14121832644058888,
    "minimum_robot_table_aabb_clearance_m": 2.173460948331139,
    "robot_collision_bundle_file_count": 24,
    "shelf_collision_tolerance_m": 0.00075,
    "stationary_world_delta_m": [0.0, 0.0, 0.0],
    "zero_command_state_only": true
  },
  "checkpoint": {
    "base_quaternion_xyzw": [-7.841987717116337e-07, -1.8111510164022872e-06, -0.38261328459685773, 0.92390858554626965],
    "base_world_position_m": [2.0026756657405862, -0.14949491213554689, -1.4878387192088314e-06],
    "base_world_yaw_rad": -0.78524631099223996,
    "book_position_world_m": [2.8906623656072785, -0.15317203567178594, 1.5775255289762071],
    "book_quaternion_xyzw": [-0.0041921970354157952, 0.71006386026635582, 0.0041818392611605477, 0.70411238594989312],
    "gripper_master_m": 0.029008248114871713,
    "head_q2_measured": [-2.129024192630906e-10, -2.154373341162415e-12],
    "left_q8_measured": [0.34999960263322505, -0.3090982737139299, 0.6175550756809001, 0.08219257607711211, -1.4857083412082002, 0.17395238960120715, 1.0415884404391076, 0.21481052689977764],
    "right_q7_measured": [8.928280789134892e-06, -7.879013991233926e-06, 7.637038921104753e-06, -6.830823170602171e-06, 2.0678424885728507e-07, 9.92501358043243e-10, -1.1777669089316835e-09],
    "target_model": "book_col_3_row_2_red",
    "world_iterations": 169181,
    "world_paused": true,
    "world_sim_time_s": 338.362
  },
  "comparison_reference": {
    "book_position_world_m": [2.891599226313614, -0.15309109581985894, 1.5775425288657587],
    "book_quaternion_xyzw": [-0.003867323503119562, 0.7101335529054696, 0.003847067873253381, 0.704045865633451],
    "kind": "pre_failed_outward_scaleup_continue_start",
    "reference_planned_world_delta_m": [-0.001, 0.0, 0.0]
  },
  "endpoint_policy": {
    "floor_signed_max_m": 0.00025,
    "floor_signed_min_m": -0.0001,
    "maximum_book_from_hand_translation_change_m": 0.00018,
    "maximum_cumulative_rotation_rad": 0.0008,
    "maximum_cumulative_yaw_component_rad": 0.0008,
    "maximum_settle_rotation_growth_rad": 0.00005,
    "minimum_book_center_cumulative_outward_progress_m": 0.0007,
    "minimum_deepest_extent_cumulative_outward_progress_m": 0.00065,
    "minimum_left_force_n": 1.031,
    "minimum_right_force_n": 0.913,
    "require_base_stable": true,
    "require_fresh_bilateral_exact_target_pressure": true,
    "require_non_target_scene_stable": true
  },
  "execution_policy": {
    "base_command_count": 0,
    "gazebo_entity_pose_mutation_allowed": false,
    "gripper_command_count": 0,
    "head_command_count": 0,
    "left_arm_command_count": 0,
    "repause_after_settle": true,
    "right_arm_command_count": 0,
    "settle_duration_s": 0.2,
    "trajectory_command_rows": 0,
    "world_resume_allowed_for_settle_only": true
  },
  "kind": "read_only_seed101_outward_scaleup_settle_resample_certificate",
  "prior_pressure_evidence": {
    "left_force_n": 2.0616649794995525,
    "left_samples": 64,
    "measured_width_m": 0.029008188497949033,
    "reason": "bilateral_contact_verified",
    "right_force_n": 1.8254921687182613,
    "right_samples": 64,
    "stage": "pre_extraction_step_1",
    "target_model": "book_col_3_row_2_red"
  },
  "route": [
    {
      "minimum_duration_s": 0.0,
      "phase": "checkpoint",
      "q8": [0.34999960263322505, -0.3090982737139299, 0.6175550756809001, 0.08219257607711211, -1.4857083412082002, 0.17395238960120715, 1.0415884404391076, 0.21481052689977764],
      "row": 0,
      "world_delta_m": [0.0, 0.0, 0.0]
    }
  ],
  "route_q8_sha256": "08b34ec0a5f142cf6aa24d1ab2b61c3af7730a145805c07329d1aa8498aac1da"
}'''

CERTIFICATE: Mapping[str, object] = json.loads(CERTIFICATE_JSON)
CERTIFICATE_KIND = str(CERTIFICATE['kind'])
ROUTE_Q8_SHA256 = str(CERTIFICATE['route_q8_sha256'])
CHECKPOINT = CERTIFICATE['checkpoint']
COMPARISON_REFERENCE = CERTIFICATE['comparison_reference']
AUDIT = CERTIFICATE['audit']
ENDPOINT_POLICY = CERTIFICATE['endpoint_policy']
EXECUTION_POLICY = CERTIFICATE['execution_policy']
PRIOR_PRESSURE_EVIDENCE = CERTIFICATE['prior_pressure_evidence']

CHECKPOINT_BASE_WORLD_POSITION_M = tuple(CHECKPOINT['base_world_position_m'])
CHECKPOINT_BASE_QUATERNION_XYZW = tuple(CHECKPOINT['base_quaternion_xyzw'])
CHECKPOINT_BASE_WORLD_YAW_RAD = float(CHECKPOINT['base_world_yaw_rad'])
CHECKPOINT_BOOK_POSITION_WORLD_M = tuple(CHECKPOINT['book_position_world_m'])
CHECKPOINT_BOOK_QUATERNION_XYZW = tuple(CHECKPOINT['book_quaternion_xyzw'])
CHECKPOINT_LEFT_Q8 = tuple(CHECKPOINT['left_q8_measured'])
CHECKPOINT_RIGHT_Q7 = tuple(CHECKPOINT['right_q7_measured'])
CHECKPOINT_HEAD_Q2 = tuple(CHECKPOINT['head_q2_measured'])
CHECKPOINT_GRIPPER_MASTER_M = float(CHECKPOINT['gripper_master_m'])
REFERENCE_BOOK_POSITION_WORLD_M = tuple(
    COMPARISON_REFERENCE['book_position_world_m']
)
REFERENCE_BOOK_QUATERNION_XYZW = tuple(
    COMPARISON_REFERENCE['book_quaternion_xyzw']
)
REFERENCE_PLANNED_WORLD_DELTA_M = tuple(
    COMPARISON_REFERENCE['reference_planned_world_delta_m']
)
SETTLE_DURATION_S = float(EXECUTION_POLICY['settle_duration_s'])
FLOOR_SIGNED_RANGE_M = (
    float(ENDPOINT_POLICY['floor_signed_min_m']),
    float(ENDPOINT_POLICY['floor_signed_max_m']),
)
MINIMUM_FORCE_N = (
    float(ENDPOINT_POLICY['minimum_left_force_n']),
    float(ENDPOINT_POLICY['minimum_right_force_n']),
)
SETTLE_ROTATION_GROWTH_LIMIT_RAD = float(
    ENDPOINT_POLICY['maximum_settle_rotation_growth_rad']
)
ROUTE = tuple(CERTIFICATE['route'])
ROUTE_Q8 = tuple(tuple(row['q8']) for row in ROUTE)
COMMAND_Q8 = ROUTE_Q8[1:]


def _canonical_certificate_bytes() -> bytes:
    return json.dumps(
        CERTIFICATE, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode('utf-8')


def semantic_digest() -> str:
    """Return the deterministic digest of all embedded certificate data."""

    return hashlib.sha256(_canonical_certificate_bytes()).hexdigest()


def source_json_digest() -> str:
    """Return the digest of the exact embedded JSON text."""

    return hashlib.sha256(CERTIFICATE_JSON.encode('utf-8')).hexdigest()


def route_q8_digest() -> str:
    """Return the digest of the single, non-command checkpoint row."""

    payload = json.dumps(
        [list(row) for row in ROUTE_Q8],
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _finite_vector(values: object, length: int) -> bool:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return False
    return len(vector) == length and all(math.isfinite(value) for value in vector)


def validate_certificate() -> Tuple[bool, Mapping[str, float]]:
    """Fail closed if the checkpoint, no-command policy, or limits change."""

    route_valid = bool(
        len(ROUTE) == 1
        and len(COMMAND_Q8) == 0
        and isinstance(ROUTE[0], Mapping)
        and int(ROUTE[0].get('row', -1)) == 0
        and ROUTE[0].get('phase') == 'checkpoint'
        and _finite_vector(ROUTE[0].get('q8'), 8)
        and tuple(ROUTE[0].get('world_delta_m', ())) == (0.0, 0.0, 0.0)
        and float(ROUTE[0].get('minimum_duration_s', math.nan)) == 0.0
        and ROUTE_Q8[0] == CHECKPOINT_LEFT_Q8
    )
    checkpoint_valid = bool(
        CHECKPOINT.get('world_paused') is True
        and int(CHECKPOINT.get('world_iterations', -1)) == 169181
        and float(CHECKPOINT.get('world_sim_time_s', math.nan)) == 338.362
        and CHECKPOINT.get('target_model') == 'book_col_3_row_2_red'
        and _finite_vector(CHECKPOINT_BASE_WORLD_POSITION_M, 3)
        and _finite_vector(CHECKPOINT_BASE_QUATERNION_XYZW, 4)
        and math.isfinite(CHECKPOINT_BASE_WORLD_YAW_RAD)
        and _finite_vector(CHECKPOINT_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(CHECKPOINT_BOOK_QUATERNION_XYZW, 4)
        and _finite_vector(CHECKPOINT_LEFT_Q8, 8)
        and _finite_vector(CHECKPOINT_RIGHT_Q7, 7)
        and _finite_vector(CHECKPOINT_HEAD_Q2, 2)
        and math.isfinite(CHECKPOINT_GRIPPER_MASTER_M)
        and _finite_vector(REFERENCE_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(REFERENCE_BOOK_QUATERNION_XYZW, 4)
        and REFERENCE_PLANNED_WORLD_DELTA_M == (-0.001, 0.0, 0.0)
    )
    execution_valid = bool(
        SETTLE_DURATION_S == 0.2
        and int(EXECUTION_POLICY.get('trajectory_command_rows', -1)) == 0
        and all(
            int(EXECUTION_POLICY.get(key, -1)) == 0
            for key in (
                'base_command_count',
                'gripper_command_count',
                'head_command_count',
                'left_arm_command_count',
                'right_arm_command_count',
            )
        )
        and EXECUTION_POLICY.get('gazebo_entity_pose_mutation_allowed') is False
        and EXECUTION_POLICY.get('world_resume_allowed_for_settle_only') is True
        and EXECUTION_POLICY.get('repause_after_settle') is True
    )
    endpoint_valid = bool(
        float(ENDPOINT_POLICY['maximum_cumulative_rotation_rad']) == 0.0008
        and float(ENDPOINT_POLICY['maximum_cumulative_yaw_component_rad'])
        == 0.0008
        and SETTLE_ROTATION_GROWTH_LIMIT_RAD == 0.00005
        and float(
            ENDPOINT_POLICY['maximum_book_from_hand_translation_change_m']
        ) == 0.00018
        and float(
            ENDPOINT_POLICY['minimum_book_center_cumulative_outward_progress_m']
        ) == 0.0007
        and float(
            ENDPOINT_POLICY[
                'minimum_deepest_extent_cumulative_outward_progress_m'
            ]
        ) == 0.00065
        and FLOOR_SIGNED_RANGE_M == (-0.0001, 0.00025)
        and MINIMUM_FORCE_N == (1.031, 0.913)
        and ENDPOINT_POLICY.get('require_fresh_bilateral_exact_target_pressure')
        is True
        and ENDPOINT_POLICY.get('require_base_stable') is True
        and ENDPOINT_POLICY.get('require_non_target_scene_stable') is True
    )
    audit_valid = bool(
        AUDIT.get('collision_free') is True
        and AUDIT.get('zero_command_state_only') is True
        and AUDIT.get('current_geometry_revalidation_required') is True
        and int(AUDIT.get('asset_count', 0)) == 13
        and int(AUDIT.get('robot_collision_bundle_file_count', 0)) == 24
        and int(AUDIT.get('dense_joint_samples', 0)) == 8
        and tuple(AUDIT.get('stationary_world_delta_m', ())) == (0.0, 0.0, 0.0)
        and min(
            float(AUDIT.get(
                'minimum_nonadjacent_robot_aabb_clearance_m', math.nan
            )),
            float(AUDIT.get(
                'minimum_15mm_padded_payload_robot_aabb_clearance_m', math.nan
            )),
            float(AUDIT.get(
                'minimum_closed_gripper_shelf_clearance_after_tolerance_m',
                math.nan,
            )),
        ) > 0.0
    )
    valid = bool(
        CERTIFICATE_KIND
        == 'read_only_seed101_outward_scaleup_settle_resample_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and route_valid
        and checkpoint_valid
        and execution_valid
        and endpoint_valid
        and audit_valid
    )
    return valid, {
        'route_rows': float(len(ROUTE_Q8)),
        'command_rows': float(len(COMMAND_Q8)),
        'settle_duration_s': SETTLE_DURATION_S,
        'minimum_audit_margin_m': min(
            float(AUDIT['minimum_nonadjacent_robot_aabb_clearance_m']),
            float(AUDIT['minimum_15mm_padded_payload_robot_aabb_clearance_m']),
            float(
                AUDIT[
                    'minimum_closed_gripper_shelf_clearance_after_tolerance_m'
                ]
            ),
        ),
    }


__all__ = (
    'AUDIT',
    'CERTIFICATE',
    'CERTIFICATE_KIND',
    'CERTIFICATE_SEMANTIC_DIGEST',
    'CERTIFICATE_SOURCE_JSON_SHA256',
    'CHECKPOINT',
    'CHECKPOINT_BASE_QUATERNION_XYZW',
    'CHECKPOINT_BASE_WORLD_POSITION_M',
    'CHECKPOINT_BASE_WORLD_YAW_RAD',
    'CHECKPOINT_BOOK_POSITION_WORLD_M',
    'CHECKPOINT_BOOK_QUATERNION_XYZW',
    'CHECKPOINT_GRIPPER_MASTER_M',
    'CHECKPOINT_HEAD_Q2',
    'CHECKPOINT_LEFT_Q8',
    'CHECKPOINT_RIGHT_Q7',
    'COMMAND_Q8',
    'COMPARISON_REFERENCE',
    'ENDPOINT_POLICY',
    'EXECUTION_POLICY',
    'FLOOR_SIGNED_RANGE_M',
    'MINIMUM_FORCE_N',
    'PRIOR_PRESSURE_EVIDENCE',
    'REFERENCE_BOOK_POSITION_WORLD_M',
    'REFERENCE_BOOK_QUATERNION_XYZW',
    'REFERENCE_PLANNED_WORLD_DELTA_M',
    'ROUTE',
    'ROUTE_Q8',
    'ROUTE_Q8_SHA256',
    'SETTLE_DURATION_S',
    'SETTLE_ROTATION_GROWTH_LIMIT_RAD',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
