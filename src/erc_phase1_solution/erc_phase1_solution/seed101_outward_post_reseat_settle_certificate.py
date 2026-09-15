"""Inert certificate for zero-command post-reseat settling and verification.

The certificate binds the exact paused state left by the failed reverse-reseat
endpoint gate.  Its sole route row is a checkpoint, so it authorizes no joint,
gripper, base, head, or entity-pose command.  A live caller may establish fresh
feedback and exact-target pressure monitoring, resume physics for at least
0.40 s, repause, and apply the embedded fail-closed stationary-state policy.
Importing this module performs no external operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    '68085dc223002e29c9c0802cf518dee1f77dca11d19c519f44c547afa9d3f8f5'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    'af1f8466f55e3b140ca0333a31b31cc9db14ab0438aeec19a4783ad674b42d89'
)
CERTIFICATE_JSON = r'''{
  "attached_book_corners_padding_15mm_in_grasp": [
    [0.05979776529546938, 0.02996295613951966, -0.13594799664501234],
    [0.22573488795772992, 0.0281828777588545, -0.04341852759039904],
    [0.05930620490610954, -0.030034410538734096, -0.13622068705330595],
    [0.22524332756836998, -0.031814488919399246, -0.043691217998692466],
    [-0.07656738998873869, 0.029968712352647642, 0.1086016800899607],
    [0.08936973267352175, 0.02818863397198251, 0.2011311491445742],
    [-0.07705895037809862, -0.0300286543256061, 0.10832898968166728],
    [0.08887817228416191, -0.03180873270627125, 0.20085845873628055]
  ],
  "attached_book_corners_physical_in_grasp": [
    [0.0654698981797365, 0.014823390334779923, -0.11561033562445316],
    [0.2052064225269034, 0.013324376961588216, -0.037690782736357675],
    [0.06522411798505659, -0.015175293004346961, -0.11574668082859996],
    [0.2049606423322235, -0.016674306377538667, -0.03782712794050449],
    [-0.056284704752592216, 0.01482852981078707, 0.10273759003177273],
    [0.0834518195945747, 0.01332951643759536, 0.18065714291986817],
    [-0.056530484947272135, -0.015170153528339817, 0.10260124482762592],
    [0.0832060393998948, -0.016669166901531522, 0.18052079771572138]
  ],
  "audit": {
    "asset_count": 13,
    "attached_corners_recomputed_from_current_checkpoint": true,
    "attached_corners_source": "exact_book_pose_base_pose_left_q8_and_tiago_urdf_fk",
    "collision_free": true,
    "current_geometry_revalidation_required": true,
    "dense_joint_samples": 8,
    "inherited_collision_audit_applies_to_stationary_checkpoint_only": true,
    "inherited_from_certificate_kind": "read_only_seed101_outward_reverse_reseat_certificate",
    "inherited_source_json_sha256": "ce6975263734cbb5cde58d2466ecb67686bc08bf1810ff7ca1ce14879b72211f",
    "inherited_source_semantic_digest": "e66fcf5bdf648ac1f9a370d92043b63faf145463e81f4e48faf46ae8fa320527",
    "inherited_source_route_q8_sha256": "bab2900d3c1c74b1dad9d94c490498d2b925bf538be07fb2fba86de7bd483fba",
    "maximum_15mm_padded_corner_change_from_prior_attachment_m": 0.00023346430696964435,
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
    "base_quaternion_xyzw": [4.9649765503099407e-08, 7.9098610137010558e-08, -0.382572923442866, 0.92392529906284648],
    "base_world_position_m": [2.0026561063355421, -0.14951695526078862, -7.2401578769641856e-08],
    "base_world_yaw_rad": -0.7851589413319934,
    "book_floor_signed_distance_m": -6.796354348637124e-06,
    "book_physical_bounds_world_m": [
      [2.81019889910172, -0.1689583865527535, 1.4518516705086513],
      [2.9725912773849252, -0.13721715805085302, 1.703177232325387]
    ],
    "book_position_world_m": [2.8913950882433226, -0.15308777230180326, 1.5775144514170192],
    "book_quaternion_xyzw": [-0.0038177587397655315, 0.71003865841669134, 0.0037855082890519117, 0.70414217186518768],
    "gripper_master_m": 0.0290081304296433,
    "head_q2_measured": [9.39826583820943e-12, -2.0856175280534895e-12],
    "left_q8_measured": [0.3499999965486979, -0.30951489249886954, 0.6178794341067166, 0.08205576832470303, -1.4824724103134208, 0.1737299607249708, 1.038614878890176, 0.21480386196601237],
    "right_q7_measured": [7.034048053706401e-08, 4.423280813918232e-06, 5.8105495467438036e-08, -7.514830644261804e-06, 1.7459349433664546e-09, 1.166947670158128e-10, -1.0919903091720797e-10],
    "target_model": "book_col_3_row_2_red",
    "world_iterations": 172142,
    "world_paused": true,
    "world_sim_time_s": 344.284
  },
  "checkpoint_stable_reference_evidence": {
    "absolute_rotation_rad": 0.0003132162351817864,
    "absolute_yaw_component_rad": 0.00015708376298213892,
    "maximum_x_error_m": 0.00023998376269451782,
    "position_error_m": 0.00020608673087378054
  },
  "endpoint_policy": {
    "floor_signed_max_m": 0.00025,
    "floor_signed_min_m": -0.0001,
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0008,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.0008,
    "maximum_stable_reference_maximum_x_error_m": 0.00030,
    "maximum_stable_reference_position_error_m": 0.00025,
    "minimum_left_force_n": 1.03591172618656,
    "minimum_right_force_n": 0.905239502936712,
    "require_base_stable": true,
    "require_fresh_bilateral_exact_target_pressure": true,
    "require_gripper_stable": true,
    "require_head_stable": true,
    "require_non_target_scene_stable": true,
    "require_right_arm_stable": true
  },
  "execution_policy": {
    "base_command_count": 0,
    "gazebo_entity_pose_mutation_allowed": false,
    "gripper_command_count": 0,
    "head_command_count": 0,
    "left_arm_command_count": 0,
    "maximum_continuous_absolute_rotation_rad": 0.0012,
    "maximum_continuous_absolute_yaw_rad": 0.0012,
    "maximum_rotation_growth_during_settle_rad": 0.00005,
    "maximum_short_dwell_rotation_change_rad": 0.000075,
    "maximum_short_dwell_translation_change_m": 0.00005,
    "maximum_startup_rotation_change_rad": 0.0002,
    "maximum_startup_translation_change_m": 0.00005,
    "repause_after_settle": true,
    "right_arm_command_count": 0,
    "settle_duration_s": 0.4,
    "settle_stability_rotation_limit_rad": 0.000075,
    "settle_stability_translation_limit_m": 0.00005,
    "trajectory_command_rows": 0,
    "world_resume_allowed_for_settle_only": true
  },
  "kind": "read_only_seed101_outward_post_reseat_settle_certificate",
  "pressure_reference": {
    "left_force_n": 2.07182345237312,
    "left_samples": 64,
    "measured_width_m": 0.02900826371252373,
    "minimum_each_side_retention_fraction": 0.5,
    "reason": "bilateral_contact_verified",
    "right_force_n": 1.810479005873424,
    "right_samples": 64,
    "source": "seed101_outward_reverse_reseat_certificate",
    "stage": "resume_transport_lock",
    "target_model": "book_col_3_row_2_red"
  },
  "route": [
    {
      "minimum_duration_s": 0.0,
      "phase": "checkpoint",
      "q8": [0.3499999965486979, -0.30951489249886954, 0.6178794341067166, 0.08205576832470303, -1.4824724103134208, 0.1737299607249708, 1.038614878890176, 0.21480386196601237],
      "row": 0,
      "world_delta_m": [0.0, 0.0, 0.0]
    }
  ],
  "route_q8_sha256": "5b07d21689f8d1f2c98c4de7874f3bcfbf099900f69eed1f262c194fada006d9",
  "stable_reference": {
    "book_maximum_world_x_m": 2.9728312611476198,
    "book_physical_bounds_world_m": [
      [2.810367191479608, -0.16897240693940888, 1.4518581475789596],
      [2.9728312611476198, -0.137209784700309, 1.703226910152558]
    ],
    "book_position_world_m": [2.891599226313614, -0.15309109581985894, 1.5775425288657587],
    "book_quaternion_xyzw": [-0.003867323503119562, 0.7101335529054696, 0.003847067873253381, 0.704045865633451],
    "source": "seed101_outward_reverse_reseat_certificate.stable_reference",
    "support_floor_world_z_m": 1.451858466863
  },
  "stage": "outward-post-reseat-settle"
}'''

CERTIFICATE: Mapping[str, object] = json.loads(CERTIFICATE_JSON)
CERTIFICATE_KIND = str(CERTIFICATE['kind'])
ROUTE_Q8_SHA256 = str(CERTIFICATE['route_q8_sha256'])
CHECKPOINT = CERTIFICATE['checkpoint']
STABLE_REFERENCE = CERTIFICATE['stable_reference']
CHECKPOINT_STABLE_REFERENCE_EVIDENCE = CERTIFICATE[
    'checkpoint_stable_reference_evidence'
]
AUDIT = CERTIFICATE['audit']
ENDPOINT_POLICY = CERTIFICATE['endpoint_policy']
EXECUTION_POLICY = CERTIFICATE['execution_policy']
PRESSURE_REFERENCE = CERTIFICATE['pressure_reference']
STAGE = str(CERTIFICATE['stage'])

ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP = tuple(
    tuple(row) for row in CERTIFICATE['attached_book_corners_physical_in_grasp']
)
ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP = tuple(
    tuple(row)
    for row in CERTIFICATE['attached_book_corners_padding_15mm_in_grasp']
)
CHECKPOINT_BASE_WORLD_POSITION_M = tuple(CHECKPOINT['base_world_position_m'])
CHECKPOINT_BASE_QUATERNION_XYZW = tuple(CHECKPOINT['base_quaternion_xyzw'])
CHECKPOINT_BASE_WORLD_YAW_RAD = float(CHECKPOINT['base_world_yaw_rad'])
CHECKPOINT_BOOK_POSITION_WORLD_M = tuple(CHECKPOINT['book_position_world_m'])
CHECKPOINT_BOOK_QUATERNION_XYZW = tuple(CHECKPOINT['book_quaternion_xyzw'])
CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M = tuple(
    tuple(row) for row in CHECKPOINT['book_physical_bounds_world_m']
)
CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M = float(
    CHECKPOINT['book_floor_signed_distance_m']
)
CHECKPOINT_LEFT_Q8 = tuple(CHECKPOINT['left_q8_measured'])
CHECKPOINT_RIGHT_Q7 = tuple(CHECKPOINT['right_q7_measured'])
CHECKPOINT_HEAD_Q2 = tuple(CHECKPOINT['head_q2_measured'])
CHECKPOINT_GRIPPER_MASTER_M = float(CHECKPOINT['gripper_master_m'])
STABLE_REFERENCE_BOOK_POSITION_WORLD_M = tuple(
    STABLE_REFERENCE['book_position_world_m']
)
STABLE_REFERENCE_BOOK_QUATERNION_XYZW = tuple(
    STABLE_REFERENCE['book_quaternion_xyzw']
)
STABLE_REFERENCE_BOOK_PHYSICAL_BOUNDS_WORLD_M = tuple(
    tuple(row) for row in STABLE_REFERENCE['book_physical_bounds_world_m']
)
SUPPORT_FLOOR_WORLD_Z_M = float(STABLE_REFERENCE['support_floor_world_z_m'])
SETTLE_DURATION_S = float(EXECUTION_POLICY['settle_duration_s'])
SETTLE_STABILITY_TRANSLATION_LIMIT_M = float(
    EXECUTION_POLICY['settle_stability_translation_limit_m']
)
SETTLE_STABILITY_ROTATION_LIMIT_RAD = float(
    EXECUTION_POLICY['settle_stability_rotation_limit_rad']
)
SETTLE_ROTATION_GROWTH_LIMIT_RAD = float(
    EXECUTION_POLICY['maximum_rotation_growth_during_settle_rad']
)
SETTLE_STABILITY_ROTATION_LIMIT_RAD = float(
    EXECUTION_POLICY['maximum_short_dwell_rotation_change_rad']
)
SETTLE_STABILITY_TRANSLATION_LIMIT_M = float(
    EXECUTION_POLICY['maximum_short_dwell_translation_change_m']
)
STARTUP_STABILITY_ROTATION_LIMIT_RAD = float(
    EXECUTION_POLICY['maximum_startup_rotation_change_rad']
)
STARTUP_STABILITY_TRANSLATION_LIMIT_M = float(
    EXECUTION_POLICY['maximum_startup_translation_change_m']
)
FLOOR_SIGNED_RANGE_M = (
    float(ENDPOINT_POLICY['floor_signed_min_m']),
    float(ENDPOINT_POLICY['floor_signed_max_m']),
)
MINIMUM_FORCE_N = (
    float(ENDPOINT_POLICY['minimum_left_force_n']),
    float(ENDPOINT_POLICY['minimum_right_force_n']),
)
ROUTE = tuple(CERTIFICATE['route'])
ROUTE_Q8 = tuple(tuple(row['q8']) for row in ROUTE)
COMMAND_Q8 = ROUTE_Q8[1:]


def _canonical_certificate_bytes() -> bytes:
    return json.dumps(
        CERTIFICATE, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode('utf-8')


def semantic_digest() -> str:
    return hashlib.sha256(_canonical_certificate_bytes()).hexdigest()


def source_json_digest() -> str:
    return hashlib.sha256(CERTIFICATE_JSON.encode('utf-8')).hexdigest()


def route_q8_digest() -> str:
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
    """Fail closed if the checkpoint, geometry, no-command policy, or gates change."""

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
        and int(CHECKPOINT.get('world_iterations', -1)) == 172142
        and float(CHECKPOINT.get('world_sim_time_s', math.nan)) == 344.284
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
        and len(CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M) == 2
        and all(
            _finite_vector(row, 3)
            for row in CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M
        )
        and CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M
        == -6.796354348637124e-06
    )
    reference_valid = bool(
        _finite_vector(STABLE_REFERENCE_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(STABLE_REFERENCE_BOOK_QUATERNION_XYZW, 4)
        and len(STABLE_REFERENCE_BOOK_PHYSICAL_BOUNDS_WORLD_M) == 2
        and all(
            _finite_vector(row, 3)
            for row in STABLE_REFERENCE_BOOK_PHYSICAL_BOUNDS_WORLD_M
        )
        and SUPPORT_FLOOR_WORLD_Z_M == 1.451858466863
        and math.isclose(
            min(row[2] for row in CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M)
            - CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M,
            SUPPORT_FLOOR_WORLD_Z_M,
            rel_tol=0.0,
            abs_tol=5e-16,
        )
        and float(CHECKPOINT_STABLE_REFERENCE_EVIDENCE['position_error_m'])
        <= float(ENDPOINT_POLICY['maximum_stable_reference_position_error_m'])
        and float(CHECKPOINT_STABLE_REFERENCE_EVIDENCE['maximum_x_error_m'])
        <= float(
            ENDPOINT_POLICY['maximum_stable_reference_maximum_x_error_m']
        )
        and float(CHECKPOINT_STABLE_REFERENCE_EVIDENCE['absolute_rotation_rad'])
        <= float(
            ENDPOINT_POLICY['maximum_absolute_rotation_to_stable_reference_rad']
        )
        and float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE[
                'absolute_yaw_component_rad'
            ]
        )
        <= float(
            ENDPOINT_POLICY['maximum_absolute_yaw_to_stable_reference_rad']
        )
    )
    corners_valid = bool(
        all(
            len(corners) == 8 and all(_finite_vector(row, 3) for row in corners)
            for corners in (
                ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP,
                ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP,
            )
        )
        and AUDIT.get('attached_corners_recomputed_from_current_checkpoint')
        is True
        and AUDIT.get('attached_corners_source')
        == 'exact_book_pose_base_pose_left_q8_and_tiago_urdf_fk'
    )
    execution_valid = bool(
        STAGE == 'outward-post-reseat-settle'
        and SETTLE_DURATION_S == 0.4
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
        and SETTLE_STABILITY_TRANSLATION_LIMIT_M == 0.00005
        and SETTLE_STABILITY_ROTATION_LIMIT_RAD == 0.000075
        and float(
            EXECUTION_POLICY['maximum_continuous_absolute_rotation_rad']
        ) == 0.0012
        and float(
            EXECUTION_POLICY['maximum_continuous_absolute_yaw_rad']
        ) == 0.0012
        and SETTLE_ROTATION_GROWTH_LIMIT_RAD == 0.00005
        and SETTLE_STABILITY_ROTATION_LIMIT_RAD == 0.000075
        and SETTLE_STABILITY_TRANSLATION_LIMIT_M == 0.00005
        and STARTUP_STABILITY_ROTATION_LIMIT_RAD == 0.0002
        and STARTUP_STABILITY_TRANSLATION_LIMIT_M == 0.00005
    )
    endpoint_valid = bool(
        float(
            ENDPOINT_POLICY['maximum_stable_reference_position_error_m']
        ) == 0.00025
        and float(
            ENDPOINT_POLICY['maximum_stable_reference_maximum_x_error_m']
        ) == 0.00030
        and float(
            ENDPOINT_POLICY['maximum_absolute_rotation_to_stable_reference_rad']
        ) == 0.0008
        and float(
            ENDPOINT_POLICY['maximum_absolute_yaw_to_stable_reference_rad']
        ) == 0.0008
        and FLOOR_SIGNED_RANGE_M == (-0.0001, 0.00025)
        and MINIMUM_FORCE_N == (1.03591172618656, 0.905239502936712)
        and ENDPOINT_POLICY.get('require_fresh_bilateral_exact_target_pressure')
        is True
        and all(
            ENDPOINT_POLICY.get(key) is True
            for key in (
                'require_base_stable',
                'require_gripper_stable',
                'require_head_stable',
                'require_non_target_scene_stable',
                'require_right_arm_stable',
            )
        )
    )
    pressure_valid = bool(
        PRESSURE_REFERENCE.get('target_model') == 'book_col_3_row_2_red'
        and PRESSURE_REFERENCE.get('reason') == 'bilateral_contact_verified'
        and PRESSURE_REFERENCE.get('source')
        == 'seed101_outward_reverse_reseat_certificate'
        and int(PRESSURE_REFERENCE.get('left_samples', 0)) == 64
        and int(PRESSURE_REFERENCE.get('right_samples', 0)) == 64
        and float(PRESSURE_REFERENCE['minimum_each_side_retention_fraction'])
        == 0.5
        and MINIMUM_FORCE_N == (
            0.5 * float(PRESSURE_REFERENCE['left_force_n']),
            0.5 * float(PRESSURE_REFERENCE['right_force_n']),
        )
    )
    audit_valid = bool(
        AUDIT.get('collision_free') is True
        and AUDIT.get('zero_command_state_only') is True
        and AUDIT.get('current_geometry_revalidation_required') is True
        and AUDIT.get(
            'inherited_collision_audit_applies_to_stationary_checkpoint_only'
        ) is True
        and tuple(AUDIT.get('stationary_world_delta_m', ()))
        == (0.0, 0.0, 0.0)
        and int(AUDIT.get('asset_count', 0)) == 13
        and int(AUDIT.get('robot_collision_bundle_file_count', 0)) == 24
        and int(AUDIT.get('dense_joint_samples', 0)) == 8
        and min(
            float(AUDIT['minimum_nonadjacent_robot_aabb_clearance_m']),
            float(AUDIT['minimum_15mm_padded_payload_robot_aabb_clearance_m']),
            float(
                AUDIT[
                    'minimum_closed_gripper_shelf_clearance_after_tolerance_m'
                ]
            ),
        ) > 0.0
    )
    valid = bool(
        CERTIFICATE_KIND
        == 'read_only_seed101_outward_post_reseat_settle_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and route_valid
        and checkpoint_valid
        and reference_valid
        and corners_valid
        and execution_valid
        and endpoint_valid
        and pressure_valid
        and audit_valid
    )
    return valid, {
        'route_rows': float(len(ROUTE_Q8)),
        'command_rows': float(len(COMMAND_Q8)),
        'settle_duration_s': SETTLE_DURATION_S,
        'checkpoint_stable_position_error_m': float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE['position_error_m']
        ),
        'checkpoint_stable_maximum_x_error_m': float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE['maximum_x_error_m']
        ),
        'checkpoint_stable_rotation_rad': float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE['absolute_rotation_rad']
        ),
        'checkpoint_stable_yaw_component_rad': float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE[
                'absolute_yaw_component_rad'
            ]
        ),
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
    'ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP',
    'ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP',
    'AUDIT',
    'CERTIFICATE',
    'CERTIFICATE_KIND',
    'CERTIFICATE_SEMANTIC_DIGEST',
    'CERTIFICATE_SOURCE_JSON_SHA256',
    'CHECKPOINT',
    'CHECKPOINT_BASE_QUATERNION_XYZW',
    'CHECKPOINT_BASE_WORLD_POSITION_M',
    'CHECKPOINT_BASE_WORLD_YAW_RAD',
    'CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M',
    'CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M',
    'CHECKPOINT_BOOK_POSITION_WORLD_M',
    'CHECKPOINT_BOOK_QUATERNION_XYZW',
    'CHECKPOINT_GRIPPER_MASTER_M',
    'CHECKPOINT_HEAD_Q2',
    'CHECKPOINT_LEFT_Q8',
    'CHECKPOINT_RIGHT_Q7',
    'CHECKPOINT_STABLE_REFERENCE_EVIDENCE',
    'COMMAND_Q8',
    'ENDPOINT_POLICY',
    'EXECUTION_POLICY',
    'FLOOR_SIGNED_RANGE_M',
    'MINIMUM_FORCE_N',
    'PRESSURE_REFERENCE',
    'ROUTE',
    'ROUTE_Q8',
    'ROUTE_Q8_SHA256',
    'SETTLE_DURATION_S',
    'SETTLE_ROTATION_GROWTH_LIMIT_RAD',
    'SETTLE_STABILITY_ROTATION_LIMIT_RAD',
    'SETTLE_STABILITY_TRANSLATION_LIMIT_M',
    'STARTUP_STABILITY_ROTATION_LIMIT_RAD',
    'STARTUP_STABILITY_TRANSLATION_LIMIT_M',
    'SETTLE_STABILITY_ROTATION_LIMIT_RAD',
    'SETTLE_STABILITY_TRANSLATION_LIMIT_M',
    'STABLE_REFERENCE',
    'STABLE_REFERENCE_BOOK_PHYSICAL_BOUNDS_WORLD_M',
    'STABLE_REFERENCE_BOOK_POSITION_WORLD_M',
    'STABLE_REFERENCE_BOOK_QUATERNION_XYZW',
    'STAGE',
    'SUPPORT_FLOOR_WORLD_Z_M',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
