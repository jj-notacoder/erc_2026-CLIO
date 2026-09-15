"""Inert certificate for one seed-101 inward reverse-reseat leg.

The two-row route starts at the exact paused state left by the failed
settle/resample and returns 1 mm in world +X to the previously certified
stable outward-scale-up-continuation checkpoint.  It commands only the left
arm, keeps the current gripper pressure, and fixes hand orientation and torso.
Importing this module performs no ROS, Gazebo, controller, filesystem, or
network operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    'ce6975263734cbb5cde58d2466ecb67686bc08bf1810ff7ca1ce14879b72211f'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    'e66fcf5bdf648ac1f9a370d92043b63faf145463e81f4e48faf46ae8fa320527'
)
CERTIFICATE_JSON = r'''{
  "attached_book_corners_padding_15mm_in_grasp": [
    [0.060014259002197125, 0.029927536090246835, -0.13589033895440072],
    [0.2259259284762027, 0.02809600567485028, -0.04331624199047786],
    [0.05950900184834385, -0.030069676167643656, -0.1361718287235101],
    [0.2254206713223494, -0.03190120658304023, -0.04359773175958724],
    [-0.0764170501746636, 0.02992928700441636, 0.10862243754863231],
    [0.08949461929934194, 0.028097756589019775, 0.20119653451255518],
    [-0.07692230732851688, -0.03006792525347416, 0.10834094777952293],
    [0.08898936214548867, -0.03189945566887072, 0.2009150447434458]
  ],
  "attached_book_corners_physical_in_grasp": [
    [0.06567741427457541, 0.014783732318268907, -0.11555333131618704],
    [0.20539250435794856, 0.013241390915829692, -0.03759619703077835],
    [0.06542478569764897, -0.015214873810676351, -0.11569407620074164],
    [0.20513987578102172, -0.01675721521311557, -0.03773694191533314],
    [-0.056136254633335936, 0.014785295634491687, 0.1027616477043782],
    [0.08357883545003683, 0.013242954232052469, 0.1807187819897867],
    [-0.056388883210262766, -0.015213310494453574, 0.1026209028198234],
    [0.0833262068731104, -0.016755651896892793, 0.1805780371052321]
  ],
  "audit": {
    "asset_count": 13,
    "certified_reverse_start_q8": [0.3499998174377888, -0.30909607616392276, 0.6175535922346728, 0.08219232062926038, -1.4857090774553214, 0.17395264707175168, 1.0415889282352255, 0.21481052701719683],
    "collision_free": true,
    "current_geometry_revalidation_required": true,
    "current_to_certified_reverse_start_maximum_joint_error_rad": 2.536561498067691e-06,
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
    "reverse_of_source_route_rows": [1, 0],
    "robot_collision_bundle_file_count": 24,
    "shelf_collision_tolerance_m": 0.00075
  },
  "checkpoint": {
    "base_quaternion_xyzw": [1.8196506007855779e-07, 3.006148335564496e-07, -0.3826074383522628, 0.923911006600417],
    "base_world_position_m": [2.0026729276880491, -0.14949948904604238, -3.977426593568196e-07],
    "base_world_yaw_rad": -0.7852336555462195,
    "book_position_world_m": [2.890661892167218, -0.15317035347670219, 1.5775238509060145],
    "book_quaternion_xyzw": [-0.004188475048371351, 0.71005656164804254, 0.004174723293775891, 0.70411981055978623],
    "gripper_master_m": 0.029008297335297823,
    "head_q2_measured": [4.837166206112712e-12, -2.158367100045046e-12],
    "left_q8_measured": [0.34999953751554436, -0.30909861272542083, 0.6175553554062443, 0.08219275225689975, -1.4857088139359278, 0.17395238931347662, 1.0415889844466004, 0.2148105279821962],
    "right_q7_measured": [1.0360774759655906e-05, -1.0042776704687532e-05, 8.783570150086181e-06, -6.745791004783519e-06, 2.4298609371527956e-07, 5.51294518680262e-09, -2.491546018600574e-09],
    "target_model": "book_col_3_row_2_red",
    "world_iterations": 169708,
    "world_paused": true,
    "world_sim_time_s": 339.416
  },
  "endpoint_policy": {
    "floor_signed_max_m": 0.00025,
    "floor_signed_min_m": -0.0001,
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0008,
    "maximum_absolute_rotation_growth_from_start_rad": 0.00005,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.0008,
    "maximum_book_from_hand_translation_change_m": 0.00018,
    "maximum_cross_track_motion_m": 0.00018,
    "maximum_incremental_rotation_rad": 0.0012,
    "maximum_reference_depth_overshoot_m": 0.00015,
    "maximum_stable_reference_position_error_m": 0.00025,
    "minimum_book_center_cumulative_inward_progress_m": 0.0007,
    "minimum_book_maximum_x_cumulative_inward_return_m": 0.00065,
    "minimum_each_side_force_retention_fraction": 0.5,
    "pause_after_endpoint": true,
    "require_fresh_bilateral_exact_target_pressure": true
  },
  "execution_policy": {
    "base_motion_allowed": false,
    "fixed_hand_orientation": true,
    "gazebo_entity_pose_mutation_allowed": false,
    "head_motion_allowed": false,
    "keep_current_gripper_pressure": true,
    "left_arm_only": true,
    "maximum_continuous_absolute_rotation_rad": 0.0012,
    "maximum_continuous_absolute_yaw_rad": 0.0012,
    "minimum_duration_s": 2.5,
    "pause_after_endpoint": true,
    "planned_world_delta_m": [0.001, 0.0, 0.0],
    "right_arm_motion_allowed": false,
    "torso_fixed": true
  },
  "kind": "read_only_seed101_outward_reverse_reseat_certificate",
  "pressure_reference": {
    "left_force_n": 2.07182345237312,
    "left_samples": 64,
    "measured_width_m": 0.02900826371252373,
    "reason": "bilateral_contact_verified",
    "right_force_n": 1.810479005873424,
    "right_samples": 64,
    "stage": "resume_transport_lock",
    "target_model": "book_col_3_row_2_red"
  },
  "route": [
    {
      "minimum_duration_s": 0.0,
      "phase": "checkpoint",
      "q8": [0.34999953751554436, -0.30909861272542083, 0.6175553554062443, 0.08219275225689975, -1.4857088139359278, 0.17395238931347662, 1.0415889844466004, 0.2148105279821962],
      "row": 0,
      "world_delta_m": [0.0, 0.0, 0.0]
    },
    {
      "maximum_absolute_rotation_to_stable_reference_rad": 0.0008,
      "maximum_book_from_hand_translation_change_m": 0.00018,
      "maximum_incremental_rotation_rad": 0.0012,
      "minimum_book_center_cumulative_inward_progress_m": 0.0007,
      "minimum_book_maximum_x_cumulative_inward_return_m": 0.00065,
      "minimum_duration_s": 2.5,
      "phase": "shelf_inward",
      "q8": [0.34999953751554436, -0.3095148967058795, 0.6178794379281761, 0.08205575594538433, -1.4824722381676605, 0.17372994976888595, 1.0386147095992404, 0.2148038614518491],
      "row": 1,
      "world_delta_m": [0.001, 0.0, 0.0]
    }
  ],
  "route_q8_sha256": "bab2900d3c1c74b1dad9d94c490498d2b925bf538be07fb2fba86de7bd483fba",
  "stage": "outward-reverse-reseat",
  "stable_reference": {
    "book_position_world_m": [2.891599226313614, -0.15309109581985894, 1.5775425288657587],
    "book_quaternion_xyzw": [-0.003867323503119562, 0.7101335529054696, 0.003847067873253381, 0.704045865633451],
    "left_q8": [0.34999953751554436, -0.3095148967058795, 0.6178794379281761, 0.08205575594538433, -1.4824722381676605, 0.17372994976888595, 1.0386147095992404, 0.2148038614518491],
    "source": "outward_scaleup_continue_checkpoint"
  }
}'''

CERTIFICATE: Mapping[str, object] = json.loads(CERTIFICATE_JSON)
CERTIFICATE_KIND = str(CERTIFICATE['kind'])
ROUTE_Q8_SHA256 = str(CERTIFICATE['route_q8_sha256'])
CHECKPOINT = CERTIFICATE['checkpoint']
STABLE_REFERENCE = CERTIFICATE['stable_reference']
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
TARGET_LEFT_Q8 = tuple(STABLE_REFERENCE['left_q8'])
PLANNED_WORLD_DELTA_M = tuple(EXECUTION_POLICY['planned_world_delta_m'])
MINIMUM_DURATION_S = float(EXECUTION_POLICY['minimum_duration_s'])
FLOOR_SIGNED_RANGE_M = (
    float(ENDPOINT_POLICY['floor_signed_min_m']),
    float(ENDPOINT_POLICY['floor_signed_max_m']),
)
MINIMUM_FORCE_RETENTION_FRACTION = float(
    ENDPOINT_POLICY['minimum_each_side_force_retention_fraction']
)
REFERENCE_PRESSURE_N = (
    float(PRESSURE_REFERENCE['left_force_n']),
    float(PRESSURE_REFERENCE['right_force_n']),
)
MINIMUM_FORCE_N = tuple(
    force * MINIMUM_FORCE_RETENTION_FRACTION for force in REFERENCE_PRESSURE_N
)
MINIMUM_RETAINED_FORCE_N = MINIMUM_FORCE_N
ROUTE = tuple(CERTIFICATE['route'])
ROUTE_Q8 = tuple(tuple(row['q8']) for row in ROUTE)
REVERSE_RESEAT_Q8 = ROUTE_Q8[1]


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
    """Fail closed if any checkpoint, reverse leg, audit, or gate changes."""

    route_valid = bool(
        len(ROUTE) == 2
        and all(
            isinstance(row, Mapping)
            and int(row.get('row', -1)) == index
            and _finite_vector(row.get('q8'), 8)
            and _finite_vector(row.get('world_delta_m'), 3)
            for index, row in enumerate(ROUTE)
        )
        and ROUTE[0]['phase'] == 'checkpoint'
        and ROUTE[1]['phase'] == 'shelf_inward'
        and tuple(ROUTE[0]['world_delta_m']) == (0.0, 0.0, 0.0)
        and tuple(ROUTE[1]['world_delta_m']) == PLANNED_WORLD_DELTA_M
        and float(ROUTE[1]['minimum_duration_s']) == MINIMUM_DURATION_S
        and ROUTE_Q8[0] == CHECKPOINT_LEFT_Q8
        and REVERSE_RESEAT_Q8 == TARGET_LEFT_Q8
    )
    checkpoint_valid = bool(
        CHECKPOINT.get('world_paused') is True
        and int(CHECKPOINT.get('world_iterations', -1)) == 169708
        and float(CHECKPOINT.get('world_sim_time_s', math.nan)) == 339.416
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
        and _finite_vector(STABLE_REFERENCE_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(STABLE_REFERENCE_BOOK_QUATERNION_XYZW, 4)
        and _finite_vector(TARGET_LEFT_Q8, 8)
        and all(
            len(corners) == 8 and all(_finite_vector(row, 3) for row in corners)
            for corners in (
                ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP,
                ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP,
            )
        )
    )
    execution_valid = bool(
        STAGE == 'outward-reverse-reseat'
        and PLANNED_WORLD_DELTA_M == (0.001, 0.0, 0.0)
        and MINIMUM_DURATION_S == 2.5
        and EXECUTION_POLICY.get('fixed_hand_orientation') is True
        and EXECUTION_POLICY.get('torso_fixed') is True
        and EXECUTION_POLICY.get('left_arm_only') is True
        and EXECUTION_POLICY.get('keep_current_gripper_pressure') is True
        and EXECUTION_POLICY.get('pause_after_endpoint') is True
        and EXECUTION_POLICY.get('base_motion_allowed') is False
        and EXECUTION_POLICY.get('head_motion_allowed') is False
        and EXECUTION_POLICY.get('right_arm_motion_allowed') is False
        and EXECUTION_POLICY.get('gazebo_entity_pose_mutation_allowed') is False
        and float(
            EXECUTION_POLICY['maximum_continuous_absolute_rotation_rad']
        ) == 0.0012
        and float(
            EXECUTION_POLICY['maximum_continuous_absolute_yaw_rad']
        ) == 0.0012
        and math.isclose(
            REVERSE_RESEAT_Q8[0], CHECKPOINT_LEFT_Q8[0],
            rel_tol=0.0, abs_tol=5e-7,
        )
    )
    endpoint_valid = bool(
        float(
            ENDPOINT_POLICY['minimum_book_center_cumulative_inward_progress_m']
        ) == 0.0007
        and float(
            ENDPOINT_POLICY[
                'minimum_book_maximum_x_cumulative_inward_return_m'
            ]
        ) == 0.00065
        and float(
            ENDPOINT_POLICY['maximum_book_from_hand_translation_change_m']
        ) == 0.00018
        and float(ENDPOINT_POLICY['maximum_incremental_rotation_rad']) == 0.0012
        and float(
            ENDPOINT_POLICY['maximum_absolute_rotation_to_stable_reference_rad']
        ) == 0.0008
        and float(
            ENDPOINT_POLICY['maximum_absolute_yaw_to_stable_reference_rad']
        ) == 0.0008
        and float(
            ENDPOINT_POLICY['maximum_absolute_rotation_growth_from_start_rad']
        ) == 0.00005
        and float(ENDPOINT_POLICY['maximum_cross_track_motion_m']) == 0.00018
        and float(
            ENDPOINT_POLICY['maximum_reference_depth_overshoot_m']
        ) == 0.00015
        and float(
            ENDPOINT_POLICY['maximum_stable_reference_position_error_m']
        ) == 0.00025
        and FLOOR_SIGNED_RANGE_M == (-0.0001, 0.00025)
        and MINIMUM_FORCE_RETENTION_FRACTION == 0.5
        and ENDPOINT_POLICY.get('require_fresh_bilateral_exact_target_pressure')
        is True
        and ENDPOINT_POLICY.get('pause_after_endpoint') is True
    )
    certified_start_q8 = tuple(AUDIT['certified_reverse_start_q8'])
    start_error = max(
        abs(actual - certified)
        for actual, certified in zip(CHECKPOINT_LEFT_Q8, certified_start_q8)
    )
    audit_valid = bool(
        AUDIT.get('collision_free') is True
        and AUDIT.get('current_geometry_revalidation_required') is True
        and tuple(AUDIT.get('reverse_of_source_route_rows', ())) == (1, 0)
        and _finite_vector(certified_start_q8, 8)
        and math.isclose(
            start_error,
            float(AUDIT[
                'current_to_certified_reverse_start_maximum_joint_error_rad'
            ]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and start_error < 5e-6
        and int(AUDIT.get('asset_count', 0)) == 13
        and int(AUDIT.get('robot_collision_bundle_file_count', 0)) == 24
        and int(AUDIT.get('dense_joint_samples', 0)) == 8
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
    pressure_valid = bool(
        PRESSURE_REFERENCE.get('reason') == 'bilateral_contact_verified'
        and PRESSURE_REFERENCE.get('target_model') == 'book_col_3_row_2_red'
        and int(PRESSURE_REFERENCE.get('left_samples', 0)) == 64
        and int(PRESSURE_REFERENCE.get('right_samples', 0)) == 64
        and all(math.isfinite(force) and force > 0.0 for force in REFERENCE_PRESSURE_N)
        and all(force > 0.0 for force in MINIMUM_FORCE_N)
    )
    valid = bool(
        CERTIFICATE_KIND
        == 'read_only_seed101_outward_reverse_reseat_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and route_valid
        and checkpoint_valid
        and execution_valid
        and endpoint_valid
        and audit_valid
        and pressure_valid
    )
    return valid, {
        'route_rows': float(len(ROUTE_Q8)),
        'command_rows': float(len(ROUTE_Q8) - 1),
        'maximum_joint_delta_rad': max(
            abs(target - start)
            for start, target in zip(CHECKPOINT_LEFT_Q8, REVERSE_RESEAT_Q8)
        ),
        'certified_start_joint_error_rad': start_error,
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
    'CHECKPOINT_BOOK_POSITION_WORLD_M',
    'CHECKPOINT_BOOK_QUATERNION_XYZW',
    'CHECKPOINT_GRIPPER_MASTER_M',
    'CHECKPOINT_HEAD_Q2',
    'CHECKPOINT_LEFT_Q8',
    'CHECKPOINT_RIGHT_Q7',
    'ENDPOINT_POLICY',
    'EXECUTION_POLICY',
    'FLOOR_SIGNED_RANGE_M',
    'MINIMUM_DURATION_S',
    'MINIMUM_FORCE_N',
    'MINIMUM_FORCE_RETENTION_FRACTION',
    'MINIMUM_RETAINED_FORCE_N',
    'PLANNED_WORLD_DELTA_M',
    'PRESSURE_REFERENCE',
    'REFERENCE_PRESSURE_N',
    'REVERSE_RESEAT_Q8',
    'ROUTE',
    'ROUTE_Q8',
    'ROUTE_Q8_SHA256',
    'STABLE_REFERENCE',
    'STABLE_REFERENCE_BOOK_POSITION_WORLD_M',
    'STABLE_REFERENCE_BOOK_QUATERNION_XYZW',
    'TARGET_LEFT_Q8',
    'STAGE',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
