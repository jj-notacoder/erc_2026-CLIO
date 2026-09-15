"""Inert certificate for the final two seed-101 outward scale-up rungs."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    'c48b1d01d43bb2f461e4e4c83b42ad05037629f8c21afb6a5d886715d4e5be1c'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    '3f73080bd147c3b61e1060b3ac3348ef1d09d45dba77cd3cb7c4182619f0a1ee'
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
    "collision_free": true,
    "dense_joint_samples": 8,
    "maximum_dense_joint_increment_rad": 0.002,
    "minimum_15mm_padded_payload_robot_aabb_clearance_m": 0.09931032286966568,
    "minimum_carried_book_other_book_aabb_clearance_m": 0.25845650009185483,
    "minimum_closed_gripper_shelf_clearance_after_tolerance_m": 0.011773523189314156,
    "minimum_nonadjacent_robot_aabb_clearance_m": 0.002157269827602226,
    "minimum_robot_bin_aabb_clearance_m": 2.3313741399156047,
    "minimum_robot_other_book_aabb_clearance_m": 0.14121832644058888,
    "minimum_robot_table_aabb_clearance_m": 2.173460948331139,
    "shelf_collision_tolerance_m": 0.00075
  },
  "endpoint_policy": {
    "floor_signed_max_m": 0.00025,
    "floor_signed_min_m": -0.0001,
    "keep_current_gripper_pressure": true,
    "maximum_cumulative_rotation_rad": [0.0008, 0.0018],
    "maximum_cumulative_translation_change_m": [0.00018, 0.00032],
    "minimum_book_center_cumulative_outward_progress_m": [0.0007, 0.0028],
    "minimum_deepest_extent_cumulative_outward_progress_m": [0.00065, 0.0026],
    "minimum_each_side_force_retention_fraction": 0.5,
    "pause_after_every_endpoint": true,
    "require_fresh_bilateral_exact_target_pressure": true
  },
  "execution_policy": {
    "cumulative_outward_world_minus_x_m": [0.001, 0.004],
    "fixed_hand_orientation": true,
    "minimum_durations_s": [2.5, 4.5],
    "pause_after_every_endpoint": true,
    "torso_fixed": true
  },
  "input": {
    "base_quaternion_xyzw": [2.3714089973631522e-08, 2.7489961682582755e-08, -0.3826253888614286, 0.9239035727816184],
    "base_world_position_m": [2.0026896388252298, -0.1494827573688263, -3.5579804881440684e-07],
    "base_world_yaw_rad": -0.7852725133609783,
    "book_floor_signed_distance_m": -3.192840403176689e-07,
    "book_physical_bounds_world_m": [[2.810367191479608, -0.16897240693940888, 1.4518581475789596], [2.9728312611476198, -0.137209784700309, 1.703226910152558]],
    "book_position_world_m": [2.891599226313614, -0.15309109581985894, 1.5775425288657587],
    "book_quaternion_xyzw": [-0.003867323503119562, 0.7101335529054696, 0.003847067873253381, 0.704045865633451],
    "gripper_master_m": 0.029008128166538853,
    "head_q2_measured": [-5.341047622287229e-12, -2.104500678413703e-12],
    "left_q8_measured": [0.3499998174377888, -0.3095148967058795, 0.6178794379281761, 0.08205575594538433, -1.4824722381676605, 0.17372994976888595, 1.0386147095992404, 0.2148038614518491],
    "right_q7_measured": [1.1017502906726425e-06, 2.930440907271199e-06, 9.312227186970225e-07, -7.4406713585614e-06, 2.550846057202886e-08, 5.052158441872368e-10, -2.9108287519463966e-10]
  },
  "kind": "read_only_seed101_outward_scaleup_continue_certificate",
  "route": [
    {"minimum_duration_s": 0.0, "phase": "checkpoint", "q8": [0.3499998174377888, -0.3095148967058795, 0.6178794379281761, 0.08205575594538433, -1.4824722381676605, 0.17372994976888595, 1.0386147095992404, 0.2148038614518491], "row": 0, "world_delta_m": [0.0, 0.0, 0.0]},
    {"maximum_cumulative_rotation_rad": 0.0008, "maximum_cumulative_translation_change_m": 0.00018, "minimum_book_center_cumulative_outward_progress_m": 0.0007, "minimum_deepest_extent_cumulative_outward_progress_m": 0.00065, "minimum_duration_s": 2.5, "phase": "shelf_outward", "q8": [0.3499998174377888, -0.30909607616392276, 0.6175535922346728, 0.08219232062926038, -1.4857090774553214, 0.17395264707175168, 1.0415889282352255, 0.21481052701719683], "row": 1, "world_delta_m": [-0.001, 0.0, 0.0]},
    {"maximum_cumulative_rotation_rad": 0.0018, "maximum_cumulative_translation_change_m": 0.00032, "minimum_book_center_cumulative_outward_progress_m": 0.0028, "minimum_deepest_extent_cumulative_outward_progress_m": 0.0026, "minimum_duration_s": 4.5, "phase": "shelf_outward", "q8": [0.3499998174377888, -0.3078346417560695, 0.6166324597930262, 0.08260676597407056, -1.49534711034959, 0.17462952039489882, 1.0504967340163236, 0.21482665883406432], "row": 2, "world_delta_m": [-0.004, 0.0, 0.0]}
  ],
  "route_q8_sha256": "8a88bf78e9350e4fa2625735a61fb2f0aa2b6e4e037ed598df167384dfcfe7e3"
}
'''

CERTIFICATE: Mapping[str, object] = json.loads(CERTIFICATE_JSON)
CERTIFICATE_KIND = str(CERTIFICATE['kind'])
ROUTE_Q8_SHA256 = str(CERTIFICATE['route_q8_sha256'])
CHECKPOINT = CERTIFICATE['input']
AUDIT = CERTIFICATE['audit']
ENDPOINT_POLICY = CERTIFICATE['endpoint_policy']
EXECUTION_POLICY = CERTIFICATE['execution_policy']

CHECKPOINT_BASE_WORLD_POSITION_M = tuple(CHECKPOINT['base_world_position_m'])
CHECKPOINT_BASE_QUATERNION_XYZW = tuple(CHECKPOINT['base_quaternion_xyzw'])
CHECKPOINT_BASE_WORLD_YAW_RAD = float(CHECKPOINT['base_world_yaw_rad'])
CHECKPOINT_LEFT_Q8 = tuple(CHECKPOINT['left_q8_measured'])
CHECKPOINT_RIGHT_Q7 = tuple(CHECKPOINT['right_q7_measured'])
CHECKPOINT_HEAD_Q2 = tuple(CHECKPOINT['head_q2_measured'])
CHECKPOINT_BOOK_POSITION_WORLD_M = tuple(CHECKPOINT['book_position_world_m'])
CHECKPOINT_BOOK_QUATERNION_XYZW = tuple(CHECKPOINT['book_quaternion_xyzw'])
CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M = tuple(
    tuple(row) for row in CHECKPOINT['book_physical_bounds_world_m']
)
CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M = float(
    CHECKPOINT['book_floor_signed_distance_m']
)
CHECKPOINT_GRIPPER_MASTER_M = float(CHECKPOINT['gripper_master_m'])
ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP = tuple(
    tuple(row) for row in CERTIFICATE['attached_book_corners_physical_in_grasp']
)
ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP = tuple(
    tuple(row)
    for row in CERTIFICATE['attached_book_corners_padding_15mm_in_grasp']
)

ROUTE = tuple(CERTIFICATE['route'])
ROUTE_Q8 = tuple(tuple(row['q8']) for row in ROUTE)
SCALEUP_Q8 = ROUTE_Q8[1:]
PLANNED_WORLD_DELTA_M = tuple(
    tuple(row['world_delta_m']) for row in ROUTE[1:]
)
ROW_MINIMUM_DURATIONS_S = tuple(
    float(row['minimum_duration_s']) for row in ROUTE[1:]
)
MINIMUM_DURATION_S = min(ROW_MINIMUM_DURATIONS_S)


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
        [list(row) for row in ROUTE_Q8], separators=(',', ':'), allow_nan=False
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _finite_vector(values: object, length: int) -> bool:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return False
    return len(vector) == length and all(math.isfinite(value) for value in vector)


def validate_certificate() -> Tuple[bool, Mapping[str, float]]:
    rows_valid = bool(
        len(ROUTE) == 3
        and all(
            isinstance(row, Mapping)
            and int(row.get('row', -1)) == index
            and _finite_vector(row.get('q8'), 8)
            and _finite_vector(row.get('world_delta_m'), 3)
            for index, row in enumerate(ROUTE)
        )
        and ROUTE[0]['phase'] == 'checkpoint'
        and all(row['phase'] == 'shelf_outward' for row in ROUTE[1:])
        and PLANNED_WORLD_DELTA_M == (
            (-0.001, 0.0, 0.0), (-0.004, 0.0, 0.0)
        )
        and ROW_MINIMUM_DURATIONS_S == (2.5, 4.5)
    )
    checkpoint_valid = bool(
        _finite_vector(CHECKPOINT_BASE_WORLD_POSITION_M, 3)
        and _finite_vector(CHECKPOINT_BASE_QUATERNION_XYZW, 4)
        and math.isfinite(CHECKPOINT_BASE_WORLD_YAW_RAD)
        and _finite_vector(CHECKPOINT_LEFT_Q8, 8)
        and _finite_vector(CHECKPOINT_RIGHT_Q7, 7)
        and _finite_vector(CHECKPOINT_HEAD_Q2, 2)
        and _finite_vector(CHECKPOINT_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(CHECKPOINT_BOOK_QUATERNION_XYZW, 4)
        and math.isfinite(CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M)
        and math.isfinite(CHECKPOINT_GRIPPER_MASTER_M)
        and CHECKPOINT_LEFT_Q8 == ROUTE_Q8[0]
        and all(
            len(corners) == 8 and all(_finite_vector(row, 3) for row in corners)
            for corners in (
                ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP,
                ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP,
            )
        )
    )
    motion_valid = bool(
        EXECUTION_POLICY.get('fixed_hand_orientation') is True
        and EXECUTION_POLICY.get('torso_fixed') is True
        and EXECUTION_POLICY.get('pause_after_every_endpoint') is True
        and ENDPOINT_POLICY.get('pause_after_every_endpoint') is True
        and ENDPOINT_POLICY.get('keep_current_gripper_pressure') is True
        and ENDPOINT_POLICY.get('require_fresh_bilateral_exact_target_pressure')
        is True
        and all(
            math.isclose(row[0], CHECKPOINT_LEFT_Q8[0], abs_tol=1e-12)
            for row in ROUTE_Q8
        )
    )
    audit_valid = bool(
        AUDIT.get('collision_free') is True
        and int(AUDIT.get('dense_joint_samples', 0)) == 8
        and min(
            float(AUDIT.get('minimum_nonadjacent_robot_aabb_clearance_m', math.nan)),
            float(AUDIT.get('minimum_15mm_padded_payload_robot_aabb_clearance_m', math.nan)),
            float(AUDIT.get('minimum_closed_gripper_shelf_clearance_after_tolerance_m', math.nan)),
            float(AUDIT.get('minimum_robot_other_book_aabb_clearance_m', math.nan)),
            float(AUDIT.get('minimum_carried_book_other_book_aabb_clearance_m', math.nan)),
        ) > 0.0
    )
    endpoint_valid = bool(
        tuple(ENDPOINT_POLICY['maximum_cumulative_rotation_rad'])
        == (0.0008, 0.0018)
        and tuple(ENDPOINT_POLICY['maximum_cumulative_translation_change_m'])
        == (0.00018, 0.00032)
        and tuple(
            ENDPOINT_POLICY['minimum_book_center_cumulative_outward_progress_m']
        ) == (0.0007, 0.0028)
        and tuple(
            ENDPOINT_POLICY['minimum_deepest_extent_cumulative_outward_progress_m']
        ) == (0.00065, 0.0026)
        and float(ENDPOINT_POLICY['minimum_each_side_force_retention_fraction'])
        == 0.5
        and float(ENDPOINT_POLICY['floor_signed_min_m']) == -0.0001
        and float(ENDPOINT_POLICY['floor_signed_max_m']) == 0.00025
    )
    maximum_joint_delta = max(
        max(abs(b - a) for a, b in zip(first, second))
        for first, second in zip(ROUTE_Q8, ROUTE_Q8[1:])
    )
    valid = bool(
        CERTIFICATE_KIND
        == 'read_only_seed101_outward_scaleup_continue_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and rows_valid and checkpoint_valid and motion_valid
        and audit_valid and endpoint_valid
    )
    return valid, {
        'route_rows': float(len(ROUTE_Q8)),
        'command_rows': float(len(SCALEUP_Q8)),
        'maximum_joint_delta_rad': maximum_joint_delta,
        'minimum_audit_margin_m': min(
            float(AUDIT['minimum_nonadjacent_robot_aabb_clearance_m']),
            float(AUDIT['minimum_15mm_padded_payload_robot_aabb_clearance_m']),
            float(AUDIT['minimum_closed_gripper_shelf_clearance_after_tolerance_m']),
        ),
    }


__all__ = (
    'ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP',
    'ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP',
    'AUDIT', 'CERTIFICATE', 'CERTIFICATE_KIND',
    'CERTIFICATE_SEMANTIC_DIGEST', 'CERTIFICATE_SOURCE_JSON_SHA256',
    'CHECKPOINT', 'CHECKPOINT_BASE_QUATERNION_XYZW',
    'CHECKPOINT_BASE_WORLD_POSITION_M', 'CHECKPOINT_BASE_WORLD_YAW_RAD',
    'CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M',
    'CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M',
    'CHECKPOINT_BOOK_POSITION_WORLD_M', 'CHECKPOINT_BOOK_QUATERNION_XYZW',
    'CHECKPOINT_GRIPPER_MASTER_M', 'CHECKPOINT_HEAD_Q2', 'CHECKPOINT_LEFT_Q8',
    'CHECKPOINT_RIGHT_Q7', 'ENDPOINT_POLICY', 'EXECUTION_POLICY',
    'MINIMUM_DURATION_S', 'PLANNED_WORLD_DELTA_M', 'ROUTE', 'ROUTE_Q8',
    'ROUTE_Q8_SHA256', 'ROW_MINIMUM_DURATIONS_S', 'SCALEUP_Q8',
    'route_q8_digest', 'semantic_digest', 'source_json_digest',
    'validate_certificate',
)
