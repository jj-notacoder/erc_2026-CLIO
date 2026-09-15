"""Inert certificate for one seed-101 shelf-supported outward step.

Importing this module performs no ROS, controller, Gazebo, filesystem, or
network operation.  The two-row route is bound to one paused held-book
checkpoint and commands only 0.50 mm in world -X at fixed hand orientation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    '85d615c78a2dbfd20fe0978a1495b3a07919827735bca2ea597a00448d1cf193'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    '2fd3ad97fbc66a8e6fd80eca5d5c0ca4f8a64df96823e50cf6a02e9cfb828eac'
)
CERTIFICATE_JSON = r'''{
  "attached_book_corners_padding_15mm_in_grasp": [
    [0.06008593380669869, 0.02997304664829531, -0.1359911890157554],
    [0.22592991784984945, 0.02808178355490994, -0.04329708952389563],
    [0.0595651835230593, -0.03002398079288136, -0.13628362253774406],
    [0.22540916756621004, -0.03191524388626672, -0.0435895230458843],
    [-0.07652271355130051, 0.029967450043853743, 0.1084225533059395],
    [0.08932127049185024, 0.02807618695046834, 0.20111665279779928],
    [-0.07704346383493992, -0.030029577397322944, 0.10813011978395083],
    [0.0888005202082108, -0.03192084049070832, 0.2008242192758106]
  ],
  "attached_book_corners_physical_in_grasp": [
    [0.06573037195028009, 0.014824179725691423, -0.11565274883454883],
    [0.20538846377609116, 0.01323153712073531, -0.0375945597887722],
    [0.06546999680846038, -0.015174333994896904, -0.11579896559554316],
    [0.20512808863427145, -0.016766976599853017, -0.03774077654976653],
    [-0.05624163461936193, 0.014819182757440012, 0.10257380680982174],
    [0.08341645720644913, 0.013226540152483899, 0.18063199585559836],
    [-0.05650200976118163, -0.015179330963148316, 0.1024275900488274],
    [0.08315608206462943, -0.016771973568104427, 0.18048577909460403]
  ],
  "audit": {
    "collision_free": true,
    "dense_joint_samples": 9,
    "maximum_dense_joint_increment_rad": 0.000204,
    "minimum_15mm_padded_payload_robot_aabb_clearance_m": 0.09910554786951353,
    "minimum_closed_gripper_shelf_clearance_after_tolerance_m": 0.011773667244485503,
    "minimum_nonadjacent_robot_aabb_clearance_m": 0.0021614295280663054,
    "shelf_collision_tolerance_m": 0.00075,
    "target_fk_orientation_error_rad": 4.22e-11,
    "target_fk_position_error_m": 5.61e-11
  },
  "endpoint_policy": {
    "floor_signed_max_m": 0.00025,
    "floor_signed_min_m": -0.0001,
    "maximum_book_from_hand_translation_change_m": 0.00035,
    "maximum_book_incremental_rotation_rad": 0.002,
    "maximum_deepest_extent_inward_motion_m": 0.00015,
    "minimum_book_center_outward_progress_m": 0.00025,
    "minimum_deepest_extent_outward_progress_m": 0.0002,
    "require_fresh_bilateral_exact_target_pressure": true
  },
  "execution_policy": {
    "fixed_hand_orientation": true,
    "keep_current_gripper_pressure": true,
    "minimum_duration_s": 1.5,
    "planned_world_delta_m": [-0.0005, 0.0, 0.0],
    "require_endpoint_gate_before_further_motion": true,
    "torso_fixed": true
  },
  "input": {
    "base_quaternion_xyzw": [2.0738319846978343e-08, 6.100898975558632e-08, -0.3826463512946239, 0.9238948911217699],
    "base_world_position_m": [2.0026952797877713, -0.14944553030518343, -3.240684055017811e-07],
    "base_world_yaw_rad": -0.7853178915417812,
    "book_floor_signed_distance_m": -2.313e-06,
    "book_physical_bounds_world_m": [
      [2.81168131, -0.16900588, 1.45185646],
      [2.97433327, -0.13720673, 1.70334069]
    ],
    "book_position_world_m": [2.893007288445724, -0.1531063074070775, 1.577598577430115],
    "book_quaternion_xyzw": [-0.0039514559842917674, 0.710388296852809, 0.003953043185958488, 0.7037877713769537],
    "gripper_master_m": 0.029008128166538853,
    "head_q2_measured": [-5.341047622287229e-12, -2.104500678413703e-12],
    "left_q8_measured": [0.3499998174377888, -0.31014159591450025, 0.6183859306984962, 0.08185239525203032, -1.4775941032640676, 0.17339864886753584, 1.0341486100334276, 0.2147926251332205],
    "right_q7_measured": [1.1017502906726425e-06, 2.930440907271199e-06, 9.312227186970225e-07, -7.4406713585614e-06, 2.550846057202886e-08, 5.052158441872368e-10, -2.9108287519463966e-10]
  },
  "kind": "read_only_seed101_pure_outward_step_certificate",
  "observed_diagonal_probe": {
    "book_center_delta_world_m": [-0.000636303, -3.037e-06, 0.000247076],
    "book_from_hand_translation_change_m": 0.00027732,
    "book_incremental_rotation_rad": 0.003151524,
    "deepest_extent_outward_progress_m": 0.000247592,
    "pressure_monitor_remained_valid": true,
    "rotation_axis_world_y": 0.99824,
    "yaw_component_rad": 0.0001823
  },
  "route": [
    {
      "minimum_duration_s": 0.0,
      "phase": "checkpoint",
      "q8": [0.3499998174377888, -0.31014159591450025, 0.6183859306984962, 0.08185239525203032, -1.4775941032640676, 0.17339864886753584, 1.0341486100334276, 0.2147926251332205],
      "row": 0,
      "world_delta_m": [0.0, 0.0, 0.0]
    },
    {
      "minimum_duration_s": 1.5,
      "phase": "shelf_outward",
      "q8": [0.3499998174377888, -0.3099328933601225, 0.618214728582325, 0.0819199889501402, -1.479223208250511, 0.17350872560919878, 1.0356379503462736, 0.21479652461032916],
      "row": 1,
      "world_delta_m": [-0.0005, 0.0, 0.0]
    }
  ],
  "route_q8_sha256": "434c45d323c23d8ab897a71e788d928238eed3e7014930b0b4923fa4bbc035c9"
}
'''

CERTIFICATE: Mapping[str, object] = json.loads(CERTIFICATE_JSON)
CERTIFICATE_KIND = str(CERTIFICATE['kind'])
ROUTE_Q8_SHA256 = str(CERTIFICATE['route_q8_sha256'])
CHECKPOINT = CERTIFICATE['input']
AUDIT = CERTIFICATE['audit']
ENDPOINT_POLICY = CERTIFICATE['endpoint_policy']
EXECUTION_POLICY = CERTIFICATE['execution_policy']
OBSERVED_DIAGONAL_PROBE = CERTIFICATE['observed_diagonal_probe']

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

PLANNED_WORLD_DELTA_M = tuple(EXECUTION_POLICY['planned_world_delta_m'])
MINIMUM_DURATION_S = float(EXECUTION_POLICY['minimum_duration_s'])
ROUTE = tuple(CERTIFICATE['route'])
ROUTE_Q8 = tuple(tuple(row['q8']) for row in ROUTE)
PURE_OUTWARD_Q8 = ROUTE_Q8[1]

MAXIMUM_TRANSLATION_CHANGE_M = float(
    ENDPOINT_POLICY['maximum_book_from_hand_translation_change_m']
)
MAXIMUM_INCREMENTAL_ROTATION_RAD = float(
    ENDPOINT_POLICY['maximum_book_incremental_rotation_rad']
)
MINIMUM_CENTER_OUTWARD_PROGRESS_M = float(
    ENDPOINT_POLICY['minimum_book_center_outward_progress_m']
)
MINIMUM_DEEPEST_EXTENT_OUTWARD_PROGRESS_M = float(
    ENDPOINT_POLICY['minimum_deepest_extent_outward_progress_m']
)
FLOOR_SIGNED_RANGE_M = (
    float(ENDPOINT_POLICY['floor_signed_min_m']),
    float(ENDPOINT_POLICY['floor_signed_max_m']),
)


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
    """Return the deterministic digest of the exact two-row joint route."""

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
    """Fail closed if any checkpoint, route, policy, or audit fact changed."""

    rows_valid = bool(
        len(ROUTE) == 2
        and all(
            isinstance(row, Mapping)
            and int(row.get('row', -1)) == index
            and _finite_vector(row.get('q8'), 8)
            and _finite_vector(row.get('world_delta_m'), 3)
            for index, row in enumerate(ROUTE)
        )
        and ROUTE[0]['phase'] == 'checkpoint'
        and ROUTE[1]['phase'] == 'shelf_outward'
        and tuple(ROUTE[0]['world_delta_m']) == (0.0, 0.0, 0.0)
        and tuple(ROUTE[1]['world_delta_m']) == PLANNED_WORLD_DELTA_M
        and float(ROUTE[1]['minimum_duration_s']) == MINIMUM_DURATION_S
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
    maximum_joint_delta = max(
        abs(after - before)
        for before, after in zip(CHECKPOINT_LEFT_Q8, PURE_OUTWARD_Q8)
    )
    motion_valid = bool(
        PLANNED_WORLD_DELTA_M == (-0.0005, 0.0, 0.0)
        and MINIMUM_DURATION_S >= 1.5
        and EXECUTION_POLICY.get('fixed_hand_orientation') is True
        and EXECUTION_POLICY.get('keep_current_gripper_pressure') is True
        and EXECUTION_POLICY.get('torso_fixed') is True
        and EXECUTION_POLICY.get('require_endpoint_gate_before_further_motion')
        is True
        and math.isclose(
            PURE_OUTWARD_Q8[0], CHECKPOINT_LEFT_Q8[0],
            rel_tol=0.0, abs_tol=1e-12,
        )
    )
    audit_valid = bool(
        AUDIT.get('collision_free') is True
        and int(AUDIT.get('dense_joint_samples', 0)) == 9
        and float(AUDIT.get('target_fk_position_error_m', math.inf)) < 1e-9
        and float(AUDIT.get('target_fk_orientation_error_rad', math.inf)) < 1e-9
        and min(
            float(AUDIT.get('minimum_nonadjacent_robot_aabb_clearance_m', math.nan)),
            float(AUDIT.get('minimum_15mm_padded_payload_robot_aabb_clearance_m', math.nan)),
            float(AUDIT.get('minimum_closed_gripper_shelf_clearance_after_tolerance_m', math.nan)),
        ) > 0.0
    )
    endpoint_policy_valid = bool(
        ENDPOINT_POLICY.get('require_fresh_bilateral_exact_target_pressure')
        is True
        and MAXIMUM_TRANSLATION_CHANGE_M <= 0.00035
        and MAXIMUM_INCREMENTAL_ROTATION_RAD <= 0.002
        and MINIMUM_CENTER_OUTWARD_PROGRESS_M >= 0.00025
        and MINIMUM_DEEPEST_EXTENT_OUTWARD_PROGRESS_M >= 0.0002
        and float(ENDPOINT_POLICY['maximum_deepest_extent_inward_motion_m'])
        <= 0.00015
        and FLOOR_SIGNED_RANGE_M == (-0.0001, 0.00025)
    )
    valid = bool(
        CERTIFICATE_KIND == 'read_only_seed101_pure_outward_step_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and rows_valid
        and checkpoint_valid
        and motion_valid
        and audit_valid
        and endpoint_policy_valid
    )
    return valid, {
        'route_rows': float(len(ROUTE_Q8)),
        'maximum_joint_delta_rad': maximum_joint_delta,
        'minimum_duration_s': MINIMUM_DURATION_S,
        'minimum_audit_margin_m': min(
            float(AUDIT['minimum_nonadjacent_robot_aabb_clearance_m']),
            float(AUDIT['minimum_15mm_padded_payload_robot_aabb_clearance_m']),
            float(AUDIT['minimum_closed_gripper_shelf_clearance_after_tolerance_m']),
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
    'ENDPOINT_POLICY',
    'EXECUTION_POLICY',
    'FLOOR_SIGNED_RANGE_M',
    'MAXIMUM_INCREMENTAL_ROTATION_RAD',
    'MAXIMUM_TRANSLATION_CHANGE_M',
    'MINIMUM_CENTER_OUTWARD_PROGRESS_M',
    'MINIMUM_DEEPEST_EXTENT_OUTWARD_PROGRESS_M',
    'MINIMUM_DURATION_S',
    'OBSERVED_DIAGONAL_PROBE',
    'PLANNED_WORLD_DELTA_M',
    'PURE_OUTWARD_Q8',
    'ROUTE',
    'ROUTE_Q8',
    'ROUTE_Q8_SHA256',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
