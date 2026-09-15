"""Exact offline certificate for the paused seed-101 diagonal peel step.

This module is inert: importing it performs no ROS, controller, Gazebo,
filesystem, or network operation.  The certificate applies only to the exact
recorded held-book checkpoint.  It describes one fixed-orientation move of
0.75 mm in world -X and 0.50 mm in world +Z, with the torso and gripper held.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    'bfbfc8e78929f205cfb55a32e0e06b9258c7dcd4c59b2c9d4514e8f078b99d79'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    '405fb42d305e68977eec553d7a83fead05d7efd6010912fc39fb6dfb7df5238f'
)
CERTIFICATE_JSON = r'''{
  "attached_book_corners_padding_15mm_in_grasp": [
    [
      0.0596105041931412,
      0.0300944716983146,
      -0.136214299735423
    ],
    [
      0.225737849631093,
      0.0280366341074724,
      -0.0440325748535948
    ],
    [
      0.0590427449893652,
      -0.0299020090531596,
      -0.1365304435706
    ],
    [
      0.225170090427318,
      -0.0319598466440018,
      -0.0443487186887723
    ],
    [
      -0.0762442592673814,
      0.0300899741763813,
      0.108619282979159
    ],
    [
      0.0898830861705709,
      0.0280321365855391,
      0.200801007860987
    ],
    [
      -0.0768120184711573,
      -0.0299065065750928,
      0.108303139143981
    ],
    [
      0.0893153269667949,
      -0.031964344165935,
      0.200484864025809
    ]
  ],
  "audit": {
    "collision_coverage": "diagnostic waypoint is the first half-waypoint inside an independently collision-certified neighborhood; collision margins below are inherited conservative bounds",
    "diagnostic_waypoint_is_certified_neighborhood_subset": true,
    "inherited_closed_gripper_shelf_collision_tolerance_m": 0.00075,
    "inherited_conservative_minimum_15mm_padded_payload_robot_clearance_m": 0.00525063,
    "inherited_full_route_minimum_nonadjacent_robot_aabb_clearance_m": 0.002051274,
    "inherited_minimum_carried_book_other_book_aabb_clearance_m": 0.258436,
    "inherited_minimum_closed_gripper_shelf_clearance_after_tolerance_m": 0.011274706,
    "inherited_minimum_closed_gripper_shelf_raw_clearance_m": 0.012024706,
    "current_book_ceiling_clearance_m": 0.03900462,
    "target_fk_orientation_error_rad": 2.02e-06,
    "target_fk_position_error_m": 9.06e-07
  },
  "endpoint_policy": {
    "max_book_center_inward_motion_m": 0.00025,
    "max_book_deepest_extent_increase_m": 0.00015,
    "max_book_from_hand_translation_change_m": 0.0005,
    "max_floor_signed_distance_m": 0.0005,
    "max_incremental_rotation_rad": 0.002,
    "max_shelf_penetration_m": 0.0001,
    "max_yaw_component_rad": 0.0015,
    "min_abs_rotation_axis_dot_world_y": 0.95,
    "min_book_center_outward_progress_m": 0.00025,
    "min_floor_signed_distance_m": -0.0001,
    "require_fresh_bilateral_exact_target_pressure": true
  },
  "execution_policy": {
    "fixed_hand_orientation": true,
    "keep_current_gripper_pressure": true,
    "minimum_duration_s": 1.2,
    "planned_world_delta_m": [
      -0.00075,
      0.0,
      0.0005
    ],
    "require_endpoint_gate_before_any_further_motion": true,
    "torso_fixed": true
  },
  "input": {
    "base_world_xyyaw": [
      2.0027012233,
      -0.1494027021,
      -0.7853981633974483
    ],
    "book_floor_signed_distance_m": -1.26e-06,
    "book_physical_bounds_world_m": [
      [
        2.8126908599,
        -0.1691300411,
        1.4518572102
      ],
      [
        2.9746007829,
        -0.1371945473,
        1.7028538471
      ]
    ],
    "book_position_world_m": [
      2.893645821378262,
      -0.15316229419440927,
      1.5773555286545407
    ],
    "book_quaternion_xyzw": [
      -0.004258650506574088,
      0.7093019208353551,
      0.004250302763156808,
      0.7048791271711482
    ],
    "gripper_master_m": 0.02900820842,
    "head_q2_measured": [
      -5.341047622287229e-12,
      -2.104500678413703e-12
    ],
    "left_q8_measured": [
      0.3499998174377888,
      -0.31065478099691884,
      0.6171232383242858,
      0.08169638870230944,
      -1.4769923426656921,
      0.1731037765037025,
      1.0321934140278828,
      0.21509846856833278
    ],
    "right_q7_measured": [
      1.1017502906726425e-06,
      2.930440907271199e-06,
      9.312227186970225e-07,
      -7.4406713585614e-06,
      2.550846057202886e-08,
      5.052158441872368e-10,
      -2.9108287519463966e-10
    ]
  },
  "kind": "read_only_seed101_diagonal_peel_step_certificate",
  "observed_prior_micro_lift": {
    "book_delta_world_m": [
      0.00015429798358690405,
      -7.676886050031473e-05,
      0.000495531218393408
    ],
    "book_rotation_axis_world": [
      0.004602250155556919,
      0.9897869558323922,
      0.1424801788488944
    ],
    "book_rotation_rad": 0.0062686420544828345,
    "commanded_world_delta_m": [
      0.0,
      0.0,
      0.001
    ],
    "diagnosis": "book rocked about a deep shelf-supported edge; front grasp followed the hand",
    "effective_pivot_line_world_xz_m": [
      2.97318,
      1.55161
    ]
  },
  "route": [
    {
      "minimum_duration_s": 0.0,
      "phase": "checkpoint",
      "q8": [
        0.3499998174377888,
        -0.31065478099691884,
        0.6171232383242858,
        0.08169638870230944,
        -1.4769923426656921,
        0.1731037765037025,
        1.0321934140278828,
        0.21509846856833278
      ],
      "row": 0,
      "world_delta_m": [
        0.0,
        0.0,
        0.0
      ]
    },
    {
      "minimum_duration_s": 1.2,
      "phase": "diagonal_peel",
      "q8": [
        0.3499998174377888,
        -0.31014159591450025,
        0.6183859306984962,
        0.08185239525203032,
        -1.4775941032640676,
        0.17339864886753584,
        1.0341486100334276,
        0.2147926251332205
      ],
      "row": 1,
      "world_delta_m": [
        -0.00075,
        0.0,
        0.0005
      ]
    }
  ],
  "route_q8_sha256": "7dc56385293dd59e595348db03a3b3708a43f578b75f7ffca6127cbf65d0132d"
}
'''

CERTIFICATE: Mapping[str, object] = json.loads(CERTIFICATE_JSON)

CERTIFICATE_KIND = str(CERTIFICATE['kind'])
ROUTE_Q8_SHA256 = str(CERTIFICATE['route_q8_sha256'])

CHECKPOINT = CERTIFICATE['input']
AUDIT = CERTIFICATE['audit']
ENDPOINT_POLICY = CERTIFICATE['endpoint_policy']
EXECUTION_POLICY = CERTIFICATE['execution_policy']
OBSERVED_PRIOR_MICRO_LIFT = CERTIFICATE['observed_prior_micro_lift']

ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP = tuple(
    tuple(row)
    for row in CERTIFICATE['attached_book_corners_padding_15mm_in_grasp']
)

CHECKPOINT_BASE_WORLD_XYYAW = tuple(CHECKPOINT['base_world_xyyaw'])
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

PLANNED_WORLD_DELTA_M = tuple(EXECUTION_POLICY['planned_world_delta_m'])
MINIMUM_DURATION_S = float(EXECUTION_POLICY['minimum_duration_s'])

ROUTE = tuple(CERTIFICATE['route'])
ROUTE_Q8 = tuple(tuple(row['q8']) for row in ROUTE)
DIAGONAL_PEEL_Q8 = ROUTE_Q8[1]


def _canonical_certificate_bytes() -> bytes:
    return json.dumps(
        CERTIFICATE,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')


def semantic_digest() -> str:
    """Return the deterministic digest of all embedded certificate data."""

    return hashlib.sha256(_canonical_certificate_bytes()).hexdigest()


def source_json_digest() -> str:
    """Return the digest of the exact embedded source JSON text."""

    return hashlib.sha256(CERTIFICATE_JSON.encode('utf-8')).hexdigest()


def route_q8_digest() -> str:
    """Return the deterministic digest of the exact two-row q8 route."""

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
        and ROUTE[1]['phase'] == 'diagonal_peel'
        and tuple(ROUTE[0]['world_delta_m']) == (0.0, 0.0, 0.0)
        and tuple(ROUTE[1]['world_delta_m']) == PLANNED_WORLD_DELTA_M
        and float(ROUTE[1]['minimum_duration_s']) == MINIMUM_DURATION_S
    )
    checkpoint_valid = bool(
        _finite_vector(CHECKPOINT_BASE_WORLD_XYYAW, 3)
        and _finite_vector(CHECKPOINT_LEFT_Q8, 8)
        and _finite_vector(CHECKPOINT_RIGHT_Q7, 7)
        and _finite_vector(CHECKPOINT_HEAD_Q2, 2)
        and _finite_vector(CHECKPOINT_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(CHECKPOINT_BOOK_QUATERNION_XYZW, 4)
        and math.isfinite(CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M)
        and math.isfinite(CHECKPOINT_GRIPPER_MASTER_M)
        and CHECKPOINT_LEFT_Q8 == ROUTE_Q8[0]
        and len(ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP) == 8
        and all(
            _finite_vector(row, 3)
            for row in ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP
        )
    )
    maximum_joint_delta = max(
        abs(after - before)
        for before, after in zip(CHECKPOINT_LEFT_Q8, DIAGONAL_PEEL_Q8)
    )
    motion_valid = bool(
        PLANNED_WORLD_DELTA_M == (-0.00075, 0.0, 0.0005)
        and MINIMUM_DURATION_S >= 1.2
        and EXECUTION_POLICY.get('fixed_hand_orientation') is True
        and EXECUTION_POLICY.get('keep_current_gripper_pressure') is True
        and EXECUTION_POLICY.get('torso_fixed') is True
        and EXECUTION_POLICY.get(
            'require_endpoint_gate_before_any_further_motion'
        ) is True
        and math.isclose(
            DIAGONAL_PEEL_Q8[0], CHECKPOINT_LEFT_Q8[0],
            rel_tol=0.0, abs_tol=1e-12,
        )
    )
    audit_valid = bool(
        AUDIT.get('diagnostic_waypoint_is_certified_neighborhood_subset')
        is True
        and float(AUDIT.get('target_fk_position_error_m', math.inf)) < 1e-6
        and float(AUDIT.get('target_fk_orientation_error_rad', math.inf))
        < 3e-6
        and min(
            float(
                AUDIT.get(
                    'inherited_full_route_minimum_nonadjacent_robot_aabb_clearance_m',
                    math.nan,
                )
            ),
            float(
                AUDIT.get(
                    'inherited_conservative_minimum_15mm_padded_payload_robot_clearance_m',
                    math.nan,
                )
            ),
            float(
                AUDIT.get(
                    'inherited_minimum_carried_book_other_book_aabb_clearance_m',
                    math.nan,
                )
            ),
            float(
                AUDIT.get(
                    'inherited_minimum_closed_gripper_shelf_clearance_after_tolerance_m',
                    math.nan,
                )
            ),
        ) > 0.0
    )
    endpoint_policy_valid = bool(
        ENDPOINT_POLICY.get('require_fresh_bilateral_exact_target_pressure')
        is True
        and float(ENDPOINT_POLICY['min_book_center_outward_progress_m'])
        == 0.00025
        and float(ENDPOINT_POLICY['max_book_center_inward_motion_m'])
        <= 0.00025
        and float(ENDPOINT_POLICY['max_book_from_hand_translation_change_m'])
        <= 0.0005
        and float(ENDPOINT_POLICY['max_incremental_rotation_rad']) <= 0.002
        and float(ENDPOINT_POLICY['min_abs_rotation_axis_dot_world_y']) >= 0.95
        and float(ENDPOINT_POLICY['max_yaw_component_rad']) <= 0.0015
        and float(ENDPOINT_POLICY['min_floor_signed_distance_m']) >= -0.0001
        and float(ENDPOINT_POLICY['max_floor_signed_distance_m']) <= 0.0005
    )
    valid = bool(
        CERTIFICATE_KIND
        == 'read_only_seed101_diagonal_peel_step_certificate'
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
            float(
                AUDIT[
                    'inherited_full_route_minimum_nonadjacent_robot_aabb_clearance_m'
                ]
            ),
            float(
                AUDIT[
                    'inherited_conservative_minimum_15mm_padded_payload_robot_clearance_m'
                ]
            ),
            float(
                AUDIT[
                    'inherited_minimum_carried_book_other_book_aabb_clearance_m'
                ]
            ),
        ),
    }


__all__ = (
    'ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP',
    'AUDIT',
    'CERTIFICATE',
    'CERTIFICATE_KIND',
    'CERTIFICATE_SEMANTIC_DIGEST',
    'CERTIFICATE_SOURCE_JSON_SHA256',
    'CHECKPOINT',
    'CHECKPOINT_BASE_WORLD_XYYAW',
    'CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M',
    'CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M',
    'CHECKPOINT_BOOK_POSITION_WORLD_M',
    'CHECKPOINT_BOOK_QUATERNION_XYZW',
    'CHECKPOINT_GRIPPER_MASTER_M',
    'CHECKPOINT_HEAD_Q2',
    'CHECKPOINT_LEFT_Q8',
    'CHECKPOINT_RIGHT_Q7',
    'DIAGONAL_PEEL_Q8',
    'ENDPOINT_POLICY',
    'EXECUTION_POLICY',
    'MINIMUM_DURATION_S',
    'OBSERVED_PRIOR_MICRO_LIFT',
    'PLANNED_WORLD_DELTA_M',
    'ROUTE',
    'ROUTE_Q8',
    'ROUTE_Q8_SHA256',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
