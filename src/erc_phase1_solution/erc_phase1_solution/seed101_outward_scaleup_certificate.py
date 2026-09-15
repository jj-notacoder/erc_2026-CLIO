"""Inert certificate for the seed-101 shelf-supported outward scale-up.

The route is bound to one paused held-book checkpoint.  It contains three
fixed-orientation, left-arm-only world-minus-X endpoints at cumulative 1, 2,
and 5 mm.  Importing this module performs no external operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    'c21a5f9b93eed26e737f241ad1440c8b85e95f3988774b2394ebc56e1bd3a5dc'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    'e873670368d741af9216cf3653a5806439d4c7621099bf020f748e17281c47a0'
)
CERTIFICATE_JSON = r'''{
  "attached_book_corners_padding_15mm_in_grasp": [
    [0.06007191148917354, 0.029911643230651097, -0.1359750837141643],
    [0.22592892283456978, 0.02809551515565323, -0.04330279299904112],
    [0.059570470300909735, -0.030085615698311856, -0.13625343195951156],
    [0.22542748164630594, -0.03190174377330972, -0.043581141244388416],
    [-0.07650395715457589, 0.029919102418094923, 0.1084569764755045],
    [0.08935305419082033, 0.02810297434309706, 0.20112926719062768],
    [-0.0770053983428397, -0.030078156510868027, 0.10817862823015723],
    [0.08885161300255653, -0.03189428458586587, 0.20085091894528037]
  ],
  "attached_book_corners_physical_in_grasp": [
    [0.06572396138037766, 0.014769349565501024, -0.11563386264497338],
    [0.2053930235659749, 0.013239978554976517, -0.03759403888486966],
    [0.065473240786246, -0.015229279898980451, -0.11577303676764703],
    [0.20514230297184274, -0.016758650909504957, -0.037733213007543315],
    [-0.05621877848011268, 0.014776009554290167, 0.1026090482386594],
    [0.08345028370548407, 0.013246638543765659, 0.18064887199876312],
    [-0.05646949907424483, -0.01522261991019131, 0.10246987411598575],
    [0.0831995631113524, -0.016751990920715814, 0.18050969787608948]
  ],
  "audit": {
    "collision_free": true,
    "dense_joint_samples": 10,
    "maximum_dense_joint_increment_rad": 0.002,
    "minimum_15mm_padded_payload_robot_aabb_clearance_m": 0.09916101371496033,
    "minimum_carried_book_other_book_aabb_clearance_m": 0.2584458909551794,
    "minimum_closed_gripper_shelf_clearance_after_tolerance_m": 0.011773834255501187,
    "minimum_nonadjacent_robot_aabb_clearance_m": 0.0021572671943710375,
    "minimum_robot_bin_aabb_clearance_m": 2.331373870770816,
    "minimum_robot_other_book_aabb_clearance_m": 0.1407570981139426,
    "minimum_robot_table_aabb_clearance_m": 2.173458775887552,
    "shelf_collision_tolerance_m": 0.00075,
    "target_maximum_fk_orientation_error_rad": 1.33e-10,
    "target_maximum_fk_position_error_m": 3.11e-10
  },
  "endpoint_policy": {
    "floor_signed_max_m": 0.00025,
    "floor_signed_min_m": -0.0001,
    "keep_current_gripper_pressure": true,
    "maximum_cumulative_rotation_rad": [0.0006, 0.001, 0.002],
    "maximum_cumulative_translation_change_m": [0.00015, 0.0002, 0.00035],
    "minimum_book_center_cumulative_outward_progress_m": [0.0007, 0.0014, 0.0035],
    "minimum_deepest_extent_cumulative_outward_progress_m": [0.00065, 0.0013, 0.00325],
    "minimum_each_side_force_retention_fraction": 0.5,
    "pause_after_every_endpoint": true,
    "require_fresh_bilateral_exact_target_pressure": true
  },
  "execution_policy": {
    "cumulative_outward_world_minus_x_m": [0.001, 0.002, 0.005],
    "fixed_hand_orientation": true,
    "minimum_durations_s": [2.5, 2.5, 4.5],
    "pause_after_every_endpoint": true,
    "torso_fixed": true
  },
  "input": {
    "base_quaternion_xyzw": [2.4516683782643835e-09, 4.3372223619149105e-09, -0.3826348055997708, 0.9238996728777565],
    "base_world_position_m": [2.00268588774987, -0.1494545965687467, -6.3358907793141e-08],
    "base_world_yaw_rad": -0.7852928980818393,
    "book_floor_signed_distance_m": -3.3e-07,
    "book_physical_bounds_world_m": [[2.81122042, -0.16895152, 1.45185814], [2.97382859, -0.13719986, 1.70331975]],
    "book_position_world_m": [2.8925245058893223, -0.15307569024398282, 1.5775889442600233],
    "book_quaternion_xyzw": [-0.003837929463147951, 0.7103418934244442, 0.0038047704464166656, 0.703836052263211],
    "gripper_master_m": 0.02900815633201123,
    "head_q2_measured": [-5.341047622287229e-12, -2.104500678413703e-12],
    "left_q8_measured": [0.3499998174377888, -0.3099328933601225, 0.618214728582325, 0.0819199889501402, -1.479223208250511, 0.17350872560919878, 1.0356379503462736, 0.21479652461032916],
    "right_q7_measured": [1.1017502906726425e-06, 2.930440907271199e-06, 9.312227186970225e-07, -7.4406713585614e-06, 2.550846057202886e-08, 5.052158441872368e-10, -2.9108287519463966e-10]
  },
  "kind": "read_only_seed101_outward_scaleup_certificate",
  "observed_prior_outward_step": {
    "bilateral_normal_forces_n": [2.016, 1.486],
    "book_center_delta_world_m": [-0.000483219, -0.000002622, -0.000007976],
    "book_from_hand_translation_mismatch_m": 0.000018764,
    "book_incremental_rotation_rad": 0.000109233,
    "deepest_extent_outward_progress_m": 0.000496067
  },
  "route": [
    {"minimum_duration_s": 0.0, "phase": "checkpoint", "q8": [0.3499998174377888, -0.3099328933601225, 0.618214728582325, 0.0819199889501402, -1.479223208250511, 0.17350872560919878, 1.0356379503462736, 0.21479652461032916], "row": 0, "world_delta_m": [0.0, 0.0, 0.0]},
    {"maximum_cumulative_rotation_rad": 0.0006, "maximum_cumulative_translation_change_m": 0.00015, "minimum_book_center_cumulative_outward_progress_m": 0.0007, "minimum_deepest_extent_cumulative_outward_progress_m": 0.00065, "minimum_duration_s": 2.5, "phase": "shelf_outward", "q8": [0.3499998174377888, -0.3095148967058795, 0.6178794379281761, 0.08205575594538433, -1.4824722381676605, 0.17372994976888595, 1.0386147095992404, 0.2148038614518491], "row": 1, "world_delta_m": [-0.001, 0.0, 0.0]},
    {"maximum_cumulative_rotation_rad": 0.001, "maximum_cumulative_translation_change_m": 0.0002, "minimum_book_center_cumulative_outward_progress_m": 0.0014, "minimum_deepest_extent_cumulative_outward_progress_m": 0.0013, "minimum_duration_s": 2.5, "phase": "shelf_outward", "q8": [0.3499998174377888, -0.30909607616392276, 0.6175535922346728, 0.08219232062926038, -1.4857090774553214, 0.17395264707175168, 1.0415889282352255, 0.21481052701719683], "row": 2, "world_delta_m": [-0.002, 0.0, 0.0]},
    {"maximum_cumulative_rotation_rad": 0.002, "maximum_cumulative_translation_change_m": 0.00035, "minimum_book_center_cumulative_outward_progress_m": 0.0035, "minimum_deepest_extent_cumulative_outward_progress_m": 0.00325, "minimum_duration_s": 4.5, "phase": "shelf_outward", "q8": [0.3499998174377888, -0.3078346417560695, 0.6166324597930262, 0.08260676597407056, -1.49534711034959, 0.17462952039489882, 1.0504967340163236, 0.21482665883406432], "row": 3, "world_delta_m": [-0.005, 0.0, 0.0]}
  ],
  "route_q8_sha256": "74a01efc7f6742f7d60a610fa996d5bd84ee8750d6f625b0d2b61302d55302ee"
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
        len(ROUTE) == 4
        and all(
            isinstance(row, Mapping)
            and int(row.get('row', -1)) == index
            and _finite_vector(row.get('q8'), 8)
            and _finite_vector(row.get('world_delta_m'), 3)
            for index, row in enumerate(ROUTE)
        )
        and ROUTE[0]['phase'] == 'checkpoint'
        and all(row['phase'] == 'shelf_outward' for row in ROUTE[1:])
        and tuple(ROUTE[0]['world_delta_m']) == (0.0, 0.0, 0.0)
        and PLANNED_WORLD_DELTA_M == (
            (-0.001, 0.0, 0.0),
            (-0.002, 0.0, 0.0),
            (-0.005, 0.0, 0.0),
        )
        and ROW_MINIMUM_DURATIONS_S == (2.5, 2.5, 4.5)
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
        and int(AUDIT.get('dense_joint_samples', 0)) == 10
        and float(AUDIT.get('target_maximum_fk_position_error_m', math.inf))
        < 1e-9
        and float(AUDIT.get('target_maximum_fk_orientation_error_rad', math.inf))
        < 1e-9
        and min(
            float(AUDIT.get('minimum_nonadjacent_robot_aabb_clearance_m', math.nan)),
            float(AUDIT.get('minimum_15mm_padded_payload_robot_aabb_clearance_m', math.nan)),
            float(AUDIT.get('minimum_closed_gripper_shelf_clearance_after_tolerance_m', math.nan)),
            float(AUDIT.get('minimum_robot_other_book_aabb_clearance_m', math.nan)),
            float(AUDIT.get('minimum_carried_book_other_book_aabb_clearance_m', math.nan)),
        ) > 0.0
    )
    endpoint_valid = bool(
        tuple(ENDPOINT_POLICY['maximum_cumulative_translation_change_m'])
        == (0.00015, 0.0002, 0.00035)
        and tuple(ENDPOINT_POLICY['maximum_cumulative_rotation_rad'])
        == (0.0006, 0.001, 0.002)
        and tuple(
            ENDPOINT_POLICY['minimum_book_center_cumulative_outward_progress_m']
        ) == (0.0007, 0.0014, 0.0035)
        and tuple(
            ENDPOINT_POLICY['minimum_deepest_extent_cumulative_outward_progress_m']
        ) == (0.00065, 0.0013, 0.00325)
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
        CERTIFICATE_KIND == 'read_only_seed101_outward_scaleup_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and rows_valid
        and checkpoint_valid
        and motion_valid
        and audit_valid
        and endpoint_valid
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
    'MINIMUM_DURATION_S',
    'PLANNED_WORLD_DELTA_M',
    'ROUTE',
    'ROUTE_Q8',
    'ROUTE_Q8_SHA256',
    'ROW_MINIMUM_DURATIONS_S',
    'SCALEUP_Q8',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
