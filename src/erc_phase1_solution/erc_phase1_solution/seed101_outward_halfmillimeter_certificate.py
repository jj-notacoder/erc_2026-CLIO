"""Inert exact-state certificate for one 0.5 mm shelf-outward rung.

The two-row route is bound to the authoritative paused seed-101 state left
after the post-reseat verifier stopped.  Row 1 is an offline-solved exact
world-minus-X target with fixed torso and hand orientation; it is not a joint
midpoint.  Importing this module performs no ROS, Gazebo, network, or process
operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    '2fa98fe7e61268b96a2b628bf5c6e74f31d2dd97122d5fe81fec648c3aa2a967'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    '87e1b4e75a96f88e6a7e01ed8a8c24f12c2407ad4dddef8ce2c857bcec48dbff'
)
CERTIFICATE_JSON = r'''{
  "attached_book_corners_padding_15mm_in_grasp": [
    [0.05977907090353527, 0.02994036866298485, -0.13595129528322045],
    [0.2257222597172832, 0.02820525926664631, -0.04343185147972861],
    [0.05930000165547603, -0.03005712938871359, -0.13621722796644112],
    [0.22524319046922398, -0.031792238785052135, -0.043697784162949296],
    [-0.0765710203297692, 0.02994512136604614, 0.10860678084202495],
    [0.08937216848397873, 0.0282100119697076, 0.2011262246455168],
    [-0.07705008957782841, -0.0300523766856523, 0.10834084815880426],
    [0.08889309923591951, -0.03178748608199084, 0.2008602919622961]
  ],
  "attached_book_corners_physical_in_grasp": [
    [0.06545561248255072, 0.014804266438389248, -0.11561228633125106],
    [0.20519724516781263, 0.013343121683577836, -0.03770117575988931],
    [0.06521607785852106, -0.015194482587459972, -0.1157452526728613],
    [0.20495771054378292, -0.01665562734227138, -0.037834142101499546],
    [-0.05628554040432814, 0.014808509923265387, 0.10274313878057521],
    [0.08345609228093372, 0.013347365168453976, 0.18065424935193697],
    [-0.05652507502835781, -0.015190239102583832, 0.10261017243896497],
    [0.08321655765690404, -0.016651383857395244, 0.18052128301032672]
  ],
  "audit": {
    "asset_count": 13,
    "collision_free": true,
    "current_geometry_revalidation_required": true,
    "dense_joint_samples": 5,
    "endpoint_is_exact_ik_target": true,
    "expected_physical_book_shelf_support_contact_only": true,
    "gripper_geometry_joint_count": 8,
    "joint_midpoint_used": false,
    "maximum_attached_corner_reconstruction_error_m": 1.6653345369377348e-16,
    "maximum_dense_joint_increment_rad": 0.000404985040551864,
    "minimum_15mm_padded_payload_robot_aabb_clearance_m": 0.10135749232271651,
    "minimum_carried_book_other_book_aabb_clearance_m": 0.2584731440119574,
    "minimum_closed_gripper_shelf_clearance_after_tolerance_m": 0.029351485230199443,
    "minimum_closed_gripper_shelf_triangle_aabb_clearance_raw_m": 0.030101485230199444,
    "minimum_nonadjacent_robot_aabb_clearance_m": 0.002159137713071191,
    "minimum_robot_bin_aabb_clearance_m": 2.3092402665518317,
    "minimum_robot_other_book_aabb_clearance_m": 0.141249651565878,
    "minimum_robot_table_aabb_clearance_m": 2.173411050228734,
    "robot_collision_bundle_file_count": 24,
    "shelf_collision_tolerance_m": 0.00075,
    "target_fk_orientation_error_rad": 3.3753133726508316e-14,
    "target_fk_position_error_m": 4.87957859962302e-14
  },
  "checkpoint": {
    "base_quaternion_xyzw": [-9.165407789741888e-07, -2.2788479610232315e-06, -0.38256738165539433, 0.923927593747098],
    "base_world_position_m": [2.0026505847001035, -0.14951724023818858, -1.5047256624677901e-06],
    "base_world_yaw_rad": -0.7851469451686999,
    "book_floor_signed_distance_m": -1.7112924011186692e-07,
    "book_maximum_world_x_m": 2.9725668573568997,
    "book_physical_bounds_world_m": [
      [2.8101970505264386, -0.16892364469703047, 1.4518582957337598],
      [2.9725668573568997, -0.13721825034281387, 1.7031736414582506]
    ],
    "book_position_world_m": [2.891381953941669, -0.15307094751992217, 1.5775159685960052],
    "book_quaternion_xyzw": [-0.003738838979782266, 0.7100161204589502, 0.0037055847263033443, 0.7041657464072936],
    "gripper_left_geometry_joints_measured": {
      "gripper_left_finger_joint": 0.029008139290647626,
      "gripper_left_finger_right_joint": 0.006381790926295473,
      "gripper_left_fingertip_left_joint": 0.24018740390867532,
      "gripper_left_fingertip_right_joint": 0.24018740382371734,
      "gripper_left_inner_finger_left_joint": -0.24954122988783836,
      "gripper_left_inner_finger_right_joint": -0.24436293615140162,
      "gripper_left_outer_finger_left_joint": -0.24018740395486662,
      "gripper_left_outer_finger_right_joint": -0.2401874039548664
    },
    "gripper_master_m": 0.029008139290647626,
    "head_q2_measured": [-6.10080487965487e-13, -2.1016303763273987e-12],
    "left_q8_measured": [0.34999997571431013, -0.3095150715488276, 0.6178795182789477, 0.08205578083125306, -1.482472215523375, 0.17372993398979747, 1.0386147125774767, 0.21480386142221175],
    "right_q7_measured": [5.781426359381836e-07, 3.7819114738292264e-06, 4.876258033148757e-07, -7.479478725963625e-06, 1.3110881245031847e-08, 2.5247796150951337e-10, -1.1318231342399575e-10],
    "target_model": "book_col_3_row_2_red",
    "world_iterations": 172681,
    "world_paused": true,
    "world_sim_time_s": 345.362
  },
  "checkpoint_policy": {
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0008,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.0006,
    "maximum_book_position_error_m": 0.0001,
    "maximum_book_rotation_error_rad": 0.0003
  },
  "checkpoint_stable_reference_evidence": {
    "absolute_rotation_rad": 0.0005086748800145877,
    "absolute_yaw_component_rad": 0.0003817176759775565,
    "maximum_x_error_m": 0.00026440379072001363,
    "position_error_m": 0.00021981511670081903
  },
  "continuous_policy": {
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0012,
    "maximum_absolute_world_y_motion_m": 0.00007,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.0009,
    "maximum_book_center_outward_progress_m": 0.00065,
    "maximum_cross_track_motion_m": 0.0001,
    "maximum_relative_rotation_from_start_rad": 0.00085,
    "maximum_relative_yaw_from_start_rad": 0.00045,
    "minimum_book_center_outward_progress_m": -0.0001
  },
  "endpoint_policy": {
    "floor_signed_max_m": 0.00025,
    "floor_signed_min_m": -0.0001,
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0012,
    "maximum_absolute_world_y_motion_m": 0.00007,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.0009,
    "maximum_book_center_outward_progress_m": 0.00065,
    "maximum_book_from_hand_translation_mismatch_m": 0.0002,
    "maximum_cross_track_motion_m": 0.0001,
    "maximum_incremental_rotation_rad": 0.0008,
    "maximum_incremental_yaw_rad": 0.00045,
    "minimum_book_center_outward_progress_m": 0.0003,
    "minimum_deepest_extent_outward_progress_m": 0.00025,
    "pause_after_endpoint": true
  },
  "execution_policy": {
    "automatic_next_rung_allowed": false,
    "base_motion_allowed": false,
    "fixed_hand_orientation": true,
    "gazebo_entity_pose_mutation_allowed": false,
    "gripper_command_count": 0,
    "head_motion_allowed": false,
    "joint_midpoint_allowed": false,
    "left_arm_command_count": 1,
    "left_arm_only": true,
    "minimum_duration_s": 2.5,
    "pause_after_endpoint": true,
    "planned_world_delta_m": [-0.0005, 0.0, 0.0],
    "right_arm_motion_allowed": false,
    "torso_fixed": true,
    "trajectory_command_rows": 1
  },
  "final_dwell_policy": {
    "duration_s": 0.3,
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0009,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.00065,
    "maximum_relative_rotation_from_start_rad": 0.0006,
    "maximum_relative_yaw_from_start_rad": 0.00035,
    "maximum_rotation_change_rad": 0.000075,
    "maximum_rotation_growth_from_immediate_endpoint_rad": 0.00005,
    "maximum_translation_change_m": 0.00005,
    "repause_after_dwell": true
  },
  "kind": "read_only_seed101_outward_halfmillimeter_certificate",
  "pressure_policy": {
    "automatic_next_rung_allowed": false,
    "emergency_minimum_fraction": 0.5,
    "gripper_squeeze_allowed": false,
    "historical_minimum_left_force_n": 1.839596895813086,
    "historical_minimum_right_force_n": 1.810479005873424,
    "maximum_left_right_balance_change_fraction": 0.3,
    "minimum_post_to_pre_force_ratio_each_side": 0.6,
    "require_fresh_bilateral_exact_target_pressure": true,
    "target_model": "book_col_3_row_2_red"
  },
  "route": [
    {
      "minimum_duration_s": 0.0,
      "phase": "checkpoint",
      "q8": [0.34999997571431013, -0.3095150715488276, 0.6178795182789477, 0.08205578083125306, -1.482472215523375, 0.17372993398979747, 1.0386147125774767, 0.21480386142221175],
      "row": 0,
      "world_delta_m": [0.0, 0.0, 0.0]
    },
    {
      "minimum_duration_s": 2.5,
      "phase": "shelf_outward",
      "q8": [0.34999997571431013, -0.30930585016382295, 0.6177154645826406, 0.0821239227235539, -1.484092155685582, 0.17384098428524442, 1.0401021615240358, 0.21480741731946937],
      "row": 1,
      "world_delta_m": [-0.0005, 0.0, 0.0]
    }
  ],
  "route_q8_sha256": "806f3bff0edfa3bc9c487ccb2bd36dec1655c1566edd537669692556e4c0b6cd",
  "stable_reference": {
    "book_maximum_world_x_m": 2.9728312611476198,
    "book_position_world_m": [2.891599226313614, -0.15309109581985894, 1.5775425288657587],
    "book_quaternion_xyzw": [-0.003867323503119562, 0.7101335529054696, 0.003847067873253381, 0.704045865633451],
    "source": "seed101_outward_post_reseat_settle_certificate.stable_reference",
    "support_floor_world_z_m": 1.451858466863
  },
  "stage": "outward-halfmillimeter"
}'''

CERTIFICATE: Mapping[str, object] = json.loads(CERTIFICATE_JSON)
CERTIFICATE_KIND = str(CERTIFICATE['kind'])
ROUTE_Q8_SHA256 = str(CERTIFICATE['route_q8_sha256'])
STAGE = str(CERTIFICATE['stage'])
CHECKPOINT = CERTIFICATE['checkpoint']
STABLE_REFERENCE = CERTIFICATE['stable_reference']
CHECKPOINT_STABLE_REFERENCE_EVIDENCE = CERTIFICATE[
    'checkpoint_stable_reference_evidence'
]
AUDIT = CERTIFICATE['audit']
CHECKPOINT_POLICY = CERTIFICATE['checkpoint_policy']
CONTINUOUS_POLICY = CERTIFICATE['continuous_policy']
ENDPOINT_POLICY = CERTIFICATE['endpoint_policy']
FINAL_DWELL_POLICY = CERTIFICATE['final_dwell_policy']
PRESSURE_POLICY = CERTIFICATE['pressure_policy']
EXECUTION_POLICY = CERTIFICATE['execution_policy']

CHECKPOINT_BASE_WORLD_POSITION_M = tuple(CHECKPOINT['base_world_position_m'])
CHECKPOINT_BASE_QUATERNION_XYZW = tuple(CHECKPOINT['base_quaternion_xyzw'])
CHECKPOINT_BASE_WORLD_YAW_RAD = float(CHECKPOINT['base_world_yaw_rad'])
CHECKPOINT_BASE_WORLD_XYYAW = (
    *CHECKPOINT_BASE_WORLD_POSITION_M[:2],
    CHECKPOINT_BASE_WORLD_YAW_RAD,
)
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
CHECKPOINT_GRIPPER_LEFT_GEOMETRY_JOINTS = dict(
    CHECKPOINT['gripper_left_geometry_joints_measured']
)

STABLE_REFERENCE_BOOK_POSITION_WORLD_M = tuple(
    STABLE_REFERENCE['book_position_world_m']
)
STABLE_REFERENCE_BOOK_QUATERNION_XYZW = tuple(
    STABLE_REFERENCE['book_quaternion_xyzw']
)
SUPPORT_FLOOR_WORLD_Z_M = float(STABLE_REFERENCE['support_floor_world_z_m'])
ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP = tuple(
    tuple(row) for row in CERTIFICATE['attached_book_corners_physical_in_grasp']
)
ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP = tuple(
    tuple(row)
    for row in CERTIFICATE['attached_book_corners_padding_15mm_in_grasp']
)

ROUTE = tuple(CERTIFICATE['route'])
ROUTE_Q8 = tuple(tuple(row['q8']) for row in ROUTE)
COMMAND_Q8 = ROUTE_Q8[1:]
ENDPOINT_Q8 = ROUTE_Q8[1]
PLANNED_WORLD_DELTA_M = tuple(EXECUTION_POLICY['planned_world_delta_m'])
MINIMUM_DURATION_S = float(EXECUTION_POLICY['minimum_duration_s'])
FINAL_DWELL_DURATION_S = float(FINAL_DWELL_POLICY['duration_s'])
MINIMUM_FORCE_N = (
    float(PRESSURE_POLICY['emergency_minimum_fraction'])
    * float(PRESSURE_POLICY['historical_minimum_left_force_n']),
    float(PRESSURE_POLICY['emergency_minimum_fraction'])
    * float(PRESSURE_POLICY['historical_minimum_right_force_n']),
)


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
    """Fail closed if the exact state, route, audit, or pragmatic gates change."""

    maximum_joint_delta = max(
        abs(after - before)
        for before, after in zip(ROUTE_Q8[0], ROUTE_Q8[1])
    )
    dense_samples = int(AUDIT.get('dense_joint_samples', 0))
    expected_dense_samples = int(math.ceil(maximum_joint_delta / 0.0005)) + 1
    expected_dense_increment = maximum_joint_delta / max(1, dense_samples - 1)

    route_valid = bool(
        len(ROUTE) == 2
        and len(ROUTE_Q8) == 2
        and len(COMMAND_Q8) == 1
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
        and tuple(ROUTE[1]['world_delta_m']) == (-0.0005, 0.0, 0.0)
        and float(ROUTE[0]['minimum_duration_s']) == 0.0
        and float(ROUTE[1]['minimum_duration_s']) == 2.5
        and CHECKPOINT_LEFT_Q8 == ROUTE_Q8[0]
        and ENDPOINT_Q8 != tuple(
            0.5 * (before + after)
            for before, after in zip(CHECKPOINT_LEFT_Q8, ENDPOINT_Q8)
        )
    )
    checkpoint_valid = bool(
        CHECKPOINT.get('world_paused') is True
        and int(CHECKPOINT.get('world_iterations', -1)) == 172681
        and float(CHECKPOINT.get('world_sim_time_s', math.nan)) == 345.362
        and CHECKPOINT.get('target_model') == 'book_col_3_row_2_red'
        and _finite_vector(CHECKPOINT_BASE_WORLD_POSITION_M, 3)
        and _finite_vector(CHECKPOINT_BASE_QUATERNION_XYZW, 4)
        and math.isfinite(CHECKPOINT_BASE_WORLD_YAW_RAD)
        and _finite_vector(CHECKPOINT_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(CHECKPOINT_BOOK_QUATERNION_XYZW, 4)
        and len(CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M) == 2
        and all(
            _finite_vector(row, 3)
            for row in CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M
        )
        and _finite_vector(CHECKPOINT_LEFT_Q8, 8)
        and _finite_vector(CHECKPOINT_RIGHT_Q7, 7)
        and _finite_vector(CHECKPOINT_HEAD_Q2, 2)
        and math.isfinite(CHECKPOINT_GRIPPER_MASTER_M)
        and len(CHECKPOINT_GRIPPER_LEFT_GEOMETRY_JOINTS) == 8
        and all(
            math.isfinite(float(value))
            for value in CHECKPOINT_GRIPPER_LEFT_GEOMETRY_JOINTS.values()
        )
        and math.isclose(
            min(row[2] for row in CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M)
            - CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M,
            SUPPORT_FLOOR_WORLD_Z_M,
            rel_tol=0.0,
            abs_tol=5e-16,
        )
    )
    corners_valid = bool(
        all(
            len(corners) == 8
            and all(_finite_vector(row, 3) for row in corners)
            for corners in (
                ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP,
                ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP,
            )
        )
        and float(AUDIT['maximum_attached_corner_reconstruction_error_m'])
        < 2e-15
    )
    stable_valid = bool(
        _finite_vector(STABLE_REFERENCE_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(STABLE_REFERENCE_BOOK_QUATERNION_XYZW, 4)
        and float(CHECKPOINT_STABLE_REFERENCE_EVIDENCE['absolute_rotation_rad'])
        <= float(
            CHECKPOINT_POLICY[
                'maximum_absolute_rotation_to_stable_reference_rad'
            ]
        )
        and float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE[
                'absolute_yaw_component_rad'
            ]
        )
        <= float(
            CHECKPOINT_POLICY['maximum_absolute_yaw_to_stable_reference_rad']
        )
    )
    execution_valid = bool(
        STAGE == 'outward-halfmillimeter'
        and PLANNED_WORLD_DELTA_M == (-0.0005, 0.0, 0.0)
        and MINIMUM_DURATION_S == 2.5
        and int(EXECUTION_POLICY.get('trajectory_command_rows', -1)) == 1
        and int(EXECUTION_POLICY.get('left_arm_command_count', -1)) == 1
        and int(EXECUTION_POLICY.get('gripper_command_count', -1)) == 0
        and EXECUTION_POLICY.get('fixed_hand_orientation') is True
        and EXECUTION_POLICY.get('torso_fixed') is True
        and EXECUTION_POLICY.get('left_arm_only') is True
        and EXECUTION_POLICY.get('base_motion_allowed') is False
        and EXECUTION_POLICY.get('right_arm_motion_allowed') is False
        and EXECUTION_POLICY.get('head_motion_allowed') is False
        and EXECUTION_POLICY.get('gazebo_entity_pose_mutation_allowed') is False
        and EXECUTION_POLICY.get('pause_after_endpoint') is True
        and EXECUTION_POLICY.get('joint_midpoint_allowed') is False
        and EXECUTION_POLICY.get('automatic_next_rung_allowed') is False
        and all(
            math.isclose(row[0], CHECKPOINT_LEFT_Q8[0], abs_tol=1e-15)
            for row in ROUTE_Q8
        )
    )
    policy_valid = bool(
        CHECKPOINT_POLICY == {
            'maximum_absolute_rotation_to_stable_reference_rad': 0.0008,
            'maximum_absolute_yaw_to_stable_reference_rad': 0.0006,
            'maximum_book_position_error_m': 0.0001,
            'maximum_book_rotation_error_rad': 0.0003,
        }
        and CONTINUOUS_POLICY == {
            'maximum_absolute_rotation_to_stable_reference_rad': 0.0012,
            'maximum_absolute_world_y_motion_m': 0.00007,
            'maximum_absolute_yaw_to_stable_reference_rad': 0.0009,
            'maximum_book_center_outward_progress_m': 0.00065,
            'maximum_cross_track_motion_m': 0.0001,
            'maximum_relative_rotation_from_start_rad': 0.00085,
            'maximum_relative_yaw_from_start_rad': 0.00045,
            'minimum_book_center_outward_progress_m': -0.0001,
        }
        and float(ENDPOINT_POLICY['minimum_book_center_outward_progress_m'])
        == 0.0003
        and float(
            ENDPOINT_POLICY['minimum_deepest_extent_outward_progress_m']
        ) == 0.00025
        and float(ENDPOINT_POLICY['maximum_book_center_outward_progress_m'])
        == 0.00065
        and float(
            ENDPOINT_POLICY['maximum_book_from_hand_translation_mismatch_m']
        ) == 0.0002
        and float(ENDPOINT_POLICY['maximum_cross_track_motion_m']) == 0.0001
        and float(ENDPOINT_POLICY['maximum_absolute_world_y_motion_m'])
        == 0.00007
        and float(ENDPOINT_POLICY['maximum_incremental_rotation_rad'])
        == 0.0008
        and float(ENDPOINT_POLICY['maximum_incremental_yaw_rad']) == 0.00045
        and float(ENDPOINT_POLICY['floor_signed_min_m']) == -0.0001
        and float(ENDPOINT_POLICY['floor_signed_max_m']) == 0.00025
        and ENDPOINT_POLICY.get('pause_after_endpoint') is True
        and FINAL_DWELL_DURATION_S == 0.3
        and float(FINAL_DWELL_POLICY['maximum_translation_change_m'])
        == 0.00005
        and float(FINAL_DWELL_POLICY['maximum_rotation_change_rad'])
        == 0.000075
        and float(
            FINAL_DWELL_POLICY[
                'maximum_rotation_growth_from_immediate_endpoint_rad'
            ]
        ) == 0.00005
        and float(
            FINAL_DWELL_POLICY['maximum_relative_rotation_from_start_rad']
        ) == 0.0006
        and float(
            FINAL_DWELL_POLICY['maximum_relative_yaw_from_start_rad']
        ) == 0.00035
        and float(
            FINAL_DWELL_POLICY[
                'maximum_absolute_rotation_to_stable_reference_rad'
            ]
        ) == 0.0009
        and float(
            FINAL_DWELL_POLICY['maximum_absolute_yaw_to_stable_reference_rad']
        ) == 0.00065
        and FINAL_DWELL_POLICY.get('repause_after_dwell') is True
    )
    pressure_valid = bool(
        PRESSURE_POLICY.get('target_model') == 'book_col_3_row_2_red'
        and float(PRESSURE_POLICY['emergency_minimum_fraction']) == 0.5
        and float(
            PRESSURE_POLICY['minimum_post_to_pre_force_ratio_each_side']
        ) == 0.6
        and float(
            PRESSURE_POLICY['maximum_left_right_balance_change_fraction']
        ) == 0.3
        and PRESSURE_POLICY.get('require_fresh_bilateral_exact_target_pressure')
        is True
        and PRESSURE_POLICY.get('gripper_squeeze_allowed') is False
        and PRESSURE_POLICY.get('automatic_next_rung_allowed') is False
        and MINIMUM_FORCE_N == (
            0.919798447906543,
            0.905239502936712,
        )
    )
    audit_valid = bool(
        AUDIT.get('collision_free') is True
        and AUDIT.get('current_geometry_revalidation_required') is True
        and AUDIT.get('endpoint_is_exact_ik_target') is True
        and AUDIT.get('joint_midpoint_used') is False
        and int(AUDIT.get('asset_count', 0)) == 13
        and int(AUDIT.get('robot_collision_bundle_file_count', 0)) == 24
        and dense_samples == expected_dense_samples == 5
        and math.isclose(
            float(AUDIT['maximum_dense_joint_increment_rad']),
            expected_dense_increment,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and float(AUDIT['maximum_dense_joint_increment_rad']) <= 0.0005
        and float(AUDIT['target_fk_position_error_m']) < 1e-10
        and float(AUDIT['target_fk_orientation_error_rad']) < 1e-10
        and min(
            float(AUDIT['minimum_nonadjacent_robot_aabb_clearance_m']),
            float(AUDIT['minimum_15mm_padded_payload_robot_aabb_clearance_m']),
            float(
                AUDIT[
                    'minimum_closed_gripper_shelf_clearance_after_tolerance_m'
                ]
            ),
            float(AUDIT['minimum_robot_other_book_aabb_clearance_m']),
            float(AUDIT['minimum_carried_book_other_book_aabb_clearance_m']),
            float(AUDIT['minimum_robot_table_aabb_clearance_m']),
            float(AUDIT['minimum_robot_bin_aabb_clearance_m']),
        ) > 0.0
    )
    valid = bool(
        CERTIFICATE_KIND
        == 'read_only_seed101_outward_halfmillimeter_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and route_valid
        and checkpoint_valid
        and corners_valid
        and stable_valid
        and execution_valid
        and policy_valid
        and pressure_valid
        and audit_valid
    )
    return valid, {
        'route_rows': float(len(ROUTE_Q8)),
        'command_rows': float(len(COMMAND_Q8)),
        'maximum_joint_delta_rad': maximum_joint_delta,
        'maximum_dense_joint_increment_rad': expected_dense_increment,
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
        'minimum_left_force_n': MINIMUM_FORCE_N[0],
        'minimum_right_force_n': MINIMUM_FORCE_N[1],
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
    'CHECKPOINT_BASE_WORLD_XYYAW',
    'CHECKPOINT_BASE_WORLD_YAW_RAD',
    'CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M',
    'CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M',
    'CHECKPOINT_BOOK_POSITION_WORLD_M',
    'CHECKPOINT_BOOK_QUATERNION_XYZW',
    'CHECKPOINT_GRIPPER_LEFT_GEOMETRY_JOINTS',
    'CHECKPOINT_GRIPPER_MASTER_M',
    'CHECKPOINT_HEAD_Q2',
    'CHECKPOINT_LEFT_Q8',
    'CHECKPOINT_POLICY',
    'CHECKPOINT_RIGHT_Q7',
    'CHECKPOINT_STABLE_REFERENCE_EVIDENCE',
    'COMMAND_Q8',
    'CONTINUOUS_POLICY',
    'ENDPOINT_POLICY',
    'ENDPOINT_Q8',
    'EXECUTION_POLICY',
    'FINAL_DWELL_DURATION_S',
    'FINAL_DWELL_POLICY',
    'MINIMUM_DURATION_S',
    'MINIMUM_FORCE_N',
    'PLANNED_WORLD_DELTA_M',
    'PRESSURE_POLICY',
    'ROUTE',
    'ROUTE_Q8',
    'ROUTE_Q8_SHA256',
    'STABLE_REFERENCE',
    'STABLE_REFERENCE_BOOK_POSITION_WORLD_M',
    'STABLE_REFERENCE_BOOK_QUATERNION_XYZW',
    'STAGE',
    'SUPPORT_FLOOR_WORLD_Z_M',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
