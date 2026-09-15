"""Inert certificate for a zero-command mid-route settling observation.

The live half-millimetre shelf-outward rung stopped after one relative-yaw
sample exceeded its soft limit by 4.389 microradians.  This certificate binds
the resulting paused state and authorizes one absolute run-to-simulation-time
request covering exactly 150 2 ms physics steps (0.30 seconds total), with
Gazebo automatically pausing at the endpoint.  It authorizes no robot,
gripper, base, head, trajectory, or entity-pose command.

Gazebo's paused state service preserves joints and contact data as binary
protobuf doubles, but serializes model poses through a six-significant-digit
Pose3 stream.  The last persisted full-double dynamic-pose sample is retained
as the checkpoint reference; the exact current serialized strings and their
conservative quantization bounds are independently bound below.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Tuple


CERTIFICATE_SOURCE_JSON_SHA256 = (
    '3d6b0a05d2c34796189e91a5fbb60bc15087f5579228a74d4f928675f9995942'
)
CERTIFICATE_SEMANTIC_DIGEST = (
    '1204d643501b2dd27a16aff5e1db6db1e04c874c73f369652a9fb668a07cd1f4'
)
CERTIFICATE_JSON = r'''{
  "attached_book_corners_padding_15mm_in_grasp": [
    [0.059758050424108665, 0.02991539839418801, -0.13594385907209872],
    [0.22571277731972517, 0.028196869027448564, -0.04344480352115318],
    [0.05928194719212951, -0.030082147103096445, -0.13620435861870148],
    [0.2252366740877464, -0.031800676469835885, -0.043705303067755705],
    [-0.0765617663827755, 0.029935243226923702, 0.10863109294040842],
    [0.08939296051284142, 0.028216713860184255, 0.2011301484913542],
    [-0.07703786961475426, -0.030062302270360754, 0.10837059339380588],
    [0.08891685728086224, -0.03178083163710021, 0.20086964894475143]
  ],
  "attached_book_corners_physical_in_grasp": [
    [0.06543786625261687, 0.014781401764847947, -0.11560419736947093],
    [0.2051892152173469, 0.01333421914022523, -0.03771025585288511],
    [0.06519981463662748, -0.01521737098379428, -0.11573444714277219],
    [0.20495116360135754, -0.016664553608416968, -0.03784050562618639],
    [-0.056276255896386636, 0.014799120365504783, 0.10276629549883909],
    [0.08347509306834341, 0.013351937740882091, 0.18066023701542488],
    [-0.05651430751237603, -0.01519965238313742, 0.10263604572553782],
    [0.08323704145235405, -0.016646835007760142, 0.18052998724212363]
  ],
  "audit": {
    "asset_count": 13,
    "attached_corners_recomputed_from_current_exact_dynamic_pose": true,
    "attached_corners_source": "current_exact_book_pose_base_pose_left_q8_and_tiago_urdf_fk",
    "collision_free": true,
    "current_q8_fraction_of_halfmillimeter_route": 0.0956405834366858,
    "current_q8_maximum_residual_from_route_rad": 9.771754976850566e-07,
    "current_state_pose_maximum_position_quantization_m": 5e-06,
    "current_state_pose_maximum_rotation_quantization_rad": 2e-05,
    "current_state_service": "/world/erc_world/state",
    "current_state_service_response_type": "gz.msgs.SerializedStepMap",
    "inherited_collision_audit_applies_to_stationary_checkpoint_only": true,
    "inherited_from_certificate_kind": "read_only_seed101_outward_halfmillimeter_certificate",
    "inherited_minimum_15mm_padded_payload_robot_aabb_clearance_m": 0.10135749232271651,
    "inherited_minimum_closed_gripper_shelf_clearance_after_tolerance_m": 0.029351485230199443,
    "inherited_minimum_nonadjacent_robot_aabb_clearance_m": 0.002159137713071191,
    "maximum_15mm_padded_corner_change_from_prior_attachment_m": 3.365447569996587e-05,
    "no_live_mutation_during_capture": true,
    "robot_collision_bundle_file_count": 24,
    "stationary_world_delta_m": [0.0, 0.0, 0.0],
    "zero_command_state_only": true
  },
  "checkpoint": {
    "base_quaternion_xyzw": [9.978588766464317e-08, 2.3035211189417835e-07, -0.38256263060418283, 0.923929560986737],
    "base_world_position_m": [2.002648039946845, -0.1495205290917588, -1.6957408982813647e-07],
    "base_world_yaw_rad": -0.7851366607105121,
    "book_floor_signed_distance_m": -1.7587426781595639e-07,
    "book_maximum_world_x_m": 2.9725116515837784,
    "book_physical_bounds_world_m": [
      [2.81017388582961, -0.16891399592327924, 1.451858290988732],
      [2.9725116515837784, -0.13720806303553643, 1.703156322483965]
    ],
    "book_position_world_m": [2.891342768706694, -0.15306102947940783, 1.5775073067363485],
    "book_quaternion_xyzw": [-0.0037302502014055204, 0.7099740343741349, 0.0036603078421359346, 0.704208461958851],
    "gripper_left_geometry_joints_measured": {
      "gripper_left_finger_joint": 0.029008131671675274,
      "gripper_left_finger_right_joint": 0.006381789059757022,
      "gripper_left_fingertip_left_joint": 0.24018733309685808,
      "gripper_left_fingertip_right_joint": 0.2401873330656241,
      "gripper_left_inner_finger_left_joint": -0.24987151315243236,
      "gripper_left_inner_finger_right_joint": -0.244223243580627,
      "gripper_left_outer_finger_left_joint": -0.2401873334298026,
      "gripper_left_outer_finger_right_joint": -0.2401873334804848
    },
    "gripper_master_m": 0.029008131671675274,
    "head_q2_measured": [5.1484347686901044e-11, -2.1052812478162052e-12],
    "last_exact_dynamic_pose_sim_time_s": 346.66,
    "left_q8_measured": [0.349999950536834, -0.30949603866899605, 0.6178645057946034, 0.08206232481188921, -1.4826272057601544, 0.17374047013531607, 1.0387571269500129, 0.21480420336300401],
    "pose_reference_source": "current_paused_full_double_gz_dynamic_pose_sample",
    "right_gripper_q8_measured": [2.272303802860943e-06, 4.999942089480129e-07, 1.881785942205744e-05, 1.8817964821676555e-05, -1.8817851503797678e-05, -1.8818014906062548e-05, -1.88178820724499e-05, -1.8817966588333597e-05],
    "right_q7_measured": [9.4989315444127e-07, 3.0243413967052386e-06, 8.046789091558862e-07, -7.445534058304117e-06, 2.5429898449800823e-08, 1.4969459015825713e-09, -4.60363339585441e-10],
    "serialized_state_corroboration": {
      "base_derived_quaternion_xyzw": [9.978593405025428e-08, 2.3035211651254125e-07, -0.38256278734397114, 0.9239294960869842],
      "base_pose_text": "2.00265 -0.149521 -1.69574e-07 8.14204e-09 5.02007e-07 -0.785137",
      "base_pose_xyz_rpy": [2.00265, -0.149521, -1.69574e-07, 8.14204e-09, 5.02007e-07, -0.785137],
      "base_reference_position_error_m": 2.0158281031909113e-06,
      "base_reference_rotation_error_rad": 3.392894878384613e-07,
      "base_reference_yaw_error_rad": 3.392894878384583e-07,
      "book_derived_quaternion_xyzw": [-0.00372955692141751, 0.7099747259105619, 0.003659613873483387, 0.7042077720386857],
      "book_pose_text": "2.89134 -0.153061 1.57751 -3.13469 1.56264 -3.12424",
      "book_pose_xyz_rpy": [2.89134, -0.153061, 1.57751, -3.13469, 1.56264, -3.12424],
      "book_reference_position_error_m": 3.862677166131498e-06,
      "book_reference_rotation_error_rad": 2.7687124061485808e-06,
      "book_reference_yaw_component_error_rad": 1.961927342403991e-06
    },
    "target_model": "book_col_3_row_2_red",
    "torso_lift_joint_measured_m": 0.349999950536834,
    "world_iterations": 173330,
    "world_paused": true,
    "world_sim_time_s": 346.66
  },
  "checkpoint_policy": {
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0009,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.00065,
    "maximum_book_position_error_m": 0.0001,
    "maximum_book_rotation_error_rad": 0.0003
  },
  "checkpoint_stable_reference_evidence": {
    "current_exact_absolute_rotation_rad": 0.000649775736903032,
    "current_exact_absolute_yaw_component_rad": 0.0004576731837770565,
    "current_exact_floor_signed_distance_m": -1.7587426781595639e-07,
    "current_exact_maximum_x_error_m": 0.00031960956384136097,
    "current_exact_position_error_m": 0.0002606052328240994,
    "prior_exact_dynamic_absolute_rotation_rad": 0.0005086748800145877,
    "prior_exact_dynamic_absolute_yaw_component_rad": 0.0003817176759775565
  },
  "continuous_policy": {
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0012,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.0009,
    "maximum_passive_scene_rotation_from_reference_rad": 0.0002,
    "maximum_passive_scene_translation_from_reference_m": 0.0001,
    "maximum_relative_rotation_from_checkpoint_rad": 0.00085,
    "maximum_stationary_translation_from_checkpoint_m": 0.0001,
    "require_exact_target_bilateral_pressure": true,
    "require_non_target_scene_stable": true
  },
  "execution_policy": {
    "automatic_next_stage_allowed": false,
    "base_command_count": 0,
    "base_motion_allowed": false,
    "gazebo_entity_pose_command_count": 0,
    "gazebo_entity_pose_mutation_allowed": false,
    "gripper_command_count": 0,
    "gripper_squeeze_allowed": false,
    "head_command_count": 0,
    "left_arm_command_count": 0,
    "maximum_resume_request_count": 1,
    "maximum_resume_duration_s": 0.3,
    "minimum_resume_duration_s": 0.3,
    "physics_step_size_s": 0.002,
    "repause_after_settle": true,
    "run_to_sim_time_required": true,
    "right_arm_command_count": 0,
    "safety_zero_velocity_publish_allowed": true,
    "settle_step_count": 150,
    "trajectory_command_rows": 0,
    "world_resume_allowed_for_settle_only": true
  },
  "final_dwell_policy": {
    "duration_s": 0.3,
    "floor_signed_max_m": 0.00025,
    "floor_signed_min_m": -0.0001,
    "maximum_absolute_rotation_to_stable_reference_rad": 0.0009,
    "maximum_absolute_yaw_to_stable_reference_rad": 0.00065,
    "maximum_passive_scene_rotation_from_reference_rad": 0.0002,
    "maximum_passive_scene_translation_from_reference_m": 0.0001,
    "maximum_rotation_change_rad": 0.0002,
    "maximum_translation_change_m": 0.0001,
    "repause_after_dwell": true,
    "require_exact_target_bilateral_pressure": true,
    "require_non_target_scene_stable": true
  },
  "kind": "read_only_seed101_outward_halfmillimeter_midroute_settle_certificate",
  "passive_scene_reference": {
    "book_col_1_row_2_red": "2.9 1.98357 1.57686 0.00371287 1.57079 0.00371287",
    "book_col_1_row_3_blue": "2.9 2.08174 1.24686 -0.00148204 1.57079 -0.00148204",
    "book_col_1_row_4_yellow": "2.9 1.85726 0.916858 0.00656179 1.57079 0.00656179",
    "book_col_1_row_5_green": "2.9 1.86085 0.586856 0.00656126 1.57079 0.00656127",
    "book_col_2_row_2_blue": "2.9 0.914519 1.57686 0.00505171 1.57079 0.00505171",
    "book_col_2_row_3_red": "2.9 0.968991 1.24686 0.00272248 1.57079 0.00272248",
    "book_col_2_row_4_green": "2.9 0.877621 0.916858 0.00600895 1.57079 0.00600895",
    "book_col_2_row_5_yellow": "2.9 0.830533 0.586856 0.00654027 1.57079 0.00654027",
    "book_col_3_row_3_green": "2.9 0.143139 1.24686 -0.00608876 1.57079 -0.00608876",
    "book_col_3_row_4_yellow": "2.9 -0.13059 0.916858 0.00536571 1.57079 0.00536571",
    "book_col_3_row_5_blue": "2.9 0.145597 0.586858 -0.00611561 1.57079 -0.00611561",
    "book_col_4_row_2_green": "2.9 -1.03492 1.57686 -0.000781476 1.57079 -0.000781476",
    "book_col_4_row_3_yellow": "2.9 -1.09786 1.24686 0.00256359 1.57079 0.00256359",
    "book_col_4_row_4_red": "2.9 -1.1946 0.916858 0.00605084 1.57079 0.00605084",
    "book_col_4_row_5_blue": "2.9 -0.838304 0.586857 -0.00630588 1.57079 -0.00630588",
    "book_col_5_row_2_red": "2.9 -2.05268 1.57686 -0.00163805 1.57079 -0.00163805",
    "book_col_5_row_3_green": "2.9 -2.06036 1.24686 -0.00120174 1.57079 -0.00120174",
    "book_col_5_row_4_blue": "2.9 -2.03852 0.916858 -0.00229614 1.57079 -0.00229614",
    "book_col_5_row_5_yellow": "2.9 -2.04772 0.586858 -0.00195894 1.57079 -0.00195894",
    "erc_collection_bin": "-1 2.2797e-10 0.845 1.5708 -2.16827e-09 -1.5708",
    "erc_shelf": "3 0 1.1 1.5708 -5.55112e-17 -1.5708",
    "erc_table": "-1 0 0.7 1.5708 -5.55112e-17 -1.5708"
  },
  "passive_scene_reference_sha256": "2185f1ef95297a18426a45b830877bc30a16f865961100acf0ba96a35316c9f3",
  "pressure_policy": {
    "automatic_next_stage_allowed": false,
    "current_paused_left_force_n": 1.9585742341509955,
    "current_paused_left_target_contact": true,
    "current_paused_right_force_n": 2.2315443416534047,
    "current_paused_right_target_contact": true,
    "current_paused_sample_is_last_physics_step": true,
    "emergency_minimum_fraction": 0.5,
    "fresh_pressure_acquisition_deadline_s": 0.1,
    "gripper_squeeze_allowed": false,
    "historical_minimum_left_force_n": 1.839596895813086,
    "historical_minimum_right_force_n": 1.6923703650649269,
    "maximum_left_right_balance_change_fraction": 0.3,
    "minimum_post_to_pre_force_ratio_each_side": 0.6,
    "preleg_left_force_n": 1.9095794869575362,
    "preleg_left_samples": 64,
    "preleg_right_force_n": 1.6923703650649269,
    "preleg_right_samples": 64,
    "require_fresh_bilateral_exact_target_pressure_after_resume": true,
    "target_model": "book_col_3_row_2_red"
  },
  "relative_yaw_debounce_policy": {
    "consecutive_samples_required": 2,
    "distinct_pose_generations_required": true,
    "reset_below_soft_limit": true,
    "relative_yaw_immediate_hard_limit_rad": 0.00085,
    "relative_yaw_soft_limit_rad": 0.00045,
    "soft_limit_comparison": "strictly_greater_than"
  },
  "route": [
    {
      "minimum_duration_s": 0.0,
      "phase": "checkpoint",
      "q8": [0.349999950536834, -0.30949603866899605, 0.6178645057946034, 0.08206232481188921, -1.4826272057601544, 0.17374047013531607, 1.0387571269500129, 0.21480420336300401],
      "row": 0,
      "world_delta_m": [0.0, 0.0, 0.0]
    }
  ],
  "route_q8_sha256": "ffe33de1a93fc590dda48d8bf76de0c00182ce58b24593bd1569bb5816cb3a0d",
  "stable_reference": {
    "book_maximum_world_x_m": 2.9728312611476198,
    "book_position_world_m": [2.891599226313614, -0.15309109581985894, 1.5775425288657587],
    "book_quaternion_xyzw": [-0.003867323503119562, 0.7101335529054696, 0.003847067873253381, 0.704045865633451],
    "source": "seed101_outward_post_reseat_settle_certificate.stable_reference",
    "support_floor_world_z_m": 1.451858466863
  },
  "stage": "outward-halfmillimeter-midroute-settle"
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
SERIALIZED_STATE_CORROBORATION = CHECKPOINT['serialized_state_corroboration']
AUDIT = CERTIFICATE['audit']
CHECKPOINT_POLICY = CERTIFICATE['checkpoint_policy']
CONTINUOUS_POLICY = CERTIFICATE['continuous_policy']
RELATIVE_YAW_DEBOUNCE_POLICY = CERTIFICATE['relative_yaw_debounce_policy']
FINAL_DWELL_POLICY = CERTIFICATE['final_dwell_policy']
PRESSURE_POLICY = CERTIFICATE['pressure_policy']
EXECUTION_POLICY = CERTIFICATE['execution_policy']
PASSIVE_SCENE_REFERENCE = CERTIFICATE['passive_scene_reference']
PASSIVE_SCENE_REFERENCE_SHA256 = str(
    CERTIFICATE['passive_scene_reference_sha256']
)

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
CHECKPOINT_RIGHT_GRIPPER_Q8 = tuple(CHECKPOINT['right_gripper_q8_measured'])
CHECKPOINT_HEAD_Q2 = tuple(CHECKPOINT['head_q2_measured'])
CHECKPOINT_GRIPPER_MASTER_M = float(CHECKPOINT['gripper_master_m'])
CHECKPOINT_GRIPPER_LEFT_GEOMETRY_JOINTS = dict(
    CHECKPOINT['gripper_left_geometry_joints_measured']
)
CURRENT_SERIALIZED_BOOK_POSITION_WORLD_M = tuple(
    SERIALIZED_STATE_CORROBORATION['book_pose_xyz_rpy'][:3]
)
CURRENT_SERIALIZED_BOOK_RPY_RAD = tuple(
    SERIALIZED_STATE_CORROBORATION['book_pose_xyz_rpy'][3:]
)
CURRENT_SERIALIZED_BOOK_QUATERNION_XYZW = tuple(
    SERIALIZED_STATE_CORROBORATION['book_derived_quaternion_xyzw']
)
CURRENT_SERIALIZED_BASE_POSITION_WORLD_M = tuple(
    SERIALIZED_STATE_CORROBORATION['base_pose_xyz_rpy'][:3]
)
CURRENT_SERIALIZED_BASE_RPY_RAD = tuple(
    SERIALIZED_STATE_CORROBORATION['base_pose_xyz_rpy'][3:]
)
CURRENT_SERIALIZED_BASE_QUATERNION_XYZW = tuple(
    SERIALIZED_STATE_CORROBORATION['base_derived_quaternion_xyzw']
)
STABLE_REFERENCE_BOOK_POSITION_WORLD_M = tuple(
    STABLE_REFERENCE['book_position_world_m']
)
STABLE_REFERENCE_BOOK_QUATERNION_XYZW = tuple(
    STABLE_REFERENCE['book_quaternion_xyzw']
)
SUPPORT_FLOOR_WORLD_Z_M = float(STABLE_REFERENCE['support_floor_world_z_m'])
SETTLE_DURATION_S = float(FINAL_DWELL_POLICY['duration_s'])
PHYSICS_STEP_SIZE_S = float(EXECUTION_POLICY['physics_step_size_s'])
SETTLE_STEP_COUNT = int(EXECUTION_POLICY['settle_step_count'])
SETTLE_STABILITY_TRANSLATION_LIMIT_M = float(
    FINAL_DWELL_POLICY['maximum_translation_change_m']
)
SETTLE_STABILITY_ROTATION_LIMIT_RAD = float(
    FINAL_DWELL_POLICY['maximum_rotation_change_rad']
)
FLOOR_SIGNED_RANGE_M = (
    float(FINAL_DWELL_POLICY['floor_signed_min_m']),
    float(FINAL_DWELL_POLICY['floor_signed_max_m']),
)
MINIMUM_FORCE_N = (
    float(PRESSURE_POLICY['emergency_minimum_fraction'])
    * float(PRESSURE_POLICY['historical_minimum_left_force_n']),
    float(PRESSURE_POLICY['emergency_minimum_fraction'])
    * float(PRESSURE_POLICY['historical_minimum_right_force_n']),
)
ROUTE = tuple(CERTIFICATE['route'])
ROUTE_Q8 = tuple(tuple(row['q8']) for row in ROUTE)
COMMAND_Q8 = ROUTE_Q8[1:]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode('utf-8')


def semantic_digest() -> str:
    return hashlib.sha256(_canonical_bytes(CERTIFICATE)).hexdigest()


def source_json_digest() -> str:
    return hashlib.sha256(CERTIFICATE_JSON.encode('utf-8')).hexdigest()


def route_q8_digest() -> str:
    return hashlib.sha256(
        _canonical_bytes([list(row) for row in ROUTE_Q8])
    ).hexdigest()


def passive_scene_digest() -> str:
    return hashlib.sha256(_canonical_bytes(PASSIVE_SCENE_REFERENCE)).hexdigest()


def _finite_vector(values: object, length: int) -> bool:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return False
    return len(vector) == length and all(math.isfinite(value) for value in vector)


def validate_certificate() -> Tuple[bool, Mapping[str, float]]:
    """Fail closed if the paused state or zero-command settle policy changes."""

    route_valid = bool(
        len(ROUTE) == 1
        and len(ROUTE_Q8) == 1
        and len(COMMAND_Q8) == 0
        and isinstance(ROUTE[0], Mapping)
        and int(ROUTE[0].get('row', -1)) == 0
        and ROUTE[0].get('phase') == 'checkpoint'
        and _finite_vector(ROUTE[0].get('q8'), 8)
        and ROUTE_Q8[0] == CHECKPOINT_LEFT_Q8
        and tuple(ROUTE[0].get('world_delta_m', ())) == (0.0, 0.0, 0.0)
        and float(ROUTE[0].get('minimum_duration_s', math.nan)) == 0.0
    )
    checkpoint_valid = bool(
        CHECKPOINT.get('world_paused') is True
        and int(CHECKPOINT.get('world_iterations', -1)) == 173330
        and float(CHECKPOINT.get('world_sim_time_s', math.nan)) == 346.66
        and CHECKPOINT.get('target_model') == 'book_col_3_row_2_red'
        and CHECKPOINT.get('pose_reference_source')
        == 'current_paused_full_double_gz_dynamic_pose_sample'
        and _finite_vector(CHECKPOINT_BASE_WORLD_POSITION_M, 3)
        and _finite_vector(CHECKPOINT_BASE_QUATERNION_XYZW, 4)
        and math.isfinite(CHECKPOINT_BASE_WORLD_YAW_RAD)
        and _finite_vector(CHECKPOINT_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(CHECKPOINT_BOOK_QUATERNION_XYZW, 4)
        and _finite_vector(CHECKPOINT_LEFT_Q8, 8)
        and _finite_vector(CHECKPOINT_RIGHT_Q7, 7)
        and _finite_vector(CHECKPOINT_RIGHT_GRIPPER_Q8, 8)
        and _finite_vector(CHECKPOINT_HEAD_Q2, 2)
        and math.isfinite(CHECKPOINT_GRIPPER_MASTER_M)
        and len(CHECKPOINT_GRIPPER_LEFT_GEOMETRY_JOINTS) == 8
        and all(
            math.isfinite(float(value))
            for value in CHECKPOINT_GRIPPER_LEFT_GEOMETRY_JOINTS.values()
        )
        and SERIALIZED_STATE_CORROBORATION['book_pose_text']
        == '2.89134 -0.153061 1.57751 -3.13469 1.56264 -3.12424'
        and SERIALIZED_STATE_CORROBORATION['base_pose_text']
        == '2.00265 -0.149521 -1.69574e-07 8.14204e-09 5.02007e-07 -0.785137'
        and float(
            SERIALIZED_STATE_CORROBORATION[
                'book_reference_position_error_m'
            ]
        ) <= float(CHECKPOINT_POLICY['maximum_book_position_error_m'])
        and float(
            SERIALIZED_STATE_CORROBORATION[
                'book_reference_rotation_error_rad'
            ]
        ) <= float(CHECKPOINT_POLICY['maximum_book_rotation_error_rad'])
    )
    reference_valid = bool(
        _finite_vector(STABLE_REFERENCE_BOOK_POSITION_WORLD_M, 3)
        and _finite_vector(STABLE_REFERENCE_BOOK_QUATERNION_XYZW, 4)
        and SUPPORT_FLOOR_WORLD_Z_M == 1.451858466863
        and CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M
        == -1.7587426781595639e-07
        and math.isclose(
            min(row[2] for row in CHECKPOINT_BOOK_PHYSICAL_BOUNDS_WORLD_M)
            - CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M,
            SUPPORT_FLOOR_WORLD_Z_M,
            rel_tol=0.0,
            abs_tol=5e-16,
        )
        and float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE[
                'current_exact_absolute_rotation_rad'
            ]
        ) <= float(
            CHECKPOINT_POLICY[
                'maximum_absolute_rotation_to_stable_reference_rad'
            ]
        )
        and float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE[
                'current_exact_absolute_yaw_component_rad'
            ]
        ) <= float(
            CHECKPOINT_POLICY['maximum_absolute_yaw_to_stable_reference_rad']
        )
    )
    execution_valid = bool(
        STAGE == 'outward-halfmillimeter-midroute-settle'
        and SETTLE_DURATION_S == 0.3
        and int(EXECUTION_POLICY.get('trajectory_command_rows', -1)) == 0
        and all(
            int(EXECUTION_POLICY.get(key, -1)) == 0
            for key in (
                'base_command_count',
                'gazebo_entity_pose_command_count',
                'gripper_command_count',
                'head_command_count',
                'left_arm_command_count',
                'right_arm_command_count',
            )
        )
        and EXECUTION_POLICY.get('world_resume_allowed_for_settle_only') is True
        and EXECUTION_POLICY.get('repause_after_settle') is True
        and EXECUTION_POLICY.get('gazebo_entity_pose_mutation_allowed') is False
        and EXECUTION_POLICY.get('base_motion_allowed') is False
        and EXECUTION_POLICY.get('gripper_squeeze_allowed') is False
        and EXECUTION_POLICY.get('automatic_next_stage_allowed') is False
        and EXECUTION_POLICY.get('run_to_sim_time_required') is True
        and float(EXECUTION_POLICY['physics_step_size_s']) == 0.002
        and int(EXECUTION_POLICY['settle_step_count']) == 150
        and int(EXECUTION_POLICY['maximum_resume_request_count']) == 1
        and math.isclose(
            float(EXECUTION_POLICY['physics_step_size_s'])
            * int(EXECUTION_POLICY['settle_step_count']),
            SETTLE_DURATION_S,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and float(EXECUTION_POLICY['minimum_resume_duration_s']) == 0.3
        and float(EXECUTION_POLICY['maximum_resume_duration_s']) == 0.3
    )
    policy_valid = bool(
        CHECKPOINT_POLICY == {
            'maximum_absolute_rotation_to_stable_reference_rad': 0.0009,
            'maximum_absolute_yaw_to_stable_reference_rad': 0.00065,
            'maximum_book_position_error_m': 0.0001,
            'maximum_book_rotation_error_rad': 0.0003,
        }
        and CONTINUOUS_POLICY == {
            'maximum_absolute_rotation_to_stable_reference_rad': 0.0012,
            'maximum_absolute_yaw_to_stable_reference_rad': 0.0009,
            'maximum_passive_scene_rotation_from_reference_rad': 0.0002,
            'maximum_passive_scene_translation_from_reference_m': 0.0001,
            'maximum_relative_rotation_from_checkpoint_rad': 0.00085,
            'maximum_stationary_translation_from_checkpoint_m': 0.0001,
            'require_exact_target_bilateral_pressure': True,
            'require_non_target_scene_stable': True,
        }
        and RELATIVE_YAW_DEBOUNCE_POLICY == {
            'consecutive_samples_required': 2,
            'distinct_pose_generations_required': True,
            'reset_below_soft_limit': True,
            'relative_yaw_immediate_hard_limit_rad': 0.00085,
            'relative_yaw_soft_limit_rad': 0.00045,
            'soft_limit_comparison': 'strictly_greater_than',
        }
        and FINAL_DWELL_POLICY == {
            'duration_s': 0.3,
            'floor_signed_max_m': 0.00025,
            'floor_signed_min_m': -0.0001,
            'maximum_absolute_rotation_to_stable_reference_rad': 0.0009,
            'maximum_absolute_yaw_to_stable_reference_rad': 0.00065,
            'maximum_passive_scene_rotation_from_reference_rad': 0.0002,
            'maximum_passive_scene_translation_from_reference_m': 0.0001,
            'maximum_rotation_change_rad': 0.0002,
            'maximum_translation_change_m': 0.0001,
            'repause_after_dwell': True,
            'require_exact_target_bilateral_pressure': True,
            'require_non_target_scene_stable': True,
        }
        and SETTLE_STABILITY_TRANSLATION_LIMIT_M == 0.0001
        and SETTLE_STABILITY_ROTATION_LIMIT_RAD == 0.0002
        and FLOOR_SIGNED_RANGE_M == (-0.0001, 0.00025)
    )
    pressure_valid = bool(
        PRESSURE_POLICY.get('target_model') == 'book_col_3_row_2_red'
        and PRESSURE_POLICY.get('current_paused_left_target_contact') is True
        and PRESSURE_POLICY.get('current_paused_right_target_contact') is True
        and PRESSURE_POLICY.get('current_paused_sample_is_last_physics_step')
        is True
        and PRESSURE_POLICY.get(
            'require_fresh_bilateral_exact_target_pressure_after_resume'
        ) is True
        and float(PRESSURE_POLICY['emergency_minimum_fraction']) == 0.5
        and float(PRESSURE_POLICY['fresh_pressure_acquisition_deadline_s']) == 0.1
        and float(PRESSURE_POLICY['current_paused_left_force_n'])
        >= MINIMUM_FORCE_N[0]
        and float(PRESSURE_POLICY['current_paused_right_force_n'])
        >= MINIMUM_FORCE_N[1]
        and int(PRESSURE_POLICY['preleg_left_samples']) == 64
        and int(PRESSURE_POLICY['preleg_right_samples']) == 64
        and float(PRESSURE_POLICY['minimum_post_to_pre_force_ratio_each_side'])
        == 0.6
        and float(
            PRESSURE_POLICY['maximum_left_right_balance_change_fraction']
        ) == 0.3
        and PRESSURE_POLICY.get('gripper_squeeze_allowed') is False
        and PRESSURE_POLICY.get('automatic_next_stage_allowed') is False
        and MINIMUM_FORCE_N == (
            0.919798447906543,
            0.8461851825324634,
        )
    )
    audit_valid = bool(
        AUDIT.get('collision_free') is True
        and AUDIT.get('zero_command_state_only') is True
        and AUDIT.get('no_live_mutation_during_capture') is True
        and AUDIT.get('attached_corners_recomputed_from_current_exact_dynamic_pose')
        is True
        and AUDIT.get(
            'inherited_collision_audit_applies_to_stationary_checkpoint_only'
        ) is True
        and tuple(AUDIT.get('stationary_world_delta_m', ()))
        == (0.0, 0.0, 0.0)
        and int(AUDIT.get('asset_count', 0)) == 13
        and int(AUDIT.get('robot_collision_bundle_file_count', 0)) == 24
        and float(AUDIT['current_state_pose_maximum_position_quantization_m'])
        <= 5e-06
        and float(AUDIT['current_state_pose_maximum_rotation_quantization_rad'])
        <= 2e-05
        and min(
            float(
                AUDIT[
                    'inherited_minimum_nonadjacent_robot_aabb_clearance_m'
                ]
            ),
            float(
                AUDIT[
                    'inherited_minimum_15mm_padded_payload_robot_aabb_clearance_m'
                ]
            ),
            float(
                AUDIT[
                    'inherited_minimum_closed_gripper_shelf_clearance_after_tolerance_m'
                ]
            ),
        ) > 0.0
        and len(ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP) == 8
        and len(ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP) == 8
        and all(
            _finite_vector(row, 3)
            for corners in (
                ATTACHED_BOOK_CORNERS_PHYSICAL_IN_GRASP,
                ATTACHED_BOOK_CORNERS_PADDING_15MM_IN_GRASP,
            )
            for row in corners
        )
        and len(PASSIVE_SCENE_REFERENCE) == 22
        and passive_scene_digest() == PASSIVE_SCENE_REFERENCE_SHA256
    )
    valid = bool(
        CERTIFICATE_KIND
        == 'read_only_seed101_outward_halfmillimeter_midroute_settle_certificate'
        and semantic_digest() == CERTIFICATE_SEMANTIC_DIGEST
        and source_json_digest() == CERTIFICATE_SOURCE_JSON_SHA256
        and route_q8_digest() == ROUTE_Q8_SHA256
        and route_valid
        and checkpoint_valid
        and reference_valid
        and execution_valid
        and policy_valid
        and pressure_valid
        and audit_valid
    )
    return valid, {
        'route_rows': float(len(ROUTE_Q8)),
        'command_rows': float(len(COMMAND_Q8)),
        'settle_duration_s': SETTLE_DURATION_S,
        'checkpoint_reference_position_error_m': float(
            SERIALIZED_STATE_CORROBORATION['book_reference_position_error_m']
        ),
        'checkpoint_reference_rotation_error_rad': float(
            SERIALIZED_STATE_CORROBORATION['book_reference_rotation_error_rad']
        ),
        'checkpoint_stable_rotation_rad': float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE[
                'current_exact_absolute_rotation_rad'
            ]
        ),
        'checkpoint_stable_yaw_component_rad': float(
            CHECKPOINT_STABLE_REFERENCE_EVIDENCE[
                'current_exact_absolute_yaw_component_rad'
            ]
        ),
        'checkpoint_floor_signed_distance_m': CHECKPOINT_BOOK_FLOOR_SIGNED_DISTANCE_M,
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
    'CHECKPOINT_RIGHT_GRIPPER_Q8',
    'CHECKPOINT_RIGHT_Q7',
    'CHECKPOINT_STABLE_REFERENCE_EVIDENCE',
    'COMMAND_Q8',
    'CONTINUOUS_POLICY',
    'CURRENT_SERIALIZED_BASE_POSITION_WORLD_M',
    'CURRENT_SERIALIZED_BASE_QUATERNION_XYZW',
    'CURRENT_SERIALIZED_BASE_RPY_RAD',
    'CURRENT_SERIALIZED_BOOK_POSITION_WORLD_M',
    'CURRENT_SERIALIZED_BOOK_QUATERNION_XYZW',
    'CURRENT_SERIALIZED_BOOK_RPY_RAD',
    'EXECUTION_POLICY',
    'FINAL_DWELL_POLICY',
    'FLOOR_SIGNED_RANGE_M',
    'MINIMUM_FORCE_N',
    'PASSIVE_SCENE_REFERENCE',
    'PASSIVE_SCENE_REFERENCE_SHA256',
    'PHYSICS_STEP_SIZE_S',
    'PRESSURE_POLICY',
    'RELATIVE_YAW_DEBOUNCE_POLICY',
    'ROUTE',
    'ROUTE_Q8',
    'ROUTE_Q8_SHA256',
    'SERIALIZED_STATE_CORROBORATION',
    'SETTLE_DURATION_S',
    'SETTLE_STABILITY_ROTATION_LIMIT_RAD',
    'SETTLE_STABILITY_TRANSLATION_LIMIT_M',
    'SETTLE_STEP_COUNT',
    'STABLE_REFERENCE',
    'STABLE_REFERENCE_BOOK_POSITION_WORLD_M',
    'STABLE_REFERENCE_BOOK_QUATERNION_XYZW',
    'STAGE',
    'SUPPORT_FLOOR_WORLD_Z_M',
    'passive_scene_digest',
    'route_q8_digest',
    'semantic_digest',
    'source_json_digest',
    'validate_certificate',
)
