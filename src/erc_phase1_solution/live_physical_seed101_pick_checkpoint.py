#!/usr/bin/env python3
"""Strict physical seed-101 pickup with only the left arm.

The checkpoint uses normal controllers, never mutates Gazebo, and stops after
an exact, bilateral, pressure-verified close.  It does not extract or recover.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Iterable, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


EXPECTED_SEED = 101
BOOK = 'book_col_3_row_2_red'
LAYOUT_DIGEST = '9617b0d8728be43efa543a74f4ac24a63bfe166775c31724af40a779453f1987'

# These are the exact files used when the dense route was collision checked.
# Runtime resolves them through the ROS package index, so a stale install or a
# changed world / model fails before any controller command is sent.
CERTIFIED_ASSET_SHA256 = {
    'simulation_launch': '6a7020a9c6a4a28441cb61540bb8238ba24ea38b2ed0936dda55e856e6e7775e',
    'trajectory_controller_params': '3532c84f5d8562443da95dcd347952eac13d1df4f9048450a174eef88c9c4058',
    'controller_manager_params': '80f6df3cc9fb07729c63597a39478c5e5d2d978e654103c978025ac8dc1e1e54',
    'gripper_command_clamp': '1e2a09bb14df0733377bd0c3975b7afde1363f4c166a1de725c00e8eb189089c',
    'world_sdf': 'b4a45ff3b29144f9742bf4b254859102ede140f47a38fc15d565266525b1c681',
    'book_sdf': '778fe65e8db261b31b640104029c140beaf3cffc7ade7e6a081ee6b85171d345',
    'shelf_sdf': '1b1d861ae433396c8db5281d854c1a4cab94bb0f41ababfe92dec4586198e907',
    'shelf_mesh': '137f24a44db896e0fba4b018fa128453b4c70ff3df7cee3071cac6789adc8367',
    'table_sdf': '90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f',
    'table_mesh': '9a662650c305d09e67d112b1c3311c78c8b75454eb05c8c7981292eb1d354977',
    'collection_bin_sdf': '7e5a6af39497c377b7551cc876f9a4ee7a142b18e9bf3cef35e8c23475d7e6e0',
    'collection_bin_mesh': '7c036fe096a50eadc8c32f30837964c3bbbb012cf7a45b874eb7a6f96973aa2a',
    'robot_urdf': '15e6e4618a33c92d4a786092ccc9f1d82ed96bb42615557a865d6edc4dea14bd',
}
TEXT_CERTIFICATE_ASSETS = frozenset({
    'simulation_launch', 'trajectory_controller_params',
    'controller_manager_params', 'gripper_command_clamp', 'world_sdf',
    'book_sdf', 'shelf_sdf', 'table_sdf', 'collection_bin_sdf', 'robot_urdf',
})
# Filled from the URDF plus every unique mesh referenced by a <collision>
# element.  The digest therefore binds the robot collision geometry as well as
# the URDF text that places it.
CERTIFIED_ROBOT_COLLISION_BUNDLE_SHA256 = (
    'ec16d6e3b70e7d02c52178551a2ce2ea57cb4e70d5075edd7496637b15920260'
)

# Named, fail-closed certificate hooks.  The start certificate proves that the
# two official adjacent-link contacts disappear monotonically during the
# torso-only measured-start->S1 leg (last at .088 m, clear at .089 m) and that
# the bounded measured S1->HOME transition is collision-free.  Neither
# certificate applies to resumed state or a different scene.
OFFICIAL_START_MONOTONIC_ESCAPE_CERTIFIED = True
OFFICIAL_START_CERTIFICATE_SAMPLES = 101
OFFICIAL_START_FIRST_CLEAR_TORSO_M = 0.089
OBLIQUE_ROUTE_CERTIFIED = True
OBLIQUE_ROUTE_MINIMUM_CLEARANCE_M = 0.0029094

START_Q8 = (0.0,) * 8
START_TORSO_CLEAR_Q8 = (0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
HOME_Q8 = (0.10, 0.36, -1.83, 0.47, -2.35, 0.0, -1.20, 0.0)
HOME_LIFT_Q8 = (0.35, 0.36, -1.83, 0.47, -2.35, 0.0, -1.20, 0.0)

# Row zero is the clearance endpoint; only rows one through seven are then
# issued as insertion commands.
OBLIQUE_GRASP_LINK_WORLD_X = (
    2.579761107, 2.639761107, 2.699761107, 2.739761107,
    2.779761107, 2.819761107, 2.859761107, 2.879761107,
)
OBLIQUE_Q8 = (
    (0.35, -0.197657721832, 0.768592787207, 0.133023008060, -1.977213808290, 0.251840407735, 1.701923559526, 0.224450639220),
    (0.35, -0.233084183311, 0.686617995752, 0.114536390598, -1.896556833829, 0.223191706085, 1.532939271518, 0.222396745788),
    (0.35, -0.265019034052, 0.634154340120, 0.099303707415, -1.783838667779, 0.198989285023, 1.362571452788, 0.222543357503),
    (0.35, -0.284553472825, 0.615419502924, 0.090889044218, -1.691235997296, 0.185257690371, 1.248054113690, 0.223417752901),
    (0.35, -0.302683347205, 0.610306159167, 0.083780084347, -1.583440038062, 0.173482800222, 1.132297432283, 0.224453712754),
    (0.35, -0.319505654987, 0.619487382496, 0.077981342178, -1.458447345112, 0.163745911876, 1.013945351911, 0.225087748864),
    (0.35, -0.335041596449, 0.644223489607, 0.073523430619, -1.313236104335, 0.156356311417, 0.891417155256, 0.224559615323),
    (0.35, -0.342517593645, 0.662142376208, 0.071729532802, -1.233345168443, 0.152946889496, 0.828621692895, 0.224447061045),
)
# The full eight-row route remains the offline collision certificate.  Live
# evidence plus exact mesh intersection identified the unsensed gripper palm
# as the source of the 24.1 mm book push: row 6 penetrates the nominal book by
# 4.664 mm and row 7 by 24.352 mm.  Stop at collision-free row 5, then make a
# sub-10-milliradian lateral centering correction at the same safe depth so
# adaptive close reaches both fingertips almost simultaneously.
PRESSURE_CLOSE_CENTER_Q8 = (
    0.35, -0.313895266409, 0.616658542385, 0.080672246204,
    -1.458039913137, 0.171356896902, 1.012260273118, 0.215670921836,
)
PRESSURE_CLOSE_Q8 = (*OBLIQUE_Q8[:6], PRESSURE_CLOSE_CENTER_Q8)
GRASP_LINK_TARGET_WORLD = (2.81999684, -0.154194330902, 1.56882200)

START_BASE_WORLD = (0.0, 0.0, math.pi / 2.0)
BASE_ROUTE_WORLD = (
    (1.3, -0.149392781, math.pi / 2.0),
    (1.3, -0.149392781, -math.pi / 4.0),
    (2.002948701, -0.149392781, -math.pi / 4.0),
)

FRESH_ODOM_TRANSLATION_LIMIT_M = 0.05
FRESH_ODOM_YAW_LIMIT_RAD = 0.05
FRESH_WORLD_TRANSLATION_LIMIT_M = 0.02
FRESH_WORLD_YAW_ERROR_LIMIT_RAD = 0.02
PASSIVE_JOINT_STATIONARY_LIMIT_RAD = 0.001
PASSIVE_GRIPPER_STATIONARY_LIMIT_M = 0.00010
OFFICIAL_START_TORSO_LIMIT_M = 0.0005
OFFICIAL_SETTLED_PROFILE_TOLERANCE_RAD = 0.001
ARM_ENDPOINT_LIMIT_RAD = 0.00025
FINAL_BASE_POSITION_LIMIT_M = 0.0005
FINAL_BASE_YAW_LIMIT_RAD = 0.0005
WORLD_WAYPOINT_MAXIMUM_ATTEMPTS = 3
WORLD_WAYPOINT_SETTLE_SIM_SECONDS = 0.10
WORLD_CORRECTION_POSITION_ENVELOPE_M = 0.05
WORLD_CORRECTION_YAW_ENVELOPE_RAD = 0.02
MAXIMUM_ARM_TRAJECTORY_INCREMENT_RAD = 0.010
LEFT_OPEN_APERTURE_LIMIT_M = 0.00025
JOINT_STATE_MAXIMUM_WALL_AGE_S = 0.50
STRICT_ENDPOINT_SETTLE_WALL_TIMEOUT_S = 2.0
OFFICIAL_TORSO_MAXIMUM_VELOCITY_MPS = 0.035
START_TORSO_ESCAPE_SECONDS = 3.25
HOME_LIFT_TORSO_SECONDS = 8.0

POST_CLOSE_TRANSLATION_LIMIT_M = 0.005
POST_CLOSE_ROTATION_LIMIT_RAD = math.radians(3.0)
POST_CLOSE_STABILITY_TRANSLATION_LIMIT_M = 0.001
POST_CLOSE_STABILITY_ROTATION_LIMIT_RAD = 0.010
POST_CLOSE_SHELF_TOP_GAP_CHANGE_LIMIT_M = 0.0005
SHELF_FRONT_X_M = 2.755
POST_CLOSE_MINIMUM_SHELF_OVERLAP_M = 0.200
BOOK_HALF_EXTENTS_M = (0.125, 0.015, 0.080)

RIGHT_GRIPPER_JOINTS = ('gripper_right_finger_joint',)
LEFT_GRIPPER_JOINTS = ('gripper_left_finger_joint',)

# The official controllers can latch one of several repeatable gravity-settled
# states while they activate.  Only profiles already swept against the exact
# official meshes are accepted; the continuous monitor is then bound to the
# exact measured snapshot.  This never authorizes a right/head command.
CERTIFIED_SETTLED_START_PROFILES = (
    ((0.0,) * 7, (0.0,) * 7, (0.0, 0.0)),
    (
        (0.0,) * 7,
        (
            1.701357668739e-11, 0.000989736840008,
            1.600678358404e-11, -0.005921724243012,
            1.281169048628e-11, -0.005016253239820,
            9.359936329961e-13,
        ),
        (0.0, 0.0),
    ),
    (
        (
            4.4042350624378974e-12, 0.0009578563512194929,
            -2.980511465206689e-11, -0.006042738723240624,
            -0.003643569029497347, -0.004934205171561421,
            -4.4307381830317705e-13,
        ),
        (
            3.311670097717337e-12, 4.5328655040660165e-06,
            -2.8107202023774933e-12, -7.519431574886939e-06,
            4.771432479495819e-12, -4.196474827386196e-12,
            1.5842093011497615e-13,
        ),
        (2.0624371330624625e-14, 0.015846024557683646),
    ),
    (
        (
            8.627405863011813e-12, -0.029201528294114047,
            -2.982487601904129e-11, -0.0171047186076895,
            -4.833957166129986e-13, -0.004934201351368813,
            1.6641155792772265e-13,
        ),
        (
            1.713767364858076e-11, 0.0010194590198962788,
            1.5492205070991183e-11, -0.011673811509834844,
            1.293740190513598e-11, -0.005016323577279667,
            8.684385342740072e-13,
        ),
        (2.4903429511158132e-14, 0.022735905198066616),
    ),
)
CERTIFIED_START_LEFT_PROFILES = tuple(
    profile[0] for profile in CERTIFIED_SETTLED_START_PROFILES
)
OFFICIAL_PASSIVE_RIGHT_GRIPPER_Q = (0.0,)

TARGET_POSITION_LIMIT_M = 0.0005
TARGET_ROTATION_LIMIT_RAD = 0.002
OTHER_BOOK_POSITION_LIMIT_M = 0.010
OTHER_BOOK_ROTATION_LIMIT_RAD = 0.020
TARGET_STABILITY_POSITION_LIMIT_M = 0.00025
TARGET_STABILITY_ROTATION_LIMIT_RAD = 0.001
SHELF_POSITION_LIMIT_M = 0.00025
SHELF_ROTATION_LIMIT_RAD = 0.00025
DYNAMIC_POSE_TOPIC = '/world/erc_world/dynamic_pose/info'
TARGET_POSE_MAXIMUM_WALL_AGE_S = 0.25

SHELF = 'erc_shelf'
SHELF_POSITION = (3.0, 0.0, 1.1)
SHELF_QUATERNION = (0.5, -0.5, -0.5, 0.5)

ROW_Z = {2: 1.576857, 3: 1.246857, 4: 0.916857, 5: 0.586857}
BOOK_LAYOUT = {
    'book_col_1_row_2_red': (2.9, 1.9835693391, ROW_Z[2]),
    'book_col_1_row_3_blue': (2.9, 2.0817353223, ROW_Z[3]),
    'book_col_1_row_4_yellow': (2.9, 1.8572614849, ROW_Z[4]),
    'book_col_1_row_5_green': (2.9, 1.8608481248, ROW_Z[5]),
    'book_col_2_row_2_blue': (2.9, 0.9145187474, ROW_Z[2]),
    'book_col_2_row_3_red': (2.9, 0.9689910116, ROW_Z[3]),
    'book_col_2_row_4_green': (2.9, 0.8776209660, ROW_Z[4]),
    'book_col_2_row_5_yellow': (2.9, 0.8305332850, ROW_Z[5]),
    BOOK: (2.9, -0.154194330902091, ROW_Z[2]),
    'book_col_3_row_3_green': (2.9, 0.1431389873, ROW_Z[3]),
    'book_col_3_row_4_yellow': (2.9, -0.1305898367, ROW_Z[4]),
    'book_col_3_row_5_blue': (2.9, 0.1455974892, ROW_Z[5]),
    'book_col_4_row_2_green': (2.9, -1.0349229818, ROW_Z[2]),
    'book_col_4_row_3_yellow': (2.9, -1.0978550940, ROW_Z[3]),
    'book_col_4_row_4_red': (2.9, -1.1946004270, ROW_Z[4]),
    'book_col_4_row_5_blue': (2.9, -0.8383042528, ROW_Z[5]),
    'book_col_5_row_2_red': (2.9, -2.0526785337, ROW_Z[2]),
    'book_col_5_row_3_green': (2.9, -2.0603608331, ROW_Z[3]),
    'book_col_5_row_4_blue': (2.9, -2.0385164839, ROW_Z[4]),
    'book_col_5_row_5_yellow': (2.9, -2.0477227839, ROW_Z[5]),
}
NOMINAL_BOOK_QUATERNION = (
    0.0, math.sin(1.5708 / 2.0), 0.0, math.cos(1.5708 / 2.0),
)


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x, self.y, self.yaw)):
            raise ValueError('planar pose must contain three finite values')


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reason: str
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class EntityPose:
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]

    @property
    def planar(self) -> Pose2:
        qx, qy, qz, qw = _unit_quaternion(self.quaternion)
        return Pose2(
            self.position[0], self.position[1],
            math.atan2(2.0 * (qw * qz + qx * qy),
                       1.0 - 2.0 * (qy * qy + qz * qz)),
        )


def _canonical_text_bytes(payload: bytes) -> bytes:
    return bytes(payload).replace(b'\r\n', b'\n').replace(b'\r', b'\n')


def _file_sha256(path: Path, *, text: bool = False) -> str:
    digest = hashlib.sha256()
    payload = Path(path).read_bytes()
    digest.update(_canonical_text_bytes(payload) if text else payload)
    return digest.hexdigest()


def _manifest_digest(entries: Sequence[Tuple[str, bytes]]) -> str:
    """Hash labelled byte streams without ambiguous concatenation."""
    digest = hashlib.sha256()
    for label, payload in sorted(entries, key=lambda item: item[0]):
        name = str(label).encode('utf-8')
        data = bytes(payload)
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.hexdigest()


def robot_collision_bundle_digest(
    urdf_path: Path,
    package_shares: Mapping[str, Path],
) -> Tuple[str, int]:
    """Digest the robot URDF and every unique collision-mesh dependency."""
    urdf = _canonical_text_bytes(Path(urdf_path).read_bytes())
    root = ET.fromstring(urdf)
    uris = sorted({
        str(mesh.attrib.get('filename', '')).strip()
        for mesh in root.findall('.//collision/geometry/mesh')
    })
    if not uris or any(not uri.startswith('package://') for uri in uris):
        raise ValueError('robot collision mesh URI set is empty or unsupported')
    entries = [('robot_urdf', urdf)]
    for uri in uris:
        package_and_path = uri[len('package://'):].split('/', 1)
        if len(package_and_path) != 2:
            raise ValueError(f'malformed robot collision mesh URI: {uri}')
        package, relative = package_and_path
        if package not in package_shares:
            raise KeyError(f'package share unavailable for {package}')
        mesh_path = Path(package_shares[package]) / relative
        entries.append((uri, mesh_path.read_bytes()))
    return _manifest_digest(entries), len(entries)


def certificate_asset_gate(
    asset_paths: Mapping[str, Path],
    package_shares: Mapping[str, Path],
) -> GateResult:
    """Fail closed unless the live install matches the collision certificate."""
    expected_names = set(CERTIFIED_ASSET_SHA256)
    if set(asset_paths) != expected_names:
        return GateResult(False, 'certificate_asset_set_mismatch', {
            'asset_count': float(len(asset_paths)),
        })
    try:
        for name, expected in CERTIFIED_ASSET_SHA256.items():
            if _file_sha256(
                Path(asset_paths[name]), text=name in TEXT_CERTIFICATE_ASSETS,
            ) != expected:
                return GateResult(False, f'certificate_asset_hash_mismatch:{name}', {})
        robot_digest, bundle_count = robot_collision_bundle_digest(
            Path(asset_paths['robot_urdf']), package_shares,
        )
    except (OSError, KeyError, ET.ParseError, ValueError) as exc:
        return GateResult(False, f'certificate_asset_unavailable:{exc}', {})
    if robot_digest != CERTIFIED_ROBOT_COLLISION_BUNDLE_SHA256:
        return GateResult(False, 'robot_collision_bundle_hash_mismatch', {
            'robot_collision_bundle_file_count': float(bundle_count),
        })
    return GateResult(True, 'collision_certificate_assets_match', {
        'asset_count': float(len(asset_paths)),
        'robot_collision_bundle_file_count': float(bundle_count),
    })


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def compose(first: Pose2, second: Pose2) -> Pose2:
    cosine, sine = math.cos(first.yaw), math.sin(first.yaw)
    return Pose2(
        first.x + cosine * second.x - sine * second.y,
        first.y + sine * second.x + cosine * second.y,
        normalize_angle(first.yaw + second.yaw),
    )


def inverse(pose: Pose2) -> Pose2:
    cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
    return Pose2(
        -cosine * pose.x - sine * pose.y,
        sine * pose.x - cosine * pose.y,
        normalize_angle(-pose.yaw),
    )


def world_from_odom(world_base: Pose2, odom_base: Pose2) -> Pose2:
    return compose(world_base, inverse(odom_base))


def world_goal_to_odom(world_goal: Pose2, world_base: Pose2, odom_base: Pose2) -> Pose2:
    return compose(inverse(world_from_odom(world_base, odom_base)), world_goal)


def startup_gate(odom_pose: Pose2, world_pose: Pose2) -> GateResult:
    metrics = {
        'odom_translation_m': math.hypot(odom_pose.x, odom_pose.y),
        'odom_yaw_error_rad': abs(normalize_angle(odom_pose.yaw)),
        'world_translation_m': math.hypot(world_pose.x, world_pose.y),
        'world_yaw_error_rad': abs(normalize_angle(world_pose.yaw - math.pi / 2.0)),
    }
    for key, limit, reason in (
        ('odom_translation_m', FRESH_ODOM_TRANSLATION_LIMIT_M, 'not_fresh_odom_translation'),
        ('odom_yaw_error_rad', FRESH_ODOM_YAW_LIMIT_RAD, 'not_fresh_odom_yaw'),
        ('world_translation_m', FRESH_WORLD_TRANSLATION_LIMIT_M, 'not_fresh_world_translation'),
        ('world_yaw_error_rad', FRESH_WORLD_YAW_ERROR_LIMIT_RAD, 'not_fresh_world_yaw'),
    ):
        if metrics[key] > limit:
            return GateResult(False, reason, metrics)
    return GateResult(True, 'fresh_start', metrics)


def expected_joint_gate(observed: Sequence[float], expected: Sequence[float], *, limit: float, failure_reason: str) -> GateResult:
    try:
        actual = tuple(float(value) for value in observed)
        target = tuple(float(value) for value in expected)
    except (TypeError, ValueError):
        return GateResult(False, 'joint_state_malformed', {})
    if (len(actual) != len(target) or not actual
            or not all(math.isfinite(value) for value in (*actual, *target, limit))
            or limit < 0.0):
        return GateResult(False, 'joint_state_malformed', {})
    maximum = max(abs(value - goal) for value, goal in zip(actual, target))
    return GateResult(maximum <= limit,
                      'joint_endpoint_verified' if maximum <= limit else failure_reason,
                      {'maximum_joint_error': maximum})


def certified_joint_profile_gate(
    observed: Sequence[float],
    profiles: Sequence[Sequence[float]],
    *,
    limit: float,
    failure_reason: str,
) -> GateResult:
    """Accept only a narrow neighborhood of an explicitly swept profile."""
    try:
        actual = tuple(float(value) for value in observed)
        candidates = tuple(tuple(float(value) for value in profile) for profile in profiles)
    except (TypeError, ValueError):
        return GateResult(False, 'joint_profile_malformed', {})
    if (
        not actual
        or not candidates
        or any(len(profile) != len(actual) for profile in candidates)
        or not all(math.isfinite(value) for value in (*actual, *sum(candidates, ()), limit))
        or limit < 0.0
    ):
        return GateResult(False, 'joint_profile_malformed', {})
    errors = tuple(
        max(abs(value - target) for value, target in zip(actual, profile))
        for profile in candidates
    )
    profile_index = min(range(len(errors)), key=errors.__getitem__)
    maximum = errors[profile_index]
    return GateResult(
        maximum <= limit,
        'certified_joint_profile' if maximum <= limit else failure_reason,
        {
            'maximum_joint_error': maximum,
            'certified_profile_index': float(profile_index),
            'profile_tolerance_rad': float(limit),
        },
    )


def certified_settled_start_gate(
    left: Sequence[float],
    right: Sequence[float],
    head: Sequence[float],
) -> GateResult:
    """Bind one correlated left/right/head equilibrium already swept offline."""
    actual = (*tuple(left), *tuple(right), *tuple(head))
    profiles = tuple(
        (*left_profile, *right_profile, *head_profile)
        for left_profile, right_profile, head_profile
        in CERTIFIED_SETTLED_START_PROFILES
    )
    return certified_joint_profile_gate(
        actual,
        profiles,
        limit=OFFICIAL_SETTLED_PROFILE_TOLERANCE_RAD,
        failure_reason='outside_certified_settled_start_profile',
    )


def official_start_escape_gate(measured_q8: Sequence[float], *, certificate_available: bool = OFFICIAL_START_MONOTONIC_ESCAPE_CERTIFIED) -> GateResult:
    if not certificate_available:
        return GateResult(False, 'official_start_escape_not_certified', {})
    torso = expected_joint_gate(
        tuple(measured_q8)[:1],
        START_Q8[:1],
        limit=OFFICIAL_START_TORSO_LIMIT_M,
        failure_reason='not_official_zero_start_torso',
    )
    if not torso.ok:
        return torso
    arm = certified_joint_profile_gate(
        tuple(measured_q8)[1:],
        CERTIFIED_START_LEFT_PROFILES,
        limit=OFFICIAL_SETTLED_PROFILE_TOLERANCE_RAD,
        failure_reason='left_arm_outside_certified_start_profile',
    )
    if not arm.ok:
        return arm
    return GateResult(True, 'official_start_monotonic_escape_certified', {
        'maximum_torso_error_m': torso.metrics['maximum_joint_error'],
        'maximum_arm_error_rad': arm.metrics['maximum_joint_error'],
        'left_start_profile_index': arm.metrics['certified_profile_index'],
        'left_start_profile_tolerance_rad': OFFICIAL_SETTLED_PROFILE_TOLERANCE_RAD,
        'certificate_samples': float(OFFICIAL_START_CERTIFICATE_SAMPLES),
        'first_clear_torso_m': OFFICIAL_START_FIRST_CLEAR_TORSO_M,
    })


def seed101_layout_digest() -> str:
    payload = json.dumps(
        BOOK_LAYOUT,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def oblique_route_gate(*, certificate_available: bool = OBLIQUE_ROUTE_CERTIFIED) -> GateResult:
    valid = (
        certificate_available and len(OBLIQUE_Q8) == 8
        and len(PRESSURE_CLOSE_Q8) == 7
        and PRESSURE_CLOSE_Q8[:-1] == OBLIQUE_Q8[:6]
        and PRESSURE_CLOSE_Q8[-1] == PRESSURE_CLOSE_CENTER_Q8
        and max(abs(b - a) for a, b in zip(OBLIQUE_Q8[5][1:], PRESSURE_CLOSE_CENTER_Q8[1:]))
        <= MAXIMUM_ARM_TRAJECTORY_INCREMENT_RAD
        and seed101_layout_digest() == LAYOUT_DIGEST
        and all(len(row) == 8 and all(math.isfinite(v) for v in row) for row in OBLIQUE_Q8)
        and all(math.isfinite(v) for v in (*PRESSURE_CLOSE_CENTER_Q8, *GRASP_LINK_TARGET_WORLD))
        and all(abs(row[0] - 0.35) <= 1e-12 for row in OBLIQUE_Q8)
        and all(b > a for a, b in zip(OBLIQUE_GRASP_LINK_WORLD_X, OBLIQUE_GRASP_LINK_WORLD_X[1:]))
    )
    return GateResult(valid,
                      'certified_oblique_route' if valid else 'oblique_route_not_certified',
                      {'minimum_official_mesh_clearance_m': OBLIQUE_ROUTE_MINIMUM_CLEARANCE_M})


def dense_arm_waypoints(start_q8: Sequence[float], end_q8: Sequence[float], *, maximum_increment: float = MAXIMUM_ARM_TRAJECTORY_INCREMENT_RAD) -> Tuple[Tuple[float, ...], ...]:
    start = tuple(float(value) for value in start_q8)
    end = tuple(float(value) for value in end_q8)
    if (len(start) != 8 or len(end) != 8
            or not all(math.isfinite(value) for value in (*start, *end, maximum_increment))
            or maximum_increment <= 0.0 or abs(start[0] - end[0]) > 1e-12):
        raise ValueError('dense left-arm leg requires finite q8 endpoints at one torso height')
    distance = max(abs(after - before) for before, after in zip(start[1:], end[1:]))
    steps = max(1, int(math.ceil(distance / maximum_increment)))
    return tuple(tuple(before + (after - before) * index / steps
                       for before, after in zip(start, end))
                 for index in range(1, steps + 1))


def stationary_joint_gate(reference: Sequence[float], current: Sequence[float], *, limit: float = PASSIVE_JOINT_STATIONARY_LIMIT_RAD) -> GateResult:
    return expected_joint_gate(current, reference, limit=limit,
                               failure_reason='passive_joint_moved')


def final_base_gate(observed: Pose2, target: Pose2) -> GateResult:
    position_error = math.hypot(observed.x - target.x, observed.y - target.y)
    yaw_error = abs(normalize_angle(observed.yaw - target.yaw))
    metrics = {'final_base_position_error_m': position_error,
               'final_base_yaw_error_rad': yaw_error}
    if position_error > FINAL_BASE_POSITION_LIMIT_M:
        return GateResult(False, 'final_base_position_outside_pick_gate', metrics)
    if yaw_error > FINAL_BASE_YAW_LIMIT_RAD:
        return GateResult(False, 'final_base_yaw_outside_pick_gate', metrics)
    return GateResult(True, 'final_base_aligned', metrics)


def _unit_quaternion(values: Sequence[float]) -> Tuple[float, float, float, float]:
    quaternion = tuple(float(value) for value in values)
    if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion):
        raise ValueError('quaternion must contain four finite values')
    magnitude = math.sqrt(sum(value * value for value in quaternion))
    if magnitude < 1e-9:
        raise ValueError('quaternion magnitude is zero')
    return tuple(value / magnitude for value in quaternion)


def quaternion_distance(first: Sequence[float], second: Sequence[float]) -> float:
    left, right = _unit_quaternion(first), _unit_quaternion(second)
    dot = min(1.0, max(0.0, abs(sum(a * b for a, b in zip(left, right)))))
    return 2.0 * math.acos(dot)


def book_maximum_world_x(pose: EntityPose) -> float:
    """Return the maximum world-x corner of the official box-shaped book."""
    qx, qy, qz, qw = _unit_quaternion(pose.quaternion)
    first_rotation_row = (
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy - qz * qw),
        2.0 * (qx * qz + qy * qw),
    )
    projected_half_extent = sum(
        abs(axis) * half_extent
        for axis, half_extent in zip(first_rotation_row, BOOK_HALF_EXTENTS_M)
    )
    return float(pose.position[0]) + projected_half_extent


def book_minimum_world_z(pose: EntityPose) -> float:
    """Return the minimum world-z corner of the official box-shaped book."""
    qx, qy, qz, qw = _unit_quaternion(pose.quaternion)
    third_rotation_row = (
        2.0 * (qx * qz - qy * qw),
        2.0 * (qy * qz + qx * qw),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    projected_half_extent = sum(
        abs(axis) * half_extent
        for axis, half_extent in zip(third_rotation_row, BOOK_HALF_EXTENTS_M)
    )
    return float(pose.position[2]) - projected_half_extent


def postclose_target_gate(
    preclose: EntityPose,
    first: EntityPose,
    second: EntityPose,
) -> GateResult:
    """Require a stable, upright target that remains deeply shelf-supported."""
    translation = max(
        math.sqrt(sum((after - before) ** 2 for before, after in zip(
            preclose.position, observed.position,
        )))
        for observed in (first, second)
    )
    rotation = max(
        quaternion_distance(preclose.quaternion, observed.quaternion)
        for observed in (first, second)
    )
    stability_translation = math.sqrt(sum(
        (after - before) ** 2
        for before, after in zip(first.position, second.position)
    ))
    stability_rotation = quaternion_distance(first.quaternion, second.quaternion)
    shelf_overlap = min(
        book_maximum_world_x(observed) - SHELF_FRONT_X_M
        for observed in (first, second)
    )
    preclose_shelf_contact_z = book_minimum_world_z(preclose)
    shelf_top_gap_change = max(
        abs(book_minimum_world_z(observed) - preclose_shelf_contact_z)
        for observed in (first, second)
    )
    metrics = {
        'postclose_translation_m': translation,
        'postclose_rotation_rad': rotation,
        'postclose_stability_translation_m': stability_translation,
        'postclose_stability_rotation_rad': stability_rotation,
        'postclose_minimum_shelf_overlap_m': shelf_overlap,
        'postclose_shelf_top_gap_change_m': shelf_top_gap_change,
    }
    if translation > POST_CLOSE_TRANSLATION_LIMIT_M:
        return GateResult(False, 'target_shifted_during_close', metrics)
    if rotation > POST_CLOSE_ROTATION_LIMIT_RAD:
        return GateResult(False, 'target_tilted_during_close', metrics)
    if shelf_top_gap_change > POST_CLOSE_SHELF_TOP_GAP_CHANGE_LIMIT_M:
        return GateResult(False, 'target_lost_shelf_top_support', metrics)
    if stability_translation > POST_CLOSE_STABILITY_TRANSLATION_LIMIT_M:
        return GateResult(False, 'target_not_stable_after_close', metrics)
    if stability_rotation > POST_CLOSE_STABILITY_ROTATION_LIMIT_RAD:
        return GateResult(False, 'target_rotation_not_stable_after_close', metrics)
    if shelf_overlap < POST_CLOSE_MINIMUM_SHELF_OVERLAP_M:
        return GateResult(False, 'target_lost_shelf_support_during_close', metrics)
    return GateResult(True, 'target_pressure_pick_pose_verified', metrics)


def _pose_errors(observed: EntityPose, position: Sequence[float]) -> Tuple[float, float]:
    translation = math.sqrt(sum((a - b) ** 2 for a, b in zip(observed.position, position)))
    return translation, quaternion_distance(observed.quaternion, NOMINAL_BOOK_QUATERNION)


def scene_gate(first: Mapping[str, EntityPose], second: Mapping[str, EntityPose]) -> GateResult:
    expected_names = set(BOOK_LAYOUT)
    if set(first) != expected_names or set(second) != expected_names:
        return GateResult(False, 'seed101_book_set_mismatch', {
            'first_book_count': float(len(first)), 'second_book_count': float(len(second))})
    maximum_other_translation = 0.0
    maximum_other_rotation = 0.0
    for snapshot in (first, second):
        for name, expected_position in BOOK_LAYOUT.items():
            translation, rotation = _pose_errors(snapshot[name], expected_position)
            if name == BOOK:
                if translation > TARGET_POSITION_LIMIT_M:
                    return GateResult(False, 'target_position_outside_seed101_gate', {'target_position_error_m': translation})
                if rotation > TARGET_ROTATION_LIMIT_RAD:
                    return GateResult(False, 'target_rotation_outside_seed101_gate', {'target_rotation_error_rad': rotation})
            else:
                maximum_other_translation = max(maximum_other_translation, translation)
                maximum_other_rotation = max(maximum_other_rotation, rotation)
                if translation > OTHER_BOOK_POSITION_LIMIT_M:
                    return GateResult(False, 'non_target_position_outside_seed101_gate', {'non_target_position_error_m': translation})
                if rotation > OTHER_BOOK_ROTATION_LIMIT_RAD:
                    return GateResult(False, 'non_target_rotation_outside_seed101_gate', {'non_target_rotation_error_rad': rotation})
    target_motion = math.sqrt(sum((b - a) ** 2 for a, b in zip(first[BOOK].position, second[BOOK].position)))
    target_rotation = quaternion_distance(first[BOOK].quaternion, second[BOOK].quaternion)
    metrics = {
        'target_stability_translation_m': target_motion,
        'target_stability_rotation_rad': target_rotation,
        'maximum_non_target_translation_error_m': maximum_other_translation,
        'maximum_non_target_rotation_error_rad': maximum_other_rotation,
    }
    if target_motion > TARGET_STABILITY_POSITION_LIMIT_M:
        return GateResult(False, 'target_not_stable_in_fresh_dwell', metrics)
    if target_rotation > TARGET_STABILITY_ROTATION_LIMIT_RAD:
        return GateResult(False, 'target_rotated_in_fresh_dwell', metrics)
    return GateResult(True, 'exact_seed101_scene_stable', metrics)


def shelf_gate(first: EntityPose, second: EntityPose) -> GateResult:
    errors = []
    for observed in (first, second):
        translation = math.sqrt(sum(
            (actual - expected) ** 2
            for actual, expected in zip(observed.position, SHELF_POSITION)
        ))
        rotation = quaternion_distance(observed.quaternion, SHELF_QUATERNION)
        errors.append((translation, rotation))
    motion = math.sqrt(sum(
        (after - before) ** 2
        for before, after in zip(first.position, second.position)
    ))
    rotation_motion = quaternion_distance(first.quaternion, second.quaternion)
    metrics = {
        'shelf_position_error_m': max(value[0] for value in errors),
        'shelf_rotation_error_rad': max(value[1] for value in errors),
        'shelf_motion_m': motion,
        'shelf_rotation_motion_rad': rotation_motion,
    }
    if metrics['shelf_position_error_m'] > SHELF_POSITION_LIMIT_M:
        return GateResult(False, 'shelf_position_outside_certificate', metrics)
    if metrics['shelf_rotation_error_rad'] > SHELF_ROTATION_LIMIT_RAD:
        return GateResult(False, 'shelf_rotation_outside_certificate', metrics)
    if motion > 1e-6 or rotation_motion > 1e-6:
        return GateResult(False, 'shelf_not_static', metrics)
    return GateResult(True, 'certified_shelf_pose_static', metrics)


def _protobuf_scalar(block: str, field: str, default: float = 0.0) -> float:
    match = re.search(rf'(?m)^\s*{re.escape(field)}:\s*([^\s]+)', block)
    return float(default if match is None else match.group(1))


def entity_pose_from_dynamic_pose(message: str, name: str) -> EntityPose:
    marker = f'name: "{name}"'
    marker_index = message.find(marker)
    if marker_index < 0:
        raise RuntimeError(f'Gazebo dynamic pose omitted {name!r}')
    start = message.rfind('pose {', 0, marker_index)
    next_pose = message.find('\npose {', marker_index)
    if start < 0:
        raise RuntimeError(f'Gazebo dynamic pose for {name!r} was malformed')
    block = message[start:] if next_pose < 0 else message[start:next_pose]
    position_match = re.search(r'position\s*\{([^}]*)\}', block, re.DOTALL)
    orientation_match = re.search(r'orientation\s*\{([^}]*)\}', block, re.DOTALL)
    if position_match is None or orientation_match is None:
        raise RuntimeError(f'Gazebo dynamic pose for {name!r} was incomplete')
    position = tuple(_protobuf_scalar(position_match.group(1), axis) for axis in ('x', 'y', 'z'))
    quaternion = tuple(_protobuf_scalar(orientation_match.group(1), axis) for axis in ('x', 'y', 'z', 'w'))
    if not all(math.isfinite(v) for v in (*position, *quaternion)):
        raise RuntimeError(f'Gazebo dynamic pose for {name!r} was non-finite')
    return EntityPose(position, quaternion)


BOOK_NAME_PATTERN = re.compile(r'(?m)^\s*name: "(book_col_[1-5]_row_[2-5]_(?:red|blue|yellow|green))"\s*$')


def book_poses_from_dynamic_pose(message: str) -> Mapping[str, EntityPose]:
    counts = Counter(BOOK_NAME_PATTERN.findall(message))
    if set(counts) != set(BOOK_LAYOUT) or any(count != 1 for count in counts.values()):
        raise RuntimeError('Gazebo dynamic pose did not contain the exact seed-101 book set once')
    return {name: entity_pose_from_dynamic_pose(message, name) for name in BOOK_LAYOUT}


def read_dynamic_pose_message() -> str:
    completed = subprocess.run(
        ['gz', 'topic', '-e', '-t', '/world/erc_world/dynamic_pose/info', '-n', '1'],
        check=True, capture_output=True, text=True, timeout=5.0,
    )
    return completed.stdout


def read_full_pose_message() -> str:
    """Read the pose stream that includes static models such as the shelf."""
    completed = subprocess.run(
        ['gz', 'topic', '-e', '-t', '/world/erc_world/pose/info', '-n', '1'],
        check=True, capture_output=True, text=True, timeout=5.0,
    )
    return completed.stdout


def _emit(event: str, **fields: object) -> None:
    print(json.dumps({'event': event, **fields}, sort_keys=True), flush=True)


def _joint_snapshot(node: object, names: Iterable[str]) -> Tuple[float, ...]:
    requested = tuple(names)
    lock = getattr(node, '_lock', None)
    if lock is None:
        joints = dict(getattr(node, 'joints'))
        sample_times = dict(getattr(node, '_strict_joint_state_wall_times', {}))
    else:
        with lock:
            joints = dict(getattr(node, 'joints'))
            sample_times = dict(getattr(node, '_strict_joint_state_wall_times', {}))
    missing = [name for name in requested if name not in joints]
    if missing:
        raise RuntimeError(f'joint state omitted {missing}')
    if hasattr(node, '_strict_joint_state_wall_times'):
        unstamped = [name for name in requested if name not in sample_times]
        if unstamped:
            raise RuntimeError(f'joint state freshness omitted {unstamped}')
        maximum_age = max(time.monotonic() - sample_times[name] for name in requested)
        if maximum_age < 0.0 or maximum_age > JOINT_STATE_MAXIMUM_WALL_AGE_S:
            raise RuntimeError(f'joint state stale by {maximum_age:.3f}s')
    values = tuple(float(joints[name]) for name in requested)
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError('joint state contained non-finite values')
    return values


def _joint_generation_snapshot(node: object, names: Iterable[str]) -> Tuple[int, ...]:
    requested = tuple(names)
    lock = getattr(node, '_lock', None)
    if lock is None:
        generations = dict(getattr(node, '_strict_joint_state_generations', {}))
    else:
        with lock:
            generations = dict(getattr(node, '_strict_joint_state_generations', {}))
    missing = [name for name in requested if name not in generations]
    if missing:
        raise RuntimeError(f'joint state generation omitted {missing}')
    return tuple(int(generations[name]) for name in requested)


def _wait_for_joint_update(
    node: object,
    names: Iterable[str],
    previous_generations: Sequence[int],
    *,
    timeout: float = 2.0,
) -> Tuple[float, ...]:
    requested = tuple(names)
    previous = tuple(int(value) for value in previous_generations)
    if len(previous) != len(requested):
        raise ValueError('joint generation reference has the wrong length')
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        try:
            current = _joint_generation_snapshot(node, requested)
            if all(after > before for before, after in zip(previous, current)):
                return _joint_snapshot(node, requested)
        except RuntimeError:
            pass
        time.sleep(0.01)
    raise RuntimeError('no fresh joint state arrived after the command')


def _cmd_vel_gate(nav: object) -> GateResult:
    infos = tuple(nav.get_publishers_info_by_topic('/cmd_vel'))
    namespace = str(nav.get_namespace()).rstrip('/')
    own = f'{namespace}/{nav.get_name()}' if namespace else f'/{nav.get_name()}'
    identities = []
    for info in infos:
        ns = str(info.node_namespace).rstrip('/')
        identities.append(f'{ns}/{info.node_name}' if ns else f'/{info.node_name}')
    metrics = {'cmd_vel_publishers': float(len(infos)),
               'cmd_vel_subscribers': float(nav.cmd_pub.get_subscription_count())}
    if len(infos) != 1 or identities != [own]:
        return GateResult(False, 'foreign_cmd_vel_publisher', metrics)
    if nav.cmd_pub.get_subscription_count() < 1:
        return GateResult(False, 'cmd_vel_bridge_unavailable', metrics)
    return GateResult(True, 'cmd_vel_exclusive', metrics)


def _require(result: GateResult) -> None:
    _emit('gate', passed=result.ok, reason=result.reason, **result.metrics)
    if not result.ok:
        raise RuntimeError(result.reason)


def main() -> None:
    parser = argparse.ArgumentParser(description='Strict one-left-arm seed-101 pickup')
    parser.add_argument('--confirm-seed', type=int, required=True, choices=(EXPECTED_SEED,))
    parser.add_argument('--navigation-wall-timeout', type=float, default=180.0)
    args = parser.parse_args()
    if not 30.0 <= args.navigation_wall_timeout <= 300.0:
        raise SystemExit('--navigation-wall-timeout must be within [30, 300] s')

    import rclpy
    from action_msgs.msg import GoalStatus
    from builtin_interfaces.msg import Duration as DurationMessage
    from control_msgs.action import FollowJointTrajectory
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node as GzTransportNode
    from rclpy.executors import MultiThreadedExecutor
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from ament_index_python.packages import (
        get_package_prefix,
        get_package_share_directory,
    )
    from erc_phase1_solution.common import decode_event
    from erc_phase1_solution.manipulation_node import HEAD_JOINTS, ManipulationNode
    from erc_phase1_solution.motion_profiles import ARM_JOINTS, IK_JOINTS, RIGHT_ARM_JOINTS
    from erc_phase1_solution.navigation_node import NavigationNode

    class StrictLeftNode(ManipulationNode):
        """Production node with one dense synchronized left-only primitive."""

        def __init__(self) -> None:
            self._strict_dispatch_lock = threading.Lock()
            self._strict_joint_state_wall_times = {}
            self._strict_joint_state_generations = {}
            self._strict_joint_state_counter = 0
            self._gz_target_pose_lock = threading.Lock()
            self._gz_target_pose_generation = 0
            self._gz_target_pose_wall_ns = 0
            self._gz_target_pose: Optional[Tuple[float, ...]] = None
            self._gz_target_watch_armed = False
            self._gz_target_reference: Optional[EntityPose] = None
            super().__init__()
            # This direct read-only Gazebo subscription covers target movement
            # caused by the fixed gripper palm, which has no official contact
            # sensor / ROS bridge.  Keeping the transport node on self keeps
            # the subscription alive for the whole checkpoint.
            self._gz_pose_node = GzTransportNode()
            self._gz_pose_node.subscribe(
                Pose_V, DYNAMIC_POSE_TOPIC, self._on_gz_dynamic_pose,
            )

        def _on_joint_state(self, message: object) -> None:
            super()._on_joint_state(message)
            observed_at = time.monotonic()
            names = tuple(str(name) for name in getattr(message, 'name', ()))
            with self._lock:
                self._strict_joint_state_counter += 1
                generation = self._strict_joint_state_counter
                for name in names:
                    self._strict_joint_state_wall_times[name] = observed_at
                    self._strict_joint_state_generations[name] = generation

        def _on_command(self, message: object) -> None:
            # This focused launcher owns the actuators for its lifetime.  A
            # concurrently running mission manager must not be able to enter a
            # generic pick/stow path (the latter commands the right arm).
            payload = decode_event(message.data)
            command = str(payload.get('event', message.data)).strip().lower()
            if command in ('cancel', 'abort', 'stop'):
                with self._strict_dispatch_lock:
                    super()._on_command(message)
                return
            self._publish_status(
                'rejected',
                command=command,
                reason='strict_physical_checkpoint_owns_actuators',
            )

        def target_robot_contact_latched(self) -> bool:
            with self._lock:
                return bool(self._target_robot_contact_latched)

        def _on_gz_dynamic_pose(self, message: object) -> None:
            for pose in getattr(message, 'pose', ()):
                if str(getattr(pose, 'name', '')) != BOOK:
                    continue
                position = getattr(pose, 'position', None)
                orientation = getattr(pose, 'orientation', None)
                sample = (
                    float(getattr(position, 'x', math.nan)),
                    float(getattr(position, 'y', math.nan)),
                    float(getattr(position, 'z', math.nan)),
                    float(getattr(orientation, 'x', math.nan)),
                    float(getattr(orientation, 'y', math.nan)),
                    float(getattr(orientation, 'z', math.nan)),
                    float(getattr(orientation, 'w', math.nan)),
                )
                # Invalid data never refreshes freshness.  The active watchdog
                # will fail closed when the last valid sample becomes stale.
                if not all(math.isfinite(value) for value in sample):
                    return
                with self._gz_target_pose_lock:
                    self._gz_target_pose = sample
                    self._gz_target_pose_wall_ns = time.monotonic_ns()
                    self._gz_target_pose_generation += 1
                return

        def arm_target_pose_watchdog(self) -> GateResult:
            with self._gz_target_pose_lock:
                sample = self._gz_target_pose
                stamp_ns = self._gz_target_pose_wall_ns
                generation = self._gz_target_pose_generation
            age = (
                math.inf if stamp_ns <= 0
                else (time.monotonic_ns() - stamp_ns) / 1e9
            )
            metrics = {
                'target_pose_generation': float(generation),
                'target_pose_wall_age_s': float(age),
            }
            if sample is None:
                return GateResult(False, 'target_dynamic_pose_unavailable', metrics)
            if age < 0.0 or age > TARGET_POSE_MAXIMUM_WALL_AGE_S:
                return GateResult(False, 'target_dynamic_pose_stale', metrics)
            observed = EntityPose(tuple(sample[:3]), tuple(sample[3:]))
            reference = EntityPose(BOOK_LAYOUT[BOOK], NOMINAL_BOOK_QUATERNION)
            translation = math.dist(observed.position, reference.position)
            rotation = quaternion_distance(observed.quaternion, reference.quaternion)
            metrics.update({
                'target_position_error_m': translation,
                'target_rotation_error_rad': rotation,
            })
            if translation > TARGET_POSITION_LIMIT_M:
                return GateResult(False, 'target_position_outside_seed101_gate', metrics)
            if rotation > TARGET_ROTATION_LIMIT_RAD:
                return GateResult(False, 'target_rotation_outside_seed101_gate', metrics)
            with self._gz_target_pose_lock:
                self._gz_target_reference = reference
                self._gz_target_watch_armed = True
            return GateResult(True, 'target_dynamic_pose_watchdog_armed', metrics)

        def target_pose_watchdog_reason(self) -> Optional[str]:
            with self._gz_target_pose_lock:
                if not self._gz_target_watch_armed:
                    return None
                sample = self._gz_target_pose
                stamp_ns = self._gz_target_pose_wall_ns
                reference = self._gz_target_reference
            if sample is None or reference is None or stamp_ns <= 0:
                return 'target_dynamic_pose_unavailable'
            age = (time.monotonic_ns() - stamp_ns) / 1e9
            if age < 0.0 or age > TARGET_POSE_MAXIMUM_WALL_AGE_S:
                return 'target_dynamic_pose_stale'
            observed = EntityPose(tuple(sample[:3]), tuple(sample[3:]))
            if math.dist(observed.position, reference.position) > TARGET_POSITION_LIMIT_M:
                return 'target_moved_during_open_insertion'
            if quaternion_distance(observed.quaternion, reference.quaternion) > TARGET_ROTATION_LIMIT_RAD:
                return 'target_rotated_during_open_insertion'
            return None

        def disarm_target_pose_watchdog(self) -> None:
            with self._gz_target_pose_lock:
                self._gz_target_watch_armed = False
                self._gz_target_reference = None

        def _strict_motion_hazard_reason(
            self,
            *,
            watch_target_contact: bool,
        ) -> Optional[str]:
            if watch_target_contact and self.target_robot_contact_latched():
                return 'target_robot_contact'
            return self.target_pose_watchdog_reason()

        def cancel_active_goals(self) -> bool:
            """Cancel every registered left/torso goal before ROS shutdown."""
            self._cancel.set()
            with self._lock:
                goal_handles = tuple(self._goal_handles)
            all_confirmed = True
            for goal_handle in goal_handles:
                try:
                    result_future = goal_handle.get_result_async()
                    confirmed = (
                        result_future.done()
                        or self._cancel_goal_and_confirm(goal_handle, result_future)
                    )
                except Exception:
                    confirmed = False
                    try:
                        goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                all_confirmed = all_confirmed and confirmed
                if confirmed:
                    with self._lock:
                        if goal_handle in self._goal_handles:
                            self._goal_handles.remove(goal_handle)
            return all_confirmed

        def _adaptive_overload_reason(self) -> Optional[str]:
            if self.target_robot_contact_latched():
                with self._strict_dispatch_lock:
                    self._cancel.set()
                return 'target_robot_contact'
            return super()._adaptive_overload_reason()

        def _publish_gripper_position(
            self,
            position: float,
            *,
            motion_seconds: float,
            wait_seconds: float,
        ) -> bool:
            """Publish only while atomically outside the cancel interlock."""
            message = JointTrajectory()
            message.joint_names = ['gripper_left_finger_joint']
            point = JointTrajectoryPoint()
            point.positions = [min(0.069, max(0.0, float(position)))]
            nanoseconds = int(round(float(motion_seconds) * 1e9))
            point.time_from_start = DurationMessage(
                sec=nanoseconds // 1_000_000_000,
                nanosec=nanoseconds % 1_000_000_000,
            )
            message.points = [point]
            for _ in range(2):
                with self._strict_dispatch_lock:
                    if self._cancel.is_set():
                        return False
                    self.gripper_pub.publish(message)
                time.sleep(0.04)
            return self._wait_sim_duration(wait_seconds)

        def _follow(
            self,
            client: object,
            joints: Sequence[str],
            positions: Sequence[float],
            duration: float,
        ) -> bool:
            forbidden = set(RIGHT_ARM_JOINTS) | set(HEAD_JOINTS)
            if client is getattr(self, 'right_arm_client', None) or forbidden.intersection(joints):
                raise RuntimeError('strict checkpoint rejected a right-arm/head command')
            raise RuntimeError('strict checkpoint rejected an inherited trajectory path')

        def _wait_for_strict_endpoint(
            self,
            goal: object,
            *,
            watch_target_contact: bool,
        ) -> bool:
            """Require post-result feedback at the exact commanded endpoint."""
            joints = tuple(str(name) for name in goal.trajectory.joint_names)
            if not goal.trajectory.points:
                raise RuntimeError('strict trajectory omitted its endpoint')
            expected = tuple(float(value) for value in goal.trajectory.points[-1].positions)
            if not joints or len(joints) != len(expected):
                raise RuntimeError('strict trajectory endpoint was malformed')

            previous = _joint_generation_snapshot(self, joints)
            deadline = time.monotonic() + STRICT_ENDPOINT_SETTLE_WALL_TIMEOUT_S
            last_gate: Optional[GateResult] = None
            while time.monotonic() < deadline:
                hazard = self._strict_motion_hazard_reason(
                    watch_target_contact=watch_target_contact,
                )
                with self._strict_dispatch_lock:
                    cancelled = self._cancel.is_set()
                    if hazard is not None:
                        self._cancel.set()
                if hazard is not None:
                    self._publish_status(
                        'payload_hazard', reason=hazard,
                    )
                if cancelled or hazard is not None:
                    return False

                try:
                    current = _joint_generation_snapshot(self, joints)
                    if all(after > before for before, after in zip(previous, current)):
                        previous = current
                        last_gate = expected_joint_gate(
                            _joint_snapshot(self, joints),
                            expected,
                            limit=ARM_ENDPOINT_LIMIT_RAD,
                            failure_reason='strict_trajectory_endpoint_missed',
                        )
                        if last_gate.ok:
                            hazard = self._strict_motion_hazard_reason(
                                watch_target_contact=watch_target_contact,
                            )
                            with self._strict_dispatch_lock:
                                cancelled = self._cancel.is_set()
                                if hazard is not None:
                                    self._cancel.set()
                            if not cancelled and hazard is None:
                                return True
                            return False
                except RuntimeError:
                    pass
                time.sleep(0.01)

            if last_gate is None:
                _emit(
                    'gate', passed=False,
                    reason='strict_trajectory_endpoint_feedback_timeout',
                    endpoint_settle_timeout_s=STRICT_ENDPOINT_SETTLE_WALL_TIMEOUT_S,
                )
            else:
                _emit(
                    'gate', passed=False, reason=last_gate.reason,
                    endpoint_settle_timeout_s=STRICT_ENDPOINT_SETTLE_WALL_TIMEOUT_S,
                    **last_gate.metrics,
                )
            return False

        def _execute_strict_trajectory(
            self,
            client: object,
            goal: object,
            duration: float,
            *,
            watch_target_contact: bool,
        ) -> bool:
            if not client.wait_for_server(timeout_sec=min(5.0, self.timeout)):
                raise RuntimeError('strict trajectory action server unavailable')
            hazard = self._strict_motion_hazard_reason(
                watch_target_contact=watch_target_contact,
            )
            with self._strict_dispatch_lock:
                if self._cancel.is_set() or hazard is not None:
                    if hazard is not None:
                        self._cancel.set()
                    return False
                goal_future = client.send_goal_async(goal)
            goal_handle = self._wait_future(goal_future, self.timeout)
            if goal_handle is None or not goal_handle.accepted:
                return False
            with self._lock:
                self._goal_handles.append(goal_handle)
            keep_registered = True
            try:
                result_future = goal_handle.get_result_async()
                deadline = time.monotonic() + self.timeout + 4.0 * float(duration)
                while not result_future.done():
                    hazard = self._strict_motion_hazard_reason(
                        watch_target_contact=watch_target_contact,
                    )
                    if hazard is not None:
                        self._cancel.set()
                        self._publish_status(
                            'payload_hazard', reason=hazard,
                        )
                    if hazard is not None or self._cancel.is_set() or time.monotonic() >= deadline:
                        if not self._cancel_goal_and_confirm(goal_handle, result_future):
                            raise RuntimeError('strict trajectory cancellation was not confirmed')
                        keep_registered = False
                        return False
                    time.sleep(0.02)
                hazard = self._strict_motion_hazard_reason(
                    watch_target_contact=watch_target_contact,
                )
                with self._strict_dispatch_lock:
                    cancelled = self._cancel.is_set()
                    if hazard is not None:
                        self._cancel.set()
                if cancelled or hazard is not None:
                    keep_registered = False
                    return False
                if result_future.result().status != GoalStatus.STATUS_SUCCEEDED:
                    keep_registered = False
                    return False
                try:
                    return self._wait_for_strict_endpoint(
                        goal, watch_target_contact=watch_target_contact,
                    )
                finally:
                    keep_registered = False
            finally:
                if not keep_registered:
                    with self._lock:
                        if goal_handle in self._goal_handles:
                            self._goal_handles.remove(goal_handle)

        def move_torso_strict(self, position: float, duration: float) -> bool:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = ['torso_lift_joint']
            point = JointTrajectoryPoint()
            point.positions = [float(position)]
            nanoseconds = int(round(float(duration) * 1e9))
            point.time_from_start = DurationMessage(
                sec=nanoseconds // 1_000_000_000,
                nanosec=nanoseconds % 1_000_000_000,
            )
            goal.trajectory.points = [point]
            return self._execute_strict_trajectory(
                self.torso_client, goal, duration, watch_target_contact=False,
            )

        def follow_dense_left(self, start_q8: Sequence[float], end_q8: Sequence[float], duration: float) -> bool:
            hazard = self._strict_motion_hazard_reason(watch_target_contact=True)
            if self._cancel.is_set() or hazard is not None:
                self._cancel.set()
                return False
            gate = expected_joint_gate(self._measured_left_solution(), start_q8,
                                       limit=ARM_ENDPOINT_LIMIT_RAD,
                                       failure_reason='dense_arm_start_mismatch')
            if not gate.ok:
                return False
            points = dense_arm_waypoints(start_q8, end_q8)
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(ARM_JOINTS)
            for index, q8 in enumerate(points, start=1):
                point = JointTrajectoryPoint()
                point.positions = [float(v) for v in q8[1:]]
                ns = int(round(float(duration) * index / len(points) * 1e9))
                point.time_from_start = DurationMessage(sec=ns // 1_000_000_000,
                                                        nanosec=ns % 1_000_000_000)
                goal.trajectory.points.append(point)
            return self._execute_strict_trajectory(
                self.arm_client, goal, duration, watch_target_contact=True,
            )

    class CheckpointNavigationNode(NavigationNode):
        def __init__(self) -> None:
            self.checkpoint_terminal = threading.Event()
            self.checkpoint_outcome: Optional[Tuple[str, Mapping[str, object]]] = None
            super().__init__()

        def _finish_goal(self, event: str, **fields: object) -> None:
            super()._finish_goal(event, **fields)
            self.checkpoint_outcome = (event, dict(fields))
            self.checkpoint_terminal.set()

    bringup_share = Path(get_package_share_directory('erc_bringup'))
    bringup_prefix = Path(get_package_prefix('erc_bringup'))
    description_share = Path(get_package_share_directory('erc_description'))
    asset_paths = {
        'simulation_launch': bringup_share / 'launch' / 'simulation.launch.py',
        'trajectory_controller_params': bringup_share / 'config' / 'controller_params.yaml',
        'controller_manager_params': bringup_share / 'config' / 'gazebo_controller_manager_cfg.yaml',
        'gripper_command_clamp': (
            bringup_prefix / 'lib' / 'erc_bringup' / 'gripper_command_clamp.py'
        ),
        'world_sdf': description_share / 'worlds' / 'erc_world.sdf',
        'book_sdf': description_share / 'models' / 'book' / 'sdf' / 'erc_book.sdf',
        'shelf_sdf': description_share / 'models' / 'shelf' / 'sdf' / 'erc_shelf.sdf',
        'shelf_mesh': description_share / 'models' / 'shelf' / 'meshes' / 'erc_base_shelf.STL',
        'table_sdf': description_share / 'models' / 'table' / 'sdf' / 'erc_table.sdf',
        'table_mesh': description_share / 'models' / 'table' / 'meshes' / 'erc_base_table.STL',
        'collection_bin_sdf': description_share / 'models' / 'collection_bin' / 'sdf' / 'erc_collection_bin.sdf',
        'collection_bin_mesh': description_share / 'models' / 'collection_bin' / 'meshes' / 'erc_base_collection_bin.STL',
        'robot_urdf': description_share / 'urdf' / 'tiago_pro.urdf',
    }
    urdf_root = ET.fromstring(asset_paths['robot_urdf'].read_bytes())
    collision_packages = sorted({
        str(mesh.attrib['filename'])[len('package://'):].split('/', 1)[0]
        for mesh in urdf_root.findall('.//collision/geometry/mesh')
        if str(mesh.attrib.get('filename', '')).startswith('package://')
    })
    package_shares = {
        package: Path(get_package_share_directory(package))
        for package in collision_packages
    }
    _require(certificate_asset_gate(asset_paths, package_shares))

    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    node, nav = StrictLeftNode(), CheckpointNavigationNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(nav)
    threading.Thread(target=executor.spin, daemon=True).start()
    monitor_stop, monitor_fault = threading.Event(), threading.Event()
    monitor_reason = {'value': ''}
    right_reference: Optional[Tuple[float, ...]] = None
    right_gripper_reference: Optional[Tuple[float, ...]] = None
    head_reference: Optional[Tuple[float, ...]] = None
    monitor_thread: Optional[threading.Thread] = None

    def latch_passive_fault(reason: str) -> None:
        if monitor_fault.is_set():
            return
        monitor_reason['value'] = reason
        monitor_fault.set()
        with node._strict_dispatch_lock:
            node._cancel.set()
        nav._publish_zero()
        if nav.goal is not None:
            nav._finish_goal('cancelled', reason=reason)

    def passive_monitor() -> None:
        assert (
            right_reference is not None
            and right_gripper_reference is not None
            and head_reference is not None
        )
        while not monitor_stop.is_set():
            try:
                if not stationary_joint_gate(right_reference, _joint_snapshot(node, RIGHT_ARM_JOINTS)).ok:
                    latch_passive_fault('right_arm_moved')
                    return
                if not stationary_joint_gate(
                    right_gripper_reference,
                    _joint_snapshot(node, RIGHT_GRIPPER_JOINTS),
                    limit=PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
                ).ok:
                    latch_passive_fault('right_gripper_moved')
                    return
                if not stationary_joint_gate(head_reference, _joint_snapshot(node, HEAD_JOINTS)).ok:
                    latch_passive_fault('head_moved')
                    return
            except Exception as exc:
                latch_passive_fault(f'passive_joint_monitor_failed:{exc}')
                return
            time.sleep(0.01)

    def require_monitor_clear() -> None:
        if monitor_fault.is_set():
            raise RuntimeError(monitor_reason['value'] or 'passive_joint_monitor_failed')

    def require_not_cancelled(stage: str) -> None:
        if node._cancel.is_set():
            nav._publish_zero()
            raise RuntimeError(f'checkpoint cancelled:{stage}')

    def require_no_target_robot_contact(stage: str) -> None:
        if node.target_robot_contact_latched():
            node._cancel.set()
            nav._publish_zero()
            raise RuntimeError(f'target contacted a non-gripper robot link:{stage}')

    def require_target_pose_watchdog_clear(stage: str) -> None:
        reason = node.target_pose_watchdog_reason()
        if reason is not None:
            with node._strict_dispatch_lock:
                node._cancel.set()
            nav._publish_zero()
            raise RuntimeError(f'{reason}:{stage}')

    def require_left_open() -> None:
        _require(
            expected_joint_gate(
                _joint_snapshot(node, LEFT_GRIPPER_JOINTS),
                (float(node.gripper_open),),
                limit=LEFT_OPEN_APERTURE_LIMIT_M,
                failure_reason='left_gripper_not_physically_open',
            )
        )

    def stable_scene() -> Mapping[str, EntityPose]:
        first_message = read_dynamic_pose_message()
        first_full_message = read_full_pose_message()
        first = book_poses_from_dynamic_pose(first_message)
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('scene stability dwell was interrupted')
        second_message = read_dynamic_pose_message()
        second_full_message = read_full_pose_message()
        second = book_poses_from_dynamic_pose(second_message)
        _require(scene_gate(first, second))
        _require(shelf_gate(
            entity_pose_from_dynamic_pose(first_full_message, SHELF),
            entity_pose_from_dynamic_pose(second_full_message, SHELF),
        ))
        require_monitor_clear()
        return second

    def insertion_scene_guard(stage: str) -> None:
        """Reject any book or shelf displacement after an insertion leg."""
        require_target_pose_watchdog_clear(stage)
        dynamic_message = read_dynamic_pose_message()
        books = book_poses_from_dynamic_pose(dynamic_message)
        # Using the same instantaneous snapshot on both sides deliberately
        # applies the absolute seed-101 pose gates without inventing motion
        # evidence.  The full temporal stability dwell still runs immediately
        # before pressure close.
        _require(scene_gate(books, books))
        full_message = read_full_pose_message()
        shelf = entity_pose_from_dynamic_pose(full_message, SHELF)
        _require(shelf_gate(shelf, shelf))
        require_monitor_clear()
        require_target_pose_watchdog_clear(stage)
        _emit(
            'insertion_scene_clear', stage=stage,
            target_position=list(books[BOOK].position),
            target_quaternion=list(books[BOOK].quaternion),
        )

    def navigate(stage: str, goal: Pose2) -> None:
        """Reach one certified world waypoint despite wheel-odometry slip."""
        last_world_gate: Optional[GateResult] = None
        for attempt in range(1, WORLD_WAYPOINT_MAXIMUM_ATTEMPTS + 1):
            # Rebind world->odom only while stationary.  Corrections repeat the
            # same certified world goal; they never invent an off-route target.
            nav._publish_zero()
            require_not_cancelled(f'before_{stage}_navigation_attempt_{attempt}')
            require_monitor_clear()
            if nav.goal is not None or not nav._sensors_fresh(nav.get_clock().now()):
                raise RuntimeError('navigation was not ready before dispatch')
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                HOME_Q8,
                limit=ARM_ENDPOINT_LIMIT_RAD,
                failure_reason='left_home_drifted_before_navigation',
            ))
            require_left_open()
            _require(_cmd_vel_gate(nav))
            calibration_world = entity_pose_from_dynamic_pose(
                read_dynamic_pose_message(), 'tiago_pro',
            ).planar
            if nav.pose is None:
                raise RuntimeError('odometry unavailable during world recalibration')
            calibration_odom = Pose2(*nav.pose)
            converted = world_goal_to_odom(goal, calibration_world, calibration_odom)

            nav.checkpoint_outcome = None
            nav.checkpoint_terminal.clear()
            nav._accept_goal((converted.x, converted.y, converted.yaw), profile='normal')
            deadline = time.monotonic() + args.navigation_wall_timeout
            while not nav.checkpoint_terminal.wait(0.02):
                require_monitor_clear()
                if node._cancel.is_set():
                    nav._finish_goal('cancelled', reason='checkpoint_cancel_requested')
                    raise RuntimeError(f'{stage} navigation cancelled')
                if time.monotonic() >= deadline:
                    nav._finish_goal('cancelled', reason='checkpoint_wall_timeout')
                    raise RuntimeError(f'{stage} navigation exceeded wall timeout')
            if node._cancel.is_set():
                nav._publish_zero()
                if nav.goal is not None:
                    nav._finish_goal('cancelled', reason='checkpoint_cancel_requested')
                raise RuntimeError(f'{stage} navigation cancelled')
            if nav.checkpoint_outcome is None or nav.checkpoint_outcome[0] != 'reached':
                raise RuntimeError(f'{stage} navigation failed: {nav.checkpoint_outcome!r}')
            nav._publish_zero()
            require_monitor_clear()
            if not node._wait_sim_duration(WORLD_WAYPOINT_SETTLE_SIM_SECONDS):
                raise RuntimeError(f'{stage} world-pose settle was interrupted')
            require_not_cancelled(f'after_{stage}_navigation_attempt_{attempt}')
            require_monitor_clear()
            _require(expected_joint_gate(
                _joint_snapshot(node, IK_JOINTS),
                HOME_Q8,
                limit=ARM_ENDPOINT_LIMIT_RAD,
                failure_reason='left_home_drifted_during_navigation',
            ))
            require_left_open()
            observed = entity_pose_from_dynamic_pose(
                read_dynamic_pose_message(), 'tiago_pro',
            ).planar
            last_world_gate = final_base_gate(observed, goal)
            _emit(
                'navigation_world_check', stage=stage, attempt=attempt,
                passed=last_world_gate.ok, reason=last_world_gate.reason,
                **last_world_gate.metrics,
            )
            if last_world_gate.ok:
                _emit(
                    'navigation_reached', stage=stage, attempt=attempt,
                    odom=list(nav.pose or ()),
                    world=[observed.x, observed.y, observed.yaw],
                )
                return
            if attempt < WORLD_WAYPOINT_MAXIMUM_ATTEMPTS:
                position_error = last_world_gate.metrics['final_base_position_error_m']
                yaw_error = last_world_gate.metrics['final_base_yaw_error_rad']
                correction_ok = (
                    position_error <= WORLD_CORRECTION_POSITION_ENVELOPE_M
                    and yaw_error <= WORLD_CORRECTION_YAW_ENVELOPE_RAD
                )
                _emit(
                    'navigation_correction_gate', stage=stage, attempt=attempt,
                    passed=correction_ok,
                    reason=(
                        'certified_same_waypoint_correction'
                        if correction_ok
                        else 'world_residual_outside_correction_envelope'
                    ),
                    position_error_m=position_error,
                    yaw_error_rad=yaw_error,
                    position_envelope_m=WORLD_CORRECTION_POSITION_ENVELOPE_M,
                    yaw_envelope_rad=WORLD_CORRECTION_YAW_ENVELOPE_RAD,
                )
                if not correction_ok:
                    raise RuntimeError(
                        f'{stage} world residual outside certified correction envelope'
                    )
        assert last_world_gate is not None
        _require(last_world_gate)

    try:
        _require(oblique_route_gate())
        deadline = time.monotonic() + 40.0
        required = (
            *IK_JOINTS,
            *RIGHT_ARM_JOINTS,
            *RIGHT_GRIPPER_JOINTS,
            *HEAD_JOINTS,
            *LEFT_GRIPPER_JOINTS,
        )
        while time.monotonic() < deadline:
            ready = (all(name in node.joints for name in required)
                     and node.arm_client.server_is_ready()
                     and node.torso_client.server_is_ready()
                     and node.gripper_pub.get_subscription_count() > 0
                     and nav.pose is not None and nav.last_odom_time is not None
                     and nav.last_front_scan_time is not None and nav.last_rear_scan_time is not None)
            if ready:
                try:
                    _joint_snapshot(node, required)
                except RuntimeError:
                    ready = False
                if ready:
                    break
            time.sleep(0.05)
        else:
            raise RuntimeError('left-arm, joint, or navigation state is unavailable')

        initial_message = read_dynamic_pose_message()
        initial_world = entity_pose_from_dynamic_pose(initial_message, 'tiago_pro').planar
        initial_odom = Pose2(*nav.pose)
        _require(startup_gate(initial_odom, initial_world))
        _require(_cmd_vel_gate(nav))
        initial_start_q8 = _joint_snapshot(node, IK_JOINTS)
        initial_right = _joint_snapshot(node, RIGHT_ARM_JOINTS)
        initial_right_gripper = _joint_snapshot(node, RIGHT_GRIPPER_JOINTS)
        initial_head = _joint_snapshot(node, HEAD_JOINTS)
        _require(official_start_escape_gate(initial_start_q8))
        settled_start = certified_settled_start_gate(
            initial_start_q8[1:], initial_right, initial_head,
        )
        _require(settled_start)
        settled_profile_index = int(settled_start.metrics['certified_profile_index'])
        _require(expected_joint_gate(
            initial_right_gripper,
            OFFICIAL_PASSIVE_RIGHT_GRIPPER_Q,
            limit=PASSIVE_GRIPPER_STATIONARY_LIMIT_M,
            failure_reason='right_gripper_outside_certified_passive_start',
        ))
        right_reference = initial_right
        right_gripper_reference = initial_right_gripper
        head_reference = initial_head
        monitor_thread = threading.Thread(target=passive_monitor, daemon=True)
        monitor_thread.start()

        initial_books = book_poses_from_dynamic_pose(initial_message)
        initial_shelf = entity_pose_from_dynamic_pose(read_full_pose_message(), SHELF)
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('initial scene stability dwell was interrupted')
        initial_second_message = read_dynamic_pose_message()
        _require(scene_gate(
            initial_books,
            book_poses_from_dynamic_pose(initial_second_message),
        ))
        _require(shelf_gate(
            initial_shelf,
            entity_pose_from_dynamic_pose(read_full_pose_message(), SHELF),
        ))

        # The certified base sweep assumes the left gripper is open and the
        # left arm is in HOME.  Establish that exact posture at the empty start
        # area before any base motion.  The torso-only leg is the certified
        # monotonic escape from the official model's two initial shoulder / torso
        # contacts.  It preserves the exact measured left-arm settling pose;
        # the following dense left-only leg then reaches HOME.
        require_not_cancelled('before_startup_gripper_open')
        open_generation = _joint_generation_snapshot(node, LEFT_GRIPPER_JOINTS)
        if not node._open_gripper():
            raise RuntimeError('left gripper startup open failed')
        _wait_for_joint_update(node, LEFT_GRIPPER_JOINTS, open_generation)
        require_left_open()
        measured_start_q8 = _joint_snapshot(node, IK_JOINTS)
        _require(official_start_escape_gate(measured_start_q8))
        _require(expected_joint_gate(
            measured_start_q8[1:],
            CERTIFIED_SETTLED_START_PROFILES[settled_profile_index][0],
            limit=OFFICIAL_SETTLED_PROFILE_TOLERANCE_RAD,
            failure_reason='left_arm_changed_certified_start_profile',
        ))
        measured_torso_clear_q8 = (
            START_TORSO_CLEAR_Q8[0], *measured_start_q8[1:],
        )
        require_not_cancelled('before_startup_torso_escape')
        if not node.move_torso_strict(
            START_TORSO_CLEAR_Q8[0], START_TORSO_ESCAPE_SECONDS,
        ):
            raise RuntimeError('certified torso-only start escape failed')
        _require(expected_joint_gate(_joint_snapshot(node, IK_JOINTS), measured_torso_clear_q8,
                                     limit=ARM_ENDPOINT_LIMIT_RAD,
                                     failure_reason='torso-only_escape_endpoint_missed'))
        require_not_cancelled('before_left_home_escape')
        if not node.follow_dense_left(measured_torso_clear_q8, HOME_Q8, 4.5):
            raise RuntimeError('certified left HOME escape failed')
        _require(expected_joint_gate(_joint_snapshot(node, IK_JOINTS), HOME_Q8,
                                     limit=ARM_ENDPOINT_LIMIT_RAD,
                                     failure_reason='left_home_endpoint_missed'))
        require_monitor_clear()

        # Drive the exact HOME-posture corridor used by the 340-sample
        # certificate.  Each waypoint refreshes the physical world-to-odom
        # transform so accumulated wheel slip cannot shift the grasp standoff.
        nav.position_tolerance = 0.00025
        nav.yaw_tolerance = 0.00025
        for stage, values in zip(('side_translation', 'oblique_rotation', 'book_grasp_standoff'), BASE_ROUTE_WORLD):
            navigate(stage, Pose2(*values))
        nav._publish_zero()
        aligned = entity_pose_from_dynamic_pose(read_dynamic_pose_message(), 'tiago_pro').planar
        _require(final_base_gate(aligned, Pose2(*BASE_ROUTE_WORLD[-1])))
        _require(expected_joint_gate(
            _joint_snapshot(node, IK_JOINTS),
            HOME_Q8,
            limit=ARM_ENDPOINT_LIMIT_RAD,
            failure_reason='left_home_drifted_during_navigation',
        ))
        require_left_open()
        stable_scene()

        # Refresh the one-shot adaptive-close authorization and clear stale
        # contact identity while the open hand is still outside the shelf.
        require_not_cancelled('before_final_gripper_open')
        open_generation = _joint_generation_snapshot(node, LEFT_GRIPPER_JOINTS)
        if not node._open_gripper():
            raise RuntimeError('left gripper open failed')
        _wait_for_joint_update(node, LEFT_GRIPPER_JOINTS, open_generation)
        require_left_open()
        require_not_cancelled('before_home_lift')
        if not node.move_torso_strict(
            HOME_LIFT_Q8[0], HOME_LIFT_TORSO_SECONDS,
        ):
            raise RuntimeError('HOME_LIFT torso command failed')
        _require(expected_joint_gate(_joint_snapshot(node, IK_JOINTS), HOME_LIFT_Q8,
                                     limit=ARM_ENDPOINT_LIMIT_RAD,
                                     failure_reason='home_lift_endpoint_missed'))
        require_not_cancelled('before_oblique_clearance')
        if not node.follow_dense_left(HOME_LIFT_Q8, OBLIQUE_Q8[0], 4.5):
            raise RuntimeError('synchronized oblique clearance motion failed')
        _require(expected_joint_gate(_joint_snapshot(node, IK_JOINTS), OBLIQUE_Q8[0],
                                     limit=ARM_ENDPOINT_LIMIT_RAD,
                                     failure_reason='oblique_clearance_endpoint_missed'))
        require_left_open()

        # Bind the exact target before insertion.  This makes any target contact
        # with a non-gripper robot link latch as a hard fault, and prevents a
        # neighboring book from satisfying pressure acquisition.
        with node._lock:
            node._target_book_model = BOOK
        _require(node.arm_target_pose_watchdog())

        previous = PRESSURE_CLOSE_Q8[0]
        for index, target in enumerate(PRESSURE_CLOSE_Q8[1:], start=1):
            require_not_cancelled(f'before_insertion_leg_{index}')
            require_no_target_robot_contact(f'before_insertion_leg_{index}')
            maximum_delta = max(abs(b - a) for a, b in zip(previous[1:], target[1:]))
            if not node.follow_dense_left(previous, target, max(0.8, maximum_delta / 0.14)):
                watchdog_reason = node.target_pose_watchdog_reason()
                raise RuntimeError(
                    watchdog_reason or f'oblique insertion leg {index} failed'
                )
            require_no_target_robot_contact(f'after_insertion_leg_{index}')
            _require(expected_joint_gate(_joint_snapshot(node, IK_JOINTS), target,
                                         limit=ARM_ENDPOINT_LIMIT_RAD,
                                         failure_reason=f'oblique_insertion_endpoint_{index}_missed'))
            require_monitor_clear()
            require_left_open()
            insertion_scene_guard(f'after_insertion_leg_{index}')
            previous = target

        # No attached corners are invented or monitor armed: no motion follows
        # the close.  The measured pose is retained as truthful evidence.
        preclose_target = stable_scene()[BOOK]
        require_target_pose_watchdog_clear('before_adaptive_close')
        require_not_cancelled('before_adaptive_close')
        require_no_target_robot_contact('before_adaptive_close')
        node.disarm_target_pose_watchdog()
        if str(node.target_colour).strip().lower() != 'red':
            raise RuntimeError('configured target colour is not red')
        verified, width, left_contact, right_contact, plausible = node._adaptive_close_for_grasp()
        nav._publish_zero()
        require_monitor_clear()
        require_no_target_robot_contact('after_adaptive_close')
        if not (verified and plausible and left_contact and right_contact
                and node._target_book_model == BOOK
                and not node.target_robot_contact_latched()):
            # Fail in place: never reopen or retreat after an ambiguous close.
            raise RuntimeError('adaptive close lacked exact bilateral pressure evidence')
        if not node._fresh_retention_probe('strict_physical_pick', 'post_close', require_new_sample=True):
            raise RuntimeError('fresh post-close retention was not verified')
        left_contact, right_contact = node._target_contact_sides(max_age=0.15)
        if not (left_contact and right_contact and node._target_book_model == BOOK
                and not node.target_robot_contact_latched()):
            raise RuntimeError('fresh retention did not preserve exact bilateral identity')
        require_monitor_clear()

        first_final_message = read_dynamic_pose_message()
        first_final_full_message = read_full_pose_message()
        first_final_book = entity_pose_from_dynamic_pose(first_final_message, BOOK)
        if not node._wait_sim_duration(0.20):
            raise RuntimeError('post-close pose stability dwell was interrupted')
        if not node._fresh_retention_probe(
            'strict_physical_pick', 'post_close_pose_dwell', require_new_sample=True,
        ):
            raise RuntimeError('retention was lost during post-close pose dwell')
        require_no_target_robot_contact('post_close_pose_dwell')
        final_message = read_dynamic_pose_message()
        final_full_message = read_full_pose_message()
        final_base = entity_pose_from_dynamic_pose(final_message, 'tiago_pro')
        final_book = entity_pose_from_dynamic_pose(final_message, BOOK)
        _require(postclose_target_gate(preclose_target, first_final_book, final_book))
        _require(shelf_gate(
            entity_pose_from_dynamic_pose(first_final_full_message, SHELF),
            entity_pose_from_dynamic_pose(final_full_message, SHELF),
        ))
        _require(final_base_gate(final_base.planar, Pose2(*BASE_ROUTE_WORLD[-1])))
        left_contact, right_contact = node._target_contact_sides(max_age=0.15)
        if not (left_contact and right_contact and node._target_book_model == BOOK):
            raise RuntimeError('final pressure identity was not fresh')
        require_monitor_clear()
        require_no_target_robot_contact('final')
        _emit('result', passed=True, stage='strict_oblique_pressure_pick_no_extraction',
              target_model=node._target_book_model, measured_gripper_width_m=float(width),
              left_contact=bool(left_contact), right_contact=bool(right_contact),
              target_preclose_position=list(preclose_target.position),
              target_preclose_quaternion=list(preclose_target.quaternion),
              final_base_position=list(final_base.position),
              final_base_quaternion=list(final_base.quaternion),
              final_book_position=list(final_book.position),
              final_book_quaternion=list(final_book.quaternion),
              extraction_commanded=False, base_retreat_commanded=False,
              gripper_reopened=False, gazebo_mutation_used=False,
              right_arm_commanded=False, head_commanded=False,
              attached_payload_monitor_armed=False)
    except Exception as exc:
        nav._publish_zero()
        _emit('result', passed=False, reason=str(exc), extraction_commanded=False,
              base_retreat_commanded=False, gripper_reopened_after_close=False,
              gazebo_mutation_used=False, right_arm_commanded=False,
              head_commanded=False)
        raise
    finally:
        node.disarm_target_pose_watchdog()
        monitor_stop.set()
        node._cancel.set()
        nav._publish_zero()
        if nav.goal is not None:
            nav._finish_goal('cancelled', reason='strict_checkpoint_shutdown')
        goals_cancelled = node.cancel_active_goals()
        _emit('cleanup', passed=goals_cancelled,
              reason='all_controller_goals_terminal' if goals_cancelled
              else 'controller_goal_cancellation_unconfirmed')
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
