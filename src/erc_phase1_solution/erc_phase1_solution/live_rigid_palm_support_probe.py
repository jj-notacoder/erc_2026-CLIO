#!/usr/bin/env python3
"""Fail-closed live proof of rigid-palm support and bilateral caging.

This is a deliberately narrow seed-101 engineering diagnostic.  It accepts
only the recorded, shelf-supported release state, follows the complete audited
free-space route to V710_TOUCH, proves support with one 0.25 mm upward hand
command, and only then closes the fingers in three guarded stages.  It never
pulls the book or commands the mobile base.

The diagnostic expects the dedicated palm collision selector on the existing
arm-left-7 contact sensor.  Exact target-to-palm contact is required during or
after the proof lift, but it is never sufficient by itself: correlated measured
hand and book-center motion remains mandatory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

TARGET_BOOK_MODEL = 'book_col_3_row_2_red'
BOOK_HALF_EXTENTS_M = np.asarray([0.125, 0.015, 0.080], dtype=float)

# Exact clean seed-101 state recorded after guarded reset and staged release on
# 2026-09-05.  The reset target was x=2.775, but the accepted physical state
# settled at x=2.767236585; using the target here would admit the wrong state.
EXPECTED_RELEASED_JOINTS = np.asarray(
    [
        0.3499999999982953,
        0.6851897165837914,
        0.9177062765585426,
        -0.06739602835387239,
        -1.5473529669718242,
        0.28311018646610975,
        1.4223476858040753,
        -0.497003799721225,
    ],
    dtype=float,
)
EXPECTED_RELEASED_BASE_POSE = np.asarray(
    [
        2.126890015059356,
        -0.09328628788094377,
        0.0008626277379560718,
    ],
    dtype=float,
)
EXPECTED_RELEASED_BOOK_POSITION = np.asarray(
    [
        2.7672365851550045,
        -0.14866867839922135,
        1.5768569967968757,
    ],
    dtype=float,
)
EXPECTED_RELEASED_BOOK_QUATERNION = np.asarray(
    [
        0.0019002743619455518,
        0.7071028941387457,
        -0.0018988993998644151,
        0.7071055651308567,
    ],
    dtype=float,
)
EXPECTED_RELEASED_BOOK_MINIMUM = np.asarray(
    [
        2.6871566746778295,
        -0.16409852957937304,
        1.4518566657092717,
    ],
    dtype=float,
)
EXPECTED_RELEASED_BOOK_MAXIMUM = np.asarray(
    [
        2.8473164956321795,
        -0.13323882721906966,
        1.7018573278844797,
    ],
    dtype=float,
)
EXPECTED_RELEASED_BOOK_RELATIVE_POSITION = np.asarray(
    [
        0.6402985574658704,
        -0.05593475055728616,
        1.5768570126799601,
    ],
    dtype=float,
)
EXPECTED_RELEASED_BOOK_RELATIVE_QUATERNION = np.asarray(
    [
        0.00220522770422451,
        0.7071019824717732,
        -0.0022039134072331453,
        0.7071047066177356,
    ],
    dtype=float,
)
EXPECTED_OPEN_GRIPPER_M = 0.069

INITIAL_JOINT_TOLERANCE_RAD = 0.003
INITIAL_GRIPPER_TOLERANCE_M = 0.001
INITIAL_BASE_POSITION_TOLERANCE_M = 0.001
INITIAL_BASE_YAW_TOLERANCE_RAD = 0.001
INITIAL_RELATIVE_BOOK_POSITION_TOLERANCE_M = 0.00075
INITIAL_RELATIVE_BOOK_ROTATION_TOLERANCE_RAD = 0.003
INITIAL_BOOK_EXTENT_TOLERANCE_M = 0.0005

# A fresh deterministic pick can close on a slightly different point along the
# book, so the guarded reset can finish a few millimetres from the one recorded
# release without changing the supported shelf state.  Such a state may be
# used only by translating every audited Cartesian waypoint by the measured
# book offset and resolving IK; the fixed recorded joint route is never used
# against the shifted book.
REANCHOR_JOINT_TOLERANCE_RAD = 0.008
REANCHOR_BOOK_TRANSLATION_LIMIT_M = 0.0035
REANCHOR_BOOK_AXIS_LIMIT_M = np.asarray([0.0035, 0.0015, 0.0010], dtype=float)
REANCHOR_BOOK_ROTATION_TOLERANCE_RAD = 0.006
REANCHOR_SHELF_FRONT_X_M = 2.755
REANCHOR_SHELF_SUPPORT_MARGIN_M = 0.008
REANCHOR_IK_POSITION_TOLERANCE_M = 0.00025
REANCHOR_IK_ORIENTATION_TOLERANCE_RAD = 0.002
# The arm is close to a wrist singularity at the far-high transit pose: a
# measured 2.02 mm Cartesian correction legitimately changes one joint by
# 0.103 rad.  The complete interpolated joint sweep is still checked for
# self/environment collision, so retain a bounded 0.12 rad same-branch cap.
REANCHOR_IK_MAXIMUM_JOINT_DELTA_RAD = 0.12

JOINT_SAMPLE_STEP_RAD = 0.025
ROUTE_TRANSLATION_LIMIT_M = 0.0006
ROUTE_ROTATION_LIMIT_RAD = math.radians(0.8)
TANGENT_NOISE_SAMPLE_COUNT = 7
TANGENT_NOISE_SAMPLE_INTERVAL_SECONDS = 0.10
TANGENT_CENTER_NOISE_LIMIT_M = 0.00003
TANGENT_AABB_NOISE_LIMIT_M = 0.00004
TANGENT_HORIZONTAL_NOISE_LIMIT_M = 0.00004
TANGENT_ROTATION_NOISE_LIMIT_RAD = 0.0005

LIFT_DISTANCE_M = 0.00025
LIFT_MINIMUM_CENTER_RISE_M = 0.00010
LIFT_MAXIMUM_CENTER_RISE_M = 0.00050
LIFT_AABB_RISE_LOWER_M = -0.00008
LIFT_AABB_RISE_UPPER_M = 0.00065
LIFT_HORIZONTAL_LIMIT_M = 0.00050
LIFT_ROTATION_LIMIT_RAD = math.radians(1.0)
HAND_LIFT_TOLERANCE_M = 0.00004
HAND_LIFT_HORIZONTAL_LIMIT_M = 0.00010
HAND_LIFT_ROTATION_LIMIT_RAD = 0.002
BOOK_HAND_RISE_MISMATCH_LIMIT_M = 0.00012

CAGE_MAXIMUM_DROP_M = 0.00020
CAGE_BASE_UPWARD_LIMIT_M = 0.00005
CAGE_AABB_EXTRA_LIMIT_M = 0.00010
CAGE_TRANSLATION_LIMIT_M = 0.00075
CAGE_HORIZONTAL_LIMIT_M = 0.00050
CAGE_ROTATION_LIMIT_RAD = math.radians(1.0)
CAGE_PRECLOSE_M = 0.035
CAGE_PRELOAD_M = 0.030
CAGE_LOCK_M = 0.029
CAGE_COMMAND_TOLERANCE_M = 0.00001
CAGE_MEASURED_TARGET_TOLERANCE_M = 0.0025
CAGE_MAXIMUM_PRECLOSE_TO_PRELOAD_DELTA_M = 0.007
CAGE_MINIMUM_PRECLOSE_TO_PRELOAD_DELTA_M = 0.001
CAGE_MAXIMUM_PRELOAD_TO_LOCK_DELTA_M = 0.002

# The support command is only 0.25 mm, so base drift is bounded well below it.
BASE_STATIONARY_POSITION_LIMIT_M = 0.00008
BASE_STATIONARY_YAW_LIMIT_RAD = 0.00010


def _q(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


# Ordered route retained from live_open_hook_touch.py,
# live_hook_extract_continue.py, live_rigid_palm_touch.py, and
# live_rigid_palm_touch_continue.py.  Keeping the vectors here makes the
# installed package diagnostic independent of top-level checkpoint scripts.
D1 = _q(
    [
        0.35,
        0.737201066950,
        0.896045020690,
        -0.040096002315,
        -1.848248971086,
        0.330283539283,
        1.702658129373,
        -0.465898993781,
    ]
)
HIGH = _q(
    [
        0.35,
        0.694973014385,
        1.006893992766,
        -0.063808759991,
        -1.682065701134,
        0.295259337612,
        1.638822520566,
        -0.442359128450,
    ]
)
Q_EXT = _q(
    [
        0.3499999954440325,
        0.6943560328120175,
        1.0020372683310048,
        -0.06441115695003245,
        -1.6789008214877463,
        0.2942356082976997,
        1.6309414247928256,
        -0.443853762084003,
    ]
)
U260 = _q(
    [
        0.35,
        0.697351187207,
        1.061690013751,
        -0.059854191621,
        -1.720810605813,
        0.303940533832,
        1.729358644769,
        -0.419189885638,
    ]
)
VFAR_HI = _q(
    [
        0.35,
        0.379294706031,
        0.752071665989,
        0.358453269586,
        -1.867268396732,
        0.984559479831,
        0.856536206464,
        -1.037809183768,
    ]
)
VFAR = _q(
    [
        0.35,
        0.413637756743,
        0.613901603249,
        0.403682837994,
        -2.027458764675,
        1.066868822073,
        0.911817792474,
        -1.021105724351,
    ]
)
V58 = _q(
    [
        0.35,
        0.388867592051,
        0.529467885533,
        0.433868597509,
        -1.914378797115,
        1.241782052030,
        0.831650403791,
        -1.254685724378,
    ]
)
V62 = _q(
    [
        0.35,
        0.381317684635,
        0.512856422371,
        0.443246040390,
        -1.836432401396,
        1.331440181664,
        0.806342323957,
        -1.379518448179,
    ]
)
V65 = _q(
    [
        0.35,
        0.377139882245,
        0.508195586664,
        0.451922547376,
        -1.772059069625,
        1.399586802762,
        0.796707938344,
        -1.475523881540,
    ]
)
V68 = _q(
    [
        0.35,
        0.374652507563,
        0.510087002278,
        0.462065241341,
        -1.702060018459,
        1.466908478494,
        0.795150684498,
        -1.572536035926,
    ]
)
V70_LOW = _q(
    [
        0.35,
        0.373949124210,
        0.515024715620,
        0.469910600391,
        -1.651921868096,
        1.510466746480,
        0.798676132414,
        -1.637346684743,
    ]
)
V705_LOW = _q(
    [
        0.35,
        0.373876355599,
        0.516733063469,
        0.472061125985,
        -1.638911196863,
        1.521109075047,
        0.800145620698,
        -1.653542517342,
    ]
)
V710_LOW = _q(
    [
        0.35,
        0.373878203317,
        0.518623469387,
        0.474247055263,
        -1.625706006669,
        1.531643485168,
        0.801831162366,
        -1.669719061621,
    ]
)
V710_U4 = _q(
    [
        0.35,
        0.372956005871,
        0.528346486547,
        0.471505214545,
        -1.616424321031,
        1.527301257255,
        0.798504362357,
        -1.670678403714,
    ]
)
V710_U5 = _q(
    [
        0.35,
        0.372729064154,
        0.530781702803,
        0.470824516415,
        -1.614083212578,
        1.526235525289,
        0.797672704588,
        -1.670935669871,
    ]
)
V710_TOUCH = _q(
    [
        0.35,
        0.372648636041,
        0.531649146839,
        0.470582682252,
        -1.613247656781,
        1.525858120652,
        0.797376699454,
        -1.671029065894,
    ]
)


@dataclass(frozen=True)
class BookSnapshot:
    """Physical target-book pose and axis-aligned bounds from Gazebo."""

    position: np.ndarray
    quaternion: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass(frozen=True)
class MotionMetrics:
    """Book-center motion plus independent AABB and attitude evidence."""

    translation_m: float
    horizontal_m: float
    rotation_rad: float
    center_rise_m: float
    minimum_rise_m: float
    maximum_rise_m: float


@dataclass(frozen=True)
class CartesianMotionMetrics:
    """Measured world-frame hand motion between two FK transforms."""

    rise_m: float
    horizontal_m: float
    rotation_rad: float


@dataclass(frozen=True)
class TangentNoise:
    """No-command motion floor learned immediately before the proof lift."""

    reference: BookSnapshot
    center_z_peak_to_peak_m: float
    minimum_z_peak_to_peak_m: float
    maximum_z_peak_to_peak_m: float
    horizontal_max_deviation_m: float
    rotation_max_deviation_rad: float


PROBE_STAGES = (
    'created',
    'initial',
    'route',
    'tangent',
    'support',
    'preload',
    'caged',
)


@dataclass
class ProbeProgress:
    """Small fail-closed state machine used by the live orchestration."""

    stage: str = 'created'
    failure: str | None = None

    def advance(self, stage: str) -> None:
        if self.failure is not None:
            raise RuntimeError('cannot advance a failed rigid-palm probe')
        try:
            current_index = PROBE_STAGES.index(self.stage)
        except ValueError as exc:
            raise RuntimeError(f'unknown current probe stage: {self.stage}') from exc
        expected_index = current_index + 1
        if expected_index >= len(PROBE_STAGES) or PROBE_STAGES[expected_index] != stage:
            raise RuntimeError(
                f'illegal probe transition {self.stage!r} -> {stage!r}'
            )
        self.stage = stage

    def fail(self, reason: str) -> dict[str, Any]:
        self.failure = str(reason)
        return {
            'event': 'result',
            'passed': False,
            'stage': self.stage,
            'reason': self.failure,
            'next_motion_authorized': False,
        }

    def success_payload(self, **fields: Any) -> dict[str, Any]:
        if self.failure is not None or self.stage != 'caged':
            raise RuntimeError('success is unavailable before the caged state')
        return {
            'event': 'result',
            'passed': True,
            'stage': 'rigid_palm_support_and_bilateral_cage',
            'next_motion_authorized': False,
            **fields,
        }


def sampled_joint_segment(
    start: Sequence[float],
    end: Sequence[float],
    maximum_step: float = JOINT_SAMPLE_STEP_RAD,
) -> list[np.ndarray]:
    """Return every checked configuration, including both endpoints."""

    first = np.asarray(start, dtype=float)
    last = np.asarray(end, dtype=float)
    if first.shape != (8,) or last.shape != (8,):
        raise ValueError('joint endpoints must contain torso plus seven joints')
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(last)):
        raise ValueError('joint endpoints must be finite')
    if not np.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError('maximum joint sample step must be positive and finite')
    largest_delta = float(np.max(np.abs(last - first)))
    steps = max(1, math.ceil(largest_delta / float(maximum_step)))
    return [
        first + (last - first) * (index / steps)
        for index in range(steps + 1)
    ]


def quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert one finite, non-zero xyzw quaternion to a rotation matrix."""

    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError('quaternion must contain four finite xyzw values')
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError('quaternion must be non-zero')
    x, y, z, w = (values / norm).tolist()
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the shortest angular distance between xyzw quaternions."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError('quaternions must contain four xyzw values')
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if (
        not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
        or first_norm <= 1e-12
        or second_norm <= 1e-12
    ):
        raise ValueError('quaternions must be finite and non-zero')
    cosine = abs(float(np.dot(first / first_norm, second / second_norm)))
    return 2.0 * math.acos(float(np.clip(cosine, -1.0, 1.0)))


def rotation_matrix_distance(
    reference: Sequence[Sequence[float]],
    current: Sequence[Sequence[float]],
) -> float:
    """Return the geodesic angle between two finite 3x3 rotations."""

    first = np.asarray(reference, dtype=float)
    second = np.asarray(current, dtype=float)
    if (
        first.shape != (3, 3)
        or second.shape != (3, 3)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError('rotation matrices must be finite 3x3 arrays')
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _validated_snapshot(snapshot: BookSnapshot) -> BookSnapshot:
    arrays = (
        np.asarray(snapshot.position, dtype=float),
        np.asarray(snapshot.quaternion, dtype=float),
        np.asarray(snapshot.minimum, dtype=float),
        np.asarray(snapshot.maximum, dtype=float),
    )
    if (
        arrays[0].shape != (3,)
        or arrays[1].shape != (4,)
        or arrays[2].shape != (3,)
        or arrays[3].shape != (3,)
        or not all(np.all(np.isfinite(value)) for value in arrays)
        or np.any(arrays[3] < arrays[2])
    ):
        raise ValueError('book snapshot is malformed or non-finite')
    quaternion_matrix(arrays[1])
    return BookSnapshot(*arrays)


def motion_metrics(reference: BookSnapshot, current: BookSnapshot) -> MotionMetrics:
    """Measure center, AABB, horizontal, and rotational motion separately."""

    first = _validated_snapshot(reference)
    second = _validated_snapshot(current)
    delta = second.position - first.position
    return MotionMetrics(
        translation_m=float(np.linalg.norm(delta)),
        horizontal_m=float(np.linalg.norm(delta[:2])),
        rotation_rad=quaternion_distance(first.quaternion, second.quaternion),
        center_rise_m=float(second.position[2] - first.position[2]),
        minimum_rise_m=float(second.minimum[2] - first.minimum[2]),
        maximum_rise_m=float(second.maximum[2] - first.maximum[2]),
    )


def cartesian_motion_metrics(
    reference: Sequence[Sequence[float]],
    current: Sequence[Sequence[float]],
) -> CartesianMotionMetrics:
    """Measure achieved world-frame motion from two 4x4 transforms."""

    first = np.asarray(reference, dtype=float)
    second = np.asarray(current, dtype=float)
    if (
        first.shape != (4, 4)
        or second.shape != (4, 4)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError('grasp transforms must be finite 4x4 arrays')
    delta = second[:3, 3] - first[:3, 3]
    return CartesianMotionMetrics(
        rise_m=float(delta[2]),
        horizontal_m=float(np.linalg.norm(delta[:2])),
        rotation_rad=rotation_matrix_distance(first[:3, :3], second[:3, :3]),
    )


def world_hand_pose(
    base_pose: Sequence[float],
    base_hand_pose: Sequence[Sequence[float]],
) -> np.ndarray:
    """Transform a base-frame FK result into the measured world frame."""

    base = np.asarray(base_pose, dtype=float)
    hand = np.asarray(base_hand_pose, dtype=float)
    if (
        base.shape != (3,)
        or hand.shape != (4, 4)
        or not np.all(np.isfinite(base))
        or not np.all(np.isfinite(hand))
    ):
        raise ValueError('base and hand poses must be finite')
    cosine = math.cos(float(base[2]))
    sine = math.sin(float(base[2]))
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    transform[:2, 3] = base[:2]
    return transform @ hand


def relative_book_pose(
    base_pose: Sequence[float],
    snapshot: BookSnapshot,
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-book position and attitude in the planar base frame."""

    base = np.asarray(base_pose, dtype=float)
    book = _validated_snapshot(snapshot)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError('base pose must contain finite x, y, and yaw')
    cosine = math.cos(float(base[2]))
    sine = math.sin(float(base[2]))
    world_rotation = np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    origin = np.asarray([base[0], base[1], 0.0], dtype=float)
    relative_position = world_rotation.T @ (book.position - origin)
    relative_rotation = world_rotation.T @ quaternion_matrix(book.quaternion)
    return relative_position, relative_rotation


def released_state_is_expected(
    snapshot: BookSnapshot,
    base_pose: Sequence[float],
    joints: Sequence[float],
    gripper_position: float,
) -> bool:
    """Recognize only the tightly recorded supported-release state."""

    try:
        book = _validated_snapshot(snapshot)
        base = np.asarray(base_pose, dtype=float)
        arm = np.asarray(joints, dtype=float)
        gripper = float(gripper_position)
        if (
            base.shape != (3,)
            or arm.shape != (8,)
            or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(arm))
            or not np.isfinite(gripper)
        ):
            return False
        relative_position, relative_rotation = relative_book_pose(base, book)
        relative_rotation_error = rotation_matrix_distance(
            quaternion_matrix(EXPECTED_RELEASED_BOOK_RELATIVE_QUATERNION),
            relative_rotation,
        )
        position_error = float(
            np.linalg.norm(
                relative_position - EXPECTED_RELEASED_BOOK_RELATIVE_POSITION
            )
        )
        extent_error = float(
            np.max(
                np.abs(
                    (book.maximum - book.minimum)
                    - (
                        EXPECTED_RELEASED_BOOK_MAXIMUM
                        - EXPECTED_RELEASED_BOOK_MINIMUM
                    )
                )
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        float(np.max(np.abs(arm - EXPECTED_RELEASED_JOINTS)))
        <= INITIAL_JOINT_TOLERANCE_RAD
        and abs(gripper - EXPECTED_OPEN_GRIPPER_M)
        <= INITIAL_GRIPPER_TOLERANCE_M
        and float(np.linalg.norm(base[:2] - EXPECTED_RELEASED_BASE_POSE[:2]))
        <= INITIAL_BASE_POSITION_TOLERANCE_M
        and _angle_error(float(base[2]), float(EXPECTED_RELEASED_BASE_POSE[2]))
        <= INITIAL_BASE_YAW_TOLERANCE_RAD
        and position_error <= INITIAL_RELATIVE_BOOK_POSITION_TOLERANCE_M
        and relative_rotation_error
        <= INITIAL_RELATIVE_BOOK_ROTATION_TOLERANCE_RAD
        and extent_error <= INITIAL_BOOK_EXTENT_TOLERANCE_M
    )


def supported_release_reanchor(
    snapshot: BookSnapshot,
    base_pose: Sequence[float],
    joints: Sequence[float],
    gripper_position: float,
) -> np.ndarray | None:
    """Return a small base-frame book offset for a safe supported release.

    The exact recorded state remains the regression anchor.  A nearby state is
    accepted only while it is open, upright, shelf-supported, dimensionally
    unchanged, and close to the recorded robot/base configuration.  Callers
    must apply the returned translation to every Cartesian route waypoint.
    """

    try:
        book = _validated_snapshot(snapshot)
        base = np.asarray(base_pose, dtype=float)
        arm = np.asarray(joints, dtype=float)
        gripper = float(gripper_position)
        if (
            base.shape != (3,)
            or arm.shape != (8,)
            or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(arm))
            or not np.isfinite(gripper)
        ):
            return None
        relative_position, relative_rotation = relative_book_pose(base, book)
        offset = relative_position - EXPECTED_RELEASED_BOOK_RELATIVE_POSITION
        rotation_error = rotation_matrix_distance(
            quaternion_matrix(EXPECTED_RELEASED_BOOK_RELATIVE_QUATERNION),
            relative_rotation,
        )
        extent_error = float(
            np.max(
                np.abs(
                    (book.maximum - book.minimum)
                    - (
                        EXPECTED_RELEASED_BOOK_MAXIMUM
                        - EXPECTED_RELEASED_BOOK_MINIMUM
                    )
                )
            )
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if not (
        float(np.max(np.abs(arm - EXPECTED_RELEASED_JOINTS)))
        <= REANCHOR_JOINT_TOLERANCE_RAD
        and abs(gripper - EXPECTED_OPEN_GRIPPER_M)
        <= INITIAL_GRIPPER_TOLERANCE_M
        and float(np.linalg.norm(base[:2] - EXPECTED_RELEASED_BASE_POSE[:2]))
        <= INITIAL_BASE_POSITION_TOLERANCE_M
        and _angle_error(float(base[2]), float(EXPECTED_RELEASED_BASE_POSE[2]))
        <= INITIAL_BASE_YAW_TOLERANCE_RAD
        and float(np.linalg.norm(offset))
        <= REANCHOR_BOOK_TRANSLATION_LIMIT_M
        and bool(np.all(np.abs(offset) <= REANCHOR_BOOK_AXIS_LIMIT_M))
        and rotation_error <= REANCHOR_BOOK_ROTATION_TOLERANCE_RAD
        and extent_error <= INITIAL_BOOK_EXTENT_TOLERANCE_M
        and float(book.position[0])
        >= REANCHOR_SHELF_FRONT_X_M + REANCHOR_SHELF_SUPPORT_MARGIN_M
    ):
        return None
    return offset.copy()


def learn_tangent_noise(samples: Sequence[BookSnapshot]) -> TangentNoise:
    """Learn a robust no-command reference and a multi-sample noise floor."""

    if len(samples) < 5:
        raise ValueError('at least five no-command samples are required')
    checked = [_validated_snapshot(sample) for sample in samples]
    positions = np.stack([sample.position for sample in checked])
    minimums = np.stack([sample.minimum for sample in checked])
    maximums = np.stack([sample.maximum for sample in checked])
    anchor = checked[0].quaternion / np.linalg.norm(checked[0].quaternion)
    aligned_quaternions = []
    for sample in checked:
        quaternion = sample.quaternion / np.linalg.norm(sample.quaternion)
        if float(np.dot(quaternion, anchor)) < 0.0:
            quaternion = -quaternion
        aligned_quaternions.append(quaternion)
    median_quaternion = np.median(np.stack(aligned_quaternions), axis=0)
    norm = float(np.linalg.norm(median_quaternion))
    if norm <= 1e-12:
        raise ValueError('no-command quaternions produced no valid median')
    reference = BookSnapshot(
        position=np.median(positions, axis=0),
        quaternion=median_quaternion / norm,
        minimum=np.median(minimums, axis=0),
        maximum=np.median(maximums, axis=0),
    )
    horizontal_deviations = np.linalg.norm(
        positions[:, :2] - reference.position[:2],
        axis=1,
    )
    rotation_deviations = [
        quaternion_distance(reference.quaternion, sample.quaternion)
        for sample in checked
    ]
    return TangentNoise(
        reference=reference,
        center_z_peak_to_peak_m=float(np.ptp(positions[:, 2])),
        minimum_z_peak_to_peak_m=float(np.ptp(minimums[:, 2])),
        maximum_z_peak_to_peak_m=float(np.ptp(maximums[:, 2])),
        horizontal_max_deviation_m=float(np.max(horizontal_deviations)),
        rotation_max_deviation_rad=float(np.max(rotation_deviations)),
    )


def tangent_noise_is_safe(noise: TangentNoise) -> bool:
    values = (
        noise.center_z_peak_to_peak_m,
        noise.minimum_z_peak_to_peak_m,
        noise.maximum_z_peak_to_peak_m,
        noise.horizontal_max_deviation_m,
        noise.rotation_max_deviation_rad,
    )
    return bool(
        all(np.isfinite(value) and value >= 0.0 for value in values)
        and noise.center_z_peak_to_peak_m <= TANGENT_CENTER_NOISE_LIMIT_M
        and noise.minimum_z_peak_to_peak_m <= TANGENT_AABB_NOISE_LIMIT_M
        and noise.maximum_z_peak_to_peak_m <= TANGENT_AABB_NOISE_LIMIT_M
        and noise.horizontal_max_deviation_m
        <= TANGENT_HORIZONTAL_NOISE_LIMIT_M
        and noise.rotation_max_deviation_rad
        <= TANGENT_ROTATION_NOISE_LIMIT_RAD
    )


def no_finger_motion_is_safe(
    metrics: MotionMetrics,
    *,
    left_contact: bool,
    right_contact: bool,
    unexpected_contacts: bool,
    palm_contact: bool = False,
    allow_target_palm_contact: bool = False,
) -> bool:
    """Gate free motion; only the final tangent may touch the target palm."""

    values = (
        metrics.translation_m,
        metrics.horizontal_m,
        metrics.rotation_rad,
        metrics.center_rise_m,
        metrics.minimum_rise_m,
        metrics.maximum_rise_m,
    )
    return bool(
        all(np.isfinite(value) for value in values)
        and not left_contact
        and not right_contact
        and not unexpected_contacts
        and (allow_target_palm_contact or not palm_contact)
        and metrics.translation_m <= ROUTE_TRANSLATION_LIMIT_M
        and metrics.rotation_rad <= ROUTE_ROTATION_LIMIT_RAD
    )


def rigid_support_lift_is_safe(
    metrics: MotionMetrics,
    noise: TangentNoise,
    *,
    left_contact: bool,
    right_contact: bool,
    unexpected_contacts: bool,
    palm_contact: bool,
) -> bool:
    """Require center rise; an AABB-only rise can never prove support."""

    values = (
        metrics.translation_m,
        metrics.horizontal_m,
        metrics.rotation_rad,
        metrics.center_rise_m,
        metrics.minimum_rise_m,
        metrics.maximum_rise_m,
    )
    required_center_rise = max(
        LIFT_MINIMUM_CENTER_RISE_M,
        5.0 * noise.center_z_peak_to_peak_m + 0.00002,
    )
    return bool(
        all(np.isfinite(value) for value in values)
        and tangent_noise_is_safe(noise)
        and not left_contact
        and not right_contact
        and not unexpected_contacts
        and palm_contact
        and required_center_rise
        <= metrics.center_rise_m
        <= LIFT_MAXIMUM_CENTER_RISE_M
        and LIFT_AABB_RISE_LOWER_M
        <= metrics.minimum_rise_m
        <= LIFT_AABB_RISE_UPPER_M
        and LIFT_AABB_RISE_LOWER_M
        <= metrics.maximum_rise_m
        <= LIFT_AABB_RISE_UPPER_M
        and metrics.horizontal_m <= LIFT_HORIZONTAL_LIMIT_M
        and metrics.rotation_rad <= LIFT_ROTATION_LIMIT_RAD
    )


def measured_hand_lift_is_safe(
    hand: CartesianMotionMetrics,
    book: MotionMetrics,
    noise: TangentNoise,
) -> bool:
    """Correlate book-center rise to an exactly measured 0.25 mm hand rise."""

    values = (
        hand.rise_m,
        hand.horizontal_m,
        hand.rotation_rad,
        book.center_rise_m,
    )
    correlation_limit = max(
        BOOK_HAND_RISE_MISMATCH_LIMIT_M,
        5.0 * noise.center_z_peak_to_peak_m + 0.00004,
    )
    return bool(
        all(np.isfinite(value) for value in values)
        and abs(hand.rise_m - LIFT_DISTANCE_M) <= HAND_LIFT_TOLERANCE_M
        and hand.horizontal_m <= HAND_LIFT_HORIZONTAL_LIMIT_M
        and hand.rotation_rad <= HAND_LIFT_ROTATION_LIMIT_RAD
        and abs(book.center_rise_m - hand.rise_m) <= correlation_limit
    )


def cage_motion_is_safe(
    metrics: MotionMetrics,
    noise: TangentNoise,
    *,
    unexpected_contacts: bool,
) -> bool:
    """Bound all book motion and prohibit uncommanded upward cage motion."""

    values = (
        metrics.translation_m,
        metrics.horizontal_m,
        metrics.rotation_rad,
        metrics.center_rise_m,
        metrics.minimum_rise_m,
        metrics.maximum_rise_m,
    )
    upward_limit = max(
        CAGE_BASE_UPWARD_LIMIT_M,
        3.0 * noise.center_z_peak_to_peak_m + 0.00002,
    )
    return bool(
        all(np.isfinite(value) for value in values)
        and not unexpected_contacts
        and metrics.translation_m <= CAGE_TRANSLATION_LIMIT_M
        and metrics.horizontal_m <= CAGE_HORIZONTAL_LIMIT_M
        and metrics.rotation_rad <= CAGE_ROTATION_LIMIT_RAD
        and -CAGE_MAXIMUM_DROP_M <= metrics.center_rise_m <= upward_limit
        and -CAGE_MAXIMUM_DROP_M - CAGE_AABB_EXTRA_LIMIT_M
        <= metrics.minimum_rise_m
        <= upward_limit + CAGE_AABB_EXTRA_LIMIT_M
        and -CAGE_MAXIMUM_DROP_M - CAGE_AABB_EXTRA_LIMIT_M
        <= metrics.maximum_rise_m
        <= upward_limit + CAGE_AABB_EXTRA_LIMIT_M
    )


def cage_stage_width_is_safe(
    stage: str,
    commanded: float,
    measured: float,
    previous_measured: float | None,
) -> bool:
    """Audit the measured jaw change for each mandatory cage stage."""

    commands = {
        'cage_preclose': CAGE_PRECLOSE_M,
        'cage_preload': CAGE_PRELOAD_M,
        'cage_transport_lock': CAGE_LOCK_M,
    }
    if (
        stage not in commands
        or not np.isfinite(commanded)
        or not np.isfinite(measured)
        or abs(commanded - commands[stage]) > CAGE_COMMAND_TOLERANCE_M
        or abs(measured - commanded) > CAGE_MEASURED_TARGET_TOLERANCE_M
    ):
        return False
    if stage == 'cage_preclose':
        return previous_measured is None
    if previous_measured is None or not np.isfinite(previous_measured):
        return False
    inward_delta = float(previous_measured - measured)
    if stage == 'cage_preload':
        return bool(
            CAGE_MINIMUM_PRECLOSE_TO_PRELOAD_DELTA_M
            <= inward_delta
            <= CAGE_MAXIMUM_PRECLOSE_TO_PRELOAD_DELTA_M
        )
    return bool(
        -0.0002
        <= inward_delta
        <= CAGE_MAXIMUM_PRELOAD_TO_LOCK_DELTA_M
        and measured >= CAGE_LOCK_M - CAGE_MEASURED_TARGET_TOLERANCE_M
    )


def _angle_error(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def _protobuf_scalar(block: str, field: str, default: float = 0.0) -> float:
    match = re.search(
        rf'(?m)^\s*{re.escape(field)}:\s*([^\s]+)',
        block,
    )
    return default if match is None else float(match.group(1))


def _entity_pose(message: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    marker = f'name: "{name}"'
    marker_index = message.find(marker)
    if marker_index < 0:
        raise RuntimeError(f'Gazebo pose message omitted {name!r}')
    start = message.rfind('pose {', 0, marker_index)
    next_pose = message.find('\npose {', marker_index)
    block = message[start:] if next_pose < 0 else message[start:next_pose]
    position_match = re.search(r'position\s*\{([^}]*)\}', block, re.DOTALL)
    orientation_match = re.search(
        r'orientation\s*\{([^}]*)\}',
        block,
        re.DOTALL,
    )
    if position_match is None or orientation_match is None:
        raise RuntimeError(f'Gazebo pose for {name!r} was incomplete')
    position = np.asarray(
        [
            _protobuf_scalar(position_match.group(1), axis)
            for axis in ('x', 'y', 'z')
        ],
        dtype=float,
    )
    quaternion = np.asarray(
        [
            _protobuf_scalar(orientation_match.group(1), axis)
            for axis in ('x', 'y', 'z', 'w')
        ],
        dtype=float,
    )
    return position, quaternion


def _gazebo_entity_pose(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one fresh entity pose from Gazebo's world-frame pose topic."""

    if not name:
        raise ValueError('Gazebo entity name must be concrete')
    return _entity_pose(_gazebo_dynamic_pose_message(), name)


def _gazebo_dynamic_pose_message() -> str:
    """Read one coherent Gazebo world-pose sample for diagnostic preflight."""

    return _gazebo_pose_message('/world/erc_world/dynamic_pose/info')


def _gazebo_all_pose_message() -> str:
    """Read the all-entity stream used to locate the static shelf once."""

    return _gazebo_pose_message('/world/erc_world/pose/info')


def _gazebo_pose_message(topic: str) -> str:
    if not topic.startswith('/world/erc_world/'):
        raise ValueError('Gazebo pose topic is outside the audited world')
    completed = subprocess.run(
        [
            'gz',
            'topic',
            '-e',
            '-t',
            topic,
            '-n',
            '1',
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if not completed.stdout.strip():
        raise RuntimeError(f'Gazebo pose topic {topic!r} returned an empty sample')
    return completed.stdout


def _gazebo_book_pose() -> tuple[np.ndarray, np.ndarray]:
    return _gazebo_entity_pose(TARGET_BOOK_MODEL)


def _physical_base_pose() -> np.ndarray:
    """Return fresh Gazebo-world x, y, yaw for the TIAGo model."""

    position, quaternion = _gazebo_entity_pose('tiago_pro')
    rotation = quaternion_matrix(quaternion)
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.asarray([position[0], position[1], yaw], dtype=float)


def _physical_book_bounds() -> tuple[np.ndarray, ...]:
    position, quaternion = _gazebo_book_pose()
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=float,
    )
    corners = (
        (signs * BOOK_HALF_EXTENTS_M) @ quaternion_matrix(quaternion).T
        + position
    )
    return position, quaternion, np.min(corners, axis=0), np.max(corners, axis=0)


def _load_runtime() -> SimpleNamespace:
    """Load ROS dependencies only for a real module invocation."""

    import rclpy
    from ament_index_python.packages import get_package_share_directory
    from rclpy.executors import MultiThreadedExecutor

    from erc_phase1_solution.kinematics import pose_matrix
    from erc_phase1_solution.manipulation_node import ManipulationNode
    from erc_phase1_solution.navigation_node import NavigationNode
    from erc_phase1_solution.rigid_palm_live_preflight import (
        RigidPalmEnvironmentPreflight,
    )

    description_share = Path(get_package_share_directory('erc_description'))
    environment_preflight = RigidPalmEnvironmentPreflight(
        urdf_path=description_share / 'urdf' / 'tiago_pro.urdf',
        shelf_mesh_path=(
            description_share
            / 'models'
            / 'shelf'
            / 'meshes'
            / 'erc_base_shelf.STL'
        ),
        package_resolver=get_package_share_directory,
        pose_message_reader=_gazebo_dynamic_pose_message,
        static_pose_message_reader=_gazebo_all_pose_message,
        target_book=TARGET_BOOK_MODEL,
    )

    return SimpleNamespace(
        rclpy=rclpy,
        MultiThreadedExecutor=MultiThreadedExecutor,
        pose_matrix=pose_matrix,
        NavigationNode=NavigationNode,
        ManipulationNode=ManipulationNode,
        physical_book_bounds=_physical_book_bounds,
        physical_base_pose=_physical_base_pose,
        environment_preflight=environment_preflight,
        BOOK=TARGET_BOOK_MODEL,
    )


LEFT_FINGER_TOKENS = (
    'base_finger_left',
    'inner_finger_left',
    'outer_finger_left',
    'fingertip_left',
)
RIGHT_FINGER_TOKENS = (
    'base_finger_right',
    'inner_finger_right',
    'outer_finger_right',
    'fingertip_right',
)
PALM_COLLISION_TOKEN = (
    'arm_left_7_link_fixed_joint_lump__'
    'gripper_left_base_link_collision_1'
)
BOOK_TOKENS = ('book_col_', 'erc_book', 'book_base_link', 'book_collision')
SCORED_GEOMETRY_TOKENS = (
    'erc_shelf',
    'erc_table',
    'collection_bin',
    *BOOK_TOKENS,
)


def _probe_node_type(
    base_type: type,
    target_book: str,
    *,
    preflight_only: bool = False,
) -> type:
    """Add contact evidence and an optional immutable no-actuation guard."""

    if target_book != TARGET_BOOK_MODEL:
        raise ValueError('the rigid-palm probe requires one concrete target model')
    target_lower = target_book.lower()
    actuation_enabled = not preflight_only

    class PalmSupportProbeNode(base_type):
        def __init__(self) -> None:
            # Set this before the base constructor registers its bound command
            # callback.  The guards below use the immutable closure value, so
            # later mutation of this diagnostic attribute cannot enable motion.
            self._probe_actuation_enabled = actuation_enabled
            self.probe_unexpected_pairs: set[tuple[str, str]] = set()
            self.probe_any_left_latched = False
            self.probe_any_right_latched = False
            self.probe_exact_left_ns = 0
            self.probe_exact_right_ns = 0
            self.probe_exact_palm_ns = 0
            self.probe_exact_palm_latched = False
            super().__init__()

        def _on_command(self, message: Any) -> None:
            if actuation_enabled:
                return super()._on_command(message)
            command = str(getattr(message, 'data', '')).strip().lower()
            self._publish_status(
                'rejected',
                command=command,
                reason='preflight_only',
            )

        def _run_command(self, command: str) -> None:
            if actuation_enabled:
                return super()._run_command(command)
            self._publish_status(
                'rejected',
                command=str(command).strip().lower(),
                reason='preflight_only',
            )

        def _follow(self, *args: Any, **kwargs: Any) -> bool:
            if not actuation_enabled:
                raise RuntimeError(
                    'preflight-only mode forbids controller action goals'
                )
            return super()._follow(*args, **kwargs)

        def _cancel_goal_and_confirm(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> bool:
            if not actuation_enabled:
                raise RuntimeError(
                    'preflight-only mode forbids controller cancellation'
                )
            return super()._cancel_goal_and_confirm(*args, **kwargs)

        def _send_retained_arm_trajectory(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[bool, bool]:
            if not actuation_enabled:
                raise RuntimeError(
                    'preflight-only mode forbids retained arm goals'
                )
            return super()._send_retained_arm_trajectory(*args, **kwargs)

        def _command_gripper(self, position: float) -> bool:
            if not actuation_enabled:
                raise RuntimeError(
                    'preflight-only mode forbids gripper commands'
                )
            return super()._command_gripper(position)

        def _with_probe_lock(self, callback: Callable[[], Any]) -> Any:
            lock = getattr(self, '_lock', None)
            if lock is None:
                return callback()
            with lock:
                return callback()

        def reset_probe_audit(self) -> None:
            def reset() -> None:
                self.probe_unexpected_pairs.clear()
                self.probe_any_left_latched = False
                self.probe_any_right_latched = False
                self.probe_exact_left_ns = 0
                self.probe_exact_right_ns = 0
                self.probe_exact_palm_ns = 0
                self.probe_exact_palm_latched = False

            self._with_probe_lock(reset)

        def clear_probe_finger_evidence(self) -> None:
            def clear() -> None:
                self.probe_any_left_latched = False
                self.probe_any_right_latched = False
                self.probe_exact_left_ns = 0
                self.probe_exact_right_ns = 0

            self._with_probe_lock(clear)

        def clear_probe_palm_evidence(self) -> None:
            def clear() -> None:
                self.probe_exact_palm_ns = 0
                self.probe_exact_palm_latched = False

            self._with_probe_lock(clear)

        def probe_finger_latches(self) -> tuple[bool, bool]:
            return self._with_probe_lock(
                lambda: (
                    bool(self.probe_any_left_latched),
                    bool(self.probe_any_right_latched),
                )
            )

        def probe_exact_finger_sides(
            self,
            max_age: float = 0.25,
        ) -> tuple[bool, bool]:
            now_ns = self.get_clock().now().nanoseconds

            def recent(timestamp: int) -> bool:
                age_ns = now_ns - int(timestamp)
                return bool(
                    timestamp > 0
                    and -100_000_000 <= age_ns <= int(max_age * 1e9)
                )

            return self._with_probe_lock(
                lambda: (
                    recent(self.probe_exact_left_ns),
                    recent(self.probe_exact_right_ns),
                )
            )

        def probe_exact_palm_contact(
            self,
            *,
            max_age: float = 0.25,
            since_ns: int = 0,
        ) -> bool:
            now_ns = self.get_clock().now().nanoseconds

            def observed() -> bool:
                timestamp = int(self.probe_exact_palm_ns)
                age_ns = now_ns - timestamp
                return bool(
                    self.probe_exact_palm_latched
                    and timestamp >= int(since_ns)
                    and timestamp > 0
                    and -100_000_000 <= age_ns <= int(max_age * 1e9)
                )

            return bool(self._with_probe_lock(observed))

        def probe_palm_latched(self) -> bool:
            return bool(
                self._with_probe_lock(
                    lambda: self.probe_exact_palm_latched
                )
            )

        def _on_contacts(self, message: Any) -> None:
            super()._on_contacts(message)
            now_ns = self.get_clock().now().nanoseconds
            saw_left = False
            saw_right = False
            exact_left = False
            exact_right = False
            exact_palm = False
            unexpected: set[tuple[str, str]] = set()
            for contact in getattr(message, 'contacts', []):
                first = str(
                    getattr(getattr(contact, 'collision1', None), 'name', '')
                )
                second = str(
                    getattr(getattr(contact, 'collision2', None), 'name', '')
                )
                names = (first, second)
                lowered = tuple(name.lower() for name in names)
                combined = ' '.join(lowered)
                involves_book = any(token in combined for token in BOOK_TOKENS)
                involves_palm = any(
                    PALM_COLLISION_TOKEN in name for name in lowered
                )
                involves_exact_target = any(
                    target_lower in name for name in lowered
                )
                if involves_palm:
                    if involves_exact_target:
                        exact_palm = True
                    else:
                        # Palm-to-shelf and palm-to-anonymous/non-target book
                        # contacts are always faults.  Treat any other palm
                        # collision conservatively as a fault as well.
                        unexpected.add(tuple(sorted(names)))
                if not involves_book:
                    continue
                left_gripper_names = [
                    name for name in lowered if 'gripper_left_' in name
                ]
                involves_any_gripper = any(
                    'gripper_' in name for name in lowered
                )
                recognized_left_finger = False
                for name in left_gripper_names:
                    if any(token in name for token in LEFT_FINGER_TOKENS):
                        recognized_left_finger = True
                        saw_left = True
                        exact_left |= involves_exact_target
                    if any(token in name for token in RIGHT_FINGER_TOKENS):
                        recognized_left_finger = True
                        saw_right = True
                        exact_right |= involves_exact_target

                concrete_other_book = any(
                    'book_col_' in name and target_lower not in name
                    for name in lowered
                )
                exact_allowed_gripper_contact = bool(
                    involves_exact_target
                    and (involves_palm or recognized_left_finger)
                )
                involves_robot = any('tiago_pro::' in name for name in lowered)
                involves_scored_geometry = any(
                    token in combined for token in SCORED_GEOMETRY_TOKENS
                )
                anonymous_or_wrong_gripper_book = bool(
                    involves_any_gripper
                    and involves_book
                    and (
                        not involves_exact_target
                        or concrete_other_book
                        or not left_gripper_names
                    )
                )
                target_non_gripper_robot = bool(
                    involves_exact_target
                    and involves_robot
                    and not exact_allowed_gripper_contact
                )
                robot_scored_fault = bool(
                    involves_robot
                    and involves_scored_geometry
                    and not exact_allowed_gripper_contact
                )
                if (
                    anonymous_or_wrong_gripper_book
                    or target_non_gripper_robot
                    or robot_scored_fault
                    or (involves_exact_target and concrete_other_book)
                ):
                    unexpected.add(tuple(sorted(names)))

            def apply() -> None:
                self.probe_any_left_latched |= saw_left
                self.probe_any_right_latched |= saw_right
                if exact_left:
                    self.probe_exact_left_ns = now_ns
                if exact_right:
                    self.probe_exact_right_ns = now_ns
                if exact_palm:
                    self.probe_exact_palm_ns = now_ns
                    self.probe_exact_palm_latched = True
                self.probe_unexpected_pairs.update(unexpected)

            self._with_probe_lock(apply)

    return PalmSupportProbeNode


def _snapshot(bounds_reader: Callable[[], Sequence[Any]]) -> BookSnapshot:
    values = bounds_reader()
    if len(values) != 4:
        raise RuntimeError('physical book bounds returned an invalid tuple')
    try:
        return _validated_snapshot(BookSnapshot(*values))
    except ValueError as exc:
        raise RuntimeError('physical book bounds were malformed') from exc


def _validated_base_pose(
    base_reader: Callable[[], Sequence[float]],
) -> np.ndarray:
    pose = np.asarray(base_reader(), dtype=float)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise RuntimeError('base pose is malformed or non-finite')
    return pose


def _require_audited_base_pose(
    base_reader: Callable[[], Sequence[float]],
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    pose = _validated_base_pose(base_reader)
    position_error = float(
        np.linalg.norm(pose[:2] - EXPECTED_RELEASED_BASE_POSE[:2])
    )
    yaw_error = _angle_error(
        float(pose[2]),
        float(EXPECTED_RELEASED_BASE_POSE[2]),
    )
    if (
        position_error > INITIAL_BASE_POSITION_TOLERANCE_M
        or yaw_error > INITIAL_BASE_YAW_TOLERANCE_RAD
    ):
        raise RuntimeError(
            'base is not at the recorded seed-101 release pose '
            f'(position_error={position_error:.6f}, yaw_error={yaw_error:.6f})'
        )
    if reference is not None:
        first = np.asarray(reference, dtype=float)
        if first.shape != (3,) or not np.all(np.isfinite(first)):
            raise RuntimeError('reference base pose is malformed or non-finite')
        drift = float(np.linalg.norm(pose[:2] - first[:2]))
        yaw_drift = _angle_error(float(pose[2]), float(first[2]))
        if (
            drift > BASE_STATIONARY_POSITION_LIMIT_M
            or yaw_drift > BASE_STATIONARY_YAW_LIMIT_RAD
        ):
            raise RuntimeError(
                'base moved during rigid-palm proof '
                f'(drift={drift:.6f}, yaw_drift={yaw_drift:.6f})'
            )
    return pose


def _unexpected_pairs(node: Any) -> list[tuple[str, ...]]:
    # The production parser permits the exact fixed-joint-lumped palm against a
    # target book.  This probe remains stricter: it allows that pair only for
    # the concrete target and records every other robot/scored-geometry pair.
    return sorted(
        set(getattr(node, 'probe_unexpected_pairs', set()))
    )


def _metrics_payload(metrics: MotionMetrics) -> dict[str, float]:
    return {
        'translation_m': metrics.translation_m,
        'horizontal_drift_m': metrics.horizontal_m,
        'rotation_rad': metrics.rotation_rad,
        'book_center_rise_m': metrics.center_rise_m,
        'book_aabb_minimum_rise_m': metrics.minimum_rise_m,
        'book_aabb_maximum_rise_m': metrics.maximum_rise_m,
    }


def _emit_observation(
    event: str,
    node: Any,
    snapshot: BookSnapshot,
    metrics: MotionMetrics,
    left: bool,
    right: bool,
    **fields: Any,
) -> None:
    print(
        json.dumps(
            {
                'event': event,
                **_metrics_payload(metrics),
                'left_exact_target_contact': bool(left),
                'right_exact_target_contact': bool(right),
                'unexpected_contacts': [
                    list(pair) for pair in _unexpected_pairs(node)
                ],
                'position': snapshot.position.tolist(),
                'minimum': snapshot.minimum.tolist(),
                'maximum': snapshot.maximum.tolist(),
                'solution': node._measured_left_solution().tolist(),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _preflight_joint_sweep(
    node: Any,
    target: np.ndarray,
    event: str,
    environment_preflight: Callable[..., Any] | None = None,
    *,
    planned_start: Sequence[float] | None = None,
) -> None:
    """Check every joint sample and invoke the environment-preflight adapter.

    The adapter boundary is intentionally small.  A live integrator can turn
    these joint samples into rigid_palm_preflight.GripperSample values, add the
    fresh shelf triangles and BookOBBs, and call preflight_gripper_sweep.
    """

    start = np.asarray(
        node._measured_left_solution()
        if planned_start is None
        else planned_start,
        dtype=float,
    )
    samples = sampled_joint_segment(start, target)
    for index, sample in enumerate(samples):
        collision = node._robot_self_collision(sample)
        if collision is not None:
            raise RuntimeError(
                f'sampled self-collision during {event} at sample {index}: '
                f'{collision}'
            )
    if environment_preflight is None:
        return
    result = environment_preflight(
        node=node,
        event=event,
        joint_samples=tuple(sample.copy() for sample in samples),
    )
    if not bool(getattr(result, 'safe', False)):
        code = getattr(result, 'code', 'invalid_environment_preflight_result')
        detail = getattr(result, 'detail', repr(result))
        raise RuntimeError(
            f'environment preflight rejected {event}: {code}: {detail}'
        )


def _require_no_finger_move(
    node: Any,
    bounds_reader: Callable[[], Sequence[Any]],
    baseline: BookSnapshot,
    target: np.ndarray,
    duration: float,
    event: str,
    *,
    allow_target_palm_contact: bool = False,
    environment_preflight: Callable[..., Any] | None = None,
) -> BookSnapshot:
    transition_samples = sampled_joint_segment(
        node._measured_left_solution(),
        target,
    )
    _preflight_joint_sweep(
        node,
        target,
        event,
        environment_preflight,
    )
    node._clear_target_contact_samples(reset_robot_contact=True)
    node.clear_probe_finger_evidence()
    node.clear_probe_palm_evidence()
    if not node._move_arm_solution(target, duration):
        raise RuntimeError(f'controller failed during {event}')
    if not node._wait_sim_duration(0.18):
        raise RuntimeError(f'contact dwell interrupted during {event}')
    current = _snapshot(bounds_reader)
    left, right = node.probe_exact_finger_sides(max_age=0.22)
    latched_left, latched_right = node.probe_finger_latches()
    palm_contact = node.probe_palm_latched()
    metrics = motion_metrics(baseline, current)
    _emit_observation(
        event,
        node,
        current,
        metrics,
        left,
        right,
        transition_joint_samples=len(transition_samples),
        exact_target_palm_contact=palm_contact,
        latched_any_left_book_contact=latched_left,
        latched_any_right_book_contact=latched_right,
    )
    if not no_finger_motion_is_safe(
        metrics,
        left_contact=left or latched_left,
        right_contact=right or latched_right,
        unexpected_contacts=bool(_unexpected_pairs(node)),
        palm_contact=palm_contact,
        allow_target_palm_contact=allow_target_palm_contact,
    ):
        raise RuntimeError(f'fail-closed motion/contact gate rejected {event}')
    return current


def _ordered_tangent_route() -> list[tuple[np.ndarray, float, str]]:
    """Return the full coarse and deep free-space route to V710_TOUCH."""

    return [
        (D1, 0.55, 'rigid_return_d1'),
        (HIGH, 0.55, 'rigid_return_high'),
        (Q_EXT, 0.45, 'rigid_return_qext'),
        (U260, 0.60, 'rigid_u260'),
        (VFAR_HI, 1.20, 'rigid_vertical_far_high'),
        (VFAR, 0.70, 'rigid_vertical_far'),
        (V58, 0.50, 'rigid_insert_58'),
        (V62, 0.45, 'rigid_insert_62'),
        (V65, 0.40, 'rigid_insert_65'),
        (V68, 0.36, 'rigid_insert_68'),
        (V70_LOW, 0.34, 'rigid_insert_70_low'),
        (V70_LOW, 0.40, 'rigid_deep_lower_70'),
        (V705_LOW, 0.32, 'rigid_deep_insert_70_5'),
        (V710_LOW, 0.32, 'rigid_deep_insert_71'),
        (V710_U4, 0.45, 'rigid_deep_up_4mm'),
        (V710_U5, 0.32, 'rigid_deep_up_5mm'),
        (V710_TOUCH, 0.30, 'rigid_deep_tangent'),
    ]


def _reanchored_tangent_route(
    node: Any,
    pose_matrix: Any,
    snapshot: BookSnapshot,
    base_pose: Sequence[float],
) -> tuple[list[tuple[np.ndarray, float, str]], np.ndarray]:
    """Resolve the audited route at the measured supported-book offset."""

    start = np.asarray(node._measured_left_solution(), dtype=float)
    gripper = float(node.joints.get('gripper_left_finger_joint', math.nan))
    offset = supported_release_reanchor(snapshot, base_pose, start, gripper)
    if offset is None:
        raise RuntimeError('supported release is outside the re-anchor envelope')

    route: list[tuple[np.ndarray, float, str]] = []
    previous = start
    for recorded, duration, event in _ordered_tangent_route():
        recorded_pose = np.asarray(node.chain.forward(recorded), dtype=float)
        target_position = recorded_pose[:3, 3] + offset
        solution, _ = node.chain.solve(
            pose_matrix(target_position, recorded_pose[:3, :3]),
            [previous, recorded],
            position_tolerance=REANCHOR_IK_POSITION_TOLERANCE_M,
            orientation_tolerance=REANCHOR_IK_ORIENTATION_TOLERANCE_RAD,
            max_iterations=360,
            fixed_positions={'torso_lift_joint': float(start[0])},
        )
        if solution is None:
            raise RuntimeError(f're-anchored tangent IK failed during {event}')
        solved = np.asarray(solution, dtype=float)
        if (
            solved.shape != (8,)
            or not np.all(np.isfinite(solved))
        ):
            raise RuntimeError(
                f're-anchored tangent IK returned malformed joints during {event}'
            )
        maximum_joint_delta = float(np.max(np.abs(solved - recorded)))
        if maximum_joint_delta > REANCHOR_IK_MAXIMUM_JOINT_DELTA_RAD:
            raise RuntimeError(
                're-anchored tangent IK left the audited branch during '
                f'{event}: maximum joint delta {maximum_joint_delta:.9f} rad'
            )
        solved_pose = np.asarray(node.chain.forward(solved), dtype=float)
        if (
            float(np.linalg.norm(solved_pose[:3, 3] - target_position))
            > REANCHOR_IK_POSITION_TOLERANCE_M
            or rotation_matrix_distance(
                solved_pose[:3, :3], recorded_pose[:3, :3]
            )
            > REANCHOR_IK_ORIENTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                f're-anchored tangent IK missed its pose during {event}'
            )
        route.append((solved, duration, event))
        previous = solved
    return route, offset


def _collect_tangent_noise(
    node: Any,
    base_reader: Callable[[], Sequence[float]],
    bounds_reader: Callable[[], Sequence[Any]],
    terminal: BookSnapshot,
    base_start: np.ndarray,
) -> TangentNoise:
    """Latch all finger evidence while learning a no-command noise floor."""

    node._clear_target_contact_samples(reset_robot_contact=True)
    node.clear_probe_finger_evidence()
    samples: list[BookSnapshot] = []
    for index in range(TANGENT_NOISE_SAMPLE_COUNT):
        if not node._wait_sim_duration(TANGENT_NOISE_SAMPLE_INTERVAL_SECONDS):
            raise RuntimeError('tangent no-command sampling was interrupted')
        current = _snapshot(bounds_reader)
        samples.append(current)
        left, right = node.probe_exact_finger_sides(max_age=0.22)
        latched_left, latched_right = node.probe_finger_latches()
        palm_contact = node.probe_palm_latched()
        metrics = motion_metrics(terminal, current)
        _emit_observation(
            'rigid_tangent_noise_sample',
            node,
            current,
            metrics,
            left,
            right,
            sample=index + 1,
            sample_count=TANGENT_NOISE_SAMPLE_COUNT,
            exact_target_palm_contact=palm_contact,
            latched_any_left_book_contact=latched_left,
            latched_any_right_book_contact=latched_right,
        )
        if not no_finger_motion_is_safe(
            metrics,
            left_contact=left or latched_left,
            right_contact=right or latched_right,
            unexpected_contacts=bool(_unexpected_pairs(node)),
            palm_contact=palm_contact,
            allow_target_palm_contact=True,
        ):
            raise RuntimeError('tangent was not stable and finger-clear')
        _require_audited_base_pose(base_reader, reference=base_start)
    noise = learn_tangent_noise(samples)
    if not tangent_noise_is_safe(noise):
        raise RuntimeError(
            'tangent no-command noise floor is too large for a 0.25 mm proof'
        )
    return noise


def _solve_exact_lift(
    node: Any,
    pose_matrix: Any,
    start: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(node.chain.forward(start), dtype=float)
    position = pose[:3, 3].copy()
    position[2] += LIFT_DISTANCE_M
    solution, _ = node.chain.solve(
        pose_matrix(position, pose[:3, :3]),
        [start],
        position_tolerance=0.00002,
        orientation_tolerance=0.0005,
        max_iterations=400,
        fixed_positions={'torso_lift_joint': float(start[0])},
    )
    if solution is None:
        raise RuntimeError('exact 0.25 mm rigid-support lift IK failed')
    solved = np.asarray(solution, dtype=float)
    commanded_metrics = cartesian_motion_metrics(
        pose,
        np.asarray(node.chain.forward(solved), dtype=float),
    )
    if (
        abs(commanded_metrics.rise_m - LIFT_DISTANCE_M) > 0.00003
        or commanded_metrics.horizontal_m > 0.00003
        or commanded_metrics.rotation_rad > 0.0005
    ):
        raise RuntimeError('IK solution did not encode the exact 0.25 mm lift')
    return solved


def _require_rigid_support_lift(
    node: Any,
    runtime: SimpleNamespace,
    noise: TangentNoise,
    base_start: np.ndarray,
) -> BookSnapshot:
    # Do not clear the probe latch here: any transient finger contact observed
    # during tangent dwell must remain disqualifying through the proof lift.
    start_base = _require_audited_base_pose(
        runtime.physical_base_pose,
        reference=base_start,
    )
    start = np.asarray(node._measured_left_solution(), dtype=float)
    start_hand_world = world_hand_pose(
        start_base,
        np.asarray(node.chain.forward(start), dtype=float),
    )
    target = _solve_exact_lift(node, runtime.pose_matrix, start)
    _preflight_joint_sweep(
        node,
        target,
        'rigid_support_lift_250um',
        getattr(runtime, 'environment_preflight', None),
    )
    node.clear_probe_palm_evidence()
    lift_started_ns = node.get_clock().now().nanoseconds
    if not node._move_arm_solution(target, 0.28):
        raise RuntimeError('controller failed during exact 0.25 mm lift')
    if not node._wait_sim_duration(0.18):
        raise RuntimeError('rigid-support lift dwell interrupted')

    current = _snapshot(runtime.physical_book_bounds)
    after_base = _require_audited_base_pose(
        runtime.physical_base_pose,
        reference=base_start,
    )
    measured = np.asarray(node._measured_left_solution(), dtype=float)
    achieved_hand_world = world_hand_pose(
        after_base,
        np.asarray(node.chain.forward(measured), dtype=float),
    )
    hand_metrics = cartesian_motion_metrics(
        start_hand_world,
        achieved_hand_world,
    )
    metrics = motion_metrics(noise.reference, current)
    left, right = node.probe_exact_finger_sides(max_age=0.22)
    latched_left, latched_right = node.probe_finger_latches()
    palm_contact = node.probe_exact_palm_contact(
        max_age=0.22,
        since_ns=lift_started_ns,
    )
    _emit_observation(
        'rigid_support_lift_250um',
        node,
        current,
        metrics,
        left,
        right,
        commanded_rise_m=LIFT_DISTANCE_M,
        measured_world_hand_rise_m=hand_metrics.rise_m,
        measured_world_hand_horizontal_m=hand_metrics.horizontal_m,
        measured_world_hand_rotation_rad=hand_metrics.rotation_rad,
        tangent_center_noise_peak_to_peak_m=(
            noise.center_z_peak_to_peak_m
        ),
        exact_target_palm_contact_during_or_after_lift=palm_contact,
        latched_any_left_book_contact=latched_left,
        latched_any_right_book_contact=latched_right,
    )
    if not rigid_support_lift_is_safe(
        metrics,
        noise,
        left_contact=left or latched_left,
        right_contact=right or latched_right,
        unexpected_contacts=bool(_unexpected_pairs(node)),
        palm_contact=palm_contact,
    ) or not measured_hand_lift_is_safe(hand_metrics, metrics, noise):
        raise RuntimeError('0.25 mm lift did not prove rigid-palm support')
    return current


def _inspect_cage_stage(
    node: Any,
    bounds_reader: Callable[[], Sequence[Any]],
    reference: BookSnapshot,
    noise: TangentNoise,
    event: str,
    commanded: float,
    previous_measured: float | None,
) -> tuple[bool, bool, BookSnapshot, float]:
    current = _snapshot(bounds_reader)
    left, right = node.probe_exact_finger_sides(max_age=0.25)
    measured = float(node.joints['gripper_left_finger_joint'])
    metrics = motion_metrics(reference, current)
    _emit_observation(
        event,
        node,
        current,
        metrics,
        left,
        right,
        commanded_aperture=commanded,
        measured_aperture=measured,
        previous_measured_aperture=previous_measured,
    )
    if not cage_stage_width_is_safe(
        event,
        commanded,
        measured,
        previous_measured,
    ):
        raise RuntimeError(f'measured jaw transition was unsafe during {event}')
    if not cage_motion_is_safe(
        metrics,
        noise,
        unexpected_contacts=bool(_unexpected_pairs(node)),
    ):
        raise RuntimeError(f'book moved unsafely during {event}')
    return bool(left), bool(right), current, measured


def _require_bilateral_cage(
    node: Any,
    bounds_reader: Callable[[], Sequence[Any]],
    lifted: BookSnapshot,
    noise: TangentNoise,
    progress: ProbeProgress,
    environment_preflight: Any | None = None,
) -> BookSnapshot:
    configured = (
        float(node.gripper_preclose),
        float(node.gripper_preload),
    )
    expected = (CAGE_PRECLOSE_M, CAGE_PRELOAD_M)
    if any(
        abs(actual - wanted) > CAGE_COMMAND_TOLERANCE_M
        for actual, wanted in zip(configured, expected, strict=True)
    ):
        raise RuntimeError('configured cage positions differ from audited values')

    node._transport_lock_engaged = False
    previous_measured: float | None = None
    node.clear_probe_finger_evidence()
    _preflight_cage_aperture(
        node,
        environment_preflight,
        'cage_preclose',
        CAGE_PRECLOSE_M,
    )
    if not node._command_gripper(CAGE_PRECLOSE_M):
        raise RuntimeError('gripper command failed during cage_preclose')
    _, _, _, previous_measured = _inspect_cage_stage(
        node,
        bounds_reader,
        lifted,
        noise,
        'cage_preclose',
        CAGE_PRECLOSE_M,
        previous_measured,
    )

    # Preclose contact never authorizes a direct jump to a narrow lock.  A
    # separately measured 30 mm preload and fresh bilateral exact-target
    # evidence are mandatory.  The palm now carries gravity, so this preload
    # is also the final lateral cage: the exact mesh preflight showed that the
    # former 29 mm step exceeded the explicit 1 mm fingertip-overlap cap.
    node.clear_probe_finger_evidence()
    _preflight_cage_aperture(
        node,
        environment_preflight,
        'cage_preload',
        CAGE_PRELOAD_M,
    )
    if not node._command_gripper(CAGE_PRELOAD_M):
        raise RuntimeError('gripper command failed during cage_preload')
    left, right, _, preload_measured = _inspect_cage_stage(
        node,
        bounds_reader,
        lifted,
        noise,
        'cage_preload',
        CAGE_PRELOAD_M,
        previous_measured,
    )
    if not left or not right:
        raise RuntimeError(
            'audited preload did not acquire fresh bilateral target contact'
        )
    progress.advance('preload')

    # Discard callbacks from the preload command itself and require new
    # exact-target contact evidence while the 30 mm cage is stationary.
    node.clear_probe_finger_evidence()
    if not node._wait_sim_duration(0.20):
        raise RuntimeError('fresh bilateral cage dwell was interrupted')
    current = _snapshot(bounds_reader)
    left, right = node.probe_exact_finger_sides(max_age=0.22)
    final_measured = float(node.joints['gripper_left_finger_joint'])
    metrics = motion_metrics(lifted, current)
    _emit_observation(
        'cage_fresh_bilateral',
        node,
        current,
        metrics,
        left,
        right,
        commanded_aperture=CAGE_PRELOAD_M,
        measured_aperture=final_measured,
    )
    if (
        not left
        or not right
        or not cage_stage_width_is_safe(
            'cage_preload',
            CAGE_PRELOAD_M,
            final_measured,
            previous_measured,
        )
        or not cage_motion_is_safe(
            metrics,
            noise,
            unexpected_contacts=bool(_unexpected_pairs(node)),
        )
    ):
        raise RuntimeError('final fresh bilateral exact-target gate failed')
    node._transport_lock_engaged = False
    progress.advance('caged')
    return current


def _preflight_cage_aperture(
    node: Any,
    environment_preflight: Any | None,
    event: str,
    target_aperture_m: float,
) -> None:
    """Run the nine-link environment gate before one gripper command."""

    if environment_preflight is None:
        return
    method = getattr(environment_preflight, 'preflight_aperture_sweep', None)
    if not callable(method):
        raise RuntimeError('environment preflight cannot check cage motion')
    result = method(
        node=node,
        event=event,
        joint_positions=node._measured_left_solution(),
        target_aperture_m=target_aperture_m,
    )
    if not bool(getattr(result, 'safe', False)):
        code = getattr(result, 'code', 'invalid_environment_preflight_result')
        detail = getattr(result, 'detail', repr(result))
        raise RuntimeError(
            f'environment preflight rejected {event}: {code}: {detail}'
        )


def _require_initial_state(
    node: Any,
    runtime: SimpleNamespace,
) -> tuple[BookSnapshot, np.ndarray]:
    base = _require_audited_base_pose(runtime.physical_base_pose)
    arm = np.asarray(node._measured_left_solution(), dtype=float)
    gripper = float(node.joints.get('gripper_left_finger_joint', math.nan))
    if (
        getattr(node, '_held_book_corners', None) is not None
        or getattr(node, '_transport_lock_engaged', False)
    ):
        raise RuntimeError('retained-payload state was not cleared by release')
    initial = _snapshot(runtime.physical_book_bounds)
    initial_offset = supported_release_reanchor(initial, base, arm, gripper)
    if initial_offset is None:
        raise RuntimeError(
            'robot/book state is outside the guarded supported-release envelope'
        )

    node._clear_target_contact_samples(reset_robot_contact=True)
    node.clear_probe_finger_evidence()
    node.clear_probe_palm_evidence()
    if not node._wait_sim_duration(
        TANGENT_NOISE_SAMPLE_INTERVAL_SECONDS
        * TANGENT_NOISE_SAMPLE_COUNT
    ):
        raise RuntimeError('initial shelf-support dwell interrupted')
    settled = _snapshot(runtime.physical_book_bounds)
    settled_base = _require_audited_base_pose(
        runtime.physical_base_pose,
        reference=base,
    )
    settled_arm = np.asarray(node._measured_left_solution(), dtype=float)
    settled_gripper = float(
        node.joints.get('gripper_left_finger_joint', math.nan)
    )
    left, right = node.probe_exact_finger_sides(max_age=0.22)
    latched_left, latched_right = node.probe_finger_latches()
    palm_contact = node.probe_palm_latched()
    metrics = motion_metrics(initial, settled)
    _emit_observation(
        'initial_supported_open_state',
        node,
        settled,
        metrics,
        left,
        right,
        joint_max_error_rad=float(
            np.max(np.abs(settled_arm - EXPECTED_RELEASED_JOINTS))
        ),
        relative_book_position_tolerance_m=(
            INITIAL_RELATIVE_BOOK_POSITION_TOLERANCE_M
        ),
        latched_any_left_book_contact=latched_left,
        latched_any_right_book_contact=latched_right,
        exact_target_palm_contact=palm_contact,
        route_reanchor_translation_m=initial_offset.tolist(),
    )
    if (
        supported_release_reanchor(
            settled,
            settled_base,
            settled_arm,
            settled_gripper,
        )
        is None
        or not no_finger_motion_is_safe(
            metrics,
            left_contact=left or latched_left,
            right_contact=right or latched_right,
            unexpected_contacts=bool(_unexpected_pairs(node)),
            palm_contact=palm_contact,
        )
    ):
        raise RuntimeError('initial supported/open state is not stable and clear')
    return settled, base


def _run_preflight_only(
    node: Any,
    runtime: SimpleNamespace,
    base_start: np.ndarray,
) -> None:
    """Check the complete planned route without issuing any controller goal."""

    environment_preflight = getattr(runtime, 'environment_preflight', None)
    if environment_preflight is None:
        raise RuntimeError('nine-link environment preflight is unavailable')
    baseline = _snapshot(runtime.physical_book_bounds)
    route, reanchor = _reanchored_tangent_route(
        node, runtime.pose_matrix, baseline, base_start
    )
    planned_start = np.asarray(node._measured_left_solution(), dtype=float)
    checked_events: list[str] = []
    for target, _, event in route:
        _preflight_joint_sweep(
            node,
            target,
            event,
            environment_preflight,
            planned_start=planned_start,
        )
        checked_events.append(event)
        planned_start = np.asarray(target, dtype=float).copy()

    lift_target = _solve_exact_lift(
        node,
        runtime.pose_matrix,
        planned_start,
    )
    _preflight_joint_sweep(
        node,
        lift_target,
        'rigid_support_lift_250um',
        environment_preflight,
        planned_start=planned_start,
    )
    checked_events.append('rigid_support_lift_250um')

    aperture_method = getattr(
        environment_preflight,
        'preflight_aperture_sweep',
        None,
    )
    if not callable(aperture_method):
        raise RuntimeError('environment preflight cannot check cage motion')
    aperture_start = EXPECTED_OPEN_GRIPPER_M
    for event, aperture_target in (
        ('cage_preclose', CAGE_PRECLOSE_M),
        ('cage_preload', CAGE_PRELOAD_M),
    ):
        result = aperture_method(
            node=node,
            event=event,
            joint_positions=lift_target,
            start_aperture_m=aperture_start,
            target_aperture_m=aperture_target,
            target_book_translation=(0.0, 0.0, LIFT_DISTANCE_M),
        )
        if not bool(getattr(result, 'safe', False)):
            code = getattr(
                result,
                'code',
                'invalid_environment_preflight_result',
            )
            detail = getattr(result, 'detail', repr(result))
            raise RuntimeError(
                f'environment preflight rejected {event}: {code}: {detail}'
            )
        checked_events.append(event)
        aperture_start = aperture_target

    _require_audited_base_pose(
        runtime.physical_base_pose,
        reference=base_start,
    )
    print(
        json.dumps(
            {
                'event': 'preflight_result',
                'passed': True,
                'motion_executed': False,
                'planned_sweeps': checked_events,
                'route_reanchor_translation_m': reanchor.tolist(),
                'next_motion_authorized': False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run(runtime: SimpleNamespace, *, preflight_only: bool = False) -> None:
    if runtime.BOOK != TARGET_BOOK_MODEL:
        raise RuntimeError('runtime target identity is not the audited book')
    if getattr(runtime, 'environment_preflight', None) is None:
        raise RuntimeError('nine-link environment preflight is unavailable')
    ProbeNode = _probe_node_type(
        runtime.ManipulationNode,
        runtime.BOOK,
        preflight_only=preflight_only,
    )
    node = ProbeNode()
    nav = None if preflight_only else runtime.NavigationNode()
    executor = runtime.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    if nav is not None:
        executor.add_node(nav)
    spin_errors: list[BaseException] = []

    def spin_executor() -> None:
        try:
            executor.spin()
        except BaseException as exc:
            # A worker-thread exception must be surfaced by _run rather than
            # leaking through threading/rclpy while node handles are destroyed.
            spin_errors.append(exc)

    spin_thread = threading.Thread(
        target=spin_executor,
        name='rigid-palm-probe-executor',
        daemon=True,
    )
    spin_thread.start()
    progress = ProbeProgress()
    run_error: Exception | None = None
    try:
        deadline = time.monotonic() + 30.0
        while len(node.joints) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
        if len(node.joints) < 8:
            raise RuntimeError('live robot state is unavailable')

        node._target_book_model = TARGET_BOOK_MODEL
        node._payload_monitor_enabled = False
        node._retention_probe_active = True
        node._gravity_supported_payload = False
        node._target_robot_contact_latched = False
        node.reset_probe_audit()

        baseline, base_start = _require_initial_state(node, runtime)
        if preflight_only:
            _run_preflight_only(node, runtime, base_start)
            return
        progress.advance('initial')
        terminal = baseline
        route, reanchor = _reanchored_tangent_route(
            node, runtime.pose_matrix, baseline, base_start
        )
        print(
            json.dumps(
                {
                    'event': 'rigid_route_reanchored',
                    'translation_m': reanchor.tolist(),
                    'waypoints': len(route),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for target, duration, event in route:
            terminal = _require_no_finger_move(
                node,
                runtime.physical_book_bounds,
                baseline,
                target,
                duration,
                event,
                allow_target_palm_contact=(event == 'rigid_deep_tangent'),
                environment_preflight=runtime.environment_preflight,
            )
        progress.advance('route')

        _require_audited_base_pose(
            runtime.physical_base_pose,
            reference=base_start,
        )
        noise = _collect_tangent_noise(
            node,
            runtime.physical_base_pose,
            runtime.physical_book_bounds,
            terminal,
            base_start,
        )
        progress.advance('tangent')

        lifted = _require_rigid_support_lift(
            node,
            runtime,
            noise,
            base_start,
        )
        _require_audited_base_pose(
            runtime.physical_base_pose,
            reference=base_start,
        )
        progress.advance('support')

        caged = _require_bilateral_cage(
            node,
            runtime.physical_book_bounds,
            lifted,
            noise,
            progress,
            runtime.environment_preflight,
        )
        # The final base check occurs after the entire close and before success.
        _require_audited_base_pose(
            runtime.physical_base_pose,
            reference=base_start,
        )
        final_metrics = motion_metrics(lifted, caged)
        payload = progress.success_payload(
            target_palm_contact_required=True,
            commanded_support_lift_m=LIFT_DISTANCE_M,
            post_lift_cage_motion=_metrics_payload(final_metrics),
            position=caged.position.tolist(),
            minimum=caged.minimum.tolist(),
            maximum=caged.maximum.tolist(),
            measured_aperture=float(
                node.joints['gripper_left_finger_joint']
            ),
            solution=node._measured_left_solution().tolist(),
        )
        print(json.dumps(payload, sort_keys=True), flush=True)
    except Exception as exc:
        run_error = exc
        print(json.dumps(progress.fail(str(exc)), sort_keys=True), flush=True)
        raise
    finally:
        node._retention_probe_active = False
        if nav is not None:
            nav._publish_zero()
        executor.shutdown()
        spin_thread.join()
        node.destroy_node()
        if nav is not None:
            nav.destroy_node()
        if spin_errors and run_error is None:
            error = spin_errors[0]
            raise RuntimeError(f'executor spin failed: {error}') from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the seed-101 rigid-palm engineering diagnostic.',
    )
    parser.add_argument(
        '--preflight-only',
        action='store_true',
        help='validate every planned sweep without commanding the robot',
    )
    arguments = parser.parse_args()
    runtime = _load_runtime()
    runtime.rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    try:
        _run(runtime, preflight_only=arguments.preflight_only)
    finally:
        if runtime.rclpy.ok():
            runtime.rclpy.shutdown()


if __name__ == '__main__':
    main()
