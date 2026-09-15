"""Pure motion presets and shelf-grasp orientation candidates."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


ARM_JOINTS = tuple(f'arm_left_{index}_joint' for index in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f'arm_right_{index}_joint' for index in range(1, 8))
IK_JOINTS = ('torso_lift_joint', *ARM_JOINTS)

# All vectors contain torso lift followed by the seven left-arm joints. These
# values are inside the limits baked into the official v1.0.3 TIAGo Pro URDF.
HOME = np.asarray([0.10, 0.36, -1.83, 0.47, -2.35, 0.0, -1.20, 0.0])
# PAL's official ``home_right`` posture, excluding the separately controlled
# torso joint.  The right arm is parked here once and is never used to handle
# the book; leaving its simulated start pose near zero extends it into the
# shelf during the left-arm grasp approach.
RIGHT_HOME = np.asarray([-0.36, -1.83, -0.47, -2.35, 0.0, -1.20, 0.0])
OFFER = np.asarray(
    [0.14, -0.25843, -0.57522, 0.50314, -2.0337, 0.0, 1.0543, 1.5708]
)
PREGRASP = np.asarray(
    [0.22, 3.1646, -2.1672, -0.40768, -1.9953, 0.0, -1.0548, 0.0]
)

# Compact carried-book posture.  Unlike HOME, this keeps the grasp link upright
# and the inflated official book envelope in front of the robot instead of
# folding it across the proximal arm.  It is never commanded blindly: the
# manipulation planner checks the measured book envelope, shelf plane and
# official robot collision meshes over every subdivided transition first.
CARRY = np.asarray(
    [
        0.10,
        3.45924571,
        -2.20780209,
        -1.51348316,
        -1.90602684,
        -2.50362934,
        -1.45613156,
        -0.89091652,
    ]
)

# Gravity-supported compact posture for books extracted from the top row.
# The wrist keeps the finger-closing axis mostly vertical while preserving a
# 0.27 rad margin to its lower soft limit and fitting the payload and arm
# inside the mobile base envelope.
SUPPORTED_CARRY = np.asarray(
    [
        0.10,
        3.43734870845,
        -2.26760092740,
        -1.60197601886,
        -1.92696895395,
        -2.47218391424,
        -1.52315137806,
        -2.10000000000,
    ]
)


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=float,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=float,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _shelf_pitches(height: float) -> Tuple[float, ...]:
    if not math.isfinite(height):
        raise ValueError('target height must be finite')
    if height >= 1.42:
        return (-0.50, -0.25, 0.0)
    if height >= 1.10:
        return (-0.25, 0.0, -0.50)
    if height <= 0.76:
        return (0.35, 0.0, -0.25)
    return (0.0, -0.25, 0.25)


def shelf_grasp_orientations(height: float) -> Tuple[np.ndarray, ...]:
    """Return broad end-effector rotations for a shelf target height.

    The gripper x-axis remains primarily shelf-normal. Tilting it upward for
    high rows and downward for the bottom row avoids a hard reach singularity.
    A -90 degree roll is preferred because it matches the nominal
    forward-facing PAL gripper pose.
    """
    pitches = _shelf_pitches(height)
    rolls = (-math.pi / 2.0, 0.0, math.pi / 2.0, math.pi)
    return tuple(
        _rotation_y(pitch) @ _rotation_x(roll)
        for pitch in pitches
        for roll in rolls
    )


def shelf_pinch_orientations(height: float) -> Tuple[np.ndarray, ...]:
    """Return shelf poses whose finger-closing axis stays horizontal.

    The books are thin along the robot's lateral axis.  Roll angles of zero
    or pi keep the gripper's local y-axis aligned with that thickness; the
    vertical-roll candidates are useful for reach checks but cannot pinch a
    vertical book reliably.
    """

    rolls = (0.0, math.pi)
    standard = tuple(
        _rotation_y(pitch) @ _rotation_x(roll)
        for pitch in _shelf_pitches(height)
        for roll in rolls
    )
    if height < 1.42:
        return standard
    # The high-row shelf aperture is too tight for a yawed gripper: even a
    # modest yaw makes one fingertip lead the other into the shelf.  This
    # single roll-pi pose keeps both fingers shelf-tangent.  Its reachable arm
    # branch is recovered by endpoint-first planning in ManipulationNode.
    return (_rotation_y(-0.50) @ _rotation_x(math.pi),)


def bin_place_orientations() -> Tuple[np.ndarray, ...]:
    """Return horizontal-jaw poses suitable for lowering into the bin.

    A modest downward pitch avoids the reach singularity seen when the base is
    aligned at the nominal bin standoff.  Every candidate preserves a level
    finger-closing axis.  Roll-pi variants let a shelf-grasped book retain its
    attitude instead of being inverted during the bin transition.
    """

    # The level roll-pi pose is the shortest non-inverting route from the
    # grasp-specific staging posture.  Put it first so loaded planning does
    # not spend its timeout proving the lower-ranked roll-zero poses unsafe.
    pitches = (0.0, 0.25, -0.25, 0.50, -0.50)
    return tuple(
        _rotation_y(pitch) @ _rotation_x(roll)
        for roll in (math.pi, 0.0)
        for pitch in pitches
    )


def supported_bin_place_orientations() -> Tuple[np.ndarray, ...]:
    """Return vertical-jaw bin poses for a mechanically cradled book."""

    # +0.25 rad is reachable from the top-row cradle without approaching the
    # wrist soft limit.  Keeping +pi/2 roll preserves lower-finger support and
    # releases the book horizontally over the collection bin.
    return tuple(
        _rotation_y(pitch) @ _rotation_x(math.pi / 2.0)
        for pitch in (0.25, 0.50, 0.0)
    )


__all__ = [
    'ARM_JOINTS',
    'CARRY',
    'HOME',
    'IK_JOINTS',
    'OFFER',
    'PREGRASP',
    'RIGHT_ARM_JOINTS',
    'RIGHT_HOME',
    'SUPPORTED_CARRY',
    'bin_place_orientations',
    'shelf_grasp_orientations',
    'shelf_pinch_orientations',
    'supported_bin_place_orientations',
]
