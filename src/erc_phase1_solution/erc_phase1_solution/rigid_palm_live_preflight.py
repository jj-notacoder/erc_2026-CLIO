"""Gazebo-backed adapter for the rigid-palm diagnostic preflight.

This module is intentionally diagnostic-only.  It reads Gazebo entity poses
to validate one recorded engineering checkpoint; the competition solution
must instead obtain the shelf and book scene from vision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import URDFChain, load_stl_triangles
from .rigid_palm_preflight import (
    DEFAULT_MAX_VERTEX_STEP_M,
    LEFT_GRIPPER_COLLISION_LINKS,
    PALM_COLLISION_LINK,
    BookOBB,
    GripperCollisionModel,
    GripperSample,
    PreflightConfig,
    PreflightResult,
    ProbePhase,
    load_left_gripper_collision_model,
    preflight_gripper_sweep,
)


BOOK_NAME_PATTERN = re.compile(
    r'^book_col_([1-5])_row_([2-5])_(red|green|yellow|blue)$'
)
EXPECTED_BOOK_SLOTS = frozenset(
    (column, row)
    for column in range(1, 6)
    for row in range(2, 6)
)
EXPECTED_COLUMN_COLOURS = frozenset(('red', 'green', 'yellow', 'blue'))
BOOK_HALF_EXTENTS_M = np.asarray([0.125, 0.015, 0.080], dtype=float)
ROBOT_MODEL_NAME = 'tiago_pro'
SHELF_MODEL_NAME = 'erc_shelf'
MOVING_GRIPPER_LINKS = (
    'gripper_left_inner_finger_left_link',
    'gripper_left_outer_finger_left_link',
    'gripper_left_fingertip_left_link',
    'gripper_left_inner_finger_right_link',
    'gripper_left_outer_finger_right_link',
    'gripper_left_fingertip_right_link',
)
OPEN_APERTURE_M = 0.069
MAX_DENSE_SAMPLES = 20_000
FK_POSITION_TOLERANCE_M = 0.00002
# The loaded passive finger linkage can settle about 0.49 degrees away from
# the ideal mimic angle while its link origin remains within a few microns.
# Together with the 20-micrometre origin tolerance, 0.01 rad lets the farthest
# vertex of the audited finger meshes differ by at most 0.589 mm, still inside
# the 0.75 mm collision padding used below.
FK_ROTATION_TOLERANCE_RAD = 0.01
STATE_JOINT_STABILITY_RAD = 0.0002
STATE_APERTURE_STABILITY_M = 0.0002
SCENE_POSITION_STABILITY_M = 0.00008
SCENE_ROTATION_STABILITY_RAD = 0.0001
# Repeated stationary loaded-cage samples show about 0.19--0.22 mm of passive
# fingertip settling.  The sweep starts from the measured geometry, so a
# 0.25 mm stability allowance still leaves 0.50 mm of the 0.75 mm collision
# padding for unmodelled separation error.
MEASURED_FINGER_VERTEX_STABILITY_M = 0.00025
BOX_SIGNS = np.asarray(
    [
        (x, y, z)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ],
    dtype=float,
)

APPROACH_EVENTS = frozenset(
    (
        'rigid_return_high',
        'rigid_return_qext',
        'rigid_u260',
        'rigid_vertical_far_high',
        'rigid_vertical_far',
        'rigid_insert_58',
        'rigid_insert_62',
        'rigid_insert_65',
        'rigid_insert_68',
        'rigid_insert_70_low',
        'rigid_deep_lower_70',
        'rigid_deep_insert_70_5',
        'rigid_deep_insert_71',
        'rigid_deep_up_4mm',
    )
)
DEPARTURE_EVENT = 'rigid_return_d1'
TANGENT_APPROACH_EVENT = 'rigid_deep_up_5mm'
FINAL_TANGENT_EVENT = 'rigid_deep_tangent'
LIFT_EVENT = 'rigid_support_lift_250um'
CAGE_EVENTS = frozenset(
    (
        'cage_preclose',
        'cage_preload',
        'cage_transport_lock',
        'rigid_caged_extraction',
    )
)
CAGED_EXTRACTION_EVENT = 'rigid_caged_extraction'


@dataclass(frozen=True)
class WorldScene:
    """One coherent Gazebo-world snapshot used by a planned sweep."""

    base_transform: np.ndarray
    shelf_triangles: np.ndarray
    books: Mapping[str, BookOBB]
    model_link_transforms: Mapping[str, np.ndarray]
    expected_book_names: frozenset[str]
    observed_at: float


@dataclass(frozen=True)
class DenseJointSample:
    positions: np.ndarray
    transforms: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class MeasuredRobotState:
    positions: np.ndarray
    aperture_m: float


def conservative_tool_vertex_displacement(
    model: GripperCollisionModel,
    first_transforms: Mapping[str, np.ndarray],
    second_transforms: Mapping[str, np.ndarray],
) -> float:
    """Bound mesh-vertex motion using each local mesh's AABB corners.

    Every real mesh vertex lies inside this box.  Rigid-transform displacement
    is convex in the local point, so the maximum over the enclosing box occurs
    at a corner and is a conservative upper bound for every real vertex.
    """

    maximum = 0.0
    for mesh in model.meshes:
        bounds = np.asarray(mesh.bounds, dtype=float)
        center = 0.5 * (bounds[0] + bounds[1])
        half = 0.5 * (bounds[1] - bounds[0])
        corners = center + BOX_SIGNS * half
        first = np.asarray(first_transforms[mesh.link], dtype=float)
        second = np.asarray(second_transforms[mesh.link], dtype=float)
        first_world = corners @ first[:3, :3].T + first[:3, 3]
        second_world = corners @ second[:3, :3].T + second[:3, 3]
        maximum = max(
            maximum,
            float(np.max(np.linalg.norm(second_world - first_world, axis=1))),
        )
    return maximum


def _protobuf_scalar(block: str, field: str, default: float = 0.0) -> float:
    match = re.search(rf'(?m)^\s*{re.escape(field)}:\s*([^\s]+)', block)
    return default if match is None else float(match.group(1))


def _pose_blocks(message: str) -> tuple[str, ...]:
    """Extract the repeated top-level ``pose { ... }`` protobuf blocks."""

    blocks: list[str] = []
    for match in re.finditer(r'(?m)^pose\s*\{', message):
        opening = message.find('{', match.start())
        depth = 0
        for index in range(opening, len(message)):
            character = message[index]
            if character == '{':
                depth += 1
            elif character == '}':
                depth -= 1
                if depth == 0:
                    blocks.append(message[match.start():index + 1])
                    break
        else:
            raise ValueError('Gazebo pose message contains an open pose block')
    return tuple(blocks)


def _quaternion_rotation(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm((x, y, z, w)))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('entity orientation quaternion is invalid')
    if abs(norm - 1.0) > 1e-3:
        raise ValueError('entity orientation quaternion is not normalized')
    x, y, z, w = np.asarray((x, y, z, w), dtype=float) / norm
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


def _pose_transform(block: str) -> tuple[str, np.ndarray]:
    name_match = re.search(r'(?m)^\s*name:\s*"([^"]+)"', block)
    position_match = re.search(r'position\s*\{([^}]*)\}', block, re.DOTALL)
    orientation_match = re.search(
        r'orientation\s*\{([^}]*)\}',
        block,
        re.DOTALL,
    )
    if name_match is None or position_match is None or orientation_match is None:
        raise ValueError('Gazebo pose block is missing a name or transform')
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
    if not np.all(np.isfinite(position)):
        raise ValueError(f'Gazebo pose for {name_match.group(1)!r} is nonfinite')
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _quaternion_rotation(quaternion)
    transform[:3, 3] = position
    return name_match.group(1), transform


def parse_pose_transforms(message: str) -> tuple[tuple[str, np.ndarray], ...]:
    """Parse named transforms from one textual Gazebo ``Pose_V`` sample."""

    if not isinstance(message, str) or not message.strip():
        raise ValueError('Gazebo pose message is empty')
    # ``gz topic -n 1`` occasionally delivers the latched sample followed by
    # one live sample before its subscriber exits.  Treating that concatenated
    # stdout as one Pose_V duplicates every entity and makes an otherwise safe
    # preflight fail nondeterministically.  The final header starts the newest
    # complete sample, so consistently parse that coherent frame only.
    headers = tuple(re.finditer(r'(?m)^header\s*\{', message))
    frame = message[headers[-1].start():] if headers else message
    blocks = _pose_blocks(frame)
    if not blocks:
        raise ValueError('Gazebo pose message contains no poses')
    return tuple(_pose_transform(block) for block in blocks)


def pose_message_timestamp(message: str) -> float:
    """Return the simulation timestamp carried by one Gazebo Pose_V."""

    headers = tuple(re.finditer(r'(?m)^header\s*\{', message))
    frame = message[headers[-1].start():] if headers else message
    header = re.search(
        r'header\s*\{.*?stamp\s*\{([^}]*)\}',
        frame,
        re.DOTALL,
    )
    if header is None:
        raise ValueError('Gazebo pose message has no simulation timestamp')
    seconds = _protobuf_scalar(header.group(1), 'sec')
    nanoseconds = _protobuf_scalar(header.group(1), 'nsec')
    if (
        not math.isfinite(seconds)
        or not math.isfinite(nanoseconds)
        or seconds < 0.0
        or not 0.0 <= nanoseconds < 1_000_000_000.0
    ):
        raise ValueError('Gazebo pose timestamp is invalid')
    return seconds + nanoseconds * 1e-9


def _validated_rigid_transform(
    transform: Sequence[Sequence[float]],
    *,
    label: str,
) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f'{label} must be a finite 4x4 matrix')
    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-8,
        rtol=0.0,
    ):
        raise ValueError(f'{label} must have a homogeneous final row')
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-6,
        rtol=0.0,
    ) or not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        abs_tol=1e-6,
        rel_tol=0.0,
    ):
        raise ValueError(f'{label} rotation must be proper and orthonormal')
    return matrix


def named_pose_transform(message: str, name: str) -> np.ndarray:
    """Return one uniquely named transform from a Gazebo pose sample."""

    matches = [
        transform
        for observed_name, transform in parse_pose_transforms(message)
        if observed_name == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f'Gazebo scene must contain exactly one {name!r} pose'
        )
    return matches[0]


def world_scene_from_pose_message(
    message: str,
    *,
    shelf_triangles_local: Sequence[Sequence[Sequence[float]]],
    target_book: str,
    observed_at: float,
    shelf_transform: Sequence[Sequence[float]] | None = None,
) -> WorldScene:
    """Validate and construct the complete 20-book diagnostic world scene."""

    timestamp = float(observed_at)
    if not math.isfinite(timestamp):
        raise ValueError('world-scene timestamp must be finite')
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for name, transform in parse_pose_transforms(message):
        if name.startswith('book_col_') and not BOOK_NAME_PATTERN.fullmatch(
            name
        ):
            raise ValueError(f'malformed or unmodelled Gazebo book {name!r}')
        if (
            name in {ROBOT_MODEL_NAME, SHELF_MODEL_NAME}
            or name in MOVING_GRIPPER_LINKS
            or BOOK_NAME_PATTERN.fullmatch(name)
        ):
            grouped[name].append(transform)

    required_dynamic = (ROBOT_MODEL_NAME,)
    for required in required_dynamic:
        if len(grouped[required]) != 1:
            raise ValueError(
                f'Gazebo scene must contain exactly one {required!r} pose'
            )
    for link in MOVING_GRIPPER_LINKS:
        if len(grouped[link]) != 1:
            raise ValueError(
                f'Gazebo scene must contain exactly one {link!r} pose'
            )
    if shelf_transform is None:
        if len(grouped[SHELF_MODEL_NAME]) != 1:
            raise ValueError(
                f'Gazebo scene must contain exactly one '
                f'{SHELF_MODEL_NAME!r} pose'
            )
        shelf_pose = grouped[SHELF_MODEL_NAME][0]
    else:
        shelf_pose = _validated_rigid_transform(
            shelf_transform,
            label='cached shelf transform',
        )

    book_entries = {
        name: transforms[0]
        for name, transforms in grouped.items()
        if BOOK_NAME_PATTERN.fullmatch(name) and len(transforms) == 1
    }
    duplicate_books = sorted(
        name
        for name, transforms in grouped.items()
        if BOOK_NAME_PATTERN.fullmatch(name) and len(transforms) != 1
    )
    if duplicate_books:
        raise ValueError(f'duplicate Gazebo book poses: {duplicate_books}')

    slots: dict[tuple[int, int], str] = {}
    colours_by_column: dict[int, set[str]] = defaultdict(set)
    for name in book_entries:
        match = BOOK_NAME_PATTERN.fullmatch(name)
        if match is None:
            raise AssertionError('validated book regex unexpectedly failed')
        column, row = int(match.group(1)), int(match.group(2))
        slot = (column, row)
        if slot in slots:
            raise ValueError(
                f'multiple Gazebo books occupy shelf slot {slot}: '
                f'{slots[slot]!r} and {name!r}'
            )
        slots[slot] = name
        colours_by_column[column].add(match.group(3))
    missing_slots = sorted(EXPECTED_BOOK_SLOTS - set(slots))
    unexpected_slots = sorted(set(slots) - EXPECTED_BOOK_SLOTS)
    if missing_slots or unexpected_slots or len(book_entries) != 20:
        raise ValueError(
            'Gazebo scene does not contain the complete 20-book layout; '
            f'missing slots={missing_slots}, unexpected slots={unexpected_slots}'
        )
    for column in range(1, 6):
        if colours_by_column[column] != EXPECTED_COLUMN_COLOURS:
            raise ValueError(
                f'column {column} does not contain one book of every colour'
            )
    if target_book not in book_entries:
        raise ValueError(f'audited target {target_book!r} is absent')

    local_shelf = np.asarray(shelf_triangles_local, dtype=float)
    if (
        local_shelf.ndim != 3
        or local_shelf.shape[1:] != (3, 3)
        or not len(local_shelf)
        or not np.all(np.isfinite(local_shelf))
    ):
        raise ValueError('official shelf mesh must be a finite (N, 3, 3) array')
    shelf_world = (
        local_shelf @ shelf_pose[:3, :3].T
        + shelf_pose[:3, 3]
    )
    books = {
        name: BookOBB.from_pose(
            name,
            transform,
            BOOK_HALF_EXTENTS_M,
            timestamp,
        )
        for name, transform in book_entries.items()
    }
    return WorldScene(
        base_transform=grouped[ROBOT_MODEL_NAME][0],
        shelf_triangles=shelf_world,
        books=books,
        model_link_transforms={
            link: grouped[link][0]
            for link in MOVING_GRIPPER_LINKS
        },
        expected_book_names=frozenset(book_entries),
        observed_at=timestamp,
    )


class MimicGripperKinematics:
    """Mimic-aware branched FK below ``gripper_left_base_link``."""

    def __init__(
        self,
        chains: Mapping[str, URDFChain],
        mimic_rules: Mapping[str, tuple[float, float]],
        aperture_bounds: tuple[float, float],
    ) -> None:
        self._chains = dict(chains)
        self._mimic_rules = dict(mimic_rules)
        self._aperture_bounds = tuple(float(value) for value in aperture_bounds)
        if set(self._chains) != set(LEFT_GRIPPER_COLLISION_LINKS):
            raise ValueError('gripper FK must contain all nine collision links')

    @classmethod
    def from_urdf(cls, urdf_path: str | Path) -> 'MimicGripperKinematics':
        root = ET.parse(str(urdf_path)).getroot()
        joint_elements = {
            element.get('name', ''): element
            for element in root.findall('joint')
        }
        driver = joint_elements.get('gripper_left_finger_joint')
        if driver is None or driver.find('limit') is None:
            raise ValueError('left gripper aperture joint or limits are missing')
        driver_limit = driver.find('limit')
        assert driver_limit is not None
        aperture_bounds = (
            float(driver_limit.get('lower', 'nan')),
            float(driver_limit.get('upper', 'nan')),
        )
        if not np.all(np.isfinite(aperture_bounds)):
            raise ValueError('left gripper aperture limits are invalid')

        chains: dict[str, URDFChain] = {}
        rules: dict[str, tuple[float, float]] = {}
        for link in LEFT_GRIPPER_COLLISION_LINKS:
            chain = URDFChain.from_urdf(
                urdf_path,
                PALM_COLLISION_LINK,
                link,
            )
            chains[link] = chain
            for joint_name in chain.active_names:
                element = joint_elements.get(joint_name)
                mimic = None if element is None else element.find('mimic')
                if mimic is None or mimic.get('joint') != (
                    'gripper_left_finger_joint'
                ):
                    raise ValueError(
                        f'gripper collision joint {joint_name!r} is not driven '
                        'by the left aperture mimic'
                    )
                rules[joint_name] = (
                    float(mimic.get('multiplier', '1.0')),
                    float(mimic.get('offset', '0.0')),
                )
        return cls(chains, rules, aperture_bounds)

    def relative_transforms(
        self,
        aperture_m: float,
    ) -> dict[str, np.ndarray]:
        aperture = float(aperture_m)
        lower, upper = self._aperture_bounds
        if (
            not math.isfinite(aperture)
            or aperture < lower - 1e-9
            or aperture > upper + 1e-9
        ):
            raise ValueError(
                f'left gripper aperture {aperture!r} is outside '
                f'[{lower}, {upper}]'
            )
        transforms: dict[str, np.ndarray] = {}
        for link, chain in self._chains.items():
            positions = []
            for index, joint_name in enumerate(chain.active_names):
                multiplier, offset = self._mimic_rules[joint_name]
                value = multiplier * aperture + offset
                if (
                    value < float(chain.lower[index]) - 1e-9
                    or value > float(chain.upper[index]) + 1e-9
                ):
                    raise ValueError(
                        f'mimic joint {joint_name!r} is outside its limits'
                    )
                positions.append(value)
            transforms[link] = chain.forward(positions)
        return transforms


def densify_joint_samples(
    *,
    model: GripperCollisionModel,
    joint_samples: Sequence[Sequence[float]],
    transform_factory: Callable[[np.ndarray], Mapping[str, np.ndarray]],
    max_vertex_step_m: float = DEFAULT_MAX_VERTEX_STEP_M,
    max_samples: int = MAX_DENSE_SAMPLES,
) -> tuple[DenseJointSample, ...]:
    """Recursively densify until every tool vertex moves at most the limit."""

    step = float(max_vertex_step_m)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError('maximum tool-vertex step must be finite and positive')
    if max_samples < 2:
        raise ValueError('dense-sweep sample cap must be at least two')
    positions = tuple(np.asarray(sample, dtype=float) for sample in joint_samples)
    if not positions:
        raise ValueError('joint sweep contains no samples')
    shape = positions[0].shape
    if len(shape) != 1 or not shape[0]:
        raise ValueError('joint samples must be nonempty vectors')
    if any(sample.shape != shape for sample in positions):
        raise ValueError('joint samples must have one consistent shape')
    if any(not np.all(np.isfinite(sample)) for sample in positions):
        raise ValueError('joint samples must be finite')

    first_transform = transform_factory(positions[0])
    dense = [DenseJointSample(positions[0].copy(), first_transform)]

    def append_segment(
        first_position: np.ndarray,
        first_transforms: Mapping[str, np.ndarray],
        second_position: np.ndarray,
        second_transforms: Mapping[str, np.ndarray],
        depth: int,
    ) -> None:
        displacement = conservative_tool_vertex_displacement(
            model,
            first_transforms,
            second_transforms,
        )
        if displacement <= step + 1e-12:
            dense.append(
                DenseJointSample(second_position.copy(), second_transforms)
            )
            if len(dense) > max_samples:
                raise ValueError('dense-sweep sample cap was exceeded')
            return
        if depth >= 24:
            raise ValueError('tool sweep could not be safely densified')
        midpoint = 0.5 * (first_position + second_position)
        midpoint_transforms = transform_factory(midpoint)
        append_segment(
            first_position,
            first_transforms,
            midpoint,
            midpoint_transforms,
            depth + 1,
        )
        append_segment(
            midpoint,
            midpoint_transforms,
            second_position,
            second_transforms,
            depth + 1,
        )

    for second_position in positions[1:]:
        first = dense[-1]
        second_transforms = transform_factory(second_position)
        append_segment(
            first.positions,
            first.transforms,
            second_position,
            second_transforms,
            0,
        )
    return tuple(dense)


def _event_phase(event: str) -> ProbePhase:
    if event == DEPARTURE_EVENT:
        return ProbePhase.DEPARTURE
    if event in APPROACH_EVENTS:
        return ProbePhase.APPROACH
    if event == TANGENT_APPROACH_EVENT:
        return ProbePhase.TANGENT_APPROACH
    if event == FINAL_TANGENT_EVENT:
        # This entire final sub-millimetre segment is the guarded tangent
        # approach.  Restricting permission to its endpoint is incompatible
        # with a 0.75 mm obstacle envelope and a 0.50 mm sweep step.
        return ProbePhase.FINAL_TANGENT
    if event == LIFT_EVENT:
        return ProbePhase.LIFT
    if event in CAGE_EVENTS:
        return ProbePhase.CAGE
    raise ValueError(f'unrecognized rigid-palm preflight event {event!r}')


def _measured_aperture(node: Any) -> float:
    lock = getattr(node, '_lock', None)
    if lock is None:
        joints = dict(node.joints)
    else:
        with lock:
            joints = dict(node.joints)
    if 'gripper_left_finger_joint' not in joints:
        raise ValueError('measured left gripper aperture is unavailable')
    aperture = float(joints['gripper_left_finger_joint'])
    if not math.isfinite(aperture):
        raise ValueError('measured left gripper aperture is nonfinite')
    return aperture


def _measured_robot_state(node: Any) -> MeasuredRobotState:
    positions = np.asarray(node._measured_left_solution(), dtype=float)
    if positions.ndim != 1 or not len(positions):
        raise ValueError('measured torso/left-arm state must be a vector')
    if not np.all(np.isfinite(positions)):
        raise ValueError('measured torso/left-arm state is nonfinite')
    return MeasuredRobotState(positions, _measured_aperture(node))


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _node_time_seconds(node: Any) -> float:
    nanoseconds = int(node.get_clock().now().nanoseconds)
    if nanoseconds < 0:
        raise ValueError('ROS simulation clock is invalid')
    return nanoseconds * 1e-9


def _validate_fk_snapshot(
    node: Any,
    scene: WorldScene,
    state: MeasuredRobotState,
    gripper_kinematics: MimicGripperKinematics,
) -> None:
    """Prove model/base and mimic FK agree with Gazebo's measured links."""

    arm_links = node.chain.link_transforms(state.positions)
    if PALM_COLLISION_LINK not in arm_links:
        raise ValueError('arm FK omitted the left gripper base link')
    model_palm = arm_links[PALM_COLLISION_LINK]
    relative = gripper_kinematics.relative_transforms(state.aperture_m)
    for link in MOVING_GRIPPER_LINKS:
        predicted = model_palm @ relative[link]
        measured = scene.model_link_transforms[link]
        position_error = float(
            np.linalg.norm(predicted[:3, 3] - measured[:3, 3])
        )
        rotation_error = _rotation_distance(predicted, measured)
        if (
            position_error > FK_POSITION_TOLERANCE_M
            or rotation_error > FK_ROTATION_TOLERANCE_RAD
        ):
            raise ValueError(
                f'Gazebo/URDF FK mismatch for {link}: '
                f'{position_error:.9f} m, {rotation_error:.9f} rad'
            )


def _measured_finger_transforms_relative_to_palm(
    node: Any,
    scene: WorldScene,
    state: MeasuredRobotState,
) -> dict[str, np.ndarray]:
    """Capture the six loaded passive-link poses in the predicted palm frame."""

    arm_links = node.chain.link_transforms(state.positions)
    if PALM_COLLISION_LINK not in arm_links:
        raise ValueError('arm FK omitted the left gripper base link')
    world_palm = _validated_rigid_transform(
        scene.base_transform @ arm_links[PALM_COLLISION_LINK],
        label='predicted world caged-palm transform',
    )
    palm_from_world = np.linalg.inv(world_palm)
    relative: dict[str, np.ndarray] = {}
    for link in MOVING_GRIPPER_LINKS:
        if link not in scene.model_link_transforms:
            raise ValueError(f'Gazebo scene omitted measured finger link {link!r}')
        # Gazebo reports these link poses in the model frame; express both
        # operands in world before deriving the palm-relative calibration.
        measured_world = _validated_rigid_transform(
            scene.base_transform @ scene.model_link_transforms[link],
            label=f'measured world transform for {link}',
        )
        relative[link] = _validated_rigid_transform(
            palm_from_world @ measured_world,
            label=f'measured palm-relative transform for {link}',
        )
    return relative


def measured_finger_relative_vertex_drift(
    model: GripperCollisionModel,
    before: Mapping[str, Sequence[Sequence[float]]],
    after: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[float, str | None]:
    """Return conservative passive-link mesh drift in the palm frame."""

    if set(before) != set(MOVING_GRIPPER_LINKS) or set(after) != set(
        MOVING_GRIPPER_LINKS
    ):
        raise ValueError('measured finger calibration must contain all six links')
    first = {
        link: _validated_rigid_transform(
            before[link],
            label=f'initial measured relative transform for {link}',
        )
        for link in MOVING_GRIPPER_LINKS
    }
    second = {
        link: _validated_rigid_transform(
            after[link],
            label=f'final measured relative transform for {link}',
        )
        for link in MOVING_GRIPPER_LINKS
    }
    maximum = 0.0
    maximum_link: str | None = None
    for mesh in model.meshes:
        if mesh.link not in first:
            continue
        bounds = np.asarray(mesh.bounds, dtype=float)
        center = 0.5 * (bounds[0] + bounds[1])
        half = 0.5 * (bounds[1] - bounds[0])
        corners = center + BOX_SIGNS * half
        first_corners = (
            corners @ first[mesh.link][:3, :3].T
            + first[mesh.link][:3, 3]
        )
        second_corners = (
            corners @ second[mesh.link][:3, :3].T
            + second[mesh.link][:3, 3]
        )
        displacement = float(
            np.max(np.linalg.norm(second_corners - first_corners, axis=1))
        )
        if displacement > maximum:
            maximum = displacement
            maximum_link = mesh.link
    return maximum, maximum_link


def _require_stable_measured_finger_geometry(
    model: GripperCollisionModel,
    before: Mapping[str, Sequence[Sequence[float]]],
    after: Mapping[str, Sequence[Sequence[float]]],
) -> None:
    drift, link = measured_finger_relative_vertex_drift(model, before, after)
    if drift > MEASURED_FINGER_VERTEX_STABILITY_M + 1e-12:
        raise ValueError(
            'loaded palm-relative finger geometry moved during preflight: '
            f'{drift:.9f} m at {link}'
        )


def _require_stable_preflight_inputs(
    before_scene: WorldScene,
    after_scene: WorldScene,
    before_state: MeasuredRobotState,
    after_state: MeasuredRobotState,
) -> None:
    """Reject a safe result if its measured inputs changed during checking."""

    if after_scene.observed_at + 1e-12 < before_scene.observed_at:
        raise ValueError('Gazebo scene timestamps moved backwards')
    joint_drift = float(
        np.max(np.abs(after_state.positions - before_state.positions))
    )
    if joint_drift > STATE_JOINT_STABILITY_RAD:
        raise ValueError(
            f'arm moved {joint_drift:.9f} rad during geometric preflight'
        )
    aperture_drift = abs(after_state.aperture_m - before_state.aperture_m)
    if aperture_drift > STATE_APERTURE_STABILITY_M:
        raise ValueError(
            f'gripper moved {aperture_drift:.9f} m during geometric preflight'
        )
    base_translation = float(
        np.linalg.norm(
            after_scene.base_transform[:3, 3]
            - before_scene.base_transform[:3, 3]
        )
    )
    base_rotation = _rotation_distance(
        before_scene.base_transform,
        after_scene.base_transform,
    )
    if (
        base_translation > SCENE_POSITION_STABILITY_M
        or base_rotation > SCENE_ROTATION_STABILITY_RAD
    ):
        raise ValueError(
            'robot base moved during geometric preflight: '
            f'{base_translation:.9f} m, {base_rotation:.9f} rad'
        )
    if before_scene.expected_book_names != after_scene.expected_book_names:
        raise ValueError('book identities changed during geometric preflight')
    for name in before_scene.expected_book_names:
        displacement = float(
            np.max(
                np.linalg.norm(
                    after_scene.books[name].corners
                    - before_scene.books[name].corners,
                    axis=1,
                )
            )
        )
        if displacement > SCENE_POSITION_STABILITY_M:
            raise ValueError(
                f'book {name!r} moved {displacement:.9f} m during preflight'
            )


def _scene_with_target_translation(
    scene: WorldScene,
    target_book: str,
    translation: Sequence[float] | None,
) -> WorldScene:
    if translation is None:
        return scene
    delta = np.asarray(translation, dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError('virtual target translation must be a finite vector')
    books = dict(scene.books)
    target = books[target_book]
    books[target_book] = BookOBB(
        name=target.name,
        corners=np.asarray(target.corners, dtype=float) + delta,
        observed_at=target.observed_at,
    )
    return WorldScene(
        base_transform=scene.base_transform,
        shelf_triangles=scene.shelf_triangles,
        books=books,
        model_link_transforms=scene.model_link_transforms,
        expected_book_names=scene.expected_book_names,
        observed_at=scene.observed_at,
    )


def _scene_with_rigidly_attached_target(
    scene: WorldScene,
    target_book: str,
    reference_palm: Sequence[Sequence[float]],
    current_palm: Sequence[Sequence[float]],
) -> WorldScene:
    """Move the target OBB by the same world rigid motion as the caged palm."""

    first = _validated_rigid_transform(
        reference_palm,
        label='reference caged-palm transform',
    )
    current = _validated_rigid_transform(
        current_palm,
        label='current caged-palm transform',
    )
    motion = current @ np.linalg.inv(first)
    books = dict(scene.books)
    target = books[target_book]
    corners = np.asarray(target.corners, dtype=float)
    books[target_book] = BookOBB(
        name=target.name,
        corners=corners @ motion[:3, :3].T + motion[:3, 3],
        observed_at=target.observed_at,
    )
    return WorldScene(
        base_transform=scene.base_transform,
        shelf_triangles=scene.shelf_triangles,
        books=books,
        model_link_transforms=scene.model_link_transforms,
        expected_book_names=scene.expected_book_names,
        observed_at=scene.observed_at,
    )


class RigidPalmEnvironmentPreflight:
    """Convert planned arm/gripper motion into the pure collision checker."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        shelf_mesh_path: str | Path,
        package_resolver: Callable[[str], str | Path],
        pose_message_reader: Callable[[], str],
        static_pose_message_reader: Callable[[], str],
        target_book: str,
        config: PreflightConfig = PreflightConfig(),
    ) -> None:
        if not BOOK_NAME_PATTERN.fullmatch(target_book):
            raise ValueError('diagnostic target must be a concrete shelf book')
        self.model = load_left_gripper_collision_model(
            urdf_path,
            package_resolver,
        )
        self.gripper_kinematics = MimicGripperKinematics.from_urdf(urdf_path)
        self.shelf_triangles_local = load_stl_triangles(shelf_mesh_path)
        self.pose_message_reader = pose_message_reader
        self.static_pose_message_reader = static_pose_message_reader
        self.target_book = target_book
        self.config = config
        self._shelf_transform: np.ndarray | None = None

    def _read_scene(self) -> WorldScene:
        if self._shelf_transform is None:
            self._shelf_transform = named_pose_transform(
                self.static_pose_message_reader(),
                SHELF_MODEL_NAME,
            )
        message = self.pose_message_reader()
        observed_at = pose_message_timestamp(message)
        return world_scene_from_pose_message(
            message,
            shelf_triangles_local=self.shelf_triangles_local,
            target_book=self.target_book,
            observed_at=observed_at,
            shelf_transform=self._shelf_transform,
        )

    def _world_transforms(
        self,
        node: Any,
        scene: WorldScene,
        positions: np.ndarray,
        aperture_m: float,
        measured_finger_relative: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        arm_links = node.chain.link_transforms(positions)
        if PALM_COLLISION_LINK not in arm_links:
            raise ValueError(
                'arm FK omitted the left gripper base collision link'
            )
        world_palm = scene.base_transform @ arm_links[PALM_COLLISION_LINK]
        relative = self.gripper_kinematics.relative_transforms(aperture_m)
        if measured_finger_relative is not None:
            if set(measured_finger_relative) != set(MOVING_GRIPPER_LINKS):
                raise ValueError(
                    'measured finger calibration must contain all six links'
                )
            for link in MOVING_GRIPPER_LINKS:
                relative[link] = _validated_rigid_transform(
                    measured_finger_relative[link],
                    label=f'measured palm-relative transform for {link}',
                )
        return {
            link: world_palm @ relative[link]
            for link in LEFT_GRIPPER_COLLISION_LINKS
        }

    def _evaluate_joint_sweep(
        self,
        *,
        node: Any,
        event: str,
        joint_samples: Sequence[Sequence[float]],
        scene: WorldScene,
        aperture_m: float,
        reference_time: float,
        measured_finger_relative: Mapping[str, np.ndarray] | None = None,
    ) -> PreflightResult:
        phase = _event_phase(event)
        if event == CAGED_EXTRACTION_EVENT:
            if measured_finger_relative is None:
                raise ValueError(
                    'caged extraction requires measured passive-link geometry'
                )
        elif measured_finger_relative is not None:
            raise ValueError(
                'measured passive-link geometry is restricted to caged extraction'
            )
        def transforms(positions: np.ndarray) -> Mapping[str, np.ndarray]:
            if measured_finger_relative is None:
                return self._world_transforms(
                    node,
                    scene,
                    positions,
                    aperture_m,
                )
            return self._world_transforms(
                node,
                scene,
                positions,
                aperture_m,
                measured_finger_relative,
            )

        dense = densify_joint_samples(
            model=self.model,
            joint_samples=joint_samples,
            transform_factory=transforms,
            max_vertex_step_m=self.config.max_vertex_step_m,
        )
        for index, sample in enumerate(dense):
            collision = node._robot_self_collision(sample.positions)
            if collision is not None:
                return PreflightResult(
                    False,
                    'dense_robot_self_collision',
                    f'dense sample {index} has self-collision: {collision}',
                    sample_index=index,
                )
        samples = tuple(
            GripperSample(
                transforms=sample.transforms,
                measured_aperture_m=aperture_m,
                expected_aperture_m=(
                    aperture_m
                    if phase is ProbePhase.CAGE
                    else OPEN_APERTURE_M
                ),
                observed_at=scene.observed_at,
                phase=phase,
            )
            for sample in dense
        )
        if event == CAGED_EXTRACTION_EVENT:
            reference_palm = dense[0].transforms[PALM_COLLISION_LINK]
            for index, (dense_sample, sample) in enumerate(zip(dense, samples)):
                attached_scene = _scene_with_rigidly_attached_target(
                    scene,
                    self.target_book,
                    reference_palm,
                    dense_sample.transforms[PALM_COLLISION_LINK],
                )
                result = preflight_gripper_sweep(
                    model=self.model,
                    samples=(sample,),
                    shelf_triangles=attached_scene.shelf_triangles,
                    books=attached_scene.books,
                    target_book=self.target_book,
                    expected_book_names=attached_scene.expected_book_names,
                    reference_time=reference_time,
                    config=self.config,
                )
                if not result.safe:
                    return PreflightResult(
                        False,
                        result.code,
                        result.detail,
                        sample_index=index,
                        link=result.link,
                        obstacle=result.obstacle,
                    )
            return PreflightResult(
                True,
                'clear',
                f'{len(samples)} rigidly attached caged states passed preflight',
            )
        return preflight_gripper_sweep(
            model=self.model,
            samples=samples,
            shelf_triangles=scene.shelf_triangles,
            books=scene.books,
            target_book=self.target_book,
            expected_book_names=scene.expected_book_names,
            reference_time=reference_time,
            config=self.config,
        )

    def __call__(
        self,
        *,
        node: Any,
        event: str,
        joint_samples: Sequence[Sequence[float]],
    ) -> PreflightResult:
        try:
            before_state = _measured_robot_state(node)
            scene = self._read_scene()
            reference_time = _node_time_seconds(node)
            measured_finger_relative = None
            if event == CAGED_EXTRACTION_EVENT:
                measured_finger_relative = (
                    _measured_finger_transforms_relative_to_palm(
                        node,
                        scene,
                        before_state,
                    )
                )
            else:
                _validate_fk_snapshot(
                    node,
                    scene,
                    before_state,
                    self.gripper_kinematics,
                )
            result = self._evaluate_joint_sweep(
                node=node,
                event=event,
                joint_samples=joint_samples,
                scene=scene,
                aperture_m=before_state.aperture_m,
                reference_time=reference_time,
                measured_finger_relative=measured_finger_relative,
            )
            if not result.safe:
                return result
            after_scene = self._read_scene()
            after_state = _measured_robot_state(node)
            if measured_finger_relative is None:
                _validate_fk_snapshot(
                    node,
                    after_scene,
                    after_state,
                    self.gripper_kinematics,
                )
            else:
                after_measured_finger_relative = (
                    _measured_finger_transforms_relative_to_palm(
                        node,
                        after_scene,
                        after_state,
                    )
                )
                _require_stable_measured_finger_geometry(
                    self.model,
                    measured_finger_relative,
                    after_measured_finger_relative,
                )
            _require_stable_preflight_inputs(
                scene,
                after_scene,
                before_state,
                after_state,
            )
            return result
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return PreflightResult(False, 'live_adapter_error', str(error))

    def preflight_aperture_sweep(
        self,
        *,
        node: Any,
        event: str,
        joint_positions: Sequence[float],
        target_aperture_m: float,
        start_aperture_m: float | None = None,
        target_book_translation: Sequence[float] | None = None,
    ) -> PreflightResult:
        """Preflight one stationary-arm cage motion before commanding it."""

        try:
            if event not in CAGE_EVENTS:
                raise ValueError(f'{event!r} is not a cage event')
            before_state = _measured_robot_state(node)
            start_aperture = (
                before_state.aperture_m
                if start_aperture_m is None
                else float(start_aperture_m)
            )
            if not math.isfinite(start_aperture):
                raise ValueError('start cage aperture is nonfinite')
            target_aperture = float(target_aperture_m)
            if not math.isfinite(target_aperture):
                raise ValueError('target cage aperture is nonfinite')
            positions = np.asarray(joint_positions, dtype=float)
            if positions.ndim != 1 or not np.all(np.isfinite(positions)):
                raise ValueError('cage arm state must be a finite vector')
            scene = self._read_scene()
            reference_time = _node_time_seconds(node)
            _validate_fk_snapshot(
                node,
                scene,
                before_state,
                self.gripper_kinematics,
            )
            evaluation_scene = _scene_with_target_translation(
                scene,
                self.target_book,
                target_book_translation,
            )

            def transforms(aperture: float) -> Mapping[str, np.ndarray]:
                return self._world_transforms(
                    node,
                    evaluation_scene,
                    positions,
                    aperture,
                )

            scalar_samples = [start_aperture, target_aperture]
            dense_values: list[tuple[float, Mapping[str, np.ndarray]]] = [
                (start_aperture, transforms(start_aperture))
            ]

            def append_aperture(
                first: float,
                first_transforms: Mapping[str, np.ndarray],
                second: float,
                second_transforms: Mapping[str, np.ndarray],
                depth: int,
            ) -> None:
                displacement = conservative_tool_vertex_displacement(
                    self.model,
                    first_transforms,
                    second_transforms,
                )
                if displacement <= self.config.max_vertex_step_m + 1e-12:
                    dense_values.append((second, second_transforms))
                    if len(dense_values) > MAX_DENSE_SAMPLES:
                        raise ValueError('dense cage sample cap was exceeded')
                    return
                if depth >= 24:
                    raise ValueError('cage sweep could not be safely densified')
                midpoint = 0.5 * (first + second)
                midpoint_transforms = transforms(midpoint)
                append_aperture(
                    first,
                    first_transforms,
                    midpoint,
                    midpoint_transforms,
                    depth + 1,
                )
                append_aperture(
                    midpoint,
                    midpoint_transforms,
                    second,
                    second_transforms,
                    depth + 1,
                )

            for target in scalar_samples[1:]:
                first, first_transforms = dense_values[-1]
                target_transforms = transforms(target)
                append_aperture(
                    first,
                    first_transforms,
                    target,
                    target_transforms,
                    0,
                )

            samples = tuple(
                GripperSample(
                    transforms=sample_transforms,
                    measured_aperture_m=aperture,
                    expected_aperture_m=aperture,
                    observed_at=scene.observed_at,
                    phase=ProbePhase.CAGE,
                )
                for aperture, sample_transforms in dense_values
            )
            result = preflight_gripper_sweep(
                model=self.model,
                samples=samples,
                shelf_triangles=evaluation_scene.shelf_triangles,
                books=evaluation_scene.books,
                target_book=self.target_book,
                expected_book_names=evaluation_scene.expected_book_names,
                reference_time=reference_time,
                config=self.config,
            )
            if not result.safe:
                return result
            after_scene = self._read_scene()
            after_state = _measured_robot_state(node)
            _validate_fk_snapshot(
                node,
                after_scene,
                after_state,
                self.gripper_kinematics,
            )
            _require_stable_preflight_inputs(
                scene,
                after_scene,
                before_state,
                after_state,
            )
            return result
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return PreflightResult(False, 'live_adapter_error', str(error))
