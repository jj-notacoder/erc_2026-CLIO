"""Tests for the Gazebo-to-rigid-palm preflight adapter."""

from itertools import pairwise
from pathlib import Path
import sys

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from erc_phase1_solution.kinematics import (  # noqa: E402
    CollisionMesh,
    URDFChain,
)
from erc_phase1_solution.rigid_palm_live_preflight import (  # noqa: E402
    APPROACH_EVENTS,
    CAGED_EXTRACTION_EVENT,
    DEPARTURE_EVENT,
    FINAL_TANGENT_EVENT,
    LIFT_EVENT,
    MEASURED_FINGER_VERTEX_STABILITY_M,
    MOVING_GRIPPER_LINKS,
    MeasuredRobotState,
    RigidPalmEnvironmentPreflight,
    TANGENT_APPROACH_EVENT,
    WorldScene,
    MimicGripperKinematics,
    _event_phase,
    _measured_finger_transforms_relative_to_palm,
    _require_stable_measured_finger_geometry,
    _validate_fk_snapshot,
    conservative_tool_vertex_displacement,
    densify_joint_samples,
    measured_finger_relative_vertex_drift,
    named_pose_transform,
    parse_pose_transforms,
    pose_message_timestamp,
    world_scene_from_pose_message,
)
from erc_phase1_solution.rigid_palm_preflight import (  # noqa: E402
    LEFT_GRIPPER_COLLISION_LINKS,
    PALM_COLLISION_LINK,
    GripperCollisionModel,
    PreflightConfig,
    ProbePhase,
    maximum_tool_vertex_displacement,
)
from erc_phase1_solution.motion_profiles import IK_JOINTS  # noqa: E402


URDF_PATH = (
    PACKAGE_ROOT.parent
    / 'erc_description'
    / 'urdf'
    / 'tiago_pro.urdf'
)
TARGET = 'book_col_3_row_2_red'


def _pose(name, position=(0.0, 0.0, 0.0), quaternion=(0, 0, 0, 1)):
    x, y, z = position
    qx, qy, qz, qw = quaternion
    return f'''pose {{
  name: "{name}"
  id: 1
  position {{
    x: {x}
    y: {y}
    z: {z}
  }}
  orientation {{
    x: {qx}
    y: {qy}
    z: {qz}
    w: {qw}
  }}
}}'''


def _dynamic_scene(*, missing_slot=None, include_shelf=False):
    blocks = [
        'header {\n  stamp {\n    sec: 10\n    nsec: 250000000\n  }\n}',
        _pose('tiago_pro', (2.1, -0.1, 0.0)),
    ]
    blocks.extend(_pose(link) for link in MOVING_GRIPPER_LINKS)
    if include_shelf:
        blocks.append(_pose('erc_shelf', (3.0, 0.0, 1.1)))
    colours = ('red', 'green', 'yellow', 'blue')
    for column in range(1, 6):
        for row, colour in zip(range(2, 6), colours, strict=True):
            if missing_slot == (column, row):
                continue
            name = f'book_col_{column}_row_{row}_{colour}'
            blocks.append(_pose(name, (3.0, float(column), float(row))))
    return '\n'.join(blocks)


def _tiny_model():
    triangle = np.asarray(
        [[[0.0, 0.0, 0.0], [0.0001, 0.0, 0.0], [0.0, 0.0001, 0.0]]]
    )
    bounds = np.asarray(
        [np.min(triangle, axis=(0, 1)), np.max(triangle, axis=(0, 1))]
    )
    return GripperCollisionModel(
        tuple(
            CollisionMesh(link, triangle.copy(), bounds.copy(), False)
            for link in LEFT_GRIPPER_COLLISION_LINKS
        )
    )


def test_pose_parser_extracts_nested_position_and_orientation_blocks():
    message = '\n'.join(
        (
            _pose('tiago_pro', (1.0, 2.0, 3.0)),
            _pose('erc_shelf', (3.0, 0.0, 1.1)),
        )
    )

    poses = dict(parse_pose_transforms(message))

    np.testing.assert_allclose(poses['tiago_pro'][:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        named_pose_transform(message, 'erc_shelf')[:3, 3],
        [3.0, 0.0, 1.1],
    )

    assert pose_message_timestamp(_dynamic_scene()) == pytest.approx(10.25)


def test_pose_parser_uses_latest_frame_when_gz_echo_returns_two_samples():
    stale = _dynamic_scene().replace(
        'sec: 10',
        'sec: 9',
        1,
    ).replace(
        _pose('tiago_pro', (2.1, -0.1, 0.0)),
        _pose('tiago_pro', (1.0, 2.0, 3.0)),
        1,
    )
    latest = _dynamic_scene()
    message = stale + '\n' + latest

    poses = parse_pose_transforms(message)

    assert sum(name == 'tiago_pro' for name, _ in poses) == 1
    np.testing.assert_allclose(
        named_pose_transform(message, 'tiago_pro')[:3, 3],
        [2.1, -0.1, 0.0],
    )
    assert pose_message_timestamp(message) == pytest.approx(10.25)


def test_world_scene_requires_every_one_of_the_20_concrete_book_slots():
    shelf = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    shelf_transform = np.eye(4)
    shelf_transform[:3, 3] = [3.0, 0.0, 1.1]

    scene = world_scene_from_pose_message(
        _dynamic_scene(),
        shelf_triangles_local=shelf,
        target_book=TARGET,
        observed_at=10.0,
        shelf_transform=shelf_transform,
    )

    assert len(scene.books) == 20
    assert scene.expected_book_names == frozenset(scene.books)
    assert TARGET in scene.books
    np.testing.assert_allclose(scene.shelf_triangles[0, 0], [3.0, 0.0, 1.1])

    with pytest.raises(ValueError, match='complete 20-book layout'):
        world_scene_from_pose_message(
            _dynamic_scene(missing_slot=(5, 5)),
            shelf_triangles_local=shelf,
            target_book=TARGET,
            observed_at=10.0,
            shelf_transform=shelf_transform,
        )


def test_world_scene_can_read_shelf_from_all_pose_stream_when_not_cached():
    shelf = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    scene = world_scene_from_pose_message(
        _dynamic_scene(include_shelf=True),
        shelf_triangles_local=shelf,
        target_book=TARGET,
        observed_at=10.0,
    )

    np.testing.assert_allclose(scene.shelf_triangles[0, 0], [3.0, 0.0, 1.1])


def test_world_scene_rejects_a_malformed_unmodelled_book_entity():
    shelf = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    shelf_transform = np.eye(4)

    with pytest.raises(ValueError, match='malformed or unmodelled'):
        world_scene_from_pose_message(
            _dynamic_scene() + '\n' + _pose('book_col_6_row_2_red'),
            shelf_triangles_local=shelf,
            target_book=TARGET,
            observed_at=10.0,
            shelf_transform=shelf_transform,
        )


def test_mimic_fk_covers_all_links_and_matches_audited_apertures():
    kinematics = MimicGripperKinematics.from_urdf(URDF_PATH)

    opened = kinematics.relative_transforms(0.069)
    locked = kinematics.relative_transforms(0.029)

    assert set(opened) == set(LEFT_GRIPPER_COLLISION_LINKS)
    np.testing.assert_allclose(opened[PALM_COLLISION_LINK], np.eye(4))
    assert opened['gripper_left_fingertip_left_link'][1, 3] == pytest.approx(
        -0.041014412,
        abs=2e-8,
    )
    assert locked['gripper_left_fingertip_left_link'][1, 3] == pytest.approx(
        -0.025868227,
        abs=2e-8,
    )
    assert opened['gripper_left_fingertip_right_link'][1, 3] == pytest.approx(
        -opened['gripper_left_fingertip_left_link'][1, 3],
        abs=1e-6,
    )
    with pytest.raises(ValueError, match='outside'):
        kinematics.relative_transforms(0.071)


def test_joint_sweep_is_densified_by_world_tool_vertex_motion():
    model = _tiny_model()

    def transforms(positions):
        transform = np.eye(4)
        transform[0, 3] = float(positions[0])
        return {
            link: transform.copy()
            for link in LEFT_GRIPPER_COLLISION_LINKS
        }

    dense = densify_joint_samples(
        model=model,
        joint_samples=(np.asarray([0.0]), np.asarray([0.0021])),
        transform_factory=transforms,
        max_vertex_step_m=0.0005,
    )

    assert len(dense) > 2
    np.testing.assert_allclose(dense[0].positions, [0.0])
    np.testing.assert_allclose(dense[-1].positions, [0.0021])
    assert all(
        maximum_tool_vertex_displacement(
            model,
            first.transforms,
            second.transforms,
        ) <= 0.0005 + 1e-12
        for first, second in pairwise(dense)
    )
    assert all(
        conservative_tool_vertex_displacement(
            model,
            first.transforms,
            second.transforms,
        ) >= maximum_tool_vertex_displacement(
            model,
            first.transforms,
            second.transforms,
        ) - 1e-12
        for first, second in pairwise(dense)
    )


def test_gazebo_moving_links_validate_the_model_frame_and_mimic_fk():
    kinematics = MimicGripperKinematics.from_urdf(URDF_PATH)
    chain = URDFChain.from_urdf(
        URDF_PATH,
        'base_footprint',
        'gripper_left_grasping_link',
        IK_JOINTS,
    )
    positions = np.zeros(len(IK_JOINTS))
    aperture = 0.029
    model_palm = chain.link_transforms(positions)[PALM_COLLISION_LINK]
    relative = kinematics.relative_transforms(aperture)
    link_transforms = {
        link: model_palm @ relative[link]
        for link in MOVING_GRIPPER_LINKS
    }
    scene = WorldScene(
        base_transform=np.eye(4),
        shelf_triangles=np.ones((1, 3, 3)),
        books={},
        model_link_transforms=link_transforms,
        expected_book_names=frozenset(),
        observed_at=1.0,
    )
    node = type('Node', (), {'chain': chain})()
    state = MeasuredRobotState(positions, aperture)

    _validate_fk_snapshot(node, scene, state, kinematics)

    wrong = dict(link_transforms)
    wrong_link = MOVING_GRIPPER_LINKS[-1]
    wrong[wrong_link] = wrong[wrong_link].copy()
    wrong[wrong_link][0, 3] += 0.00002
    wrong_scene = WorldScene(
        base_transform=scene.base_transform,
        shelf_triangles=scene.shelf_triangles,
        books=scene.books,
        model_link_transforms=wrong,
        expected_book_names=scene.expected_book_names,
        observed_at=scene.observed_at,
    )
    with pytest.raises(ValueError, match='Gazebo/URDF FK mismatch'):
        _validate_fk_snapshot(node, wrong_scene, state, kinematics)


def test_caged_extraction_uses_loaded_measured_finger_geometry():
    kinematics = MimicGripperKinematics.from_urdf(URDF_PATH)
    chain = URDFChain.from_urdf(
        URDF_PATH,
        'base_footprint',
        'gripper_left_grasping_link',
        IK_JOINTS,
    )
    positions = np.zeros(len(IK_JOINTS))
    aperture = 0.030
    model_palm = chain.link_transforms(positions)[PALM_COLLISION_LINK]
    ideal_relative = kinematics.relative_transforms(aperture)
    model_links = {
        link: model_palm @ ideal_relative[link]
        for link in MOVING_GRIPPER_LINKS
    }
    loaded_link = MOVING_GRIPPER_LINKS[-1]
    model_links[loaded_link] = model_links[loaded_link].copy()
    model_links[loaded_link][0, 3] += 0.000392740
    scene = WorldScene(
        base_transform=np.asarray(
            [
                [0.0, -1.0, 0.0, 2.0],
                [1.0, 0.0, 0.0, -0.1],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        shelf_triangles=np.ones((1, 3, 3)),
        books={},
        model_link_transforms=model_links,
        expected_book_names=frozenset(),
        observed_at=1.0,
    )
    node = type('Node', (), {'chain': chain})()
    state = MeasuredRobotState(positions, aperture)

    measured_relative = _measured_finger_transforms_relative_to_palm(
        node,
        scene,
        state,
    )
    adapter = object.__new__(RigidPalmEnvironmentPreflight)
    adapter.gripper_kinematics = kinematics
    world = adapter._world_transforms(
        node,
        scene,
        positions,
        aperture,
        measured_relative,
    )
    world_palm = scene.base_transform @ model_palm

    np.testing.assert_allclose(
        world[loaded_link],
        world_palm @ measured_relative[loaded_link],
    )
    np.testing.assert_allclose(
        world[PALM_COLLISION_LINK],
        world_palm,
    )
    assert not np.allclose(
        world[loaded_link],
        world_palm @ ideal_relative[loaded_link],
    )
    with pytest.raises(ValueError, match='Gazebo/URDF FK mismatch'):
        _validate_fk_snapshot(node, scene, state, kinematics)


def test_loaded_finger_relative_mesh_drift_is_bounded_fail_closed():
    model = _tiny_model()
    before = {link: np.eye(4) for link in MOVING_GRIPPER_LINKS}
    stable = {link: transform.copy() for link, transform in before.items()}
    stable[MOVING_GRIPPER_LINKS[-1]][0, 3] = (
        MEASURED_FINGER_VERTEX_STABILITY_M - 0.000001
    )

    drift, link = measured_finger_relative_vertex_drift(
        model,
        before,
        stable,
    )

    assert drift == pytest.approx(
        MEASURED_FINGER_VERTEX_STABILITY_M - 0.000001
    )
    assert link == MOVING_GRIPPER_LINKS[-1]
    _require_stable_measured_finger_geometry(model, before, stable)

    unstable = {
        name: transform.copy() for name, transform in before.items()
    }
    unstable[MOVING_GRIPPER_LINKS[-1]][0, 3] = (
        MEASURED_FINGER_VERTEX_STABILITY_M + 0.000001
    )
    with pytest.raises(ValueError, match='finger geometry moved'):
        _require_stable_measured_finger_geometry(model, before, unstable)


def test_caged_extraction_refuses_ideal_passive_link_geometry():
    adapter = object.__new__(RigidPalmEnvironmentPreflight)
    adapter.model = _tiny_model()
    adapter.config = PreflightConfig(max_state_age_seconds=1.0)

    with pytest.raises(ValueError, match='requires measured passive-link'):
        adapter._evaluate_joint_sweep(
            node=object(),
            event=CAGED_EXTRACTION_EVENT,
            joint_samples=(np.asarray([0.0]),),
            scene=object(),
            aperture_m=0.030,
            reference_time=1.0,
        )


def test_dense_midpoint_self_collision_rejects_the_sweep():
    model = _tiny_model()
    scene = world_scene_from_pose_message(
        _dynamic_scene(include_shelf=True),
        shelf_triangles_local=(
            np.asarray(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
            )
            + 100.0
        ),
        target_book=TARGET,
        observed_at=10.25,
    )
    adapter = object.__new__(RigidPalmEnvironmentPreflight)
    adapter.model = model
    adapter.config = PreflightConfig(max_state_age_seconds=1.0)

    def transforms(node, current_scene, positions, aperture):
        del node, current_scene, aperture
        transform = np.eye(4)
        transform[0, 3] = float(positions[0])
        return {
            link: transform.copy()
            for link in LEFT_GRIPPER_COLLISION_LINKS
        }

    adapter._world_transforms = transforms

    class Node:
        @staticmethod
        def _robot_self_collision(positions):
            if abs(float(positions[0]) - 0.00105) < 1e-12:
                return ('arm_left_4_link', 'arm_left_6_link')
            return None

    result = adapter._evaluate_joint_sweep(
        node=Node(),
        event=next(iter(APPROACH_EVENTS)),
        joint_samples=(np.asarray([0.0]), np.asarray([0.0021])),
        scene=scene,
        aperture_m=0.069,
        reference_time=10.25,
    )

    assert not result.safe
    assert result.code == 'dense_robot_self_collision'


def test_event_policy_recognizes_only_the_audited_motion_phases():
    assert all(_event_phase(event) is ProbePhase.APPROACH for event in APPROACH_EVENTS)
    assert _event_phase(DEPARTURE_EVENT) is ProbePhase.DEPARTURE
    assert _event_phase(TANGENT_APPROACH_EVENT) is ProbePhase.TANGENT_APPROACH
    assert _event_phase(FINAL_TANGENT_EVENT) is ProbePhase.FINAL_TANGENT
    assert _event_phase(LIFT_EVENT) is ProbePhase.LIFT
    assert _event_phase('cage_preclose') is ProbePhase.CAGE
    with pytest.raises(ValueError, match='unrecognized'):
        _event_phase('unreviewed_motion')
