"""Offline tests for the diagnostic continuous-contact tangent slide."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from erc_phase1_solution.kinematics import URDFChain  # noqa: E402
from erc_phase1_solution import (  # noqa: E402
    live_rigid_palm_tangent_slide_probe as tangent_module,
)
from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (  # noqa
    EXPECTED_STEP18_BOOK_MAXIMUM_X_M,
    EXPECTED_STEP18_BOOK_POSITION,
    EXPECTED_STEP18_JOINTS,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (  # noqa: E402
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
)
from erc_phase1_solution.live_rigid_palm_tangent_slide_probe import (  # noqa
    APERTURE_ENDPOINT_TOLERANCE_M,
    BASE_HOLD_LIMIT_M,
    BASE_YAW_HOLD_LIMIT_RAD,
    EVIDENCE_APERTURE_M,
    HAND_ENDPOINT_POSITION_LIMIT_M,
    LOCAL_Y_RECENTER_M,
    LOCAL_Y_STAGE_LIMIT_M,
    LOCAL_Z_INSERTION_M,
    LOCAL_Z_STAGE_LIMIT_M,
    OPEN_APERTURE_M,
    RECAGE_APERTURE_M,
    START_APERTURE_M,
    RESUME_CAGE_ROTATION,
    RouteGeometrySample,
    STABILITY_BOOK_CENTER_SPAN_M,
    StageObservation,
    TangentWaypoint,
    _build_aperture_trajectory,
    _dense_aperture_segment,
    _join_dense_segments,
    aperture_endpoint_is_settled,
    bounded_stage_targets,
    classify_step18_resume,
    endpoint_book_hand_transforms,
    fixed_height_tangent_pose,
    measured_local_y_recenter,
    opening_recovery_guard,
    opening_stage_targets,
    recage_stage_targets,
    solve_tangent_waypoints,
    stage_stability_guard,
    tangent_stage_guard,
)
from erc_phase1_solution.motion_profiles import IK_JOINTS  # noqa: E402
from erc_phase1_solution.rigid_palm_live_preflight import (  # noqa: E402
    MOVING_GRIPPER_LINKS,
    conservative_tool_vertex_displacement,
)
from erc_phase1_solution.rigid_palm_preflight import (  # noqa: E402
    LEFT_GRIPPER_COLLISION_LINKS,
    PALM_COLLISION_LINK,
    PreflightResult,
)


LIVE_REANCHORED_Q18 = np.asarray(
    [
        0.35,
        0.29690028,
        0.54589121,
        0.54535862,
        -1.81096689,
        1.31185176,
        0.84234442,
        -1.41568237,
    ]
)
LIVE_REANCHORED_BOOK_POSITION = np.asarray(
    [2.68586484, -0.14743711, 1.57695550]
)
LIVE_REANCHORED_BOOK_MAXIMUM_X_M = 2.76604393


@pytest.fixture(scope='module')
def official_chain() -> URDFChain:
    urdf = (
        Path(__file__).resolve().parents[2]
        / 'erc_description'
        / 'urdf'
        / 'tiago_pro.urdf'
    )
    if not urdf.exists():
        pytest.skip('official erc_description URDF is unavailable')
    return URDFChain.from_urdf(
        urdf,
        'base_footprint',
        'gripper_left_grasping_link',
        IK_JOINTS,
    )


def _book(
    *,
    position=EXPECTED_STEP18_BOOK_POSITION,
    maximum_x=EXPECTED_STEP18_BOOK_MAXIMUM_X_M,
    quaternion=EXPECTED_RELEASED_BOOK_QUATERNION,
) -> BookSnapshot:
    center = np.asarray(position, dtype=float)
    half = np.asarray([0.0803, 0.0154, 0.1252])
    maximum = center + half
    maximum[0] = float(maximum_x)
    return BookSnapshot(
        center.copy(),
        np.asarray(quaternion, dtype=float).copy(),
        center - half,
        maximum,
    )


def _resume(chain, *, q=EXPECTED_STEP18_JOINTS, book=None, **overrides):
    arm = np.asarray(q, dtype=float)
    values = {
        'book': _book() if book is None else book,
        'base': EXPECTED_RELEASED_BASE_POSE,
        'arm': arm,
        'hand_base': chain.forward(arm),
        'aperture_m': START_APERTURE_M,
        'observed_at': 10.0,
        'reference_time': 10.1,
        'left_contact': True,
        'right_contact': True,
        'palm_contact': True,
        'unexpected_contacts': False,
    }
    values.update(overrides)
    return classify_step18_resume(**values)


def test_resume_accepts_recorded_30mm_cage(official_chain):
    result, nominal = _resume(official_chain)

    assert result.safe
    assert nominal == START_APERTURE_M
    assert result.metrics['cage_center_y_m'] == pytest.approx(
        LOCAL_Y_RECENTER_M, abs=1e-8
    )


def test_resume_accepts_live_reanchored_31mm_palm_support(official_chain):
    book = _book(
        position=LIVE_REANCHORED_BOOK_POSITION,
        maximum_x=LIVE_REANCHORED_BOOK_MAXIMUM_X_M,
    )
    result, nominal = _resume(
        official_chain,
        q=LIVE_REANCHORED_Q18,
        book=book,
        aperture_m=EVIDENCE_APERTURE_M,
        left_contact=False,
        right_contact=False,
    )

    assert result.safe
    assert nominal == EVIDENCE_APERTURE_M
    assert result.metrics['cage_center_z_m'] == pytest.approx(
        0.058977, abs=2e-6
    )
    assert result.metrics['bilateral_contact_required'] is False
    assert np.max(np.abs(LIVE_REANCHORED_Q18 - EXPECTED_STEP18_JOINTS)) > 0.08


@pytest.mark.parametrize(
    ('overrides', 'reason'),
    (
        ({'aperture_m': 0.0305}, 'wrong_aperture'),
        ({'left_contact': False}, 'left_target_contact_missing'),
        ({'right_contact': False}, 'right_target_contact_missing'),
        ({'palm_contact': False}, 'exact_target_palm_contact_missing'),
        ({'unexpected_contacts': True}, 'unexpected_contact'),
        ({'reference_time': 10.251}, 'scene_stale'),
    ),
)
def test_resume_rejects_ambiguous_or_unproven_state(
    official_chain, overrides, reason
):
    result, nominal = _resume(official_chain, **overrides)

    assert not result.safe
    assert result.reason == reason
    assert nominal is None


def test_resume_rejects_wrong_measured_book_hand_geometry(official_chain):
    hand = official_chain.forward(EXPECTED_STEP18_JOINTS).copy()
    hand[:3, 3] += hand[:3, :3] @ np.asarray([0.0, 0.009, 0.0])
    result, nominal = _resume(official_chain, hand_base=hand)

    assert not result.safe
    assert result.reason == 'book_not_centered_between_fingers'
    assert nominal is None


def test_aperture_schedules_are_exact_one_millimetre_stages():
    from_30 = opening_stage_targets(START_APERTURE_M)
    from_31 = opening_stage_targets(EVIDENCE_APERTURE_M)
    closing = recage_stage_targets()

    assert len(from_30) == 17 and from_30[0] == 0.031
    assert len(from_31) == 16 and from_31[0] == 0.032
    assert from_30[-1] == from_31[-1] == OPEN_APERTURE_M
    assert len(closing) == 17
    assert closing[0] == 0.046 and closing[-1] == RECAGE_APERTURE_M
    for route in (from_30, from_31, closing):
        assert all(
            abs(second - first) <= 0.001 + 1e-12
            for first, second in zip(route, route[1:])
        )


def test_aperture_endpoint_requires_tight_quiet_convergence():
    assert not aperture_endpoint_is_settled(0.03170, 0.03175, 0.032)
    assert not aperture_endpoint_is_settled(0.03196, 0.03199, 0.032)
    assert aperture_endpoint_is_settled(0.03198, 0.031995, 0.032)


def test_tangent_scalar_stages_use_full_steps_then_residual():
    y = bounded_stage_targets(
        0.0, LOCAL_Y_RECENTER_M, LOCAL_Y_STAGE_LIMIT_M
    )
    z = bounded_stage_targets(
        0.0, LOCAL_Z_INSERTION_M, LOCAL_Z_STAGE_LIMIT_M
    )

    assert y == pytest.approx((-0.001, -0.002, -0.003, -0.004, -0.004184678))
    assert z == pytest.approx(
        tuple(index * 0.005 for index in range(1, 13)) + (0.061,)
    )
    assert all(
        abs(b - a) <= LOCAL_Y_STAGE_LIMIT_M + 1e-12
        for a, b in zip((0.0, *y), y)
    )
    assert all(
        abs(b - a) <= LOCAL_Z_STAGE_LIMIT_M + 1e-12
        for a, b in zip((0.0, *z), z)
    )


def test_tangent_pose_uses_measured_local_axes_once():
    angle = math.radians(31.0)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    reference = np.eye(4)
    reference[:3, :3] = rotation
    reference[:3, 3] = [0.4, -0.2, 1.5]
    target = fixed_height_tangent_pose(
        reference, LOCAL_Y_RECENTER_M, LOCAL_Z_INSERTION_M
    )
    expected_local = np.asarray(
        [0.0, LOCAL_Y_RECENTER_M, LOCAL_Z_INSERTION_M]
    )

    np.testing.assert_allclose(target[:3, :3], rotation, atol=1e-12)
    np.testing.assert_allclose(
        target[:3, 3] - reference[:3, 3],
        rotation @ expected_local,
        atol=1e-12,
    )
    recovered = rotation.T @ (target[:3, 3] - reference[:3, 3])
    np.testing.assert_allclose(recovered, expected_local, atol=1e-12)


def test_actual_urdf_tangent_sign_and_frame_regression(official_chain):
    reference = official_chain.forward(EXPECTED_STEP18_JOINTS)
    target = fixed_height_tangent_pose(
        reference, LOCAL_Y_RECENTER_M, LOCAL_Z_INSERTION_M
    )
    delta = target[:3, 3] - reference[:3, 3]

    assert delta == pytest.approx(
        [0.0610042864, 0.0041214513, 0.0000470065], abs=2e-9
    )
    yaw = EXPECTED_RELEASED_BASE_POSE[2]
    world_from_base = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert world_from_base @ delta == pytest.approx(
        [0.061000708, 0.004174074, 0.000047006], abs=2e-9
    )
    assert delta[0] > 0.060
    assert delta[1] > 0.004
    assert abs(delta[2]) < 0.000050


def test_dynamic_ik_is_anchored_to_measured_reanchored_q18(official_chain):
    waypoints = solve_tangent_waypoints(
        official_chain, LIVE_REANCHORED_Q18
    )
    start_pose = official_chain.forward(LIVE_REANCHORED_Q18)

    assert len(waypoints) == 18
    assert [waypoint.phase for waypoint in waypoints[:5]] == ['recenter'] * 5
    assert [waypoint.phase for waypoint in waypoints[5:]] == ['insert'] * 13
    assert waypoints[4].local_y_m == pytest.approx(LOCAL_Y_RECENTER_M)
    assert waypoints[-1].local_z_m == pytest.approx(LOCAL_Z_INSERTION_M)
    assert all(
        waypoint.positions[0] == pytest.approx(LIVE_REANCHORED_Q18[0])
        for waypoint in waypoints
    )
    expected_end = fixed_height_tangent_pose(
        start_pose, LOCAL_Y_RECENTER_M, LOCAL_Z_INSERTION_M
    )
    error = official_chain.pose_error(
        official_chain.forward(waypoints[-1].positions), expected_end
    )
    assert np.linalg.norm(error[:3]) <= 0.00020
    assert np.linalg.norm(error[3:]) <= 0.00050


def test_live_recenter_is_measured_from_book_hand_geometry(official_chain):
    book = _book(
        position=LIVE_REANCHORED_BOOK_POSITION,
        maximum_x=LIVE_REANCHORED_BOOK_MAXIMUM_X_M,
    )
    hand = official_chain.forward(LIVE_REANCHORED_Q18)
    recenter = measured_local_y_recenter(
        book,
        EXPECTED_RELEASED_BASE_POSE,
        hand,
    )
    waypoints = solve_tangent_waypoints(
        official_chain,
        LIVE_REANCHORED_Q18,
        local_y_recenter_m=recenter,
    )

    assert recenter == pytest.approx(-0.00078224, abs=2e-6)
    assert len(waypoints) == 14
    assert waypoints[0].phase == 'recenter'
    assert waypoints[0].local_y_m == pytest.approx(recenter)
    assert waypoints[-1].local_y_m == pytest.approx(recenter)


def _observation(
    *,
    delta=(0.0, 0.0, 0.0),
    base_delta=(0.0, 0.0, 0.0),
    arm_delta=0.0,
    hand_delta=(0.0, 0.0, 0.0),
    aperture=RECAGE_APERTURE_M,
    stamp=10.0,
) -> StageObservation:
    hand = np.eye(4)
    hand[:3, 3] = np.asarray(hand_delta, dtype=float)
    return StageObservation(
        book=_book(
            position=EXPECTED_STEP18_BOOK_POSITION + np.asarray(delta),
            maximum_x=(
                EXPECTED_STEP18_BOOK_MAXIMUM_X_M + float(delta[0])
            ),
        ),
        base=EXPECTED_RELEASED_BASE_POSE + np.asarray(base_delta),
        arm=EXPECTED_STEP18_JOINTS + float(arm_delta),
        hand_base=hand,
        aperture_m=float(aperture),
        observed_at=float(stamp),
    )


def _stage_guard(**changes):
    reference = _observation()
    previous = changes.pop('previous', reference)
    current = changes.pop('current', reference)
    return tangent_stage_guard(
        reference,
        previous,
        current,
        expected_arm=changes.pop('expected_arm', reference.arm),
        expected_hand_base=changes.pop('expected_hand', reference.hand_base),
        expected_aperture_m=changes.pop(
            'expected_aperture', reference.aperture_m
        ),
        reference_time=changes.pop('reference_time', 10.1),
        palm_contact=changes.pop('palm', True),
        unexpected_contacts=changes.pop('unexpected', False),
        **changes,
    )


def test_stage_guard_accepts_stationary_palm_supported_endpoint():
    result = _stage_guard()

    assert result.safe
    assert result.reason == 'ok'


@pytest.mark.parametrize(
    ('changes', 'reason'),
    (
        (
            {'current': _observation(delta=(0.000501, 0.0, 0.0))},
            'target_step_motion',
        ),
        (
            {
                'current': _observation(
                    base_delta=(BASE_HOLD_LIMIT_M + 1e-6, 0.0, 0.0)
                )
            },
            'base_moved',
        ),
        (
            {
                'current': _observation(
                    base_delta=(0.0, 0.0, BASE_YAW_HOLD_LIMIT_RAD + 1e-6)
                )
            },
            'base_rotated',
        ),
        (
            {
                'current': _observation(
                    hand_delta=(
                        HAND_ENDPOINT_POSITION_LIMIT_M + 1e-6,
                        0.0,
                        0.0,
                    )
                )
            },
            'hand_endpoint_position_missed',
        ),
        (
            {
                'current': _observation(
                    aperture=(
                        RECAGE_APERTURE_M
                        + APERTURE_ENDPOINT_TOLERANCE_M
                        + 1e-6
                    )
                )
            },
            'aperture_endpoint_missed',
        ),
        ({'palm': False}, 'exact_target_palm_contact_missing'),
        ({'unexpected': True}, 'unexpected_contact'),
    ),
)
def test_stage_guard_fails_at_each_safety_boundary(changes, reason):
    result = _stage_guard(**changes)

    assert not result.safe
    assert result.reason == reason


def test_stage_guard_requires_bilateral_only_when_requested():
    open_result = _stage_guard(
        require_bilateral=False, left_contact=False, right_contact=False
    )
    caged_result = _stage_guard(
        require_bilateral=True, left_contact=False, right_contact=True
    )

    assert open_result.safe
    assert not caged_result.safe
    assert caged_result.reason == 'left_target_contact_missing'


def test_stability_guard_accepts_three_quiet_frames():
    samples = tuple(_observation(stamp=10.0 + 0.03 * i) for i in range(3))

    result = stage_stability_guard(samples)

    assert result.safe
    assert result.metrics['stability_duration_s'] == pytest.approx(0.06)


def test_stability_guard_rejects_subthreshold_window_and_book_creep():
    too_short = stage_stability_guard(
        tuple(_observation(stamp=10.0 + 0.01 * i) for i in range(3))
    )
    moved = stage_stability_guard(
        (
            _observation(stamp=10.0),
            _observation(stamp=10.03),
            _observation(
                delta=(STABILITY_BOOK_CENTER_SPAN_M + 1e-6, 0.0, 0.0),
                stamp=10.06,
            ),
        )
    )

    assert not too_short.safe
    assert too_short.reason == 'stability_window_too_short'
    assert not moved.safe
    assert moved.reason == 'target_not_stable'


def test_opening_recovery_requires_strictly_unchanged_pose():
    reference = _observation(aperture=RECAGE_APERTURE_M)
    accepted = opening_recovery_guard(
        reference,
        _observation(aperture=0.040),
        reference_time=10.1,
        palm_contact=True,
        unexpected_contacts=False,
    )
    moved = opening_recovery_guard(
        reference,
        _observation(delta=(0.000501, 0.0, 0.0), aperture=0.040),
        reference_time=10.1,
        palm_contact=True,
        unexpected_contacts=False,
    )
    stale = opening_recovery_guard(
        reference,
        _observation(aperture=0.040, stamp=9.0),
        reference_time=10.1,
        palm_contact=True,
        unexpected_contacts=False,
    )

    assert accepted.safe
    assert not moved.safe
    assert moved.reason == 'target_step_motion'
    assert not stale.safe
    assert stale.reason == 'scene_stale'


class _FakeGripperKinematics:
    def relative_transforms(self, aperture):
        transforms = {}
        value = float(aperture)
        for index, link in enumerate(LEFT_GRIPPER_COLLISION_LINKS):
            transform = np.eye(4)
            if link in MOVING_GRIPPER_LINKS:
                side = -1.0 if '_left_' in link else 1.0
                angle = side * 4.0 * value
                transform[:3, :3] = [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
                transform[1, 3] = side * (value + index * 0.0001)
            transforms[link] = transform
        return transforms


def _fake_dense_inputs():
    kinematics = _FakeGripperKinematics()
    ideal = kinematics.relative_transforms(RECAGE_APERTURE_M)
    bias = np.eye(4)
    bias[:3, 3] = [0.0002, -0.0001, 0.0003]
    measured = {
        link: bias @ ideal[link] for link in MOVING_GRIPPER_LINKS
    }
    meshes = tuple(
        SimpleNamespace(
            link=link,
            bounds=np.asarray(
                [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]
            ),
        )
        for link in LEFT_GRIPPER_COLLISION_LINKS
    )
    environment = SimpleNamespace(
        gripper_kinematics=kinematics,
        model=SimpleNamespace(meshes=meshes),
        config=SimpleNamespace(max_vertex_step_m=0.00020),
    )
    chain = SimpleNamespace(
        link_transforms=lambda unused: {PALM_COLLISION_LINK: np.eye(4)}
    )
    node = SimpleNamespace(chain=chain)
    scene = SimpleNamespace(base_transform=np.eye(4))
    return environment, node, scene, measured


def test_dense_aperture_route_is_seeded_and_below_mesh_step():
    environment, node, scene, measured = _fake_dense_inputs()
    dense = _dense_aperture_segment(
        environment=environment,
        node=node,
        scene=scene,
        positions=EXPECTED_STEP18_JOINTS,
        measured_start=measured,
        start_aperture_m=RECAGE_APERTURE_M,
        target_aperture_m=OPEN_APERTURE_M,
    )

    assert dense[0].aperture_m == RECAGE_APERTURE_M
    assert dense[-1].aperture_m == OPEN_APERTURE_M
    assert set(dense[0].transforms) == set(LEFT_GRIPPER_COLLISION_LINKS)
    for first, second in zip(dense, dense[1:]):
        displacement = conservative_tool_vertex_displacement(
            environment.model, first.transforms, second.transforms
        )
        assert displacement <= environment.config.max_vertex_step_m + 1e-12


def test_dense_segment_join_removes_only_identical_boundary():
    environment, node, scene, measured = _fake_dense_inputs()
    opening = _dense_aperture_segment(
        environment=environment,
        node=node,
        scene=scene,
        positions=EXPECTED_STEP18_JOINTS,
        measured_start=measured,
        start_aperture_m=RECAGE_APERTURE_M,
        target_aperture_m=EVIDENCE_APERTURE_M,
    )
    joined = _join_dense_segments(opening, opening[-1:])

    assert len(joined) == len(opening)
    assert joined[-1].aperture_m == EVIDENCE_APERTURE_M


def _dense_evaluation_inputs():
    signs = np.asarray(
        [
            (x, y, z)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
    )
    target = tangent_module.TARGET_BOOK_MODEL
    books = {
        target: SimpleNamespace(corners=signs * [0.08, 0.015, 0.125])
    }
    books.update(
        {
            f'book_{index}': SimpleNamespace(
                corners=signs * [0.08, 0.015, 0.125] + [index, 0.0, 0.0]
            )
            for index in range(19)
        }
    )
    scene = SimpleNamespace(
        base_transform=np.eye(4),
        books=books,
        shelf_triangles=np.zeros((1, 3, 3)),
        expected_book_names=frozenset(books),
        observed_at=10.0,
    )
    node = SimpleNamespace(
        carried_book_padding=0.0,
        _robot_self_collision=lambda unused: None,
        _carried_robot_collision=lambda unused_q, unused_book: None,
    )
    environment = SimpleNamespace(
        model=SimpleNamespace(meshes=()),
        config=SimpleNamespace(
            max_vertex_step_m=0.0005,
            collision_padding_m=0.00075,
        ),
    )
    transforms = {
        link: np.eye(4) for link in LEFT_GRIPPER_COLLISION_LINKS
    }
    samples = tuple(
        RouteGeometrySample(
            positions=np.asarray([0.35, index, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            aperture_m=OPEN_APERTURE_M,
            transforms=transforms,
        )
        for index in (0.0, 0.01, 0.02)
    )
    return node, environment, scene, samples


@pytest.mark.parametrize(
    ('fault', 'code'),
    (
        ('self', 'robot_self_collision'),
        ('payload', 'target_robot_collision'),
        ('environment', 'arm_environment_collision'),
        ('tool', 'tool_robot_collision'),
    ),
)
def test_dense_route_checks_every_robot_and_scene_collision(
    monkeypatch, fault, code
):
    node, environment, scene, samples = _dense_evaluation_inputs()
    if fault == 'self':
        node._robot_self_collision = lambda q: (
            ('arm_left_2_link', 'arm_left_4_link')
            if q[1] == pytest.approx(0.01)
            else None
        )
    if fault == 'payload':
        node._carried_robot_collision = lambda q, unused: (
            'arm_left_3_link' if q[1] == pytest.approx(0.01) else None
        )
    monkeypatch.setattr(
        tangent_module,
        '_moving_arm_environment_collision',
        lambda unused_node, unused_scene, q, padding_m: (
            ('arm_left_4_link', 'shelf')
            if fault == 'environment' and q[1] == pytest.approx(0.01)
            else None
        ),
    )
    monkeypatch.setattr(
        tangent_module,
        '_tool_robot_collision',
        lambda unused_node, unused_scene, q, unused_transforms, unused_model: (
            ('gripper_left_inner_finger_left_link', 'arm_left_5_link')
            if fault == 'tool' and q[1] == pytest.approx(0.01)
            else None
        ),
    )
    monkeypatch.setattr(
        tangent_module,
        'preflight_gripper_sweep',
        lambda **unused: PreflightResult(True, 'clear', 'ok'),
    )

    result = tangent_module._evaluate_dense_route(
        node=node,
        environment=environment,
        scene=scene,
        samples=samples,
        reference_time=10.0,
    )

    assert not result.safe
    assert result.code == code
    assert result.sample_index == 1


def test_dense_route_passes_complete_twenty_book_scene_to_gripper(monkeypatch):
    node, environment, scene, samples = _dense_evaluation_inputs()
    captured = {}
    monkeypatch.setattr(
        tangent_module,
        '_moving_arm_environment_collision',
        lambda *unused, **unused_keywords: None,
    )
    monkeypatch.setattr(
        tangent_module,
        '_tool_robot_collision',
        lambda *unused, **unused_keywords: None,
    )

    def accept(**kwargs):
        captured.update(kwargs)
        return PreflightResult(True, 'clear', 'ok')

    monkeypatch.setattr(tangent_module, 'preflight_gripper_sweep', accept)

    result = tangent_module._evaluate_dense_route(
        node=node,
        environment=environment,
        scene=scene,
        samples=samples,
        reference_time=10.0,
    )

    assert result.safe
    assert len(captured['books']) == 20
    assert captured['expected_book_names'] == scene.expected_book_names
    assert captured['target_book'] == tangent_module.TARGET_BOOK_MODEL
    assert all(sample.phase.value == 'cage' for sample in captured['samples'])


def test_endpoint_reanchor_returns_explicit_inverse(official_chain):
    hand = official_chain.forward(LIVE_REANCHORED_Q18)
    book = _book(
        position=LIVE_REANCHORED_BOOK_POSITION,
        maximum_x=LIVE_REANCHORED_BOOK_MAXIMUM_X_M,
    )
    hand_from_book, book_from_hand = endpoint_book_hand_transforms(
        book, EXPECTED_RELEASED_BASE_POSE, hand
    )

    np.testing.assert_allclose(
        hand_from_book @ book_from_hand, np.eye(4), atol=1e-12
    )


def test_trajectory_builder_uses_real_rclpy_duration_message():
    message = _build_aperture_trajectory(OPEN_APERTURE_M, 0.55)

    assert message.joint_names == ['gripper_left_finger_joint']
    assert list(message.points[0].positions) == [OPEN_APERTURE_M]
    assert message.points[0].time_from_start.sec == 0
    assert message.points[0].time_from_start.nanosec == 550_000_000


def test_source_forbids_navigation_and_rigid_target_shortcuts():
    source_path = (
        PACKAGE_ROOT
        / 'erc_phase1_solution'
        / 'live_rigid_palm_tangent_slide_probe.py'
    )
    source = source_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert '_accept_goal' not in called_attributes
    assert '_move_arm_solution' not in called_attributes
    assert '_preflight_leg' not in source
    assert 'CAGED_EXTRACTION_EVENT' not in source
    assert 'builtin_interfaces.msg import Duration' not in source
    assert 'rclpy.duration import Duration' in source
    assert '--confirm-diagnostic-tangent-slide' in source
    assert 'Gazebo entity truth' in source
