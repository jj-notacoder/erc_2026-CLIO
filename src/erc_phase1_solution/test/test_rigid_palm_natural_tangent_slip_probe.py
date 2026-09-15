"""Pure tests for the one-command natural-cage tangent diagnostic."""

import inspect
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import erc_phase1_solution.live_rigid_palm_natural_tangent_slip_probe as probe
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    GuardResult,
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
)
from erc_phase1_solution.live_rigid_palm_natural_settle_probe import (
    EXPECTED_CAGED_APERTURE_M,
    WorldSample,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    BOOK_HALF_EXTENTS_M,
    EXPECTED_RELEASED_BASE_POSE,
    BookSnapshot,
    quaternion_matrix,
    world_hand_pose,
)
from erc_phase1_solution.kinematics import URDFChain
from erc_phase1_solution.motion_profiles import IK_JOINTS
from erc_phase1_solution.rigid_palm_live_preflight import MeasuredRobotState
from erc_phase1_solution.rigid_palm_live_preflight import WorldScene
from erc_phase1_solution.rigid_palm_preflight import (
    PALM_COLLISION_LINK,
    PreflightResult,
)


SIGNS = np.asarray(
    [
        (x, y, z)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ],
    dtype=float,
)


def _settled_quaternion(angle_deg=25.9):
    # Preserve the actual q6 shelf-lip pose as the baseline, then apply test
    # deltas in the same radial plane.  Its apparent natural-angle metric also
    # contains the small authored released-pose offset.
    measured_absolute = math.degrees(2.0 * math.atan2(0.843917, 0.536473))
    half = 0.5 * math.radians(measured_absolute + angle_deg - 25.9)
    return np.asarray([0.0, math.sin(half), 0.0, math.cos(half)])


def _scene_geometry(*, center_shift=(0.0, 0.0, 0.0), angle_deg=25.9):
    quaternion = _settled_quaternion(angle_deg)
    rotation = quaternion_matrix(quaternion)
    center = np.asarray([2.71248, -0.14634, 1.54962], dtype=float)
    center += np.asarray(center_shift, dtype=float)
    corners = (SIGNS * BOOK_HALF_EXTENTS_M) @ rotation.T + center
    shelf_z = 1.451858
    shelf = np.asarray(
        [
            [
                [SHELF_FRONT_X_M, -0.4, shelf_z],
                [SHELF_FRONT_X_M + 0.3, -0.4, shelf_z],
                [SHELF_FRONT_X_M + 0.3, 0.4, shelf_z],
            ],
            [
                [SHELF_FRONT_X_M, -0.4, shelf_z],
                [SHELF_FRONT_X_M + 0.3, 0.4, shelf_z],
                [SHELF_FRONT_X_M, 0.4, shelf_z],
            ],
        ],
        dtype=float,
    )
    book = BookSnapshot(
        center.copy(),
        quaternion,
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )
    return book, corners, shelf


def _world_sample(
    *,
    arm=None,
    hand_base=None,
    center_shift=(0.0, 0.0, 0.0),
    angle_deg=25.9,
    base_shift=(0.0, 0.0, 0.0),
    aperture=EXPECTED_CAGED_APERTURE_M,
    stamp=10.0,
):
    book, corners, shelf = _scene_geometry(
        center_shift=center_shift, angle_deg=angle_deg
    )
    base = EXPECTED_RELEASED_BASE_POSE + np.asarray(base_shift, dtype=float)
    q = probe.EXPECTED_Q6.copy() if arm is None else np.asarray(arm, dtype=float)
    hand = np.eye(4) if hand_base is None else np.asarray(hand_base, dtype=float)
    return WorldSample(
        book=book,
        base=base,
        arm=q,
        hand_world=world_hand_pose(base, hand),
        aperture_m=float(aperture),
        observed_at=float(stamp),
        corners=corners,
        all_book_corners={TARGET_BOOK_MODEL: corners.copy()},
        shelf_triangles=shelf,
    )


def _recovery_observation(**kwargs):
    return probe.RecoveryObservation(
        _world_sample(**kwargs), np.zeros(7, dtype=float)
    )


def _safe_geometry():
    return GuardResult(True, 'ok', {})


def test_measured_book_positive_z_is_the_audited_world_tangent():
    sample = _world_sample()

    direction = probe.measured_book_positive_z(sample.book)

    expected = probe.EXPECTED_TANGENT_WORLD_DELTA_M.copy()
    expected /= np.linalg.norm(expected)
    assert np.dot(direction, expected) > math.cos(math.radians(1.0))
    assert direction[0] > 0.0
    assert direction[2] < 0.0


def test_shelf_lip_guard_requires_full_measured_section_at_front():
    sample = _world_sample()

    accepted = probe.shelf_lip_guard(
        sample.corners, sample.shelf_triangles
    )
    shifted = probe.shelf_lip_guard(
        sample.corners + [0.00060, 0.0, 0.0],
        sample.shelf_triangles,
    )

    assert accepted.safe
    assert math.isclose(
        accepted.metrics['shelf_edge_section_x_span_m'],
        probe.SHELF_SECTION_EXPECTED_X_SPAN_M,
        abs_tol=probe.SHELF_SECTION_X_SPAN_TOLERANCE_M,
    )
    assert accepted.metrics['shelf_edge_section_y_span_m'] == 0.03
    assert not shifted.safe
    assert shifted.reason in {'edge_not_near_section', 'shelf_edge_mismatch'}


def test_shelf_solid_penetration_is_signed_and_fail_closed():
    sample = _world_sample()
    accepted = probe.shelf_lip_guard(
        sample.corners, sample.shelf_triangles
    )
    # Lowering the entire OBB while holding its x footprint creates a true
    # below-top volume behind the shelf front.
    lowered = probe.shelf_solid_penetration_m(
        sample.corners - [0.0, 0.0, 0.001],
        shelf_plane_z_m=accepted.metrics['shelf_edge_plane_z_m'],
    )

    assert accepted.metrics['shelf_solid_penetration_m'] < 0.0001
    assert lowered > probe.SHELF_SOLID_PENETRATION_LIMIT_M


def test_finite_palm_clip_ignores_remote_tilted_corners_but_catches_local_depth():
    angle = math.radians(45.0)
    rotation = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    center = np.asarray([0.0, 0.0, math.sqrt(2.0) * 0.01])
    corners = (SIGNS * [0.10, 0.02, 0.01]) @ rotation.T + center
    # Only a 0.4 mm strip of the slanted book lies over this finite pad.  The
    # remote OBB corner is 63 mm behind the plane but cannot touch the pad.
    polygon = np.asarray(
        [
            [-0.0002, -0.015, 0.0],
            [0.0002, -0.015, 0.0],
            [0.0002, 0.015, 0.0],
            [-0.0002, 0.015, 0.0],
        ]
    )

    minimum, maximum, count = probe.clipped_palm_book_signed_interval(
        corners,
        polygon,
        [0.0, 0.0, 1.0],
        0.0,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        polygon_inset_m=0.0,
    )
    lowered_minimum, _, _ = probe.clipped_palm_book_signed_interval(
        corners - [0.0, 0.0, 0.002],
        polygon,
        [0.0, 0.0, 1.0],
        0.0,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        polygon_inset_m=0.0,
    )

    assert np.min(corners[:, 2]) < -0.060
    assert math.isclose(minimum, -0.0002, abs_tol=1e-9)
    assert maximum > 0.020
    assert count >= 4
    assert -lowered_minimum > probe.PALM_FACE_PENETRATION_LIMIT_M


def test_pressure_guard_uses_clipped_local_penetration_not_global_corner():
    angle = math.radians(45.0)
    rotation = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    center = np.asarray([0.0, 0.0, math.sqrt(2.0) * 0.01])
    corners = (SIGNS * [0.10, 0.02, 0.01]) @ rotation.T + center
    polygon = np.asarray(
        [
            [-0.0002, -0.015, 0.0],
            [0.0002, -0.015, 0.0],
            [0.0002, 0.015, 0.0],
            [-0.0002, 0.015, 0.0],
        ]
    )

    def assessment_for(book):
        return SimpleNamespace(
            safe=False,
            metrics={},
            raw_geometry=SimpleNamespace(
                polygon_world=polygon,
                support_normal_world=np.asarray([0.0, 0.0, 1.0]),
                plane_offset_m=0.0,
                depth_world=np.asarray([1.0, 0.0, 0.0]),
                lateral_world=np.asarray([0.0, 1.0, 0.0]),
                book_minimum_signed_plane_distance_m=float(
                    np.min(np.asarray(book)[:, 2])
                ),
            ),
        )

    with patch.object(
        probe,
        'palm_bearing_support_guard',
        side_effect=lambda *args, **kwargs: assessment_for(args[2]),
    ):
        accepted = probe.palm_pressure_interface_guard(
            np.zeros((1, 3, 3)),
            np.eye(4),
            corners,
            [1.0, 0.0, 0.0],
            polygon_inset_m=0.0,
        )
        rejected = probe.palm_pressure_interface_guard(
            np.zeros((1, 3, 3)),
            np.eye(4),
            corners - [0.0, 0.0, 0.002],
            [1.0, 0.0, 0.0],
            polygon_inset_m=0.0,
        )

    assert accepted.safe
    assert accepted.metrics[
        'global_obb_minimum_signed_m_diagnostic_only'
    ] < -0.060
    assert accepted.metrics['palm_pressure_face_penetration_m'] < 0.001
    assert not rejected.safe
    assert rejected.reason == 'book_palm_penetration_exceeds_limit'


def test_q6_resume_requires_exact_cage_tilt_aperture_and_base():
    sample = _world_sample()
    shelf = probe.shelf_lip_guard(
        sample.corners, sample.shelf_triangles
    )

    accepted = probe.recovery_checkpoint_guard(
        sample,
        reference_time=sample.observed_at,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=shelf,
    )
    missing_right = probe.recovery_checkpoint_guard(
        sample,
        reference_time=sample.observed_at,
        left_contact=True,
        right_contact=False,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=shelf,
    )
    wrong_aperture = probe.recovery_checkpoint_guard(
        _world_sample(aperture=EXPECTED_CAGED_APERTURE_M + 0.00036),
        reference_time=sample.observed_at,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=shelf,
    )
    # The live natural-settle endpoint is 1.87 mm / 1.57 mrad from the older
    # released checkpoint.  Initial recognition allows that known passive
    # drift; the live watchdog still holds the freshly captured base to the
    # much tighter cumulative and yaw limits.
    current_live_drift = probe.recovery_checkpoint_guard(
        _world_sample(base_shift=(0.00187, 0.0, 0.00157)),
        reference_time=sample.observed_at,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=shelf,
    )
    outside_checkpoint = probe.recovery_checkpoint_guard(
        _world_sample(base_shift=(0.00501, 0.0, 0.0)),
        reference_time=sample.observed_at,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=shelf,
    )

    assert accepted.safe
    assert current_live_drift.safe
    assert not outside_checkpoint.safe
    assert outside_checkpoint.reason == 'wrong_base_checkpoint'
    assert not missing_right.safe
    assert missing_right.reason == 'right_target_contact_missing'
    assert not wrong_aperture.safe
    assert wrong_aperture.reason == 'wrong_cage_aperture'


def test_q6_resume_rejects_each_missing_contact_and_stale_scene():
    sample = _world_sample()
    shelf = probe.shelf_lip_guard(sample.corners, sample.shelf_triangles)
    for contacts, reason in (
        ((False, True, True), 'left_target_contact_missing'),
        ((True, False, True), 'right_target_contact_missing'),
        ((True, True, False), 'palm_target_contact_missing'),
    ):
        result = probe.recovery_checkpoint_guard(
            sample,
            reference_time=sample.observed_at,
            left_contact=contacts[0],
            right_contact=contacts[1],
            palm_contact=contacts[2],
            unexpected_contacts=False,
            shelf=shelf,
        )
        assert not result.safe
        assert result.reason == reason
    stale = probe.recovery_checkpoint_guard(
        sample,
        reference_time=sample.observed_at + 0.251,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=shelf,
    )
    assert not stale.safe
    assert stale.reason == 'scene_stale'


def test_official_chain_dynamic_solution_stays_on_audited_tangent_branch():
    source_root = __import__('pathlib').Path(__file__).resolve().parents[2]
    chain = URDFChain.from_urdf(
        source_root / 'erc_description' / 'urdf' / 'tiago_pro.urdf',
        'base_footprint',
        'gripper_left_grasping_link',
        IK_JOINTS,
    )
    node = type('Node', (), {'chain': chain})()
    sample = _world_sample()

    waypoint = probe.solve_book_tangent_waypoint(node, sample)

    assert np.max(np.abs(
        waypoint.positions - probe.EXPECTED_TANGENT_TARGET_Q
    )) < 0.00005
    assert np.max(np.abs(waypoint.positions - sample.arm)) < 0.003
    assert np.dot(
        waypoint.expected_delta_world, waypoint.book_axis_world
    ) >= probe.TANGENT_HAND_PROGRESS_RANGE_M[0]
    assert np.linalg.norm(
        waypoint.expected_delta_world
        - np.dot(
            waypoint.expected_delta_world, waypoint.book_axis_world
        ) * waypoint.book_axis_world
    ) <= probe.TANGENT_HAND_OFF_AXIS_LIMIT_M


def test_fixed_book_preflight_keeps_closed_aperture_and_stability_gate():
    sample = _world_sample()
    book_obb = SimpleNamespace(corners=sample.corners.copy())
    scene = SimpleNamespace(
        base_transform=np.eye(4),
        shelf_triangles=sample.shelf_triangles,
        books={TARGET_BOOK_MODEL: book_obb},
    )
    environment = SimpleNamespace(
        model=SimpleNamespace(),
        config=SimpleNamespace(collision_padding_m=0.0001),
        _read_scene=lambda: scene,
    )
    state = MeasuredRobotState(
        probe.EXPECTED_Q6.copy(), EXPECTED_CAGED_APERTURE_M
    )
    dense = (
        SimpleNamespace(
            positions=probe.EXPECTED_Q6.copy(),
            aperture_m=EXPECTED_CAGED_APERTURE_M,
            transforms={PALM_COLLISION_LINK: np.eye(4)},
        ),
        SimpleNamespace(
            positions=probe.EXPECTED_TANGENT_TARGET_Q.copy(),
            aperture_m=EXPECTED_CAGED_APERTURE_M,
            transforms={PALM_COLLISION_LINK: np.eye(4)},
        ),
    )
    checked = {'stable': False}

    def evaluate(**kwargs):
        assert kwargs['scene'].books[TARGET_BOOK_MODEL].corners is book_obb.corners
        assert all(
            item.aperture_m == EXPECTED_CAGED_APERTURE_M
            for item in kwargs['samples']
        )
        return PreflightResult(True, 'clear', 'fixed target passed')

    def stable(*unused):
        checked['stable'] = True

    safe_interface = GuardResult(
        True,
        'ok',
        {
            'palm_pressure_face_gap_m': 0.0,
            'palm_pressure_face_penetration_m': 0.0,
        },
    )
    with patch.multiple(
        probe,
        _measured_robot_state=lambda unused: state,
        _measured_finger_transforms_relative_to_palm=lambda *unused: {},
        _dense_arm_segment=lambda **unused: dense,
        _evaluate_dense_route=evaluate,
        _node_time_seconds=lambda unused: 10.0,
        _palm_local_triangles=lambda unused: np.zeros((1, 3, 3)),
        shelf_lip_guard=lambda *unused: _safe_geometry(),
        palm_pressure_interface_guard=lambda *args, **kwargs: safe_interface,
        _require_stable_measured_finger_geometry=lambda *unused: None,
        _require_stable_fixed_target_preflight_inputs=stable,
    ):
        result = probe.preflight_fixed_book_tangent(
            node=SimpleNamespace(),
            environment=environment,
            expected_start_arm=probe.EXPECTED_Q6,
            target_arm=probe.EXPECTED_TANGENT_TARGET_Q,
        )

    assert result.safe
    assert result.fixed_target_hypothesis_count == 1
    assert result.route_sample_count == 2
    assert checked['stable']


def test_fixed_book_preflight_fails_if_inputs_change_during_geometry():
    sample = _world_sample()
    scene = SimpleNamespace(
        base_transform=np.eye(4),
        shelf_triangles=sample.shelf_triangles,
        books={TARGET_BOOK_MODEL: SimpleNamespace(corners=sample.corners)},
    )
    environment = SimpleNamespace(
        model=SimpleNamespace(),
        config=SimpleNamespace(collision_padding_m=0.0001),
        _read_scene=lambda: scene,
    )
    state = MeasuredRobotState(
        probe.EXPECTED_Q6.copy(), EXPECTED_CAGED_APERTURE_M
    )
    dense = (
        SimpleNamespace(
            positions=probe.EXPECTED_Q6.copy(),
            aperture_m=EXPECTED_CAGED_APERTURE_M,
            transforms={PALM_COLLISION_LINK: np.eye(4)},
        ),
    )
    safe_interface = GuardResult(
        True,
        'ok',
        {
            'palm_pressure_face_gap_m': 0.0,
            'palm_pressure_face_penetration_m': 0.0,
        },
    )
    with patch.multiple(
        probe,
        _measured_robot_state=lambda unused: state,
        _measured_finger_transforms_relative_to_palm=lambda *unused: {},
        _dense_arm_segment=lambda **unused: dense,
        _evaluate_dense_route=lambda **unused: PreflightResult(
            True, 'clear', 'fixed target passed'
        ),
        _node_time_seconds=lambda unused: 10.0,
        _palm_local_triangles=lambda unused: np.zeros((1, 3, 3)),
        shelf_lip_guard=lambda *unused: _safe_geometry(),
        palm_pressure_interface_guard=lambda *args, **kwargs: safe_interface,
        _require_stable_measured_finger_geometry=lambda *unused: None,
        _require_stable_fixed_target_preflight_inputs=lambda *unused: (
            _ for _ in ()
        ).throw(
            ValueError('book moved 0.000665000 m during preflight')
        ),
    ):
        result = probe.preflight_fixed_book_tangent(
            node=SimpleNamespace(),
            environment=environment,
            expected_start_arm=probe.EXPECTED_Q6,
            target_arm=probe.EXPECTED_TANGENT_TARGET_Q,
        )

    assert not result.safe
    assert 'book moved 0.000665000 m' in result.reason


def test_fixed_target_input_stability_allows_98um_but_not_201um():
    sample = _world_sample()
    state = MeasuredRobotState(
        probe.EXPECTED_Q6.copy(), EXPECTED_CAGED_APERTURE_M
    )

    def scene(target_shift=0.0, neighbour_shift=0.0, stamp=10.0):
        target = SimpleNamespace(
            corners=sample.corners + [target_shift, 0.0, 0.0]
        )
        neighbour = SimpleNamespace(
            corners=sample.corners + [0.5 + neighbour_shift, 0.0, 0.0]
        )
        return WorldScene(
            base_transform=np.eye(4),
            shelf_triangles=sample.shelf_triangles,
            books={TARGET_BOOK_MODEL: target, 'neighbour': neighbour},
            model_link_transforms={},
            expected_book_names=frozenset((TARGET_BOOK_MODEL, 'neighbour')),
            observed_at=stamp,
        )

    before = scene()
    probe._require_stable_fixed_target_preflight_inputs(
        before, scene(target_shift=0.000098, stamp=10.1), state, state
    )
    try:
        probe._require_stable_fixed_target_preflight_inputs(
            before, scene(target_shift=0.000201, stamp=10.1), state, state
        )
    except ValueError as exc:
        assert 'target book moved' in str(exc)
    else:
        raise AssertionError('target drift above 0.20 mm was accepted')
    try:
        probe._require_stable_fixed_target_preflight_inputs(
            before, scene(neighbour_shift=0.000081, stamp=10.1), state, state
        )
    except ValueError as exc:
        assert "book 'neighbour' moved" in str(exc)
    else:
        raise AssertionError('non-target drift above 80 um was accepted')


def test_fixed_book_watch_guard_enforces_quarter_mm_and_quarter_degree():
    reference = _recovery_observation()
    accepted = probe.static_book_motion_guard(
        reference,
        reference,
        _recovery_observation(center_shift=(0.00024, 0.0, 0.0)),
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=_safe_geometry(),
        palm_interface=_safe_geometry(),
        reference_time=10.0,
    )
    translated = probe.static_book_motion_guard(
        reference,
        reference,
        _recovery_observation(center_shift=(0.000251, 0.0, 0.0)),
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=_safe_geometry(),
        palm_interface=_safe_geometry(),
        reference_time=10.0,
    )
    rotated = probe.static_book_motion_guard(
        reference,
        reference,
        _recovery_observation(angle_deg=26.151),
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=_safe_geometry(),
        palm_interface=_safe_geometry(),
        reference_time=10.0,
    )

    assert accepted.safe
    assert not translated.safe
    assert translated.reason == 'fixed_book_translation_exceeded'
    assert not rotated.safe
    assert rotated.reason == 'fixed_book_rotation_exceeded'


def test_watch_guard_enforces_base_step_and_idle_right_arm():
    reference = _recovery_observation()
    base_motion = probe.static_book_motion_guard(
        reference,
        reference,
        _recovery_observation(base_shift=(0.000351, 0.0, 0.0)),
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=_safe_geometry(),
        palm_interface=_safe_geometry(),
        reference_time=10.0,
    )
    right = probe.RecoveryObservation(
        _world_sample(), np.asarray([0.0021, 0, 0, 0, 0, 0, 0], dtype=float)
    )
    right_motion = probe.static_book_motion_guard(
        reference,
        reference,
        right,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=_safe_geometry(),
        palm_interface=_safe_geometry(),
        reference_time=10.0,
    )

    assert not base_motion.safe
    assert base_motion.reason == 'base_step_exceeded'
    assert not right_motion.safe
    assert right_motion.reason == 'right_arm_moved'


def test_watch_guard_rejects_cumulative_base_creep_and_stale_frame():
    reference = _recovery_observation()
    previous = _recovery_observation(base_shift=(0.00030, 0.0, 0.0))
    cumulative = probe.static_book_motion_guard(
        reference,
        previous,
        _recovery_observation(base_shift=(0.00076, 0.0, 0.0)),
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=_safe_geometry(),
        palm_interface=_safe_geometry(),
        reference_time=10.0,
    )
    stale = probe.static_book_motion_guard(
        reference,
        reference,
        reference,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=_safe_geometry(),
        palm_interface=_safe_geometry(),
        reference_time=10.251,
    )

    assert not cumulative.safe
    # The last jump also exceeds the step cap, which is intentionally checked
    # before cumulative drift; either bound prevents dispatch continuation.
    assert cumulative.reason in {
        'base_step_exceeded', 'base_cumulative_motion_exceeded'
    }
    assert not stale.safe
    assert stale.reason == 'scene_stale'


def test_endpoint_requires_one_mm_hand_motion_but_static_book():
    start_hand = np.eye(4)
    reference = _recovery_observation(hand_base=start_hand)
    axis = probe.measured_book_positive_z(reference.world.book)
    target_hand = start_hand.copy()
    # Convert the world tangent into base coordinates at the measured yaw.
    yaw = reference.world.base[2]
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target_hand[:3, 3] = rotation.T @ (axis * 0.001)
    final = _recovery_observation(
        arm=probe.EXPECTED_TANGENT_TARGET_Q,
        hand_base=target_hand,
    )
    waypoint = probe.TangentWaypoint(
        probe.EXPECTED_TANGENT_TARGET_Q.copy(),
        target_hand,
        axis,
        axis * 0.001,
        start_hand[:3, :3].T @ rotation.T @ (axis * 0.001),
    )

    result = probe.endpoint_guard(
        reference,
        final,
        waypoint,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        shelf=_safe_geometry(),
        palm_interface=_safe_geometry(),
        reference_time=10.0,
    )

    assert result.safe
    assert math.isclose(
        result.metrics['hand_book_tangent_progress_m'], 0.001, abs_tol=1e-12
    )
    assert result.metrics['book_translation_m'] == 0.0


def test_module_has_one_left_arm_goal_and_no_other_motion_command_path():
    module_source = inspect.getsource(probe)
    run_source = inspect.getsource(probe._run)
    command_source = inspect.getsource(probe._run_one_tangent_leg)
    watcher_source = inspect.getsource(probe._watch_one_tangent_leg)

    assert command_source.count('_make_retained_arm_trajectory_goal') == 1
    assert module_source.count('._send_retained_arm_trajectory(') == 1
    assert 'legs = ((waypoint.positions' in command_source
    assert 'SOFT_' not in watcher_source
    assert '_publish_zero' not in run_source
    assert 'gripper_pub.publish' not in run_source
    assert '_move_arm_solution' not in module_source
    assert 'ProbeNavigation(' not in module_source
    assert 'right_arm' not in command_source
    assert 'next_motion_authorized=False' in run_source
    assert 'gravity_support_authorized=False' in run_source
