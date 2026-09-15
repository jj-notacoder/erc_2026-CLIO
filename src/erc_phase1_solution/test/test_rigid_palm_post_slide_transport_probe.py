"""Pure tests for the diagnostic-only post-slide transport continuation."""

import math
from pathlib import Path

import numpy as np
import pytest


from erc_phase1_solution.live_rigid_palm_post_slide_transport_probe import (
    APERTURE_ENDPOINT_TOLERANCE_M,
    BASE_FINAL_SHELF_CLEARANCE_M,
    BASE_RETREAT_QUANTUM_M,
    MINIMUM_SHELF_CLEARANCE_M,
    MINIMUM_SUPPORT_DEPTH_MARGIN_M,
    RECAGE_APERTURE_M,
    SupportAssessment,
    TransportObservation,
    _box_triangles,
    arm_transport_step_guard,
    base_retreat_sample_guard,
    capture_attachment_anchor,
    choose_base_retreat_distance,
    finite_palm_support_guard,
    post_slide_cage_guard,
    predict_attached_corners,
    shelf_clearance_m,
    solve_outward_waypoints,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (
    BookSnapshot,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


SIGNS = np.asarray(
    [
        (x, y, z)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ],
    dtype=float,
)


def _rotation_z(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _rotation_y(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=float,
    )


def _transform(position=(0.0, 0.0, 0.0), rotation=None):
    result = np.eye(4)
    result[:3, :3] = np.eye(3) if rotation is None else rotation
    result[:3, 3] = position
    return result


def _ordered_box(center, half=(0.010, 0.010, 0.040), rotation=None):
    basis = (
        np.eye(3)
        if rotation is None
        else np.asarray(rotation, dtype=float)
    )
    return (SIGNS * np.asarray(half, dtype=float)) @ basis.T + np.asarray(
        center, dtype=float
    )


def _snapshot(corners):
    points = np.asarray(corners, dtype=float)
    return BookSnapshot(
        position=np.mean(points, axis=0),
        quaternion=np.asarray([0.0, 0.0, 0.0, 1.0]),
        minimum=np.min(points, axis=0),
        maximum=np.max(points, axis=0),
    )


def _palm_box(depth=0.080, width=0.050, thickness=0.010):
    corners = SIGNS * np.asarray(
        [0.5 * depth, 0.5 * width, 0.5 * thickness]
    )
    return _box_triangles(corners)


def _support(book_x=0.0, book_y=0.0, *, palm=None, transform=None):
    triangles = _palm_box() if palm is None else palm
    palm_world = np.eye(4) if transform is None else transform
    # Bottom face touches the palm's +z support plane at z=5 mm.
    book = _ordered_box((book_x, book_y, 0.045))
    return finite_palm_support_guard(
        triangles,
        palm_world,
        book,
        (-1.0, 0.0, 0.0),
    )


def test_support_polygon_is_derived_from_measured_mesh_and_transform():
    nominal = _support()
    narrower = _support(palm=_palm_box(depth=0.006))
    yawed = _support(transform=_transform(rotation=_rotation_z(0.4)))

    assert nominal.safe
    assert nominal.depth_margin_m == pytest.approx(0.03925, abs=1e-9)
    assert nominal.lateral_margin_m == pytest.approx(0.02425, abs=1e-9)
    assert nominal.support_area_m2 == pytest.approx(0.004, abs=1e-12)
    assert not narrower.safe
    assert narrower.reason == 'support_depth_margin_too_small'
    assert yawed.safe
    assert yawed.support_area_m2 == pytest.approx(nominal.support_area_m2)
    assert yawed.depth_margin_m != pytest.approx(nominal.depth_margin_m)


def test_support_guard_rejects_depth_lateral_and_tilt_failures():
    shallow = _support(book_x=0.038)
    sideways = _support(book_y=0.023)
    tilted = _support(
        transform=_transform(rotation=_rotation_y(math.radians(13)))
    )

    assert not shallow.safe
    assert shallow.reason == 'support_depth_margin_too_small'
    assert shallow.depth_margin_m < MINIMUM_SUPPORT_DEPTH_MARGIN_M
    assert not sideways.safe
    assert sideways.reason == 'support_lateral_margin_too_small'
    assert not tilted.safe
    assert tilted.reason == 'support_surface_too_tilted'


def test_support_guard_rejects_com_projection_outside_finite_pad():
    outside = _support(book_x=0.050)

    assert not outside.safe
    assert outside.reason == 'book_com_projection_outside_palm'


def test_attachment_anchor_preserves_actual_rotated_obb_under_new_hand_pose():
    hand = _transform((0.4, -0.2, 0.8), _rotation_z(0.3))
    book_rotation = _rotation_z(-0.2)
    corners = _ordered_box((0.45, -0.18, 0.85), rotation=book_rotation)
    book = _snapshot(corners)
    # The quaternion is only used to record hand_T_book in this test; set it
    # consistently with the synthetic book rotation.
    book = BookSnapshot(
        book.position,
        np.asarray([0.0, 0.0, math.sin(-0.1), math.cos(-0.1)]),
        book.minimum,
        book.maximum,
    )
    anchor = capture_attachment_anchor(
        book, corners, hand, observed_at=4.2
    )
    moved_hand = _transform((0.3, -0.1, 0.9), _rotation_z(0.6))
    predicted = predict_attached_corners(anchor, moved_hand)
    rigid_motion = moved_hand @ np.linalg.inv(hand)
    expected = corners @ rigid_motion[:3, :3].T + rigid_motion[:3, 3]

    np.testing.assert_allclose(predicted, expected, atol=1e-12)
    np.testing.assert_allclose(
        anchor.book_from_hand @ anchor.hand_from_book, np.eye(4), atol=1e-12
    )


class _CartesianFakeChain:
    """Eight-joint fake whose q[1:4] are Cartesian hand translation."""

    def forward(self, positions):
        q = np.asarray(positions, dtype=float)
        return _transform(q[1:4])

    def solve(self, target, seeds, **_kwargs):
        result = np.asarray(seeds[0], dtype=float).copy()
        result[1:4] = np.asarray(target, dtype=float)[:3, 3]
        return result, 0.0

    def pose_error(self, achieved, target):
        error = np.zeros(6)
        error[:3] = np.asarray(target)[:3, 3] - np.asarray(achieved)[:3, 3]
        return error


def test_dynamic_waypoints_use_measured_state_and_exact_five_mm_steps():
    chain = _CartesianFakeChain()
    start = np.zeros(8)
    book_corners = _ordered_box((2.755, 0.0, 1.0), half=(0.010, 0.010, 0.040))
    anchor = capture_attachment_anchor(
        _snapshot(book_corners), book_corners, np.eye(4), observed_at=1.0
    )

    waypoints = solve_outward_waypoints(chain, start, (0.0, 0.0, 0.0), anchor)

    assert len(waypoints) == 6
    np.testing.assert_allclose(
        [point.outward_distance_m for point in waypoints],
        np.arange(1, 7) * 0.005,
    )
    assert waypoints[-1].predicted_shelf_clearance_m == pytest.approx(0.020)
    assert all(
        point.positions[1] == pytest.approx(-point.outward_distance_m)
        for point in waypoints
    )


def _safe_support():
    return SupportAssessment(
        True,
        'ok',
        0.010,
        0.010,
        0.01075,
        0.01075,
        np.zeros(3),
        np.asarray([0.0, 0.0, 1.0]),
        _ordered_box((0.0, 0.0, 0.0), half=(0.02, 0.02, 0.001))[:4],
        0.0016,
    )


def _observation(
    *,
    hand_x=0.0,
    book_x=2.755,
    book_y=0.0,
    base_x=0.0,
    base_y=0.0,
    base_yaw=0.0,
    aperture=RECAGE_APERTURE_M,
    arm_delta=0.0,
    stamp=1.0,
):
    corners = _ordered_box((book_x, book_y, 1.0))
    hand = _transform((hand_x, 0.0, 0.0))
    arm = np.zeros(8)
    arm[4] = arm_delta
    base = np.asarray([base_x, base_y, base_yaw], dtype=float)
    return TransportObservation(
        book=_snapshot(corners),
        corners=corners,
        base=base,
        arm=arm,
        hand_base=hand.copy(),
        hand_world=hand,
        aperture_m=aperture,
        observed_at=stamp,
    )


def _anchor_from_observation(observation):
    return capture_attachment_anchor(
        observation.book,
        observation.corners,
        observation.hand_world,
        observed_at=observation.observed_at,
    )


def test_dynamic_post_slide_cage_requires_exact_contacts_support_and_30mm():
    observation = _observation(book_x=2.750, stamp=10.0)
    values = dict(
        observation=observation,
        support=_safe_support(),
        reference_time=10.1,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
    )

    assert post_slide_cage_guard(**values).safe
    assert post_slide_cage_guard(
        **{**values, 'left_contact': False}
    ).reason == 'left_target_contact_missing'
    assert post_slide_cage_guard(
        **{**values, 'support': SupportAssessment(
            False, 'support_depth_margin_too_small', 0.0, 0.01,
            0.00075, 0.01075, np.zeros(3), np.asarray([0, 0, 1.0]),
            np.empty((0, 3)), 0.0,
        )}
    ).reason == 'support_depth_margin_too_small'
    opened = _observation(
        book_x=2.750,
        aperture=RECAGE_APERTURE_M + APERTURE_ENDPOINT_TOLERANCE_M + 1e-6,
        stamp=10.0,
    )
    assert post_slide_cage_guard(
        **{**values, 'observation': opened}
    ).reason == 'wrong_cage_aperture'


def _arm_guard(
    reference, previous, current, anchor, support=None, **overrides
):
    values = dict(
        reference=reference,
        previous=previous,
        current=current,
        anchor=anchor,
        support=_safe_support() if support is None else support,
        expected_hand_world=current.hand_world,
        reference_time=current.observed_at + 0.1,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        expected_arm=current.arm,
    )
    values.update(overrides)
    return arm_transport_step_guard(**values)


def test_arm_step_guard_accepts_rigid_five_mm_motion_and_rejects_slip():
    reference = _observation(hand_x=0.0, book_x=2.755)
    anchor = _anchor_from_observation(reference)
    current = _observation(hand_x=-0.005, book_x=2.750, stamp=1.1)
    slipped = _observation(
        hand_x=-0.005, book_x=2.752, book_y=0.003, stamp=1.1
    )

    accepted = _arm_guard(reference, reference, current, anchor)
    rejected = _arm_guard(reference, reference, slipped, anchor)

    assert accepted.safe
    assert accepted.metrics['book_outward_progress_m'] == pytest.approx(0.005)
    assert not rejected.safe
    assert rejected.reason == 'book_did_not_follow_hand'


def test_arm_step_guard_fails_closed_on_contact_base_and_support():
    reference = _observation()
    anchor = _anchor_from_observation(reference)
    current = _observation(hand_x=-0.005, book_x=2.750, stamp=1.1)
    moved_base = _observation(
        hand_x=-0.005, book_x=2.750, base_y=0.001, stamp=1.1
    )
    unsafe_support = SupportAssessment(
        False, 'support_lateral_margin_too_small', 0.01, 0.0,
        0.01075, 0.00075, np.zeros(3), np.asarray([0, 0, 1.0]),
        np.empty((0, 3)), 0.0,
    )

    assert _arm_guard(
        reference, reference, current, anchor, left_contact=False
    ).reason == 'left_target_contact_missing'
    assert _arm_guard(
        reference, reference, moved_base, anchor
    ).reason in {'base_moved_during_arm_step', 'base_moved_during_arm_route'}
    assert _arm_guard(
        reference, reference, current, anchor, support=unsafe_support
    ).reason == 'support_lateral_margin_too_small'


def test_base_retreat_distance_is_minimal_quantized_and_bounded():
    corners = _ordered_box((2.720, 0.0, 1.0))  # max x=2.730, 25 mm clear
    distance = choose_base_retreat_distance(corners, (-1.0, 0.0, 0.0))
    already_buffered = _ordered_box((2.695, 0.0, 1.0))  # exactly 50 mm

    assert shelf_clearance_m(corners) == pytest.approx(0.025)
    assert distance == pytest.approx(0.025)
    assert choose_base_retreat_distance(
        already_buffered, (-1.0, 0.0, 0.0)
    ) == pytest.approx(BASE_RETREAT_QUANTUM_M)
    assert BASE_FINAL_SHELF_CLEARANCE_M > MINIMUM_SHELF_CLEARANCE_M
    with pytest.raises(ValueError, match='geometry'):
        choose_base_retreat_distance(corners, (1.0, 0.0, 0.0))


def _base_motion_observation(reference, distance, *, lateral=0.0, yaw=0.0,
                             arm_delta=0.0, aperture=RECAGE_APERTURE_M,
                             book_extra=(0.0, 0.0, 0.0), stamp=1.1):
    delta = np.asarray((-distance, lateral, 0.0))
    book_delta = delta + np.asarray(book_extra, dtype=float)
    corners = reference.corners + book_delta
    hand = reference.hand_world.copy()
    hand[:3, 3] += delta
    hand_base = reference.hand_base.copy()
    arm = reference.arm.copy()
    arm[3] += arm_delta
    return TransportObservation(
        _snapshot(corners),
        corners,
        np.asarray([
            reference.base[0] - distance,
            reference.base[1] + lateral,
            reference.base[2] + yaw,
        ]),
        arm,
        hand_base,
        hand,
        aperture,
        stamp,
    )


def _base_guard(reference, previous, current, anchor, **overrides):
    values = dict(
        reference=reference,
        previous=previous,
        current=current,
        anchor=anchor,
        support=_safe_support(),
        reference_time=current.observed_at + 0.1,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
    )
    values.update(overrides)
    return base_retreat_sample_guard(**values)


def test_base_retreat_guard_accepts_rigid_straight_motion():
    reference = _observation(hand_x=0.0, book_x=2.720)
    anchor = _anchor_from_observation(reference)
    previous = _base_motion_observation(reference, 0.005, stamp=1.05)
    current = _base_motion_observation(reference, 0.010)

    result = _base_guard(reference, previous, current, anchor)

    assert result.safe
    assert result.metrics['base_outward_travel_m'] == pytest.approx(0.010)
    assert result.metrics['shelf_clearance_m'] == pytest.approx(0.035)


@pytest.mark.parametrize(
    ('changes', 'reason'),
    (
        ({'lateral': 0.0011}, 'base_retreat_not_straight'),
        ({'yaw': 0.0011}, 'base_rotated_during_retreat'),
        ({'arm_delta': 0.0041}, 'arm_moved_during_retreat'),
        (
            {
                'aperture': RECAGE_APERTURE_M
                + APERTURE_ENDPOINT_TOLERANCE_M
                + 1e-6
            },
            'cage_aperture_changed',
        ),
        ({'book_extra': (0.004, 0.0, 0.0)}, 'attachment_position_slip'),
    ),
)
def test_base_retreat_guard_rejects_each_transport_violation(changes, reason):
    reference = _observation(hand_x=0.0, book_x=2.720)
    anchor = _anchor_from_observation(reference)
    previous = _base_motion_observation(reference, 0.005, stamp=1.05)
    current = _base_motion_observation(reference, 0.010, **changes)

    assert _base_guard(reference, previous, current, anchor).reason == reason


def test_probe_source_is_explicitly_diagnostic_and_never_opens():
    source = (
        PACKAGE_ROOT
        / 'erc_phase1_solution'
        / 'live_rigid_palm_post_slide_transport_probe.py'
    ).read_text(encoding='utf-8')

    assert '--confirm-diagnostic-post-slide-transport' in source
    assert 'diagnostic_truth_and_contacts_only=True' in source
    assert 'gripper_opened=False' in source
    assert 'next_motion_authorized=False' in source
    assert '_command_gripper(' not in source
    assert '_publish_aperture(' not in source
