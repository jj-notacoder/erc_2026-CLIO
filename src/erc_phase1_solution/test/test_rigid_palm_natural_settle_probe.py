"""Pure safety gates for the diagnostic shelf-edge natural-settle probe."""

import math
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from erc_phase1_solution.live_rigid_palm_natural_settle_probe import (  # noqa
    EDGE_REAR_CLEARANCE_X_M,
    EXPECTED_CAGED_APERTURE_M,
    INITIAL_SHELF_CONTACT_TOLERANCE_M,
    NATURAL_SETTLE_MINIMUM_SHELF_OVERLAP_M,
    NATURAL_SETTLE_STEP18_BOOK_MAXIMUM_X_M,
    NATURAL_SETTLE_STEP18_BOOK_POSITION,
    NATURAL_SETTLE_STEP18_JOINTS,
    NON_TARGET_BOOK_MOTION_LIMIT_M,
    POST_SETTLE_CANDIDATE_DISTANCES_M,
    SETTLE_CENTER_SPAN_LIMIT_M,
    SOFT_SETTLE_TRIGGER_REASONS,
    LegWatchResult,
    SettleSample,
    WorldSample,
    _edge_to_settle_trigger,
    _hazard_is_latched,
    _probe_node_types,
    _watch_commanded_leg,
    choose_shortest_clearance_candidate,
    commanded_edge_leg_guard,
    commanded_motion_envelope_guard,
    initial_shelf_support_geometry_guard,
    natural_settle_step18_resume_guard,
    non_target_book_motion_guard,
    padded_book_shelf_clearance_guard,
    settle_attitude_metrics,
    settle_envelope_guard,
    settle_stability_guard,
    settle_trigger_guard,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (  # noqa: E402
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
)


BOX_SIGNS = np.asarray(
    [
        (x, y, z)
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ],
    dtype=float,
)


def _quaternion_y(angle_rad):
    return np.asarray(
        [0.0, math.sin(0.5 * angle_rad), 0.0, math.cos(0.5 * angle_rad)],
        dtype=float,
    )


def _rotation_y(angle_rad):
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=float,
    )


def _snapshot(
    *,
    center=(2.6847, -0.1442, 1.5770),
    angle_rad=0.0,
    maximum_x=None,
):
    position = np.asarray(center, dtype=float)
    rotation = _rotation_y(angle_rad)
    corners = (
        BOX_SIGNS * np.asarray([0.080, 0.015, 0.125])
    ) @ rotation.T + position
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    if maximum_x is not None:
        maximum = maximum.copy()
        maximum[0] = float(maximum_x)
    return BookSnapshot(
        position=position,
        quaternion=_quaternion_y(angle_rad),
        minimum=minimum,
        maximum=maximum,
    ), corners


def _shelf_top(
    *,
    front=2.755,
    back=3.055,
    minimum_y=-0.5,
    maximum_y=0.5,
    z=1.4519,
):
    return np.asarray(
        [
            (
                (front, minimum_y, z),
                (back, minimum_y, z),
                (back, maximum_y, z),
            ),
            (
                (front, minimum_y, z),
                (back, maximum_y, z),
                (front, maximum_y, z),
            ),
        ],
        dtype=float,
    )


def _world_sample(
    *,
    center=(2.6847, -0.1442, 1.5770),
    hand_x=2.62,
    angle_rad=0.0,
    maximum_x=None,
    observed_at=1.0,
    arm_delta=0.0,
    base=(2.1269, -0.0933, 0.00086),
    aperture=EXPECTED_CAGED_APERTURE_M,
    neighbours=None,
):
    book, corners = _snapshot(
        center=center,
        angle_rad=angle_rad,
        maximum_x=maximum_x,
    )
    hand = np.eye(4)
    hand[:3, 3] = [hand_x, -0.1442, 1.50]
    arm = np.zeros(8)
    arm[1] = arm_delta
    all_books = {'book_col_3_row_2_red': corners.copy()}
    if neighbours is not None:
        all_books.update(neighbours)
    return WorldSample(
        book=book,
        base=np.asarray(base, dtype=float),
        arm=arm,
        hand_world=hand,
        aperture_m=float(aperture),
        observed_at=float(observed_at),
        corners=corners,
        all_book_corners=all_books,
        shelf_triangles=_shelf_top(),
    )


def _edge_guard(before, after, **contacts):
    values = {
        'left_contact': True,
        'right_contact': True,
        'palm_contact': True,
        'unexpected_contacts': False,
    }
    values.update(contacts)
    return commanded_edge_leg_guard(before, after, **values)


class _StopAfterIterations:
    def __init__(self, iterations):
        self.iterations = int(iterations)
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > self.iterations


class _WatchNode:
    def __init__(self):
        self.latched = []

    def latch_transfer_watchdog(self, reason):
        self.latched.append(str(reason))


def _natural_settle_resume_book(*, maximum_x=None):
    position = NATURAL_SETTLE_STEP18_BOOK_POSITION.copy()
    rear = (
        NATURAL_SETTLE_STEP18_BOOK_MAXIMUM_X_M
        if maximum_x is None
        else float(maximum_x)
    )
    return BookSnapshot(
        position=position,
        quaternion=EXPECTED_RELEASED_BOOK_QUATERNION.copy(),
        minimum=position - np.asarray([0.0805, 0.0155, 0.1255]),
        maximum=np.asarray(
            [rear, position[1] + 0.0155, position[2] + 0.1255],
            dtype=float,
        ),
    )


def test_natural_settle_resume_accepts_only_newer_step18_checkpoint():
    accepted = natural_settle_step18_resume_guard(
        _natural_settle_resume_book(),
        EXPECTED_RELEASED_BASE_POSE,
        NATURAL_SETTLE_STEP18_JOINTS,
        aperture_m=EXPECTED_CAGED_APERTURE_M,
    )
    old_route_joints = np.asarray(
        [
            0.34999998897070606,
            0.3822254566345725,
            0.5258644813930865,
            0.4355273966236304,
            -1.8212814698501696,
            1.327510835219393,
            0.8002464027363253,
            -1.3830228375779812,
        ],
        dtype=float,
    )
    wrong_arm = natural_settle_step18_resume_guard(
        _natural_settle_resume_book(),
        EXPECTED_RELEASED_BASE_POSE,
        old_route_joints,
        aperture_m=EXPECTED_CAGED_APERTURE_M,
    )
    old_position = np.asarray(
        [2.6846923463259027, -0.14421938761973457, 1.576999694477571],
        dtype=float,
    )
    old_book = BookSnapshot(
        position=old_position,
        quaternion=EXPECTED_RELEASED_BOOK_QUATERNION.copy(),
        minimum=old_position - np.asarray([0.0805, 0.0155, 0.1255]),
        maximum=np.asarray(
            [
                2.7649729506091196,
                old_position[1] + 0.0155,
                old_position[2] + 0.1255,
            ],
            dtype=float,
        ),
    )
    old_route = natural_settle_step18_resume_guard(
        old_book,
        EXPECTED_RELEASED_BASE_POSE,
        old_route_joints,
        aperture_m=EXPECTED_CAGED_APERTURE_M,
    )

    assert accepted.safe
    assert accepted.reason == 'ok'
    assert math.isclose(
        accepted.metrics['shelf_overlap_m'],
        0.005192466654996508,
        abs_tol=1e-12,
    )
    assert not wrong_arm.safe
    assert wrong_arm.reason == 'wrong_step18_arm'
    assert not old_route.safe
    assert old_route.reason == 'wrong_step18_book_position'


def test_natural_settle_resume_keeps_local_five_millimetre_overlap_floor():
    accepted = natural_settle_step18_resume_guard(
        _natural_settle_resume_book(
            maximum_x=(
                2.755 + NATURAL_SETTLE_MINIMUM_SHELF_OVERLAP_M
            )
        ),
        EXPECTED_RELEASED_BASE_POSE,
        NATURAL_SETTLE_STEP18_JOINTS,
        aperture_m=EXPECTED_CAGED_APERTURE_M,
    )
    rejected = natural_settle_step18_resume_guard(
        _natural_settle_resume_book(
            maximum_x=(
                2.755
                + NATURAL_SETTLE_MINIMUM_SHELF_OVERLAP_M
                - 0.00001
            )
        ),
        EXPECTED_RELEASED_BASE_POSE,
        NATURAL_SETTLE_STEP18_JOINTS,
        aperture_m=EXPECTED_CAGED_APERTURE_M,
    )

    assert accepted.safe
    assert not rejected.safe
    assert rejected.reason == 'shelf_overlap_lost'


def test_natural_settle_resume_preserves_aperture_and_base_gates():
    wrong_aperture = natural_settle_step18_resume_guard(
        _natural_settle_resume_book(),
        EXPECTED_RELEASED_BASE_POSE,
        NATURAL_SETTLE_STEP18_JOINTS,
        aperture_m=EXPECTED_CAGED_APERTURE_M + 0.01,
    )
    moved_base = EXPECTED_RELEASED_BASE_POSE.copy()
    moved_base[0] += 0.01
    wrong_base = natural_settle_step18_resume_guard(
        _natural_settle_resume_book(),
        moved_base,
        NATURAL_SETTLE_STEP18_JOINTS,
        aperture_m=EXPECTED_CAGED_APERTURE_M,
    )

    assert not wrong_aperture.safe
    assert wrong_aperture.reason == 'gripper_not_at_proven_cage'
    assert not wrong_base.safe
    assert wrong_base.reason == 'wrong_step18_base'


def test_initial_shelf_support_geometry_accepts_contact_shell_and_overlap():
    _, corners = _snapshot()

    result = initial_shelf_support_geometry_guard(corners, _shelf_top())

    assert result.safe
    assert result.reason == 'ok'
    assert math.isclose(
        result.metrics['projected_shelf_overlap_m'],
        0.0097,
        abs_tol=1e-12,
    )
    assert result.metrics['near_contact_triangle_count'] >= 1.0
    assert math.isclose(
        result.metrics['minimum_support_gap_m'],
        0.0001,
        abs_tol=1e-12,
    )


def test_initial_shelf_support_geometry_requires_five_millimetre_overlap():
    accepted_center_x = 2.755 - 0.080 + 0.005
    _, accepted_corners = _snapshot(
        center=(accepted_center_x, -0.1442, 1.5770)
    )
    _, rejected_corners = _snapshot(
        center=(accepted_center_x - 0.00001, -0.1442, 1.5770)
    )

    accepted = initial_shelf_support_geometry_guard(
        accepted_corners, _shelf_top()
    )
    rejected = initial_shelf_support_geometry_guard(
        rejected_corners, _shelf_top()
    )

    assert accepted.safe
    assert not rejected.safe
    assert rejected.reason == 'projected_shelf_overlap_below_minimum'


def test_initial_shelf_support_geometry_rejects_gap_and_deep_penetration():
    shelf_z = 1.4519
    half_height = 0.125
    excessive_offset = INITIAL_SHELF_CONTACT_TOLERANCE_M + 0.00001
    _, hovering = _snapshot(
        center=(2.6847, -0.1442, shelf_z + half_height + excessive_offset)
    )
    _, penetrating = _snapshot(
        center=(2.6847, -0.1442, shelf_z + half_height - excessive_offset)
    )

    gap = initial_shelf_support_geometry_guard(hovering, _shelf_top())
    penetration = initial_shelf_support_geometry_guard(
        penetrating, _shelf_top()
    )

    assert not gap.safe
    assert gap.reason == 'book_lower_face_not_near_shelf'
    assert not penetration.safe
    assert penetration.reason == 'book_shelf_penetration_exceeds_tolerance'


def test_initial_shelf_support_geometry_requires_upward_facing_surface():
    _, corners = _snapshot()
    downward_winding = _shelf_top()[:, ::-1, :]

    result = initial_shelf_support_geometry_guard(corners, downward_winding)

    assert not result.safe
    assert result.reason == 'upward_shelf_surface_unavailable'


def test_initial_shelf_support_uses_lower_face_not_upper_aabb_edge():
    angle = 0.004
    shelf_front = 2.755
    lower_face_maximum_x = shelf_front + 0.00499
    center_x = (
        lower_face_maximum_x
        - 0.080 * math.cos(angle)
        + 0.125 * math.sin(angle)
    )
    center_z = (
        1.4519
        + 0.0001
        + 0.080 * math.sin(angle)
        + 0.125 * math.cos(angle)
    )
    _, corners = _snapshot(
        center=(center_x, -0.1442, center_z),
        angle_rad=angle,
    )
    assert np.max(corners[:, 0]) - shelf_front > 0.0059

    result = initial_shelf_support_geometry_guard(corners, _shelf_top())

    assert not result.safe
    assert result.reason == 'projected_shelf_overlap_below_minimum'
    assert math.isclose(
        result.metrics['projected_shelf_overlap_m'],
        0.00499,
        abs_tol=1e-12,
    )


def test_initial_shelf_support_rejects_xy_disjoint_top_surface():
    _, corners = _snapshot()

    result = initial_shelf_support_geometry_guard(
        corners,
        _shelf_top(minimum_y=1.0, maximum_y=2.0),
    )

    assert not result.safe
    assert result.reason == 'book_lower_face_has_no_projected_support'


def test_completed_edge_leg_requires_rigid_one_millimetre_following():
    before = _world_sample(center=(2.6847, -0.1442, 1.5770), hand_x=2.62)
    after = _world_sample(center=(2.6837, -0.1442, 1.5770), hand_x=2.619)

    result = _edge_guard(before, after)

    assert result.safe
    assert result.reason == 'ok'
    assert math.isclose(
        result.metrics['book_outward_progress_m'], 0.001, abs_tol=1e-9
    )


def test_completed_edge_leg_rejects_slip_rotation_and_missing_palm():
    before = _world_sample(hand_x=2.62)
    slipped = _world_sample(center=(2.6827, -0.1442, 1.5770), hand_x=2.619)
    rotated = _world_sample(
        center=(2.6837, -0.1442, 1.5770),
        hand_x=2.619,
        angle_rad=math.radians(1.1),
    )

    assert _edge_guard(before, slipped).reason == 'book_progress_out_of_bounds'
    assert _edge_guard(before, rotated).reason == 'nonrigid_rotation_started'
    assert _edge_guard(before, _world_sample(
        center=(2.6837, -0.1442, 1.5770), hand_x=2.619
    ), palm_contact=False).reason == 'palm_contact_missing'


def test_settle_trigger_accepts_only_near_edge_or_clearance_crossing():
    step18 = _world_sample(maximum_x=2.765)
    before = _world_sample(maximum_x=2.758, hand_x=2.62)
    near_edge = _world_sample(
        center=(2.660, -0.1442, 1.557),
        hand_x=2.619,
        angle_rad=math.radians(22.0),
        maximum_x=2.7565,
    )
    premature = _world_sample(
        center=(2.660, -0.1442, 1.557),
        hand_x=2.619,
        angle_rad=math.radians(22.0),
        maximum_x=2.7580,
    )
    rigid_clear = _world_sample(
        center=(2.6837, -0.1442, 1.5770),
        hand_x=2.619,
        maximum_x=EDGE_REAR_CLEARANCE_X_M - 1e-5,
    )
    contact = dict(
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
    )

    accepted = settle_trigger_guard(step18.book, before, near_edge, **contact)
    rejected = settle_trigger_guard(step18.book, before, premature, **contact)
    cleared = settle_trigger_guard(step18.book, before, rigid_clear, **contact)

    assert accepted.safe
    assert accepted.reason == 'nonrigid_motion_started'
    assert not rejected.safe
    assert rejected.reason == 'premature_settle'
    assert cleared.safe
    assert cleared.reason == 'rear_clearance_reached'


def _watch_samples():
    first = _world_sample(
        center=(2.6600, -0.1442, 1.5570),
        hand_x=2.6190,
        angle_rad=math.radians(22.0),
        maximum_x=2.7565,
        observed_at=2.0,
    )
    second = _world_sample(
        center=(2.6599, -0.1442, 1.5569),
        hand_x=2.6189,
        angle_rad=math.radians(22.1),
        maximum_x=2.7564,
        observed_at=3.0,
    )
    return first, second


def _run_watch(samples, contacts):
    module = (
        'erc_phase1_solution.live_rigid_palm_natural_settle_probe'
    )
    step18 = _world_sample(maximum_x=2.765, observed_at=0.0)
    before = _world_sample(
        maximum_x=2.758, hand_x=2.620, observed_at=1.0
    )
    node = _WatchNode()
    result = {'reason': None, 'sample': None, 'metrics': {}}
    with (
        patch(f'{module}._scene_sample', side_effect=samples),
        patch(f'{module}._exact_contacts', side_effect=contacts),
        patch(f'{module}._hazard_is_latched', return_value=False),
    ):
        _watch_commanded_leg(
            node,
            SimpleNamespace(environment_preflight=object()),
            step18,
            before,
            _StopAfterIterations(len(samples)),
            result,
            allow_settle_trigger=True,
        )
    return node, result


def test_expected_settle_trigger_is_recorded_without_cancelling_leg():
    first, second = _watch_samples()

    node, result = _run_watch(
        [first, second],
        [(True, True, True), (True, True, True)],
    )

    assert result['reason'] in SOFT_SETTLE_TRIGGER_REASONS
    assert result['reason'] == 'nonrigid_motion_started'
    assert result['sample'] is first
    assert node.latched == []


def test_rigid_rear_clearance_trigger_also_does_not_cancel_leg():
    cleared = _world_sample(
        center=(2.6837, -0.1442, 1.5770),
        hand_x=2.6190,
        maximum_x=EDGE_REAR_CLEARANCE_X_M - 0.00001,
        observed_at=2.0,
    )

    node, result = _run_watch([cleared], [(True, True, True)])

    assert result['reason'] == 'rear_clearance_reached'
    assert result['sample'] is cleared
    assert node.latched == []


def test_hard_fault_after_soft_trigger_overrides_and_latches():
    first, second = _watch_samples()

    node, result = _run_watch(
        [first, second],
        [(True, True, True), (True, True, False)],
    )

    assert result['reason'] == 'palm_contact_missing'
    assert result['sample'] is second
    assert node.latched == ['palm_contact_missing']


def test_cancelled_leg_cannot_be_accepted_from_recorded_soft_trigger():
    module = (
        'erc_phase1_solution.live_rigid_palm_natural_settle_probe'
    )
    step18 = _world_sample(maximum_x=2.765, observed_at=1.0)
    watch = LegWatchResult(
        reason='nonrigid_motion_started',
        sample=step18,
        metrics={},
    )
    with (
        patch(f'{module}._solve_translation', return_value=np.zeros(8)),
        patch(f'{module}._preflight_translation'),
        patch(
            f'{module}._run_arm_leg_with_watchdog',
            return_value=(False, True, watch),
        ),
    ):
        try:
            _edge_to_settle_trigger(object(), SimpleNamespace(), step18)
        except RuntimeError as exc:
            assert 'stopped unsafely' in str(exc)
        else:
            raise AssertionError('a cancelled soft-trigger leg was accepted')


def test_controller_hazard_callback_latches_only_actual_faults():
    class BaseProbe:
        def __init__(self):
            self.inherited_reason = None

        def _with_probe_lock(self, callback):
            return callback()

        def _payload_hazard_reason(self, *, max_age=0.20):
            del max_age
            return self.inherited_reason

    module = (
        'erc_phase1_solution.live_rigid_palm_natural_settle_probe'
    )
    with patch(f'{module}._probe_types', return_value=(BaseProbe, object)):
        NaturalSettleProbeNode, _ = _probe_node_types(SimpleNamespace())
    node = NaturalSettleProbeNode()
    node.begin_transfer_watchdog()
    with patch(f'{module}._exact_contacts', return_value=(True, True, True)):
        assert node._payload_hazard_reason() is None
    assert node.probe_transfer_watchdog_reason is None

    node.inherited_reason = 'contact_lost'
    assert node._payload_hazard_reason() == 'contact_lost'
    assert node.end_transfer_watchdog() == 'contact_lost'

    node.inherited_reason = None
    node.begin_transfer_watchdog()

    def latch_during_contact_read(*args, **kwargs):
        del args, kwargs
        node.latch_transfer_watchdog('late_hard_fault')
        return True, True, True

    with patch(f'{module}._exact_contacts', side_effect=latch_during_contact_read):
        assert node._payload_hazard_reason() == 'late_hard_fault'
    assert node.end_transfer_watchdog() == 'late_hard_fault'

    node.begin_transfer_watchdog()
    with patch(f'{module}._exact_contacts', return_value=(True, True, False)):
        assert node._payload_hazard_reason() == (
            'exact_three_point_contact_lost'
        )
    assert node.end_transfer_watchdog() == 'exact_three_point_contact_lost'

    with patch(f'{module}._unexpected_pairs', return_value=()):
        assert _hazard_is_latched(node)


def test_settle_envelope_caps_drop_tilt_and_drift():
    reference = _world_sample(maximum_x=2.765)
    settled = _world_sample(
        center=(2.660, -0.1440, 1.550),
        angle_rad=math.radians(25.0),
        maximum_x=2.757,
    )

    accepted = settle_envelope_guard(
        reference,
        np.zeros(8),
        settled,
        unexpected_contacts=False,
    )
    excessive_drop = _world_sample(
        center=(2.660, -0.1440, 1.516),
        angle_rad=math.radians(25.0),
    )
    dropped = settle_envelope_guard(
        reference,
        np.zeros(8),
        excessive_drop,
        unexpected_contacts=False,
    )

    assert accepted.safe
    assert dropped.reason == 'settle_drop_limit'


def test_commanded_envelope_allows_arm_motion_but_not_base_or_aperture():
    reference = _world_sample()
    moving_arm = _world_sample(arm_delta=0.05)
    moved_base = _world_sample(base=(2.1274, -0.0933, 0.00086))
    opened = _world_sample(aperture=EXPECTED_CAGED_APERTURE_M + 0.0011)

    assert commanded_motion_envelope_guard(
        reference,
        moving_arm,
        unexpected_contacts=False,
    ).safe
    assert commanded_motion_envelope_guard(
        reference,
        moved_base,
        unexpected_contacts=False,
    ).reason == 'base_moved_during_transfer'
    assert commanded_motion_envelope_guard(
        reference,
        opened,
        unexpected_contacts=False,
    ).reason == 'aperture_moved_during_transfer'


def _settle_observation(stamp, *, x_jitter=0.0, angle_deg=25.0):
    world = _world_sample(
        center=(2.660 + x_jitter, -0.1440, 1.550),
        angle_rad=math.radians(angle_deg),
        observed_at=stamp,
    )
    return SettleSample(
        observed_at=stamp,
        book=world.book,
        corners=world.corners,
        arm=world.arm,
        base=world.base,
        aperture_m=world.aperture_m,
    )


def test_stability_guard_requires_half_second_and_tight_spans():
    stable = [
        _settle_observation(1.0, x_jitter=0.0),
        _settle_observation(1.25, x_jitter=0.00004),
        _settle_observation(1.50, x_jitter=0.00002),
    ]
    too_short = stable[:2]
    moving = [
        stable[0],
        stable[1],
        _settle_observation(
            1.50, x_jitter=SETTLE_CENTER_SPAN_LIMIT_M + 0.00001
        ),
    ]

    assert settle_stability_guard(stable).safe
    assert settle_stability_guard(too_short).reason == (
        'insufficient_stability_samples'
    )
    assert settle_stability_guard(moving).reason == 'center_not_stable'


def test_attitude_metrics_separate_radial_tilt_from_sideways_tilt():
    reference = _world_sample(base=(2.1269, -0.0933, 0.0))
    radial = _world_sample(
        angle_rad=math.radians(25.0), base=(2.1269, -0.0933, 0.0)
    )
    radial_metrics = settle_attitude_metrics(
        reference.book, radial.book, reference.base
    )

    assert math.isclose(
        radial_metrics['tilt_rad'], math.radians(25.0), abs_tol=1e-9
    )
    assert radial_metrics['off_axis_tilt_rad'] < 1e-9


def test_non_target_guard_rejects_half_millimetre_excess():
    _, neighbour = _snapshot(center=(2.8, 0.1, 1.5))
    reference = {
        'book_col_3_row_2_red': np.zeros((8, 3)),
        'book_col_1_row_1_blue': neighbour,
    }
    accepted = {name: corners.copy() for name, corners in reference.items()}
    moved = {name: corners.copy() for name, corners in reference.items()}
    moved['book_col_1_row_1_blue'] += np.asarray(
        [NON_TARGET_BOOK_MOTION_LIMIT_M + 1e-5, 0.0, 0.0]
    )

    assert non_target_book_motion_guard(reference, accepted).safe
    assert non_target_book_motion_guard(reference, moved).reason == (
        'non_target_book_moved'
    )


def _small_box(center_x):
    return BOX_SIGNS * np.asarray([0.001, 0.001, 0.001]) + np.asarray(
        [center_x, 0.0, 0.0]
    )


def _shelf_front_plane():
    return np.asarray(
        [
            [[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0]],
            [[0.0, -1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]],
        ],
        dtype=float,
    )


def test_padded_clearance_requires_geometric_and_rear_edge_clearance():
    shelf = _shelf_front_plane()
    clear = padded_book_shelf_clearance_guard(
        _small_box(-0.002), shelf, shelf_front_x_m=0.0
    )
    too_close = padded_book_shelf_clearance_guard(
        _small_box(-0.0015), shelf, shelf_front_x_m=0.0
    )

    assert clear.safe
    assert too_close.reason == 'rear_clearance_below_padding'


def test_clearance_selector_chooses_only_shortest_5_or_10_mm_candidate():
    shelf = _shelf_front_plane()
    five, ten = POST_SETTLE_CANDIDATE_DISTANCES_M
    distance, result = choose_shortest_clearance_candidate(
        {
            five: _small_box(-0.0015),
            ten: _small_box(-0.0020),
        },
        shelf,
        shelf_front_x_m=0.0,
    )
    shortest, _ = choose_shortest_clearance_candidate(
        {
            five: _small_box(-0.0020),
            ten: _small_box(-0.0030),
        },
        shelf,
        shelf_front_x_m=0.0,
    )

    assert result.safe
    assert math.isclose(distance, ten)
    assert math.isclose(shortest, five)


def test_clearance_selector_rejects_any_other_candidate_set():
    try:
        choose_shortest_clearance_candidate(
            {0.005: _small_box(-0.002)}, _shelf_front_plane(),
            shelf_front_x_m=0.0,
        )
    except ValueError:
        return
    raise AssertionError(
        'candidate set outside exact 5/10 mm policy must fail'
    )
