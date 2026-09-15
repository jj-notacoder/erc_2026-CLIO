from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import GuardResult
from erc_phase1_solution.live_rigid_palm_micro_aperture_probe import (
    APERTURE_REVERSE_LIMIT_M,
    COMMAND_DURATION_S,
    EXPECTED_PARTIAL_TANGENT_BASE,
    EXPECTED_PARTIAL_TANGENT_Q,
    MICRO_APERTURE_EVENT,
    START_APERTURE_M,
    TARGET_APERTURE_M,
    CombinedSupportAssessment,
    MicroObservation,
    _convex_intersection,
    combined_shelf_palm_support_guard,
    micro_state_guard,
    resume_guard,
)
from erc_phase1_solution.live_rigid_palm_natural_settle_probe import WorldSample
from erc_phase1_solution.live_rigid_palm_support_probe import BookSnapshot


def _corners(
    minimum=(2.6263, -0.161, 1.4518),
    maximum=(2.7550, -0.131, 1.6518),
):
    low = np.asarray(minimum, dtype=float)
    high = np.asarray(maximum, dtype=float)
    return np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=float,
    )


def _world(
    *,
    position_delta=(0.0, 0.0, 0.0),
    aperture=START_APERTURE_M,
    base=None,
    arm=None,
    observed_at=1.0,
):
    delta = np.asarray(position_delta, dtype=float)
    corners = _corners() + delta
    position = np.mean(corners, axis=0)
    book = BookSnapshot(
        position,
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )
    return WorldSample(
        book,
        EXPECTED_PARTIAL_TANGENT_BASE.copy() if base is None else np.asarray(base, dtype=float),
        EXPECTED_PARTIAL_TANGENT_Q.copy() if arm is None else np.asarray(arm, dtype=float),
        np.eye(4),
        aperture,
        observed_at,
        corners,
        {'target_book': corners.copy()},
        np.zeros((1, 3, 3)),
    )


def _observation(**kwargs):
    return MicroObservation(_world(**kwargs), np.zeros(7, dtype=float))


def _support(safe=True, reason='ok'):
    hull = np.asarray(
        [[2.626, -0.161], [2.755, -0.161], [2.755, -0.131], [2.626, -0.131]]
    )
    return CombinedSupportAssessment(
        safe,
        reason,
        hull,
        hull[:2],
        {'combined_support_com_inset_slack_m': 0.010},
    )


def _edge(*, distance=0.0001):
    return SimpleNamespace(
        available=True,
        reason='geometry_consistent',
        geometry_consistent=True,
        shelf_plane_z_m=1.4518,
        section_minimum_x_m=2.6263,
        section_maximum_x_m=2.7550,
        section_minimum_y_m=-0.161,
        section_maximum_y_m=-0.131,
        section_edge_distance_m=distance,
        section_shelf_y_overlap_m=0.030,
        metrics={'shelf_edge_section_distance_m': distance},
    )


def _install_support_geometry(monkeypatch, *, distance=0.0001):
    module = 'erc_phase1_solution.live_rigid_palm_micro_aperture_probe'
    monkeypatch.setattr(f'{module}.shelf_edge_proximity_geometry', lambda *args, **kwargs: _edge(distance=distance))
    monkeypatch.setattr(f'{module}.shelf_solid_penetration_m', lambda *args, **kwargs: 0.0001)
    palm = np.asarray(
        [
            [2.620, -0.170, 1.452],
            [2.640, -0.170, 1.452],
            [2.640, -0.120, 1.452],
            [2.620, -0.120, 1.452],
        ]
    )
    monkeypatch.setattr(
        f'{module}._raw_palm_support_geometry',
        lambda *args, **kwargs: SimpleNamespace(polygon_world=palm),
    )


def test_probe_is_exactly_one_quarter_millimetre_left_gripper_leg():
    assert MICRO_APERTURE_EVENT != 'rigid_palm_natural_one_mm_tangent_slip'
    assert TARGET_APERTURE_M - START_APERTURE_M == pytest.approx(0.00025)
    assert 0.75 <= COMMAND_DURATION_S <= 1.0


def test_convex_intersection_returns_only_local_overlap():
    first = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    second = np.asarray([[1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [1.0, 1.0]])
    overlap = _convex_intersection(first, second)
    assert np.min(overlap, axis=0) == pytest.approx([1.0, 0.0])
    assert np.max(overlap, axis=0) == pytest.approx([2.0, 1.0])


def test_combined_support_requires_local_palm_and_inset_com(monkeypatch):
    _install_support_geometry(monkeypatch)
    assessment = combined_shelf_palm_support_guard(
        _corners(),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3, 3)),
        np.eye(4),
        [-1.0, 0.0, 0.0],
        polygon_inset_m=0.001,
    )
    assert assessment.safe, assessment.reason
    assert assessment.metrics['local_palm_feature_vertex_count'] >= 3
    assert assessment.metrics['combined_support_x_span_m'] >= 0.120
    assert assessment.metrics['combined_support_y_span_m'] >= 0.020
    assert assessment.metrics['combined_support_com_original_slack_m'] >= 0.003
    assert assessment.metrics['raw_wrench_support_proven'] == 0.0
    assert assessment.metrics['force_closure_proven'] == 0.0


def test_combined_support_rejects_more_than_quarter_mm_edge_mismatch(monkeypatch):
    _install_support_geometry(monkeypatch, distance=0.000251)
    assessment = combined_shelf_palm_support_guard(
        _corners(),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3, 3)),
        np.eye(4),
        [-1.0, 0.0, 0.0],
        polygon_inset_m=0.001,
    )
    assert not assessment.safe
    assert assessment.reason == 'shelf_edge_mismatch'


def test_resume_requires_checkpoint_contacts_and_combined_support():
    observation = _observation(observed_at=5.0)
    passed = resume_guard(
        observation,
        support=_support(),
        reference_time=5.1,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
    )
    assert passed.safe
    failed = resume_guard(
        observation,
        support=_support(False, 'combined_support_x_span_too_small'),
        reference_time=5.1,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
    )
    assert not failed.safe
    assert failed.reason == 'combined_support_x_span_too_small'


def test_resume_rejects_wrong_base_checkpoint():
    observation = _observation(
        base=EXPECTED_PARTIAL_TANGENT_BASE + [0.0011, 0.0, 0.0],
        observed_at=5.0,
    )
    result = resume_guard(
        observation,
        support=_support(),
        reference_time=5.1,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
    )
    assert not result.safe
    assert result.reason == 'wrong_partial_tangent_base'


def test_endpoint_accepts_0p25mm_progress_with_stationary_scene():
    reference = _observation(aperture=START_APERTURE_M)
    previous = _observation(aperture=START_APERTURE_M + 0.00018, observed_at=1.1)
    endpoint = _observation(aperture=TARGET_APERTURE_M, observed_at=1.2)
    result = micro_state_guard(
        reference,
        previous,
        endpoint,
        support=_support(),
        endpoint=True,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
    )
    assert result.safe, result.reason
    assert result.metrics['aperture_progress_m'] == pytest.approx(0.00025)


def test_motion_guard_rejects_tenth_mm_book_step():
    reference = _observation()
    current = _observation(
        position_delta=(0.000101, 0.0, 0.0),
        aperture=TARGET_APERTURE_M,
        observed_at=1.1,
    )
    result = micro_state_guard(
        reference,
        reference,
        current,
        support=_support(),
        endpoint=True,
    )
    assert not result.safe
    assert result.reason == 'target_step_motion'


def test_motion_guard_rejects_half_mm_corner_motion_even_with_fixed_com():
    reference = _observation()
    shifted = reference.world.corners.copy()
    shifted[0, 2] += 0.000501
    current = MicroObservation(
        replace(
            reference.world,
            corners=shifted,
            aperture_m=TARGET_APERTURE_M,
            observed_at=1.1,
        ),
        reference.right_arm.copy(),
    )
    result = micro_state_guard(
        reference,
        reference,
        current,
        support=_support(),
        endpoint=True,
    )
    assert not result.safe
    assert result.reason == 'target_corner_motion'


def test_motion_guard_rejects_more_than_0p05mm_reversal():
    reference = _observation()
    previous = _observation(aperture=START_APERTURE_M + 0.00020, observed_at=1.1)
    current = _observation(
        aperture=START_APERTURE_M + 0.00020 - APERTURE_REVERSE_LIMIT_M - 1e-6,
        observed_at=1.2,
    )
    result = micro_state_guard(
        reference,
        previous,
        current,
        support=_support(),
        endpoint=False,
    )
    assert not result.safe
    assert result.reason == 'aperture_reversed'


def test_endpoint_requires_fresh_exact_three_point_contacts():
    reference = _observation()
    endpoint = _observation(aperture=TARGET_APERTURE_M, observed_at=1.1)
    result = micro_state_guard(
        reference,
        reference,
        endpoint,
        support=_support(),
        endpoint=True,
        left_contact=True,
        right_contact=False,
        palm_contact=True,
    )
    assert not result.safe
    assert result.reason == 'right_target_contact_missing'


def test_in_flight_watchdog_requires_age_bounded_exact_contacts():
    reference = _observation()
    current = _observation(
        aperture=START_APERTURE_M + 0.0001, observed_at=1.1
    )
    result = micro_state_guard(
        reference,
        reference,
        current,
        support=_support(),
        endpoint=False,
        require_contacts=True,
        left_contact=True,
        right_contact=True,
        palm_contact=False,
    )
    assert not result.safe
    assert result.reason == 'palm_target_contact_missing'


def test_support_failure_is_a_hard_motion_failure():
    reference = _observation()
    current = _observation(aperture=START_APERTURE_M + 0.0001, observed_at=1.1)
    result = micro_state_guard(
        reference,
        reference,
        current,
        support=_support(False, 'target_com_outside_inset_combined_support'),
        endpoint=False,
    )
    assert not result.safe
    assert result.reason == 'target_com_outside_inset_combined_support'
