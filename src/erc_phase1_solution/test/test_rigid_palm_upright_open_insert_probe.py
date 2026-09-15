"""Focused offline tests for the isolated upright open/insert probe."""

from dataclasses import replace
import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import erc_phase1_solution.live_rigid_palm_upright_open_insert_probe as probe
from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (
    SHELF_FRONT_X_M,
    TARGET_BOOK_MODEL,
)
from erc_phase1_solution.live_rigid_palm_natural_settle_probe import WorldSample
from erc_phase1_solution.live_rigid_palm_support_probe import (
    EXPECTED_RELEASED_BOOK_QUATERNION,
    BookSnapshot,
    world_hand_pose,
)


def _corners(*, shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    low = np.asarray([2.605, -0.160, 1.452], dtype=float)
    high = np.asarray([2.765, -0.130, 1.702], dtype=float)
    result = np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=float,
    )
    return result + np.asarray(shift, dtype=float)


def _book(*, shift=(0.0, 0.0, 0.0)) -> BookSnapshot:
    corners = _corners(shift=shift)
    return BookSnapshot(
        np.mean(corners, axis=0),
        EXPECTED_RELEASED_BOOK_QUATERNION.copy(),
        np.min(corners, axis=0),
        np.max(corners, axis=0),
    )


def _world(
    *,
    hand_shift=(0.0, 0.0, 0.0),
    book_shift=(0.0, 0.0, 0.0),
    arm=None,
    base=(2.126, -0.091, 0.0),
    aperture=probe.OPEN_APERTURE_M,
    stamp=1.0,
) -> WorldSample:
    corners = _corners(shift=book_shift)
    hand_base = np.eye(4, dtype=float)
    hand_base[:3, 3] = np.asarray(hand_shift, dtype=float)
    base_pose = np.asarray(base, dtype=float)
    return WorldSample(
        book=_book(shift=book_shift),
        base=base_pose,
        arm=(
            np.zeros(8, dtype=float)
            if arm is None
            else np.asarray(arm, dtype=float)
        ),
        hand_world=world_hand_pose(base_pose, hand_base),
        aperture_m=float(aperture),
        observed_at=float(stamp),
        corners=corners,
        all_book_corners={TARGET_BOOK_MODEL: corners.copy()},
        shelf_triangles=np.zeros((1, 3, 3), dtype=float),
    )


def _observation(*, right=None, **kwargs) -> probe.ProbeObservation:
    return probe.ProbeObservation(
        _world(**kwargs),
        (
            np.zeros(7, dtype=float)
            if right is None
            else np.asarray(right, dtype=float)
        ),
    )


def _support(
    safe=True, reason='ok'
) -> probe.UprightCombinedSupportAssessment:
    hull = np.asarray(
        [
            [2.605, -0.160],
            [SHELF_FRONT_X_M, -0.160],
            [SHELF_FRONT_X_M, -0.130],
            [2.605, -0.130],
        ],
        dtype=float,
    )
    return probe.UprightCombinedSupportAssessment(
        safe,
        reason,
        hull,
        hull[:2],
        {'combined_support_com_inset_slack_m': 0.010},
    )


def _waypoint(command=(0.0, 0.0, probe.INSERT_LOCAL_Z_M)):
    target = np.eye(4, dtype=float)
    target[:3, 3] = np.asarray(command, dtype=float)
    return probe.ReanchoredWaypoint(
        phase='single_0p25mm_local_z_insert',
        start_positions=np.zeros(8, dtype=float),
        positions=np.zeros(8, dtype=float),
        start_hand_base=np.eye(4, dtype=float),
        target_hand_base=target,
        command_local_m=np.asarray(command, dtype=float),
    )


def _edge(*, overlap=0.010):
    return SimpleNamespace(
        available=True,
        reason='geometry_consistent',
        geometry_consistent=True,
        shelf_plane_z_m=1.452,
        section_minimum_x_m=2.605,
        section_maximum_x_m=2.765,
        section_minimum_y_m=-0.160,
        section_maximum_y_m=-0.130,
        section_rear_signed_overlap_m=overlap,
        section_edge_distance_m=abs(overlap),
        section_shelf_y_overlap_m=0.030,
        metrics={
            'shelf_edge_section_rear_signed_overlap_m': overlap,
            'shelf_edge_support_geometry_consistent': 1.0,
        },
    )


def _install_support_geometry(
    monkeypatch, *, overlap=0.010, palm_y=(-0.165, -0.125)
):
    module = (
        'erc_phase1_solution.'
        'live_rigid_palm_upright_open_insert_probe'
    )
    monkeypatch.setattr(
        f'{module}.shelf_edge_proximity_geometry',
        lambda *args, **kwargs: _edge(overlap=overlap),
    )
    monkeypatch.setattr(
        f'{module}.shelf_solid_penetration_m',
        lambda *args, **kwargs: 0.0001,
    )
    palm = np.asarray(
        [
            [2.595, palm_y[0], 1.452],
            [2.680, palm_y[0], 1.452],
            [2.680, palm_y[1], 1.452],
            [2.595, palm_y[1], 1.452],
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        f'{module}._raw_palm_support_geometry',
        lambda *args, **kwargs: SimpleNamespace(polygon_world=palm),
    )


class _TranslationChain:
    """Minimal exact xyz chain for reanchor/IK unit tests."""

    lower = np.full(8, -1.0, dtype=float)
    upper = np.full(8, 1.0, dtype=float)

    @staticmethod
    def forward(positions):
        q = np.asarray(positions, dtype=float)
        result = np.eye(4, dtype=float)
        result[:3, 3] = q[1:4]
        return result

    @staticmethod
    def solve(target, seeds, **kwargs):
        result = np.asarray(seeds[0], dtype=float).copy()
        result[1:4] = np.asarray(target, dtype=float)[:3, 3]
        return result, {'exact': True}

    @staticmethod
    def pose_error(achieved, target):
        result = np.zeros(6, dtype=float)
        result[:3] = (
            np.asarray(target, dtype=float)[:3, 3]
            - np.asarray(achieved, dtype=float)[:3, 3]
        )
        return result


def test_opening_targets_are_exact_guarded_30_to_47mm_stages():
    targets = probe.opening_stage_targets(probe.START_APERTURE_M)

    assert targets == pytest.approx(
        tuple(value / 1000.0 for value in range(31, 48))
    )
    assert max(np.diff(targets)) == pytest.approx(0.001)
    assert targets[-1] == probe.OPEN_APERTURE_M


def test_upright_combined_support_uses_local_palm_and_shelf_line(monkeypatch):
    _install_support_geometry(monkeypatch)

    result = probe.upright_combined_support_guard(
        _book(),
        _corners(),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3, 3)),
        np.eye(4),
        [-1.0, 0.0, 0.0],
        polygon_inset_m=0.001,
    )

    assert result.safe, result.reason
    assert result.metrics['shelf_signed_overlap_m'] == pytest.approx(0.010)
    assert result.metrics['local_palm_feature_x_span_m'] > 0.0005
    assert result.metrics['local_palm_feature_y_span_m'] >= 0.010
    assert result.metrics['combined_support_x_span_m'] >= 0.120
    assert result.metrics['combined_support_com_original_slack_m'] >= 0.003
    assert result.metrics['raw_wrench_support_proven'] == 0.0
    assert result.metrics['force_closure_proven'] == 0.0


def test_upright_combined_support_rejects_overlap_outside_11mm_band(
    monkeypatch,
):
    _install_support_geometry(monkeypatch, overlap=0.015001)

    result = probe.upright_combined_support_guard(
        _book(),
        _corners(),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3, 3)),
        np.eye(4),
        [-1.0, 0.0, 0.0],
        polygon_inset_m=0.001,
    )

    assert not result.safe
    assert result.reason == 'shelf_overlap_outside_upright_band'


def test_upright_combined_support_rejects_tiny_local_palm_feature(
    monkeypatch,
):
    _install_support_geometry(
        monkeypatch, palm_y=(-0.146, -0.144)
    )

    result = probe.upright_combined_support_guard(
        _book(),
        _corners(),
        np.zeros((1, 3, 3)),
        np.zeros((1, 3, 3)),
        np.eye(4),
        [-1.0, 0.0, 0.0],
        polygon_inset_m=0.001,
    )

    assert not result.safe
    assert result.reason == 'local_palm_feature_y_span_too_small'


def test_solver_reanchors_exact_quarter_mm_from_each_measured_state():
    chain = _TranslationChain()
    first = np.zeros(8, dtype=float)
    first[1:4] = [0.10, -0.20, 0.30]
    second = first.copy()
    second[1:4] += [0.001, 0.002, 0.003]

    one = probe.solve_reanchored_tangent_waypoint(
        chain,
        first,
        phase='single_0p25mm_local_z_insert',
        local_y_m=0.0,
        local_z_m=probe.INSERT_LOCAL_Z_M,
    )
    two = probe.solve_reanchored_tangent_waypoint(
        chain,
        second,
        phase='single_0p25mm_local_z_insert',
        local_y_m=0.0,
        local_z_m=probe.INSERT_LOCAL_Z_M,
    )

    assert one.target_hand_base[:3, 3] - one.start_hand_base[:3, 3] == (
        pytest.approx([0.0, 0.0, 0.00025])
    )
    assert two.target_hand_base[:3, 3] - two.start_hand_base[:3, 3] == (
        pytest.approx([0.0, 0.0, 0.00025])
    )
    assert two.start_positions == pytest.approx(second)
    assert two.positions != pytest.approx(one.positions)


def test_solver_allows_one_measured_local_y_recenter_but_not_large_one():
    waypoint = probe.solve_reanchored_tangent_waypoint(
        _TranslationChain(),
        np.zeros(8),
        phase='measured_local_y_recenter',
        local_y_m=0.00026,
        local_z_m=0.0,
    )

    assert waypoint.command_local_m == pytest.approx([0.0, 0.00026, 0.0])
    with pytest.raises(ValueError, match='exceeds one millimetre'):
        probe.solve_reanchored_tangent_waypoint(
            _TranslationChain(),
            np.zeros(8),
            phase='measured_local_y_recenter',
            local_y_m=0.001001,
            local_z_m=0.0,
        )


def test_relative_endpoint_accepts_historical_like_60_percent_book_follow():
    reference = _observation(stamp=1.0)
    endpoint = _observation(
        hand_shift=(0.0, 0.0, 0.00025),
        book_shift=(0.0, 0.0, 0.00015),
        stamp=1.1,
    )

    result = probe.relative_progress_endpoint_guard(
        reference,
        reference,
        reference,
        endpoint,
        _waypoint(),
        support=_support(),
        reference_time=1.1,
        palm_contact=True,
        unexpected_contacts=False,
    )

    assert result.safe, result.reason
    assert result.metrics['hand_axis_progress_m'] == pytest.approx(0.00025)
    assert result.metrics['book_axis_progress_m'] == pytest.approx(0.00015)
    assert result.metrics['hand_book_relative_progress_m'] == pytest.approx(
        0.00010
    )
    assert result.metrics['book_follow_fraction'] == pytest.approx(0.60)


def test_relative_endpoint_rejects_eighty_percent_book_follow():
    reference = _observation(stamp=1.0)
    endpoint = _observation(
        hand_shift=(0.0, 0.0, 0.00025),
        book_shift=(0.0, 0.0, 0.00020),
        stamp=1.1,
    )

    result = probe.relative_progress_endpoint_guard(
        reference,
        reference,
        reference,
        endpoint,
        _waypoint(),
        support=_support(),
        reference_time=1.1,
        palm_contact=True,
        unexpected_contacts=False,
    )

    assert not result.safe
    assert result.reason == 'target_followed_hand_too_far'
    assert result.metrics['book_follow_fraction'] == pytest.approx(0.80)


def test_relative_endpoint_rejects_too_little_relative_gain():
    reference = _observation(stamp=1.0)
    endpoint = _observation(
        hand_shift=(0.0, 0.0, 0.00015),
        book_shift=(0.0, 0.0, 0.00011),
        stamp=1.1,
    )

    result = probe.relative_progress_endpoint_guard(
        reference,
        reference,
        reference,
        endpoint,
        _waypoint(),
        support=_support(),
        reference_time=1.1,
        palm_contact=True,
        unexpected_contacts=False,
    )

    assert not result.safe
    assert result.reason == 'insufficient_hand_book_relative_progress'


def test_opening_checkpoint_requires_support_palm_and_idle_right_arm():
    reference = _observation(aperture=0.030, stamp=1.0)
    current = _observation(
        aperture=0.031,
        right=[0.002001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        stamp=1.1,
    )

    moved_right = probe.opening_checkpoint_guard(
        reference,
        current,
        expected_aperture_m=0.031,
        support=_support(),
        reference_time=1.1,
        palm_contact=True,
        unexpected_contacts=False,
    )
    no_support = probe.opening_checkpoint_guard(
        reference,
        replace(current, right_arm=np.zeros(7)),
        expected_aperture_m=0.031,
        support=_support(False, 'local_palm_feature_x_span_too_small'),
        reference_time=1.1,
        palm_contact=True,
        unexpected_contacts=False,
    )
    no_palm = probe.opening_checkpoint_guard(
        reference,
        replace(current, right_arm=np.zeros(7)),
        expected_aperture_m=0.031,
        support=_support(),
        reference_time=1.1,
        palm_contact=False,
        unexpected_contacts=False,
    )

    assert moved_right.reason == 'right_arm_moved'
    assert no_support.reason == 'local_palm_feature_x_span_too_small'
    assert no_palm.reason == 'exact_target_palm_contact_missing'


def test_runtime_contains_one_insert_and_no_forbidden_motion_path():
    source = inspect.getsource(probe._run)
    tree = ast.parse(source)
    insert_solve_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == 'solve_reanchored_tangent_waypoint'
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        local_z = keywords.get('local_z_m')
        if isinstance(local_z, ast.Name) and local_z.id == 'INSERT_LOCAL_Z_M':
            insert_solve_calls.append(node)

    assert '_execute_aperture_stage(' in source
    assert 'opening_stage_targets(START_APERTURE_M)' in source
    assert len(insert_solve_calls) == 1
    assert '_execute_guarded_arm_leg(' in source
    assert 'ProbeNavigation' not in source
    assert '_verified_opening_reclose' not in source
    assert '_move_arm_solution' not in source
    assert '_send_base' not in source
    assert 'set_model' not in source
    assert 'recage_stage_targets' not in source
