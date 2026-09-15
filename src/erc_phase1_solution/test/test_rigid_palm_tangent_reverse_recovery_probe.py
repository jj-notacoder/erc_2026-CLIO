"""Offline tests for the partial tangent reverse-and-recage recovery."""

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
from erc_phase1_solution.live_rigid_palm_support_probe import (  # noqa: E402
    BOOK_HALF_EXTENTS_M,
    EXPECTED_RELEASED_BASE_POSE,
    BookSnapshot,
    quaternion_matrix,
    rotation_matrix_distance,
    world_hand_pose,
)
from erc_phase1_solution.live_rigid_palm_tangent_reverse_recovery_probe import (  # noqa: E501,E402
    EXPECTED_PARTIAL_ARM,
    EXPECTED_PARTIAL_BOOK_MAXIMUM_X_M,
    EXPECTED_PARTIAL_BOOK_POSITION,
    EXPECTED_PARTIAL_BOOK_QUATERNION,
    EXPECTED_PREINSERT_ARM,
    OPEN_APERTURE_M,
    RECAGE_APERTURE_M,
    REVERSE_LOCAL_Z_M,
    REVERSE_MINIMUM_SHELF_OVERLAP_M,
    TARGET_BOOK_MODEL,
    HypothesisGeometrySample,
    ReverseOutcome,
    build_target_hypothesis_routes,
    classify_partial_tangent_resume,
    recovery_start_unchanged_guard,
    recage_stage_targets,
    reverse_endpoint_guard,
    solve_reverse_waypoint,
    target_follow_hypothesis_fractions,
)
from erc_phase1_solution.live_rigid_palm_tangent_slide_probe import (  # noqa
    RouteGeometrySample,
    StageObservation,
)
from erc_phase1_solution.motion_profiles import IK_JOINTS  # noqa: E402
from erc_phase1_solution.rigid_palm_preflight import BookOBB  # noqa: E402


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
    position=EXPECTED_PARTIAL_BOOK_POSITION,
    quaternion=EXPECTED_PARTIAL_BOOK_QUATERNION,
) -> BookSnapshot:
    center = np.asarray(position, dtype=float)
    orientation = np.asarray(quaternion, dtype=float)
    extent = np.abs(quaternion_matrix(orientation)) @ BOOK_HALF_EXTENTS_M
    return BookSnapshot(
        center.copy(),
        orientation.copy(),
        center - extent,
        center + extent,
    )


def _translated_book(book: BookSnapshot, delta) -> BookSnapshot:
    translation = np.asarray(delta, dtype=float)
    return BookSnapshot(
        book.position + translation,
        book.quaternion.copy(),
        book.minimum + translation,
        book.maximum + translation,
    )


def _partial_observation(
    chain: URDFChain,
    *,
    book=None,
    arm=EXPECTED_PARTIAL_ARM,
    aperture=OPEN_APERTURE_M,
    stamp=10.0,
) -> StageObservation:
    q = np.asarray(arm, dtype=float)
    return StageObservation(
        book=_book() if book is None else book,
        base=EXPECTED_RELEASED_BASE_POSE.copy(),
        arm=q.copy(),
        hand_base=chain.forward(q),
        aperture_m=float(aperture),
        observed_at=float(stamp),
    )


def _resume(chain: URDFChain, **overrides):
    state = _partial_observation(chain)
    values = {
        'book': state.book,
        'base': state.base,
        'arm': state.arm,
        'hand_base': state.hand_base,
        'aperture_m': state.aperture_m,
        'observed_at': state.observed_at,
        'reference_time': state.observed_at + 0.10,
        'palm_contact': True,
        'unexpected_contacts': False,
    }
    values.update(overrides)
    return classify_partial_tangent_resume(**values)


def test_recorded_partial_state_geometry_is_self_consistent(official_chain):
    book = _book()

    assert book.maximum[0] == pytest.approx(
        EXPECTED_PARTIAL_BOOK_MAXIMUM_X_M, abs=2e-10
    )
    assert (
        book.maximum[0] - 2.755
    ) == pytest.approx(0.0159714847, abs=2e-10)
    result = _resume(official_chain)
    assert result.safe
    assert result.metrics['relative_z_error_m'] < 2e-7


@pytest.mark.parametrize(
    ('overrides', 'reason'),
    (
        ({'aperture_m': 0.0467}, 'wrong_aperture'),
        ({'palm_contact': False}, 'exact_target_palm_contact_missing'),
        ({'unexpected_contacts': True}, 'unexpected_contact'),
        ({'reference_time': 10.30}, 'scene_stale'),
        ({'arm': EXPECTED_PARTIAL_ARM + 0.004}, 'wrong_partial_arm'),
    ),
)
def test_partial_resume_fails_closed(official_chain, overrides, reason):
    result = _resume(official_chain, **overrides)

    assert not result.safe
    assert result.reason == reason


def test_reverse_retraces_exact_preinsert_waypoint(official_chain):
    waypoint = solve_reverse_waypoint(
        official_chain, EXPECTED_PARTIAL_ARM
    )
    before = official_chain.forward(EXPECTED_PARTIAL_ARM)
    after = official_chain.forward(waypoint.positions)
    local_delta = before[:3, :3].T @ (
        after[:3, 3] - before[:3, 3]
    )

    np.testing.assert_allclose(
        waypoint.positions, EXPECTED_PREINSERT_ARM, atol=1e-14
    )
    assert local_delta[2] == pytest.approx(REVERSE_LOCAL_Z_M, abs=2e-10)
    assert np.linalg.norm(local_delta[:2]) < 4e-6
    assert rotation_matrix_distance(
        before[:3, :3], after[:3, :3]
    ) < 0.00013
    assert np.max(np.abs(
        waypoint.positions - EXPECTED_PARTIAL_ARM
    )) < 0.0152


def test_reverse_refuses_a_different_arm_checkpoint(official_chain):
    different_arm = EXPECTED_PARTIAL_ARM.copy()
    # The torso is already at its upper stop. Perturb one valid arm joint so
    # this exercises checkpoint identity rather than the earlier limit guard.
    different_arm[1] += 0.004
    assert np.all(different_arm >= official_chain.lower)
    assert np.all(different_arm <= official_chain.upper)
    with pytest.raises(ValueError, match='partial tangent checkpoint'):
        solve_reverse_waypoint(official_chain, different_arm)


def test_reverse_rejects_an_arm_outside_hard_limits(official_chain):
    outside_limits = EXPECTED_PARTIAL_ARM.copy()
    outside_limits[0] = official_chain.upper[0] + 0.004
    with pytest.raises(ValueError, match='outside its hard limits'):
        solve_reverse_waypoint(official_chain, outside_limits)


def test_follow_fraction_grid_covers_static_and_attached():
    fractions = target_follow_hypothesis_fractions(0.004961)

    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert len(fractions) == 11
    assert max(np.diff(fractions)) * 0.004961 <= 0.0005 + 1e-12
    with pytest.raises(ValueError, match='cap'):
        target_follow_hypothesis_fractions(0.020)


def test_hypothesis_routes_include_exact_static_and_following_endpoints(
    official_chain,
):
    start = EXPECTED_PARTIAL_ARM
    waypoint = solve_reverse_waypoint(official_chain, start)
    book = _book()
    signs = np.asarray([
        [x, y, z]
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ])
    corners = (
        signs * BOOK_HALF_EXTENTS_M
    ) @ quaternion_matrix(book.quaternion).T + book.position
    scene = SimpleNamespace(
        base_transform=np.asarray([
            [math.cos(EXPECTED_RELEASED_BASE_POSE[2]),
             -math.sin(EXPECTED_RELEASED_BASE_POSE[2]), 0.0,
             EXPECTED_RELEASED_BASE_POSE[0]],
            [math.sin(EXPECTED_RELEASED_BASE_POSE[2]),
             math.cos(EXPECTED_RELEASED_BASE_POSE[2]), 0.0,
             EXPECTED_RELEASED_BASE_POSE[1]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        books={
            TARGET_BOOK_MODEL: BookOBB(TARGET_BOOK_MODEL, corners, 10.0),
            'book_01': BookOBB('book_01', corners + 0.25, 10.0),
        },
    )
    dense = (
        RouteGeometrySample(start, OPEN_APERTURE_M, {}),
        RouteGeometrySample(
            waypoint.positions, OPEN_APERTURE_M, {}
        ),
    )
    node = SimpleNamespace(chain=official_chain)

    routes = build_target_hypothesis_routes(
        scene=scene, node=node, dense_arm=dense
    )
    static = routes[0][1]
    following = routes[-1][1]
    start_hand = scene.base_transform @ official_chain.forward(start)
    final_hand = (
        scene.base_transform @ official_chain.forward(waypoint.positions)
    )
    corners_in_hand = (
        corners - start_hand[:3, 3]
    ) @ start_hand[:3, :3]
    expected_following = (
        corners_in_hand @ final_hand[:3, :3].T + final_hand[:3, 3]
    )

    np.testing.assert_allclose(static[-1].target_corners, corners)
    np.testing.assert_allclose(
        following[-1].target_corners, expected_following
    )
    assert isinstance(static[-1], HypothesisGeometrySample)


def _reverse_outcome(
    chain: URDFChain,
    fraction: float,
    *,
    palm_contact: bool = True,
    unexpected_contacts: bool = False,
) -> ReverseOutcome:
    reference = _partial_observation(chain)
    waypoint = solve_reverse_waypoint(chain, reference.arm)
    reference_hand = world_hand_pose(
        reference.base, reference.hand_base
    )
    final_hand = world_hand_pose(
        reference.base, waypoint.target_pose
    )
    hand_delta = final_hand[:3, 3] - reference_hand[:3, 3]
    current = _partial_observation(
        chain,
        book=_translated_book(reference.book, fraction * hand_delta),
        arm=waypoint.positions,
        stamp=10.10,
    )
    return reverse_endpoint_guard(
        reference,
        current,
        expected_arm=waypoint.positions,
        expected_hand_base=waypoint.target_pose,
        reference_time=10.15,
        palm_contact=palm_contact,
        unexpected_contacts=unexpected_contacts,
    )


@pytest.mark.parametrize(
    ('fraction', 'mode'),
    ((0.0, 'static'), (0.45, 'bounded_slip'), (1.0, 'following')),
)
def test_reverse_endpoint_accepts_entire_preflighted_motion_union(
    official_chain, fraction, mode
):
    outcome = _reverse_outcome(official_chain, fraction)

    assert outcome.safe
    assert outcome.mode == mode
    assert outcome.metrics['shelf_overlap_m'] >= (
        REVERSE_MINIMUM_SHELF_OVERLAP_M
    )
    if fraction == 0.0:
        assert outcome.metrics[
            'palm_relative_support_gain_m'
        ] == pytest.approx(0.004961, abs=5e-6)
    if fraction == 1.0:
        assert outcome.metrics[
            'book_hand_translation_mismatch_m'
        ] < 1e-12


def test_following_reverse_returns_to_eleven_millimetres_overlap(
    official_chain,
):
    outcome = _reverse_outcome(official_chain, 1.0)

    assert outcome.metrics['shelf_overlap_m'] == pytest.approx(
        # This fixture translates the recorded book without rotating it.
        # Its URDF-derived reverse displacement leaves 11.0104 mm overlap.
        0.0110104, abs=2e-6
    )


def test_reverse_endpoint_requires_fresh_palm_and_clean_contacts(
    official_chain,
):
    no_palm = _reverse_outcome(official_chain, 1.0, palm_contact=False)
    hazard = _reverse_outcome(
        official_chain, 1.0, unexpected_contacts=True
    )

    assert not no_palm.safe
    assert no_palm.reason == 'exact_target_palm_contact_missing'
    assert not hazard.safe
    assert hazard.reason == 'unexpected_contact'


def test_reverse_endpoint_rejects_target_motion_farther_into_shelf(
    official_chain,
):
    reference = _partial_observation(official_chain)
    waypoint = solve_reverse_waypoint(official_chain, reference.arm)
    current = _partial_observation(
        official_chain,
        book=_translated_book(reference.book, [0.001, 0.0, 0.0]),
        arm=waypoint.positions,
        stamp=10.1,
    )
    outcome = reverse_endpoint_guard(
        reference,
        current,
        expected_arm=waypoint.positions,
        expected_hand_base=waypoint.target_pose,
        reference_time=10.15,
        palm_contact=True,
        unexpected_contacts=False,
    )

    assert not outcome.safe
    assert outcome.reason == 'target_reverse_progress'


def test_post_preflight_start_guard_rejects_book_drift(official_chain):
    reference = _partial_observation(official_chain)
    current = _partial_observation(
        official_chain,
        book=_translated_book(reference.book, [0.0006, 0.0, 0.0]),
        stamp=10.1,
    )
    result = recovery_start_unchanged_guard(
        reference,
        current,
        reference_time=10.15,
        palm_contact=True,
        unexpected_contacts=False,
    )

    assert not result.safe
    assert result.reason == 'book_changed_during_preflight'


def test_recage_schedule_only_closes_in_bounded_stages():
    targets = recage_stage_targets()

    assert targets[0] < OPEN_APERTURE_M
    assert targets[-1] == RECAGE_APERTURE_M
    assert all(second < first for first, second in zip(
        (OPEN_APERTURE_M, *targets), targets
    ))
    assert all(first - second <= 0.001 + 1e-12 for first, second in zip(
        (OPEN_APERTURE_M, *targets), targets
    ))


def test_live_entrypoint_is_explicit_sim_time_and_has_no_navigation():
    source_path = (
        PACKAGE_ROOT
        / 'erc_phase1_solution'
        / 'live_rigid_palm_tangent_reverse_recovery_probe.py'
    )
    source = source_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    retained_arm_calls = [
        call for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == '_send_retained_arm_trajectory'
    ]

    assert '--confirm-diagnostic-reverse-and-recage' in source
    assert "'use_sim_time:=true'" in source
    assert len(retained_arm_calls) == 1
    assert '_accept_goal' not in source
    assert 'NavigationNode' not in source
    assert 'cmd_vel' not in source
    assert 'gripper_opening_commanded=False' in source
    assert 'automatic_arm_recovery_commanded=False' in source


def test_probe_remains_explicitly_diagnostic_only():
    source_path = (
        PACKAGE_ROOT
        / 'erc_phase1_solution'
        / 'live_rigid_palm_tangent_reverse_recovery_probe.py'
    )
    source = source_path.read_text(encoding='utf-8')

    assert 'diagnostic_truth_and_contacts_only=True' in source
    assert 'next_motion_authorized=False' in source
    assert 'static_target_checked=True' in source
    assert 'following_target_checked=True' in source
    assert 'bounded_slip_checked=True' in source
