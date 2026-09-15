"""Pure guards for the diagnostic rigid-palm extraction continuation."""

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from erc_phase1_solution.live_rigid_palm_extract_retreat_probe import (  # noqa: E402
    CONTACT_DWELL_S,
    EXPECTED_CAGED_APERTURE_M,
    EXPECTED_CAGED_BOOK_POSITION,
    EXPECTED_RELEASED_BASE_POSE,
    EXPECTED_RELEASED_BOOK_QUATERNION,
    EXTRACTION_STEP_M,
    _attached_corners,
    _book_corners,
    _extraction_tangent_reference_pose,
    _is_transient_preflight_stability_rejection,
    _outward_world,
    _preflight_leg,
    intermediate_extraction_resume_guard,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (  # noqa: E402
    BOOK_HALF_EXTENTS_M,
    BookSnapshot,
)


PADDING_M = 0.00075


class _IdentityChain:
    @staticmethod
    def forward(_positions):
        return np.eye(4)


def _snapshot(quaternion):
    provisional = BookSnapshot(
        position=np.asarray([2.77, -0.15, 1.58]),
        quaternion=np.asarray(quaternion, dtype=float),
        minimum=np.zeros(3),
        maximum=np.zeros(3),
    )
    corners = _book_corners(provisional)
    return BookSnapshot(
        position=provisional.position,
        quaternion=provisional.quaternion,
        minimum=np.min(corners, axis=0),
        maximum=np.max(corners, axis=0),
    )


def _assert_same_corner_set(first, second):
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    assert left.shape == right.shape == (8, 3)
    assert all(
        np.min(np.linalg.norm(right - point, axis=1)) <= 1e-12
        for point in left
    )
    assert all(
        np.min(np.linalg.norm(left - point, axis=1)) <= 1e-12
        for point in right
    )


def _ordered_basis(corners):
    points = np.asarray(corners, dtype=float)
    edges = np.column_stack(
        (points[4] - points[0], points[2] - points[0], points[1] - points[0])
    )
    return edges / np.linalg.norm(edges, axis=0)


def _attached_world_corners(snapshot):
    node = SimpleNamespace(
        carried_book_padding=PADDING_M,
        chain=_IdentityChain(),
    )
    return _attached_corners(node, snapshot, np.zeros(3), np.zeros(8))


def test_attached_corners_put_current_y_rotated_book_vertical_axis_up():
    half_angle = 0.25 * math.pi
    book = _snapshot([0.0, math.sin(half_angle), 0.0, math.cos(half_angle)])

    attached = _attached_world_corners(book)
    original = _book_corners(book, padding_m=PADDING_M)
    basis = _ordered_basis(attached)

    _assert_same_corner_set(attached, original)
    assert attached[1, 2] > attached[0, 2]
    assert np.argmax(np.abs(basis[2, :])) == 2
    assert np.linalg.det(basis) > 1.0 - 1e-12
    np.testing.assert_allclose(
        np.linalg.norm(attached[1] - attached[0]),
        2.0 * (BOOK_HALF_EXTENTS_M[0] + PADDING_M),
        atol=1e-12,
        rtol=0.0,
    )


def test_attached_corners_reorder_different_x_rotated_obb_without_change():
    half_angle = 0.25 * math.pi
    book = _snapshot([math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)])

    attached = _attached_world_corners(book)
    original = _book_corners(book, padding_m=PADDING_M)
    basis = _ordered_basis(attached)

    _assert_same_corner_set(attached, original)
    assert attached[1, 2] > attached[0, 2]
    assert np.argmax(np.abs(basis[2, :])) == 2
    assert np.linalg.det(basis) > 1.0 - 1e-12
    np.testing.assert_allclose(
        np.linalg.norm(attached[1] - attached[0]),
        2.0 * (BOOK_HALF_EXTENTS_M[1] + PADDING_M),
        atol=1e-12,
        rtol=0.0,
    )


def _resume_book(progress_m):
    delta = _outward_world(EXPECTED_RELEASED_BASE_POSE) * float(progress_m)
    position = EXPECTED_CAGED_BOOK_POSITION + delta
    return BookSnapshot(
        position=position,
        quaternion=EXPECTED_RELEASED_BOOK_QUATERNION.copy(),
        minimum=position - np.asarray([0.0805, 0.0155, 0.1255]),
        maximum=position + np.asarray([0.0805, 0.0155, 0.1255]),
    )


def _intermediate_guard(*, hand_step=8, book_progress=0.039, stamp=9.9,
                        palm=True):
    expected_hand = np.eye(4)
    expected_hand[:3, 3] = [2.7, -0.15, 1.50]
    current_hand = expected_hand.copy()
    current_hand[:3, 3] += (
        _outward_world(EXPECTED_RELEASED_BASE_POSE)
        * (float(hand_step) * EXTRACTION_STEP_M)
    )
    return intermediate_extraction_resume_guard(
        _resume_book(book_progress),
        EXPECTED_RELEASED_BASE_POSE.copy(),
        current_hand,
        expected_hand,
        aperture_m=EXPECTED_CAGED_APERTURE_M,
        observed_at=stamp,
        reference_time=10.0,
        max_state_age_seconds=0.25,
        future_tolerance_seconds=0.02,
        left_contact=True,
        right_contact=True,
        palm_contact=palm,
        unexpected_contacts=False,
    )


def test_intermediate_resume_accepts_unique_step8_with_small_caged_lag():
    result = _intermediate_guard()

    assert result.safe
    assert result.reason == 'ok'
    assert result.metrics['inferred_extraction_step'] == 8.0
    assert result.metrics['book_step_residual_m'] < 0.00101


def test_intermediate_resume_rejects_book_hand_step_disagreement():
    result = _intermediate_guard(book_progress=0.034)

    assert not result.safe
    assert result.reason == 'book_hand_step_disagreement'


def test_intermediate_resume_rejects_stale_scene_or_missing_contact():
    stale = _intermediate_guard(stamp=9.0)
    no_palm = _intermediate_guard(palm=False)

    assert not stale.safe
    assert stale.reason == 'stale_or_future_scene'
    assert not no_palm.safe
    assert no_palm.reason == 'palm_contact_missing'


def test_checkpoint_reference_reconstructs_global_step_zero_pose():
    expected = np.eye(4)
    expected[:3, 3] = [0.45, -0.05, 1.49]
    outward = np.asarray([-1.0, 0.0, 0.0])
    current = expected.copy()
    current[:3, 3] += outward * (8 * EXTRACTION_STEP_M)

    reconstructed = _extraction_tangent_reference_pose(current, outward, 8)

    np.testing.assert_allclose(reconstructed, expected, atol=1e-12, rtol=0.0)


def test_preflight_retry_classifier_is_narrow():
    moving_book = SimpleNamespace(
        safe=False,
        code='live_adapter_error',
        detail=(
            "book 'book_col_3_row_2_red' moved 0.000085639 m "
            'during preflight'
        ),
    )
    collision = SimpleNamespace(
        safe=False,
        code='target_palm_overlap_exceeds_phase_cap',
        detail='palm target overlap exceeds cap',
    )
    unrelated_adapter_error = SimpleNamespace(
        safe=False,
        code='live_adapter_error',
        detail='Gazebo scene timestamps moved backwards',
    )

    assert _is_transient_preflight_stability_rejection(moving_book)
    assert not _is_transient_preflight_stability_rejection(collision)
    assert not _is_transient_preflight_stability_rejection(
        unrelated_adapter_error
    )


class _PreflightNode:
    def __init__(self):
        self.dwell_durations = []

    @staticmethod
    def _measured_left_solution():
        return np.zeros(8)

    @staticmethod
    def _robot_self_collision(_sample):
        return None

    def _wait_sim_duration(self, duration):
        self.dwell_durations.append(float(duration))
        return True


class _SequenceEnvironment:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def __call__(self, **_kwargs):
        index = min(self.calls, len(self.results) - 1)
        self.calls += 1
        return self.results[index]


def test_preflight_retries_transient_stability_then_accepts():
    transient = SimpleNamespace(
        safe=False,
        code='live_adapter_error',
        detail=(
            "book 'book_col_3_row_2_red' moved 0.000085639 m "
            'during preflight'
        ),
    )
    clear = SimpleNamespace(safe=True, code='clear', detail='clear')
    environment = _SequenceEnvironment([transient, clear])

    node = _PreflightNode()
    _preflight_leg(node, environment, np.full(8, 0.001))

    assert environment.calls == 2
    assert node.dwell_durations == [CONTACT_DWELL_S]


def test_preflight_stops_if_transient_retry_dwell_is_interrupted():
    transient = SimpleNamespace(
        safe=False,
        code='live_adapter_error',
        detail=(
            "book 'book_col_3_row_2_red' moved 0.000085639 m "
            'during preflight'
        ),
    )
    clear = SimpleNamespace(safe=True, code='clear', detail='clear')
    environment = _SequenceEnvironment([transient, clear])
    node = _PreflightNode()
    node._wait_sim_duration = lambda _duration: False

    try:
        _preflight_leg(node, environment, np.full(8, 0.001))
    except RuntimeError as error:
        assert 'settling dwell was interrupted' in str(error)
    else:
        raise AssertionError('interrupted settling dwell must fail closed')
    assert environment.calls == 1


def test_preflight_never_retries_collision_and_caps_stability_retries():
    collision = SimpleNamespace(
        safe=False,
        code='tool_shelf_collision',
        detail='arm_left_7_link intersects shelf',
    )
    collision_environment = _SequenceEnvironment([collision])
    collision_node = _PreflightNode()
    try:
        _preflight_leg(
            collision_node, collision_environment, np.full(8, 0.001)
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError('collision preflight must fail')
    assert collision_environment.calls == 1
    assert collision_node.dwell_durations == []

    transient = SimpleNamespace(
        safe=False,
        code='live_adapter_error',
        detail='arm moved 0.000200000 rad during geometric preflight',
    )
    unstable_environment = _SequenceEnvironment([transient])
    unstable_node = _PreflightNode()
    try:
        _preflight_leg(
            unstable_node, unstable_environment, np.full(8, 0.001)
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError('exhausted stability retries must fail')
    # One initial check plus no more than three retries.
    assert unstable_environment.calls == 4
    assert unstable_node.dwell_durations == [CONTACT_DWELL_S] * 3
