"""Pure guards for the diagnostic shelf-edge cradle probe."""

import math
from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from erc_phase1_solution.live_rigid_palm_shelf_edge_cradle_probe import (  # noqa: E402
    CRADLE_MINIMUM_SHELF_OVERLAP_M,
    EXPECTED_CAGED_APERTURE_M,
    PALM_OUTER_TOP_EDGE_IN_GRASP_M,
    cradle_step_guard,
    pivoted_cradle_pose,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (  # noqa: E402
    BookSnapshot,
)


def _snapshot(*, delta=(0.0, 0.0, 0.0), maximum_x=2.765):
    offset = np.asarray(delta, dtype=float)
    position = np.asarray([2.6847, -0.1442, 1.5770]) + offset
    return BookSnapshot(
        position=position,
        quaternion=np.asarray([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]),
        minimum=np.asarray([2.6045, -0.1592, 1.4520]) + offset,
        maximum=np.asarray([maximum_x, -0.1292, 1.7020]) + offset,
    )


def test_pivoted_cradle_pose_keeps_outer_top_edge_fixed_and_lowers_pad():
    reference = np.eye(4)
    reference[:3, 3] = [0.49, -0.055, 1.499]
    angle = math.radians(5.0)

    result = pivoted_cradle_pose(reference, angle)
    before = (
        reference[:3, :3] @ PALM_OUTER_TOP_EDGE_IN_GRASP_M
        + reference[:3, 3]
    )
    after = (
        result[:3, :3] @ PALM_OUTER_TOP_EDGE_IN_GRASP_M
        + result[:3, 3]
    )

    np.testing.assert_allclose(after, before, atol=1e-12, rtol=0.0)
    expected = reference[:3, :3] @ np.asarray(
        [
            [math.cos(angle), 0.0, -math.sin(angle)],
            [0.0, 1.0, 0.0],
            [math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    np.testing.assert_allclose(result[:3, :3], expected, atol=1e-12, rtol=0.0)

    inner_edge = np.asarray([-0.04650, 0.0, 0.011463], dtype=float)
    inner_before = reference[:3, :3] @ inner_edge + reference[:3, 3]
    inner_after = result[:3, :3] @ inner_edge + result[:3, 3]
    # With identity as the reference orientation, local +x is world +x.
    # The whole pad beyond the fixed outer edge must fall away rather than
    # being rotated upward into the nominally stationary book.
    assert inner_after[0] < inner_before[0]


def test_pivoted_cradle_pose_rejects_unvalidated_extension():
    try:
        pivoted_cradle_pose(np.eye(4), math.radians(5.01))
    except ValueError:
        return
    raise AssertionError('angles beyond the five-degree evidence probe must fail')


def _guard(*, after_book=None, left=True, right=True, palm=True):
    book = _snapshot()
    base = np.asarray([2.1269, -0.0933, 0.00086])
    reference_hand = np.eye(4)
    reference_hand[:3, 3] = [2.62, -0.144, 1.499]
    after_hand = pivoted_cradle_pose(reference_hand, math.radians(1.0))
    return cradle_step_guard(
        book,
        book,
        book if after_book is None else after_book,
        reference_hand,
        reference_hand,
        after_hand,
        base,
        base,
        base,
        expected_angle_rad=math.radians(1.0),
        left_contact=left,
        right_contact=right,
        palm_contact=palm,
        unexpected_contacts=False,
        measured_aperture_m=EXPECTED_CAGED_APERTURE_M,
    )


def test_cradle_step_guard_accepts_stationary_supported_book():
    result = _guard()

    assert result.safe
    assert result.reason == 'ok'
    assert result.metrics['palm_pivot_drift_m'] < 1e-12
    assert result.metrics['shelf_overlap_m'] > CRADLE_MINIMUM_SHELF_OVERLAP_M


def test_cradle_step_guard_rejects_book_motion_or_missing_contact():
    shifted = _guard(after_book=_snapshot(delta=(0.0008, 0.0, 0.0)))
    no_palm = _guard(palm=False)

    assert not shifted.safe
    assert shifted.reason == 'book_shifted_during_cradle_step'
    assert not no_palm.safe
    assert no_palm.reason == 'palm_contact_missing'


def test_cradle_step_guard_rejects_loss_of_shelf_overlap():
    unsupported = _snapshot(
        maximum_x=2.755 + CRADLE_MINIMUM_SHELF_OVERLAP_M - 0.00001
    )

    result = _guard(after_book=unsupported)

    assert not result.safe
    assert result.reason == 'shelf_support_overlap_lost'
