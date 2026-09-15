"""Pure tests for the diagnostic-only supported compaction continuation."""

import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from erc_phase1_solution.live_rigid_palm_post_slide_transport_probe import (  # noqa: E402,E501
    RECAGE_APERTURE_M,
    SupportAssessment,
    TransportObservation,
    capture_attachment_anchor,
)
from erc_phase1_solution.live_rigid_palm_support_probe import (  # noqa: E402
    BookSnapshot,
)
from erc_phase1_solution.live_rigid_palm_supported_compaction_probe import (  # noqa: E402,E501
    APERTURE_ENDPOINT_TOLERANCE_M,
    ENDPOINT_ATTACHMENT_POSITION_LIMIT_M,
    MINIMUM_SHELF_CLEARANCE_M,
    SupportedCompactionPlan,
    WATCHDOG_BASE_TRANSLATION_LIMIT_M,
    build_compaction_goal,
    compaction_endpoint_guard,
    compaction_watchdog_reason,
    deep_cage_resume_guard,
    measured_attached_envelope,
    supported_compaction_candidates,
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


def _rotation_z(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _quaternion_z(angle):
    return np.asarray([0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)])


def _transform(position=(0.0, 0.0, 0.0), rotation=None):
    result = np.eye(4)
    result[:3, :3] = np.eye(3) if rotation is None else rotation
    result[:3, 3] = position
    return result


def _ordered_box(center, half=(0.080, 0.015, 0.125), rotation=None):
    basis = (
        np.eye(3)
        if rotation is None
        else np.asarray(rotation, dtype=float)
    )
    return (SIGNS * np.asarray(half, dtype=float)) @ basis.T + np.asarray(
        center, dtype=float
    )


def _snapshot(corners, yaw=0.0):
    points = np.asarray(corners, dtype=float)
    return BookSnapshot(
        np.mean(points, axis=0),
        _quaternion_z(yaw),
        np.min(points, axis=0),
        np.max(points, axis=0),
    )


def _observation(
    *,
    book_x=2.650,
    book_y=0.0,
    hand_x=0.0,
    base_x=0.0,
    base_y=0.0,
    base_yaw=0.0,
    aperture=RECAGE_APERTURE_M,
    arm_delta=0.0,
    stamp=1.0,
):
    corners = _ordered_box((book_x, book_y, 1.5))
    hand = _transform((hand_x, 0.0, 0.0))
    arm = np.zeros(8)
    arm[3] = arm_delta
    return TransportObservation(
        book=_snapshot(corners),
        corners=corners,
        base=np.asarray([base_x, base_y, base_yaw], dtype=float),
        arm=arm,
        hand_base=hand.copy(),
        hand_world=hand,
        aperture_m=aperture,
        observed_at=stamp,
    )


def _support(safe=True, reason='ok'):
    return SupportAssessment(
        safe,
        reason,
        0.010 if safe else 0.0,
        0.012,
        0.01075,
        0.01275,
        np.zeros(3),
        np.asarray([0.0, 0.0, 1.0]),
        np.zeros((4, 3)),
        0.001,
    )


def _resume_values(reference, current):
    anchor = capture_attachment_anchor(
        reference.book,
        reference.corners,
        reference.hand_world,
        observed_at=reference.observed_at,
    )
    return dict(
        reference=reference,
        current=current,
        anchor=anchor,
        support=_support(),
        reference_time=current.observed_at + 0.05,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        navigation_idle=True,
    )


def test_resume_accepts_fresh_stationary_deep_cage_after_retreat():
    reference = _observation(stamp=1.0)
    current = _observation(stamp=1.18)

    result = deep_cage_resume_guard(**_resume_values(reference, current))

    assert result.safe
    assert result.metrics['measured_aperture_m'] == pytest.approx(0.0303)
    assert result.metrics['shelf_clearance_m'] > MINIMUM_SHELF_CLEARANCE_M


@pytest.mark.parametrize(
    ('change', 'reason'),
    (
        (
            {
                'aperture': (
                    RECAGE_APERTURE_M
                    + APERTURE_ENDPOINT_TOLERANCE_M
                    + 1e-6
                )
            },
            'wrong_cage_aperture',
        ),
        ({'book_x': 2.6551}, 'target_not_shelf_clear'),
        ({'base_x': 0.00051}, 'base_not_settled'),
        ({'arm_delta': 0.0011}, 'arm_not_settled'),
    ),
)
def test_resume_rejects_wrong_geometry_or_motion(change, reason):
    reference = _observation(stamp=1.0)
    current = _observation(stamp=1.18, **change)

    result = deep_cage_resume_guard(**_resume_values(reference, current))

    assert not result.safe
    assert result.reason == reason


def test_resume_requires_exact_three_point_contact_support_and_idle_base():
    reference = _observation(stamp=1.0)
    current = _observation(stamp=1.18)
    values = _resume_values(reference, current)

    assert deep_cage_resume_guard(
        **{**values, 'left_contact': False}
    ).reason == 'left_target_contact_missing'
    assert deep_cage_resume_guard(
        **{**values, 'palm_contact': False}
    ).reason == 'exact_target_palm_contact_missing'
    assert deep_cage_resume_guard(
        **{**values, 'navigation_idle': False}
    ).reason == 'navigation_still_active'
    assert deep_cage_resume_guard(
        **{**values, 'support': _support(False, 'support_too_shallow')}
    ).reason == 'support_too_shallow'


def test_measured_attachment_envelope_preserves_pose_and_adds_padding():
    rotation = _rotation_z(0.37)
    corners = _ordered_box((0.2, -0.1, 0.4), rotation=rotation)
    observation = TransportObservation(
        _snapshot(corners, yaw=0.37),
        corners,
        np.zeros(3),
        np.zeros(8),
        np.eye(4),
        np.eye(4),
        RECAGE_APERTURE_M,
        1.0,
    )
    anchor = capture_attachment_anchor(
        observation.book,
        observation.corners,
        observation.hand_world,
        observed_at=1.0,
    )

    inflated = measured_attached_envelope(anchor, 0.015)

    np.testing.assert_allclose(
        np.mean(inflated, axis=0), np.mean(corners, axis=0)
    )
    original_distances = np.sort(np.linalg.norm(
        corners - np.mean(corners, axis=0), axis=1
    ))
    inflated_distances = np.sort(np.linalg.norm(
        inflated - np.mean(inflated, axis=0), axis=1
    ))
    assert np.all(inflated_distances > original_distances)


class _PlanningNode:
    carried_navigation_radius_limit = 0.45

    def __init__(self):
        self.starts = []

    def _supported_compact_goals(self, start):
        first = np.asarray(start, dtype=float)
        return [first + 0.01, first + 0.02]

    def _solve_supported_post_retreat_staging(self, start):
        first = np.asarray(start, dtype=float)
        return [], [first + 0.003], [first + 0.006]

    def _plan_carried_joint_route(
        self, start, goals, attached, *, require_gravity_support
    ):
        self.starts.append(np.asarray(start, dtype=float).copy())
        assert np.asarray(attached).shape == (8, 3)
        assert require_gravity_support is True
        return [np.asarray(goal, dtype=float).copy() for goal in goals]

    def _carried_navigation_radius(self, terminal, attached):
        assert np.asarray(attached).shape == (8, 3)
        return 0.44 if float(np.asarray(terminal)[0]) < 0.04 else 0.46


def test_candidates_are_recomputed_from_measured_start_without_cache():
    node = _PlanningNode()
    measured = np.asarray([0.011, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -1.2])
    attached = _ordered_box((0.1, 0.0, 0.0), half=(0.08, 0.015, 0.125))

    plans = supported_compaction_candidates(node, measured, attached)

    assert plans
    assert all(isinstance(plan, SupportedCompactionPlan) for plan in plans)
    np.testing.assert_allclose(node.starts[0], measured)
    assert all(plan.terminal_radius_m <= 0.45 for plan in plans)
    assert not hasattr(node, '_cached_post_retreat_plan')


class _DirectFailurePlanningNode(_PlanningNode):
    def __init__(self):
        super().__init__()
        self.compact_calls = 0

    def _supported_compact_goals(self, start):
        self.compact_calls += 1
        if self.compact_calls == 1:
            raise RuntimeError('direct branch unavailable')
        return super()._supported_compact_goals(start)


def test_staged_candidate_survives_a_direct_helper_failure():
    node = _DirectFailurePlanningNode()
    measured = np.zeros(8)
    attached = _ordered_box((0.1, 0.0, 0.0))

    plans = supported_compaction_candidates(node, measured, attached)

    assert [plan.label for plan in plans] == [
        'supported_lower_retract_compact'
    ]


def _watchdog_values():
    return dict(
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
        measured_aperture_m=RECAGE_APERTURE_M,
        navigation_pose=(1.0, 2.0, 0.1),
        reference_navigation_pose=(1.0, 2.0, 0.1),
        navigation_active=False,
    )


def test_watchdog_requires_contacts_aperture_and_stationary_base():
    values = _watchdog_values()

    assert compaction_watchdog_reason(**values) is None
    assert compaction_watchdog_reason(
        **{**values, 'right_contact': False}
    ) == 'exact_right_target_contact_lost'
    assert compaction_watchdog_reason(
        **{**values, 'palm_contact': False}
    ) == 'exact_palm_target_contact_lost'
    assert compaction_watchdog_reason(
        **{
            **values,
            'measured_aperture_m': RECAGE_APERTURE_M + 0.00036,
        }
    ) == 'cage_aperture_changed'
    assert compaction_watchdog_reason(
        **{
            **values,
            'navigation_pose': (
                1.0 + WATCHDOG_BASE_TRANSLATION_LIMIT_M + 1e-6,
                2.0,
                0.1,
            ),
        }
    ) == 'base_moved_during_compaction'


def test_endpoint_requires_measured_radius_contacts_and_attachment():
    reference = _observation(stamp=1.0)
    current = _observation(stamp=1.2)
    anchor = capture_attachment_anchor(
        reference.book,
        reference.corners,
        reference.hand_world,
        observed_at=reference.observed_at,
    )
    values = dict(
        reference=reference,
        current=current,
        anchor=anchor,
        support=_support(),
        expected_arm=current.arm,
        navigation_radius_m=0.449,
        navigation_radius_limit_m=0.45,
        reference_time=1.25,
        left_contact=True,
        right_contact=True,
        palm_contact=True,
        unexpected_contacts=False,
    )

    assert compaction_endpoint_guard(**values).safe
    assert compaction_endpoint_guard(
        **{**values, 'navigation_radius_m': 0.451}
    ).reason == 'navigation_radius_exceeded'
    slipped = _observation(
        book_x=2.650 + ENDPOINT_ATTACHMENT_POSITION_LIMIT_M + 0.001,
        stamp=1.2,
    )
    assert compaction_endpoint_guard(
        **{**values, 'current': slipped}
    ).reason == 'attachment_position_slip'


class _Duration:
    def __init__(self, *, seconds):
        self.seconds = seconds

    def to_msg(self):
        return self.seconds


class _Point:
    def __init__(self):
        self.positions = []
        self.time_from_start = None


class _Goal:
    def __init__(self):
        self.trajectory = SimpleNamespace(joint_names=[], points=[])


def test_one_goal_uses_rclpy_duration_and_cumulative_times():
    runtime = SimpleNamespace(
        FollowJointTrajectory=SimpleNamespace(Goal=_Goal),
        JointTrajectoryPoint=_Point,
        Duration=_Duration,
        ARM_JOINTS=tuple(f'joint_{index}' for index in range(7)),
    )
    first = np.arange(8, dtype=float)
    second = first + 0.1

    goal, duration = build_compaction_goal(
        runtime,
        ((first, 0.4, 'first'), (second, 0.6, 'second')),
    )

    assert duration == pytest.approx(1.0)
    assert len(goal.trajectory.points) == 2
    assert goal.trajectory.points[0].time_from_start == pytest.approx(0.4)
    assert goal.trajectory.points[1].time_from_start == pytest.approx(1.0)
    np.testing.assert_allclose(
        goal.trajectory.points[-1].positions, second[1:]
    )


def test_source_is_diagnostic_one_trajectory_and_has_no_task_continuation():
    source = (
        PACKAGE_ROOT
        / 'erc_phase1_solution'
        / 'live_rigid_palm_supported_compaction_probe.py'
    ).read_text(encoding='utf-8')

    assert '--confirm-diagnostic-supported-compaction' in source
    assert 'from rclpy.duration import Duration' in source
    assert 'diagnostic_truth_and_contacts_only=True' in source
    assert 'navigation_commanded=False' in source
    assert 'placement_commanded=False' in source
    assert 'stopped_at_compact_cage=True' in source
    assert 'next_motion_authorized=False' in source
    assert '_command_gripper(' not in source
    assert '_accept_goal(' not in source
