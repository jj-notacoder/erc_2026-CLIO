"""Bounded pressure recheck regressions, including the archived Run09 window.

No ROS nodes or actuator interfaces are created. The contact evaluator remains
real; fake time makes every wait and publication count observable.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from erc_phase1_solution.adaptive_grasp import ForceSample, GripperFeedback
from test_fine_gripper_close import pressure_fixture, set_feedback
from test_contact_force_timing import LEFT, RIGHT, force_message


DEFICIENCIES = (
    'bilateral_contact_missing', 'bilateral_contact_stale', 'unilateral_contact',
    'left_force_below_minimum', 'right_force_below_minimum',
    'insufficient_force_samples', 'force_sample_span_too_short',
    'force_sample_gap', 'bilateral_sample_skew',
)
HARD_REASONS = (
    'force_history_invalid', 'joint_feedback_invalid', 'joint_feedback_unavailable',
    'joint_feedback_stale', 'force_overload', 'effort_overload',
    'effort_delta_overload', 'width_too_narrow', 'width_too_wide',
    'joint_still_moving', 'unrecognized_future_evaluator_reason',
)


def evidence(reason, verified=False):
    return SimpleNamespace(reason=reason, verified=verified)


def pair(reason='left_force_below_minimum'):
    return (evidence('bilateral_contact_verified', True), evidence(reason))


def recheck_fixture(monkeypatch):
    limits = fine.FineGripLimits(maximum_force=20., minimum_force=6.)
    values = pressure_fixture(monkeypatch, limits=limits)
    c, node, now, clock, _, _ = values
    c._fine_motion_end_ns = now.nanoseconds
    def sensors():
        set_feedback(node, now, c._fine_command)
        c._fine_side_last = {'right': (now.nanoseconds, .2)}
    clock.on_sleep = sensors
    return values


@pytest.mark.parametrize('reason', DEFICIENCIES)
@pytest.mark.parametrize('side', [0, 1])
def test_explicit_pressure_deficiencies_do_not_become_acquisition_or_hard_faults(
        monkeypatch, reason, side):
    c, _, _, _, _, _ = recheck_fixture(monkeypatch)
    values = list(pair())
    values[side] = evidence(reason)
    assert c._micro_pressure_fault(tuple(values)) is None
    assert not values[1].verified


@pytest.mark.parametrize('reason', HARD_REASONS)
@pytest.mark.parametrize('side', [0, 1])
def test_any_hard_evaluator_failure_prevents_waiting_and_further_closure(
        monkeypatch, reason, side):
    c, _, _, clock, messages, statuses = recheck_fixture(monkeypatch)
    values = list(pair())
    values[side] = evidence(reason)
    c._micro_pressure_snapshot = lambda *a, **kw: pytest.fail('hard failure re-evaluated')
    result = c._recheck_micro_pressure(1, c._fine_command, tuple(values), wall_deadline=5.)
    assert result is None
    assert c._fine_fault and reason in c._fine_fault
    assert clock.wall == 0.
    assert not any(f.get('stage') == 'public_position_command' for _, f in statuses)
    assert all(m.points[0].positions[0] == .01825 for m in messages)


def test_whitelist_is_exact_and_unknown_pressure_outcomes_fail_closed():
    assert fine.MICRO_PRESSURE_DEFICIENCIES == frozenset(DEFICIENCIES)


def test_recheck_preserves_target_epoch_and_histories_without_actuation(monkeypatch):
    c, node, now, clock, messages, statuses = recheck_fixture(monkeypatch)
    initial_time, initial_epoch = now.nanoseconds, c._fine_micro_stationary_start_ns
    target = c._fine_command
    stored = tuple(ForceSample(initial_time - i*2_000_000, .2) for i in range(31, -1, -1))
    node._adaptive_force_histories = lambda: (c.expected_model, (), stored, None)
    node._clear_target_contact_samples = lambda: pytest.fail('recheck cleared history')
    original = c._micro_pressure_snapshot
    calls = []
    def snapshot(step, position, **kwargs):
        calls.append((step, position, now.nanoseconds, kwargs))
        return original(step, position, **kwargs)
    c._micro_pressure_snapshot = snapshot
    result = c._recheck_micro_pressure(3, target, pair(), wall_deadline=5.)
    assert result is not None and not result[1].verified
    assert c._fine_fault is None and not messages
    assert c._fine_command == target and c._fine_micro_stationary_start_ns == initial_epoch
    assert node._adaptive_force_histories()[2] is stored
    assert len(calls) == 1 and calls[0][:2] == (3, target)
    assert 50_000_000 <= now.nanoseconds-initial_time <= 150_000_000
    records = [f for _, f in statuses if f.get('stage') == 'pressure_measurement']
    assert len(records) == 1
    assert records[0]['raw_force_histories']['right'] == [
        {'stamp_ns': s.stamp_ns, 'force_newtons': s.force_newtons} for s in stored]


@pytest.mark.parametrize('fault', ['cancel', 'force', 'identity', 'stream', 'position', 'velocity'])
def test_recheck_live_fault_interrupts_before_final_snapshot_or_inward_motion(monkeypatch, fault):
    c, node, now, clock, messages, statuses = recheck_fixture(monkeypatch)
    original = clock.on_sleep
    def sensors():
        original()
        if fault == 'cancel':
            node._cancel.set()
        elif fault == 'force':
            book = c.expected_model + '::book_base_link::collision'
            node._on_contacts(force_message(now.nanoseconds, (RIGHT, book, 21.)))
        elif fault == 'identity':
            node._target_book_model = 'book_col_1_row_1_blue'
        elif fault == 'stream':
            c._fine_side_last = {'right': (now.nanoseconds-151_000_000, .2)}
        elif fault == 'position':
            set_feedback(node, now, c._fine_command+26e-9)
        elif fault == 'velocity':
            set_feedback(node, now, c._fine_command, velocity=2.01e-6)
    clock.on_sleep = sensors
    c._micro_pressure_snapshot = lambda *a, **kw: pytest.fail('fault reached final snapshot')
    result = c._recheck_micro_pressure(1, c._fine_command, pair(), wall_deadline=5.)
    assert result is None and c._fine_fault
    assert clock.wall < .05
    assert not any(f.get('stage') == 'public_position_command' for _, f in statuses)
    assert all(m.points[0].positions[0] >= .01825 for m in messages)


@pytest.mark.parametrize('bound', ['step', 'total'])
def test_stalled_clock_recheck_uses_existing_wall_deadlines(monkeypatch, bound):
    c, _, now, clock, messages, _ = recheck_fixture(monkeypatch)
    clock.advancing = False
    start = now.nanoseconds
    if bound == 'total':
        c._fine_wall_deadline = .012
        deadline = 5.
    else:
        deadline = .012
    result = c._recheck_micro_pressure(1, c._fine_command, pair(), wall_deadline=deadline)
    assert result is None and c._fine_fault
    assert now.nanoseconds == start and clock.wall <= .016
    assert all(m.points[0].positions[0] == .01825 for m in messages)


def load_run09():
    path = Path(__file__).parent/'fixtures'/'run09_pressure_step1.json'
    return json.loads(path.read_text(encoding='utf-8-sig'))['payload']


def test_recorded_run09_skew_replay_rechecks_without_fabricating_pressure(monkeypatch):
    c, node, now, clock, messages, statuses = recheck_fixture(monkeypatch)
    record = load_run09()
    now.nanoseconds = record['evaluation_ros_ns']
    c.expected_model = node._target_book_model = record['expected_model']
    target = c._fine_command = record['commanded_position']
    c._fine_motion_end_ns = record['stationary_start_ros_ns']
    c._fine_micro_reference = record['reference_position']
    c._fine_micro_stationary_start_ns = record['stationary_start_ros_ns']
    node._gripper_feedback_samples.append(GripperFeedback(**record['feedback']))
    histories = {side: tuple(ForceSample(**s) for s in values)
                 for side, values in record['force_histories'].items()}
    node._adaptive_force_histories = lambda: (c.expected_model, histories['left'], histories['right'], None)
    c._fine_side_last = {side: (values[-1].stamp_ns, values[-1].force_newtons)
                         for side, values in histories.items()}
    before = c._micro_pressure_snapshot(1, target)
    assert before[0].reason == 'bilateral_sample_skew'
    assert before[0].left_samples == 32 and before[0].right_samples == 26
    assert before[1].reason == 'left_force_below_minimum' and not before[1].verified
    after = c._recheck_micro_pressure(1, target, before, wall_deadline=5.)
    assert after is not None and not after[1].verified
    assert c._fine_fault is None and not messages
    assert c._fine_micro_stationary_start_ns == record['stationary_start_ros_ns']
    assert c._fine_command == target
    assert len([f for _, f in statuses if f.get('stage') == 'pressure_measurement']) == 2


def test_same_target_recheck_can_verify_real_six_newton_history_without_command(monkeypatch):
    c, node, now, clock, messages, statuses = recheck_fixture(monkeypatch)
    book = c.expected_model+'::book_base_link::collision'
    def sensors():
        set_feedback(node, now, c._fine_command)
        node._on_contacts(force_message(now.nanoseconds, (LEFT, book, 6.5), (RIGHT, book, 6.5)))
    sensors()
    clock.on_sleep = sensors
    result = c._recheck_micro_pressure(1, c._fine_command, pair(), wall_deadline=5.)
    assert result is not None and result[1].verified
    assert result[1].left_samples >= 3 and result[1].right_samples >= 3
    assert not messages and c._fine_fault is None
    assert clock.wall <= .15
    measurement = next(f for _, f in statuses if f.get('stage') == 'pressure_measurement')
    assert measurement['pressure_windows']['left']['span_seconds'] >= .05
    assert measurement['pressure_windows']['right']['span_seconds'] >= .05
    assert measurement['retention_verified'] is False


def test_recheck_fixed_future_snapshot_waits_without_chasing_newer_frames(monkeypatch):
    c, node, now, clock, messages, statuses = recheck_fixture(monkeypatch)
    start = now.nanoseconds
    calls = []
    def histories():
        calls.append(now.nanoseconds)
        samples = tuple(ForceSample(now.nanoseconds+offset*1_000_000, 6.5)
                        for offset in range(18,81,2))
        return c.expected_model, samples, samples, None
    node._adaptive_force_histories = histories
    result = c._recheck_micro_pressure(1, c._fine_command, pair(), wall_deadline=5.)
    assert result is not None and result[1].verified
    assert len(calls) == 1  # no chasing the live producer stream
    assert now.nanoseconds-start == 130_000_000
    assert not messages and c._fine_fault is None


@pytest.mark.parametrize('change', ['epoch', 'clock_forward', 'clock_reverse'])
def test_recheck_rejects_epoch_or_clock_discontinuity(monkeypatch, change):
    c, _, now, clock, messages, _ = recheck_fixture(monkeypatch)
    sensors = clock.on_sleep
    def disturb():
        sensors()
        if change == 'epoch':
            c._fine_micro_stationary_start_ns += 1
        elif change == 'clock_forward':
            now.nanoseconds += 160_000_000
        else:
            now.nanoseconds -= 10_000_000
    clock.on_sleep = disturb
    c._micro_pressure_snapshot = lambda *a, **kw: pytest.fail('discontinuity evaluated')
    result = c._recheck_micro_pressure(1, c._fine_command, pair(), wall_deadline=5.)
    assert result is None and c._fine_fault
    assert all(m.points[0].positions[0] == .01825 for m in messages)


def test_recheck_does_not_count_unfinished_trajectory_as_stationary(monkeypatch):
    c, _, now, clock, messages, _ = recheck_fixture(monkeypatch)
    c._fine_motion_end_ns = now.nanoseconds+1
    result = c._recheck_micro_pressure(1, c._fine_command, pair(), wall_deadline=5.)
    assert result is None and c._fine_fault == 'joint_still_moving'
    assert clock.wall == 0.
    assert all(m.points[0].positions[0] == .01825 for m in messages)


def test_recheck_checks_live_deadline_after_fixed_snapshot_computation(monkeypatch):
    c, _, now, clock, messages, _ = recheck_fixture(monkeypatch)
    def delayed(*args, **kwargs):
        now.nanoseconds += 110_000_000
        clock.on_sleep()
        return pair()
    c._micro_pressure_snapshot = delayed
    result = c._recheck_micro_pressure(1, c._fine_command, pair(), wall_deadline=5.)
    assert result is None and c._fine_fault == 'micro_pressure_recheck_timeout'
    assert all(m.points[0].positions[0] == .01825 for m in messages)
