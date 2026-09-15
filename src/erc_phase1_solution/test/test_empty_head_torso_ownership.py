"""Actual composed sender boundaries with synthetic feedback; no physics proof."""
import pytest

from erc_phase1_solution.motion_profiles import IK_JOINTS
import test_empty_torso_planning_overlap as torso
import test_empty_head_timing as head


def test_two_explicit_owners_refuse_before_either_owner_or_sender_runs(monkeypatch):
    n, checker, owner, clock, sent, *_ = torso.fixture(monkeypatch)
    n.empty_head_timing_enabled = True
    with pytest.raises(ValueError, match='mutually exclusive'):
        n._follow(n.torso_client, [IK_JOINTS[0]], [.35], 2.5,
                  empty_torso_owner=owner, empty_head_owner=object())
    assert sent == [] and owner.follow_token is None and not owner.dispatched
    assert not n._pending_retained_acceptances and not n._goal_handles
    assert n._empty_torso_planning_owner is owner


def test_foreign_head_owner_appearing_at_torso_locked_admission_refuses_send(monkeypatch):
    foreign = object()
    def install(n, goal):
        assert n._lock.locked()
        n._empty_head_timing_owner = foreign
    n, checker, owner, clock, sent, *_ = torso.fixture(monkeypatch, locked_hook=install)
    n.empty_head_timing_enabled = True
    with pytest.raises(ValueError, match='head owner exists'):
        n._move_torso(.35, 2.5, empty_torso_owner=owner)
    assert sent == [] and not owner.dispatched
    assert n._empty_head_timing_owner is foreign and n._empty_torso_planning_owner is owner
    assert not n._pending_retained_acceptances and not n._goal_handles


def test_foreign_torso_owner_appearing_at_head_locked_admission_refuses_send(monkeypatch):
    n, owner, clock, sent, *_ = head.fixture(monkeypatch)
    n.empty_torso_planning_overlap_enabled = True
    foreign = object()
    admitted = owner.admit_locked
    def install(client, goal):
        assert n.command.depth == n._lock.depth == 1
        n._empty_torso_planning_owner = foreign
        return admitted(client, goal)
    owner.admit_locked = install
    with pytest.raises(RuntimeError, match='ownership or physical context'):
        n._move_head(*owner.target, empty_head_timing=owner)
    assert sent == [] and not owner.dispatched
    assert n._empty_torso_planning_owner is foreign and n._empty_head_timing_owner is owner
    assert not n._pending_retained_acceptances and not n._goal_handles


def test_torso_path_with_both_flags_keeps_exact_goal_and_action_ownership(monkeypatch):
    n, checker, owner, clock, sent, *_ = torso.fixture(monkeypatch)
    n.empty_head_timing_enabled = True
    n._empty_head_timing_owner = None
    assert n._move_torso(.35, 2.5, empty_torso_owner=owner)
    assert len(sent) == 1
    point = sent[0].trajectory.points[0]
    assert tuple(sent[0].trajectory.joint_names) == (IK_JOINTS[0],)
    assert tuple(point.positions) == (.35,)
    assert (point.time_from_start.sec, point.time_from_start.nanosec) == (2, 500_000_000)
    assert n._empty_torso_planning_owner is owner and not owner.measured_stopped
    assert n._empty_head_timing_owner is None
    assert not n._pending_retained_acceptances and not n._goal_handles


def test_head_path_with_both_flags_keeps_own_fresh_stop_and_cannot_release_torso(monkeypatch):
    n, owner, clock, sent, *_ = head.fixture(monkeypatch)
    n.empty_torso_planning_overlap_enabled = True
    n._empty_torso_planning_owner = None
    assert n._move_head(*owner.target, empty_head_timing=owner)
    assert len(sent) == 1 and owner.completion['empty_head_measured_stop'] is True
    assert (sent[0].trajectory.points[0].time_from_start.sec,
            sent[0].trajectory.points[0].time_from_start.nanosec) == (0, 600_000_000)
    assert n._empty_torso_planning_owner is None and n._empty_head_timing_owner is None
    assert not n._pending_retained_acceptances and not n._goal_handles


@pytest.mark.parametrize('kind', ['torso', 'head'])
def test_combined_sender_still_rechecks_fresh_start_after_server_wait(monkeypatch, kind):
    if kind == 'torso':
        n, checker, owner, clock, sent, *_ = torso.fixture(monkeypatch)
        n.empty_head_timing_enabled = True
        def wait(**unused):
            n._joint_stamps_ns[IK_JOINTS[0]] -= 350_000_001
            return True
        n.torso_client.wait_for_server = wait
        invoke = lambda: n._move_torso(.35, 2.5, empty_torso_owner=owner)
    else:
        n, owner, clock, sent, *_ = head.fixture(monkeypatch)
        n.empty_torso_planning_overlap_enabled = True
        def wait(**unused):
            n._joint_stamps_ns[head.timing.HEAD[0]] -= 150_000_001
            return True
        n.head_client.wait_for_server = wait
        invoke = lambda: n._move_head(*owner.target, empty_head_timing=owner)
    with pytest.raises(RuntimeError):
        invoke()
    assert not sent and not n._pending_retained_acceptances and not n._goal_handles


def test_both_flags_do_not_retime_unmarked_ordinary_head_call(monkeypatch):
    n, unused_owner, clock, sent, *_ = head.fixture(monkeypatch)
    n.empty_torso_planning_overlap_enabled = True
    n._empty_torso_planning_owner = None
    assert n._move_head(0., .2)
    assert len(sent) == 1
    point = sent[0].trajectory.points[0]
    assert (point.time_from_start.sec, point.time_from_start.nanosec) == (1, 200_000_000)
    assert n._empty_head_timing_owner is None and n._empty_torso_planning_owner is None
    assert not unused_owner.dispatched and unused_owner.completion is None
