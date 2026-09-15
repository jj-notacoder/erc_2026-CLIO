"""Actual callback and failure-handler timing; diagnostics cannot admit motion."""
import json
from types import SimpleNamespace as NS
from erc_phase1_solution import release_sensor_adapter as adapter

import pytest

from erc_phase1_solution import release_pose_finish as finish
from erc_phase1_solution import place_contact_guard as contacts
from test_release_pose_finish import (
    fixture, record_feedback, contact_message, ready, command_fixture,
    actual_methods, IDENTITY, BOOK, BIN,
)


def held_at_boundary(monkeypatch, stale):
    n, owner = fixture(monkeypatch)
    for stamp in (1_050_000_000, 1_100_000_000, 1_150_000_000):
        n.refresh(stamp)
        header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9))
        if stale not in ('odom', 'both'):
            adapter.record_raw_odom(n,NS(header=header,twist=NS(twist=NS(
                linear=NS(x=0.,y=0.),angular=NS(z=0.)))))
        if stale not in ('joint', 'both'):
            names=list(n.joints)
            adapter.record_raw_joints(n,NS(header=header,name=names,
                position=[n.joints[name] for name in names],velocity=[0.]*len(names)))
        contacts.observe(n, contact_message(stamp, [(BOOK, BIN)]), '/bin_contacts')
        # No raw contact event is produced: silence cannot supply feedback.
    return n, owner


@pytest.mark.parametrize('stale', ['joint', 'odom', 'both'])
def test_each_actual_callback_branch_keeps_exact_150ms_refusal_and_records_operands(monkeypatch, stale):
    n, owner = held_at_boundary(monkeypatch, stale)
    assert finish._admit(n, owner) is False  # exactly150ms is unchanged, not ready
    assert not n._cancel.is_set() and not n.events
    n.now += 1
    with pytest.raises(finish.ReleasePoseFeedbackStale) as captured:
        finish.finish(n, owner)
    error = captured.value
    assert str(error) == 'release_pose_hold_feedback_stale'
    details = error.details
    assert details['now_ns'] == 1_150_000_001
    assert details['open_epoch_ns'] == 1_000_000_000
    assert details['joint_stale'] is (stale in ('joint', 'both'))
    assert details['odom_stale'] is (stale in ('odom', 'both'))
    assert details['raw_contact_stale'] is True
    assert details['joint_age_ns'] == (150_000_001 if details['joint_stale'] else 1)
    assert details['raw_contact_age_ns'] == 150_000_001
    assert details['odom_age_ns'] == (150_000_001 if details['odom_stale'] else 1)
    assert details['latest_joint_stamp_ns'] == (1_000_000_000 if details['joint_stale'] else 1_150_000_000)
    assert details['latest_raw_contact_stamp_ns'] is None
    assert details['freshness_limit_ns'] == 150_000_000
    assert details['raw_sample_sequence'] == (1 if details['joint_stale'] else 4)
    assert details['raw_contact_frame_count'] == 0
    assert details['pending_acceptance_count'] == details['active_goal_count'] == 0
    assert details['evidence_fault'] is details['fault_latched'] is None
    assert details['owner_phase'] == 'opened' and details['cancelled'] is False
    assert owner.evidence.release.contact_stamps == {1_050_000_000, 1_100_000_000, 1_150_000_000}
    assert json.loads(json.dumps(details, allow_nan=False)) == details
    details['now_ns'] = -1
    assert error.details['now_ns'] == 1_150_000_001
    assert n._cancel.is_set() and n._release_pose_owner is owner and not n.events
    finish.deactivate(n)
    assert n._release_pose_owner is None and n._release_pose_fault_latched == 'release_pose_hold_failed'


@pytest.mark.parametrize('stale', ['joint', 'odom', 'both'])
def test_actual_command_failure_surfaces_frozen_details_outside_lock_without_success(monkeypatch, stale):
    n, owner = held_at_boundary(monkeypatch, stale)
    n.now += 1
    command_fixture(n)
    n._place = lambda payload: finish.finish(n, owner)
    publish = n._publish_status
    def observed(event, **fields):
        if event == 'failed':
            assert n._lock.acquire(blocking=False)
            n._lock.release()
            assert fields['reason'] == 'release_pose_hold_feedback_stale'
            assert fields['release_pose_feedback']['joint_stale'] is (stale in ('joint', 'both'))
            assert fields['release_pose_feedback']['odom_stale'] is (stale in ('odom', 'both'))
            assert n._release_pose_owner is owner
        publish(event, **fields)
    n._publish_status = observed
    actual_methods()['_run_command'](n, 'place', dict(vars(IDENTITY), completion_mode='release_pose',release_evidence_contract=finish.CONTRACT))
    assert [event for event, fields in n.events] == ['started', 'failed']
    assert n._cancel.is_set() and n._release_pose_owner is None and not n._busy
    assert n._release_pose_fault_latched == 'release_pose_hold_failed'
    assert not n.commands


def test_diagnostic_publication_error_preserves_original_fault_and_owned_cleanup(monkeypatch):
    n, owner = held_at_boundary(monkeypatch, 'odom')
    n.now += 1
    command_fixture(n)
    n._place = lambda payload: finish.finish(n, owner)
    publish = n._publish_status
    def broken(event, **fields):
        if event == 'failed':
            assert fields['release_pose_feedback']['odom_stale']
            raise ValueError('failed status transport unavailable')
        publish(event, **fields)
    n._publish_status = broken
    with pytest.raises(finish.ReleasePoseFeedbackStale, match='^release_pose_hold_feedback_stale$') as captured:
        actual_methods()['_run_command'](n, 'place', dict(vars(IDENTITY), completion_mode='release_pose',release_evidence_contract=finish.CONTRACT))
    assert isinstance(captured.value.__cause__, ValueError)
    assert [event for event, fields in n.events] == ['started']
    assert n._cancel.is_set() and n._release_pose_owner is None and not n._busy
    assert n._release_pose_fault_latched == 'release_pose_hold_failed'


def test_fresh_actual_callback_hold_completion_has_no_diagnostic_field(monkeypatch):
    n, owner = fixture(monkeypatch)
    ready(n, owner)
    command_fixture(n)
    actual_methods()['_run_command'](n, 'place', dict(vars(IDENTITY), completion_mode='release_pose',release_evidence_contract=finish.CONTRACT))
    assert [event for event, fields in n.events] == ['started', 'placement_release_pose_measured', 'succeeded']
    assert not any('release_pose_feedback' in fields for event, fields in n.events)
    assert n._release_pose_fault_latched is None and not n._cancel.is_set()
    assert not n.commands


def test_unrelated_actual_place_exception_keeps_existing_reason_and_no_details(monkeypatch):
    n, owner = fixture(monkeypatch)
    command_fixture(n)
    def fail(payload):
        raise RuntimeError('ordinary_place_failure')
    n._place = fail
    actual_methods()['_run_command'](n, 'place', dict(vars(IDENTITY), completion_mode='release_pose',release_evidence_contract=finish.CONTRACT))
    assert [event for event, fields in n.events] == ['started', 'failed']
    assert n.events[-1][1]['reason'] == 'ordinary_place_failure'
    assert 'release_pose_feedback' not in n.events[-1][1]
    assert n._release_pose_fault_latched == 'release_pose_hold_failed'
