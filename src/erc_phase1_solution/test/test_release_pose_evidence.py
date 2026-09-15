"""Attempt policy cannot be switched by a status or renamed return event."""
import pytest

from erc_phase1_solution.release_evidence import AttemptIdentity, ReleaseEvidence
from erc_phase1_solution.release_pose_evidence import (
    EVENT, MODE, ReleasePoseEvidence, checked_enabled, contract_fields,
)


def fixture(*, opened=True, contacts=True, finish=True, terminal=True):
    identity = AttemptIdentity('trial', 'attempt', 'book_col_3_row_2_red')
    evidence = ReleasePoseEvidence(identity)
    fields = vars(identity)
    opening = dict(fields, event='placement_open_measured', verified=True,
                   producer_stamp_ns=1_000_000_000)
    if opened:
        assert evidence.opening(opening, 1_000_000_000)
    if contacts:
        for stamp in range(1_100_000_000, 1_600_000_001, 100_000_000):
            evidence.contact(stamp, stamp, exact_target_bin_pair=True)
    measured = dict(fields, event=EVENT, command='place', completion_mode=MODE,
                    verified=True, hand_return=False,
                    **contract_fields(), release_measurement_stamp_ns=1_600_000_000,
                    odom_producer_stamp_ns=1_600_000_000,
                    bin_contact_start_ns=1_100_000_000, bin_contact_last_ns=1_600_000_000,
                    bin_contact_samples=6, bin_contact_max_gap_ns=100_000_000,
                    producer_stamp_ns=1_600_000_000,
                    stationary_start_ns=1_450_000_000)
    if finish:
        assert evidence.finish_pose(measured, 1_600_000_000)
    if terminal:
        evidence.motion_terminal(dict(fields, event='succeeded', command='place',
                                      completion_mode=MODE, hand_return=False,
                                      release_terminal_stamp_ns=1_600_000_000, **contract_fields()),
                                 1_600_000_000, 10.)
    return evidence, measured


def test_explicit_finish_is_not_hand_return_or_containment():
    evidence, _ = fixture()
    result = evidence.evaluate(1_600_000_000, 10.)
    assert result['status'] == 'operation_completed'
    assert result['release_pose_measured'] is True
    assert result['hand_return'] is result['hand_return_measured'] is False
    assert result['hand_clear_verified'] is result['containment_verified'] is False
    assert result['physical_delivery_verified'] is False
    assert result['physical_inside_verified'] is None
    assert evidence.return_measurement is None


@pytest.mark.parametrize('missing', ['contacts', 'finish', 'terminal'])
def test_every_original_and_new_completion_condition_is_required(missing):
    evidence, _ = fixture(**{missing: False})
    assert evidence.evaluate(1_600_000_000, 10.)['operation_completed'] is False


@pytest.mark.parametrize('field,value', [
    ('event', 'placement_hand_return_measured'), ('completion_mode', 'hand_return'),
    ('verified', False), ('hand_return', True),
    ('no_observed_target_robot_contact_verified', False),
    ('hand_clear_verified', True), ('physical_inside_verified', True),
    ('producer_stamp_ns', None), ('stationary_start_ns', 1_550_000_000),
    ('stationary_start_ns', 1_000_000_000),
])
def test_false_or_unproven_milestones_cannot_complete(field, value):
    evidence, measured = fixture(finish=False)
    measured[field] = value
    assert not evidence.finish_pose(measured, 1_600_000_000)
    assert evidence.evaluate(1_600_000_000, 10.)['status'] == 'invalid'


@pytest.mark.parametrize('field', ['trial_id', 'placement_attempt_id', 'target_model'])
def test_cross_attempt_measurement_is_ignored(field):
    evidence, measured = fixture(finish=False)
    measured[field] += '_other'
    assert not evidence.finish_pose(measured, 1_600_000_000)
    assert evidence.evaluate(1_600_000_000, 10.)['status'] == 'pending'


def test_ordinary_attempt_still_requires_actual_hand_return():
    _, measured = fixture()
    normal = ReleaseEvidence(AttemptIdentity('trial', 'attempt', 'book_col_3_row_2_red'))
    assert not hasattr(normal, 'finish_pose')
    assert not normal.hand_return(measured, 1_600_000_000)
    assert normal.evaluate(1_600_000_000, 10.)['operation_completed'] is False


def test_hand_return_cannot_satisfy_release_pose_mode():
    evidence, measured = fixture(finish=False)
    measured['event'] = 'placement_hand_return_measured'
    assert not evidence.hand_return(measured, 1_600_000_000)
    assert evidence.return_measurement is None
    assert evidence.evaluate(1_600_000_000, 10.)['status'] == 'invalid'


@pytest.mark.parametrize('stamps', [
    (1_100_000_000, 1_200_000_000, 1_600_000_000),
    (1_400_000_000, 1_500_000_000, 1_600_000_000),
    (1_600_000_000,) * 10,
])
def test_duplicate_short_or_gapped_contacts_keep_original_gate(stamps):
    evidence, _ = fixture(contacts=False)
    for stamp in stamps:
        evidence.contact(stamp, 1_600_000_000, exact_target_bin_pair=True)
    assert evidence.evaluate(1_600_000_000, 10.)['operation_completed'] is False


@pytest.mark.parametrize('value', [0, 1, None, 'true', [], {}])
def test_flag_is_typed_boolean(value):
    with pytest.raises(ValueError):
        checked_enabled(value)


def test_mode_has_no_setter_and_terminal_deadline_remains_bounded():
    evidence, _ = fixture(finish=False)
    with pytest.raises(AttributeError):
        evidence.completion_mode = 'hand_return'
    assert evidence.evaluate(1_600_000_000, 41.)['status'] == 'expired'
