"""Actual contact callbacks and opening handoffs; no physical success claim."""
from types import SimpleNamespace as NS

import pytest

from erc_phase1_solution import mission_manager as mission
from erc_phase1_solution import release_pose_finish as finish
from erc_phase1_solution.release_evidence import AttemptIdentity, ReleaseEvidence
from erc_phase1_solution.release_pose_evidence import PendingContactEvents
from test_release_contact_events import mission_fixture, status, stream
from test_release_pose_finish import (
    fixture, record_feedback, real_open_measurement, contact_message,
    IDENTITY, BOOK, BIN, ROBOT,
)

OTHER = 'unrelated_model::link::collision'
FLOOR = 'ground_plane::link::collision'
TABLE = 'erc_table::table_link::collision'


def measured_open(n, stamp):
    status(n, dict(vars(IDENTITY), event='placement_open_measured',
        command='place', producer_stamp_ns=stamp, verified=True))


@pytest.mark.parametrize('vary_stamps', [False, True])
def test_actual_mission_dense_irrelevant_events_do_not_fill_pending_history(vary_stamps):
    n,evidence=mission_fixture(opened=False)
    for i in range(5000):
        n.now=1_000_000_000+(i if vary_stamps else 0)
        mission.MissionManager._on_contacts(n,contact_message(n.now,[(OTHER,FLOOR)]))
    pending=evidence.pending_contact_events
    assert len(pending)==0 and len(pending.last_stamps)==1
    assert evidence.fault is None
    n.now=1_100_000_000;measured_open(n,1_050_000_000)
    assert evidence.open_epoch==1_050_000_000 and evidence.fault is None
    assert not evidence.last_received_stamps
    assert not evidence.evaluate(n.now,1.)['operation_completed']


@pytest.mark.parametrize('kind', ['invalid_stamp','malformed'])
def test_actual_mission_unknown_head_is_latched_separately_from_later_noise(kind):
    n,evidence=mission_fixture(opened=False)
    message=contact_message(0 if kind=='invalid_stamp' else n.now,[(OTHER,FLOOR)])
    if kind=='malformed':message.contacts=None
    try:mission.MissionManager._on_contacts(n,message)
    except (TypeError,AttributeError):pass
    for i in range(5000):
        n.now=2_000_000_000+i
        mission.MissionManager._on_contacts(n,contact_message(n.now,[(OTHER,FLOOR)]))
    assert len(evidence.pending_contact_events)==1
    n.now=2_100_000_000;measured_open(n,2_050_000_000)
    expected=('release_pose_contact_stamp_invalid' if kind=='invalid_stamp'
              else 'release_pose_contacts_malformed')
    assert evidence.fault==expected
    assert evidence.pending_contact_events.fault_witness['reason']==expected
    assert not evidence.evaluate(n.now,1.)['operation_completed']


@pytest.mark.parametrize('relative',[-1,0,1])
@pytest.mark.parametrize('negative_first',[False,True])
def test_actual_mission_epoch_boundary_and_benign_noise_cannot_replace_veto(relative,negative_first):
    n,evidence=mission_fixture(opened=False);epoch=1_050_000_000;n.now=1_100_000_000
    bad=contact_message(epoch+relative,[(BOOK,ROBOT)])
    good=contact_message(epoch+relative,[(OTHER,FLOOR)])
    ordered=(bad,good) if negative_first else (good,bad)
    for message in ordered:
        for _ in range(1500):mission.MissionManager._on_contacts(n,message)
    assert len(evidence.pending_contact_events)==1
    measured_open(n,epoch)
    assert (evidence.fault is None)==(relative<=0)
    if relative>0:
        assert evidence.fault=='release_pose_target_robot_contact'
        witness=evidence.pending_contact_events.fault_witness
        assert witness['stamp_ns']==epoch+relative and witness['pair']==(BOOK,ROBOT)
    assert not evidence.evaluate(n.now,1.)['operation_completed']


def test_greatest_actual_witness_keeps_its_pair_and_attempt_identity():
    pending=PendingContactEvents(IDENTITY)
    left=(BOOK,ROBOT)
    right=(BOOK,ROBOT.replace('left','right'))
    pending.observe(990_000_000,1_100_000_000,[left],'/contacts')
    pending.observe(1_050_000_000,1_100_000_000,[right],'/contacts')
    pending.observe(1_090_000_000,1_100_000_000,[(OTHER,FLOOR)],'/contacts')
    release=ReleaseEvidence(IDENTITY);release.open_epoch=1_000_000_000
    pending.resolve(release,release.invalidate,{},1_100_000_000)
    assert release.fault=='release_pose_target_robot_contact'
    assert pending.fault_witness['pair']==right and pending.identity is IDENTITY
    newer=PendingContactEvents(AttemptIdentity('other','new',IDENTITY.target_model))
    assert not newer and newer.fault_witness is None


@pytest.mark.parametrize('kind',['future','reversed','pair_invalid'])
@pytest.mark.parametrize('post_open',[False,True])
def test_stamped_received_faults_are_classified_against_eventual_epoch(kind,post_open):
    n,evidence=mission_fixture(opened=False);n.now=1_100_000_000
    if kind=='future':
        bad_stamp=1_200_000_001
    else:
        bad_stamp=1_050_000_000
    if kind=='reversed':
        mission.MissionManager._on_contacts(n,contact_message(bad_stamp+1,[(OTHER,FLOOR)]))
    message=contact_message(bad_stamp,[(OTHER,FLOOR)])
    if kind=='pair_invalid':message.contacts[0].collision1.name=''
    mission.MissionManager._on_contacts(n,message)
    epoch=bad_stamp-1 if post_open else bad_stamp
    n.now=max(n.now,epoch);measured_open(n,epoch)
    expected={'future':'release_pose_contact_not_fresh',
        'reversed':'release_pose_contact_stamp_reversed',
        'pair_invalid':'release_pose_contact_pair_invalid'}[kind]
    assert evidence.fault==(expected if post_open else None)
    assert not evidence.evaluate(n.now,1.)['operation_completed']


def test_stale_preopening_event_does_not_poison_newer_valid_open_epoch():
    n,evidence=mission_fixture(opened=False);n.now=1_400_000_000
    mission.MissionManager._on_contacts(n,contact_message(1_000_000_000,[(OTHER,FLOOR)]))
    measured_open(n,1_350_000_000)
    assert evidence.fault is None
    # The same received-event age remains a refusal after opening is known.
    n.now=1_700_000_001
    mission.MissionManager._on_contacts(n,contact_message(1_450_000_000,[(OTHER,FLOOR)]))
    assert evidence.fault=='release_pose_contact_not_fresh'


def test_actual_mission_positive_window_survives_dense_preopening_noise(monkeypatch):
    n,owner=fixture(monkeypatch);stream(n,owner)
    assert finish.finish(n,owner) and finish.publish_terminal(n,vars(IDENTITY))
    m,evidence=mission_fixture(opened=False)
    for _ in range(5000):
        mission.MissionManager._on_contacts(m,contact_message(m.now,[(OTHER,FLOOR)]))
    measured_open(m,m.now)
    for stamp in range(1_050_000_000,1_600_000_001,50_000_000):
        m.now=stamp
        mission.MissionManager._on_bin_contacts(m,contact_message(stamp,[(BOOK,BIN)]))
    for event,fields in n.events:status(m,dict(fields,event=event))
    result=evidence.evaluate(m.now,1.)
    assert result['operation_completed'] and result['contact_samples']==12
    assert result['physical_inside_verified'] is None
    assert result['contact_stream_coverage_verified'] is result['hand_clear_verified'] is False


@pytest.mark.parametrize('hazard',[False,True])
def test_actual_manipulation_open_publication_handoff_uses_same_summary(monkeypatch,hazard):
    n,owner=fixture(monkeypatch,opened=False)
    def during_publication(fields):
        epoch=fields['producer_stamp_ns'];n.now=epoch+1
        for _ in range(5000):
            finish.observe_contacts(n,contact_message(n.now,[(OTHER,FLOOR)]),'/contacts')
        if hazard:
            finish.observe_contacts(n,contact_message(n.now,[(BOOK,ROBOT)]),'/contacts')
    fields=real_open_measurement(n,monkeypatch,publication_callback=during_publication)
    assert finish.remember_open(n,owner,fields) is (not hazard)
    assert n._release_pose_owner is owner and not n.commands
    assert n._cancel.is_set() is hazard
    assert not any(event in ('placement_release_pose_measured','succeeded') for event,_ in n.events)
    if hazard:
        assert owner.evidence.fault=='release_pose_target_robot_contact'
        assert owner.opening_contacts.fault_witness['pair']==(BOOK,ROBOT)


def test_actual_manipulation_invalid_stamp_cannot_pin_or_escape_handoff(monkeypatch):
    n,owner=fixture(monkeypatch,opened=False)
    finish.observe_contacts(n,contact_message(0,[(OTHER,FLOOR)]),'/contacts')
    for _ in range(5000):
        finish.observe_contacts(n,contact_message(n.now,[(OTHER,FLOOR)]),'/contacts')
    fields=real_open_measurement(n,monkeypatch)
    assert not finish.remember_open(n,owner,fields)
    assert owner.evidence.fault=='release_pose_contact_stamp_invalid'
    assert n._cancel.is_set() and n._release_pose_owner is owner and not n.commands


def test_duplicate_positive_events_do_not_consume_unique_history():
    pending=PendingContactEvents(IDENTITY)
    for stamp in range(1_050_000_000,1_600_000_001,50_000_000):
        for _ in range(1100):pending.observe(stamp,stamp,[(BOOK,BIN)],'/bin_contacts')
    assert len(pending.bin_stamps)==12 and pending.unknown is None
    release=ReleaseEvidence(IDENTITY);release.open_epoch=1_550_000_000
    pending.resolve(release,release.invalidate,{},1_600_000_000)
    assert release.contact_stamps=={1_600_000_000} and release.fault is None
    assert not release.evaluate(1_600_000_000,1.)['operation_completed']


def test_positive_unique_history_limit_is_not_relaxed():
    pending=PendingContactEvents(IDENTITY)
    for i in range(1025):
        pending.observe(1_050_000_000+i,1_100_000_000,[(BOOK,BIN)],'/bin_contacts')
    release=ReleaseEvidence(IDENTITY);release.open_epoch=1_000_000_000
    pending.resolve(release,release.invalidate,{},1_100_000_000)
    assert release.fault=='contact_history_rate_overflow' and not release.contact_stamps


@pytest.mark.parametrize('fault',['robot','reversed','invalid_stamp'])
def test_diagnostic_ring_eviction_never_erases_received_fault_checks(monkeypatch,fault):
    n,owner=fixture(monkeypatch)
    for i in range(1100):
        n.now=1_000_000_001+i
        finish.observe_contacts(n,contact_message(n.now,[(OTHER,FLOOR)]),'/contacts')
    assert owner.evidence.fault is None and not n._cancel.is_set()
    assert len(owner.evidence.raw_contact_frames)==1024
    assert owner.evidence.raw_contact_history_truncated
    stamp=0 if fault=='invalid_stamp' else n.now-1 if fault=='reversed' else n.now+1
    pairs=[(BOOK,ROBOT)] if fault=='robot' else [(OTHER,FLOOR)]
    finish.observe_contacts(n,contact_message(stamp,pairs),'/contacts')
    expected={'robot':'release_pose_target_robot_contact',
        'reversed':'release_pose_contact_stamp_reversed',
        'invalid_stamp':'release_pose_contact_stamp_invalid'}[fault]
    assert owner.evidence.fault==expected and n._cancel.is_set()
    assert n._release_pose_owner is owner and not n.commands and not n.events


def test_expired_diagnostic_ring_prunes_before_append(monkeypatch):
    n,owner=fixture(monkeypatch)
    for i in range(1024):
        n.now=1_000_000_001+i
        finish.observe_contacts(n,contact_message(n.now,[(OTHER,FLOOR)]),'/contacts')
    n.now=2_000_000_000
    finish.observe_contacts(n,contact_message(n.now,[(OTHER,FLOOR)]),'/contacts')
    assert list(owner.evidence.raw_contact_frames)==[n.now]
    assert not owner.evidence.raw_contact_history_truncated
    assert owner.evidence.fault is None


def test_actual_mission_benign_flood_spans_delayed_open_epoch():
    n,evidence=mission_fixture(opened=False);epoch=1_050_000_000
    for stamp in (epoch-1,epoch,epoch+1):
        n.now=1_100_000_000
        for _ in range(1600):
            mission.MissionManager._on_contacts(n,contact_message(stamp,[(OTHER,FLOOR)]))
    measured_open(n,epoch)
    assert evidence.fault is None and not evidence.pending_contact_events
    assert evidence.last_received_stamps=={'/contacts':epoch+1}
    # The summary transfers ordering, not just the absence of a veto.
    mission.MissionManager._on_contacts(n,contact_message(epoch,[(BOOK,ROBOT)]))
    assert evidence.fault is None  # original <= epoch exemption
    mission.MissionManager._on_contacts(n,contact_message(epoch+2,[(OTHER,FLOOR)]))
    mission.MissionManager._on_contacts(n,contact_message(epoch+1,[(OTHER,FLOOR)]))
    assert evidence.fault=='release_pose_contact_stamp_reversed'


@pytest.mark.parametrize('damage',['clock_reversal','invalid_clock','invalid_sources'])
def test_summary_unknown_clock_and_fixed_source_namespace_fail_closed(damage):
    pending=PendingContactEvents(IDENTITY)
    pending.observe(1_050_000_000,1_100_000_000,[(OTHER,FLOOR)],'/contacts')
    if damage=='invalid_sources':
        for i in range(2000):
            pending.observe(1_050_000_001,1_100_000_000,[(OTHER,FLOOR)],'/unknown_'+str(i))
        assert pending.unknown is not None and not pending.witnesses
        expected='release_pose_contact_source_invalid'
    else:
        now=1_099_999_999 if damage=='clock_reversal' else 0
        pending.observe(1_050_000_001,now,[(OTHER,FLOOR)],'/contacts')
        expected='invalid_or_reversed_clock'
    release=ReleaseEvidence(IDENTITY);release.open_epoch=1_000_000_000
    pending.resolve(release,release.invalidate,{},1_100_000_000)
    assert release.fault==expected
    release.invalidate('later_cannot_replace_first_fault')
    assert release.fault==expected


def test_fresh_capture_and_exact_open_receipt_bound_preserve_pending_bin_stamp():
    n,evidence=mission_fixture(opened=False);epoch=1_000_000_000
    n.now=1_050_000_000
    mission.MissionManager._on_bin_contacts(n,contact_message(epoch+1,[(BOOK,BIN)]))
    n.now=epoch+250_000_000;measured_open(n,epoch)
    assert evidence.fault is None and evidence.contact_stamps=={epoch+1}
    assert not evidence.evaluate(n.now,1.)['operation_completed']
    # The original opening receipt limit itself remains exact and mandatory.
    other,invalid=mission_fixture(opened=False);other.now=epoch+250_000_001
    measured_open(other,epoch)
    assert invalid.fault=='invalid_measured_open'


def test_pre_epoch_invalid_source_preserves_original_classifier_guard():
    from erc_phase1_solution.release_pose_evidence import received_contact_event
    epoch=1_100_000_000;stamp=1_050_000_000
    reason,_=received_contact_event(IDENTITY,epoch,None,stamp,epoch,[(OTHER,FLOOR)],'/invalid')
    assert reason=='release_pose_contact_source_invalid'
    pending=PendingContactEvents(IDENTITY)
    pending.observe(stamp,epoch,[(OTHER,FLOOR)],'/invalid')
    release=ReleaseEvidence(IDENTITY);release.open_epoch=epoch
    pending.resolve(release,release.invalidate,{},epoch)
    assert release.fault==reason and not release.contact_stamps
