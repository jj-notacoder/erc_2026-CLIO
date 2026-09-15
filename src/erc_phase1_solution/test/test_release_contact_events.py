"""Nonempty-only producer contract through actual callbacks; no physical claim."""
import json
from types import SimpleNamespace as NS

import pytest

from erc_phase1_solution import release_pose_finish as finish
from erc_phase1_solution import place_contact_guard as contacts
from erc_phase1_solution import mission_manager as mission
from erc_phase1_solution import release_sensor_adapter as adapter
from erc_phase1_solution.release_pose_evidence import (
    ReleasePoseEvidence, CONTRACT, contract_fields,
)
from test_release_pose_finish import (
    fixture, record_feedback, contact_message, IDENTITY, BOOK, BIN, ROBOT,
)


def stream(n, owner, *, topic='/bin_contacts', pairs=None, contact=True):
    start=n.now
    for stamp in range(start+50_000_000,start+600_000_001,50_000_000):
        record_feedback(n,stamp)
        if contact:
            contacts.observe(n,contact_message(stamp,[(BOOK,BIN)] if pairs is None else pairs),topic)
    return n.now


def mission_fixture(*, opened=True):
    evidence=ReleasePoseEvidence(IDENTITY)
    n=NS(delivery_release_evidence=evidence,delivery_evidence_enabled=True,
         place_finish_at_release_enabled=True,target_book_model=IDENTITY.target_model,
         target_colour='red',state='PLACE',now=1_000_000_000,
         collision_episodes=0,finished=False,events=[],bin_contact_confirmed=False,
         first_target_bin_contact=None)
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=n.now))
    n._expected_target_book_model=lambda:IDENTITY.target_model
    n._contact_pairs=mission.MissionManager._contact_pairs
    n.contact_tracker=NS(observe=lambda *a:[],active_pairs=lambda *a:set())
    n._log=lambda event,**fields:n.events.append((event,fields))
    n._record_first_target_bin_contact=lambda *a:None
    if opened:
        status(n,dict(vars(IDENTITY),event='placement_open_measured',command='place',
                      producer_stamp_ns=n.now,verified=True))
    return n,evidence


def status(n,payload):
    mission.MissionManager._on_manipulation_status(n,NS(data=json.dumps(payload)))


def assert_qualified(payload):
    assert all(payload[k]==v and type(payload[k]) is type(v) for k,v in contract_fields().items())
    assert payload['contact_stream_coverage_verified'] is payload['detachment_verified'] is False
    assert not any(k in payload for k in ('raw_contact_start_ns','raw_contact_last_ns',
                                         'no_observed_target_robot_contact_verified'))


def test_actual_bin_only_nonempty_callbacks_complete_honest_locked_operation(monkeypatch):
    n,owner=fixture(monkeypatch);stream(n,owner)
    assert owner.evidence.last_raw_contact_stamp is None
    assert len(owner.evidence.raw_contact_frames)==0
    assert finish.finish(n,owner)
    assert finish.publish_terminal(n,vars(IDENTITY))
    measured=dict(n.events[0][1],event=n.events[0][0])
    terminal=dict(n.events[1][1],event=n.events[1][0])
    assert_qualified(measured);assert_qualified(terminal)
    assert measured['release_measurement_stamp_ns']==terminal['release_terminal_stamp_ns']==n.now
    assert measured['bin_contact_samples']==12
    assert measured['bin_contact_last_ns']-measured['bin_contact_start_ns']==550_000_000
    assert measured['bin_contact_max_gap_ns']==50_000_000
    assert not n.commands
    m,evidence=mission_fixture()
    for stamp in range(1_050_000_000,1_600_000_001,50_000_000):
        m.now=stamp
        mission.MissionManager._on_bin_contacts(m,contact_message(stamp,[(BOOK,BIN)]))
    status(m,measured);status(m,terminal)
    result=evidence.evaluate(m.now,1.)
    assert result['operation_completed'] and result['hand_return'] is False
    assert_qualified(result)


@pytest.mark.parametrize('topic',['/contacts','/table_contacts'])
def test_positive_pair_on_wrong_topic_cannot_supply_bin_window(monkeypatch,topic):
    n,owner=fixture(monkeypatch);stream(n,owner,topic=topic)
    assert owner.evidence.release.contact_stamps==set()
    assert finish._admit(n,owner) is False
    assert not n._cancel.is_set() and not n.events


@pytest.mark.parametrize('pairs',[[],[(BOOK.replace('_red','_blue'),BIN)]])
def test_empty_or_wrong_target_events_cannot_complete(monkeypatch,pairs):
    n,owner=fixture(monkeypatch);stream(n,owner,pairs=pairs)
    assert finish._admit(n,owner) is False
    assert owner.evidence.release.contact_stamps==set()


def test_initial_silence_waits_only_with_fresh_feedback_and_original_deadline(monkeypatch):
    n,owner=fixture(monkeypatch);stream(n,owner,contact=False)
    assert finish._admit(n,owner) is False
    assert owner.evidence.fault is None and not n._cancel.is_set()
    assert not n.events and not n.commands
    monkeypatch.setattr(finish.time,'monotonic',lambda:31.000000001)
    with pytest.raises(RuntimeError,match='hold_clock_or_deadline'):
        finish.finish(n,owner)
    assert n._cancel.is_set() and n._release_pose_owner is owner
    finish.deactivate(n)
    assert n._release_pose_fault_latched and not n.events and not n.commands


@pytest.mark.parametrize('fault',['robot','robot_bin','robot_table','malformed','invalid_stamp','stale','reversed'])
def test_received_event_fault_after_ready_survives_later_silence(monkeypatch,fault):
    n,owner=fixture(monkeypatch);stream(n,owner);assert finish.finish(n,owner)
    stamp=n.now+1
    pairs=[(BOOK,ROBOT)]
    if fault=='robot_bin':pairs=[(ROBOT,BIN)]
    if fault=='robot_table':pairs=[(ROBOT,'erc_table::table_link::collision')]
    if fault in ('invalid_stamp','stale','reversed'):
        pairs=[(BOOK,BIN)]
        stamp={'invalid_stamp':0,'stale':n.now-250_000_001,'reversed':n.now-1}[fault]
    message=contact_message(stamp,pairs)
    if fault=='malformed':message.contacts=[NS(collision1=None,collision2=None)]
    topic='/bin_contacts' if fault=='reversed' else '/contacts'
    contacts.observe(n,message,topic)
    assert owner.evidence.fault and n._cancel.is_set()
    with pytest.raises(RuntimeError):finish.publish_terminal(n,vars(IDENTITY))
    finish.deactivate(n)
    assert n._release_pose_fault_latched and not n.commands
    assert not any(event in ('placement_release_pose_measured','succeeded') for event,_ in n.events)
    if fault in ('robot_bin','robot_table'):
        assert len(n.events)==1 and n.events[0][0]=='payload_hazard'
        fields=n.events[0][1]
        assert fields['command']=='place' and fields['reason']=='place_scene_contact'
        assert IDENTITY.matches(fields) and fields['collision_pair']==list(pairs[0])
    else:
        assert n.events==[]


def test_expired_bin_window_cannot_publish_despite_current_joint_and_odom(monkeypatch):
    n,owner=fixture(monkeypatch);stream(n,owner);assert finish.finish(n,owner)
    start=n.now
    for delta in (50_000_000,100_000_000,150_000_000):record_feedback(n,start+delta)
    assert finish._admit(n,owner) is True  # exact150ms latest-bin boundary
    record_feedback(n,start+150_000_001)
    with pytest.raises(RuntimeError,match='terminal_evidence_changed'):
        finish.publish_terminal(n,vars(IDENTITY))
    assert not n.events and not n.commands


@pytest.mark.parametrize('linear,angular,allowed',[
    (.005,0.,True),(0.,.008,True),(.005000001,0.,False),
    (0.,.008000001,False),(float('nan'),0.,False),(0.,float('inf'),False),
])
def test_newer_actual_odom_cannot_hide_behind_last_joint_pair(monkeypatch,linear,angular,allowed):
    n,owner=fixture(monkeypatch);stream(n,owner);assert finish.finish(n,owner)
    paired=dict(owner.evidence.latest_sample['odom'])
    n.now+=1
    header=NS(stamp=NS(sec=n.now//10**9,nanosec=n.now%10**9))
    adapter.record_raw_odom(n,NS(header=header,twist=NS(twist=NS(
        linear=NS(x=linear,y=0.),angular=NS(z=angular)))))
    assert owner.evidence.latest_sample['odom']==paired
    if allowed:
        assert finish.publish_terminal(n,vars(IDENTITY))
    else:
        with pytest.raises(RuntimeError,match='latest_odom_not_stationary'):
            finish.publish_terminal(n,vars(IDENTITY))
        assert not n.events
    assert not n.commands


@pytest.mark.parametrize('kind',['short','gap','duplicate'])
def test_positive_window_duration_gap_and_unique_count_are_retained(monkeypatch,kind):
    n,owner=fixture(monkeypatch)
    for offset in range(50_000_000,600_000_001,50_000_000):
        record_feedback(n,1_000_000_000+offset)
        send=(offset>=400_000_000 if kind=='short' else
              offset not in (350_000_000,400_000_000) if kind=='gap' else offset==600_000_000)
        if send:
            for _ in range(3 if kind=='duplicate' else 1):
                contacts.observe(n,contact_message(n.now,[(BOOK,BIN)]),'/bin_contacts')
    assert finish._admit(n,owner) is False and not n.events


@pytest.mark.parametrize('relative',['before','after'])
def test_actual_mission_negative_before_delayed_open_status_uses_producer_epoch(relative):
    n,evidence=mission_fixture(opened=False)
    event_stamp=990_000_000 if relative=='before' else 1_050_000_000
    n.now=1_060_000_000
    mission.MissionManager._on_contacts(n,contact_message(event_stamp,[(BOOK,ROBOT)]))
    assert len(evidence.pending_contact_events)==1 and evidence.fault is None
    n.now=1_070_000_000
    status(n,dict(vars(IDENTITY),event='placement_open_measured',command='place',
                  producer_stamp_ns=1_000_000_000,verified=True))
    assert not evidence.pending_contact_events
    assert (evidence.fault is None)==(relative=='before')
    if relative=='after':
        assert evidence.fault=='release_pose_target_robot_contact'
        assert not evidence.evaluate(n.now,1.)['operation_completed']


@pytest.mark.parametrize('fault',['overflow','malformed'])
def test_mission_pending_event_history_cannot_erase_unknown_hazard(fault):
    n,evidence=mission_fixture(opened=False);n.now=1_060_000_000
    if fault=='overflow':
        for _ in range(1025):
            mission.MissionManager._on_contacts(n,contact_message(1_050_000_000,[(BOOK,ROBOT)]))
        assert evidence.fault is None
        assert len(evidence.pending_contact_events)==1  # repeated veto, no message-count gate
    else:
        message=contact_message(1_050_000_000,[(BOOK,ROBOT)]);message.contacts=None
        try:mission.MissionManager._on_contacts(n,message)
        except (TypeError,AttributeError):pass
    n.now=1_070_000_000
    status(n,dict(vars(IDENTITY),event='placement_open_measured',command='place',
                  producer_stamp_ns=1_000_000_000,verified=True))
    assert evidence.fault
    assert not evidence.evaluate(n.now,1.)['operation_completed']


def test_actual_mission_pending_future_opening_replays_later_received_hazard():
    n,evidence=mission_fixture(opened=False)
    status(n,dict(vars(IDENTITY),event='placement_open_measured',command='place',
                  producer_stamp_ns=1_050_000_000,verified=True))
    assert evidence.open_epoch is None and 'opening' in evidence.pending_milestones
    n.now=1_100_000_000
    mission.MissionManager._on_contacts(n,contact_message(n.now,[(BOOK,ROBOT)]))
    assert evidence.open_epoch==1_050_000_000
    assert evidence.fault=='release_pose_target_robot_contact'
    assert not evidence.pending_contact_events
    assert not evidence.evaluate(n.now,1.)['operation_completed']


@pytest.mark.parametrize('topic',['/contacts','/bin_contacts'])
def test_actual_mission_received_malformed_event_latches(topic):
    n,evidence=mission_fixture();n.now+=50_000_000
    message=contact_message(n.now,[(BOOK,BIN)]);message.contacts=None
    method=mission.MissionManager._on_contacts if topic=='/contacts' else mission.MissionManager._on_bin_contacts
    # Ordinary scoring parser may also reject malformed input, but the
    # release evidence must already be invalidated before that parser runs.
    try:method(n,message)
    except (TypeError,AttributeError):pass
    assert evidence.fault=='release_pose_contacts_malformed'
    assert not evidence.evaluate(n.now,1.)['operation_completed']


@pytest.mark.parametrize('lag',[-50_000_000,200_000_000])
def test_actual_admission_clocks_separate_pending_or_delayed_status_receipt(monkeypatch,lag):
    n,owner=fixture(monkeypatch);stream(n,owner);assert finish.finish(n,owner)
    assert finish.publish_terminal(n,vars(IDENTITY))
    measured=dict(n.events[0][1],event=n.events[0][0]);terminal=dict(n.events[1][1],event='succeeded')
    m,evidence=mission_fixture()
    # Mission contacts may be independently received later than manipulation.
    for stamp in range(1_050_000_000,1_500_000_001,50_000_000):
        m.now=stamp;mission.MissionManager._on_bin_contacts(m,contact_message(stamp,[(BOOK,BIN)]))
    m.now=n.now+lag
    status(m,measured);status(m,terminal)
    assert evidence.fault is None
    if lag<0:
        assert not evidence.evaluate(m.now,1.)['operation_completed']
        assert evidence.pending_finish_measurement is not None
    m.now=max(m.now,n.now)
    mission.MissionManager._on_bin_contacts(m,contact_message(m.now,[(BOOK,BIN)]))
    # A delayed summary may need new positive bin events; it cannot fabricate
    # a contiguous window across a real gap in this independent callback stream.
    for offset in range(50_000_000,600_000_001,50_000_000):
        m.now=max(n.now,n.now+lag)+offset
        mission.MissionManager._on_bin_contacts(m,contact_message(m.now,[(BOOK,BIN)]))
    assert evidence.evaluate(m.now,1.)['operation_completed']


@pytest.mark.parametrize('field,value',[
    ('release_evidence_contract','old_raw_heartbeat'),
    ('contact_stream_coverage_verified',True),('detachment_verified',True),
    ('hand_clear_verified',True),('physical_inside_verified',True),
    ('raw_contact_start_ns',1_000_000_000),
    ('bin_contact_max_gap_ns',1_000_000),
])
def test_mission_cannot_complete_with_old_or_false_scope(monkeypatch,field,value):
    n,owner=fixture(monkeypatch);stream(n,owner);assert finish.finish(n,owner)
    assert finish.publish_terminal(n,vars(IDENTITY))
    payload=dict(n.events[0][1],event=n.events[0][0]);payload[field]=value
    m,evidence=mission_fixture();m.now=n.now
    status(m,payload)
    assert evidence.fault and not evidence.evaluate(m.now,1.)['operation_completed']


def test_source_contract_is_required_by_actual_command_mode():
    for payload in ({'completion_mode':'release_pose'},
                    {'completion_mode':'release_pose','release_evidence_contract':'old'}):
        with pytest.raises(RuntimeError,match='mode_mismatch'):
            finish.validate_mode(NS(place_finish_at_release_enabled=True),payload)
    finish.validate_mode(NS(place_finish_at_release_enabled=True),dict(
        completion_mode='release_pose',release_evidence_contract=CONTRACT))
    finish.validate_mode(NS(place_finish_at_release_enabled=False),{})


def test_actual_mission_rejects_unequal_locked_measurement_and_terminal_clocks(monkeypatch):
    n,owner=fixture(monkeypatch);stream(n,owner);assert finish.finish(n,owner)
    assert finish.publish_terminal(n,vars(IDENTITY))
    measured=dict(n.events[0][1],event=n.events[0][0])
    terminal=dict(n.events[1][1],event='succeeded')
    terminal['release_terminal_stamp_ns']+=1
    m,evidence=mission_fixture();m.now=n.now
    status(m,measured);status(m,terminal)
    m.now+=1
    result=evidence.evaluate(m.now,1.)
    assert result['status']=='invalid'
    assert evidence.fault=='release_pose_terminal_measurement_stale'
    assert not result['operation_completed']
