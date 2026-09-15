"""Actual PLACE/mission/perception methods with delayed independent delivery."""
from collections import deque
import copy
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from rclpy.time import Time
from rclpy.clock import ClockType
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

from erc_phase1_solution import manipulation_node as manipulation
from erc_phase1_solution.manipulation_node import ManipulationNode
from erc_phase1_solution.mission_manager import MissionManager
from erc_phase1_solution.perception_node import PerceptionNode
from erc_phase1_solution.common import encode_event, decode_event
from erc_phase1_solution import place_input_handoff as handoff
from test_bin_scene_admission import fixture as registered_scene_fixture

TRIAL='current_trial'
REQUEST='1'*32


def payload(**fields):
    return dict(event='place',trial_id=TRIAL,place_input_request_id=REQUEST,**fields)


def message(event, **fields):
    return String(data=encode_event(event,**fields))


class PlanningReached(Exception):
    pass


@pytest.fixture
def bus(monkeypatch):
    scene,entry,context=registered_scene_fixture()
    node=object.__new__(ManipulationNode)
    node._lock=threading.Lock();node._cancel=threading.Event()
    node.latest_bin=node.bin_candidate=None
    node.bin_verified_ns=node.bin_invalidated_ns=-1
    node.table_scene_required=node.bin_scene_required=True
    node.delivery_evidence_enabled=False
    node.perception_wait=.14;node.perception_max_age=.5;node.tf_timeout=.05
    node.get_clock=lambda:SimpleNamespace(now=lambda:Time(nanoseconds=context['stamp_ns'],clock_type=ClockType.ROS_TIME))
    node.tf_buffer=SimpleNamespace(transform=lambda *a,**k:SimpleNamespace(
        point=SimpleNamespace(**dict(zip(('x','y','z'),entry['bin_floor_point_base'])))))
    node._held_book_corners=np.zeros((8,3));node._carried_staging_solution=object()
    node._fresh_retention_probe=Mock(return_value=True)
    node._measured_left_solution=Mock(return_value=np.zeros(8))
    monkeypatch.setattr(manipulation,'measured_scene_context',lambda *a:copy.deepcopy(context))
    activate=Mock(side_effect=PlanningReached)
    monkeypatch.setattr(manipulation,'_activate_place_contacts',activate)
    mission=object.__new__(MissionManager)
    mission.trial_id=TRIAL;mission.state='PLACE';mission.finished=False
    mission.delivery_evidence_enabled=False;mission.target_column=2;mission.target_colour='red'
    mission.mode_pub=object();mission.manip_command_pub=object()
    mission._place_input_request=handoff.InputRequest(TRIAL,REQUEST)
    mission._log=Mock();mission._perception_mode=Mock();mission.manip_event=None
    modes=[];statuses=[];ready=threading.Event();acks=[]
    def command(pub,event,**fields):
        value=message(event,trial_id=TRIAL,**fields)
        if pub is mission.mode_pub:modes.append(value)
    mission._command=command
    def status(event,**fields):
        statuses.append((event,fields))
        mission._on_manipulation_status(message(event,**fields))
        if event=='place_input_ready':ready.set()
    node._publish_status=status
    perception=object.__new__(PerceptionNode)
    perception.mode='bin';perception.target_column=2;perception.target_colour='red'
    perception.confirmed_book_row=None
    for name in ('marker_history','book_history','bin_history','bin_rgb_frames','bin_depth_frames'):
        setattr(perception,name,deque([1]))
    perception.bin_tracker=SimpleNamespace(reset=Mock())
    perception.target_tracker=SimpleNamespace(reset=Mock())
    perception._publish_tracking_status=Mock()
    def perceived(event,**fields):
        value=message(event,mode=perception.mode,**fields);acks.append(decode_event(value.data))
        node._on_bin_status(value)
    perception._publish_status=perceived
    point=PointStamped();point.header.frame_id='depth_optical_frame';point.header.stamp.sec=1
    point.point.z=entry['bin_floor_point_base'][2]
    status_fields=dict(mode='bin',bin_valid=True,**entry)
    results=[]
    def run():
        try:node._place(payload())
        except BaseException as exc:results.append(exc)
    thread=threading.Thread(target=run)
    current=SimpleNamespace(node=node,mission=mission,perception=perception,point=point,
        entry=entry,context=context,status_fields=status_fields,modes=modes,statuses=statuses,
        ready=ready,acks=acks,activate=activate,results=results,thread=thread)
    yield current
    node._cancel.set()
    if thread.ident is not None:thread.join(1.)
    assert not thread.is_alive()


@pytest.mark.parametrize('order',[('point','status'),('status','point')])
def test_delayed_independent_point_and_registered_status_keep_producer_until_capture(bus,order):
    b=bus;b.thread.start()
    assert not b.ready.wait(.015) and b.perception.mode=='bin'
    for part in order:
        if part=='point':b.node._on_bin(b.point)
        else:b.node._on_bin_status(message('bin_verified',**b.status_fields))
        if part==order[0]:assert not b.ready.wait(.015)
    assert b.ready.wait(.3)
    assert b.perception.mode=='bin' and len(b.modes)==1
    assert b.activate.call_count==0 and b.thread.is_alive()
    # Publishing idle alone does not release planning; its applied ack does.
    b.perception._on_mode(b.modes.pop())
    b.thread.join(.5)
    assert not b.thread.is_alive() and isinstance(b.results[0],PlanningReached)
    assert b.perception.mode=='idle'
    assert b.acks[-1]['event']=='place_input_idle'
    assert b.acks[-1]['place_input_request_id']==REQUEST
    assert b.node._place_input_waiter is None
    assert b.node._selected_place_bin_scene==b.entry['bin_scene']
    assert b.node._selected_place_table_scene==b.entry['table_scene']


@pytest.mark.parametrize('missing',['point','status','table','bin','invalidated','stale','mismatched_stamp','nonpositive'])
def test_original_fresh_and_registered_checks_fail_without_capture_ack(bus,missing):
    b=bus;fields=copy.deepcopy(b.status_fields)
    if missing=='nonpositive':b.point.point.z=0.0
    if missing=='table':fields.pop('table_scene')
    if missing=='bin':fields.pop('bin_scene')
    if missing=='invalidated':fields['bin_valid']=False
    if missing=='stale':b.context['stamp_ns']=2_000_000_000
    if missing=='mismatched_stamp':fields['observation_stamp_ns']+=1
    if missing!='point':b.node._on_bin(b.point)
    if missing!='status':b.node._on_bin_status(message('bin_verified',**fields))
    b.thread.start();b.thread.join(.8)
    assert not b.thread.is_alive() and isinstance(b.results[0],RuntimeError)
    assert not b.ready.is_set() and not b.modes and b.activate.call_count==0
    assert b.perception.mode=='bin'


def admit_input(b):
    b.node._on_bin(b.point)
    b.node._on_bin_status(message('bin_verified',**b.status_fields))
    b.thread.start();assert b.ready.wait(.3)


@pytest.mark.parametrize('wrong',[
    {'trial_id':'old_trial'}, {'place_input_request_id':'2'*32},
    {'mode':'bin'}, {'event':'mode_changed'}, {'place_input_request_id':True},
])
def test_old_unrelated_or_malformed_idle_ack_cannot_release_planning(bus,wrong):
    b=bus;admit_input(b)
    value=dict(event='place_input_idle',mode='idle',trial_id=TRIAL,place_input_request_id=REQUEST)
    value.update(wrong);b.node._on_bin_status(String(data=encode_event(**value)))
    assert b.thread.is_alive() and not b.activate.called
    b.perception._on_mode(b.modes.pop());b.thread.join(.5)
    assert isinstance(b.results[0],PlanningReached)


def test_matching_idle_before_capture_cannot_release_later_waiter(bus):
    b=bus;b.node._on_bin_status(message('place_input_idle',mode='idle',trial_id=TRIAL,place_input_request_id=REQUEST))
    admit_input(b)
    assert b.thread.is_alive() and not b.activate.called
    b.perception._on_mode(b.modes.pop());b.thread.join(.5)
    assert isinstance(b.results[0],PlanningReached)


@pytest.mark.parametrize('cause',['missing_ack','cancel','publish_error'])
def test_bounded_wait_cancellation_and_publication_failure_clear_waiter(bus,monkeypatch,cause):
    b=bus;monkeypatch.setattr(handoff,'IDLE_ACK_SECONDS',.05)
    if cause=='publish_error':b.node._publish_status=Mock(side_effect=RuntimeError('publisher_failed'))
    b.node._on_bin(b.point);b.node._on_bin_status(message('bin_verified',**b.status_fields))
    b.thread.start()
    if cause=='cancel':assert b.ready.wait(.3);b.node._cancel.set()
    b.thread.join(.5)
    assert not b.thread.is_alive() and isinstance(b.results[0],RuntimeError)
    assert b.node._place_input_waiter is None and not b.activate.called
    assert str(b.results[0])=={'missing_ack':'place_input_idle_ack_timeout','cancel':'place_input_handoff_cancelled','publish_error':'publisher_failed'}[cause]


@pytest.mark.parametrize('fields',[None,{}, {'trial_id':'legacy'}])
def test_direct_legacy_call_needs_no_protocol_or_lock(fields):
    node=SimpleNamespace()
    handoff.await_idle_after_capture(node,fields)
    assert vars(node)=={}


@pytest.mark.parametrize('field,value',[('trial_id',''),('trial_id',True),('place_input_request_id','old'),('place_input_request_id',1)])
def test_invalid_request_refused_before_ready_publication(bus,field,value):
    data=payload();data[field]=value
    with pytest.raises(RuntimeError,match='request_invalid'):handoff.await_idle_after_capture(bus.node,data)
    assert not bus.statuses


@pytest.mark.parametrize('change',[{'trial_id':'old'}, {'place_input_request_id':'2'*32}, {'command':'pick'}])
def test_mission_ignores_stale_or_wrong_command_ready(bus,change):
    b=bus;fields=payload();fields.update(event='place_input_ready',command='place');fields.update(change)
    b.mission._on_manipulation_status(String(data=encode_event(**fields)))
    assert not b.modes and b.mission._place_input_request is not None


def test_ready_requires_current_delivery_attempt_when_enabled(bus):
    b=bus;b.mission.delivery_evidence_enabled=True
    b.mission.delivery_release_evidence=SimpleNamespace(identity=SimpleNamespace(matches=lambda p:p.get('placement_attempt_id')=='current'))
    for attempt,count in [('old',0),('current',1)]:
        b.mission._on_manipulation_status(message('place_input_ready',command='place',trial_id=TRIAL,place_input_request_id=REQUEST,placement_attempt_id=attempt))
        assert len(b.modes)==count


@pytest.mark.parametrize('event',['succeeded','failed','rejected','cancelled'])
def test_matching_terminal_cleans_idle_and_preserves_terminal_handling(bus,event):
    b=bus;b.mission._on_manipulation_status(message(event,command='place',trial_id=TRIAL,place_input_request_id=REQUEST))
    assert b.mission._place_input_request is None
    b.mission._perception_mode.assert_called_once_with('idle')
    assert b.mission.manip_event['event']==event


def test_old_terminal_cannot_stop_current_input_request(bus):
    b=bus;b.mission._on_manipulation_status(message('failed',command='place',trial_id=TRIAL,place_input_request_id='2'*32))
    assert b.mission._place_input_request is not None and b.mission.manip_event is None
    b.mission._perception_mode.assert_not_called()


def test_new_place_requests_are_unique_and_included_even_without_delivery_evidence(bus):
    b=bus;calls=[];b.mission._command=lambda pub,event,**fields:calls.append((event,fields))
    b.mission._manipulate('place');first=b.mission._place_input_request
    b.mission._manipulate('place');second=b.mission._place_input_request
    assert first!=second and all(r.trial_id==TRIAL for r in (first,second))
    assert calls[0][1]['place_input_request_id']==first.request_id
    assert calls[1][1]['place_input_request_id']==second.request_id


def test_reacquire_bin_dispatch_keeps_producer_active_and_creates_request(bus):
    b=bus;m=b.mission;m.state='REACQUIRE_BIN';m.wall_started=time.monotonic();m.trial_timeout=1000.
    m.bin_point=b.point;m.bin_invalidated_ns=-1;m.get_clock=b.node.get_clock
    m._set_state=lambda state:setattr(m,'state',state)
    m._tick()
    assert m.state=='PLACE' and m._place_input_request.trial_id==TRIAL
    m._perception_mode.assert_not_called()
    assert b.perception.mode=='bin'


def test_abort_publishes_idle_before_cancels_and_clears_request(bus):
    b=bus;events=[];m=b.mission;m.active_nav_pending=False
    m.nav_command_pub=object();m._perception_mode=lambda mode:events.append(mode)
    m._command=lambda *a,**k:events.append(a[1]);m._set_state=Mock();m._write_summary=Mock()
    m._abort('shutdown')
    assert events==['idle','cancel','cancel'] and m._place_input_request is None


def test_invalid_ros_shutdown_clears_request_without_illegal_publication(bus):
    b=bus;m=b.mission;m._write_summary=Mock();m._finalize_without_ros('shutdown')
    assert m.finished and m.state=='ABORTED' and m._place_input_request is None
    m._perception_mode.assert_not_called();m._write_summary.assert_called_once_with(False,'shutdown',log_event=False)


def test_applied_idle_ack_is_repeated_for_same_mode_but_never_unrelated_mode(bus):
    b=bus;msg=message('idle',trial_id=TRIAL,place_input_request_id=REQUEST)
    b.perception._on_mode(msg);b.perception._on_mode(msg)
    assert len([x for x in b.acks if x['event']=='place_input_idle'])==2
    b.perception._on_mode(message('bin',trial_id=TRIAL,place_input_request_id=REQUEST))
    assert len([x for x in b.acks if x['event']=='place_input_idle'])==2


@pytest.mark.parametrize('attribute,value,reason',[
    ('_raw_contacts_first_failure',{'reason':'bad'},'raw_contacts_decode_failed'),
    ('_busy',True,'busy'),
    ('_pending_retained_acceptances',[object()],'retained_goal_acceptance_unresolved'),
    ('_goal_handles',[object()],'controller_goal_unresolved'),
])
def test_actual_early_command_refusals_keep_request_and_cleanup(bus,attribute,value,reason):
    b=bus;n=b.node;n._busy=False;n._raw_contacts_first_failure=None
    n._pending_retained_acceptances=[];n._goal_handles=[]
    setattr(n,attribute,value)
    n._on_command(String(data=encode_event(**payload())))
    event,fields=b.statuses[-1]
    assert event=='rejected' and fields['reason']==reason
    assert fields['place_input_request_id']==REQUEST and fields['trial_id']==TRIAL
    assert b.mission._place_input_request is None
    b.mission._perception_mode.assert_called_once_with('idle')


@pytest.mark.parametrize('dry_run',[False,True])
def test_actual_dispatch_passes_payload_without_delivery_feature_and_terminal_keeps_request(bus,monkeypatch,dry_run):
    b=bus;n=b.node;n.dry_run=dry_run;n.book_row_tilts=[];n._place=Mock(return_value=True)
    monkeypatch.setattr(manipulation,'_deactivate_place_contacts',Mock())
    n._run_command('place',payload())
    if dry_run:n._place.assert_not_called()
    else:n._place.assert_called_once_with(payload())
    assert b.statuses[-1][0]=='succeeded'
    assert b.statuses[-1][1]['place_input_request_id']==REQUEST
    assert b.mission._place_input_request is None
