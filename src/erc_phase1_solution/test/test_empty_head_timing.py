"""Actual sender/method wiring with controlled futures/feedback; no physics claim."""
import ast
import copy
from concurrent.futures import Future
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration

from erc_phase1_solution import empty_head_timing as timing
from erc_phase1_solution.completed_torso_hold import (
    TorsoHoldCancellation,TorsoHoldState,track_completed_torso_hold,require_follow_token_locked)
from erc_phase1_solution.runtime_utils import frames_are_synchronized,stamp_to_nanoseconds
import test_optional_arm_timing as fixtures

ROOT=Path(__file__).resolve().parents[1]
IDENTITY={'trial_id':'0123456789ab',timing.ID:'a'*32}


class Clock:
    def __init__(self):self.ns=1_000_000_000;self.wall=0.;self.on_sleep=lambda:None
    def monotonic(self):return self.wall
    def sleep(self,s):self.wall+=s;self.ns+=int(s*1e9);self.on_sleep()


class Lock:
    def __init__(self, node, label):
        self.node,self.label,self.depth=node,label,0
        self.raw=threading.RLock() if label=='command' else threading.Lock()
    def __enter__(self):
        if self.label=='command':assert self.node._lock.depth==0
        assert self.raw.acquire(timeout=.1);self.depth+=1;return self
    def __exit__(self,*unused):self.depth-=1;self.raw.release()


def done(value):
    f=Future();f.set_result(value);return f


def method(module,cls,name,**extra):
    path=ROOT/'erc_phase1_solution'/module
    owner=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=copy.deepcopy(next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name))
    scope=dict(__package__='erc_phase1_solution',__name__='erc_phase1_solution.'+Path(module).stem,
        _empty_head_timing=timing,np=np,**extra)
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),scope)
    return scope[name]


def fixture(monkeypatch, *, acceptance_pending=False):
    clock=Clock();monkeypatch.setattr(timing,'time',clock)
    n=NS(empty_head_timing_enabled=True,_empty_head_timing_owner=None,_empty_head_pick_started=False,
        _empty_head_timing_limits=timing.HeadLimits((-1.24,-.98),(1.24,.785),(3.,3.),'pinned.urdf','a'*64),
        head_chain=object(),_cancel=TorsoHoldCancellation(),_torso_hold_state=TorsoHoldState(),
        settled_place_torso_skip_enabled=True,_goal_handles=[],_pending_retained_acceptances=set(),
        _held_book_corners=None,_last_completed_head_target=None,timeout=.3,
        book_overview_tilt=-.1,book_row_tilts=[.2,-.1,-.4,-.6],
        joints=dict.fromkeys((*timing.BODY,*timing.HEAD),0.),
        _joint_stamps_ns=dict.fromkeys((*timing.BODY,*timing.HEAD),clock.ns),
        _joint_velocities=dict.fromkeys(timing.HEAD,0.),
        _staging_odom=dict(stamp_ns=clock.ns,pose=(1.,2.,.3),linear_speed=0.,angular_speed=0.))
    n._lock=Lock(n,'sensor');n.command=Lock(n,'command');n._adaptive_command_guard=lambda:n.command
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=clock.ns))
    checks=[];n._try_completed_head_hold=lambda *args:(checks.append('original_hold') or None)
    n._carried_head_transition_is_safe=lambda *args:pytest.fail('selected empty request is never loaded')
    n._publish_status=lambda *a,**k:None
    def clear(owner):
        checks.append('original_contact')
        if getattr(owner,'contact_fault',False):raise RuntimeError('original contact fault')
    terminal=done(NS(status=GoalStatus.STATUS_SUCCEEDED,result=FollowJointTrajectory.Result()))
    cancels=[];handle=NS(accepted=True,get_result_async=lambda:terminal,
        cancel_goal_async=lambda:(cancels.append(True) or done(NS())))
    acceptance=Future() if acceptance_pending else done(handle);sent=[]
    def send(goal):
        assert n.command.depth==n._lock.depth==1
        sent.append(copy.deepcopy(goal));return acceptance
    n.head_client=NS(wait_for_server=lambda **kw:True,send_goal_async=send)
    n.arm_client=object();n.torso_client=object()
    n._wait_future=lambda f,t:f.result()
    n._cancel_goal_and_confirm=lambda *args:False
    n._valid_retained_terminal_result=fixtures.load('_valid_retained_terminal_result',GoalStatus=GoalStatus)
    cancel=fixtures.load('_cancel_retained_goal_and_confirm')
    n._cancel_retained_goal_and_confirm=lambda *a:cancel(n,*a)
    late=fixtures.load('_cancel_late_retained_goal');n._cancel_late_retained_goal=lambda *a:late(n,*a)
    follow=fixtures.load('_follow',time=clock,Duration=Duration,JointTrajectoryPoint=JointTrajectoryPoint,
        FollowJointTrajectory=FollowJointTrajectory,GoalStatus=GoalStatus,
        require_follow_token_locked=require_follow_token_locked,_require_place_contact_clear=clear)
    tracked=track_completed_torso_hold(follow);n._follow=lambda *a,**k:tracked(n,*a,**k)
    move=fixtures.load('_move_head',_empty_head_timing=timing)
    n._move_head=lambda *a,**k:move(n,*a,**k)
    owner=timing.prepare(n,'look_markers',dict(IDENTITY,empty_head_before_pick=True))
    def update():
        n._joint_stamps_ns=dict.fromkeys(n.joints,clock.ns);n._staging_odom['stamp_ns']=clock.ns
        n.joints.update(zip(timing.HEAD,owner.target))
    clock.on_sleep=update
    return n,owner,clock,sent,checks,acceptance,handle,cancels


@pytest.mark.parametrize('v',[None,0,1,'true',[],float('nan')])
def test_option_is_strict_bool(v):
    with pytest.raises(ValueError):timing.checked_enabled(v)


def test_official_limit_reader_uses_actual_named_revolute_entries(tmp_path):
    p=tmp_path/'head.urdf'
    p.write_text('<robot>'+''.join(f'<joint name="{n}" type="revolute"><limit lower="-1" upper="1" velocity="3"/></joint>' for n in timing.HEAD)+'</robot>')
    limits=timing.load_limits(p)
    assert limits.velocity==(3.,3.) and limits.lower==(-1.,-1.) and len(limits.sha256)==64
    p.write_text(p.read_text().replace('velocity="3"','velocity="3.1"',1))
    with pytest.raises(ValueError):timing.load_limits(p)


def test_actual_sender_original_hold_first_goal_retime_and_post_action_measurement(monkeypatch):
    n,o,c,sent,checks,*_=fixture(monkeypatch)
    assert n._move_head(0.,.2,empty_head_timing=o)
    assert checks[1]=='original_hold'  # original contact check remains first
    assert len(sent)==1 and tuple(sent[0].trajectory.points[0].positions)==(0.,.2)
    p=sent[0].trajectory.points[0]
    assert (p.time_from_start.sec,p.time_from_start.nanosec)==(0,600_000_000)
    assert not p.velocities and not p.accelerations and not p.effort
    assert o.completion['empty_head_measured_stop'] is True
    assert all(v>o.completion['empty_head_action_return_ros_ns'] for v in o.completion['empty_head_producer_stamps_ns'])
    assert c.ns>=1_040_000_000 and n._empty_head_timing_owner is None
    assert not n._goal_handles and not n._pending_retained_acceptances


def test_duration_is_ceiled_from_fresh_start_after_server_wait(monkeypatch):
    n,o,c,sent,*_=fixture(monkeypatch)
    def server(**kw):n.joints[timing.HEAD[1]]=-.9;return True
    n.head_client.wait_for_server=server
    assert n._move_head(0.,.2,empty_head_timing=o)
    assert o.duration_ns==733_333_334
    assert sent[0].trajectory.points[0].time_from_start.nanosec==733_333_334


@pytest.mark.parametrize('fault',['moving','stale','future','nan','base','pending','held','pick','limits','cancel'])
def test_actual_final_admission_after_server_wait_rejects_unsafe_start(monkeypatch,fault):
    n,o,c,sent,*_=fixture(monkeypatch)
    def fail(**kw):
        if fault=='moving':n._joint_velocities[timing.HEAD[0]]=.001
        elif fault=='stale':n._joint_stamps_ns[timing.HEAD[0]]-=150_000_001
        elif fault=='future':n._joint_stamps_ns[timing.HEAD[0]]+=50_000_001
        elif fault=='nan':n.joints[timing.HEAD[0]]=float('nan')
        elif fault=='base':n._staging_odom['linear_speed']=.006
        elif fault=='pending':n._pending_retained_acceptances.add(object())
        elif fault=='held':n._held_book_corners=np.zeros((8,3))
        elif fault=='pick':n._empty_head_pick_started=True
        elif fault=='limits':n._empty_head_timing_limits=None
        else:n._cancel.set()
        return True
    n.head_client.wait_for_server=fail
    with pytest.raises(RuntimeError):n._move_head(0.,.2,empty_head_timing=o)
    assert not sent and n._empty_head_timing_owner is o and n._cancel.is_set()


def test_locked_token_mutation_is_checked_before_dispatch(monkeypatch):
    n,o,c,sent,*_=fixture(monkeypatch)
    original=o.admit_locked
    def mutate(client,goal):
        n.joints[timing.HEAD[0]]=float('nan');return original(client,goal)
    o.admit_locked=mutate
    with pytest.raises(RuntimeError):n._move_head(0.,.2,empty_head_timing=o)
    assert not sent


def test_late_acceptance_is_cancelled_and_failed_owner_still_blocks(monkeypatch):
    n,o,c,sent,checks,future,handle,cancels=fixture(monkeypatch,acceptance_pending=True)
    c.on_sleep=n._cancel.set
    with pytest.raises(RuntimeError):n._move_head(0.,.2,empty_head_timing=o)
    assert o.token in n._pending_retained_acceptances
    future.set_result(handle)
    assert cancels and not n._pending_retained_acceptances and not n._goal_handles
    assert n._empty_head_timing_owner is o and o.completion is None


def test_broken_result_accessor_still_cancels_known_handle(monkeypatch):
    n,o,c,sent,checks,future,handle,cancels=fixture(monkeypatch)
    def fail():raise RuntimeError('result unavailable')
    handle.get_result_async=fail
    with pytest.raises(RuntimeError):n._move_head(0.,.2,empty_head_timing=o)
    assert cancels and handle in n._goal_handles and o.token in n._pending_retained_acceptances
    assert n._empty_head_timing_owner is o


@pytest.mark.parametrize('fault',['pending_cancel','terminal_hazard','unknown_terminal','rejected'])
def test_actual_sender_terminal_and_pending_failures_never_release_owner(monkeypatch,fault):
    n,o,c,sent,checks,acceptance,handle,cancels=fixture(monkeypatch)
    if fault=='pending_cancel':
        pending=Future();handle.get_result_async=lambda:pending
        c.on_sleep=n._cancel.set
    elif fault=='terminal_hazard':
        result=handle.get_result_async()
        def poison():
            n._raw_contacts_first_failure='new fault';return result
        handle.get_result_async=poison
    elif fault=='unknown_terminal':
        handle.get_result_async=lambda:done(NS(status=GoalStatus.STATUS_EXECUTING,result=FollowJointTrajectory.Result()))
    else:handle.accepted=False
    with pytest.raises(RuntimeError):n._move_head(0.,.2,empty_head_timing=o)
    assert len(sent)==1 and o.completion is None and n._empty_head_timing_owner is o
    if fault!='rejected':assert cancels
    if fault in ('pending_cancel','unknown_terminal'):assert handle in n._goal_handles
    if fault=='rejected':assert not n._pending_retained_acceptances and not n._goal_handles


def test_larger_measured_delta_cannot_exceed_original_duration(monkeypatch):
    n,o,c,sent,*_=fixture(monkeypatch)
    o.target=(1.24,.2);n.joints[timing.HEAD[0]]=-1.24
    with pytest.raises(RuntimeError,match='bounded timing'):
        n._move_head(*o.target,empty_head_timing=o)
    assert not sent


@pytest.mark.parametrize('fault',['moving','stale','regressed','cancel','body','paused_clock'])
def test_controller_success_is_not_fresh_stationary_completion(monkeypatch,fault):
    n,o,c,sent,*_=fixture(monkeypatch);update=c.on_sleep;previous=c.ns
    def fail():
        update()
        if fault=='moving':n._joint_velocities[timing.HEAD[1]]=.001
        elif fault=='stale':n._joint_stamps_ns[timing.HEAD[0]]=previous
        elif fault=='regressed':c.ns=previous-1
        elif fault=='cancel':n._cancel.set()
        elif fault=='body':n.joints[timing.BODY[1]]=.004
        else:c.ns=previous
    c.on_sleep=fail
    with pytest.raises((RuntimeError,TimeoutError)):n._move_head(0.,.2,empty_head_timing=o)
    assert len(sent)==1 and o.completion is None and n._empty_head_timing_owner is o


def test_original_watchdog_uses_one_point_two_not_retimed_duration():
    fn=fixtures.method_ast('_follow');text=ast.unparse(fn)
    assert 'time.monotonic() + self.timeout + 4.0 * duration' in text
    move=fixtures.method_ast('_move_head')
    ordinary=move.body[-1].value
    assert ordinary.args[-1].value==1.2
    assert len([n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='duration' for t in n.targets)])==0


def test_missing_metadata_uses_default_and_first_pick_disables_opt_in(monkeypatch):
    n,o,*_=fixture(monkeypatch)
    assert timing.prepare(n,'look_bin',{}) is None
    assert timing.prepare(n,'look_books',{}) is None
    n._empty_head_pick_started=True
    with pytest.raises(RuntimeError):timing.prepare(n,'look_books',dict(IDENTITY,empty_head_before_pick=True))


def test_actual_command_gate_preserves_failed_owner_interlock(monkeypatch):
    n,o,c,sent,*_=fixture(monkeypatch);n._empty_head_timing_owner=o;n._busy=False;events=[]
    n._publish_status=lambda *a,**k:events.append((a,k))
    fn=fixtures.load('_on_command',String=NS,decode_event=lambda x:{'event':x})
    fn(n,NS(data='stow'))
    assert events[-1][1]['reason']=='empty_head_timing_owner_active' and not sent and not n._busy


def manager(state='INIT_STOW',attempts=0):
    n=NS(empty_head_timing_enabled=True,state=state,pick_attempts=attempts,trial_id=IDENTITY['trial_id'],
        detected_row=1,row_confirmed=False,target_column=1,target_colour='red',_empty_head_request=None,
        _head_return_request=None,marker_search_negative_enabled=False,book_point=object(),marker_cloud=object(),
        manip_event=None,mode_pub='mode',manip_command_pub='manip',get_clock=lambda:NS(now=lambda:NS(nanoseconds=2_000_000_000)))
    n.calls=[];n._command=lambda *a,**k:n.calls.append((a,k));n._log=lambda *a,**k:None
    mode=method('mission_manager.py','MissionManager','_perception_mode')
    n._perception_mode=lambda *a:mode(n,*a)
    manipulate=method('mission_manager.py','MissionManager','_manipulate')
    n._manipulate=lambda *a,**k:manipulate(n,*a,**k)
    return n


@pytest.mark.parametrize('state,command,attempts,expected',[
    ('INIT_STOW','look_markers',0,True),('NAVIGATE_SHELF','look_books',0,True),
    ('ALIGN_BOOK','look_book_row_1',0,True),('ALIGN_BOOK','look_book_row_2',0,False),
    ('RETRY_HEAD_BOOKS','look_book_row_1',1,False),('RETURN_START','look_bin',0,False),
    ('INIT_STOW','look_markers',1,False)])
def test_actual_mission_dispatch_has_scoped_ids_and_idle_before_head(state,command,attempts,expected):
    n=manager(state,attempts);n._manipulate(command)
    assert (timing.ID in n.calls[-1][1]) is expected
    if expected:
        assert n.calls[0][0]==('mode','idle')
        assert n.calls[0][1][timing.ID]==n.calls[-1][1][timing.ID]
        assert n.book_point is None and n.marker_cloud is None
    else:assert len(n.calls)==1


def test_old_same_command_success_and_buffered_positive_do_not_advance_current_request():
    n=manager();n._manipulate('look_markers');r=n._empty_head_request
    old=dict(IDENTITY,event='succeeded',command='look_markers')
    assert timing.mission_status(n,old) and r['epoch'] is None
    current=dict(r,event='succeeded',empty_head_measured_stop=True,empty_head_completed_ros_ns=1_990_000_000,
        empty_head_action_return_ros_ns=1_950_000_000,empty_head_producer_stamps_ns=[1_980_000_000]*2)
    assert not timing.mission_status(n,current)
    n._perception_mode('markers');assert n.calls[-1][1][timing.EPOCH]==2_000_000_000
    marker=method('mission_manager.py','MissionManager','_on_markers')
    message=NS(header=NS(stamp=NS(sec=1,nanosec=999_000_000)))
    marker(n,message);assert n.marker_cloud is None
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=2_100_000_000))
    message.header.stamp=NS(sec=2,nanosec=50_000_000)
    marker(n,message);assert n.marker_cloud is message


@pytest.mark.parametrize('rgb,depth,allowed',[(1_999_000_000,2_010_000_000,False),
    (2_010_000_000,1_999_000_000,False),(2_010_000_000,2_012_000_000,True),
    (2_101_000_000,2_012_000_000,False)])
def test_actual_perception_ready_rejects_each_old_or_future_pair_member(rgb,depth,allowed):
    stamp=lambda value:NS(sec=value//1_000_000_000,nanosec=value%1_000_000_000)
    n=NS(mode='books',_empty_head_camera=dict(IDENTITY,mode='books',epoch=2_000_000_000),
        latest_rgb=np.zeros((2,2,3)),latest_depth=np.zeros((2,2)),
        latest_rgb_message=NS(header=NS(stamp=stamp(rgb))),latest_depth_message=NS(header=NS(stamp=stamp(depth))),
        camera_info=NS(k=[1.,0.,0.,0.,1.]),maximum_frame_skew=.05,maximum_frame_age=.35,
        get_clock=lambda:NS(now=lambda:NS(nanoseconds=2_100_000_000)))
    ready=method('perception_node.py','PerceptionNode','_ready',frames_are_synchronized=frames_are_synchronized,
        stamp_to_nanoseconds=stamp_to_nanoseconds)
    assert ready(n) is allowed
    timing.camera_mode(n,{'event':'books'})
    assert n._empty_head_camera is None


def test_default_mission_does_not_add_metadata_or_idle():
    n=manager();n.empty_head_timing_enabled=False;n._manipulate('look_markers')
    assert n.calls==[(('manip','look_markers'),{})]


@pytest.mark.parametrize('fault',['wrong_id','wrong_mode','future','old','reversed'])
def test_camera_mode_reset_then_exact_request_epoch_only(fault):
    resets=[]
    n=NS(marker_history=[1],book_history=[2],last_tracking_depth_ns=10,
        target_tracker=NS(reset=lambda:resets.append(True)),
        get_clock=lambda:NS(now=lambda:NS(nanoseconds=2_000_000_000)))
    timing.camera_mode(n,dict(IDENTITY,event='idle',empty_head_expected_mode='books'))
    assert not n.marker_history and not n.book_history and resets==[True]
    assert n.last_tracking_depth_ns==-1 and n._empty_head_camera['epoch'] is None
    request=dict(IDENTITY,event='books',empty_head_not_before_ns=1_950_000_000)
    timing.camera_mode(n,request)
    if fault=='wrong_id':request[timing.ID]='b'*32
    elif fault=='wrong_mode':request['event']='markers'
    elif fault=='future':request[timing.EPOCH]=2_000_000_001
    elif fault=='old':request[timing.EPOCH]=1_649_999_999
    else:request[timing.EPOCH]=1_949_999_999
    with pytest.raises(RuntimeError):timing.camera_mode(n,request)
    assert n._empty_head_camera['epoch']==1_950_000_000


@pytest.mark.parametrize('fault',['missing_stop','producer_before_action','producer_future','completion_future'])
def test_current_request_success_requires_valid_measured_completion(fault):
    n=manager();n._manipulate('look_markers');r=n._empty_head_request
    status=dict(r,event='succeeded',empty_head_measured_stop=True,
        empty_head_completed_ros_ns=1_990_000_000,empty_head_action_return_ros_ns=1_950_000_000,
        empty_head_producer_stamps_ns=[1_980_000_000,1_980_000_000])
    if fault=='missing_stop':status['empty_head_measured_stop']=False
    elif fault=='producer_before_action':status['empty_head_producer_stamps_ns'][0]=1_950_000_000
    elif fault=='producer_future':status['empty_head_producer_stamps_ns'][1]=2_040_000_001
    else:status['empty_head_completed_ros_ns']=2_000_000_001
    assert timing.mission_status(n,status) and r['epoch'] is None
