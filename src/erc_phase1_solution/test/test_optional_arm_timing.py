"""Prepared pure boundary/workflow tests; no ROS or geometry imports.

These model command/future behavior. They do not establish tracking, acceleration,
retention under faster dynamics, or physical collision safety.
"""
from __future__ import annotations
import ast
import hashlib
import json
from candidate_composition_support import without_torso_tool
from erc_phase1_solution.raised_place_finish import checked_enabled as checked_raised_place_finish_enabled
import copy
import importlib.util
import math
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import numpy as np
import pytest
from erc_phase1_solution.arm_velocity_admission import (
    ArmVelocityAdmissionRejected, require_arm_velocity_locked, publish_arm_velocity_admission,
)

SOURCE=Path(__file__).resolve().parents[1]
PACKET=SOURCE.parent
spec=importlib.util.spec_from_file_location('candidate_arm_timing',SOURCE/'erc_phase1_solution/arm_trajectory_timing.py')
timing=importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)
ARM_JOINTS=tuple(f'arm_left_{i}_joint' for i in range(1,8))
HEAD_JOINTS=('head_1_joint','head_2_joint')
HOME=np.array([.1,.36,-1.83,.47,-2.35,0,-1.2,0])

class RetainedMotionNotStopped(RuntimeError): pass
class LiftPressureRejected(RuntimeError): pass
class Duration:
    def __init__(self,*,seconds): self.seconds=seconds
    def to_msg(self):
        ns=int(self.seconds*1_000_000_000)
        return NS(sec=ns//1_000_000_000,nanosec=ns%1_000_000_000)

def seconds(stamp):return stamp.sec+stamp.nanosec/1_000_000_000
def point():return NS(positions=[],velocities=[],accelerations=[],effort=[])
def goal():return NS(trajectory=NS(joint_names=[],points=[],header=NS(stamp=NS(sec=0,nanosec=0))))

def node_tree(baseline=False):
    path=(SOURCE/'test/fixtures/arm_timing_original_follow.py' if baseline else SOURCE/'erc_phase1_solution/manipulation_node.py')
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')

def method_ast(name,baseline=False):
    return copy.deepcopy(next(n for n in node_tree(baseline).body if isinstance(n,ast.FunctionDef) and n.name==name))

def load(name,baseline=False,**extra):
    n=method_ast(name,baseline); n.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])
    scope=dict(__package__='erc_phase1_solution', __name__='erc_phase1_solution.manipulation_node', np=np,math=math,ARM_JOINTS=ARM_JOINTS,HEAD_JOINTS=HEAD_JOINTS,HOME=HOME,
        Duration=Duration,JointTrajectoryPoint=point,
        FollowJointTrajectory=NS(Goal=goal),
        GoalStatus=NS(STATUS_SUCCEEDED=4),
        RetainedMotionNotStopped=RetainedMotionNotStopped,LiftPressureRejected=LiftPressureRejected,
        checked_arm_speed_scale=timing.checked_arm_speed_scale,scaled_arm_seconds=timing.scaled_arm_seconds,
        checked_raised_place_finish_enabled=checked_raised_place_finish_enabled,
        ArmVelocityAdmissionRejected=ArmVelocityAdmissionRejected,
        require_arm_velocity_locked=require_arm_velocity_locked,publish_arm_velocity_admission=publish_arm_velocity_admission,
        _require_place_contact_clear=lambda n:None,measured_scene_context=lambda n,r:None)
    scope.update(extra)
    exec(compile(ast.fix_missing_locations(module),'<actual candidate method>','exec'),scope)
    return scope[name]

@pytest.mark.parametrize('value',[1.,1.125,1.25,2.,2.25,2.5,2.500001,2.75,3.0])
def test_valid_scale(value):
    assert timing.checked_arm_speed_scale(value)==value

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),0.,.999999,3.000001,math.nextafter(3.0, math.inf),-1.])
def test_bad_scale_rejected(value):
    with pytest.raises(ValueError): timing.checked_arm_speed_scale(value)

@pytest.mark.parametrize('seconds',[.35,.65,.8,1.6,2.8,10.513066998547497])
def test_default_exact_duration(seconds):
    assert timing.scaled_arm_seconds(seconds,1.)==seconds

@pytest.mark.parametrize('seconds,expected',[(.35,.35),(.4,.35),(.8,.64),(1.6,1.28)])
def test_supported_floor(seconds,expected):
    assert timing.scaled_arm_seconds(seconds,1.25,minimum=.35)==pytest.approx(expected)

@pytest.mark.parametrize('seconds,minimum',[(0,0),(-1,0),(float('nan'),0),(float('inf'),0),(1,float('nan')),(1,-1),(1,2)])
def test_bad_duration_floor_rejected(seconds,minimum):
    with pytest.raises(ValueError): timing.scaled_arm_seconds(seconds,1.25,minimum=minimum)

def follow_node(*,late=False):
    sent=[]; statuses=[]; sleeps=[]
    calls=iter([False,True] if late else [True])
    result=NS(done=lambda:next(calls),result=lambda:NS(status=4))
    handle=NS(accepted=True,get_result_async=lambda:result)
    client=NS(wait_for_server=lambda **kw:True,
        send_goal_async=lambda g:(sent.append(g) or NS(result=lambda:handle)))
    n=NS(_cancel=threading.Event(),_lock=threading.Lock(),arm_client=client,
        _goal_handles=[],timeout=1.,_publish_status=lambda *a,**kw:statuses.append((a,kw)),
        _wait_future=lambda f,t:f.result(),_cancel_goal_and_confirm=lambda *a:True,
        _adaptive_command_guard=lambda:threading.RLock(),
        joints=dict.fromkeys(ARM_JOINTS,0.),_joint_stamps_ns=dict.fromkeys(ARM_JOINTS,1_000_000_000),
        _faster_arm_velocity_limits=(1.95,1.95,3.95,3.95,3.95,3.95,3.95),
        get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_000_000_000)))
    times=iter([0.,11.]) if late else iter([0.])
    clock=NS(monotonic=lambda:next(times),sleep=lambda v:sleeps.append(v))
    return n,sent,statuses,sleeps,clock

@pytest.mark.parametrize('baseline',[True,False])
def test_default_follow_goal_and_lifecycle(baseline):
    n,sent,_,_,clock=follow_node()
    assert load('_follow',baseline,time=clock)(n,n.arm_client,ARM_JOINTS,[.1]*7,2.8)
    assert seconds(sent[0].trajectory.points[0].time_from_start)==2.8
    assert sent[0].trajectory.points[0].positions==[.1]*7
    assert not n._goal_handles

def test_scaled_follow_keeps_original_watchdog_allowance():
    n,sent,_,sleeps,clock=follow_node(late=True)
    # 11 wall seconds is beyond the scaled 9.96-second allowance, but inside
    # the unchanged 12.2-second allowance (timeout1 + 4*nominal2.8).
    assert load('_follow',time=clock)(n,n.arm_client,ARM_JOINTS,[.1]*7,2.8,trajectory_duration=2.24)
    assert seconds(sent[0].trajectory.points[0].time_from_start)==2.24
    assert sleeps==[.02] and not n._goal_handles

@pytest.mark.parametrize('guard',[False,True])
def test_cancel_in_pre_send_hook_still_prevents_scaled_goal(guard):
    n,sent,_,_,clock=follow_node()
    n._place_contact_guard=object() if guard else None
    assert not load('_follow',time=clock)(n,n.arm_client,ARM_JOINTS,[.1]*7,2.8,
        trajectory_duration=2.24,pre_send_check=n._cancel.set)
    assert not sent

def test_latched_hazard_prevents_scaled_goal():
    n,sent,statuses,_,clock=follow_node()
    n._payload_robot_watchdog_enabled=True; n._target_robot_contact_latched=True
    assert not load('_follow',time=clock)(n,n.arm_client,ARM_JOINTS,[.1]*7,2.8,trajectory_duration=2.24)
    assert not sent and n._payload_hazard_latched=='payload_robot_contact'

@pytest.mark.parametrize('duration',[0.,.9,float('nan'),float('inf'),3.])
def test_invalid_override_does_not_send(duration):
    n,sent,_,_,clock=follow_node()
    with pytest.raises(ValueError):
        load('_follow',time=clock)(n,n.arm_client,ARM_JOINTS,[.1]*7,2.8,trajectory_duration=duration)
    assert not sent

def test_override_cannot_change_head_or_torso_controller():
    n,sent,_,_,clock=follow_node()
    with pytest.raises(ValueError):
        load('_follow',time=clock)(n,n.arm_client,HEAD_JOINTS,[0.,-.6],2.8,trajectory_duration=2.24)
    assert not sent

def retained_node(failure=None):
    calls=[]; probes=[]
    make=load('_make_retained_arm_trajectory_goal')
    n=NS(_goal_handles=[],_cancel=threading.Event(),_publish_status=lambda *a,**kw:None)
    n._make_retained_arm_trajectory_goal=lambda legs:make(n,legs)
    def send(goal,allowance,legs,command,**kw):
        calls.append((goal,allowance,legs,command,kw))
        if isinstance(failure,Exception): raise failure
        return (False,True) if failure=='contact' else (True,False)
    n._send_retained_arm_trajectory=send
    n._retention_after_leg=lambda *a:(probes.append(('ordinary',a)) or True)
    n._fresh_retention_probe=lambda *a,**kw:(probes.append(('fresh',a,kw)) or True)
    return n,calls,probes

@pytest.mark.parametrize('scale',[1.,1.25,2.,2.5])
def test_retained_goal_scales_command_not_q_watchdog_or_probes(scale):
    n,calls,probes=retained_node()
    q=np.array([.35,.1,.2,.3,.4,.5,.6,.7]); original=q.copy()
    legs=[(q,.8,'bin_transition'),(q,2.8,'bin_clearance')]
    assert load('_execute_retained_arm_legs')(n,legs,'place',arm_speed_scale=scale,
        fresh_retention_phases=('bin_clearance',),leg_offset=5)==(True,2,False)
    for call,duration in zip(calls,[.8,2.8]):
        assert call[1]==duration
        assert seconds(call[0].trajectory.points[0].time_from_start)==pytest.approx(duration/scale)
        assert call[0].trajectory.points[0].positions==list(q[1:])
        assert call[2][0][0] is q
    assert np.array_equal(q,original)
    assert legs[0][1]==.8 and probes[0]==('ordinary',('place','bin_transition',5))
    assert probes[1]==('fresh',('place','bin_clearance'),{'leg':6})

def test_retained_contact_failure_stops_before_next_point():
    n,calls,probes=retained_node('contact')
    assert load('_execute_retained_arm_legs')(n,[(np.zeros(8),.8,'a')]*2,'place',arm_speed_scale=1.25)==(False,0,True)
    assert len(calls)==1 and not probes

def test_uncertain_retained_stop_still_propagates():
    n,calls,probes=retained_node(RetainedMotionNotStopped('pending controller'))
    with pytest.raises(RetainedMotionNotStopped):
        load('_execute_retained_arm_legs')(n,[(np.zeros(8),.8,'a')],'place',arm_speed_scale=1.25)
    assert not probes

def test_return_scales_only_arm_and_forwards_all_exact_points():
    calls=[]; solutions=[np.full(8,v) for v in [0.,.1,.2]]
    n=NS(table_scene_required=False,
        _move_arm_solution=lambda q,d,**kw:(calls.append(('arm',q,d,kw)) or True),
        _execute_unloaded_home=lambda q,**kw:(calls.append(('home',q,kw)) or True))
    empty=[]
    assert load('_return_from_bin')(n,solutions,[],solutions[0],[],direct_empty_home=empty,arm_speed_scale=1.25)
    assert calls[0][1] is solutions[1] and calls[1][1] is solutions[0]
    assert all(c[2]==.65 and c[3]=={'trajectory_duration':.52} for c in calls[:2])
    assert calls[2][1] is empty and calls[2][2]['arm_speed_scale']==1.25
    assert calls[2][2]['final_arm_duration']==max(2.8,4*max(abs(HOME[1:])))

def test_unloaded_endpoint_waits_and_torso_duration_are_unchanged():
    calls=[]; waits=[]; q=np.array([.35,.1,.2,.3,.4,.5,.6,.7])
    client=object()
    n=NS(arm_client=client,
        _move_arm_solution=lambda goal,d,**kw:(calls.append(('arm',goal,d,kw)) or True),
        _follow=lambda c,names,goal,d,**kw:(calls.append(('follow',c,tuple(names),goal,d,kw)) or True),
        _move_torso=lambda height,d:(calls.append(('torso',height,d)) or True))
    assert load('_execute_unloaded_home')(n,[q],final_arm_duration=10.,endpoint_start=q,
        endpoint_wait=lambda *a:waits.append(a),arm_speed_scale=1.25)
    assert calls[0][2]==2.2 and calls[0][3]['trajectory_duration']==pytest.approx(1.76)
    assert calls[1][4:]==(10.,{'trajectory_duration':8.})
    assert calls[2]==('torso',.1,2.)
    assert [w[2] for w in waits]==['empty_return_waypoint','empty_return_arm_home','empty_return_torso_home']

def test_default_declared_and_actual_constructor_validation_wiring():
    declared=method_ast('_declare_parameters')
    values=[dict(zip([k.value for k in d.keys if isinstance(k,ast.Constant)],d.values)) for d in ast.walk(declared) if isinstance(d,ast.Dict) and all(isinstance(k,ast.Constant) for k in d.keys)]
    assert any(isinstance(v.get('placement_transport_speed_scale'),ast.Constant) and v['placement_transport_speed_scale'].value==1.0 for v in values)
    init=method_ast('__init__')
    assignment=next(n for n in ast.walk(init) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='placement_transport_speed_scale' for t in n.targets))
    assert isinstance(assignment.value,ast.Call) and assignment.value.func.id=='checked_arm_speed_scale'

@pytest.mark.parametrize('table,bin_required,direct,expected',[
    (True,True,[],True),(False,True,[],False),(True,False,[],False),(True,True,None,False)])
def test_actual_place_timing_requires_registered_bin_table_path(table,bin_required,direct,expected):
    f=method_ast('_place')
    start=next(i for i,n in enumerate(f.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='arm_timing_options' for t in n.targets))
    scope=dict(self=NS(placement_transport_speed_scale=1.25,table_scene_required=table,bin_scene_required=bin_required),
        direct_empty_home=direct,checked_arm_speed_scale=timing.checked_arm_speed_scale)
    initialization = f.body[0]
    assert isinstance(initialization, ast.Assign)
    assert [target.id for target in initialization.targets] == ['release_only_request', 'release_only_endpoint']
    assert isinstance(initialization.value, ast.Constant) and initialization.value.value is None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[initialization,*f.body[start:start+3]],type_ignores=[])),'<actual timing admission>','exec'),scope)
    assert scope['arm_timing_options']==({'arm_speed_scale':1.25} if expected else {})

def test_actual_compact_scaling_block_excludes_extension_roll_and_keeps_phase_clock():
    f=method_ast('_execute_cached_post_retreat_compaction')
    start=next(i for i,n in enumerate(f.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='supported_watchdog_duration' for t in n.targets))
    block=f.body[start:start+3]
    q=np.zeros(8); groups=[[(q,.4,'extension')],[(q,.75,'roll')],[(q,.35,'lower'),(q,.8,'tuck')]]
    saved=copy.deepcopy(groups)
    scope=dict(self=NS(placement_transport_speed_scale=1.25),timed_groups=groups,
        checked_arm_speed_scale=timing.checked_arm_speed_scale,scaled_arm_seconds=timing.scaled_arm_seconds)
    exec(compile(ast.fix_missing_locations(ast.Module(body=block,type_ignores=[])),'<actual group timing block>','exec'),scope)
    assert groups[0][0][1]==.4 and groups[1][0][1]==.75
    assert [v[1] for v in groups[2]]==[.35,.64]
    assert [v[2] for v in groups[2]]==['lower','tuck']
    assert all(v[0] is q for group in groups for v in group)
    assert scope['supported_watchdog_duration']==pytest.approx(1.15)

def test_geometry_recovery_shelf_and_gripper_method_asts_unchanged():
    scope_file=SOURCE/'test/fixtures/arm_timing_original_scope.json'
    assert hashlib.sha256(scope_file.read_bytes()).hexdigest()=='e1620e42eed5679e58638ef015766c7400235512f22af12671ba5a2037a99cf3'
    scope=json.loads(scope_file.read_text())
    assert hashlib.sha256((SOURCE/'test/fixtures/arm_timing_original_follow.py').read_bytes()).hexdigest()==scope['original_follow_fixture_sha256']
    old=scope['method_ast_sha256']
    restored=ast.parse(without_torso_tool((SOURCE/'erc_phase1_solution/manipulation_node.py').read_text()))
    owner=next(n for n in restored.body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    new={n.name:n for n in owner.body if isinstance(n,ast.FunctionDef)}
    changed={'__init__','_declare_parameters','_follow','_move_arm_solution','_send_retained_arm_trajectory','_execute_retained_arm_legs',
        '_execute_cached_post_retreat_compaction','_return_from_bin','_execute_unloaded_home','_place'}
    assert old.keys()==new.keys()
    for name in old.keys()-changed:
        assert old[name]==hashlib.sha256(ast.dump(new[name],include_attributes=False).encode()).hexdigest(),name
    # Normal PLACE passes its option explicitly; recovery has no such kwargs.
    place=method_ast('_place')
    option_calls=[n.func.attr for n in ast.walk(place) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and any(k.arg is None and isinstance(k.value,ast.Name) and k.value.id=='arm_timing_options' for k in n.keywords)]
    assert sorted(option_calls)==['_return_from_bin','_return_from_bin']
    loaded_calls=[n.func.attr for n in ast.walk(place) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and any(k.arg is None and isinstance(k.value,ast.Name) and k.value.id=='loaded_arm_timing_options' for k in n.keywords)]
    assert loaded_calls==['_execute_retained_arm_legs']
