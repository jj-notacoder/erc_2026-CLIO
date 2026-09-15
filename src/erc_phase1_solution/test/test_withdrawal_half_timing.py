"""Actual retained sender, route and pressure/probe contracts; no physics proof."""
import ast
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration

from erc_phase1_solution import withdrawal_half_timing as half
from erc_phase1_solution import additional_arm_timing as extra
from erc_phase1_solution import arm_velocity_admission as velocity
from erc_phase1_solution.completed_torso_hold import TorsoHoldCancellation
from erc_phase1_solution.lift_first_extraction import LiftFirstPlan
from erc_phase1_solution.lift_pressure_gate import LiftPressureGate
from erc_phase1_solution.motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS
from erc_phase1_solution.withdrawal_timing import withdrawal_leg_speed_scales
import test_optional_arm_timing as timing
import test_arm_velocity_admission as senders

ROOT=Path(__file__).resolve().parents[1]
MODEL='book_col_1_row_1_red'


def load(name,**kwargs):
    scope=dict(IK_JOINTS=IK_JOINTS,RIGHT_ARM_JOINTS=RIGHT_ARM_JOINTS,
        FollowJointTrajectory=FollowJointTrajectory,JointTrajectoryPoint=JointTrajectoryPoint,
        Duration=Duration,GoalStatus=GoalStatus,
        withdrawal_leg_speed_scales=withdrawal_leg_speed_scales,
        validate_withdrawal_half_execution=half.validate_execution,
        withdrawal_half_timing_input=half.timing_input,
        require_withdrawal_half_publication_locked=half.require_publication_locked,
        require_withdrawal_half_endpoint=half.require_endpoint,
        withdrawal_half_normal_options=half.normal_options,
        retime_admitted_arm_goal=extra.retime_admitted_arm_goal,
        require_retimed_arm_headroom=extra.require_retimed_arm_headroom)
    scope.update(kwargs)
    return timing.load(name,**scope)


def install_context(n):
    n.withdrawal_half_timing_enabled=True;n.withdrawal_speed_scale=3.;n.additional_arm_time_scale=2.
    n.lift_first_extraction_lift_m=.020;n._cancel=TorsoHoldCancellation()
    n._target_book_model=MODEL;n._contact_epoch=7;n._held_book_corners=np.zeros((8,3))
    n._active_place_scene_reference=None;n._gravity_supported_payload=False
    n._payload_monitor_enabled=True;n._gripper_open_confirmed=False
    for field in half.MODEL_NAMES:setattr(n,field,NS(chains={}) if field=='_shelf_cradle_geometry' else object())
    names=(*IK_JOINTS,*RIGHT_ARM_JOINTS,*half.HEAD,'gripper_left_finger_joint')
    n.joints=dict.fromkeys(names,0.);n.joints[IK_JOINTS[0]]=.35;n.joints[names[-1]]=.0181
    n._joint_stamps_ns=dict.fromkeys(names,senders.NOW)
    n._staging_odom=dict(stamp_ns=senders.NOW,pose=[0.,0.,0.],linear_speed=0.,angular_speed=0.)
    n.grasp_contact_max_age=.2
    measured=load('_lift_first_measurements');n._lift_first_measurements=lambda r=None:measured(n,r)
    route=tuple(np.array([.35,*([.02*i]*7)]) for i in range(5))
    legs=[(q,1. if i==0 else 5.8,'initial_shelf_lift' if i==0 else 'extraction') for i,q in enumerate(route)]
    plan=LiftFirstPlan(route,route[-1],n._held_book_corners,{'lift_m':.020})
    gate=LiftPressureGate(n,{},n._lift_first_measurements())
    options=half.normal_options(n,legs,gate,plan,ordinary=True)
    return legs,plan,gate,options


def fixture(*,late=False,send_error=None,admission_hook=None):
    n,events,sent,records,acceptance,clock=senders.sender_node(retained=True,late=late,send_error=send_error)
    legs,plan,gate,options=install_context(n)
    checks=[]
    def check(owner,goal):
        assert owner.command.depth==owner._lock.depth==1
        checks.append(copy.deepcopy(goal));record=velocity.require_arm_velocity_locked(owner,goal)
        if admission_hook is not None:admission_hook(owner,goal,len(checks))
        return record
    sender=load('_send_retained_arm_trajectory',time=clock,require_arm_velocity_locked=check)
    n._send_retained_arm_trajectory=lambda *a,**kw:sender(n,*a,**kw)
    make=load('_make_retained_arm_trajectory_goal');n._make_retained_arm_trajectory_goal=lambda x:make(n,x)
    gate.used=gate.send_started=True;n._contact_epoch+=1
    marker=half.WithdrawalLeg(options['withdrawal_timing'],1)
    goal,_=n._make_retained_arm_trajectory_goal([(legs[1][0],2.9/3.,'extraction')])
    def run():
        return n._send_retained_arm_trajectory(goal,5.8,[(legs[1][0],2.9/3.,'extraction')],
            'pick',leg_offset=1,velocity_admission=True,velocity_headroom=True,withdrawal_admission=marker)
    return n,legs,plan,gate,options,marker,goal,run,sent,records,checks,acceptance,clock


@pytest.mark.parametrize('value',[0,1,None,'true',[],{},np.bool_(True)])
def test_strict_boolean(value):
    with pytest.raises(ValueError):half.checked_enabled(value)


@pytest.mark.parametrize('enabled,ordinary',[(False,True),(False,False),(True,False),(True,None)])
def test_disabled_and_nonordinary_calls_do_not_touch_route_or_payload(enabled,ordinary):
    assert half.normal_options(NS(withdrawal_half_timing_enabled=enabled),None,None,None,ordinary=ordinary)=={}


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_normal_nonlift_call_does_not_evaluate_unbound_lift_plan(enabled):
    tree=timing.node_tree();pick=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick')
    calls=[n for n in ast.walk(pick) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='_execute_retained_arm_legs' and any(k.arg is None and isinstance(k.value,ast.Call)
           and isinstance(k.value.func,ast.Name) and k.value.func.id=='withdrawal_half_normal_options' for k in n.keywords)]
    assert len(calls)==1
    observed=[];n=NS(withdrawal_half_timing_enabled=enabled,_execute_retained_arm_legs=lambda *a,**kw:observed.append((a,kw)))
    scope=dict(self=n,first_segment=[],initial_pressure_gate=None,lift_enabled=False,top_row=True,
        staged_empty_gripper=False,deferred_post_retreat_return=True,withdrawal_half_normal_options=half.normal_options)
    assert 'lift_plan' not in scope
    assignments=[n for n in pick.body if isinstance(n,ast.Assign) and any(
        isinstance(target,ast.Name) and target.id=='ordinary_top_withdrawal'
        for target in n.targets)]
    assert len(assignments)==1
    exec(compile(ast.Module(body=assignments,type_ignores=[]),
        '<actual normal PICK timing scope>','exec'),scope)
    assert scope['ordinary_top_withdrawal'] is False
    eval(compile(ast.Expression(calls[0]),'<actual normal PICK call>','eval'),scope)
    assert observed==[(([],'pick'),{})]


@pytest.mark.parametrize('fault',['duration','phase','point','count','lift','gate'])
def test_selected_route_is_bound_to_exact_original_plan_and_gate(fault):
    n,legs,plan,gate,*_=fixture()
    gate.used=gate.send_started=False;n._contact_epoch-=1
    if fault=='duration':legs[2]=(legs[2][0],2.9,'extraction')
    elif fault=='phase':legs[2]=(legs[2][0],5.8,'recovery')
    elif fault=='point':legs[2]=(legs[2][0]+.001,5.8,'extraction')
    elif fault=='count':legs.append(legs[-1])
    elif fault=='lift':plan=LiftFirstPlan(plan.route,plan.terminal,plan.attached_corners,{'lift_m':.019})
    else:gate=object()
    with pytest.raises(RuntimeError,match='pick_recovery_failed'):
        half.normal_options(n,legs,gate,plan,ordinary=True)


def test_real_pressure_send_then_original_fresh_probe_permits_exactly_one_epoch(monkeypatch):
    import test_fine_lift_pressure_gate as pressure
    gate,n,now,wall,sent,events,joint,send=pressure.fixture(monkeypatch)
    n.withdrawal_half_timing_enabled=True;n.withdrawal_speed_scale=3.;n.additional_arm_time_scale=2.
    n.lift_first_extraction_lift_m=.020;n._cancel=TorsoHoldCancellation()
    n._held_book_corners=np.zeros((8,3));n._payload_monitor_enabled=True
    n._gravity_supported_payload=False;n._gripper_open_confirmed=False;n._active_place_scene_reference=None
    for field in half.MODEL_NAMES:setattr(n,field,object())
    route=tuple(np.asarray(gate.checked['left'])+np.array([0.,*([.001*i]*7)]) for i in range(5))
    legs=[(q,1. if i==0 else 5.8,'initial_shelf_lift' if i==0 else 'extraction') for i,q in enumerate(route)]
    plan=LiftFirstPlan(route,route[-1],n._held_book_corners,{'lift_m':.020})
    token=half.normal_options(n,legs,gate,plan,ordinary=True)['withdrawal_timing']
    assert gate.send(send)=='accepted-future' and gate.used is True and gate.send_started is True
    with n._lock:
        with pytest.raises(velocity.ArmVelocityAdmissionRejected):half._scope_locked(n,token,after_lift=True)
    n.grasp_contact_max_age=.2;n._wait_sim_duration=lambda seconds:True
    n._pinch_sample=lambda **kw:(True,.0183,True,True,True)
    epoch=n._contact_epoch
    assert load('_fresh_retention_probe')(n,'pick','initial_shelf_lift',leg=0)
    assert n._contact_epoch==epoch+1 and n._target_book_model==token.model
    with n._lock:half._scope_locked(n,token,after_lift=True)
    n._clear_target_contact_samples()
    with n._lock:
        with pytest.raises(velocity.ArmVelocityAdmissionRejected):half._scope_locked(n,token,after_lift=True)
    assert len(sent)==1


@pytest.mark.parametrize('used,started,epoch_delta',[(False,False,1),(True,False,1),(False,True,1),(True,True,0),(True,True,2)])
def test_withdrawal_needs_consumed_actual_gate_and_exact_postlift_epoch(used,started,epoch_delta):
    n,legs,plan,gate,options,marker,goal,run,sent,*_=fixture()
    gate.used=used;gate.send_started=started
    n._contact_epoch=options['withdrawal_timing'].contact_epoch+epoch_delta
    with pytest.raises(velocity.ArmVelocityAdmissionRejected):run()
    assert not sent and not n._goal_handles and not n._pending_retained_acceptances


@pytest.mark.parametrize('fault',['command','offset','gate','arm_scale','withdrawal_scale','probe','clearance'])
def test_marker_cannot_be_reused_in_recovery_or_another_execution_scope(fault):
    n,legs,plan,gate,options,marker,goal,run,sent,*_=fixture()
    gate.used=gate.send_started=False;n._contact_epoch-=1
    kw=dict(leg_offset=0,initial_pressure_gate=gate,arm_speed_scale=1.,withdrawal_speed_scale=3.,
        fresh_retention_phases=('initial_shelf_lift',),clearance_timing=None)
    command='pick'
    if fault=='command':command='place'
    elif fault=='offset':kw['leg_offset']=1
    elif fault=='gate':kw['initial_pressure_gate']=copy.copy(gate)
    elif fault=='arm_scale':kw['arm_speed_scale']=2.
    elif fault=='withdrawal_scale':kw['withdrawal_speed_scale']=1.
    elif fault=='probe':kw['fresh_retention_phases']=()
    else:kw['clearance_timing']=object()
    with pytest.raises(RuntimeError,match='pick_recovery_failed'):
        load('_execute_retained_arm_legs')(n,legs,command,**options,**kw)
    assert not sent and not n._goal_handles and not n._pending_retained_acceptances


@pytest.mark.parametrize('when',[1,2])
@pytest.mark.parametrize('fault',['epoch','gate','model','corners','cancel','base','right','stale'])
def test_final_locked_publication_rejects_mutation_during_either_velocity_gate(when,fault):
    def mutate(n,goal,count):
        if count!=when:return
        if fault=='epoch':n._contact_epoch+=1
        elif fault=='gate':state['gate'].send_started=False
        elif fault=='model':n.chain=object()
        elif fault=='corners':n._held_book_corners[0,0]=.001
        elif fault=='cancel':n._cancel.set()
        elif fault=='base':n._staging_odom['pose'][0]=.003
        elif fault=='right':n.joints[RIGHT_ARM_JOINTS[0]]=.002
        else:n._joint_stamps_ns[IK_JOINTS[1]]=senders.NOW-150_000_001
    state={};n,legs,plan,gate,options,marker,goal,run,sent,records,*_=fixture(admission_hook=mutate);state['gate']=gate
    with pytest.raises(velocity.ArmVelocityAdmissionRejected):run()
    assert not sent and not n._goal_handles and not n._pending_retained_acceptances
    assert len(records)==1 and records[0]['admitted'] is False


def test_real_goal_copy_half_time_and_original_watchdog_are_kept():
    n,legs,plan,gate,options,marker,goal,run,sent,records,checks,acceptance,clock=fixture()
    original=copy.deepcopy(goal)
    assert run()==(True,False)
    assert len(sent)==1 and len(checks)==2 and goal==original
    point=sent[0].trajectory.points[0]
    assert abs(timing.seconds(point.time_from_start)-.483333333)<=1e-9
    assert list(point.positions)==list(legs[1][0][1:])
    record=records[0]['additional_arm_timing']
    assert record['nominal_duration_seconds']==5.8 and record['unchanged_nominal_watchdog']
    assert not n._goal_handles and not n._pending_retained_acceptances


def test_velocity_floor_uses_exact_eighty_percent_and_does_not_edit_path():
    n,legs,plan,gate,options,marker,goal,run,sent,records,*_=fixture()
    # A conservative test limit forces an extension above the nominal half time.
    n._faster_arm_velocity_limits=(.025,)*7
    assert run()==(True,False)
    point=sent[0].trajectory.points[0];ns=point.time_from_start.sec*10**9+point.time_from_start.nanosec
    assert ns>483_333_334 and tuple(point.positions)==tuple(legs[1][0][1:])
    for delta,limit in zip(point.positions,n._faster_arm_velocity_limits):
        assert abs(Fraction.from_float(delta))*10**9<=Fraction(4,5)*Fraction.from_float(limit)*ns


def test_actual_executor_and_four_senders_keep_initial_lift_probe_order_and_watchdogs():
    n,legs,plan,gate,options,marker,goal,run,sent,records,*_=fixture()
    gate.used=gate.send_started=False;n._contact_epoch-=1
    original=copy.deepcopy(legs);calls=[];sequence=[];ordinary_sender=n._send_retained_arm_trajectory
    def send(g,watchdog,active,command,**kwargs):
        index=kwargs['leg_offset'];calls.append((copy.deepcopy(g),watchdog,kwargs))
        sequence.append(('send',index))
        if index==0:
            # The unchanged pressure sender has its separate actual send test.
            assert kwargs==dict(leg_offset=0,initial_pressure_gate=gate)
            gate.used=gate.send_started=True
            result=True,False
        else:result=ordinary_sender(g,watchdog,active,command,**kwargs)
        n.joints.update(zip(IK_JOINTS,legs[index][0]));return result
    n._send_retained_arm_trajectory=send
    clear_unlocked=load('_clear_target_contact_samples_unlocked')
    n._clear_target_contact_samples_unlocked=lambda **kw:clear_unlocked(n,**kw)
    clear=load('_clear_target_contact_samples');n._clear_target_contact_samples=lambda **kw:clear(n,**kw)
    n._wait_sim_duration=lambda seconds:True;n._pinch_sample=lambda **kw:(True,.0181,True,True,True)
    fresh=load('_fresh_retention_probe')
    n._fresh_retention_probe=lambda *a,**kw:(sequence.append(('probe',kw['leg'])) or fresh(n,*a,**kw))
    n._retention_after_leg=lambda *a:(sequence.append(('retention',a[-1])) or True)
    n._measured_left_solution=lambda:np.asarray([n.joints[name] for name in IK_JOINTS])
    wait=load('_wait_for_retained_endpoint',time=NS(monotonic=lambda:0.,sleep=lambda _:pytest.fail('already measured endpoint')))
    n._wait_for_retained_endpoint=lambda *a,**kw:(sequence.append(('endpoint',kw['leg'])) or wait(n,*a,**kw))
    assert load('_execute_retained_arm_legs')(n,legs,'pick',initial_pressure_gate=gate,
        fresh_retention_phases=('initial_shelf_lift',),withdrawal_speed_scale=3.,**options)==(True,5,False)
    assert [c[1] for c in calls]==[1.,5.8,5.8,5.8,5.8]
    assert timing.seconds(calls[0][0].trajectory.points[0].time_from_start)==1.
    assert len(sent)==4 and all(abs(timing.seconds(g.trajectory.points[0].time_from_start)-.483333333)<=1e-9 for g in sent)
    assert all(np.array_equal(a[0],b[0]) and a[1:]==b[1:] for a,b in zip(legs,original))
    assert [tuple(g.trajectory.points[0].positions) for g in sent]==[tuple(q[0][1:]) for q in legs[1:]]
    assert sequence==[('send',0),('probe',0),*sum(([('send',i),('endpoint',i),('retention',i)] for i in range(1,5)),[])]
    assert n._contact_epoch==options['withdrawal_timing'].contact_epoch+1
    assert not n._pending_retained_acceptances and not n._goal_handles


@pytest.mark.parametrize('fault',['endpoint','hazard','epoch','stale'])
def test_endpoint_failure_after_action_cannot_advance_retention_or_next_leg(fault):
    n,legs,plan,gate,options,marker,goal,run,sent,records,*_=fixture()
    gate.used=gate.send_started=False;n._contact_epoch-=1
    sequence=[]
    def send(g,allowance,active,command,**kw):
        index=kw['leg_offset'];sequence.append(('send',index,allowance))
        n.joints.update(zip(IK_JOINTS,legs[index][0]))
        if index==0:gate.used=gate.send_started=True
        return True,False
    n._send_retained_arm_trajectory=send
    n._fresh_retention_probe=lambda *a,**kw:(setattr(n,'_contact_epoch',n._contact_epoch+1) or sequence.append(('probe',kw['leg'])) or True)
    n._retention_after_leg=lambda *a:(sequence.append(('retention',a[-1])) or True)
    def wait(*a,**kw):
        sequence.append(('endpoint',kw['leg']))
        if fault=='endpoint':return None
        if fault=='hazard':n._payload_hazard_reason=lambda **kw:'lost'
        elif fault=='epoch':n._contact_epoch+=1
        else:n._joint_stamps_ns[IK_JOINTS[1]]=senders.NOW-150_000_001
        return legs[1][0]
    n._wait_for_retained_endpoint=wait
    with pytest.raises(RuntimeError,match='pick_recovery_failed'):
        load('_execute_retained_arm_legs')(n,legs,'pick',initial_pressure_gate=gate,
            fresh_retention_phases=('initial_shelf_lift',),withdrawal_speed_scale=3.,**options)
    assert sequence==[('send',0,1.),('probe',0),('send',1,5.8),('endpoint',1)]


@pytest.mark.parametrize('fault',['late','client_error'])
def test_unknown_acceptance_keeps_original_pending_ownership(fault):
    error=velocity.ArmVelocityAdmissionRejected('client-side exception after registration') if fault=='client_error' else None
    n,legs,plan,gate,options,marker,goal,run,sent,records,checks,acceptance,clock=fixture(late=fault=='late',send_error=error)
    with pytest.raises(timing.RetainedMotionNotStopped):run()
    assert len(sent)==1 and n._cancel.is_set() and len(n._pending_retained_acceptances)==1
    if fault=='late':assert len(acceptance.callbacks)==1


def test_accepted_unknown_terminal_retains_handle_and_requests_cancel():
    n,legs,plan,gate,options,marker,goal,run,sent,records,checks,acceptance,clock=fixture()
    handle=acceptance.result();cancelled=[]
    handle.get_result_async=lambda:(_ for _ in ()).throw(RuntimeError('no result future'))
    handle.cancel_goal_async=lambda:(cancelled.append(True) or senders.Future(None))
    with pytest.raises(timing.RetainedMotionNotStopped):run()
    assert sent and cancelled==[True] and n._goal_handles==[handle] and n._cancel.is_set()
    assert not n._pending_retained_acceptances


def test_default_executor_exact_parent_messages_watchdogs_checks_and_indices():
    path=ROOT/'erc_phase1_solution/manipulation_node.py';text=path.read_text()
    from candidate_composition_support import restore_all_books_source
    text=restore_all_books_source(text)
    inverse=json.loads((ROOT/'test/fixtures/withdrawal_half_timing_inverse.json').read_bytes())
    for row in reversed(inverse['fragments']):
        assert text.count(row['new'])==1;text=text.replace(row['new'],row['old'],1)
    assert hashlib.sha256(text.encode()).hexdigest()==inverse['parent_sha256']
    owner=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    fn=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_execute_retained_arm_legs')
    current=load('_execute_retained_arm_legs');scope=dict(current.__globals__)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<exact34 executor>','exec'),scope)
    observed=[]
    for f in (current,scope['_execute_retained_arm_legs']):
        n,calls,probes=timing.retained_node();n.withdrawal_half_timing_enabled=False
        legs=[(np.array([.35,*([.02*i]*7)]),1. if i==0 else 5.8,'initial_shelf_lift' if i==0 else 'extraction') for i in range(5)]
        result=f(n,legs,'pick',withdrawal_speed_scale=3.,initial_pressure_gate='unchanged',fresh_retention_phases=('initial_shelf_lift',))
        observed.append((result,[(list(c[0].trajectory.points[0].positions),timing.seconds(c[0].trajectory.points[0].time_from_start),c[1],c[4]) for c in calls],probes))
    assert observed[0]==observed[1]
