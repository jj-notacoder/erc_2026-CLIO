"""Actual guarded withdrawal flow/sender; controlled feedback is not physics."""
import ast
import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import withdrawal_half_timing as half
from erc_phase1_solution import arm_velocity_admission as velocity
from erc_phase1_solution.motion_profiles import IK_JOINTS
import test_withdrawal_half_timing as original

ROOT=Path(__file__).resolve().parents[1]
FLAG='withdrawal_quarter_timing_enabled'


def fixture(flag=True, **kwargs):
    n,legs,plan,gate,unused,unused_marker,unused_goal,unused_run,sent,records,checks,acceptance,clock=original.fixture(**kwargs)
    if flag!='absent':setattr(n,FLAG,flag)
    gate.used=gate.send_started=False;n._contact_epoch-=1
    options=half.normal_options(n,legs,gate,plan,ordinary=True)
    gate.used=gate.send_started=True;n._contact_epoch+=1
    seconds,marker=half.timing_input(n,options['withdrawal_timing'],1,*legs[1])
    goal,_=n._make_retained_arm_trajectory_goal([(legs[1][0],seconds/3.,'extraction')])
    def run():
        return n._send_retained_arm_trajectory(goal,5.8,[(legs[1][0],seconds/3.,'extraction')],
            'pick',leg_offset=1,velocity_admission=True,velocity_headroom=True,withdrawal_admission=marker)
    return NS(n=n,legs=legs,plan=plan,gate=gate,options=options,marker=marker,seconds=seconds,
              goal=goal,run=run,sent=sent,records=records,checks=checks,acceptance=acceptance,clock=clock)


def constructor(flag, half_enabled):
    owner=original.timing.node_tree()
    init=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    at=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr==FLAG for t in n.targets))
    block=init.body[at:at+3]
    assert isinstance(block[0],ast.Assign) and all(isinstance(n,ast.If) for n in block[1:])
    n=NS(get_parameter=lambda key:NS(value=flag),withdrawal_half_timing_enabled=half_enabled)
    exec(compile(ast.fix_missing_locations(ast.Module(body=block,type_ignores=[])),'actual_quarter_constructor','exec'),{'self':n})
    return n


@pytest.mark.parametrize('flag',[None,0,1,'true',[],{},np.bool_(True)])
def test_actual_constructor_rejects_non_boolean_before_any_motion(flag):
    with pytest.raises(ValueError,match='must be Boolean'):constructor(flag,True)


@pytest.mark.parametrize('flag,half_enabled,accepted',[(False,False,True),(False,True,True),(True,False,False),(True,True,True)])
def test_quarter_requires_the_existing_guarded_half_option(flag,half_enabled,accepted):
    if accepted:assert getattr(constructor(flag,half_enabled),FLAG) is flag
    else:
        with pytest.raises(ValueError,match='requires withdrawal_half'):constructor(flag,half_enabled)


@pytest.mark.parametrize('flag,expected_ns',[('absent',483333333),(False,483333333),(True,241666667)])
def test_actual_serialized_goal_uses_selected_fraction_and_original_watchdog(flag,expected_ns):
    f=fixture(flag);before=copy.deepcopy(f.goal)
    assert f.run()==(True,False)
    assert len(f.sent)==1 and len(f.checks)==2 and f.goal==before
    point=f.sent[0].trajectory.points[0]
    assert point.time_from_start.sec*10**9+point.time_from_start.nanosec==expected_ns
    assert tuple(point.positions)==tuple(f.legs[1][0][1:])
    assert f.records[0]['additional_arm_timing']['nominal_duration_seconds']==5.8
    assert f.records[0]['additional_arm_timing']['unchanged_nominal_watchdog'] is True
    assert not f.n._goal_handles and not f.n._pending_retained_acceptances
    with pytest.raises(FrozenInstanceError):f.marker.owner.input_fraction=.125


@pytest.mark.parametrize('when',[1,2])
@pytest.mark.parametrize('value',[False,None,'true'])
def test_fraction_change_during_either_actual_velocity_gate_refuses_publication(when,value):
    def mutate(n,goal,count):
        if count==when:setattr(n,FLAG,value)
    f=fixture(admission_hook=mutate)
    with pytest.raises(velocity.ArmVelocityAdmissionRejected):f.run()
    assert not f.sent and not f.n._goal_handles and not f.n._pending_retained_acceptances
    assert len(f.records)==1 and f.records[0]['admitted'] is False


def test_unlisted_fraction_in_a_replaced_owner_cannot_publish():
    f=fixture();marker=half.WithdrawalLeg(replace(f.marker.owner,input_fraction=.125),1)
    with pytest.raises(velocity.ArmVelocityAdmissionRejected):
        f.n._send_retained_arm_trajectory(f.goal,5.8,[(f.legs[1][0],f.seconds/3.,'extraction')],
            'pick',leg_offset=1,velocity_admission=True,velocity_headroom=True,withdrawal_admission=marker)
    assert not f.sent and not f.n._pending_retained_acceptances and not f.n._goal_handles


def test_current_start_velocity_floor_can_lengthen_quarter_goal_without_changing_path():
    f=fixture();f.n._faster_arm_velocity_limits=(.05,)*7
    assert f.run()==(True,False)
    point=f.sent[0].trajectory.points[0];ns=point.time_from_start.sec*10**9+point.time_from_start.nanosec
    assert ns>241666667 and tuple(point.positions)==tuple(f.legs[1][0][1:])
    for delta,limit in zip(point.positions,f.n._faster_arm_velocity_limits):
        assert abs(Fraction.from_float(delta))*10**9<=Fraction(4,5)*Fraction.from_float(limit)*ns
    assert f.records[0]['additional_arm_timing']['unchanged_nominal_watchdog'] is True


def test_actual_executor_retains_lift_epoch_four_endpoints_and_original_watchdogs():
    f=fixture();n=f.n;gate=f.gate;legs=f.legs
    gate.used=gate.send_started=False;n._contact_epoch-=1
    before=copy.deepcopy(legs);calls=[];sequence=[];send_original=n._send_retained_arm_trajectory
    def send(goal,watchdog,active,command,**kw):
        index=kw['leg_offset'];calls.append((copy.deepcopy(goal),watchdog,kw));sequence.append(('send',index))
        if index==0:
            assert kw==dict(leg_offset=0,initial_pressure_gate=gate)
            gate.used=gate.send_started=True;result=True,False
        else:result=send_original(goal,watchdog,active,command,**kw)
        n.joints.update(zip(IK_JOINTS,legs[index][0]));return result
    n._send_retained_arm_trajectory=send
    clear_unlocked=original.load('_clear_target_contact_samples_unlocked')
    n._clear_target_contact_samples_unlocked=lambda **kw:clear_unlocked(n,**kw)
    clear=original.load('_clear_target_contact_samples');n._clear_target_contact_samples=lambda **kw:clear(n,**kw)
    n._wait_sim_duration=lambda seconds:True;n._pinch_sample=lambda **kw:(True,.0181,True,True,True)
    fresh=original.load('_fresh_retention_probe')
    n._fresh_retention_probe=lambda *a,**kw:(sequence.append(('probe',kw['leg'])) or fresh(n,*a,**kw))
    n._retention_after_leg=lambda *a:(sequence.append(('retention',a[-1])) or True)
    n._measured_left_solution=lambda:np.asarray([n.joints[name] for name in IK_JOINTS])
    wait=original.load('_wait_for_retained_endpoint',time=NS(monotonic=lambda:0.,sleep=lambda _:pytest.fail('already measured endpoint')))
    n._wait_for_retained_endpoint=lambda *a,**kw:(sequence.append(('endpoint',kw['leg'])) or wait(n,*a,**kw))
    assert original.load('_execute_retained_arm_legs')(n,legs,'pick',initial_pressure_gate=gate,
        fresh_retention_phases=('initial_shelf_lift',),withdrawal_speed_scale=3.,**f.options)==(True,5,False)
    assert [c[1] for c in calls]==[1.,5.8,5.8,5.8,5.8]
    assert original.timing.seconds(calls[0][0].trajectory.points[0].time_from_start)==1.
    assert len(f.sent)==4
    assert all(g.trajectory.points[0].time_from_start.sec==0 and g.trajectory.points[0].time_from_start.nanosec==241666667 for g in f.sent)
    assert [tuple(g.trajectory.points[0].positions) for g in f.sent]==[tuple(q[0][1:]) for q in legs[1:]]
    assert all(np.array_equal(a[0],b[0]) and a[1:]==b[1:] for a,b in zip(legs,before))
    assert sequence==[('send',0),('probe',0),*sum(([('send',i),('endpoint',i),('retention',i)] for i in range(1,5)),[])]
    assert n._contact_epoch==f.marker.owner.contact_epoch+1
    assert not n._pending_retained_acceptances and not n._goal_handles


@pytest.mark.parametrize('fault',['endpoint','hazard','epoch','stale'])
def test_after_send_endpoint_failure_aborts_before_retention_or_next_leg(fault):
    f=fixture();n=f.n;gate=f.gate;legs=f.legs
    gate.used=gate.send_started=False;n._contact_epoch-=1;sequence=[]
    def send(goal,watchdog,active,command,**kw):
        index=kw['leg_offset'];sequence.append(('send',index,watchdog))
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
        else:n._joint_stamps_ns[IK_JOINTS[1]]=original.senders.NOW-150000001
        return legs[1][0]
    n._wait_for_retained_endpoint=wait
    with pytest.raises(RuntimeError,match='pick_recovery_failed'):
        original.load('_execute_retained_arm_legs')(n,legs,'pick',initial_pressure_gate=gate,
            fresh_retention_phases=('initial_shelf_lift',),withdrawal_speed_scale=3.,**f.options)
    assert sequence==[('send',0,1.),('probe',0),('send',1,5.8),('endpoint',1)]


@pytest.mark.parametrize('fault',['late','client_error'])
def test_unknown_acceptance_keeps_original_pending_owner_and_cancel(fault):
    error=velocity.ArmVelocityAdmissionRejected('client failure after registration') if fault=='client_error' else None
    f=fixture(late=fault=='late',send_error=error)
    with pytest.raises(original.timing.RetainedMotionNotStopped):f.run()
    assert len(f.sent)==1 and f.n._cancel.is_set() and len(f.n._pending_retained_acceptances)==1
    if fault=='late':assert len(f.acceptance.callbacks)==1


def test_unknown_terminal_retains_handle_and_requests_cancel():
    f=fixture();handle=f.acceptance.result();cancelled=[]
    handle.get_result_async=lambda:(_ for _ in ()).throw(RuntimeError('result unavailable'))
    handle.cancel_goal_async=lambda:(cancelled.append(True) or original.senders.Future(None))
    with pytest.raises(original.timing.RetainedMotionNotStopped):f.run()
    assert f.sent and cancelled==[True] and f.n._goal_handles==[handle] and f.n._cancel.is_set()
    assert not f.n._pending_retained_acceptances


def test_declared_option_is_default_off():
    path=ROOT/'erc_phase1_solution/manipulation_node.py'
    owner=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    declarations=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_declare_parameters')
    values=[v for d in ast.walk(declarations) if isinstance(d,ast.Dict) for k,v in zip(d.keys,d.values)
            if isinstance(k,ast.Constant) and k.value==FLAG]
    assert len(values)==1 and isinstance(values[0],ast.Constant) and values[0].value is False
