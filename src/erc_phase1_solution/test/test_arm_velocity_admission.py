"""Prepared small actual-helper/actual-sender cases; no physical safety claim.

Not executed during Run44. Serialized ROS-like messages and real plain locks are
used; fake immediate/delayed futures explicitly limit these lifecycle models.
"""
import ast
from candidate_composition_support import without_torso_tool
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import numpy as np
import pytest
from erc_phase1_solution import arm_velocity_admission as gate
import test_optional_arm_timing as timing

NAMES=timing.ARM_JOINTS
LIMITS=(1.95,1.95,3.95,3.95,3.95,3.95,3.95)
NOW=2_000_000_000
HERE=Path(__file__).resolve().parent


def stamp(ns):return NS(sec=ns//1_000_000_000,nanosec=ns%1_000_000_000)

def goal(points=None):
    points=[([.2]*7,1_000_000_000)] if points is None else points
    return NS(trajectory=NS(joint_names=list(NAMES),header=NS(stamp=stamp(0)),
        points=[NS(positions=list(q),time_from_start=stamp(t),velocities=[],accelerations=[],effort=[])
                for q,t in points]))

def node():
    return NS(joints=dict.fromkeys(NAMES,0.),_joint_stamps_ns=dict.fromkeys(NAMES,NOW),
              _faster_arm_velocity_limits=LIMITS,_faster_arm_velocity_urdf='fixture/official.urdf',
              get_clock=lambda:NS(now=lambda:NS(nanoseconds=NOW)))


def test_all_exact_points_and_serialized_times_recorded_without_mutation():
    g=goal([([.2]*7,500_000_000),([.3]*7,1_000_000_001)])
    before=copy.deepcopy(g)
    record=gate.require_arm_velocity_locked(node(),g)
    assert vars(g.trajectory)==vars(before.trajectory)
    assert record['points'][1]['time_from_start_ns']==1_000_000_001
    assert record['points'][0]['positions']==[.2]*7
    assert record['start_positions']==[0.]*7 and record['velocity_limits']==list(LIMITS)
    assert record['producer_stamps_ns']==[NOW]*7 and record['record_complete']
    record['points'][0]['positions'][0]=100.
    assert g.trajectory.points[0].positions[0]==.2


@pytest.mark.parametrize('age,accepted',[(150_000_000,True),(150_000_001,False),
    (-50_000_000,True),(-50_000_001,False),(0,True)])
def test_feedback_exact_age_boundaries(age,accepted):
    n=node();n._joint_stamps_ns[NAMES[3]]=NOW-age
    if accepted:assert gate.require_arm_velocity_locked(n,goal())
    else:
        with pytest.raises(gate.ArmVelocityAdmissionRejected,match='feedback stale') as caught:
            gate.require_arm_velocity_locked(n,goal())
        assert caught.value.record['producer_stamps_ns'][3]==NOW-age


@pytest.mark.parametrize('kind',['missing','nonfinite','future','bad_stamp','missing_limits','bad_clock'])
def test_unusable_feedback_is_typed_pre_send_rejection(kind):
    n=node()
    if kind=='missing':del n.joints[NAMES[0]]
    elif kind=='nonfinite':n.joints[NAMES[0]]=math.nan
    elif kind=='future':n._joint_stamps_ns[NAMES[0]]=NOW+50_000_001
    elif kind=='bad_stamp':n._joint_stamps_ns[NAMES[0]]=True
    elif kind=='missing_limits':del n._faster_arm_velocity_limits
    else:n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=0))
    with pytest.raises(gate.ArmVelocityAdmissionRejected):gate.require_arm_velocity_locked(n,goal())


@pytest.mark.parametrize('kind',['joint_order','delayed','velocities','accelerations','effort',
    'zero_time','descending','bad_nsec','huge_sec','bool_sec','empty','too_many',
    'nonfinite','wrong_positions','zero_limit','overflow','missing_header'])
def test_invalid_serialized_goal_rejects_with_bounded_record(kind):
    g=goal();limits=LIMITS
    p=g.trajectory.points[0]
    if kind=='joint_order':g.trajectory.joint_names.reverse()
    elif kind=='delayed':g.trajectory.header.stamp=stamp(1)
    elif kind in ('velocities','accelerations','effort'):setattr(p,kind,[0.]*7)
    elif kind=='zero_time':p.time_from_start=stamp(0)
    elif kind=='descending':g.trajectory.points.append(copy.deepcopy(p))
    elif kind=='bad_nsec':p.time_from_start.nanosec=1_000_000_000
    elif kind=='huge_sec':p.time_from_start.sec=2_147_483_648
    elif kind=='bool_sec':p.time_from_start.sec=True
    elif kind=='empty':g.trajectory.points=[]
    elif kind=='too_many':g.trajectory.points=[p]*257
    elif kind=='nonfinite':p.positions[0]=math.inf
    elif kind=='wrong_positions':p.positions.pop()
    elif kind=='zero_limit':limits=(0.,*LIMITS[1:])
    elif kind=='overflow':p.positions[0]=1.7e308
    else:del g.trajectory.header
    with pytest.raises(gate.ArmVelocityAdmissionRejected) as caught:
        gate.check_serialized_arm_goal(g,[-1.7e308 if kind=='overflow' else 0.]*7,limits)
    assert len(caught.value.record.get('points',()))<=256


def test_one_nanosecond_rounding_and_later_segment_use_actual_times():
    g=goal([([1.]*7,1_000_000_000)])
    assert gate.check_serialized_arm_goal(g,[0.]*7,[1.]*7)
    g.trajectory.points[0].time_from_start=stamp(999_999_999)
    with pytest.raises(gate.ArmVelocityAdmissionRejected,match='segment 0'):
        gate.check_serialized_arm_goal(g,[0.]*7,[1.]*7)
    g=goal([([.1]*7,1_000_000_000),([.3]*7,1_100_000_000),([.4]*7,2_000_000_000)])
    with pytest.raises(gate.ArmVelocityAdmissionRejected,match='segment 1') as caught:
        gate.check_serialized_arm_goal(g,[0.]*7,[1.]*7)
    assert len(caught.value.record['points'])==3


class Lock:
    def __init__(self,label,events,reentrant=False):
        self.lock=threading.RLock() if reentrant else threading.Lock()
        self.label,self.events,self.depth=label,events,0
    def __enter__(self):
        assert self.lock.acquire(timeout=1.),'deadlock'
        self.depth+=1;self.events.append(('enter',self.label));return self
    def __exit__(self,*args):
        self.events.append(('leave',self.label));self.depth-=1;self.lock.release()


class Future:
    def __init__(self,result=None,done=True):self.value,self.ready,self.callbacks=result,done,[]
    def done(self):return self.ready
    def result(self):return self.value
    def add_done_callback(self,callback):self.callbacks.append(callback)


def sender_node(*,retained=False,send_error=None,late=False,publisher_error=False):
    n=node();events=[];sent=[];records=[]
    n._lock=Lock('sensor',events);n.command=Lock('command',events,True)
    n._adaptive_command_guard=lambda:n.command
    n._cancel=threading.Event();n._goal_handles=[];n._pending_retained_acceptances=set()
    n.timeout=1.;n._payload_hazard_reason=lambda **kw:None
    n._trajectory_leg_at_elapsed_time=lambda legs,elapsed:(0,'fixture')
    n._valid_retained_terminal_result=lambda wrapped:wrapped.status in (4,5,6)
    n._cancel_retained_goal_and_confirm=lambda *a:True
    n._cancel_goal_and_confirm=lambda *a:True
    n._cancel_late_retained_goal=lambda *args:events.append(('late_cancel',))
    handle=NS(accepted=True,get_result_async=lambda:Future(NS(status=4)))
    acceptance=Future(handle,done=not late)
    def send(g):
        assert n.command.depth and n._lock.depth
        if retained:assert len(n._pending_retained_acceptances)==1
        events.append(('send',));sent.append(copy.deepcopy(g))
        if send_error is not None:raise send_error
        return acceptance
    n.arm_client=NS(wait_for_server=lambda **kw:True,send_goal_async=send)
    n._wait_future=lambda f,t:f.result()
    def publish(event,**data):
        if event=='arm_velocity_admission':
            assert not n.command.depth and not n._lock.depth
            records.append(data);events.append(('record',))
            if publisher_error:raise RuntimeError('telemetry failure')
    n._publish_status=publish
    ticks=iter([0.,2.]) if late else None
    clock=NS(monotonic=(lambda:next(ticks)) if late else (lambda:0.),sleep=lambda _:None)
    return n,events,sent,records,acceptance,clock


def run_sender(n,clock,*,retained,g=None):
    if retained:
        return timing.load('_send_retained_arm_trajectory',time=clock)(n,goal() if g is None else g,
            1.,[(np.zeros(8),.8,'fixture')],'place',velocity_admission=True)
    return timing.load('_follow',time=clock)(n,n.arm_client,NAMES,[.2]*7,1.,trajectory_duration=.8)


@pytest.mark.parametrize('retained',[False,True])
@pytest.mark.parametrize('publisher_error',[False,True])
def test_actual_sender_has_lock_order_once_evidence_and_unchanged_lifecycle(retained,publisher_error):
    n,events,sent,records,_,clock=sender_node(retained=retained,publisher_error=publisher_error)
    result=[]
    worker=threading.Thread(target=lambda:result.append(run_sender(n,clock,retained=retained)),daemon=True)
    worker.start();worker.join(2.)
    assert not worker.is_alive() and result==([(True,False)] if retained else [True])
    assert len(sent)==len(records)==1 and records[0]['admitted']
    assert not n._pending_retained_acceptances and not n._goal_handles
    at=events.index(('send',))
    assert events[at-2:at]==[('enter','command'),('enter','sensor')]
    assert events[at+1:at+3]==[('leave','sensor'),('leave','command')]


@pytest.mark.parametrize('retained',[False,True])
@pytest.mark.parametrize('fault',['stale','slope'])
def test_rejection_occurs_before_send_or_pending_token_without_uncertain_motion(retained,fault):
    n,_,sent,records,_,clock=sender_node(retained=retained)
    if fault=='stale':n._joint_stamps_ns[NAMES[0]]=NOW-150_000_001
    else:n.joints[NAMES[0]]=-10.
    with pytest.raises(gate.ArmVelocityAdmissionRejected):run_sender(n,clock,retained=retained)
    assert not sent and not n._pending_retained_acceptances and not n._goal_handles
    assert not n._cancel.is_set()
    assert len(records)==1 and not records[0]['admitted']
    assert records[0]['start_positions'][0]==n.joints[NAMES[0]]


@pytest.mark.parametrize('retained',[False,True])
def test_preexisting_cancel_and_hazard_still_prevent_scaled_send(retained):
    n,_,sent,records,_,clock=sender_node(retained=retained)
    n._cancel.set()
    result=run_sender(n,clock,retained=retained)
    assert result==((False,False) if retained else False)
    assert not sent and not records
    n._cancel.clear()
    if retained:n._payload_hazard_reason=lambda **kw:'payload_robot_contact'
    else:n._payload_robot_watchdog_enabled=True;n._target_robot_contact_latched=True
    assert run_sender(n,clock,retained=retained)==((False,True) if retained else False)
    assert not sent


def test_pre_send_hook_can_cancel_and_server_wait_can_make_feedback_stale():
    n,_,sent,records,_,clock=sender_node()
    f=timing.load('_follow',time=clock)
    assert not f(n,n.arm_client,NAMES,[.2]*7,1.,trajectory_duration=.8,pre_send_check=n._cancel.set)
    assert not sent and not records
    n._cancel.clear()
    n.arm_client.wait_for_server=lambda **kw:(n._joint_stamps_ns.update({NAMES[0]:NOW-150_000_001}) or True)
    with pytest.raises(gate.ArmVelocityAdmissionRejected):f(n,n.arm_client,NAMES,[.2]*7,1.,trajectory_duration=.8)
    assert not sent and len(records)==1


def test_same_exception_class_from_client_remains_unknown_acceptance():
    error=gate.ArmVelocityAdmissionRejected('client raised after registration')
    n,_,sent,records,_,clock=sender_node(retained=True,send_error=error)
    with pytest.raises(timing.RetainedMotionNotStopped) as caught:run_sender(n,clock,retained=True)
    assert caught.value.__cause__ is error and n._cancel.is_set()
    assert len(sent)==1 and len(n._pending_retained_acceptances)==1
    assert len(records)==1 and records[0]['admitted']


def test_late_acceptance_registration_and_original_cancel_callback_are_retained():
    n,events,sent,records,acceptance,clock=sender_node(retained=True,late=True)
    with pytest.raises(timing.RetainedMotionNotStopped):run_sender(n,clock,retained=True)
    assert n._cancel.is_set() and len(n._pending_retained_acceptances)==1
    assert len(acceptance.callbacks)==1 and len(records)==1 and records[0]['admitted']
    acceptance.callbacks[0](acceptance)
    assert ('late_cancel',) in events


def test_retained_executor_propagates_clean_admission_rejection_without_recovery():
    error=gate.ArmVelocityAdmissionRejected('too fast')
    n,calls,probes=timing.retained_node(error)
    with pytest.raises(gate.ArmVelocityAdmissionRejected):
        timing.load('_execute_retained_arm_legs')(n,[(np.zeros(8),.8,'a')],'place',arm_speed_scale=1.25)
    assert len(calls)==1 and calls[0][-1]['velocity_admission'] is True
    assert not probes and not n._cancel.is_set()


def test_only_scaled_supported_group_and_scaled_place_legs_opt_in():
    fn=timing.method_ast('_execute_cached_post_retreat_compaction')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='_send_retained_arm_trajectory']
    marked=[n for n in calls if any('velocity_admission' in ast.unparse(k.value) for k in n.keywords)]
    assert len(calls)==3 and len(marked)==2
    intended={ast.unparse(call.args[2]):call for call in marked}
    assert set(intended)=={'timed_groups[0]','timed_groups[2]'}
    extension,=[k.value for k in intended['timed_groups[0]'].keywords if k.arg is None]
    assert isinstance(extension,ast.IfExp) and ast.unparse(extension.test)=='extension_half_enabled'
    assert ast.literal_eval(extension.body)=={'velocity_admission':True,'velocity_headroom':True}
    assert ast.literal_eval(extension.orelse)=={}
    supported,=[k.value for k in intended['timed_groups[2]'].keywords if k.arg is None]
    assert isinstance(supported,ast.IfExp) and ast.unparse(supported.test)=='arm_speed_scale != 1.0'
    assert ast.literal_eval(supported.body)=={'velocity_admission':True}
    assert ast.literal_eval(supported.orelse)=={}
    n,calls,_=timing.retained_node()
    timing.load('_execute_retained_arm_legs')(n,[(np.zeros(8),.8,'a')],'place')
    assert 'velocity_admission' not in calls[0][-1]


def test_optional_constructor_loads_same_urdf_limits_once_and_keeps_tuple(tmp_path,monkeypatch):
    from erc_phase1_solution import empty_pickup_collision as module
    urdf=tmp_path/'test.urdf'
    names=('torso_lift_joint',*NAMES)
    urdf.write_text('<robot>'+''.join(f'<joint name="{name}"><limit velocity="{i+1}"/></joint>' for i,name in enumerate(names))+'</robot>')
    actual=module.joint_velocity_limits;calls=[]
    monkeypatch.setattr(module,'joint_velocity_limits',lambda p:(calls.append(p) or actual(p)))
    init=timing.method_ast('__init__')
    blocks=[n for n in init.body if isinstance(n,ast.If) and any(
        isinstance(x,ast.Assign) and any(isinstance(t,ast.Attribute)
            and t.attr=='_faster_arm_velocity_limits' for t in x.targets) for x in n.body)]
    assert len(blocks)==1
    block=ast.Module(body=blocks,type_ignores=[])
    n=NS(placement_transport_speed_scale=1.,withdrawal_speed_scale=1.,empty_pickup_setup_retiming_enabled=False)
    scope=dict(self=n,urdf=urdf,__name__='erc_phase1_solution.fixture',__package__='erc_phase1_solution')
    exec(compile(ast.fix_missing_locations(block),'<actual optional init>','exec'),scope)
    assert not calls and not hasattr(n,'_faster_arm_velocity_limits')
    n.placement_transport_speed_scale=1.25
    exec(compile(ast.fix_missing_locations(block),'<actual optional init>','exec'),scope)
    assert calls==[urdf] and n._faster_arm_velocity_limits==tuple(float(i) for i in range(2,9))
    assert n._faster_arm_velocity_urdf==str(urdf)


def test_complete_default_source_ast_restores_to_sealed_arm_candidate():
    path=HERE/'fixtures/arm_velocity_node_inverse.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==INVERSE_SHA
    inverse=json.loads(path.read_text())
    source=(HERE.parent/'erc_phase1_solution/manipulation_node.py').read_text()
    source=without_torso_tool(source)
    for item in reversed(inverse['replacements']):
        assert source.count(item['after'])==item['count']
        source=source.replace(item['after'],item['before'])
    digest=hashlib.sha256(ast.dump(ast.parse(source),include_attributes=False).encode()).hexdigest()
    assert digest==inverse['parent_ast_sha256']


# Bound during final source sealing, before any test execution.
INVERSE_SHA='f145a03af5faf5b3de9b8fbf9f77e7076cd70ff4a37ac1d67626028336218841'
