"""Real capability/sender/executor boundaries; synthetic scene and action inputs.

The existing raw stop collector and geometry algorithms are unchanged. Boundary
fixtures explicitly complete a typed stop token instead of claiming a measured
stationary robot. These cases make no tracking or physical timing claim.
"""
import copy
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import release_only_clearance_timing as connector
from erc_phase1_solution import release_only_place_planning as release
from erc_phase1_solution import place_transition_stop as stop
from erc_phase1_solution import additional_arm_timing as extra
from erc_phase1_solution import arm_velocity_admission as gate
from erc_phase1_solution.motion_profiles import IK_JOINTS, ARM_JOINTS
from test_release_only_place_planning import request_node, IDENTITY, PAYLOAD
import test_optional_arm_timing as timing
import test_arm_velocity_admission as velocity


def route():
    return [(np.asarray([.30]+[x]*7), duration, phase) for x,duration,phase in
            [(.05,.8,'bin_transition'),(.10,.8,'bin_transition'),
             (.20,2.8,'bin_clearance'),(.25,.65,'bin_approach')]]


def prepared(monkeypatch, n=None):
    n=request_node(monkeypatch,n)
    n.release_only_clearance_timing_enabled=True
    n.place_transition_stop_enabled=True
    n.loaded_place_speed_scale_cap=2.;n.additional_arm_time_scale=2.
    n._release_only_clearance_owner=None;n._place_transition_stop_owner=None
    n._delivery_measurement_active=False;n._contact_epoch=10
    n._payload_monitor_enabled=True;n._gripper_open_confirmed=False
    n._retention_probe_active=False;n.grasp_contact_max_age=.15
    n._payload_hazard_reason=lambda **kw:None
    n.joints=dict(zip(IK_JOINTS,[.30]+[0.]*7))
    n._joint_stamps_ns=dict.fromkeys(IK_JOINTS,velocity.NOW)
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=velocity.NOW))
    n.statuses=[];n._publish_status=lambda event,**kw:n.statuses.append((event,kw))
    legs=route();request=release.capture(n,PAYLOAD)
    endpoint=request.endpoint(legs[-1][0],NS(sample=lambda *a:True))
    qualification=stop.Qualification(n,IDENTITY,connector._route(legs)[0],.017)
    plan=connector.normal_options(n,IDENTITY,endpoint,legs,qualification,2.)['release_clearance']
    return n,legs,qualification,plan


def execute_options(qualification):
    return dict(arm_speed_scale=2.,leg_offset=0,fresh_retention_phases=(),
                initial_pressure_gate=None,clearance_timing=None,withdrawal_timing=None,
                withdrawal_speed_scale=1.,transition_stop=qualification)


def complete_prefix(n,legs,qualification,plan):
    plan.require_execution(n,legs,'place',**execute_options(qualification))
    plan.retained(0)
    qualification.qualified=qualification.published=qualification.active=True
    n._place_transition_stop_owner=qualification;n._delivery_measurement_active=True
    qualification.close()
    plan.retained(1)


def admission(monkeypatch):
    n,legs,q,p=prepared(monkeypatch)
    complete_prefix(n,legs,q,p)
    return n,legs,q,p,p.admit(2,*legs[2])


def publish(n,token,goal=None):
    goal=velocity.goal([(token.plan.route[token.plan.index][0][1:],350_000_000)]) if goal is None else goal
    with n._adaptive_command_guard():
        with n._lock:token.require_publication_locked(n,goal,'place',2)


def endpoint_feedback(n,legs):
    n._wait_for_retained_endpoint=lambda target,**kw:np.asarray(target)
    n.joints.update(zip(IK_JOINTS,legs[2][0]))


@pytest.mark.parametrize('value',[None,0,1,'true',.0,{},[]])
def test_flag_rejects_non_boolean(value):
    with pytest.raises(ValueError):connector.checked_enabled(value)


@pytest.mark.parametrize('present',[False,True])
def test_default_absent_or_false_never_reads_context(present):
    n=NS(**({'release_only_clearance_timing_enabled':False} if present else {}))
    assert connector.normal_options(n,None,None,None,None,None)=={}


@pytest.mark.parametrize('fault',['endpoint','request','qualification','other_node','scale','bool_scale',
                                 'endpoint_goal','route','already_qualified'])
def test_selected_option_requires_original_typed_capabilities(monkeypatch,fault):
    n,legs,q,p=prepared(monkeypatch);ep=p.endpoint;scale=2.
    if fault=='endpoint':ep=NS(request=ep.request,goal=ep.goal)
    elif fault=='request':ep=release.Endpoint(NS(),ep.goal)
    elif fault=='qualification':q=NS(node=n,route=q.route)
    elif fault=='other_node':q.node=NS()
    elif fault=='scale':scale=1.5
    elif fault=='bool_scale':scale=True
    elif fault=='endpoint_goal':ep=release.Endpoint(ep.request,tuple([.0]*8))
    elif fault=='route':legs[1]=(legs[1][0]+.01,.8,'bin_transition')
    else:q.qualified=True
    with pytest.raises(gate.ArmVelocityAdmissionRejected):
        connector.normal_options(n,IDENTITY,ep,legs,q,scale)
    assert n._release_only_clearance_owner is None


@pytest.mark.parametrize('fault',['duration','bool_duration','nan','short_q','one_transition',
                                 'two_connectors','no_approach','wrong_phase'])
def test_route_restriction_retains_exact_original_segment_structure(fault):
    legs=route()
    if fault=='duration':legs[2]=(legs[2][0],math.nextafter(2.8,0.),'bin_clearance')
    elif fault=='bool_duration':legs[2]=(legs[2][0],True,'bin_clearance')
    elif fault=='nan':legs[2][0][2]=math.nan
    elif fault=='short_q':legs[2]=(legs[2][0][1:],2.8,'bin_clearance')
    elif fault=='one_transition':legs.pop(0)
    elif fault=='two_connectors':legs.insert(2,legs[2])
    elif fault=='no_approach':legs.pop()
    else:legs[1]=(legs[1][0],.8,'withdrawal')
    with pytest.raises(gate.ArmVelocityAdmissionRejected):connector._route(legs)


@pytest.mark.parametrize('fault',['cancel','active','pending','old_clearance','release_owner','serial_owner',
    'retention_owner','monitor','open','cap','additional','stop_flag','new_flag',
    'payload_fault','raw_fault','scene_identity','scene_value','guard_fault'])
def test_context_invalidation_before_connector_cannot_create_owner(monkeypatch,fault):
    n,legs,q,p=prepared(monkeypatch);complete_prefix(n,legs,q,p)
    if fault=='cancel':n._cancel.set()
    elif fault=='active':n._goal_handles.append(object())
    elif fault=='pending':n._pending_retained_acceptances.append(object())
    elif fault=='old_clearance':n.bin_clearance_timing_enabled=True
    elif fault=='release_owner':n._release_pose_owner=object()
    elif fault=='serial_owner':n._initial_stow_serial_owner=object()
    elif fault=='retention_owner':n._retention_probe_active=True
    elif fault=='monitor':n._payload_monitor_enabled=False
    elif fault=='open':n._gripper_open_confirmed=True
    elif fault=='cap':n.loaded_place_speed_scale_cap=1.5
    elif fault=='additional':n.additional_arm_time_scale=1.
    elif fault=='stop_flag':n.place_transition_stop_enabled=False
    elif fault=='new_flag':n.release_only_clearance_timing_enabled=False
    elif fault=='payload_fault':n._payload_hazard_latched='held fault'
    elif fault=='raw_fault':n._raw_contacts_first_failure='wire fault'
    elif fault=='scene_identity':n._active_place_scene_reference=copy.deepcopy(n._active_place_scene_reference)
    elif fault=='scene_value':n._active_place_scene_reference['base_pose'][0]=.01
    else:n._place_contact_guard.fault={'reason':'contact'}
    with pytest.raises((gate.ArmVelocityAdmissionRejected,RuntimeError)):p.admit(2,*legs[2])
    assert n._release_only_clearance_owner is None and p.admission is None


@pytest.mark.parametrize('fault',['unqualified','unpublished','active','owner','measurement','epoch','reference'])
def test_completed_transition_must_remain_closed_before_connector(monkeypatch,fault):
    n,legs,q,p=prepared(monkeypatch);complete_prefix(n,legs,q,p)
    if fault=='unqualified':q.qualified=False
    elif fault=='unpublished':q.published=False
    elif fault=='active':q.active=True
    elif fault=='owner':n._place_transition_stop_owner=q
    elif fault=='measurement':n._delivery_measurement_active=True
    elif fault=='epoch':n._contact_epoch+=1
    else:q.reference=copy.deepcopy(q.reference)
    with pytest.raises(gate.ArmVelocityAdmissionRejected):p.admit(2,*legs[2])
    assert n._release_only_clearance_owner is None


def test_admission_requires_every_prefix_retention_and_is_single_use(monkeypatch):
    n,legs,q,p=prepared(monkeypatch)
    with pytest.raises(gate.ArmVelocityAdmissionRejected):p.admit(2,*legs[2])
    complete_prefix(n,legs,q,p)
    with pytest.raises(gate.ArmVelocityAdmissionRejected):p.retained(1)
    token=p.admit(2,*legs[2]);assert n._release_only_clearance_owner is token
    with pytest.raises(gate.ArmVelocityAdmissionRejected):p.admit(2,*legs[2])
    with pytest.raises(gate.ArmVelocityAdmissionRejected):p.require_execution(n,legs,'place',**execute_options(q))
    token.close();assert n._release_only_clearance_owner is None


@pytest.mark.parametrize('key,value',[('arm_speed_scale',1.5),('leg_offset',1),
    ('fresh_retention_phases',('bin_clearance',)),('initial_pressure_gate','owned'),
    ('clearance_timing','old'),('withdrawal_timing','owned'),('withdrawal_speed_scale',2.)])
def test_execution_options_cannot_overlap_other_timing_owners(monkeypatch,key,value):
    n,legs,q,p=prepared(monkeypatch);options=execute_options(q);options[key]=value
    with pytest.raises(gate.ArmVelocityAdmissionRejected):p.require_execution(n,legs,'place',**options)
    assert p.started is False and n._release_only_clearance_owner is None


@pytest.mark.parametrize('ns,accepted',[(349_999_999,False),(350_000_000,True),
    (350_000_001,True),(700_000_000,True),(700_000_001,False)])
def test_serialized_final_time_exact_boundaries(monkeypatch,ns,accepted):
    n,legs,q,p,t=admission(monkeypatch);g=velocity.goal([(legs[2][0][1:],ns)])
    if accepted:publish(n,t,g);assert t.published
    else:
        with pytest.raises(gate.ArmVelocityAdmissionRejected):publish(n,t,g)
        assert not t.published
    t.close()


@pytest.mark.parametrize('fault',['owner','closed','repeated','target','joint_order','point_count',
                                 'bool_sec','bad_nsec'])
def test_publication_scope_rejects_before_registration(monkeypatch,fault):
    n,legs,q,p,t=admission(monkeypatch);g=velocity.goal([(legs[2][0][1:],350_000_000)])
    if fault=='owner':n._release_only_clearance_owner=object()
    elif fault=='closed':t.close()
    elif fault=='repeated':publish(n,t,g)
    elif fault=='target':g.trajectory.points[0].positions[0]+=.001
    elif fault=='joint_order':g.trajectory.joint_names.reverse()
    elif fault=='point_count':g.trajectory.points.append(copy.deepcopy(g.trajectory.points[0]))
    elif fault=='bool_sec':g.trajectory.points[0].time_from_start.sec=False
    else:g.trajectory.points[0].time_from_start.nanosec=1_000_000_000
    with pytest.raises(gate.ArmVelocityAdmissionRejected):publish(n,t,g)
    foreign=n._release_only_clearance_owner
    t.close()
    assert n._release_only_clearance_owner is (foreign if fault=='owner' else None)
    assert not n._goal_handles and not n._pending_retained_acceptances


def sender_fixture(monkeypatch,**kw):
    n,events,sent,records,acceptance,clock=velocity.sender_node(retained=True,**kw)
    sensor,command_guard,publisher=n._lock,n._adaptive_command_guard,n._publish_status
    n,legs,q,p=prepared(monkeypatch,n)
    n._lock=sensor;n._adaptive_command_guard=command_guard;n._publish_status=publisher
    n._pending_retained_acceptances=set()
    complete_prefix(n,legs,q,p);t=p.admit(2,*legs[2])
    return n,legs,t,events,sent,records,acceptance,clock


def send_actual(n,legs,t,clock,check=None):
    fn=timing.load('_send_retained_arm_trajectory',time=clock,_place_transition_stop=stop,
        require_arm_velocity_locked=check or gate.require_arm_velocity_locked,
        retime_admitted_arm_goal=extra.retime_admitted_arm_goal,
        require_retimed_arm_headroom=extra.require_retimed_arm_headroom)
    return fn(n,velocity.goal([(legs[2][0][1:],700_000_000)]),2.8,
              [(legs[2][0],.7,'bin_clearance')],'place',leg_offset=2,
              velocity_admission=True,velocity_headroom=True,release_clearance_admission=t)


def test_actual_sender_has_two_locked_velocity_gates_and_original_watchdog(monkeypatch):
    n,legs,t,events,sent,records,_,clock=sender_fixture(monkeypatch)
    checks=[]
    def checked(node,goal):
        assert n._lock.depth and n.command.depth
        checks.append(timing.seconds(goal.trajectory.points[0].time_from_start))
        return gate.require_arm_velocity_locked(node,goal)
    try:assert send_actual(n,legs,t,clock,checked)==(True,False)
    finally:t.close()
    assert checks==[.7,.35] and len(sent)==len(records)==1
    assert timing.seconds(sent[0].trajectory.points[0].time_from_start)==.35
    assert sent[0].trajectory.points[0].positions==list(legs[2][0][1:])
    assert records[0]['additional_arm_timing']['nominal_duration_seconds']==2.8
    assert records[0]['maximum_commanded_velocity_limit_ratio']<=.8
    assert not n._goal_handles and not n._pending_retained_acceptances
    assert t.closed and t.published and n._release_only_clearance_owner is None


def test_actual_sender_extends_retiming_for_fresh_eighty_percent_headroom(monkeypatch):
    n,legs,t,events,sent,records,_,clock=sender_fixture(monkeypatch)
    # Connector target remains .20. A fresh start at -.40 requires >.35s under
    # the original 1.95rad/s limit, so the existing checked retimer must extend.
    n.joints[ARM_JOINTS[0]]=-.4
    try:assert send_actual(n,legs,t,clock)==(True,False)
    finally:t.close()
    actual=sent[0].trajectory.points[0].time_from_start
    ns=actual.sec*1_000_000_000+actual.nanosec
    assert 350_000_000<ns<=700_000_000
    assert ns==records[0]['additional_arm_timing']['segments'][0]['velocity_minimum_ns']
    assert records[0]['maximum_commanded_velocity_limit_ratio']<=.8
    assert records[0]['additional_arm_timing']['nominal_duration_seconds']==2.8


@pytest.mark.parametrize('stage',[1,2])
@pytest.mark.parametrize('fault',['stale','slope','cancel','request'])
def test_locked_rejection_never_registers_unknown_action(monkeypatch,stage,fault):
    n,legs,t,events,sent,records,_,clock=sender_fixture(monkeypatch);calls=[]
    def checked(node,goal):
        calls.append(1)
        if len(calls)==stage:
            if fault=='stale':n._joint_stamps_ns[ARM_JOINTS[0]]=velocity.NOW-150_000_001
            elif fault=='slope':n.joints[ARM_JOINTS[0]]=-10.
            elif fault=='cancel':n._cancel.set()
            else:n._selected_place_bin_scene=copy.deepcopy(n._selected_place_bin_scene)
        return gate.require_arm_velocity_locked(node,goal)
    try:
        with pytest.raises(gate.ArmVelocityAdmissionRejected):send_actual(n,legs,t,clock,checked)
    finally:t.close()
    assert not sent and not n._pending_retained_acceptances and not n._goal_handles
    assert len(records)==1 and records[0]['admitted'] is False
    assert not t.published and n._release_only_clearance_owner is None


@pytest.mark.parametrize('kind',['client_exception','late_acceptance'])
def test_connector_close_keeps_original_uncertain_action_ownership(monkeypatch,kind):
    kw={'send_error':RuntimeError('wire')} if kind=='client_exception' else {'late':True}
    n,legs,t,events,sent,records,acceptance,clock=sender_fixture(monkeypatch,**kw)
    try:
        with pytest.raises(timing.RetainedMotionNotStopped):send_actual(n,legs,t,clock)
    finally:t.close()
    assert t.published and t.closed and n._release_only_clearance_owner is None
    assert n._cancel.is_set() and len(n._pending_retained_acceptances)==1
    assert len(records)==1 and records[0]['admitted'] is True
    if kind=='late_acceptance':
        assert len(acceptance.callbacks)==1
        acceptance.callbacks[0](acceptance)
        assert ('late_cancel',) in events


def executor_fixture(monkeypatch,failure=None):
    n,calls,probes=timing.retained_node();n,legs,q,p=prepared(monkeypatch,n)
    endpoint_feedback(n,legs);events=[]
    def qualify():
        events.append(('qualified',));q.qualified=q.active=True
        n._place_transition_stop_owner=q;n._delivery_measurement_active=True
    q.qualify=qualify
    def send(goal,watchdog,active,command,**kw):
        index=kw['leg_offset'];events.append(('send',index))
        calls.append((goal,watchdog,active,command,kw))
        if index==1:
            assert kw['transition_stop'] is q and n._release_only_clearance_owner is None
            q.published=True
        if index==2:
            token=kw['release_clearance_admission']
            assert not q.active and n._place_transition_stop_owner is None
            assert 'transition_stop' not in kw and kw['velocity_headroom'] is True
            if failure=='sender_exception':raise RuntimeError('sender fixture')
            token.published=True
        return (False,False) if failure==('motion',index) else (True,False)
    def retained(command,phase,index):
        events.append(('retained',index));probes.append(index)
        return failure!=('retention',index)
    n._send_retained_arm_trajectory=send;n._retention_after_leg=retained
    return n,legs,q,p,calls,probes,events


def execute(n,legs,q,p):
    return timing.load('_execute_retained_arm_legs')(n,legs,'place',arm_speed_scale=2.,
                                                   transition_stop=q,release_clearance=p)


def test_actual_executor_only_shortens_connector_and_keeps_all_retention(monkeypatch):
    n,legs,q,p,calls,probes,events=executor_fixture(monkeypatch)
    assert execute(n,legs,q,p)==(True,4,False)
    assert [c[1] for c in calls]==[.8,.8,2.8,.65]
    assert [timing.seconds(c[0].trajectory.points[0].time_from_start) for c in calls]==[.4,.4,.7,.325]
    assert all(c[2][0][0] is leg[0] for c,leg in zip(calls,legs))
    assert probes==[0,1,2,3] and p.completed_legs==4
    assert events.index(('retained',1))<events.index(('send',2))
    assert n._release_only_clearance_owner is None and n._place_transition_stop_owner is None
    assert [x[0] for x in n.statuses].count('release_only_clearance_endpoint_verified')==1


@pytest.mark.parametrize('failure',[('motion',0),('motion',1),('retention',0),('retention',1)])
def test_failed_prefix_never_admits_connector(monkeypatch,failure):
    n,legs,q,p,calls,probes,events=executor_fixture(monkeypatch,failure)
    result=execute(n,legs,q,p)
    assert result[0] is False and len(calls)<=2 and p.admission is None
    assert n._release_only_clearance_owner is None and n._place_transition_stop_owner is None


def test_actual_executor_closes_connector_on_sender_exception(monkeypatch):
    n,legs,q,p,calls,probes,events=executor_fixture(monkeypatch,'sender_exception')
    assert execute(n,legs,q,p)==(False,2,False)
    assert len(calls)==3 and probes==[0,1]
    assert p.admission.closed and n._release_only_clearance_owner is None


def test_actual_executor_closes_connector_without_forgetting_accepted_handle(monkeypatch):
    n,legs,q,p,calls,probes,events=executor_fixture(monkeypatch)
    original=n._send_retained_arm_trajectory;handle=object()
    def send(*args,**kw):
        if kw['leg_offset']==2:
            n._goal_handles.append(handle)
            raise RuntimeError('accepted action result failed')
        return original(*args,**kw)
    n._send_retained_arm_trajectory=send
    with pytest.raises(timing.RetainedMotionNotStopped):execute(n,legs,q,p)
    assert n._goal_handles==[handle] and n._cancel.is_set()
    assert p.admission.closed and n._release_only_clearance_owner is None
    assert probes==[0,1] and p.completed_legs==2


@pytest.mark.parametrize('fault',['miss','stale','future','residual','cancel','hazard','scene'])
def test_endpoint_refusal_raises_before_route_advance_or_any_next_dispatch(monkeypatch,fault):
    n,legs,q,p,calls,probes,events=executor_fixture(monkeypatch)
    def wait(target,**kw):
        assert kw==dict(command='place',phase='bin_clearance',leg=2)
        if fault=='miss':return None
        if fault=='stale':n._joint_stamps_ns[IK_JOINTS[2]]=velocity.NOW-150_000_001
        elif fault=='future':n._joint_stamps_ns[IK_JOINTS[2]]=velocity.NOW+50_000_001
        elif fault=='residual':n.joints[IK_JOINTS[2]]+=.0051
        elif fault=='cancel':n._cancel.set()
        elif fault=='hazard':n._payload_hazard_reason=lambda **kw:'contact_lost'
        elif fault=='scene':n._active_place_scene_reference['base_pose'][0]=.01
        return np.asarray(target)
    n._wait_for_retained_endpoint=wait
    with pytest.raises(RuntimeError):execute(n,legs,q,p)
    assert len(calls)==3 and probes==[0,1] and p.completed_legs==2
    assert p.admission.closed and n._release_only_clearance_owner is None
    assert not any(event=='release_only_clearance_endpoint_verified' for event,_ in n.statuses)


def test_default_executor_does_not_add_connector_kwargs_or_endpoint_gate():
    n,calls,probes=timing.retained_node();legs=route()
    n._wait_for_retained_endpoint=lambda *a,**kw:pytest.fail('default endpoint changed')
    assert timing.load('_execute_retained_arm_legs')(n,legs,'place',arm_speed_scale=2.)==(True,4,False)
    assert [timing.seconds(c[0].trajectory.points[0].time_from_start) for c in calls]==[.4,.4,1.4,.325]
    assert all(c[4]=={'leg_offset':i,'velocity_admission':True} for i,c in enumerate(calls))
    assert len(probes)==4


# Exercise the real selected _place handoff that the earlier helper/executor
# fixtures bypassed. Geometry and action completion below remain synthetic;
# option construction, capabilities, executor and velocity retiming are real.
import ast
import hashlib
from pathlib import Path
import yaml


FAILED95_CALLSITE = """        if getattr(self, 'release_only_clearance_timing_enabled', False):
            arm_timing_options.update(_release_clearance.normal_options(
                self, correlation, release_only_endpoint, approach_legs,
                arm_timing_options.get('transition_stop'), arm_speed_scale))
"""
FIXED68_CALLSITE = """        if getattr(self, 'release_only_clearance_timing_enabled', False):
            loaded_arm_timing_options.update(_release_clearance.normal_options(
                self, correlation, release_only_endpoint, approach_legs,
                loaded_arm_timing_options.get('transition_stop'),
                loaded_arm_timing_options.get('arm_speed_scale', 1.0)))
"""


def selected_place_call_source(*, failed95=False):
    path=timing.SOURCE/'erc_phase1_solution/manipulation_node.py'
    source=path.read_text()
    if failed95:
        assert source.count(FIXED68_CALLSITE)==1
        source=source.replace(FIXED68_CALLSITE,FAILED95_CALLSITE,1)
        assert hashlib.sha256(source.encode()).hexdigest()=='c9de32ba5dfcf687d47ede00e8e51febb018c2699602f8f8a84b56cddef2da8c'
    return source


def run_actual_place_options(scope, *, failed95=False):
    source=selected_place_call_source(failed95=failed95)
    owner=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    fn=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='_place')
    first=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='arm_timing_options' for t in n.targets))
    last=next(i for i,n in enumerate(fn.body) if i>first and isinstance(n,ast.Assign)
              and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
              and n.value.func.attr=='_execute_retained_arm_legs')
    assert last>first
    # The entire contiguous block includes the unmodified loaded cap, real
    # stop creation, connector creation and actual **loaded forwarding call.
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
          *copy.deepcopy(fn.body[first:last+1])]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),
                 '<actual selected PLACE options>','exec'),scope)
    return scope


def selected_place_call_fixture(monkeypatch, *, failure=None):
    n,calls,probes=timing.retained_node();n,legs,unused_stop,unused_plan=prepared(monkeypatch,n)
    selected=yaml.safe_load((timing.SOURCE/'config/collision_quality.yaml').read_bytes())['erc_manipulation']['ros__parameters']
    names=('placement_transport_speed_scale','loaded_place_speed_scale_cap','additional_arm_time_scale',
           'stock_gripper_close_diagnostic_enabled','place_transition_stop_enabled',
           'release_only_clearance_timing_enabled')
    for name in names:setattr(n,name,selected[name])
    assert (n.placement_transport_speed_scale,n.loaded_place_speed_scale_cap,n.additional_arm_time_scale)==(3.,2.,2.)
    n._faster_arm_velocity_limits=(1.95,1.95,3.95,3.95,3.95,3.95,3.95)
    n._faster_arm_velocity_urdf='synthetic named official limits'
    n._wait_for_retained_endpoint=lambda target,**kw:np.asarray(target)
    observed=[];serialized=[];retimed=[]
    executor=timing.load('_execute_retained_arm_legs')
    def execute(actual_legs,command,**options):
        observed.append((actual_legs,command,options))
        q=options['transition_stop'];p=options.get('release_clearance')
        assert type(q) is stop.Qualification and q is not unused_stop
        if n.release_only_clearance_timing_enabled:
            assert type(p) is connector.Plan and p is not unused_plan and p.transition is q
        def qualify():
            # Existing70 stop tests cover sensor collection. Here only the
            # action/stop completion is synthetic so this test reaches the
            # actual caller-to-executor-to-connector wiring.
            q.qualified=q.active=True;n._place_transition_stop_owner=q;n._delivery_measurement_active=True
        q.qualify=qualify
        def send(goal,watchdog,active,command,**kw):
            index=kw['leg_offset'];calls.append((goal,watchdog,active,command,kw))
            if index==1:
                assert kw['transition_stop'] is q
                q.published=True
            with n._adaptive_command_guard():
                with n._lock:
                    admission=gate.require_arm_velocity_locked(n,goal)
                    actual,actual_legs,detail=extra.retime_admitted_arm_goal(goal,admission,
                        n.additional_arm_time_scale,nominal_duration=watchdog,legs=active)
                    extra.require_retimed_arm_headroom(gate.require_arm_velocity_locked(n,actual))
                    if index==2 and p is not None:
                        assert kw['release_clearance_admission'] is p.admission
                        assert not q.active and 'transition_stop' not in kw
                        p.admission.require_publication_locked(n,actual,command,index)
            serialized.append(actual);retimed.append(detail)
            n.joints.update(zip(IK_JOINTS,active[-1][0]))
            return (False,False) if failure==('motion',index) else (True,False)
        n._send_retained_arm_trajectory=send
        n._retention_after_leg=lambda command,phase,index:(probes.append(index) or failure!=('retention',index))
        return executor(n,actual_legs,command,**options)
    n._execute_retained_arm_legs=execute
    scope=dict(self=n,correlation=IDENTITY,direct_empty_home=None,release_only_endpoint=unused_plan.endpoint,
        carried_transition_waypoints=[leg[0] for leg in legs[:2]],solutions=[leg[0] for leg in legs[2:]],
        place_scene_diagnostics=dict(measured_master=.017),checked_arm_speed_scale=timing.timing.checked_arm_speed_scale,
        _place_transition_stop=stop,_release_clearance=connector,
        bin_clearance_normal_options=lambda *a:pytest.fail('old clearance mode activated'))
    return NS(n=n,legs=legs,scope=scope,observed=observed,calls=calls,probes=probes,serialized=serialized,retimed=retimed)


def test_actual_selected_place_passes_capped_stop_and_plan_through_real_executor(monkeypatch):
    f=selected_place_call_fixture(monkeypatch);run_actual_place_options(f.scope)
    assert (f.scope['approach_ok'],f.scope['next_leg'],f.scope['contact_lost'])==(True,4,False)
    assert f.scope['arm_timing_options']=={'arm_speed_scale':3.}
    loaded=f.scope['loaded_arm_timing_options'];p=loaded['release_clearance'];q=loaded['transition_stop']
    assert loaded['arm_speed_scale']==2. and p.transition is q
    assert f.observed[0][2]==loaded and p.started and p.completed_legs==4
    assert [c[1] for c in f.calls]==[.8,.8,2.8,.65]
    assert [timing.seconds(c[0].trajectory.points[0].time_from_start) for c in f.calls]==[.4,.4,.7,.325]
    assert [timing.seconds(g.trajectory.points[0].time_from_start) for g in f.serialized]==[.2,.2,.35,.1625]
    assert f.retimed[2]['nominal_duration_seconds']==2.8 and f.probes==[0,1,2,3]
    assert p.admission.closed and p.admission.published and not q.active
    assert f.n._release_only_clearance_owner is f.n._place_transition_stop_owner is None


def test_exact_failed95_callsite_is_rejected_before_executor_or_any_arm_send(monkeypatch):
    f=selected_place_call_fixture(monkeypatch)
    with pytest.raises(gate.ArmVelocityAdmissionRejected,match='typed release-only endpoint and cap2 stop required'):
        run_actual_place_options(f.scope,failed95=True)
    assert f.scope['arm_speed_scale']==3. and f.scope['arm_timing_options'].get('transition_stop') is None
    assert f.scope['loaded_arm_timing_options']['arm_speed_scale']==2.
    assert type(f.scope['loaded_arm_timing_options']['transition_stop']) is stop.Qualification
    assert f.observed==f.calls==f.probes==[]
    assert f.n._release_only_clearance_owner is f.n._place_transition_stop_owner is None


@pytest.mark.parametrize('missing',[False,True])
def test_actual_disabled_place_callsite_preserves_prior_loaded_options(monkeypatch,missing):
    f=selected_place_call_fixture(monkeypatch)
    if missing:del f.n.release_only_clearance_timing_enabled
    else:f.n.release_only_clearance_timing_enabled=False
    def capture(legs,command,**options):f.observed.append((legs,command,options));return True,len(legs),False
    f.n._execute_retained_arm_legs=capture
    run_actual_place_options(f.scope)
    assert set(f.scope['loaded_arm_timing_options'])=={'arm_speed_scale','transition_stop'}
    assert f.scope['loaded_arm_timing_options']['arm_speed_scale']==2.
    assert f.scope['arm_timing_options']=={'arm_speed_scale':3.}
    assert f.observed[0][2]==f.scope['loaded_arm_timing_options'] and not f.calls


@pytest.mark.parametrize('fault',['cancel','scene','cap','stop_disabled','raw_fault'])
def test_actual_selected_callsite_preserves_live_admission_refusals(monkeypatch,fault):
    f=selected_place_call_fixture(monkeypatch)
    if fault=='cancel':f.n._cancel.set()
    elif fault=='scene':f.n._active_place_scene_reference=copy.deepcopy(f.n._active_place_scene_reference)
    elif fault=='cap':f.n.loaded_place_speed_scale_cap=1.5
    elif fault=='stop_disabled':f.n.place_transition_stop_enabled=False
    else:f.n._raw_contacts_first_failure='wire fault'
    with pytest.raises((RuntimeError,gate.ArmVelocityAdmissionRejected)):
        run_actual_place_options(f.scope)
    assert f.observed==f.calls==f.probes==[]
    assert f.n._release_only_clearance_owner is f.n._place_transition_stop_owner is None


@pytest.mark.parametrize('failure',[('motion',1),('retention',1)])
def test_actual_selected_callsite_prefix_failure_never_reaches_connector(monkeypatch,failure):
    f=selected_place_call_fixture(monkeypatch,failure=failure);run_actual_place_options(f.scope)
    p=f.scope['loaded_arm_timing_options']['release_clearance']
    assert f.scope['approach_ok'] is False and len(f.calls)==2
    assert p.admission is None and p.completed_legs==1
    assert f.n._release_only_clearance_owner is f.n._place_transition_stop_owner is None
