"""Actual raw collectors, normal executor and sender on controlled feedback.

These tests establish admission/lifecycle semantics, not physical settling gain.
"""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from erc_phase1_solution import place_transition_stop as stop
from erc_phase1_solution import additional_arm_timing as extra
from erc_phase1_solution import arm_velocity_admission as velocity
from erc_phase1_solution.placement_scene_context import measured_scene_context
from test_preopen_stationary import Node, raw, NAMES, GOAL, MASTER
import test_optional_arm_timing as actual

ROUTE=[(GOAL,.8,'bin_transition'),(tuple(v+.01 if i else v for i,v in enumerate(GOAL)),.8,'bin_transition')]
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def model(monkeypatch):
    n=Node();n.place_transition_stop_enabled=True
    n.loaded_place_speed_scale_cap=2.;n.additional_arm_time_scale=2.
    n.stock_gripper_close_diagnostic_enabled=True;n.delivery_evidence_enabled=True
    n._place_transition_stop_owner=None;n._place_transition_scene_failure=None
    n._pending_retained_acceptances=set();n._gravity_supported_payload=False
    n.joints=dict(zip(NAMES,(*GOAL,MASTER)));n._joint_stamps_ns=dict.fromkeys(NAMES,n.now)
    n._left_target_contact_ns=n._right_target_contact_ns=n.now
    n._gripper_feedback_samples=[NS(stamp_ns=n.now,position=MASTER,velocity=0.,effort=0.)]
    original_feed=n.feed
    def feed(sample=None):
        s=raw(n.now) if sample is None else sample
        original_feed(s)
        n.joints.update(s['positions']);n._joint_stamps_ns.update(dict.fromkeys(NAMES,s['producer_stamp_ns']))
        n._left_target_contact_ns=n._right_target_contact_ns=n.now
        n._gripper_feedback_samples=[NS(stamp_ns=n.now,position=n.joints[NAMES[-1]],
                                      velocity=s['velocities'][NAMES[-1]],effort=0.)]
    n.feed=feed
    clock=NS(monotonic=lambda:n.wall,sleep=n.sleep)
    monkeypatch.setattr(stop,'time',clock)
    monkeypatch.setattr(stop,'measured_scene_context',n.context)
    identity=dict(target_model=n._target_book_model,trial_id='trial',placement_attempt_id='attempt')
    token=stop.normal_options(n,identity,ROUTE,2.,MASTER)['transition_stop']
    return NS(n=n,token=token,clock=clock,identity=identity)


def publication(f, **changes):
    g=actual.goal();g.trajectory.joint_names=list(stop.ARM_JOINTS)
    p=actual.point();p.positions=list(ROUTE[1][0][1:]);p.time_from_start=actual.Duration(seconds=.2).to_msg()
    g.trajectory.points=[p]
    for key,value in changes.items():setattr(g.trajectory,key,value)
    with f.n._adaptive_command_guard():
        with f.n._lock:f.token.require_publication_locked(f.n,g,'place',1)


def test_distinct_joint_and_odom_span_uses_original_collectors(model):
    f=model;f.token.qualify()
    assert f.n.ticks>=5 and f.token.previous_joint-f.token.first_joint>=100_000_000
    assert f.token.previous_odom-f.token.first_odom>=100_000_000
    assert f.n._payload_monitor_enabled and not f.n._retention_probe_active
    assert f.n._delivery_measurement_active and f.n._place_transition_stop_owner is f.token
    assert len(f.n.events)==1 and f.n.events[0][1]['verified'] is True
    publication(f);assert f.token.published
    f.token.close();assert not f.n._delivery_measurement_active and f.n._place_transition_stop_owner is None


@pytest.mark.parametrize('kind',['joint_duplicate','odom_duplicate','stale','no_samples','moving_arm','moving_master'])
def test_insufficient_additional_feedback_never_publishes_and_expires_bounded(model,kind):
    f=model
    if kind=='no_samples':f.n.feeds=False
    def mutate(s):
        if kind=='joint_duplicate':s['producer_stamp_ns']=1_025_000_000
        elif kind=='odom_duplicate':s['odom']['stamp_ns']=1_025_000_000
        elif kind=='stale':s['producer_stamp_ns']=800_000_000
        elif kind=='moving_arm':s['velocities'][NAMES[2]]=.00101
        elif kind=='moving_master':s['velocities'][NAMES[-1]]=.000101
    f.n.mutate=mutate
    with pytest.raises(stop.TransitionStopRejected,match='deadline'):f.token.qualify()
    assert not f.token.published and not f.n._delivery_measurement_active
    assert f.n.now<=1_500_000_000 and f.n.wall<=3.01
    assert all(not fields['verified'] for _,fields in f.n.events)


@pytest.mark.parametrize('kind',['joint_reversed','odom_reversed','joint_nan','odom_nan','moving_base','moving_yaw','future'])
def test_invalid_or_original_negative_evidence_is_fatal_during_wait(model,kind):
    f=model
    def mutate(s):
        if f.n.ticks<2:return
        if kind=='joint_reversed':s['producer_stamp_ns']=1_010_000_000
        elif kind=='odom_reversed':s['odom']['stamp_ns']=1_010_000_000
        elif kind=='joint_nan':s['velocities'][NAMES[1]]=math.nan
        elif kind=='odom_nan':s['odom']['linear_speed']=math.nan
        elif kind=='moving_base':s['odom']['linear_speed']=.005001
        elif kind=='moving_yaw':s['odom']['angular_speed']=.008001
        elif kind=='future':s['producer_stamp_ns']=f.n.now+100_000_001
    f.n.mutate=mutate
    with pytest.raises(stop.TransitionStopRejected):f.token.qualify()
    assert f.n.ticks<=3 and not f.token.published and not f.n._delivery_measurement_active


def test_initial_arm_residual_can_settle_but_cannot_erase_an_original_fault(model):
    f=model
    f.n.mutate=lambda s:s['velocities'].__setitem__(NAMES[1],.002 if f.n.ticks<=2 else 0.)
    f.token.qualify()
    assert f.token.first_joint>=1_075_000_000 and f.n.ticks>=7
    assert f.n._payload_monitor_enabled
    f.token.close()


@pytest.mark.parametrize('fault',['cancel','payload','contact','pinch','scene','active','pending','identity','owner'])
def test_every_existing_fault_or_owner_change_blocks_qualification_without_dispatch(model,fault):
    f=model
    def tick():
        if f.n.ticks!=2:return
        if fault=='cancel':f.n._cancel.set()
        elif fault=='payload':f.n._payload_hazard_latched='payload_fault'
        elif fault=='contact':f.n.contact_fault='contact_lost'
        elif fault=='pinch':f.n.pinch=False
        elif fault=='scene':f.n.contact_fault='placement_scene_base_not_stationary'
        elif fault=='active':f.n._goal_handles.append(object())
        elif fault=='pending':f.n._pending_retained_acceptances.add(object())
        elif fault=='identity':f.n._contact_epoch+=1
        elif fault=='owner':f.n._place_transition_stop_owner=object()
    f.n.on_tick=tick
    with pytest.raises(stop.TransitionStopRejected):f.token.qualify()
    assert not f.token.published and not any(x[1]['verified'] for x in f.n.events)
    if fault!='owner':assert not f.n._delivery_measurement_active


@pytest.mark.parametrize('kind',['new_moving_joint','new_moving_odom','invalid_joint_stamp','invalid_odom_stamp','reversed_odom','stale','cancel','epoch','goal'])
def test_fresh_final_locked_publication_rechecks_newest_unpaired_feedback(model,kind):
    f=model;f.token.qualify()
    if kind=='new_moving_joint':
        s=raw(f.n.now);s['velocities'][NAMES[1]]=.002;f.n.feed(s)
    elif kind=='new_moving_odom':f.n._delivery_raw_odom['linear_speed']=.006
    elif kind in ('invalid_joint_stamp','invalid_odom_stamp'):
        f.n.feed(raw(f.n.now))
        if kind=='invalid_joint_stamp':f.n._delivery_raw_samples[-1]['producer_stamp_ns']=None
        else:f.n._delivery_raw_samples[-1]['odom']['stamp_ns']=None
    elif kind=='reversed_odom':f.n._delivery_raw_odom['stamp_ns']-=1
    elif kind=='stale':f.n.now+=150_000_001
    elif kind=='cancel':f.n._cancel.set()
    elif kind=='epoch':f.n._contact_epoch+=1
    with pytest.raises(stop.TransitionStopRejected) as caught:
        publication(f,**({'joint_names':list(reversed(stop.ARM_JOINTS))} if kind=='goal' else {}))
    assert caught.value is f.token.publication_rejection and not f.token.published
    f.token.close();assert not f.n._delivery_measurement_active


def test_stalled_clock_expires_on_wall_bound_without_dispatch(model):
    f=model;entered=f.n.now
    def wall_only_sleep(_):f.n.wall+=.1
    f.clock.sleep=wall_only_sleep
    with pytest.raises(stop.TransitionStopRejected,match='deadline'):f.token.qualify()
    assert f.n.now==entered and 3.0<=f.n.wall<=3.11
    assert not f.token.published and not f.n._delivery_measurement_active


def test_slow_simulation_allows_original_half_second_settling_window(model):
    f=model;n=f.n
    n.get_clock=lambda:NS(ros_time_is_active=True, now=lambda:NS(nanoseconds=n.now))
    n.timeout=60.
    # Residual arm motion settles after 0.3 sim seconds; the required new
    # 0.1-second stopped span finishes after the former 3-wall-second limit.
    n.mutate=lambda s:s['velocities'].__setitem__(NAMES[1], .002 if n.now<1_300_000_000 else 0.)
    def slow_sleep(_):
        n.wall+=.02;n.now+=2_600_000;n.ticks+=1;n.feed()
    f.clock.sleep=slow_sleep
    f.token.qualify()
    assert 3.<n.wall<4. and n.now<1_500_000_000
    assert f.token.previous_joint-f.token.first_joint>=100_000_000
    publication(f)
    f.token.close()


def test_slow_simulation_advancing_delayed_feedback_keeps_watchdog_alive(model):
    f=model;n=f.n;entered=n.now
    n.get_clock=lambda:NS(ros_time_is_active=True, now=lambda:NS(nanoseconds=n.now))
    n.timeout=60.
    # At RTF .02 a valid 100 ms producer lag takes five wall seconds to
    # cross the entry clock. Those samples show progress but cannot qualify.
    def slow_sleep(_):
        n.wall+=.1;n.now+=2_000_000;n.ticks+=1
        n.feed(raw(n.now-100_000_000))
    f.clock.sleep=slow_sleep
    f.token.qualify()
    assert n.wall>10. and n.now<entered+500_000_000
    assert f.token.first_joint>entered and f.token.first_odom>entered
    assert f.token.previous_joint-f.token.first_joint>=100_000_000
    assert f.token.previous_odom-f.token.first_odom>=100_000_000
    publication(f)
    f.token.close()


@pytest.mark.parametrize('frozen', ['joint', 'odom'])
def test_delayed_feedback_baseline_cannot_renew_without_both_producers(model, frozen):
    f=model;n=f.n;entered=n.now
    n.get_clock=lambda:NS(ros_time_is_active=True, now=lambda:NS(nanoseconds=n.now))
    n.timeout=60.
    def slow_sleep(_):
        n.wall+=.1;n.now+=2_000_000;n.ticks+=1
        sample=raw(n.now-100_000_000)
        if frozen=='joint':sample['producer_stamp_ns']=entered-100_000_000
        else:sample['odom']['stamp_ns']=entered-100_000_000
        n.feed(sample)
    f.clock.sleep=slow_sleep
    with pytest.raises(stop.TransitionStopRejected, match='deadline'):
        f.token.qualify()
    assert 3.<=n.wall<=3.11 and not f.token.published
    assert not n._delivery_measurement_active


@pytest.mark.parametrize('frozen', ['clock', 'feedback'])
def test_simulated_transition_still_refuses_missing_progress(model, frozen):
    f=model;n=f.n;entered=n.now
    n.get_clock=lambda:NS(ros_time_is_active=True, now=lambda:NS(nanoseconds=n.now))
    n.timeout=60.
    def sleep(_):
        n.wall+=.02
        if frozen!='clock':n.now+=200_000
    f.clock.sleep=sleep
    with pytest.raises(stop.TransitionStopRejected, match='deadline'):
        f.token.qualify()
    assert 3.<=n.wall<=3.03 and n.now<entered+500_000_000
    assert not f.token.published and not n._delivery_measurement_active


def install_actual_sender(f, *, in_flight_fault=False, send_error=None):
    n=f.n;n.timeout=1.;sent=[];cancelled=[];velocity_checks=[]
    n._faster_arm_velocity_limits=(1.95,1.95,3.95,3.95,3.95,3.95,3.95)
    n._faster_arm_velocity_urdf='controlled/official-named-limits'
    completed=[not in_flight_fault]
    result=NS(done=lambda:completed[0],result=lambda:NS(status=4))
    handle=NS(accepted=True,get_result_async=lambda:result)
    acceptance=NS(done=lambda:True,result=lambda:handle,add_done_callback=lambda callback:None)
    def send(g):
        assert n._lock.locked() and n.command_lock._is_owned()
        sent.append(copy.deepcopy(g))
        if send_error:raise send_error
        if in_flight_fault:n.contact_fault='placement_scene_base_not_stationary'
        return acceptance
    n.arm_client=NS(send_goal_async=send)
    def cancel(*args):cancelled.append(True);completed[0]=True;return True
    n._cancel_retained_goal_and_confirm=cancel
    n._valid_retained_terminal_result=lambda wrapped:wrapped.status==4
    n._trajectory_leg_at_elapsed_time=lambda legs,elapsed:(0,legs[0][2])
    n._cancel_late_retained_goal=lambda *args:None
    def check(node,goal):
        assert node._lock.locked() and node.command_lock._is_owned()
        velocity_checks.append(True)
        return velocity.require_arm_velocity_locked(node,goal)
    sender=actual.load('_send_retained_arm_trajectory',time=f.clock,_place_transition_stop=stop,
        require_arm_velocity_locked=check,retime_admitted_arm_goal=extra.retime_admitted_arm_goal,
        require_retimed_arm_headroom=extra.require_retimed_arm_headroom)
    n._send_retained_arm_trajectory=lambda *a,**kw:sender(n,*a,**kw)
    make=actual.load('_make_retained_arm_trajectory_goal',IK_JOINTS=stop.IK_JOINTS)
    n._make_retained_arm_trajectory_goal=lambda legs:make(n,legs)
    return sent,cancelled,velocity_checks


def test_actual_sender_uses_both_original_velocity_gates_and_consumes_one_token(model):
    f=model;f.token.qualify();sent,cancelled,checks=install_actual_sender(f)
    goal,total=f.n._make_retained_arm_trajectory_goal([(ROUTE[1][0],.4,'bin_transition')])
    original=copy.deepcopy(goal)
    assert f.n._send_retained_arm_trajectory(goal,.8,[(ROUTE[1][0],.4,'bin_transition')],
        'place',leg_offset=1,velocity_admission=True,transition_stop=f.token)==(True,False)
    assert len(sent)==1 and len(checks)==2 and not cancelled and f.token.published
    assert sent[0].trajectory.points[0].positions==original.trajectory.points[0].positions
    assert not f.n._goal_handles and not f.n._pending_retained_acceptances
    f.token.close()


def test_actual_sender_still_cancels_an_inflight_scene_fault(model):
    f=model;f.token.qualify();sent,cancelled,_=install_actual_sender(f,in_flight_fault=True)
    goal,_=f.n._make_retained_arm_trajectory_goal([(ROUTE[1][0],.4,'bin_transition')])
    assert f.n._send_retained_arm_trajectory(goal,.8,[(ROUTE[1][0],.4,'bin_transition')],
        'place',leg_offset=1,velocity_admission=True,transition_stop=f.token)==(False,True)
    assert len(sent)==len(cancelled)==1 and not f.n._goal_handles
    assert any(e=='grasp_lost' and d['reason']=='placement_scene_base_not_stationary' for e,d in f.n.events)
    f.token.close()


@pytest.mark.parametrize('fault',['cancel','new_velocity','client_error'])
def test_actual_sender_prepublication_refusal_or_unknown_acceptance_preserves_ownership(model,fault):
    f=model;f.token.qualify();sent,_,_=install_actual_sender(f,send_error=RuntimeError('client') if fault=='client_error' else None)
    if fault=='cancel':f.n._cancel.set()
    elif fault=='new_velocity':f.n._delivery_raw_odom['angular_speed']=.009
    goal,_=f.n._make_retained_arm_trajectory_goal([(ROUTE[1][0],.4,'bin_transition')])
    invoke=lambda:f.n._send_retained_arm_trajectory(goal,.8,[(ROUTE[1][0],.4,'bin_transition')],
        'place',leg_offset=1,velocity_admission=True,transition_stop=f.token)
    if fault=='cancel':assert invoke()==(False,False)
    elif fault=='new_velocity':
        with pytest.raises(stop.TransitionStopRejected):invoke()
    else:
        with pytest.raises(actual.RetainedMotionNotStopped):invoke()
    if fault=='client_error':assert len(sent)==1 and len(f.n._pending_retained_acceptances)==1 and f.n._cancel.is_set()
    else:assert not sent and not f.n._pending_retained_acceptances
    f.token.close()


@pytest.mark.parametrize('stop_on_second',[False,True])
def test_actual_executor_qualifies_only_after_success_and_retention_then_cleans_second_leg(model,stop_on_second):
    f=model;n=f.n;events=[];n._make_retained_arm_trajectory_goal=lambda legs:(object(),legs[0][1])
    def sender(goal,watchdog,legs,command,**kw):
        i=kw['leg_offset'];events.append(('send',i));assert watchdog==.8
        if i==0:assert 'transition_stop' not in kw and not f.token.active
        else:
            assert f.token.qualified and n._delivery_measurement_active and kw['transition_stop'] is f.token
            if stop_on_second:raise RuntimeError('controlled pre-send failure')
        return True,False
    n._send_retained_arm_trajectory=sender
    n._retention_after_leg=lambda *a:(events.append(('retention',a[-1])) or True)
    fn=actual.load('_execute_retained_arm_legs',time=f.clock)
    result=fn(n,ROUTE,'place',arm_speed_scale=2.,transition_stop=f.token)
    assert result==((False,1,False) if stop_on_second else (True,2,False))
    assert events==([('send',0),('retention',0),('send',1)] if stop_on_second else
                   [('send',0),('retention',0),('send',1),('retention',1)])
    assert not n._delivery_measurement_active and n._place_transition_stop_owner is None


@pytest.mark.parametrize('failure',['first_action','first_retention','qualification'])
def test_actual_executor_never_dispatches_second_leg_after_failed_prefix(model,failure):
    f=model;n=f.n;sent=[];n._make_retained_arm_trajectory_goal=lambda legs:(object(),legs[0][1])
    n._send_retained_arm_trajectory=lambda *a,**kw:(sent.append(kw['leg_offset']) or (failure!='first_action',False))
    n._retention_after_leg=lambda *a:failure!='first_retention'
    if failure=='qualification':n.contact_fault='placement_scene_base_not_stationary'
    fn=actual.load('_execute_retained_arm_legs',time=f.clock)
    if failure=='qualification':
        with pytest.raises(stop.TransitionStopRejected):fn(n,ROUTE,'place',arm_speed_scale=2.,transition_stop=f.token)
    else:assert not fn(n,ROUTE,'place',arm_speed_scale=2.,transition_stop=f.token)[0]
    assert sent==[0] and not getattr(n,'_delivery_measurement_active',False)


@pytest.mark.parametrize('value',[0,1,None,'true',[],{}])
def test_non_boolean_option_is_refused(value):
    with pytest.raises(ValueError):stop.checked_enabled(value)


def test_disabled_options_do_not_touch_any_unneeded_route_or_node_fields():
    assert stop.normal_options(NS(place_transition_stop_enabled=False),None,None,None,None)=={}


@pytest.mark.parametrize('kind',['cap','stock','raw','scene','phase','master'])
def test_enabled_option_rejects_unsupported_normal_place_inputs_before_motion(model,kind):
    f=model;route=copy.deepcopy(ROUTE);master=MASTER
    if kind=='cap':f.n.loaded_place_speed_scale_cap=1.5
    elif kind=='stock':f.n.stock_gripper_close_diagnostic_enabled=False
    elif kind=='raw':f.n.delivery_evidence_enabled=False
    elif kind=='scene':f.n.table_scene_required=False
    elif kind=='phase':route[1]=(route[1][0],.8,'recovery')
    else:master=math.nan
    with pytest.raises(stop.TransitionStopRejected):stop.normal_options(f.n,f.identity,route,2.,master)
    assert not getattr(f.n,'_delivery_measurement_active',False) and not f.n._goal_handles


@pytest.mark.parametrize('kind',['command','offset','speed','route','probe','clearance','withdrawal'])
def test_actual_executor_rejects_changed_scope_before_first_goal(model,kind):
    f=model;n=f.n;route=copy.deepcopy(ROUTE);command='place';kw=dict(arm_speed_scale=2.,transition_stop=f.token)
    if kind=='command':command='recover'
    elif kind=='offset':kw['leg_offset']=1
    elif kind=='speed':kw['arm_speed_scale']=1.5
    elif kind=='route':route[0]=(tuple(v+.001 for v in GOAL),.8,'bin_transition')
    elif kind=='probe':kw['fresh_retention_phases']=('bin_transition',)
    elif kind=='clearance':kw['clearance_timing']=object()
    else:kw['withdrawal_timing']=object()
    n._make_retained_arm_trajectory_goal=lambda legs:pytest.fail('scope refusal must precede construction')
    with pytest.raises(stop.TransitionStopRejected):actual.load('_execute_retained_arm_legs')(n,route,command,**kw)
    assert not getattr(n,'_delivery_measurement_active',False)


def test_default_executor_preserves_original_calls_without_qualification(model):
    n=model.n;calls=[];n.place_transition_stop_enabled=False
    n._make_retained_arm_trajectory_goal=lambda legs:(object(),legs[0][1])
    n._send_retained_arm_trajectory=lambda *a,**kw:(calls.append((a[1],kw)) or (True,False))
    n._retention_after_leg=lambda *a:True
    assert actual.load('_execute_retained_arm_legs')(n,ROUTE,'place')==(True,2,False)
    assert calls==[(.8,{'leg_offset':0}),(.8,{'leg_offset':1})]
    assert not getattr(n,'_delivery_measurement_active',False)


@pytest.mark.parametrize('kind',['linear','angular','pose','speed_nan'])
def test_actual_scene_veto_keeps_reason_and_records_exact_disjunct(model,kind):
    n=model.n
    n._staging_odom=dict(stamp_ns=n.now,producer_stamp_ns=n.now-1,pose=[0.,0.,0.],linear_speed=0.,angular_speed=0.)
    if kind=='linear':n._staging_odom['linear_speed']=.005001
    elif kind=='angular':n._staging_odom['angular_speed']=.008001
    elif kind=='pose':n._staging_odom['pose']=[0.,math.nan,0.]
    else:n._staging_odom['linear_speed']=math.nan
    with pytest.raises(RuntimeError,match='^placement_scene_base_not_stationary$'):measured_scene_context(n)
    fields=stop.diagnostic_fields(n,'placement_scene_base_not_stationary')
    detail=fields['scene_stationarity_diagnostic']
    assert detail['odom_producer_stamp_ns']==n.now-1 and detail['evaluated_ros_ns']==n.now
    assert detail['failed_predicates']==[{'linear':'linear_speed_above_0.005','angular':'angular_speed_above_0.008',
        'pose':'pose_nonfinite','speed_nan':'speed_nonfinite'}[kind]]
    assert not fields['raw_velocity_window']


def test_actual_constructor_default_is_false_and_owner_unset():
    defaults={}
    actual.load('_declare_parameters')(NS(declare_parameter=lambda name,value:defaults.__setitem__(name,value)))
    assert defaults['place_transition_stop_enabled'] is False
    owner=actual.node_tree();init=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    at=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='place_transition_stop_enabled' for t in n.targets))
    guards=[n for n in init.body if isinstance(n,ast.If) and ast.unparse(n.test)=="self.place_transition_stop_enabled and self.loaded_place_speed_scale_cap != 2.0"]
    assert len(guards)==1
    block=copy.deepcopy(init.body[at:at+3]+guards);n=NS(get_parameter=lambda name:NS(value=False),loaded_place_speed_scale_cap=3.)
    exec(compile(ast.fix_missing_locations(ast.Module(body=block,type_ignores=[])),'actual constructor slice','exec'),{'self':n,'_place_transition_stop':stop})
    assert n.place_transition_stop_enabled is False and n._place_transition_stop_owner is None
    assert n._place_transition_scene_failure is None
