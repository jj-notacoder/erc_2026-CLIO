"""Actual sender/executor and integer retimer; controlled feedback, no physics."""
import copy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from erc_phase1_solution import manipulation_node as manipulation
from erc_phase1_solution import place_transition_stop as stop
from erc_phase1_solution import additional_arm_timing as extra
from erc_phase1_solution import arm_velocity_admission as velocity
from erc_phase1_solution.scene_checked_place import SceneCheckedPlacePlan
from test_place_transition_stop import model, install_actual_sender
from test_preopen_stationary import NAMES, GOAL, MASTER
from test_additional_arm_timing import make, admit


def prepared(f,monkeypatch,positive=True):
    n=f.n
    start=np.array(GOAL);start[-1]=.7 if positive else -.7
    n.joints.update(zip(stop.IK_JOINTS,start))
    q0=start.copy();q0[1]+=.12
    q1=q0.copy();q1[2]+=.01
    q2=q1.copy();q2[3]+=.01
    q3=q2.copy();q3[4]+=.01
    plan=SceneCheckedPlacePlan([q2,q3],0,0.,[q0,q1,q2],[],dict(
        cartesian_search={'wrist_policy':'measured_positive'},registered_bin_scene={'valid':True},
        whole_table_pose_modeled=True,actual_carry_start=start.tolist(),torso_ready=start.tolist()))
    route=[(q0,.8,'bin_transition'),(q1,.8,'bin_transition'),(q2,2.8,'bin_clearance'),(q3,.65,'bin_approach')]
    token=stop.normal_options(n,f.identity,route,2.,MASTER,
        **({'positive_entry_plan':plan}if positive else {}))['transition_stop']
    sent,cancelled,_=install_actual_sender(f)
    send=n.arm_client.send_goal_async
    def observe_send(goal):
        result=send(goal)
        n.joints.update(zip(stop.ARM_JOINTS,goal.trajectory.points[-1].positions))
        return result
    n.arm_client.send_goal_async=observe_send
    def keep_feedback(sample):sample['positions'].update({name:n.joints[name]for name in stop.IK_JOINTS})
    n.mutate=keep_feedback
    checks=[]
    original=velocity.require_arm_velocity_locked
    def checked(node,goal):
        assert node._lock.locked() and node.command_lock._is_owned()
        record=original(node,goal);checks.append(copy.deepcopy(record));return record
    monkeypatch.setattr(manipulation,'require_arm_velocity_locked',checked)
    n._send_retained_arm_trajectory=lambda *a,**kw:manipulation.ManipulationNode._send_retained_arm_trajectory(n,*a,**kw)
    n._make_retained_arm_trajectory_goal=lambda legs:manipulation.ManipulationNode._make_retained_arm_trajectory_goal(n,legs)
    n._retention_after_leg=lambda *args:True
    return NS(n=n,plan=plan,route=route,token=token,sent=sent,cancelled=cancelled,checks=checks)


def ns(goal):
    t=goal.trajectory.points[0].time_from_start
    return t.sec*1_000_000_000+t.nanosec


@pytest.mark.parametrize('positive',[False,True])
def test_actual_executor_changes_only_bound_positive_first_segment(model,monkeypatch,positive):
    f=prepared(model,monkeypatch,positive)
    result=manipulation.ManipulationNode._execute_retained_arm_legs(
        f.n,f.route,'place',arm_speed_scale=2.,transition_stop=f.token)
    assert result==(True,4,False)
    assert [ns(goal)for goal in f.sent]==[800_000_000 if positive else 200_000_000,
        200_000_000,700_000_000,162_500_000]
    assert len(f.checks)==8 and not f.cancelled
    for i,((q,_,_),goal)in enumerate(zip(f.route,f.sent)):
        np.testing.assert_array_equal(goal.trajectory.points[0].positions,q[1:])
        velocity.check_serialized_arm_goal(goal,f.checks[2*i]['start_positions'],f.checks[2*i]['velocity_limits'])
        extra.require_retimed_arm_headroom(f.checks[2*i+1])
    assert f.token.qualified and f.token.published and not f.token.active
    assert f.token.entry_published is positive
    reports=[v for e,v in f.n.events if e=='placement_transition_stop']
    assert reports[-1]['positive_entry_original_duration_restored'] is positive
    assert reports[-1]['elapsed_ros_ns']==125_000_000
    assert not f.n._delivery_measurement_active and not f.n._goal_handles


@pytest.mark.parametrize('fault',['phase','goal','plan','identity','negative_feedback','stale_feedback','watchdog'])
def test_bound_entry_refusal_never_dispatches_or_authorizes_recovery(model,monkeypatch,fault):
    f=prepared(model,monkeypatch);legs=[(f.route[0][0],.4,'bin_transition')]
    goal,_=f.n._make_retained_arm_trajectory_goal(legs);watchdog=.8
    if fault=='phase':legs=[(legs[0][0],.4,'recovery')]
    if fault=='goal':goal.trajectory.points[0].positions[0]+=.01
    if fault=='plan':f.plan.setup[1][1]+=.01
    if fault=='identity':f.n._contact_epoch+=1
    if fault=='negative_feedback':f.n.joints[stop.ARM_JOINTS[-1]]=-.1
    if fault=='stale_feedback':f.n._joint_stamps_ns[stop.ARM_JOINTS[0]]=1
    if fault=='watchdog':watchdog=.81
    with pytest.raises((stop.TransitionStopRejected,velocity.ArmVelocityAdmissionRejected)):
        f.n._send_retained_arm_trajectory(goal,watchdog,legs,'place',leg_offset=0,
            velocity_admission=True,transition_stop=f.token)
    assert not f.sent and not f.token.entry_published
    assert not f.n._pending_retained_acceptances and not f.n._goal_handles and not f.n._cancel.is_set()


def test_direct_sender_cannot_bypass_both_fresh_velocity_admissions(model,monkeypatch):
    f=prepared(model,monkeypatch);legs=[(f.route[0][0],.8,'bin_transition')]
    goal,_=f.n._make_retained_arm_trajectory_goal(legs)
    with pytest.raises(velocity.ArmVelocityAdmissionRejected,match='exclusive fresh'):
        f.n._send_retained_arm_trajectory(goal,.8,legs,'place',leg_offset=0,transition_stop=f.token)
    assert not f.sent and not f.checks and not f.n._pending_retained_acceptances


def test_integer_floor_restores_exact800ms_without_fractional_nanosecond_overrun():
    goal=make([([.1]*7,400_000_000)]);record=admit(goal)
    owned,_,diagnostic=extra.retime_admitted_arm_goal(goal,record,2.,nominal_duration=.8,
        minimum_segment_ns=800_000_000)
    assert ns(owned)==800_000_000 and ns(goal)==400_000_000
    assert diagnostic['minimum_segment_ns']==800_000_000
    assert diagnostic['new_total_ns']==800_000_000
    extra.require_retimed_arm_headroom(admit(owned))


@pytest.mark.parametrize('floor',[True,87_499_999,800_000_000.,800_000_001])
def test_invalid_or_over_budget_floor_is_rejected(floor):
    goal=make([([.1]*7,400_000_000)])
    with pytest.raises(velocity.ArmVelocityAdmissionRejected):
        extra.retime_admitted_arm_goal(goal,admit(goal),2.,nominal_duration=.8,minimum_segment_ns=floor)


@pytest.mark.parametrize('search',[None,{'wrist_policy':'negative'},{'wrist_policy':'measured_positive'}])
def test_actual_place_callsite_passes_only_selected_positive_plan_to_owner(monkeypatch,search):
    from test_book_centered_place import place_fixture
    from test_bin_scene_admission import fixture as bin_fixture
    n,_,_=place_fixture(True)
    monkeypatch.setattr(manipulation,'_activate_place_contacts',lambda *args:None)
    n.place_transition_stop_enabled=True;n.placement_transport_speed_scale=3.;n.loaded_place_speed_scale_cap=2.
    n.table_scene_required=n.bin_scene_required=True
    n._selected_place_bin_scene=bin_fixture()[0]
    n._selected_place_table_scene={};n._selected_place_scene_reference=object()
    n._wait_for_perception_point=lambda _:np.asarray(n._selected_place_bin_scene['floor_center'])
    n._publish_status=lambda *args,**kwargs:None
    n._move_torso=lambda *args,**kwargs:True
    n._retention_after_leg=lambda *args,**kwargs:True
    q=n._carried_staging_solution.copy()
    plan=SceneCheckedPlacePlan([q.copy(),q.copy()],0,0.,[q.copy(),q.copy(),q.copy()],[],
                              {'cartesian_search':search,'measured_master':MASTER},[])
    monkeypatch.setattr(manipulation,'plan_scene_checked_place',lambda *args,**kwargs:plan)
    observed=[]
    class ReachedOwner(Exception):pass
    def options(node,identity,legs,scale,master,**kwargs):
        observed.append(kwargs);assert node is n and scale==2. and master==MASTER
        raise ReachedOwner()
    monkeypatch.setattr(stop,'normal_options',options)
    with pytest.raises(ReachedOwner):n._place()
    assert observed==([{'positive_entry_plan':plan}] if search and search['wrist_policy']=='measured_positive' else [{}])
