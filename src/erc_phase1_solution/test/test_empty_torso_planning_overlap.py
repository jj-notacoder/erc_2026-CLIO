"""Actual ROS goal/sender and original measured endpoint contracts; no physics claim."""
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

from erc_phase1_solution import empty_torso_planning_overlap as overlap
from erc_phase1_solution import empty_pickup_collision as collision
from erc_phase1_solution.completed_torso_hold import (
    TorsoHoldCancellation, TorsoHoldState, track_completed_torso_hold, require_follow_token_locked,
)
from erc_phase1_solution.motion_profiles import IK_JOINTS, RIGHT_ARM_JOINTS
from erc_phase1_solution.empty_pickup_collision import HEAD, MASTER, NAMES, EmptyPickupCollision
import test_optional_arm_timing as fixtures

ROOT=Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.wall=0.;self.ns=1_000_000_000;self.on_sleep=lambda:None
    def monotonic(self):return self.wall
    def sleep(self,seconds):
        self.wall+=seconds;self.ns+=int(seconds*1e9);self.on_sleep()


def completed(value):
    future=Future();future.set_result(value);return future


def fixture(monkeypatch, *, delayed_acceptance=False, locked_hook=None):
    clock=Clock();monkeypatch.setattr(overlap,'time',clock)
    monkeypatch.setattr(collision,'time',clock)
    n=NS(_lock=threading.Lock(),_cancel=TorsoHoldCancellation(),_torso_hold_state=TorsoHoldState(),
        settled_place_torso_skip_enabled=True,empty_torso_planning_overlap_enabled=True,
        _goal_handles=[],_pending_retained_acceptances=set(),_held_book_corners=None,
        _target_book_model=None,_contact_epoch=0,_empty_arm_contact_guard=True,
        _active_place_scene_reference=None,_cached_post_retreat_plan={'provisional':True},
        gripper_open=.069,adaptive_endpoint_tolerance=.001,pick_torso_height=.35,timeout=10.,
        joints=dict.fromkeys(NAMES,0.),_joint_stamps_ns=dict.fromkeys(NAMES,clock.ns),
        _joint_velocities=dict.fromkeys(NAMES,0.),get_clock=lambda:NS(now=lambda:NS(nanoseconds=clock.ns)),
        chain=NS(lower=np.array([-.001,*([-3.]*7)]),upper=np.array([.35,*([3.]*7)])),
        right_chain=object(),head_chain=object(),carried_collision_meshes=[],
        _shelf_cradle_geometry=NS(chains={}),_staging_odom=dict(stamp_ns=clock.ns,pose=np.array([1.,2.,.3]),linear_speed=0.,angular_speed=0.))
    n.joints[IK_JOINTS[0]]=.1;n.joints[MASTER]=.069
    command_lock=threading.RLock();n._adaptive_command_guard=lambda:command_lock
    events=[];n._publish_status=lambda *a,**k:events.append((a,k))
    fresh=fixtures.load('_lift_first_measurements',IK_JOINTS=IK_JOINTS,RIGHT_ARM_JOINTS=RIGHT_ARM_JOINTS)
    n._lift_first_measurements=lambda r=None:fresh(n,r)
    original_valid=fixtures.load('_valid_retained_terminal_result',GoalStatus=GoalStatus)
    n._valid_retained_terminal_result=original_valid
    result=completed(NS(status=GoalStatus.STATUS_SUCCEEDED,result=FollowJointTrajectory.Result()))
    cancels=[]
    handle=NS(accepted=True,get_result_async=lambda:result,cancel_goal_async=lambda:(cancels.append('cancel') or completed(NS())))
    acceptance=Future() if delayed_acceptance else completed(handle)
    sent=[];n.torso_client=NS(wait_for_server=lambda **k:True,send_goal_async=lambda goal:(sent.append(copy.deepcopy(goal)) or acceptance))
    n.arm_client=object();n.head_client=object()
    n._wait_future=lambda f,t:f.result()
    n._cancel_goal_and_confirm=lambda *a:True
    n._cancel_retained_goal_and_confirm=lambda *a:True
    late=fixtures.load('_cancel_late_retained_goal')
    n._cancel_late_retained_goal=lambda f,t:late(n,f,t)
    def locked(owner, token, goal):
        require_follow_token_locked(owner, token, goal)
        if locked_hook is not None:locked_hook(owner, goal)
    follow=fixtures.load('_follow',time=clock,Duration=Duration,JointTrajectoryPoint=JointTrajectoryPoint,
        FollowJointTrajectory=FollowJointTrajectory,GoalStatus=GoalStatus,
        require_follow_token_locked=locked)
    tracked=track_completed_torso_hold(follow)
    n._follow=lambda *a,**k:tracked(n,*a,**k)
    move=fixtures.load('_move_torso')
    n._move_torso=lambda *a,**k:move(n,*a,**k)
    start=np.array([n.joints[x] for x in IK_JOINTS])
    g=EmptyPickupCollision(n,start,[0.]*7,[0.]*2,.069,geometry=n._shelf_cradle_geometry,screen=object())
    g.velocity_limits=np.array([.035,*([2.]*7)])
    reference=n._lift_first_measurements()
    owner=overlap.EmptyTorsoPlanningOwner(n,g,reference)
    n._empty_torso_planning_owner=owner;owner.contact_epoch=0
    def update():
        n._joint_stamps_ns=dict.fromkeys(NAMES,clock.ns)
        n._staging_odom['stamp_ns']=clock.ns
    clock.on_sleep=update
    return n,g,owner,clock,sent,events,acceptance,handle,cancels


@pytest.mark.parametrize('value',[False,True])
def test_boolean_parameter(value):assert overlap.checked_enabled(value) is value


@pytest.mark.parametrize('value',[0,1,None,'true',[],float('nan')])
def test_non_boolean_parameter(value):
    with pytest.raises(ValueError):overlap.checked_enabled(value)


@pytest.mark.parametrize('enabled,ordinary,checker',[(False,True,object()),(True,False,object()),(True,True,object())])
def test_disabled_or_nonordinary_scope_never_opens_or_starts(enabled,ordinary,checker):
    n=NS(empty_torso_planning_overlap_enabled=enabled)
    with overlap.planning_scope(n,checker,None,ordinary=ordinary) as owner:assert owner is None
    assert not hasattr(n,'_empty_torso_planning_owner')


def test_actual_goal_preserves_original_torso_positions_duration_and_cleanup(monkeypatch):
    n,g,o,clock,sent,events,*_=fixture(monkeypatch)
    assert n._move_torso(.35,2.5,empty_torso_owner=o)
    assert len(sent)==1 and sent[0].trajectory.joint_names==[IK_JOINTS[0]]
    point=sent[0].trajectory.points[0]
    assert tuple(point.positions)==(.35,) and point.time_from_start==Duration(seconds=2.5).to_msg()
    assert not point.velocities and not point.accelerations and not point.effort
    assert not n._goal_handles and not n._pending_retained_acceptances
    # Nominal action success while still at.10 cannot release physical ownership.
    assert n._empty_torso_planning_owner is o and not o.measured_stopped and not o.joined
    assert n._torso_hold_state.completed[2]==.35


@pytest.mark.parametrize('fault',['head','right','left','torso','aperture','stale','cancel','base','model','epoch','target','contact'])
def test_actual_sender_rechecks_after_server_wait_and_never_publishes_invalid_context(monkeypatch,fault):
    n,g,o,clock,sent,*_=fixture(monkeypatch)
    def fail(**unused):
        if fault=='head':n.joints[HEAD[0]]=.002
        elif fault=='right':n.joints[RIGHT_ARM_JOINTS[0]]=.002
        elif fault=='left':n.joints[IK_JOINTS[1]]=.009
        elif fault=='torso':n.joints[IK_JOINTS[0]]=.103
        elif fault=='aperture':n.joints[MASTER]=.06
        elif fault=='stale':n._joint_stamps_ns[MASTER]-=350_000_001
        elif fault=='cancel':n._cancel.set()
        elif fault=='base':n._staging_odom['pose'][0]+=.003
        elif fault=='model':n.right_chain=object()
        elif fault=='epoch':n._contact_epoch+=1
        elif fault=='target':n._target_book_model='other'
        elif fault=='contact':n._empty_arm_contact_latched=True
        return True
    n.torso_client.wait_for_server=fail
    try:out=n._move_torso(.35,2.5,empty_torso_owner=o)
    except RuntimeError:out=False
    assert not out and sent==[] and n._empty_torso_planning_owner is o


def test_cancel_during_acceptance_keeps_token_and_late_goal_is_cancelled(monkeypatch):
    n,g,o,clock,sent,events,acceptance,handle,cancels=fixture(monkeypatch,delayed_acceptance=True)
    clock.on_sleep=n._cancel.set
    with pytest.raises(RuntimeError):n._move_torso(.35,2.5,empty_torso_owner=o)
    assert len(sent)==1 and o.token in n._pending_retained_acceptances
    acceptance.set_result(handle)
    assert cancels==['cancel'] and not n._pending_retained_acceptances and not n._goal_handles
    assert n._empty_torso_planning_owner is o and not o.joined
    assert n._torso_hold_state.uncertain


@pytest.mark.parametrize('regression',['clock','producer'])
def test_regression_during_planning_never_revives_owner(monkeypatch,regression):
    n,g,o,clock,*_=fixture(monkeypatch)
    o.check()
    if regression=='clock':clock.ns-=1
    else:n._joint_stamps_ns[IK_JOINTS[0]]-=1
    with pytest.raises(RuntimeError,match='regress'):o.check()


def test_original_endpoint_and_two_new_stopped_samples_are_both_required(monkeypatch):
    n,g,o,clock,sent,events,*_=fixture(monkeypatch)
    original_update=clock.on_sleep
    snapshots=iter([(.1879,.035),(.3482,.035),(.35,0.),(.35,0.)])
    def progress():
        original_update()
        q,v=next(snapshots,(.35,0.))
        n.joints[IK_JOINTS[0]]=q;n._joint_velocities[IK_JOINTS[0]]=v
    clock.on_sleep=progress
    o._motion()
    assert o.done.is_set() and o.motion_succeeded and o.measured_stopped
    assert n._empty_torso_planning_owner is o
    verified=[e for e in events if e[0][0]=='empty_pickup_endpoint_verified']
    assert len(verified)==1 and verified[0][1]['last_evaluated_measurement']['measured_left'][0]==.3482
    assert clock.ns>=1_080_000_000
    o.close(planner_failed=False)
    assert o.joined and n._empty_torso_planning_owner is None


@pytest.mark.parametrize('rate', [.05, .13])
def test_slow_simulation_preserves_physical_torso_settling_window(monkeypatch, rate):
    n,g,o,clock,*_=fixture(monkeypatch)
    n.get_clock=lambda:NS(ros_time_is_active=True, now=lambda:NS(nanoseconds=clock.ns))
    n.joints[IK_JOINTS[0]]=.35
    n._joint_velocities[IK_JOINTS[0]]=1e-4
    update=clock.on_sleep
    def sleep(seconds):
        clock.wall+=seconds;clock.ns+=round(seconds*rate*1e9)
        update()
        if clock.ns>=2_000_000_000:n._joint_velocities[IK_JOINTS[0]]=0.
    clock.sleep=sleep
    o._stop_evidence()
    assert clock.wall>5. and o.measured_stopped
    assert o.stopped_stamp>2_000_000_000


@pytest.mark.parametrize('frozen', ['clock', 'feedback'])
def test_simulated_torso_stop_still_bounds_missing_progress(monkeypatch, frozen):
    n,g,o,clock,*_=fixture(monkeypatch)
    n.get_clock=lambda:NS(ros_time_is_active=True, now=lambda:NS(nanoseconds=clock.ns))
    n.joints[IK_JOINTS[0]]=.35
    n._joint_velocities[IK_JOINTS[0]]=1e-4
    def sleep(seconds):
        clock.wall+=seconds
        if frozen!='clock':clock.ns+=round(seconds*.01*1e9)
    clock.sleep=sleep
    with pytest.raises(TimeoutError, match='measured stop'):
        o._stop_evidence()
    assert 5.<=clock.wall<=5.03 and not o.measured_stopped


def test_simulated_torso_stop_does_not_extend_five_ros_seconds(monkeypatch):
    n,g,o,clock,*_=fixture(monkeypatch)
    n.get_clock=lambda:NS(ros_time_is_active=True, now=lambda:NS(nanoseconds=clock.ns))
    n.joints[IK_JOINTS[0]]=.35
    n._joint_velocities[IK_JOINTS[0]]=1e-4
    with pytest.raises(TimeoutError, match='measured stop'):
        o._stop_evidence()
    assert clock.ns<=6_020_000_000 and not o.measured_stopped


def test_pure_planning_failure_waits_for_owned_ascent_then_keeps_failed_interlock(monkeypatch):
    n,g,o,clock,sent,events,*_=fixture(monkeypatch)
    original_update=clock.on_sleep
    def arrived():
        original_update();n.joints[IK_JOINTS[0]]=.35
    clock.on_sleep=arrived
    o._motion()
    assert o.motion_succeeded and o.measured_stopped and len(sent)==1
    o.close(planner_failed=True)
    assert n._empty_torso_planning_owner is o and o.fault
    assert n._cached_post_retreat_plan is None and n._cancel.is_set()


def test_unknown_physical_stop_is_not_released_by_terminal_success(monkeypatch):
    n,g,o,clock,sent,events,*_=fixture(monkeypatch)
    assert n._move_torso(.35,2.5,empty_torso_owner=o)
    o.done.set();o._fault('original endpoint stale after controller success')
    with pytest.raises(RuntimeError):o.close(planner_failed=False)
    assert n._empty_torso_planning_owner is o and not o.measured_stopped
    assert n._cached_post_retreat_plan is None and n._cancel.is_set()


@pytest.mark.parametrize('fault',['epoch','model','head','contact'])
def test_context_fault_after_action_success_during_original_endpoint_wait(monkeypatch,fault):
    n,g,o,clock,sent,events,*_=fixture(monkeypatch)
    update=clock.on_sleep
    def fail():
        update()
        n.joints[IK_JOINTS[0]]=.2
        if fault=='epoch':n._contact_epoch+=1
        elif fault=='model':n.head_chain=object()
        elif fault=='head':n.joints[HEAD[0]]=.002
        else:n._empty_arm_contact_latched=True
    clock.on_sleep=fail
    o._motion()
    assert len(sent)==1 and o.done.is_set() and o.fault and n._cancel.is_set()
    assert not o.motion_succeeded and not o.measured_stopped and not o.joined
    assert n._empty_torso_planning_owner is o
    assert any(e[0][0]=='empty_pickup_endpoint_rejected' for e in events)


@pytest.mark.parametrize('velocity',[.000002,float('nan'),None])
def test_early_stop_does_not_admit_later_moving_or_unknown_join(monkeypatch,velocity):
    n,g,o,clock,sent,events,*_=fixture(monkeypatch)
    update=clock.on_sleep
    def arrived():
        update();n.joints[IK_JOINTS[0]]=.35
    clock.on_sleep=arrived
    o._motion()
    assert o.motion_succeeded and o.measured_stopped
    clock.sleep(.02)
    n._joint_velocities[IK_JOINTS[0]]=velocity
    with pytest.raises(RuntimeError,match='current measured stop'):o.close(planner_failed=False)
    assert n._empty_torso_planning_owner is o and n._cancel.is_set()
    assert n._cached_post_retreat_plan is None


def test_stop_is_rechecked_at_owner_release_after_join_status_callback(monkeypatch):
    n,g,o,clock,sent,events,*_=fixture(monkeypatch)
    update=clock.on_sleep
    def arrived():
        update();n.joints[IK_JOINTS[0]]=.35
    clock.on_sleep=arrived;o._motion()
    publish=n._publish_status
    def publish_then_move(event,**kw):
        publish(event,**kw)
        if event=='empty_torso_planning_overlap_joined':n._joint_velocities[IK_JOINTS[0]]=.035
    n._publish_status=publish_then_move
    with pytest.raises(RuntimeError,match='current measured stop'):o.close(planner_failed=False)
    assert n._empty_torso_planning_owner is o and o.fault and n._cancel.is_set()


@pytest.mark.parametrize('field',['_payload_hazard_latched','_held_grip_sensor_fault',
    '_raw_contacts_first_failure','_target_robot_contact_latched','_empty_arm_contact_latched'])
def test_preopen_latch_cannot_be_erased_by_opening(monkeypatch,field):
    n,g,o,clock,sent,*_=fixture(monkeypatch)
    n._empty_torso_planning_owner=None
    setattr(n,field,True)
    opened=[];n._open_gripper=lambda:(opened.append(True) or True)
    with pytest.raises(RuntimeError,match='existing motion/pool/payload'):
        with overlap.planning_scope(n,g,o.reference,ordinary=True):pytest.fail('entered planning')
    assert opened==[] and sent==[] and n._empty_torso_planning_owner is None


@pytest.mark.parametrize('field',['start','right','head','velocity_limits','initial_aperture','open_aperture','tolerance'])
def test_frozen_checker_context_cannot_be_replaced(monkeypatch,field):
    n,g,o,clock,sent,*_=fixture(monkeypatch)
    if field=='tolerance':n.adaptive_endpoint_tolerance=.002
    elif field.endswith('aperture'):setattr(g,field,.068)
    else:
        changed=getattr(g,field).copy();changed[-1]+=.0001;setattr(g,field,changed)
    with pytest.raises(RuntimeError,match='model/plan changed'):o.check()
    assert sent==[]


@pytest.mark.parametrize('fault',['goal','epoch','target'])
def test_actual_sender_final_locked_admission_rejects_race(monkeypatch,fault):
    def mutate(n,goal):
        if fault=='goal':goal.trajectory.points[0].positions=[.349]
        elif fault=='epoch':n._contact_epoch+=1
        else:n._target_book_model='replaced'
    n,g,o,clock,sent,*_=fixture(monkeypatch,locked_hook=mutate)
    with pytest.raises(RuntimeError):n._move_torso(.35,2.5,empty_torso_owner=o)
    assert sent==[] and not n._goal_handles and not n._pending_retained_acceptances


def test_actual_sender_result_accessor_failure_still_requests_cancel(monkeypatch):
    n,g,o,clock,sent,events,acceptance,handle,cancels=fixture(monkeypatch)
    calls=[]
    def fail():calls.append('result');raise RuntimeError('result accessor failed')
    handle.get_result_async=fail
    o._motion()
    assert len(sent)==1 and calls==['result','result'] and cancels==['cancel']
    assert handle in n._goal_handles and o.token in n._pending_retained_acceptances
    assert n._empty_torso_planning_owner is o and o.fault and n._cancel.is_set()
    assert not o.measured_stopped and not o.joined


@pytest.mark.parametrize('confirmation',[False,True])
def test_unconfirmed_cancel_never_forgets_pending_accepted_action(monkeypatch,confirmation):
    n,g,o,clock,sent,events,acceptance,handle,cancels=fixture(monkeypatch)
    pending=Future();handle.get_result_async=lambda:pending
    n._cancel_retained_goal_and_confirm=lambda *args:confirmation
    clock.on_sleep=n._cancel.set
    o._motion()
    assert len(sent)==1 and cancels and not pending.done()
    assert handle in n._goal_handles and n._empty_torso_planning_owner is o
    assert o.fault and not o.joined


def bind_actual_open(n, opened):
    clear=fixtures.load('_clear_target_contact_samples_unlocked')
    n._clear_target_contact_samples_unlocked=lambda **kw:clear(n,**kw)
    original=fixtures.load('_open_gripper',_place_contact_fault=lambda owner:None)
    n._open_gripper=lambda:original(n)
    def command(position,**kw):
        assert position==.069 and kw=={'respect_cancel':True}
        opened.append(position);n.joints[MASTER]=position;return True
    n._command_gripper=command


def test_actual_scope_opens_before_new_plan_and_blocks_arm_until_thread_stop_join(monkeypatch):
    n,g,old_owner,clock,sent,events,*_=fixture(monkeypatch)
    n._empty_torso_planning_owner=None;n._target_book_model='book_before_open'
    opened=[];bind_actual_open(n,opened)
    endpoint_waiting=threading.Event();allow_endpoint=threading.Event()
    join_entered=threading.Event();arm_ready=threading.Event();failures=[];new_plan=object()
    original_join=overlap.EmptyTorsoPlanningOwner.join
    def join(owner):
        join_entered.set();return original_join(owner)
    monkeypatch.setattr(overlap.EmptyTorsoPlanningOwner,'join',join)
    update=clock.on_sleep
    def progress():
        endpoint_waiting.set()
        assert allow_endpoint.wait(3.),'test endpoint release timed out'
        update();n.joints[IK_JOINTS[0]]=.35;n._joint_velocities[IK_JOINTS[0]]=0.
    clock.on_sleep=progress
    def planning():
        try:
            with overlap.planning_scope(n,g,old_owner.reference,ordinary=True) as owner:
                assert endpoint_waiting.wait(3.)
                assert n._cached_post_retreat_plan is None and n._target_book_model is None
                assert n._contact_epoch==1 and n._gripper_open_confirmed
                assert owner.thread.is_alive() and not owner.measured_stopped
                n._cached_post_retreat_plan=new_plan
            assert owner.joined and not owner.thread.is_alive()
            arm_ready.set()
        except BaseException as error:failures.append(error)
    planner=threading.Thread(target=planning)
    planner.start()
    try:
        assert join_entered.wait(3.),failures
        assert not arm_ready.is_set() and planner.is_alive()
        assert n._empty_torso_planning_owner is not None
        assert n._cached_post_retreat_plan is new_plan and opened==[.069]
    finally:
        allow_endpoint.set();planner.join(3.)
    assert not planner.is_alive() and not failures
    assert arm_ready.is_set() and n._empty_torso_planning_owner is None
    assert n._cached_post_retreat_plan is new_plan and opened==[.069] and len(sent)==1


def test_actual_command_gate_blocks_owner_even_after_busy_cleared(monkeypatch):
    n,g,o,clock,sent,events,*_=fixture(monkeypatch)
    n._busy=False
    method=fixtures.load('_on_command',String=NS,decode_event=lambda value:{'event':value})
    method(n,NS(data='stow'))
    assert events[-1][0]==('rejected',) and events[-1][1]['reason']=='empty_torso_planning_owner_active'
    assert sent==[] and not n._busy


def test_source_keeps_original_empty_scope_before_overlap_and_all_arm_calls_after_join():
    tree=fixtures.node_tree()
    pick=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick')
    scopes=[n for n in ast.walk(pick) if isinstance(n,ast.With)]
    named=lambda n,name:any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name)
        and i.context_expr.func.id==name for i in n.items)
    empty=next(n for n in scopes if named(n,'empty_pickup_geometry_scope'))
    torso=next(n for n in scopes if named(n,'empty_torso_planning_scope'))
    assert empty.end_lineno<torso.lineno
    calls=[n for n in ast.walk(torso) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in
        ('_move_arm_solution','_move_torso','_follow','_command_gripper','_open_gripper') for n in calls)
    assert any(isinstance(n.func,ast.Name) and n.func.id=='run_pickup_geometry' for n in calls)
    ordinary=next(k.value for i in torso.items for k in i.context_expr.keywords if k.arg=='ordinary')
    assert ast.dump(ordinary)==ast.dump(ast.parse('bool(top_row and lift_enabled and not staged_empty_gripper)',mode='eval').body)
    assert "'empty_torso_planning_overlap_enabled': False" in (ROOT/'erc_phase1_solution/manipulation_node.py').read_text()
