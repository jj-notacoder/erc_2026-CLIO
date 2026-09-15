"""Actual callback/action/open method ordering with bounded fake transports; no ROS.

Only method ASTs are loaded, so importing this test cannot create a node or
subscription. The actual scene-context and bin-freshness implementations run.
"""
from pathlib import Path
from erc_phase1_solution.completed_torso_hold import track_completed_torso_hold
import ast
import copy
import json
import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution.bin_geometry import bin_message_is_current, bin_message_is_verified
from erc_phase1_solution.placement_scene_context import measured_scene_context, matching_table_scene
from erc_phase1_solution.place_contact_guard import fault_reason, require_clear


PACKAGE = Path(__file__).resolve().parents[1]/'erc_phase1_solution'


class Clock:
    def __init__(self): self.ns=1_000_000_000
    def now(self): return SimpleNamespace(nanoseconds=self.ns)
    def monotonic(self): return self.ns/1e9
    def sleep(self,duration): self.ns+=round(duration*1e9)


class RetainedMotionNotStopped(RuntimeError):
    pass


def methods(clock):
    tree=ast.parse((PACKAGE/'manipulation_node.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    wanted={'_on_bin','_on_bin_status','_wait_for_perception_point','_follow','_open_gripper',
            '_payload_hazard_reason','_send_retained_arm_trajectory'}
    functions=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in wanted]
    namespace=dict(__package__="erc_phase1_solution",track_completed_torso_hold=track_completed_torso_hold,np=np,time=clock,decode_event=json.loads,
                   _place_contact_fault=fault_reason,_require_place_contact_clear=require_clear,
                   measured_scene_context=measured_scene_context,matching_table_scene=matching_table_scene,
                   bin_message_is_current=bin_message_is_current,bin_message_is_verified=bin_message_is_verified,
                   FollowJointTrajectory=SimpleNamespace(Goal=lambda:SimpleNamespace(trajectory=SimpleNamespace())),
                   JointTrajectoryPoint=SimpleNamespace,
                   Duration=lambda **kw:SimpleNamespace(to_msg=lambda:kw),
                   GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4),
                   _stock_diagnostic=lambda node:False, RetainedMotionNotStopped=RetainedMotionNotStopped,
                   LiftPressureRejected=type('LiftPressureRejected',(RuntimeError,),{}))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(PACKAGE/'manipulation_node.py'),'exec'),namespace)
    return namespace


def node_fixture():
    clock=Clock();functions=methods(clock)
    parked=[*[f'arm_right_{i}_joint' for i in range(1,8)],'head_1_joint','head_2_joint']
    node=SimpleNamespace(_lock=threading.RLock(),_cancel=threading.Event(),
                         joints=dict.fromkeys(parked,0.),_joint_stamps_ns=dict.fromkeys(parked,clock.ns),
                         _staging_odom=dict(stamp_ns=clock.ns,pose=[0.,0.,0.],linear_speed=0.,angular_speed=0.),
                         right_chain=SimpleNamespace(active_names=('torso_lift_joint',*parked[:7])),
                         head_chain=SimpleNamespace(active_names=('torso_lift_joint',*parked[7:])),
                         get_clock=lambda:clock,timeout=.2,_goal_handles=[],
                         _payload_robot_watchdog_enabled=False,_target_robot_contact_latched=False,
                         table_scene_required=True,bin_candidate=None,latest_bin=None,
                         bin_verified_ns=-1,bin_invalidated_ns=-1,perception_wait=.12)
    events=[]
    node._publish_status=lambda *a,**kw:events.append((a,kw))
    node._wait_future=lambda future,timeout:future
    node._point_in_base=lambda msg:np.array([msg.point.x,msg.point.y,msg.point.z])
    for name in ['_on_bin','_on_bin_status','_wait_for_perception_point','_follow','_open_gripper',
                 '_payload_hazard_reason','_send_retained_arm_trajectory']:
        setattr(node,name,MethodType(functions[name],node))
    node._active_place_scene_reference=measured_scene_context(node)
    node._target_contact_recent=lambda **kw:True
    node._trajectory_leg_at_elapsed_time=lambda legs,elapsed:(0,'place')
    node._valid_retained_terminal_result=lambda wrapped:wrapped.status==4
    return node,clock,events


def point(stamp=1_000_000_000):
    return SimpleNamespace(header=SimpleNamespace(frame_id='base_footprint',
             stamp=SimpleNamespace(sec=stamp//1_000_000_000,nanosec=stamp%1_000_000_000)),
             point=SimpleNamespace(x=.86,y=.01,z=.75))


def table_status(stamp=1_000_000_000,**updates):
    result=dict(mode='bin',event='bin_verified',bin_valid=True,observation_stamp_ns=stamp,
                table_scene=dict(valid=True,frame='base_footprint',model='erc_table_two_edge_rgbd_v1'),
                bin_floor_point_base=[.86,.01,.75])
    result.update(updates)
    return SimpleNamespace(data=json.dumps(result))


def action(node,*,on_wait=None,on_result=None,done=False,cancel_confirmed=True):
    sent=[];cancelled=[]
    class Result:
        called=False
        def done(self):
            if not self.called:
                self.called=True
                if on_result:on_result()
            return done
        def result(self):return SimpleNamespace(status=4)
    result=Result()
    handle=SimpleNamespace(accepted=True,get_result_async=lambda:result)
    def wait(**kw):
        if on_wait:on_wait()
        return True
    client=SimpleNamespace(wait_for_server=wait,send_goal_async=lambda goal:sent.append(goal) or handle)
    node._cancel_goal_and_confirm=lambda *a:cancelled.append(a) or cancel_confirmed
    return client,sent,cancelled,handle


def call_follow(node,client):
    return node._follow(client,('arm_left_1_joint',),(.1,),.2)


@pytest.mark.parametrize('status_first',[False,True])
def test_exact_table_binding_handles_both_callback_orders(status_first):
    node,_,_=node_fixture()
    if status_first:node._on_bin_status(table_status())
    node._on_bin(point())
    if not status_first:
        assert node.latest_bin is None
        node._on_bin_status(table_status())
    np.testing.assert_allclose(node._wait_for_perception_point('latest_bin'),[.86,.01,.75])
    assert node._selected_place_scene_reference['parked_joints']==node.joints
    assert node._selected_place_table_scene['frame']=='base_footprint'


@pytest.mark.parametrize('entry,reason',[
    (None,'matching_table'),
    (table_status(999_000_000),'matching_table'),
    (table_status(table_scene=dict(valid=False,reason='no_two_edges')),'no_two_edges'),
    (table_status(bin_floor_point_base=[.86,.02,.75]),'frame_mismatch'),
])
def test_no_point_admission_without_its_fresh_registered_table(entry,reason):
    node,_,_=node_fixture();node.bin_verified_ns=1_000_000_000;node._on_bin(point())
    if entry is not None:node._on_bin_status(entry)
    with pytest.raises(RuntimeError,match=reason):node._wait_for_perception_point('latest_bin')
    assert not hasattr(node,'_selected_place_scene_reference')


def test_table_from_wrong_coordinate_frame_rejected_at_binding():
    node,_,_=node_fixture();node._on_bin(point())
    node._on_bin_status(table_status(table_scene=dict(valid=True,frame='world')))
    with pytest.raises(RuntimeError,match='frame'):
        node._wait_for_perception_point('latest_bin')


def test_stale_bin_point_cannot_reuse_old_matching_geometry():
    node,clock,_=node_fixture();node._on_bin(point());node._on_bin_status(table_status())
    clock.ns+=600_000_000
    with pytest.raises(RuntimeError,match='fresh verified'):
        node._wait_for_perception_point('latest_bin')


def test_new_invalid_observation_revokes_prior_point_and_scene():
    node,_,_=node_fixture();node._on_bin(point());node._on_bin_status(table_status())
    node._on_bin_status(table_status(1_010_000_000,bin_valid=False,table_scene=dict(valid=False,reason='occluded')))
    assert node.latest_bin is None
    with pytest.raises(RuntimeError,match='fresh verified'):
        node._wait_for_perception_point('latest_bin')


def test_older_callback_cannot_replace_newer_scene_entry():
    node,_,_=node_fixture();node._on_bin_status(table_status())
    selected=node._latest_table_scene_entry
    node._on_bin_status(table_status(900_000_000,table_scene=dict(valid=False)))
    assert node._latest_table_scene_entry is selected


def test_cancel_before_goal_does_not_wait_or_publish():
    node,_,_=node_fixture();node._cancel.set()
    client,sent,cancelled,_=action(node,on_wait=lambda:pytest.fail('waited after cancel'))
    assert not call_follow(node,client)
    assert sent==[] and cancelled==[]


def test_cancel_during_server_wait_cannot_publish_goal():
    node,_,_=node_fixture();client,sent,_,_=action(node,on_wait=node._cancel.set)
    assert not call_follow(node,client)
    assert sent==[]


def test_scene_change_during_server_wait_cannot_publish_goal():
    node,_,_=node_fixture()
    client,sent,_,_=action(node,on_wait=lambda:node.joints.update(head_2_joint=.003))
    with pytest.raises(RuntimeError,match='parked_geometry_moved'):call_follow(node,client)
    assert sent==[]


@pytest.mark.parametrize('confirmed',[True,False])
def test_scene_change_during_motion_cancels_and_preserves_unresolved_handle(confirmed):
    node,_,_=node_fixture()
    client,sent,cancelled,handle=action(node,on_result=lambda:node._staging_odom.update(pose=[.003,0,0]),
                                      cancel_confirmed=confirmed)
    reason='base_moved' if confirmed else 'cancellation_unconfirmed'
    with pytest.raises(RuntimeError,match=reason):call_follow(node,client)
    assert len(sent)==1 and len(cancelled)==1
    assert (handle in node._goal_handles) is (not confirmed)
    if not confirmed:assert node._cancel.is_set()


def test_cancel_during_motion_requests_cancellation():
    node,_,_=node_fixture();client,sent,cancelled,_=action(node,on_result=node._cancel.set)
    assert not call_follow(node,client)
    assert len(sent)==1 and len(cancelled)==1 and node._goal_handles==[]


def test_same_turn_terminal_result_cannot_hide_scene_drift():
    node,_,_=node_fixture()
    client,sent,_,_=action(node,on_result=lambda:node.joints.update(arm_right_7_joint=.003),done=True)
    with pytest.raises(RuntimeError,match='parked_geometry_moved'):call_follow(node,client)
    assert len(sent)==1
    assert node._goal_handles==[]  # already terminal, not an unresolved controller


def test_same_turn_terminal_result_cannot_hide_cancel():
    node,_,_=node_fixture();client,_,_,_=action(node,on_result=node._cancel.set,done=True)
    assert not call_follow(node,client)


def test_successful_follow_rechecks_stationary_scene_and_removes_handle():
    node,_,_=node_fixture();client,sent,_,_=action(node,done=True)
    assert call_follow(node,client)
    assert len(sent)==1 and node._goal_handles==[]


@pytest.mark.parametrize('fault',['stale','moving','cancel'])
def test_release_is_not_published_after_failed_scene_guard(fault):
    node,clock,_=node_fixture()
    held=object();node._held_book_corners=held;node._payload_monitor_enabled=True
    node._retention_probe_active=False;node._gripper_open_confirmed=False
    published=[];measured=[]
    node._command_gripper=lambda *a,**kw:published.append(a) or True
    if fault=='stale':clock.ns+=400_000_000
    if fault=='moving':node._staging_odom['linear_speed']=.006
    if fault=='cancel':node._cancel.set()
    if fault=='cancel':assert not node._open_gripper(verify_measurement=lambda:measured.append(True))
    else:
        with pytest.raises(RuntimeError,match='placement_scene_'):
            node._open_gripper(verify_measurement=lambda:measured.append(True))
    assert published==[] and measured==[]
    assert node._held_book_corners is held and node._payload_monitor_enabled


def test_unconfirmed_motion_stop_blocks_following_release_even_if_scene_recovers():
    node,_,_=node_fixture()
    client,_,_,_=action(node,on_result=lambda:node.joints.update(head_1_joint=.003),cancel_confirmed=False)
    with pytest.raises(RuntimeError,match='cancellation_unconfirmed'):call_follow(node,client)
    node.joints['head_1_joint']=0.
    published=[];node._command_gripper=lambda *a,**kw:published.append(a) or True
    assert not node._open_gripper()
    assert published==[]


def retained_action(node,*,acceptance_hook=None,result_hook=None,result_done=False,cancel_confirmed=True):
    sent=[];cancelled=[]
    class Future:
        def __init__(self,hook,terminal,value):
            self.checks=0;self.hook=hook;self.terminal=terminal;self.value=value
        def done(self):
            self.checks+=1
            if self.hook:self.hook(self.checks)
            return self.terminal(self.checks)
        def result(self):return self.value
        def add_done_callback(self,callback):pass
    result=Future(result_hook,lambda count:result_done,SimpleNamespace(status=4))
    handle=SimpleNamespace(accepted=True,get_result_async=lambda:result,cancel_goal_async=lambda:None)
    acceptance=Future(acceptance_hook,lambda count:count>(1 if acceptance_hook else 0),handle)
    node.arm_client=SimpleNamespace(send_goal_async=lambda goal:sent.append(goal) or acceptance)
    node._cancel_retained_goal_and_confirm=lambda *a:cancelled.append(a) or cancel_confirmed
    return sent,cancelled,handle


def call_retained(node):
    return node._send_retained_arm_trajectory(object(),.2,[(np.zeros(8),.2,'place')],'place')


def test_retained_sender_scene_fault_before_admission_sends_no_goal():
    node,_,events=node_fixture();sent,cancelled,_=retained_action(node)
    node.joints['head_1_joint']=.002
    assert call_retained(node)==(False,True)
    assert sent==[] and cancelled==[]
    assert events[-1][1]['reason']=='placement_scene_parked_geometry_moved'


def test_retained_sender_remembers_transient_scene_fault_during_acceptance():
    node,_,events=node_fixture()
    def accepted(count):node.joints['head_1_joint']=.002 if count==1 else 0.
    sent,cancelled,_=retained_action(node,acceptance_hook=accepted,result_done=True)
    assert call_retained(node)==(False,True)
    assert len(sent)==1 and len(cancelled)==1 and node._goal_handles==[]
    assert events[-1][1]['reason']=='placement_scene_parked_geometry_moved'


@pytest.mark.parametrize('confirmed',[True,False])
def test_retained_motion_context_fault_cancels_and_preserves_uncertain_stop(confirmed):
    node,_,events=node_fixture()
    sent,cancelled,handle=retained_action(node,result_hook=lambda count:node._staging_odom.update(pose=[.003,0,0]),
                                         cancel_confirmed=confirmed)
    if confirmed:assert call_retained(node)==(False,True)
    else:
        with pytest.raises(RetainedMotionNotStopped,match='could not be cancelled'):call_retained(node)
        assert node._cancel.is_set()
    assert len(sent)==1 and len(cancelled)==1
    assert (handle in node._goal_handles) is (not confirmed)
    assert events[-1][1]['reason']=='placement_scene_base_moved'


def test_retained_terminal_result_does_not_hide_same_turn_scene_fault():
    node,_,events=node_fixture()
    sent,cancelled,_=retained_action(node,result_hook=lambda count:node.joints.update(arm_right_1_joint=.002),result_done=True)
    assert call_retained(node)==(False,True)
    assert len(sent)==1 and cancelled==[] and node._goal_handles==[]
    assert events[-1][1]['reason']=='placement_scene_parked_geometry_moved'
