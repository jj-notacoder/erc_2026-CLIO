"""Portable exact-pair, lifecycle and actual-node AST command race tests."""
import ast
from pathlib import Path
from erc_phase1_solution.completed_torso_hold import track_completed_torso_hold
from types import SimpleNamespace as NS
import threading
import math

import numpy as np
import pytest

from erc_phase1_solution import place_contact_guard as guard
from erc_phase1_solution import planning_cpu_diagnostics as diagnostics

BIN='erc_collection_bin::collection_bin_base_link::base_link_collection_bin_collision'
TABLE='erc_table::table_base_link::base_link_table_collision'
ARM='tiago_pro::arm_left_3_link::arm_left_3_link_collision'
BOOK='book_col_3_row_2_red::book_base_link::base_link_book_collision'
FINGER='tiago_pro::gripper_left_fingertip_left_link::fingertip_collision'


def message(first=ARM,second=TABLE,stamp=10_050_000_000):
    header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9)) if stamp is not None else None
    return NS(header=header,contacts=[NS(collision1=NS(name=first),collision2=NS(name=second))])


def node():
    n=NS(_lock=threading.Lock(),_adaptive_command_lock=threading.RLock(),_cancel=threading.Event(),table_scene_required=True)
    n._adaptive_command_guard=lambda:n._adaptive_command_lock
    n.now=10_000_000_000;n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=n.now))
    n.holds=[];n.events=[]
    def hold(reason):
        assert n._lock.acquire(blocking=False),'sensor lock held during command'
        n._lock.release();n.holds.append(reason);n._adaptive_hold_sent=True
    n._hold_adaptive_gripper=hold
    n._publish_status=lambda *a,**kw:n.events.append((a,kw))
    return n


@pytest.mark.parametrize('side',['left','right'])
@pytest.mark.parametrize('joint',range(1,8))
@pytest.mark.parametrize('scene',[BIN,TABLE])
def test_exact_arm_pair_both_orders(side,joint,scene):
    arm=f'tiago_pro::arm_{side}_{joint}_link::collision'
    assert guard.collision_pair(arm,scene)['robot_link']==f'arm_{side}_{joint}_link'
    assert guard.collision_pair(scene,arm)['scene_model'] in guard.SCENE_MODELS


@pytest.mark.parametrize('link',['base','inner_finger_left','inner_finger_right','outer_finger_left','outer_finger_right','fingertip_left','fingertip_right','screw_left','screw_right'])
def test_all_nine_nominal_tool_links(link):
    assert guard.collision_pair(f'tiago_pro::gripper_left_{link}_link::collision',BIN)


@pytest.mark.parametrize('enabled',[False,True])
def test_bin_subscription_only_feeds_attempt_guard_without_grip_aggregation(enabled):
    path=Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    block=next(x for x in init.body if isinstance(x,ast.If)
               and ast.unparse(x.test)=='self.table_scene_required'
               and "'/bin_contacts'" in ast.unparse(x))
    n=node();n.table_scene_required=enabled;subscriptions=[]
    n.create_subscription=lambda *args:subscriptions.append(args)
    n._on_contacts=lambda *_:pytest.fail('bin-side contact duplicated into grip aggregation')
    scope=dict(self=n,Contacts=object(),SENSOR_QOS=object(),_observe_place_contacts=guard.observe,
               _planning_cpu_callback=diagnostics.callback)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),str(path),'exec'),scope)
    if not enabled:
        assert not subscriptions
        return
    assert len(subscriptions)==1 and subscriptions[0][1]=='/bin_contacts'
    callback=subscriptions[0][2]
    frame=message('tiago_pro::gripper_left_base_link::base_collision',BIN)
    callback(frame);assert not n.holds  # No active PLACE yet.
    guard.activate(n);callback(frame)
    assert n._cancel.is_set() and n.events[-1][1]['source_topic']=='/bin_contacts'
    assert n.events[-1][1]['producer_stamp_ns']==10_050_000_000


@pytest.mark.parametrize('first,second',[(BOOK,BIN),(BOOK,FINGER),(BIN,TABLE),('tiago_pro::wheel_left_link::collision',TABLE),(ARM,'ground_plane::link::collision'),('other_tiago_pro::arm_left_3_link::collision',BIN),(ARM,'not_erc_table::table_base_link::collision')])
def test_intended_pairs_and_unrelated_names_do_not_latch(first,second):
    assert guard.collision_pair(first,second) is None


def test_attempt_scope_old_frames_ignored_and_missing_stamp_stops():
    n=node();guard.observe(n,message());assert not n.holds
    guard.activate(n,{'trial_id':'trial','placement_attempt_id':'attempt','target_model':'book'})
    guard.observe(n,message(stamp=9_999_999_999));assert not n.holds
    guard.observe(n,message(stamp=None));assert n._cancel.is_set()
    fields=n.events[-1][1]
    assert fields['producer_stamp_ns'] is None and fields['producer_stamp_valid'] is False
    assert fields['trial_id']=='trial' and fields['activation_ns']==10_000_000_000


def test_latch_survives_empty_frames_release_and_is_first_snapshot():
    n=node();guard.activate(n);m=message();guard.observe(n,m)
    original=dict(n._place_contact_guard.fault)
    m.contacts[0].collision1.name='changed';n.now+=10**9;n._held_book_corners=None
    guard.observe(n,NS(contacts=[],header=None));guard.observe(n,message(FINGER,BIN,stamp=11_000_000_000))
    assert n._place_contact_guard.fault==original and len(n.holds)==1 and len(n.events)==1
    assert guard.fault_reason(n)=='place_scene_contact'
    guard.deactivate(n);assert guard.fault_reason(n) is None
    n._cancel.clear();guard.activate(n);guard.observe(n,message(stamp=10_050_000_000))
    assert not n._cancel.is_set()  # no queued old collision in the new attempt


def test_parse_race_cannot_latch_into_replacement_attempt():
    n=node();guard.activate(n)
    class Contacts:
        def __iter__(self):
            guard.deactivate(n);n.now+=10**9;guard.activate(n)
            yield message().contacts[0]
    m=message();m.contacts=Contacts();guard.observe(n,m)
    assert not n._cancel.is_set() and not n.holds


def test_default_disabled_has_no_guard_or_command_lock_requirement():
    n=NS(table_scene_required=False);guard.activate(n);guard.deactivate(n)
    guard.observe(n,message());assert guard.fault_reason(n) is None


def test_new_contact_attempt_can_stop_an_open_after_an_older_hold():
    n=node();guard.activate(n);n._adaptive_hold_sent=True;guard.observe(n,message())
    assert len(n.holds)==1 and n._adaptive_hold_sent


def test_hold_failure_keeps_cancel_and_reports_error():
    n=node();guard.activate(n)
    def failed(_):raise RuntimeError('feedback unavailable')
    n._hold_adaptive_gripper=failed;guard.observe(n,message())
    assert n._cancel.is_set() and guard.fault_reason(n)
    assert n.events[-1][1]['hold_error']=='feedback unavailable'


def methods():
    path=Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    cls=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    names={'_open_gripper','_publish_gripper_position','_command_gripper','_recover_closed_place','_best_effort','_follow','_payload_hazard_reason','_run_command'}
    selected=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in names]
    from erc_phase1_solution import empty_head_timing as actual_empty_head_timing
    from erc_phase1_solution import release_pose_finish
    scope=dict(__package__="erc_phase1_solution",release_pose_finish=release_pose_finish,_empty_head_timing=actual_empty_head_timing,track_completed_torso_hold=track_completed_torso_hold,np=np,math=math,time=NS(sleep=lambda _:None,monotonic=lambda:0.),
        _place_contact_fault=guard.fault_reason,_require_place_contact_clear=guard.require_clear,
        _deactivate_place_contacts=guard.deactivate,measured_scene_context=lambda *a:None,
        _stock_diagnostic=lambda n:False,JointTrajectory=lambda:NS(),JointTrajectoryPoint=lambda:NS(),
        Duration=lambda **kw:NS(to_msg=lambda:kw),FollowJointTrajectory=NS(Goal=lambda:NS(trajectory=NS())),
        GoalStatus=NS(STATUS_SUCCEEDED=4))
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='Actual',bases=[],keywords=[],body=selected,decorator_list=[])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),scope)
    return scope['Actual'],scope


def release_fixture():
    cls,scope=methods();n=node();actual=cls();actual.__dict__.update(n.__dict__)
    # Rebind fixture closures to the actual object's original shared lists/locks.
    actual.gripper_open=.069;actual.gripper_settle=1.;actual._held_book_corners=object()
    actual._payload_monitor_enabled=True;actual._retention_probe_active=False
    actual._clear_target_contact_samples_unlocked=lambda **kw:None
    actual.commands=[];actual.gripper_pub=NS(publish=lambda msg:actual.commands.append(msg.points[0].positions[0]))
    actual._wait_sim_duration=lambda _:not actual._cancel.is_set()
    guard.activate(actual)
    return actual,scope


def test_contact_between_duplicate_open_commands_leaves_hold_last():
    n,scope=release_fixture();held=n._held_book_corners
    def hold(reason):n.commands.append(.017)
    n._hold_adaptive_gripper=hold
    def sleep(_):
        if len(n.commands)==1:guard.observe(n,message(FINGER,BIN))
    scope['time'].sleep=sleep
    assert not n._open_gripper()
    assert n.commands==[.069,.017] and n._held_book_corners is held
    assert n._payload_monitor_enabled


def test_contact_prevents_recovery_open_even_if_cancel_flag_cleared():
    n,_=release_fixture();guard.observe(n,message());n._cancel.clear()
    assert not n._open_gripper() and not n.commands
    with pytest.raises(RuntimeError,match='place_recovery_failed'):
        n._recover_closed_place(cause='release_failed',remaining_approach_legs=(),cartesian_solutions=(),carried_transition_waypoints=(),carried_start=np.zeros(8),unloaded_home_waypoints=())
    assert not n.commands


def test_fault_is_retained_hazard_without_any_book_contact_queries():
    n,_=release_fixture();guard.observe(n,message())
    assert n._payload_hazard_reason()=='place_scene_contact'


def test_empty_action_cancels_on_contact_and_preserves_latch():
    n,_=release_fixture();n._held_book_corners=None;n._active_place_scene_reference={};n.timeout=1.;n._goal_handles=[]
    state={'terminal':False,'triggered':False};cancelled=[]
    def done():
        if not state['triggered']:
            state['triggered']=True;guard.observe(n,message())
        return state['terminal']
    result=NS(done=done,result=lambda:NS(status=4))
    handle=NS(accepted=True,get_result_async=lambda:result)
    client=NS(wait_for_server=lambda **kw:True,send_goal_async=lambda goal:NS(result=lambda:handle))
    n._wait_future=lambda future,timeout:future.result()
    def cancel(h,f):cancelled.append(h);state['terminal']=True;return True
    n._cancel_goal_and_confirm=cancel
    with pytest.raises(RuntimeError,match='place_scene_contact'):
        n._follow(client,['arm_left_3_joint'],[0.],1.)
    assert cancelled==[handle] and not n._goal_handles and guard.fault_reason(n)


def test_node_wiring_preserves_release_and_command_finally_scope():
    path=Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    body={x.name:x for x in cls.body if isinstance(x,ast.FunctionDef)}
    on=body['_on_contacts'];assert isinstance(on.body[0],ast.Expr) and on.body[0].value.func.id=='_observe_place_contacts'
    text=ast.unparse(body['_run_command']);assert '_deactivate_place_contacts(self)' in text.split('finally:')[1]
    place=ast.unparse(body['_place']);assert place.index('_activate_place_contacts')<place.index('plan_node_scene_checked_place(')
    assert 'verify_measurement=verify_open' in place and 'placement_release_unverified' in place


def test_contact_during_measurement_blocks_cleanup_and_return_admission():
    n,_=release_fixture();held=n._held_book_corners
    n._hold_adaptive_gripper=lambda reason:n.commands.append(.017)
    def verify():guard.observe(n,message());return True
    assert not n._open_gripper(verify_measurement=verify)
    assert n.commands==[.069,.069,.017] and n._held_book_corners is held
    assert n._payload_monitor_enabled and guard.fault_reason(n)


@pytest.mark.parametrize('outcome',[True,False,'raise'])
def test_actual_place_command_finally_clears_only_attempt_guard(outcome):
    n,_=release_fixture();n.delivery_evidence_enabled=False;n._busy=True;n.dry_run=False;n.book_row_tilts=[]
    n._active_place_scene_reference={};n._stow=lambda:True;n._move_head=lambda *a:True
    n._pick=lambda *a:True;n._compact_transport=lambda:True
    n.get_logger=lambda:NS(error=lambda message:None)
    def place():
        if outcome=='raise':raise RuntimeError('planning rejected')
        return outcome
    n._place=place
    n._run_command('place')
    assert n._place_contact_guard is None and n._active_place_scene_reference is None and not n._busy


def test_contact_at_final_empty_action_send_prevents_publication():
    from contextlib import contextmanager
    n,_=release_fixture();n._held_book_corners=None;n._active_place_scene_reference={};n.timeout=1.;n._goal_handles=[]
    sent=[];once=[True]
    @contextmanager
    def command_guard():
        with n._adaptive_command_lock:
            if once[0]:
                once[0]=False;guard.observe(n,message())
            yield
    n._adaptive_command_guard=command_guard
    client=NS(wait_for_server=lambda **kw:True,send_goal_async=lambda goal:sent.append(goal))
    with pytest.raises(RuntimeError,match='place_scene_contact'):
        n._follow(client,['arm_left_3_joint'],[0.],1.)
    assert not sent and n._cancel.is_set()
