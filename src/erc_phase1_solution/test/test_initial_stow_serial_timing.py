"""Actual original serial sender, frozen clocks and controlled action futures.

These tests verify software admission/ownership; they do not model faster plant
tracking, acceleration, stopping distance or physical retention.
"""
from concurrent.futures import Future
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import ast
import copy
import math
import threading

import pytest
from control_msgs.action import FollowJointTrajectory
from erc_phase1_solution import initial_stow_serial_timing as serial
from erc_phase1_solution import manipulation_node as runtime
from erc_phase1_solution.completed_torso_hold import TorsoHoldCancellation, TorsoHoldState


def done(value):
    f = Future(); f.set_result(value); return f


class Clock:
    def __init__(self):
        self.wall = 10.; self.ns = 1_000_000_000
        self.hook = lambda: None
    def monotonic(self):
        return self.wall
    def sleep(self, seconds):
        self.wall += seconds; self.ns += round(seconds*1e9); self.hook()


def fixture(monkeypatch, *, enabled=True):
    clock = Clock(); events = []
    command = threading.RLock()
    positions = dict.fromkeys(serial.NAMES, 0.)
    positions.update(zip(serial.IK_JOINTS, serial.HOME))
    positions.update(zip(serial.RIGHT_ARM_JOINTS, serial.RIGHT_HOME))
    positions[serial.ARM_JOINTS[0]] = .1
    positions[serial.RIGHT_ARM_JOINTS[0]] = -.1
    positions[serial.MASTERS[0]] = .069
    n = NS(initial_stow_serial_timing_enabled=enabled, _initial_stow_serial_owner=None,
        _initial_stow_serial_command_seen=False, _initial_stow_serial_limits=serial.Limits(
            tuple(serial.ARM_NAMES), (-4.,)*14, (4.,)*14, (1.,)*14, 'a'*64),
        settled_place_torso_skip_enabled=True, _torso_hold_state=TorsoHoldState(),
        _cancel=TorsoHoldCancellation(), _lock=threading.Lock(), _busy=True,
        _adaptive_command_guard=lambda:command, _right_parked=False, _empty_arm_staged=False,
        dry_run=False, timeout=1., gripper_open=.069, additional_arm_time_scale=2.,
        chain=object(), right_chain=object(), head_chain=object(), carried_collision_meshes=(),
        _goal_handles=[], _pending_retained_acceptances=set(), joints=positions,
        _joint_stamps_ns=dict.fromkeys(serial.NAMES, clock.ns),
        _joint_velocities=dict.fromkeys(serial.NAMES, 0.),
        _staging_odom=dict(stamp_ns=clock.ns, pose=(0.,0.,0.), linear_speed=0., angular_speed=0.),
        _active_place_scene_reference=None)
    n.get_clock = lambda:NS(now=lambda:NS(nanoseconds=clock.ns))
    n.get_logger = lambda:NS(error=lambda text:events.append(('error', text)))
    n._publish_status = lambda event,**data:events.append(('status', event, data))
    n._open_gripper = lambda:events.append(('open',)) or True
    n._valid_retained_terminal_result = runtime.ManipulationNode._valid_retained_terminal_result
    n._cancel_late_retained_goal = MethodType(runtime.ManipulationNode._cancel_late_retained_goal,n)
    n._wait_future = lambda f,timeout:f.result()
    def cancel(h, f):
        events.append(('confirm_cancel', h.side));h.cancel_goal_async();return f.done()
    n._cancel_goal_and_confirm = cancel
    n._cancel_retained_goal_and_confirm = cancel
    clients = []
    for side in ('torso', 'left', 'right'):
        client = NS(side=side, goals=[], acceptance=None, handle=None, terminal=None,
                    delayed_acceptance=False, delayed_terminal=False, update_endpoint=True,
                    status=4,error_code=0)
        def send(goal, client=client, side=side):
            owner = n._initial_stow_serial_owner
            if owner is not None and side != 'torso':
                assert owner.tokens[owner.side] in n._pending_retained_acceptances
                assert owner.follow_token in n._torso_hold_state.pending
                assert owner.follow_token.dispatched
                assert not n._goal_handles
                if side == 'right':
                    assert owner.side == 1 and all(abs(n.joints[j]-q) <= .005
                        for j,q in zip(serial.ARM_JOINTS,serial.HOME[1:]))
            events.append(('send',side,clock.ns))
            client.goals.append(copy.deepcopy(goal))
            terminal_value=NS(status=client.status,result=FollowJointTrajectory.Result())
            terminal_value.result.error_code=client.error_code
            client.terminal=Future() if client.delayed_terminal else done(terminal_value)
            client.handle=NS(side=side,accepted=True,get_result_async=lambda:client.terminal,
                cancel_goal_async=lambda:events.append(('cancel',side)) or done(NS()))
            client.acceptance=Future() if client.delayed_acceptance else done(client.handle)
            return client.acceptance
        client.send_goal_async=send
        client.wait_for_server=lambda **kw:True
        clients.append(client)
    n.torso_client,n.arm_client,n.right_arm_client=clients
    n.head_client=object()
    def update():
        n._joint_stamps_ns.update(dict.fromkeys(serial.NAMES,clock.ns))
        n._staging_odom['stamp_ns']=clock.ns
        for client in clients:
            if client.goals and client.update_endpoint:
                goal=client.goals[-1]
                n.joints.update(zip(goal.trajectory.joint_names,goal.trajectory.points[0].positions))
    clock.hook=update
    monkeypatch.setattr(serial,'time',clock)
    monkeypatch.setattr(runtime,'time',clock)
    monkeypatch.setattr(runtime,'_require_place_contact_clear',lambda node:events.append(('contact',)))
    n._follow=MethodType(runtime.ManipulationNode._follow,n)
    n._stow=MethodType(runtime.ManipulationNode._stow,n)
    n._move_ik=MethodType(runtime.ManipulationNode._move_ik,n)
    return n,clock,events


def start(n):
    return serial.begin_command(n,'stow')


def run(n):
    return n._stow(initial_stow_serial=start(n))


@pytest.mark.parametrize('value',[None,0,1,'true',[],float('nan')])
def test_option_rejects_ambiguous_values(value):
    with pytest.raises(ValueError):serial.checked_enabled(value)


def test_declaration_is_default_off_and_noninitial_scope_stays_original(monkeypatch):
    tree=ast.parse(Path(runtime.__file__).read_text())
    declarations=next(f for c in tree.body if isinstance(c,ast.ClassDef) for f in c.body
                      if isinstance(f,ast.FunctionDef) and f.name=='_declare_parameters')
    assert any(isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant)
        and k.value=='initial_stow_serial_timing_enabled' and isinstance(v,ast.Constant)
        and v.value is False for k,v in zip(n.keys,n.values)) for n in ast.walk(declarations))
    n,_,_=fixture(monkeypatch,enabled=False)
    assert start(n) is None and not n._initial_stow_serial_command_seen
    n.initial_stow_serial_timing_enabled=True
    assert serial.begin_command(n,'look_markers') is None
    assert start(n) is None
    n._initial_stow_serial_command_seen=False;n.dry_run=True
    assert start(n) is None


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_stow_serial_order_original_torso_and_exact_arm_timing(monkeypatch,enabled):
    n,clock,events=fixture(monkeypatch,enabled=enabled)
    assert run(n) is True
    assert [e[1] for e in events if e[0]=='send']==['torso','left','right']
    assert events[0]==('open',)
    for client,names,target,duration in (
        (n.torso_client,[serial.IK_JOINTS[0]],[serial.HOME[0]],(2,500_000_000)),
        (n.arm_client,serial.ARM_JOINTS,serial.HOME[1:],(2,0) if enabled else (3,500_000_000)),
        (n.right_arm_client,serial.RIGHT_ARM_JOINTS,serial.RIGHT_HOME,(2,0) if enabled else (3,500_000_000))):
        goal,=client.goals;point,=goal.trajectory.points
        assert goal.trajectory.joint_names==list(names) and list(point.positions)==list(target)
        assert (point.time_from_start.sec,point.time_from_start.nanosec)==duration
        assert not point.velocities and not point.accelerations and not point.effort
        assert not goal.path_tolerance and not goal.goal_tolerance
    assert not n._goal_handles and not n._pending_retained_acceptances
    assert n._initial_stow_serial_owner is None and n._right_parked
    assert not n._torso_hold_state.pending and not n._torso_hold_state.uncertain
    if enabled:
        left,right=[e[2] for e in events if e[0]=='send' and e[1]!='torso']
        assert right-left>=100_000_000  # Actual measured handoff, not callback success alone.


@pytest.mark.parametrize('field,value',[('_held_book_corners',(1,)),('_raw_contacts_first_failure','decode'),
    ('_payload_hazard_latched','hazard'),('_empty_arm_staged',True),('_retention_probe_active',True)])
def test_existing_faults_refuse_before_original_open(monkeypatch,field,value):
    n,_,events=fixture(monkeypatch);setattr(n,field,value)
    with pytest.raises(RuntimeError):run(n)
    assert not any(e[0] in ('open','send') for e in events)


@pytest.mark.parametrize('fault',['stale','future','reverse','nan','moving_base','moving_arm','torso','head','odom_reverse'])
def test_feedback_admission_refuses_without_an_arm_goal(monkeypatch,fault):
    n,clock,events=fixture(monkeypatch);owner=start(n)
    owner.check();name=serial.ARM_JOINTS[0]
    if fault=='stale':n._joint_stamps_ns[name]-=150_000_001
    if fault=='future':n._joint_stamps_ns[name]+=1
    if fault=='reverse':n._joint_stamps_ns[name]-=1
    if fault=='nan':n.joints[name]=math.nan
    if fault=='moving_base':n._staging_odom['linear_speed']=.005001
    if fault=='moving_arm':n._joint_velocities[name]=.010001
    if fault=='torso':n.joints[serial.IK_JOINTS[0]]+=.002001
    if fault=='head':n._joint_velocities[serial.HEAD[0]]=.001001
    if fault=='odom_reverse':n._staging_odom['stamp_ns']-=1
    with pytest.raises(RuntimeError):owner.check(stationary=True)
    assert not any(e[0]=='send' for e in events)


@pytest.mark.parametrize('freeze',['clock','joints','odom'])
def test_initial_stop_requires_both_advancing_producers_and_is_wall_bounded(monkeypatch,freeze):
    n,clock,_=fixture(monkeypatch);owner=start(n)
    original=clock.hook;stamps=dict(n._joint_stamps_ns);odom=n._staging_odom['stamp_ns']
    if freeze=='clock':clock.sleep=lambda seconds:setattr(clock,'wall',clock.wall+seconds)
    else:
        def update():
            original()
            if freeze=='joints':n._joint_stamps_ns.update(stamps)
            else:n._staging_odom['stamp_ns']=odom
        clock.hook=update
    with pytest.raises((RuntimeError,TimeoutError)):owner.wait_stopped()
    assert clock.wall<13.1


@pytest.mark.parametrize('side',['left','right'])
def test_both_named_official_limits_are_used_before_server_wait(monkeypatch,side):
    n,_,_=fixture(monkeypatch)
    index=0 if side=='left' else 7
    velocities=list(n._initial_stow_serial_limits.velocity);velocities[index]=.1
    n._initial_stow_serial_limits=serial.Limits(tuple(serial.ARM_NAMES),(-4.,)*14,(4.,)*14,tuple(velocities),'a'*64)
    with pytest.raises(RuntimeError,match='80percent'):run(n)
    assert not getattr(n,'arm_client' if side=='left' else 'right_arm_client').goals
    assert bool(n.arm_client.goals) is (side=='right')


@pytest.mark.parametrize('change',['joint','stamp','limit','owner','cancel_clear','fault','target','duration'])
def test_actual_sender_rechecks_under_final_lock_after_server_wait(monkeypatch,change):
    n,_,_=fixture(monkeypatch)
    def wait(**unused):
        if change=='joint':n.joints[serial.ARM_JOINTS[0]]+=.001001
        if change=='stamp':n._joint_stamps_ns[serial.ARM_JOINTS[0]]+=1
        if change=='limit':n._initial_stow_serial_limits=object()
        if change=='owner':n._initial_stow_serial_owner=object()
        if change=='cancel_clear':n._cancel.set();n._cancel.clear()
        if change=='fault':n._raw_contacts_first_failure='wire'
        return True
    n.arm_client.wait_for_server=wait
    if change in ('target','duration'):
        original=serial.Owner.admit_locked
        def mutate(owner,client,goal,token):
            if change=='target':goal.trajectory.points[0].positions[0]+=.001
            else:goal.trajectory.points[0].time_from_start.nanosec=1
            return original(owner,client,goal,token)
        monkeypatch.setattr(serial.Owner,'admit_locked',mutate)
    with pytest.raises(RuntimeError):run(n)
    assert not n.arm_client.goals and not n.right_arm_client.goals


def test_fresh_final_slope_can_reject_when_first_slope_passed(monkeypatch):
    n,_,_=fixture(monkeypatch)
    # Start exactly on the 80% boundary, then move .0005 rad before locked send.
    n.joints[serial.ARM_JOINTS[0]]=serial.HOME[1]-1.6
    n.arm_client.wait_for_server=lambda **kw:(n.joints.__setitem__(serial.ARM_JOINTS[0],serial.HOME[1]-1.6005) or True)
    with pytest.raises(RuntimeError,match='80percent'):run(n)
    assert not n.arm_client.goals


def test_result_success_alone_cannot_start_right_or_mark_parked(monkeypatch):
    n,clock,_=fixture(monkeypatch);n.arm_client.update_endpoint=False
    with pytest.raises(TimeoutError,match='measured stop'):run(n)
    assert n.arm_client.goals and not n.right_arm_client.goals and not n._right_parked
    assert n._initial_stow_serial_owner is not None and n._cancel.is_set()


@pytest.mark.parametrize('phase',['acceptance','terminal'])
def test_unknown_action_is_owned_and_original_watchdogs_remain_bounded(monkeypatch,phase):
    n,clock,events=fixture(monkeypatch)
    n.arm_client.delayed_acceptance=phase=='acceptance'
    n.arm_client.delayed_terminal=phase=='terminal'
    with pytest.raises((TimeoutError,RuntimeError)):run(n)
    owner=n._initial_stow_serial_owner
    assert owner is not None and n._cancel.is_set() and not n.right_arm_client.goals
    if phase=='acceptance':
        assert owner.tokens[0] in n._pending_retained_acceptances
        assert 11.1<=clock.wall<11.3
        n.arm_client.acceptance.set_result(n.arm_client.handle)
        assert ('cancel','left') in events and not n._pending_retained_acceptances
    else:
        assert 25.1<=clock.wall<25.3  # timeout1 +4*original3.5, never4*2.
        assert n.arm_client.handle in n._goal_handles and n._torso_hold_state.uncertain


@pytest.mark.parametrize('failure',['send','get_result','controller_error','late_fault'])
def test_failed_left_action_never_starts_right_and_preserves_interlock(monkeypatch,failure):
    n,clock,events=fixture(monkeypatch)
    original=n.arm_client.send_goal_async
    def send(goal):
        if failure=='send':raise RuntimeError('publication uncertain')
        f=original(goal)
        if failure=='get_result':
            n.arm_client.handle.get_result_async=lambda:(_ for _ in ()).throw(RuntimeError('result unavailable'))
        if failure=='late_fault':n._raw_contacts_first_failure='wire'
        return f
    n.arm_client.send_goal_async=send
    if failure=='controller_error':n.arm_client.error_code=-4
    with pytest.raises(RuntimeError):run(n)
    owner=n._initial_stow_serial_owner
    assert owner is not None and n._cancel.is_set() and not n.right_arm_client.goals
    if failure=='send':assert owner.tokens[0] in n._pending_retained_acceptances
    else:assert ('cancel','left') in events
    assert n._torso_hold_state.uncertain


def test_real_command_receiver_retains_failed_owner_before_cancel_can_clear(monkeypatch):
    n,_,events=fixture(monkeypatch);owner=start(n);owner.cancel_owned();n._busy=False
    runtime.ManipulationNode._on_command(n,NS(data='{"event":"stow"}'))
    assert n._initial_stow_serial_owner is owner and n._cancel.is_set()
    assert events[-1]==('status','rejected',{'command':'stow','reason':'initial_stow_serial_owner_active'})


def test_no_geometry_or_additional_retimer_is_called_by_serial_branch(monkeypatch):
    n,_,_=fixture(monkeypatch)
    monkeypatch.setattr(runtime,'retime_admitted_arm_goal',lambda *a,**k:pytest.fail('PLACE timing entered'))
    monkeypatch.setattr(runtime,'require_arm_velocity_locked',lambda *a,**k:pytest.fail('left-only gate used on serial owner'))
    assert run(n)


@pytest.mark.parametrize('problem',['missing','duplicate','wrong_type','nonfinite','negative'])
def test_official_named_limit_loader_refuses_bad_urdf(tmp_path,problem):
    rows=[]
    for name in serial.ARM_NAMES:
        velocity='nan' if problem=='nonfinite' and name==serial.ARM_NAMES[0] else '-1' if problem=='negative' and name==serial.ARM_NAMES[0] else '1.95'
        kind='continuous' if problem=='wrong_type' and name==serial.ARM_NAMES[0] else 'revolute'
        rows.append(f'<joint name="{name}" type="{kind}"><limit lower="-4" upper="4" velocity="{velocity}"/></joint>')
    if problem=='missing':rows.pop()
    if problem=='duplicate':rows.append(rows[0])
    path=tmp_path/'robot.urdf';path.write_text('<robot>'+''.join(rows)+'</robot>')
    with pytest.raises(ValueError):serial.load_limits(path)


def test_named_limit_loader_preserves_distinct_right_arm_values(tmp_path):
    rows=[f'<joint name="{name}" type="revolute"><limit lower="-4" upper="4" velocity="{1+i/10}"/></joint>'
          for i,name in enumerate(serial.ARM_NAMES)]
    path=tmp_path/'robot.urdf';path.write_text('<robot>'+''.join(reversed(rows))+'</robot>')
    limits=serial.load_limits(path)
    assert limits.names==tuple(serial.ARM_NAMES)
    assert limits.velocity==tuple(1+i/10 for i in range(14))
    assert limits.lower==(-4.,)*14 and limits.upper==(4.,)*14 and len(limits.urdf_sha256)==64


def test_accepted_goal_stays_pending_if_fresh_fault_and_cancellation_result_both_fail(monkeypatch):
    n,_,events=fixture(monkeypatch)
    original=n.arm_client.send_goal_async
    def send(goal):
        acceptance=original(goal)
        n._raw_contacts_first_failure='wire_after_acceptance'
        def failed_cancel():
            events.append(('cancel_failed','left'))
            raise RuntimeError('cancel transport unavailable')
        n.arm_client.handle.cancel_goal_async=failed_cancel
        n.arm_client.handle.get_result_async=lambda:(_ for _ in ()).throw(RuntimeError('result unavailable'))
        return acceptance
    n.arm_client.send_goal_async=send
    with pytest.raises(RuntimeError,match='wire|occupied/fault'):run(n)
    owner=n._initial_stow_serial_owner
    assert owner is not None and owner.handle is n.arm_client.handle and owner.handle.accepted
    assert owner.tokens[0] in n._pending_retained_acceptances
    assert ('cancel_failed','left') in events and n._cancel.is_set()
    assert not n.right_arm_client.goals and not n._right_parked
    assert n._torso_hold_state.uncertain
