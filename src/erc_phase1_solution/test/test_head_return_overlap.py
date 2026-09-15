"""Behavior and ownership tests for the actual isolated helper; no ROS server."""
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
import os
BASE=Path(os.environ['HEAD_RETURN_BASE_SOURCE'])
assert (BASE/'erc_phase1_solution/manipulation_node.py').is_file(), 'Explicit pinned candidate20 source required'
from erc_phase1_solution import head_return_overlap as h

IDENT=dict(trial_id='a'*12,head_return_id='b'*32,target_model='book_col_3_row_2_red')

def node():
    n=NS(now=1_000_000_000,joints={k:0. for k in h.NAMES},
        _joint_velocities={k:0. for k in h.NAMES},_joint_stamps_ns={k:1_000_000_000 for k in h.NAMES},
        _held_book_corners=np.zeros((8,3)),_target_book_model=IDENT['target_model'],
        _contact_epoch=7,_cancel=threading.Event(),_lock=threading.Lock(),
        _payload_hazard_latched=None,_held_grip_sensor_fault=None,_target_robot_contact_latched=False,
        _staging_odom=dict(stamp_ns=1_000_000_000,pose=(0.,0.,0.),linear_speed=0.,angular_speed=0.),
        _goal_handles=[],_pending_retained_acceptances=set(),_head_return_state=None,
        head_return_overlap_enabled=True,timeout=1.,events=[])
    n.joints['head_2_joint']=.2
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=n.now))
    n._payload_hazard_reason=lambda **k:None
    n._publish_status=lambda event,**kw:n.events.append((event,kw))
    n._adaptive_command_guard=nullcontext
    return n

@pytest.mark.parametrize('value',[0,1,'false',None,[],np.bool_(False)])
def test_flag_is_strict(value):
    with pytest.raises(ValueError):h.checked_enabled(value)

@pytest.mark.parametrize('value',[True,False])
def test_flag_preserves_bool(value):assert h.checked_enabled(value) is value

@pytest.mark.parametrize('key,value',[('trial_id','x'),('trial_id',None),('head_return_id','a'*31),('target_model','other')])
def test_identity_rejects_uncorrelated_commands(key,value):
    with pytest.raises(RuntimeError):h.identity({**IDENT,key:value})

def test_frozen_context_is_copy_and_accepts_stationary_fresh_state():
    n=node();r=h.capture(n,stopped=True)
    n.joints[h.IK_JOINTS[1]]=.5;n._held_book_corners[0,0]=1
    assert r['joints'][h.IK_JOINTS[1]]==0 and r['attached'][0,0]==0

@pytest.mark.parametrize('which,change,match',[
    ('stamp',-150_000_001,'stale'),('stamp',50_000_001,'stale'),
    ('position',float('nan'),'stale'),('velocity',float('inf'),'stale'),
    ('velocity',.001,'parked_joint_moving'),('drift',.00001,'parked_joint_drift'),
    ('attachment',1,'carry_identity_changed'),('epoch',1,'carry_identity_changed'),
    ('cancel',1,'interlock'),('target',1,'carry_identity_changed'),
])
def test_context_rejects_actual_change(which,change,match):
    n=node();r=h.capture(n);k=h.IK_JOINTS[1]
    if which=='stamp':n._joint_stamps_ns[k]=n.now+change
    if which=='position':n.joints[k]=change
    if which=='velocity':n._joint_velocities[k]=change
    if which=='drift':n.joints[k]+=change
    if which=='attachment':n._held_book_corners[0,0]+=change
    if which=='epoch':n._contact_epoch+=change
    if which=='cancel':n._cancel.set()
    if which=='target':n._target_book_model='book_col_4_row_2_red'
    with pytest.raises(RuntimeError,match=match):h.capture(n,reference=r,moving_head=True)

def test_only_head_can_move_inside_original_sweep():
    n=node();r=h.capture(n);n.joints['head_2_joint']=-.3;n._joint_velocities['head_2_joint']=.2
    h.capture(n,reference=r,moving_head=True)
    n.joints['head_2_joint']=-.61
    with pytest.raises(RuntimeError,match='checked_interval'):h.capture(n,reference=r,moving_head=True)

def test_presend_head_change_and_poststop_target_distinct():
    n=node();r=h.capture(n);n.joints['head_2_joint']=-.6
    with pytest.raises(RuntimeError,match='parked_joint_drift'):h.capture(n,reference=r,stopped=True)
    h.capture(n,reference=r,target=True,stopped=True)

@pytest.mark.parametrize('field,value',[('linear_speed',.006),('angular_speed',.009),('stamp_ns',0)])
def test_poststop_needs_fresh_measured_stopped_base(field,value):
    n=node();n._staging_odom[field]=value
    with pytest.raises(RuntimeError,match='base_not_measured_stopped'):h.capture(n,stopped=True)

def test_base_frozen_until_acceptance_then_can_navigate():
    n=node();r=h.capture(n,stopped=True);r['base_must_match']=True;n._staging_odom['pose']=(.003,0.,0.)
    with pytest.raises(RuntimeError,match='base_moved_before_head_acceptance'):h.capture(n,reference=r,stopped=True)
    r['base_must_match']=False;h.capture(n,reference=r,stopped=True)

def geometry_node():
    n=node();n.head_chain=NS(lower=np.array([0.,-1.,-1.]),upper=np.array([.35,1.,1.]))
    n.chain=NS(forward=lambda q:np.eye(4));n.carried_collision_meshes=[]
    n._held_book_corners[:]=[.4,0.,1.]
    n._collision_link_transforms=lambda *a,**k:{}
    n.calls=[];n._robot_self_collision=lambda *a,**k:None;n._carried_robot_collision=lambda *a,**k:None
    n._head_motion_collision=lambda q,head,corners,**kw:n.calls.append(head.copy())
    n.carried_transition_samples=61;n.carried_navigation_radius_limit=.45
    return n

def test_actual_geometry_visits_full_grid_and_same_radius():
    n=geometry_node();r=h.admit_sweep(n,h.capture(n))
    assert len(n.calls)==61 and r['baseline_radius_m']==r['maximum_radius_m']==.4
    np.testing.assert_array_equal(n.calls[0],[0.,.2]);np.testing.assert_allclose(n.calls[-1],h.TARGET)

@pytest.mark.parametrize('failure',['body','payload','sweep','radius','limit'])
def test_geometry_rejection_before_action(monkeypatch,failure):
    n=geometry_node()
    if failure=='body':n._robot_self_collision=lambda *a,**k:'hit'
    if failure=='payload':n._carried_robot_collision=lambda *a,**k:'hit'
    if failure=='sweep':n._head_motion_collision=lambda *a,**k:'hit'
    if failure=='radius':monkeypatch.setattr(h,'planar_radius',lambda *a,**k:.4+max(0.,-float(k['head'][1])))
    if failure=='limit':n.head_chain.lower[2]=-.5
    with pytest.raises(RuntimeError):h.admit_sweep(n,h.capture(n))
    assert n._goal_handles==[] and not n._pending_retained_acceptances

def manager():
    m=NS(trial_id=IDENT['trial_id'],target_book_model=IDENT['target_model'],events=[],nav='none',
        _head_return_request=None,manipulation_timeout=10.,navigation_timeout=20.,elapsed=0.,now=1_000_000_000)
    m._log=lambda *a,**k:None;m._perception_mode=lambda mode:m.events.append(('mode',mode))
    m._manipulate=lambda cmd,**kw:m.events.append(('manip',cmd,kw))
    m._navigate=lambda *args:m.events.append(('nav',args))
    m._set_state=lambda state:setattr(m,'state',state)
    m._abort=lambda reason:m.events.append(('abort',reason))
    m._elapsed_state=lambda:m.elapsed;m._nav_failed=lambda:m.nav=='failed';m._nav_reached=lambda:m.nav=='reached'
    m.get_clock=lambda:NS(now=lambda:NS(nanoseconds=m.now))
    h.begin_mission(m,(1.,2.,3.));return m

def status(m,event,**kw):return h.mission_status(m,dict(command=h.COMMAND,event=event,**{**m._head_return_request['identity'],**kw}))

@pytest.mark.parametrize('head_first',[True,False])
def test_two_results_both_required_and_nav_waits_acceptance(head_first):
    m=manager();h.tick_mission(m);assert not any(e[0]=='nav' for e in m.events)
    status(m,'head_return_accepted');h.tick_mission(m);h.tick_mission(m)
    assert sum(e[0]=='nav' for e in m.events)==1
    if head_first:status(m,'succeeded')
    else:m.nav='reached'
    h.tick_mission(m);assert m.state=='RETURN_START'
    status(m,'succeeded');m.nav='reached';h.tick_mission(m)
    assert m.state=='HEAD_BIN' and m.events[-1][1]=='look_bin'
    assert m.events[-1][2]['head_return_id']==m._head_return_request['identity']['head_return_id']

def test_stale_success_cannot_complete_current_request():
    m=manager();status(m,'succeeded',head_return_id='c'*32);assert not m._head_return_request['succeeded']
    assert h.mission_status(m,{'command':'look_bin','event':'succeeded'}) is False

@pytest.mark.parametrize('failure',['head','nav','timeout'])
def test_any_terminal_failure_aborts(failure):
    m=manager();status(m,'head_return_accepted');h.tick_mission(m)
    if failure=='head':status(m,'failed',reason='lost')
    elif failure=='nav':m.nav='failed'
    else:m.elapsed=21
    h.tick_mission(m);assert m.events[-1][0]=='abort'

def completed_node():
    n=node();r=h.capture(n);n.joints.update(dict(zip(h.HEAD,h.TARGET)))
    n._head_return_state=dict(identity=IDENT.copy(),reference=r,phase='completed',fault=None)
    n._move_head=lambda *a:n.events.append(('ordinary_head',a)) or True
    return n

def test_poststop_preserves_ordinary_hold_and_clears_owner():
    n=completed_node();assert h.finish(n,IDENT);assert n.events==[('ordinary_head',h.TARGET)] and n._head_return_state is None

@pytest.mark.parametrize('failure',['identity','goal','pending','movingbase','headposition','carrydrift'])
def test_poststop_never_skips_admission(failure):
    n=completed_node();payload=IDENT.copy()
    if failure=='identity':payload['head_return_id']='c'*32
    if failure=='goal':n._goal_handles.append(object())
    if failure=='pending':n._pending_retained_acceptances.add(object())
    if failure=='movingbase':n._staging_odom['linear_speed']=.1
    if failure=='headposition':n.joints[h.HEAD[1]]=-.5
    if failure=='carrydrift':n.joints[h.IK_JOINTS[1]]=.001
    with pytest.raises(RuntimeError):h.finish(n,payload)
    assert not n.events and n._head_return_state is not None

def image(stamp):return NS(header=NS(stamp=NS(sec=stamp//10**9,nanosec=stamp%10**9)))

def camera():
    n=node();n.mode='bin';n.bin_history=[1];n.bin_rgb_frames=[1];n.bin_depth_frames=[1]
    n.bin_tracker=NS(reset=lambda:n.events.append('reset'))
    n.latest_rgb_message=image(n.now+1);n.latest_depth_message=image(n.now+1)
    return n

def test_fresh_camera_epoch_clears_history():
    n=camera();h.camera_epoch(n,dict(event='bin',head_return_not_before_ns=n.now))
    assert not n.bin_history and not n.bin_rgb_frames and not n.bin_depth_frames and n.events==['reset']


def actual_bin_methods():
    path=ROOT/'erc_phase1_solution/perception_node.py'
    assert path.is_file()
    parsed=ast.parse(path.read_text(encoding='utf-8'))
    methods=[n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef) and n.name in ('_process','_process_bin')]
    for method in methods:method.decorator_list=[]
    env=dict(stamp_to_nanoseconds=lambda stamp:stamp.sec*10**9+stamp.nanosec,
        bin_stamp_is_fresh=lambda stamp,now:0<=now-stamp<=2_000_000_000,
        BIN_MAX_SKEW_NS=100_000_000,Time=NS(from_msg=lambda stamp:stamp),
        camera_transform=lambda *args:np.eye(4),TransformException=ValueError,
        red_bin_candidates=lambda rgb:[])
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'),env)
    return env['_process'],env['_process_bin']


@pytest.mark.parametrize('rgb_offset,depth_offset,admitted',[
    (-1,-1,False),(-1,1,False),(1,-1,False),(0,1,False),(1,0,False),(1,1,True)])
def test_actual_bin_dispatch_rejects_queued_preboundary_pairs(rgb_offset,depth_offset,admitted):
    process,bin_process=actual_bin_methods();n=camera()
    h.camera_epoch(n,dict(event='bin',head_return_not_before_ns=n.now))
    boundary=n.now;n.now+=50_000_000;n.camera_info=NS(k=[])
    n.bin_rgb_frames=[(image(boundary+rgb_offset),np.zeros((1,1,3)))]
    n.bin_depth_frames=[(image(boundary+depth_offset),np.zeros((1,1)))]
    for frames in (n.bin_rgb_frames,n.bin_depth_frames):frames[0][0].header.frame_id='camera'
    # Latest frames are fresh in every case, including rejected buffered pairs.
    n.latest_rgb_message=image(n.now);n.latest_depth_message=image(n.now)
    n.last_bin_consumed_ns=n.last_bin_depth_ns=n.last_bin_observed_ns=-1
    n.tf_calls=[]
    n.tf_buffer=NS(lookup_transform=lambda *args:n.tf_calls.append(args) or NS(transform=NS(translation=None,rotation=None)))
    n._invalidate_bin=lambda *args:n.events.append(('invalid',args))
    n._ready=lambda:(_ for _ in ()).throw(AssertionError('bin must use actual special branch'))
    n._process_bin=lambda:bin_process(n)
    process(n)
    assert bool(n.tf_calls) is admitted
    assert n.last_bin_consumed_ns==(boundary+rgb_offset if admitted else -1)
    assert n.last_bin_depth_ns==(boundary+depth_offset if admitted else -1)


def test_no_epoch_preserves_original_bin_selection():
    process,bin_process=actual_bin_methods();n=camera();n.camera_info=NS(k=[])
    n.bin_rgb_frames=[(image(n.now-2),np.zeros((1,1,3)))];n.bin_depth_frames=[(image(n.now-1),np.zeros((1,1)))]
    n.bin_depth_frames[0][0].header.frame_id='camera'
    n.last_bin_consumed_ns=n.last_bin_depth_ns=n.last_bin_observed_ns=-1
    n.tf_buffer=NS(lookup_transform=lambda *args:NS(transform=NS(translation=None,rotation=None)))
    n._invalidate_bin=lambda *args:None;n._process_bin=lambda:bin_process(n)
    process(n);assert n.last_bin_consumed_ns==n.now-2 and n.last_bin_depth_ns==n.now-1


@pytest.mark.parametrize('value',[True,1.0,0,-1,1,1_051_000_000])
def test_epoch_rejects_old_future_or_wrong_type(value):
    n=camera()
    with pytest.raises(RuntimeError):h.camera_epoch(n,dict(event='bin',head_return_not_before_ns=value))
    assert n.bin_history==[1]

def test_epoch_only_after_dual_stop_boundary():
    m=manager();fields={};h.bin_epoch(m,'bin',fields);assert not fields
    m._head_return_request['poststop']=True;h.bin_epoch(m,'bin',fields)
    assert fields['head_return_not_before_ns']==m.now==m.bin_invalidated_ns

class Future:
    def __init__(self,value=None,done=True):self.value=value;self.ready=done;self.callbacks=[]
    def done(self):return self.ready
    def result(self):return self.value
    def add_done_callback(self,callback):self.callbacks.append(callback)

def fake_actions(monkeypatch,n,*,acceptance=True,terminal=True,error=0):
    for name in ('control_msgs.action','trajectory_msgs.msg','rclpy.duration','action_msgs.msg'):
        monkeypatch.setitem(sys.modules,name,ModuleType(name))
    sys.modules['control_msgs.action'].FollowJointTrajectory=NS(Goal=lambda:NS(trajectory=NS()))
    sys.modules['trajectory_msgs.msg'].JointTrajectoryPoint=lambda:NS()
    sys.modules['rclpy.duration'].Duration=lambda seconds:NS(to_msg=lambda:seconds)
    sys.modules['action_msgs.msg'].GoalStatus=NS(STATUS_SUCCEEDED=4)
    terminal_future=Future(NS(status=4,result=NS(error_code=error)),done=terminal)
    handle=NS(accepted=acceptance,get_result_async=lambda:terminal_future)
    accepted=Future(handle)
    n.sent=[];n.head_client=NS(wait_for_server=lambda **k:True,send_goal_async=lambda goal:n.sent.append(goal) or accepted)
    n._valid_retained_terminal_result=lambda result:result.status in (4,5,6)
    n._cancel_retained_goal_and_confirm=lambda *a:False
    n._cancel_late_retained_goal=lambda f,t:n.events.append(('late_cancel',t))
    t=[0.]
    monkeypatch.setattr(h.time,'monotonic',lambda:t[0])
    def tick(seconds):
        t[0]+=seconds;n.now+=20_000_000
        n._joint_stamps_ns={k:n.now for k in h.NAMES};n._staging_odom['stamp_ns']=n.now
        n.joints.update(dict(zip(h.HEAD,h.TARGET)))
    monkeypatch.setattr(h.time,'sleep',tick)
    monkeypatch.setattr(h,'admit_sweep',lambda *a:dict(samples=61))
    return accepted,terminal_future,handle

def test_action_success_requires_actual_new_goal_then_fresh_measured_sample(monkeypatch):
    n=node();fake_actions(monkeypatch,n)
    assert h.run(n,IDENT)
    assert n.sent[0].trajectory.joint_names==list(h.HEAD)
    point=n.sent[0].trajectory.points[0]
    assert point.positions==list(h.TARGET) and point.time_from_start==1.2
    assert n._head_return_state['phase']=='completed' and not n._goal_handles and not n._pending_retained_acceptances
    assert n._last_completed_head_target[0]==h.TARGET
    assert n._last_completed_head_target[1]<n._joint_stamps_ns[h.HEAD[0]]
    assert [e[0] for e in n.events]==['head_return_accepted','head_return_measured']

def test_unknown_acceptance_retains_token_and_registers_late_cancellation(monkeypatch):
    n=node();acceptance,_,_=fake_actions(monkeypatch,n);acceptance.ready=False
    with pytest.raises(TimeoutError,match='acceptance_timeout'):h.run(n,IDENT)
    assert n._cancel.is_set() and len(n._pending_retained_acceptances)==1 and len(acceptance.callbacks)==1
    acceptance.ready=True;acceptance.callbacks[0](acceptance)
    assert n.events[0][0]=='late_cancel' and not any(e[0]=='head_return_accepted' for e in n.events)

def test_rejected_goal_removes_pending_but_never_publishes_acceptance(monkeypatch):
    n=node();fake_actions(monkeypatch,n,acceptance=False)
    with pytest.raises(RuntimeError,match='goal_rejected'):h.run(n,IDENT)
    assert not n._pending_retained_acceptances and not n._goal_handles and n._cancel.is_set() and not n.events

def test_unconfirmed_cancellation_keeps_handle_interlock(monkeypatch):
    n=node();_,_,handle=fake_actions(monkeypatch,n,terminal=False)
    with pytest.raises(TimeoutError,match='motion_timeout'):h.run(n,IDENT)
    assert n._goal_handles==[handle] and n._cancel.is_set()

def test_nonzero_controller_error_is_failure_despite_terminal_success(monkeypatch):
    n=node();fake_actions(monkeypatch,n,error=-4)
    with pytest.raises(RuntimeError,match='trajectory_failed'):h.run(n,IDENT)
    assert not n._goal_handles and n._cancel.is_set()
    assert getattr(n,'_last_completed_head_target',None) is None

def test_monitor_latches_context_hazard_once():
    n=completed_node();n.joints[h.IK_JOINTS[1]]=.01
    n.stock_gripper_close_diagnostic_enabled=True;n.holds=[]
    def hold(reason):
        assert n._lock.acquire(blocking=False), 'gripper hold must occur outside sensor lock'
        n._lock.release();n.holds.append(reason)
    n._hold_adaptive_gripper=hold
    h.monitor(n);h.monitor(n)
    assert len(n.holds)==1 and n.holds[0].startswith('head_return_context:')
    assert n._cancel.is_set() and n._head_return_state['fault'] and len(n.events)==1
    assert n.events[0][0]=='payload_hazard' and n.events[0][1]['head_return_id']==IDENT['head_return_id']

def test_original_motion_functions_unchanged_in_overlay():
    assert (BASE/'erc_phase1_solution/manipulation_node.py').is_file()
    def functions(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    expected=(BASE/'erc_phase1_solution/manipulation_node.py').read_text(encoding='utf-8')
    # Only reviewed raised-finish and clearance _place call sites differ from the explicit parent.
    changes=[("                raise RuntimeError('placement_release_unverified')\n            if not self._return_from_bin(solutions, carried_transition_waypoints,\n", "                raise RuntimeError('placement_release_unverified')\n            finish_options, measured_return_goal = raised_place_normal_finish(\n                self, correlation, torso_ready, direct_empty_home, HOME)\n            if not self._return_from_bin(solutions, carried_transition_waypoints,\n"), ('                       if direct_empty_home is not None else {}), **arm_timing_options):\n                return False\n            return observe_measured_open_pose(self, correlation, HOME,\n', '                       if direct_empty_home is not None else {}), **arm_timing_options, **finish_options):\n                return False\n            return observe_measured_open_pose(self, correlation, measured_return_goal,\n'), ('        approach_ok, next_leg, contact_lost = self._execute_retained_arm_legs(\n', "        if getattr(self, 'bin_clearance_timing_enabled', False):\n            loaded_arm_timing_options.update(bin_clearance_normal_options(\n                self, correlation, direct_empty_home, approach_legs))\n        approach_ok, next_leg, contact_lost = self._execute_retained_arm_legs(\n")]
    for old,new in changes:
        assert expected.count(old)==1
        expected=expected.replace(old,new,1)
    # The reviewed empty planning scope is the only additional PICK delta.
    empty_old,empty_new=("            empty_elevated[0] = self.pick_torso_height\n            if not empty_setup_guard.opening() or not empty_setup_guard.edge(\n                empty_setup_guard.start, empty_elevated,\n            ):\n                raise RuntimeError(f'Empty pickup opening/torso rejected: {empty_setup_guard.last_rejection}')\n            empty_setup_options = dict(\n                transition_start=empty_elevated,\n                transition_edge_validator=empty_setup_guard.retracted_edge,\n                setup_transition_planner=empty_setup_guard.plan_transition,\n                candidate_validator=empty_setup_guard.candidate,\n            )\n        solutions, orientation_index, path_score, transition_waypoints = (\n            self._solve_cartesian_path(\n                positions,\n                rotations,\n                self.pick_torso_height,\n                endpoint_first=top_row,\n                position_tolerance=pick_position_tolerance,\n                orientation_tolerance=pick_orientation_tolerance,\n                **empty_setup_options,\n            )\n        )\n", "            empty_elevated[0] = self.pick_torso_height\n            empty_setup_options = dict(\n                transition_start=empty_elevated,\n                transition_edge_validator=empty_setup_guard.retracted_edge,\n                setup_transition_planner=empty_setup_guard.plan_transition,\n                candidate_validator=empty_setup_guard.candidate,\n            )\n        # This empty-only epoch is completely closed before any loaded pool\n        # or motion. Disabled/custom paths keep their ordinary serial checker.\n        with empty_pickup_geometry_scope(self, empty_setup_guard):\n            if empty_setup_guard is not None:\n                if not empty_setup_guard.opening() or not empty_setup_guard.edge(\n                    empty_setup_guard.start, empty_elevated,\n                ):\n                    raise RuntimeError(f'Empty pickup opening/torso rejected: {empty_setup_guard.last_rejection}')\n            solutions, orientation_index, path_score, transition_waypoints = (\n                self._solve_cartesian_path(\n                    positions,\n                    rotations,\n                    self.pick_torso_height,\n                    endpoint_first=top_row,\n                    position_tolerance=pick_position_tolerance,\n                    orientation_tolerance=pick_orientation_tolerance,\n                    **empty_setup_options,\n                )\n            )\n")
    assert expected.count(empty_old)==1
    expected=expected.replace(empty_old,empty_new,1)
    before=functions(expected)
    import hashlib,json
    text=(ROOT/'erc_phase1_solution/manipulation_node.py').read_text(encoding='utf-8')
    from candidate_composition_support import restore_all_books_source
    text=restore_all_books_source(text)
    root=ROOT
    head = json.loads((root/'test/fixtures/empty_head_timing_inverse.json').read_text())
    for fragment in reversed(head['node_fragments']):
        assert text.count(fragment['new']) == 1
        text = text.replace(fragment['new'], fragment['old'])
    assert hashlib.sha256(text.encode()).hexdigest() == head['parent_node_sha256']
    after=functions(text)
    for name in ('_follow','_stow','_pick','_compact_transport','_move_head','_place','_cancel_late_retained_goal'):
        assert before[name]==after[name],name
