"""Actual planner orchestration/capability boundaries; synthetic geometry only."""
import ast
import copy
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import release_only_place_planning as policy
from erc_phase1_solution import scene_checked_place as planner
from erc_phase1_solution import release_pose_finish as finish
from erc_phase1_solution.place_contact_guard import PlaceContactGuard
from test_scene_checked_place_planner import planning


IDENTITY=dict(trial_id='trial',placement_attempt_id='attempt',target_model='book_col_3_row_2_red')
from erc_phase1_solution.release_pose_evidence import CONTRACT
PAYLOAD=dict(IDENTITY,completion_mode='release_pose',release_evidence_contract=CONTRACT)
ROOT=Path(__file__).resolve().parents[1]


def request_node(monkeypatch, node=None):
    n=node or NS()
    n._lock=threading.RLock();n._command_lock=threading.RLock()
    n._adaptive_command_guard=lambda:n._command_lock
    n._cancel=threading.Event();n._busy=True;n._target_book_model=IDENTITY['target_model']
    n._held_book_corners=np.zeros((8,3));n._goal_handles=[];n._pending_retained_acceptances=[]
    n._release_pose_owner=None;n._release_pose_fault_latched=None
    n.release_only_place_planning_enabled=True;n.place_finish_at_release_enabled=True
    n.delivery_evidence_enabled=True;n.book_centered_place_enabled=True
    n.table_scene_required=True;n.bin_scene_required=True;n.bin_clearance_timing_enabled=False
    n.gripper_open=.06;n.chain=object();n.right_chain=object();n.head_chain=object()
    n._active_place_scene_reference=dict(base_pose=[0.,0.,0.],parked_joints={})
    n._selected_place_scene_reference=n._active_place_scene_reference
    n._selected_place_bin_scene=dict(mesh_sha256='synthetic-bin')
    n._selected_place_table_scene=dict(valid=True)
    n._place_contact_guard=PlaceContactGuard(10,dict(IDENTITY))
    n.context_checks=0
    def context(node, reference):
        assert node is n and reference is n._active_place_scene_reference
        n.context_checks+=1
    monkeypatch.setattr(policy,'measured_scene_context',context)
    monkeypatch.setattr(finish,'measured_scene_context',context)
    return n


@pytest.fixture
def selected_planning(planning,monkeypatch,tmp_path):
    n,events,ordinary=planning
    request_node(monkeypatch,n)
    # The original fixture's geometry checks remain explicit recording stubs;
    # asset admission stubs enable the registered planner branch, not a claim
    # about actual meshes. No original planner statement is replaced.
    for relative in ('models/table/meshes/erc_base_table.STL','models/table/sdf/erc_table.sdf'):
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(relative)
    original_obstacle=planner.NominalBinObstacle
    def obstacle(*a,bin_scene=None,**kw):
        value=original_obstacle(*a,**kw);value.registered_scene=bin_scene;return value
    monkeypatch.setattr(planner,'NominalBinObstacle',obstacle)
    monkeypatch.setattr(planner,'validate_bin_scene',lambda v:v)
    monkeypatch.setattr(planner,'TableSceneObstacle',lambda scene:object())
    def hashed(data):
        value=(planner.TABLE_MESH_SHA256 if data==b'models/table/meshes/erc_base_table.STL'
               else '90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f'
               if data==b'models/table/sdf/erc_table.sdf' else 'synthetic-bin')
        return NS(hexdigest=lambda:value)
    monkeypatch.setattr(planner,'hashlib',NS(sha256=hashed))
    def solve(node,positions,rotation,height,seed,scene,candidate,**kw):
        return node._solve_cartesian_path(positions,(rotation,),height,transition_start=seed,
            skip_setup_transition=True,candidate_validator=candidate,first_valid=True,joint_limit_margin=.10)
    monkeypatch.setattr(planner,'solve_scene_cartesian',solve)
    base_scene=planner.PlaceSceneChecker
    class Scene(base_scene):
        def sample(self,q,master,loaded):
            events.append(('static_sample' if not loaded else 'loaded_sample',master,q.copy()))
            if not loaded:return not getattr(n,'reject_static',False)
            return super().sample(q,master,loaded)
    monkeypatch.setattr(planner,'PlaceSceneChecker',Scene)
    request=policy.capture(n,PAYLOAD)
    start=np.asarray([.35,0.,0.,0.,0.,0.,0.,0.]);torso=start.copy();torso[0]=.30
    def call(selected=True):
        return planner.plan_scene_checked_place(n,[[0,0,0]],np.eye(3),start,torso,np.ones(8),
            [.8,0,.75],lambda p:tmp_path,table_scene=n._selected_place_table_scene,
            bin_scene=n._selected_place_bin_scene,
            **({'release_only_request':request} if selected else {}))
    return n,events,request,call


def test_actual_planner_omits_only_return_work_and_keeps_open_static_endpoint(selected_planning):
    n,events,request,call=selected_planning
    plan=call()
    assert not any(e[0]=='unloaded_home' for e in events)
    assert not any(e[0]=='scene_leg' and e[2] is False for e in events)
    assert sum(e[0]=='static_sample' for e in events)==1
    opening=[e[1] for e in events if e[0]=='opening']
    assert opening[0]==.017 and opening[-1]==n.gripper_open
    assert max(np.diff(opening))<=.0010000000001
    assert any(e[0]=='loaded' for e in events) and any(e[0]=='support' for e in events)
    assert any(e[0]=='controller_route' for e in events)
    assert plan.empty_return_from_clearance is None
    assert type(plan.unloaded_home) is policy.OmittedReturn
    assert plan.diagnostics['open_return_legs']==0
    assert plan.diagnostics['static_open_hand_endpoint_checked'] is True
    assert policy.require_plan(request,plan,n,IDENTITY) is plan.release_only_endpoint


def test_actual_default_planner_keeps_complete_return_and_no_capability(selected_planning):
    n,events,request,call=selected_planning
    n.release_only_place_planning_enabled=False
    plan=call(False)
    assert any(e[0]=='unloaded_home' for e in events)
    assert any(e[0]=='scene_leg' and e[2] is False for e in events)
    assert plan.empty_return_from_clearance==[]
    assert plan.release_only_endpoint is None and type(plan.unloaded_home) is list
    assert 'unused_empty_return_omitted' not in plan.diagnostics


@pytest.mark.parametrize('failure',['torso','setup','cartesian','opening','static'])
def test_each_retained_geometry_stage_can_reject(selected_planning,failure):
    n,events,request,call=selected_planning
    if failure=='torso':n._carried_robot_transition_is_safe=lambda *a:False
    if failure=='setup':n._plan_carried_joint_route=lambda *a,**kw:None
    if failure=='cartesian':
        count=[0]
        def volume(*a):count[0]+=1;return count[0]==1
        n._carried_robot_transition_is_safe=volume
    if failure=='opening':n.reject_opening=True
    if failure=='static':n.reject_static=True
    with pytest.raises(RuntimeError):call()
    assert not any(e[0]=='unloaded_home' for e in events)


def test_changed_guard_after_successful_search_discards_plan(selected_planning):
    n,events,request,call=selected_planning;original=n._solve_cartesian_path
    def changed(*a,**kw):
        result=original(*a,**kw);n._place_contact_guard=PlaceContactGuard(11,dict(IDENTITY));return result
    n._solve_cartesian_path=changed
    with pytest.raises(RuntimeError,match='attempt_scene_or_policy_changed'):call()


@pytest.mark.parametrize('fault',['cancel','policy','finish','delivery','centered','table_required',
    'bin_required','clearance_timing','busy','target','scene_identity','scene_value','bin_identity',
    'table_identity','guard_identity','guard_epoch','guard_fault','held_identity','held_value','chain','open_position'])
def test_request_invalidation_rejects_before_endpoint(monkeypatch,fault):
    n=request_node(monkeypatch);request=policy.capture(n,PAYLOAD)
    if fault=='cancel':n._cancel.set()
    if fault=='policy':n.release_only_place_planning_enabled=False
    if fault=='finish':n.place_finish_at_release_enabled=False
    if fault=='delivery':n.delivery_evidence_enabled=False
    if fault=='centered':n.book_centered_place_enabled=False
    if fault=='table_required':n.table_scene_required=False
    if fault=='bin_required':n.bin_scene_required=False
    if fault=='clearance_timing':n.bin_clearance_timing_enabled=True
    if fault=='busy':n._busy=False
    if fault=='target':n._target_book_model='book_col_1_row_2_red'
    if fault=='scene_identity':n._active_place_scene_reference=copy.deepcopy(n._active_place_scene_reference)
    if fault=='scene_value':n._active_place_scene_reference['base_pose'][0]=.1
    if fault=='bin_identity':n._selected_place_bin_scene=copy.deepcopy(n._selected_place_bin_scene)
    if fault=='table_identity':n._selected_place_table_scene=copy.deepcopy(n._selected_place_table_scene)
    if fault=='guard_identity':n._place_contact_guard=PlaceContactGuard(10,dict(IDENTITY))
    if fault=='guard_epoch':n._place_contact_guard.started_ns=11
    if fault=='guard_fault':n._place_contact_guard.fault={'reason':'contact'}
    if fault=='held_identity':n._held_book_corners=n._held_book_corners.copy()
    if fault=='held_value':n._held_book_corners[0,0]=1.
    if fault=='chain':n.chain=object()
    if fault=='open_position':n.gripper_open=.069
    scene=NS(sample=lambda *a:pytest.fail('static geometry reached after invalidation'))
    with pytest.raises(RuntimeError):request.endpoint(np.zeros(8),scene)


def method(name):
    source=(ROOT/'erc_phase1_solution/manipulation_node.py').read_text()
    cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)
    scope=dict(__package__='erc_phase1_solution',
               __name__='erc_phase1_solution.manipulation_node',np=np)
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(ROOT), 'exec'),scope)
    return scope[name]


@pytest.mark.parametrize('name',['_return_from_bin','_execute_unloaded_home','_recover_closed_place'])
def test_actual_legacy_helpers_reject_omitted_marker_before_any_actuator(monkeypatch,name):
    n=request_node(monkeypatch);marker=policy.OmittedReturn(policy.capture(n,PAYLOAD))
    n._open_gripper=n._move_arm_solution=n._move_torso=lambda *a,**k:pytest.fail('actuator touched')
    fn=method(name)
    with pytest.raises(RuntimeError,match='no_return_or_recovery_route'):
        if name=='_execute_unloaded_home':fn(n,marker)
        elif name=='_return_from_bin':fn(n,[np.zeros(8)],[],np.zeros(8),marker)
        else:fn(n,cause='release_failed',remaining_approach_legs=[],cartesian_solutions=[np.zeros(8)],
                carried_transition_waypoints=[],carried_start=np.zeros(8),unloaded_home_waypoints=marker)


def test_actual_finish_prepare_accepts_only_checked_endpoint_and_retains_owner(selected_planning):
    n,events,request,call=selected_planning;plan=call()
    proof=policy.require_plan(request,plan,n,IDENTITY)
    owner=finish.prepare(n,IDENTITY,plan.solutions[-1],None,release_only_endpoint=proof)
    assert owner is n._release_pose_owner
    assert owner.binding.goal==tuple(plan.solutions[-1])
    assert owner.binding.guard is request.guard
    assert owner.phase=='opening' and owner.evidence.release.open_epoch is None


@pytest.mark.parametrize('kind',['missing','wrong_goal','wrong_attempt','legacy_direct_list','disabled_after_plan'])
def test_incompatible_plan_cannot_enter_finish(selected_planning,kind):
    n,events,request,call=selected_planning;plan=call();proof=plan.release_only_endpoint
    correlation=dict(IDENTITY);goal=plan.solutions[-1].copy();direct=None
    if kind=='missing':proof=object()
    if kind=='wrong_goal':goal[-1]+=.01
    if kind=='wrong_attempt':correlation['placement_attempt_id']='wrong'
    if kind=='legacy_direct_list':direct=[]
    if kind=='disabled_after_plan':n.place_finish_at_release_enabled=False
    with pytest.raises(RuntimeError):finish.prepare(n,correlation,goal,direct,release_only_endpoint=proof)
    assert n._release_pose_owner is None


def test_default_capture_is_inert_and_flag_strict():
    assert policy.capture(NS(),None) is None
    with pytest.raises(ValueError):policy.checked_enabled(1)
    with pytest.raises(TypeError):list(policy.OmittedReturn(None))


@pytest.mark.parametrize('fault',[None,'after_plan','after_loaded_motion'])
def test_actual_place_wiring_requires_capability_before_motion_and_open(monkeypatch,fault):
    from test_scene_checked_place_flow import PlaceWiring
    from erc_phase1_solution.release_evidence import AttemptIdentity
    from erc_phase1_solution.arm_trajectory_timing import checked_arm_speed_scale
    wiring=PlaceWiring();wiring.setUp();n=wiring.node
    original_corners=n._held_book_corners.copy()
    request_node(monkeypatch,n);n._held_book_corners=original_corners
    # Original upright CAD convention: local X left, Y up, Z forward.
    n._selected_place_bin_scene['rotation']=[[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]
    n._selected_place_table_scene['valid']=True
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=100))
    n.placement_transport_speed_scale=3.;n.loaded_place_speed_scale_cap=1.25
    scope=wiring.namespace
    scope.update(AttemptIdentity=AttemptIdentity,release_only_place_planning=policy,
                 checked_arm_speed_scale=checked_arm_speed_scale)
    calls=[]
    def actual_planner(node,positions,rotation,carried,torso,seed,point,resolver,**kw):
        request=kw['release_only_request'];request.require()
        q=torso.copy();q[1]=.6;goal=q.copy();goal[1]=.7
        proof=request.endpoint(goal,NS(sample=lambda *a:True))
        plan=planner.SceneCheckedPlacePlan([q,goal],0,1.,[q],policy.OmittedReturn(request),{},
            release_only_endpoint=proof)
        calls.append('plan')
        if fault=='after_plan':node._place_contact_guard.fault={'reason':'contact'}
        return plan
    scope['plan_scene_checked_place']=actual_planner
    real_prepare=finish.prepare
    def prepare(*a,**kw):
        calls.append('prepare');assert type(kw['release_only_endpoint']) is policy.Endpoint
        return real_prepare(*a,**kw)
    scope['release_pose_finish']=NS(validate_mode=finish.validate_mode,prepare=prepare,
        remember_open=lambda *a:True,finish=lambda *a:calls.append('finish') or True)
    scope['observe_measured_open_pose']=lambda *a:dict(verified=True)
    n._move_torso=lambda *a,**kw:calls.append('torso') or True
    def loaded(legs,command,**kw):
        calls.append('loaded');assert kw['arm_speed_scale']==1.25
        if fault=='after_loaded_motion':n.place_finish_at_release_enabled=False
        return True,len(legs),False
    n._execute_retained_arm_legs=loaded
    n._open_gripper=lambda **kw:calls.append('open') or kw['verify_measurement']()
    n._return_from_bin=lambda *a,**kw:pytest.fail('legacy return used')
    n._recover_closed_place=lambda **kw:pytest.fail('legacy recovery used')
    if fault is None:
        assert scope['_place'](n,PAYLOAD)
        assert calls==['plan','torso','loaded','prepare','open','finish']
    else:
        with pytest.raises(RuntimeError):scope['_place'](n,PAYLOAD)
        assert 'open' not in calls
        assert ('torso' in calls) is (fault=='after_loaded_motion')


@pytest.mark.parametrize('loader',['optional','empty_endpoint','direct_return','raised'])
def test_existing_ast_loaders_reach_actual_omitted_return_refusal(loader):
    if loader=='optional':
        from test_optional_arm_timing import load
        execute=load('_execute_unloaded_home')
    elif loader=='empty_endpoint':
        from test_empty_pickup_endpoint_wait import ReturnWiringTests
        execute,_=ReturnWiringTests().method()
    elif loader=='direct_return':
        from test_scene_direct_empty_return import actual_return_methods
        execute=actual_return_methods()['_execute_unloaded_home']
    else:
        from test_raised_place_finish import method as raised_method
        execute=raised_method('_execute_unloaded_home')
    def forbidden(*a,**kw):pytest.fail('actuator reached before omitted-route refusal')
    node=NS(_follow=forbidden,_move_torso=forbidden,_move_arm_solution=forbidden)
    with pytest.raises(RuntimeError,match='no_return_or_recovery_route'):
        execute(node,policy.OmittedReturn(None))


def test_actual_contiguous_release_suffix_rejects_endpoint_without_owner_before_open():
    from test_raised_place_finish import normal_suffix,node as raised_node,IDENTITY as raised_identity
    node=raised_node();node.place_finish_at_release_enabled=False
    def forbidden(*a,**kw):pytest.fail('release or return reached without owner')
    node._open_gripper=node._return_from_bin=node._recover_closed_place=forbidden
    tail=normal_suffix(forbidden)
    with pytest.raises(RuntimeError,match='cannot enter legacy release/return'):
        tail(node,raised_identity,[np.zeros(8)],[],np.zeros(8),[],[],{},
             release_only_endpoint=object())
