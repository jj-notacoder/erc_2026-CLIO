"""Actual optional helper and actual method statements; no controller dispatch."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType, MethodType

import numpy as np
import pytest

from erc_phase1_solution import raised_place_finish as finish
from erc_phase1_solution.motion_profiles import HOME, ARM_JOINTS, IK_JOINTS
from erc_phase1_solution.release_evidence import measured_pose_is_open
from erc_phase1_solution.place_contact_guard import PlaceContactGuard, fault_reason

NODE=Path(finish.__file__).with_name('manipulation_node.py')
OWNER=next(x for x in ast.parse(NODE.read_text(encoding='utf-8')).body
           if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
METHODS={x.name:x for x in OWNER.body if isinstance(x,ast.FunctionDef)}
IDENTITY=dict(trial_id='trial',placement_attempt_id='attempt',target_model='book_col_1_row_1_red')


def method(name,extra=None):
    scope=dict(__package__='erc_phase1_solution', __name__='erc_phase1_solution.manipulation_node', np=np,HOME=HOME,ARM_JOINTS=ARM_JOINTS,
        checked_raised_place_finish_enabled=finish.checked_enabled,
        raised_place_normal_finish=finish.normal_finish)
    scope.update(extra or {})
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
                         copy.deepcopy(METHODS[name])],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(NODE),'exec'),scope)
    return scope[name]


def node(enabled=True):
    return SimpleNamespace(place_finish_keep_torso_height_enabled=enabled,
        delivery_evidence_enabled=True,table_scene_required=True,bin_scene_required=True,
        _active_place_scene_reference=object(),_target_book_model=None,
        _place_contact_guard=PlaceContactGuard(1,dict(IDENTITY)),_gripper_open_confirmed=True,
        _held_book_corners=None,chain=SimpleNamespace(lower=[0.],upper=[.35]))


@pytest.mark.parametrize('value',[None,0,1,'false','true',.0,.1])
def test_flag_requires_actual_boolean(value):
    with pytest.raises(ValueError):finish.checked_enabled(value)


@pytest.mark.parametrize('height',[0.,.1,.2991,.349999,.35])
def test_goal_uses_checked_height_and_does_not_mutate_home(height):
    q=HOME.copy();q[0]=height;before=HOME.copy()
    options,goal=finish.normal_finish(node(),IDENTITY,q,[],HOME)
    assert options=={'keep_torso_height':True}
    assert goal[0]==height and np.array_equal(goal[1:],HOME[1:])
    assert goal is not HOME and np.array_equal(HOME,before)


def test_false_and_absent_flag_preserve_goal_and_ignore_optional_context():
    for n in (SimpleNamespace(),SimpleNamespace(place_finish_keep_torso_height_enabled=False)):
        options,goal=finish.normal_finish(n,None,None,None,HOME)
        assert options=={} and goal is HOME


@pytest.mark.parametrize('attribute,value',[
    ('delivery_evidence_enabled',False),('table_scene_required',False),('bin_scene_required',False),
    ('_active_place_scene_reference',None),('_target_book_model','book_col_2_row_1_red'),
    ('_gripper_open_confirmed',False),('_place_contact_guard',None),
    ('_place_contact_guard',PlaceContactGuard(1,dict(IDENTITY,target_model='book_col_2_row_1_red'))),
    ('_place_contact_guard',PlaceContactGuard(1,dict(IDENTITY,trial_id='different'))),
    ('_place_contact_guard',PlaceContactGuard(1,dict(IDENTITY,placement_attempt_id='different'))),
    ('_place_contact_guard',PlaceContactGuard(1,dict(IDENTITY),fault={'reason':'contact'})),
    ('_place_contact_guard',PlaceContactGuard(0,dict(IDENTITY))),
    ('_held_book_corners',[]),('_held_book_corners',np.zeros((8,3))),
])
def test_missing_checked_release_context_rejects(attribute,value):
    n=node();setattr(n,attribute,value)
    with pytest.raises(RuntimeError):finish.normal_finish(n,IDENTITY,HOME,[],HOME)


@pytest.mark.parametrize('q',[[.36]*8,[-.01]*8,[.2]*7,[float('nan')]*8,[float('inf')]*8])
def test_bad_checked_state_rejects(q):
    with pytest.raises(ValueError):finish.normal_finish(node(),IDENTITY,q,[],HOME)


def test_uncorrelated_or_unchecked_route_rejects():
    with pytest.raises(RuntimeError):finish.normal_finish(node(),IDENTITY,HOME,None,HOME)
    with pytest.raises((TypeError,ValueError)):finish.normal_finish(node(),None,HOME,[],HOME)


def movement_node(*,fail=None,endpoint_error=False):
    events=[];n=SimpleNamespace(arm_client=object())
    def move(q,seconds,**kw):
        events.append(('arm',list(q),seconds,kw));return fail!='arm'
    def follow(client,names,q,seconds,**kw):
        events.append(('follow',list(q),seconds,kw));return fail!='follow'
    def torso(height,seconds):
        events.append(('torso',height,seconds));return fail!='torso'
    def endpoint(first,goal,phase):
        events.append(('endpoint',phase,list(first),list(goal)))
        if endpoint_error:raise RuntimeError('actual endpoint unconfirmed')
    n._move_arm_solution=move;n._follow=follow;n._move_torso=torso
    return n,events,endpoint


def test_enabled_retains_exact_arm_prefix_then_skips_only_lowering():
    execute=method('_execute_unloaded_home');start=HOME.copy();start[0]=.35
    waypoint=start.copy();waypoint[1]+=.1
    traces=[]
    for enabled in (False,True):
        n,events,endpoint=movement_node()
        assert execute(n,[waypoint],endpoint_wait=endpoint,endpoint_start=start,
                       final_arm_duration=3.1,keep_torso_height=enabled)
        traces.append(events)
    assert traces[1]==traces[0][:-2]
    assert traces[0][-2]==('torso',.1,2.)
    assert traces[0][-1][1]=='empty_return_torso_home'
    assert traces[1][-1][1]=='empty_return_arm_home'
    assert traces[1][-1][3]==[.35,*HOME[1:]]


def test_default_unmeasured_legacy_return_still_lowers():
    n,events,_=movement_node()
    assert method('_execute_unloaded_home')(n,[])
    assert events[-1]==('torso',.1,2.)


@pytest.mark.parametrize('adapter,start',[(None,HOME),(False,HOME),(lambda *a:None,None),
                                         (lambda *a:None,[.1]*7)])
def test_enabled_rejects_missing_adapter_or_bad_start_before_actions(adapter,start):
    n,events,_=movement_node()
    with pytest.raises((RuntimeError,ValueError)):
        method('_execute_unloaded_home')(n,[],keep_torso_height=True,
                                        endpoint_wait=adapter,endpoint_start=start)
    assert not events


@pytest.mark.parametrize('failed',['arm','follow'])
def test_failed_arm_goal_cannot_return_success(failed):
    n,events,endpoint=movement_node(fail=failed)
    assert not method('_execute_unloaded_home')(n,[HOME],keep_torso_height=True,
                                               endpoint_wait=endpoint,endpoint_start=HOME)
    assert not any(e[0]=='torso' for e in events)


def test_failed_measured_arm_endpoint_cannot_return_success():
    n,events,endpoint=movement_node(endpoint_error=True)
    with pytest.raises(RuntimeError,match='endpoint unconfirmed'):
        method('_execute_unloaded_home')(n,[],keep_torso_height=True,
                                        endpoint_wait=endpoint,endpoint_start=HOME)
    assert not any(e[0]=='torso' for e in events)


def test_return_rejects_enabled_unchecked_path_before_actions():
    n,events,_=movement_node();n.table_scene_required=True
    with pytest.raises(RuntimeError,match='measured direct return'):
        method('_return_from_bin')(n,[HOME],[],HOME,[],keep_torso_height=True)
    assert not events


def test_actual_direct_return_keeps_context_checks_and_forwards_only_to_checked_fold(monkeypatch):
    module=ModuleType('erc_phase1_solution.empty_pickup_collision');events=[]
    module.joint_velocity_limits=lambda path: {'official':'limits'}
    module.measured_context=lambda owner: ({'right':0.,'head':0.},None)
    def endpoint(owner,first,goal,**kw):
        kw['context_check']();events.append(('endpoint',kw['phase']))
    module.wait_for_geometry_endpoint=endpoint
    monkeypatch.setitem(sys.modules,module.__name__,module)
    n,commands,_=movement_node();n.table_scene_required=True;n.gripper_open=.069
    n._active_place_scene_reference=object()
    def execute(route,**kw):
        assert callable(kw['endpoint_wait'])
        assert kw['keep_torso_height'] is True and route==[]
        events.append(('fold',kw));return True
    n._execute_unloaded_home=execute
    f=method('_return_from_bin',dict(__package__='erc_phase1_solution',Path=Path,
        RIGHT_ARM_JOINTS=['right'],HEAD_JOINTS=['head'],get_package_share_directory=lambda p:'/official',
        _require_place_contact_clear=lambda owner:events.append(('contact_clear',)),
        measured_scene_context=lambda owner,reference:events.append(('scene_context',))))
    q=HOME.copy();q[0]=.35
    assert f(n,[q,q],[],q,[],direct_empty_home=[],keep_torso_height=True)
    assert [e[0] for e in events]==['contact_clear','scene_context','contact_clear','scene_context','endpoint','fold']
    assert len(commands)==1 and commands[0][0]=='arm'


def actual_open(n, events, *, command_passes=True):
    n.gripper_open=.069
    n._held_book_corners=np.zeros((8,3))
    n._target_book_model=IDENTITY['target_model']
    n._gripper_open_confirmed=False
    def command(position,**kw):
        events.append(('gripper_command',position));return command_passes
    n._command_gripper=command
    n._clear_target_contact_samples_unlocked=MethodType(method('_clear_target_contact_samples_unlocked'),n)
    n._open_gripper=MethodType(method('_open_gripper',dict(_place_contact_fault=fault_reason,
        measured_scene_context=lambda owner,scene:events.append(('open_scene_check',)))),n)


def normal_suffix(observer):
    branch=next(x for x in METHODS['_place'].body if isinstance(x,ast.If)
        and any(isinstance(z,ast.FunctionDef) and z.name=='verify_open' for z in x.body))
    index=METHODS['_place'].body.index(branch)
    assert index>=2
    initialization,guard=METHODS['_place'].body[index-2:index]
    assert isinstance(initialization,ast.Assign) and len(initialization.targets)==1
    assert isinstance(initialization.targets[0],ast.Name) and initialization.targets[0].id=='release_owner'
    expected_guard=ast.parse("if release_only_endpoint is not None and release_owner is None:\n"
        "    raise RuntimeError('release-only plan cannot enter legacy release/return')").body[0]
    assert ast.dump(guard,include_attributes=False)==ast.dump(expected_guard,include_attributes=False)
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in
        ('self','correlation','solutions','carried_transition_waypoints','torso_ready',
         'unloaded_home_waypoints','direct_empty_home','arm_timing_options','release_only_endpoint')],
         kwonlyargs=[],kw_defaults=[],defaults=[ast.Constant(value=None)])
    tail=ast.FunctionDef(name='tail',args=args,
        body=[copy.deepcopy(x) for x in METHODS['_place'].body[index-2:index+1]],decorator_list=[])
    from erc_phase1_solution import release_pose_finish as actual_release_finish
    scope=dict(__package__='erc_phase1_solution', __name__='erc_phase1_solution.manipulation_node', release_pose_finish=actual_release_finish,
               HOME=HOME,raised_place_normal_finish=finish.normal_finish,
               observe_measured_open_pose=observer)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tail],type_ignores=[])),str(NODE),'exec'),scope)
    return scope['tail']


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_normal_suffix_orders_measured_open_return_and_correct_final_observation(enabled):
    n=node(enabled);events=[];q=HOME.copy();q[0]=.3499
    def observer(owner,identity,goal,event):
        events.append((event,list(goal)));return {'verified':True}
    actual_open(n,events)
    guard=n._place_contact_guard
    def returned(*a,**kw):events.append(('return',kw));return True
    n._return_from_bin=returned
    assert normal_suffix(observer)(n,IDENTITY,[q],[],q,[],[],{})
    assert [x[0] for x in events]==['open_scene_check','gripper_command','placement_open_measured','return','placement_hand_return_measured']
    assert n._target_book_model is None and n._held_book_corners is None and n._gripper_open_confirmed
    assert n._place_contact_guard is guard and guard.correlation==IDENTITY
    assert events[3][1]==({'direct_empty_home':[],'keep_torso_height':True} if enabled else {'direct_empty_home':[]})
    assert events[-1][1]==[q[0] if enabled else HOME[0],*HOME[1:]]


@pytest.mark.parametrize('fault',['open','return','final_measurement'])
def test_actual_normal_suffix_does_not_fabricate_success(fault):
    n=node();events=[]
    def observer(owner,identity,goal,event):
        events.append(event);return {'verified':not (fault=='final_measurement' and event=='placement_hand_return_measured')}
    actual_open(n,events,command_passes=fault!='open')
    n._return_from_bin=lambda *a,**kw: fault!='return'
    tail=normal_suffix(observer)
    if fault=='open':
        with pytest.raises(RuntimeError,match='placement_release_unverified'):
            tail(n,IDENTITY,[HOME],[],HOME,[],[],{})
        assert events==[('open_scene_check',),('gripper_command',.069)]
        assert n._target_book_model==IDENTITY['target_model'] and n._held_book_corners is not None
    else:
        assert not tail(n,IDENTITY,[HOME],[],HOME,[],[],{})
        if fault=='return':assert 'placement_hand_return_measured' not in events


def test_original_measurement_requires_actual_raised_home_and_stationarity():
    goal=HOME.copy();goal[0]=.35;names=(*IK_JOINTS,'gripper_left_finger_joint')
    raw=dict(producer_stamp_ns=1_000_000_000,positions=dict(zip(names,[*goal,.069])),
             velocities=dict.fromkeys(names,0.))
    odom=dict(stamp_ns=1_000_000_000,linear_speed=0.,angular_speed=0.)
    assert measured_pose_is_open(raw,odom,goal,1_000_000_000,IK_JOINTS)[0]
    assert not measured_pose_is_open(raw,odom,HOME,1_000_000_000,IK_JOINTS)[0]
    raw['velocities']['torso_lift_joint']=.002
    assert not measured_pose_is_open(raw,odom,goal,1_000_000_000,IK_JOINTS)[0]


def test_only_normal_suffix_opts_in_and_default_remains_false():
    calls=[n for n in ast.walk(OWNER) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='raised_place_normal_finish']
    assert len(calls)==1
    for name in ('_recover_closed_place','_recover_pick','_stow'):
        if name in METHODS:
            assert not any(isinstance(n,ast.keyword) and n.arg=='keep_torso_height'
                           for n in ast.walk(METHODS[name]))
    values=next(n.value for n in METHODS['_declare_parameters'].body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='values' for t in n.targets))
    assert ast.literal_eval(values)['place_finish_keep_torso_height_enabled'] is False
