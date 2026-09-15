"""Independent planner/actual-method return ordering tests; no ROS or meshes."""
import ast
import hashlib
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import scene_checked_place as mod
from erc_phase1_solution.motion_profiles import HOME, ARM_JOINTS
import test_scene_checked_place_flow as flow


def actual_return_methods():
    path=Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    wanted={'_return_from_bin','_execute_unloaded_home'}
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in wanted]
    from erc_phase1_solution.raised_place_finish import checked_enabled
    namespace=dict(__package__='erc_phase1_solution', __name__='erc_phase1_solution.manipulation_node', np=np,HOME=HOME,ARM_JOINTS=ARM_JOINTS,checked_raised_place_finish_enabled=checked_enabled)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*methods],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace


def execution_fixture():
    n=NS(_cancel=threading.Event(),arm_client=object());events=[]
    for name,fn in actual_return_methods().items():
        if name.startswith('_') and callable(fn):setattr(n,name,MethodType(fn,n))
    n._move_arm_solution=lambda q,t:events.append(('arm',np.asarray(q).copy(),t)) or not n._cancel.is_set()
    n._follow=lambda client,names,q,t:events.append(('home_arm',np.asarray(q).copy(),t)) or not n._cancel.is_set()
    n._move_torso=lambda z,t:events.append(('torso',z,t)) or not n._cancel.is_set()
    first=HOME.copy();first[0]=.35;first[6]+=2.6175347001886404
    second=first.copy();second[1]+=.1
    last=second.copy();last[1]+=.1
    carry=HOME.copy();carry[0]=.35;carry[1]+=.4
    staging=carry.copy();staging[1]+=.2
    return n,events,[first,second,last],carry,staging


def test_empty_list_selects_direct_fold_and_preserves_explicit_torso_action():
    n,events,path,carry,staging=execution_fixture()
    assert n._return_from_bin(path,[staging],carry,[staging],direct_empty_home=[])
    assert [e[0] for e in events]==['arm','arm','home_arm','torso']
    np.testing.assert_array_equal(events[0][1],path[1])
    np.testing.assert_array_equal(events[1][1],path[0])
    np.testing.assert_array_equal(events[2][1],HOME[1:])
    assert events[2][2]==pytest.approx(4*2.6175347001886404)
    assert events[3]==('torso',float(HOME[0]),2.)
    assert not any(e[0]=='arm' and np.array_equal(e[1],carry) for e in events)


def test_none_preserves_legacy_retrace_and_default_fold_duration():
    n,events,path,carry,staging=execution_fixture()
    assert n._return_from_bin(path,[staging],carry,[staging])
    assert [e[0] for e in events]==['arm']*5+['home_arm','torso']
    np.testing.assert_array_equal(events[2][1],staging)
    np.testing.assert_array_equal(events[3][1],carry)
    assert events[-2][2]==2.8


def test_direct_small_fold_retains_minimum_existing_duration():
    n,events,path,carry,staging=execution_fixture()
    path=[HOME.copy(),HOME.copy()]
    assert n._return_from_bin(path,[],carry,[],direct_empty_home=[])
    assert events[-2][2]==2.8


@pytest.mark.parametrize('failure',['reverse','fold','lower','cancel_before_return'])
def test_return_stops_at_failed_or_cancelled_action(failure):
    n,events,path,carry,staging=execution_fixture()
    if failure=='reverse':
        n._move_arm_solution=lambda q,t:events.append(('failed_reverse',)) or False
    elif failure=='fold':
        n._follow=lambda *a:events.append(('failed_fold',)) or False
    elif failure=='lower':
        n._move_torso=lambda *a:events.append(('failed_lower',)) or False
    else:n._cancel.set()
    assert not n._return_from_bin(path,[staging],carry,[staging],direct_empty_home=[])
    if failure in ('reverse','cancel_before_return'):
        assert not any(e[0] in ('home_arm','torso') for e in events)
    if failure=='fold':assert not any(e[0]=='torso' for e in events)


@pytest.fixture
def table_plan(monkeypatch,tmp_path):
    # Asset loading and collision predicates are deliberate protocol doubles;
    # this tests the actual orchestration, not synthetic geometry admission.
    payloads={'urdf/tiago_pro.urdf':b'urdf',
        'models/collection_bin/meshes/erc_base_collection_bin.STL':b'bin',
        'models/table/meshes/erc_base_table.STL':b'tablemesh',
        'models/table/sdf/erc_table.sdf':b'tablesdf'}
    for relative,data in payloads.items():
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    real_hash=hashlib.sha256
    def asset_hash(data):
        if data==b'tablemesh':return NS(hexdigest=lambda:mod.TABLE_MESH_SHA256)
        if data==b'tablesdf':return NS(hexdigest=lambda:'90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f')
        return real_hash(data)
    monkeypatch.setattr(mod,'hashlib',NS(sha256=asset_hash))
    monkeypatch.setattr(mod,'load_stl_triangles',lambda _:mod._box_triangles([.3,.2,.5]))
    monkeypatch.setattr(mod,'ScreenEnvelope',lambda _:object())
    monkeypatch.setattr(mod,'TableSceneObstacle',lambda _:object())
    torso=np.zeros(8);torso[0]=.35
    first=torso.copy();first[1]=.1
    last=first.copy();last[1]=.2
    middle=torso.copy();middle[1]=.05
    unloaded=torso.copy();unloaded[1]=-.1
    home=HOME.copy();home[0]=.35
    n=NS(_cancel=threading.Event(),gripper_open=.069,place_torso_height=.35,
         place_joint_limit_margin=.1,carried_supported_jaw_vertical_component=.75,
         _shelf_cradle_geometry=object(),_held_book_corners=np.zeros((8,3)),chain=object(),
         reject=None,cancel_after_search=False)
    n._adaptive_motion_feedback=lambda:(NS(position=.017,stamp_ns=1),None)
    n._carried_robot_transition_is_safe=lambda *a:True
    n._gravity_supported_transition_is_safe=lambda *a:True
    n._plan_retracted_transition=lambda *a:[unloaded]
    n._plan_carried_joint_route=lambda start,goals,book,**kw:list(goals)
    n._solve_cartesian_path=lambda *a,**kw:pytest.fail('table path used legacy solver')
    monkeypatch.setattr(mod,'coordinated_supported_goals',lambda *a,**kw:[middle,first])
    events=[]
    class Scene:
        def __init__(self,node,obstacle,tool,attached,**kw):
            self.attached=attached;self.samples=self.cache_hits=0
            self.minimum_moving_left_z=.8;self.last_rejection=None
        def leg(self,a,b,q,loaded):
            self.samples+=1;events.append((a.copy(),b.copy(),q,loaded))
            if not loaded and n.reject is not None and n.reject(a,b):
                self.last_rejection={'reason':'deliberate_return_rejection'};return False
            return True
        def sample(self,*a):self.samples+=1;return True
    monkeypatch.setattr(mod,'PlaceSceneChecker',Scene)
    def solve(node,positions,rotation,height,seed,scene,candidate,**kw):
        if not candidate([first,last],[]):raise RuntimeError('no validated candidate')
        if n.cancel_after_search:n._cancel.set()
        return [first,last],0,.1,[]
    monkeypatch.setattr(mod,'solve_scene_cartesian',solve)
    def call():
        return mod.plan_scene_checked_place(n,[[0,0,0]],np.eye(3),torso,torso,torso,
            [.9,0,.75],lambda _:tmp_path,table_scene={'valid':True})
    return NS(node=n,events=events,call=call,first=first,last=last,middle=middle,
              torso=torso,home=home,unloaded=unloaded)


def test_direct_plan_matches_exact_empty_dispatch_geometry(table_plan):
    p=table_plan;result=p.call()
    assert result.empty_return_from_clearance==[]
    assert result.diagnostics['direct_empty_fold_from_clearance']
    empty=[e for e in p.events if not e[3]]
    assert len(empty)==3 and all(e[2]==.069 for e in empty)
    for event,(a,b) in zip(empty,[(p.last,p.first),(p.first,p.home),(p.home,HOME)]):
        np.testing.assert_array_equal(event[0],a);np.testing.assert_array_equal(event[1],b)


def test_direct_rejection_selects_only_fully_checked_legacy_return(table_plan):
    p=table_plan
    p.node.reject=lambda a,b:np.array_equal(a,p.first) and np.array_equal(b,p.home)
    result=p.call()
    assert result.empty_return_from_clearance is None
    assert not result.diagnostics['direct_empty_fold_from_clearance']
    empty=[e for e in p.events if not e[3]]
    expected=[p.last,p.first,p.middle,p.torso,p.unloaded,p.home,HOME]
    for event,(a,b) in zip(empty[-6:],zip(expected,expected[1:])):
        np.testing.assert_array_equal(event[0],a);np.testing.assert_array_equal(event[1],b)


def test_final_torso_leg_rejection_prevents_either_return_admission(table_plan):
    p=table_plan;p.node.reject=lambda a,b:np.array_equal(b,HOME)
    with pytest.raises(RuntimeError,match='no validated candidate'):p.call()


def test_cancel_after_candidate_acceptance_discards_direct_plan(table_plan):
    p=table_plan;p.node.cancel_after_search=True
    with pytest.raises(RuntimeError,match='no_accepted_route'):p.call()


@pytest.mark.parametrize('correlated',[False,True])
@pytest.mark.parametrize('released',[False,True])
def test_actual_place_routes_direct_only_after_successful_release(correlated,released):
    fixture=flow.PlaceWiring();fixture.setUp()
    n,ns,calls=fixture.node,fixture.namespace,fixture.calls
    n.table_scene_required=True;n._selected_place_scene_reference={};n._selected_place_table_scene={}
    n._lock=threading.RLock();lock=threading.RLock();n._adaptive_command_guard=lambda:lock
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=1_000_000_000))
    old=ns['plan_scene_checked_place']
    def planner(*a,**kw):
        result=old(*a,**kw);result.empty_return_from_clearance=[];return result
    ns['plan_scene_checked_place']=planner
    n._return_from_bin=lambda *a,**kw:calls.append(('direct_return',kw)) or True
    n._recover_closed_place=lambda **kw:pytest.fail('table release failure attempted recovery/open again')
    n.delivery_evidence_enabled=correlated
    payload=None
    if correlated:
        n._target_book_model='target';payload=dict(trial_id='trial',placement_attempt_id='attempt',target_model='target')
        ns['AttemptIdentity']=lambda **kw:NS(**kw)
        ns['observe_measured_open_pose']=lambda *a,**kw:{'verified':released}
    def opening(**kw):
        calls.append(('release_attempt',None))
        return kw['verify_measurement']() if 'verify_measurement' in kw else released
    n._open_gripper=opening
    if released:
        assert ns['_place'](n,payload)
        selected=[e for e in calls if e[0]=='direct_return']
        assert len(selected)==1 and selected[0][1]=={'direct_empty_home':[]}
        assert next(i for i,e in enumerate(calls) if e[0]=='release_attempt')<next(i for i,e in enumerate(calls) if e[0]=='direct_return')
    else:
        with pytest.raises(RuntimeError,match='placement_release_unverified'):ns['_place'](n,payload)
        assert not any(e[0]=='direct_return' for e in calls)
