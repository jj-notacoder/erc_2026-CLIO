"""Actual helper policy orchestration plus frozen-baseline coordinator replay.

Synthetic gate backends inject rejection at each existing admission boundary;
they are not a robot/scene feasibility certificate.
"""
import copy
import math
from pathlib import Path
import runpy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution import scene_checked_place as mod


class SignedChain:
    def __init__(self, wanted=lambda q:-2.2+.5*q[1]):
        self.wanted=wanted
        self.lower=np.array([0.]+[-3.]*7);self.upper=np.array([.35]+[3.]*7)
    def forward(self,q):
        angle=float(q[-1])-self.wanted(q)+np.pi/2
        c,s=np.cos(angle),np.sin(angle);result=np.eye(4)
        result[:3,:3]=[[1.,0.,0.],[0.,c,-s],[0.,s,c]]
        return result


def endpoints():
    start=np.zeros(8);start[0]=.30;start[-1]=-2.2
    goal=start.copy();goal[1:7]=[1.,.8,-.6,-1.5,.5,1.2];goal[-1]=-1.75
    return start,goal


def test_default_twenty_is_bitwise_equal_to_previous_production_coordinator():
    baseline=runpy.run_path(str(Path(__file__).parent/'fixtures/coordinator_twenty_step_original.py'),
        init_globals=dict(np=np,math=math,_finite=mod._finite))['coordinated_supported_goals']
    for policy in ({},{'shoulder_progress_power':.5},
                   {'joint_progress_powers':(1.,.5,1.,1.,.5,1.)}):
        start,goal=endpoints();original_start=start.copy();original_goal=goal.copy()
        old=baseline(SignedChain(),start,goal,margin=.10,minimum_support=.75,**policy)
        new=mod.coordinated_supported_goals(SignedChain(),start,goal,margin=.10,minimum_support=.75,**policy)
        explicit=mod.coordinated_supported_goals(SignedChain(),start,goal,margin=.10,
            minimum_support=.75,proximal_steps=20,**policy)
        np.testing.assert_array_equal(new,old);np.testing.assert_array_equal(explicit,old)
        np.testing.assert_array_equal(start,original_start);np.testing.assert_array_equal(goal,original_goal)
    # Preserve the prior failure behavior for a signed-support wrap barrier too.
    chain=SignedChain(lambda q:np.pi);chain.lower[-1]=-.3;chain.upper[-1]=.3
    start=np.zeros(8);start[0]=.3;goal=start.copy();goal[1]=.2
    assert baseline(chain,start,goal,margin=.10,minimum_support=.75) is None
    assert mod.coordinated_supported_goals(chain,start,goal,margin=.10,minimum_support=.75) is None


def test_ten_candidates_preserve_exact_endpoint_legal_margin_and_signed_roll_continuity():
    start,goal=endpoints();chain=SignedChain()
    result=mod.coordinated_supported_goals(chain,start,goal,margin=.10,minimum_support=.75,
        proximal_steps=10,joint_progress_powers=(1.,.5,1.,1.,.5,1.))
    assert result is not None and len(result)==11  # ten coordinated states plus the untouched IK endpoint
    np.testing.assert_array_equal(result[-1],goal)
    values=np.asarray(result)
    assert np.all(values[:,1:]>=chain.lower[1:]+.10) and np.all(values[:,1:]<=chain.upper[1:]-.10)
    assert all(chain.forward(q)[2,1]>=.75 for q in result)
    assert np.max(np.abs(np.diff(np.r_[start[-1],values[:,-1]])))<np.pi


@pytest.mark.parametrize('count',[0,9,11,20.,True,np.int64(10),float('nan')])
def test_only_explicit_python_integer_ten_or_twenty_is_accepted(count):
    with pytest.raises(ValueError,match='coordinator policy'):
        mod.coordinated_supported_goals(SignedChain(),*endpoints(),margin=.10,
            minimum_support=.75,proximal_steps=count)


@pytest.fixture
def orchestration(monkeypatch,tmp_path):
    state=SimpleNamespace(policy=None,events=[],policies=[],reject=None,reject_custom=False,cancel_after_ten=False)
    start=np.zeros(8);start[0]=.35;torso=start.copy();torso[0]=.30
    mid=torso.copy();mid[1]=.1;first=torso.copy();first[1]=.2;last=torso.copy();last[1]=.3
    node=SimpleNamespace(_cancel=threading.Event(),gripper_open=.069,place_torso_height=.30,
        place_joint_limit_margin=.03,carried_supported_jaw_vertical_component=.75,
        _shelf_cradle_geometry=object(),_held_book_corners=np.zeros((8,3)),chain=object(),bin_scene_required=True)
    def check(stage):
        state.events.append((state.policy,stage))
        reject=state.policy==10 and state.reject==stage
        if reject and state.cancel_after_ten:node._cancel.set()
        return not reject
    node._adaptive_motion_feedback=lambda:(SimpleNamespace(position=.017,stamp_ns=12),None)
    node._carried_robot_transition_is_safe=lambda a,b,book:check('torso_body' if state.policy is None else 'cartesian_body')
    node._gravity_supported_transition_is_safe=lambda a,b:check('torso_support' if state.policy is None else
        ('cartesian_support' if np.array_equal(b,last) else 'setup_support'))
    node._plan_retracted_transition=lambda a,b:[]
    def route(a,goals,book,**kwargs):
        state.events.append((state.policy,('route_arguments',kwargs)))
        assert kwargs==dict(require_gravity_support=True)
        assert np.array_equal(book,node._held_book_corners)
        return [q.copy() for q in goals] if check('controller_route') else None
    node._plan_carried_joint_route=route
    def coordinator(*args,**kwargs):
        state.policy=kwargs.get('proximal_steps',20);state.policies.append(copy.deepcopy(kwargs))
        assert kwargs['margin']==.10 and kwargs['minimum_support']==.75
        if state.reject_custom and 'joint_progress_powers' in kwargs:return None
        return [mid.copy(),first.copy()]
    monkeypatch.setattr(mod,'coordinated_supported_goals',coordinator)
    class Scene:
        def __init__(self,node,obstacle,tool,attached,**kwargs):
            self.attached=attached;self.samples=0;self.cache_hits=0
            self.last_rejection={'reason':'synthetic fault'};self.minimum_moving_left_z=.8
        def leg(self,a,b,q,loaded):
            self.samples+=1
            if not loaded:
                assert q==node.gripper_open
                return check('return')
            if np.array_equal(a,b):return check('cartesian_endpoint')
            if state.policy is None:return check('torso_scene')
            return check('cartesian_scene' if np.array_equal(b,last) else 'setup_scene')
        def sample(self,q,aperture,loaded):
            assert loaded and .017<=aperture<=.069
            self.samples+=1;return check('opening')
    monkeypatch.setattr(mod,'PlaceSceneChecker',Scene)
    class Obstacle:
        def __init__(self,point,bounds,*,cavity_bounds,bin_scene):
            self.point=np.asarray(point);self.origin=self.point.copy();self.rotation=np.eye(3)
            self.bounds=np.asarray(bounds);self.cavity_bounds=cavity_bounds;self.margin=.005
            self.material_bounds=[bounds];self.registered_scene=bin_scene
    monkeypatch.setattr(mod,'NominalBinObstacle',Obstacle)
    monkeypatch.setattr(mod,'validate_bin_scene',lambda scene:copy.deepcopy(scene))
    monkeypatch.setattr(mod,'TableSceneObstacle',lambda scene:object())
    monkeypatch.setattr(mod,'ScreenEnvelope',lambda path:object())
    monkeypatch.setattr(mod,'load_stl_triangles',lambda path:mod._box_triangles([.3,.2,.5]))
    assets={'urdf/tiago_pro.urdf':b'urdf','models/collection_bin/meshes/erc_base_collection_bin.STL':b'bin',
        'models/table/meshes/erc_base_table.STL':b'table','models/table/sdf/erc_table.sdf':b'sdf'}
    for path,content in assets.items():
        file=tmp_path/path;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(content)
    pins={b'bin':'b'*64,b'table':mod.TABLE_MESH_SHA256,b'sdf':'90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f',b'urdf':'a'*64}
    monkeypatch.setattr(mod,'hashlib',SimpleNamespace(sha256=lambda data:SimpleNamespace(hexdigest=lambda:pins[data])))
    def solve(n,positions,rotation,height,seed,scene,candidate,**kwargs):
        if not candidate([first,last],[]):raise RuntimeError('all candidates rejected')
        return [first,last],0,.5,[]
    monkeypatch.setattr(mod,'solve_scene_cartesian',solve)
    def call(registered=True):
        node.bin_scene_required=registered
        return mod.plan_scene_checked_place(node,[[0,0,0]],np.eye(3),start,torso,np.ones(8),[.8,0,.75],
            lambda _:tmp_path,table_scene={'valid':True},bin_scene={'mesh_sha256':'b'*64} if registered else None)
    return state,node,call


def test_ten_is_selected_only_after_all_existing_loaded_opening_and_return_gates_pass(orchestration):
    state,_,call=orchestration;result=call()
    assert [p.get('proximal_steps',20) for p in state.policies]==[10]
    assert result.diagnostics['proximal_setup_steps']==10
    stages=[stage for policy,stage in state.events if policy==10 and isinstance(stage,str)]
    for required in ('setup_support','controller_route','setup_scene','cartesian_body',
                     'cartesian_support','cartesian_scene','opening','return'):
        assert required in stages
    assert stages.index('controller_route')<stages.index('setup_scene')<stages.index('opening')<stages.index('return')
    assert [(p,s) for p,s in state.events if p is None][:3]==[(None,'torso_body'),(None,'torso_support'),(None,'torso_scene')]


@pytest.mark.parametrize('stage',['setup_support','controller_route','setup_scene','cartesian_body',
                                 'cartesian_support','cartesian_scene','opening','return'])
def test_ten_rejection_at_each_mandatory_gate_falls_back_to_fully_checked_twenty(orchestration,stage):
    state,_,call=orchestration;state.reject=stage
    result=call()
    assert [p.get('proximal_steps',20) for p in state.policies]==[10,20]
    assert result.diagnostics['proximal_setup_steps']==20
    assert (10,stage) in state.events
    assert (20,'controller_route') in state.events and (20,'opening') in state.events and (20,'return') in state.events


def test_both_custom_routes_rejected_then_original_legacy_policy_remains_available(orchestration):
    state,_,call=orchestration;state.reject_custom=True
    result=call()
    assert [p.get('proximal_steps',20) for p in state.policies]==[10,20,20]
    assert 'joint_progress_powers' not in state.policies[-1]
    assert result.diagnostics['joint_progress_powers']==[1.]*6
    assert result.diagnostics['proximal_setup_steps']==20


def test_nonregistered_table_path_keeps_twenty_as_first_policy(orchestration):
    state,_,call=orchestration;result=call(registered=False)
    assert [p.get('proximal_steps',20) for p in state.policies]==[20]
    assert result.diagnostics['proximal_setup_steps']==20


def test_cancellation_after_failed_ten_stops_before_twenty(orchestration):
    state,node,call=orchestration;state.reject='controller_route';state.cancel_after_ten=True
    with pytest.raises(RuntimeError,match='candidates rejected'):call()
    assert node._cancel.is_set() and len(state.policies)==1
    assert (10,'opening') not in state.events


def test_controller_leg_sampling_keeps_sixty_one_samples_and_detects_interior_rejection():
    scene=object.__new__(mod.PlaceSceneChecker);scene.node=SimpleNamespace(carried_transition_samples=61)
    first=np.zeros(8);last=first.copy();last[1]=.10
    samples=[];scene.sample=lambda q,a,loaded:samples.append(q.copy()) or True
    assert scene.leg(first,last,.017,True)
    assert len(samples)==61
    np.testing.assert_array_equal(samples[0],first);np.testing.assert_array_equal(samples[-1],last)
    scene.last_rejection={'reason':'interior'}
    scene.sample=lambda q,a,loaded:not .049<=q[1]<=.051
    assert not scene.leg(first,last,.017,True)
    assert scene.last_rejection['fraction']==pytest.approx(.5)
