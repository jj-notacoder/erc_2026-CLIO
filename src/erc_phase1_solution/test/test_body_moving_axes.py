"""Prepared body-pair proof controls; no ROS, robot assets or replay.

Actual old/new methods and the pinned existing helper are used. A rejecting
narrow spy exposes any bypass; analytic separation witnesses explain why that
bypass is safe, not universal equivalence to the old numerical oracle.
"""
import ast
import copy
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path
import struct
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/'erc_phase1_solution/manipulation_node.py'
ORIGINAL=ROOT/'test/fixtures/body_axes_original_node.py'
HELPER=ROOT/'erc_phase1_solution/static_pair_separation.py'
UTILS=ROOT/'erc_phase1_solution/runtime_utils.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='1dea6499c9f594a60cd51936d52862433730a8373efd99700b7e76ecdf7585f9'
spec=importlib.util.spec_from_file_location('body_existing_axes',HELPER)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def rotation(angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def box(center=(0.,0.,0.),size=(2.,.1,.1)):
    vertices=np.array(list(itertools.product((-1.,1.),repeat=3)))*np.array(size)/2+center
    faces=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
           [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
    return vertices[np.asarray(faces, dtype=int)]


def load_method(path,fixture):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    consts={'LEFT_COLLISION_LINKS','RIGHT_COLLISION_LINKS','HEAD_COLLISION_LINKS',
            'CARRIED_COLLISION_LINKS','DIRECTLY_CONNECTED_COLLISION_LINKS'}
    selected=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.Assign)
              and isinstance(n.targets[0],ast.Name) and n.targets[0].id in consts]
    assert len(selected)==len(consts)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_robot_self_collision')
    key=next(n for n in ast.parse(UTILS.read_text(encoding='utf-8')).body
             if isinstance(n,ast.FunctionDef) and n.name=='joint_state_cache_key')
    selected += [copy.deepcopy(key),ast.ClassDef(name='ActualNode',bases=[],keywords=[],
                      body=[copy.deepcopy(method)],decorator_list=[])]
    env=dict(np=np,math=math,struct=struct,PreparedTriangleMesh=fixture.prepare,
             triangle_meshes_intersect=fixture.narrow)
    unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
                         *selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit),str(path),'exec'),env)
    return env['ActualNode'](),env


class Fixture:
    def __init__(self,gap=.1,first='torso_base_link',second='arm_left_3_link'):
        self.events=[]; self.projections=[]; self.names={}
        self.q=np.zeros(8); self.right=np.zeros(7); self.head=np.zeros(2)
        self.first,self.second=first,second
        r=rotation(.6)
        self.local={first:box()@r.T+[0.,0.,1.],second:box([0.,.1+gap,0.])@r.T+[0.,0.,1.]}
        self.helper=module.StaticPairSeparation()
        self.proof=NS(separated=self.separated)
        self.narrow_result=True
        self.closed={first,second}

    def world(self,q,**context):
        self.events.append('world')
        # Deliberate small meshes, not a model of the actual robot FK.
        r=rotation(q[2]); axis=rotation(.6+q[2])[:,1]
        surfaces={name:s@r.T+(axis*(-q[1]) if name.startswith('arm_left_')
                    else axis*(context['right_positions'][0]+context['head_positions'][0]))
                  for name,s in self.local.items()}
        self.names={id(s):name for name,s in surfaces.items()}
        return surfaces

    def separated(self,a,b,first,second):
        self.events.append(('proof',a,b))
        self.projections.append((a,b,first.copy(),second.copy()))
        return self.helper.separated(a,b,first,second)

    def prepare(self,surface):
        name=self.names[id(surface)]
        self.events.append(('prepare',name))
        return NS(surface=surface,name=name)

    def narrow(self,a,b,**flags):
        self.events.append(('narrow',a.name,b.name,flags))
        if isinstance(self.narrow_result,Exception): raise self.narrow_result
        return self.narrow_result

    def node(self,path=CURRENT,attach=True):
        node,env=load_method(path,self)
        node._self_collision_cache={}; node._static_self_collision_cache={}
        node._resolved_right_positions=lambda p: np.array(self.right if p is None else p,copy=True)
        node._resolved_head_positions=lambda p: np.array(self.head if p is None else p,copy=True)
        node._world_collision_surfaces=self.world
        node._watertight_collision_links=lambda:self.closed
        if attach: node._static_pair_separation=self.proof
        return node


def test_only_original_static_pair_qualifiers_removed():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()=='4f641380bcc0103fe3e0a05e0803ed34c774034ae13536dda2456bdbd554951d'
    def method_source(path):
        source=path.read_text(encoding='utf-8')
        tree=ast.parse(source)
        cls=next(n for n in tree.body
                 if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
        method=next(n for n in cls.body
                    if isinstance(n,ast.FunctionDef) and n.name=='_robot_self_collision')
        pair=next(n for n in method.body
                  if isinstance(n,ast.FunctionDef) and n.name=='links_intersect')
        return ast.get_source_segment(source,pair)

    # Pin the complete pair predicate and exact qualifier-only inverse. The
    # separate snapshot test restores the complete outer body method AST.
    before=method_source(ORIGINAL)
    after=method_source(CURRENT)
    old='''            if (
                collision_surfaces is None
                and first_link not in LEFT_COLLISION_LINKS
                and second_link not in LEFT_COLLISION_LINKS
            ):'''
    new='''            if collision_surfaces is None:'''
    anchor="\n                separator = getattr(self, '_static_pair_separation', None)"
    assert after.count(new+anchor)==1
    assert after.replace(new+anchor,old+anchor)==before
    assert ast.dump(ast.parse(after.replace(new+anchor,old+anchor)))==ast.dump(ast.parse(before))


@pytest.mark.parametrize('pair',[('torso_base_link','arm_left_3_link'),
                                ('arm_left_3_link','arm_left_5_link'),
                                ('arm_left_3_link','arm_right_4_link')])
def test_new_moving_pair_proof_precedes_preparation(pair):
    f=Fixture(first=pair[0],second=pair[1]); n=f.node()
    assert n._robot_self_collision(f.q) is None
    assert f.events==['world',('proof',*pair)]
    a,b,first,second=f.projections[0]
    assert np.all(first.max(axis=(0,1))>=second.min(axis=(0,1)))
    assert np.all(second.max(axis=(0,1))>=first.min(axis=(0,1)))
    old=Fixture(first=pair[0],second=pair[1]); old_n=old.node(ORIGINAL)
    assert old_n._robot_self_collision(old.q)==pair
    assert not any(isinstance(x,tuple) and x[0]=='proof' for x in old.events)


def test_existing_static_pair_uses_identical_helper_and_cache():
    records=[]
    for path in (ORIGINAL,CURRENT):
        f=Fixture(first='torso_lift_link',second='arm_right_3_link'); n=f.node(path)
        assert n._robot_self_collision(f.q) is None
        records.append((f.events,n._self_collision_cache,n._static_self_collision_cache))
    assert records[0]==records[1]


@pytest.mark.parametrize('gap',[-.01,0.,99e-6,100e-6])
def test_moving_touch_overlap_and_margin_keep_original_rejection(gap):
    outcomes=[]
    for path in (ORIGINAL,CURRENT):
        f=Fixture(gap=gap); n=f.node(path)
        outcomes.append(n._robot_self_collision(f.q))
        assert f.events[-1]==('narrow',f.first,f.second,dict(first_watertight=True,second_watertight=True))
    assert outcomes==[('torso_base_link','arm_left_3_link')]*2


@pytest.mark.parametrize('geometry',['contained','compound','nan'])
def test_current_compound_containment_or_uncertainty_keeps_fallback(geometry):
    f=Fixture()
    if geometry=='contained': f.local[f.first]=box(size=(4.,4.,4.))+[0.,0.,1.]
    elif geometry=='compound': f.local[f.second]=np.concatenate([f.local[f.second],f.local[f.first]])
    else: f.local[f.second][0,0,0]=np.nan
    n=f.node()
    assert n._robot_self_collision(f.q)==(f.first,f.second)
    assert f.events[-1][0]=='narrow'


@pytest.mark.parametrize('change',['left','right','head','rotate_and_overlap'])
def test_new_pose_reprojects_current_meshes_despite_old_successful_hint(change):
    f=Fixture(); n=f.node(); assert n._robot_self_collision(f.q) is None
    first=f.projections[-1]
    if change=='left': f.q[1]=.2
    elif change=='right': f.right[0]=.2
    elif change=='head': f.head[0]=.2
    else: f.q[1]=.2; f.q[2]=1.3
    assert n._robot_self_collision(f.q)==(f.first,f.second)
    assert len(f.projections)==2
    assert not (np.array_equal(first[2],f.projections[-1][2]) and np.array_equal(first[3],f.projections[-1][3]))
    assert f.events[-1][0]=='narrow'


def test_hint_alone_does_not_cache_mutated_geometry_or_ignore_components():
    helper=module.StaticPairSeparation()
    first=box(); second=box([0.,.2,0.]); saved=first.copy()
    assert helper.separated('left3','left5',first,second)
    second[...,1]-=.2
    assert not helper.separated('left3','left5',first,second)
    second[...,1]+=.2
    assert not helper.separated('left3','left5',first,np.concatenate([second,first]))
    assert helper.separated('left3','left5',first,second)
    np.testing.assert_array_equal(first,saved)


@pytest.mark.parametrize('pair',[('arm_left_1_link','arm_left_2_link'),
                                ('torso_lift_link','arm_left_1_link'),
                                ('head_1_link','head_2_link')])
def test_existing_adjacent_pair_exclusions_precede_proof(pair):
    f=Fixture(gap=-.1,first=pair[0],second=pair[1]); n=f.node(attach=False)
    assert n._robot_self_collision(f.q) is None
    assert f.events==['world'] and not hasattr(n,'_static_pair_separation')


def test_world_aabb_rejection_precedes_proof_and_lazy_creation():
    f=Fixture(); f.local[f.second]+=[0.,10.,0.]; n=f.node(attach=False)
    assert n._robot_self_collision(f.q) is None
    assert f.events==['world'] and not hasattr(n,'_static_pair_separation')


def test_custom_surfaces_bypass_all_verdict_caches_and_proof():
    f=Fixture(); n=f.node(); assert n._robot_self_collision(f.q) is None
    old_caches=(dict(n._self_collision_cache),dict(n._static_self_collision_cache))
    surfaces=f.world(f.q,right_positions=f.right,head_positions=f.head)
    surfaces[f.second][...]=surfaces[f.first]
    f.events.clear()
    assert n._robot_self_collision(f.q,collision_surfaces=surfaces)==(f.first,f.second)
    assert not any(isinstance(x,tuple) and x[0]=='proof' for x in f.events)
    assert 'world' not in f.events
    assert old_caches==(n._self_collision_cache,n._static_self_collision_cache)


def test_custom_geometry_cannot_create_hint_helper():
    f=Fixture(gap=-.1); n=f.node(attach=False)
    surfaces=f.world(f.q,right_positions=f.right,head_positions=f.head); f.events.clear()
    assert n._robot_self_collision(f.q,collision_surfaces=surfaces)==(f.first,f.second)
    assert not hasattr(n,'_static_pair_separation')


def test_repeated_state_preserves_existing_full_result_cache():
    f=Fixture(); n=f.node(); assert n._robot_self_collision(f.q) is None
    seen=list(f.events)
    assert n._robot_self_collision(f.q) is None and f.events==seen
    f.q[1]=.2
    assert n._robot_self_collision(f.q)==(f.first,f.second)
    seen=list(f.events)
    assert n._robot_self_collision(f.q)==(f.first,f.second) and f.events==seen


def test_static_rejection_precedes_moving_proof_and_retains_first_pair():
    f=Fixture()
    f.local['head_2_link']=f.local['torso_base_link'].copy()
    n=f.node()
    assert n._robot_self_collision(f.q)==('torso_base_link','head_2_link')
    assert [x for x in f.events if isinstance(x,tuple) and x[0]=='proof']==[('proof','torso_base_link','head_2_link')]


def test_clear_first_moving_pair_does_not_hide_next_rejection():
    f=Fixture()
    f.local['arm_left_5_link']=f.local['torso_base_link'].copy()
    n=f.node()
    assert n._robot_self_collision(f.q)==('torso_base_link','arm_left_5_link')
    assert [x for x in f.events if isinstance(x,tuple) and x[0]=='proof']==[
        ('proof','torso_base_link','arm_left_3_link'),('proof','torso_base_link','arm_left_5_link')]


def test_watertight_flags_and_narrow_exceptions_remain():
    f=Fixture(gap=-.1); f.closed={f.second}; n=f.node()
    assert n._robot_self_collision(f.q)==(f.first,f.second)
    assert f.events[-1][-1]==dict(first_watertight=False,second_watertight=True)
    f=Fixture(gap=-.1); f.narrow_result=RuntimeError('detailed predicate failure')
    with pytest.raises(RuntimeError,match='detailed predicate failure'): f.node()._robot_self_collision(f.q)


def test_empty_surface_retains_original_early_error():
    for path in (ORIGINAL,CURRENT):
        f=Fixture(); f.local[f.second]=np.empty((0,3,3)); n=f.node(path)
        with pytest.raises(ValueError): n._robot_self_collision(f.q)
        assert not any(isinstance(x,tuple) and x[0]=='proof' for x in f.events)


def test_nonfinite_joint_context_rejects_before_geometry():
    f=Fixture(); f.right[0]=np.nan; n=f.node()
    with pytest.raises(ValueError,match='finite'): n._robot_self_collision(f.q)
    assert f.events==[]


def test_lazy_helper_creation_uses_existing_class_once(monkeypatch):
    f=Fixture(); n=f.node(attach=False); created=[]
    imported=types.ModuleType('erc_phase1_solution.static_pair_separation')
    imported.StaticPairSeparation=lambda: created.append(True) or f.proof
    if 'erc_phase1_solution' not in sys.modules:
        package=types.ModuleType('erc_phase1_solution'); package.__path__=[]
        monkeypatch.setitem(sys.modules,'erc_phase1_solution',package)
    monkeypatch.setitem(sys.modules,'erc_phase1_solution.static_pair_separation',imported)
    assert n._robot_self_collision(f.q) is None
    f.q[1]=.001
    assert n._robot_self_collision(f.q) is None
    assert created==[True]


def test_hull_failure_preserves_narrow_and_cached_empty_hints(monkeypatch):
    calls=[]
    def fail(*args): calls.append(1); raise module.QhullError('deliberate fixture')
    monkeypatch.setattr(module,'ConvexHull',fail)
    f=Fixture(); n=f.node()
    assert n._robot_self_collision(f.q)==(f.first,f.second)
    f.q[1]=.001
    assert n._robot_self_collision(f.q)==(f.first,f.second)
    assert len(calls)==2


def test_worst_hint_loop_is_finite_and_no_verdict_reused(monkeypatch):
    helper=module.StaticPairSeparation()
    assert (helper.maximum_links,helper.maximum_pairs,helper.maximum_axes)==(32,128,256)
    many=np.tile([1.,0.,0.],(256,1)); many.setflags(write=False)
    helper._link_hints['left3']=many; helper._link_hints['left5']=many
    helper._pair_hints[('left3','left5')]=np.array([0.,1.,0.])
    calls=[]
    monkeypatch.setattr(helper,'_separates',lambda a,b,d,s: calls.append((a.copy(),b.copy())) or False)
    assert not helper.separated('left3','left5',box(),box([2.,0.,0.]))
    assert len(calls)==513  # One previous pair direction plus two 256-axis lists.
    assert all(a.shape==(36,3) and b.shape==(36,3) for a,b in calls)


def test_bounded_hint_eviction_does_not_clear_overlap():
    helper=module.StaticPairSeparation(maximum_links=2,maximum_pairs=1,maximum_axes=3)
    for i in range(4):
        assert helper.separated(f'left{i}',f'other{i}',box(),box([2.1,0.,0.]))
    assert len(helper._link_hints)<=2 and len(helper._pair_hints)==1
    assert all(len(v)<=3 and not v.flags.writeable for v in helper._link_hints.values())
    assert not helper.separated('left0','other0',box(),box())
