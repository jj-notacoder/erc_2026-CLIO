import ast
import copy
import hashlib
from pathlib import Path
import types

import numpy as np
import pytest

from erc_phase1_solution.kinematics import _box_triangles, _rpy_matrix, PreparedTriangleMesh, triangle_meshes_intersect
from erc_phase1_solution.runtime_utils import joint_state_cache_key
from erc_phase1_solution.static_pair_separation import StaticPairSeparation

ROOT = Path(__file__).resolve().parents[1]


def method(path, class_name):
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_robot_self_collision')


BASE = method(Path(__file__).parent/'fixtures/static_pair_baseline.py', 'Baseline')
NEW = method(ROOT/'erc_phase1_solution/manipulation_node.py', 'ManipulationNode')


def instantiate(selected):
    scope = dict(np=np, joint_state_cache_key=joint_state_cache_key,
        PreparedTriangleMesh=PreparedTriangleMesh, triangle_meshes_intersect=triangle_meshes_intersect,
        CARRIED_COLLISION_LINKS=('base', 'right', 'head', 'left'),
        LEFT_COLLISION_LINKS=frozenset(('left',)), DIRECTLY_CONNECTED_COLLISION_LINKS=frozenset())
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),copy.deepcopy(selected)],type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), '<actual-method>', 'exec'), scope)
    node=types.SimpleNamespace(_self_collision_cache={}, _static_self_collision_cache={})
    node._resolved_right_positions=lambda value:np.asarray(value)
    node._resolved_head_positions=lambda value:np.asarray(value)
    node._watertight_collision_links=lambda:frozenset(('base','right','head','left'))
    node._robot_self_collision=types.MethodType(scope['_robot_self_collision'], node)
    node._world_collision_surfaces=lambda *a,**kw:node.surfaces
    return node


def surfaces(offset=.2):
    a=_box_triangles((2.,.01,.01))@_rpy_matrix((0.,0.,np.pi/4)).T
    return dict(base=a,right=a+(0.,offset,0.),head=_box_triangles((.2,.2,.2))+(0.,0.,5.),left=_box_triangles((.2,.2,.2))+(0.,0.,8.))


def call(node, index, *, custom=None):
    return node._robot_self_collision([.3+index*.001]+[0.]*7, custom,
        right_positions=[index*1e-6]*7,head_positions=[index*2e-6,0.])


def test_only_added_static_certificate_changes_complete_pair_predicate_ast():
    fixture = Path(__file__).parent/'fixtures/static_pair_baseline.py'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == 'b275e493bf3684b59715c693eaa458d3e7eeb9e690db672ae14137afd62a79cc'
    revised=copy.deepcopy(NEW)
    nested=next(n for n in revised.body if isinstance(n,ast.FunctionDef) and n.name=='links_intersect')
    added=[n for n in nested.body if isinstance(n,ast.If) and '_static_pair_separation' in ast.unparse(n)]
    assert len(added)==1
    nested.body.remove(added[0])
    # The snapshot suite separately restores the complete body/sample ASTs.
    original=next(n for n in BASE.body if isinstance(n,ast.FunctionDef) and n.name=='links_intersect')
    assert ast.dump(nested,include_attributes=False)==ast.dump(original,include_attributes=False)


def test_changing_torso_right_head_and_collision_transitions_matches_baseline():
    old,new=instantiate(BASE),instantiate(NEW)
    for index,offset in enumerate((.2,.3,0.,.4,0.,.2)):
        old.surfaces=new.surfaces=surfaces(offset)
        assert call(new,index)==call(old,index)
        assert new._self_collision_cache==old._self_collision_cache
        assert new._static_self_collision_cache==old._static_self_collision_cache


def test_original_first_collision_order_preserved():
    old,new=instantiate(BASE),instantiate(NEW)
    shape=_box_triangles((1.,1.,1.))
    old.surfaces=new.surfaces=dict(base=shape,right=shape,head=shape,left=shape)
    assert call(old,0)==call(new,0)==('base','right')


def test_custom_surface_bypasses_certificate_and_all_old_caches():
    old,new=instantiate(BASE),instantiate(NEW)
    class Bomb:
        def separated(self,*args):pytest.fail('custom surfaces used certificate')
    new._static_pair_separation=Bomb()
    for index,offset in enumerate((.2,0.)):
        data=surfaces(offset)
        assert call(new,index,custom=data)==call(old,index,custom=data)
    assert not new._self_collision_cache and not new._static_self_collision_cache


def test_moving_certificate_retains_overlap_rejection():
    new=instantiate(NEW);seen=[]
    class Spy(StaticPairSeparation):
        def separated(self,a,b,*args):
            result=super().separated(a,b,*args)
            seen.append((a,b,result))
            if 'left' in (a,b):
                assert result is False  # Identical current meshes must fall through.
            return result
    new._static_pair_separation=Spy();new.surfaces=surfaces()
    new.surfaces['left']=new.surfaces['base'].copy()
    assert call(new,0)==('base','left')
    assert ('base','left',False) in seen


def test_existing_whole_cache_admission_is_unchanged():
    old,new=instantiate(BASE),instantiate(NEW)
    key=joint_state_cache_key([.3]+[0.]*7,[0.]*7,[0.]*2)
    for node in (old,new):
        node._self_collision_cache[key]=('cached','result')
        node._world_collision_surfaces=lambda *a,**kw:pytest.fail('cache hit rebuilt surfaces')
    assert call(old,0)==call(new,0)==('cached','result')


def test_degenerate_refinement_occurs_only_through_explicit_new_certificate():
    old,new=instantiate(BASE),instantiate(NEW)
    old.surfaces=new.surfaces=surfaces()
    assert call(old,0) is None and call(new,0) is None
    a=np.asarray([[(0.,0.,0.),(1.,1.,0.),(2.,2.,0.)]])
    old.surfaces=new.surfaces=dict(base=a,right=a+(0.,.2,0.),head=surfaces()['head'],left=surfaces()['left'])
    assert call(old,1)==('base','right')
    assert call(new,1) is None
    new.surfaces['right']=a
    old.surfaces=new.surfaces
    assert call(old,2)==call(new,2)==('base','right')
