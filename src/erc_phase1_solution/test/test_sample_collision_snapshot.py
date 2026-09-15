"""Prepared actual-method differentials; no execution during the live trial."""
import ast
from tool_local_ast_support import restore_tool_handle_sample
import copy
import itertools
import math
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest

from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.kinematics import CollisionMesh
from erc_phase1_solution.runtime_utils import joint_state_cache_key
from erc_phase1_solution.sample_collision_snapshot import _capture_scene_robot_snapshot

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_NODE = ROOT/'test/fixtures/sample_collision_original_node.py'
ORIGINAL_SCENE = ROOT/'test/fixtures/sample_collision_original_scene.py'


def box(shift=0.):
    vertices = np.array(list(itertools.product((-.1, .1), repeat=3))) + [shift, 0., 1.]
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=int)
    return np.array(vertices[faces], copy=True, dtype=np.float64, order='C')


def signature(value):
    return value.shape, value.dtype.str, value.tobytes()


class Bin:
    def intersects(self, *args): return False
    def book_intersects(self, *args): return False


class Tool:
    watertight = {'tool': True}
    def local_surfaces(self, aperture): return {'tool': box(3.)}


def actual_class(path, fixture, scene=False):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = ({'_finite', '_SampleWorldAabbs', 'PlaceSceneChecker'} if scene else
             {'LEFT_COLLISION_LINKS', 'RIGHT_COLLISION_LINKS', 'HEAD_COLLISION_LINKS',
              'CARRIED_COLLISION_LINKS', 'DIRECTLY_CONNECTED_COLLISION_LINKS'})
    nodes = [copy.deepcopy(n) for n in tree.body if
             isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if not scene:
        nodes += [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ManipulationNode')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_robot_self_collision')
        aliases = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_scene_snapshot_body']
        nodes += [ast.ClassDef(name='ActualNode', bases=[], keywords=[], decorator_list=[],
                              body=[copy.deepcopy(method), *aliases])]
    scope = dict(np=np, math=math, joint_state_cache_key=joint_state_cache_key,
        PreparedTriangleMesh=fixture.prepare, triangle_meshes_intersect=fixture.narrow,
        oriented_box_intersects_triangles=lambda *a, **k: False,
        separated_on_axes=lambda *a: False, PALM_COLLISION_LINK='palm',
        NominalBinObstacle=Bin, TableSceneObstacle=type('Table', (), {}),
        ScreenEnvelope=type('Screen', (), {}), ShelfCradleGeometry=Tool)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), scope)
    if not scene:
        scope['ManipulationNode'] = scope['ActualNode']
    return scope['PlaceSceneChecker' if scene else 'ActualNode']


class Fixture:
    def __init__(self, *, changed=True, rejection=False, immutable=True):
        self.events = []; self.world_calls = 0; self.rejection = rejection
        self.q=np.zeros(8); self.right=np.zeros(7); self.head=np.zeros(2)
        node_path = ROOT/'erc_phase1_solution/manipulation_node.py' if changed else ORIGINAL_NODE
        scene_path = ROOT/'erc_phase1_solution/scene_checked_place.py' if changed else ORIGINAL_SCENE
        self.node=actual_class(node_path, self)()
        meshes=[]
        for index, link in enumerate(('torso_base_link', 'head_1_link', 'arm_left_3_link')):
            raw=box(index*.02)
            handle=ModelLocalMesh(raw) if immutable else None
            surface=handle.snapshot()[0] if handle else raw
            meshes.append(CollisionMesh(link, surface,
                np.asarray([raw.min(axis=(0,1)),raw.max(axis=(0,1))]), True, handle))
        n=self.node
        n.carried_collision_meshes=tuple(meshes)
        n._self_collision_cache={};n._static_self_collision_cache={}
        n._resolved_right_positions=lambda p: np.array(self.right if p is None else p,copy=True)
        n._resolved_head_positions=lambda p: np.array(self.head if p is None else p,copy=True)
        # The actual body method's producer-identity gate compares these class
        # defaults. Instance/subclass overrides are separate fallback cases.
        type(n)._world_collision_surfaces=lambda node,q,**context:self.world(q,**context)
        type(n)._collision_link_transforms=lambda node,q,**context:self.transforms(q,**context)
        n._watertight_collision_links=lambda:frozenset(m.link for m in n.carried_collision_meshes)
        n._static_pair_separation=NS(separated=self.proof)
        n._cancel=threading.Event()
        n.chain=NS(forward=lambda q: self.matrix(q,self.right,self.head))
        self.scene=actual_class(scene_path,self,True)

    def matrix(self,q,right,head):
        t=np.eye(4);t[0,3]=q[0]+right[0]+head[0]
        return t

    def transforms(self,q,**context):
        return {m.link:self.matrix(q,context['right_positions'],context['head_positions'])
                for m in self.node.carried_collision_meshes}

    def robot(self,q=None,right=None,head=None):
        q=self.q if q is None else q;right=self.right if right is None else right;head=self.head if head is None else head
        transforms=self.transforms(q,right_positions=right,head_positions=head)
        grouped={}
        for mesh in self.node.carried_collision_meshes:
            t=transforms[mesh.link]
            grouped.setdefault(mesh.link,[]).append(mesh.triangles@t[:3,:3].T+t[:3,3])
        return {link:values[0] if len(values)==1 else np.concatenate(values) for link,values in grouped.items()}

    def world(self,q,**context):
        self.world_calls+=1
        return self.robot(q,context['right_positions'],context['head_positions'])

    def proof(self,a,b,first,second):
        self.events.append(('proof',a,b,signature(first),signature(second)))
        return False

    def prepare(self,surface):
        self.events.append(('prepare',signature(surface)))
        return surface

    def narrow(self,first,second,**flags):
        self.events.append(('narrow',signature(first),signature(second),flags))
        if isinstance(self.rejection,Exception): raise self.rejection
        return self.rejection

    def snapshot(self,robot=None):
        return _capture_scene_robot_snapshot(self.node,self.q,self.right,self.head,
                                             self.robot() if robot is None else robot)

    def call(self,snapshot=None,**kwargs):
        return self.node._robot_self_collision(self.q,right_positions=self.right,
            head_positions=self.head,**({'_scene_snapshot':snapshot} if snapshot is not None else {}),**kwargs)


@pytest.mark.parametrize('rejection',[False,True])
def test_actual_body_method_order_first_result_and_cache_semantics_equal(rejection):
    original=Fixture(changed=False,rejection=rejection)
    changed=Fixture(rejection=rejection)
    expected=original.call(); snapshot=changed.snapshot()
    assert changed.call(snapshot)==expected
    assert changed.events==original.events
    assert changed.node._self_collision_cache==original.node._self_collision_cache
    assert changed.node._static_self_collision_cache==original.node._static_self_collision_cache
    assert original.world_calls==1 and changed.world_calls==0
    assert snapshot.resolved(changed.node,changed.q,changed.right,changed.head) is not None


def test_full_state_cache_hit_does_not_freeze_or_reduce_shared_geometry():
    fixture=Fixture();fixture.call()
    snapshot=fixture.snapshot();before=list(fixture.events)
    assert fixture.call(snapshot) is None
    assert fixture.events==before and fixture.world_calls==1
    assert snapshot.resolved(fixture.node,fixture.q,fixture.right,fixture.head) is None


@pytest.mark.parametrize('field',['q','right','head'])
def test_exact_context_change_falls_back_even_below_rounded_cache_precision(field):
    fixture=Fixture();snapshot=fixture.snapshot()
    getattr(fixture,field)[0]=np.nextafter(0.,1.)
    assert fixture.call(snapshot) is None
    assert fixture.world_calls==1
    assert snapshot.resolved(fixture.node,fixture.q,fixture.right,fixture.head) is None


def test_different_owner_or_model_replacement_falls_back():
    fixture=Fixture();snapshot=fixture.snapshot();other=Fixture()
    assert other.call(snapshot) is None and other.world_calls==1
    mesh=fixture.node.carried_collision_meshes[0]
    replacement=ModelLocalMesh(box(.04))
    changed=CollisionMesh(mesh.link,replacement.snapshot()[0],mesh.bounds,True,replacement)
    fixture.node.carried_collision_meshes=(changed,*fixture.node.carried_collision_meshes[1:])
    assert fixture.call(snapshot) is None and fixture.world_calls==1


def test_explicit_custom_surfaces_keep_original_bypass_and_do_not_consume_snapshot():
    fixture=Fixture();snapshot=fixture.snapshot()
    raw=fixture.robot()
    assert fixture.call(snapshot,collision_surfaces=raw) is None
    assert fixture.world_calls==0 and fixture.node._self_collision_cache=={}
    assert fixture.node._static_self_collision_cache=={}
    assert not any(e[0]=='proof' for e in fixture.events)
    assert snapshot.resolved(fixture.node,fixture.q,fixture.right,fixture.head) is None


@pytest.mark.parametrize('method',['_world_collision_surfaces','_collision_link_transforms'])
def test_custom_geometry_producer_override_uses_original_path(method):
    fixture=Fixture();snapshot=fixture.snapshot()
    original=getattr(fixture.node,method)
    setattr(fixture.node,method,lambda *args,**kwargs:original(*args,**kwargs))
    assert fixture.call(snapshot) is None
    assert fixture.world_calls==1
    assert snapshot.resolved(fixture.node,fixture.q,fixture.right,fixture.head) is None


def test_custom_body_consumer_keeps_original_signature_without_private_keyword():
    fixture=Fixture(); calls=[]
    def custom(q, *, head_positions, right_positions):
        calls.append((q.copy(), head_positions.copy(), right_positions.copy()))
        return None
    fixture.node._robot_self_collision=custom
    assert fixture.snapshot() is None
    checker=fixture.scene(fixture.node,Bin(),Tool(),np.ones((8,3)))
    assert checker.sample(fixture.q,.017,False)
    assert len(calls)==1


def test_immutable_views_and_bounds_survive_prior_array_metadata_mutation():
    fixture=Fixture();robot=fixture.robot();snapshot=fixture.snapshot(robot)
    first=snapshot.resolve(fixture.node,fixture.q,fixture.right,fixture.head)
    expected={k:(signature(v),signature(first[1][k])) for k,v in first[0].items()}
    for value in robot.values():value[:]=999.
    for value in first[0].values():
        with pytest.raises(ValueError):value.flags.writeable=True
        value.shape=(value.size,)
    for value in first[1].values():value.shape=(6,)
    second=snapshot.resolved(fixture.node,fixture.q,fixture.right,fixture.head)
    assert {k:(signature(v),signature(second[1][k])) for k,v in second[0].items()}==expected


@pytest.mark.parametrize('variant',['raw_model','view','subclass','wrong_order','wrong_shape'])
def test_unsupported_sources_fall_back(variant):
    fixture=Fixture(immutable=variant!='raw_model');robot=fixture.robot()
    key=next(iter(robot))
    if variant=='view':robot[key]=robot[key].view()
    elif variant=='subclass':
        class Custom(np.ndarray):pass
        robot[key]=robot[key].view(Custom)
    elif variant=='wrong_order':robot=dict(reversed(tuple(robot.items())))
    elif variant=='wrong_shape':robot[key]=robot[key][:-1].copy()
    assert fixture.snapshot(robot) is None


def test_compound_groups_retain_all_facets_in_order():
    fixture=Fixture();mesh=fixture.node.carried_collision_meshes[0]
    handle=ModelLocalMesh(box(.08))
    extra=CollisionMesh(mesh.link,handle.snapshot()[0],mesh.bounds,True,handle)
    fixture.node.carried_collision_meshes=(mesh,extra,*fixture.node.carried_collision_meshes[1:])
    robot=fixture.robot();snapshot=fixture.snapshot(robot)
    surfaces,bounds=snapshot.resolve(fixture.node,fixture.q,fixture.right,fixture.head)
    assert tuple(surfaces)==tuple(robot)
    assert all(signature(surfaces[k])==signature(robot[k]) for k in robot)
    assert all(signature(bounds[k])==signature(np.asarray([np.min(v,axis=(0,1)),np.max(v,axis=(0,1))])) for k,v in robot.items())


@pytest.mark.parametrize('changed',[False,True])
def test_original_predicate_error_is_preserved(changed):
    fixture=Fixture(changed=changed,rejection=RuntimeError('narrow failure'))
    with pytest.raises(RuntimeError,match='narrow failure'):
        fixture.call(fixture.snapshot() if changed else None)


def test_actual_scene_pipeline_reuses_bounds_and_preserves_exact_minimum_z():
    rows=[]
    for changed in (False,True):
        fixture=Fixture(changed=changed)
        checker=fixture.scene(fixture.node,Bin(),Tool(),np.array(list(itertools.product((2.,2.1),repeat=3))))
        result=checker.sample(fixture.q,.017,True)
        rows.append((result,checker.minimum_moving_left_z,checker.last_rejection,
                     fixture.events,checker.samples,checker.cache_hits,fixture.world_calls))
    assert rows[0][:-1]==rows[1][:-1]
    assert rows[0][-1]==1 and rows[1][-1]==0


def test_custom_scene_obstacle_keeps_normal_body_call_and_raw_path():
    class CustomBin(Bin):pass
    fixture=Fixture()
    checker=fixture.scene(fixture.node,CustomBin(),Tool(),np.ones((8,3)))
    assert checker.sample(fixture.q,.017,False)
    assert fixture.world_calls==1


def test_complete_original_body_and_sample_asts_recovered_from_optional_plumbing():
    assert hashlib.sha256(ORIGINAL_NODE.read_bytes()).hexdigest() == 'ab3fcfa5d572c8db5cc54b9056db535ab336ea3915342db3a7ee9a0691ab49e5'
    assert hashlib.sha256(ORIGINAL_SCENE.read_bytes()).hexdigest() == 'b660946ad9a18870f674351c0f77f7ddfe80404e36a0c63b9fdee1f23b61500f'
    def method(path, clsname, name):
        source=path.read_text(encoding='utf-8')
        if path.name == 'scene_checked_place.py':
            from world_geometry_test_support import restore_world_geometry_source
            source=restore_world_geometry_source(source, path.name)
        tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==clsname)
        return copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
    def assigned(node,name):
        return (isinstance(node,ast.Assign) and len(node.targets)==1
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name)
    changed=method(ROOT/'erc_phase1_solution/manipulation_node.py','ManipulationNode','_robot_self_collision')
    index=next(i for i,arg in enumerate(changed.args.kwonlyargs) if arg.arg=='_scene_snapshot')
    assert isinstance(changed.args.kw_defaults[index],ast.Constant) and changed.args.kw_defaults[index].value is None
    del changed.args.kwonlyargs[index];del changed.args.kw_defaults[index]
    index=next(i for i,n in enumerate(changed.body) if assigned(n,'shared_geometry'))
    assert isinstance(changed.body[index+1],ast.If)
    del changed.body[index:index+2]
    for item in changed.body:
        if assigned(item,'surfaces') or assigned(item,'bounds'):
            assert isinstance(item.value,ast.IfExp)
            item.value=item.value.body
    original=method(ORIGINAL_NODE,'ManipulationNode','_robot_self_collision')
    assert ast.dump(changed,include_attributes=False)==ast.dump(original,include_attributes=False)
    changed=method(ROOT/'erc_phase1_solution/scene_checked_place.py','PlaceSceneChecker','sample')
    changed=restore_tool_handle_sample(changed)
    index=next(i for i,n in enumerate(changed.body) if assigned(n,'snapshot'))
    assert isinstance(changed.body[index+1],ast.If) and assigned(changed.body[index+2],'collision')
    del changed.body[index:index+2]
    assert isinstance(changed.body[index].value,ast.IfExp)
    changed.body[index].value=changed.body[index].value.body
    index=next(i for i,n in enumerate(changed.body) if assigned(n,'shared_geometry'))
    branch=changed.body[index+1]
    assert isinstance(branch,ast.If) and len(branch.orelse)==1 and assigned(branch.orelse[0],'bounds')
    changed.body[index:index+2]=branch.orelse
    original=method(ORIGINAL_SCENE,'PlaceSceneChecker','sample')
    assert ast.dump(changed,include_attributes=False)==ast.dump(original,include_attributes=False)
