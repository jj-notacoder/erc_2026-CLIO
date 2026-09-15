"""Behavior tests for the unused pure-owner extraction; no source-mirror tests.

The reference tests import the actual ROS node class but never construct it,
start ROS, subscribe, publish or create an action. Small synthetic chains and
closed facets isolate geometry behavior; full official-model parity is a
separate required experiment before production dispatch.
"""
from dataclasses import replace
import itertools
import json
import math
from types import SimpleNamespace
from pathlib import Path
import threading

import numpy as np
import pytest

from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.exact_world_geometry import ordinary_model_producer
from erc_phase1_solution.kinematics import CollisionMesh
from erc_phase1_solution.pure_geometry_owner import (
    GeometryQuery, RobotGeometryOwner,
    PinnedGeometryAssets,
    _BoundedExactCache,
    CARRIED_COLLISION_LINKS, RIGHT_COLLISION_LINKS, HEAD_COLLISION_LINKS,
    HEAD_JOINTS,
)
from erc_phase1_solution.motion_profiles import RIGHT_ARM_JOINTS
from erc_phase1_solution.scene_checked_place import PlaceSceneChecker
from erc_phase1_solution.sample_collision_snapshot import _capture_scene_robot_snapshot
from erc_phase1_solution.runtime_utils import joint_state_cache_key

IDS = dict(source_id='1'*64, model_id='2'*64, scene_id='3'*64)


def query(index=0, **overrides):
    values = dict(request_id=index, epoch=1, **IDS, q=np.zeros(8), right=np.zeros(7),
                  head=np.zeros(2), aperture=.04, loaded=False)
    values.update(overrides)
    return GeometryQuery.capture(**values)


def box(x):
    corners = np.array(list(itertools.product((-.05, .05), repeat=3))) + [x, 0., 1.]
    faces = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
             [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
    return np.array(corners[np.asarray(faces, dtype=np.intp)], dtype=np.float64, order='C')


def robot(cls=RobotGeometryOwner, overlapping=False):
    node = cls.__new__(cls)
    node._lock = threading.Lock(); node._cancel = threading.Event()
    node._self_collision_cache = {}; node._static_self_collision_cache = {}
    node.joints = dict(zip((*RIGHT_ARM_JOINTS, *HEAD_JOINTS), [0.]*9))
    def transform(q, offset):
        value = np.eye(4)
        value[:3, 3] = [q[1] if len(q)>1 else 0., offset, q[0]]
        return value
    node.chain = SimpleNamespace(
        link_transforms=lambda q: {name: transform(q, 0.) for name in CARRIED_COLLISION_LINKS},
        forward=lambda q: transform(q, 0.))
    node.right_chain = SimpleNamespace(
        link_transforms=lambda q: {name: transform(q, 0.2) for name in RIGHT_COLLISION_LINKS})
    node.head_chain = SimpleNamespace(
        link_transforms=lambda q: {name: transform(q, 0.4) for name in HEAD_COLLISION_LINKS})
    models=[]
    for link, shift in [('torso_base_link',0.),('arm_left_3_link',0. if overlapping else 1.),
                        ('head_1_link',3.),('arm_right_4_link',4.)]:
        model=ModelLocalMesh(box(shift));surface=model.snapshot()[0]
        models.append(CollisionMesh(link,surface,
            np.array([surface.min(axis=(0,1)),surface.max(axis=(0,1))]),True,model))
    node.carried_collision_meshes=tuple(models)
    node.assets=SimpleNamespace(**{k:v for k,v in IDS.items() if k!='scene_id'})
    node._epoch=1;node._scene_id=IDS['scene_id'];node._last_request_id=-1
    node._queries_seen=0;node._apertures=set()
    return node


class Bin:
    def __init__(self, reject=False): self.reject=reject
    def intersects(self, *args): return self.reject
    def book_intersects(self, *args): return False


class Tool:
    watertight={'tool':True}
    def local_surfaces(self, aperture): return {'tool':box(8.)}


def scene(node, reject=False):
    return PlaceSceneChecker(node,Bin(reject),Tool(),np.array(list(itertools.product((-.02,.02),repeat=3))))


def test_query_owns_exact_values_and_caller_mutation_cannot_change_it():
    q=np.zeros(8);q[1]=-0.;right=np.arange(7,dtype=float);head=np.array([.2,.3])
    sample=query(q=q,right=right,head=head)
    expected=(q.tobytes(),right.tobytes(),head.tobytes())
    q[:]=9.;right[:]=9.;head[:]=9.
    assert (sample.q,sample.right,sample.head)==expected
    assert np.signbit(np.frombuffer(sample.q,dtype=np.float64)[1])
    assert not np.frombuffer(sample.q,dtype=np.float64).flags.writeable
    assert sample.validated()==sample
    changed=replace(sample,q=np.zeros(8).tobytes())
    assert changed.input_sha256!=sample.input_sha256


@pytest.mark.parametrize('name,count',[('q',8),('right',7),('head',2)])
@pytest.mark.parametrize('bad',['nan','infinity','wrong_length'])
def test_invalid_vectors_are_rejected(name,count,bad):
    values=np.zeros(count)
    if bad=='wrong_length':values=values[:-1]
    else:values[0]=math.nan if bad=='nan' else math.inf
    with pytest.raises(ValueError):query(**{name:values})


@pytest.mark.parametrize('field,value',[('request_id',True),('request_id',-1),('epoch',-1),
    ('source_id','wrong'),('model_id','g'*64),('scene_id',''),('loaded',1),
    ('aperture',math.inf),('aperture',math.nan),('aperture',True)])
def test_invalid_query_metadata_is_rejected(field,value):
    with pytest.raises(ValueError):query(**{field:value})


@pytest.mark.parametrize('field,value',[('q',b'bad'),('right',bytearray(56)),('head',b'')])
def test_wire_query_is_revalidated_before_geometry(field,value):
    node=robot();node._scene=scene(node)
    with pytest.raises(ValueError):node.evaluate(replace(query(),**{field:value}))
    assert node._scene.samples==0


@pytest.mark.parametrize('field,value',[('epoch',2),('source_id','4'*64),
    ('model_id','5'*64),('scene_id','6'*64)])
def test_mixed_source_model_epoch_or_scene_never_reaches_geometry(field,value):
    node=robot();node._scene=scene(node)
    with pytest.raises(RuntimeError,match='identity mismatch'):
        node.evaluate(query(**{field:value}))
    assert node._scene.samples==0


def test_ordered_distinct_queries_allow_gaps_but_reject_duplicates_and_reordering():
    node=robot();node._scene=scene(node)
    assert node.evaluate(query(2)).verdict
    assert node.evaluate(query(8)).verdict
    for index in (8,7,0):
        with pytest.raises(RuntimeError,match='increasing'):node.evaluate(query(index))
    assert node._scene.samples==2


def test_cancel_is_checked_before_a_previously_successful_sample_cache_hit():
    node=robot();node._scene=scene(node)
    first=node.evaluate(query(0));node.cancel();second=node.evaluate(query(1))
    assert first.verdict and not second.verdict
    assert json.loads(second.rejection_json)=={'reason':'cancelled'}
    assert second.owner_samples==first.owner_samples


def test_early_rejection_has_no_fabricated_minimum():
    node=robot();node._scene=scene(node,True)
    reply=node.evaluate(query())
    assert not reply.verdict and reply.owner_prefix_minimum is None
    assert json.loads(reply.rejection_json)['reason']=='robot_bin'


def test_geometry_exception_latches_cancellation_for_later_queries():
    node=robot();node._scene=scene(node)
    def broken(*args):raise ValueError('bad geometry')
    node._scene.obstacle.intersects=broken
    with pytest.raises(ValueError,match='bad geometry'):node.evaluate(query())
    assert node._cancel.is_set()
    reply=node.evaluate(query(1))
    assert not reply.verdict and json.loads(reply.rejection_json)['reason']=='cancelled'


@pytest.mark.parametrize('overlapping',[False,True])
@pytest.mark.parametrize('delta',[0.,.2,.4])
def test_actual_node_and_extracted_owner_match_geometry_order_away_from_rounding_aliases(overlapping,delta):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    original=robot(ManipulationNode,overlapping);owned=robot(RobotGeometryOwner,overlapping)
    for value in (0.,delta,delta,0.):
        q=np.zeros(8);q[1]=value
        right=np.full(7,value);head=np.full(2,value)
        a=original._world_collision_surfaces(q,right_positions=right,head_positions=head)
        b=owned._world_collision_surfaces(q,right_positions=right,head_positions=head)
        assert tuple(a)==tuple(b)
        assert all(a[k].tobytes()==b[k].tobytes() for k in a)
        assert original._robot_self_collision(q,right_positions=right,head_positions=head)==owned._robot_self_collision(q,right_positions=right,head_positions=head)
        assert list(original._self_collision_cache.values())==list(owned._self_collision_cache.values())
        assert list(original._static_self_collision_cache.values())==list(owned._static_self_collision_cache.values())


@pytest.mark.parametrize('overlapping',[False,True])
def test_snapshot_provenance_reuses_owned_geometry_with_original_result(overlapping):
    ordinary=robot(overlapping=overlapping);shared=robot(overlapping=overlapping)
    q=np.zeros(8);right=np.zeros(7);head=np.zeros(2)
    robot_world=shared._world_collision_surfaces(q,right_positions=right,head_positions=head)
    snapshot=_capture_scene_robot_snapshot(shared,q,right,head,robot_world)
    assert snapshot is not None and ordinary_model_producer(shared)
    expected=ordinary._robot_self_collision(q,right_positions=right,head_positions=head)
    actual=shared._robot_self_collision(q,right_positions=right,head_positions=head,_scene_snapshot=snapshot)
    assert actual==expected
    assert snapshot.resolved(shared,q,right,head) is not None
    changed=right.copy();changed[0]=np.nextafter(0.,1.)
    assert snapshot.resolve(shared,q,changed,head) is None


def test_custom_geometry_producer_retains_fallback():
    node=robot()
    assert ordinary_model_producer(node)
    node._world_collision_surfaces=lambda *a,**k:{}
    assert not ordinary_model_producer(node)


def test_successful_cache_hit_preserves_prior_rejection_and_minimum():
    node=robot();node._scene=scene(node)
    first=node.evaluate(query())
    node._scene.last_rejection={'reason':'earlier_candidate'}
    second=node.evaluate(query(1))
    assert first.verdict and second.verdict
    assert first.input_sha256==query().input_sha256
    assert second.input_sha256==query(1).input_sha256
    assert second.owner_cache_hits==first.owner_cache_hits+1
    assert second.owner_prefix_minimum==first.owner_prefix_minimum
    assert json.loads(second.rejection_json)=={'reason':'earlier_candidate'}


@pytest.mark.parametrize('static',[False,True])
@pytest.mark.parametrize('clear_first',[False,True])
def test_exact_owner_correctly_rechecks_nearby_states_that_share_legacy_rounded_key(static,clear_first):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    def boundary_node(cls):
        node=robot(cls)
        moving='head_1_link' if static else 'arm_left_3_link'
        models=[]
        for link,x in [('torso_base_link',0.),(moving,.1)]:
            mesh=ModelLocalMesh(box(x));surface=mesh.snapshot()[0]
            models.append(CollisionMesh(link,surface,np.array([
                surface.min(axis=(0,1)),surface.max(axis=(0,1))]),True,mesh))
        node.carried_collision_meshes=tuple(models)
        def transforms(q,names):
            result={name:np.eye(4) for name in names}
            if moving in result:result[moving][0,3]=q[1]-1.
            return result
        node.chain=SimpleNamespace(link_transforms=lambda q:transforms(q,CARRIED_COLLISION_LINKS))
        if static:
            node.head_chain=SimpleNamespace(link_transforms=lambda q:transforms(q,HEAD_COLLISION_LINKS))
        return node
    def state(value):
        q=np.zeros(8);head=np.zeros(2);right=np.zeros(7)
        if static:head[0]=value
        else:q[1]=value
        return q,right,head
    clear=state(1.+2e-11);hit=state(1.-2e-11)
    assert joint_state_cache_key(*clear)==joint_state_cache_key(*hit)
    first,second=(clear,hit) if clear_first else (hit,clear)
    old=boundary_node(ManipulationNode);new=boundary_node(RobotGeometryOwner)
    def call(node,values):
        q,right,head=values
        return node._robot_self_collision(q,right_positions=right,head_positions=head)
    first_expected=call(boundary_node(ManipulationNode),first)
    second_expected=call(boundary_node(ManipulationNode),second)
    assert (first_expected is None)==clear_first
    assert (second_expected is None)!=clear_first
    assert call(old,first)==call(new,first)==first_expected
    if static:
        # Isolate the persistent static cache from the full-body cache.
        old._self_collision_cache.clear();new._self_collision_cache.clear()
    assert call(old,second)==first_expected
    assert call(new,second)==second_expected
    assert first_expected!=second_expected


@pytest.mark.parametrize('capacity',[0,-1,True])
def test_exact_cache_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):_BoundedExactCache(capacity)


def test_exact_cache_keeps_none_results_and_eviction_only_removes_old_entry():
    cache=_BoundedExactCache(2)
    cache[b'a']=None;cache[b'b']=('one','two');cache[b'b']=None
    assert b'a' in cache and b'b' in cache and cache[b'b'] is None
    cache[b'c']=('three','four')
    assert b'a' not in cache and list(cache)==[b'b',b'c']


@pytest.mark.parametrize('limit',['queries','apertures'])
def test_owner_resource_bound_fails_closed_before_excess_geometry(limit):
    node=robot();node._scene=scene(node)
    if limit=='queries':node.maximum_queries=1
    else:node.maximum_distinct_apertures=1
    assert node.evaluate(query()).verdict
    with pytest.raises(RuntimeError,match='resource bound'):
        node.evaluate(query(1,aperture=.041))
    assert node._scene.samples==1 and node._cancel.is_set()


@pytest.mark.parametrize('changed',['bin','table','sdf',None])
def test_scene_model_registration_gates_reject_changed_declared_assets(monkeypatch,changed):
    from erc_phase1_solution import pure_geometry_owner as owner_module
    from erc_phase1_solution.table_scene import TABLE_MESH_SHA256
    assets=PinnedGeometryAssets.__new__(PinnedGeometryAssets)
    assets.bin_mesh=Path('bin.stl');assets.table_mesh=Path('table.stl');assets.table_sdf=Path('table.sdf')
    expected={assets.bin_mesh:'a'*64,assets.table_mesh:TABLE_MESH_SHA256,
        assets.table_sdf:'90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f'}
    if changed is not None:
        expected[{'bin':assets.bin_mesh,'table':assets.table_mesh,'sdf':assets.table_sdf}[changed]]='b'*64
    monkeypatch.setattr(owner_module,'_digest',lambda path:expected[path])
    if changed is None:assets.verify_scene_models({'mesh_sha256':'a'*64})
    else:
        with pytest.raises(RuntimeError,match='model_changed'):
            assets.verify_scene_models({'mesh_sha256':'a'*64})
