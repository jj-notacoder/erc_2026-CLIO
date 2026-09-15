"""Actual predicate/counter differentials for exact empty-pickup facet reuse."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from erc_phase1_solution import empty_pickup_collision as current
from erc_phase1_solution.exact_local_bounds import ModelLocalMesh
from erc_phase1_solution.exact_world_geometry import ExactWorldGeometry
from erc_phase1_solution.kinematics import CollisionMesh
from erc_phase1_solution.rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS
from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
from test_sample_collision_snapshot import Fixture, box, signature

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('erc_phase1_solution._empty_world_original',
    ROOT/'test/fixtures/empty_world_geometry_original.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


def make(module, *, rejection=False, immutable=True, compound=False, custom=None):
    f=Fixture(immutable=immutable,rejection=rejection);n=f.node
    n.chain.lower=np.full(8,-3.);n.chain.upper=np.full(8,3.)
    n.gripper_open=.069;n.carried_transition_samples=61
    if compound:
        mesh=n.carried_collision_meshes[0];handle=ModelLocalMesh(box(.08))
        n.carried_collision_meshes=(mesh,CollisionMesh(mesh.link,handle.snapshot()[0],mesh.bounds,True,handle),*n.carried_collision_meshes[1:])
    if custom:
        function=getattr(n,custom)
        setattr(n,custom,lambda *a,**k:function(*a,**k))
    geometry=object.__new__(ShelfCradleGeometry)
    geometry.local_surfaces=lambda aperture:{name:box(30.+i*3.) for i,name in enumerate(LEFT_GRIPPER_COLLISION_LINKS)}
    geometry.watertight={name:True for name in LEFT_GRIPPER_COLLISION_LINKS}
    screen=object.__new__(module.ScreenEnvelope)
    screen.world=lambda torso,head:np.array([[x,y,z] for x in (100.,101.) for y in (100.,101.) for z in (100.,101.)])
    c=module.EmptyPickupCollision(n,f.q,f.right,f.head,.069,geometry=geometry,screen=screen)
    return f,c


def diagnostics(c):
    return dict(checked=c.checked_samples,hits=c.cache_hits,rejection=c.last_rejection,
        rejected_q=c.last_rejected_q,rejected_aperture=c.last_rejected_aperture,cache=c.cache)


@pytest.mark.parametrize('rejection',[False,True])
@pytest.mark.parametrize('immutable',[False,True])
@pytest.mark.parametrize('compound',[False,True])
def test_original_predicate_order_reasons_and_diagnostics_match(rejection,immutable,compound):
    a,old=make(original,rejection=rejection,immutable=immutable,compound=compound)
    b,new=make(current,rejection=rejection,immutable=immutable,compound=compound)
    for shift,aperture in [(0.,.01),(0.,.02),(0.,.02),(.1,.02),(.1,.03),(0.,.04)]:
        q=np.zeros(8);q[0]=shift
        assert old.sample(q,aperture)==new.sample(q,aperture)
        assert a.events==b.events
        assert diagnostics(old)==diagnostics(new)
    if immutable:
        assert new._world_geometry_cache.hits>0
        assert b.world_calls==0 and a.world_calls>0
    else:
        assert new._world_geometry_cache is None
        assert a.world_calls==b.world_calls


@pytest.mark.parametrize('method',['_robot_self_collision','_world_collision_surfaces','_collision_link_transforms'])
def test_custom_producers_retain_original_calls(method):
    a,old=make(original,custom=method);b,new=make(current,custom=method)
    assert old.sample(a.q,.069)==new.sample(b.q,.069)
    assert new._world_geometry_cache is None
    assert a.events==b.events and a.world_calls==b.world_calls


@pytest.mark.parametrize('compound',[False,True])
def test_current_facets_and_bounds_are_bitwise_equal_across_hit_miss_and_context_changes(compound):
    f,c=make(current,compound=compound)
    for shift in (0.,.1,.1,np.nextafter(.1,1.),0.):
        q=f.q.copy();q[0]=shift
        expected=f.robot(q,c.right,c.head)
        actual,handles=c._cached_robot_world(q)
        assert tuple(actual)==tuple(expected)
        for name in expected:
            assert signature(actual[name])==signature(expected[name])
            if name in handles:
                assert signature(handles[name].views()[1])==signature(np.asarray([
                    expected[name].min(axis=(0,1)),expected[name].max(axis=(0,1))]))
    if compound:assert 'torso_base_link' not in handles
    assert c._world_geometry_cache.hits>0


def test_model_owner_replacement_is_a_miss_and_changed_transform_uses_current_bytes():
    f,c=make(current);c._cached_robot_world(f.q)
    previous=c._world_geometry_cache.misses
    mesh=f.node.carried_collision_meshes[0];handle=ModelLocalMesh(box(.07))
    f.node.carried_collision_meshes=(CollisionMesh(mesh.link,handle.snapshot()[0],mesh.bounds,True,handle),*f.node.carried_collision_meshes[1:])
    actual,_=c._cached_robot_world(f.q)
    assert c._world_geometry_cache.misses==previous+1
    assert signature(actual[mesh.link])==signature(f.robot()[mesh.link])
    c.context['right_positions']=np.r_[.01,np.zeros(6)]
    actual,_=c._cached_robot_world(f.q)
    expected=f.robot(f.q,c.context['right_positions'],c.head)
    assert all(signature(actual[k])==signature(expected[k]) for k in actual)


def test_bounded_eviction_keeps_the_current_geometry():
    f,c=make(current);c._world_geometry_cache=ExactWorldGeometry(maximum_entries=1,maximum_bytes=1024*1024)
    for shift in (0.,.1,0.):
        q=f.q.copy();q[0]=shift
        actual,_=c._cached_robot_world(q);expected=f.robot(q,c.right,c.head)
        assert all(signature(actual[k])==signature(expected[k]) for k in actual)
        assert len(c._world_geometry_cache._entries)<=1


def test_body_exception_and_cancellation_do_not_commit_geometry_verdicts():
    a,old=make(original,rejection=RuntimeError('body failure'))
    b,new=make(current,rejection=RuntimeError('body failure'))
    for f,c in ((a,old),(b,new)):
        with pytest.raises(RuntimeError,match='body failure'):c.sample(f.q,.069)
        assert not c.cache
        f.node._cancel.set();assert c.sample(f.q,.069)=='empty_pickup_cancelled'
    assert a.events==b.events and diagnostics(old)==diagnostics(new)


def test_producer_replaced_after_capture_keeps_original_fallback():
    a,old=make(original);b,new=make(current)
    for f,c in ((a,old),(b,new)):
        old_world=f.node._world_collision_surfaces
        f.node._world_collision_surfaces=lambda *a,_world=old_world,**k:_world(*a,**k)
    assert old.sample(a.q,.069)==new.sample(b.q,.069)
    assert a.events==b.events and a.world_calls==b.world_calls
    assert new._world_geometry_cache.hits==new._world_geometry_cache.misses==0


def test_body_cache_hits_reuse_exact_bounds_without_skipping_new_aperture_checks():
    a,old=make(original);b,new=make(current)
    for aperture in (.01,.02,.03):
        assert old.sample(a.q,aperture)==new.sample(b.q,aperture)
    assert diagnostics(old)==diagnostics(new)
    assert old.checked_samples==3 and old.cache_hits==0
    assert a.events==b.events
    assert new._world_geometry_cache.misses==3
    assert new._world_geometry_cache.hits==6


@pytest.mark.parametrize('failure',['screen','robot_floor','tool_floor','tool_robot','inventory'])
def test_current_screen_floor_and_tool_counterexamples_preserve_first_reason(failure):
    a,old=make(original);b,new=make(current)
    for f,c in ((a,old),(b,new)):
        if failure=='screen':
            c.screen.world=lambda torso,head:np.array([[x,y,z] for x in (-.3,.3) for y in (-.3,.3) for z in (.7,1.3)])
        elif failure=='robot_floor':
            mesh=f.node.carried_collision_meshes[-1];raw=box(.04);raw[:,:,2]-=.99
            handle=ModelLocalMesh(raw)
            f.node.carried_collision_meshes=(*f.node.carried_collision_meshes[:-1],
                CollisionMesh(mesh.link,handle.snapshot()[0],mesh.bounds,True,handle))
        else:
            local=c.geometry.local_surfaces(.069)
            if failure=='inventory':local.pop(next(iter(local)))
            else:
                key=next(iter(local));local[key]=box()
                if failure=='tool_floor':local[key][:,:,2]-=.99
            c.geometry.local_surfaces=lambda aperture,_local=local:_local
    expected=old.sample(a.q,.069);actual=new.sample(b.q,.069)
    assert expected==actual and failure in actual
    assert diagnostics(old)==diagnostics(new) and a.events==b.events


def test_unsupported_transform_view_retains_dense_expression():
    a,old=make(original);b,new=make(current)
    for f in (a,b):
        matrix=f.matrix
        f.matrix=lambda *args,_matrix=matrix:_matrix(*args).view()
    assert old.sample(a.q,.069)==new.sample(b.q,.069)
    assert a.events==b.events and diagnostics(old)==diagnostics(new)
    assert new._world_geometry_cache.fallbacks==3
    assert new._world_geometry_cache.misses==new._world_geometry_cache.hits==0
