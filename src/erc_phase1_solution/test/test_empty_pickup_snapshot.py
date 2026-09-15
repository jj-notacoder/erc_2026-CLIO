"""Compare real body predicates and world facets against the frozen parent."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution.empty_pickup_collision import EmptyPickupCollision, ScreenEnvelope
from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
from erc_phase1_solution.rigid_palm_preflight import LEFT_GRIPPER_COLLISION_LINKS
from test_sample_collision_snapshot import Fixture, box
from empty_pickup_snapshot_support import restore_empty_pickup_snapshot

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('erc_phase1_solution._empty_snapshot_parent',
    ROOT/'test/fixtures/empty_pickup_snapshot_parent.py')
parent=importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)

def make(*, current, immutable=True, rejection=False, q0=0., custom=False):
    f=Fixture(immutable=immutable,rejection=rejection)
    n=f.node
    n.chain.lower=np.full(8,-10.)
    n.chain.upper=np.full(8,10.)
    n.gripper_open=.069
    n.carried_transition_samples=61
    f.q[0]=q0
    if custom:
        original=n._robot_self_collision
        n._robot_self_collision=lambda q,**kw:original(q,**kw)
    geometry=object.__new__(ShelfCradleGeometry)
    geometry.local_surfaces=lambda aperture:{name:box(30.+i*3.) for i,name in enumerate(LEFT_GRIPPER_COLLISION_LINKS)}
    geometry.watertight={name:True for name in LEFT_GRIPPER_COLLISION_LINKS}
    cls=EmptyPickupCollision if current else parent.EmptyPickupCollision
    screen=object.__new__(ScreenEnvelope if current else parent.ScreenEnvelope)
    screen.world=lambda torso,head:np.array([[x,y,z] for x in (100.,101.) for y in (100.,101.) for z in (100.,101.)])
    c=cls(n,f.q,f.right,f.head,.069,geometry=geometry,screen=screen)
    return f,c

@pytest.mark.parametrize('q0',[0.,.1,-.1])
@pytest.mark.parametrize('rejection',[False,True])
@pytest.mark.parametrize('immutable',[False,True])
def test_exact_verdict_and_narrow_phase_trace(q0,rejection,immutable):
    a,old=make(current=False,q0=q0,rejection=rejection,immutable=immutable)
    b,new=make(current=True,q0=q0,rejection=rejection,immutable=immutable)
    assert old.sample(a.q,.069)==new.sample(b.q,.069)
    assert a.events==b.events
    if immutable and not rejection:
        assert a.world_calls==2 and b.world_calls==0
    elif not immutable:
        assert a.world_calls==b.world_calls

def test_body_cache_hit_reuses_exact_world_without_skipping_predicates():
    a,old=make(current=False)
    b,new=make(current=True)
    for f,c in ((a,old),(b,new)):
        assert c.sample(f.q,.069) is None
        c.cache.clear()
    a.world_calls=b.world_calls=0
    assert old.sample(a.q,.069)==new.sample(b.q,.069)
    assert a.world_calls==1 and b.world_calls==0
    assert new._world_geometry_cache.hits>0
    assert a.events==b.events

def test_changed_parked_context_and_model_are_not_old_snapshot():
    a,old=make(current=False)
    b,new=make(current=True)
    for f,c in ((a,old),(b,new)):
        assert c.sample(f.q,.069) is None
        c.cache.clear()
        c.right=np.frombuffer(np.array([.1,0.,0.,0.,0.,0.,0.]).tobytes(),dtype=float)
    assert old.sample(a.q,.069)==new.sample(b.q,.069)
    assert a.events==b.events

def test_custom_consumer_keeps_original_calls():
    a,old=make(current=False,custom=True)
    b,new=make(current=True,custom=True)
    assert old.sample(a.q,.069)==new.sample(b.q,.069)
    assert a.events==b.events and a.world_calls==b.world_calls==2

def test_cancellation_runs_no_geometry():
    f,c=make(current=True)
    f.node._cancel.set()
    assert c.sample(f.q,.069)=='empty_pickup_cancelled'
    assert f.world_calls==0 and not f.events

def test_body_exception_still_fails_closed():
    f,c=make(current=True,rejection=RuntimeError('body failure'))
    with pytest.raises(RuntimeError,match='body failure'):
        c.sample(f.q,.069)
    assert not c.cache

def test_complete_runtime_inverse_is_exact():
    source=(ROOT/'erc_phase1_solution/empty_pickup_collision.py').read_text()
    assert restore_empty_pickup_snapshot(source)==(ROOT/'test/fixtures/empty_pickup_snapshot_parent.py').read_text()
    with pytest.raises(AssertionError):
        restore_empty_pickup_snapshot(source+'\n')
