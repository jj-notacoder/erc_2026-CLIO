"""Ordered empty-sample commits, actual grids/fresh guards and owned scope.

Small synthetic sample reasons isolate cache/order mechanics. Real model and
unchanged-predicate parity are separately exercised by the recorded replay.
"""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from erc_phase1_solution import empty_pickup_geometry_backend as backend
from erc_phase1_solution.empty_pickup_collision import EmptyPickupCollision, NAMES, MASTER
from erc_phase1_solution.empty_pickup_geometry_owner import EmptyPickupGeometryQuery, SCENE_KIND
from erc_phase1_solution.geometry_process_protocol import SampleDelta
from erc_phase1_solution.pure_geometry_owner import GeometryQuery


def fixture():
    node=SimpleNamespace(chain=SimpleNamespace(lower=np.full(8,-4.),upper=np.full(8,4.)),
        right_chain=object(),head_chain=object(),carried_collision_meshes=[],
        gripper_open=.069,carried_transition_samples=61,_cancel=threading.Event(),
        pick_torso_height=.35,adaptive_endpoint_tolerance=.0006,
        cartesian_clearance=.45,cartesian_joint_step=.4,
        _lock=threading.Lock(),joints={n:0. for n in NAMES},
        _joint_stamps_ns={n:1_000_000_000 for n in NAMES},
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1_100_000_000)))
    node.joints[MASTER]=.069
    checker=EmptyPickupCollision(node,np.zeros(8),np.zeros(7),np.zeros(2),.069,
        geometry=SimpleNamespace(),screen=SimpleNamespace())
    checker._sample_uncached=lambda q,a: 'collision:first' if .25<q[1]<.5 else None
    return checker,node


def query(q=None, aperture=.069, index=0):
    return EmptyPickupGeometryQuery(GeometryQuery.capture(request_id=index,epoch=1,
        source_id='1'*64,model_id='2'*64,scene_id='3'*64,
        q=np.zeros(8) if q is None else q,right=np.zeros(7),head=np.zeros(2),
        aperture=aperture,loaded=False))


def delta(q,reason=None):
    return SampleDelta(q.request_id,q.epoch,q.source_id,q.model_id,q.scene_id,
        q.input_sha256,reason is None,None,None if reason is None else {'reason':reason},False,None)


def state(c):
    return (dict(c.cache),c.checked_samples,c.cache_hits,c.last_rejection,
            c.last_rejected_q,c.last_rejected_aperture)


@pytest.mark.parametrize('values',[[0.,.3,0.,.3],[.3,.6,.3,.6],[0.,-0.,.3,.6]])
def test_committed_samples_match_original_cache_and_rejection_history(values):
    serial,_=fixture();parallel,_=fixture()
    for i,value in enumerate(values):
        q=np.zeros(8);q[1]=value;item=query(q,index=i)
        expected=serial.sample(q,.069)
        fresh=parallel._sample_uncached(q,.069)
        assert backend.commit_empty_sample(parallel,item,delta(item,fresh)) is (expected is None)
        assert state(serial)==state(parallel)


@pytest.mark.parametrize('cached',[False,True])
def test_cancel_precedes_delta_cache_and_all_counters(cached):
    c,n=fixture();item=query()
    if cached:c.sample(np.zeros(8),.069)
    before=state(c);n._cancel.set()
    assert not backend.commit_empty_sample(c,item,None)
    assert state(c)[:3]==before[:3]
    assert c.last_rejection=='empty_pickup_cancelled'
    assert state(c)[4:]==before[4:]


@pytest.mark.parametrize('aperture,value',[(-.001,0.),(.070,0.),(.069,4.1),(.069,-4.1)])
def test_invalid_finite_samples_keep_original_uncached_limit_rejection(aperture,value):
    serial,_=fixture();parallel,_=fixture();q=np.full(8,value);item=query(q,aperture)
    assert serial.sample(q,aperture)=='empty_pickup_joint_limit'
    assert not backend.commit_empty_sample(parallel,item,delta(item,'empty_pickup_joint_limit'))
    assert state(serial)==state(parallel)
    assert parallel.checked_samples==0 and not parallel.cache


@pytest.mark.parametrize('mutation',[
    lambda d:replace(d,request_id=7),lambda d:replace(d,input_sha256='4'*64),
    lambda d:replace(d,minimum_update=0.),lambda d:replace(d,table_touched=True),
    lambda d:replace(d,verdict=False,rejection_update={'reason':'collision','unrelated':True}),
])
def test_unrelated_or_mismatched_uncached_result_cannot_commit(mutation):
    c,_=fixture();item=query();before=state(c)
    with pytest.raises((ValueError,backend.GeometryProcessError)):
        backend.commit_empty_sample(c,item,mutation(delta(item)))
    assert state(c)==before


def test_worker_limit_claim_must_match_original_local_limit_predicate():
    c,_=fixture();item=query()
    with pytest.raises(backend.GeometryProcessError,match='limit result differs'):
        backend.commit_empty_sample(c,item,delta(item,'empty_pickup_joint_limit'))


@pytest.mark.parametrize('last,transition_samples',[(0.,61),(.17,61),(1.5,3),(.3,95)])
def test_edge_callback_receives_exact_original_grid_and_prefix(last,transition_samples):
    serial,n=fixture();parallel,m=fixture()
    n.carried_transition_samples=m.carried_transition_samples=transition_samples
    a=np.zeros(8);b=np.zeros(8);b[1]=last
    serial_seen=[];parallel_seen=[]
    ordinary=serial.sample
    def record(q,aperture):
        serial_seen.append((q.tobytes(),np.float64(aperture).tobytes()))
        return ordinary(q,aperture)
    serial.sample=record
    def sequence(items):
        for i,(q,aperture) in enumerate(items):
            parallel_seen.append((q.tobytes(),np.float64(aperture).tobytes()))
            item=query(q,aperture,i)
            reason=parallel._sample_uncached(q,aperture)
            if not backend.commit_empty_sample(parallel,item,delta(item,reason)):return False
        return True
    parallel._parallel_sequence=sequence
    assert parallel.edge(a,b)==serial.edge(a,b)
    assert parallel_seen==serial_seen
    assert state(parallel)==state(serial)


@pytest.mark.parametrize('initial',[-.0000005,0.,.019,.068,.0690000005])
def test_opening_callback_keeps_clipped_grid_and_unclipped_measurement(initial):
    serial,_=fixture();parallel,_=fixture();serial.initial_aperture=parallel.initial_aperture=initial
    expected=[];actual=[]
    serial.sample=lambda q,a:expected.append((q.tobytes(),np.float64(a).tobytes())) or None
    def sequence(items):
        actual.extend((q.tobytes(),np.float64(a).tobytes()) for q,a in items);return True
    parallel._parallel_sequence=sequence
    assert serial.opening() and parallel.opening()
    assert actual==expected and parallel.initial_aperture==initial


@pytest.mark.parametrize('aperture',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_edge_preserves_original_sample_limit_reason_without_dispatch(aperture):
    c,_=fixture();c._parallel_sequence=lambda _:pytest.fail('nonfinite query dispatched')
    assert not c.edge(np.zeros(8),np.ones(8)*.1,aperture)
    assert c.last_rejection=='empty_pickup_joint_limit' and c.checked_samples==0


def test_original_fresh_admission_returns_context_only_on_request():
    c,n=fixture();assert c.require_fresh(c.start,c.initial_aperture) is None
    actual,stamps=c.require_fresh(c.start,c.initial_aperture,return_context=True)
    assert actual==n.joints and stamps==n._joint_stamps_ns
    n.joints[MASTER]=0.;n._joint_stamps_ns[MASTER]=0
    assert actual[MASTER]==.069 and stamps[MASTER]==1_000_000_000


@pytest.mark.parametrize('name',NAMES)
def test_return_context_does_not_skip_any_original_freshness_guard(name):
    c,n=fixture();n._joint_stamps_ns[name]=0
    with pytest.raises(RuntimeError,match='stale'):
        c.require_fresh(c.start,c.initial_aperture,return_context=True)


@pytest.mark.parametrize('name,value',[('torso_lift_joint',.0021),('arm_left_1_joint',.0081),
    ('arm_right_2_joint',.0031),('head_2_joint',.0031),(MASTER,.0683)])
def test_return_context_preserves_original_joint_and_aperture_drift_guards(name,value):
    c,n=fixture();n.joints[name]=value
    with pytest.raises(RuntimeError):c.require_fresh(c.start,c.initial_aperture,return_context=True)


@pytest.mark.parametrize('bad',[0,1,None,'true',np.bool_(True)])
def test_flag_is_strict_boolean(bad):
    with pytest.raises(ValueError):backend.checked_empty_pickup_parallel_geometry_enabled(bad)


def scope_fixture(monkeypatch):
    c,n=fixture();events=[]
    n.empty_pickup_parallel_geometry_enabled=True
    n.geometry_process_workers=4
    n._pickup_parallel_geometry_identity=SimpleNamespace(source_id='1'*64,model_id='2'*64,
        verify=lambda:events.append('verify'))
    n._pickup_parallel_geometry_epoch=0;n._pickup_parallel_geometry_active=False
    n._publish_status=lambda event,**kw:events.append((event,kw))
    monkeypatch.setattr(backend.os,'name','posix')
    monkeypatch.setattr(backend.sys,'platform','linux')
    monkeypatch.setattr(backend,'empty_model_signature',lambda c,s:'4'*64)
    class Pool:
        def __init__(self,identity,**kw):
            assert n._pickup_parallel_geometry_active
            self.epoch=kw['epoch'];self.scene_id='3'*64
            events.append('bootstrap');self.closed=False
        def finish(self):events.append('finish')
        def close(self):self.closed=True;events.append('close')
        def worker_status(self):assert self.closed;return dict(children_closed=True)
    monkeypatch.setattr(backend,'GeometryProcessPool',Pool)
    return c,n,events,Pool


@pytest.mark.parametrize('exception',[False,True])
def test_empty_scope_reaps_and_removes_hook_before_later_loaded_work(monkeypatch,exception):
    c,n,events,_=scope_fixture(monkeypatch)
    def attempt():
        with backend.empty_pickup_geometry_scope(n,c):
            assert callable(c._parallel_sequence)
            assert n._pickup_parallel_geometry_active
            events.append('planning')
            if exception:raise RuntimeError('original planner failed')
        events.append('later_loaded_scope')
    if exception:
        with pytest.raises(RuntimeError,match='original planner failed'):attempt()
    else:attempt()
    assert not hasattr(c,'_parallel_sequence') and not n._pickup_parallel_geometry_active
    assert n._pickup_parallel_geometry_epoch==1
    assert events.index('bootstrap')<events.index('planning')<events.index('close')
    if not exception:assert events.index('close')<events.index('later_loaded_scope')
    assert ('finish' in events) is (not exception)
    reports=[x for x in events if isinstance(x,tuple)]
    assert reports[-1][0]=='empty_pickup_geometry_process_completed'
    assert reports[-1][1]['passed'] is (not exception)


def test_bootstrap_failure_is_not_hidden_as_serial_success(monkeypatch):
    c,n,events,_=scope_fixture(monkeypatch)
    def fail(*a,**kw):
        e=backend.GeometryProcessError('owned bootstrap failed')
        e.geometry_worker_status=dict(children_closed=True,workers_created=2,workers_reaped=2)
        raise e
    monkeypatch.setattr(backend,'GeometryProcessPool',fail)
    with pytest.raises(backend.GeometryProcessError,match='owned bootstrap failed'):
        with backend.empty_pickup_geometry_scope(n,c):pytest.fail('planner reached after bootstrap failure')
    assert not hasattr(c,'_parallel_sequence') and not n._pickup_parallel_geometry_active
    report=[x for x in events if isinstance(x,tuple)][-1][1]
    assert report['workers_created']==report['workers_reaped']==2 and not report['passed']


@pytest.mark.parametrize('active',['_pickup_parallel_geometry_active','_place_parallel_geometry_active'])
def test_no_overlapping_owned_pool_can_be_created(monkeypatch,active):
    c,n,events,_=scope_fixture(monkeypatch);setattr(n,active,True)
    with pytest.raises(backend.GeometryProcessError,match='pool active'):
        with backend.empty_pickup_geometry_scope(n,c):pytest.fail('overlap admitted')
    assert 'bootstrap' not in events


@pytest.mark.parametrize('mode',['disabled','unsupported'])
def test_disabled_and_prebootstrap_unsupported_checker_stay_serial(monkeypatch,mode):
    c,n,events,_=scope_fixture(monkeypatch)
    if mode=='disabled':n.empty_pickup_parallel_geometry_enabled=False
    else:
        def unsupported(*a):raise ValueError('custom producer')
        monkeypatch.setattr(backend,'empty_model_signature',unsupported)
    with backend.empty_pickup_geometry_scope(n,c):assert not hasattr(c,'_parallel_sequence')
    assert not events and n._pickup_parallel_geometry_epoch==0


def test_fresh_capture_binds_actual_full_context_and_empty_query(monkeypatch):
    c,n,events,_=scope_fixture(monkeypatch)
    with backend.empty_pickup_geometry_scope(n,c):
        instance=c._parallel_sequence.__self__
        item=instance._capture(7,(np.ones(8)*.1,.02))
        assert type(item) is EmptyPickupGeometryQuery and item.loaded is False
        assert item.q==np.full(8,.1).tobytes()
        assert item.right==c.right.tobytes() and item.head==c.head.tobytes()
        assert instance.last_admission['actual_joints']==n.joints
        assert instance.last_admission['joint_stamps_ns']==n._joint_stamps_ns
        assert instance.last_admission['measured_master_position']==.069
        n._joint_stamps_ns['head_2_joint']=0
        with pytest.raises(RuntimeError,match='stale'):instance._capture(8,(np.zeros(8),.069))
        n._joint_stamps_ns['head_2_joint']=1_000_000_000


def test_postplanning_drift_rejects_after_closing_children(monkeypatch):
    c,n,events,_=scope_fixture(monkeypatch)
    with pytest.raises(RuntimeError,match='context changed'):
        with backend.empty_pickup_geometry_scope(n,c):n.joints['arm_right_1_joint']=.004
    assert events.index('finish')<events.index('close')
    assert not hasattr(c,'_parallel_sequence') and not n._pickup_parallel_geometry_active


def test_node_scope_contains_original_empty_checks_and_solver_before_loaded_work():
    path=Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ManipulationNode')
    pick=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_pick')
    scopes=[x for x in ast.walk(pick) if isinstance(x,ast.With) and any(
        isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name)
        and i.context_expr.func.id=='empty_pickup_geometry_scope' for i in x.items)]
    assert len(scopes)==1
    calls=[x for x in ast.walk(scopes[0]) if isinstance(x,ast.Call)]
    names=[x.func.attr for x in calls if isinstance(x.func,ast.Attribute)]
    assert names.count('opening')==1 and names.count('edge')==1 and names.count('_solve_cartesian_path')==1
    assert not any(name.startswith('_move') or name.startswith('_command') for name in names)
    loaded=[x for x in ast.walk(pick) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='run_pickup_geometry']
    assert loaded and all(x.lineno>scopes[0].end_lineno for x in loaded)
