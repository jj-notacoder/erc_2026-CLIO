"""Behavioral prefix, wire and scheduling tests; no source-mirror assertions."""
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution.geometry_process_protocol import (
    SampleDelta, commit_sample, decode_frame, encode_frame,
    query_from_wire, query_to_wire, sample_key,
)
from erc_phase1_solution.geometry_process_pool import GeometryProcessPool, GeometryProcessError
from test_pure_geometry_owner import query, robot, scene


def delta(q, *, accepted=True, minimum=1., table_touched=False, table=None):
    return SampleDelta(q.request_id,q.epoch,q.source_id,q.model_id,q.scene_id,
        q.input_sha256,accepted,minimum,None if accepted else {'reason':'robot_self'},
        table_touched,table)


def state():
    result=SimpleNamespace(cache={},samples=0,cache_hits=0,
        minimum_moving_left_z=math.inf,last_rejection={'reason':'old'},
        table=SimpleNamespace(last_intersection='old'))
    def reject(reason):
        result.last_rejection={'reason':reason}
        return False
    result._reject=reject
    return result


def test_wire_roundtrip_preserves_owned_negative_zero_and_exact_query_binding():
    q=query(q=np.array([-0.,1.,2.,3.,4.,5.,6.,7.]),aperture=-0.)
    restored=query_from_wire(decode_frame(encode_frame(query_to_wire(q))))
    assert restored==q and restored.input_sha256==q.input_sha256
    assert np.signbit(np.frombuffer(restored.q,dtype=np.float64)[0])
    assert math.copysign(1.,restored.aperture)==-1.


@pytest.mark.parametrize('data',[b'{"x":1,"x":2}\n',b'{"x":NaN}\n',b'{"x":Infinity}\n',
    b'{}\n{}\n',b'{}\n\n',b'{}',b'[]\n',b'{}'+b' '*9000+b'\n'])
def test_wire_ambiguous_nonfinite_unframed_and_oversized_values_fail(data):
    with pytest.raises(ValueError):decode_frame(data,8192)


@pytest.mark.parametrize('field,value',[('input_sha256','0'*64),('q','00'),('right','z'*112),
    ('loaded',1),('head','00'*16),('epoch',99)])
def test_changed_query_payload_cannot_keep_the_original_hash(field,value):
    q=query(head=np.ones(2))
    payload=query_to_wire(q);payload[field]=value
    with pytest.raises(ValueError):query_from_wire(payload)


@pytest.mark.parametrize('change',[{'minimum_update':math.nan},{'minimum_update':False},
    {'verdict':False},{'rejection_update':{'reason':'unexpected'}},
    {'table_intersection':'floor'}, {'source_id':'no'},
    {'verdict':False,'rejection_update':{'reason':'bad','value':math.inf}}])
def test_invalid_delta_never_mutates_committed_scene(change):
    q=query();s=state();bad=replace(delta(q),**change)
    with pytest.raises((ValueError,RuntimeError)):commit_sample(s,q,bad,cancelled=lambda:False)
    assert s.samples==0 and s.minimum_moving_left_z==math.inf and not s.cache


def test_cancel_precedes_cache_and_does_not_count_a_sample():
    q=query();s=state();s.cache[sample_key(q)]=True
    assert not commit_sample(s,q,None,cancelled=lambda:True)
    assert s.last_rejection=={'reason':'cancelled'} and s.samples==s.cache_hits==0


def test_success_cache_hit_preserves_minimum_rejection_and_table():
    q=query();s=state();s.cache[sample_key(q)]=True
    assert commit_sample(s,q,None,cancelled=lambda:False)
    assert (s.samples,s.cache_hits)==(1,1)
    assert s.minimum_moving_left_z==math.inf and s.last_rejection=={'reason':'old'}
    assert s.table.last_intersection=='old'


def test_rejected_sample_still_commits_earlier_minimum_and_table_reset():
    q=query();s=state()
    assert not commit_sample(s,q,delta(q,accepted=False,minimum=.012,table_touched=True),cancelled=lambda:False)
    assert s.minimum_moving_left_z==.012 and s.last_rejection=={'reason':'robot_self'}
    assert s.table.last_intersection is None and not s.cache and s.samples==1


def test_duplicate_speculation_becomes_parent_hit_only_after_ordered_success():
    first=query(0);second=query(1);s=state()
    assert commit_sample(s,first,delta(first),cancelled=lambda:False)
    # A duplicate computed on another worker cannot reset committed diagnostics.
    assert commit_sample(s,second,delta(second,minimum=.1),cancelled=lambda:False)
    assert s.minimum_moving_left_z==1. and (s.samples,s.cache_hits)==(2,1)


def test_independent_owner_uses_actual_scene_and_does_not_keep_success_cache():
    owner=robot();owner._scene=scene(owner)
    initial=owner._scene
    first=owner.evaluate_independent(query(0))
    second=owner.evaluate_independent(query(1))
    assert first.verdict and second.verdict
    assert first.minimum_update==second.minimum_update and first.minimum_update is not None
    assert initial.samples==initial.cache_hits==0 and initial.cache=={}
    assert initial.minimum_moving_left_z==math.inf and initial.last_rejection is None


def test_independent_actual_body_rejection_keeps_preceding_book_minimum():
    owner=robot(overlapping=True);owner._scene=scene(owner)
    q=np.zeros(8);q[0]=1.
    result=owner.evaluate_independent(query(q=q,loaded=True))
    assert not result.verdict and result.rejection_update['reason']=='robot_self'
    assert result.minimum_update==pytest.approx(.98)
    assert owner._scene.minimum_moving_left_z==math.inf


def test_discarded_low_suffix_does_not_pollute_a_later_actual_query():
    owner=robot();owner._scene=scene(owner);committed=state()
    samples=[]
    for index,height in enumerate((.2,.1,.4)):
        q=np.zeros(8);q[0]=height
        request=query(index,q=q)
        samples.append((request,owner.evaluate_independent(request)))
    assert samples[1][1].minimum_update < samples[0][1].minimum_update
    for index in (0,2):
        assert commit_sample(committed,*samples[index],cancelled=lambda:False)
    assert committed.minimum_moving_left_z==samples[0][1].minimum_update
    assert committed.samples==2


def test_independent_table_reset_is_reported_and_prior_table_state_restored():
    owner=robot();owner._scene=scene(owner)
    class Table:
        last_intersection='prior'
        def intersects(self,*args):
            self.last_intersection=None
            return False
    owner._scene.table=Table()
    result=owner.evaluate_independent(query())
    assert result.table_touched and result.table_intersection is None
    assert owner._scene.table.last_intersection=='prior'


class SchedulingPool(GeometryProcessPool):
    """Exercise the actual scheduling loop with controlled out-of-order replies."""
    def __init__(self):
        self.active=self.closed=False;self.next_request_id=0
        self.last_worker_requests=[-1]*4
        self.identity=SimpleNamespace(source_id='1'*64,model_id='2'*64)
        self.epoch=1;self.scene_id='3'*64;self.token='token'
        self.deadline=math.inf;self.query_seconds=15.;self.sent=[];self.replies=[]
        self.cancelled=lambda:False
        self.workers=[SimpleNamespace(send=lambda value,i=i:self.sent.append((i,value))) for i in range(4)]
        self.statistics=dict(submitted=0,consumed=0,discarded=0,cached_at_enqueue=0)
    def _check(self):
        if self.closed or self.cancelled():raise GeometryProcessError('cancelled or closed')
    def _reply(self,pending):
        number=max(pending,key=lambda i:pending[i][0].request_id)
        q,_=pending.pop(number);self.replies.append(q.request_id)
        return number,q,delta(q)
    def close(self):self.closed=True


def capture(index,item):
    q=np.zeros(8);q[0]=item
    return query(index,q=q)


def test_four_worker_replies_commit_in_order_despite_reverse_arrival():
    pool=SchedulingPool();committed=[]
    assert pool.evaluate_sequence(range(9),capture=capture,
        consume=lambda q,d:committed.append(q.request_id) is None)
    assert committed==list(range(9)) and pool.replies[:4]==[3,2,1,0]
    assert len(pool.sent)==9 and not pool.closed


def test_rejected_ordered_prefix_discards_suffix_and_next_sequence_is_clean():
    pool=SchedulingPool();committed=[]
    def consume(q,d):
        committed.append(q.request_id)
        return q.request_id!=1
    assert not pool.evaluate_sequence(range(9),capture=capture,consume=consume)
    assert committed==[0,1] and len(pool.sent)==4 and pool.statistics['discarded']==2
    assert pool.evaluate_sequence([99],capture=capture,consume=lambda q,d:True)
    assert pool.sent[-1][1]['query']['request_id']==4 and not pool.closed


def test_cached_requests_capture_each_context_without_sending_geometry():
    pool=SchedulingPool();seen=[]
    def cached_capture(index,item):
        seen.append(index)
        return capture(index,item)
    assert pool.evaluate_sequence(range(7),capture=cached_capture,
        consume=lambda q,d:d is None,is_cached=lambda q:True)
    assert seen==list(range(7)) and pool.sent==[]


def test_cancellation_after_enqueue_commits_no_results_and_closes_epoch():
    pool=SchedulingPool();count=[0];committed=[]
    def cancels(index,item):
        result=capture(index,item);count[0]+=1
        if count[0]==2:pool.cancelled=lambda:True
        return result
    with pytest.raises(GeometryProcessError):
        pool.evaluate_sequence(range(8),capture=cancels,consume=lambda q,d:committed.append(q))
    assert committed==[] and pool.closed


@pytest.mark.parametrize('kind',['capture_identity','consumer_exception','bad_cache_flag','count_bound'])
def test_parent_contract_error_closes_whole_epoch(kind):
    pool=SchedulingPool();options=dict(capture=capture,consume=lambda q,d:True)
    if kind=='capture_identity':options['capture']=lambda i,x:query(i,epoch=2)
    elif kind=='consumer_exception':
        def bad(q,d):raise ValueError('consumer failed')
        options['consume']=bad
    elif kind=='bad_cache_flag':options['is_cached']=lambda q:1
    else:pool.maximum_queries=0
    with pytest.raises((GeometryProcessError,ValueError)):
        pool.evaluate_sequence([0],**options)
    assert pool.closed


def test_iterator_construction_failure_closes_epoch_and_releases_active_flag():
    pool=SchedulingPool()
    class Broken:
        def __iter__(self):raise ValueError('broken iterator')
    with pytest.raises(ValueError,match='broken iterator'):
        pool.evaluate_sequence(Broken(),capture=capture,consume=lambda q,d:True)
    assert pool.closed and not pool.active


def test_cancellation_after_reply_before_consume_commits_nothing():
    pool=SchedulingPool();committed=[]
    original=pool._reply
    def reply(pending):
        result=original(pending)
        pool.cancelled=lambda:True
        return result
    pool._reply=reply
    with pytest.raises(GeometryProcessError):
        pool.evaluate_sequence([0],capture=capture,consume=lambda q,d:committed.append(q))
    assert committed==[] and pool.closed


def test_partial_thread_start_failure_still_reaps_registered_child(monkeypatch):
    import queue
    import erc_phase1_solution.geometry_process_pool as module
    import erc_phase1_solution.installed_geometry_identity as installed
    calls=[]
    class Identity:
        source_id='1'*64;model_id='2'*64
        def verify(self):pass
        def to_descriptor(self):return {}
    class Process:
        pid=1234;returncode=None
        stdin=stdout=stderr=SimpleNamespace(close=lambda:None)
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=-15;calls.append('reaped')
    class BrokenWorker:
        def __init__(self,number,*args):
            self.number=number;self.process=Process();self.threads=[];self.outgoing=queue.Queue()
            self.stderr_bytes=0;self.stderr_prefix=b'';self.fault=None
        def start(self):raise RuntimeError('thread start failed')
        def capture_identity(self):pass
    monkeypatch.setattr(installed,'InstalledGeometryIdentity',Identity)
    monkeypatch.setattr(module,'_Worker',BrokenWorker)
    monkeypatch.setattr(module,'os',SimpleNamespace(name='posix',environ={},getpid=lambda:9999,killpg=lambda pid,sig:calls.append(pid)))
    monkeypatch.setattr(module,'sys',SimpleNamespace(platform='linux'))
    with pytest.raises(RuntimeError,match='thread start failed'):
        module.GeometryProcessPool(Identity(),epoch=1,scene={},geometry_id='4'*64,cancelled=lambda:False)
    assert 1234 in calls and 'reaped' in calls
