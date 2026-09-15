"""Full setup admission, cancellation and bounded fallback; no IK or mesh jobs."""
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from erc_phase1_solution import empty_shelf_setup as setup


def fixture():
    first=np.array([.1,0.,0.,0.,0.,0.,0.,0.])
    last=np.array([.1,.4,-.2,.1,-.3,.2,-.2,.1])
    node=NS(_cancel=threading.Event(),_retracted_transition_is_safe=Mock(return_value=True),
            _plan_retracted_transition=Mock(return_value=None))
    guard=NS(open_aperture=.069,sample=Mock(return_value=None),
             retracted_edge=Mock(return_value=True),last_rejection='dense body rejected')
    bounds=NS(sample=Mock(return_value=None),edge=Mock(return_value=None))
    return NS(node=node,guard=guard,bounds=bounds,first=first,last=last)


def run(f,**kwargs):
    return setup.plan_registered_empty_setup(f.node,f.first,f.last,guard=f.guard,bounds=f.bounds,**kwargs)


def edge_key(a,b):return (np.asarray(a).tobytes(),np.asarray(b).tobytes())


def test_every_returned_prefix_edge_received_both_dense_admissions():
    f=fixture();route=run(f)
    assert route is not None
    expected=[edge_key(a,b) for a,b in zip([f.first,*route],[*route,f.last])]
    body=[edge_key(*call.args) for call in f.guard.retracted_edge.call_args_list]
    shelf=[edge_key(*call.args) for call in f.bounds.edge.call_args_list]
    assert body==shelf==expected
    assert all(call.kwargs=={'allow_entry':False} for call in f.bounds.edge.call_args_list)
    f.node._plan_retracted_transition.assert_not_called()


@pytest.mark.parametrize('failure',['shelf','body'])
def test_coarse_success_cannot_admit_a_dense_failure(failure):
    f=fixture()
    if failure=='shelf':f.bounds.edge.return_value='hidden shelf crossing'
    else:f.guard.retracted_edge.return_value=False
    assert run(f) is None
    assert f.bounds.sample.call_count>0
    assert f.bounds.edge.call_count>0
    if failure=='shelf':f.guard.retracted_edge.assert_not_called()


def force_fallback(f):
    # Prefixes are only numerical proposals. A planner-returned route still
    # needs the full guard, even if that planner ignores its coarse callback.
    f.node._retracted_transition_is_safe.return_value=False
    middle=f.first.copy();middle[6]=.7
    return middle


def test_fallback_cannot_return_unchecked_route_or_skip_failed_edge():
    f=fixture();middle=force_fallback(f)
    f.node._plan_retracted_transition.return_value=[middle]
    f.bounds.edge.return_value='narrow dense-only obstruction'
    assert run(f) is None
    assert f.node._plan_retracted_transition.call_count==8
    assert f.bounds.edge.call_count==1  # Same failed exact edge stays blocked/cached.
    f.guard.retracted_edge.assert_not_called()


def test_failed_proposal_does_not_poison_different_fully_admitted_fallback():
    f=fixture();bad=force_fallback(f);good=bad.copy();good[6]=.5
    f.node._plan_retracted_transition.side_effect=[[bad],[good]]
    f.bounds.edge.side_effect=lambda a,b,**kw:'shelf' if np.array_equal(b,bad) else None
    result=run(f)
    assert len(result)==1
    np.testing.assert_array_equal(result[0],good)
    checked=[edge_key(*call.args) for call in f.guard.retracted_edge.call_args_list]
    assert checked==[edge_key(f.first,good),edge_key(good,f.last)]


def test_fallback_deduplicates_only_repeated_neighbors_and_exact_terminal():
    f=fixture();middle=force_fallback(f)
    f.node._plan_retracted_transition.return_value=[f.first.copy(),middle,middle.copy(),f.last.copy()]
    route=run(f)
    assert len(route)==1
    np.testing.assert_array_equal(route[0],middle)
    assert f.bounds.edge.call_count==f.guard.retracted_edge.call_count==2
    route[0][6]=123.
    assert middle[6]==.7 and f.first[6]==0. and f.last[6]==-.2


def test_exact_admission_cache_does_not_cross_independent_calls():
    f=fixture();assert run(f) is not None
    old=f.guard.retracted_edge.call_count
    assert run(f) is not None
    assert f.guard.retracted_edge.call_count==2*old


def test_initial_cancellation_precedes_all_geometry_and_search():
    f=fixture();f.node._cancel.set()
    with pytest.raises(RuntimeError,match='registered_setup_cancelled'):run(f)
    f.bounds.sample.assert_not_called();f.guard.sample.assert_not_called()
    f.node._plan_retracted_transition.assert_not_called()


@pytest.mark.parametrize('family',['prefix','fallback'])
@pytest.mark.parametrize('when',['last_dense_edge','admitted_emit'])
def test_cancellation_cannot_race_final_admission(family,when):
    f=fixture()
    if family=='fallback':
        middle=force_fallback(f);f.node._plan_retracted_transition.return_value=[middle]
    if when=='last_dense_edge':
        def body(a,b):
            if np.array_equal(b,f.last):f.node._cancel.set()
            return True
        f.guard.retracted_edge.side_effect=body
    def emit(event,**fields):
        if when=='admitted_emit' and event in ('setup_fully_admitted','setup_fully_admitted_prefix'):
            f.node._cancel.set()
    with pytest.raises(RuntimeError,match='registered_setup_cancelled'):run(f,emit=emit)


@pytest.mark.parametrize('family',['prefix','fallback'])
def test_budget_exhaustion_during_last_dense_edge_cannot_return_route(monkeypatch,family):
    f=fixture();clock=NS(now=0.)
    monkeypatch.setattr(setup.time,'monotonic',lambda:clock.now)
    if family=='fallback':
        middle=force_fallback(f);f.node._plan_retracted_transition.return_value=[middle]
    def body(a,b):
        if np.array_equal(b,f.last):clock.now=2.
        return True
    f.guard.retracted_edge.side_effect=body
    with pytest.raises(RuntimeError,match='registered_setup_budget'):run(f,wall_budget=1.)


@pytest.mark.parametrize('budget',[0.,-.1,300.1,np.nan,np.inf])
def test_invalid_wall_budget_precedes_geometry(budget):
    f=fixture()
    with pytest.raises(ValueError,match='wall_budget'):run(f,wall_budget=budget)
    f.bounds.sample.assert_not_called()


@pytest.mark.parametrize('variant',['shape','nonfinite','torso'])
def test_invalid_endpoints_precede_geometry(variant):
    f=fixture()
    if variant=='shape':f.last=f.last[:-1]
    elif variant=='nonfinite':f.last[4]=np.nan
    else:f.last[0]+=.01
    with pytest.raises(ValueError,match='endpoint_shape'):run(f)
    f.bounds.sample.assert_not_called()


def test_invalid_static_endpoint_stops_before_edge_search():
    f=fixture();f.guard.sample.return_value='static body collision'
    assert run(f) is None
    f.bounds.edge.assert_not_called();f.node._plan_retracted_transition.assert_not_called()
