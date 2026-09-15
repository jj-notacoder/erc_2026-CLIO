"""Protocol lifecycle tests; official geometry is separately replayed A/B."""
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
from test_lower_shelf_pick import protocol, open_tool, lower_pick, FRONT


@pytest.fixture
def owned(protocol, monkeypatch):
    p=protocol
    p.events=[]
    p.node._pickup_parallel_geometry_active=False
    p.guard.cache={b'existing': None}
    p.fault={}
    p.epochs=0
    @contextmanager
    def scope(node, checker):
        assert node is p.node and checker is p.guard
        assert not node._pickup_parallel_geometry_active
        if p.fault.get('enter'):
            raise p.fault['enter']
        p.epochs+=1
        epoch=p.epochs
        p.events.append(('enter',epoch,node.pick_torso_height,id(checker.cache)))
        node._pickup_parallel_geometry_active=True
        def sequence(items):
            if p.fault.get('sequence'):
                raise p.fault['sequence']
            return True
        checker._parallel_sequence=sequence
        try:
            yield
        finally:
            del checker._parallel_sequence
            node._pickup_parallel_geometry_active=False
            p.events.append(('close',epoch))
            if p.fault.get('exit'):
                raise p.fault['exit']
    monkeypatch.setattr(lower_pick,'empty_pickup_geometry_scope',scope)
    for name in ('opening','edge','candidate'):
        original=getattr(p.guard,name)
        def guarded(*args,_name=name,_original=original,**kwargs):
            assert p.node._pickup_parallel_geometry_active
            p.events.append((_name,p.epochs))
            return _original(*args,**kwargs) and p.guard._parallel_sequence(iter(()))
        setattr(p.guard,name,guarded)
    def setup(*args,**kwargs):
        assert p.node._pickup_parallel_geometry_active
        p.events.append(('setup',p.epochs))
        return []
    monkeypatch.setattr(lower_pick,'plan_registered_empty_setup',setup)
    lift=p.lift
    def loaded(grasp,extraction):
        assert not p.node._pickup_parallel_geometry_active
        assert not hasattr(p.guard,'_parallel_sequence')
        p.events.append(('lift',p.epochs))
        return lift(grasp,extraction)
    p.lift=loaded
    carry=p.node._plan_carried_return
    def carried(*args,**kwargs):
        assert not p.node._pickup_parallel_geometry_active
        p.events.append(('carry',p.epochs))
        return carry(*args,**kwargs)
    p.node._plan_carried_return=carried
    return p


def run(p):
    return lower_pick.plan_lower_shelf_pick(p.node,FRONT,empty_guard=p.guard,
                                          lift_planner=p.lift,bay=p.bay)


def assert_closed(p):
    assert not p.node._pickup_parallel_geometry_active
    assert not hasattr(p.guard,'_parallel_sequence')
    assert p.node.pick_torso_height==.35
    p.node._move_arm.assert_not_called()
    p.node._move_torso.assert_not_called()


def test_all_empty_checks_finish_before_loaded_planning(owned):
    p=owned
    plan=run(p)
    assert p.epochs==1
    names=[v[0] for v in p.events]
    assert names.index('enter')<names.index('opening')<names.index('candidate')
    assert names.index('candidate')<names.index('close')<names.index('lift')<names.index('carry')
    assert plan.pick_torso_height==.10
    assert_closed(p)


def test_loaded_rejection_reopens_same_checker_only_for_next_ik_approach(owned):
    p=owned
    calls=[]
    original_lift=p.lift
    def lift(grasp,extraction):
        calls.append(grasp.copy())
        assert not p.node._pickup_parallel_geometry_active
        if len(calls)==1:
            raise RuntimeError('ordinary lift geometry rejection')
        return original_lift(grasp,extraction)
    p.lift=lift
    def solve(positions,rotations,torso,**kwargs):
        for attempt in range(2):
            solutions=[np.array([torso,i*.05+attempt*.001,*([0.]*6)])
                       for i in range(len(positions))]
            transition=kwargs['setup_transition_planner'](p.guard.start,solutions[0])
            if kwargs['candidate_validator'](solutions,transition):
                return solutions,0,1.25,transition
        pytest.fail('second equivalent protocol candidate should pass')
    p.node._solve_cartesian_path=solve
    run(p)
    assert p.epochs==2 and len(calls)==2
    entries=[v for v in p.events if v[0]=='enter']
    assert entries[0][3]==entries[1][3]==id(p.guard.cache)
    assert p.guard.cache=={b'existing':None}
    names=[v[0] for v in p.events]
    assert names.index('close')<names.index('enter',1)
    assert_closed(p)


@pytest.mark.parametrize('where',['enter','sequence','exit'])
@pytest.mark.parametrize('kind',[RuntimeError,lower_pick.GeometryProcessError])
def test_pool_fault_cannot_become_candidate_fallback(owned,monkeypatch,where,kind):
    p=owned
    error=kind('exact injected backend fault')
    p.fault[where]=error
    monkeypatch.setattr(lower_pick,'_candidates',lambda _: (
        lower_pick.LowerShelfCandidate('first',.10,-.5),
        lower_pick.LowerShelfCandidate('second',.15,-.5)))
    with pytest.raises(kind,match='exact injected backend fault') as caught:
        run(p)
    assert caught.value is error
    assert p.epochs<=1
    assert not any(v[0] in ('lift','carry') for v in p.events)
    assert p.node._cached_post_retreat_plan is None
    assert_closed(p)


@pytest.mark.parametrize('kind',[TypeError,KeyboardInterrupt])
def test_unexpected_planning_error_closes_owned_pool(owned,kind):
    p=owned
    p.node._solve_cartesian_path=Mock(side_effect=kind('planner stopped'))
    with pytest.raises(kind,match='planner stopped'):
        run(p)
    assert p.epochs==1
    assert_closed(p)


def test_cancelled_empty_admission_closes_before_any_loaded_scope(owned):
    p=owned
    original=p.guard.candidate
    def cancel(*args):
        result=original(*args)
        p.node._cancel.set()
        return result
    p.guard.candidate=cancel
    with pytest.raises(lower_pick.LowerShelfPlanningCancelled):
        run(p)
    assert not any(v[0] in ('lift','carry') for v in p.events)
    assert_closed(p)
