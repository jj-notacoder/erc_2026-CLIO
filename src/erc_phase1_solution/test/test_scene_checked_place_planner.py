"""Actual helper orchestration with synthetic geometry backends; no ROS."""
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution import scene_checked_place as mod


@pytest.fixture
def planning(monkeypatch, tmp_path):
    events=[]
    for relative in ('urdf/tiago_pro.urdf','models/collection_bin/meshes/erc_base_collection_bin.STL'):
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic asset')
    monkeypatch.setattr(mod,'load_stl_triangles',lambda p:mod._box_triangles([.3,.2,.5]))
    monkeypatch.setattr(mod,'ScreenEnvelope',lambda p:object())
    start=np.asarray([.35,0.,0.,0.,0.,0.,0.,0.]);torso=start.copy();torso[0]=.30
    first=torso.copy();first[1]=.1;last=first.copy();last[1]=.2
    node=SimpleNamespace(_cancel=threading.Event(),gripper_open=.06,place_torso_height=.30,
        place_joint_limit_margin=.03,carried_supported_jaw_vertical_component=.75,
        _shelf_cradle_geometry=object(),_held_book_corners=np.zeros((8,3)),chain=object())
    node._adaptive_motion_feedback=lambda:(SimpleNamespace(position=.017,stamp_ns=12),None)
    node._carried_robot_transition_is_safe=lambda a,b,book:events.append(('loaded',a.copy(),b.copy())) or True
    node._gravity_supported_transition_is_safe=lambda a,b:events.append(('support',a.copy(),b.copy())) or True
    node._plan_retracted_transition=lambda a,b:events.append(('unloaded_home',a.copy(),b.copy())) or []
    node._plan_carried_joint_route=lambda a,g,b,**kw:events.append(('controller_route',kw)) or list(g)
    monkeypatch.setattr(mod,'coordinated_supported_goals',lambda *a,**kw:events.append(('coordinator',kw)) or [first])
    class Scene:
        def __init__(self,node,obstacle,tool,attached,table=None,screen=None):
            self.attached=attached;self.samples=0;self.cache_hits=0;self.last_rejection=None
            self.minimum_moving_left_z=.8
        def leg(self,a,b,q,loaded):
            events.append(('scene_leg',q,loaded,a.copy(),b.copy()));self.samples+=1
            return not getattr(node,'reject_return',False) or loaded
        def sample(self,q,master,loaded):
            events.append(('opening',master));self.samples+=1
            return not getattr(node,'reject_opening',False)
    monkeypatch.setattr(mod,'PlaceSceneChecker',Scene)
    def solve(p,r,height,**kw):
        events.append(('solve',kw))
        assert kw['skip_setup_transition'] and callable(kw['candidate_validator'])
        assert kw['joint_limit_margin']==.10
        if not kw['candidate_validator']([first,last],[]):raise RuntimeError('all candidates rejected')
        return [first,last],0,.5,[]
    node._solve_cartesian_path=solve
    call=lambda:mod.plan_scene_checked_place(node,[[0,0,0]],np.eye(3),start,torso,
        np.ones(8),[.8,0,.75],lambda p:tmp_path)
    return node,events,call


def test_torso_precedes_search_and_all_open_return_legs_use_configured_target(planning):
    node,events,call=planning
    result=call()
    assert events[0][0]=='loaded' and events[1][0]=='support' and events[2][0]=='scene_leg'
    assert events[2][2] is True
    solver=next(data for kind,data,*_ in events if kind=='solve')
    np.testing.assert_array_equal(solver['transition_start'],np.ones(8))
    assert solver['skip_setup_transition'] and callable(solver['candidate_validator'])
    opening=[event[1] for event in events if event[0]=='opening']
    assert opening[0]==.017 and opening[-1]==node.gripper_open
    assert max(np.diff(opening))<=.0010000000001
    empty=[event for event in events if event[0]=='scene_leg' and not event[2]]
    assert empty and all(event[1]==node.gripper_open for event in empty)
    assert result.diagnostics['modeled_open_master']==.06


def test_failed_torso_validation_never_starts_ik(planning):
    node,events,call=planning
    node._carried_robot_transition_is_safe=lambda *a:False
    with pytest.raises(RuntimeError,match='torso_rejected'):call()
    assert not any(event[0]=='solve' for event in events)


@pytest.mark.parametrize('flag',('reject_opening','reject_return'))
def test_opening_and_entire_return_rejections_are_candidate_rejections(planning,flag):
    node,events,call=planning
    setattr(node,flag,True)
    with pytest.raises(RuntimeError,match='candidates rejected'):call()
    assert any(event[0]=='opening' for event in events)


def test_preexisting_cancel_stops_before_feedback_assets_or_ik(planning):
    node,events,call=planning
    node._cancel.set()
    node._adaptive_motion_feedback=lambda:pytest.fail('feedback read after cancel')
    with pytest.raises(RuntimeError,match='cancelled'):call()
    assert events==[]


def test_failed_controller_sized_loaded_route_is_never_admitted(planning):
    node,events,call=planning
    node._plan_carried_joint_route=lambda *a,**k:None
    with pytest.raises(RuntimeError,match='candidates rejected'):call()
    assert not any(event[0]=='opening' for event in events)


def test_cancel_after_successful_search_still_discards_plan(planning):
    node,events,call=planning
    original=node._solve_cartesian_path
    def cancelled(*a,**kw):
        result=original(*a,**kw);node._cancel.set();return result
    node._solve_cartesian_path=cancelled
    with pytest.raises(RuntimeError,match='no_accepted_route'):call()


@pytest.mark.parametrize('position',(-.001,.07,float('nan')))
def test_invalid_configured_opening_never_starts_plan(planning,position):
    node,events,call=planning
    node.gripper_open=position
    with pytest.raises(ValueError,match='opening'):call()
    assert events==[]
