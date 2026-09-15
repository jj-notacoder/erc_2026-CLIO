"""Production caller chooses wrist policy from the admitted measured carry."""
import numpy as np
import pytest
from erc_phase1_solution import scene_checked_place as planner
from test_scene_checked_place_planner import planning
from test_release_only_place_planning import selected_planning


@pytest.mark.parametrize('measured,staging,expected',[
    (.9,-.9,'measured_positive'),(-.9,.9,None),(0.,.9,None)])
def test_measured_carry_selects_policy_independently_of_staging_seed(
        selected_planning,tmp_path,monkeypatch,measured,staging,expected):
    node,events,request,_=selected_planning
    start=np.array([.1,0.,0.,0.,0.,0.,0.,measured]);torso=start.copy();torso[0]=.30
    seed=torso.copy();seed[-1]=staging
    node.cartesian_step=.06
    monkeypatch.setattr(planner,'registered_place_proposals',lambda *args,**kwargs:())
    observed=[]
    class ReachedSolver(Exception):pass
    def solve(*args,**kwargs):
        observed.append(kwargs)
        assert args[4][-1]==staging and kwargs['measured_seed'][-1]==measured
        assert kwargs['limits'].max_ik_calls==512
        assert callable(args[6])
        raise ReachedSolver()
    monkeypatch.setattr(planner,'solve_scene_cartesian',solve)
    with pytest.raises(ReachedSolver):
        planner.plan_scene_checked_place(node,[[0.,0.,0.]],np.eye(3),start,torso,
            seed,[.8,0.,.75],lambda _:tmp_path,table_scene=node._selected_place_table_scene,
            bin_scene=node._selected_place_bin_scene,release_only_request=request)
    assert observed[0].get('wrist_policy')==expected
    assert [row[0] for row in events[:3]]==['loaded','support','scene_leg']
