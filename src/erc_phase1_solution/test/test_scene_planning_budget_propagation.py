"""Actual scene helper passes the chosen wall bound to the full-route solver."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from erc_phase1_solution import scene_checked_place as mod
from test_scene_checked_place_planner import planning


@pytest.mark.parametrize('value',[None,420.,900.,1800.])
def test_scene_helper_forwards_wall_budget_without_replacing_candidate_guard(planning,monkeypatch,tmp_path,value):
    node,events,_=planning
    if value is not None:node.scene_cartesian_wall_seconds=value
    for relative,data in (
        ('models/table/meshes/erc_base_table.STL',b'table-mesh'),
        ('models/table/sdf/erc_table.sdf',b'table-sdf')):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    original=hashlib.sha256
    def digest(data):
        value=(mod.TABLE_MESH_SHA256 if data==b'table-mesh' else
               '90f3aa39affdadcf6ade532ed19a7898d84b813478791b38f40e2316e5d07f4f' if data==b'table-sdf' else None)
        if value is None:return original(data)
        return type('Digest',(),{'hexdigest':lambda self:value})()
    monkeypatch.setattr(mod.hashlib,'sha256',digest)
    monkeypatch.setattr(mod,'TableSceneObstacle',lambda scene:object())
    observed=[]
    def solve(node_arg,positions,rotation,height,staging,scene,candidate,**kwargs):
        limits=kwargs['limits'];observed.append(limits)
        assert (limits.max_ik_calls,limits.max_paths,limits.max_candidates)==(512,24,6)
        # Invoke the unchanged real helper's candidate, including setup/open/return.
        return node._solve_cartesian_path(positions,(rotation,),height,
            skip_setup_transition=True,candidate_validator=candidate,joint_limit_margin=.1)
    monkeypatch.setattr(mod,'solve_scene_cartesian',solve)
    start=np.asarray([.35,0.,0.,0.,0.,0.,0.,0.]);torso=start.copy();torso[0]=.30
    result=mod.plan_scene_checked_place(node,[[0,0,0]],np.eye(3),start,torso,np.ones(8),
        [.8,0,.75],lambda package:tmp_path,table_scene={'synthetic_fixture':True})
    assert len(observed)==1 and observed[0].wall_seconds==(420. if value is None else value)
    assert any(event[0]=='controller_route' for event in events)
    assert any(event[0]=='opening' for event in events)
    assert any(event[0]=='scene_leg' and event[2] is False for event in events)
    assert result.diagnostics['open_return_legs']>0
