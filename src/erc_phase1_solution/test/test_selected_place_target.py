"""The actual PLACE method reports the selected target used by its solution."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from erc_phase1_solution.book_centered_place import book_centered_place_target
from test_book_centered_place import place_fixture


def test_selected_target_replaces_release_and_center_before_place_dispatch(monkeypatch):
    node,_,_=place_fixture(True)
    nominal=book_centered_place_target(node._held_book_corners,[.868,.004,.751],.27)
    delta=np.array([-.05,0.,0.])
    selected=replace(nominal,book_center=nominal.book_center+delta,tool_position=nominal.tool_position+delta)
    q=node._carried_staging_solution.copy()
    def scene_plan(*args,**kwargs):
        return SimpleNamespace(solutions=[q.copy()],orientation_index=0,path_score=0.,
            setup=[q.copy()],unloaded_home=[],diagnostics={},selected_target=selected)
    monkeypatch.setattr('erc_phase1_solution.manipulation_node.plan_scene_checked_place',scene_plan)
    observed={}
    class StopBeforeMotion(Exception):pass
    def publish(event,**fields):
        if event=='ik_ready':observed.update(fields);raise StopBeforeMotion()
    node._publish_status=publish
    with pytest.raises(StopBeforeMotion):node._place()
    np.testing.assert_array_equal(observed['release'],selected.tool_position)
    np.testing.assert_array_equal(observed['nominal_book_center'],selected.book_center)
    np.testing.assert_array_equal(observed['tool_rotation'],selected.tool_rotation)
