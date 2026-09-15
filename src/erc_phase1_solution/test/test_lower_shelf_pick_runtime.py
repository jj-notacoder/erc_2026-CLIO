"""The actual PICK workflow admits a full plan before motion and uses its torso."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
pytest.importorskip('rclpy')
from erc_phase1_solution import manipulation_node as manipulation
from erc_phase1_solution import lower_shelf_pick as lower
from test_shutdown import _mocked_pick_trace
from test_lift_first_integration import _enabled
from erc_phase1_solution.lift_first_extraction import LiftFirstPlan


def test_infeasible_lower_route_stops_actual_pick_before_any_motion(monkeypatch):
    motions = []
    def configure(node, seen):
        node.lower_shelf_pick_enabled = True
        node._move_torso = lambda *a, **k: motions.append('torso')
        node._move_arm_solution = lambda *a, **k: motions.append('arm')
        node._open_gripper = lambda *a, **k: motions.append('open')
    _, setup = _enabled(monkeypatch, configure=configure)
    planner = Mock(side_effect=RuntimeError('no complete lower route'))
    monkeypatch.setattr(lower, 'plan_lower_shelf_pick', planner)
    with pytest.raises(RuntimeError, match='no complete lower route'):
        _mocked_pick_trace([.67, -.05, 1.25], configure_node=setup)
    planner.assert_called_once()
    assert motions == []
    assert planner.call_args.kwargs['empty_guard'] is not None
    assert callable(planner.call_args.kwargs['lift_planner'])


def test_selected_torso_and_cached_plan_reach_actual_pick_dispatch(monkeypatch):
    heights = []
    def configure(node, seen):
        node.lower_shelf_pick_enabled = True
        node._move_torso = lambda height, *a, **k: heights.append(height) or False
        node._plan_carried_return = Mock(side_effect=AssertionError('accepted carry was recomputed'))
    seen, setup = _enabled(monkeypatch, configure=configure)
    def planner(node, front, **kwargs):
        qs = tuple(np.array([.10, *([v]*7)]) for v in (10.,11.,12.))
        raised = tuple(np.array([.10, *([v]*7)]) for v in (40.,41.))
        lift = LiftFirstPlan(raised, raised[-1], np.zeros((8,3)), {})
        return NS(positions=tuple(np.asarray(front) for _ in qs),
            grasp=np.asarray(front)+[.025,-.001,-.015], rotations=(np.eye(3),),
            solutions=qs, orientation_index=0, path_score=1., transition_waypoints=(),
            pick_torso_height=.10, grasp_depth_offset=.025, grasp_lateral_offset=-.001,
            grasp_vertical_offset=-.015, loaded_clearance_index=1, endpoint_first=True,
            lift_plan=lift, return_result=([],[],None,[],np.zeros((8,3))),
            cached_post_retreat_plan={'requires_gravity_support':False,'sentinel':'selected_lower_route'})
    monkeypatch.setattr(lower, 'plan_lower_shelf_pick', planner)
    with pytest.raises(RuntimeError, match='pick_recovery_failed'):
        _mocked_pick_trace([.67, -.05, 1.25], configure_node=setup)
    assert heights == [.10]
    assert seen['node'].pick_torso_height == .35
    seen['node']._plan_carried_return.assert_not_called()
