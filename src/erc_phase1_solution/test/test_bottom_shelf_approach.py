"""Bottom proposal admission, preserved withdrawal and registered shelf bounds."""
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from erc_phase1_solution import bottom_shelf_approach as draft


class Chain:
    active_names = draft.IK_JOINTS
    lower = np.full(8, -10.)
    upper = np.full(8, 10.)
    def __init__(self):
        self.calls = []; self.bad_fixed = False; self.jump = False
    def forward(self, q):
        pose = np.eye(4); pose[:3, :3] = draft.shelf_pinch_orientations(.604)[3]
        pose[:3, 3] = np.asarray(q)[[1, 2, 4]]
        return pose
    def pose_error(self, actual, desired):
        return np.r_[desired[:3, 3]-actual[:3, 3], np.zeros(3)]
    def solve(self, target, seeds, **kwargs):
        self.calls.append(kwargs)
        q = seeds[0].copy(); q[[1, 2, 4]] = target[:3, 3]
        if self.bad_fixed:q[3] += .05
        if self.jump:q[5] += 1.
        return q, .001


@pytest.fixture
def context(monkeypatch):
    chain = Chain()
    grasp = np.array([.35, .695, -.056, .528693, .604, 0., 0., 0.])
    withdrawal = grasp.copy(); withdrawal[1] = .648281
    start = grasp.copy(); start[0] = .10; start[1] = .10; start[3] = 1.5
    node = NS(chain=chain, _cancel=threading.Event(), pick_position_tolerance=.0005,
        pick_orientation_tolerance=.01, cartesian_joint_step=.40,
        carried_transition_samples=61, pick_torso_height=.35)
    node._plan_retracted_transition = Mock(return_value=[])
    node._arm_route_cost = Mock(return_value=.1)
    node._move_arm_solution = Mock(side_effect=AssertionError('must not dispatch'))
    node._move_torso = Mock(side_effect=AssertionError('must not dispatch'))
    node._command_gripper = Mock(side_effect=AssertionError('must not dispatch'))
    guard = NS(node=node, open_aperture=.069, initial_aperture=.069, start=start,
        context=dict(right_positions=np.zeros(7), head_positions=[0., -.7]),
        opening=Mock(return_value=True), edge=Mock(return_value=True),
        retracted_edge=Mock(return_value=True), last_rejection='fixture_reject')
    bounds = NS(edge=Mock(return_value=None), samples=488, minimum_floor=.02,
                minimum_roof=.04, minimum_side=.10)
    monkeypatch.setattr(draft, '_ShelfEntryBounds', Mock(return_value=bounds))
    monkeypatch.setattr(draft, 'check_gripper_opening', Mock(return_value=None))
    monkeypatch.setattr(draft, 'check_cradle_tool_sweep', Mock(return_value=None))
    monkeypatch.setattr(draft, 'check_open_gripper_approach', Mock(return_value=NS(ok=True)))
    args = dict(empty_guard=guard, withdrawal_solutions=[withdrawal], bay=object())
    return NS(node=node, guard=guard, bounds=bounds, grasp=grasp, withdrawal=withdrawal,
              args=args, front=np.array([.670, -.056, .604]))


def run(c):
    return draft.plan_bottom_shelf_approach(c.node, c.front, c.grasp, **c.args)


def test_preserves_exact_grasp_and_withdrawal_and_checks_all_edges(context):
    c=context; plan=run(c)
    np.testing.assert_array_equal(plan.solutions[-1], c.grasp)
    np.testing.assert_array_equal(plan.extraction_solutions[0], c.withdrawal)
    assert not plan.solutions[-1].flags.writeable
    c.grasp[1]=.700; c.withdrawal[1]=.640
    assert plan.solutions[-1][1]==.695 and plan.extraction_solutions[0][1]==.648281
    assert len(c.node.chain.calls)==7 and len(plan.solutions)==8
    assert c.guard.edge.call_count==8  # torso plus seven Cartesian edges
    assert c.bounds.edge.call_args_list[0].kwargs==dict(allow_entry=False)
    assert all(call.kwargs==dict(allow_entry=True) for call in c.bounds.edge.call_args_list[1:8])
    assert c.bounds.edge.call_args_list[-1].kwargs==dict(allow_entry=False)
    assert c.node.pick_torso_height==.35
    for call, (_, arm3) in zip(c.node.chain.calls, draft.BACKWARD_ENTRY_SCHEDULE):
        assert call['fixed_positions']=={'torso_lift_joint':.35, 'arm_left_3_joint':arm3}
    for command in (c.node._move_arm_solution,c.node._move_torso,c.node._command_gripper):
        command.assert_not_called()


def test_body_rejection_stops_before_setup_or_book_admission(context):
    c=context; c.guard.edge.side_effect=[True, False]
    with pytest.raises(RuntimeError, match='reverse_body'):run(c)
    c.node._plan_retracted_transition.assert_not_called()
    draft.check_open_gripper_approach.assert_not_called()


def test_bay_rejection_stops_route(context):
    c=context; c.bounds.edge.side_effect=[None,'bottom_empty_bay_clearance:test']
    with pytest.raises(RuntimeError, match='bay_clearance'):run(c)
    assert len(c.node.chain.calls)==1
    c.node._plan_retracted_transition.assert_not_called()


@pytest.mark.parametrize('field',['bad_fixed','jump'])
def test_ik_success_does_not_bypass_fixed_joint_or_step_check(context,field):
    c=context; setattr(c.node.chain,field,True)
    with pytest.raises(RuntimeError, match='residual_or_step'):run(c)
    assert c.guard.edge.call_count==1


def test_returned_setup_is_rechecked_even_if_planner_ignores_callback(context):
    c=context; c.guard.retracted_edge.return_value=False
    with pytest.raises(RuntimeError, match='setup_route_rejected'):run(c)


def test_tool_book_check_is_required(context):
    c=context; draft.check_open_gripper_approach.return_value=NS(ok=False, reason='fingertip')
    with pytest.raises(RuntimeError, match='approach_book:fingertip'):run(c)
    c.node._plan_retracted_transition.assert_not_called()


def test_closed_guard_from_another_context_is_rejected(context):
    c=context; c.guard.node=object()
    with pytest.raises(ValueError, match='guard_context'):run(c)
    c.guard.opening.assert_not_called()


def test_invalid_withdrawal_is_not_substituted(context):
    c=context; c.withdrawal[1]=.53
    with pytest.raises(ValueError, match='withdrawal_proposal'):run(c)


def test_cancellation_prevents_any_guard_or_ik_work(context):
    c=context; c.node._cancel.set()
    with pytest.raises(RuntimeError, match='cancelled'):run(c)
    c.guard.opening.assert_not_called(); assert not c.node.chain.calls


def test_clipping_includes_plane_crossings():
    triangles=np.array([[[-1.,-2.,0.],[1.,2.,0.],[1.,0.,0.]]])
    points=draft._forward_vertices(triangles,np.zeros(3),np.array([1.,0.,0.]),0.)
    assert points.shape==(4,3)
    assert points[:,0].min()==0. and points[:,1].min()==-1.


def test_clipping_ignores_a_triangle_entirely_behind_plane():
    triangles=np.array([[[-2.,-100.,0.],[-1.,100.,0.],[-1.,0.,0.]]])
    assert draft._forward_vertices(triangles,np.zeros(3),np.array([1.,0.,0.]),.015).shape==(0,3)


def _real_bounds(robot_triangle):
    from erc_phase1_solution.lift_first_extraction import RelativeShelfBay
    node=NS(_cancel=threading.Event(),carried_transition_samples=61,carried_book_dimensions=np.array([.16,.02,.25]),
        chain=NS(forward=lambda q:np.eye(4)),
        _world_collision_surfaces=lambda q,**kwargs:{'arm_left_test':np.array([robot_triangle])})
    behind=np.array([[[.10,-.01,.60],[.11,.01,.60],[.10,.01,.61]]])
    local={name:behind for name in draft.LEFT_GRIPPER_COLLISION_LINKS}
    guard=NS(geometry=NS(local_surfaces=lambda aperture:local),open_aperture=.069,
        context={'right_positions':np.zeros(7),'head_positions':np.array([0.,-.7])})
    bay=RelativeShelfBay(marker_center_base=[.605,0.,2.26],inward_axis_base=[1.,0.,0.],
        physical_column=3,lateral_uncertainty_m=.200,roof_uncertainty_m=.010,
        assume_upright_supported=True,source='unit registered bay')
    return draft._ShelfEntryBounds(node,np.array([.67,-.056,.604]),np.zeros(8),guard,bay)


def test_actual_bay_bounds_allow_inside_entry_but_reject_arbitrary_setup():
    triangle=[[.65,-.01,.60],[.66,.01,.60],[.65,.01,.61]]
    bounds=_real_bounds(triangle)
    assert bounds.sample(np.zeros(8),True) is None
    assert 'setup_enters_shelf' in bounds.sample(np.zeros(8),False)


@pytest.mark.parametrize('triangle',[
    [[.65,-.01,.45],[.66,.01,.45],[.65,.01,.46]],
    [[.65,-.01,.80],[.66,.01,.80],[.65,.01,.81]],
    [[.65,.35,.60],[.66,.36,.60],[.65,.35,.61]],
])
def test_actual_bay_bounds_reject_floor_roof_and_side(triangle):
    assert 'bay_clearance' in _real_bounds(triangle).sample(np.zeros(8),True)


def test_actual_bay_bounds_catch_floor_violation_at_triangle_crossing():
    # Original vertices ahead of the plane fit the bay; its crossing vertices
    # are below the shelf floor and must still reject the entire triangle.
    triangle=[[.58,0.,-.10],[.70,-.01,.60],[.70,.01,.60]]
    assert 'bay_clearance' in _real_bounds(triangle).sample(np.zeros(8),True)


def test_normal_uncertainty_rejects_setup_in_new_ten_millimetre_band():
    # x=.585 was behind the older .605-.015 setup plane. The independently
    # registered model retains the earlier .605 plane and adds .010 uncertainty.
    triangle=[[.584,-.01,.60],[.585,.01,.60],[.584,.01,.61]]
    bounds=_real_bounds(triangle)
    assert bounds.normal_uncertainty_m == .010
    assert bounds.registered_lip[0] == pytest.approx(.606016)
    assert bounds.plane_point[0] == pytest.approx(.605)
    assert bounds.sample(np.zeros(8),False).startswith('empty_setup_enters_shelf:')
    assert bounds.sample(np.zeros(8),True) is None


def test_registered_marker_plane_is_independent_of_book_motion():
    triangle=[[.65,-.01,.60],[.66,.01,.60],[.65,.01,.61]]
    bounds=_real_bounds(triangle)
    from erc_phase1_solution.lift_first_extraction import RelativeShelfBay
    bay=RelativeShelfBay(marker_center_base=[.605,0.,2.26],inward_axis_base=[1.,0.,0.],
        physical_column=3,lateral_uncertainty_m=.200,roof_uncertainty_m=.010,
        assume_upright_supported=True,source='unit independent registered marker')
    deeper=draft._ShelfEntryBounds(bounds.node,np.array([.72,-.056,.604]),
        np.zeros(8),bounds.guard,bay)
    assert deeper.registered_lip[0] == bounds.registered_lip[0]
    assert deeper.plane_point[0] == pytest.approx(.606016)
    assert deeper.book_depth_m > bounds.book_depth_m
