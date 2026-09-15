"""Actual planner and Request orchestration with explicit synthetic geometry."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import scene_checked_place as planner
from erc_phase1_solution import release_only_place_planning as policy
from erc_phase1_solution.place_contact_guard import PlaceContactGuard
from test_scene_checked_place_planner import planning
from test_release_only_place_planning import selected_planning, PAYLOAD


@pytest.fixture
def bottom_flow(selected_planning, monkeypatch, tmp_path):
    node, events, _, _ = selected_planning
    identity = dict(PAYLOAD, target_model='book_col_4_row_5_yellow')
    node._target_book_model = identity['target_model']
    node._place_contact_guard = PlaceContactGuard(10, identity)
    node._transport_lock_engaged = node._gravity_supported_payload = node._payload_monitor_enabled = True
    node.cartesian_step = .06
    request = policy.capture(node, identity)
    chosen = NS(tool_position=np.array([.7, .1, 1.]))
    proposals = [NS(positions=[[0., 0., 0.]], target=chosen, diagnostics={'fixture': True})]
    monkeypatch.setattr(planner, 'registered_place_proposals',
        lambda *a, **kw: events.append(('registered_proposals',)) or proposals)
    original_solver = planner.solve_scene_cartesian
    def solver(*a, **kw):
        events.append(('search_kwargs', dict(kw)))
        kw['diagnostics_out']['strategy'] = {'position_proposal_index': 0}
        return original_solver(*a, **kw)
    monkeypatch.setattr(planner, 'solve_scene_cartesian', solver)
    def nearest(chain, start, goals, *, margin, minimum_support, checkpoint):
        checkpoint()
        events.append(('nearest_proposal', start.copy(), margin, minimum_support))
        return goals
    monkeypatch.setattr(planner, 'nearest_negative_wrist_goals', nearest)
    # Force the four original policies to fail so the new fifth proposal must
    # still pass every original body/support/scene/open/static-endpoint gate.
    route_calls = []
    def route(start, goals, attached, **kw):
        route_calls.append(kw)
        events.append(('body_route', len(route_calls), kw))
        return list(goals) if len(route_calls) > 4 else None
    node._plan_carried_joint_route = route
    start = np.array([.35, 0., 0., 0., 0., 0., 0., -.8])
    torso = start.copy(); torso[0] = .30
    def call():
        return planner.plan_scene_checked_place(node, [[0., 0., 0.]], np.eye(3),
            start, torso, np.ones(8), [.8, 0., .75], lambda p: tmp_path,
            table_scene=node._selected_place_table_scene, bin_scene=node._selected_place_bin_scene,
            release_only_request=request)
    return node, events, request, chosen, call


def test_fallback_keeps_original_policies_and_full_admission_and_selected_target(bottom_flow):
    node, events, request, chosen, call = bottom_flow
    plan = call()
    assert [e[1] for e in events if e[0] == 'body_route'] == [1, 2, 3, 4, 5]
    assert sum(e[0] == 'nearest_proposal' for e in events) == 1
    options = next(e[1] for e in events if e[0] == 'search_kwargs')
    assert 'wrist_policy' not in options  # Original negative default.
    assert options['limits'].max_ik_calls == 512 and options['limits'].max_candidates == 6
    assert options['position_proposals'] == ([[0., 0., 0.]],)
    assert plan.selected_target is chosen
    assert plan.diagnostics['bottom_nearest_negative_wrist_proposal'] is True
    assert plan.diagnostics['proposal_only_support_reserve'] == .005
    assert all(e[2]['require_gravity_support'] for e in events if e[0] == 'body_route')
    assert sum(e[0] == 'static_sample' for e in events) == 1
    assert any(e[0] == 'loaded' for e in events)
    assert any(e[0] == 'support' for e in events)
    assert any(e[0] == 'scene_leg' and e[2] for e in events)
    assert policy.require_plan(request, plan, node, vars(request.identity)) is plan.release_only_endpoint


@pytest.mark.parametrize('failure', ('body', 'cartesian', 'opening', 'static'))
def test_extra_proposal_cannot_bypass_original_admission(bottom_flow, failure):
    node, events, _, _, call = bottom_flow
    if failure == 'body': node._plan_carried_joint_route = lambda *a, **kw: None
    if failure == 'cartesian':
        checks = []
        def volume(*a): checks.append(None); return len(checks) == 1
        node._carried_robot_transition_is_safe = volume
    if failure == 'opening': node.reject_opening = True
    if failure == 'static': node.reject_static = True
    with pytest.raises(RuntimeError): call()
    assert not any(e[0] == 'unloaded_home' for e in events)


def test_retained_state_change_at_final_static_sample_discards_plan(bottom_flow, monkeypatch):
    node, events, _, _, call = bottom_flow
    original = planner.PlaceSceneChecker.sample
    def sample(scene, q, aperture, loaded):
        result = original(scene, q, aperture, loaded)
        if not loaded: node._transport_lock_engaged = False
        return result
    monkeypatch.setattr(planner.PlaceSceneChecker, 'sample', sample)
    with pytest.raises(RuntimeError, match='retained_identity_changed'): call()


def test_nonbottom_keeps_original_negative_proposal_dispatch(selected_planning, monkeypatch):
    node, events, _, call = selected_planning
    monkeypatch.setattr(planner, 'registered_place_proposals', lambda *a, **kw: pytest.fail('extra target'))
    monkeypatch.setattr(planner, 'nearest_negative_wrist_goals', lambda *a, **kw: pytest.fail('extra wrist'))
    plan = call()
    assert 'bottom_nearest_negative_wrist_proposal' not in plan.diagnostics
