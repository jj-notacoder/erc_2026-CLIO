"""The empty setup needs a complete collision certificate and real backoff."""
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution import empty_arm_staging as staging


NAMES = ['torso_lift_joint', *(f'arm_left_{j}_joint' for j in range(1, 8))]


def test_endpoint_moves_use_measured_start_and_current_target():
    start = np.asarray([.35, .11, -.21, .32, -.43, .54, -.65, .76])
    goal = np.asarray([.35, .22, .31, -.44, -.53, .66, .75, -.88])
    order = (1, 3, 6, 2, 4, 5, 7)
    route = staging.endpoint_order_route(NAMES, start, goal, order)
    previous = start
    for joint, waypoint in zip(order, route):
        assert np.flatnonzero(waypoint != previous).tolist() == [joint]
        assert waypoint[joint] == goal[joint]
        previous = waypoint
    np.testing.assert_array_equal(route[-1], goal)


def test_empty_setup_refuses_an_unchecked_torso_change():
    with pytest.raises(ValueError, match='fixed non-arm joints'):
        staging.endpoint_order_route(NAMES, np.zeros(8), np.ones(8), range(1, 8))


def fake_planner(monkeypatch):
    def forward(joints):
        transform = np.eye(4)
        # The setup reaches farther forward than either endpoint.
        transform[0, 3] = .5 + .35 * np.sin(np.pi * joints[2])
        transform[2, 3] = 1.
        return transform

    triangle = np.asarray([[[0., 0., 0.], [.03, 0., 0.], [0., .02, 0.]]])
    node = SimpleNamespace(
        chain=SimpleNamespace(active_names=NAMES, forward=forward),
        carried_transition_samples=5, carried_shelf_margin=.02,
        _shelf_cradle_geometry=SimpleNamespace(local_surfaces=lambda _: {'tool': triangle}),
        _world_collision_surfaces=lambda _: {},
    )
    start, goal = np.zeros(8), np.asarray([0., .1, 1., .1, .1, .1, .1, .1])
    calls = []
    monkeypatch.setattr(staging, 'check_cradle_tool_sweep', lambda *a, **kw: None)
    monkeypatch.setattr(
        staging, 'check_cradle_tool_route',
        lambda *a, **kw: calls.append((a[5], kw['aperture'])),
    )
    return node, start, goal, calls


def test_path_maximum_sets_backoff_and_endpoint_sets_advance_footprint(monkeypatch):
    node, start, goal, full_checks = fake_planner(monkeypatch)
    plan = staging.plan_empty_arm_staging(node, [0, 0, 1], goal, start, goal, .8)
    assert plan.minimum_sampled_backoff_m == pytest.approx(.1)
    assert plan.staging_backoff_m == pytest.approx(.11)
    assert plan.setup_maximum_robot_tool_x == pytest.approx(.88)
    assert plan.advance_robot_tool_bounds[1, 0] == pytest.approx(.53)
    assert plan.final_shelf_clearance_m == pytest.approx(.27)
    assert plan.advance_robot_tool_radius_m == pytest.approx(.53)
    assert full_checks == [(pytest.approx(.91), 0.)]


def test_available_staging_space_cannot_be_silently_exceeded(monkeypatch):
    node, start, goal, full_checks = fake_planner(monkeypatch)
    with pytest.raises(RuntimeError, match='requires 0.1100m backoff'):
        staging.plan_empty_arm_staging(
            node, [0, 0, 1], goal, start, goal, .8, maximum_backoff_m=.05,
        )
    assert not full_checks


def test_coarse_pass_does_not_replace_full_collision_check(monkeypatch):
    node, start, goal, _ = fake_planner(monkeypatch)
    monkeypatch.setattr(staging, 'check_cradle_tool_route', lambda *a, **kw: 'collision')
    with pytest.raises(RuntimeError, match='no checked empty-arm staging route'):
        staging.plan_empty_arm_staging(node, [0, 0, 1], goal, start, goal, .8)


def test_unsafe_advance_endpoint_is_rejected_before_setup_search(monkeypatch):
    node, start, goal, full_checks = fake_planner(monkeypatch)
    monkeypatch.setattr(staging, 'check_cradle_tool_sweep', lambda *a, **kw: 'shelf_contact')
    with pytest.raises(RuntimeError, match='advance endpoint rejected: shelf_contact'):
        staging.plan_empty_arm_staging(node, [0, 0, 1], goal, start, goal, .8)
    assert not full_checks
