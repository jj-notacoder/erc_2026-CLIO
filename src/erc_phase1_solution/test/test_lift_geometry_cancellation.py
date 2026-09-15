"""Cancellation boundaries and exact per-sample preprocessing reuse."""
import threading

import numpy as np
import pytest

from erc_phase1_solution import lift_first_extraction as lift
from erc_phase1_solution import shelf_cradle_geometry as geometry
from erc_phase1_solution.kinematics import PreparedTriangleMesh
from test_lift_first_extraction import scenario, plan  # noqa: F401
from test_shelf_cradle_geometry import node_with_surface, check, PALM


def validate(scenario, result):
    node, front, grasp, _, bay, *_ = scenario
    return lift.validate_lift_first_route(node, front, grasp, result.route,
        bay=bay, aperture=.0183, attached_corners=result.attached_corners)


def test_cancel_before_measured_route_has_no_collision_work(scenario):
    result = plan(scenario)
    node, *_, calls, _ = scenario
    calls.clear()
    node._cancel = threading.Event()
    node._cancel.set()
    with pytest.raises(ValueError, match='geometry check cancelled'):
        validate(scenario, result)
    assert calls == []


@pytest.mark.parametrize('during', ['payload', 'tool'])
def test_cancel_between_passes_never_checks_next_leg(scenario, monkeypatch, during):
    result = plan(scenario)
    node, *_, calls, _ = scenario
    calls.clear()
    node._cancel = threading.Event()
    if during == 'payload':
        def payload(*args):
            calls.append(('cancelled_payload',))
            node._cancel.set()
            return True
        node._carried_robot_transition_is_safe = payload
    else:
        def tool(*args, **kwargs):
            calls.append(('cancelled_tool',))
            node._cancel.set()
            return None
        monkeypatch.setattr(lift, 'check_cradle_tool_sweep', tool)
    with pytest.raises(ValueError, match='geometry check cancelled'):
        validate(scenario, result)
    assert [c[0] for c in calls] == (['cancelled_payload'] if during == 'payload'
                                    else ['robot', 'cancelled_tool'])


def test_cancel_inside_bay_samples_stops_before_remaining_samples(scenario, monkeypatch):
    result = plan(scenario)
    node = scenario[0]
    node._cancel = threading.Event()
    forward, samples = node.chain.forward, []
    bay_started = False
    original_tool = lift.check_cradle_tool_sweep
    def tool(*args, **kwargs):
        nonlocal bay_started
        value = original_tool(*args, **kwargs)
        bay_started = True
        return value
    def during(q):
        if bay_started:
            samples.append(np.asarray(q).copy())
            if len(samples) == 7:
                node._cancel.set()
        return forward(q)
    monkeypatch.setattr(lift, 'check_cradle_tool_sweep', tool)
    node.chain.forward = during
    with pytest.raises(ValueError, match='geometry check cancelled'):
        validate(scenario, result)
    assert len(samples) == 7


def test_tool_cancel_precedes_mesh_checks():
    node = node_with_surface(palm_x=.05)
    node._cancel = threading.Event()
    node._cancel.set()
    assert check(node) == 'cradle_tool_cancelled'


def test_tool_cancel_is_checked_at_each_unchanged_density_sample():
    node = node_with_surface()
    node.carried_transition_samples = 61
    node._cancel = threading.Event()
    checked = []
    def self_collision(q):
        checked.append(np.asarray(q).copy())
        if len(checked) == 2:
            node._cancel.set()
        return None
    node._robot_self_collision = self_collision
    assert check(node, end=(.3,)) == 'cradle_tool_cancelled'
    assert len(checked) == 2
    assert [q[0] for q in checked] == pytest.approx([0., .005])


def test_prepared_mesh_reused_only_within_exact_world_sample(monkeypatch):
    local = np.asarray([[[-.1,-.1,-.1],[-.1,.1,-.1],[-.1,0.,.1]]])
    world = local + [0.,0.,1.]
    node = node_with_surface(robot={'arm_left_3_link':world.copy(), 'torso_lift_link':world.copy()})
    tip = 'gripper_left_fingertip_left_link'
    node._shelf_cradle_geometry.local_surfaces = lambda *_: {PALM:local.copy(),tip:local.copy()}
    node._shelf_cradle_geometry.watertight = {PALM:False,tip:False}
    pairs = []
    def no_intersection(first, second, **kwargs):
        assert isinstance(first,PreparedTriangleMesh) and isinstance(second,PreparedTriangleMesh)
        assert np.array_equal(first.surface,world) and np.array_equal(second.surface,world)
        pairs.append((first,second))
        return False
    monkeypatch.setattr(geometry,'triangle_meshes_intersect',no_intersection)
    assert check(node) is None
    assert len(pairs) == 4
    assert pairs[0][0] is pairs[1][0]
    assert pairs[0][1] is pairs[2][1]
    assert pairs[1][1] is pairs[3][1]
    original_objects = {id(value) for pair in pairs for value in pair}
    assert check(node) is None
    assert original_objects.isdisjoint({id(value) for pair in pairs[4:] for value in pair})
