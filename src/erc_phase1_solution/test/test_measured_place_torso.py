"""Measured torso selection, complete plan wiring and post-plan rejection."""
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution.settled_torso import (
    choose_measured_place_height, require_planned_place_height,
)
from test_scene_checked_place_planner import planning
import test_scene_checked_place_flow as flow

JOINT = 'torso_lift_joint'
HEIGHT = .35 - 13.472334e-9


def feedback(position=HEIGHT, velocity=0., stamp=1_990_000_000):
    return SimpleNamespace(
        _lock=threading.Lock(), joints={JOINT: position},
        _joint_velocities={JOINT: velocity}, _joint_stamps_ns={JOINT: stamp},
        chain=SimpleNamespace(lower=[0.], upper=[.35]),
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=2_000_000_000)),
    )


def test_measured_height_is_preserved_exactly_and_does_not_mutate_inputs():
    node = feedback()
    start = np.array([HEIGHT, 1., 2., 3., 4., 5., 6., 7.])
    before = start.copy()
    assert choose_measured_place_height(node, start, .35) == HEIGHT
    np.testing.assert_array_equal(start, before)
    assert node.joints == {JOINT: HEIGHT}
    require_planned_place_height(node, HEIGHT)


@pytest.mark.parametrize('field,value', [
    ('position', math.nan), ('position', math.inf), ('position', -.001),
    ('position', .35001), ('position', HEIGHT-2e-6),
    ('velocity', math.nan), ('velocity', math.inf), ('velocity', 2e-6),
    ('velocity', -2e-6), ('stamp', 0), ('stamp', 1_649_999_999),
    ('stamp', 2_050_000_001), ('stamp', math.nan),
])
def test_invalid_stale_moving_or_changed_feedback_preserves_nominal_and_rejects_hold(field, value):
    node = feedback(**{field:value})
    assert choose_measured_place_height(node, [HEIGHT], .35) is None
    with pytest.raises(RuntimeError, match='measured_torso_changed'):
        require_planned_place_height(node, HEIGHT)


@pytest.mark.parametrize('field', ['joints', '_joint_velocities', '_joint_stamps_ns', '_lock', 'chain'])
def test_missing_measurements_do_not_enable_optimization(field):
    node = feedback()
    delattr(node, field)
    assert choose_measured_place_height(node, [HEIGHT], .35) is None
    with pytest.raises(RuntimeError, match='measured_torso_changed'):
        require_planned_place_height(node, HEIGHT)


@pytest.mark.parametrize('start,nominal', [
    ([math.nan], .35), ([math.inf], .35), ([], .35),
    ([HEIGHT], math.nan), ([HEIGHT], .30), ([HEIGHT], .35001),
    ([HEIGHT-2e-6], .35),
])
def test_distinct_invalid_or_out_of_limits_target_uses_original_nominal_path(start, nominal):
    assert choose_measured_place_height(feedback(), start, nominal) is None


def test_scene_planner_uses_supplied_ready_height_for_ik_and_home(planning):
    node, events, call = planning
    # The supplied torso-ready state is .30; the immutable nominal config
    # differs. Both solver and HOME must follow the actual planned coordinate.
    node.place_torso_height = .35
    original = node._solve_cartesian_path
    heights = []
    def solve(positions, rotations, height, **kwargs):
        heights.append(height)
        return original(positions, rotations, height, **kwargs)
    node._solve_cartesian_path = solve
    result = call()
    assert heights == [.30]
    homes = [row[2][0] for row in events if row[0] == 'unloaded_home']
    assert homes == [.30]
    assert result.diagnostics['torso_ready'][0] == .30
    assert node.place_torso_height == .35


def test_registered_scene_solver_receives_the_same_effective_height(planning, monkeypatch, tmp_path):
    from test_scene_planning_budget_propagation import (
        test_scene_helper_forwards_wall_budget_without_replacing_candidate_guard,
    )
    node, _, _ = planning
    node.place_torso_height = .35
    original = node._solve_cartesian_path
    heights = []
    def solve(positions, rotations, height, **kwargs):
        heights.append(height)
        return original(positions, rotations, height, **kwargs)
    node._solve_cartesian_path = solve
    test_scene_helper_forwards_wall_budget_without_replacing_candidate_guard(
        planning, monkeypatch, tmp_path, 1800.)
    assert heights == [.30]
    assert node.place_torso_height == .35


def wired_place():
    case = flow.PlaceWiring(methodName='runTest')
    case.setUp()
    node = case.node
    measurement = feedback()
    for name, value in vars(measurement).items():
        setattr(node, name, value)
    node.place_torso_height = .35
    node.measured_place_height_enabled = True  # Explicit legacy experiment.
    node.table_scene_required = True
    node._selected_place_scene_reference = None
    node._selected_place_table_scene = {'synthetic': True}
    command_lock = threading.RLock()
    node._adaptive_command_guard = lambda:command_lock
    node._measured_left_solution = lambda:np.array([HEIGHT]+[1.]*7)
    case.namespace['choose_measured_place_height'] = choose_measured_place_height
    case.namespace['require_planned_place_height'] = require_planned_place_height
    return case


def test_actual_place_uses_exact_measured_height_and_still_dispatches_normal_torso_action():
    case = wired_place()
    assert case.namespace['_place'](case.node)
    _, _, carried, ready, staging, _ = case.plan_calls[0]
    np.testing.assert_array_equal(carried, ready)
    assert ready[0] == staging[0] == HEIGHT
    torso = [row[1] for row in case.calls if row[0] == 'torso']
    assert torso == [(HEIGHT, 2.2)]
    legs = next(row[1] for row in case.calls if row[0] == 'execute')
    assert all(leg[0][0] == HEIGHT for leg in legs)
    assert case.node.place_torso_height == .35
    assert ('retention', ('place', 'pre_place_motion')) in case.calls
    assert ('retention', ('place', 'torso')) in case.calls


@pytest.mark.parametrize('change', ['position', 'velocity', 'stale', 'missing'])
def test_postplanning_measurement_change_stops_before_any_action_or_release(change):
    case = wired_place()
    original = case.namespace['plan_scene_checked_place']
    def changed(*args, **kwargs):
        plan = original(*args, **kwargs)
        if change == 'position': case.node.joints[JOINT] -= 2e-6
        elif change == 'velocity': case.node._joint_velocities[JOINT] = 2e-6
        elif change == 'stale': case.node._joint_stamps_ns[JOINT] = 1
        else: case.node._joint_velocities.clear()
        return plan
    case.namespace['plan_scene_checked_place'] = changed
    with pytest.raises(RuntimeError, match='measured_torso_changed'):
        case.namespace['_place'](case.node)
    assert not any(row[0] in ('torso', 'execute', 'open', 'return') for row in case.calls)


def test_nonselected_place_keeps_its_nominal_height():
    case = wired_place()
    case.node.table_scene_required = False
    assert case.namespace['_place'](case.node)
    assert case.plan_calls[0][3][0] == .35
    assert next(row[1] for row in case.calls if row[0] == 'torso') == (.35, 2.2)


def actual_follow_node(guarded, on_server_wait):
    from erc_phase1_solution.manipulation_node import ManipulationNode
    from erc_phase1_solution.place_contact_guard import PlaceContactGuard
    node = object.__new__(ManipulationNode)
    for name, value in vars(feedback()).items():
        setattr(node, name, value)
    node._cancel = threading.Event()
    node._goal_handles = []
    node.timeout = 1.
    node._place_contact_guard = PlaceContactGuard(1, {}) if guarded else None
    sent = []
    def ready(**kwargs):
        on_server_wait(node)
        return True
    node.torso_client = SimpleNamespace(
        wait_for_server=ready,
        send_goal_async=lambda goal:sent.append(goal) or SimpleNamespace(accepted=False),
    )
    node._wait_future = lambda value, timeout:value
    return node, sent


@pytest.mark.parametrize('guarded', [False, True])
@pytest.mark.parametrize('change', ['position', 'velocity', 'stamp'])
def test_real_follow_rechecks_torso_after_server_wait_before_send(guarded, change):
    def drift(node):
        if change == 'position': node.joints[JOINT] -= 2e-6
        elif change == 'velocity': node._joint_velocities[JOINT] = 2e-6
        else: node._joint_stamps_ns[JOINT] = 1
    node, sent = actual_follow_node(guarded, drift)
    with pytest.raises(RuntimeError, match='measured_torso_changed'):
        node._move_torso(HEIGHT, 2.2, planned_start=HEIGHT)
    assert sent == []


@pytest.mark.parametrize('guarded', [False, True])
def test_real_follow_keeps_normal_action_and_exact_selected_height(guarded):
    node, sent = actual_follow_node(guarded, lambda node:None)
    # The fake server rejects acceptance after recording the correctly guarded
    # request; the method must not turn that response into success.
    assert node._move_torso(HEIGHT, 2.2, planned_start=HEIGHT) is False
    assert len(sent) == 1
    assert list(sent[0].trajectory.points[0].positions) == [HEIGHT]


@pytest.mark.parametrize('guarded', [False, True])
def test_cancel_during_pre_send_check_cannot_send_a_goal(guarded):
    node, sent = actual_follow_node(guarded, lambda node:None)
    assert node._follow(node.torso_client, [JOINT], [HEIGHT], 2.2,
                        pre_send_check=node._cancel.set) is False
    assert sent == []


@pytest.mark.parametrize('mode', ['missing', 'false'])
def test_normal_place_uses_full_nominal_height_route_without_measured_shortcut(mode):
    case = wired_place()
    if mode == 'missing':
        del case.node.measured_place_height_enabled
    else:
        case.node.measured_place_height_enabled = False
    def unexpected_shortcut(*args, **kwargs):
        raise AssertionError('normal nominal route must not use measured-height admission')
    case.namespace['choose_measured_place_height'] = unexpected_shortcut
    case.namespace['require_planned_place_height'] = unexpected_shortcut
    assert case.namespace['_place'](case.node)
    _, _, carried, ready, staging, _ = case.plan_calls[0]
    assert carried[0] == HEIGHT
    assert ready[0] == staging[0] == .35
    assert ready[0] != carried[0]
    np.testing.assert_array_equal(ready[1:], carried[1:])
    assert [row[1] for row in case.calls if row[0] == 'torso'] == [(.35, 2.2)]
    legs = next(row[1] for row in case.calls if row[0] == 'execute')
    assert all(leg[0][0] == .35 for leg in legs)
    assert ('retention', ('place', 'pre_place_motion')) in case.calls
    assert ('retention', ('place', 'torso')) in case.calls
    assert case.node.place_torso_height == .35
