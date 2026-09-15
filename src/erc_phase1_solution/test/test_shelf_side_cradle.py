"""Regression checks for the opt-in support transfer before base motion."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rclpy')

from erc_phase1_solution import manipulation_node


def _planner(monkeypatch):
    from erc_phase1_solution import shelf_cradle_geometry

    node = object.__new__(manipulation_node.ManipulationNode)
    node.carried_shelf_retreat_clearance = 0.25
    node.carried_navigation_radius_limit = 0.45
    node.carried_book_dimensions = np.array([0.20, 0.04, 0.28])
    node._cached_post_retreat_plan = None
    node._solve_carried_cradle = lambda q: np.asarray(q) + np.array(
        [0, 0, 0, 0, 0, 0, 0, -1.10]
    )
    node._solve_supported_post_retreat_staging = lambda q: (
        [], [np.asarray(q) + 0.01], [np.asarray(q) + 0.02]
    )
    node._supported_compact_goals = lambda q: [np.asarray(q) + 0.03]
    node._carried_navigation_radius = lambda *args: 0.439
    calls = []

    def plan(start, goals, corners, **kwargs):
        calls.append(kwargs)
        return list(goals)

    node._plan_carried_joint_route = plan
    monkeypatch.setattr(shelf_cradle_geometry, 'check_cradle_tool_sweep',
                        lambda *a, **kw: None)
    monkeypatch.setattr(shelf_cradle_geometry, 'check_cradle_tool_route',
                        lambda *a, **kw: None)
    return node, calls


def test_support_is_established_before_the_cached_base_retreat(monkeypatch):
    node, calls = _planner(monkeypatch)
    start = np.zeros(8)
    front = np.array([0.79, -0.06, 1.58])
    corners = np.zeros((8, 3))
    route = node._plan_shelf_side_cradle(front, start, start, corners)

    assert len(route) == 1
    assert route[0][-1] == pytest.approx(-1.10)
    assert calls[0]['post_retreat_shelf_front_x'] == pytest.approx(0.725)
    assert all(c['require_gravity_support'] for c in calls[1:])
    assert all(c['post_retreat_shelf_front_x'] == pytest.approx(0.975)
               for c in calls[1:])
    cache = node._cached_post_retreat_plan
    assert cache['requires_gravity_support'] is True
    np.testing.assert_array_equal(cache['start'], route[-1])
    assert [phase for _, phase in cache['legs']] == [
        'supported_cradle_lowering', 'supported_cradle_retraction', 'compact_transport'
    ]


def test_a_palm_collision_blocks_the_plan_before_any_cache_is_armed(monkeypatch):
    from erc_phase1_solution import shelf_cradle_geometry

    node, calls = _planner(monkeypatch)
    monkeypatch.setattr(shelf_cradle_geometry, 'check_cradle_tool_sweep',
                        lambda *a, **kw: 'book_palm_overlap')
    with pytest.raises(RuntimeError, match='book_palm_overlap'):
        node._plan_shelf_side_cradle([0.79, 0, 1.58], np.zeros(8),
                                    np.zeros(8), np.zeros((8, 3)))
    assert len(calls) == 1
    assert node._cached_post_retreat_plan is None


def test_a_cached_tool_collision_rejects_the_whole_plan_before_cache_is_armed(
    monkeypatch,
):
    from erc_phase1_solution import shelf_cradle_geometry

    node, _ = _planner(monkeypatch)
    front = np.array([0.79, 0.0, 1.58])
    start = np.zeros(8)

    def reject_route(actual_node, book, grasp, cradle, goals, plane, **kwargs):
        assert actual_node is node
        np.testing.assert_array_equal(book, front)
        np.testing.assert_array_equal(grasp, start)
        assert cradle[-1] == pytest.approx(-1.10)
        assert len(goals) == 3  # Includes lowering, retraction, and compaction.
        assert plane == pytest.approx(0.975)
        assert kwargs == {'aperture': 0.04}
        return 'leg1:cradle_tool_robot_intersection'

    monkeypatch.setattr(shelf_cradle_geometry, 'check_cradle_tool_route',
                        reject_route)
    with pytest.raises(RuntimeError, match='Supported compact tool sweep rejected'):
        node._plan_shelf_side_cradle(front, start, start, np.zeros((8, 3)))
    assert node._cached_post_retreat_plan is None


def test_a_supported_posture_does_not_bypass_the_navigation_radius(monkeypatch):
    node, _ = _planner(monkeypatch)
    node._carried_navigation_radius = lambda *args: 0.451
    with pytest.raises(RuntimeError, match='navigation radius'):
        node._plan_shelf_side_cradle([0.79, 0, 1.58], np.zeros(8),
                                    np.zeros(8), np.zeros((8, 3)))
    assert node._cached_post_retreat_plan is None


def test_a_stale_unsupported_cache_cannot_be_used_as_a_supported_cache():
    node = object.__new__(manipulation_node.ManipulationNode)
    node._held_book_corners = np.zeros((8, 3))
    node._measured_left_solution = lambda: np.zeros(8)
    node._post_retreat_shelf_front_x = 0.975
    node._gravity_supported_payload = True
    node._supported_post_retreat_staging_required = True
    node._cached_post_retreat_plan = {'requires_gravity_support': False}
    node._execute_cached_post_retreat_compaction = lambda *args: pytest.fail(
        'a supported payload must not repeat the vertical-to-supported roll'
    )
    with pytest.raises(RuntimeError, match='requires a vertical bilateral pinch'):
        node._compact_transport()


def test_supported_cache_executes_once_without_repeating_roll_and_cleans_up():
    node = object.__new__(manipulation_node.ManipulationNode)
    start = np.array([0.30, 0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -1.10])
    lowering = start.copy()
    lowering[2] += 0.03
    staging = lowering.copy()
    staging[3] += 0.04
    terminal = staging.copy()
    terminal[4] += 0.05
    corners = np.arange(24, dtype=float).reshape(8, 3) / 1000
    phases = ['supported_cradle_lowering', 'supported_cradle_retraction',
              'compact_transport']
    waypoints = [lowering, staging, terminal]
    node._held_book_corners = corners.copy()
    node._carried_staging_solution = start.copy()
    node._post_retreat_shelf_front_x = 0.975
    node._gravity_supported_payload = True
    node._supported_post_retreat_staging_required = True
    node.carried_navigation_radius_limit = 0.45
    node.timeout = 10.0
    cache = {
        'start': start.copy(),
        'shelf_front_x': 0.975,
        'attached_corners': corners.copy(),
        'requires_gravity_support': True,
        'legs': list(zip(waypoints, phases)),
        'staging_terminal': staging.copy(),
        'terminal': terminal.copy(),
        'compact_radius': 0.439,
    }
    node._cached_post_retreat_plan = cache
    events = []
    measured = iter([start, start, terminal])

    def measure():
        events.append('measure')
        return next(measured).copy()

    def shelf_safe(first, last, attached, plane):
        np.testing.assert_array_equal(attached, corners)
        assert plane == pytest.approx(0.975)
        events.append(('shelf', first.copy(), last.copy()))
        return True

    def support_safe(first, last):
        events.append(('support', first.copy(), last.copy()))
        return True

    def server(**kwargs):
        events.append('server')
        return True

    def retained(command, phase, **kwargs):
        assert command == 'compact_transport'
        events.append(phase)
        if phase == 'post_retreat':
            assert kwargs == {'require_new_sample': False}
        else:
            assert phase == 'compact_transport_final'
            assert kwargs == {'leg': 3}
        return True

    def send(goal, duration, legs, command):
        events.append('dispatch')
        assert command == 'compact_transport'
        assert [phase for _, _, phase in legs] == phases
        assert duration == pytest.approx(sum(seconds for _, seconds, _ in legs))
        assert len(goal.trajectory.points) == 3
        assert goal.trajectory.joint_names == list(manipulation_node.ARM_JOINTS)
        for point, expected in zip(goal.trajectory.points, waypoints):
            np.testing.assert_array_equal(point.positions, expected[1:])
            # Support was established before the base retreat: no new q7 roll.
            assert point.positions[-1] == start[-1]
        return True, False

    def forbidden(*args, **kwargs):
        pytest.fail('supported cache must not replan, reroll, or dispatch per leg')

    node._measured_left_solution = measure
    node._carried_post_retreat_transition_is_safe = shelf_safe
    node._gravity_supported_transition_is_safe = support_safe
    node.arm_client = SimpleNamespace(wait_for_server=server)
    node._fresh_retention_probe = retained
    node._send_retained_arm_trajectory = send
    node._carried_navigation_radius = lambda *args: 0.438
    node._publish_status = lambda event, **kwargs: events.append((event, kwargs))
    node._execute_cached_post_retreat_compaction = forbidden
    node._execute_retained_arm_legs = forbidden
    node._plan_carried_joint_route = forbidden
    node._solve_carried_cradle = forbidden

    assert node._compact_transport() is True

    # Check the measured start and first sweep before the last contact proof,
    # then remeasure before one continuous dispatch. Validate the endpoint too.
    labels = [entry if isinstance(entry, str) else entry[0] for entry in events]
    assert labels == [
        'measure', 'shelf', 'support', 'shelf', 'support', 'server',
        'post_retreat', 'measure', 'dispatch', 'measure', 'shelf', 'support',
        'compact_transport_final', 'transport_compact',
    ]
    sweeps = [entry for entry in events if isinstance(entry, tuple)
              and entry[0] == 'shelf']
    for (_, first, last), expected in zip(
        sweeps, [(start, start), (start, lowering), (terminal, terminal)]
    ):
        np.testing.assert_array_equal(first, expected[0])
        np.testing.assert_array_equal(last, expected[1])
    assert events[-1][1]['planar_radius'] == pytest.approx(0.438)
    assert node._cached_post_retreat_plan is None
    assert node._post_retreat_shelf_front_x is None
    assert node._supported_post_retreat_staging_required is False
    assert node._gravity_supported_payload is True
    np.testing.assert_array_equal(node._held_book_corners, corners)
    np.testing.assert_array_equal(node._carried_staging_solution, staging)
    assert node._carried_staging_solution is not cache['staging_terminal']
