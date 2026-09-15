"""Pure fixed-context propagation without ROS or simulator inputs."""
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution import lift_first_extraction as lift
from erc_phase1_solution import shelf_cradle_geometry as tool
from test_lift_first_extraction import scenario, plan
from test_shelf_cradle_geometry import node_with_surface


def test_all_route_passes_share_private_exact_context_while_live_state_changes(scenario, monkeypatch):
    node, front, grasp, _, bay, _, _ = scenario
    planned = plan(scenario)
    right = np.asarray([.1, .2, .3, .4, .5, .6, .7])
    head = np.asarray([-.2, .1])
    expected_right, expected_head = right.copy(), head.copy()
    node.joints = {'live': 0.}
    visits = []
    actual_start = grasp.copy()
    actual_start[1] += .00001
    fingers = {'gripper_left_finger_joint': .01825}

    def verify(phase, first, **kwargs):
        r, h = kwargs['right_positions'], kwargs['head_positions']
        assert np.array_equal(r, expected_right)
        assert np.array_equal(h, expected_head)
        assert not r.flags.writeable and not h.flags.writeable
        visits.append((phase, first.copy(), r, h))
        right[:] = 88.  # Original caller arrays and callback state can change.
        head[:] = 99.
        node.joints['live'] += 1

    def robot(first, last, attached, **kwargs):
        verify('robot', first, **kwargs)
        assert np.array_equal(attached, planned.attached_corners)
        return True

    def checked_tool(node, point, original, first, last, plane, **kwargs):
        verify('tool', first, **kwargs)
        assert kwargs['aperture'] == .01825
        assert kwargs['finger_positions'] is fingers
        return None

    node._carried_robot_transition_is_safe = robot
    monkeypatch.setattr(lift, 'check_cradle_tool_sweep', checked_tool)
    result = lift.validate_lift_first_route(node, front, actual_start, planned.route,
        bay=bay, aperture=.01825, finger_positions=fingers,
        attached_corners=planned.attached_corners,
        right_positions=right, head_positions=head)
    assert [v[0] for v in visits] == ['robot', 'tool'] * 3
    assert all(v[2] is visits[0][2] and v[3] is visits[0][3] for v in visits)
    assert np.array_equal(visits[0][1], actual_start)
    assert node.joints['live'] == 6
    assert [r['samples'] for r in result['legs']] == [61, 61, 61]


@pytest.mark.parametrize('key,value', [
    ('right_positions', [0.] * 6), ('right_positions', [0.] * 6 + [np.nan]),
    ('head_positions', [0.] * 3), ('head_positions', [0., np.inf]),
])
def test_invalid_context_rejected_before_route_collision_work(scenario, key, value):
    node, front, grasp, _, bay, calls, _ = scenario
    planned = plan(scenario)
    calls.clear()
    with pytest.raises(ValueError, match=key):
        lift.validate_lift_first_route(node, front, grasp, planned.route,
            bay=bay, aperture=.0183, **{key: value})
    assert calls == []


def test_tool_self_and_mesh_passes_use_same_snapshot_at_every_sample():
    node = node_with_surface()
    right, head = np.arange(7, dtype=float), np.asarray([.2, .3])
    original_right, original_head = right.copy(), head.copy()
    visits = []

    def verify(phase, q, **kwargs):
        assert np.array_equal(kwargs['right_positions'], original_right)
        assert np.array_equal(kwargs['head_positions'], original_head)
        assert not kwargs['right_positions'].flags.writeable
        visits.append((phase, q.copy()))
        right[:] += 1.
        head[:] += 1.

    def self_collision(q, **kwargs):
        verify('self', q, **kwargs)
        return None

    def surfaces(q, **kwargs):
        verify('surfaces', q, **kwargs)
        return {}

    node._robot_self_collision = self_collision
    node._world_collision_surfaces = surfaces
    assert tool.check_cradle_tool_sweep(node, [0., 0., 1.], [0.], [0.], [.06], None,
        right_positions=right, head_positions=head) is None
    assert [r[0] for r in visits] == ['self', 'surfaces'] * 4
    assert [r[1][0] for r in visits[::2]] == pytest.approx([0., .02, .04, .06])


@pytest.mark.parametrize('key,value', [
    ('right_positions', []), ('right_positions', [0.] * 6 + [np.inf]),
    ('head_positions', [0.]), ('head_positions', [np.nan, 0.]),
])
def test_tool_invalid_explicit_context_never_uses_live_fallback(key, value):
    node = node_with_surface()
    node._robot_self_collision = lambda *a, **kw: pytest.fail('invalid context reached collision')
    assert tool.check_cradle_tool_sweep(node, [0., 0., 1.], [0.], [0.], [0.], None,
        **{key: value}) == f'cradle_tool_{key}_invalid'


def test_pass_timing_reports_each_unchanged_leg_without_actuation(scenario, monkeypatch):
    node, front, grasp, _, bay, _, _ = scenario
    planned = plan(scenario)
    events = []
    node._publish_status = lambda event, **fields: events.append((event, fields))
    ticks = iter(np.arange(1000, dtype=float) * .01)
    monkeypatch.setattr(lift, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    checked = lift.validate_lift_first_route(node, front, grasp, planned.route,
        bay=bay, aperture=.0183)
    assert len(events) == 9
    assert all(name == 'lift_first_geometry_progress' for name, _ in events)
    assert [f['geometry_pass'] for _, f in events] == ['robot_payload', 'tool', 'relative_bay'] * 3
    assert [f['leg'] for _, f in events] == [0] * 3 + [1] * 3 + [2] * 3
    assert all(f['pass_wall_seconds'] > 0 for _, f in events)
    assert checked['validation_wall_seconds'] > 0
    assert all(set(r['pass_wall_seconds']) == {'robot_payload', 'tool', 'bay'} for r in checked['legs'])
