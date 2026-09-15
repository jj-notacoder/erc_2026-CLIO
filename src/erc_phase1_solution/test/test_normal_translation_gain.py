"""Actual controller-law regressions; ideal command integration is not a plant proof."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import Twist
from rclpy.time import Time
from erc_phase1_solution import navigation_node as nav
import test_normal_linear_convergence as normal

ROOT = Path(__file__).resolve().parents[1]
BASE_NAV_SHA256 = '5a060155334edcb6f79d73cedaa68bfabc9dfb2fcc0d95edaa6e1c817c7fdb97'


def without_yaw_source():
    source = (ROOT/'erc_phase1_solution/navigation_node.py').read_bytes()
    record = json.loads((ROOT/'test/fixtures/normal_linear_convergence_inverse.json').read_bytes())
    for edit in reversed(record['replacements'][-4:]):
        assert source.count(edit['new'].encode()) == 1
        source = source.replace(edit['new'].encode(), edit['old'].encode(), 1)
    assert hashlib.sha256(source).hexdigest() == '364397dd1c780f162f62450515ca1925201a0d7a8eaea3c3363388ff84302cc0'
    return source


def previous_source():
    source = without_yaw_source()
    record = json.loads((ROOT/'test/fixtures/normal_linear_convergence_inverse.json').read_bytes())
    # After the exact four-fragment yaw inverse, retain the original translation audit.
    for edit in reversed(record['replacements'][-10:-4]):
        assert source.count(edit['new'].encode()) == 1
        source = source.replace(edit['new'].encode(), edit['old'].encode(), 1)
    assert hashlib.sha256(source).hexdigest() == BASE_NAV_SHA256
    return source


def previous_method(name):
    cls = next(n for n in ast.parse(previous_source()).body
               if isinstance(n, ast.ClassDef) and n.name == 'NavigationNode')
    method = copy.deepcopy(next(n for n in cls.body
                               if isinstance(n, ast.FunctionDef) and n.name == name))
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    scope = dict(nav.__dict__)
    exec(compile(ast.fix_missing_locations(module), '<successful55 method>', 'exec'), scope)
    return scope[name]


def node(gain=1.1, enabled=True, profile='normal'):
    n = normal.control_node(enabled=enabled, profile=profile)
    n.normal_translation_gain = nav._checked_normal_translation_gain(gain)
    return n


def vector(command):
    return command.linear.x, command.linear.y, command.angular.z


@pytest.mark.parametrize('gain', [.85, 1, 1.1, 1.25, 1.5, math.nextafter(1.1, math.inf), math.nextafter(1.5, 0.), 1.8, math.nextafter(1.8, 0.), math.nextafter(1.5, math.inf), 1.5001])
def test_valid_gain_and_actual_constructor_assignment(gain):
    cls = next(n for n in ast.parse((ROOT/'erc_phase1_solution/navigation_node.py').read_bytes()).body
               if isinstance(n, ast.ClassDef) and n.name == 'NavigationNode')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    assignment = [n for n in init.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == 'normal_translation_gain' for t in n.targets)]
    assert len(assignment) == 1
    seen = []
    def parameter(name):
        seen.append(name)
        return NS(value=gain)
    receiver = NS(get_parameter=parameter)
    scope = dict(nav.__dict__, self=receiver)
    exec(compile(ast.fix_missing_locations(ast.Module(body=assignment, type_ignores=[])),
                 '<actual gain assignment>', 'exec'), scope)
    assert seen == ['normal_translation_gain']
    assert receiver.normal_translation_gain == float(gain)
    assert type(receiver.normal_translation_gain) is float


@pytest.mark.parametrize('gain', [True, False, None, '1.1', float('nan'), float('inf'),
    -float('inf'), math.nextafter(.85, 0.), math.nextafter(1.8, math.inf), 1.8001, 0., -1., 2, 10**1000])
def test_gain_rejects_invalid_types_nonfinite_and_out_of_range(gain):
    with pytest.raises(ValueError, match='normal_translation_gain'):
        nav._checked_normal_translation_gain(gain)


def test_actual_declarations_keep_gain_and_all_motion_limits_conservative():
    values = {}
    nav.NavigationNode._declare_parameters(NS(declare_parameter=lambda name, value: values.update({name: value})))
    assert values['normal_translation_gain'] == .85
    assert values['normal_linear_convergence_enabled'] is False
    assert {k: values[k] for k in ('control_rate_hz', 'max_linear_speed', 'max_lateral_speed',
        'linear_acceleration_limit', 'position_tolerance', 'settle_time_seconds',
        'emergency_stop_distance')} == dict(control_rate_hz=20., max_linear_speed=.42,
        max_lateral_speed=.32, linear_acceleration_limit=.55, position_tolerance=.045,
        settle_time_seconds=.45, emergency_stop_distance=.28)


@pytest.mark.parametrize('xy', [(0., 0.), (.04, 0.), (.14, -.09), (.349, .01), (.7, -.8), (-3., 2.)])
@pytest.mark.parametrize('setting', ['missing', .85])
def test_default_gain_preserves_exact_successful55_translation(xy, setting):
    n = normal.control_node()
    if setting != 'missing':
        n.normal_translation_gain = setting
    args = (*xy, math.hypot(*xy))
    assert n._desired_translation(*args) == previous_method('_desired_translation')(n, *args)


@pytest.mark.parametrize('profile,enabled', [('normal', False), ('carried_retreat', True),
    ('empty_arm_staging', True), ('empty_arm_advance', True), ('custom', True)])
@pytest.mark.parametrize('xy', [(.12, -.08), (.7, .6)])
@pytest.mark.parametrize('gain', [1.1, 1.5, 1.8])
def test_selected_gain_is_inert_outside_enabled_normal_goals(profile, enabled, xy, gain):
    n = node(gain, profile=profile, enabled=enabled)
    args = (*xy, math.hypot(*xy))
    assert n._desired_translation(*args) == previous_method('_desired_translation')(n, *args)


@pytest.mark.parametrize('xy,expected', [((.1, 0.), (.11, 0.)), ((-.1, .1), (-.11, .11)),
    ((1., 1.), (.42, .32)), ((-1., -1.), (-.42, -.32)),
    ((.2, -.1), (.22, -.11)), ((0., 0.), (0., 0.))])
def test_selected_gain_preserves_direction_caps_and_zero(xy, expected):
    n = node()
    assert n._desired_translation(*xy, math.hypot(*xy)) == pytest.approx(expected)


def test_only_constructor_declaration_and_desired_law_change_in_node():
    def methods(source):
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'NavigationNode')
        return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
    old, current = methods(previous_source()), methods(without_yaw_source())
    assert old.keys() == current.keys()
    assert {k for k in old if old[k] != current[k]} == {'__init__', '_declare_parameters', '_desired_translation'}
    assert old['_limited_command'] == current['_limited_command']
    assert old['_control'] == current['_control']
    assert old['_control_empty_arm'] == current['_control_empty_arm']


@pytest.mark.parametrize('fault', ['odom_stale', 'front_stale', 'rear_stale', 'missing_scan',
    'front_obstacle', 'rear_obstacle', 'timeout', 'no_goal', 'no_pose'])
@pytest.mark.parametrize('gain', [1.1, 1.5, 1.8])
def test_actual_control_preserves_failure_stops_without_extra_motion(fault, gain):
    n = node(gain);n.last_command.linear.x = .1
    if fault.endswith('_stale'):
        attribute = {'odom_stale': 'last_odom_time', 'front_stale': 'last_front_scan_time',
                     'rear_stale': 'last_rear_scan_time'}[fault]
        setattr(n, attribute, Time(nanoseconds=0));normal.advance(n, 0)
        # advance refreshes all three producers; restore only the deliberately stale one.
        setattr(n, attribute, Time(nanoseconds=0));n.now_ns = 1_000_000_001
    elif fault == 'missing_scan': n.last_rear_scan_time = None
    elif fault == 'front_obstacle': n.front_clearance = .279
    elif fault == 'rear_obstacle': n.rear_clearance = .279
    elif fault == 'timeout':
        n.goal_started = Time(nanoseconds=0);n.now_ns = 46_000_000_000;normal.advance(n, 0)
    elif fault == 'no_goal': n.goal = None
    else: n.pose = None
    n._control()
    assert n.commands and all(vector(c) == (0., 0., 0.) for c in n.commands)
    if fault.endswith('_stale') or fault == 'missing_scan':
        assert n.goal is None and n.statuses[-1][1]['reason'] == 'stale_navigation_sensor'
    elif fault == 'timeout':
        assert n.goal is None and n.statuses[-1][1]['reason'] == 'timeout'
    elif fault.endswith('_obstacle'):
        assert n.blocked and n.statuses[-1][0] == 'blocked' and n.goal is not None


@pytest.mark.parametrize('event', ['stop', 'cancel', 'abort'])
@pytest.mark.parametrize('gain', [1.1, 1.5, 1.8])
def test_actual_cancel_wire_clears_goal_and_never_resumes_motion(event, gain):
    n = node(gain);n._control();assert any(vector(c) != (0., 0., 0.) for c in n.commands)
    n.commands.clear();n._on_command(NS(data=json.dumps({'event': event})))
    assert n.goal is None and n.statuses[-1][0] == 'cancelled'
    normal.advance(n);n._control()
    assert all(vector(c) == (0., 0., 0.) for c in n.commands)


@pytest.mark.parametrize('side', ['front', 'rear'])
@pytest.mark.parametrize('gain', [1.1, 1.5, 1.8])
def test_obstacle_during_terminal_taper_prevents_reached_and_still_stops(side, gain):
    n = node(gain);n.pose = (.1, 0., 0.);n.last_command.linear.x = .08
    setattr(n, side+'_clearance', .279)
    n._control()
    assert n.blocked and n.goal is not None and n.settle_started is None
    assert vector(n.commands[-1]) == (0., 0., 0.)
    assert all(event != 'reached' for event, _ in n.statuses)


@pytest.mark.parametrize('gain', [1.1, 1.5, 1.8])
def test_terminal_taper_keeps_original_limiter_and_full_settle_window(gain):
    n = node(gain);n.pose = (.1, 0., 0.)
    n.last_command.linear.x = .0495;n.last_command.linear.y = -.03;n.last_command.angular.z = .06
    previous = vector(n.last_command);first_settle = None;seen_taper = False
    for _ in range(30):
        n._control();current = vector(n.commands[-1])
        assert abs(current[0]-previous[0]) <= .55/20+1e-12
        assert abs(current[1]-previous[1]) <= .55/20+1e-12
        assert abs(current[2]-previous[2]) <= .8/20+1e-12
        if current != (0., 0., 0.):
            seen_taper = True;assert n.settle_started is None
        if n.settle_started is not None and first_settle is None:
            first_settle = n.settle_started.nanoseconds
        if n.goal is None: break
        previous = current;normal.advance(n)
    assert seen_taper and n.goal is None and n.statuses[-1][0] == 'reached'
    assert first_settle is not None and n.now_ns-first_settle >= 450_000_000


@pytest.mark.parametrize('goal', [(.046, 0., .04), (.04, 0., .046)])
@pytest.mark.parametrize('gain', [1.1, 1.5, 1.8])
def test_same_position_and_yaw_tolerances_gate_settle(goal, gain):
    n = node(gain);n.goal = goal;n._control()
    assert n.goal == goal and n.settle_started is None


@pytest.mark.parametrize('goal,yaw', [((.137, 0.), 0.), ((1.4, .3), math.pi/4), ((-.7, -.3), math.pi/2)])
@pytest.mark.parametrize('gain', [1.1, 1.5, 1.8])
def test_ideal_command_following_reaches_same_endpoint_with_original_steps(goal, yaw, gain):
    # Ideal odometry only exercises actual code ordering and limiter; it does not model tracking or braking.
    n = node(gain);n.pose = (0., 0., yaw);n.goal = (*goal, yaw);target = n.goal
    previous = (0., 0., 0.)
    for _ in range(600):
        n._control();command = n.commands[-1];current = vector(command)
        assert abs(current[0]) <= .42 and abs(current[1]) <= .32 and abs(current[2]) <= .55
        assert abs(current[0]-previous[0]) <= .55/20+1e-12
        assert abs(current[1]-previous[1]) <= .55/20+1e-12
        assert abs(current[2]-previous[2]) <= .8/20+1e-12
        if n.goal is None: break
        assert n.goal == target
        x, y, theta = n.pose
        n.pose = (x+(math.cos(theta)*current[0]-math.sin(theta)*current[1])/20,
                  y+(math.sin(theta)*current[0]+math.cos(theta)*current[1])/20,
                  theta+current[2]/20)
        previous = current;normal.advance(n)
    assert n.goal is None and n.statuses[-1][0] == 'reached'
    assert math.hypot(target[0]-n.pose[0], target[1]-n.pose[1]) <= .045
    assert abs(nav.normalize_angle(target[2]-n.pose[2])) <= .045


def test_default_control_trace_matches_successful55_with_original_desired_method():
    traces = []
    for historical in (False, True):
        n = node(.85);n.pose = (.1, 0., 0.);n.last_command.linear.x = .03
        if historical:
            method = previous_method('_desired_translation')
            n._desired_translation = lambda *args: method(n, *args)
        for _ in range(15):
            n._control();normal.advance(n)
        traces.append(([vector(c) for c in n.commands], n.goal, n.statuses))
    assert traces[0] == traces[1]


@pytest.mark.parametrize('axis,sign,edge', [
    ('x', 1., 'below'), ('x', 1., 'equal'), ('x', 1., 'above'),
    ('x', -1., 'below'), ('x', -1., 'equal'), ('x', -1., 'above'),
    ('y', 1., 'below'), ('y', 1., 'equal'), ('y', 1., 'above'),
    ('y', -1., 'below'), ('y', -1., 'equal'), ('y', -1., 'above')])
def test_gain15_saturation_boundary_preserves_each_original_axis_cap(axis, sign, edge):
    n = node(1.5)
    cap = n.max_vx if axis == 'x' else n.max_vy
    boundary = cap / 1.5
    offset = {'below': -1e-9, 'equal': 0., 'above': 1e-9}[edge]
    error = sign * (boundary + offset)
    xy = (error, 0.) if axis == 'x' else (0., error)
    vx, vy = n._desired_translation(*xy, math.hypot(*xy))
    observed, other = (vx, vy) if axis == 'x' else (vy, vx)
    assert other == 0. and abs(observed) <= cap
    assert observed == pytest.approx(sign * min(cap, 1.5 * abs(error)), abs=1e-15)
    if edge == 'below':
        assert abs(observed) < cap
    else:
        assert abs(observed) == pytest.approx(cap, abs=1e-15)


@pytest.mark.parametrize('xy,expected', [((.1, -.1), (.15, -.15)),
    ((.3, .3), (.42, .32)), ((-.3, -.3), (-.42, -.32)),
    ((.05, .02), (.075, .03)), ((0., 0.), (0., 0.))])
def test_gain15_diagonal_and_zero_commands_keep_original_limiter_steps(xy, expected):
    n = node(1.5)
    desired = Twist()
    desired.linear.x, desired.linear.y = n._desired_translation(*xy, math.hypot(*xy))
    assert (desired.linear.x, desired.linear.y) == pytest.approx(expected)
    limited = n._limited_command(desired)
    assert abs(limited.linear.x) <= .55 / 20 + 1e-15
    assert abs(limited.linear.y) <= .55 / 20 + 1e-15
    assert limited.angular.z == 0.
    for command, error in ((limited.linear.x, xy[0]), (limited.linear.y, xy[1])):
        assert command == 0. if error == 0. else command * error > 0.


@pytest.mark.parametrize('gain', [1.1, 1.5, 1.8])
def test_saturated_entry_requires_original_deceleration_before_settling(gain):
    n = node(gain)
    n.pose = (n.goal[0], n.goal[1], n.goal[2])
    n.last_command.linear.x = n.max_vx
    n.last_command.linear.y = -n.max_vy
    n.last_command.angular.z = n.max_wz
    previous = vector(n.last_command)
    original_goal = n.goal
    first_settle = None
    deceleration_samples = 0
    for _ in range(50):
        n._control()
        current = vector(n.commands[-1])
        assert abs(current[0] - previous[0]) <= .55 / 20 + 1e-12
        assert abs(current[1] - previous[1]) <= .55 / 20 + 1e-12
        assert abs(current[2] - previous[2]) <= .8 / 20 + 1e-12
        if max(map(abs, current)) > .005:
            deceleration_samples += 1
            assert n.goal == original_goal and n.settle_started is None
        if n.settle_started is not None and first_settle is None:
            first_settle = n.settle_started.nanoseconds
        if n.goal is None:
            break
        previous = current
        normal.advance(n)
    assert deceleration_samples >= 10
    assert first_settle is not None and n.now_ns - first_settle >= 450_000_000
    assert n.goal is None and n.statuses[-1][0] == 'reached'
    assert vector(n.commands[-1]) == (0., 0., 0.)


@pytest.mark.parametrize('axis,sign,edge', [
    ('x', 1., 'below'), ('x', 1., 'equal'), ('x', 1., 'above'),
    ('x', -1., 'below'), ('x', -1., 'equal'), ('x', -1., 'above'),
    ('y', 1., 'below'), ('y', 1., 'equal'), ('y', 1., 'above'),
    ('y', -1., 'below'), ('y', -1., 'equal'), ('y', -1., 'above')])
def test_gain18_saturation_boundary_preserves_each_original_axis_cap(axis, sign, edge):
    n = node(1.8)
    cap = n.max_vx if axis == 'x' else n.max_vy
    boundary = cap / 1.8
    offset = {'below': -1e-9, 'equal': 0., 'above': 1e-9}[edge]
    error = sign * (boundary + offset)
    xy = (error, 0.) if axis == 'x' else (0., error)
    vx, vy = n._desired_translation(*xy, math.hypot(*xy))
    observed, other = (vx, vy) if axis == 'x' else (vy, vx)
    assert other == 0. and abs(observed) <= cap
    assert observed == pytest.approx(sign * min(cap, 1.8 * abs(error)), abs=1e-15)
    if edge == 'below':
        assert abs(observed) < cap
    else:
        assert abs(observed) == pytest.approx(cap, abs=1e-15)



@pytest.mark.parametrize('xy,expected', [((.1, -.1), (.18, -.18)),
    ((.3, .3), (.42, .32)), ((-.3, -.3), (-.42, -.32)),
    ((.05, .02), (.09, .036)), ((0., 0.), (0., 0.))])
def test_gain18_diagonal_and_zero_commands_keep_original_limiter_steps(xy, expected):
    n = node(1.8)
    desired = Twist()
    desired.linear.x, desired.linear.y = n._desired_translation(*xy, math.hypot(*xy))
    assert (desired.linear.x, desired.linear.y) == pytest.approx(expected)
    limited = n._limited_command(desired)
    assert abs(limited.linear.x) <= .55 / 20 + 1e-15
    assert abs(limited.linear.y) <= .55 / 20 + 1e-15
    assert limited.angular.z == 0.
    for command, error in ((limited.linear.x, xy[0]), (limited.linear.y, xy[1])):
        assert command == 0. if error == 0. else command * error > 0.

