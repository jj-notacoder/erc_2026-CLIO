"""Actual mission schedule and existing perception gates; no visibility proof."""
import ast
import copy
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest

import test_marker_search_completion as original

ROOT = Path(__file__).resolve().parents[1]


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def mission(*, elapsed=1., negative=True, limit=8):
    node, calls = original.mission(enabled=negative, elapsed=elapsed)
    node.marker_search_first_probe_enabled = True
    node._marker_search_probe_origin = None
    node.search_turn = -math.pi/4
    node.max_search_turns = limit
    node.robot_pose = (1.2, -3.4, .37)
    node.now = 1_400_000_000
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=node.now))
    node._marker_search_epoch = 1
    cls = original.methods('mission_manager.py',
        ['_marker_search_probe_yaw', '_nav_reached', '_nav_failed'], dict(math=math))
    for name in ('_marker_search_probe_yaw', '_nav_reached', '_nav_failed'):
        setattr(node, name, MethodType(getattr(cls, name), node))
    node._tick.__func__.__globals__['TransformException'] = ValueError
    node.nav_event = None
    node.navigation_timeout = 45.
    return node, calls


def goals(calls):
    return [call[1] for call in calls if call[0] == 'navigate']


def negatives(node):
    request = node._marker_search_negative_gate.request
    for sequence in (1, 2, 3):
        stamp = max(node.now, request.not_before_ns)+70_000_000
        data = original.event(sequence)
        data.update(request.fields(), rgb_stamp_ns=stamp, depth_stamp_ns=stamp,
                    completed_ros_ns=stamp+10_000_000,
                    frame_valid_until_ns=stamp+original.m.MAX_AGE_NS)
        node.now = data['completed_ros_ns']
        node._on_perception_status(NS(data=json.dumps(data)))


def reached(node, yaw):
    node.robot_pose = (node.robot_pose[0]+.001, node.robot_pose[1]-.001, yaw)
    node.nav_event = {'event': 'reached'}
    node.now += 100_000_000
    node._tick()
    assert node.state == 'SEARCH_COLUMN'
    assert node.marker_cloud is None and node._marker_search_negative_gate.count == 0
    node.nav_event = None


@pytest.mark.parametrize('initial', [.37, math.pi-.01, -math.pi+.01])
def test_actual_eight_goal_schedule_uses_frozen_origin_not_accumulated_endpoint_error(initial):
    node, calls = mission()
    node.robot_pose = (1.2, -3.4, initial)
    expected = [-math.pi/3, *(-i*math.pi/4 for i in range(1, 8))]
    for index, offset in enumerate(expected):
        current_xy = node.robot_pose[:2]
        node._tick()
        assert node.state == 'SEARCH_TURN' and node.search_turns == index+1
        assert len(goals(calls)) == index+1
        target = goals(calls)[-1]
        assert target[:2] == current_xy and target[3] == 'visual_search_turn'
        assert target[2] == pytest.approx(wrap(initial+offset), abs=1e-12)
        reached(node, wrap(target[2]+(.04 if index % 2 == 0 else -.04)))
        before = len(goals(calls))
        node._tick()
        assert len(goals(calls)) == before  # no stale negative streak is reused
        negatives(node)
    node._tick()
    assert len(goals(calls)) == 8 and node.search_turns == 8
    assert calls[-1] == ('abort', 'shelf_markers_not_found')
    assert node._marker_search_probe_origin == (original.TRIAL, initial)
    # Including the original zero view, the fallback visits each old nominal
    # 45-degree direction once. Actual attained poses remain tolerance-bounded.
    offsets = [0., *(wrap(g[2]-initial) for g in goals(calls)[1:])]
    assert all(any(abs(wrap(actual-(-i*math.pi/4))) < 1e-12 for actual in offsets)
               for i in range(8))


@pytest.mark.parametrize('limit,expected', [(0,0), (1,1), (3,3), (8,8), (99,8)])
def test_configured_lower_limit_and_hard_eight_dispatch_bound(limit, expected):
    node, calls = mission(limit=limit)
    for unused in range(expected):
        node._tick()
        reached(node, goals(calls)[-1][2])
        negatives(node)
    node._tick()
    assert len(goals(calls)) == expected and node.search_turns == expected
    assert calls[-1] == ('abort', 'shelf_markers_not_found')
    if expected == 0: assert node._marker_search_probe_origin is None


def test_negative_first_probe_returns_to_original_45_heading_without_resetting_budget():
    node, calls = mission()
    initial = node.robot_pose[2]
    node._tick()
    attained = wrap(goals(calls)[0][2]+.04)
    reached(node, attained)
    negatives(node)
    node._tick()
    second = goals(calls)[1][2]
    assert wrap(second-attained) > 0.  # intentional small reverse to the fallback
    assert second == pytest.approx(wrap(initial-math.pi/4))
    assert node.search_turns == 2 and node._marker_search_probe_origin == (original.TRIAL, initial)


def test_reentering_head_marker_state_does_not_reset_an_existing_search_budget():
    node, calls = mission()
    initial = node.robot_pose[2]
    node._tick()
    reached(node, goals(calls)[-1][2])
    node.state = 'HEAD_MARKERS'
    node._tick()
    assert node.state == 'SEARCH_COLUMN' and node.search_turns == 1
    assert node._marker_search_probe_origin == (original.TRIAL, initial)
    negatives(node)
    node._tick()
    assert len(goals(calls)) == 2 and node.search_turns == 2
    assert goals(calls)[1][2] == pytest.approx(wrap(initial-math.pi/4))


def test_missing_live_odometry_keeps_original_refusal_before_origin_capture():
    node, calls = mission()
    node.robot_pose = None
    node._tick()
    assert goals(calls) == [] and node.search_turns == 0
    assert node._marker_search_probe_origin is None
    assert calls[-1] == ('abort', 'shelf_markers_not_found')


@pytest.mark.parametrize('case', ['missing', 'false'])
def test_default_path_keeps_measured_relative_angle_and_original_configured_limit(case):
    node, calls = mission(limit=10)
    if case == 'missing': del node.marker_search_first_probe_enabled
    else: node.marker_search_first_probe_enabled = False
    node.search_turn = .71
    node.search_turns = 8
    node._tick()
    assert goals(calls) == [(1.2, -3.4, .37+.71, 'visual_search_turn')]
    assert node.search_turns == 9 and node._marker_search_probe_origin is None


@pytest.mark.parametrize('negative', [False, True])
def test_original_timeout_fallback_remains_when_negative_evidence_is_unavailable(negative):
    node, calls = mission(elapsed=5.1, negative=negative)
    node._marker_search_negative_gate.reset()
    node._tick()
    assert len(goals(calls)) == 1 and node.search_turns == 1
    assert goals(calls)[0][2] == pytest.approx(wrap(.37-math.pi/3))
    assert not any(c[0] == 'log' and c[1] == ('marker_search_early_turn',) for c in calls)


@pytest.mark.parametrize('case', ['absent', 'stale', 'wrong_epoch'])
def test_before_timeout_only_current_completed_negative_frames_allow_a_probe(case):
    node, calls = mission()
    node._marker_search_negative_gate.reset()
    if case != 'absent':
        for sequence in (1,2,3):
            data = original.event(sequence, epoch=2 if case == 'wrong_epoch' else 1)
            node._on_perception_status(NS(data=json.dumps(data)))
        if case == 'stale': node.now = 3_000_000_000
    node._tick()
    assert goals(calls) == [] and node.search_turns == 0
    assert node._marker_search_probe_origin is None and node.state == 'SEARCH_COLUMN'


@pytest.mark.parametrize('after_probe', [False, True])
def test_live_positive_marker_geometry_keeps_priority_over_the_optional_schedule(after_probe):
    node, calls = mission()
    if after_probe:
        node._tick()
        reached(node, goals(calls)[-1][2])
    before = node.search_turns
    cloud = NS(header=NS(stamp=original.stamp(node.now)))
    node._on_markers(cloud)
    node._shelf_geometry = lambda value: ('live-target', 'live-normal', 3) if value is cloud else pytest.fail('wrong live cloud')
    node._republish_score = lambda: None
    node._goal_from_target = lambda *args: (7.1, 8.2, -.43)
    node.shelf_standoff = 1.3
    node._marker_search_probe_yaw = lambda *a: pytest.fail('positive cloud entered scan schedule')
    node._tick()
    assert node.state == 'NAVIGATE_SHELF' and node.search_turns == before
    assert goals(calls)[-1] == (7.1, 8.2, -.43, 'shelf_observation_pose')
    assert node.target_marker_odom == 'live-target' and node.shelf_normal == 'live-normal'


def test_invalid_positive_geometry_is_unavailable_not_a_fresh_negative_result():
    node, calls = mission()
    node._on_markers(NS(header=NS(stamp=original.stamp(node.now))))
    def reject(cloud): raise ValueError('requested digit absent')
    node._shelf_geometry = reject
    node._tick()
    assert node.marker_cloud is None and goals(calls) == []
    assert node.state == 'SEARCH_COLUMN' and node.search_turns == 0


def test_marker_arrival_during_turn_does_not_cancel_or_admit_another_goal():
    node, calls = mission()
    node._tick()
    before = list(calls)
    node._on_markers(object())
    node._tick()
    assert calls == before and node.state == 'SEARCH_TURN' and node.search_turns == 1
    reached(node, goals(calls)[-1][2])
    assert node.marker_cloud is None
    assert node._marker_search_negative_gate.request.request_id == 2
    assert not any(c[0] == 'mode' and c[1] in ('cancel', 'stop') for c in calls)


@pytest.mark.parametrize('event', ['failed', 'rejected', 'cancelled'])
def test_original_navigation_failures_abort_without_retry_or_new_probe(event):
    node, calls = mission()
    node._tick()
    node.nav_event = {'event': event}
    node._tick()
    assert len(goals(calls)) == 1 and node.search_turns == 1
    assert calls[-1] == ('abort', 'visual_search_motion_failed')


def test_original_navigation_and_trial_timeouts_remain():
    node, calls = mission()
    node._tick()
    node._elapsed_state = lambda: 46.
    node._tick()
    assert calls[-1] == ('abort', 'visual_search_motion_failed') and len(goals(calls)) == 1
    node, calls = mission()
    node.trial_timeout = -1.
    node._tick()
    assert calls == [('abort', 'trial_timeout')] and node.search_turns == 0


@pytest.mark.parametrize('case', ['missing_origin', 'changed_trial', 'reset_count',
                                 'wrong_step', 'nonfinite_yaw', 'invalid_count'])
def test_bad_or_restarted_schedule_is_refused_before_dispatch(case):
    node, calls = mission()
    if case == 'missing_origin': node.search_turns = 1
    elif case == 'changed_trial':
        node._marker_search_probe_origin = ('another-trial', .37)
        node.search_turns = 1
    elif case == 'reset_count': node._marker_search_probe_origin = (node.trial_id, .37)
    elif case == 'wrong_step': node.search_turn = -math.pi/3
    elif case == 'nonfinite_yaw': node.robot_pose = (1.2, -3.4, float('nan'))
    else: node.search_turns = -.5
    before = node.search_turns
    node._tick()
    assert goals(calls) == [] and node.search_turns == before
    assert calls[-1] == ('abort', 'visual_search_schedule_invalid')


@pytest.mark.parametrize('flag', [None, 0, 1, 'true', []])
def test_actual_constructor_flag_validation_rejects_non_boolean(flag):
    tree = ast.parse((ROOT/'erc_phase1_solution/mission_manager.py').read_bytes())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'MissionManager')
    init = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '__init__')
    start = next(i for i,x in enumerate(init.body) if isinstance(x, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == 'marker_search_first_probe_enabled' for t in x.targets))
    fragment = ast.Module(body=copy.deepcopy(init.body[start:start+3]), type_ignores=[])
    assert isinstance(fragment.body[1], ast.If) and isinstance(fragment.body[2], ast.If)
    node = NS(get_parameter=lambda name: NS(value=flag), search_turn=-math.pi/4)
    with pytest.raises(ValueError, match='must be Boolean'):
        exec(compile(ast.fix_missing_locations(fragment), 'actual_constructor_fragment', 'exec'),
             {'self': node, 'math': math})


def test_declared_option_is_default_off_and_camera_navigation_methods_unchanged_by_schedule():
    tree = ast.parse((ROOT/'erc_phase1_solution/mission_manager.py').read_bytes())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'MissionManager')
    declare = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_declare_parameters')
    values = next(x.value for x in declare.body if isinstance(x, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'values' for t in x.targets))
    i = next(i for i,k in enumerate(values.keys) if isinstance(k, ast.Constant)
             and k.value == 'marker_search_first_probe_enabled')
    assert isinstance(values.values[i], ast.Constant) and values.values[i].value is False
    # Full historical source restoration exercises all six exact additions;
    # behavioral tests above always use the current optional schedule.
    from candidate_composition_support import restore_current_extensions
    restored = restore_current_extensions((ROOT/'erc_phase1_solution/mission_manager.py').read_bytes(), 'mission_manager.py')
    assert b'marker_search_first_probe_enabled' not in restored


def _actual_search_turn_constructor(value, *, enabled=True):
    tree=ast.parse((ROOT/'erc_phase1_solution/mission_manager.py').read_bytes())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MissionManager')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    at=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=='search_turn' for t in n.targets))
    statements=copy.deepcopy(init.body[at:at+4])
    assert all(isinstance(n,ast.Assign) for n in statements[:2])
    assert all(isinstance(n,ast.If) for n in statements[2:])
    params={'search_turn_radians':value,'marker_search_first_probe_enabled':enabled}
    node=NS(get_parameter=lambda name:NS(value=params[name]))
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),
                 'actual_search_turn_constructor','exec'),{'self':node,'math':math})
    return node.search_turn


def _existing_search_turn_value(source):
    if source=='installed_yaml':
        import yaml
        return yaml.safe_load((ROOT/'config/solution.yaml').read_bytes())['erc_mission_manager']['ros__parameters']['search_turn_radians']
    tree=ast.parse((ROOT/'erc_phase1_solution/mission_manager.py').read_bytes())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MissionManager')
    declare=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_declare_parameters')
    value=next(v for d in ast.walk(declare) if isinstance(d,ast.Dict) for k,v in zip(d.keys,d.values)
               if isinstance(k,ast.Constant) and k.value=='search_turn_radians')
    return eval(compile(ast.Expression(copy.deepcopy(value)),'actual_search_default','eval'),{'math':math})


@pytest.mark.parametrize('source',['declared_default','installed_yaml'])
def test_actual_supported_default_and_yaml_keep_the_exact_probe_and_fallback_headings(source):
    value=_existing_search_turn_value(source)
    assert value==(-math.pi/4.0 if source=='declared_default' else -0.7853981634)
    configured=_actual_search_turn_constructor(value)
    assert configured==value
    node,calls=mission();node.search_turn=configured;initial=node.robot_pose[2]
    for index in range(8):
        node._tick()
        assert len(goals(calls))==index+1
        expected=wrap(initial+(-math.pi/3.0 if index==0 else -index*math.pi/4.0))
        assert goals(calls)[-1][2]==expected
        reached(node,expected+.01)
        negatives(node)
    node._tick()
    assert len(goals(calls))==8 and calls[-1]==('abort','shelf_markers_not_found')
    assert node._marker_search_probe_origin==(node.trial_id,initial)


@pytest.mark.parametrize('value',[-math.pi/3.0,-0.7853981635,
    math.nextafter(-math.pi/4.0,math.inf),math.nextafter(-0.7853981634,-math.inf),
    0.,True,False,float('nan'),float('inf'),float('-inf')])
def test_other_nearby_nonfinite_and_boolean_turns_refuse_ctor_and_probe(value):
    with pytest.raises(ValueError,match='original -pi/4 fallback'):
        _actual_search_turn_constructor(value)
    node,calls=mission();node.search_turn=value
    node._tick()
    assert goals(calls)==[] and node.search_turns==0 and node._marker_search_probe_origin is None
    assert calls[-1]==('abort','visual_search_schedule_invalid')


@pytest.mark.parametrize('value',[.71,-math.pi/3.0])
def test_disabled_probe_keeps_existing_configured_relative_turn(value):
    configured=_actual_search_turn_constructor(value,enabled=False)
    node,calls=mission(limit=10);node.marker_search_first_probe_enabled=False
    node.search_turn=configured;initial=node.robot_pose[2]
    node._tick()
    assert len(goals(calls))==1 and goals(calls)[0][2]==initial+value
    assert node._marker_search_probe_origin is None
