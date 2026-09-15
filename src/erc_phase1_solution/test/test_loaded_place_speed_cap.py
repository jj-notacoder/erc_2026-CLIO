"""Exercise actual selected constructor and loaded PLACE dispatch cap statements."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from erc_phase1_solution.arm_trajectory_timing import checked_arm_speed_scale

SOURCE = Path(__file__).resolve().parents[1]/'erc_phase1_solution/manipulation_node.py'


def method(name):
    owner = next(n for n in ast.parse(SOURCE.read_bytes()).body
                 if isinstance(n, ast.ClassDef) and n.name == 'ManipulationNode')
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def assignment_to(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


def dispatch(options, cap=None):
    body = method('_place').body
    start = next(i for i,n in enumerate(body) if assignment_to(n, 'loaded_arm_timing_options'))
    end = next(i for i,n in enumerate(body[start:], start) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == '_execute_retained_arm_legs')
    received = []; legs = [object()]
    owner = NS(_execute_retained_arm_legs=lambda *args, **kwargs:
               (received.append((args, kwargs)) or (True, 1, False)))
    if cap is not None: owner.loaded_place_speed_scale_cap = cap
    scope = dict(self=owner, arm_timing_options=options, approach_legs=legs,
                 checked_arm_speed_scale=checked_arm_speed_scale)
    initialization = body[0]
    assert isinstance(initialization, ast.Assign)
    assert [target.id for target in initialization.targets] == ['release_only_request', 'release_only_endpoint']
    assert isinstance(initialization.value, ast.Constant) and initialization.value.value is None
    block = ast.Module(body=copy.deepcopy([initialization,*body[start:end+1]]), type_ignores=[])
    exec(compile(ast.fix_missing_locations(block), '<actual loaded PLACE dispatch>', 'exec'), scope)
    assert scope['approach_ok'] and scope['next_leg'] == 1 and not scope['contact_lost']
    assert received[0][0] == (legs, 'place')
    return received[0][1], scope['loaded_arm_timing_options']


@pytest.mark.parametrize('global_scale,cap,expected', [
    (3., None, 3.), (3., 3., 3.), (3., 1.25, 1.25), (2., 1.25, 1.25),
    (1.1, 1.25, 1.1), (1., 1.25, 1.), (1.25, 3., 1.25), (2., 1., 1.),
])
def test_actual_loaded_dispatch_caps_without_increasing_or_mutating(global_scale, cap, expected):
    sentinel = object(); original = dict(arm_speed_scale=global_scale, marker=sentinel)
    received, copied = dispatch(original, cap)
    assert received == dict(arm_speed_scale=expected, marker=sentinel)
    assert original == dict(arm_speed_scale=global_scale, marker=sentinel)
    assert copied is not original and copied == received


def test_unselected_normal_path_stays_without_timing_override():
    original = {}; received, copied = dispatch(original, 1.25)
    assert original == received == copied == {} and copied is not original


@pytest.mark.parametrize('value,accepted', [(1., True), (1.25, True), (3., True),
                                         (.99, False), (3.01, False), (float('nan'), False),
                                         (float('inf'), False)])
def test_actual_constructor_cap_uses_existing_bounded_validator(value, accepted):
    statement = next(n for n in ast.walk(method('__init__')) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Attribute) and t.attr == 'loaded_place_speed_scale_cap'
                             for t in n.targets))
    assert isinstance(statement.value, ast.Call) and statement.value.func.id == 'checked_arm_speed_scale'
    requested = []
    owner = NS(get_parameter=lambda name: (requested.append(name) or NS(value=value)))
    scope = dict(self=owner, checked_arm_speed_scale=checked_arm_speed_scale)
    block = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(statement)], type_ignores=[])),
                    '<actual loaded PLACE cap constructor>', 'exec')
    if accepted:
        exec(block, scope); assert owner.loaded_place_speed_scale_cap == value
    else:
        with pytest.raises(ValueError): exec(block, scope)
    assert requested == ['loaded_place_speed_scale_cap']


def test_default_cap_is_three_and_only_normal_loaded_dispatch_uses_its_copy():
    declaration = method('_declare_parameters')
    defaults = [v for d in ast.walk(declaration) if isinstance(d, ast.Dict)
                for k,v in zip(d.keys,d.values) if isinstance(k, ast.Constant)
                and k.value == 'loaded_place_speed_scale_cap']
    assert len(defaults) == 1 and ast.literal_eval(defaults[0]) == 3.
    calls = [n for n in ast.walk(method('_place')) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)]
    copied = [n.func.attr for n in calls if any(k.arg is None and isinstance(k.value, ast.Name)
              and k.value.id == 'loaded_arm_timing_options' for k in n.keywords)]
    original = [n.func.attr for n in calls if any(k.arg is None and isinstance(k.value, ast.Name)
                and k.value.id == 'arm_timing_options' for k in n.keywords)]
    assert copied == ['_execute_retained_arm_legs']
    assert sorted(original) == ['_return_from_bin', '_return_from_bin']
    for n in calls:
        if n.func.attr == '_recover_closed_place':
            assert all(not (isinstance(k.value, ast.Name) and k.value.id == 'loaded_arm_timing_options')
                       for k in n.keywords)


def test_compact_and_unloaded_methods_do_not_reference_the_loaded_cap():
    for name in ('_execute_cached_post_retreat_compaction', '_execute_unloaded_home',
                 '_return_from_bin', '_recover_closed_place'):
        assert all(not (isinstance(n, ast.Attribute) and n.attr == 'loaded_place_speed_scale_cap')
                   for n in ast.walk(method(name)))
