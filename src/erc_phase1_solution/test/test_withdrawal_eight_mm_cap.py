"""8 mm is a software range expansion; no physical clearance claim."""
import ast
import hashlib
from installed_profile_test_support import restore_installed_profile_bytes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from erc_phase1_solution import lift_first_extraction as lift
from candidate_composition_support import (
    restore_to_r51_source, restore_withdrawal_lift_ceiling_bytes,
)
from test_lift_first_extraction import scenario, plan  # noqa: F401
from test_withdrawal_maintained_floor import prepared, interior_height  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def constructor_guard(value):
    """Execute the unique actual constructor range guard without constructing ROS."""
    root = ast.parse((ROOT / 'erc_phase1_solution/manipulation_node.py').read_bytes())
    owner = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == 'ManipulationNode')
    init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    matches = [n for n in ast.walk(init) if isinstance(n, ast.If)
               and any(isinstance(c, ast.Constant) and isinstance(c.value, str)
                       and c.value.startswith('lift_first_extraction_lift_m must be')
                       for c in ast.walk(n))]
    assert len(matches) == 1
    block = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    exec(compile(block, '<actual-lift-constructor-admission>', 'exec'),
         {'np': np, 'self': SimpleNamespace(lift_first_extraction_lift_m=value)})


def call_with_lift(prepared, mode, value, *, route=None, bay=None):
    scene, original = prepared
    node, front, grasp, goals, original_bay, *_ = scene
    bay = original_bay if bay is None else bay
    if mode == 'constructor':
        return constructor_guard(value)
    if mode == 'planned':
        return lift.plan_lift_first_extraction(node, front, grasp, goals,
            bay=bay, aperture=.017, lift_m=value)
    return lift.validate_lift_first_route(node, front, grasp,
        original.route if route is None else route,
        bay=bay, aperture=.017, lift_m=value)


@pytest.mark.parametrize('mode', ['constructor', 'planned', 'measured'])
@pytest.mark.parametrize('value, accepted', [
    (float('nan'), False), (float('inf'), False), (-float('inf'), False),
    (-.001, False), (0., False), (.005, True), (.006, True),
    (np.nextafter(.008, 0.), True), (.008, True),
    (np.nextafter(.020, float('inf')), False), (.021, False),
])
def test_three_actual_software_admissions_preserve_finite_inclusive_cap(prepared, mode, value, accepted):
    route = None
    if accepted and mode == 'measured':
        route = call_with_lift(prepared, 'planned', value).route
    if accepted:
        call_with_lift(prepared, mode, value, route=route)
    else:
        with pytest.raises(ValueError):
            call_with_lift(prepared, mode, value)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_eight_mm_uses_seven_mm_floor_at_unchanged_default_tolerance(scenario, mode):
    node, front, grasp, goals, bay, *_ = scenario
    before = [q.copy() for q in (grasp, *goals)]
    result = lift.plan_lift_first_extraction(node, front, grasp, goals,
        bay=bay, aperture=.017, lift_m=.008)
    metrics = result.metrics if mode == 'planned' else lift.validate_lift_first_route(
        node, front, grasp, result.route, bay=bay, aperture=.017, lift_m=.008)
    assert metrics['minimum_withdrawal_corner_rise_m'] == .007
    assert metrics['legs'][0]['minimum_corner_rise_m'] == pytest.approx(0.)
    assert all(v['required_minimum_corner_rise_m'] == .007 for v in metrics['legs'][1:])
    assert all(v['minimum_corner_rise_m'] == pytest.approx(.008) for v in metrics['legs'][1:])
    assert all(np.array_equal(a, b) for a, b in zip(before, (grasp, *goals)))
    assert metrics['full_shelf_collision_certificate'] is False


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_positive_six_and_half_mm_sag_is_rejected_for_eight_mm_route(prepared, mode):
    result = call_with_lift(prepared, 'planned', .008)
    interior_height(prepared, .0065)
    with pytest.raises(ValueError, match='maintained floor rise at leg 1'):
        call_with_lift(prepared, mode, .008, route=result.route)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_roof_that_admits_five_mm_can_reject_eight_mm(prepared, mode):
    from dataclasses import replace
    bay = replace(prepared[0][4], roof_uncertainty_m=.028)
    five = call_with_lift(prepared, 'planned', .005, bay=bay)
    call_with_lift(prepared, 'measured', .005, route=five.route, bay=bay)
    eight = call_with_lift(prepared, 'planned', .008)
    with pytest.raises(ValueError, match='clearance'):
        call_with_lift(prepared, mode, .008, route=eight.route, bay=bay)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_eight_mm_does_not_bypass_uncertain_side_wall(prepared, mode):
    from dataclasses import replace
    eight = call_with_lift(prepared, 'planned', .008)
    bay = replace(prepared[0][4], marker_center_base=[.61, .48, 2.26])
    with pytest.raises(ValueError, match='clearance'):
        call_with_lift(prepared, mode, .008, route=eight.route, bay=bay)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_separate_five_mm_modeled_tool_allowance_did_not_expand(prepared, mode):
    scene, _ = prepared
    node, front, grasp, goals, bay, *_ = scene
    route = call_with_lift(prepared, 'planned', .008).route
    with pytest.raises(ValueError, match='modeled tool allowance'):
        if mode == 'planned':
            lift.plan_lift_first_extraction(node, front, grasp, goals,
                bay=bay, aperture=.017, lift_m=.008, modeled_tool_allowance_m=.006)
        else:
            lift.validate_lift_first_route(node, front, grasp, route,
                bay=bay, aperture=.017, lift_m=.008, modeled_tool_allowance_m=.006)


@pytest.mark.parametrize('kind, relative', [
    ('node', 'erc_phase1_solution/manipulation_node.py'),
    ('helper', 'erc_phase1_solution/lift_first_extraction.py'),
])
@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'unrelated'])
def test_cap_inverse_refuses_undeclared_source(kind, relative, damage):
    data = (ROOT / relative).read_bytes()
    record = json.loads((ROOT / 'test/fixtures/withdrawal_twenty_mm_cap_inverse.json').read_bytes())[kind]
    item = record['replacements'][0]
    if damage == 'missing':
        data = data.replace(item['new'].encode(), item['old'].encode(), 1)
    elif damage == 'duplicate':
        data += item['new'].encode()
    else:
        data += b'\n# unrelated source edit\n'
    with pytest.raises(AssertionError):
        restore_withdrawal_lift_ceiling_bytes(data, kind)


def test_old_complete_node_structural_chain_still_reaches_exact_r51():
    source = (ROOT / 'erc_phase1_solution/manipulation_node.py').read_text()
    assert restore_to_r51_source(source)


def test_default_selected_profile_and_all_noncap_dispatch_bytes_stay_r57():
    data = (ROOT / 'erc_phase1_solution/manipulation_node.py').read_bytes()
    original = restore_withdrawal_lift_ceiling_bytes(data, 'node')
    assert hashlib.sha256(original).hexdigest() == '4754b83123b43cb183856d279ce5dc935eb291642c5dcb4f27165f1707a043f6'
    assert b"'lift_first_extraction_lift_m': .005" in data
    assert b'1.0 if lift_enabled and index == 0 else 5.8' in data
    assert hashlib.sha256(restore_installed_profile_bytes((ROOT / 'config/collision_quality.yaml').read_bytes())).hexdigest() == '98ef332abbec93aa4af1d9874cf01c2308a8dc8575a3b22ab790843ebebe4bea'
