"""20 mm software admission only; current geometry and physical proof stay separate."""
import ast
from dataclasses import replace
import hashlib
from installed_profile_test_support import restore_installed_profile_bytes
import importlib.util
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

from erc_phase1_solution import lift_first_extraction as lift
from candidate_composition_support import restore_withdrawal_twenty_mm_ceiling_bytes, restore_supported_compact_floor_bytes
from test_lift_first_extraction import scenario, plan  # noqa: F401
from test_withdrawal_eight_mm_cap import call_with_lift
from test_withdrawal_maintained_floor import prepared, interior_height  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['constructor', 'planned', 'measured'])
@pytest.mark.parametrize('value, admitted', [
    (np.nextafter(.008, float('inf')), True), (.012, True),
    (np.nextafter(.020, 0.), True), (.020, True),
    (np.nextafter(.020, float('inf')), False), (.021, False),
])
def test_three_real_admissions_include_twenty_but_reject_above_it(prepared, mode, value, admitted):
    route = call_with_lift(prepared, 'planned', value).route if admitted and mode == 'measured' else None
    if admitted:
        call_with_lift(prepared, mode, value, route=route)
    else:
        with pytest.raises(ValueError):
            call_with_lift(prepared, mode, value)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_twenty_requires_nineteen_mm_without_relaxing_samples_or_input_ownership(scenario, mode):
    node, front, grasp, goals, bay, calls, _ = scenario
    node.pick_position_tolerance = .0005
    old_inputs = [q.copy() for q in (front, grasp, *goals)]
    result = lift.plan_lift_first_extraction(node, front, grasp, goals,
        bay=bay, aperture=.017, lift_m=.020)
    solved = len(node.chain.targets)
    calls.clear()
    metrics = result.metrics if mode == 'planned' else lift.validate_lift_first_route(
        node, front, grasp, result.route, bay=bay, aperture=.017, lift_m=.020)
    assert metrics['minimum_withdrawal_corner_rise_m'] == pytest.approx(.019)
    assert metrics['legs'][0]['required_minimum_corner_rise_m'] == 0.
    assert metrics['legs'][0]['minimum_corner_rise_m'] == pytest.approx(0.)
    assert all(row['required_minimum_corner_rise_m'] == pytest.approx(.019) for row in metrics['legs'][1:])
    assert all(row['minimum_corner_rise_m'] == pytest.approx(.020) for row in metrics['legs'][1:])
    assert all(row['samples'] >= 61 for row in metrics['legs'])
    assert all(np.array_equal(a,b) for a,b in zip(old_inputs,(front,grasp,*goals)))
    assert metrics['full_shelf_collision_certificate'] is False
    if mode == 'measured':
        assert len(node.chain.targets) == solved
        assert [row[0] for row in calls] == ['robot','tool']*len(result.route)
        assert metrics['route_replanned'] is False


@pytest.mark.parametrize('mode', ['planned', 'measured'])
@pytest.mark.parametrize('delta', [-2.**-40, 0., 2.**-40])
def test_higher_route_retains_exact_interior_floor_boundary(prepared, mode, delta):
    # Exact binary values isolate the predicate from decimal subtraction noise.
    scene, _ = prepared
    node, front, grasp, goals, bay, *_ = scene
    rise, tolerance = .01953125, .00048828125
    node.pick_position_tolerance = tolerance
    floor = rise - 2*tolerance
    route = call_with_lift(prepared, 'planned', rise).route
    interior_height(prepared, floor+delta)
    if delta < 0:
        with pytest.raises(ValueError, match='maintained floor rise at leg 1'):
            call_with_lift(prepared, mode, rise, route=route)
    else:
        checked = call_with_lift(prepared, mode, rise, route=route)
        metrics = checked.metrics if mode == 'planned' else checked
        assert metrics['legs'][1]['minimum_corner_rise_m'] == floor+delta
        assert metrics['legs'][1]['required_minimum_corner_rise_m'] == floor


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_roof_can_admit_eight_and_refuse_twenty_with_same_fifteen_mm_margin(prepared, mode):
    bay = replace(prepared[0][4], roof_uncertainty_m=.025)
    eight = call_with_lift(prepared, 'planned', .008, bay=bay)
    call_with_lift(prepared, 'measured', .008, route=eight.route, bay=bay)
    twenty = call_with_lift(prepared, 'planned', .020)
    assert bay.margin_m == .015
    with pytest.raises(ValueError, match='clearance'):
        call_with_lift(prepared, mode, .020, route=twenty.route, bay=bay)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_twenty_keeps_side_uncertainty_and_minimum_margin_admission(prepared, mode):
    route = call_with_lift(prepared, 'planned', .020).route
    bay = replace(prepared[0][4], marker_center_base=[.61,.48,2.26])
    with pytest.raises(ValueError, match='clearance'):
        call_with_lift(prepared, mode, .020, route=route, bay=bay)
    bay = replace(prepared[0][4], margin_m=np.nextafter(.015,0.))
    with pytest.raises(ValueError, match='at least 15 mm'):
        call_with_lift(prepared, mode, .020, route=route, bay=bay)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
@pytest.mark.parametrize('veto', ['robot', 'tool', 'cancel'])
def test_twenty_still_obeys_actual_collision_and_cancellation_vetoes(prepared, monkeypatch, mode, veto):
    node, *_, calls, _ = prepared[0]
    route = call_with_lift(prepared, 'planned', .020).route
    calls.clear()
    if veto == 'robot':
        node._carried_robot_transition_is_safe = lambda *a,**kw: False
    elif veto == 'tool':
        monkeypatch.setattr(lift,'check_cradle_tool_sweep',lambda *a,**kw:'retained_tool_veto')
    else:
        node._cancel = threading.Event()
        node._cancel.set()
    if veto == 'cancel' and mode == 'planned':
        # Planning delegates cancellation to its sweeps. Restore the actual
        # tool checker after the original scenario's success-only test double;
        # the measured validator independently checks cancellation at entry.
        from erc_phase1_solution.shelf_cradle_geometry import check_cradle_tool_sweep
        monkeypatch.setattr(lift, 'check_cradle_tool_sweep', check_cradle_tool_sweep)
        expected = 'gripper sweep rejected leg 0: cradle_tool_cancelled'
    else:
        expected = 'cancelled' if veto == 'cancel' else 'sweep rejected leg 0'
    with pytest.raises(ValueError, match=expected):
        call_with_lift(prepared, mode, .020, route=route)
    if veto == 'cancel':
        if mode == 'planned':
            assert [row[0] for row in calls] == ['robot']
        else:
            assert calls == []


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_twenty_does_not_expand_separate_tool_deflection_allowance(prepared, mode):
    node, front, grasp, goals, bay, *_ = prepared[0]
    route = call_with_lift(prepared, 'planned', .020).route
    with pytest.raises(ValueError, match='modeled tool allowance'):
        if mode == 'planned':
            lift.plan_lift_first_extraction(node,front,grasp,goals,
                bay=bay,aperture=.017,lift_m=.020,modeled_tool_allowance_m=.006)
        else:
            lift.validate_lift_first_route(node,front,grasp,route,
                bay=bay,aperture=.017,lift_m=.020,modeled_tool_allowance_m=.006)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_exact_eight_mm_parent_is_a_rejecting_positive_control(prepared, monkeypatch, tmp_path, mode):
    current = (ROOT/'erc_phase1_solution/lift_first_extraction.py').read_bytes()
    parent_bytes = restore_withdrawal_twenty_mm_ceiling_bytes(current, 'helper')
    path = tmp_path/'old_eight_mm_helper.py'
    path.write_bytes(parent_bytes)
    name = 'erc_phase1_solution._twenty_mm_parent_control'
    spec = importlib.util.spec_from_file_location(name,path)
    parent = importlib.util.module_from_spec(spec)
    sys.modules[name] = parent
    try:
        spec.loader.exec_module(parent)
        monkeypatch.setattr(parent,'check_cradle_tool_sweep',lift.check_cradle_tool_sweep)
        node, front, grasp, goals, bay, *_ = prepared[0]
        route = call_with_lift(prepared,'planned',.020).route
        call_with_lift(prepared,mode,.020,route=route)
        old_bay = parent.RelativeShelfBay(**vars(bay))
        with pytest.raises(ValueError):
            if mode == 'planned':
                parent.plan_lift_first_extraction(node,front,grasp,goals,bay=old_bay,aperture=.017,lift_m=.020)
            else:
                parent.validate_lift_first_route(node,front,grasp,route,bay=old_bay,aperture=.017,lift_m=.020)
    finally:
        del sys.modules[name]


def test_complete_node_inverse_preserves_pressure_retention_timing_and_defaults():
    node = (ROOT/'erc_phase1_solution/manipulation_node.py').read_bytes()
    parent = restore_withdrawal_twenty_mm_ceiling_bytes(node,'node')
    assert hashlib.sha256(parent).hexdigest() == '9fd743c3b645e13893fd9f6205e50580ee038b684ebf1b3764b352631461f750'
    current_ast, parent_ast = ast.parse(restore_supported_compact_floor_bytes(node)), ast.parse(parent)
    def methods(tree):
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ManipulationNode')
        return {n.name:ast.dump(n,include_attributes=False) for n in owner.body if isinstance(n,ast.FunctionDef)}
    a,b=methods(current_ast),methods(parent_ast)
    assert set(a)==set(b)
    assert all(a[name]==b[name] for name in a if name!='__init__')
    assert b"'lift_first_extraction_lift_m': .005" in node
    assert b"'withdrawal_speed_scale': 1.0" in node
    assert b'1.0 if lift_enabled and index == 0 else 5.8' in node
    assert hashlib.sha256(restore_installed_profile_bytes((ROOT/'config/collision_quality.yaml').read_bytes())).hexdigest() == '98ef332abbec93aa4af1d9874cf01c2308a8dc8575a3b22ab790843ebebe4bea'


@pytest.mark.parametrize('kind, path', [('helper','lift_first_extraction.py'),('node','manipulation_node.py')])
def test_twenty_inverse_rejects_otherwise_plausible_unrelated_source_edit(kind,path):
    source=(ROOT/'erc_phase1_solution'/path).read_bytes()
    with pytest.raises(AssertionError):
        restore_withdrawal_twenty_mm_ceiling_bytes(source+b'\n# undeclared\n',kind)
