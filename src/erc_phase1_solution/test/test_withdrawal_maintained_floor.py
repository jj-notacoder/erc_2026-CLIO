"""Maintained nominal rise, distinct from measured slip/physical clearance."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

from erc_phase1_solution import lift_first_extraction as lift
from test_lift_first_extraction import scenario, plan  # noqa: F401


# Binary-exact floor boundary avoids testing decimal/cancellation roundoff.
LIFT = .0048828125
TOLERANCE = .00048828125
FLOOR = .00390625
EPSILON = 2. ** -40


@pytest.fixture
def prepared(scenario):
    node, front, grasp, goals, *_ = scenario
    front[2] = grasp[3] = 0.
    for q in goals:
        q[3] = 0.
    node.pick_position_tolerance = TOLERANCE
    result = plan(scenario, lift_m=LIFT)
    return scenario, result


def invoke(prepared, mode, module=lift, *, measured=None, bay=None):
    scene, original = prepared
    node, front, grasp, goals, original_bay, *_ = scene
    bay = original_bay if bay is None else bay
    if mode == 'planned':
        return module.plan_lift_first_extraction(node, front, grasp, goals,
            bay=bay, aperture=.017, lift_m=LIFT).metrics
    return module.validate_lift_first_route(node, front,
        grasp if measured is None else measured, original.route,
        bay=bay, aperture=.017, lift_m=LIFT,
        attached_corners=original.attached_corners)


def interior_height(prepared, value, *, second=False):
    chain = prepared[0][0].chain
    forward = chain.forward
    lower, upper = ((.63, .65) if second else (.67, .69))

    def changed(q):
        pose = forward(q)
        if lower < q[1] < upper:
            pose[2, 3] = value
        return pose

    chain.forward = changed


def restored_parent_source():
    path = Path(lift.__file__)
    record = json.loads((Path(__file__).parent / 'fixtures/withdrawal_floor_inverse.json').read_bytes())
    current = path.read_bytes()
    from candidate_composition_support import restore_withdrawal_lift_ceiling_bytes
    current = restore_withdrawal_lift_ceiling_bytes(current, 'helper')
    assert hashlib.sha256(current).hexdigest() == record['candidate_sha256']
    source = current.decode()
    for change in reversed(record['changes']):
        assert source.count(change['new']) == change['count']
        source = source.replace(change['new'], change['old'])
    assert hashlib.sha256(source.encode()).hexdigest() == record['parent_sha256']
    return source


@pytest.fixture
def parent_module(tmp_path):
    path = tmp_path / 'lift_parent.py'
    path.write_bytes(restored_parent_source().encode())
    name = 'erc_phase1_solution._withdrawal_floor_parent_reference'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[name]


def parent_bay(parent, current):
    return parent.RelativeShelfBay(**vars(current))


@pytest.mark.parametrize('mode', ['planned', 'measured'])
@pytest.mark.parametrize('difference', [-EPSILON, 0., EPSILON])
def test_each_loop_checks_exact_positive_floor_boundary(prepared, mode, difference):
    interior_height(prepared, FLOOR + difference)
    if difference < 0:
        with pytest.raises(ValueError, match='maintained floor rise at leg 1: observed_m=.*required_m=0.00390625'):
            invoke(prepared, mode)
    else:
        result = invoke(prepared, mode)
        leg = result['legs'][1]
        assert leg['minimum_corner_rise_m'] == FLOOR + difference
        assert leg['required_minimum_corner_rise_m'] == FLOOR
        assert leg['corner_rise_reserve_m'] == difference


@pytest.mark.parametrize('mode', ['planned', 'measured'])
@pytest.mark.parametrize('second', [False, True])
def test_old_positive_but_unmaintained_withdrawal_is_now_rejected(prepared, parent_module, monkeypatch, mode, second):
    interior_height(prepared, FLOOR / 2, second=second)
    monkeypatch.setattr(parent_module, 'check_cradle_tool_sweep', lift.check_cradle_tool_sweep)
    old = invoke(prepared, mode, parent_module,
                 bay=parent_bay(parent_module, prepared[0][4]))
    assert old['legs'][2 if second else 1]['minimum_corner_rise_m'] == FLOOR / 2
    with pytest.raises(ValueError, match=f'maintained floor rise at leg {2 if second else 1}'):
        invoke(prepared, mode)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_initial_lift_still_starts_at_zero_and_reports_separate_requirement(prepared, mode):
    result = invoke(prepared, mode)
    assert result['minimum_withdrawal_corner_rise_m'] == FLOOR
    assert result['maintained_floor_reference'] == 'initial_upright_book_corners'
    first = result['legs'][0]
    assert first['minimum_corner_rise_m'] == 0.
    assert first['required_minimum_corner_rise_m'] == 0.
    for leg in result['legs'][1:]:
        assert leg['required_minimum_corner_rise_m'] == FLOOR
        assert leg['minimum_corner_rise_m'] == LIFT
        assert leg['corner_rise_reserve_m'] == LIFT - FLOOR
    assert result['full_shelf_collision_certificate'] is False
    assert result['modeled_tool_allowance_is_certified_deflection_bound'] is False


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_all_corners_are_checked_when_book_center_remains_raised(prepared, mode):
    chain = prepared[0][0].chain
    forward = chain.forward

    def tilt(q):
        pose = forward(q)
        if .67 < q[1] < .69:
            angle = .012
            assert LIFT - .055 * np.sin(angle) > FLOOR
            pose[:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
                           [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
        return pose

    chain.forward = tilt
    with pytest.raises(ValueError, match='maintained floor rise'):
        invoke(prepared, mode)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
@pytest.mark.parametrize('edge', ['roof', 'side'])
def test_existing_uncertain_roof_and_side_rejections_remain(prepared, mode, edge):
    bay = prepared[0][4]
    if edge == 'roof':
        bay = replace(bay, roof_uncertainty_m=.040)
    else:
        bay = replace(bay, marker_center_base=[.61, .48, 2.26])
    with pytest.raises(ValueError, match='clearance'):
        invoke(prepared, mode, bay=bay)


@pytest.mark.parametrize('mode', ['planned', 'measured'])
@pytest.mark.parametrize('which', ['robot', 'tool'])
def test_original_collision_rejections_still_precede_bay_admission(prepared, monkeypatch, mode, which):
    if which == 'robot':
        prepared[0][0]._carried_robot_transition_is_safe = lambda *a, **k: False
    else:
        monkeypatch.setattr(lift, 'check_cradle_tool_sweep', lambda *a, **k: 'retained_tool_rejection')
    with pytest.raises(ValueError, match='sweep rejected leg 0'):
        invoke(prepared, mode)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.0001, 0., .000501])
def test_measured_floor_threshold_cannot_be_invalid(prepared, value):
    node, *_, calls, _ = prepared[0]
    node.pick_position_tolerance = value
    calls.clear()
    with pytest.raises(ValueError, match='finite positive floor reserve'):
        invoke(prepared, 'measured')
    assert calls == []


def test_measured_floor_is_relative_to_fresh_start_and_never_replans(prepared):
    scene, result = prepared
    node, _, grasp, *_ = scene
    before = [q.copy() for q in result.route]
    count = len(node.chain.targets)
    measured = grasp.copy()
    measured[3] += EPSILON
    checked = invoke(prepared, 'measured', measured=measured)
    assert checked['legs'][1]['minimum_corner_rise_m'] == LIFT - EPSILON
    assert len(node.chain.targets) == count
    assert all(np.array_equal(a, b) for a, b in zip(before, result.route))


def test_measured_start_can_consume_floor_reserve_while_first_endpoint_still_passes(prepared):
    measured = prepared[0][2].copy()
    measured[3] = LIFT - FLOOR - EPSILON
    interior_height(prepared, LIFT - 2 * EPSILON)
    with pytest.raises(ValueError, match='maintained floor rise at leg 1'):
        invoke(prepared, 'measured', measured=measured)


def test_cancellation_still_wins_before_any_measured_pass(prepared):
    node, *_, calls, _ = prepared[0]
    node._cancel = threading.Event()
    node._cancel.set()
    calls.clear()
    with pytest.raises(ValueError, match='geometry check cancelled'):
        invoke(prepared, 'measured')
    assert calls == []


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_valid_route_preserves_original_samples_checks_and_clearance_values(prepared, parent_module, monkeypatch, mode):
    node, _, _, _, bay, calls, _ = prepared[0]
    monkeypatch.setattr(parent_module, 'check_cradle_tool_sweep', lift.check_cradle_tool_sweep)
    calls.clear()
    old = invoke(prepared, mode, parent_module, bay=parent_bay(parent_module, bay))
    old_calls = [(c[0], c[1].copy(), c[2].copy()) for c in calls]
    calls.clear()
    new = invoke(prepared, mode)
    assert [c[0] for c in calls] == [c[0] for c in old_calls]
    for a, b in zip(calls, old_calls):
        assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    for a, b in zip(old['legs'], new['legs']):
        for field in ('samples', 'minimum_corner_rise_m',
                      'roof_clearance_after_margin_uncertainty_m',
                      'side_clearance_after_margin_uncertainty_m'):
            assert a[field] == b[field]


def test_whole_runtime_inverse_is_exact_and_dispatch_module_is_untouched():
    restored_parent_source()
    package = Path(lift.__file__).parent
    from candidate_composition_support import restore_withdrawal_lift_ceiling_bytes
    original = restore_withdrawal_lift_ceiling_bytes((package / 'manipulation_node.py').read_bytes(), 'node')
    assert hashlib.sha256(original).hexdigest() == '4754b83123b43cb183856d279ce5dc935eb291642c5dcb4f27165f1707a043f6'


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_eight_mm_retains_the_same_floor_formula(prepared, mode):
    node, front, grasp, goals, bay, *_ = prepared[0]
    result = lift.plan_lift_first_extraction(node, front, grasp, goals,
        bay=bay, aperture=.017, lift_m=.008)
    metrics = (result.metrics if mode == 'planned' else
        lift.validate_lift_first_route(node, front, grasp, result.route,
            bay=bay, aperture=.017, lift_m=.008))
    assert metrics['minimum_withdrawal_corner_rise_m'] == .008 - 2*TOLERANCE
    assert all(leg['required_minimum_corner_rise_m'] == .008 - 2*TOLERANCE
               for leg in metrics['legs'][1:])


@pytest.mark.parametrize('mode', ['planned', 'measured'])
def test_default_five_mm_lift_requires_four_mm_without_timing_changes(scenario, mode):
    node, front, grasp, _, bay, *_ = scenario
    original = plan(scenario)
    result = (original.metrics if mode == 'planned' else
        lift.validate_lift_first_route(node, front, grasp, original.route,
            bay=bay, aperture=.017))
    assert result['minimum_withdrawal_corner_rise_m'] == .004
    assert all(leg['required_minimum_corner_rise_m'] == .004
               for leg in result['legs'][1:])


@pytest.mark.parametrize('value', [TOLERANCE, 2*TOLERANCE])
def test_measured_lift_must_exceed_existing_position_error_allowance(prepared, value):
    node, front, grasp, _, bay, *_ = prepared[0]
    with pytest.raises(ValueError, match='finite positive floor reserve'):
        lift.validate_lift_first_route(node, front, grasp, prepared[1].route,
            bay=bay, aperture=.017, lift_m=value)
