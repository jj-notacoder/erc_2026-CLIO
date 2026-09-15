"""Lower-shelf selection protocol and complete official-mesh geometry regressions."""
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest


from erc_phase1_solution import lower_shelf_pick as lower_pick


FRONT = np.array([.670, -.056, 1.254])


@pytest.fixture
def protocol(open_tool, monkeypatch):
    """Records ownership/control order; it makes no physical geometry claim."""
    calls = []
    shelf_failure = {}
    class Bounds:
        def __init__(self, *args, **kwargs):
            self.samples = 0
            self.plane_point = np.array([.605, 0., 0.])
            self.normal_uncertainty_m = .010
            self.minimum_floor = self.minimum_roof = self.minimum_side = self.minimum_back = .04
        def edge(self, first, last, *, allow_entry):
            self.samples += 1
            return shelf_failure.get('entry' if allow_entry else 'setup')
    monkeypatch.setattr(lower_pick, 'EmptyShelfBounds', Bounds)
    node = NS(pick_torso_height=.35, pick_position_tolerance=.0005,
        pick_orientation_tolerance=.01, pregrasp_offset=.14, cartesian_step=.06,
        chain=NS(lower=np.array([-.001, *([-3.]*7)]),
                 upper=np.array([.35, *([3.]*7)])),
        _cancel=threading.Event(), _cached_post_retreat_plan={'old': True},
        _publish_status=Mock(), _move_arm=Mock(side_effect=AssertionError('motion')),
        _move_torso=Mock(side_effect=AssertionError('motion')))
    start = np.array([.10, *([0.]*7)])
    guard = NS(start=start, last_rejection=None)
    def record(label, result=True):
        def callback(*args, **kwargs):
            calls.append((label, node.pick_torso_height))
            return result
        return callback
    guard.opening = record('opening')
    guard.edge = record('torso_edge')
    guard.candidate = record('empty_candidate')
    guard.retracted_edge = record('retracted_edge')
    guard.plan_transition = record('setup_transition', [])
    node._interpolate_positions = lambda a, b, _: list(np.linspace(a, b, 3)[1:])
    def solve(positions, rotations, torso, **kwargs):
        calls.append(('solver', torso))
        assert kwargs['first_valid']
        assert kwargs['endpoint_first'] is getattr(node, 'expected_endpoint_first', True)
        assert callable(kwargs['transition_edge_validator'])
        assert callable(kwargs['setup_transition_planner'])
        assert kwargs['position_tolerance'] == .0005
        assert kwargs['orientation_tolerance'] == .01
        solutions = [np.array([torso, i*.05, *([0.]*6)]) for i in range(len(positions))]
        if not kwargs['candidate_validator'](solutions, []):
            raise RuntimeError('no admitted Cartesian path')
        return solutions, 0, 1.25, []
    node._solve_cartesian_path = solve
    def lift(grasp, extraction):
        calls.append(('lift', node.pick_torso_height))
        return NS(terminal=extraction[-1].copy(), route=tuple(extraction))
    def carry(front, grasp, terminal, rotation, torso, **kwargs):
        calls.append(('carry', torso, kwargs))
        node._cached_post_retreat_plan = {'start': terminal.copy(), 'selected_torso': torso}
        return ([], [], None, [], np.zeros((8, 3)))
    node._plan_carried_return = carry
    node._carried_robot_transition_is_safe = record('extraction_volume')
    node._plan_retracted_transition = record('recovery', [])
    return NS(node=node, guard=guard, lift=lift, calls=calls, bay=object(), shelf_failure=shelf_failure)


@pytest.fixture
def open_tool(monkeypatch):
    checker = Mock(return_value=NS(ok=True, reason='open_approach_clear'))
    monkeypatch.setattr(lower_pick, 'check_open_gripper_approach', checker)
    return checker


def test_acceptance_requires_complete_route_and_returns_local_torso(protocol, open_tool):
    p = protocol
    plan = lower_pick.plan_lower_shelf_pick(p.node, FRONT, empty_guard=p.guard, bay=p.bay, lift_planner=p.lift)
    assert plan.pick_torso_height == .10 and p.node.pick_torso_height == .35
    assert plan.grasp == pytest.approx(FRONT+[.025, -.001, .005])
    assert plan.loaded_clearance_index == 1 and plan.deferred_return
    assert plan.endpoint_first and not plan.staged_empty_gripper
    assert plan.cached_post_retreat_plan['selected_torso'] == .10
    assert p.node._cached_post_retreat_plan is plan.cached_post_retreat_plan
    assert len(plan.unloaded_recovery_route) == 1  # Includes the HOME endpoint.
    assert [x[0] for x in p.calls] == ['opening', 'torso_edge', 'solver',
        'empty_candidate', 'lift', 'carry', 'extraction_volume', 'extraction_volume',
        'extraction_volume', 'recovery']
    assert open_tool.call_args.kwargs == {'torso_height': .10}
    p.node._move_arm.assert_not_called(); p.node._move_torso.assert_not_called()
    assert not plan.grasp.flags.writeable and not plan.solutions[0].flags.writeable


def test_rejected_complete_candidate_does_not_hide_valid_alternative(protocol, monkeypatch):
    p = protocol
    candidates = (lower_pick.LowerShelfCandidate('first', .10, -.50),
                  lower_pick.LowerShelfCandidate('second', .15, -.50,
                      carry_options={'extension_distance': .12}))
    monkeypatch.setattr(lower_pick, '_candidates', lambda _: candidates)
    original = p.lift
    def reject_first(grasp, extraction):
        if p.node.pick_torso_height == .10:
            p.node._cached_post_retreat_plan = {'rejected': True}
            raise RuntimeError('lift corner descent')
        assert p.node._cached_post_retreat_plan is None
        return original(grasp, extraction)
    plan = lower_pick.plan_lower_shelf_pick(p.node, FRONT, empty_guard=p.guard, bay=p.bay, lift_planner=reject_first)
    assert plan.candidate_name == 'second' and plan.pick_torso_height == .15
    assert p.node.pick_torso_height == .35
    assert p.node._cached_post_retreat_plan == plan.cached_post_retreat_plan
    carry = next(x for x in p.calls if x[0] == 'carry')
    assert carry[2] == {'defer_return': True, 'extension_distance': .12}
    assert plan.carry_options == {'extension_distance': .12}
    assert any(c.kwargs.get('stage') == 'lift'
               for c in p.node._publish_status.call_args_list)


@pytest.mark.parametrize('stage', ['opening', 'torso', 'empty', 'shelf_setup', 'shelf_entry', 'open', 'lift',
                                  'carry', 'volume', 'recovery'])
def test_each_guard_failure_prevents_acceptance_and_clears_state(protocol, open_tool, stage):
    p = protocol
    if stage == 'opening': p.guard.opening = lambda: False
    if stage == 'torso': p.guard.edge = lambda *args: False
    if stage == 'empty': p.guard.candidate = lambda *args: False
    if stage == 'shelf_setup': p.shelf_failure['setup'] = 'setup_shelf_collision'
    if stage == 'shelf_entry': p.shelf_failure['entry'] = 'entry_shelf_collision'
    if stage == 'open': open_tool.return_value = NS(ok=False, reason='finger_book_overlap')
    if stage == 'lift': p.lift = Mock(side_effect=RuntimeError('fresh joint state stale'))
    if stage == 'carry': p.node._plan_carried_return = Mock(side_effect=ValueError('shelf sweep'))
    if stage == 'volume': p.node._carried_robot_transition_is_safe = lambda *args: False
    if stage == 'recovery': p.node._plan_retracted_transition = lambda *args: None
    with pytest.raises(RuntimeError, match='no_complete_route'):
        lower_pick.plan_lower_shelf_pick(p.node, FRONT, empty_guard=p.guard, bay=p.bay, lift_planner=p.lift)
    assert p.node.pick_torso_height == .35
    assert p.node._cached_post_retreat_plan is None
    p.node._move_arm.assert_not_called(); p.node._move_torso.assert_not_called()


def test_cancellation_after_sensor_admission_does_not_try_another_candidate(protocol, monkeypatch):
    p = protocol
    monkeypatch.setattr(lower_pick, '_candidates', lambda _: (
        lower_pick.LowerShelfCandidate('first', .10, -.50),
        lower_pick.LowerShelfCandidate('second', .15, -.50)))
    def cancel(grasp, extraction):
        p.node._cancel.set()
        return NS(terminal=extraction[-1], route=extraction)
    with pytest.raises(lower_pick.LowerShelfPlanningCancelled):
        lower_pick.plan_lower_shelf_pick(p.node, FRONT, empty_guard=p.guard, bay=p.bay, lift_planner=cancel)
    assert not any(x[0] == 'carry' or x[1] == .15 for x in p.calls)
    assert p.node.pick_torso_height == .35 and p.node._cached_post_retreat_plan is None


def test_programming_errors_are_not_relabelled_as_geometry_rejections(protocol):
    p = protocol
    p.node._solve_cartesian_path = Mock(side_effect=TypeError('unexpected API mismatch'))
    with pytest.raises(TypeError, match='API mismatch'):
        lower_pick.plan_lower_shelf_pick(p.node, FRONT, empty_guard=p.guard, bay=p.bay, lift_planner=p.lift)
    assert p.node.pick_torso_height == .35 and p.node._cached_post_retreat_plan is None


@pytest.mark.parametrize('height', [1.584])
def test_unvalidated_rows_never_fall_back_to_a_top_row_route(protocol, height):
    p = protocol
    with pytest.raises(RuntimeError, match='no_validated_candidate'):
        lower_pick.plan_lower_shelf_pick(p.node, [.67, -.056, height],
                                   empty_guard=p.guard, bay=p.bay, lift_planner=p.lift)
    assert not p.calls and p.node._cached_post_retreat_plan is None


def test_precision_cannot_be_relaxed_for_candidate_selection(protocol):
    p = protocol; p.node.pick_position_tolerance = .002
    with pytest.raises(ValueError, match='precision_tolerances'):
        lower_pick.plan_lower_shelf_pick(p.node, FRONT, empty_guard=p.guard, bay=p.bay, lift_planner=p.lift)
    assert not p.calls


def test_candidate_can_select_forward_path_without_changing_other_admission(protocol, monkeypatch):
    p = protocol; p.node.expected_endpoint_first = False
    monkeypatch.setattr(lower_pick, '_candidates', lambda _: (
        lower_pick.LowerShelfCandidate('forward_candidate', .15, -.50,
            grasp_lateral_offset=0., grasp_vertical_offset=0., endpoint_first=False),))
    plan = lower_pick.plan_lower_shelf_pick(p.node, FRONT, empty_guard=p.guard, bay=p.bay, lift_planner=p.lift)
    assert plan.endpoint_first is False
    assert plan.grasp == pytest.approx(FRONT+[.025, 0., 0.])
    assert [x[0] for x in p.calls].count('lift') == 1
    assert [x[0] for x in p.calls].count('recovery') == 1
    assert p.node.pick_torso_height == .35


@pytest.mark.parametrize('flag', [0, 1, None, 'false'])
def test_endpoint_direction_requires_explicit_boolean(protocol, monkeypatch, flag):
    p = protocol
    monkeypatch.setattr(lower_pick, '_candidates', lambda _: (
        lower_pick.LowerShelfCandidate('invalid', .10, -.50, endpoint_first=flag),))
    with pytest.raises(RuntimeError, match='candidate_parameters_invalid'):
        lower_pick.plan_lower_shelf_pick(p.node, FRONT, empty_guard=p.guard, bay=p.bay, lift_planner=p.lift)
    assert not p.calls and p.node.pick_torso_height == .35


@pytest.mark.parametrize('row, front, head_pitch', [
    pytest.param(2, FRONT, 0., id='row2'),
    pytest.param(3, np.array([.658372465565475, -.05059891482849584,
                            .9287946083436958]), -.4, id='row3_blue_column4'),
])
def test_official_lower_rows_complete_candidate_with_real_guards(row, front, head_pitch):
    """Offline initial-state geometry fixture; not a physical retention trial."""
    from ament_index_python.packages import get_package_share_directory
    from test_shutdown import _official_manipulation_planner
    from erc_phase1_solution.motion_profiles import HOME, IK_JOINTS
    from erc_phase1_solution.empty_pickup_collision import EmptyPickupCollision
    from erc_phase1_solution.lift_first_extraction import plan_lift_first_extraction, RelativeShelfBay
    from erc_phase1_solution.shelf_cradle_geometry import ShelfCradleGeometry
    node = _official_manipulation_planner()
    # Restore the production measured-joint seed; the shared fixture fixes it
    # at HOME for its original static tests. This test also checks the carry end.
    del node._current_seed
    node.joints.update(dict(zip(IK_JOINTS, HOME)))
    node.joints['head_2_joint'] = head_pitch
    node.joints['gripper_left_finger_joint'] = .069
    node._cancel = threading.Event(); node._lock = threading.Lock()
    node._joint_stamps_ns = {key: 1_000_000_000 for key in node.joints}
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=1_000_000_000))
    node._publish_status = Mock()
    node.carried_book_dimensions = np.array([.16, .02, .25])
    node.gripper_open = .069; node.pregrasp_offset = .14
    node.pick_torso_height = .35
    node.pick_position_tolerance = .0005; node.pick_orientation_tolerance = .01
    node._shelf_cradle_geometry = ShelfCradleGeometry(
        Path(get_package_share_directory('erc_description'))/'urdf/tiago_pro.urdf',
        get_package_share_directory, immutable_local=True)
    guard = EmptyPickupCollision.capture(node)
    if row == 2:
        registration = dict(marker_center_base=[.605, -.006, 2.26],
            inward_axis_base=[1., 0., 0.], physical_column=3,
            source='offline representative central bay')
    else:
        # Trial 146e2f8aaa7b, seed 101, printed column 4 / physical column 5.
        # These are recorded onboard marker geometry and reached odometry,
        # not evaluator poses. The offline fixture does not assert freshness
        # or physical retention; live PICK still requires both.
        marker_odom = np.array([-2.1196444163596513, -2.6525038137816646,
                                2.2559112905035197])
        outward_odom = np.array([.047914200682930244, .9988514551087744])
        base = np.array([-2.0762829190092655, -2.062244776306888,
                         -1.6187288774425457])
        c, s = np.cos(base[2]), np.sin(base[2])
        odom_to_base = np.array([[c, s], [-s, c]])
        marker_base = np.r_[odom_to_base @ (marker_odom[:2]-base[:2]), marker_odom[2]]
        inward_base = np.r_[odom_to_base @ -outward_odom, 0.]
        registration = dict(marker_center_base=marker_base,
            inward_axis_base=inward_base, physical_column=5,
            source='recorded onboard marker and reached odometry, trial 146e2f8aaa7b')
    bay = RelativeShelfBay(**registration,
        lateral_uncertainty_m=.200, roof_uncertainty_m=.010,
        assume_upright_supported=True)
    def lift(grasp, extraction):
        return plan_lift_first_extraction(node, front, grasp, extraction, bay=bay,
            aperture=.020, lift_m=.020, modeled_tool_allowance_m=.005)
    plan = lower_pick.plan_lower_shelf_pick(node, front, empty_guard=guard, lift_planner=lift, bay=bay)
    cache = plan.cached_post_retreat_plan
    assert node.pick_torso_height == .35 and plan.pick_torso_height == .10
    assert cache['compact_radius'] < .45
    assert plan.lift_plan.metrics['minimum_withdrawal_corner_rise_m'] == .019
    assert guard.checked_samples > 61 and plan.unloaded_recovery_route
    assert node._gravity_supported_transition_is_safe(cache['terminal'], cache['terminal'])
    assert np.array_equal(cache['start'], plan.lift_plan.terminal)
    assert cache['shelf_front_x'] == pytest.approx(front[0] + .25)
    if row == 2:
        assert plan.grasp == pytest.approx(front + [.025, -.001, .005])
        assert cache['compact_shoulder_progress_power'] == 1.0
    else:
        assert plan.candidate_name == 'row3_supported_positive_wrist'
        assert not plan.endpoint_first and plan.loaded_clearance_index == 3
        assert plan.grasp == pytest.approx(front + [.025, 0., 0.])
        assert cache['compact_path'] == 'middle'
        assert cache['staging_vertical_offset_m'] == .18
        assert cache['terminal'][-1] > 0.  # Preserve the continuous supported wrist branch.
        assert {phase for _, phase in cache['legs']} == {
            'post_retreat_clearance_extension', 'post_retreat_cradle_roll',
            'supported_cradle_lowering', 'supported_cradle_retraction', 'compact_transport'}
        previous = cache['start']
        for goal, phase in cache['legs']:
            if phase not in ('post_retreat_clearance_extension', 'post_retreat_cradle_roll'):
                assert node._gravity_supported_transition_is_safe(previous, goal)
            previous = goal
        # Same retained endpoint must also admit the ordinary look-bin head move.
        node._held_book_corners = cache['attached_corners']
        node.joints.update(dict(zip(IK_JOINTS, cache['terminal'])))
        assert node._carried_head_transition_is_safe(0., -.60)
