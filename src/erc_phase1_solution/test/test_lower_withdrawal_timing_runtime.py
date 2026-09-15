"""Real PICK/executor handoff with the timing values used by Gazebo trials.

IK and controller adapters use sentinels; these tests certify dispatch scope,
duration, and gate ordering, not the geometry or physical grasp.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip('rclpy')
from erc_phase1_solution import lift_first_extraction as lift
from erc_phase1_solution import lower_shelf_pick as lower
from erc_phase1_solution import manipulation_node as manipulation
from test_lift_first_integration import _enabled
from test_shutdown import _mocked_pick_trace


PROFILE = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / 'config/collision_quality.yaml').read_text()
)['erc_manipulation']['ros__parameters']
TIMING_PARAMETERS = (
    'withdrawal_speed_scale', 'additional_arm_time_scale',
    'withdrawal_half_timing_enabled', 'withdrawal_quarter_timing_enabled',
    'lift_first_extraction_enabled', 'lift_first_extraction_lift_m',
)


def _profile(node):
    for name in TIMING_PARAMETERS:
        setattr(node, name, PROFILE[name])
    assert node.withdrawal_speed_scale == 3.0
    assert node.additional_arm_time_scale == 2.0
    assert node.withdrawal_half_timing_enabled
    assert node.withdrawal_quarter_timing_enabled
    assert node._execute_retained_arm_legs.__func__ is (
        manipulation.ManipulationNode._execute_retained_arm_legs)


@pytest.mark.parametrize('height,candidate', [
    (1.254, lower.ROW_TWO_CANDIDATES[0]),
    (.928, lower.ROW_THREE_CANDIDATES[0]),
    (.604, lower.BOTTOM_CANDIDATES[0]),
])
def test_actual_lower_pick_keeps_full_duration_and_gates_with_trial_profile(
        monkeypatch, height, candidate):
    def configure(node, seen):
        _profile(node)
        node.lower_shelf_pick_enabled = True

    seen, setup = _enabled(monkeypatch, configure=configure)

    def lift_proposal(node, front, grasp, original, **kwargs):
        # Row three has the live failure's one lift plus two extraction legs.
        route = tuple(np.full(8, float(40 + i)) for i in range(len(original) + 1))
        seen['route'] = route
        return lift.LiftFirstPlan(route, route[-1], np.zeros((8, 3)), {})

    monkeypatch.setattr(lift, 'plan_lift_first_extraction', lift_proposal)

    def lower_proposal(node, front, **kwargs):
        approach = tuple(np.full(8, float(10 + i)) for i in range(6))
        explicit = tuple(np.full(8, float(90 + i)) for i in range(2))
        withdrawal = (explicit if candidate.loaded_clearance_index is None else
                      tuple(reversed(approach[candidate.loaded_clearance_index:-1])))
        checked_lift = kwargs['lift_planner'](approach[-1], withdrawal)
        return SimpleNamespace(
            positions=tuple(np.asarray(front) for _ in approach),
            grasp=np.asarray(front), rotations=(np.eye(3),), solutions=approach,
            orientation_index=0, path_score=1., transition_waypoints=(),
            pick_torso_height=candidate.torso_height,
            grasp_depth_offset=candidate.grasp_depth_offset,
            grasp_lateral_offset=candidate.grasp_lateral_offset,
            grasp_vertical_offset=candidate.grasp_vertical_offset,
            loaded_clearance_index=candidate.loaded_clearance_index,
            endpoint_first=candidate.endpoint_first, extraction_solutions=explicit,
            lift_plan=checked_lift,
            return_result=([], [], None, [], np.zeros((8, 3))),
            cached_post_retreat_plan={'requires_gravity_support': False},
        )

    monkeypatch.setattr(lower, 'plan_lower_shelf_pick', lower_proposal)
    result = _mocked_pick_trace([.67, -.05, height], configure_node=setup)
    count = len(seen['route'])
    assert result['succeeded']
    assert result['arm_moves'][6:] == list(range(40, 40 + count))
    assert result['arm_durations'][6:] == [1.] + [5.8] * (count - 1)
    assert seen['pressure_gate_calls'] == 1
    assert seen['pressure_gate_order'] == ['admitted', 'published']
    assert [phase for _, phase, _ in result['fresh_probes']] == [
        'before_initial_shelf_lift', 'initial_shelf_lift']
    assert result['retained_phases'] == ['extraction'] * (count - 1)
    assert not result['closed_recoveries']
    assert not any(event == 'withdrawal_timing_admission_rejected'
                   for event, _ in result['status'])


def test_invalid_optimized_prefix_reports_cause_before_any_goal(monkeypatch):
    """Keep the strict top-route timing admission and its abort-only behavior."""
    def configure(node, seen):
        _profile(node)
        # The real half/quarter admission has separate physical-gate fixtures.
        # Here the base speed certificate receives the malformed two-leg prefix.
        node.withdrawal_half_timing_enabled = False
        node.withdrawal_quarter_timing_enabled = False
        seen['moves'] = []
        move = node._move_arm_solution
        node._move_arm_solution = lambda q, duration: (
            seen['moves'].append((int(q[1]), duration)) or move(q, duration))
        seen['events'] = []
        publish = node._publish_status
        node._publish_status = lambda event, **fields: (
            seen['events'].append((event, fields)) or publish(event, **fields))

    seen, setup = _enabled(monkeypatch, configure=configure)
    with pytest.raises(RuntimeError, match='^pick_recovery_failed$') as caught:
        _mocked_pick_trace([.70, 0., 1.58], configure_node=setup)
    assert isinstance(caught.value.__cause__, ValueError)
    assert 'exactly four withdrawal legs' in str(caught.value.__cause__)
    assert [value for value, _ in seen['moves']] == [10, 11, 12]
    assert seen['pressure_gate_calls'] == 0
    rejection = [fields for event, fields in seen['events']
                 if event == 'withdrawal_timing_admission_rejected']
    assert len(rejection) == 1
    assert rejection[0] == {
        'command': 'pick', 'reason': str(caught.value.__cause__), 'leg_count': 2,
        'retained_stop': True, 'recovery_halted': True,
    }
