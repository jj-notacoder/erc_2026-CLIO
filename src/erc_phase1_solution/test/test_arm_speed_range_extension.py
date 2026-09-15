"""Optional range admission models; these do not establish physical safety.

Exercise actual extracted send methods and the unchanged serialized velocity
gate. No ROS node, model, geometry replay or simulation is constructed here.
"""
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from candidate_composition_support import restore_extended_arm_speed_source, restore_preopen_speed_timing_bytes
import test_optional_arm_timing as timing
import test_arm_velocity_admission as velocity


SOURCE = Path(__file__).resolve().parents[1]
PARENT_NODE_SHA = '14df13cfb6d80574b3a5a292b2f422a977f343e677ab9f17a986bbc0a3f71366'
PARENT_TIMING_SHA = 'ebc560051076adabb7ef3bb27705da39520d85c8c7d19e425285597404b07ceb'


def test_selected_profile_uses_new_maximum_but_default_remains_one():
    selected = (SOURCE/'config/collision_quality.yaml').read_text()
    assert selected.count('    placement_transport_speed_scale: 3.0\n') == 1
    assert 'placement_transport_speed_scale: 1.25' not in selected
    timing.test_default_declared_and_actual_constructor_validation_wiring()
    assert timing.timing.checked_arm_speed_scale(3.0) == 3.0
    assert timing.timing.scaled_arm_seconds(2.8, 1.) == 2.8


def test_whole_runtime_modules_restore_exact_r53_bytes():
    node = (SOURCE/'erc_phase1_solution/manipulation_node.py').read_bytes()
    # The published R53 node is LF; every byte outside these two fragments stays.
    restored = restore_extended_arm_speed_source(node.decode())
    assert hashlib.sha256(restored.encode()).hexdigest() == PARENT_NODE_SHA
    helper = restore_preopen_speed_timing_bytes((SOURCE/'erc_phase1_solution/arm_trajectory_timing.py').read_bytes())
    for current, parent in [
        (b'1.0 <= scale <= 2.5', b'1.0 <= scale <= 1.25'),
        (b'within [1.0, 2.5]', b'within [1.0, 1.25]'),
    ]:
        assert helper.count(current) == 1
        helper = helper.replace(current, parent)
    assert hashlib.sha256(helper).hexdigest() == PARENT_TIMING_SHA


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'unrelated'])
def test_historical_inverse_rejects_extra_or_missing_changes(damage):
    source = (SOURCE/'erc_phase1_solution/manipulation_node.py').read_text()
    fragment = 'float(duration) / 3.0 <= command_duration'
    if damage == 'missing':
        source = source.replace(fragment, 'float(duration) / 3.1 <= command_duration')
    elif damage == 'duplicate':
        source += '\n# ' + fragment + '\n'
    else:
        source += '\n# unrelated change must not be hidden by a fixture inverse\n'
    with pytest.raises(AssertionError):
        restore_extended_arm_speed_source(source)


@pytest.mark.parametrize('scale', [1.25, 2., 2.5, 3.0])
def test_actual_follow_admits_exact_serialized_time_with_original_locks(scale):
    n, events, sent, records, _, clock = velocity.sender_node()
    duration = 1. / scale
    assert timing.load('_follow', time=clock)(
        n, n.arm_client, velocity.NAMES, [.2]*7, 1., trajectory_duration=duration)
    assert len(sent) == len(records) == 1 and records[0]['admitted']
    p = sent[0].trajectory.points[0]
    exact_ns = p.time_from_start.sec*1_000_000_000 + p.time_from_start.nanosec
    assert exact_ns == int(duration*1_000_000_000)
    assert records[0]['points'][0]['time_from_start_ns'] == exact_ns
    assert records[0]['producer_stamps_ns'] == [velocity.NOW]*7
    assert records[0]['velocity_limits'] == list(velocity.LIMITS)
    assert p.positions == [.2]*7
    assert not p.velocities and not p.accelerations and not p.effort
    at = events.index(('send',))
    assert events[at-2:at] == [('enter', 'command'), ('enter', 'sensor')]
    assert events[at+1:at+3] == [('leave', 'sensor'), ('leave', 'command')]
    assert not n._goal_handles and not n._pending_retained_acceptances


@pytest.mark.parametrize('scale', [2., 2.5, 3.0])
@pytest.mark.parametrize('fault', ['stale', 'velocity'])
def test_feedback_change_during_server_wait_still_rejects_before_send(scale, fault):
    n, _, sent, records, _, clock = velocity.sender_node()
    def wait_for_server(**kwargs):
        if fault == 'stale':
            n._joint_stamps_ns[velocity.NAMES[0]] = velocity.NOW - 150_000_001
        else:
            n.joints[velocity.NAMES[0]] = -2.
        return True
    n.arm_client.wait_for_server = wait_for_server
    with pytest.raises(velocity.gate.ArmVelocityAdmissionRejected):
        timing.load('_follow', time=clock)(
            n, n.arm_client, velocity.NAMES, [.2]*7, 1., trajectory_duration=1./scale)
    assert not sent and not n._pending_retained_acceptances and not n._goal_handles
    assert len(records) == 1 and not records[0]['admitted']
    assert records[0]['start_positions'][0] == n.joints[velocity.NAMES[0]]
    assert not n._cancel.is_set()


def test_follow_exact_new_boundary_accepts_but_next_float_shorter_rejects():
    boundary = 2.8 / 3.0
    n, sent, _, _, clock = timing.follow_node()
    assert timing.load('_follow', time=clock)(
        n, n.arm_client, timing.ARM_JOINTS, [.1]*7, 2.8,
        trajectory_duration=boundary)
    assert len(sent) == 1
    n, sent, _, _, clock = timing.follow_node()
    with pytest.raises(ValueError):
        timing.load('_follow', time=clock)(
            n, n.arm_client, timing.ARM_JOINTS, [.1]*7, 2.8,
            trajectory_duration=math.nextafter(boundary, -math.inf))
    assert not sent


@pytest.mark.parametrize('scale', [2., 2.5, 3.0])
def test_new_scales_keep_original_nominal_watchdog_allowance(scale):
    n, sent, _, sleeps, clock = timing.follow_node(late=True)
    assert timing.load('_follow', time=clock)(
        n, n.arm_client, timing.ARM_JOINTS, [.1]*7, 2.8,
        trajectory_duration=2.8/scale)
    assert timing.seconds(sent[0].trajectory.points[0].time_from_start) == pytest.approx(2.8/scale)
    assert sleeps == [.02] and not n._goal_handles


@pytest.mark.parametrize('scale, expected', [(1.25, [.35, .64]), (2., [.35, .4]), (2.5, [.35, .35]), (3.0, [.35, .35])])
def test_actual_compact_block_preserves_floor_other_groups_and_phase_clock(scale, expected):
    f = timing.method_ast('_execute_cached_post_retreat_compaction')
    start = next(i for i, n in enumerate(f.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'supported_watchdog_duration' for t in n.targets))
    q = np.zeros(8)
    groups = [[(q, .4, 'extension')], [(q, .75, 'roll')], [(q, .35, 'lower'), (q, .8, 'tuck')]]
    scope = dict(self=NS(placement_transport_speed_scale=scale), timed_groups=groups,
                 checked_arm_speed_scale=timing.timing.checked_arm_speed_scale,
                 scaled_arm_seconds=timing.timing.scaled_arm_seconds)
    exec(compile(ast.fix_missing_locations(ast.Module(body=f.body[start:start+3], type_ignores=[])),
                 '<actual compact range block>', 'exec'), scope)
    assert groups[0][0][1:] == (.4, 'extension') and groups[1][0][1:] == (.75, 'roll')
    assert [v[1] for v in groups[2]] == pytest.approx(expected)
    assert [v[2] for v in groups[2]] == ['lower', 'tuck']
    assert all(v[0] is q for group in groups for v in group)
    assert scope['supported_watchdog_duration'] == pytest.approx(1.15)


@pytest.mark.parametrize('scale', [2., 2.5, 3.0])
@pytest.mark.parametrize('cancel', [False, True])
def test_new_range_keeps_pre_send_cancel_and_latched_hazard_stops(scale, cancel):
    n, sent, _, _, clock = timing.follow_node()
    kwargs = {}
    if cancel:
        kwargs['pre_send_check'] = n._cancel.set
    else:
        n._payload_robot_watchdog_enabled = True
        n._target_robot_contact_latched = True
    assert not timing.load('_follow', time=clock)(
        n, n.arm_client, timing.ARM_JOINTS, [.1]*7, 2.8,
        trajectory_duration=2.8/scale, **kwargs)
    assert not sent
