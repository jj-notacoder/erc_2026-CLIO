"""Actual cached flow and retained sender with controlled feedback/futures.

No new geometry, physical retention, acceleration or measured wall-time proof.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from erc_phase1_solution import additional_arm_timing as extra
from erc_phase1_solution import arm_velocity_admission as admission
import test_arm_velocity_admission as velocity
import test_additional_arm_timing as timing
import test_optional_arm_timing as original_timing
import test_supported_compact_floor as compact

ROOT = Path(__file__).resolve().parents[1]
FLAG = 'compact_extension_half_timing_enabled'


def fixture(*, enabled=True, fault=None):
    data = compact.compact_fixture(.175, scale=3., fault=fault)
    n, start, raw, sends, probes, endpoints, statuses, geometry = data
    # Two actual same-phase points, retaining the original terminal and all
    # later boundaries. Existing complete-flow fixture has one extension point.
    first = (start + raw[0][0]) / 2.
    new_raw = [(first, 'post_retreat_clearance_extension'), *raw]
    n._cached_post_retreat_plan['legs'] = new_raw
    setattr(n, FLAG, enabled)
    n.additional_arm_time_scale = 2.
    return (n, start, new_raw, sends, probes, endpoints, statuses, geometry)


def actual_flow(*, enabled=True, gate_failure=None, second_start_shift=False,
                fault=None, late=False, terminal_cancel=False):
    data = fixture(enabled=enabled, fault=fault)
    n, start, raw, calls, probes, endpoints, statuses, geometry = data
    base, events, unused_sent, records, acceptance, clock = velocity.sender_node(retained=True, late=late)
    if late:
        n.timeout = base.timeout
    for name in ('joints', '_joint_stamps_ns', '_faster_arm_velocity_limits',
                 '_faster_arm_velocity_urdf', 'get_clock', '_lock', 'command',
                 '_adaptive_command_guard', '_cancel', '_goal_handles',
                 '_pending_retained_acceptances', '_payload_hazard_reason',
                 '_trajectory_leg_at_elapsed_time', '_valid_retained_terminal_result',
                 '_cancel_retained_goal_and_confirm', '_cancel_goal_and_confirm',
                 '_cancel_late_retained_goal'):
        setattr(n, name, getattr(base, name))
    sent, gates, modes = [], [], []
    current_kwargs = {}
    def send(goal):
        assert n._lock.depth == 1 and len(n._pending_retained_acceptances) == 1
        if current_kwargs.get('velocity_admission'):
            assert n.command.depth == 1
        sent.append(copy.deepcopy(goal))
        modes.append((n._gravity_supported_payload, n._retention_probe_active,
                      n._payload_robot_watchdog_enabled))
        if terminal_cancel:
            n._cancel.set()
        return acceptance
    n.arm_client.send_goal_async = send
    def gate(node, goal):
        assert node._lock.depth == node.command.depth == 1
        gates.append(copy.deepcopy(goal))
        if gate_failure == len(gates):
            node._joint_stamps_ns[velocity.NAMES[0]] = velocity.NOW-150_000_001
        if second_start_shift and len(gates) == 2:
            node.joints[velocity.NAMES[0]] = -.30
        return admission.require_arm_velocity_locked(node, goal)
    sender = timing.load('_send_retained_arm_trajectory', time=clock,
                         require_arm_velocity_locked=gate,
                         retime_admitted_arm_goal=extra.retime_admitted_arm_goal)
    def invoke(goal, duration, legs, command, **kwargs):
        current_kwargs.clear(); current_kwargs.update(kwargs)
        calls.append(dict(goal=copy.deepcopy(goal), duration=duration,
                          legs=copy.deepcopy(legs), command=command, kwargs=dict(kwargs)))
        return sender(n, goal, duration, legs, command, **kwargs)
    n._send_retained_arm_trajectory = invoke
    old_endpoint = n._wait_for_retained_endpoint
    def endpoint(q, **kwargs):
        value = old_endpoint(q, **kwargs)
        if value is not None:
            n.joints.update(zip(velocity.NAMES, q[1:]))
        return value
    n._wait_for_retained_endpoint = endpoint
    old_publish = n._publish_status
    def publish(event, **fields):
        if event == 'arm_velocity_admission':
            assert n._lock.depth == n.command.depth == 0
            records.append(fields)
        else:
            old_publish(event, **fields)
    n._publish_status = publish
    return data, sent, records, gates, modes, acceptance


def run(data):
    n, start, *_ = data
    return compact.load('_execute_cached_post_retreat_compaction')(n, start, .8)


def test_actual_flow_retimes_only_extension_and_preserves_three_mode_boundaries():
    histories = []
    for enabled in (False, True):
        data, sent, records, gates, modes, _ = actual_flow(enabled=enabled)
        assert run(data) is True
        n, start, raw, calls, probes, endpoints, statuses, geometry = data
        assert len(calls) == len(sent) == 3
        assert probes == ['post_retreat', 'post_retreat_clearance_extension',
                          'post_retreat_cradle_roll', 'compact_transport_final']
        assert endpoints == ['post_retreat_clearance_extension', 'post_retreat_cradle_roll', 'compact_transport_final']
        assert modes == [(False, False, False), (True, True, True), (True, False, False)]
        assert len(geometry) == 3 and n._cached_post_retreat_plan is None
        assert statuses[-1][0] == ('transport_compact',)
        assert not n._pending_retained_acceptances and not n._goal_handles
        assert len(gates) == (4 if enabled else 2)
        assert len(records) == (2 if enabled else 1)
        assert calls[0]['kwargs'] == ({'velocity_admission': True, 'velocity_headroom': True} if enabled else {})
        assert calls[1]['kwargs'] == {} and calls[2]['kwargs'] == {'velocity_admission': True}
        assert compact.stamps(calls[0]['goal']) == [350_000_000, 700_000_000]
        assert calls[0]['duration'] == .7  # original watchdog, never .35
        assert compact.stamps(sent[0]) == ([175_000_000, 350_000_000] if enabled else [350_000_000, 700_000_000])
        if enabled:
            assert records[0]['additional_arm_timing']['nominal_duration_seconds'] == .7
            assert records[0]['maximum_commanded_velocity_limit_ratio'] <= .8
        histories.append((data, sent))
    (old, old_sent), (new, new_sent) = histories
    for a, b in zip(old[3], new[3]):
        assert a['goal'] == b['goal'] and a['duration'] == b['duration']
        assert [x[2] for x in a['legs']] == [x[2] for x in b['legs']]
        assert all(np.array_equal(x[0], y[0]) for x, y in zip(a['legs'], b['legs']))
    assert old_sent[1:] == new_sent[1:]  # roll and supported fold unchanged
    assert [tuple(p.positions) for p in old_sent[0].trajectory.points] == [tuple(p.positions) for p in new_sent[0].trajectory.points]


@pytest.mark.parametrize('which', [1, 2])
def test_either_actual_locked_admission_refuses_before_extension_send(which):
    data, sent, records, gates, modes, _ = actual_flow(gate_failure=which)
    with pytest.raises(admission.ArmVelocityAdmissionRejected, match='feedback stale'):
        run(data)
    n, _, _, calls, probes, endpoints, statuses, _ = data
    assert len(calls) == 1 and not sent and not modes and not endpoints
    assert len(gates) == which and records[-1]['admitted'] is False
    assert not n._gravity_supported_payload and not n._pending_retained_acceptances
    assert n._cached_post_retreat_plan is not None and not statuses


def test_second_fresh_start_still_requires_eighty_percent_headroom():
    data, sent, records, gates, modes, _ = actual_flow(second_start_shift=True)
    with pytest.raises(admission.ArmVelocityAdmissionRejected, match='80-percent'):
        run(data)
    assert len(gates) == 2 and not sent and not data[0]._gravity_supported_payload
    assert records[-1]['admitted'] is False and not data[0]._pending_retained_acceptances


@pytest.mark.parametrize('fault,expected_sends', [
    ('server', 0), ('initial_geometry', 0), ('post_retreat', 0),
    ('endpoint_1', 1), ('post_retreat_clearance_extension', 1),
    ('endpoint_2', 2), ('retention_after_roll', 2), ('post_retreat_cradle_roll', 2),
    ('post_roll_drift', 2), ('endpoint_3', 3), ('final_geometry', 3),
    ('compact_transport_final', 3),
])
def test_actual_full_flow_preserves_original_endpoint_retention_and_geometry_refusals(fault, expected_sends):
    data, sent, records, gates, modes, _ = actual_flow(fault=fault)
    with pytest.raises(RuntimeError):
        run(data)
    assert len(sent) == expected_sends
    assert data[0]._cached_post_retreat_plan is not None and not data[6]
    assert not data[0]._retention_probe_active and not data[0]._payload_robot_watchdog_enabled


def test_unknown_extension_acceptance_does_not_enter_roll_and_keeps_late_owner():
    data, sent, records, gates, modes, acceptance = actual_flow(late=True)
    with pytest.raises(original_timing.RetainedMotionNotStopped):
        run(data)
    n = data[0]
    assert len(sent) == len(acceptance.callbacks) == 1
    assert n._cancel.is_set() and len(n._pending_retained_acceptances) == 1
    assert not n._gravity_supported_payload and not data[5] and not data[6]
    # Actual sender attaches its existing callback; its detailed late-cancel
    # implementation remains covered by the complete original watchdog module.


def test_terminal_success_raced_by_cancel_cannot_advance_to_cradle_roll():
    data, sent, records, gates, modes, _ = actual_flow(terminal_cancel=True)
    with pytest.raises(RuntimeError, match='compact_transport_failed'):
        run(data)
    assert len(sent) == 1 and not data[0]._gravity_supported_payload
    assert not data[5] and not data[6]
    assert not data[0]._pending_retained_acceptances and not data[0]._goal_handles


def test_stale_feedback_after_actual_server_wait_refuses_first_publication():
    data, sent, records, gates, modes, _ = actual_flow()
    n = data[0]
    n.arm_client.wait_for_server = lambda **kw: (n._joint_stamps_ns.update(
        {velocity.NAMES[0]: velocity.NOW-150_000_001}) or True)
    with pytest.raises(admission.ArmVelocityAdmissionRejected, match='feedback stale'):
        run(data)
    assert not sent and not n._gravity_supported_payload and not n._pending_retained_acceptances


@pytest.mark.parametrize('factor', [1., 1.5, float('nan')])
def test_changed_factor_refuses_before_any_send(factor):
    data = fixture()
    data[0].additional_arm_time_scale = factor
    with pytest.raises(ValueError, match='requires additional_arm_time_scale=2'):
        run(data)
    assert not data[3] and not data[5] and not data[6]
    assert not data[0]._gravity_supported_payload


@pytest.mark.parametrize('flag', [None, 0, 1, 'true', []])
def test_actual_constructor_rejects_non_boolean_flag(flag):
    init = compact.method('__init__')
    i = next(i for i, x in enumerate(init.body) if isinstance(x, ast.Assign)
             and any(isinstance(t, ast.Attribute) and t.attr == FLAG for t in x.targets))
    block = ast.Module(body=init.body[i:i+3], type_ignores=[])
    n = NS(get_parameter=lambda name: NS(value=flag), additional_arm_time_scale=2.)
    with pytest.raises(ValueError, match='must be Boolean'):
        exec(compile(ast.fix_missing_locations(block), 'actual_extension_constructor', 'exec'), {'self': n})


@pytest.mark.parametrize('flag,factor,accepted', [(False, 1., True), (True, 2., True), (True, 1., False), (True, 1.5, False)])
def test_actual_constructor_requires_existing_factor_two_only_for_opt_in(flag, factor, accepted):
    init = compact.method('__init__')
    i = next(i for i, x in enumerate(init.body) if isinstance(x, ast.Assign)
             and any(isinstance(t, ast.Attribute) and t.attr == FLAG for t in x.targets))
    code = compile(ast.fix_missing_locations(ast.Module(body=init.body[i:i+3], type_ignores=[])), 'actual_extension_constructor', 'exec')
    n = NS(get_parameter=lambda name: NS(value=flag), additional_arm_time_scale=factor)
    if accepted:
        exec(code, {'self': n}); assert getattr(n, FLAG) is flag
    else:
        with pytest.raises(ValueError, match='requires additional_arm_time_scale=2'):
            exec(code, {'self': n})


def test_default_declaration_and_sender_untouched_by_runtime_source_inverse():
    values = compact.method('_declare_parameters')
    matches = [v for d in ast.walk(values) if isinstance(d, ast.Dict) for k, v in zip(d.keys, d.values)
               if isinstance(k, ast.Constant) and k.value == FLAG]
    assert len(matches) == 1 and isinstance(matches[0], ast.Constant) and matches[0].value is False
    from candidate_composition_support import restore_current_extensions
    restored = restore_current_extensions((ROOT/'erc_phase1_solution/manipulation_node.py').read_bytes())
    assert FLAG.encode() not in restored


def test_actual_optional_limit_initialization_reads_once_for_only_extension_flag(monkeypatch):
    from erc_phase1_solution import empty_pickup_collision
    init = compact.method('__init__')
    block, = [node for node in init.body if isinstance(node, ast.If)
              and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Attribute)
                      and t.attr == '_faster_arm_velocity_limits' for t in x.targets)
                      for x in node.body)]
    calls = []
    # Controlled loader return; this checks actual constructor wiring, not an
    # actual installed-URDF constructor execution (reserved for native focus).
    urdf = Path('/fixture/current/tiago_pro.urdf')
    monkeypatch.setattr(empty_pickup_collision, 'joint_velocity_limits',
                        lambda p: (calls.append(p) or [.035, *velocity.LIMITS]))
    n = NS(placement_transport_speed_scale=1., withdrawal_speed_scale=1.,
           empty_pickup_setup_retiming_enabled=False)
    setattr(n, FLAG, False)
    code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), 'actual_optional_limits', 'exec')
    scope = dict(self=n, urdf=urdf, __package__='erc_phase1_solution')
    exec(code, scope)
    assert not calls and not hasattr(n, '_faster_arm_velocity_limits')
    setattr(n, FLAG, True)
    exec(code, scope)
    assert calls == [urdf] and n._faster_arm_velocity_limits == velocity.LIMITS
    assert n._faster_arm_velocity_urdf == str(urdf)
