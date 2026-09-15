"""Explicit loaded microtarget tuning with real callbacks and frozen Run19 data.

Replay is a mechanical counterfactual over a bounded recorded tail, not a
force/acquisition proof. All publishers and time are fake; no ROS node starts.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from erc_phase1_solution.adaptive_grasp import ForceSample
from test_fine_gripper_close import fixture, pressure_fixture, set_feedback
from test_fine_pressure_resettle import records
from test_contact_force_timing import BOOK, LEFT, RIGHT, force_message


TAIL = Path(__file__).parent/'fixtures/run19_micro16_settling_tail.json'
TAIL_SHA256 = 'b66548e7321d4e82002b2fb3854120cc60e398259eb3d89908c9042f3fd9ae36'


def candidate(**kwargs):
    values = dict(maximum_force=20., minimum_force=8., total_wall_seconds=900.,
                  step_wall_seconds=15., micro_position_error=50e-9,
                  micro_stationary_velocity=10e-6, micro_preload_steps=50,
                  micro_preload_limit=5e-6)
    values.update(kwargs)
    return fine.FineGripLimits(**values)


def test_strict_defaults_and_ordinary_closing_are_unchanged():
    default = fine.FineGripLimits()
    changed = replace(default, micro_position_error=50e-9,
                      micro_stationary_velocity=10e-6,
                      micro_preload_steps=50, micro_preload_limit=5e-6)
    assert default.micro_position_error == 25e-9
    assert default.micro_stationary_velocity == default.stationary_velocity == 2e-6
    assert default.micro_preload_steps == 20 and default.micro_preload_limit == 2e-6
    assert {k for k,v in asdict(default).items() if v != asdict(changed)[k]} == {
        'micro_position_error','micro_stationary_velocity','micro_preload_steps','micro_preload_limit'}
    assert changed.stationary_velocity == 2e-6 and changed.stationary_position_error == 250e-9
    assert changed.micro_preload_step == 100e-9
    assert changed.motion_seconds == changed.dwell_seconds == .2


@pytest.mark.parametrize('overrides', [
    {'micro_position_error':50.001e-9}, {'micro_position_error':0.},
    {'micro_position_error':float('nan')}, {'micro_stationary_velocity':10.001e-6},
    {'micro_stationary_velocity':0.}, {'micro_stationary_velocity':float('inf')},
    {'micro_preload_steps':51}, {'micro_preload_steps':True}, {'micro_preload_steps':1.5},
    {'micro_preload_limit':75.001e-6}, {'micro_preload_limit':0.},
    {'micro_preload_steps':50,'micro_preload_limit':2e-6},
    {'micro_preload_step':50e-9,'micro_position_error':50e-9},
])
def test_micro_profile_bounds_and_coupled_budget_are_explicit(overrides):
    with pytest.raises(ValueError): candidate(**overrides)


@pytest.mark.parametrize('micro,phase', [
    (False,'preclose'), (False,'coarse'), (False,'fine'),
    (False,'coarse_contact_settle'), (True,'micro_preload')])
def test_candidate_envelope_is_micro_only_and_starts_after_command_end(monkeypatch,micro,phase):
    c,node,now,clock,messages,statuses = pressure_fixture(monkeypatch,limits=candidate())
    target = c._fine_command
    c._fine_phase = phase
    c._fine_motion_end_ns = now.nanoseconds+200_000_000
    command_end = c._fine_motion_end_ns
    def sensors():
        set_feedback(node,now,target+40e-9,8e-6)
        c._fine_side_last = {'right':(now.nanoseconds,.3)}
    clock.on_sleep = sensors
    sensors()
    assert c._wait_motion_and_stationary(target,micro=micro,wall_deadline=.5) is micro
    if micro:
        result = records(statuses,'stationary_endpoint')[-1]
        assert result['stationary_start_ros_ns'] >= command_end
        assert now.nanoseconds-result['stationary_start_ros_ns'] >= 200_000_000
        assert result['stationarity_diagnostic']['position_error_limit_m'] == 50e-9
        assert result['stationarity_diagnostic']['velocity_limit_mps'] == 10e-6
        assert not messages and c._fine_result is None
    else:
        assert c._fine_fault == 'step_wall_timeout'
        assert not records(statuses,'stationary_endpoint')


@pytest.mark.parametrize('error,velocity', [(50.1e-9,0.),(0.,10.1e-6)])
def test_even_candidate_outliers_cannot_complete_dwell(monkeypatch,error,velocity):
    c,node,now,clock,_,statuses = pressure_fixture(monkeypatch,limits=candidate())
    c._fine_motion_end_ns = now.nanoseconds
    def sensors():
        set_feedback(node,now,c._fine_command+error,velocity)
        c._fine_side_last = {'right':(now.nanoseconds,.3)}
    clock.on_sleep = sensors
    sensors()
    assert not c._wait_motion_and_stationary(c._fine_command,micro=True,wall_deadline=.3)
    assert not records(statuses,'stationary_endpoint') and c._fine_fault == 'step_wall_timeout'


@pytest.mark.parametrize('position_error,velocity,passes', [
    (25e-9,2e-6,False),(50e-9,2e-6,False),(50e-9,5e-6,False),(50e-9,10e-6,True)])
def test_recorded_run19_tail_through_actual_wait_has_no_pressure_claim(
        monkeypatch,position_error,velocity,passes):
    raw = TAIL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TAIL_SHA256
    archived = json.loads(raw)
    assert archived['source_sha256'] == '8fab481ec8228efd40326d713374f6eac0f77b17d7c41a4f94d902561f6e750d'
    d = archived['diagnostic']
    samples = d['observed_feedback_samples']
    assert len(samples) == 256 and d['distinct_feedback_stamps'] == 561
    c,node,now,clock,messages,statuses = pressure_fixture(monkeypatch,limits=candidate(
        micro_position_error=position_error,micro_stationary_velocity=velocity))
    c._fine_command = target = d['target']
    c._fine_micro_reference = target+16e-7
    c._fine_phase = 'micro_preload'
    c._fine_motion_end_ns = d['command_motion_end_ros_ns']
    index = 0
    def present(sample):
        now.nanoseconds = sample['observed_ros_ns']
        f = sample['feedback']
        set_feedback(node,now,f['position'],f['velocity'],f['stamp_ns'])
        # Readiness only: this synthetic fresh side must never become pressure proof.
        c._fine_side_last = {'right':(now.nanoseconds,.3)}
    present(samples[0])
    def advance(_seconds):
        nonlocal index
        clock.wall += .004
        index += 1
        if index < len(samples): present(samples[index])
        else: now.nanoseconds += 4_000_000
    clock.sleep = advance
    result = c._wait_motion_and_stationary(target,micro=True,wall_deadline=1.022)
    assert result is passes and c._fine_result is None and c._fine_evidence is None
    assert not records(statuses,'pressure_measurement')
    if passes:
        assert now.nanoseconds == 206_566_000_000 < archived['terminal_ros_ns']
        assert not messages
    else:
        assert c._fine_fault == 'step_wall_timeout'


def acquisition_fixture(monkeypatch, *, strong=False, disturb_final=None):
    c,node,now,clock,messages,statuses = fixture(monkeypatch,position=.069,limits=candidate())
    def first_contact(*args):
        c._fine_phase='fine'
        set_feedback(node,now,.01825)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,3.),(RIGHT,BOOK,3.)))
        return False
    c._step = first_contact
    def sensors():
        # The unchanged ordinary first-bilateral hold settles before entering
        # the selected micro envelope; inject loaded jitter in that phase only.
        set_feedback(node,now,c._fine_command+40e-9,8e-6 if c._fine_micro_active else 0.)
        after_step = (c._fine_micro_active and
                      c._fine_command < c._fine_micro_reference-50e-9)
        if strong and after_step and now.nanoseconds >= c._fine_motion_end_ns:
            pairs = ((LEFT,BOOK,8.5),(RIGHT,BOOK,8.7))
        else: pairs = ((RIGHT,BOOK,.3),)
        node._on_contacts(force_message(now.nanoseconds,*pairs))
    clock.on_sleep = sensors
    if disturb_final:
        original = c._micro_pressure_snapshot
        verified = []
        def snapshot(*args,**kwargs):
            result = original(*args,**kwargs)
            if result and result[1].verified:
                verified.append(now.nanoseconds)
                if len(verified)==2:
                    q = c._fine_command+(60e-9 if disturb_final=='position' else 40e-9)
                    v = 11e-6 if disturb_final=='velocity' else 8e-6
                    set_feedback(node,now,q,v)
            return result
        c._micro_pressure_snapshot = snapshot
    return c,node,now,clock,messages,statuses


@pytest.mark.parametrize('disturbance',[None,'position','velocity'])
def test_real_final_acquisition_needs_eight_newtons_and_rechecks_selected_envelope(monkeypatch,disturbance):
    c,_,_,_,messages,statuses = acquisition_fixture(monkeypatch,strong=True,disturb_final=disturbance)
    assert c.run()[0] is True and c._fine_fault is None
    assert len(messages)==2  # First bilateral hold, then one0.1um command only.
    pressure = records(statuses,'pressure_measurement')[-1]
    assert pressure['pressure_evidence']['verified'] and pressure['minimum_force']==8.
    assert pressure['pressure_windows']['left']['span_seconds'] >= .05
    assert pressure['pressure_windows']['right']['span_seconds'] >= .05
    assert pressure['retention_verified'] is False
    assert bool(records(statuses,'pressure_epoch_invalidated')) is bool(disturbance)
    assert c._fine_command == pytest.approx(c._fine_micro_reference-100e-9,abs=1e-15)


def test_fifty_steps_stay_on_one_original_reference_and_weak_pressure_never_acquires(monkeypatch):
    c,_,_,clock,messages,statuses = acquisition_fixture(monkeypatch)
    assert c.run()[0] is False and c._fine_fault is None
    assert c._fine_stop == 'micro_preload_limit_reached'
    commands = records(statuses,'public_position_command')
    closing = [e['position'] for e in commands if e['purpose']=='close']
    assert len(closing)==50 and len(messages)==51
    assert closing == [c._fine_micro_reference-i*100e-9 for i in range(1,51)]
    assert c._fine_micro_reference-closing[-1] == pytest.approx(5e-6,abs=1e-15)
    assert c._fine_wall_deadline == 900. and clock.wall < 900.
    assert all(not e['pressure_evidence']['verified'] for e in records(statuses,'pressure_measurement'))


def test_measured_cumulative_travel_stays_strict_even_with_larger_settling_envelope(monkeypatch):
    c,node,now,_,_,_ = pressure_fixture(monkeypatch,limits=candidate())
    set_feedback(node,now,c._fine_micro_reference-5e-6-1e-12)
    assert c._feedback_error() == 'micro_preload_measured_travel_limit'


@pytest.mark.parametrize('force,span,expected',[(7.9,62,False),(8.1,20,False),(8.1,62,True)])
def test_larger_envelope_never_substitutes_for_fresh_force_threshold_and_span(monkeypatch,force,span,expected):
    c,node,now,_,_,_ = pressure_fixture(monkeypatch,limits=candidate())
    set_feedback(node,now,c._fine_command+40e-9,8e-6)
    samples = tuple(ForceSample(now.nanoseconds-offset*1_000_000,
                    force if offset<=span else .1) for offset in range(62,-1,-2))
    node._adaptive_force_histories = lambda:(c.expected_model,samples,samples,None)
    result = c._micro_pressure_snapshot(1,c._fine_command)
    assert result is not None and result[1].verified is expected
