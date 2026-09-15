"""Larger bounded initial-preload probes keep stock commands and all hard gates."""
from dataclasses import asdict, replace

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from test_fine_micro_settling_profile import acquisition_fixture, candidate
from test_fine_gripper_close import pressure_fixture, set_feedback
from test_fine_pressure_resettle import records
from test_contact_force_timing import BOOK, LEFT, RIGHT, force_message


def probe_limits(**overrides):
    values = dict(micro_preload_step=500e-9,micro_preload_limit=25e-6)
    values.update(overrides)
    return candidate(**values)


def test_larger_probe_is_explicit_and_keeps_strict_defaults():
    default = fine.FineGripLimits()
    assert default.micro_preload_step==100e-9
    assert default.micro_preload_steps==20 and default.micro_preload_limit==2e-6
    old = candidate()
    selected = probe_limits()
    assert {k for k,v in asdict(old).items() if v!=asdict(selected)[k]} == {
        'micro_preload_step','micro_preload_limit'}
    assert selected.micro_preload_steps==50
    assert selected.motion_seconds==selected.dwell_seconds==.2
    assert selected.maximum_force==20. and selected.minimum_force==8.
    assert selected.micro_position_error==50e-9 and selected.micro_stationary_velocity==10e-6
    assert selected.stationary_velocity==2e-6
    assert selected.step_wall_seconds==15. and selected.total_wall_seconds==900.


@pytest.mark.parametrize('overrides',[
    {'micro_preload_step':500.001e-9}, {'micro_preload_step':0.},
    {'micro_preload_step':-1e-9}, {'micro_preload_step':float('nan')},
    {'micro_preload_step':float('inf')}, {'micro_preload_limit':75.001e-6},
    {'micro_preload_limit':24.999e-6}, {'micro_preload_steps':51},
])
def test_increment_and_total_bound_cannot_be_decoupled(overrides):
    with pytest.raises(ValueError): probe_limits(**overrides)


def test_all_fifty_larger_targets_share_reference_without_weak_pressure_acquisition(monkeypatch):
    c,_,_,clock,messages,statuses = acquisition_fixture(monkeypatch)
    c.limits = probe_limits()
    assert c.run()[0] is False and c._fine_fault is None
    assert c._fine_stop=='micro_preload_limit_reached'
    close = [x for x in records(statuses,'public_position_command') if x['purpose']=='close']
    assert len(close)==50 and len(messages)==51
    assert [x['position'] for x in close] == [c._fine_micro_reference-i*500e-9 for i in range(1,51)]
    assert all(x['motion_seconds']==.2 for x in close)
    assert c._fine_micro_reference-close[-1]['position']==pytest.approx(25e-6,abs=1e-15)
    assert min(x['position'] for x in close)>c.limits.floor
    assert c._fine_wall_deadline==900. and clock.wall<900.
    assert all(not x['pressure_evidence']['verified'] for x in records(statuses,'pressure_measurement'))
    start=records(statuses,'micro_preload_started')[-1]
    assert start['command_step_m']==500e-9 and start['maximum_additional_closure']==25e-6


def test_real_success_uses_one_larger_target_then_full_fresh_pressure_proof(monkeypatch):
    c,_,_,_,messages,statuses = acquisition_fixture(monkeypatch,strong=True)
    c.limits = probe_limits()
    assert c.run()[0] is True and c._fine_fault is None
    assert len(messages)==2
    assert c._fine_command==pytest.approx(c._fine_micro_reference-500e-9,abs=1e-15)
    pressure=records(statuses,'pressure_measurement')[-1]
    assert pressure['minimum_force']==8. and pressure['pressure_evidence']['verified']
    assert pressure['pressure_windows']['left']['span_seconds']>=.05
    assert pressure['pressure_windows']['right']['span_seconds']>=.05
    final=records(statuses,'stationary_endpoint')[-1]
    assert final['stationary_start_ros_ns']>=final['stationarity_diagnostic']['command_motion_end_ros_ns']
    assert final['stationarity_diagnostic']['longest_confirmed_stable_interval_seconds']>=.2


@pytest.mark.parametrize('fault',['force','effort','cancel','identity'])
def test_hard_callback_during_larger_step_stops_without_next_close(monkeypatch,fault):
    c,node,now,clock,messages,statuses=acquisition_fixture(monkeypatch)
    c.limits=probe_limits()
    old_sensors=clock.on_sleep
    fired=[]
    def sensors():
        old_sensors()
        if (not fired and c._fine_micro_active
                and c._fine_command<c._fine_micro_reference-250e-9
                and now.nanoseconds<c._fine_motion_end_ns):
            fired.append(now.nanoseconds)
            if fault=='force':
                node._on_contacts(force_message(now.nanoseconds,(RIGHT,BOOK,20.1)))
            elif fault=='identity':
                node._on_contacts(force_message(now.nanoseconds,
                    (RIGHT,'book_col_2_row_4_red::book_base_link::collision',.3)))
            elif fault=='effort':node._adaptive_overload_latched='effort_overload'
            else:node._cancel.set()
    clock.on_sleep=sensors
    assert c.run()[0] is False and fired
    expected={'force':'force_overload','effort':'effort_overload','cancel':'cancelled',
              'identity':'unexpected_or_anonymous_hand_contact'}[fault]
    assert c._fine_fault==expected
    close=[x for x in records(statuses,'public_position_command') if x['purpose']=='close']
    assert len(close)==1 and close[0]['position']==pytest.approx(c._fine_micro_reference-500e-9)
    count=len(messages)
    with c._fine_lock:
        assert not c._publish_once_locked(c._fine_command-500e-9,.2,'close')
    assert len(messages)==count
    assert node._cancel.is_set() is (fault=='cancel')


def test_selected_total_bound_applies_to_measured_position_not_only_commands(monkeypatch):
    c,node,now,_,_,_=pressure_fixture(monkeypatch,limits=probe_limits())
    set_feedback(node,now,c._fine_micro_reference-25e-6-1e-12)
    assert c._feedback_error()=='micro_preload_measured_travel_limit'


def test_width_floor_prevents_the_next_larger_target_before_publication(monkeypatch):
    c,node,now,_,messages,statuses=acquisition_fixture(monkeypatch)
    c.limits=probe_limits()
    def first_contact(*args):
        c._fine_phase='fine'
        set_feedback(node,now,c.limits.floor+400e-9)
        node._on_contacts(force_message(now.nanoseconds,(LEFT,BOOK,3.),(RIGHT,BOOK,3.)))
        return False
    c._step=first_contact
    assert c.run()[0] is False and c._fine_fault=='micro_preload_target_limit'
    assert not [x for x in records(statuses,'public_position_command') if x['purpose']=='close']
    assert all(m.points[0].positions[0]>=c.limits.floor for m in messages)
