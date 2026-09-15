"""Longer initial probes retain one travel reference and original hard stops."""
import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution.fine_gripper_close import FineGripLimits
from test_fine_micro_settling_profile import acquisition_fixture, candidate
from test_fine_pressure_resettle import records
from test_contact_force_timing import BOOK, RIGHT, force_message


def selected(**changes):
    values = dict(micro_preload_steps=300, micro_preload_step=250e-9,
                  micro_preload_limit=75e-6)
    values.update(changes)
    return candidate(**values)


def test_probe_keeps_defaults_timing_force_and_floor():
    default, limits = FineGripLimits(), selected()
    assert (default.micro_preload_steps, default.micro_preload_step,
            default.micro_preload_limit) == (20, 100e-9, 2e-6)
    assert limits.micro_preload_steps*limits.micro_preload_step == pytest.approx(75e-6)
    assert limits.step_wall_seconds == 15. and limits.total_wall_seconds == 900.
    assert limits.motion_seconds == limits.dwell_seconds == .2
    assert limits.minimum_force == 8. and limits.maximum_force == 20.
    assert limits.floor == .0175


@pytest.mark.parametrize('changes', [
    {'micro_preload_steps':301, 'micro_preload_step':100e-9},
    {'micro_preload_steps':True}, {'micro_preload_steps':300.5},
    {'micro_preload_limit':75.00001e-6},
    {'micro_preload_limit':74.99999e-6},
    {'micro_preload_steps':300, 'micro_preload_step':500e-9},
])
def test_count_and_total_remain_independently_bounded(changes):
    with pytest.raises(ValueError):
        selected(**changes)


def test_three_hundred_targets_do_not_turn_weak_pressure_into_success(monkeypatch):
    c, _, _, clock, messages, statuses = acquisition_fixture(monkeypatch)
    c.limits = selected()
    assert c.run()[0] is False
    assert c._fine_fault is None and c._fine_stop == 'micro_preload_limit_reached'
    close = [x for x in records(statuses, 'public_position_command') if x['purpose']=='close']
    assert len(close) == 300 and len(messages) == 301
    assert [x['position'] for x in close] == [c._fine_micro_reference-i*250e-9 for i in range(1,301)]
    assert c._fine_micro_reference-close[-1]['position'] == pytest.approx(75e-6, abs=1e-15)
    assert all(not x['pressure_evidence']['verified'] for x in records(statuses,'pressure_measurement'))
    assert c._fine_wall_deadline == 900. and clock.wall < 900.


@pytest.mark.parametrize('fault', ['force', 'cancel'])
def test_late_fault_after_prior_travel_cap_never_publishes_another_close(monkeypatch, fault):
    c, node, now, clock, messages, statuses = acquisition_fixture(monkeypatch)
    c.limits = selected()
    base_sensors = clock.on_sleep
    fired = []
    def sensors():
        base_sensors()
        if (not fired and c._fine_micro_active
                and c._fine_micro_reference-c._fine_command > 55e-6
                and now.nanoseconds < c._fine_motion_end_ns):
            fired.append(c._fine_command)
            if fault == 'force':
                node._on_contacts(force_message(now.nanoseconds,(RIGHT,BOOK,20.1)))
            else:
                node._cancel.set()
    clock.on_sleep = sensors
    assert c.run()[0] is False and fired
    assert c._fine_fault == ('force_overload' if fault=='force' else 'cancelled')
    close = [x for x in records(statuses,'public_position_command') if x['purpose']=='close']
    assert len(close)>200 and close[-1]['position'] == fired[0]
    before=len(messages)
    with c._fine_lock:
        assert not c._publish_once_locked(c._fine_command-250e-9,.2,'close')
    assert len(messages)==before


def test_slow_clock_still_exhausts_original_total_before_all_targets(monkeypatch):
    c, _, now, clock, _, statuses = acquisition_fixture(monkeypatch)
    c.limits = selected()
    def slow_sleep(seconds):
        clock.wall += 8*seconds
        now.nanoseconds += round(seconds*1e9)
        if clock.on_sleep:
            clock.on_sleep()
    clock.sleep = slow_sleep
    assert c.run()[0] is False
    assert c._fine_fault == 'fine_grip_wall_timeout'
    assert c._fine_wall_deadline == 900. and 900. <= clock.wall < 901.
    close = [x for x in records(statuses,'public_position_command') if x['purpose']=='close']
    assert 200 < len(close) < 300
