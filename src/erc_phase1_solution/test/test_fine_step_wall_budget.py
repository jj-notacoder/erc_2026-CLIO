"""An opt-in slow-simulation wall allowance never changes physical evidence."""
from dataclasses import asdict, replace

import pytest

pytest.importorskip('rclpy')
from erc_phase1_solution import fine_gripper_close as fine
from test_fine_gripper_close import set_feedback
from test_fine_pressure_resettle import active_fixture, records
from test_contact_force_timing import RIGHT, force_message


def slow_fixture(monkeypatch, *, step_wall_seconds=15.):
    values=active_fixture(monkeypatch)
    c,node,now,clock,_,_=values
    c.limits=replace(c.limits,step_wall_seconds=step_wall_seconds)
    # At2.5% simulated real time, a fresh200ms dwell needs8 wall seconds.
    # Joint/contact freshness remains measured against the advancing ROS clock.
    def sleep(seconds):
        clock.wall+=seconds
        if clock.advancing:now.nanoseconds+=round(seconds*1e9*.025)
        if clock.on_sleep:clock.on_sleep()
    clock.sleep=sleep
    def sensors():
        set_feedback(node,now,c._fine_command)
        c._fine_side_last={'right':(now.nanoseconds,.3)}
    clock.on_sleep=sensors
    set_feedback(node,now,c._fine_command,6.7e-6)
    return values


def test_fifteen_seconds_is_explicit_and_only_changes_wall_allowance():
    default=fine.FineGripLimits()
    candidate=replace(default,step_wall_seconds=15.)
    assert default.step_wall_seconds==5. and candidate.step_wall_seconds==15.
    assert {k for k,v in asdict(default).items() if v!=asdict(candidate)[k]}=={'step_wall_seconds'}
    assert candidate.motion_seconds==candidate.dwell_seconds==.2
    assert candidate.micro_position_error==25e-9 and candidate.stationary_velocity==2e-6
    assert candidate.micro_preload_steps==20 and candidate.micro_preload_limit==2e-6
    assert replace(candidate,total_wall_seconds=900.).total_wall_seconds==900.


@pytest.mark.parametrize('seconds',[15.000001,0.,-1.,float('nan'),float('inf')])
def test_step_wall_allowance_stays_finite_positive_and_bounded(seconds):
    with pytest.raises(ValueError):fine.FineGripLimits(step_wall_seconds=seconds)


@pytest.mark.parametrize('allowance,passes',[(5.,False),(15.,True)])
def test_slow_simulation_requires_the_full_fresh_dwell_with_either_allowance(monkeypatch,allowance,passes):
    c,_,now,clock,messages,statuses=slow_fixture(monkeypatch,step_wall_seconds=allowance)
    start_ros=now.nanoseconds
    target,end,reference=c._fine_command,c._fine_motion_end_ns,c._fine_micro_reference
    result=c._resettle_micro_epoch(3,target,'joint_still_moving',wall_deadline=allowance)
    assert result is passes
    assert c._fine_command==target and c._fine_motion_end_ns==end and c._fine_micro_reference==reference
    assert not records(statuses,'public_position_command')
    if passes:
        assert clock.wall>5. and clock.wall<15. and not messages
        endpoint=records(statuses,'stationary_endpoint')[-1]
        assert now.nanoseconds-endpoint['stationary_start_ros_ns']>=200_000_000
        assert endpoint['stationary_start_ros_ns']>start_ros
        assert c._fine_evidence is None and c._fine_result is None
    else:
        assert c._fine_fault=='step_wall_timeout' and clock.wall<=5.0021
        assert not records(statuses,'stationary_endpoint') and len(messages)==1


def test_second_resettle_spends_remaining_original_fifteen_seconds_not_a_new_budget(monkeypatch):
    c,node,now,clock,messages,statuses=slow_fixture(monkeypatch)
    target=c._fine_command
    assert c._resettle_micro_epoch(3,target,'joint_still_moving',wall_deadline=15.)
    first_epoch=c._fine_micro_stationary_start_ns
    assert clock.wall>8.
    set_feedback(node,now,target,6.7e-6)
    assert not c._resettle_micro_epoch(3,target,'joint_still_moving',wall_deadline=15.)
    assert c._fine_fault=='step_wall_timeout' and clock.wall<=15.0021
    assert c._fine_micro_stationary_start_ns>first_epoch
    assert len(records(statuses,'pressure_epoch_invalidated'))==2
    assert len(records(statuses,'stationary_endpoint'))==1
    assert len(messages)==1 and messages[0].points[0].positions[0]==target
    assert not records(statuses,'public_position_command')


def test_original_nine_hundred_second_total_deadline_overrides_fifteen_second_step(monkeypatch):
    c,_,_,clock,messages,statuses=slow_fixture(monkeypatch)
    assert c._fine_wall_deadline==900.
    clock.wall=894.
    assert not c._resettle_micro_epoch(3,c._fine_command,'joint_still_moving',wall_deadline=909.)
    assert clock.wall<=900.0021 and c._fine_fault in ('fine_grip_wall_timeout','step_wall_timeout')
    assert len(messages)==1 and not records(statuses,'stationary_endpoint')


def test_longer_wall_policy_does_not_extend_the_one_second_clock_stall_guard(monkeypatch):
    c,_,_,clock,messages,statuses=slow_fixture(monkeypatch)
    clock.advancing=False
    assert not c._resettle_micro_epoch(3,c._fine_command,'joint_still_moving',wall_deadline=15.)
    assert c._fine_fault=='clock_stalled' and clock.wall<=1.004
    assert len(messages)==1 and not records(statuses,'stationary_endpoint')


@pytest.mark.parametrize('hazard,expected',[('force','force_overload'),('cancel','cancelled'),
                                          ('stream','contact_stream_lost')])
def test_hard_fault_after_five_seconds_interrupts_extended_wait_immediately(monkeypatch,hazard,expected):
    c,node,now,clock,messages,statuses=slow_fixture(monkeypatch)
    sensors=clock.on_sleep
    fired=[]
    def interrupted_sensors():
        sensors()
        if clock.wall>5. and not fired:
            fired.append(clock.wall)
            if hazard=='force':
                book=c.expected_model+'::book_base_link::collision'
                node._on_contacts(force_message(now.nanoseconds,(RIGHT,book,21.)))
            elif hazard=='cancel':node._cancel.set()
            else:c._fine_side_last={'right':(now.nanoseconds-151_000_000,.3)}
    clock.on_sleep=interrupted_sensors
    assert not c._resettle_micro_epoch(3,c._fine_command,'joint_still_moving',wall_deadline=15.)
    assert c._fine_fault==expected and fired and clock.wall-fired[0]<=.0021
    assert len(messages)==1 and not records(statuses,'stationary_endpoint')
    assert not records(statuses,'public_position_command')
